"""Chain Inkling 2 Mini one layer at a time in MLX (low memory) and dump, for every layer
application, the inputs and the dense-part outputs the ANE slice must reproduce:
x (layer input), blocks, message states, h (MoE input), residual, out.

Usage: PYTHONPATH=src .venv/bin/python scripts/inkling_ane_golden.py --n 48 --out <dir>
--n 48 uses the reference prompt and checks against benchmarks/inkling/reference.
"""
import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from inkling_ane.mlx_layer import ROOT, build_layer, build_message_input, load_config, tensors


def embed(ids, cfg, quant):
    w = tensors("embed_tokens.")
    qc = quant["language_model.model.embed_tokens"]
    e = nn.QuantizedEmbedding(cfg.vocab_size, cfg.hidden_size, qc["group_size"], qc["bits"])
    e.load_weights(list(w.items()))
    norm = nn.RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    norm.load_weights(list(tensors("embed_norm.").items()))
    return norm(e(ids))


class Tap(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.seen = None

    def __call__(self, *a):
        out = self.inner(*a)
        self.seen = (a[0], out)
        return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=48)
    p.add_argument("--out", required=True)
    p.add_argument("--light", action="store_true", help="save only x per step (small)")
    p.add_argument("--cpu", action="store_true", help="run MLX on the CPU (leave the GPU alone)")
    args = p.parse_args()
    if args.cpu:
        mx.set_default_device(mx.cpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg, quant = load_config()
    ref = ROOT / "benchmarks/inkling/reference"
    if args.n == 48:
        tokens = json.loads((ref / "tokens.json").read_text())
    else:
        tokens = [200 + (i * 37) % 5000 for i in range(args.n)]
    x = embed(mx.array([tokens]), cfg, quant).astype(mx.bfloat16)
    mx.eval(x)
    np.save(out / "embed.npy", np.array(x[0].astype(mx.float32)))
    start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    sched = [(i, 0) for i in range(start)]
    sched += [(i, q) for q in range(cfg.loop_num_passes) for i in range(start, end)]
    sched += [(i, 0) for i in range(end, cfg.num_hidden_layers)]
    msg_in = build_message_input(cfg)
    blocks, pre_loop, previous = [], None, None
    layers = {}
    f32 = lambda a: np.array(a[0].astype(mx.float32))
    for step, (i, q) in enumerate(sched):
        if i == start:
            if q == 0:
                pre_loop = (x, list(blocks))
            else:
                previous = (blocks[len(pre_loop[1]) + 1:], x)
                x, blocks = pre_loop[0], list(pre_loop[1])
        if i not in layers:
            layers = {i: build_layer(i, cfg, quant)}
            mx.clear_cache()
        layer = layers[i]
        moe, sconv = layer.mlp, layer.mlp_sconv
        layer.mlp, layer.mlp_sconv = Tap(moe), Tap(sconv)
        rec = {"x": f32(x), "blocks": np.stack([f32(b) for b in blocks]) if blocks else np.zeros((0, args.n, 2304), np.float32)}
        kv = RotatingKVCache(max_size=cfg.sliding_window_size, keep=0) if layer.is_sliding else KVCache()
        cache = CacheList(kv, ArraysCache(4))
        xin = x
        if q > 0:
            rec["msg_states"] = np.stack([f32(b) for b in previous[0]] + [f32(previous[1])])
            messages = msg_in(previous[0], previous[1], i - start)
            x = layer(x, blocks, cache, messages)
        else:
            x = layer(x, blocks, cache)
        mx.eval(x)
        h, sc = layer.mlp.seen[0], layer.mlp_sconv.seen[1]
        rec.update(h=f32(h), residual=f32(x.astype(mx.float32) - sc.astype(mx.float32)), out=f32(x))
        layer.mlp, layer.mlp_sconv = moe, sconv
        np.savez(out / f"step{step}.npz", **({"x": rec["x"]} if args.light else rec))
        if args.n == 48 and step + 1 < len(sched) and sched[step + 1][0] != start:
            r = np.load(ref / f"hidden_step{step + 1}.npy")
            o = rec["out"]
            cos = (o * r).sum() / np.linalg.norm(o) / np.linalg.norm(r)
            print(f"step {step:2d} layer {i:2d} pass {q}: cos vs reference {cos:.6f} maxdiff {np.abs(o - r).max():.4f}", flush=True)
        else:
            print(f"step {step:2d} layer {i:2d} pass {q}", flush=True)
    (out / "meta.json").write_text(json.dumps({"tokens": tokens, "schedule": sched}))


if __name__ == "__main__":
    main()
