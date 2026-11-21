"""GPU (MLX) cost split for one Inkling 2 Mini layer application during prefill:
full layer vs. dense-only (MoE removed) vs. MoE alone vs. projections alone.

Emulates prefilling N tokens through ONE layer in chunks of C rows with a real cache.
Usage: PYTHONPATH=src .venv/bin/python scripts/inkling_ane_gpu_breakdown.py
"""
import argparse
import time

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.cache import ArraysCache, CacheList, KVCache, RotatingKVCache

from inkling_ane.mlx_layer import build_layer, load_config
from inkling_fast import fast


class Passthrough(nn.Module):
    def __call__(self, x):
        return x


def make_cache(layer, cfg):
    kv = RotatingKVCache(max_size=cfg.sliding_window_size, keep=0) if layer.is_sliding else KVCache()
    return CacheList(kv, ArraysCache(4))


def run_prefill(fn, n, c, reps=2):
    best = 1e9
    for _ in range(reps):
        state = fn("init")
        mx.synchronize()
        t = time.perf_counter()
        for s in range(0, n, c):
            mx.eval(fn(("step", s, state)))
        mx.synchronize()
        best = min(best, time.perf_counter() - t)
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layers", type=int, nargs="+", default=[7, 11, 2, 5])
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256, 512, 2048])
    p.add_argument("--fast", action="store_true", help="apply inkling_fast fused attention/MoE")
    args = p.parse_args()
    cfg, quant = load_config()
    mx.random.seed(0)
    for li in args.layers:
        layer = build_layer(li, cfg, quant)
        if args.fast:
            fast.fuse_attention(layer.self_attn)
            fast.fuse_moe(layer.mlp)
            import mlx_vlm.models.inkling2.language as L
            L.Attention.__call__ = fast._attention_call
            L.SparseMoE.__call__ = fast._moe_call
        loop = li >= cfg.loop_layer_index_start and li < cfg.loop_layer_index_end
        nblocks = max(1, li // 6)
        X = (mx.random.normal((1, args.n, 2304)) * 0.5).astype(mx.bfloat16)
        B = [(mx.random.normal((1, args.n, 2304))).astype(mx.bfloat16) for _ in range(nblocks)]
        M = (mx.random.normal((1, args.n, 4, 2304))).astype(mx.bfloat16)
        mx.eval(X, B, M)
        moe = layer.mlp

        def full(a, use_moe=True, msgs=loop):
            if a == "init":
                return make_cache(layer, cfg)
            _, s, cache = a
            e = s + C
            layer.mlp = moe if use_moe else Passthrough()
            blocks = [b[:, s:e] for b in B]
            if msgs:
                return layer(X[:, s:e], blocks, cache, M[:, s:e])
            return layer(X[:, s:e], blocks, cache)

        def moe_only(a):
            if a == "init":
                return None
            _, s, _ = a
            return moe(X[:, s:s + C])

        attn = layer.self_attn
        projs = [attn.q_proj, attn.k_proj, attn.v_proj, attn.r_proj, attn.gate_proj, attn.sink_proj]

        def proj_only(a):
            if a == "init":
                return None
            _, s, _ = a
            x = X[:, s:s + C]
            outs = [pj(x) for pj in projs if hasattr(pj, "weight") and pj.weight.size > 1]
            outs.append(attn.o_proj(mx.zeros((1, C, 3072), mx.bfloat16) + x[..., :1]))
            return outs

        kind = ("loop" if loop else "plain") + ("/sliding" if layer.is_sliding else "/global")
        for C in args.chunks:
            if args.n < C:
                continue
            t_full = run_prefill(full, args.n, C)
            t_dense = run_prefill(lambda a: full(a, use_moe=False), args.n, C)
            t_moe = run_prefill(moe_only, args.n, C)
            t_proj = run_prefill(proj_only, args.n, C)
            us = lambda t: t / args.n * 1e6
            print(f"L{li:2d} {kind:13s} C={C:4d}  full {us(t_full):6.1f}  dense(no MoE) {us(t_dense):6.1f}  "
                  f"MoE {us(t_moe):6.1f}  proj {us(t_proj):6.1f}  us/token  "
                  f"dense share {t_dense / (t_dense + t_moe):.0%}", flush=True)
        layer.mlp = moe
        del layer, X, B, M
        mx.clear_cache()


if __name__ == "__main__":
    main()
