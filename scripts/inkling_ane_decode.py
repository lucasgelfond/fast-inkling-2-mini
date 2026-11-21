"""Decode-side ANE measurements for Inkling 2 Mini.

convert (CPU only): Q=1 'decode' + Q=128 'prefill' multifunction slices (shared weights and state)
  PYTHONPATH=src .venv/bin/python scripts/inkling_ane_decode.py convert --layer 2 --pass 0
latency (ANE, take the GPU lock): per-call wall time of the Q=1 dense slice, including the handoff
  MLX array -> numpy -> Core ML predict -> numpy -> MLX array (mx.eval on both sides).
  scripts/with-gpu-lock ane-decode env PYTHONPATH=src .venv/bin/python scripts/inkling_ane_decode.py latency
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

OUT = Path("runtime/ane/decode3")


def convert(args):
    import coremltools as ct

    from inkling_ane.convert import build
    from inkling_ane.convert import convert as conv
    OUT.mkdir(parents=True, exist_ok=True)
    tag = "int8" if args.int8 else "fp16"
    parts = {}
    for name, q in (("decode", 1), ("prefill", 128)):
        path = OUT / f"L{args.layer}p{args.pass_}_{name}_{tag}.mlpackage"
        if not path.exists():
            conv(build(args.layer, args.pass_, q, 2048), path, args.int8)
        parts[name] = path
    d = ct.utils.MultiFunctionDescriptor()
    for name, path in parts.items():
        d.add_function(str(path), src_function_name="main", target_function_name=name)
    d.default_function_name = "decode"
    out = OUT / f"L{args.layer}p{args.pass_}_multi_{tag}.mlpackage"
    ct.utils.save_multifunction(d, str(out))
    size = lambda p: sum(f.stat().st_size for f in Path(p).rglob("*") if f.is_file()) / 1e6
    print(f"{out.name}: {size(out):.0f} MB (decode alone {size(parts['decode']):.0f} MB, prefill alone {size(parts['prefill']):.0f} MB)")


def latency(args):
    import coremltools as ct
    import mlx.core as mx

    from inkling_ane.convert import n_blocks
    from inkling_ane.torch_layer import host_mask
    for layer, lp in ((2, 0), (7, 1)):
        path = OUT / f"L{layer}p{lp}_multi_{args.tag}.mlpackage"
        dec = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="decode")
        pre = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE, function_name="prefill")
        st = dec.make_state()
        # prefill 128 rows into the shared state, then decode
        f = {"x": np.random.randn(1, 128, 2304).astype(np.float16) * 0.3, "mask": host_mask(0, 128, 1024, True).astype(np.float16)}
        if n_blocks(layer):
            f["blocks"] = np.random.randn(1, n_blocks(layer), 128, 2304).astype(np.float16)
        if lp:
            f["msg_states"] = np.random.randn(1, 2, 128, 2304).astype(np.float16)
        pre.predict(f, state=st)
        x = mx.random.normal((1, 1, 2304)).astype(mx.bfloat16)
        blocks = mx.random.normal((1, n_blocks(layer), 1, 2304)).astype(mx.float16)
        msg = mx.random.normal((1, 2, 1, 2304)).astype(mx.float16)
        mask = host_mask(128, 1, 1024, True).astype(np.float16)
        mx.eval(x, blocks, msg)

        def call(x):
            feed = {"x": np.array(x.astype(mx.float16)), "mask": mask}
            if n_blocks(layer):
                feed["blocks"] = np.array(blocks)
            if lp:
                feed["msg_states"] = np.array(msg)
            o = dec.predict(feed, state=st)
            h, r = mx.array(o["h"]), mx.array(o["residual"])
            mx.eval(h, r)
            return h, r

        for _ in range(10):
            call(x)
        n = 200
        t = time.perf_counter()
        for _ in range(n):
            h, r = call(x)
            x = (r + h * 0.0).astype(mx.bfloat16)  # real dependency: next input from this output
        wall = (time.perf_counter() - t) / n
        feed = {"x": np.zeros((1, 1, 2304), np.float16), "mask": mask}
        if n_blocks(layer):
            feed["blocks"] = np.array(blocks)
        if lp:
            feed["msg_states"] = np.array(msg)
        t = time.perf_counter()
        for _ in range(n):
            dec.predict(feed, state=st)
        pred = (time.perf_counter() - t) / n
        print(f"L{layer}p{lp} decode Q=1 ({args.tag}): wall incl. MLX<->CoreML handoff {wall * 1e3:.3f} ms/call, "
              f"predict() alone {pred * 1e3:.3f} ms -> 48 calls/token = {48 * wall * 1e3:.1f} ms", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["convert", "latency"])
    p.add_argument("--layer", type=int, default=2)
    p.add_argument("--pass", dest="pass_", type=int, default=0)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--tag", default="fp16")
    args = p.parse_args()
    convert(args) if args.mode == "convert" else latency(args)


if __name__ == "__main__":
    main()
