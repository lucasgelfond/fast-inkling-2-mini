"""Run one side of the ANE/GPU overlap test for --seconds and report its throughput.

GPU side (inkling2-venv): MoE of one layer on C rows, fused fast path.
  PYTHONPATH=src .venv/bin/python scripts/inkling_ane_contention.py moe --c 512
ANE side (ane-venv): repeated predict() of a converted dense slice.
  PYTHONPATH=src .venv/bin/python scripts/inkling_ane_contention.py ane --model runtime/ane/m2/L7p1_q64_fp16.mlpackage
"""
import argparse
import time

import numpy as np


def moe(args):
    import mlx.core as mx
    import mlx_vlm.models.inkling2.language as L

    from inkling_ane.mlx_layer import build_layer, load_config
    from inkling_fast import fast
    cfg, quant = load_config()
    layer = build_layer(args.layer, cfg, quant)
    if not args.upstream:
        fast.fuse_moe(layer.mlp)
        L.SparseMoE.__call__ = fast._moe_call
    x = (mx.random.normal((1, args.c, 2304)) * 0.5).astype(mx.bfloat16)
    mx.eval(x)
    for _ in range(3):
        mx.eval(layer.mlp(x))
    n, t0 = 0, time.perf_counter()
    w0, wn = t0, 0
    while time.perf_counter() - t0 < args.seconds:
        mx.eval(layer.mlp(x))
        n += 1
        wn += 1
        now = time.perf_counter()
        if args.windows and now - w0 >= 1.0:
            print(f"  MoE window t={now - t0:4.1f}s {(now - w0) / wn / args.c * 1e6:.1f} us/token", flush=True)
            w0, wn = now, 0
    dt = (time.perf_counter() - t0) / n
    print(f"MoE L{args.layer} C={args.c}: {dt * 1e3:.2f} ms/call, {dt / args.c * 1e6:.1f} us/token", flush=True)


def ane(args):
    import coremltools as ct
    m = ct.models.MLModel(args.model, compute_units=ct.ComputeUnit.CPU_AND_NE)
    st = m.make_state()
    spec = m.get_spec()
    feed = {}
    for inp in spec.description.input:
        shape = tuple(inp.type.multiArrayType.shape)
        feed[inp.name] = (np.random.randn(*shape) * 0.1).astype(np.float16)
        if inp.name == "mask":
            feed[inp.name][:] = 0
    for _ in range(3):
        m.predict(feed, state=st)
    print("ANE start", time.strftime("%T"), flush=True)
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        m.predict(feed, state=st)
        n += 1
    dt = (time.perf_counter() - t0) / n
    print(f"ANE {args.model.split('/')[-1]}: {dt * 1e3:.2f} ms/call", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("side", choices=["moe", "ane"])
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--c", type=int, default=512)
    p.add_argument("--upstream", action="store_true")
    p.add_argument("--model", default="runtime/ane/m2/L7p1_q64_fp16.mlpackage")
    p.add_argument("--seconds", type=float, default=8)
    p.add_argument("--windows", action="store_true")
    args = p.parse_args()
    moe(args) if args.side == "moe" else ane(args)


if __name__ == "__main__":
    main()
