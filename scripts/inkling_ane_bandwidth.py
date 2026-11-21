"""Aggregate DRAM read bandwidth: ANE weight-bound matvec (Q=1) vs GPU streaming read, alone and together.

build (CPU): PYTHONPATH=src .venv/bin/python scripts/inkling_ane_bandwidth.py build
run (lock):  scripts/with-gpu-lock ane-bw zsh scripts/inkling_ane_bandwidth.sh
"""
import argparse
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np

OUT = Path("runtime/ane/bw")
K, N, PAIRS = 4096, 8192, 5   # 10 linears of 33.5M params = 335M params


def build(args):
    import coremltools as ct
    import torch
    import torch.nn as nn
    OUT.mkdir(parents=True, exist_ok=True)
    layers = []
    for _ in range(PAIRS):
        layers += [nn.Linear(K, N, bias=False), nn.Linear(N, K, bias=False)]
    for l in layers:
        nn.init.normal_(l.weight, std=0.01)
    m = nn.Sequential(*layers).eval()
    tr = torch.jit.trace(m, (torch.randn(1, 1, K),))
    mlm = ct.convert(tr, inputs=[ct.TensorType(name="x", shape=(1, 1, K), dtype=np.float16)],
                     minimum_deployment_target=ct.target.macOS15, compute_units=ct.ComputeUnit.CPU_AND_NE,
                     skip_model_load=True)
    for kind in args.kinds:
        if kind == "int8":
            oc = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
                mode="linear_symmetric", dtype="int8", granularity="per_channel"))
            q = ct.optimize.coreml.linear_quantize_weights(mlm, oc)
        elif kind == "pal4":
            oc = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpPalettizerConfig(
                mode="kmeans", nbits=4, granularity="per_grouped_channel", group_size=16))
            q = ct.optimize.coreml.palettize_weights(mlm, oc)
        else:
            q = mlm
        q.save(str(OUT / f"matvec_{kind}.mlpackage"))
        print("built", kind, flush=True)


def ane(args):
    import coremltools as ct

    from inkling_ane.convert import placement
    path = OUT / f"matvec_{args.kind}.mlpackage"
    if args.placement:
        a, t, o = placement(path)
        print(f"placement {a}/{t} {dict(o)}", flush=True)
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    f = {"x": np.random.randn(1, 1, K).astype(np.float16)}
    for _ in range(5):
        m.predict(f)
    bytes_per = 2 * PAIRS * K * N * {"int8": 1, "pal4": 0.5, "fp16": 2}[args.kind]
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.seconds:
        m.predict(f)
        n += 1
    dt = (time.perf_counter() - t0) / n
    print(f"ANE matvec {args.kind}: {dt * 1e3:.2f} ms/call, weights {bytes_per / 1e6:.0f} MB -> {bytes_per / dt / 1e9:.1f} GB/s", flush=True)


def gpu(args):
    import mlx.core as mx
    a = mx.random.uniform(shape=(512 * 1024 * 1024,)).astype(mx.float16)  # 1 GiB
    mx.eval(a)
    for _ in range(3):
        mx.eval(a.sum())
    n, t0 = 0, time.perf_counter()
    w0, wn = t0, 0
    while time.perf_counter() - t0 < args.seconds:
        mx.eval(a.sum())
        n += 1; wn += 1
        now = time.perf_counter()
        if now - w0 >= 1.0:
            print(f"  GPU window t={now - t0:4.1f}s {a.nbytes * wn / (now - w0) / 1e9:.1f} GB/s", flush=True)
            w0, wn = now, 0
    dt = (time.perf_counter() - t0) / n
    print(f"GPU stream read: {a.nbytes / dt / 1e9:.1f} GB/s (avg over {args.seconds}s)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["build", "ane", "gpu"])
    p.add_argument("--kinds", nargs="+", default=["int8", "pal4", "fp16"])
    p.add_argument("--kind", default="int8")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--placement", action="store_true")
    args = p.parse_args()
    {"build": build, "ane": ane, "gpu": gpu}[args.mode](args)


if __name__ == "__main__":
    main()
