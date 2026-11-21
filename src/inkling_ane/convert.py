"""Convert one dense layer application (or a few) to Core ML, check ANE placement.

python -m inkling_ane.convert --layer 7 --pass 1 --q 64 --out runtime/ane/models [--int8]
"""

import argparse
import time
from collections import Counter
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from .torch_layer import D, DenseLayer, G
from .weights import config, layer_weights


def n_blocks(i):
    # blocks before this layer's own append: boundaries at 0, 6, 12, 18 (x0 appended at layer 0)
    return 0 if i == 0 else (i - 1) // 6 + 1


def build(i, lp, q, s_global=2048, wscale=1.0):
    cfg, _ = config()
    w = layer_weights(f"layers.{i}.")
    msgq = layer_weights("message_input.")["query"][i - 6] if lp else None
    return DenseLayer(cfg, i, lp, q, w, msg_query=msgq, s_global=s_global, wscale=wscale).eval()


def example_inputs(m):
    q = m.q
    ex = {"x": torch.randn(1, q, D), "mask": torch.zeros(1, 1, G * q, m.S + q)}
    nb = n_blocks(m.i)
    if nb:
        ex["blocks"] = torch.randn(1, nb, q, D)
    if m.messages:
        ex["msg_states"] = torch.randn(1, 2, q, D)
    return ex


class Wrap(torch.nn.Module):
    def __init__(self, m, names):
        super().__init__()
        self.m, self.names = m, names

    def forward(self, *args):
        kw = dict(zip(self.names, args, strict=False))
        return self.m(kw["x"], kw["mask"], kw.get("blocks"), kw.get("msg_states"))


def convert(m, path, int8=False, pal=0):
    ex = example_inputs(m)
    names = list(ex)
    wrap = Wrap(m, names).eval()
    with torch.no_grad():
        traced = torch.jit.trace(wrap, tuple(ex.values()))
    states = [ct.StateType(ct.TensorType(shape=tuple(getattr(m, n).shape), dtype=np.float16), name=f"m.{n}")
              for n in m.state_names()]
    inputs = [ct.TensorType(name=n, shape=tuple(v.shape), dtype=np.float16) for n, v in ex.items()]
    outputs = [ct.TensorType(name="h", dtype=np.float16), ct.TensorType(name="residual", dtype=np.float16)]
    mlm = ct.convert(traced, inputs=inputs, outputs=outputs, states=states,
                     minimum_deployment_target=ct.target.macOS15,
                     compute_units=ct.ComputeUnit.CPU_AND_NE, compute_precision=ct.precision.FLOAT16,
                     skip_model_load=True)
    ops = [op.op_type for f in mlm._mil_program.functions.values() for op in f.operations]
    print(f"read_state {ops.count('read_state')} write_state {ops.count('coreml_update_state')}", flush=True)
    if int8:
        oc = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype="int8", granularity="per_channel", weight_threshold=200_000))
        mlm = ct.optimize.coreml.linear_quantize_weights(mlm, oc)
    elif pal:
        oc = ct.optimize.coreml.OptimizationConfig(global_config=ct.optimize.coreml.OpPalettizerConfig(
            mode="kmeans", nbits=pal, granularity="per_grouped_channel", group_size=16, weight_threshold=200_000))
        mlm = ct.optimize.coreml.palettize_weights(mlm, oc)
    mlm.save(str(path))
    return names


def placement(path):
    from coremltools.models.compute_plan import MLComputePlan
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    plan = MLComputePlan.load_from_path(m.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    prog = plan.model_structure.program
    count, other = Counter(), Counter()
    for _fname, fn in prog.functions.items():
        for op in fn.block.operations:
            if op.operator_name in ("const",) or op.operator_name.startswith("constexpr"):
                continue
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            dev = type(u.preferred_compute_device).__name__ if u else "None"
            count[dev] += 1
            if "NeuralEngine" not in dev:
                other[(op.operator_name, dev)] += 1
    total = sum(count.values())
    ane = sum(v for k, v in count.items() if "NeuralEngine" in k)
    return ane, total, other


def bench(path, names, shapes, iters=30):
    m = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
    st = m.make_state()
    feed = {n: np.random.randn(*s).astype(np.float16) * 0.1 for n, s in zip(names, shapes, strict=False)}
    feed["mask"] = np.zeros(shapes[names.index("mask")], np.float16)
    for _ in range(3):
        m.predict(feed, state=st)
    t = time.perf_counter()
    for _ in range(iters):
        m.predict(feed, state=st)
    return (time.perf_counter() - t) / iters


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--layer", type=int, default=7)
    p.add_argument("--pass", dest="lp", type=int, default=1)
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--s-global", type=int, default=2048)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--pal", type=int, default=0)
    p.add_argument("--wscale", type=float, default=1.0)
    p.add_argument("--out", default="runtime/ane/models")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = "int8" if args.int8 else (f"pal{args.pal}" if args.pal else "fp16")
    path = out / f"L{args.layer}p{args.lp}_q{args.q}_{tag}.mlpackage"
    m = build(args.layer, args.lp, args.q, args.s_global, args.wscale)
    t = time.perf_counter()
    names = convert(m, path, args.int8, args.pal)
    print(f"converted {path.name} in {time.perf_counter() - t:.0f}s", flush=True)
    ane, total, other = placement(path)
    print(f"ANE placement {ane}/{total} = {ane / total:.1%}; non-ANE: {dict(other)}", flush=True)
    shapes = [tuple(v.shape) for v in example_inputs(m).values()]
    dt = bench(path, names, shapes)
    print(f"latency {dt * 1e3:.2f} ms per Q={args.q} call -> {dt / args.q * 1e6:.1f} us/token per layer app, "
          f"48 apps -> {args.q / dt / 48:.0f} tok/s", flush=True)


if __name__ == "__main__":
    main()
