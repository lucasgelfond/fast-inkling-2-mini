"""GPU-only (MLX) prefill baseline with the same random prompts as inkling_ane_prefill.py bench.
Run under the GPU lock:
  scripts/with-gpu-lock ane-baseline env PYTHONPATH=src .venv/bin/python scripts/inkling_ane_baseline.py
"""
import argparse
import time

import mlx.core as mx
import numpy as np
from mlx_vlm import load

import inkling_fast  # noqa: F401
from inkling_fast import fast


def prefill(lm, ids, step):
    """Chunked prefill, no logits (the hybrid also stops at the final hidden state)."""
    cache = lm.make_cache()
    for s in range(0, len(ids), step):
        out = lm(mx.array([ids[s:s + step]]), cache=cache, skip_logits=True, return_hidden=True)
        mx.eval(out.hidden_states, [c.state for c in cache])
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lengths", type=int, nargs="+", default=[512, 1024, 2048])
    p.add_argument("--steps", type=int, nargs="+", default=[512, 2048])
    p.add_argument("--fast", action="store_true")
    args = p.parse_args()
    model, _ = load("weights/inkling-2-mini/mlx-3bit")
    if args.fast:
        print("fast.apply", fast.apply(model))
    lm = model.language_model
    rng = np.random.default_rng(0)
    for n in args.lengths:
        ids = [int(v) for v in rng.integers(200, 100000, n)]
        for st in args.steps:
            if st > n and st != args.steps[0]:
                continue
            prefill(lm, ids[:256], 256)
            best = 1e9
            for _ in range(2):
                t = time.perf_counter()
                prefill(lm, ids, st)
                best = min(best, time.perf_counter() - t)
            print(f"MLX {'fast' if args.fast else 'upstream'} N={n:5d} step={st:4d}: {best:6.2f}s {n / best:6.1f} tok/s", flush=True)


if __name__ == "__main__":
    main()
