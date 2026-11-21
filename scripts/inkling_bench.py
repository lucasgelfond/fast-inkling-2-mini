"""Benchmark Inkling 2 Mini decode: plain greedy versus DSpark speculation.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/inkling_bench.py \
      --model weights/inkling-2-mini/mlx-3bit --modes baseline dspark
"""

import argparse
import json
import os
import time
from pathlib import Path

import mlx.core as mx

if not os.environ.get("INKLING_NO_PATCH"):
    import inkling_fast  # noqa: F401  registers inkling2_loop
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter

PROMPTS = {
    "code": "Write a Python function that parses an RFC 3339 timestamp string into a "
    "timezone-aware datetime without using dateutil. Include type hints and a docstring.",
    "chat": "Explain to a new engineer what a hash table does, in about 150 words.",
    "math": "A train leaves at 3:15 pm going 72 km/h. A second train leaves the same "
    "station at 4:00 pm going 90 km/h on the same track. When does the second train "
    "catch the first? Show the steps.",
}


def run(model, processor, prompt, max_tokens, draft=None, kind=None, **extra):
    kwargs = dict(max_tokens=max_tokens, temperature=0.0, **extra)
    if draft is not None:
        kwargs.update(draft_model=draft, draft_kind=kind)
    text, last = "", None
    start = time.perf_counter()
    for response in stream_generate(model, processor, prompt, **kwargs):
        text += response.text
        last = response
    wall = time.perf_counter() - start
    return {
        "prompt_tokens": last.prompt_tokens,
        "generation_tokens": last.generation_tokens,
        "prompt_tps": round(last.prompt_tps, 2),
        "generation_tps": round(last.generation_tps, 2),
        "peak_memory_gb": round(last.peak_memory, 2),
        "wall_s": round(wall, 2),
        "text": text,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="weights/inkling-2-mini/mlx-3bit")
    parser.add_argument("--modes", nargs="+", default=["baseline", "dspark"])
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS))
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--reasoning-effort", default="none")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--out", default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--drafter-bits", type=int, default=0)
    args = parser.parse_args()

    model, processor = load(args.model)
    if args.fast:
        from inkling_fast import decode as fastdecode
        fastdecode.install(model)
    draft = kind = None
    if "dspark" in args.modes:
        draft, kind = load_drafter(args.model)
        if args.block_size:
            draft.config.runtime_block_size = args.block_size
        if args.drafter_bits:
            import mlx.nn as nn
            nn.quantize(
                draft, group_size=64, bits=args.drafter_bits,
                class_predicate=lambda path, m: path.startswith("layers.") and isinstance(m, nn.Linear) and m.weight.shape[-1] % 64 == 0,
            )
            mx.eval(draft.parameters())

    # Warm up kernels so the first measured prompt is not penalized.
    warm = apply_chat_template(processor, model.config, "Hi", reasoning_effort="none")
    run(model, processor, warm, 8)

    results = []
    for name in args.prompts:
        prompt = apply_chat_template(
            processor, model.config, PROMPTS[name], reasoning_effort=args.reasoning_effort
        )
        for mode in args.modes:
            r = run(
                model,
                processor,
                prompt,
                args.max_tokens,
                draft=draft if mode == "dspark" else None,
                kind=kind if mode == "dspark" else None,
            )
            r.update(prompt=name, mode=mode)
            if mode == "dspark" and draft.accept_lens:
                r["mean_accept"] = round(sum(draft.accept_lens) / len(draft.accept_lens), 2)
                draft.accept_lens.clear(); draft.draft_lens.clear()
            results.append(r)
            print(
                f"{name:5s} {mode:8s} decode {r['generation_tps']:7.2f} tok/s  "
                f"prefill {r['prompt_tps']:7.1f} tok/s  "
                f"{r['generation_tokens']} tok  peak {r['peak_memory_gb']} GB  accept {r.get('mean_accept', '-')}",
                flush=True,
            )
            mx.clear_cache()

    by_prompt = {}
    for r in results:
        by_prompt.setdefault(r["prompt"], {})[r["mode"]] = r
    for name, modes in by_prompt.items():
        if {"baseline", "dspark"} <= modes.keys():
            same = modes["baseline"]["text"] == modes["dspark"]["text"]
            speed = modes["dspark"]["generation_tps"] / modes["baseline"]["generation_tps"]
            print(f"{name}: dspark {speed:.2f}x, identical text: {same}")
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
