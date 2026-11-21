"""Time Inkling 2 Mini forward passes at several query lengths, plus DSpark acceptance."""

import argparse
import time

import mlx.core as mx
from mlx_vlm import load, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.speculative.drafters import load_drafter

import inkling_fast  # noqa: F401


def time_forward(lm, cache, n, iters=10):
    tokens = mx.array([[1000 + i for i in range(n)]])
    # Each call extends the cache; good enough for a cost profile.
    out = lm(tokens, cache=cache)
    mx.eval(out.logits)
    start = time.perf_counter()
    for _ in range(iters):
        out = lm(tokens, cache=cache)
        mx.eval(out.logits)
    return (time.perf_counter() - start) / iters * 1000


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/inkling-2-mini/mlx-3bit")
    p.add_argument("--lengths", nargs="+", type=int, default=[1, 2, 3, 4, 6, 8])
    p.add_argument("--accept", action="store_true")
    p.add_argument("--fast", action="store_true")
    p.add_argument("--block", type=int, default=None)
    args = p.parse_args()
    model, processor = load(args.model)
    if args.fast:
        from inkling_fast import decode as fastdecode
        fastdecode.install(model)
    lm = model.language_model
    cache = lm.make_cache()
    prefix = mx.array([[200 + i for i in range(256)]])
    mx.eval(lm(prefix, cache=cache).logits)
    for n in args.lengths:
        ms = time_forward(lm, cache, n)
        print(f"forward L={n}: {ms:7.2f} ms  ({ms / n:6.2f} ms/token)", flush=True)

    if args.accept:
        draft, kind = load_drafter(args.model)
        if args.block:
            draft.config.runtime_block_size = args.block
        prompt = apply_chat_template(
            processor,
            model.config,
            "Write a Python function that merges overlapping intervals.",
            reasoning_effort="none",
        )
        for _ in stream_generate(
            model, processor, prompt, max_tokens=160, temperature=0.0,
            draft_model=draft, draft_kind=kind,
        ):
            pass
        acc = draft.accept_lens
        print("block", draft.config.runtime_block_size, "steps", len(acc),
              "mean accepted", sum(acc) / max(len(acc), 1))


if __name__ == "__main__":
    main()
