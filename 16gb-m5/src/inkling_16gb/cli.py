"""inkling-16gb: Inkling 2 Mini (LookAtMyQuant 8.9 GB + DSpark drafter) on a 16 GB Apple silicon Mac.

  inkling-16gb download             # fetch the 8.87 GB model + 179 MB drafter + 637 MB vision tower into ~/.cache/inkling-16gb
  inkling-16gb run "Explain RoPE"   # generate from a prompt
  inkling-16gb chat                 # interactive chat
  inkling-16gb serve --port 8090    # OpenAI-compatible server on 127.0.0.1
  inkling-16gb bench                # decode speed, serial vs drafted, + a long-prompt prefill
"""

from __future__ import annotations

import argparse
import sys
import time


def _engine_kwargs(a) -> dict:
    return {"model": a.model, "drafter": a.drafter, "use_drafter": not a.no_drafter, "vision": not a.no_vision,
            "kv_bits": a.kv_bits, "slots": a.slots, "slot_budget_gb": a.slot_budget_gb}


def _print_stream(engine, prompt, a):
    in_thinking = False
    last = None
    for ch, text in engine.run(prompt, a.max_tokens, a.temperature):
        if ch == "done":
            last = text
            break
        if ch == "tool_call":
            print(f"\n[tool call] {text['name']}({text['arguments']})", flush=True)
            continue
        if ch == "reasoning_content" and not in_thinking:
            print("\033[2m[thinking] ", end="")
            in_thinking = True
        elif ch == "content" and in_thinking:
            print("\033[0m\n")
            in_thinking = False
        print(text, end="", flush=True)
    print("\033[0m" if in_thinking else "", flush=True)
    if last is not None:
        tpc = f", {last['tokens_per_check']:.2f} tok/check" if last.get("tokens_per_check") else ""
        print(f"[prefill {last['prompt_tokens']} tok ({last['cached_tokens']} cached) @ {last['prompt_tps']:.0f} tok/s | "
              f"decode {last['completion_tokens']} tok @ {last['generation_tps']:.1f} tok/s{tpc}]", file=sys.stderr)


def cmd_download(a):
    from .engine import download

    m, d = download()
    print(m)
    print(d)


def cmd_run(a):
    from .engine import Engine

    e = Engine(**_engine_kwargs(a), warmup=False, quiet=True)
    content = a.prompt
    if a.image:
        content = [{"type": "image_url", "image_url": {"url": a.image}}, {"type": "text", "text": a.prompt}]
    _print_stream(e, e.encode([{"role": "user", "content": content}], a.effort), a)


def cmd_chat(a):
    from .engine import Engine

    e = Engine(**_engine_kwargs(a))
    messages = []
    print("Inkling 2 Mini (8.9 GB). Empty line or Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        messages.append({"role": "user", "content": text})
        reply, thinking = [], []
        for ch, t in e.run(e.encode(messages, a.effort), a.max_tokens, a.temperature):
            if ch == "done":
                print(f"\n\033[2m[{t['cached_tokens']}/{t['prompt_tokens']} prompt tokens cached | "
                      f"{t['generation_tps']:.1f} tok/s]\033[0m", file=sys.stderr)
                break
            if ch == "tool_call":
                continue
            (reply if ch == "content" else thinking).append(t)
            print(("\033[2m" + t + "\033[0m") if ch == "reasoning_content" else t, end="", flush=True)
        # echo the reasoning back so the next turn reuses the cache with no re-prefill
        messages.append({"role": "assistant", "content": "".join(reply), "reasoning_content": "".join(thinking)})


def cmd_serve(a):
    from . import serve

    serve.main(_engine_kwargs(a), a.port, a.alias)


def cmd_bench(a):
    import mlx.core as mx

    from .engine import Engine

    e = Engine(**_engine_kwargs(a))
    prompts = {
        "prose": "Explain how a refrigerator works, step by step, to someone who has never studied physics.",
        "code": "Write a Python function that merges overlapping intervals in a list of (start, end) pairs, with a docstring and three test cases.",
        "math": "A train leaves at 9:40 and arrives at 13:05. It stops for 12 minutes on the way and covers 246 km. What is its average speed while moving, in km/h? Show each step.",
    }
    def one(prompt, n):
        last = None
        for ch, x in e.run(prompt, max_tokens=n):
            if ch == "done":
                last = x
        return last

    row = []
    for k, p in prompts.items():
        e.reset_cache()
        last = one(e.encode([{"role": "user", "content": p}]), a.max_tokens)
        tpc = f" ({last['tokens_per_check']:.2f}/chk)" if last.get("tokens_per_check") else ""
        row.append(f"{k} {last['generation_tps']:.1f}{tpc}")
    print(f"{'drafted' if e.drafter else 'serial'} decode tok/s: " + " | ".join(row), flush=True)
    filler = " ".join(f"Item {i}: the quick brown fox jumps over the lazy dog." for i in range(400))
    msgs = [{"role": "user", "content": filler + "\nHow many items were listed?"}]
    e.reset_cache()
    mx.reset_peak_memory()
    t = time.time()
    last = one(e.encode(msgs), 8)
    print(f"prefill  {last['prompt_tokens']} tok @ {last['prompt_tps']:.0f} tok/s ({time.time() - t:.1f}s), "
          f"peak GPU memory {mx.get_peak_memory() / 1e9:.2f} GB", flush=True)
    t = time.time()
    last = one(e.encode(msgs + [{"role": "assistant", "content": "400."}, {"role": "user", "content": "Are you sure?"}]), 8)
    print(f"next turn: {last['cached_tokens']}/{last['prompt_tokens']} tok cached ({last['cache']}), "
          f"{time.time() - t:.1f}s to answer", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="inkling-16gb", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default=None, help="local model dir (default ~/.cache/inkling-16gb/inkling-8.9gb)")
        p.add_argument("--drafter", default=None, help="drafter .safetensors (default: the 8.9 GB-distilled one)")
        p.add_argument("--no-drafter", action="store_true", help="plain serial decoding")
        p.add_argument("--no-vision", action="store_true", help="reject image input (never loads the vision tower)")
        p.add_argument("--kv-bits", type=int, default=4, choices=(0, 4, 8),
                       help="global-attention KV cache precision: 4 (default, 262k context fits 16 GB), 8 (~128k), 0 = fp16 (~32k)")
        p.add_argument("--slots", type=int, default=4, help="cached conversations kept (main session + subagents)")
        p.add_argument("--slot-budget-gb", type=float, default=1.5, help="memory allowed for idle cached conversations")
        p.add_argument("--max-tokens", type=int, default=1024)
        p.add_argument("--temperature", type=float, default=0.0)
        p.add_argument("--effort", default="none", help="reasoning effort: none|minimal|low|medium|high|xhigh|max")
        return p

    sub.add_parser("download").set_defaults(fn=cmd_download)
    p = common(sub.add_parser("run"))
    p.add_argument("prompt")
    p.add_argument("--image", default=None, help="image path or URL to attach")
    p.set_defaults(fn=cmd_run)
    common(sub.add_parser("chat")).set_defaults(fn=cmd_chat)
    p = common(sub.add_parser("serve"))
    p.add_argument("--port", type=int, default=8090)
    p.add_argument("--alias", default="Inkling-2-Mini-9.6GB")
    p.set_defaults(fn=cmd_serve)
    p = common(sub.add_parser("bench"))
    p.set_defaults(fn=cmd_bench, max_tokens=384)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
