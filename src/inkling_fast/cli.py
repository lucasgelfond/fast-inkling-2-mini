"""inkling-fast: download, run, serve and benchmark Inkling 2 Mini (3-bit MLX) on Apple silicon.

  inkling-fast download            # fetch the 3-bit weights (14.7 GB) into the Hugging Face cache
  inkling-fast run "Explain RoPE"  # generate from a prompt
  inkling-fast chat                # interactive chat
  inkling-fast serve --port 8080   # OpenAI-compatible server on 127.0.0.1
  inkling-fast bench               # prefill (512 tokens) and decode (128 tokens) speed
"""

from __future__ import annotations

import argparse
import os
import re
import sys
import time

REPO = "netprojects/Inkling-2-Mini-MLX-3bit"
REVISION = "af13f5abb6ef15d9b3fe96721af6845e9a4cf27e"
# The DSpark draft head (mtp.safetensors) is not used: speculation does not pay off here.
IGNORE = ["mtp.safetensors"]
_SPECIAL = re.compile(r"<\|[a-z_]+\|>")


def _clean(text: str) -> str:
    text = text.replace("<|content_thinking|>", "[thinking]\n")
    return _SPECIAL.sub("\n", text) if "<|" in text else text


def weights_path(model: str | None) -> str:
    if model:
        return model
    from huggingface_hub import snapshot_download

    return snapshot_download(REPO, revision=REVISION, ignore_patterns=IGNORE)


def load(model: str | None, fast: bool = True):
    from mlx_vlm import load as mlx_load

    import inkling_fast  # noqa: F401  registers inkling2_loop

    model_obj, processor = mlx_load(weights_path(model))
    if fast:
        from inkling_fast import decode

        decode.install(model_obj)
    return model_obj, processor


def _generate(model, processor, messages, max_tokens, temperature):
    from mlx_vlm import apply_chat_template, stream_generate

    prompt = apply_chat_template(processor, model.config, messages)
    last, parts = None, []
    for chunk in stream_generate(model, processor, prompt, max_tokens=max_tokens, temperature=temperature):
        print(_clean(chunk.text), end="", flush=True)
        parts.append(chunk.text)
        last = chunk
    print()
    return last, "".join(parts)


def _stats(last):
    if last is not None:
        print(
            f"\n[prefill {last.prompt_tokens} tok @ {last.prompt_tps:.1f} tok/s | "
            f"decode {last.generation_tokens} tok @ {last.generation_tps:.1f} tok/s]",
            file=sys.stderr,
        )


def cmd_download(a):
    print(weights_path(a.model))


def cmd_run(a):
    model, processor = load(a.model)
    last, _ = _generate(model, processor, [{"role": "user", "content": a.prompt}], a.max_tokens, a.temperature)
    _stats(last)


def cmd_chat(a):
    model, processor = load(a.model)
    messages = []
    print("Inkling 2 Mini. Empty line or Ctrl-D to quit.", file=sys.stderr)
    while True:
        try:
            text = input("> ").strip()
        except EOFError:
            break
        if not text:
            break
        messages.append({"role": "user", "content": text})
        last, reply = _generate(model, processor, messages, a.max_tokens, a.temperature)
        messages.append({"role": "assistant", "content": reply})
        _stats(last)


def cmd_serve(a):
    from . import serve

    serve.main(weights_path(a.model), a.port, a.alias, a.streams, a.sessions, a.cache_gb)


def cmd_bench(a):
    import mlx.core as mx

    model, _ = load(a.model, fast=not a.baseline)
    lm = model.language_model
    n = 512
    prompt = mx.array([[200 + (i * 37) % 50000 for i in range(n)]])

    def prefill():
        cache = lm.make_cache()
        s = time.perf_counter()
        out = lm(prompt, cache=cache).logits
        mx.eval(out)
        return n / (time.perf_counter() - s), cache, out

    def decode(cache, out, steps=128):
        tok = mx.argmax(out[:, -1], -1)[:, None]
        s = time.perf_counter()
        for _ in range(steps):
            out = lm(tok, cache=cache).logits
            tok = mx.argmax(out[:, -1], -1)[:, None]
            mx.async_eval(tok)
        mx.eval(tok)
        return steps / (time.perf_counter() - s)

    prefill()  # warm up kernels and compile
    pp, tg = [], []
    for _ in range(3):
        p, cache, out = prefill()
        pp.append(p)
        tg.append(decode(cache, out))
        del cache
        mx.clear_cache()
    engine = "standard mlx-vlm" if a.baseline else "inkling-fast"
    print(f"{engine}: prefill {max(pp):.1f} tok/s (512 tokens), decode {max(tg):.1f} tok/s (128 tokens)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="inkling-fast", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.environ.get("INKLING_MODEL"), help="local weights dir (default: download 3-bit)")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("download").set_defaults(fn=cmd_download)
    r = sub.add_parser("run")
    r.add_argument("prompt")
    c = sub.add_parser("chat")
    for s in (r, c):
        s.add_argument("--max-tokens", type=int, default=2048)
        s.add_argument("--temperature", type=float, default=0.0)
    r.set_defaults(fn=cmd_run)
    c.set_defaults(fn=cmd_chat)
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8080)
    s.add_argument("--alias", default="Inkling-2-Mini", help="model id clients send")
    s.add_argument("--streams", type=int, default=4, help="requests decoded together in one batch")
    s.add_argument("--sessions", type=int, default=4, help="conversations kept in the prefix cache")
    s.add_argument("--cache-gb", type=float, default=6.0, help="memory for prefix-cache snapshots")
    s.set_defaults(fn=cmd_serve)
    b = sub.add_parser("bench")
    b.add_argument("--baseline", action="store_true", help="measure standard mlx-vlm instead")
    b.set_defaults(fn=cmd_bench)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
