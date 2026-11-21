"""Time to first token through an OpenAI-compatible server for a ~N-token chat prompt built from
repo docs. A random nonce leads the message so APC prefix caching cannot hit."""
import argparse
import json
import random
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8766")
    p.add_argument("--chars", type=int, default=7600, help="~2k tokens of doc text")
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--model", default="weights/inkling-2-mini/mlx-3bit")
    p.add_argument("--max-tokens", type=int, default=8)
    args = p.parse_args()
    text = "\n\n".join(f.read_text() for f in sorted((ROOT / "docs").rglob("*.md")) if "inkling-ane" not in f.name)
    for r in range(args.reps):
        nonce = f"Request {random.randrange(10**9)}."
        msg = f"{nonce} Summarize the document below in five bullet points.\n\n---\n{text[:args.chars]}\n---"
        body = {"model": args.model, "messages": [{"role": "user", "content": msg}], "max_tokens": args.max_tokens,
                "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
        req = urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t, first, usage = time.perf_counter(), None, None
        with urllib.request.urlopen(req, timeout=900) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                d = json.loads(line[5:])
                usage = d.get("usage") or usage
                if first is None and any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content")
                                         or c.get("delta", {}).get("reasoning") for c in d.get("choices", [])):
                    first = time.perf_counter()
        print(f"run {r}: prompt_tokens {(usage or {}).get('prompt_tokens')}  TTFT {first - t:.2f}s", flush=True)


if __name__ == "__main__":
    main()
