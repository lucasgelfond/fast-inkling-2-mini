"""Measure streaming decode tok/s of any OpenAI-compatible local server.

Decode rate = (completion tokens - 1) / (last token time - first token time),
so prefill and connection time are excluded. Uses the same prompts as
scripts/inkling_bench.py.
"""

import argparse
import json
import time
import urllib.request

PROMPTS = {
    "code": "Write a Python function that parses an RFC 3339 timestamp string into a "
    "timezone-aware datetime without using dateutil. Include type hints and a docstring.",
    "chat": "Explain to a new engineer what a hash table does, in about 150 words.",
}


def run(url, model, prompt, max_tokens, extra):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
        **extra,
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    start = time.perf_counter()
    first = last = None
    chunks = 0
    usage = None
    with urllib.request.urlopen(req, timeout=600) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            data = json.loads(line[5:])
            if data.get("usage"):
                usage = data["usage"]
            for choice in data.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                    now = time.perf_counter()
                    first = first or now
                    last = now
                    chunks += 1
    tokens = (usage or {}).get("completion_tokens", chunks)
    return {
        "completion_tokens": tokens,
        "ttft_s": round(first - start, 2) if first else None,
        "decode_tps": round((tokens - 1) / (last - first), 2) if first and last > first else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:8765")
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--extra", default="{}", help="JSON merged into the request body")
    p.add_argument("--label", default="")
    a = p.parse_args()
    extra = json.loads(a.extra)
    run(a.url, a.model, "Hi", 8, extra)  # warm up
    for name, prompt in PROMPTS.items():
        r = run(a.url, a.model, prompt, a.max_tokens, extra)
        print(f"{a.label:12s} {name:5s} decode {r['decode_tps']} tok/s  ttft {r['ttft_s']} s  {r['completion_tokens']} tok", flush=True)


if __name__ == "__main__":
    main()
