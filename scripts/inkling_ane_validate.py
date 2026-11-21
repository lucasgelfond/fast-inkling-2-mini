"""Hybrid (ANE+GPU) vs GPU prefill on Inkling 2 Mini, with the inkling_fast decode engine installed.

validate: ~10 chat-formatted prompts built from repo docs (1k-2k tokens); greedy-decode 64 tokens
          after each prefill; report agreement and first-divergence position.
bench:    prefill tok/s at the given lengths, best of N, for both paths.
Both paths mirror mlx-vlm generate: prefill tokens[:-1], then the last token through lm() (decode engine).

Run under the GPU lock (loads the 14 GB model):
  scripts/with-gpu-lock ane-validate env PYTHONPATH=src .venv/bin/python \
      scripts/inkling_ane_validate.py validate bench
"""
import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load
from mlx_vlm.prompt_utils import apply_chat_template

import inkling_fast  # noqa: F401
from inkling_ane import hybrid
from inkling_fast import decode

ROOT = Path(__file__).resolve().parents[1]


def doc_prompts(processor, config, n=10, lo=1024, hi=2048):
    files = sorted(p for p in (ROOT / "docs").rglob("*.md") if p.stat().st_size > 3000)
    files = [f for f in files if "inkling-ane-prefill" not in f.name]
    tok = processor.tokenizer
    asks = ["Summarize the document below in five bullet points.",
            "List the main decisions recorded in this document and explain each briefly.",
            "What are the open problems described here? Answer concisely."]
    out = []
    for k in range(n):
        target = lo + (hi - lo) * k // max(n - 1, 1)
        text = "\n\n".join(f.read_text() for f in files[k % len(files):] + files[:k % len(files)])
        ask = asks[k % len(asks)]
        # binary search the text length that hits the target token count
        a, b = 200, len(text)
        best = None
        for _ in range(18):
            m = (a + b) // 2
            prompt = apply_chat_template(processor, config, f"{ask}\n\n---\n{text[:m]}\n---", reasoning_effort="none")
            ids = tok.encode(prompt, add_special_tokens=False) if hasattr(tok, "encode") else tok(prompt)["input_ids"]
            best = ids
            if len(ids) > target:
                b = m - 1
            elif len(ids) < target - 8:
                a = m + 1
            else:
                break
        out.append(best)
    return out


def gpu_prefill(lm, ids, step=2048):
    cache = lm.make_cache()
    body = ids[:-1]
    for s in range(0, len(body), step):
        lm(mx.array([body[s:s + step]]), cache=cache, skip_logits=True)
        mx.eval([c.state for c in cache])
    logits = lm(mx.array([ids[-1:]]), cache=cache).logits
    mx.eval(logits)
    return cache, logits


def hybrid_prefill(lm, hp, ids):
    cache = lm.make_cache()
    try:
        hp.run(lm.model.embed(mx.array([ids[:-1]])), cache)
    except RuntimeError as e:  # transient ANE error: report and retry once
        print(f"   ANE error, retrying: {str(e)[:100]}", flush=True)
        cache = lm.make_cache()
        hp.run(lm.model.embed(mx.array([ids[:-1]])), cache)
    logits = lm(mx.array([ids[-1:]]), cache=cache).logits
    mx.eval(logits)
    return cache, logits


def greedy(lm, cache, logits, n):
    toks = []
    tok = mx.argmax(logits[0, -1])
    for _ in range(n):
        toks.append(int(tok.item()))
        logits = lm(tok.reshape(1, 1), cache=cache).logits
        tok = mx.argmax(logits[0, -1])
    return toks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("modes", nargs="+")
    p.add_argument("--model", default=str(ROOT / "weights/inkling-2-mini/mlx-3bit"))
    p.add_argument("--n-prompts", type=int, default=10)
    p.add_argument("--lo", type=int, default=1024)
    p.add_argument("--hi", type=int, default=2048)
    p.add_argument("--gen", type=int, default=64)
    p.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048])
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--out", default=None)
    args = p.parse_args()
    t = time.perf_counter()
    model, processor = load(args.model)
    decode.install(model)
    lm = model.language_model
    hp = hybrid.get(model)
    print(f"loaded model + slices in {time.perf_counter() - t:.1f}s (slices {hp.load_s:.1f}s)", flush=True)
    results = {}
    if "validate" in args.modes:
        prompts = doc_prompts(processor, model.config, args.n_prompts, args.lo, args.hi)
        rows = []
        for k, ids in enumerate(prompts):
            cg, lg = gpu_prefill(lm, ids)
            g = greedy(lm, cg, lg, args.gen)
            del cg
            c5, l5 = gpu_prefill(lm, ids, step=512)  # calibration: GPU vs GPU with other chunking
            g5 = greedy(lm, c5, l5, args.gen)
            del c5
            first5 = next((j for j, (a, b) in enumerate(zip(g, g5)) if a != b), None)
            ch, lh = hybrid_prefill(lm, hp, ids)
            h = greedy(lm, ch, lh, args.gen)
            del ch
            pg = np.array(lg[0, -1].astype(mx.float32)); ph = np.array(lh[0, -1].astype(mx.float32))
            sm = lambda v: np.exp(v - v.max()) / np.exp(v - v.max()).sum()
            kl = float((sm(pg) * (np.log(sm(pg) + 1e-30) - np.log(sm(ph) + 1e-30))).sum())
            agree = sum(a == b for a, b in zip(g, h))
            first = next((j for j, (a, b) in enumerate(zip(g, h)) if a != b), None)
            rows.append(dict(tokens=len(ids), agree=agree, first_divergence=first, kl_first=kl,
                             gpu512_agree=sum(a == b for a, b in zip(g, g5)), gpu512_first_divergence=first5))
            print(f"prompt {k}: {len(ids)} tok  hybrid agree {agree}/{args.gen} first div "
                  f"{first if first is not None else '-'}  KL(first logits) {kl:.1e} | GPU step512 vs 2048: agree "
                  f"{rows[-1]['gpu512_agree']}/{args.gen} first div {first5 if first5 is not None else '-'}", flush=True)
            if first is not None:
                tk = processor.tokenizer
                print(f"   gpu:    {tk.decode(g[max(0, first - 3):first + 6])!r}\n   hybrid: {tk.decode(h[max(0, first - 3):first + 6])!r}", flush=True)
        print(f"GPU step512 vs step2048: identical {sum(r['gpu512_first_divergence'] is None for r in rows)}/{len(rows)}, "
              f"mean agreement {np.mean([r['gpu512_agree'] for r in rows]):.1f}/{args.gen}", flush=True)
        same = sum(r["first_divergence"] is None for r in rows)
        print(f"identical continuations: {same}/{len(rows)}; mean agreement "
              f"{np.mean([r['agree'] for r in rows]):.1f}/{args.gen}", flush=True)
        results["validate"] = rows
    if "bench" in args.modes:
        rng = np.random.default_rng(0)
        rows = []
        for n in args.lengths:
            ids = [int(v) for v in rng.integers(200, 100000, n + 1)]
            for name, fn in (("gpu", lambda: gpu_prefill(lm, ids)), ("hybrid", lambda: hybrid_prefill(lm, hp, ids))):
                fn()  # warm
                ts = []
                for _ in range(args.reps):
                    t = time.perf_counter()
                    c, _ = fn()
                    ts.append(time.perf_counter() - t)
                    del c
                best = min(ts)
                rows.append(dict(path=name, tokens=n, best_s=best, tok_s=n / best, runs=[n / x for x in ts]))
                print(f"prefill {name:6s} N={n:5d}: best {best:.2f}s = {n / best:6.1f} tok/s  "
                      f"runs {[round(n / x, 1) for x in ts]}", flush=True)
                mx.clear_cache()
        results["bench"] = rows
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
