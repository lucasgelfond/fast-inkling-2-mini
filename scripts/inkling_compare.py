"""Standard Inkling 2 Mini benchmark, same method for every engine.

  prefill: tok/s for one 512-token prompt (fresh cache; like llama-bench pp512)
  decode:  tok/s for 128 generated tokens after that prompt (like llama-bench tg128)
Best of 3 runs each. Engines: "mlx" (upstream mlx-vlm + loop module, no patches)
and "fast" (inkling_fast). llama.cpp is run with llama-bench -p 512 -n 128.

  PYTHONPATH=src scripts/with-gpu-lock cmp .venv/bin/python scripts/inkling_compare.py fast
"""

import json
import sys
import time

import mlx.core as mx
from mlx_vlm import load

import inkling_fast  # noqa: F401

engine = sys.argv[1]
import os as _os

WEIGHTS = _os.environ.get("INKLING_WEIGHTS", "weights/inkling-2-mini/mlx-3bit")
if "oq" in WEIGHTS:
    from inkling_fast.oq import load_oq
    model, _ = load_oq(WEIGHTS)
else:
    model, _ = load(WEIGHTS)
if engine == "fast":
    from inkling_fast import decode

    decode.install(model)
lm = model.language_model
prompt = mx.array([[200 + (i * 37) % 50000 for i in range(512)]])


def prefill():
    cache = lm.make_cache()
    s = time.perf_counter()
    out = lm(prompt, cache=cache).logits
    mx.eval(out)
    return 512 / (time.perf_counter() - s), cache, out


def decode_tg(cache, out, n=128):
    tok = mx.argmax(out[:, -1], -1)[:, None]
    s = time.perf_counter()
    for _ in range(n):
        out = lm(tok, cache=cache).logits
        tok = mx.argmax(out[:, -1], -1)[:, None]
        mx.async_eval(tok)
    mx.eval(tok)
    return n / (time.perf_counter() - s)


prefill()  # warm up kernels and compile
pp, tg = [], []
for _ in range(3):
    p, cache, out = prefill()
    pp.append(p)
    tg.append(decode_tg(cache, out))
    del cache
    mx.clear_cache()
res = {"engine": engine, "pp512": round(max(pp), 1), "tg128": round(max(tg), 2)}
print(json.dumps(res))
