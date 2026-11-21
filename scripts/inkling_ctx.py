"""Decode ms/token at several context lengths (compiled path)."""
import time

import mlx.core as mx
from mlx_vlm import load

from inkling_fast import decode as D

model, _ = load("weights/inkling-2-mini/mlx-3bit")
D.install(model)
lm = model.language_model
for ctx in (256, 1500, 4000):
    cache = lm.make_cache()
    ids = mx.array([[200 + (i * 31) % 50000 for i in range(ctx)]])
    for i in range(0, ctx, 512):
        mx.eval(lm(ids[:, i:i + 512], cache=cache).logits)
    tok = mx.array([[777]])
    for _ in range(5): mx.eval(lm(tok, cache=cache).logits)
    s = time.perf_counter()
    for _ in range(30):
        out = lm(tok, cache=cache).logits
        tok = mx.argmax(out[:, -1], -1)[:, None]; mx.async_eval(tok)
    mx.eval(tok)
    ms = (time.perf_counter() - s) / 30 * 1000
    print(f"context {ctx}: {ms:.2f} ms/token ({1000/ms:.2f} tok/s)", flush=True)
    del cache; mx.clear_cache()
