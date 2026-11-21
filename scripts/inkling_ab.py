"""Steady-state decode ms/token for the compiled path (env flags select features)."""
import sys
import time

import mlx.core as mx
from mlx_vlm import load

from inkling_fast import decode as D

model, _ = load("weights/inkling-2-mini/mlx-3bit")
D.install(model)
lm = model.language_model
cache = lm.make_cache()
mx.eval(lm(mx.array([[300 + i for i in range(16)]]), cache=cache).logits)
tok = mx.array([[777]])
for _ in range(8): mx.eval(lm(tok, cache=cache).logits)
best = 1e9
for rep in range(3):
    s = time.perf_counter()
    for _ in range(40):
        out = lm(tok, cache=cache).logits
        tok = mx.argmax(out[:, -1], -1)[:, None]
        mx.async_eval(tok)
    mx.eval(tok)
    best = min(best, (time.perf_counter() - s) / 40 * 1000)
print(f"{sys.argv[1] if len(sys.argv) > 1 else ''}: {best:.2f} ms/token ({1000/best:.2f} tok/s, best of 3x40)")
