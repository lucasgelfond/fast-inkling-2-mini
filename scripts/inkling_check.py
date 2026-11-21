"""Check fast patches: logits must match the reference path, then time decode."""
import sys
import time

import mlx.core as mx
from mlx_vlm import load

model, _ = load(sys.argv[1] if len(sys.argv) > 1 else "weights/inkling-2-mini/mlx-3bit")
lm = model.language_model
prompt = mx.array([[200 + (i * 37) % 5000 for i in range(48)]])

def decode(n=24):
    cache = lm.make_cache()
    out = lm(prompt, cache=cache).logits[:, -1]
    toks, logits = [], []
    tok = mx.argmax(out, -1)
    for _ in range(n):
        out = lm(tok[:, None], cache=cache).logits[:, -1]
        logits.append(out); tok = mx.argmax(out, -1); toks.append(tok.item())
    return toks, mx.stack(logits)

def speed(n=64):
    cache = lm.make_cache()
    mx.eval(lm(prompt, cache=cache).logits)
    tok = mx.array([[777]])
    for _ in range(4): mx.eval(lm(tok, cache=cache).logits)
    s = time.perf_counter()
    for _ in range(n):
        out = lm(tok, cache=cache).logits
        tok = mx.argmax(out[:, -1], -1)[:, None]
        mx.async_eval(tok)
    mx.eval(tok)
    return n / (time.perf_counter() - s)

ref_toks, ref_logits = decode(); mx.eval(ref_logits)
base = speed()
print(f"reference: {base:.2f} tok/s")
from inkling_fast import decode as fastdecode

fastdecode.install(model)
print("compiled decoder installed")
toks, logits = decode()
diff = mx.abs(logits - ref_logits).max().item()
print(f"fast: {speed():.2f} tok/s  tokens identical: {toks == ref_toks}  max |dlogit|: {diff:.4g}")
