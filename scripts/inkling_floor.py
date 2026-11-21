"""Practical floor: one decode step's weight matmuls only (real weights), serial chain."""
import time

import mlx.core as mx
from mlx_vlm import load

from inkling_fast import decode as D
from inkling_fast.kernels import message_gates

model, _ = load("weights/inkling-2-mini/mlx-3bit")
dec = D.install(model)
lm = model.language_model
def step(x, parts):
    for p in dec.passes:
        attn = p.layer.self_attn
        if "proj" in parts:
            raw = p.packed.matmul(x)
            x = (x + raw[..., :2304] * 1e-3).astype(mx.bfloat16)
            x = (x + attn.o_proj(raw[..., 2048:2048 + 3072]) * 1e-3).astype(mx.bfloat16)
        if "moe" in parts:
            x = (x + p.layer.mlp(x) * 1e-3).astype(mx.bfloat16)
        if "gates" in parts and p.loop_index is not None:
            h = mx.concatenate([x.reshape(1, -1)] * 3).astype(mx.float32)
            g = message_gates(h, h, *p.qkv_gates, 1e-6)
            x = (x + g[:1].reshape(x.shape) * 1e-3).astype(mx.bfloat16)
    if "head" in parts:
        x = (x + lm._logits_from_norm(x)[..., :2304] * 1e-3).astype(mx.bfloat16)
    return x
for parts in (("moe",), ("proj",), ("proj", "moe"), ("proj", "moe", "gates", "head")):
    x = (mx.random.normal((1, 1, 2304)) * 0.1).astype(mx.bfloat16)
    for _ in range(3): mx.eval(step(x, parts))
    s = time.perf_counter()
    for _ in range(20): x = step(x, parts); mx.eval(x)
    print(f"{'+'.join(parts):24s} {(time.perf_counter()-s)/20*1000:6.2f} ms/step", flush=True)
