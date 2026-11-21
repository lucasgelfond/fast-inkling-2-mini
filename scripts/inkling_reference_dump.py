"""Dump reference outputs of the MLX implementation for cross-engine validation.

Writes benchmarks/inkling/reference/: tokens.json (prompt ids), greedy.json
(16 greedy continuation ids), last_logits.npy (fp32 logits at the last prompt
position), and per-layer-pass hidden states for the prompt (hidden_step{k}.npy,
the input to layer application k, k = 0..47, plus final_norm.npy).
"""
import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm import load

import inkling_fast  # noqa: F401

out = Path("benchmarks/inkling/reference")
out.mkdir(parents=True, exist_ok=True)
model, processor = load("weights/inkling-2-mini/mlx-3bit")
lm = model.language_model
tokens = [200 + (i * 37) % 5000 for i in range(48)]
ids = mx.array([tokens])
total = lm.config.total_num_hidden_layers
cache = lm.make_cache()
res = lm(ids, cache=cache, capture_layer_ids=list(range(total + 1)))
logits = res.logits[0, -1].astype(mx.float32)
for k, h in enumerate(res.hidden_states):
    name = f"hidden_step{k}.npy" if k < total else "final_norm.npy"
    np.save(out / name, np.array(h[0].astype(mx.float32)))
np.save(out / "last_logits.npy", np.array(logits))
greedy, tok = [], mx.argmax(logits)
for _ in range(16):
    greedy.append(tok.item())
    tok = mx.argmax(lm(tok.reshape(1, 1), cache=cache).logits[0, -1])
(out / "tokens.json").write_text(json.dumps(tokens))
(out / "greedy.json").write_text(json.dumps(greedy))
(out / "README.md").write_text(
    "Reference: weights/inkling-2-mini/mlx-3bit via upstream mlx-vlm 0.8.0 + inkling_fast loop module\n"
    "(reference path, no fast patches). hidden_step{k} = input to layer application k (schedule order:\n"
    "layers 0-5, loop layers 6-17 x3 passes, layers 18-23). final_norm = normalized final hidden.\n"
    "Shapes [48, 2304] fp32. Greedy = argmax decode after the 48-token prompt.\n"
)
print("greedy:", greedy)
