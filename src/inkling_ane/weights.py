"""Dequantized fp32 numpy weights for single Inkling 2 Mini layers (mlx only, no mlx_vlm)."""

import json
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights/inkling-2-mini/mlx-3bit"
PREFIX = "language_model.model."


def config():
    c = json.loads((WEIGHTS / "config.json").read_text())
    return c["text_config"], c["quantization"]


def layer_weights(prefix, skip=("mlp.switch_mlp",)):
    """{name: fp32 array} for keys under PREFIX + prefix; quantized linears dequantized."""
    _, quant = config()
    index = json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"]
    keys = [k for k in index if k.startswith(PREFIX + prefix) and not any(s in k for s in skip)]
    raw = {}
    for f in sorted({index[k] for k in keys}):
        arrays = mx.load(str(WEIGHTS / f))
        raw.update({k: arrays[k] for k in keys if index[k] == f})
    out = {}
    for k, v in raw.items():
        name = k[len(PREFIX) + len(prefix):]
        if name.endswith(".scales") or name.endswith(".biases"):
            continue
        if name.endswith(".weight") and (k[:-7] + ".scales") in raw:
            q = quant.get(k[:-7], {"group_size": quant["group_size"], "bits": quant["bits"]})
            v = mx.dequantize(v, raw[k[:-7] + ".scales"], raw[k[:-7] + ".biases"],
                              group_size=q["group_size"], bits=q["bits"])
        out[name] = np.array(v.astype(mx.float32))
    return out
