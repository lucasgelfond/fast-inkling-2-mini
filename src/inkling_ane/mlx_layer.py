"""Build single Inkling 2 Mini decoder layers in MLX with real weights, without
loading the whole 14 GB model. Needs mlx_vlm + inkling_fast on the path."""

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.inkling2.language import DecoderLayer

import inkling_fast  # noqa: F401  registers inkling2_loop
from mlx_vlm.models.inkling2_loop.config import TextConfig
from mlx_vlm.models.inkling2_loop.language import LoopDecoderLayer, MessageAttentionResidual

ROOT = Path(__file__).resolve().parents[2]
WEIGHTS = ROOT / "weights/inkling-2-mini/mlx-3bit"
PREFIX = "language_model.model."


def load_config():
    cfg = json.loads((WEIGHTS / "config.json").read_text())
    return TextConfig.from_dict(cfg["text_config"]), cfg["quantization"]


def tensors(prefix):
    """Lazily load every tensor whose key starts with PREFIX + prefix."""
    index = json.loads((WEIGHTS / "model.safetensors.index.json").read_text())["weight_map"]
    want = {k: f for k, f in index.items() if k.startswith(PREFIX + prefix)}
    out = {}
    for f in sorted(set(want.values())):
        arrays = mx.load(str(WEIGHTS / f))
        out.update({k[len(PREFIX) + len(prefix):]: arrays[k] for k in want if want[k] == f})
        del arrays
    return out


def build_layer(i, config=None, quant=None):
    if config is None:
        config, quant = load_config()
    start, end = config.loop_layer_index_start, config.loop_layer_index_end
    layer = (LoopDecoderLayer if start <= i < end else DecoderLayer)(config, i)
    w = tensors(f"layers.{i}.")

    def pred(path, module):
        key = f"{PREFIX}layers.{i}.{path}"
        if key in quant:
            return quant[key]
        return f"{path}.scales" in w and {"group_size": quant["group_size"], "bits": quant["bits"]}

    nn.quantize(layer, class_predicate=pred)
    layer.load_weights(list(w.items()), strict=True)
    layer.eval()
    mx.eval(layer.parameters())
    return layer


def build_message_input(config=None):
    if config is None:
        config, _ = load_config()
    m = MessageAttentionResidual(config)
    m.load_weights(list(tensors("message_input.").items()))
    mx.eval(m.parameters())
    return m
