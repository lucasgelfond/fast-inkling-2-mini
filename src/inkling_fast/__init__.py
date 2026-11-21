"""Fast inference for Inkling 2 Mini (inkling2_loop) on MLX.

The loop model comes from internet-projects/inkling2-mlx v0.1.0, an mlx-vlm
fork that ships `mlx_vlm.models.inkling2_loop`. Importing this package
registers it with mlx-vlm.
"""

import sys

import mlx_vlm.models
import mlx_vlm.prompt_utils as prompt_utils

try:
    from mlx_vlm.models import inkling2_loop
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "inkling_fast needs mlx-vlm with the inkling2_loop model: "
        "pip install 'mlx-vlm @ git+https://github.com/internet-projects/inkling2-mlx@v0.1.0'"
    ) from e

sys.modules["mlx_vlm.models.inkling2_loop"] = inkling2_loop
mlx_vlm.models.inkling2_loop = inkling2_loop
prompt_utils.MODEL_CONFIG["inkling2_loop"] = prompt_utils.MODEL_CONFIG["inkling2"]

# netprojects/Inkling-2-Mini-MLX-3bit stores the loop controllers and message queries
# under mlx-vlm names that the fork's sanitize() does not map. Rename them first.
_GATES = {"attn": "q", "k": "k", "v": "v", "mlp": "mlp"}
_GATE_WEIGHTS = {"down_weight": "down_proj", "forget_weight": "forget_gate_proj", "gain_weight": "message_gate_proj"}
_LAYERS = "language_model.model.layers."
_sanitize = inkling2_loop.Model.sanitize


def _sanitize_mlx_names(self, weights):
    out = {}
    for key, value in weights.items():
        if key.startswith(_LAYERS) and ".controllers." in key:
            layer, _, gate, weight = key[len(_LAYERS) :].split(".")
            key = f"{_LAYERS}{layer}.{_GATES[gate]}_message_gate.{_GATE_WEIGHTS[weight]}.weight"
        elif key == "language_model.model.message_queries":
            key = "language_model.model.message_input.query"
        out[key] = value
    return _sanitize(self, out)


inkling2_loop.Model.sanitize = _sanitize_mlx_names
