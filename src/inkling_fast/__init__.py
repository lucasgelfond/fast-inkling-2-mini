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
