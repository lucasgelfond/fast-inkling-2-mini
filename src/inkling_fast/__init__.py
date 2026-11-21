"""Fast inference for Inkling 2 Mini (inkling2_loop) on MLX.

The loop model comes from internet-projects/inkling2-mlx v0.1.0 (an mlx-vlm
fork that ships `mlx_vlm.models.inkling2_loop`). Importing this package
registers the loop with mlx-vlm's prompt and DSpark drafter code.
"""

import sys

import mlx_vlm.models
import mlx_vlm.prompt_utils as prompt_utils
import mlx_vlm.speculative.drafters.inkling2_dspark.config as dspark_config
import mlx_vlm.speculative.drafters.inkling2_dspark.dspark as dspark
import mlx_vlm.speculative.drafters.mtp_split as mtp_split

try:
    from mlx_vlm.models import inkling2_loop
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "inkling_fast needs mlx-vlm with the inkling2_loop model: "
        "pip install 'mlx-vlm @ git+https://github.com/internet-projects/inkling2-mlx@v0.1.0'"
    ) from e

_SPLITTER = "mlx_vlm.speculative.drafters.inkling2_dspark.split:Inkling2DSparkSplitter"


def effective_num_layers(text: dict) -> int:
    """Decoder layer applications per token (the loop repeats its core)."""
    loop = text.get("loop_config") or {}
    if not loop:
        return text["num_hidden_layers"]
    span = loop["core_end"] - loop["core_start"]
    return text["num_hidden_layers"] + (loop["num_passes"] - 1) * span


def _register() -> None:
    sys.modules["mlx_vlm.models.inkling2_loop"] = inkling2_loop
    mlx_vlm.models.inkling2_loop = inkling2_loop
    prompt_utils.MODEL_CONFIG["inkling2_loop"] = prompt_utils.MODEL_CONFIG["inkling2"]
    for key in ("inkling2_loop", "inkling2_loop_text"):
        mtp_split.MTP_SPLITTERS[key] = _SPLITTER

    # The DSpark head taps layer applications, so its last target id counts
    # every loop pass (48), not the 24 stored layers.
    original_post_init = dspark_config.ModelConfig.__post_init__

    def post_init(self):
        text = self.text_config
        text = dict(text) if isinstance(text, dict) else text.to_dict()
        loop = text.get("loop_config")
        if loop:
            stored = text["num_hidden_layers"]
            text["num_hidden_layers"] = effective_num_layers(text)
            self.text_config = text
            original_post_init(self)
            self.text_config.num_hidden_layers = stored
            self.text_config.loop_config = loop
        else:
            original_post_init(self)

    dspark_config.ModelConfig.__post_init__ = post_init

    def validate_target_compatibility(self, target_model):
        target = getattr(target_model, "language_model", target_model)
        args = getattr(target, "config", None)
        if (
            args is None
            or args.model_type not in ("inkling2_text", "inkling2_loop_text")
            or any(
                getattr(args, name) != getattr(self.config.text_config, name)
                for name in ("hidden_size", "vocab_size", "num_hidden_layers")
            )
        ):
            raise ValueError("The DSpark head requires its matching Inkling 2 target")

    dspark.Model.validate_target_compatibility = (
        validate_target_compatibility
    )


_register()
