"""OpenAI-compatible localhost server for Inkling 2 Mini on the inkling_fast engine.

Wraps mlx-vlm's server: the model is loaded as usual, then the compiled decode
path is installed. Exact automatic prefix caching (APC) is on by default, so
repeated prompt prefixes (agent system prompts, chat history) skip prefill.

  PYTHONPATH=src .venv/bin/python scripts/inkling_serve.py \
      --model weights/inkling-2-mini/mlx-3bit --port 8766

INKLING_ANE_PREFILL=1 routes fresh prompts of >= 1024 tokens (up to the largest slice bucket,
4096) through the ANE+GPU hybrid prefill (src/inkling_ane, docs/inkling-ane-prefill.md).
It needs coremltools (pip install -e '.[ane]'). Slices load in the background; until then, and for short prompts, APC
prefix hits and any ANE error, prefill stays on the GPU.
"""

import os
import sys

os.environ.setdefault("APC_ENABLED", "1")
os.environ.setdefault("APC_DISK_MAX_GB", "4")  # disk is tight on this Mac
ANE_PREFILL = os.environ.get("INKLING_ANE_PREFILL") == "1"
if ANE_PREFILL:
    # One prefill call must carry the whole prompt (minus its last token) for the hybrid to take it.
    os.environ.setdefault("PREFILL_STEP_SIZE", "4096")

import mlx_vlm.server.generation as generation  # noqa: E402

import inkling_fast  # noqa: E402,F401  registers inkling2_loop
from inkling_fast import decode  # noqa: E402

_load = generation.load


def _is_oq(path):
    import json
    from pathlib import Path

    cfg = Path(path) / "config.json"
    if not cfg.exists():
        return False
    quant = json.loads(cfg.read_text()).get("quantization") or {}
    return any(".attn.in_proj" in k or ".mlp.experts.w13" in k for k in quant)


def _load_fast(*args, **kwargs):
    path = args[0] if args else kwargs.get("model_path")
    if path is not None and _is_oq(path):
        # oMLX "oQ" checkpoint (e.g. TML's native NVFP4 experts): our own loader.
        from inkling_fast.oq import load_oq

        model, processor = load_oq(path)
    else:
        model, processor = _load(*args, **kwargs)
    if getattr(model.config, "model_type", None) == "inkling2_loop":
        decode.install(model)
        if ANE_PREFILL:
            from inkling_ane import hybrid

            hybrid.install(model, log=lambda m: print(m, file=sys.stderr, flush=True))
    return model, processor


generation.load = _load_fast

# Model aliases: INKLING_MODEL_ALIAS="Inkling-2-Mini=/abs/path/to/weights" lets clients
# use a clean model id while the server loads the local weights directory.
_ALIASES = dict(
    item.split("=", 1) for item in os.environ.get("INKLING_MODEL_ALIAS", "").split(",") if "=" in item
)
if _ALIASES:
    import mlx_vlm.server as _server_pkg  # routes resolve get_cached_model on the package

    _app_mod = sys.modules["mlx_vlm.server.app"]  # the module (the package exports a FastAPI `app`)
    _get_cached_model = _server_pkg.get_cached_model

    def _aliased_get_cached_model(model_path, *args, **kwargs):
        return _get_cached_model(_ALIASES.get(model_path, model_path), *args, **kwargs)

    _server_pkg.get_cached_model = _aliased_get_cached_model
    _app_mod.get_cached_model = _aliased_get_cached_model

if "--host" not in sys.argv:
    sys.argv += ["--host", "127.0.0.1"]
if "--prefill-step-size" not in sys.argv and not ANE_PREFILL:
    # 512-token chunks are the fastest GPU prefill size measured, and they take the
    # compiled path. (With the ANE hybrid, PREFILL_STEP_SIZE=4096 above must win.)
    sys.argv += ["--prefill-step-size", "512"]

from mlx_vlm.server.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
