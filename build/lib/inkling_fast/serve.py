"""OpenAI-compatible localhost server for Inkling 2 Mini on the inkling_fast engine.

Wraps mlx-vlm's server: the model is loaded as usual, then the compiled decode
path is installed. Exact automatic prefix caching (APC) is on by default, so
repeated prompt prefixes (system prompts, chat history) skip prefill.
"""

from __future__ import annotations

import os
import sys


def main(weights: str, port: int, alias: str = "Inkling-2-Mini") -> None:
    os.environ.setdefault("APC_ENABLED", "1")
    os.environ.setdefault("APC_DISK_MAX_GB", "4")

    import mlx_vlm.server as server_pkg
    import mlx_vlm.server.generation as generation

    import inkling_fast  # noqa: F401  registers inkling2_loop
    from inkling_fast import decode

    original_load = generation.load

    def load_fast(*args, **kwargs):
        model, processor = original_load(*args, **kwargs)
        if getattr(model.config, "model_type", None) == "inkling2_loop":
            decode.install(model)
        return model, processor

    generation.load = load_fast

    # Clients send a clean model id; the server loads the local weights directory.
    aliases = {alias: weights}
    app_mod = sys.modules["mlx_vlm.server.app"]  # the package itself exports a FastAPI `app`
    get_cached_model = server_pkg.get_cached_model

    def aliased(model_path, *args, **kwargs):
        return get_cached_model(aliases.get(model_path, model_path), *args, **kwargs)

    server_pkg.get_cached_model = aliased
    app_mod.get_cached_model = aliased

    from mlx_vlm.server.__main__ import main as server_main

    # 512-token chunks are the fastest GPU prefill size measured, and they take the compiled path.
    sys.argv = [
        "mlx_vlm.server", "--model", weights, "--host", "127.0.0.1", "--port", str(port),
        "--max-num-seqs", "1", "--prefill-step-size", "512",
    ]
    server_main()
