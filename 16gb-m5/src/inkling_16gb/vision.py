"""Standalone MLX image encoder for Inkling 2 Mini (vision tower + projector).

Ported from the Transformers ``Inkling2VisionModel`` / ``Inkling2ImageProcessor``
and the mlx-vlm fork (internet-projects/inkling2-mlx v0.1.0,
``mlx_vlm/models/inkling2/{vision,processing_inkling2}.py``). Only needs mlx,
numpy and Pillow.

Pipeline:
  image -> RGB (RGBA composited on a gray checkerboard) -> pad to a multiple of
  40 px with value -1 (on the 0..255 scale) -> x/255, (x-0.5)/0.5 -> one
  40x40x3 tile per LLM token (raster order) -> ViT (20 px patches, 2x2 patches
  per tile, axial 2D RoPE, CLS token, learned 32x32 pos-emb bilinearly resized)
  -> concat the 2x2 patch features of each tile -> Linear(4096,4096) -> GELU ->
  Linear(4096,2304) -> one row per ``<|unused_200054|>`` token.

The rows replace the LM's *post-embed_norm* embedding rows (the packed runtime's
``PackedEmbed`` output already includes embed_norm). Nothing else changes in
the LM: plain 1D positions, plain causal attention.
"""

from __future__ import annotations

import base64
import io
import json
import math
import os
import struct
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence, Union

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ----------------------------------------------------------------------------
# Constants

HF_REPO = "thinkingmachines/Inkling-2-Mini"
HF_REVISION = "b85c3cfa615fc413cc9af3379f491b963f3b586a"  # pinned
HF_SHARD = "model-00001-of-00001.safetensors"
WEIGHT_PREFIX = "model.visual."
DEFAULT_PATH = Path(os.path.expanduser("~/.cache/inkling-16gb/vision"))
WEIGHTS_FILE = "vision.safetensors"
META_FILE = "vision_meta.json"

IMAGE_TOKEN = "<|unused_200054|>"
IMAGE_TOKEN_ID = 200054
CONTENT_IMAGE_ID = 200005  # <|content_image|>
END_MESSAGE_ID = 200010  # <|end_message|>

TILE = 40  # pixels per LLM token side (= native_patch_size * adapter_merge_size)

VISION_CONFIG = {
    # From thinkingmachines/Inkling-2-Mini config.json "vision_config", plus the
    # Transformers Inkling2VisionConfig defaults that the hub config omits.
    "text_hidden_size": 2304,
    "native_patch_size": 20,
    "adapter_merge_size": 2,
    "projection_hidden_dim": 4096,
    "projector_bias": False,
    "hidden_size": 1024,
    "num_hidden_layers": 23,
    "num_attention_heads": 16,
    "mlp_ratio": 4.0,
    "posemb_grid_size": 32,
    "layer_norm_eps": 1e-5,
    "num_channels": 3,
    "rope_theta": 10000.0,  # HF default (rope_parameters unset -> axial, 1e4)
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
    "rescale_factor": 1 / 255,
}

ImageLike = Union["PIL.Image.Image", bytes, bytearray, str, os.PathLike]  # noqa: F821


# ----------------------------------------------------------------------------
# Weights download (HTTP Range on the single BF16 shard; ~637 MB)


def _hf_headers(extra: dict | None = None) -> dict:
    h = {"User-Agent": "inkling-16gb-vision/0.1"}
    token = os.environ.get("HF_TOKEN")
    if token is None:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            token = None
    if token:
        h["Authorization"] = f"Bearer {token}"
    if extra:
        h.update(extra)
    return h


def _range_get(url: str, start: int, end_inclusive: int, timeout: float = 60.0):
    req = urllib.request.Request(url, headers=_hf_headers({"Range": f"bytes={start}-{end_inclusive}"}))
    resp = urllib.request.urlopen(req, timeout=timeout)
    if resp.status != 206:
        resp.close()
        raise RuntimeError(f"server ignored Range request (HTTP {resp.status}); refusing full download")
    return resp


def download_vision(
    path: str | os.PathLike | None = None,
    repo: str = HF_REPO,
    revision: str = HF_REVISION,
    force: bool = False,
    verbose: bool = True,
) -> Path:
    """Fetch only the ``model.visual.*`` tensors of the BF16 checkpoint via HTTP
    Range requests and write them to ``<path>/vision.safetensors``. Idempotent."""
    dest = Path(path) if path is not None else DEFAULT_PATH
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / WEIGHTS_FILE
    meta_path = dest / META_FILE
    if not force and out.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("revision") == revision and meta.get("repo") == repo and out.stat().st_size == meta.get("file_size"):
                if verbose:
                    print(f"vision weights already present: {out}")
                return dest
        except Exception:
            pass

    url = f"https://huggingface.co/{repo}/resolve/{revision}/{HF_SHARD}"
    with _range_get(url, 0, 7) as r:
        n = struct.unpack("<Q", r.read(8))[0]
    with _range_get(url, 8, 8 + n - 1) as r:
        header = json.loads(r.read())
    header.pop("__metadata__", None)
    data_start = 8 + n
    entries = sorted(
        ((k, v) for k, v in header.items() if k.startswith(WEIGHT_PREFIX)),
        key=lambda kv: kv[1]["data_offsets"][0],
    )
    if not entries:
        raise RuntimeError("no vision tensors found in checkpoint header")

    # New header with packed offsets (tensor order preserved).
    new_header, off = {}, 0
    for k, v in entries:
        a, b = v["data_offsets"]
        new_header[k] = {"dtype": v["dtype"], "shape": v["shape"], "data_offsets": [off, off + (b - a)]}
        off += b - a
    new_header["__metadata__"] = {"format": "mlx", "source": f"{repo}@{revision}/{HF_SHARD}"}
    hbytes = json.dumps(new_header, separators=(",", ":")).encode()
    hbytes += b" " * (-len(hbytes) % 8)

    # Coalesce source ranges (they are contiguous in this checkpoint -> 1 request).
    spans: list[list[int]] = []
    for _, v in entries:
        a, b = v["data_offsets"]
        if spans and spans[-1][1] == a:
            spans[-1][1] = b
        else:
            spans.append([a, b])
    total = sum(b - a for a, b in spans)
    if verbose:
        print(f"downloading {len(entries)} vision tensors ({total / 1e6:.1f} MB, {len(spans)} range(s)) from {repo}@{revision[:10]}")

    tmp = out.with_suffix(".partial")
    done = 0
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(hbytes)))
        f.write(hbytes)
        for a, b in spans:
            pos = a
            while pos < b:  # resume a span after transient errors
                try:
                    with _range_get(url, data_start + pos, data_start + b - 1, timeout=120) as r:
                        while pos < b:
                            chunk = r.read(min(8 << 20, b - pos))
                            if not chunk:
                                break
                            f.write(chunk)
                            pos += len(chunk)
                            done += len(chunk)
                            if verbose:
                                print(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB", end="", flush=True)
                except (OSError, RuntimeError) as e:  # pragma: no cover - network
                    if verbose:
                        print(f"\n  retrying after error: {e}")
    if verbose:
        print()
    if done != total:
        raise RuntimeError(f"short download: {done} of {total} bytes")
    os.replace(tmp, out)
    meta = {
        "repo": repo,
        "revision": revision,
        "shard": HF_SHARD,
        "n_tensors": len(entries),
        "data_bytes": total,
        "file_size": out.stat().st_size,
        "vision_config": VISION_CONFIG,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return dest


# ----------------------------------------------------------------------------
# Preprocessing (matches transformers Inkling2ImageProcessor / mlx-vlm port)


@dataclass
class VisionInput:
    pixel_values: np.ndarray  # [rows*cols, 40, 40, 3] float32, normalized, raster tile order
    rows: int
    cols: int

    @property
    def n_tokens(self) -> int:
        return self.rows * self.cols


def load_image(image: ImageLike):
    """PIL image from a PIL image, raw bytes, file path, http(s) URL or data URL.
    Non-PIL inputs get EXIF orientation applied (as transformers.load_image does)."""
    from PIL import Image, ImageOps

    if isinstance(image, Image.Image):
        return image
    if isinstance(image, (bytes, bytearray, memoryview)):
        data = bytes(image)
    else:
        s = os.fspath(image)
        if s.startswith("data:"):
            head, _, payload = s.partition(",")
            data = base64.b64decode(payload) if ";base64" in head else urllib.parse.unquote_to_bytes(payload)
        elif s.startswith(("http://", "https://")):
            req = urllib.request.Request(s, headers={"User-Agent": "inkling-16gb-vision/0.1"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read()
        else:
            with open(os.path.expanduser(s), "rb") as f:
                data = f.read()
    img = Image.open(io.BytesIO(data))
    img.load()
    return ImageOps.exif_transpose(img)


def _to_rgb_uint8(img) -> np.ndarray:
    """[H, W, 3] uint8, compositing alpha onto the 8px 204/153 checkerboard."""
    arr = np.asarray(img.convert("RGBA" if img.has_transparency_data else "RGB"))
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    if arr.shape[-1] == 4:
        h, w = arr.shape[:2]
        yy, xx = np.indices((h, w))
        bg = np.where((yy // 8 + xx // 8) % 2 == 0, 204, 153)[..., None]
        rgba = arr.astype(np.int32)
        alpha = rgba[..., 3:4]
        arr = ((rgba[..., :3] * alpha + bg * (255 - alpha) + 127) // 255).astype(np.uint8)
    return arr


def n_image_tokens(width: int, height: int) -> int:
    """LLM tokens for an image of this size: ceil(H/40) * ceil(W/40)."""
    return math.ceil(height / TILE) * math.ceil(width / TILE)


def preprocess(image: ImageLike, max_tokens: int | None = None) -> VisionInput:
    """Reference-exact preprocessing (no resize). ``max_tokens`` is an opt-in
    extension, NOT in the reference: if the image would exceed it, downscale
    (bicubic, aspect preserved) until ceil(H/40)*ceil(W/40) <= max_tokens."""
    from PIL import Image

    img = load_image(image)
    if max_tokens is not None and n_image_tokens(*img.size) > max_tokens:
        w, h = img.size
        s = math.sqrt(max_tokens * TILE * TILE / (w * h))
        while True:
            nw, nh = max(1, int(w * s)), max(1, int(h * s))
            if n_image_tokens(nw, nh) <= max_tokens:
                break
            s *= 0.98
        img = img.resize((nw, nh), Image.BICUBIC)
    arr = _to_rgb_uint8(img)
    h, w, _ = arr.shape
    rows, cols = math.ceil(h / TILE), math.ceil(w / TILE)
    padded = np.full((rows * TILE, cols * TILE, 3), -1.0, np.float32)
    padded[:h, :w] = arr
    mean = np.array(VISION_CONFIG["image_mean"], np.float32)
    std = np.array(VISION_CONFIG["image_std"], np.float32)
    padded = (padded * np.float32(VISION_CONFIG["rescale_factor"]) - mean) / std
    tiles = padded.reshape(rows, TILE, cols, TILE, 3).transpose(0, 2, 1, 3, 4).reshape(rows * cols, TILE, TILE, 3)
    return VisionInput(np.ascontiguousarray(tiles), rows, cols)


# ----------------------------------------------------------------------------
# Model


def _rotate_pairs(x):
    p = x.reshape(*x.shape[:-1], -1, 2)
    return mx.stack([-p[..., 1], p[..., 0]], axis=-1).reshape(x.shape)


class _Attention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.n_heads, self.head_dim = n_heads, d // n_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.out_proj = nn.Linear(d, d)

    def __call__(self, x, cos, sin):
        L = x.shape[0]
        q, k, v = (p(x).reshape(1, L, self.n_heads, self.head_dim) for p in (self.q_proj, self.k_proj, self.v_proj))

        def rope(a):
            af = a.astype(mx.float32)
            return (af * cos[None, :, None] + _rotate_pairs(af) * sin[None, :, None]).astype(a.dtype)

        q, k = rope(q), rope(k)
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5)  # bidirectional
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(L, -1))


class _MLP(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.fc1 = nn.Linear(d, hidden)
        self.fc2 = nn.Linear(hidden, d)

    def __call__(self, x):
        return self.fc2(nn.gelu(self.fc1(x)))  # exact (erf) GELU, as HF "gelu"


class _Layer(nn.Module):
    def __init__(self, d, n_heads, hidden, eps):
        super().__init__()
        self.self_attn = _Attention(d, n_heads)
        self.mlp = _MLP(d, hidden)
        self.layernorm_before = nn.LayerNorm(d, eps=eps)
        self.layernorm_after = nn.LayerNorm(d, eps=eps)
        self.lambda_1 = mx.ones((d,))
        self.lambda_2 = mx.ones((d,))

    def __call__(self, x, cos, sin):
        x = x + self.lambda_1 * self.self_attn(self.layernorm_before(x), cos, sin)
        return x + self.lambda_2 * self.mlp(self.layernorm_after(x))


def _bilinear_taps(in_size: int, out_size: int):
    """F.interpolate(mode='bilinear', align_corners=False) taps along one axis
    (same formula/op order as mlx-vlm ``_bilinear_weights_1d``)."""
    src = (mx.arange(out_size, dtype=mx.float32) + 0.5) * (in_size / out_size) - 0.5
    src = mx.maximum(src, 0.0)
    base = mx.floor(src)
    frac = src - base
    pos = base[:, None] + mx.arange(2, dtype=mx.float32)[None, :]
    idx = mx.clip(pos.astype(mx.int32), 0, in_size - 1)
    return idx, mx.stack([1.0 - frac, frac], axis=1)


_INTERP_SRC = """
    uint gx = thread_position_in_grid.x;
    uint y_out = thread_position_in_grid.y;
    int in_w = x_shape[1];
    int channels = x_shape[2];
    int out_h = iy_shape[0];
    int out_w = ix_shape[0];
    if (gx >= (uint)(out_w * channels) || y_out >= (uint)out_h) return;
    int x_out = gx / channels;
    size_t base = (size_t)(gx % channels);
    // W taps contracted inside the H loop (ATen / mlx-vlm summation order)
    float result = 0.0f;
    for (int a = 0; a < 2; a++) {
        size_t row = base + (size_t)iy[y_out * 2 + a] * in_w * channels;
        float row_result = 0.0f;
        for (int c = 0; c < 2; c++) {
            row_result += x[row + (size_t)ix[x_out * 2 + c] * channels] * wx[x_out * 2 + c];
        }
        result += row_result * wy[y_out * 2 + a];
    }
    out[(size_t)y_out * out_w * channels + gx] = result;
"""
_interp_kernel = None


def _resize_bilinear(table: mx.array, out_h: int, out_w: int) -> mx.array:
    """[H, W, C] float32 -> [out_h, out_w, C] float32, bilinear, align_corners=False."""
    global _interp_kernel
    H, W, C = table.shape
    if (H, W) == (out_h, out_w):
        return table
    iy, wy = _bilinear_taps(H, out_h)
    ix, wx = _bilinear_taps(W, out_w)
    if mx.metal.is_available():
        if _interp_kernel is None:
            _interp_kernel = mx.fast.metal_kernel(
                name="inkling_vision_bilinear", input_names=["x", "iy", "wy", "ix", "wx"],
                output_names=["out"], source=_INTERP_SRC,
            )
        return _interp_kernel(
            inputs=[table, iy, wy, ix, wx],
            grid=(out_w * C, out_h, 1),
            threadgroup=(min(256, out_w * C), 1, 1),
            output_shapes=[(out_h, out_w, C)],
            output_dtypes=[mx.float32],
        )[0]
    # CPU fallback (same math, not bit-identical summation order)
    y = (mx.take(table, iy.reshape(-1), axis=0).reshape(out_h, 2, W, C) * wy[:, :, None, None]).sum(1)
    return (mx.take(y, ix.reshape(-1), axis=1).reshape(out_h, out_w, 2, C) * wx[None, :, :, None]).sum(2)


class VisionEncoder(nn.Module):
    def __init__(self, cfg: dict | None = None):
        super().__init__()
        cfg = dict(VISION_CONFIG if cfg is None else cfg)
        self.cfg = cfg
        self.eval_every = 1  # mx.eval cadence over ViT layers (0 = single lazy graph)
        d, C = cfg["hidden_size"], cfg["num_channels"]
        p, m = cfg["native_patch_size"], cfg["adapter_merge_size"]
        g = cfg["posemb_grid_size"]
        self.patch_embedding = nn.Linear(C * p * p, d, bias=False)
        self.position_embedding = mx.zeros((g * g, d))
        self.class_embedding = mx.zeros((d,))
        self.cls_position_embedding = mx.zeros((d,))
        self.pre_layernorm = nn.LayerNorm(d, eps=cfg["layer_norm_eps"])
        hidden = int(d * cfg["mlp_ratio"])
        self.layers = [
            _Layer(d, cfg["num_attention_heads"], hidden, cfg["layer_norm_eps"]) for _ in range(cfg["num_hidden_layers"])
        ]
        self.proj_in = nn.Linear(m * m * d, cfg["projection_hidden_dim"], bias=cfg["projector_bias"])
        self.proj_out = nn.Linear(cfg["projection_hidden_dim"], cfg["text_hidden_size"], bias=cfg["projector_bias"])

    @property
    def dtype(self):
        return self.pre_layernorm.weight.dtype

    # -- helpers ------------------------------------------------------------
    def _pos_embed(self, rows: int, cols: int) -> mx.array:
        """Learned 32x32 table bilinearly resized to the (2*rows, 2*cols) patch
        grid, in merge-block order [rows, cols, 2, 2] -> [N, d], float32."""
        m, g, d = self.cfg["adapter_merge_size"], self.cfg["posemb_grid_size"], self.cfg["hidden_size"]
        t = _resize_bilinear(self.position_embedding.astype(mx.float32).reshape(g, g, d), rows * m, cols * m)
        return t.reshape(rows, m, cols, m, d).transpose(0, 2, 1, 3, 4).reshape(rows * cols * m * m, d)

    def _rope(self, rows: int, cols: int):
        """Axial 2D RoPE: CLS at (0,0), patches at 1-indexed (x, y); angles are
        [w freqs | h freqs], each repeated for one interleaved channel pair."""
        cfg = self.cfg
        m = cfg["adapter_merge_size"]
        yy, xx = np.meshgrid(np.arange(rows * m), np.arange(cols * m), indexing="ij")
        pos = np.stack([xx, yy], axis=-1).reshape(rows, m, cols, m, 2).transpose(0, 2, 1, 3, 4).reshape(-1, 2) + 1
        pos = mx.array(np.concatenate([np.zeros((1, 2)), pos]), mx.float32)
        spatial = cfg["hidden_size"] // cfg["num_attention_heads"] // 2
        inv = cfg["rope_theta"] ** (-mx.arange(0, spatial, 2) / spatial)
        ang = mx.repeat((pos[..., None] * inv).reshape(pos.shape[0], -1), 2, axis=-1)
        return mx.cos(ang), mx.sin(ang)

    # -- forward ------------------------------------------------------------
    def __call__(self, pixel_values, rows: int, cols: int) -> mx.array:
        cfg = self.cfg
        m, p, C = cfg["adapter_merge_size"], cfg["native_patch_size"], cfg["num_channels"]
        count = rows * cols
        assert pixel_values.shape == (count, m * p, m * p, C), pixel_values.shape
        px = mx.array(pixel_values).astype(self.dtype)
        patches = px.reshape(count, m, p, m, p, C).transpose(0, 1, 3, 2, 4, 5).reshape(count * m * m, -1)
        x = self.patch_embedding(patches)
        x = x + self._pos_embed(rows, cols).astype(x.dtype)
        cls = (self.class_embedding + self.cls_position_embedding).astype(x.dtype)
        x = self.pre_layernorm(mx.concatenate([cls[None], x], axis=0))
        cos, sin = self._rope(rows, cols)
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.eval(x)  # bounds peak memory: one big lazy graph retains ~32 MB/layer
        x = x[1:].reshape(count, m * m * cfg["hidden_size"])
        return self.proj_out(nn.gelu(self.proj_in(x)))

    def encode(self, image: ImageLike | VisionInput, max_tokens: int | None = None) -> mx.array:
        """-> [n_tokens, 2304] rows (model dtype, bf16) to drop into the LM
        embedding sequence in place of the expanded image placeholder ids."""
        vi = image if isinstance(image, VisionInput) else preprocess(image, max_tokens=max_tokens)
        out = self(vi.pixel_values, vi.rows, vi.cols)
        mx.eval(out)
        return out


def _remap(key: str) -> str | None:
    if not key.startswith(WEIGHT_PREFIX):
        return None
    k = key[len(WEIGHT_PREFIX):].removeprefix("backbone.").removeprefix("embeddings.")
    k = k.replace("ln_pre.", "pre_layernorm.").replace(".ln_1.", ".layernorm_before.").replace(".ln_2.", ".layernorm_after.")
    k = k.replace(".ls_1.gamma", ".lambda_1").replace(".ls_2.gamma", ".lambda_2")
    k = k.replace(".mlp.c_fc.", ".mlp.fc1.").replace(".mlp.c_proj.", ".mlp.fc2.")
    return k


def load_vision(path: str | os.PathLike | None = None, lazy: bool = False) -> VisionEncoder:
    """Load the encoder from ``<path>/vision.safetensors`` (default
    ~/.cache/inkling-16gb/vision). ``lazy=True`` defers materializing weights
    until first use. Raises FileNotFoundError if ``download_vision()`` wasn't run."""
    d = Path(path) if path is not None else DEFAULT_PATH
    f = d / WEIGHTS_FILE if d.is_dir() or not d.suffix else d
    if not f.exists():
        raise FileNotFoundError(f"{f} not found; run inkling_16gb.vision.download_vision()")
    cfg = VISION_CONFIG
    meta = f.parent / META_FILE
    if meta.exists():
        cfg = {**VISION_CONFIG, **json.loads(meta.read_text()).get("vision_config", {})}
    model = VisionEncoder(cfg)
    raw = mx.load(str(f))
    weights = [(k2, v) for k, v in raw.items() if (k2 := _remap(k)) is not None]
    model.load_weights(weights, strict=True)
    model.eval()
    if not lazy:
        mx.eval(model.parameters())
    return model


# ----------------------------------------------------------------------------
# Token-id expansion


def expand_placeholders(
    ids: Sequence[int], n_tokens_per_image: Sequence[int], image_token_id: int = IMAGE_TOKEN_ID
) -> tuple[list[int], list[tuple[int, int]]]:
    """Repeat the i-th ``image_token_id`` n_tokens_per_image[i] times.
    Returns (new_ids, [(start, n), ...]) with spans into new_ids, in image order.
    The chat template renders each image as
    ``<|content_image|> <|unused_200054|> <|end_message|>``; this turns the single
    placeholder into the N slots the encoder rows fill."""
    out: list[int] = []
    spans: list[tuple[int, int]] = []
    it = iter(n_tokens_per_image)
    for t in ids:
        if t == image_token_id:
            try:
                n = int(next(it))
            except StopIteration:
                raise ValueError("more image placeholders than images") from None
            if n < 1:
                raise ValueError("each image needs >= 1 token")
            spans.append((len(out), n))
            out.extend([image_token_id] * n)
        else:
            out.append(int(t))
    if len(spans) != len(n_tokens_per_image):
        raise ValueError(f"{len(n_tokens_per_image)} images but {len(spans)} placeholders")
    return out, spans


def merge_embeddings(h: mx.array, spans: Sequence[tuple[int, int]], rows: Sequence[mx.array], offset: int = 0) -> mx.array:
    """Replace rows of an embedding matrix ``h`` [T, D] (post embed_norm) at the
    given spans. ``offset`` is the absolute index of h[0] (for chunked prefill);
    spans partly outside [offset, offset+T) are clipped."""
    T = h.shape[0]
    pieces, cur = [], 0
    for (s, n), r in zip(spans, rows):
        a, b = max(s - offset, 0), min(s + n - offset, T)
        if a >= b:
            continue
        if a > cur:
            pieces.append(h[cur:a])
        ra = a - (s - offset)
        pieces.append(r[ra : ra + (b - a)].astype(h.dtype))
        cur = b
    if not pieces:
        return h
    if cur < T:
        pieces.append(h[cur:])
    return mx.concatenate(pieces, axis=0)


__all__ = [
    "VisionEncoder",
    "VisionInput",
    "download_vision",
    "load_vision",
    "load_image",
    "preprocess",
    "n_image_tokens",
    "expand_placeholders",
    "merge_embeddings",
    "IMAGE_TOKEN_ID",
    "IMAGE_TOKEN",
]
