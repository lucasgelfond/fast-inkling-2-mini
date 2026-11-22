# Inkling 2 Mini vision: findings

Module: `src/inkling_16gb/vision.py` (mlx + numpy + Pillow only).
References I read: Transformers 5.24 `models/inkling2/{image_processing,processing,modeling}_inkling2.py`
(it is in the venv; it imports without torch for configs only), and the mlx-vlm fork
`vendor/inkling2-mlx` (tag v0.1.0), `mlx_vlm/models/inkling2/{vision,processing_inkling2,inkling2}.py`.
`inkling2_loop` reuses the `inkling2` vision tower and processor unchanged.

## How an image becomes LM input

1. **Decode**: convert to RGB. If the image has transparency, it is converted to RGBA and
   composited onto an 8 px checkerboard (204/153 gray) with `(c*a + bg*(255-a) + 127)//255`.
   Grayscale images are converted to RGB. When the input is a path, URL or bytes, EXIF orientation
   is applied, as `transformers.load_image` does.
2. **No resize, no tiling, no dynamic resolution.** The image is padded on the bottom and right to a
   multiple of 40 px. The pad value is **-1 on the 0..255 scale**, so after normalization it is
   ≈ -1.0078. Then `x/255`, then `(x-0.5)/0.5`.
3. Cut the image into 40x40x3 tiles in raster order. **One tile is one LM token:**
   `n_tokens = ceil(H/40) * ceil(W/40)`. Examples: 640x480 → 192, 1024x768 → 520, 4096² → 10 609.
   The model card says to keep each side between 40 and 4096 px. `preprocess(max_tokens=...)` is an
   opt-in downscale that is not in the reference.
4. **Vision tower** (BF16, 422 tensors, 636.5 MB): a ViT with hidden size 1024, 23 layers, 16 heads,
   head_dim 64, MLP 4096 with exact GELU, pre-LayerNorm (eps 1e-5), and LayerScale (`lambda_1/2`).
   - Each 40 px tile holds 2x2 native 20 px patches. The patch embedding is
     `Linear(1200→1024, no bias)` over the flattened (20, 20, 3) patch. Patches are ordered
     merge-block-major: [tile row, tile col, sub-row, sub-col].
   - Learned 32x32 position table, bilinearly resized (align_corners=False) to the
     (2·rows, 2·cols) patch grid, then added.
   - One CLS token per image (`class_embedding + cls_position_embedding`) goes in front, then
     `pre_layernorm`.
   - Axial 2D RoPE (theta 1e4) inside the ViT. Patch positions are 1-indexed (x, y) and CLS is at
     (0, 0). The angle vector is [w freqs | h freqs], each repeated for one interleaved channel pair.
   - ViT attention is **bidirectional** within an image and never crosses images.
5. **Projector**: drop the CLS token and concatenate each tile's 4 patch features (4096). Then
   `Linear(4096→4096, no bias)` → GELU → `Linear(4096→2304, no bias)`. The result is
   `[n_tokens, 2304]` BF16.
6. **Placeholder expansion**: the chat template emits
   `<|content_image|>(200005) <|unused_200054|>(200054) <|end_message|>(200010)` for each image.
   The processor repeats id 200054 `n_tokens` times, in image order (`expand_placeholders`).
7. **Merge**: `inputs_embeds = embed(ids)`, which is the embedding *after* `embed_norm`. Then the
   rows where `ids == 200054` are replaced in order with the projector output (`masked_scatter`).
   The image rows are **not** passed through embed_norm. The packed runtime's `PackedEmbed` already
   returns post-embed_norm rows, so the splice goes right after `h = self.embed(input_ids)[0]`.

### What the language model needs beyond replacing rows: nothing

I checked this in both HF and mlx-vlm:
- Positions are plain 1D `arange(past, past+T)` for every token, image tokens included. There is no
  M-RoPE or 2D position for image tokens. The LM uses a relative-position bias (`rel_proj`, backward
  distance) and log-scaling, both 1D. The 2D RoPE exists only inside the ViT.
- The mask is plain causal (plus the sliding window on local layers). Image tokens get no
  bidirectional or prefix-LM mask.
- No token-type ids, per-token flags or modality embeddings. The short convolutions and the MoE
  treat image rows like any other token.
- The placeholder ids stay as 200054 in the id sequence. They only matter through the replaced
  embedding rows.
- The DSpark drafter only embeds its anchor (generated) token and reads target hidden-state taps.
  Image positions therefore need no special handling there.

## Weights

- `~/.cache/inkling-16gb/vision/vision.safetensors`: 636 563 624 bytes, 422 BF16 tensors with the
  original `model.visual.*` names. Metadata is in `vision_meta.json`.
- Source: `thinkingmachines/Inkling-2-Mini` @ `b85c3cfa615fc413cc9af3379f491b963f3b586a`, file
  `model-00001-of-00001.safetensors` (19.5 GB single shard). The vision tensors are one contiguous
  byte range, so it takes one HTTP Range request plus two small ones for the header.
  `download_vision()` is idempotent: it checks revision, repo and file size.
- `netprojects/Inkling-2-Mini-MLX-3bit` quantizes the vision Linear layers (U32 + scales), so it is
  not a BF16 source. Its BF16 tensors (position_embedding, class_embedding, LayerNorm weights,
  lambdas) are byte-identical to ours.

## Verification

- **Preprocessing**: bit-identical (max |diff| = 0) to mlx-vlm's `Inkling2ImageProcessor` on
  three synthetic images: RGB 251x173 (padding), RGBA 64x97 (alpha compositing) and L 120x80. It is
  also identical through the bytes, file-path and data-URL loaders. The mlx-vlm processor is a
  line-by-line port of the HF one. I could not run the HF one because it needs torch/torchvision.
- **Encoder**: bit-identical BF16 output to mlx-vlm's `VisionModel` with the same weights, on all
  three images and on the two-image batched path. To get there, I copied mlx-vlm's op order for the
  RoPE frequency formula and the bilinear position-table resize: same Metal kernel summation order.
  With the HF-style formulas the result differed by up to 0.03 abs, with cos ≥ 0.999, which is f32
  ulp noise amplified through BF16. **Not verified**: the torch/HF forward (no torch on this machine).
  The mlx-vlm README also says strict parity with the reference runtime is not established. The
  architecture matches HF line by line as read.
- **Expected scale**: on noise images, output row RMS ≈ 0.48. Text embedding rows after embed_norm
  have RMS ≈ 0.90 (rms of the gamma). Same order of magnitude, so nothing looks off. This is not a
  correctness test.

## Cost (M5, 16 GB, with another process holding most RAM)

Weights are 637 MB resident. Peak memory includes the weights, with `eval_every=1` (the default; it
calls `mx.eval` after each ViT layer):

| image | tokens | peak | warm time |
|---|---|---|---|
| 251x173 | 35 | 654 MB | 0.05 s |
| 640x480 | 192 | 711 MB | 0.15 s |
| 1024x768 | 520 | 814 MB | 0.4 s |
| 1600x1200 | 1200 | 1041 MB | 1.0 s |
| 2048x2048 | 2704 | 1387 MB | 3.2 s |

With a single lazy graph (`eval_every=0`) the peak was 1.57 GB at 1024x768, about 32 MB retained per
layer, with no speed benefit.

## Integration notes

- `PackedEmbed`/`_forward_core_fast` only takes ids. It needs a hook in the runtime, either an
  `embeds_override` argument or a wrapped `self.embed` that splices rows with
  `merge_embeddings(h, spans, rows, offset=cache.seq_len)`. The offset form handles chunked prefill.
- Image tokens cost LM prefill like text tokens. Above 1024 tokens an image is larger than the local
  layers' 1024 sliding window, which is the same as the reference. Consider a `max_tokens` cap of
  roughly 1024 for 16 GB.
- The prefix cache must key on image content, not just ids, because identical ids can hide
  different images. The engine already hashes the image source.
- The tower can be unloaded after encoding (`del` + `mx.clear_cache()`) to give 637 MB back to
  the LM.
