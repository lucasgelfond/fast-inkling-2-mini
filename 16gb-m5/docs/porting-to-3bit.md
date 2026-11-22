# Porting the 16gb-m5 changes to the 3-bit engine

The 3-bit build (`netprojects/Inkling-2-Mini-MLX-3bit`, 14.07 GB of weights) doesn't fit a 16 GB Mac, so none of
this was run against it. It's a map from what `16gb-m5/` changed to where the same change would go in
`src/inkling_fast/` (mlx-vlm's `inkling2_loop` + the decode patches), for a machine with 32 GB+.

| 16gb-m5 change | Where it lives here | 3-bit engine | Action |
|---|---|---|---|
| Chunked prefill (512) | `engine.py` `_install_hooks` | already: mlx-vlm server `--prefill-step-size 512` | none |
| Prefix cache, live continuation + checkpoint | `engine.py` `_Slot`, `_Checkpoint` | already: mlx-vlm APC (+ the `merge_exact` fix in `decode.py`) | check that pi turns hit APC: needs pi to echo `reasoning_content` (below). On 16gb-m5 this was verified with a pi RPC session: every request after the first reused 92-99% of its prompt |
| Several cached conversations | `engine.py` `_pick_slot` | APC is keyed by prefix, so several sessions should coexist | verify APC capacity (`APC_DISK_MAX_GB`, in-memory entries) with 2+ pi sessions |
| Tool calls / images through the chat template | `engine.py` `encode`, `_OutputParser` | mlx-vlm server | none; vision already in the 3-bit weights (quantized) |
| Quantized KV for the 8 global-attention visits (4/8-bit, group 32) | `qkv.py` (rewrites the LookAtMyQuant Metal kernels) | `decode.py` attention: mlx-vlm `KVCache` + `mx.fast.scaled_dot_product_attention` with an additive relpos mask | use mlx-lm `QuantizedKVCache` for the **global** layers only (sliding layers are already `RotatingKVCache`, capped) and `quantized_scaled_dot_product_attention` from `mlx_lm.models.base` (takes the additive mask; check `sinks` support). Leads, unverified: mlx-vlm's `--kv-bits` / `--quantized-kv-start` options |
| Split-K decode attention | `qkv.py` `_decode_split_kernel` | `mx.fast.scaled_dot_product_attention` | measure first: MLX's own sdpa already splits long KV for one query |
| Drafter windowed to 4k; drafting off above 32k context | `engine.py` `TAPS_WINDOW`, `draft_max_context` | fast-inkling doesn't use the drafter (`IGNORE = ["mtp.safetensors"]`) | none; the lesson if it ever does: each verify row re-reads the whole KV cache |
| pi: auto-start/stop the server, low thinking | `pi/inkling-16gb.ts` | same extension | change `PROVIDER`, `PORT` (8080), `BIN` (`inkling-fast`), and the spawn args (`serve --port 8080`) |
| pi: lean prompt | `../docs/pi-lean.md` | applies already (any localhost model) | none |
| pi model config | README "pi" section | README config | add `"requiresThinkingAsText": false` and `"off": "none"` in `thinkingLevelMap`; `"input": ["text", "image"]` |

## Memory math for the 3-bit build at long context

Global-attention KV is 32 KB/token in fp16 (8 visits x 2 x 8 heads x 128 x 2 bytes): 8.6 GB at 262k. With 14.07 GB of
weights that's ~23 GB before activations. A 32 GB Mac's default GPU limit is ~24 GB, so fp16 KV is borderline at
262k; 8-bit KV (17 KB/token, 4.6 GB) is comfortable, 4-bit (9.2 KB/token, 2.4 GB) leaves room for several sessions.
Measured on the 8.9 GB build: 4-bit KV is KL 0.0077 / top-1 98.6% vs fp16 KV (8-bit: 0.0020 / 99.4%), on 30k tokens.

Watch prefill memory: an additive relpos mask for a 512-token chunk against N cached keys is 512 x N x 24 heads; at
262k that's ~12 GB in fp32. Prefill at long context needs the mask built per KV tile (or a smaller chunk), not whole.

## Things that bit on the 16 GB build and may bite here

- pi only reuses the cache turn to turn if the re-rendered assistant turn is token-identical to what was generated:
  with `requiresThinkingAsText: false` pi sends thinking back as `reasoning_content`, which the template renders
  exactly; a truncated reply (hit `max_tokens`) never matches.
- A KV cache that grows by a fixed 256 slots and copies itself each time is quadratic at long context.
- Keep any per-position history (drafter taps etc.) windowed: at 262k, fp32 hidden taps alone were 9.7 GB.
