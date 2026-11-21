# inkling_fast: Inkling 2 Mini on a base M4 (32 GB)

`src/inkling_fast/` is a compiled decode engine with custom Metal kernels for the looped Inkling 2 Mini (`inkling2_loop`) on MLX. The loop model itself comes from internet-projects/inkling2-mlx v0.1.0. The default weights are `weights/inkling-2-mini/mlx-3bit` (netprojects 3-bit MLX build, 14.7 GB plus a 0.64 GB DSpark head).

## Results (standard benchmark: batch 1, greedy, 3-bit weights, best of 3, GPU lock held)

| Engine | Prefill 512 tok/s | Prefill 2048 tok/s | Decode tg128 tok/s |
|---|---|---|---|
| llama.cpp b12600, Unsloth UD-Q3_K_XL (`llama-bench -p 512/2048 -n 128`) | 37.6 | 37.8 | 12.8 |
| Standard MLX (upstream mlx-vlm 0.8.0 + loop module) | 178–184 | 168 | 18.6–19.6 |
| oMLX 0.8.1, oQ3e-mtp (its own adaptive MTP) | – | – | 22.4 |
| **inkling_fast (GPU)** | **222–230** | **195–200** | **26.8–28.6** |
| **inkling_fast + ANE hybrid prefill** | (GPU below 1024) | **325** | same |

Speedups:
- **vs llama.cpp: decode 2.1–2.2x; prefill 5.9–6.1x at 512 and 8.6x at 2048 with the ANE hybrid.**
- **vs standard MLX: decode 1.4–1.5x; prefill 1.2–1.3x at 512 and 1.9x at 2048 with the ANE hybrid.**

Through `scripts/inkling_serve.py` (OpenAI API, `scripts/inkling_server_bench.sh`):

| Mode | Decode | Fresh 2,167-token prompt, TTFT |
|---|---|---|
| GPU only | 28.7 tok/s | 11.0–11.4 s |
| GPU + ANE hybrid prefill (`INKLING_ANE_PREFILL=1`, .venv) | 28.5–28.7 tok/s | 7.6–7.8 s |

With the prefix cache on (the default), a repeated 4.4K-token context goes from 30 s to 0.9 s TTFT.

Correctness: greedy tokens are identical to the reference on the check prompt. Teacher-forced top-1 is 0.6169 (reference 0.6194; the gap comes from bf16 scales) and perplexity 5.886 (reference 5.898).

## 3-bit vs 4-bit (both on inkling_fast)

| Build | Decode | Prefill 512 | Top-1 | Perplexity | Size |
|---|---|---|---|---|---|
| `netprojects/Inkling-2-Mini-MLX-3bit` (default) | 26.8–28.8 tok/s | 222–230 | 0.6169 | 5.886 | 14.7 GB |
| `mlx-community/Inkling-2-Mini-oQ4e-mtp` (TML's NVFP4 experts untouched, 8-bit attention) | 21.6–22.7 tok/s | 230 | 0.6243 | 5.717 | 16.9 GB |

- oQ checkpoints use oMLX's layout. `src/inkling_fast/oq.py` maps them into our model at load time, with no rewritten files:
  - fused attention rows are split
  - the packed interleaved `w13` experts are read as-is
  - NVFP4 global scales are applied after the matmuls
- The ANE prefill slices come from the 3-bit weights, so the 4-bit entry prefills on the GPU.

## Serving

- `scripts/inkling-fast-serve <weights-dir> <port> [model-alias]` starts the OpenAI-compatible server on 127.0.0.1.
- It uses the ANE hybrid prefill for the 3-bit weights when coremltools is installed and the slices exist in `runtime/ane/prod_int8`. Other builds prefill on the GPU.
- `/health` reports `model_ready` and `loaded_model`.

## Other paths

- **ANE + GPU hybrid prefill** (`src/inkling_ane/`, notes in `docs/inkling-ane-prefill.md`):
  - The dense part of each layer application runs as Core ML slices, 100% on the ANE. The MoE stays on the GPU in MLX. They overlap across prompt chunks.
  - Results: 279–295 tok/s at 2048 tokens (1.5–1.6x our MLX path), 1.3x at 1024, and no gain at 512.
  - In the server it hands its caches off to the compiled decoder: 2,167-token TTFT goes from 11.0 s to 7.6 s, and 2048-token prefill reaches 325 tok/s.
  - ANE decode is not viable: 53–84 ms/token for the dense parts alone.
- **TensorFold port** (`tensorfold/inkling2-metal.patch` against TensorFold 99d2384, plan in `docs/tensorfold-inkling2-mini.md`):
  - Greedy output matches the reference (16/16 tokens). Decode is 25.0 tok/s and prefill 48 tok/s, from its dev tool.
  - Still needed: the served engine, a tiled prefill kernel for the experts, and kernel-contract registration.

## What is in the engine

- `__init__.py`: registers the loop model with mlx-vlm's prompt and DSpark code (the DSpark head taps 48 layer applications, not 24).
- `oq.py`: loads oMLX oQ checkpoints (see above).
- `fast.py`:
  - The q/k/v/r/gate projections are packed into one quantized matmul. Rows are independent, so this is exact.
  - The gate+up experts are packed into one `gather_qmm`. The originals are freed, which saves about 6 GB.
  - Opt-in, lossy: `INKLING_ATTN_BITS=4` requantizes attention to 4-bit. It costs 0.5 points of top-1 accuracy and 2.3% perplexity, for about 8% more speed.
- `decode.py`: each of the 48 layer applications is two `mx.compile`d pure functions with explicit conv state. It handles 1–16 tokens, including speculative verify with DSpark taps and cache transactions.
  - SDPA uses native sinks for one token, so there's no zero-K/V concat.
  - A ring-buffer mask attends to wrapped sliding caches in storage order, so there's no temporal re-order copy (saves about 3.8 ms/token past 1K context).
- `moe_v4.py`: a fused one-token 3-bit MoE (gather gate+up with the softcapped SwiGLU, then gather down).
  - It uses MLX's pre-scaled masked-byte trick, 16 contiguous values per lane and 4 rows per simdgroup.
  - It reaches 89 GB/s where MLX's generic `gather_qmv` gets 76 GB/s (repro at the Inkling shapes).
  - Variants v5 (strided single packs), v6 (parameter sweep) and v7 (16-bit loads) did not beat it.
- `kernels.py`, router epilogue: sigmoid, bias, parallel-rank top-8 and normalization in one kernel, after MLX's multi-core router matmul. Saves 0.6–1.0 ms/token.
- **bf16 scales** (`INKLING_BF16_SCALES`, on): the 3-bit expert layers ship float32 per-group scales and biases.
  - Storing them as bf16 matches the bf16 activations and stays on MLX's fast path. Saves 5%.
  - Cost: −0.25 top-1 points, +0.1% perplexity.
  - fp16 scales are 5x slower because of the dtype mismatch with bf16 activations.
- **bf16-activation head** (`INKLING_BF16_HEAD`, on): the 4-bit weights are unchanged. 4.6 → 2.9 ms, logits within 0.015.
- **Compiled prefill:** chunks of 256/512/1024/2048 tokens take the compiled path, and other lengths take the fused eager path, so there's no per-request retrace.
  - Prefill chunks compute the vocabulary head only for the last position.
  - 512-token prefill: 178 (standard MLX) → 230 tok/s.
  - Teacher-forced quality is identical: top-1 0.6169 for both, perplexity 5.886 compiled vs 5.905 reference.
- `kernels.py` (Metal):
  - Message gates: 2 launches per gate set, batched across q/k/v and tokens.
  - Attention-residual mixing with RMSNorm fused: +8%.
  - Sliding-layer post-projection: short conv, L2 clips and value norm in one launch, a small gain.
  - Ring mask.
  - Kept but disabled, because each measured slower than MLX's kernels: a fused MoE (exact) and a router.

## Where the time goes, and the ceiling

- **The measured peak streaming read on this M4 is 105.8 GB/s** (88% of the 120 GB/s spec).
- **Bytes per token:** experts 1.34 GB, attention 1.04 GB (6-bit, re-read on each loop pass), head 0.26 GB. That's about 2.7 GB, so the floor is about 25.5 ms/token (about 39 tok/s) with perfect kernels. The engine is at about 35 ms/token.
- The weight kernels run at about 84 GB/s (MoE) and 88 GB/s (attention).
- **A chain of only the weight matmuls costs about as much as the whole step**, so the remaining overhead is in MLX's matvec and gather efficiency.

## Speculative decoding economics on a base M4 (measured, so parked)

- A 2-token verify with taps plus rollback costs 1.45–1.53x a plain step. Each token brings its own 8 experts per layer pass.
- Consecutive tokens share 3.3 of their 8 experts. A deduplicating multi-token MoE (`moe_multi.py`) is correct, but it only breaks even with MLX at L=2.
- A DSpark draft block costs 8.5 ms even with 4-bit draft layers:
  - the full 201K-row head per draft token: 0.26 GB, about 2.9 ms
  - the folded Markov projection: 201K x 256 in bf16
  - 3 layers over a fixed 8-slot block
- Through mlx-vlm's loop with a 4-bit draft: 0.86x (code), 0.88x (chat), 1.00x (math). The best case for a lean loop is about +10% on code and about 0 on chat.

## Things that did not pay off on this machine (measured)

- **DSpark speculative decoding:** 0.45–0.9x at every block size. Each extra verify token pulls in 8 new experts per layer pass, costing about half a step. The draft accepts only 0.6–1.5 tokens per round against 3-bit weights.
- **Fewer loop passes, with or without LoopCD (arXiv 2610.02185):**
  - 2 passes: top-1 accuracy 0.618 → 0.505, perplexity 5.9 → 12.2.
  - LoopCD hidden or logits, fixed or adaptive, never recovers it. At full depth it has no effect.
  - Inkling 2 Mini is trained only at 3 passes.
- **Bigger Metal command buffers** (`MLX_MAX_OPS_PER_BUFFER`): no gain.
- **4-bit or FP4 experts instead of 3-bit:** slower. MoE time tracks bytes read.

## Tools

- `scripts/inkling_check.py`: correctness against the reference, plus speed.
- `scripts/inkling_ab.py`: steady-state ms/token. Use with env flags for A/B.
- `scripts/inkling_ctx.py`: decode speed against context length.
- `scripts/inkling_floor.py`: weight-matmul-only floor.
- `scripts/inkling_quality.py`: teacher-forced top-1 accuracy and perplexity.
- `scripts/inkling_loopcd_eval.py`: LoopCD study.
- `scripts/inkling_reference_dump.py`: reference logits and hidden states for cross-engine checks.
- `scripts/with-gpu-lock`: one heavyweight GPU/ANE job at a time.
- `scripts/inkling_compare.py`: the standard pp512/tg128 benchmark (`mlx` or `fast`; `INKLING_WEIGHTS` selects an oQ build).

Note: the quality numbers above were measured on a different 4-document English corpus. The scripts now default to this repo's docs, so absolute values will differ.
