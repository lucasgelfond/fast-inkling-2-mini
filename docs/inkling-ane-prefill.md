# Inkling 2 Mini: ANE + GPU hybrid prefill

Goal: find out whether running the dense parts of each layer application on the Apple Neural Engine
(Core ML), pipelined against the MoE on the GPU (MLX), speeds up prefill on the base M4 (32 GB).
Baseline: 160–175 tok/s prefill on the GPU at 512–2048-token prompts.

Code: `src/inkling_ane/` (torch restatement, converter, weights), `scripts/inkling_ane_*`.
Env: `.venv` (coremltools 9.0, torch 2.7.0, mlx 0.32.4).

## Progress log

### 2026-11-20 21:20: ANE slice works; projection clears 1.3x; building the prototype

- **GPU cost split** (MLX, one real layer, prefill of 2048 tokens in chunks of C, measured before the
  other agent took the GPU lock). Per token, per layer application:

  | layer | C | full | dense (no MoE) | MoE | projections only |
  |---|---|---|---|---|---|
  | 7 loop/sliding | 512 | 135 µs | 58 µs | 77 µs | 18 µs |
  | 11 loop/global | 512 | 149 µs | 68 µs | 81 µs | 20 µs |
  | 2 plain/sliding | 512 | 134 µs | 47 µs | 83 µs | 20 µs |
  | 5 plain/global | 512 | 167 µs | 76 µs | 98 µs | 22 µs |
  | 7 | 128 | 170 µs | 52 µs | 120 µs | 20 µs |
  | 7 | 2048 | 130 µs | 61 µs | 67 µs | 19 µs |

  Dense work is 36–54 % of GPU time at C ≥ 512. Most of it is attention, convs and elementwise
  work, not the projections. The MoE gets much less efficient below C = 256 (its 3-bit expert
  weights are re-read for every chunk), so the GPU must keep large MoE batches.
- **ANE dense slice** (one full layer application from the layer input to the MoE input `h` and
  `residual`, attention included, KV and conv caches in MLState): **100 % ANE** for every layer
  type. Latency per call, including Python coremltools overhead (fp16 weights):

  | slice | Q=64 | Q=128 | Q=256 |
  |---|---|---|---|
  | plain sliding (L2) | 2.25 ms (35 µs/tok) | 3.56 ms (28 µs/tok) | 7.29 ms (28 µs/tok) |
  | loop pass ≥1 sliding with message gates (L7p1) | 2.71 ms (42) | 4.90 ms (38) | |
  | plain global (L5), 2048-slot cache | 3.49 ms (55) | 6.61 ms (52) | |

  int8 per-channel: L7p1 Q=64 2.40 ms, L2 Q=128 3.29 ms (about 10 % faster).
- **Numerics on the ANE** (two chunks of 64 with carried state vs MLX): cos ≥ 0.99999 (fp16),
  ≥ 0.99996 (int8), for sliding, global, plain and loop-pass slices.
- **Projection:** the ANE needs about 1.75 ms/token for all 48 dense parts (Q=128), which is about
  570 tok/s. The GPU MoE alone needs about 3.65 ms/token at C=512 (fused fast path, 76 µs/layer),
  which caps the GPU at about 274 tok/s. With full overlap the GPU is the bottleneck, at about
  1.5–1.7x the baseline before contention. Still to measure: ANE/GPU memory contention, and the
  real end-to-end speed.

### What failed and why (so far)

- **ANECCompile() FAILED** for the whole slice. The cause was the K=4 short convs, written as 4
  slices of `cat(state, x)`. Fix: a depthwise `conv2d` (static convs). For the dynamic conv
  (global layers), a one-hot depthwise conv makes the 4 shifted copies, then a weighted sum.
- **Global attention over a 2048-slot cache was 3x slow** (9–10 ms per Q=64 call) when the
  relative bias was zero-padded to 2048. Fix: score the far slots (no bias) and the near 1024+Q
  slots (bias) as two matmuls into one softmax: 3.5 ms.
- The recipe's `write_state` count reads `coreml_update_state` in the Python MIL. The writes were
  present all along.

## Dependency graph and schedule

One layer application l on chunk c (rows of one prompt chunk) splits into:

- `A(c,l)` on the ANE: layer input x → attention-input mixing, (message gates), norms,
  projections, k/v convs, clips, banded attention with sinks over the cached K/V, gate, o_proj,
  attn conv, residual, mlp-input mixing, (mlp message gate), post norm → `h`, `residual`.
- `M(c,l)` on the GPU: `x_next = residual + mlp_sconv(MoE(h))` (the mlp short conv is tiny and
  keeps its 3-row state in MLX).

Edges:

- `A(c,l) → M(c,l) → A(c,l+1)`: within a chunk everything is sequential (96 alternating
  stages per chunk).
- `A(c-1,l) → A(c,l)`: K/V cache and the k/v/attn conv tails for layer application l
  (causality across chunks; each layer application has its own MLState set, so loop passes
  do not collide).
- `M(c-1,l) → M(c,l)`: the mlp conv tail (in MLX, ordered by the lazy graph).
- Blocks, messages and pre-loop states are per row: no cross-chunk edges.

So the only parallelism is across chunks, as a wavefront: chunk c can run layer l as soon as
chunk c-1 has finished `A(·,l)`. The lag is one stage, not a whole chunk, so the fill/drain
cost is about one stage per chunk in flight. The schedule used: sweep l = 0..47; for each l,
run chunks c = 0..K-1 in order on the ANE, and submit each `M(c,l)` with `mx.async_eval`. While
the ANE runs `A(c+1,l)`, the GPU runs `M(c,l)`. `A(c,l+1)` waits only for `M(c,l)`, which had K-1
ANE calls of time to finish. With K ≥ 2 chunks in flight and stage times D (ANE) and M (GPU),
the period per layer is K·max(D, M) + handoff, so

  prefill time ≈ 48 · K · max(D_chunk, M_chunk) + fill ≈ N · max(t_ane/token, t_moe/token).

The MoE chunk size sets M: the MoE re-reads all 128 experts' weights for every chunk, so it wants
C ≥ 512 rows. The ANE runs the same chunk as C/128 calls of Q=128 (bigger Q does not help it).
Handoff per stage: one numpy→MLX copy of h and residual (C×2304 fp16 each), one MLX→numpy copy of
x (which waits for that chunk's GPU work), and the Core ML feature conversions inside predict().

### 2026-11-20 21:50: working hybrid prototype (first run 303 tok/s at 2048; repeated runs in "Final results")

- `src/inkling_ane/hybrid.py` runs the whole prefill: 36 Core ML slices (Q=128, fp16, 100 % ANE,
  48 MLState sets) plus MLX MoE (fused fast path) with the wavefront schedule above.
  (superseded by `scripts/inkling_ane_validate.py`; see "Productized hybrid prefill").
- First-load ANE compile of all 36 slices: 43 s. Cached load: 41–73 s (36 models in Python).
- **Measured end to end** (random tokens, final hidden state, no logits; best of 2):

  | N | chunk | overlapped | serial (no overlap) |
  |---|---|---|---|
  | 2048 | 512 | **303 tok/s** (6.75 s) | 188 tok/s |
  | 2048 | 1024 | 237 tok/s | 192 tok/s |
  | 1024 | 512 | 227 tok/s | 183 tok/s |
  | 512 | 512 | 188 tok/s (one chunk, so no overlap) | 187 tok/s |

  (Baseline in the same harness: see below.)
- **Two pipeline bugs that cost the overlap** (worth knowing for any MLX + Core ML pipeline): MLX
  runs one FIFO GPU stream, so *any new lazy op* on an in-flight result (`a.astype(...)`,
  `a[0]`) is queued behind every GPU job submitted since. Reading chunk c+1's input then waits
  for chunk c's MoE, and the pipeline serializes. Fix: put every cast inside the async graph, and
  index after `np.array()`. Proof: async_eval(MoE C=512) + 4 ANE calls take 37.6 ms = max(37.6,
  18.3), not the 55.9 ms sum.
- **Contention** (dense slice L7p1 Q=128 vs MoE C=512, both looping): ANE 4.59 → 5.91 ms/call
  (+29 %), MoE 71.8 → ~79 µs/token (+10 %).
- **Numerics, whole chain vs MLX** (per-step cosine of the layer inputs, ANE fp16 vs MLX bf16):
  min 0.9965 (inside the loop passes, step ~25), 0.9998 at the last layer. The same for chunks of
  128/256/512 and for 48/128/1152 tokens (past the 1024 window), so chunking and states are exact.
  The drift comes from fp16-vs-bf16 rounding compounding over 48 applications, not from the
  pipeline. Final-norm cosine on the 48-token reference: 0.9995.

## Final results (2026-11-20 ~22:30)

All runs used the GPU lock, random tokens and the same harness for both sides. The time counts
from tokens to the final normalized hidden state of the last token (no logits). Each figure is the
best of 2–4 runs; the range covers the runs. Raw logs are in the agent task outputs; the commands
are in `scripts/inkling_ane_prefill.py` and `scripts/inkling_ane_baseline.py`.

| N | MLX upstream (step 512) | MLX + `inkling_fast.fast.apply` | **Hybrid fp16** | **Hybrid int8** | best vs fast |
|---|---|---|---|---|---|
| 2048 | 168 tok/s | 187 tok/s (183 at step 2048) | 257–282 (chunk 512) | **279–295** (chunk 512) | **1.5–1.6x** |
| 1024 | 174 | 196 | 248–257 (chunk 256) | 254–262 (chunk 256) | 1.3x |
| 512 | 181 | 208 | 186 (chunk 256) | — | 0.9x (no gain) |

- **Projected vs measured:** projected 1.5–1.7x at 2048 before contention. Measured 1.5–1.6x
  vs the fast MLX path, and 1.7–1.75x vs upstream MLX.
- Short prompts gain little. The overlap needs ≥ 2 chunks, and the MoE wants chunks of ≥ 256–512
  rows (its per-token cost doubles at 128 rows), so 512 tokens cannot pipeline efficiently.
- ANE time grows from about 4.2 s (serial) to about 5.5–6 s (overlapped) at 2048 tokens because
  of DRAM/fabric contention. With the overlap, the ANE side (5.5 s) is now close to the GPU side,
  so both units are busy.
- int8 slices: slightly faster (less ANE traffic), per-layer cos ≥ 0.99996 vs ≥ 0.99999 for fp16.
- **Output check** (48-token reference prompt, logits from the hybrid's final hidden state through
  the MLX lm_head vs the reference logits): argmax identical, top-5 overlap 4/5, KL 0.0038, logit
  cos 0.99997.

## Decode measurements (asked by the coordinator)

**1. Single-token (Q=1) dense slice on the ANE.** Multifunction packages: `decode` (Q=1) +
`prefill` (Q=128), shared weights (the package is the size of one function: 59 MB fp16 / 30 MB
int8) and the same MLState. 100 % ANE after two Q=1-specific fixes: (a) short convs as window ×
taps summed, (b) the output gate applied in head layout *before* the transpose back (gating after
it fails ANECCompile at Q=1, nowhere else). Per-call wall time includes MLX array → numpy → Core
ML predict → numpy → MLX array with `mx.eval`, and a real data dependency between calls:

| slice | fp16 wall | fp16 predict() alone | int8 wall | int8 predict() alone |
|---|---|---|---|---|
| L2 plain sliding | 1.70 ms | 1.25 ms | 1.11 ms | 0.85 ms |
| L7 loop pass ≥1 (message gates) | 1.76 ms | 1.32 ms | 1.26 ms | 0.91 ms |
| **48 calls / token** | **82–84 ms** | | **53–60 ms** | |

The GPU does the same dense matmuls in about 13 ms/token. **The ANE is 4–6x too slow for decode.**
The int8 slice streams its 27–31 MB of weights in about 0.85–0.9 ms (about 32 GB/s), and there
are about 0.3 ms of handoff per call on top.

**2. Aggregate DRAM bandwidth** (`scripts/inkling_ane_bandwidth.{py,sh}`; ANE: 10 chained
4096↔8192 linears at Q=1, 335 M params; GPU: `mx.sum` over a 1 GiB fp16 array):

| config | ANE alone | GPU alone | together: ANE | together: GPU | sum |
|---|---|---|---|---|---|
| int8 (336 MB/call) | 25.4 GB/s (13.2 ms) | 91.1 GB/s | 25.8 GB/s | 88.1 GB/s | ~114 GB/s |
| fp16 (671 MB/call) | 24.8 GB/s (27.1 ms) | | 25.0 GB/s | 93.5 GB/s | ~118 GB/s |
| pal4 (168 MB/call) | 64.4 GB/s (2.6 ms) | | 61.0 GB/s | 79.2 GB/s | ~140 GB/s |

The combined read rate does exceed the GPU alone (91.6 GB/s): about +25 % with int8/fp16
weights, about +50 % with pal4 (pal4 is the only format where the ANE streams fast: about 129 G
weights/s vs 25 G for int8). The ANE barely slows the GPU's streaming with int8/fp16; with pal4 the
GPU drops to 79 GB/s.

**What this means for a decode split.** The dense projections are 1.04 GB/token (6-bit) on the GPU
in about 13 ms. On the ANE the same weights at int8 are about 1.4 GB/token → about 55 ms; at pal4
about 0.7 GB/token → about 11 ms at best (and pal4 is a 4-bit k-means re-quantization of 6-bit
weights, a fidelity loss). An overlapped split (ANE dense of layer l+1 while the GPU does MoE of
layer l) cannot work inside one token: every layer's dense part depends on the previous layer's MoE
output, and decode has only one row, so there are no chunks to pipeline across. Running them
serially, ANE dense (≥ 11 ms pal4, + 48 × ~0.3 ms handoff ≈ 14 ms extra) + GPU MoE (18–23 ms) is
no better than today's 40 ms, and it is much worse with int8. **Recommendation: keep decode on
the GPU.** The only decode-side ANE use that might pay is speculative-style batching (several
draft rows per call), which turns decode back into small prefills. Not measured.

## What worked, what failed (summary)

Worked: the recipe's MLState-per-tensor, host-side masks, per-layer cosine checks on the ANE;
shifting K/V caches plus a constant relative-position matrix (no gather) for the learned banded
bias; split far/near global attention; one FIFO-safe async MLX pipeline.

Failed / traps:
- ANECCompile failures: (1) short convs as slices of the state window (Q=128) → depthwise conv2d
  and a one-hot shift conv; (2) at Q=1, the output gate after the head transpose → gate in head
  layout.
- Global attention 3x slow when its relative bias was zero-padded to 2048 slots → two score blocks.
- `ct.convert(..., compute_units=CPU_AND_NE)` loads (and ANE-compiles) the model unless
  `skip_model_load=True`. Use that for conversion jobs that should not touch the ANE.
- MLX FIFO stream serialization (above). It cost the whole speedup until fixed.
- tiny Core ML probe models get placed on the CPU, so they do not test the ANE compiler.

## Limits of the prototype and next steps

- **No KV handoff to MLX yet.** Prefill leaves the K/V caches in Core ML MLStates. To decode on
  the GPU afterwards, each slice must also output its k/v rows (or the host must read the states)
  and fill the MLX caches. That is cheap (C×2×1024 fp16 per layer application) but not built.
  The mlp short-conv tails already live in MLX.
- Fixed shapes: global caches hold 2048 slots, so prompts are ≤ 2048 + 128 tokens. Longer prompts
  need a bigger `--s-global` (the global slices get slower: 4096 slots cost 5.4 vs 3.5 ms at
  Q=64). The log-scaling τ of global layers is 1 below 8192 tokens and is not implemented.
- Python host: about 0.5 ms of feature conversion per Q=128 call (192 calls per 512 rows). A
  Swift/ObjC host with IOSurface-backed MLMultiArrays and `outputBackings` would remove most of it.
  Loading 36 models takes 41–73 s in Python (first compile 43 s): fine for a server, bad for
  one-shot CLI use.
- Numerics: the chain drifts to cos about 0.9965 in the loop passes (fp16 ANE vs bf16 MLX, compounding)
  and ends at 0.9998; logits KL 0.004 on the reference prompt. A wider check (greedy agreement
  over many prompts) should come before shipping.

## Productized hybrid prefill (2026-11-20, evening)

### What changed
- `src/inkling_ane/hybrid.py` now runs on the **already-loaded mlx-vlm model**: it uses the model's
  own MoE layers (fused by `inkling_fast`), mlp short convs, embedding, final mixing and norm.
  No weights are duplicated on the GPU.
- **KV/conv handoff.** After the sweep, each of the 48 slices' MLStates is read
  (`MLState.read_state("m_k_cache")` etc.; the compiled names use `_`), and the standard caches
  from `lm.make_cache()` are filled:
  - `RotatingKVCache` (sliding): the last min(N, 1024) keys/values in time order, `offset=N`,
    `_idx=len`. GPU prefill keeps ≥ 1024 keys, and its first decode or next-chunk update trims to
    the window, so after that update the two caches hold the same content in the same ring
    positions.
  - `KVCache` (global): all N positions via `.state`.
  - `ArraysCache` conv tails: k/v/attn tails from the ANE (fp16 → fp32); the mlp tail from MLX.
  Decode then continues on `inkling_fast.decode` unchanged.
- **Prompt limit lifted to 4096** with slice buckets: sliding slices are shared, and global slices
  exist for 2048- and 4096-slot caches (`runtime/ane/prod_int8/s2048`, `s4096`; the smallest
  bucket ≥ N is used). Above 4096 tokens the call falls through to GPU prefill. (Global attention
  log-scaling τ is 1 below 8192 tokens, so up to 8192 only needs another bucket.)
- **Load time:** the slices are precompiled `.mlmodelc` (int8 per-channel), loaded with
  `CompiledMLModel` on a background thread. Loading takes 39–46 s once the ANE compile cache is
  warm; requests arriving before then use GPU prefill.
- **Entry points:**
  - `hybrid.prefill(model, token_ids) -> (cache, last_logits)`.
  - `hybrid.install(model)` wraps `lm.__call__`. It takes a call when the cache is fresh, the batch
    is 1, there are no captures or speculation, and 1024 ≤ n ≤ 4096. Otherwise, and on any ANE
    error (the cache is reset first), it calls the original (GPU) path.
- **Server:** `INKLING_ANE_PREFILL=1 .venv/bin/python scripts/inkling_serve.py ...`
  (needs coremltools: `pip install -e '.[ane]'`). The flag sets
  `PREFILL_STEP_SIZE=4096`, so mlx-vlm hands the whole prompt minus its last token to one call.
  The last token then runs through the decode engine, exactly as with GPU prefill.
  `src/inkling_fast/` is not modified.

### Validation: greedy agreement (`scripts/inkling_ane_validate.py validate`)
10 chat-formatted prompts built from the repo docs, 1019–2048 tokens. Greedy decode of 64 tokens
with the decode engine, after hybrid prefill vs after GPU prefill (step 2048). Calibration: GPU
prefill with step 512 vs step 2048 (pure bf16 reordering, nothing else changed).

| prompt tokens | hybrid vs GPU: agree / first divergence | GPU step 512 vs 2048: agree / first divergence |
|---|---|---|
| 1019 | 6/64 at 6 | 64/64, identical |
| 1136 | 45/64 at 45 | 58/64 at 11 |
| 1249 | 24/64 at 24 | 64/64, identical |
| 1360 | 6/64 at 6 | 6/64 at 6 |
| 1473 | 23/64 at 21 | 23/64 at 21 |
| 1587 | 63/64 at 63 | 64/64, identical |
| 1701 | 43/64 at 43 | 20/64 at 20 |
| 1815 | 30/64 at 11 | 17/64 at 11 |
| 1932 | 54/64 at 54 | 64/64, identical |
| 2048 | 64/64, identical | 6/64 at 6 |
| **total** | **1/10 identical, mean 35.8/64** | **4/10 identical, mean 38.6/64** |

The first-token logits after hybrid prefill match GPU prefill to KL ≤ 7e-9 on every prompt. Every
divergence is a near-tie wording choice (for example "a five-bullet-point summary" vs "a summary
of the document in five"). Two of the hybrid's divergences sit at exactly the positions where GPU
prefill diverges from itself when only the chunk size changes. Conclusion: the hybrid's deviation
is about the size of GPU prefill's own rounding sensitivity. It is a little larger (1 vs 4
identical), as expected from fp16/int8 vs bf16. There is no sign of a cache-handoff bug (that
would show as large first-token KL and garbage, not near-tie paraphrases).

### Benchmark (`scripts/inkling_ane_validate.py bench`, best of 3, under the lock)
Same process and model, decode engine installed (so the GPU path is the current fast path).
Timing covers tokens → filled cache + last-token logits (prefill of N-1 tokens, then the last
token through the decode engine, as in mlx-vlm generate).

| N | GPU prefill | hybrid prefill | speedup |
|---|---|---|---|
| 1024 | 225.7 tok/s (runs 226, 226, 222) | **290.9 tok/s** (291, 284, 286) | **1.29x** |
| 2048 | 200.3 tok/s (200, 200, 200) | **324.9 tok/s** (325, 319, 323) | **1.62x** |

(The GPU baseline is faster than in the morning runs, 187 → 200 tok/s at 2048, thanks to newer
`inkling_fast` patches. The hybrid inherits the faster MoE.)

### Server time to first token (`scripts/inkling_ane_ttft.sh`, under the lock)
`scripts/inkling_serve.py` on port 8799, APC off, 2167-token chat prompt from the repo docs
(unique nonce per request), `max_tokens` 8, after one warm-up request. When the flag is on, the
requests start only after the slices report ready.

| `INKLING_ANE_PREFILL` | TTFT (3 runs) | server log |
|---|---|---|
| 0 | 10.60 / 10.87 / 11.27 s | GPU prefill |
| 1 | **7.79 / 7.80 / 7.68 s** (1.38x) | `hybrid prefill 2166 tokens in 7.5–7.6 s` |

This prompt is just over 2048 tokens, so it uses the 4096-slot global slices. Those are slower
than the 2048-slot ones, so a 2048-token prompt gains more (1.62x in the in-process benchmark).
With a warm compile cache the 42 slices load in 1.4 s (`[inkling_ane] 42 ANE slices ready in 1.4s`).
The first ANE compile after creating or changing the slices takes about 45 s.

### 4096-slot bucket (prompts of 2049–4096 tokens)
Same validation, 4 doc prompts of 2196–3996 tokens:

| tokens | hybrid vs GPU | GPU step 512 vs 2048 |
|---|---|---|
| 2196 | 29/64, first divergence 4 ("The user asks to summarize" vs "The user wants a five-bullet summary") | identical |
| 2795 | 57/64 at 57 | 21/64 at 21 |
| 3394 | identical | identical |
| 3996 | 9/64 at 9 (same position as GPU vs GPU) | 9/64 at 9 |
| total | 1/4 identical, mean 39.8/64 | 2/4 identical, mean 39.5/64 |

First-token KL ≤ 1.6e-7. Prefill: N=3072 hybrid **305 tok/s vs GPU 194 tok/s (1.57x)**;
N=4000 hybrid 308 tok/s. The GPU number at 4000 tokens (61 and 48 tok/s, 65 s) is anomalous: it
is far below its 194 tok/s at 3072, likely memory pressure from the 14 GB model plus the 4000-key
masks. I don't treat it as a fair baseline.

### Summary of the productized path
| | GPU prefill | hybrid | speedup |
|---|---|---|---|
| 1024 tokens | 226 tok/s | 291 tok/s | 1.29x |
| 2048 tokens | 200 tok/s | 325 tok/s | 1.62x |
| 3072 tokens | 194 tok/s | 305 tok/s | 1.57x |
| server TTFT, 2167-token prompt | 10.6–11.3 s | 7.7–7.8 s | 1.38x |

Disk: `runtime/ane` holds 3.8 GB in total. The production set is
`runtime/ane/prod_int8` (1.2 GB of `.mlmodelc`). The rest is the fp16 slice packages (2.0 GB, kept
for a fidelity fallback; deletable) and the small golden sets.

### Known limits / follow-ups
- Prompts > 4096 tokens use GPU prefill. Up to 8192 tokens needs only an `s8192` global bucket
  (6 slices, about 90 MB int8). τ log-scaling starts above 8192.
- Only fresh caches are taken. An APC prefix hit with a long uncached suffix stays on the GPU.
  Supporting a suffix needs the ANE states to be seeded from MLX caches (`MLState.write_state`
  can do it; not built).
- `PREFILL_STEP_SIZE=4096` under the flag also changes GPU chunking for fallback prompts > 4096
  tokens (4096-row chunks instead of 2048). Peak memory is higher there.
- Rare transient "ANE Program Inference error" seen once in the first smoke run. The server path
  resets the cache and falls back to the GPU; the validator retries.
- The server needs coremltools for this path. Without the flag nothing changes.
