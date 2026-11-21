# Inkling 2 Mini on Metal: the concrete plan for slices H1, M1 to M5

This plan finishes the Metal half of the draft plan in pull request #4274 (`docs/design/inkling-2-mini.md` on that branch).
It is local work for review. It claims no slice, and it was not posted to the issue, the pull request or ATC.
The maintainer's seven decisions in #4274 stand. One deviation is stated under "Checkpoint" below.

## Contract

- **Behavior.** `tensorfold` loads `Inkling2LoopForConditionalGeneration` (`inkling2_loop`) on Metal and runs text decode.
  Before this work the family refused `inkling2_loop`, and no backend served any Inkling 2 checkpoint.
- **Target.** TensorFold `main` at `99d2384`, family `inkling2`, backend Metal, one rank, a base M4 (10-core GPU, no tensor units), 32 GB.
- **Owning modules.** `zig/src/families/inkling2/` (config, loop plan, Metal model, state, forward, backend),
  `zig/kernels/metal/inkling2/` (the family's kernels), `zig/src/native/` (registration), `zig/tests/inkling2/` (dev tool).
- **Shared blocks reused.** `core/row_projection.zig` (affine 3-8 bit row matmuls with MLX's arithmetic) for every BF16-table
  projection, `core/checkpoint_metal.zig` and `core/quant/mlx_affine.zig` for the views, `lanes` for serving.
- **Correctness check.** The 48-token fixture in the local reference set: per layer-visit hidden cosine of at least 0.999,
  and the 16 greedy tokens equal. Then the engine's own exactness (prompt chunk equals decode rows, drafted equals plain).
- **Performance metric.** Decode and prompt tok/s on the same Mac against mlx-vlm (23.9 / 165), oMLX (22.4) and llama.cpp
  (12.0 / 36), the speed bar being decode no slower than a standard server (#4274, decision 6).
- **Hardware.** One base M4. An M5 (tensor-unit prompt path) and any CUDA chip are not available here.

## Checkpoint

Decision 4 says Metal starts from the original checkpoint (BF16 dense weights, NVFP4 and MXFP8 experts).
This Mac has about 21 GB free, and that checkpoint is about 20 GB, so it is not downloaded.
This work uses a local MLX affine conversion instead (`mlx-3bit`): attention 6-bit, experts 3-bit (layers 1-19; layers 0 and
20-23 at 4-bit), embedding and head 4-bit, all group 64. Layers 0-19 store FP32 expert scales and biases; layers 20-23 store BF16 ones.
MLX promotes the expert matmuls to FP32 when the tables are FP32 (FP32 activations and outputs); the kernels follow both.
The reference hidden states come from this same checkpoint, so the comparison is like for like.
Serving the original NVFP4/MXFP8 checkpoint is slice M3 below and stays open.

## The model, as the engine runs it

From `config.json` (`text_config`) and the MLX loop module that produced the reference:

- 24 stored layers, hidden 2,304. Layers 6 to 17 run three passes: 48 layer visits a token.
- The visit plan: layers 0-5, then 6-17 three times, then 18-23. Visit `v` of layer `i` in pass `p` uses cache slot `i` for
  pass 0 and `24 + (p - 1) * 12 + (i - 6)` after. Each slot holds its own KV and four conv states.
- Attention residuals replace residual adds: a block boundary every 6 layers. A layer's attention input is a learned-query
  softmax over the finished blocks plus the running state (one query, or three for q/k/v on a global layer).
  The MLP input is the same with one query. Keys are RMS-normalized states; the values are the normalized states too.
- Pass 2 and 3 restart from the pre-loop state. Their messages read the previous pass's block at layer 12 and its final
  output, with four learned queries a loop layer (q, k, v, MLP).
- Message gates: `(1 - tanh(Wf g)) * h + tanh(Wg g) * m`, `g = Wd [rms(h); rms(m)]`, rank 64, FP32 except `Wd`'s BF16 input.
- Attention: 24 query heads, 8 KV heads of 128. Layers 5, 11, 17, 23 global; the rest slide over 1,024 tokens.
  q is L2-clipped at 22.63 after a per-head scale; k is L2-clipped; v is RMS-normalized without a gain.
  A rank-16 relative bias `r . rel_proj[:, dist]` for distances under 1,024 (L2-clipped at 8 after a scale).
  A query-dependent sink logit with a zero value. Global layers multiply q and r by `1 + 0.1 ln(max(pos / 8192, 1))`.
  The output is gated by `sigmoid(gate_proj(x))`.
- Short convolutions, 4 taps, `(conv + x) / sqrt(5)`: on k and v (sliding layers), after attention and after the MLP.
  Global layers' k and v use an input-dependent conv: static taps plus a per-head `weight_proj(x)`, `/ 3`.
- 128 experts, top 8 by `sigmoid(score) + bias`, weights normalized and scaled by 8, softcapped SwiGLU
  (`tanh(g/4)*4*sigmoid(g) * tanh(u/25)*25`). No shared expert.
- Final: an attention residual over the four blocks and the last output, RMS norm, head, softcap 30, 200,058 real tokens.

## Slices

| Slice | What, on this plan | Status here |
|---|---|---|
| H1 | `config.zig` accepts `inkling2_loop` with `loop_config` (refuses `is_varloop`), taps up to 48 visits; `loop.zig` builds the visit plan. | done |
| H2 | Names for the gates and the message queries in the MLX layout (`*_message_gate.*`, `message_input.query`). | MLX names done; official names open |
| M1 | Attention: sliding and global, sink, relative bias, log scaling, L2 clips; decode rows and prompt rows (pre-M5 only). | done, no contract |
| M2 | Glue: embedding + norm, attention residual mixes, short convs (static and dynamic), router, softcapped SwiGLU, combine, head. | done, no contract |
| M3 | Routed experts. Here: MLX affine 3/4-bit with FP32 or bf16 tables (a family kernel). The NVFP4/MXFP8 path stays open. | MLX done |
| M4 | The loop kernels: message attention and the four rank-64 gates. | done, no contract |
| M5 | The engine: 48 visit slots, the lane backend, the Metal registration, text only, images refused. | forward done; lanes open |
| M6-M8 | DSpark, images, the served receipt and recipe. | not started |

Kernel contract: each new kernel needs a Metal contract registration (`zig/tests/contract/metal_main.zig`) and launch records.
That is not done in this local work; it is listed under remaining work.

## Milestones and notes

Kept in order as the work proceeds.

1. **Toolchain.** Zig 0.17.0 (ziglang.org tarball), Xcode Metal toolchain. `zig build native -Dcpu=apple_m1` builds in 3 min 40 s;
   `zig build test-inkling2-host` passes 24, skips 4; `zig build test-lfm2` (LFM2 family GPU checks on this Mac) passes.
2. **Host slice H1.** `config.zig` parses `inkling2_loop` (loop_config, `is_varloop` refused, MLX affine quantization_config,
   taps by visit); `loop.zig` builds the 48-visit plan. `zig build test-inkling2-host`: 28 pass, 4 skip (4 new tests).
3. **Weights load.** `families/inkling2/metal/` loads every tensor of the `mlx-3bit` checkpoint with checked dtypes and
   shapes in 7.3 s. Finding: layers 0-19 store FP32 expert tables, layers 20-23 store BF16 tables; MLX's arithmetic
   differs (FP32 vs BF16 expert outputs), and the kernels take both.
4. **One layer matches.** The first full forward ran but drifted from layer 0. An op-by-op comparison against the MLX
   layer (lazy load, layer 0 only) found the bug: a sliding layer outside the loop read k and v from unwritten input rows.
   After the fix every layer-0 op matches at cos >= 0.99998 (q, k, v, r, gate, sink, both convs, clips, KV, attention,
   o_proj, the attention conv, the MLP mix, the norm, all 384 picks, the experts and the output).
5. **Full forward and greedy match.** On the 48-token fixture: last-position logits cos 0.999936 (max abs diff 0.71 after
   the softcap), the same top token, and the 16 greedy tokens equal (16 of 16). Mean per-row hidden cosine is at least
   0.995 at every one of the 48 visits and 0.9989 after the final norm. The worst single row drops to 0.977 (layer 13).
   That is not an arithmetic error: at layer 1, rows 1, 6 and 16 pick a different 8th expert than MLX, with 8th/9th
   router-key margins of 1.7e-4 to 8.9e-4, while every input to the router matches at cos >= 0.99997. So the 0.999
   per-row bar is met only where routing agrees; matching it everywhere would need MLX's exact BF16 rounding order.
6. **Speed (dev tool, not a served receipt).** `tf-inkling2-run`, base M4, `mlx-3bit`, greedy, one stream, plain decode
   (no drafter), timed with the GPU lock held; other agents' benchmarks shared the machine between runs.

   | Engine | Decode tok/s | Prompt tok/s |
   |---|---|---|
   | TensorFold native (this work) | 25.0 (63 tokens after the 48-token fixture; 23.7-24.2 in earlier runs) | 48.0 (one cold 512-token chunk) |
   | mlx-vlm, best local path | 23.9 | 165 |
   | oMLX | 22.4 | - |
   | llama.cpp | 12.0 | 36 |

   The harnesses differ: these are TensorFold's dev-tool timings, not `tools/bench_openai.py` against a served engine.
   Profile of the 512-token prompt: without the MoE it runs at 199 tok/s, so about 80% of prompt time is the routed experts.
   Tried and lost: grouping the picks by expert on the GPU (a sort, then each weight chunk read once for up to 8 slots,
   same bits as the per-slot kernel): 39-40 tok/s against 45-48 for the per-slot kernel. It was removed.

## Remaining work

- **M5, the served engine.** A `lanes.backend.Backend` for the family (one-row rounds first), `native/inkling2.zig` and
  the `inkling2_loop` entry in `native/metal.zig`, prompt-cache snapshots of the 48 slots, admission by memory class,
  image and audio refusals with tests. Then a served receipt with `tools/bench_openai.py` and cold 2k-64k prompts.
- **Prompt path.** The experts need a tiled prompt kernel (simdgroup matrices, FP32 tables) to approach MLX's 165 tok/s;
  the core `simd_qmm_bits` tiles read bf16 tables only. The attention prompt path is a per-row loop, fine at 512 rows.
- **KV memory.** Every slot keeps every position. The 40 sliding visits need only 1,024 positions (a ring).
- **Kernel contract.** None of the 15 family kernels is registered in `zig/tests/contract/metal_main.zig` or in the
  launch records yet; that gate is required before a pull request.
- **Exactness gates.** Prompt chunk versus decode rows, resumed versus fresh, concurrent versus solo: not run yet.
  The decode and prompt expert kernels share one arithmetic, so a row's bits do not depend on the chunk.
- **Checkpoint per decision 4.** The original NVFP4/MXFP8 checkpoint (slice M3) and DSpark (M6) are not started; the
  20 GB download does not fit this Mac's free disk.
- **Process.** No ATC claim or radar session was made (local work, no publication); a pull request would need both.

## Files

- Host: `zig/src/families/inkling2/config.zig` (loop config, MLX affine format), `loop.zig`, `config_test.zig`,
  `host_test.zig`, `fixtures/mini_mlx_config.json`.
- Metal family: `zig/src/families/inkling2/metal/` (`metal.zig`, `geometry.zig`, `weights.zig`, `kernels.zig`,
  `model.zig`, `state.zig`, `encode.zig`, `layer.zig`, `forward.zig`).
- Kernels: `zig/kernels/metal/inkling2/inkling2.metal`, `inkling2_attn.metal`, `inkling2_experts.metal`;
  registered in `zig/kernels/metal/sources.zig`.
- Engine root and tool: `zig/src/tensorfold.zig` (`inkling2_metal`), `zig/build/metal_targets.zig` (`tf-inkling2-run`),
  `zig/tests/inkling2/run.zig`.
- Checks: `zig build test-inkling2-host` 29 pass, 4 skip; `python3 tools/zig/lean_check.py` 168 problems, equal to the
  clean baseline (0 new); `zig fmt --check` and `git diff --check` clean.

## Progress note (for relay)

- Reached: greedy matches (16/16 tokens), last logits cos 0.999936, decode 25.0 tok/s on the base M4 (MLX best 23.9),
  prompt 48 tok/s (MLX 165). Dev-tool timings, not yet a served receipt.
- Blocker: none hard. The served path (lane backend, Metal registration) and a tiled prompt-expert kernel are the next work.
