# fast-inkling-2-mini

Fast local inference for [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) (Thinking Machines; looped MoE, 26.4B total / 3.4B active parameters) on Apple silicon.

It is built and measured on a base M4 MacBook Pro (32 GB, 120 GB/s). The kernels use only standard Metal features, so they run on all M-series chips.

## Results (base M4, 3-bit weights, batch 1, greedy, best of 3)

| | llama.cpp b12600 | Standard MLX | **fast-inkling** | vs llama.cpp | vs standard MLX |
|---|---|---|---|---|---|
| Decode (tg128) | 12.8 tok/s | 18.6–19.6 | **26.8–28.8** | **2.1–2.2×** | **1.4–1.5×** |
| Prefill, 512 tokens | 37.6 | 178–184 | **222–230** | **5.9–6.1×** | **1.2–1.3×** |
| Prefill, 2048 tokens (ANE + GPU) | 37.8 | 168 | **325** | **8.6×** | **1.9×** |

- llama.cpp uses Unsloth UD-Q3_K_XL (`llama-bench -p 512/2048 -n 128`). Standard MLX is mlx-vlm with the loop model and no changes.
- Server, fresh 2,167-token prompt: 7.6 s time to first token with the ANE hybrid, 11.0 s on the GPU only.
- Prefix cache: a repeated 4.4K-token context goes from 30 s to 0.9 s.
- Greedy output matches unmodified MLX. bf16 scales cost 0.25 points of top-1 accuracy.
- 4-bit (`mlx-community/Inkling-2-Mini-oQ4e-mtp`, TML's NVFP4 experts): 21.6–22.7 tok/s decode, same prefill, better quality.

Full write-up, the bandwidth ceiling, and what did not work (DSpark speculation, LoopCD, ANE decode): [docs/inkling-fast.md](docs/inkling-fast.md).

## What is here

- `src/inkling_fast/`: the engine.
  - `decode.py`: each of the 48 layer applications runs as two `mx.compile`d functions with explicit state. It covers decode, 1–16-token verify and compiled prefill chunks.
  - `fast.py`: fused attention projections and fused expert gate/up, bf16 scales, routing.
  - `kernels.py`: Metal kernels for message gates, attention-residual mixing with fused RMSNorm, the sliding-layer post-projection, a ring-buffer attention mask and router top-k.
  - `moe_v4.py`: a fused one-token 3-bit MoE kernel (89 GB/s vs 76 GB/s for MLX's `gather_qmv`).
  - `oq.py`: loads oMLX oQ checkpoints (fused attention, NVFP4 experts) without rewriting files.
- `src/inkling_ane/`: ANE + GPU hybrid prefill. Dense layer slices run in Core ML on the Neural Engine, pipelined against the MoE on the GPU. See [docs/inkling-ane-prefill.md](docs/inkling-ane-prefill.md).
- `tensorfold/inkling2-metal.patch`: an Inkling 2 Mini port for [TensorFold](https://github.com/ashhart/TensorFold) (apply to commit `99d2384`). Greedy output matches the reference; 25 tok/s decode from its dev tool. See [docs/tensorfold-inkling2-mini.md](docs/tensorfold-inkling2-mini.md).
- `scripts/`: the server, benchmarks, correctness and quality checks, profiling and ANE experiments.
- `benchmarks/`: raw DSpark speculation runs (generated text removed).

## Setup

```sh
uv venv .venv && uv pip install -e .            # add '.[ane]' for the ANE hybrid prefill
hf download netprojects/Inkling-2-Mini-MLX-3bit --local-dir weights/inkling-2-mini/mlx-3bit
```

Benchmark (standard pp512 / tg128):

```sh
PYTHONPATH=src .venv/bin/python scripts/inkling_compare.py fast   # or: mlx
```

Serve (OpenAI-compatible, 127.0.0.1 only, prefix cache on):

```sh
scripts/inkling-fast-serve weights/inkling-2-mini/mlx-3bit 8080
```

The ANE prefill needs Core ML slices converted from the weights (`scripts/inkling_ane_convert_all.sh`, written to `runtime/ane/`). Without them the server prefills on the GPU.

Environment toggles (all on by default unless noted): `INKLING_BF16_SCALES`, `INKLING_BF16_HEAD`, `INKLING_MIX`, `INKLING_POSTPROJ`, `INKLING_RING`, `INKLING_PREFILL_LAST_ONLY`, `INKLING_ATTN_BITS=4` (off, lossy).

## License

Apache-2.0. The loop model comes from [internet-projects/inkling2-mlx](https://github.com/internet-projects/inkling2-mlx) and is a dependency, not vendored here.
