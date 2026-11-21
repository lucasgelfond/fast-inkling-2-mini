# fast-inkling-2-mini

Fast local inference for [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/) (Thinking Machines; looped MoE, 26.4B total / 3.4B active parameters) on Apple silicon. Download the 3-bit weights and run it.

## Quick start

```sh
uv tool install git+https://github.com/lucasgelfond/fast-inkling-2-mini
inkling-fast download              # 3-bit weights, 14.7 GB, into the Hugging Face cache
inkling-fast chat                  # interactive chat
inkling-fast run "Explain RoPE"    # one prompt
inkling-fast serve --port 8080     # OpenAI-compatible server on 127.0.0.1
inkling-fast bench                 # prefill and decode speed (add --baseline for standard MLX)
```

Needs an M-series Mac and Python 3.11+. It is tested on 32 GB; the weights alone are 14.7 GB. The weights are [`netprojects/Inkling-2-Mini-MLX-3bit`](https://huggingface.co/netprojects/Inkling-2-Mini-MLX-3bit), pinned to one revision. Use `--model DIR` for a local copy.

The server uses the model id `Inkling-2-Mini` and keeps an exact prefix cache, so a repeated system prompt or chat history skips prefill.

## Speed (base M4 MacBook Pro, 32 GB, 3-bit, batch 1, greedy)

| | llama.cpp | Standard MLX | **fast-inkling** |
|---|---|---|---|
| Decode (128 tokens) | 12.8 tok/s | 18.6–19.6 | **26.8–28.8** |
| Prefill (512 tokens) | 37.6 | 178–184 | **222–230** |

- llama.cpp b12600 runs Unsloth UD-Q3_K_XL. Standard MLX is mlx-vlm with the loop model and no changes.
- Greedy output matches standard MLX. The one approximation is bf16 quantization scales, which cost 0.25 points of top-1 agreement.

## How it works

- `decode.py`: each of the 48 layer applications runs as two `mx.compile`d functions with explicit state, for decode and for compiled prefill chunks.
- `fast.py`: fused attention projections, fused expert gate/up, bf16 scales.
- `kernels.py`: Metal kernels for the loop's message gates, attention-residual mixing with a fused RMSNorm, the sliding-layer post-projection, a ring-buffer attention mask and router top-k.
- `moe_v4.py`: a fused one-token 3-bit MoE kernel (89 GB/s against 76 GB/s for MLX's `gather_qmv`).

Details, the bandwidth ceiling and what did not work: [docs/how-it-works.md](docs/how-it-works.md).

## License

Apache-2.0. The loop model comes from [internet-projects/inkling2-mlx](https://github.com/internet-projects/inkling2-mlx), a dependency that is not vendored here.
