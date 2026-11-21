# fast-inkling-2-mini

Fast local inference for [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/). I tried this on my 32GB M4 base MacBook Pro. This got me speedups both over llama.cpp and over mlx-serve.

| | llama.cpp | Standard MLX | **fast-inkling** |
|---|---|---|---|
| Decode (128 tokens) | 12.8 tok/s | 18.6–19.6 | **26.8–28.8** |
| Prefill (512 tokens) | 37.6 | 178–184 | **222–230** |


Apache-2.0. The loop model comes from [internet-projects/inkling2-mlx](https://github.com/internet-projects/inkling2-mlx), a dependency that is not vendored here.
