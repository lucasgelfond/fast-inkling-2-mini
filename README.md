# fast-inkling-2-mini

Fast local inference for [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/). I tried this on my 32GB M4 base MacBook Pro. This got me speedups over llama.cpp, standard MLX and the other engines I tried.

| Engine | Weights | Decode (tok/s, 128 tokens) | Prefill (tok/s, 512 tokens) |
|---|---|---|---|
| **fast-inkling** | 3-bit | **26.8–28.8** | **222–236** |
| oMLX 0.8.1 | oQ3e-mtp | 22.4 | – |
| Standard MLX (mlx-vlm) | 3-bit | 18.6–19.6 | 178–184 |
| mlx-lm ([#2086](https://github.com/ml-explore/mlx-lm/pull/2086)) | original (NVFP4 experts) | 14.8 | 198 |
| llama.cpp b12600 | Unsloth UD-Q3_K_XL | 12.8 | 37.6 |
| TensorFold ([#4337](https://github.com/ashhart/TensorFold/pull/4337)) | original (NVFP4 experts) | 11.6\* | ~41\* |


The loop model comes from [internet-projects/inkling2-mlx](https://github.com/internet-projects/inkling2-mlx), a dependency that is not vendored here.

## Install

```
uv tool install git+https://github.com/lucasgelfond/fast-inkling-2-mini
inkling-fast download
inkling-fast serve --port 8080
```

I added into Pi with this configuration:


```
"inkling-fast-local": {
  "name": "Inkling 2 Mini, inkling-fast (local)",
  "baseUrl": "http://127.0.0.1:8080/v1",
  "apiKey": "inkling-fast-local",
  "api": "openai-completions",
  "models": [
    {
      "id": "Inkling-2-Mini",
      "name": "Inkling 2 Mini",
      "reasoning": true,
      "input": ["text"],
      "contextWindow": 32768,
      "maxTokens": 16384,
      "compat": {
        "supportsDeveloperRole": false,
        "supportsReasoningEffort": true,
        "supportsUsageInStreaming": true,
        "supportsStrictMode": false,
        "maxTokensField": "max_tokens"
      },
      "samplingParams": {},
      "thinkingLevelMap": {
        "minimal": "minimal",
        "low": "low",
        "medium": "medium",
        "high": "high",
        "xhigh": "xhigh",
        "max": "max"
      },
      "cost": { "input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0 }
    }
  ]
}
```
