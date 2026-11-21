# fast-inkling-2-mini

Fast local inference for [Inkling 2 Mini](https://thinkingmachines.ai/news/inkling-2-mini/). I tried this on my 32GB M4 base MacBook Pro. This got me speedups both over llama.cpp and over mlx-serve.

| | llama.cpp | Standard MLX | **fast-inkling** |
|---|---|---|---|
| Decode (128 tokens) | 12.8 tok/s | 18.6–19.6 | **26.8–28.8** |
| Prefill (512 tokens) | 37.6 | 178–184 | **222–230** |


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
