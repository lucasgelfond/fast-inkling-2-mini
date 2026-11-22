# Inkling 2 Mini on a 16 GB M5

The main [fast-inkling](../README.md) setup needs ~15 GB of 3-bit weights, so it doesn't fit a 16 GB Mac. This folder
runs [LookAtMyQuant's 8.9 GB build](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB)
(2.72 bpp, 99.2% of the BF16 model on GSM8K / HumanEval / Vision) on a base **M5 MacBook with 16 GB**, with:

- the [DSpark drafter distilled onto that build](https://huggingface.co/shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter) (`mtp_4bit_8p9gb_v2`)
- the untouched 637 MB BF16 **vision tower** (images in chat and in pi)
- **prefix caching** across requests, so agent turns only read what's new
- **tool calling** through the model's own chat template
- **several cached conversations** at once (pi + subagents), served one request at a time
- a **4-bit KV cache** for the 8 global-attention layers, for long context (target: 262k)
- an OpenAI-compatible server and a pi setup (lean prompt, low thinking by default)

Model + drafter + vision = 9.7 GB on disk, hence the pi model name `Inkling-2-Mini-9.6GB`.

## Install

```
uv tool install "git+https://github.com/lucasgelfond/fast-inkling-2-mini#subdirectory=16gb-m5"
inkling-16gb download          # 8.87 GB model + 179 MB drafter + 637 MB vision tower -> ~/.cache/inkling-16gb
inkling-16gb serve             # http://127.0.0.1:8090/v1, model id Inkling-2-Mini-9.6GB
```

Also: `inkling-16gb run "prompt" [--image photo.png]`, `inkling-16gb chat`, `inkling-16gb bench`, and
`--kv-bits 4|8|0`, `--no-drafter`, `--no-vision`, `--effort none|minimal|low|medium|high|xhigh|max`.
The first ever launch spends ~50 s compiling Metal kernels; macOS caches them (later launches: ~6 s).

## Speed (base M5, 16 GB)

Short prompts, greedy, thinking off, 384-token answers:

| | prose | code | math |
|---|---|---|---|
| serial decode | 26.5–29.4 tok/s | 28.8–30.3 | 29.0–30.7 |
| **drafted** (default) | 27.2–30.0 (2.2 tok/check) | **42.2–47.2** (3.8) | **40.3–50.2** (3.5) |

- Prefill: ~340 tok/s at short context. A new pi session with the lean prompt (~1k tokens) answers in a few seconds.
- **Prefix cache**: a follow-up turn that echoes `reasoning_content` (pi does) continues the live cache - only the new
  tokens are read. Otherwise it rolls back to a checkpoint at the end of the previous prompt. Measured: turn 2 of a
  4.8k-token chat 18 s -> 6 s; a 155-token chat turn 1.2 s.
- **Several conversations** (a pi session plus subagents) each keep their own cache slot (`--slots 4`): a request
  goes to the slot that can reuse the most of its prompt, a new conversation takes a free or the least recently used
  slot, and idle slots are held to `--slot-budget-gb 1.5` (a 4k-token conversation costs ~0.5 GB). Requests are
  served one at a time; concurrent agents queue.
- Ranges are across runs; the M5 throttles after back-to-back runs.

## Long context (base M5, 16 GB, 4-bit KV)

| prompt | first read | follow-up turn (cached) | decode | peak GPU memory |
|---|---|---|---|---|
| 16k tokens | 58 s | 4 s | ~14 tok/s drafted | ~11 GB |
| 31.6k tokens | 99 s (~330 tok/s) | - | 10.7–11.8 tok/s drafted | 11.4 GB |
| 100k tokens | 491 s (~204 tok/s) | **29 s** (100,484 of 100,505 reused) | **12.3 tok/s** serial | 12.07 GB |
| 262k tokens | not yet measured: needs the GPU limit raised, see below | | | |

Above 32k tokens the server decodes without the drafter: a drafted verify block re-reads the whole KV cache once per
drafted row, so at 100k it is slower (7.2 tok/s) than plain decoding (12.3 tok/s). Below 32k drafting stays on.

## Memory

| | |
|---|---|
| weights | 8.83 GB (+0.64 GB vision tower while encoding; unloaded afterwards) |
| short-context process footprint | ~9.4 GB |
| KV cache, 8 global layers | 4-bit: 9.2 KB/token (2.4 GB at 262k); 8-bit: 17 KB; fp16: 32 KB |
| KV cache, 40 sliding-window layers | fp16, capped at ~1.3k positions (~0.2 GB) |
| drafter + its context | 179 MB + 4k-token window |

macOS lets the GPU use ~12.7 GB of a 16 GB Mac by default. Up to ~131k tokens fits in that. For 262k (estimated peak
~13.5 GB) raise it (resets on reboot; quit Chrome and other big apps first):

```
sudo sysctl iogpu.wired_limit_mb=14336
```

## Quality of the 4-bit KV cache

Teacher-forced on 30k tokens of real code and docs (last 512 positions, vs the same model with fp16 KV):

| KV | next-token NLL | KL vs fp16 (mean / p99) | top-1 agreement |
|---|---|---|---|
| fp16 | 0.5099 | - | - |
| 8-bit | 0.5135 | 0.0020 / 0.017 | 99.4% |
| **4-bit** | 0.5107 | 0.0077 / 0.092 | 98.6% |

For scale: the 8.9 GB weight quantization itself is KL 0.0208 vs BF16. 4-bit KV adds less than that.

## pi

1. Add the provider to `~/.pi/agent/models.json` (under `"providers"`):

```json
"inkling-16gb": {
  "name": "Inkling 2 Mini (local, 16 GB)",
  "baseUrl": "http://127.0.0.1:8090/v1",
  "apiKey": "inkling-16gb-local",
  "api": "openai-completions",
  "models": [{
    "id": "Inkling-2-Mini-9.6GB",
    "name": "Inkling 2 Mini 9.6GB (local)",
    "reasoning": true,
    "input": ["text", "image"],
    "contextWindow": 131072,
    "maxTokens": 8192,
    "compat": {
      "supportsDeveloperRole": true,
      "supportsReasoningEffort": true,
      "supportsUsageInStreaming": true,
      "supportsStrictMode": false,
      "requiresThinkingAsText": false,
      "maxTokensField": "max_tokens"
    },
    "samplingParams": {},
    "thinkingLevelMap": {"off": "none", "minimal": "minimal", "low": "low", "medium": "medium", "high": "high", "xhigh": "xhigh", "max": "max"},
    "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0}
  }]
}
```

2. Copy [`pi/inkling-16gb.ts`](pi/inkling-16gb.ts) to `~/.pi/agent/extensions/`. With it, pi handles the server:
   selecting the model (or starting pi on it) launches `inkling-16gb serve` if it isn't running and holds requests
   until it's ready (footer: "Inkling: loading model…"); switching to another model or quitting pi stops it, giving the
   ~9.5 GB back. It also switches thinking to low on selection and restores your level when you switch away. For
   startup, also add `"modelThinkingLevels": {"inkling-16gb/Inkling-2-Mini-9.6GB": "low"}` to
   `~/.pi/agent/settings.json`. Server log: `~/.cache/inkling-16gb/server.log`.
3. Lean prompt: install [`lean.ts`](../docs/pi-lean.md) - local models get a ~1k-token prompt and four tools instead
   of ~10k tokens.

`requiresThinkingAsText: false` matters: pi then echoes thinking back as `reasoning_content`, which re-renders
token-identically, so each turn continues the live cache instead of re-reading the reply.

## How it works

`src/inkling_16gb/`:

- `engine.py` - loads the packed model + drafter; **chunked prefill** (512 tokens: one-pass prefill runs a 16 GB Mac
  out of GPU memory at ~4k tokens; chunking through the same cache is token-identical); **prefix cache** (live
  continuation, or a checkpoint: full-attention KV needs no copy since generation only writes past it, while the
  sliding-window KV and 4-slot conv states are copied; the attention kernels mask by absolute position so a restore is
  exact - verified token-identical); the drafter is rebuilt each request from a 4k window of target hidden taps (the
  drafter only affects speed, never output); output parsing into thinking / text / tool calls; image row injection.
- `qkv.py` - 4/8-bit KV for the 8 global-attention visits. The runtime's own Metal kernels (decode flash-decoding and
  prefill FlashAttention-2) are captured from their builders and rewritten only at the K/V loads to dequantize on load
  (kernel output vs fp16 kernel on the same dequantized values: max |diff| = 0). Decode adds a split-K variant (up to
  32 splits) - 2.4x faster at 262k keys. Buffers grow geometrically (the stock cache grows by 256 and copies
  everything, which is quadratic at long context).
- `vision.py` - the BF16 vision tower + projector in MLX, bit-identical to mlx-vlm's; weights are range-read out of
  the original 19.5 GB checkpoint (637 MB). Details: [docs/vision.md](docs/vision.md).
- `serve.py` - OpenAI-compatible `/v1/chat/completions` (streaming, tools, images, `reasoning_effort`,
  `usage.prompt_tokens_details.cached_tokens`).

Tried and dropped: fixed draft lengths 2/3/4/6 and a 3-tier adaptive block (fixed-3 wins prose by ~4 tok/s, loses
code/math). fast-inkling's kernel fusions target MLX affine-quantized layers and don't apply to trellis-packed weights.
