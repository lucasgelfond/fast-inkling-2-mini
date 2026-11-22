"""Inkling-2-Mini-LookAtMyQuant-8.9GB + its tuned DSpark drafter, set up to fit a 16 GB Mac.

What this adds on top of the LookAtMyQuant runtime:

* Prefix caching across requests (agent loops resend the whole conversation every turn):
    - live continuation: the KV cache left by the last request is reused as-is when the new prompt extends the
      exact tokens it holds (prompt + generated reply, which the chat template re-renders token-identically when the
      client echoes `reasoning_content`);
    - checkpoint: otherwise it is rolled back to a snapshot taken at the end of the previous prompt's stable prefix.
      Full-attention KV needs no copy (generation only writes past it); sliding-window KV (<=1.3k positions per slot)
      and the 4-slot causal-conv states are copied. The attention kernels mask by absolute position (`kv_start`), so
      a restored buffer is exact.
    - the drafter is rebuilt each request from a recorded history of target hidden taps; drafter state only affects
      speed (every token is verified by the target), never output.
* Chunked prefill (512 tokens). One-pass prefill runs a 16 GB Mac out of GPU memory at ~4k tokens; chunking through
  the same cache is exact (verified token-identical).
* A 512 MB MLX buffer-cache cap (12.4 GB -> 9.4 GB process footprint) and a warmup at startup (Metal JIT).
* Output parsing into thinking / text / tool calls (`<|content_invoke_tool_json|>`), tool declarations and tool
  results through the model's own chat template, and image embedding injection (see vision.py).
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import mlx.core as mx

MODEL_REPO = "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-8.9GB"
MODEL_REV = "11a9b0f151e066c0f75a30c4b20ab5c48862a528"
DRAFTER_REPO = "shinmegamiboson/Inkling-2-Mini-LookAtMyQuant-drafter"
DRAFTER_REV = "adac74d1cf00ed68a19a77233a62bd6dcc6c2607"
# Distilled onto the 8.9 GB build: held-out tokens/check 3.26 -> 3.53 vs the default mtp_4bit.safetensors.
DRAFTER_FILE = "mtp_4bit_8p9gb_v2.safetensors"

HOME = Path(os.environ.get("INKLING16_HOME", Path.home() / ".cache" / "inkling-16gb"))
MODEL_DIR = HOME / "inkling-8.9gb"
DRAFTER_DIR = HOME / "drafter"

EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")
TAPS_WINDOW = 4096   # drafter context window (target taps kept + drafter full-attention layer); speed-only
DRAFT_WINDOW = 4096
PREFILL_CHUNK = 512  # 1024 is ~10% faster prefill but peaks ~0.3 GB higher; 512 keeps headroom under the 12.7 GB GPU limit

MSG_MODEL, END_MSG, EOS = 200001, 200010, 200006
C_TEXT, C_THINK, C_TOOL = 200004, 200008, 200049
IMAGE_PLACEHOLDER = 200054


def download() -> tuple[Path, Path]:
    """Fetch (or verify/resume) the model (8.87 GB), the matching drafter (179 MB) and the vision tower (637 MB)."""
    from huggingface_hub import snapshot_download

    snapshot_download(MODEL_REPO, revision=MODEL_REV, local_dir=MODEL_DIR)
    snapshot_download(DRAFTER_REPO, revision=DRAFTER_REV, local_dir=DRAFTER_DIR,
                      allow_patterns=["config.json", "manifest.json", DRAFTER_FILE])
    from .vision import download_vision

    download_vision()  # 637 MB BF16 vision tower + projector, range-read out of the original checkpoint
    return MODEL_DIR, DRAFTER_DIR / DRAFTER_FILE


def _common_prefix(a, b) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


# ----------------------------------------------------------------------------------------------- cache bookkeeping

class _TapHistory:
    """Target hidden taps `[T, 4, H]` for every cached position, kept as chunks (no O(N) concat per decode round)."""

    def __init__(self, window: int = 4096):
        self.chunks: list[tuple[int, mx.array]] = []
        self.length = 0
        self.window = window  # the drafter only needs recent context; older taps are dropped (speed-only effect)

    def clear(self):
        self.chunks, self.length = [], 0

    def truncate(self, n: int):
        keep = []
        for s, a in self.chunks:
            if s >= n:
                break
            keep.append((s, a if s + a.shape[0] <= n else a[: n - s]))
        self.chunks, self.length = keep, min(self.length, n)

    def record(self, start: int, taps: mx.array):
        if start > self.length or not self.chunks:  # starts past what we hold (windowed prefill): restart from here
            self.chunks, self.length = [], start
        self.truncate(start)
        T = int(taps.shape[0])
        if T > self.window:
            taps, start = taps[T - self.window:], start + T - self.window
        self.chunks.append((start, taps.astype(mx.float16)))
        self.length = start + int(taps.shape[0])
        lo = self.length - self.window
        while len(self.chunks) > 1 and self.chunks[1][0] <= lo:  # drop whole chunks that fell out of the window
            self.chunks.pop(0)

    def get(self, n: int) -> tuple[mx.array | None, int]:
        """Taps for positions [start, n) where start = max(oldest kept, n - window). Returns (taps, start)."""
        if n == 0 or self.length < n or not self.chunks:
            return None, n
        lo = max(self.chunks[0][0], n - self.window)
        parts = []
        for s, a in self.chunks:
            if s >= n or s + a.shape[0] <= lo:
                continue
            a0, a1 = max(lo, s) - s, min(n, s + a.shape[0]) - s
            parts.append(a[a0:a1])
        if not parts:
            return None, n
        return (parts[0] if len(parts) == 1 else mx.concatenate(parts, axis=0)), lo


class _Checkpoint:
    """Cache state at position `n` (see module docstring for why this is exact and cheap)."""

    def __init__(self, cache, keys: list):
        self.n = cache.seq_len
        self.keys = list(keys[: self.n])
        self.slots = []
        for s in range(len(cache.kv_len)):
            buf = cache.kv_f16[s]
            copy = None
            if cache.kv_is_sliding[s] and buf is not None:
                copy = mx.contiguous(buf[:, :, : cache.kv_len[s], :])
            self.slots.append((cache.kv_len[s], cache.kv_start[s], cache.kv_cap[s], cache.kv_is_sliding[s], copy))
        if any(c is not None for *_, c in self.slots):
            mx.eval([c for *_, c in self.slots if c is not None])
        self.conv = [list(x) for x in cache.conv_states]
        self.nbytes = sum(c.nbytes for *_, c in self.slots if c is not None)

    def restore(self, cache):
        n = len(cache.kv_len)
        for s, (ln, st, cap, sliding, copy) in enumerate(self.slots):
            cache.kv_len[s], cache.kv_start[s], cache.kv_is_sliding[s] = ln, st, sliding
            if copy is not None:
                buf = mx.zeros((2, 8, cap, 128), dtype=mx.float16)
                buf[:, :, :ln, :] = copy
                cache.kv_f16[s], cache.kv_cap[s] = buf, cap
        cache.conv_states = [list(x) for x in self.conv]
        cache.seq_len = self.n
        cache._spec_recording = False
        cache._spec_kv_full = [None] * n
        cache._spec_conv_full = [[None] * 4 for _ in range(n)]
        cache._spec_conv_pair = [[None] * 4 for _ in range(n)]


def _install_hooks() -> None:
    """Patch the packed model's prefill entry points: chunking, resume-from-cache, snapshots, image rows."""
    from runtime_mlx.inkling.model import Inkling2LoopPackedMLX as M

    if getattr(M, "_inkling16_hooked", False):
        return

    def hooked(orig):
        def fwd(self, input_ids, cache=None, tap_visits=()):
            ids = input_ids.tolist() if isinstance(input_ids, mx.array) else list(input_ids)
            plan = self.__dict__.pop("_inkling16_plan", None)
            if plan is None:  # decode-time call (verify block, serial step): unchanged
                if len(ids) <= PREFILL_CHUNK:
                    return orig(self, ids, cache=cache, tap_visits=tap_visits)
                plan = {"base": cache.seq_len if cache is not None else 0}
            base = plan["base"]
            assert cache is not None and cache.seq_len == base, (cache.seq_len if cache else None, base)
            cuts = sorted({p for p in range(base, len(ids), PREFILL_CHUNK)} | {len(ids)}
                          | ({plan["snap_at"]} if base < (plan.get("snap_at") or -1) < len(ids) else set()))
            rows = plan.get("embed_rows") or []
            taps_new = []
            for a, b in zip(cuts, cuts[1:]):
                spans = [(p, r) for p, r in rows if p < b and p + r.shape[0] > a]
                if spans:
                    real = self.embed

                    def embed(x, _a=a, _spans=spans, _real=real):
                        h = _real(x)  # [1, T, H]
                        for p, r in _spans:
                            lo, hi = max(p, _a), min(p + r.shape[0], _a + h.shape[1])
                            h[0, lo - _a : hi - _a, :] = r[lo - p : hi - p].astype(h.dtype)
                        return h

                    self.embed = embed
                try:
                    res = orig(self, ids[a:b], cache=cache, tap_visits=tap_visits)
                finally:
                    if spans:
                        self.embed = real
                out, cache = res[0], res[1]
                if tap_visits:
                    taps_new.append(res[2])
                    while len(taps_new) > 1 and sum(t.shape[0] for t in taps_new[1:]) >= TAPS_WINDOW:
                        taps_new.pop(0)  # 262k-token prefill would otherwise hold ~9 GB of fp32 taps
                mx.eval(out, *taps_new[-1:])
                mx.clear_cache()
                if b == plan.get("snap_at") and plan.get("on_snap"):
                    plan["on_snap"](cache)
            if not tap_visits:
                return out, cache
            prefix, p_start = plan.get("taps_prefix") or (None, base)
            n_new = sum(t.shape[0] for t in taps_new)
            if prefix is not None and n_new < TAPS_WINDOW and len(ids) - n_new == base:
                taps = mx.concatenate([prefix] + taps_new, axis=0)
            else:
                taps = taps_new[0] if len(taps_new) == 1 else mx.concatenate(taps_new, axis=0)
            if taps.shape[0] > TAPS_WINDOW:
                taps = taps[-TAPS_WINDOW:]
            # stream_dspark pushes these with start_pos=0; tell the drafter wrapper where they really start
            self._inkling16_taps_offset = len(ids) - int(taps.shape[0])
            return out, cache, taps
        return fwd

    M.forward_argmax = hooked(M.forward_argmax)
    M.forward_ordered_logits = hooked(M.forward_ordered_logits)
    M._inkling16_hooked = True


# ------------------------------------------------------------------------------------------------- output parsing

class _OutputParser:
    """Turn raw generated token ids into ("reasoning_content"|"content", text) and ("tool_call", {...}) events."""

    def __init__(self, hf):
        self.hf = hf
        self.mode, self.channel = "body", "content"  # body text before any header is treated as content
        self.header: list[int] = []
        self.seg: list[int] = []
        self.sent = ""
        self.special = {int(i) for i, t in getattr(hf, "added_tokens_decoder", {}).items() if t.special}

    def push(self, tid: int):
        if tid == MSG_MODEL:
            yield from self._flush_tool()
            self.mode, self.header = "header", []
            return
        if self.mode == "header":
            if tid in (C_TEXT, C_THINK, C_TOOL):
                self.channel = {C_TEXT: "content", C_THINK: "reasoning_content", C_TOOL: "tool"}[tid]
                self.mode, self.seg, self.sent = "body", [], ""
            elif tid in self.special:
                self.mode, self.channel, self.seg, self.sent = "body", "ignore", [], ""
            else:
                self.header.append(tid)
            return
        if tid == END_MSG:
            yield from self._flush_tool()
            self.channel = "ignore"
            return
        if tid in self.special or self.channel == "ignore":
            return
        self.seg.append(tid)
        if self.channel == "tool":
            return
        full = self.hf.decode(self.seg, skip_special_tokens=True)
        if full.endswith("�"):  # wait for the rest of a multi-byte character
            return
        delta, self.sent = full[len(self.sent):], full
        if delta:
            yield self.channel, delta

    def _flush_tool(self):
        if self.channel == "tool" and self.seg:
            raw = self.hf.decode(self.seg, skip_special_tokens=True)
            name = self.hf.decode(self.header, skip_special_tokens=True).strip()
            try:
                obj = json.loads(raw)
                name = obj.get("name") or name
                args = obj.get("args", obj.get("arguments", {}))
                yield "tool_call", {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}
            except json.JSONDecodeError:
                yield "content", raw
        if self.channel == "tool":
            self.channel, self.seg = "ignore", []

    def finish(self):
        yield from self._flush_tool()


# ------------------------------------------------------------------------------------------------- cache slots

class _Slot:
    """One cached conversation: live KV cache + checkpoint + drafter tap history."""

    def __init__(self):
        self.cache = None
        self.live_keys: list = []
        self.ckpt: _Checkpoint | None = None
        self.taps = _TapHistory()
        self.last_used = 0.0

    def clear(self):
        self.cache, self.live_keys, self.ckpt = None, [], None
        self.taps.clear()

    def match(self, keys: list, n_ids: int) -> int:
        """How many prompt positions this slot can serve without prefill (0 = none)."""
        if self.cache is None:
            return 0
        live = self.live_keys
        if live and len(live) < n_ids and keys[: len(live)] == live and not self.cache._spec_recording:
            return len(live)
        ck = self.ckpt
        if ck is not None and ck.n < n_ids and keys[: ck.n] == ck.keys:
            return ck.n
        return 0

    def nbytes(self) -> int:
        if self.cache is None:
            return 0
        n = sum(getattr(b, "nbytes", 0) for b in self.cache.kv_f16 if b is not None)
        n += self.ckpt.nbytes if self.ckpt else 0
        n += sum(a.nbytes for _, a in self.taps.chunks)
        return n


# ------------------------------------------------------------------------------------------------------- engine

class Engine:
    def __init__(self, model: str | Path | None = None, drafter: str | Path | None = None, *, use_drafter: bool = True,
                 vision: bool = True, kv_bits: int = 4, slots: int = 4, slot_budget_gb: float = 1.5, draft_max_context: int = 32768, cache_limit_mb: int = 512, warmup: bool = True, quiet: bool = False):
        from runtime_mlx.inkling import DSparkDrafterMLX, load_inkling, resolve_drafter_files

        mx.set_cache_limit(cache_limit_mb * 1024**2)
        _install_hooks()
        from . import qkv

        qkv.BITS = kv_bits
        if kv_bits:
            qkv.install()
        self.kv_bits = kv_bits
        if model is None or (use_drafter and drafter is None):
            if not (MODEL_DIR / "manifest.json").exists() or not (DRAFTER_DIR / DRAFTER_FILE).exists():
                raise SystemExit(f"weights not found in {HOME}; run `inkling-16gb download` first (about 9.8 GB)")
        model = Path(model or MODEL_DIR)
        drafter = (drafter or DRAFTER_DIR / DRAFTER_FILE) if use_drafter else None
        self.log = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
        t = time.time()
        wrapped, tok = load_inkling(model)
        self.m = wrapped.inner if hasattr(wrapped, "inner") else wrapped
        self.tokenizer = tok
        self.hf = getattr(tok, "_tokenizer", tok)
        self.drafter = None
        if drafter:
            w_path, cfg_path = resolve_drafter_files(Path(drafter), fallback_config=model / "config.json")
            self.drafter = DSparkDrafterMLX(w_path, cfg_path)
            push = self.drafter.push_context

            d = self.drafter
            global_layers = [i for i in range(d.num_layers) if i not in d.local_layers]

            def recorded_push(target_taps, start_pos):
                off = self.m.__dict__.pop("_inkling16_taps_offset", 0)
                if start_pos == 0 and off:
                    start_pos = off
                self.taps.record(start_pos, target_taps)
                r = push(target_taps, start_pos)
                for i in global_layers:  # window the drafter's one full-attention layer (it only steers speed)
                    cur = d._ctx_cur[i]
                    if cur > DRAFT_WINDOW + 1024:
                        keep = DRAFT_WINDOW
                        cap = ((keep + 1024 + d.block_size + 255) // 256) * 256
                        K = mx.zeros((1, d.num_kv_heads, cap, d.head_dim), dtype=mx.float16)
                        V = mx.zeros((1, d.num_kv_heads, cap, d.head_dim), dtype=mx.float16)
                        K[:, :, :keep, :] = d.ctx_k[i][:, :, cur - keep:cur, :]
                        V[:, :, :keep, :] = d.ctx_v[i][:, :, cur - keep:cur, :]
                        d.ctx_k[i], d.ctx_v[i], d._ctx_cur[i], d._ctx_cap[i] = K, V, keep, cap
                return r

            self.drafter.push_context = recorded_push
        # Several cached conversations (e.g. a pi session plus subagents), served one request at a time: each request
        # goes to the slot that can reuse the most of its prompt; a new conversation takes a free or the LRU slot.
        self.slots = [_Slot() for _ in range(max(1, slots))]
        self.slot = self.slots[0]
        self.slot_budget = int(slot_budget_gb * 1e9)  # memory allowed for the *idle* slots
        self.draft_max_context = draft_max_context
        self.vision = None
        self._want_vision = vision
        self._feat_cache: dict[bytes, mx.array] = {}
        self.image_max_tokens = 1024  # = the local layers' sliding window; bigger images are downscaled to fit
        self.unload_vision = True
        self.log(f"loaded {model.name}{' + ' + Path(drafter).name if drafter else ''} in {time.time() - t:.1f}s")
        if warmup:
            t = time.time()
            for _ in self.run(self.encode([{"role": "user", "content": "Say hi."}]), max_tokens=16):
                pass
            self.reset_cache()
            self.log(f"warmed up (Metal kernel compile) in {time.time() - t:.1f}s")

    # ---- current cache slot

    cache = property(lambda self: self.slot.cache, lambda self, v: setattr(self.slot, "cache", v))
    live_keys = property(lambda self: self.slot.live_keys, lambda self, v: setattr(self.slot, "live_keys", v))
    ckpt = property(lambda self: self.slot.ckpt, lambda self, v: setattr(self.slot, "ckpt", v))
    taps = property(lambda self: self.slot.taps)

    def _pick_slot(self, prompt):
        keys, n = prompt["keys"], len(prompt["ids"])
        scored = [(sl.match(keys, n), sl) for sl in self.slots]
        best, sl = max(scored, key=lambda x: (x[0], x[1].last_used))
        if best == 0:
            empty = [s for s in self.slots if s.cache is None]
            sl = empty[0] if empty else min(self.slots, key=lambda s: s.last_used)
        self.slot = sl
        sl.last_used = time.time()
        # keep idle slots under the memory budget (LRU first)
        idle = sorted((s for s in self.slots if s is not sl and s.cache is not None), key=lambda s: s.last_used)
        budget = 0 if n > 65536 else self.slot_budget  # a very long request needs all the headroom
        while idle and sum(s.nbytes() for s in idle) > budget:
            idle.pop(0).clear()
            mx.clear_cache()

    # ---- prompts

    def _vision(self):
        if not self._want_vision:
            raise ValueError("image input is disabled (started with --no-vision)")
        if self.vision is None:
            from . import vision as V

            t = time.time()
            self.vision = V.load_vision()
            self.log(f"loaded vision tower in {time.time() - t:.1f}s")
        return self.vision

    def encode(self, messages, effort: str = "none", tools=None) -> dict:
        """Render OpenAI-style messages with the model's chat template. Returns a prompt dict for `run`."""
        if effort not in EFFORTS:
            effort = "none"
        msgs, images = [], []
        for m in messages:
            role = "system" if m.get("role") == "developer" else m.get("role", "user")
            msg = {"role": role}
            content = m.get("content")
            if isinstance(content, list):
                parts = []
                for p in content:
                    if isinstance(p, str):
                        parts.append({"type": "text", "text": p})
                    elif p.get("type") in ("text", "input_text"):
                        parts.append({"type": "text", "text": p.get("text", "")})
                    elif p.get("type") in ("image_url", "input_image", "image"):
                        src = p.get("image_url") or p.get("image") or p.get("url")
                        src = src.get("url") if isinstance(src, dict) else src
                        if role in ("user", "tool"):
                            parts.append({"type": "image"})
                            images.append(src)
                if role in ("system", "assistant") or all(p["type"] == "text" for p in parts):
                    msg["content"] = "".join(p["text"] for p in parts if p["type"] == "text")
                else:
                    msg["content"] = parts
            else:
                msg["content"] = content or ""
            if role == "assistant":
                rc = m.get("reasoning_content") or m.get("reasoning")
                if rc:
                    msg["reasoning_content"] = rc
                if m.get("tool_calls"):
                    tcs = []
                    for tc in m["tool_calls"]:
                        fn = dict(tc.get("function", {}))
                        args = fn.get("arguments") or {}
                        if isinstance(args, str):
                            try:
                                args = json.loads(args) if args.strip() else {}
                            except json.JSONDecodeError:
                                args = {"_raw": args}
                        fn["arguments"] = args
                        tcs.append({"id": tc.get("id"), "type": "function", "function": fn})
                    msg["tool_calls"] = tcs
            if role == "tool":
                if m.get("tool_call_id"):
                    msg["tool_call_id"] = m["tool_call_id"]
                if m.get("name"):
                    msg["name"] = m["name"]
            msgs.append(msg)

        def render(ms):
            from runtime_mlx.runtime import _unwrap_ids

            kw = {"tools": tools} if tools else {}
            return _unwrap_ids(self.hf.apply_chat_template(ms, tokenize=True, add_generation_prompt=True,
                                                           reasoning_effort=effort, **kw))

        ids = render(msgs)
        # The stable prefix: what the next turn's prompt will start with (everything but a trailing scaffold).
        stable = _common_prefix(ids, render(msgs + [{"role": "assistant", "content": "x"}]))

        # Images: expand each placeholder into the encoder's rows; give image positions content-derived keys so the
        # prefix cache never confuses two different images.
        keys = list(ids)
        rows = []
        if images:
            from . import vision as V

            n_ph = sum(1 for t in ids if t == IMAGE_PLACEHOLDER)
            if n_ph != len(images):
                raise ValueError(f"{len(images)} images but {n_ph} image placeholders in the rendered prompt")
            feats, digests = [], []
            for src in images:
                inp = V.preprocess(src, max_tokens=self.image_max_tokens)
                d = hashlib.sha1(inp.pixel_values.tobytes() + f"{inp.rows}x{inp.cols}".encode()).digest()
                f = self._feat_cache.get(d)
                if f is None:  # agents resend every image on every turn; encode each distinct image once
                    f = self._vision().encode(inp)
                    mx.eval(f)
                    self._feat_cache[d] = f
                    while len(self._feat_cache) > 16:
                        self._feat_cache.pop(next(iter(self._feat_cache)))
                feats.append(f)
                digests.append(d)
            if self.unload_vision and self.vision is not None:
                self.vision = None  # hand the tower's 637 MB back to the LM; reloads in ~0.3 s if a new image arrives
                mx.clear_cache()
            ph_before_stable = sum(1 for t in ids[:stable] if t == IMAGE_PLACEHOLDER)
            ids, spans = V.expand_placeholders(ids, [int(f.shape[0]) for f in feats])
            stable += sum(n - 1 for _, n in spans[:ph_before_stable])
            keys = list(ids)
            for (start, n), f, d in zip(spans, feats, digests):
                rows.append((start, f))
                tag = -int.from_bytes(d[:6], "big") - 1  # content-derived key: two images never share a prefix
                for j in range(n):
                    keys[start + j] = (tag, j)
        return {"ids": ids, "keys": keys, "stable": stable, "embed_rows": rows, "effort": effort}

    # ---- generation

    def reset_cache(self):
        for sl in self.slots:
            sl.clear()
        mx.clear_cache()

    def _plan(self, prompt, need_taps: bool) -> tuple[int, str]:
        self._pick_slot(prompt)
        ids, keys = prompt["ids"], prompt["keys"]
        live = self.live_keys
        if (self.cache is not None and live and len(live) < len(ids) and keys[: len(live)] == live
                and not self.cache._spec_recording and (not need_taps or self.taps.length >= len(live))):
            return len(live), "live"
        ck = self.ckpt
        if (self.cache is not None and ck is not None and ck.n < len(ids) and keys[: ck.n] == ck.keys
                and (not need_taps or self.taps.length >= ck.n)):
            ck.restore(self.cache)
            self.taps.truncate(ck.n)
            return ck.n, "checkpoint"
        self.cache = None
        self.ckpt = None
        self.taps.clear()
        mx.clear_cache()
        self.cache = self.m.make_cache()
        return 0, "miss"

    def run(self, prompt, max_tokens: int = 1024, temp: float = 0.0, top_p: float = 1.0):
        """Yield ("reasoning_content"|"content", text), ("tool_call", {name, arguments}), then ("done", info dict)."""
        from runtime_mlx.inkling import stream_dspark, stream_serial

        if isinstance(prompt, list):
            prompt = {"ids": prompt, "keys": list(prompt), "stable": len(prompt), "embed_rows": []}
        ids, keys = prompt["ids"], prompt["keys"]
        # Past ~32k tokens a drafted verify block (each of ~8 rows re-reads the whole KV cache) costs more than it
        # saves: measured at 100k, serial 12.3 tok/s vs drafted 7.2. Below that, drafting is 1.5-1.6x on code/math.
        drafting = self.drafter is not None and len(ids) <= self.draft_max_context
        t_plan = time.perf_counter()
        base, how = self._plan(prompt, drafting)
        t_plan = time.perf_counter() - t_plan
        cache = self.cache
        cache._inkling16_reserve = len(ids) + max_tokens + 64  # size the quantized KV once, not by repeated growth
        # one token short of the stable end, so an identical re-send (retry, regenerate) can still restore and feed 1 token
        stable = min(prompt["stable"], len(ids) - 1)
        snap_at = stable if stable > base else None

        def on_snap(c):
            self.ckpt = _Checkpoint(c, keys)

        plan = {"base": base, "snap_at": snap_at, "on_snap": on_snap, "embed_rows": prompt.get("embed_rows"),
                "taps_prefix": self.taps.get(base) if drafting else None}
        self.m._inkling16_plan = plan
        self.m.make_cache = lambda: cache  # one-shot: the stream functions call make_cache() first thing
        self.live_keys = []
        stats: dict = {}
        eos = {EOS}
        if drafting:
            gen = stream_dspark(self.m, self.drafter, ids, max_tokens=max_tokens, eos_ids=eos, temp=temp, top_p=top_p,
                                stats_out=stats)
        else:
            gen = stream_serial(self.m, ids, max_tokens=max_tokens, eos_ids=eos, temp=temp, top_p=top_p, stats_out=stats)
        parser = _OutputParser(self.hf)
        raw: list[int] = []
        stopped = False
        t0 = time.perf_counter()
        try:
            for chunk in gen:
                for tid in chunk:
                    tid = int(tid)
                    raw.append(tid)
                    if tid in eos:
                        stopped = True
                        break
                    yield from parser.push(tid)
                if stopped:
                    break
        finally:
            gen.close()
            self.m.__dict__.pop("make_cache", None)
            self.m.__dict__.pop("_inkling16_plan", None)
            if not cache._spec_recording and cache.seq_len <= len(ids) + len(raw):
                self.live_keys = (keys + raw)[: cache.seq_len]
        yield from parser.finish()
        dt = time.perf_counter() - t0
        prompt_s = float(stats.get("prompt_s", 0.0)) + float(stats.get("drafter_prompt_s", 0.0))
        rounds_s = float(stats.get("rounds_s", 0.0)) or max(dt - prompt_s, 1e-9)
        n_gen = len([t for t in raw if t not in eos])
        yield "done", {
            "finish_reason": "stop" if stopped else "length",
            "prompt_tokens": len(ids), "cached_tokens": base, "cache": how,
            "completion_tokens": n_gen,
            "prompt_tps": (len(ids) - base) / prompt_s if prompt_s > 0 else 0.0,
            "prompt_s": prompt_s,
            "timing": {"plan": t_plan, "target_prefill": float(stats.get("prompt_s", 0.0)),
                       "drafter_rebuild": float(stats.get("drafter_prompt_s", 0.0))},
            "generation_tps": stats.get("tokens", n_gen) / rounds_s if rounds_s > 0 else 0.0,
            "tokens_per_check": stats.get("tokens_per_check"),
        }
