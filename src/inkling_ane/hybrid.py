"""Hybrid Inkling 2 Mini prefill: dense parts on the ANE (Core ML), MoE on the GPU (MLX).

Every layer application l splits into
  ANE:  x_l -> (h_l, residual_l)            one Core ML slice per (layer, pass>0?), one MLState set per l
  GPU:  x_{l+1} = residual_l + mlp_sconv_l(MoE_l(h_l))   (the loaded model's own MoE / conv weights)
The prompt is cut into chunks of C rows (the ANE runs each as C/128 calls; the GPU runs the MoE
on all C rows at once). Within a chunk the layers are sequential, so the overlap is ACROSS
chunks: sweep layer by layer, chunk by chunk; the GPU's MoE for chunk c (mx.async_eval) runs while
the ANE does chunk c+1. After the sweep the ANE states (K/V windows, conv tails) are copied into
the standard mlx-vlm caches, so GPU decode (inkling_fast.decode) continues as after GPU prefill.

Entry points (model already loaded with mlx_vlm; needs coremltools):
  from inkling_ane import hybrid
  cache, logits = hybrid.prefill(model, token_ids)      # standalone
  hybrid.install(model)                                 # route fresh long prefills through the ANE

Env: INKLING_ANE_SLICES (default runtime/ane/prod_int8), INKLING_ANE_MIN_TOKENS (1024).
"""

import math
import os
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import coremltools as ct  # noqa: E402  (after the warnings filter)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from .torch_layer import host_mask  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
SLICES = Path(os.environ.get("INKLING_ANE_SLICES", ROOT / "runtime/ane/prod_int8"))
MIN_TOKENS = int(os.environ.get("INKLING_ANE_MIN_TOKENS", "1024"))
Q = 128
D = 2304


def n_blocks(i):
    return 0 if i == 0 else (i - 1) // 6 + 1


class HybridPrefill:
    def __init__(self, model, slices=SLICES, background=True):
        from mlx_vlm.models.inkling2.language import _attention_residual

        self.lm = getattr(model, "language_model", model)
        self.cfg = self.lm.config
        self.sched = self.lm.model.schedule()  # (layer, cache index, pass)
        self.start = self.cfg.loop_layer_index_start
        self.sliding = set(self.cfg.local_layer_ids)
        self.slices = Path(slices)
        self.buckets = sorted(int(p.name[1:]) for p in self.slices.glob("s*") if p.is_dir())
        if not self.buckets:
            raise FileNotFoundError(f"no slice buckets under {self.slices}")
        self._attn_res = _attention_residual
        self.models = {}  # (bucket, layer, pass01) -> CompiledMLModel
        self.ready = threading.Event()
        self.load_s = None
        self.error = None
        self._masks = {}
        if background:
            threading.Thread(target=self._load, daemon=True).start()
        else:
            self._load()

    def _load(self):
        try:
            t = time.perf_counter()
            small = self.buckets[0]
            for i, p in sorted(set((i, min(p, 1)) for i, _, p in self.sched)):
                for S in (self.buckets if i not in self.sliding else [small]):
                    path = self.slices / f"s{S}" / f"L{i}p{p}.mlmodelc"
                    if path.exists():
                        self.models[(S, i, p)] = ct.models.CompiledMLModel(
                            str(path), compute_units=ct.ComputeUnit.CPU_AND_NE)
            self.load_s = time.perf_counter() - t
            self.ready.set()
        except BaseException as e:  # keep serving on the GPU
            self.error = e

    @property
    def max_tokens(self):
        return self.buckets[-1]

    def model(self, i, p, S):
        key = (self.buckets[0], i, min(p, 1)) if i in self.sliding else (S, i, min(p, 1))
        return self.models[key]

    def mask(self, p0, sliding, S):
        key = (p0, sliding, S)
        if key not in self._masks:
            self._masks[key] = host_mask(p0, Q, 1024 if sliding else S, sliding).astype(np.float16)
        return self._masks[key]

    def eligible(self, n, cache):
        if not self.ready.is_set() or not MIN_TOKENS <= n <= self.max_tokens or cache is None:
            return False
        if len(cache) != len(self.sched):
            return False
        from mlx_vlm.models.cache import KVCache, RotatingKVCache
        for c in cache:
            kv, conv = c
            if type(kv) not in (KVCache, RotatingKVCache) or kv.keys is not None or kv.offset:
                return False
            if any(a is not None for a in conv.cache) or conv.left_padding is not None or conv.lengths is not None:
                return False
        return True

    # ---- the pipelined sweep ----
    def _ane(self, step, x, blocks, msg, p0, state, S):
        i, _, p = self.sched[step]
        model = self.model(i, p, S)
        hs, rs = [], []
        for s in range(0, x.shape[0], Q):
            feed = {"x": x[None, s:s + Q], "mask": self.mask(p0 + s, i in self.sliding, S)}
            if n_blocks(i):
                feed["blocks"] = np.stack([b[s:s + Q] for b in blocks])[None]
            if p:
                feed["msg_states"] = np.stack([m[s:s + Q] for m in msg])[None]
            o = model.predict(feed, state=state)
            hs.append(o["h"][0])
            rs.append(o["residual"][0])
        return np.concatenate(hs), np.concatenate(rs)

    def _tail(self, i, h, residual, conv):
        """x_next = residual + mlp_sconv(MoE(h)) as fp16 (inside the async graph), new conv tail."""
        layer = self.lm.model.layers[i]
        y = layer.mlp(h.astype(mx.bfloat16))
        yf = y.astype(mx.float32)
        ctx = mx.concatenate([conv, yf], axis=1)
        out = mx.conv1d(ctx, layer.mlp_sconv.conv.weight.astype(mx.float32), groups=D)
        sc = ((out + yf) / math.sqrt(5)).astype(mx.bfloat16).astype(mx.float32)
        x = (residual.astype(mx.float32) + sc).astype(mx.bfloat16)
        # Cast and index inside the graph: any later lazy op on an in-flight array queues behind
        # newer GPU work (one FIFO stream) and serializes the pipeline.
        return x.astype(mx.float16)[0], ctx[:, -3:]

    def run(self, embeds, cache, chunk=None):
        """embeds [1, N, D] (the model's input embeddings); fills `cache` (from lm.make_cache()).
        Returns the normalized final hidden state of the last token, [1, 1, D]."""
        n = embeds.shape[1]
        S = next(b for b in self.buckets if b >= n)
        chunk = chunk or (512 if n >= 1792 else 256)
        pad = (-n) % Q
        x0 = embeds[0].astype(mx.float16)
        if pad:
            x0 = mx.concatenate([mx.zeros((pad, D), x0.dtype), x0])
        total = n + pad
        bounds = [(s, min(s + chunk, total)) for s in range(0, total, chunk)]
        x0 = np.array(x0)
        states = [self.model(i, p, S).make_state() for i, _, p in self.sched]
        C = [dict(x=x0[a:b], pending=None, blocks=[], pre=None, prev=None) for a, b in bounds]
        conv = [mx.zeros((1, 3, D), mx.float32) for _ in self.sched]
        for step, (i, _, p) in enumerate(self.sched):
            for c, (a, _b) in enumerate(bounds):
                st = C[c]
                if st["pending"] is not None:
                    st["x"] = np.array(st["pending"])
                    st["pending"] = None
                x = st["x"]
                if i == self.start:
                    if p == 0:
                        st["pre"] = (x, list(st["blocks"]))
                    else:
                        st["prev"] = (st["blocks"][len(st["pre"][1]) + 1:], x)
                        x, st["blocks"] = st["pre"][0], list(st["pre"][1])
                msg = (st["prev"][0] + [st["prev"][1]]) if p else None
                h, r = self._ane(step, x, st["blocks"], msg, a - pad, states[step], S)
                if i % 6 == 0:
                    st["blocks"].append(x)
                st["pending"], conv[step] = self._tail(i, mx.array(h)[None], mx.array(r)[None], conv[step])
                mx.async_eval(st["pending"], conv[step])
        last = C[-1]
        xl = mx.array(np.array(last["pending"])[-1:]).astype(mx.bfloat16)
        blocks = [mx.array(bk[-1:]).astype(mx.bfloat16) for bk in last["blocks"]]
        h = self._attn_res(mx.stack([*blocks, xl], axis=-2), self.lm.model.final_input.query, True)[..., 0, :]
        normed = self.lm.model.norm(h)[None]
        self._fill(cache, states, conv, n, S)
        mx.eval(normed, [c.state for c in cache])
        return normed

    def _fill(self, cache, states, conv, n, S):
        """Copy the ANE windows into mlx-vlm caches, matching what GPU prefill leaves behind."""
        from mlx_vlm.models.cache import RotatingKVCache
        for step, (i, ci, _p) in enumerate(self.sched):
            st = states[step]
            kv, cv = cache[ci]
            window = 1024 if i in self.sliding else S
            keep = min(n, window)
            k = mx.array(st.read_state("m_k_cache")[:, :, window - keep:]).astype(mx.bfloat16)
            v = mx.array(st.read_state("m_v_cache")[:, :, window - keep:]).astype(mx.bfloat16)
            if isinstance(kv, RotatingKVCache):
                # GPU prefill keeps >= the last max_size keys in time order; the first decode
                # step trims to max_size and rotates. Starting from the last <= max_size keys in
                # time order gives the same cache after that step.
                kv.keys, kv.values, kv.offset, kv._idx = k, v, n, keep
            else:
                kv.state = (k, v)  # all n positions (n <= S)
            cv[0] = mx.array(st.read_state("m_k_conv")).astype(mx.float32)
            cv[1] = mx.array(st.read_state("m_v_conv")).astype(mx.float32)
            cv[2] = mx.array(st.read_state("m_a_conv")).astype(mx.float32)
            cv[3] = conv[step]


def reset(cache):
    for kv, conv in cache:
        kv.keys = kv.values = None
        kv.offset = 0
        if hasattr(kv, "_idx"):
            kv._idx = 0
        conv.cache = [None] * len(conv.cache)


_INSTANCES = {}


def get(model, wait=True):
    lm = getattr(model, "language_model", model)
    hp = _INSTANCES.get(id(lm))
    if hp is None:
        hp = _INSTANCES[id(lm)] = HybridPrefill(model)
    if wait:
        hp.ready.wait()
        if hp.error:
            raise hp.error
    return hp


def prefill(model, token_ids, chunk=None):
    """Hybrid prefill of the whole prompt. Returns (cache, last-token logits [1, 1, V])."""
    lm = getattr(model, "language_model", model)
    hp = get(model)
    cache = lm.make_cache()
    embeds = lm.model.embed(mx.array([list(token_ids)]))
    normed = hp.run(embeds, cache, chunk)
    return cache, lm._logits_from_norm(normed)


def install(model, log=print):
    """Route fresh-cache prefill calls of MIN_TOKENS..max bucket tokens through the hybrid.
    Everything else (short prompts, APC prefix hits, captures, speculation) stays on the GPU.
    Slices load in the background; requests before they are ready use the GPU."""
    from mlx_vlm.models.base import LanguageModelOutput

    lm = getattr(model, "language_model", model)
    hp = get(model, wait=False)
    original = type(lm).__call__

    def call(self, inputs=None, inputs_embeds=None, cache=None, **kwargs):
        ref = inputs_embeds if inputs_embeds is not None else inputs
        if (
            self is lm and ref is not None and ref.shape[0] == 1
            and not any(kwargs.get(k) for k in ("capture_layer_ids", "return_hidden", "speculative_verify", "return_shared_kv"))
            and hp.eligible(ref.shape[1], cache)
        ):
            t = time.perf_counter()
            embeds = inputs_embeds if inputs_embeds is not None else lm.model.embed(inputs)
            try:
                normed = hp.run(embeds, cache)
            except Exception as e:  # ANE failure: reset the cache and prefill on the GPU
                reset(cache)
                if log:
                    log(f"[inkling_ane] hybrid prefill failed ({str(e)[:120]}); GPU fallback")
                return original(self, inputs=inputs, inputs_embeds=inputs_embeds, cache=cache, **kwargs)
            if log:
                log(f"[inkling_ane] hybrid prefill {ref.shape[1]} tokens in {time.perf_counter() - t:.2f}s")
            return LanguageModelOutput(
                logits=None if kwargs.get("skip_logits") else lm._logits_from_norm(normed),
                hidden_states=None, shared_kv_states=None, gdn_states=None)
        return original(self, inputs=inputs, inputs_embeds=inputs_embeds, cache=cache, **kwargs)

    type(lm).__call__ = call
    if log:
        def announce():
            hp.ready.wait()
            log(f"[inkling_ane] {len(hp.models)} ANE slices ready in {hp.load_s:.1f}s; buckets {hp.buckets}")
        threading.Thread(target=announce, daemon=True).start()
    return hp
