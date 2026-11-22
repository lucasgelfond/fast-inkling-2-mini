"""Quantized KV cache for Inkling-2-Mini's 8 global-attention visits (slots 5, 11, ..., 47), for long context on 16 GB.

Per token, the 8 global slots hold 32 KB of fp16 KV (8.6 GB at 262k tokens) - more than a 16 GB Mac has left after the
8.8 GB of weights. The 40 sliding-window slots are capped at ~1.3k positions and stay fp16.

Storage per global slot: `d` uint8 [16 rows = 2 (K,V) x 8 kv-heads, cap, 128 * bits / 8] and `s` float16 [16, cap, 4]:
symmetric absmax quantization in groups of 32 along head_dim (K is L2-clipped and V RMS-normed upstream, so groups are
well conditioned). 4-bit: 1,152 B/token/slot (9.2 KB/token for all 8, 2.4 GB at 262k); 8-bit: 2,176 B (4.6 GB at 262k).

The attention kernels are the runtime's own (decode: online-softmax flash-decoding; prefill: simdgroup FlashAttention-2),
captured from their builders and rewritten only at the K/V load sites to dequantize on load (see `_rewrite`), so masking,
relative-position bias, sinks and accumulation order are unchanged.
"""

from __future__ import annotations

import re
from functools import lru_cache

import mlx.core as mx

BITS = 4
GROUP = 32
_HOOKED = False

_LDQ = {
    4: """
inline half4 ink_ldq(const device uint8_t* d, const device half* s, ulong row, uint d4) {
    const device uint8_t* p = d + row * 64u + d4 * 2u;
    const uint b = (uint)p[0] | ((uint)p[1] << 8);
    const float sc = (float)s[row * 4u + (d4 >> 3u)];
    return half4((float4((float)(b & 15u), (float)((b >> 4) & 15u), (float)((b >> 8) & 15u), (float)((b >> 12) & 15u)) - 8.0f) * sc);
}
""",
    8: """
inline half4 ink_ldq(const device uint8_t* d, const device half* s, ulong row, uint d4) {
    const char4 c = *(const device char4*)(d + row * 128u + d4 * 4u);
    return half4(float4(c) * (float)s[row * 4u + (d4 >> 3u)]);
}
""",
}


class QuantKV:
    """Pre-allocated quantized KV for one global slot."""

    __slots__ = ("d", "s", "cap", "bits")

    def __init__(self, cap: int, bits: int):
        self.cap, self.bits = cap, bits
        self.d = mx.zeros((16, cap, 128 * bits // 8), dtype=mx.uint8)
        self.s = mx.zeros((16, cap, 128 // GROUP), dtype=mx.float16)

    def grow(self, need: int, cur: int, reserve: int = 0):
        # Geometric growth (the stock fp16 cache grows by 256 and copies everything each time: quadratic at long context)
        cap = max(need, reserve, int(self.cap * 1.5))
        cap = ((cap + 1023) // 1024) * 1024
        d = mx.zeros((16, cap, self.d.shape[2]), dtype=mx.uint8)
        s = mx.zeros((16, cap, self.s.shape[2]), dtype=mx.float16)
        if cur:
            d[:, :cur, :] = self.d[:, :cur, :]
            s[:, :cur, :] = self.s[:, :cur, :]
        self.d, self.s, self.cap = d, s, cap

    @property
    def nbytes(self) -> int:
        return self.d.nbytes + self.s.nbytes


def quantize(kv_f16: mx.array, bits: int) -> tuple[mx.array, mx.array]:
    """[2, 8, T, 128] float16 -> (uint8 [16, T, 128*bits/8], float16 [16, T, 4])."""
    T = kv_f16.shape[2]
    x = kv_f16.reshape(16, T, 128 // GROUP, GROUP).astype(mx.float32)
    qmax = 7.0 if bits == 4 else 127.0
    scale = (mx.max(mx.abs(x), axis=-1, keepdims=True) / qmax).astype(mx.float16)
    scale = mx.maximum(scale, mx.array(1e-7, dtype=mx.float16))
    q = mx.clip(mx.round(x / scale.astype(mx.float32)), -qmax, qmax)
    if bits == 8:
        d = q.astype(mx.int8).reshape(16, T, 128).view(mx.uint8)
    else:
        u = (q + 8.0).astype(mx.uint8).reshape(16, T, 64, 2)
        d = u[..., 0] | (u[..., 1] << 4)
    return d, scale.reshape(16, T, 128 // GROUP)


def dequantize(d: mx.array, s: mx.array, bits: int) -> mx.array:
    """Inverse of `quantize` (for tests): -> float16 [16, T, 128]."""
    n, T = d.shape[0], d.shape[1]
    if bits == 8:
        q = d.view(mx.int8).astype(mx.float32)
    else:
        lo = (d & 15).astype(mx.float32) - 8.0
        hi = (d >> 4).astype(mx.float32) - 8.0
        q = mx.stack([lo, hi], axis=-1).reshape(n, T, 128)
    return (q.reshape(n, T, 4, GROUP) * s.astype(mx.float32)[..., None]).reshape(n, T, 128).astype(mx.float16)


def _capture(builder, *args) -> dict:
    """Run a runtime kernel builder with mx.fast.metal_kernel intercepted, returning its kwargs (source, names...)."""
    real = mx.fast.metal_kernel
    got = {}

    def fake(**kw):
        got.update(kw)
        return None

    mx.fast.metal_kernel = fake
    try:
        getattr(builder, "__wrapped__", builder)(*args)
    finally:
        mx.fast.metal_kernel = real
    return got


def _sub(src: str, literal: str, new: str, count: int) -> str:
    pat = r"\s+".join(re.escape(tok) for tok in literal.split())
    out, n = re.subn(pat, lambda _m: new, src)
    if n != count:
        raise RuntimeError(f"qkv kernel rewrite: expected {count} match(es) of {literal!r}, found {n}")
    return out


_BASES = (
    "const device half* Kp = KV + (ulong)kvh * kv_cap * 128u; const device half* Vp = KV + (ulong)(8u + kvh) * kv_cap * 128u;",
    "const ulong kbase = (ulong)kvh * kv_cap; const ulong vbase = (ulong)(8u + kvh) * kv_cap;",
)


@lru_cache(maxsize=4)
def _decode_kernel(bits: int):
    kw = _decode_kw()
    src = _decode_kernel_src(bits)
    names = list(kw["input_names"])
    names[names.index("KV"):names.index("KV") + 1] = ["KVd", "KVs"]
    return mx.fast.metal_kernel(name=f"ink_qkv{bits}_gqa_relpos_sdpa", input_names=names, output_names=kw["output_names"],
                                header=(kw.get("header") or "") + _LDQ[bits], source=src)


@lru_cache(maxsize=4)
def _decode_split_kernel(bits: int):
    """Split-K flash-decoding: the stock decode kernel runs one threadgroup per (token, head) over the whole cache,
    i.e. 24 threadgroups for a single token - fine at 4k context, far too little parallelism at 100k+. Here each
    (token, head) range is cut into NS splits; each writes (max, sum, unnormalized acc) and `_combine` merges them
    with the sink logit exactly as the stock kernel's in-threadgroup merge does."""
    src = _decode_kernel_src(bits)
    src = _sub(src, "const uint ht = threadgroup_position_in_grid.y;",
               "const uint gy = threadgroup_position_in_grid.y; const uint NS = meta[6]; const uint split = gy % NS; "
               "const uint ht = gy / NS;", 1)
    src = _sub(src, "for (uint j = j_lo + simd; j < j_hi; j += 4u) {",
               "const uint span = (j_hi > j_lo) ? ((j_hi - j_lo + NS - 1u) / NS) : 0u; "
               "const uint s_lo = min(j_hi, j_lo + split * span); const uint s_hi = min(j_hi, s_lo + span); "
               "for (uint j = s_lo + simd; j < s_hi; j += 4u) {", 1)
    cut = src.index("if (simd == 0u)")
    src = src[:cut] + """if (simd == 0u) {
        float m_all = -1e30f;
        for (uint s = 0; s < 4u; s++) m_all = max(m_all, tg_m[s]);
        float l_all = 0.0f;
        float4 a_all = float4(0.0f);
        for (uint s = 0; s < 4u; s++) {
            const float sc = metal::precise::exp(tg_m[s] - m_all);
            l_all += tg_l[s] * sc;
            a_all += tg_a[s][lane] * sc;
        }
        const ulong pi = (ulong)ht * NS + split;
        device float4* ap = (device float4*)(out + pi * 128u);
        ap[lane] = a_all;
        if (lane == 0u) {
            stats[pi * 3u] = m_all;
            stats[pi * 3u + 1u] = l_all;
            stats[pi * 3u + 2u] = tg_sk[0] + tg_sk[1] + tg_sk[2] + tg_sk[3];
        }
    }
"""
    kw = _decode_kw()
    names = list(kw["input_names"])
    names[names.index("KV"):names.index("KV") + 1] = ["KVd", "KVs"]
    return mx.fast.metal_kernel(name=f"ink_qkv{bits}_gqa_relpos_sdpa_splitk", input_names=names,
                                output_names=["out", "stats"], header=(kw.get("header") or "") + _LDQ[bits], source=src)


def _combine(a: mx.array, st: mx.array, T: int, NS: int) -> mx.array:
    a = a.reshape(T, 24, NS, 128)
    st = st.reshape(T, 24, NS, 3)
    m, l, sk = st[..., 0], st[..., 1], st[..., 0, 2]
    M = mx.maximum(mx.max(m, axis=-1), sk)
    w = mx.exp(m - M[..., None])
    L = mx.exp(sk - M) + mx.sum(l * w, axis=-1)
    A = mx.sum(a * w[..., None], axis=2)
    return (A / L[..., None]).reshape(T, 3072)


@lru_cache(maxsize=1)
def _decode_kw() -> dict:
    from runtime_mlx.inkling import fast_kernels as fk

    return _capture(fk._make_gqa_relpos_sdpa_kernel, False)


def _decode_kernel_src(bits: int) -> str:
    src = _sub(_decode_kw()["source"], *_BASES, 1)
    src = _sub(src, "const device half4* kp = (const device half4*)(Kp + (ulong)j * 128u); float score = simd_sum(dot(qv, float4(kp[lane])));",
               "float score = simd_sum(dot(qv, float4(ink_ldq(KVd, KVs, kbase + j, lane))));", 1)
    src = _sub(src, "const device half4* vp = (const device half4*)(Vp + (ulong)j * 128u); acc = acc * alpha + p * float4(vp[lane]);",
               "acc = acc * alpha + p * float4(ink_ldq(KVd, KVs, vbase + j, lane));", 1)
    return src


SPLIT_KEYS = 2048  # keys per split (>= this many keys before splitting at all)


@lru_cache(maxsize=4)
def _flash_kernel(bits: int):
    from runtime_mlx.inkling import fast_kernels as fk

    kw = _capture(fk._make_gqa_relpos_flash_sdpa_kernel)
    src = _sub(kw["source"], *_BASES, 1)
    src = _sub(src, "*(const device half4*)(Kp + (ulong)kj * 128u + d4 * 4u)", "ink_ldq(KVd, KVs, kbase + kj, d4)", 1)
    src = _sub(src, "*(const device half4*)(Vp + (ulong)kj * 128u + d4 * 4u)", "ink_ldq(KVd, KVs, vbase + kj, d4)", 1)
    names = list(kw["input_names"])
    names[names.index("KV"):names.index("KV") + 1] = ["KVd", "KVs"]
    return mx.fast.metal_kernel(name=f"ink_qkv{bits}_gqa_relpos_flash_sdpa", input_names=names,
                                output_names=kw["output_names"], header=(kw.get("header") or "") + _LDQ[bits], source=src)


def _sdpa_quant(q, kv: QuantKV, rel_st, rel_proj_t_f16, q_in, sink_w_f16, T, kv_len, kv_start, q_pos0, kv_cap, is_sliding,
                tree_depth=None, tree_mask=None):
    from runtime_mlx.inkling.fast_kernels import _u32

    if tree_depth is not None and tree_mask is not None:
        raise NotImplementedError("tree drafting is not supported with the quantized KV cache")
    if T >= 32:
        sinks = (q_in.reshape(T, 2304).astype(mx.float16) @ sink_w_f16.reshape(24, 2304).T).astype(mx.float32)
        nqb = (T + 31) // 32
        return _flash_kernel(kv.bits)(
            inputs=[q.reshape(-1), kv.d.reshape(-1), kv.s.reshape(-1), rel_st.reshape(-1), rel_proj_t_f16.reshape(-1),
                    sinks.reshape(-1), _u32(T, kv_len, kv_start, q_pos0, kv.cap, int(is_sliding), nqb)],
            grid=(128, 24 * nqb, 1), threadgroup=(128, 1, 1), output_shapes=[(T, 3072)], output_dtypes=[mx.float32],
        )[0]
    ins = [q.reshape(-1), kv.d.reshape(-1), kv.s.reshape(-1), rel_st.reshape(-1), rel_proj_t_f16.reshape(-1),
           q_in.reshape(-1), sink_w_f16.reshape(-1)]
    NS = min(32, kv_len // SPLIT_KEYS) if SPLIT_KEYS else 0
    if NS >= 2:
        a, st = _decode_split_kernel(kv.bits)(
            inputs=ins + [_u32(T, kv_len, kv_start, q_pos0, kv.cap, int(is_sliding), NS)],
            grid=(128, 24 * T * NS, 1), threadgroup=(128, 1, 1),
            output_shapes=[(T * 24 * NS, 128), (T * 24 * NS * 3,)], output_dtypes=[mx.float32, mx.float32],
        )
        return _combine(a, st, T, NS)
    return _decode_kernel(kv.bits)(
        inputs=ins + [_u32(T, kv_len, kv_start, q_pos0, kv.cap, int(is_sliding))],
        grid=(128, 24 * T, 1), threadgroup=(128, 1, 1), output_shapes=[(T, 3072)], output_dtypes=[mx.float32],
    )[0]


def install() -> None:
    """Route global (non-sliding) KV slots through the quantized cache. Idempotent; `BITS` = 0 disables at runtime."""
    global _HOOKED
    if _HOOKED:
        return
    from runtime_mlx.inkling import model as M

    orig_sdpa = M.gqa_relpos_sinks_sdpa_fast
    orig_update = M.Inkling2Cache.update_kv_f16

    def sdpa(q, KV_buf, *a, **kw):
        if isinstance(KV_buf, QuantKV):
            return _sdpa_quant(q, KV_buf, *a, **kw)
        return orig_sdpa(q, KV_buf, *a, **kw)

    def update_kv_f16(self, slot, kv_f16, is_sliding):
        if is_sliding or not BITS:
            return orig_update(self, slot, kv_f16, is_sliding)
        T = int(kv_f16.shape[2])
        self.kv_is_sliding[slot] = False
        cur = self.kv_len[slot]
        buf = self.kv_f16[slot]
        if not isinstance(buf, QuantKV) or buf.bits != BITS:
            buf = QuantKV(((max(cur + T, getattr(self, "_inkling16_reserve", 0)) + 1023) // 1024) * 1024, BITS)
            cur = 0 if self.kv_len[slot] == 0 else cur
            if self.kv_len[slot]:
                raise RuntimeError("KV cache was started without quantization; reset it before switching")
        need = cur + T
        if need > buf.cap:
            buf.grow(need, cur, getattr(self, "_inkling16_reserve", 0))
        d, s = quantize(kv_f16, buf.bits)
        buf.d[:, cur:need, :] = d
        buf.s[:, cur:need, :] = s
        self.kv_f16[slot] = buf
        self.kv_cap[slot] = buf.cap
        self.kv_len[slot] = need
        return buf, need, self.kv_start[slot], buf.cap

    M.gqa_relpos_sinks_sdpa_fast = sdpa
    M.Inkling2Cache.update_kv_f16 = update_kv_f16
    _HOOKED = True
