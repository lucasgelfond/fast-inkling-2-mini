"""Deduplicating MoE for L in 2..4 tokens (speculative verify), affine 3-bit.

Consecutive tokens share ~3.3 of 8 routed experts, so reading each distinct
expert's weights once for all tokens that chose it cuts verify MoE bytes ~20%.

  union:   idx [L, K] -> distinct experts uexp[S], token bitmask umask[S], count nu,
           and umap[L, K] (slot of each route)
  gate_up: per slot, per row pair: weights read once, dotted with every token in mask
  down:    same, 4 rows per simdgroup
  combine: (MLX) y[t] = sum_k w[t, k] * yu[umap[t, k], t]
"""

from functools import cache

import mlx.core as mx

from .moe_v4 import _HEADER

_UNION = r"""
    if (thread_position_in_grid.x != 0) return;
    uint n = 0;
    for (uint i = 0; i < L * K; ++i) {
        uint e = idx[i];
        uint t = i / K;
        uint s = 0;
        for (; s < n; ++s) if (uexp[s] == e) break;
        if (s == n) { uexp[n] = e; umask[n] = 0; ++n; }
        umask[s] |= (1u << t);
        umap[i] = s;
    }
    for (uint s = n; s < L * K; ++s) { uexp[s] = 0; umask[s] = 0; }
    nu[0] = n;
"""

_HEADER_M = _HEADER + r"""
    inline float qdot16_regs(const thread uint8_t* b, const thread float* x) {
        float a = 0;
        for (int p = 0; p < 2; ++p) {
            const thread uint8_t* w = b + 3 * p;
            const thread float* t = x + 8 * p;
            a += (w[0] & 0x07) * t[0];
            a += (w[0] & 0x38) * t[1];
            a += (w[0] & 0xc0) * t[2];
            a += (w[1] & 0x01) * (t[2] * 256.0f);
            a += (w[1] & 0x0e) * t[3];
            a += (w[1] & 0x70) * t[4];
            a += (w[1] & 0x80) * t[5];
            a += (w[2] & 0x03) * (t[5] * 256.0f);
            a += (w[2] & 0x1c) * t[6];
            a += (w[2] & 0xe0) * t[7];
        }
        return a;
    }
"""

_GATE_UP = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint slot = threadgroup_position_in_grid.y;
    const uint j0 = sgid * 2;
    if (j0 >= F || slot >= nu[0]) return;
    const uint e = uexp[slot];
    const uint mask = umask[slot];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + (size_t)e * 2 * F * rowbytes;
    const device T* sb = S + (size_t)e * 2 * F * groups;
    const device T* bb = B + (size_t)e * 2 * F * groups;
    const uint rows[4] = {j0, j0 + 1, F + j0, F + j0 + 1};
    float acc[L][4];
    for (uint t = 0; t < L; ++t) for (int r = 0; r < 4; ++r) acc[t][r] = 0;
    float xt[L][16];
    float xs[L];
    for (uint kk = 0; kk < COLS; kk += 512) {
        uint c = kk + lane * 16;
        if (c >= COLS) break;
        for (uint t = 0; t < L; ++t) xs[t] = (mask >> t) & 1u ? load_x16(x + t * COLS + c, xt[t]) : 0.0f;
        uint g = c / 64;
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* w = wb + (size_t)rows[r] * rowbytes + c * 3 / 8;
            uint8_t b[6] = {w[0], w[1], w[2], w[3], w[4], w[5]};
            float sc = float(sb[(size_t)rows[r] * groups + g]);
            float bi = float(bb[(size_t)rows[r] * groups + g]);
            for (uint t = 0; t < L; ++t)
                if ((mask >> t) & 1u) acc[t][r] += sc * qdot16_regs(b, xt[t]) + bi * xs[t];
        }
    }
    for (uint t = 0; t < L; ++t) {
        if (!((mask >> t) & 1u)) continue;
        float a0 = simd_sum(acc[t][0]), a1 = simd_sum(acc[t][1]), a2 = simd_sum(acc[t][2]), a3 = simd_sum(acc[t][3]);
        if (lane < 2 && j0 + lane < F) {
            float up = float(bfloat16_t(lane == 0 ? a0 : a1));
            float gate = float(bfloat16_t(lane == 0 ? a2 : a3));
            float a = metal::precise::tanh(gate / GATE_CAP) * GATE_CAP * (1.0f / (1.0f + metal::precise::exp(-gate)))
                    * metal::precise::tanh(up / UP_CAP) * UP_CAP;
            h[(slot * L + t) * F + j0 + lane] = bfloat16_t(a);
        }
    }
"""

_DOWN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint slot = threadgroup_position_in_grid.y;
    const uint r0 = sgid * 4;
    if (r0 >= D || slot >= nu[0]) return;
    const uint e = uexp[slot];
    const uint mask = umask[slot];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + ((size_t)e * D + r0) * rowbytes;
    const device T* sb = S + ((size_t)e * D + r0) * groups;
    const device T* bb = B + ((size_t)e * D + r0) * groups;
    float acc[L][4];
    for (uint t = 0; t < L; ++t) for (int r = 0; r < 4; ++r) acc[t][r] = 0;
    float xt[L][16];
    float xs[L];
    for (uint kk = 0; kk < COLS; kk += 512) {
        uint c = kk + lane * 16;
        if (c >= COLS) break;
        for (uint t = 0; t < L; ++t) xs[t] = (mask >> t) & 1u ? load_x16(h + (slot * L + t) * COLS + c, xt[t]) : 0.0f;
        uint g = c / 64;
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* w = wb + (size_t)r * rowbytes + c * 3 / 8;
            uint8_t b[6] = {w[0], w[1], w[2], w[3], w[4], w[5]};
            float sc = float(sb[r * groups + g]), bi = float(bb[r * groups + g]);
            for (uint t = 0; t < L; ++t)
                if ((mask >> t) & 1u) acc[t][r] += sc * qdot16_regs(b, xt[t]) + bi * xs[t];
        }
    }
    for (uint t = 0; t < L; ++t) {
        if (!((mask >> t) & 1u)) continue;
        float v0 = simd_sum(acc[t][0]), v1 = simd_sum(acc[t][1]), v2 = simd_sum(acc[t][2]), v3 = simd_sum(acc[t][3]);
        if (lane < 4 && r0 + lane < D) {
            float v = lane == 0 ? v0 : (lane == 1 ? v1 : (lane == 2 ? v2 : v3));
            y[(slot * L + t) * D + r0 + lane] = bfloat16_t(v);
        }
    }
"""


@cache
def _kernels(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    union = mx.fast.metal_kernel(
        name="inkling_moem_union", input_names=["idx"], output_names=["uexp", "umask", "nu", "umap"], source=_UNION,
    )
    gu = mx.fast.metal_kernel(
        name=f"inkling_moem_gu_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "uexp", "umask", "nu"], output_names=["h"], header=_HEADER_M, source=caps(_GATE_UP),
    )
    dn = mx.fast.metal_kernel(
        name="inkling_moem_down", input_names=["h", "W", "S", "B", "uexp", "umask", "nu"], output_names=["y"],
        header=_HEADER_M, source=_DOWN,
    )
    return union, gu, dn


def moe_multi(x, gu_w, dn_w, idx, route_w, gate_cap, up_cap, sg_per_tg=2):
    """x [L, D] bf16; idx [L, K] uint32; route_w [L, K] f32 -> [L, D] bf16 (3-bit affine, g64)."""
    union, k_gu, k_dn = _kernels(float(gate_cap), float(up_cap))
    Wg, Sg, Bg = gu_w
    Wd, Sd, Bd = dn_w
    L, D = x.shape
    K = idx.shape[-1]
    F = Wg.shape[1] // 2
    S = L * K
    uexp, umask, nu, umap = union(
        inputs=[idx.reshape(-1)], template=[("L", L), ("K", K)], grid=(32, 1, 1), threadgroup=(32, 1, 1),
        output_shapes=[(S,), (S,), (1,), (S,)], output_dtypes=[mx.uint32] * 4,
    )
    tg = 32 * sg_per_tg
    (h,) = k_gu(
        inputs=[x, Wg, Sg, Bg, uexp, umask, nu], template=[("T", Sg.dtype), ("F", F), ("COLS", D), ("L", L)],
        grid=(-(-F // (2 * sg_per_tg)) * tg, S, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(S * L * F,)], output_dtypes=[mx.bfloat16], init_value=0,
    )
    (yu,) = k_dn(
        inputs=[h, Wd, Sd, Bd, uexp, umask, nu], template=[("T", Sd.dtype), ("D", D), ("COLS", F), ("L", L)],
        grid=(-(-D // (4 * sg_per_tg)) * tg, S, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(S, L, D)], output_dtypes=[mx.bfloat16], init_value=0,
    )
    rows = umap.reshape(L, K).astype(mx.int32)
    tok = mx.broadcast_to(mx.arange(L, dtype=mx.int32)[:, None], (L, K))
    picked = yu[rows, tok]  # [L, K, D]
    return (picked.astype(mx.float32) * route_w[..., None]).sum(axis=1).astype(x.dtype)
