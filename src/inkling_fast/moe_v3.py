"""Fused one-token MoE kernels, v3: multi-row simdgroups with register-resident x.

Affine 3/4-bit, group 64, MLX's LSB-first bitstream (8 values per BITS bytes).
K1: h[k, j] = softcapped_swiglu(up_k[j] . x, gate_k[j] . x) for ROWS rows per simdgroup.
K2: y[k, d] = down_k[d] . h[k] for ROWS rows per simdgroup (combine stays in MLX).
Each lane owns packs p = lane + 32 * i, loads x for them once, and reuses it for
every row; per pack it accumulates sum(x * q) and sum(x) (bias term).
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
    template <int BITS>
    inline void unpack8(const device uint8_t* b, thread float* q) {
        uint v = uint(b[0]) | (uint(b[1]) << 8) | (uint(b[2]) << 16);
        if (BITS == 4) v |= uint(b[3]) << 24;
        #pragma unroll
        for (int i = 0; i < 8; ++i) q[i] = float((v >> (BITS * i)) & ((1u << BITS) - 1));
    }
"""

# NP = packs per lane (ceil(cols / 8 / 32)); COLS = row length.
_GATE_UP = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint row0 = sgid * ROWS;
    if (row0 >= F) return;
    const uint e = idx[k];
    const uint NPACK = COLS / 8;
    const uint rowbytes = COLS * BITS / 8;
    const uint groups = COLS / 64;
    // x for this lane's packs, in registers
    float xr[NP][8];
    float xs[NP];
    for (uint i = 0; i < NP; ++i) {
        uint p = lane + 32 * i;
        xs[i] = 0;
        for (uint t = 0; t < 8; ++t) {
            float v = p < NPACK ? float(x[p * 8 + t]) : 0.0f;
            xr[i][t] = v; xs[i] += v;
        }
    }
    float acc[2 * ROWS];
    for (uint r = 0; r < 2 * ROWS; ++r) acc[r] = 0;
    const device uint8_t* wbase = (const device uint8_t*)W + (size_t)e * 2 * F * rowbytes;
    for (uint r = 0; r < 2 * ROWS; ++r) {
        uint row = (r < ROWS) ? (row0 + r) : (F + row0 + r - ROWS);   // up rows, then gate rows
        const device uint8_t* wr = wbase + (size_t)row * rowbytes;
        const device T* sr = S + ((size_t)e * 2 * F + row) * groups;
        const device T* br = B + ((size_t)e * 2 * F + row) * groups;
        float a = 0;
        for (uint i = 0; i < NP; ++i) {
            uint p = lane + 32 * i;
            if (p >= NPACK) break;
            float q[8];
            unpack8<BITS>(wr + p * BITS, q);
            float d = 0;
            #pragma unroll
            for (uint t = 0; t < 8; ++t) d += xr[i][t] * q[t];
            uint g = p / 8;
            a += d * float(sr[g]) + xs[i] * float(br[g]);
        }
        acc[r] = a;
    }
    for (uint r = 0; r < 2 * ROWS; ++r) acc[r] = simd_sum(acc[r]);
    if (lane < ROWS && row0 + lane < F) {
        float up = 0, gate = 0;
        for (uint r = 0; r < ROWS; ++r) if (r == lane) { up = acc[r]; gate = acc[ROWS + r]; }
        up = float(bfloat16_t(up));
        gate = float(bfloat16_t(gate));
        float a = metal::precise::tanh(gate / GATE_CAP) * GATE_CAP * (1.0f / (1.0f + metal::precise::exp(-gate)))
                * metal::precise::tanh(up / UP_CAP) * UP_CAP;
        h[k * F + row0 + lane] = bfloat16_t(a);
    }
"""

_DOWN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint row0 = sgid * ROWS;
    if (row0 >= D) return;
    const uint e = idx[k];
    const uint NPACK = COLS / 8;
    const uint rowbytes = COLS * BITS / 8;
    const uint groups = COLS / 64;
    const device bfloat16_t* x = h + k * COLS;
    float xr[NP][8];
    float xs[NP];
    for (uint i = 0; i < NP; ++i) {
        uint p = lane + 32 * i;
        xs[i] = 0;
        for (uint t = 0; t < 8; ++t) {
            float v = p < NPACK ? float(x[p * 8 + t]) : 0.0f;
            xr[i][t] = v; xs[i] += v;
        }
    }
    float acc[ROWS];
    const device uint8_t* wbase = (const device uint8_t*)W + (size_t)e * D * rowbytes;
    for (uint r = 0; r < ROWS; ++r) {
        uint row = row0 + r;
        const device uint8_t* wr = wbase + (size_t)row * rowbytes;
        const device T* sr = S + ((size_t)e * D + row) * groups;
        const device T* br = B + ((size_t)e * D + row) * groups;
        float a = 0;
        for (uint i = 0; i < NP; ++i) {
            uint p = lane + 32 * i;
            if (p >= NPACK) break;
            float q[8];
            unpack8<BITS>(wr + p * BITS, q);
            float d = 0;
            #pragma unroll
            for (uint t = 0; t < 8; ++t) d += xr[i][t] * q[t];
            uint g = p / 8;
            a += d * float(sr[g]) + xs[i] * float(br[g]);
        }
        acc[r] = simd_sum(a);
    }
    if (lane < ROWS && row0 + lane < D) {
        float v = 0;
        for (uint r = 0; r < ROWS; ++r) if (r == lane) v = acc[r];
        y[k * D + row0 + lane] = bfloat16_t(v);
    }
"""


@cache
def _kernels(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    k1 = mx.fast.metal_kernel(
        name=f"inkling_moe3_gate_up_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"], output_names=["h"], header=_HEADER, source=caps(_GATE_UP),
    )
    k2 = mx.fast.metal_kernel(
        name="inkling_moe3_down", input_names=["h", "W", "S", "B", "idx"], output_names=["y"], header=_HEADER, source=_DOWN,
    )
    return k1, k2


def moe_v3(x, gu, dn, idx, gate_cap, up_cap, rows=4, sg_per_tg=4):
    """x [D] bf16; gu = (W [E, 2F, D*b/32], S, B, bits); dn = (W [E, D, F*b/32], S, B, bits);
    idx [K] uint32 -> per-expert down outputs y [K, D] bf16."""
    k1, k2 = _kernels(float(gate_cap), float(up_cap))
    Wg, Sg, Bg, bg = gu
    Wd, Sd, Bd, bd = dn
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    tg = 32 * sg_per_tg
    np1 = -(-D // 8 // 32)
    (h,) = k1(
        inputs=[x, Wg, Sg, Bg, idx],
        template=[("T", Sg.dtype), ("BITS", bg), ("F", F), ("COLS", D), ("ROWS", rows), ("NP", np1)],
        grid=(-(-F // (rows * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, F)], output_dtypes=[mx.bfloat16],
    )
    np2 = -(-F // 8 // 32)
    (y,) = k2(
        inputs=[h, Wd, Sd, Bd, idx],
        template=[("T", Sd.dtype), ("BITS", bd), ("D", D), ("COLS", F), ("ROWS", rows), ("NP", np2)],
        grid=(-(-D // (rows * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, D)], output_dtypes=[mx.bfloat16],
    )
    return y
