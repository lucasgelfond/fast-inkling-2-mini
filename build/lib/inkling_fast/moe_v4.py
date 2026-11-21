"""One-token 3-bit MoE gather kernels, v4: MLX qmv_fast structure for Inkling shapes.

MLX dispatches its generic gather_qmv for D=2304/F=1152 (1 pack per lane). Here:
  * x is pre-scaled per pack (1, 1/8, 1/64, 1/2, 1/16, 1/128, 1/4, 1/32) so masked
    bytes multiply directly (no shifts), as in MLX's 3-bit qdot;
  * 2 packs (16 values) per lane per step, block 512, masked tail lanes;
  * K1 computes up/gate rows j, j+1 in one simdgroup and applies the softcapped
    SwiGLU; K2 computes 4 down rows per simdgroup. Combine stays in MLX.
Affine 3-bit, group 64, LSB-first bitstream (8 values per 3 bytes).
"""

from functools import cache

import mlx.core as mx

_HEADER = r"""
    // Load 16 values (2 packs) of x pre-scaled for masked-byte products; return their sum.
    template <typename T>
    inline float load_x16(const device T* x, thread float* xt) {
        float s = 0;
        for (int p = 0; p < 2; ++p) {
            const device T* xp = x + 8 * p;
            thread float* t = xt + 8 * p;
            float v0 = float(xp[0]), v1 = float(xp[1]), v2 = float(xp[2]), v3 = float(xp[3]);
            float v4 = float(xp[4]), v5 = float(xp[5]), v6 = float(xp[6]), v7 = float(xp[7]);
            s += v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
            t[0] = v0; t[1] = v1 / 8.0f; t[2] = v2 / 64.0f; t[3] = v3 / 2.0f;
            t[4] = v4 / 16.0f; t[5] = v5 / 128.0f; t[6] = v6 / 4.0f; t[7] = v7 / 32.0f;
        }
        return s;
    }
    // Masked-byte dot of 2 packs (6 bytes) with pre-scaled x.
    inline float qdot16(const device uint8_t* w, const thread float* x) {
        float a = 0;
        for (int p = 0; p < 2; ++p) {
            const device uint8_t* b = w + 3 * p;
            const thread float* t = x + 8 * p;
            a += (b[0] & 0x07) * t[0];
            a += (b[0] & 0x38) * t[1];
            a += (b[0] & 0xc0) * t[2];
            a += (b[1] & 0x01) * (t[2] * 256.0f);
            a += (b[1] & 0x0e) * t[3];
            a += (b[1] & 0x70) * t[4];
            a += (b[1] & 0x80) * t[5];
            a += (b[2] & 0x03) * (t[5] * 256.0f);
            a += (b[2] & 0x1c) * t[6];
            a += (b[2] & 0xe0) * t[7];
        }
        return a;
    }
"""

# K1: rows (j0, j0+1) of up and of gate for expert idx[k]; COLS = D.
_GATE_UP = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint j0 = sgid * 2;
    if (j0 >= F) return;
    const uint e = idx[k];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + (size_t)e * 2 * F * rowbytes;
    const device T* sb = S + (size_t)e * 2 * F * groups;
    const device T* bb = B + (size_t)e * 2 * F * groups;
    const uint rows[4] = {j0, j0 + 1, F + j0, F + j0 + 1};
    float acc[4] = {0, 0, 0, 0};
    float xt[16];
    for (uint kk = 0; kk < COLS; kk += 512) {
        uint c = kk + lane * 16;
        if (c >= COLS) break;
        float xs = load_x16(x + c, xt);
        uint g = c / 64;
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* w = wb + (size_t)rows[r] * rowbytes + c * 3 / 8;
            float sc = float(sb[(size_t)rows[r] * groups + g]);
            float bi = float(bb[(size_t)rows[r] * groups + g]);
            acc[r] += sc * qdot16(w, xt) + bi * xs;
        }
    }
    for (int r = 0; r < 4; ++r) acc[r] = simd_sum(acc[r]);
    if (lane < 2 && j0 + lane < F) {
        float up = lane == 0 ? acc[0] : acc[1];
        float gate = lane == 0 ? acc[2] : acc[3];
        up = float(bfloat16_t(up));
        gate = float(bfloat16_t(gate));
        float a = metal::precise::tanh(gate / GATE_CAP) * GATE_CAP * (1.0f / (1.0f + metal::precise::exp(-gate)))
                * metal::precise::tanh(up / UP_CAP) * UP_CAP;
        h[k * F + j0 + lane] = bfloat16_t(a);
    }
"""

# K2: 4 down rows per simdgroup for expert idx[k]; COLS = F.
_DOWN = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint r0 = sgid * 4;
    if (r0 >= D) return;
    const uint e = idx[k];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + ((size_t)e * D + r0) * rowbytes;
    const device T* sb = S + ((size_t)e * D + r0) * groups;
    const device T* bb = B + ((size_t)e * D + r0) * groups;
    const device bfloat16_t* x = h + k * COLS;
    float acc[4] = {0, 0, 0, 0};
    float xt[16];
    for (uint kk = 0; kk < COLS; kk += 512) {
        uint c = kk + lane * 16;
        if (c >= COLS) break;
        float xs = load_x16(x + c, xt);
        uint g = c / 64;
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* w = wb + (size_t)r * rowbytes + c * 3 / 8;
            acc[r] += float(sb[r * groups + g]) * qdot16(w, xt) + float(bb[r * groups + g]) * xs;
        }
    }
    for (int r = 0; r < 4; ++r) acc[r] = simd_sum(acc[r]);
    if (lane < 4 && r0 + lane < D) {
        float v = lane == 0 ? acc[0] : (lane == 1 ? acc[1] : (lane == 2 ? acc[2] : acc[3]));
        y[k * D + r0 + lane] = bfloat16_t(v);
    }
"""


@cache
def _kernels(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    k1 = mx.fast.metal_kernel(
        name=f"inkling_moe4_gate_up_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"], output_names=["h"], header=_HEADER, source=caps(_GATE_UP),
    )
    k2 = mx.fast.metal_kernel(
        name="inkling_moe4_down", input_names=["h", "W", "S", "B", "idx"], output_names=["y"], header=_HEADER, source=_DOWN,
    )
    return k1, k2


def moe_v4(x, gu, dn, idx, gate_cap, up_cap, sg_per_tg=2):
    """x [D] bf16; gu = (W [E, 2F, D*3/32], S, B); dn = (W [E, D, F*3/32], S, B);
    idx [K] uint32 -> per-expert down outputs y [K, D] bf16. 3-bit, group 64."""
    k1, k2 = _kernels(float(gate_cap), float(up_cap))
    Wg, Sg, Bg = gu
    Wd, Sd, Bd = dn
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    tg = 32 * sg_per_tg
    (h,) = k1(
        inputs=[x, Wg, Sg, Bg, idx],
        template=[("T", Sg.dtype), ("F", F), ("COLS", D)],
        grid=(-(-F // (2 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, F)], output_dtypes=[mx.bfloat16],
    )
    (y,) = k2(
        inputs=[h, Wd, Sd, Bd, idx],
        template=[("T", Sd.dtype), ("D", D), ("COLS", F)],
        grid=(-(-D // (4 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, D)], output_dtypes=[mx.bfloat16],
    )
    return y
