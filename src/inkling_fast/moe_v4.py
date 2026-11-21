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


# ---------------------------------------------------------------------------
# v5: exact lane coverage. Rows are 288 packs (D=2304) or 144 packs (F=1152).
#  K1: one simdgroup per row pair (up/gate j, j+1); lane takes packs lane + 32*i, i < NP1.
#  K2: half-simdgroup (16 lanes) per row, 2 rows per half, packs l + 16*i, i < NP2.
# ---------------------------------------------------------------------------

_HEADER5 = _HEADER + r"""
    template <typename T>
    inline float load_x8(const device T* xp, thread float* t) {
        float v0 = float(xp[0]), v1 = float(xp[1]), v2 = float(xp[2]), v3 = float(xp[3]);
        float v4 = float(xp[4]), v5 = float(xp[5]), v6 = float(xp[6]), v7 = float(xp[7]);
        t[0] = v0; t[1] = v1 / 8.0f; t[2] = v2 / 64.0f; t[3] = v3 / 2.0f;
        t[4] = v4 / 16.0f; t[5] = v5 / 128.0f; t[6] = v6 / 4.0f; t[7] = v7 / 32.0f;
        return v0 + v1 + v2 + v3 + v4 + v5 + v6 + v7;
    }
    inline float qdot8(const device uint8_t* b, const thread float* t) {
        float a = (b[0] & 0x07) * t[0];
        a += (b[0] & 0x38) * t[1];
        a += (b[0] & 0xc0) * t[2];
        a += (b[1] & 0x01) * (t[2] * 256.0f);
        a += (b[1] & 0x0e) * t[3];
        a += (b[1] & 0x70) * t[4];
        a += (b[1] & 0x80) * t[5];
        a += (b[2] & 0x03) * (t[5] * 256.0f);
        a += (b[2] & 0x1c) * t[6];
        a += (b[2] & 0xe0) * t[7];
        return a;
    }
"""

_GATE_UP5 = r"""
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
    float t[8];
    for (uint i = 0; i < NP; ++i) {
        uint p = lane + 32 * i;
        float xs = load_x8(x + p * 8, t);
        uint g = p / 8;
        for (int r = 0; r < 4; ++r) {
            const device uint8_t* w = wb + (size_t)rows[r] * rowbytes + p * 3;
            acc[r] += float(sb[(size_t)rows[r] * groups + g]) * qdot8(w, t) + float(bb[(size_t)rows[r] * groups + g]) * xs;
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

_DOWN5 = r"""
    const uint lane = thread_index_in_simdgroup;
    const uint half_ = lane / 16, l = lane % 16;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint r0 = sgid * 4 + half_ * 2;
    if (sgid * 4 >= D) return;
    const uint e = idx[k];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + ((size_t)e * D + r0) * rowbytes;
    const device T* sb = S + ((size_t)e * D + r0) * groups;
    const device T* bb = B + ((size_t)e * D + r0) * groups;
    const device bfloat16_t* x = h + k * COLS;
    float a0 = 0, a1 = 0;
    float t[8];
    for (uint i = 0; i < NP; ++i) {
        uint p = l + 16 * i;
        float xs = load_x8(x + p * 8, t);
        uint g = p / 8;
        a0 += float(sb[g]) * qdot8(wb + p * 3, t) + float(bb[g]) * xs;
        a1 += float(sb[groups + g]) * qdot8(wb + rowbytes + p * 3, t) + float(bb[groups + g]) * xs;
    }
    for (ushort o = 8; o > 0; o >>= 1) {
        a0 += simd_shuffle_xor(a0, o);
        a1 += simd_shuffle_xor(a1, o);
    }
    if (l == 0 && r0 < D) {
        y[k * D + r0] = bfloat16_t(a0);
        if (r0 + 1 < D) y[k * D + r0 + 1] = bfloat16_t(a1);
    }
"""


@cache
def _kernels5(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    k1 = mx.fast.metal_kernel(
        name=f"inkling_moe5_gate_up_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"], output_names=["h"], header=_HEADER5, source=caps(_GATE_UP5),
    )
    k2 = mx.fast.metal_kernel(
        name="inkling_moe5_down", input_names=["h", "W", "S", "B", "idx"], output_names=["y"], header=_HEADER5, source=_DOWN5,
    )
    return k1, k2


def moe_v5(x, gu, dn, idx, gate_cap, up_cap, sg_per_tg=2):
    """Same contract as moe_v4. Requires D % 256 == 0 and F % 128 == 0."""
    k1, k2 = _kernels5(float(gate_cap), float(up_cap))
    Wg, Sg, Bg = gu
    Wd, Sd, Bd = dn
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    tg = 32 * sg_per_tg
    (h,) = k1(
        inputs=[x, Wg, Sg, Bg, idx],
        template=[("T", Sg.dtype), ("F", F), ("COLS", D), ("NP", D // 8 // 32)],
        grid=(-(-F // (2 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, F)], output_dtypes=[mx.bfloat16],
    )
    (y,) = k2(
        inputs=[h, Wd, Sd, Bd, idx],
        template=[("T", Sd.dtype), ("D", D), ("COLS", F), ("NP", F // 8 // 16)],
        grid=(-(-D // (4 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, D)], output_dtypes=[mx.bfloat16],
    )
    return y


# ---------------------------------------------------------------------------
# v6: parameterized v4. PK contiguous packs per lane per step (block = 256*PK),
# RP row pairs (up/gate) per simdgroup in K1, RD rows per simdgroup in K2.
# ---------------------------------------------------------------------------

def _src6_k1(PK, RP):
    rows = ", ".join([f"j0 + {i}" for i in range(RP)] + [f"F + j0 + {i}" for i in range(RP)])
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint j0 = sgid * {RP};
    if (j0 >= F) return;
    const uint e = idx[k];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + (size_t)e * 2 * F * rowbytes;
    const device T* sb = S + (size_t)e * 2 * F * groups;
    const device T* bb = B + (size_t)e * 2 * F * groups;
    const uint rows[{2*RP}] = {{{rows}}};
    float acc[{2*RP}];
    for (int r = 0; r < {2*RP}; ++r) acc[r] = 0;
    float xt[{8*PK}];
    for (uint kk = 0; kk < COLS; kk += {256*PK}) {{
        uint c = kk + lane * {8*PK};
        if (c >= COLS) break;
        float xs[{PK}];
        for (int p = 0; p < {PK}; ++p) xs[p] = load_x8(x + c + 8 * p, xt + 8 * p);
        for (int r = 0; r < {2*RP}; ++r) {{
            const device uint8_t* w = wb + (size_t)rows[r] * rowbytes + c * 3 / 8;
            const device T* sr = sb + (size_t)rows[r] * groups;
            const device T* br = bb + (size_t)rows[r] * groups;
            float a = 0;
            for (int p = 0; p < {PK}; ++p) {{
                uint g = (c + 8 * p) / 64;
                a += float(sr[g]) * qdot8(w + 3 * p, xt + 8 * p) + float(br[g]) * xs[p];
            }}
            acc[r] += a;
        }}
    }}
    for (int r = 0; r < {2*RP}; ++r) acc[r] = simd_sum(acc[r]);
    if (lane < {RP} && j0 + lane < F) {{
        float up = 0, gate = 0;
        for (int r = 0; r < {RP}; ++r) if ((uint)r == lane) {{ up = acc[r]; gate = acc[{RP} + r]; }}
        up = float(bfloat16_t(up));
        gate = float(bfloat16_t(gate));
        float a = metal::precise::tanh(gate / GATE_CAP) * GATE_CAP * (1.0f / (1.0f + metal::precise::exp(-gate)))
                * metal::precise::tanh(up / UP_CAP) * UP_CAP;
        h[k * F + j0 + lane] = bfloat16_t(a);
    }}
"""


def _src6_k2(PK, RD):
    return f"""
    const uint lane = thread_index_in_simdgroup;
    const uint sgid = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint k = threadgroup_position_in_grid.y;
    const uint r0 = sgid * {RD};
    if (r0 >= D) return;
    const uint e = idx[k];
    const uint rowbytes = COLS * 3 / 8;
    const uint groups = COLS / 64;
    const device uint8_t* wb = (const device uint8_t*)W + ((size_t)e * D + r0) * rowbytes;
    const device T* sb = S + ((size_t)e * D + r0) * groups;
    const device T* bb = B + ((size_t)e * D + r0) * groups;
    const device bfloat16_t* x = h + k * COLS;
    float acc[{RD}];
    for (int r = 0; r < {RD}; ++r) acc[r] = 0;
    float xt[{8*PK}];
    for (uint kk = 0; kk < COLS; kk += {256*PK}) {{
        uint c = kk + lane * {8*PK};
        if (c >= COLS) break;
        float xs[{PK}];
        for (int p = 0; p < {PK}; ++p) xs[p] = load_x8(x + c + 8 * p, xt + 8 * p);
        for (int r = 0; r < {RD}; ++r) {{
            const device uint8_t* w = wb + (size_t)r * rowbytes + c * 3 / 8;
            float a = 0;
            for (int p = 0; p < {PK}; ++p) {{
                uint g = (c + 8 * p) / 64;
                a += float(sb[r * groups + g]) * qdot8(w + 3 * p, xt + 8 * p) + float(bb[r * groups + g]) * xs[p];
            }}
            acc[r] += a;
        }}
    }}
    for (int r = 0; r < {RD}; ++r) acc[r] = simd_sum(acc[r]);
    if (lane < {RD} && r0 + lane < D) {{
        float v = 0;
        for (int r = 0; r < {RD}; ++r) if ((uint)r == lane) v = acc[r];
        y[k * D + r0 + lane] = bfloat16_t(v);
    }}
"""


@cache
def _kernels6(gate_cap, up_cap, pk1, rp, pk2, rd):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    k1 = mx.fast.metal_kernel(
        name=f"inkling_moe6_gu_{pk1}_{rp}_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"], output_names=["h"], header=_HEADER5, source=caps(_src6_k1(pk1, rp)),
    )
    k2 = mx.fast.metal_kernel(
        name=f"inkling_moe6_dn_{pk2}_{rd}", input_names=["h", "W", "S", "B", "idx"], output_names=["y"],
        header=_HEADER5, source=_src6_k2(pk2, rd),
    )
    return k1, k2


def moe_v6(x, gu, dn, idx, gate_cap, up_cap, pk1=2, rp=2, pk2=2, rd=4, sg1=2, sg2=2):
    k1, k2 = _kernels6(float(gate_cap), float(up_cap), pk1, rp, pk2, rd)
    Wg, Sg, Bg = gu
    Wd, Sd, Bd = dn
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    (h,) = k1(
        inputs=[x, Wg, Sg, Bg, idx], template=[("T", Sg.dtype), ("F", F), ("COLS", D)],
        grid=(-(-F // (rp * sg1)) * 32 * sg1, K, 1), threadgroup=(32 * sg1, 1, 1),
        output_shapes=[(K, F)], output_dtypes=[mx.bfloat16],
    )
    (y,) = k2(
        inputs=[h, Wd, Sd, Bd, idx], template=[("T", Sd.dtype), ("D", D), ("COLS", F)],
        grid=(-(-D // (rd * sg2)) * 32 * sg2, K, 1), threadgroup=(32 * sg2, 1, 1),
        output_shapes=[(K, D)], output_dtypes=[mx.bfloat16],
    )
    return y


# ---------------------------------------------------------------------------
# v7: v4 with 16-bit weight loads (each lane's 6 bytes are 2-byte aligned).
# ---------------------------------------------------------------------------

_HEADER7 = _HEADER + r"""
    inline void load6(const device uint8_t* w, thread uint8_t* b) {
        const device ushort* u = (const device ushort*)w;
        ushort a0 = u[0], a1 = u[1], a2 = u[2];
        b[0] = a0 & 0xff; b[1] = a0 >> 8; b[2] = a1 & 0xff; b[3] = a1 >> 8; b[4] = a2 & 0xff; b[5] = a2 >> 8;
    }
    inline float qdot16r(const thread uint8_t* bb, const thread float* x) {
        float a = 0;
        for (int p = 0; p < 2; ++p) {
            const thread uint8_t* b = bb + 3 * p;
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

_GATE_UP7 = _GATE_UP.replace(
    "acc[r] += sc * qdot16(w, xt) + bi * xs;",
    "uint8_t bb6[6]; load6(w, bb6); acc[r] += sc * qdot16r(bb6, xt) + bi * xs;",
)
_DOWN7 = _DOWN.replace(
    "acc[r] += float(sb[r * groups + g]) * qdot16(w, xt) + float(bb[r * groups + g]) * xs;",
    "uint8_t bb6[6]; load6(w, bb6); acc[r] += float(sb[r * groups + g]) * qdot16r(bb6, xt) + float(bb[r * groups + g]) * xs;",
)


@cache
def _kernels7(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    k1 = mx.fast.metal_kernel(
        name=f"inkling_moe7_gate_up_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"], output_names=["h"], header=_HEADER7, source=caps(_GATE_UP7),
    )
    k2 = mx.fast.metal_kernel(
        name="inkling_moe7_down", input_names=["h", "W", "S", "B", "idx"], output_names=["y"], header=_HEADER7, source=_DOWN7,
    )
    return k1, k2


def moe_v7(x, gu, dn, idx, gate_cap, up_cap, sg_per_tg=2):
    k1, k2 = _kernels7(float(gate_cap), float(up_cap))
    Wg, Sg, Bg = gu
    Wd, Sd, Bd = dn
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    tg = 32 * sg_per_tg
    (h,) = k1(
        inputs=[x, Wg, Sg, Bg, idx], template=[("T", Sg.dtype), ("F", F), ("COLS", D)],
        grid=(-(-F // (2 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, F)], output_dtypes=[mx.bfloat16],
    )
    (y,) = k2(
        inputs=[h, Wd, Sd, Bd, idx], template=[("T", Sd.dtype), ("D", D), ("COLS", F)],
        grid=(-(-D // (4 * sg_per_tg)) * tg, K, 1), threadgroup=(tg, 1, 1),
        output_shapes=[(K, D)], output_dtypes=[mx.bfloat16],
    )
    return y
