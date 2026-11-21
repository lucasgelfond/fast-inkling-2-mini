"""Custom Metal kernels for Inkling 2 Mini decode (batch 1, one token).

Decode is a long serial chain of tiny kernels, so latency per kernel, not
bandwidth, sets the speed. These kernels collapse multi-op chains into one or
two launches each.
"""

from functools import cache

import mlx.core as mx

# ---------------------------------------------------------------------------
# Message gate: out = (1 - tanh(g @ Wf^T)) * h + tanh(g @ Wg^T) * m,
# g = bf16(down(bf16(concat(rmsnorm(h), rmsnorm(m))))). G gates per launch.
# ---------------------------------------------------------------------------

_GATE_DOWN = r"""
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint r = threadgroup_position_in_grid.y;
    uint g = threadgroup_position_in_grid.z;
    const uint D = DIM;
    const uint NT = 256;
    threadgroup float red[3][8];
    const device float* h = H + g * D;
    const device float* m = M + g * D;
    float sh = 0, sm = 0;
    for (uint i = tid; i < D; i += NT) {
        float a = h[i], b = m[i];
        sh += a * a; sm += b * b;
    }
    sh = simd_sum(sh); sm = simd_sum(sm);
    if (lane == 0) { red[0][sg] = sh; red[1][sg] = sm; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        sh = lane < 8 ? red[0][lane] : 0;
        sm = lane < 8 ? red[1][lane] : 0;
        sh = simd_sum(sh); sm = simd_sum(sm);
        if (lane == 0) { red[0][0] = sh; red[1][0] = sm; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ih = metal::precise::rsqrt(red[0][0] / D + EPS);
    float im = metal::precise::rsqrt(red[1][0] / D + EPS);
    const device bfloat16_t* w = W + ((g / TOKENS) * R + r) * 2 * D;
    float acc = 0;
    for (uint i = tid; i < D; i += NT) {
        acc += float(bfloat16_t(h[i] * ih)) * float(w[i]);
        acc += float(bfloat16_t(m[i] * im)) * float(w[D + i]);
    }
    acc = simd_sum(acc);
    if (lane == 0) red[2][sg] = acc;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        acc = lane < 8 ? red[2][lane] : 0;
        acc = simd_sum(acc);
        if (lane == 0) Y[g * R + r] = float(bfloat16_t(acc));
    }
"""

_GATE_UP = r"""
    uint c = thread_position_in_grid.x;
    uint g = thread_position_in_grid.y;
    const uint D = DIM;
    if (c >= D) return;
    const device float* y = Y + g * R;
    const device bfloat16_t* wf = WF + ((g / TOKENS) * D + c) * R;
    const device bfloat16_t* wg = WG + ((g / TOKENS) * D + c) * R;
    float f = 0, n = 0;
    for (uint r = 0; r < R; ++r) {
        float v = y[r];
        f += v * float(wf[r]);
        n += v * float(wg[r]);
    }
    f = metal::precise::tanh(f);
    n = metal::precise::tanh(n);
    out[g * D + c] = (1.0f - f) * H[g * D + c] + n * M[g * D + c];
"""


@cache
def _gate_kernels(eps):
    down = mx.fast.metal_kernel(
        name=f"inkling_fast_gate_down_{abs(hash(eps))}",
        input_names=["H", "M", "W"],
        output_names=["Y"],
        source=_GATE_DOWN.replace("EPS", repr(float(eps)) + "f"),
    )
    up = mx.fast.metal_kernel(
        name="inkling_fast_gate_up",
        input_names=["Y", "WF", "WG", "H", "M"],
        output_names=["out"],
        source=_GATE_UP,
    )
    return down, up


def message_gates(h, m, wd, wf, wg, eps, tokens=1):
    """h, m: [G * tokens, D] f32 (row = gate * tokens + t); wd: [G, R, 2D]; wf, wg: [G, D, R]."""
    down, up = _gate_kernels(float(eps))
    G, D = h.shape
    R = wd.shape[1]
    tmpl = [("DIM", D), ("R", R), ("TOKENS", tokens)]
    (y,) = down(
        inputs=[h, m, wd],
        template=tmpl,
        grid=(256, R, G),
        threadgroup=(256, 1, 1),
        output_shapes=[(G, R)],
        output_dtypes=[mx.float32],
    )
    (out,) = up(
        inputs=[y, wf, wg, h, m],
        template=tmpl,
        grid=(D, G, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(G, D)],
        output_dtypes=[mx.float32],
    )
    return out


def stack_gates(gates):
    """Stack MessageGate weights for one batched launch."""
    wd = mx.stack([g.down_proj.weight for g in gates]).astype(mx.bfloat16)
    wf = mx.stack([g.forget_gate_proj.weight for g in gates]).astype(mx.bfloat16)
    wg = mx.stack([g.message_gate_proj.weight for g in gates]).astype(mx.bfloat16)
    mx.eval(wd, wf, wg)
    return wd, wf, wg


# ---------------------------------------------------------------------------
# Fused MoE decode (one token): K1 = gathered gate+up + softcapped SwiGLU,
# K2 = gathered down + routed combine. Affine 3/4-bit, group size 64,
# LSB-first bitstream with 8 values per BITS bytes (MLX's layout).
# ---------------------------------------------------------------------------

_MOE_COMMON = r"""
    // Dot of a packed affine row (BITS = 3 or 4, group 64) with bf16 x.
    // Each lane takes 32-value chunks (BITS aligned uint32 words), lane-strided.
    template <int BITS, typename T>
    inline float qdot(const device uint32_t* row, const device T* sc, const device T* bi,
                      const device bfloat16_t* x, uint nchunks, uint lane) {
        float acc = 0;
        for (uint c = lane; c < nchunks; c += 32) {
            uint w[BITS];
            for (int j = 0; j < BITS; ++j) w[j] = row[c * BITS + j];
            const device bfloat16_t* xp = x + c * 32;
            float sxq = 0, sx = 0;
            #pragma unroll
            for (int i = 0; i < 32; ++i) {
                const int bit = BITS * i, word = bit >> 5, off = bit & 31;
                uint v = w[word] >> off;
                if (off > 32 - BITS) v |= w[word + 1] << (32 - off);
                float xi = float(xp[i]);
                sxq += xi * float(v & ((1u << BITS) - 1));
                sx += xi;
            }
            uint grp = c / 2;  // 64-value groups, 32-value chunks
            acc += sxq * float(sc[grp]) + sx * float(bi[grp]);
        }
        return simd_sum(acc);
    }
"""

_MOE_GATE_UP = r"""
    uint lane = thread_index_in_simdgroup;
    uint row = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    uint k = threadgroup_position_in_grid.y;           // route slot
    if (row >= F) return;
    uint e = idx[k];
    const uint rowwords = D * BITS / 32;
    const uint groups = D / 64;
    const device uint32_t* base = W + (size_t)e * 2 * F * rowwords;
    const device T* sbase = S + (size_t)e * 2 * F * groups;
    const device T* bbase = B + (size_t)e * 2 * F * groups;
    float up = qdot<BITS, T>(base + row * rowwords, sbase + row * groups, bbase + row * groups, x, D / 32, lane);
    float gate = qdot<BITS, T>(base + (F + row) * rowwords, sbase + (F + row) * groups, bbase + (F + row) * groups, x, D / 32, lane);
    if (lane == 0) {
        up = float(bfloat16_t(up));
        gate = float(bfloat16_t(gate));
        float a = metal::precise::tanh(gate / GATE_CAP) * GATE_CAP * (1.0f / (1.0f + metal::precise::exp(-gate)))
                * metal::precise::tanh(up / UP_CAP) * UP_CAP;
        h[k * F + row] = bfloat16_t(a);
    }
"""

_MOE_DOWN = r"""
    uint lane = thread_index_in_simdgroup;
    uint d = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    uint k = threadgroup_position_in_grid.y;
    if (d >= D) return;
    uint e = idx[k];
    const uint rowwords = F * BITS / 32;
    const uint groups = F / 64;
    size_t r = (size_t)e * D + d;
    float y = qdot<BITS, T>(W + r * rowwords, S + r * groups, B + r * groups, h + k * F, F / 32, lane);
    if (lane == 0) out[k * D + d] = bfloat16_t(y);
"""


@cache
def _moe_kernels(gate_cap, up_cap):
    def caps(s):
        return s.replace("GATE_CAP", repr(float(gate_cap)) + "f").replace("UP_CAP", repr(float(up_cap)) + "f")
    gate_up = mx.fast.metal_kernel(
        name=f"inkling_fast_moe_gate_up_{abs(hash((gate_cap, up_cap)))}",
        input_names=["x", "W", "S", "B", "idx"],
        output_names=["h"],
        header=_MOE_COMMON,
        source=caps(_MOE_GATE_UP),
    )
    down = mx.fast.metal_kernel(
        name="inkling_fast_moe_down",
        input_names=["h", "W", "S", "B", "idx"],
        output_names=["out"],
        header=_MOE_COMMON,
        source=_MOE_DOWN,
    )
    return gate_up, down


def moe_decode(x, gate_up_w, down, idx, route_w, gate_cap, up_cap, rows_per_tg=4):
    """x: [D] bf16; gate_up_w = (W [E, 2F, D*b/32], S, B, bits); down = (W, S, B, bits);
    idx: [K] uint32; route_w: [K] f32 -> [D] bf16."""
    k_gate_up, k_down = _moe_kernels(float(gate_cap), float(up_cap))
    Wg, Sg, Bg, bits_g = gate_up_w
    Wd, Sd, Bd, bits_d = down
    D = x.shape[-1]
    F = Wg.shape[1] // 2
    K = idx.shape[-1]
    tg = 32 * rows_per_tg
    (h,) = k_gate_up(
        inputs=[x, Wg, Sg, Bg, idx],
        template=[("T", Sg.dtype), ("BITS", bits_g), ("D", D), ("F", F)],
        grid=(((F + rows_per_tg - 1) // rows_per_tg) * tg, K, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(K, F)],
        output_dtypes=[mx.bfloat16],
    )
    (y,) = k_down(
        inputs=[h, Wd, Sd, Bd, idx],
        template=[("T", Sd.dtype), ("BITS", bits_d), ("D", D), ("F", F)],
        grid=(((D + rows_per_tg - 1) // rows_per_tg) * tg, K, 1),
        threadgroup=(tg, 1, 1),
        output_shapes=[(K, D)],
        output_dtypes=[mx.bfloat16],
    )
    # Routed combine as plain ops so mx.compile fuses it with what follows.
    return (y.astype(mx.float32) * route_w[:, None]).sum(axis=0).astype(mx.bfloat16)


# ---------------------------------------------------------------------------
# Attention-residual mixing (learned-query softmax over block states), one
# threadgroup per token. Replaces concat + rms_norm + matmul + softmax + matmul,
# and optionally the RMSNorm that follows (per query, with weight).
#   keys_n = rmsnorm(state_n); w = softmax_n(keys_n . q / sqrt(D));
#   out_q = sum_n w_nq * keys_n   (normalize_values) ; optional: rmsnorm(out_q) * nw_q
# ---------------------------------------------------------------------------

_MIX_NT = 768


def _mix_source(n, nq, fuse_norm, normalize_values):
    lines = []
    lines.append(f"""
    const uint D = DIM;
    const uint NT = {_MIX_NT};
    const uint NS = {n};
    const uint NQ = {nq};
    const uint PER = (DIM + NT - 1) / NT;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint t = threadgroup_position_in_grid.y;
    threadgroup float red[NT / 32][NS + NS * NQ + NQ];
    threadgroup float tot[NS + NS * NQ + NQ];
    float x[NS][PER];
""")
    for i in range(n):
        lines.append(f"    for (uint j = 0; j < PER; ++j) {{ uint c = tid + j * NT; x[{i}][j] = c < D ? float(s{i}[t * D + c]) : 0.0f; }}\n")
    lines.append("""
    // pass 1: sum of squares per state, raw dots per (state, query)
    float acc[NS + NS * NQ];
    for (uint k = 0; k < NS + NS * NQ; ++k) acc[k] = 0;
    for (uint j = 0; j < PER; ++j) {
        uint c = tid + j * NT;
        if (c >= D) break;
        for (uint s = 0; s < NS; ++s) {
            float v = x[s][j];
            acc[s] += v * v;
            for (uint q = 0; q < NQ; ++q) acc[NS + s * NQ + q] += v * float(Q[q * D + c]);
        }
    }
    for (uint k = 0; k < NS + NS * NQ; ++k) {
        float v = simd_sum(acc[k]);
        if (lane == 0) red[sg][k] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NS + NS * NQ) {
        float v = 0;
        for (uint g = 0; g < NT / 32; ++g) v += red[g][tid];
        tot[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float inv[NS];
    for (uint s = 0; s < NS; ++s) inv[s] = metal::precise::rsqrt(tot[s] / D + 1e-6f);
    // softmax over states, per query
    float w[NS][NQ];
    const float scale = metal::precise::rsqrt(float(D));
    for (uint q = 0; q < NQ; ++q) {
        float m = -INFINITY;
        for (uint s = 0; s < NS; ++s) { w[s][q] = tot[NS + s * NQ + q] * inv[s] * scale; m = max(m, w[s][q]); }
        float z = 0;
        for (uint s = 0; s < NS; ++s) { w[s][q] = metal::precise::exp(w[s][q] - m); z += w[s][q]; }
        for (uint s = 0; s < NS; ++s) w[s][q] /= z;
    }
    // pass 2: weighted sum of (normalized) states
    float o[NQ][PER];
    float sq[NQ];
    for (uint q = 0; q < NQ; ++q) sq[q] = 0;
    for (uint j = 0; j < PER; ++j) {
        for (uint q = 0; q < NQ; ++q) {
            float v = 0;
            for (uint s = 0; s < NS; ++s) v += w[s][q] * x[s][j] * VALSCALE;
            o[q][j] = v;
            sq[q] += (tid + j * NT < D) ? v * v : 0.0f;
        }
    }
""".replace("VALSCALE", "inv[s]" if normalize_values else "1.0f"))
    if fuse_norm:
        lines.append("""
    for (uint q = 0; q < NQ; ++q) {
        float v = simd_sum(sq[q]);
        if (lane == 0) red[sg][q] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < NQ) {
        float v = 0;
        for (uint g = 0; g < NT / 32; ++g) v += red[g][tid];
        tot[tid] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint q = 0; q < NQ; ++q) {
        float r = metal::precise::rsqrt(tot[q] / D + EPS);
        for (uint j = 0; j < PER; ++j) {
            uint c = tid + j * NT;
            if (c < D) out[(t * NQ + q) * D + c] = OutT(o[q][j] * r * float(NW[q * D + c]));
        }
    }
""")
    else:
        lines.append("""
    for (uint q = 0; q < NQ; ++q)
        for (uint j = 0; j < PER; ++j) {
            uint c = tid + j * NT;
            if (c < D) out[(t * NQ + q) * D + c] = OutT(o[q][j]);
        }
""")
    return "".join(lines)


@cache
def _mix_kernel(n, nq, fuse_norm, normalize_values, eps):
    src = _mix_source(n, nq, fuse_norm, normalize_values).replace("EPS", repr(float(eps)) + "f")
    names = [f"s{i}" for i in range(n)] + ["Q"] + (["NW"] if fuse_norm else [])
    return mx.fast.metal_kernel(
        name=f"inkling_fast_mix_{n}_{nq}_{int(fuse_norm)}_{int(normalize_values)}_{abs(hash(eps))}",
        input_names=names,
        output_names=["out"],
        source=src,
    )


def attention_mix(states, query, normalize_values=True, norm_weights=None, eps=1e-6, out_dtype=mx.float32):
    """states: list of [1, L, D] arrays (blocks then partial); query: [D] or [NQ, D].
    Returns [1, L, NQ, D] (out_dtype); RMSNorm'd with norm_weights [NQ, D] if given."""
    n = len(states)
    q = query.reshape(-1, query.shape[-1]).astype(mx.float32)
    nq, d = q.shape
    L = states[-1].shape[-2]
    fuse = norm_weights is not None
    kernel = _mix_kernel(n, nq, fuse, bool(normalize_values), float(eps))
    inputs = [s.reshape(L, d) for s in states] + [q] + ([norm_weights] if fuse else [])
    (out,) = kernel(
        inputs=inputs,
        template=[("DIM", d), ("OutT", out_dtype)],
        grid=(_MIX_NT, L, 1),
        threadgroup=(_MIX_NT, 1, 1),
        output_shapes=[(L, nq, d)],
        output_dtypes=[out_dtype],
    )
    return out.reshape(1, L, nq, d)


# ---------------------------------------------------------------------------
# MoE router for one token: scores = sigmoid(x . Wg_e), top-K of scores + bias,
# weights = scores[top] / sum * route_scale. One threadgroup.
# ---------------------------------------------------------------------------

_ROUTE = r"""
    const uint D = DIM;
    uint tid = thread_position_in_threadgroup.x;
    uint lane = thread_index_in_simdgroup;
    uint sg = simdgroup_index_in_threadgroup;
    uint nsg = threads_per_threadgroup.x / 32;
    threadgroup float score[E];
    threadgroup float key[E];
    threadgroup float picked[K];
    threadgroup uint pidx[K];
    for (uint e = sg; e < E; e += nsg) {
        float acc = 0;
        for (uint i = lane; i < D; i += 32) acc += float(x[i]) * float(W[e * D + i]);
        acc = simd_sum(acc);
        if (lane == 0) {
            float sc = 1.0f / (1.0f + metal::precise::exp(-acc));
            score[e] = sc;
            key[e] = sc + float(bias[e]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Parallel rank: each expert counts how many keys beat it (ties broken by index).
    if (tid < E) {
        float kv = key[tid];
        uint rank = 0;
        for (uint j = 0; j < E; ++j) {
            float kj = key[j];
            rank += (kj > kv || (kj == kv && j < tid)) ? 1u : 0u;
        }
        if (rank < K) { pidx[rank] = tid; picked[rank] = score[tid]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {
        float total = 0;
        for (uint k = 0; k < K; ++k) total += picked[k];
        idx[tid] = pidx[tid];
        rw[tid] = picked[tid] / (total + 1e-20f) * SCALE;
    }
"""


@cache
def _route_kernel(scale):
    return mx.fast.metal_kernel(
        name=f"inkling_fast_route_{abs(hash(scale))}",
        input_names=["x", "W", "bias"],
        output_names=["idx", "rw"],
        source=_ROUTE.replace("SCALE", repr(float(scale)) + "f"),
    )


def route_one(x, weight, bias, top_k, route_scale):
    """x: [D]; weight: [E, D]; bias: [E] -> (idx [K] uint32, weights [K] f32)."""
    E, D = weight.shape
    idx, rw = _route_kernel(float(route_scale))(
        inputs=[x, weight, bias],
        template=[("DIM", D), ("E", E), ("K", top_k)],
        grid=(1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(top_k,), (top_k,)],
        output_dtypes=[mx.uint32, mx.float32],
    )
    return idx, rw


# ---------------------------------------------------------------------------
# Sliding-layer post-projection for one token. One simdgroup per head-vector:
#   q heads: l2_clip(q * q_scale)            -> bf16
#   k heads: sconv(k) -> l2_clip             -> bf16, new conv window
#   v heads: sconv(v) -> rmsnorm (no gamma)  -> bf16, new conv window
#   r heads: l2_clip(r * rel_scale)          -> bf16
# Rounding points follow the reference (bf16 between stages).
# ---------------------------------------------------------------------------

_POSTPROJ = r"""
    uint lane = thread_index_in_simdgroup;
    uint vec = threadgroup_position_in_grid.x * (threads_per_threadgroup.x / 32) + simdgroup_index_in_threadgroup;
    const uint HD = HEAD_DIM, NH = N_HEADS, NKV = N_KV, DR = D_REL;
    const uint PER = HD / 32;
    const float DIV = metal::precise::sqrt(float(KSZ + 1));
    if (vec < NH) {                               // q head
        uint h = vec;
        float v[PER]; float ss = 0;
        for (uint j = 0; j < PER; ++j) {
            uint i = lane * PER + j;
            v[j] = float(raw[QOFF + h * HD + i]) * float(qs[h * HD + i]);
            ss += v[j] * v[j];
        }
        ss = simd_sum(ss);
        float f = min(QK_THR / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
        for (uint j = 0; j < PER; ++j) q[h * HD + lane * PER + j] = bfloat16_t(v[j] * f);
    } else if (vec < NH + 2 * NKV) {              // k or v head (with short conv)
        bool isv = vec >= NH + NKV;
        uint h = vec - NH - (isv ? NKV : 0);
        const device float* st = isv ? sv : sk;
        const device bfloat16_t* w = isv ? wv : wk;
        uint base = isv ? VOFF : KOFF;
        float v[PER]; float ss = 0;
        for (uint j = 0; j < PER; ++j) {
            uint c = h * HD + lane * PER + j;
            const uint C = NKV * HD;
            float x = float(raw[base + c]);
            float s0 = st[c], s1 = st[C + c], s2 = st[2 * C + c];
            float o = s0 * float(w[c * KSZ + 0]) + s1 * float(w[c * KSZ + 1]) + s2 * float(w[c * KSZ + 2]) + x * float(w[c * KSZ + 3]);
            o = float(bfloat16_t((o + x) / DIV));
            device float* ctx = isv ? cv : ck;
            ctx[c] = s0; ctx[C + c] = s1; ctx[2 * C + c] = s2; ctx[3 * C + c] = x;
            v[j] = o; ss += o * o;
        }
        ss = simd_sum(ss);
        if (isv) {
            float r = metal::precise::rsqrt(ss / HD + EPS);
            for (uint j = 0; j < PER; ++j) vo[h * HD + lane * PER + j] = bfloat16_t(v[j] * r);
        } else {
            float f = min(QK_THR / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
            for (uint j = 0; j < PER; ++j) ko[h * HD + lane * PER + j] = bfloat16_t(v[j] * f);
        }
    } else if (vec < 2 * NH + 2 * NKV) {          // r head (d_rel = 16 < 32 lanes)
        uint h = vec - NH - 2 * NKV;
        float v = lane < DR ? float(raw[ROFF + h * DR + lane]) * float(rs[h * DR + lane]) : 0.0f;
        float ss = simd_sum(v * v);
        float f = min(REL_THR / (metal::precise::sqrt(ss) + 1e-6f), 1.0f);
        if (lane < DR) r[h * DR + lane] = bfloat16_t(v * f);
    }
"""


@cache
def _postproj_kernel(qk_thr, rel_thr, eps):
    src = (_POSTPROJ.replace("QK_THR", repr(float(qk_thr)) + "f")
           .replace("REL_THR", repr(float(rel_thr)) + "f").replace("EPS", repr(float(eps)) + "f"))
    return mx.fast.metal_kernel(
        name=f"inkling_fast_postproj_{abs(hash((qk_thr, rel_thr, eps)))}",
        input_names=["raw", "qs", "rs", "sk", "sv", "wk", "wv"],
        output_names=["q", "ko", "vo", "r", "ck", "cv"],
        source=src,
    )


def sliding_postproj(raw, offsets, q_scale, rel_scale, conv_k, conv_v, wk, wv, n_heads, n_kv, head_dim, d_rel, qk_thr, rel_thr, eps):
    """raw: fused projection row [k|v|q|r|g] (bf16, flat). Returns q [1,H,1,hd], k, v [1,KV,1,hd],
    r [1,1,H,dr] (bf16) and conv contexts ck, cv [1,4,KV*hd] f32."""
    kernel = _postproj_kernel(float(qk_thr), float(rel_thr), float(eps))
    nvec = 2 * n_heads + 2 * n_kv
    sg_per_tg = 8
    C = n_kv * head_dim
    q, k, v, r, ck, cv = kernel(
        inputs=[raw, q_scale, rel_scale, conv_k, conv_v, wk, wv],
        template=[("HEAD_DIM", head_dim), ("N_HEADS", n_heads), ("N_KV", n_kv), ("D_REL", d_rel), ("KSZ", wk.shape[-1]),
                  ("KOFF", offsets[0]), ("VOFF", offsets[1]), ("QOFF", offsets[2]), ("ROFF", offsets[3])],
        grid=(((nvec + sg_per_tg - 1) // sg_per_tg) * 32 * sg_per_tg, 1, 1),
        threadgroup=(32 * sg_per_tg, 1, 1),
        output_shapes=[(1, n_heads, 1, head_dim), (1, n_kv, 1, head_dim), (1, n_kv, 1, head_dim), (1, 1, n_heads, d_rel), (1, 4, C), (1, 4, C)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.float32, mx.float32],
    )
    return q, k, v, r, ck, cv


# ---------------------------------------------------------------------------
# Relative-position mask for one query over a wrapped ring buffer, in storage
# order (no temporal re-ordering copy). Physical slot s holds chronological
# index (s - idx) mod S; distance = S - 1 - chronological index.
# ---------------------------------------------------------------------------

_RING_MASK = r"""
    const int H = rel_shape[2];
    const int S = kshape_shape[2];
    uint s = thread_position_in_grid.x;
    uint h = thread_position_in_grid.y;
    if ((int)s >= S || (int)h >= H) return;
    int idx = ring_idx[0];
    int ci = ((int)s - idx + S) % S;
    int dist = S - 1 - ci;
    T val;
    if (SLIDING > 0 && dist >= (int)SLIDING) {
        val = (T)(-1e30f);
    } else if (dist < (int)REL_EXTENT) {
        float acc = 0.0f;
        size_t rbase = (size_t)h * rel_strides[2];
        for (uint d = 0; d < D_REL; ++d)
            acc += (float)rel[rbase + (size_t)d * rel_strides[3]] * (float)proj[(size_t)d * proj_strides[0] + (size_t)dist * proj_strides[1]];
        val = (T)acc;
    } else {
        val = (T)0;
    }
    out[(size_t)h * S + s] = val;
"""


@cache
def _ring_mask_kernel():
    return mx.fast.metal_kernel(
        name="inkling_fast_ring_mask",
        input_names=["rel", "proj", "kshape", "ring_idx"],
        output_names=["out"],
        source=_RING_MASK,
        ensure_row_contiguous=False,
    )


def ring_mask(rel, proj, keys, idx, sliding, rel_extent):
    """rel: [1, 1, H, d_rel]; keys: ring buffer [1, KV, S, hd] in storage order -> mask [1, H, 1, S]."""
    _, _, H, d_rel = rel.shape
    S = keys.shape[2]
    (out,) = _ring_mask_kernel()(
        inputs=[rel, proj, keys, mx.array([idx], mx.int32)],
        template=[("T", rel.dtype), ("D_REL", d_rel), ("REL_EXTENT", rel_extent), ("SLIDING", sliding)],
        grid=(((S + 255) // 256) * 256, H, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(1, H, 1, S)],
        output_dtypes=[rel.dtype],
    )
    return out


# ---------------------------------------------------------------------------
# Router epilogue for one token: raw scores (from a multi-core matmul) ->
# sigmoid, + bias, parallel-rank top-K, normalized route weights. 1 threadgroup.
# ---------------------------------------------------------------------------

_TOPK = r"""
    uint tid = thread_position_in_threadgroup.x;
    threadgroup float score[E];
    threadgroup float key[E];
    threadgroup float picked[K];
    threadgroup uint pidx[K];
    if (tid < E) {
        float sc = 1.0f / (1.0f + metal::precise::exp(-float(raw[tid])));
        score[tid] = sc;
        key[tid] = sc + float(bias[tid]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < E) {
        float kv = key[tid];
        uint rank = 0;
        for (uint j = 0; j < E; ++j) {
            float kj = key[j];
            rank += (kj > kv || (kj == kv && j < tid)) ? 1u : 0u;
        }
        if (rank < K) { pidx[rank] = tid; picked[rank] = score[tid]; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < K) {
        float total = 0;
        for (uint k = 0; k < K; ++k) total += picked[k];
        idx[tid] = pidx[tid];
        rw[tid] = picked[tid] / (total + 1e-20f) * SCALE;
    }
"""


@cache
def _topk_kernel(scale):
    return mx.fast.metal_kernel(
        name=f"inkling_fast_topk_{abs(hash(scale))}",
        input_names=["raw", "bias"], output_names=["idx", "rw"],
        source=_TOPK.replace("SCALE", repr(float(scale)) + "f"),
    )


def route_epilogue(raw, bias, top_k, route_scale):
    """raw: [E] f32 router logits -> (idx [K] uint32, weights [K] f32)."""
    E = raw.shape[-1]
    idx, rw = _topk_kernel(float(route_scale))(
        inputs=[raw, bias], template=[("E", E), ("K", top_k)],
        grid=(max(E, 32), 1, 1), threadgroup=(max(E, 32), 1, 1),
        output_shapes=[(top_k,), (top_k,)], output_dtypes=[mx.uint32, mx.float32],
    )
    return idx, rw
