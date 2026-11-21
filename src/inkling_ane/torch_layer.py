"""PyTorch restatement of the DENSE part of one Inkling 2 Mini layer application, shaped
for Core ML / the Apple Neural Engine.

One module = one layer application (layer i, loop pass p) for a fixed Q rows:
  x (layer input) -> attention-input mixing, (message gates), norms, q/k/v/r/gate/sink
  projections, k/v short convs, L2 clips, v-norm, banded relative-position attention with
  query-dependent sinks, output gate, o_proj, attn short conv, residual, mlp-input mixing,
  (mlp message gate), post-attention norm  ->  (h = MoE input, residual).
The GPU then computes out = residual + mlp_sconv(MoE(h)).

ANE shaping:
* K/V caches are SHIFTING windows held in MLState (one state per tensor): slot j of the
  S-slot cache holds position p0 - S + j, the Q new rows follow. Distances are then the same
  for every chunk, so the learned relative-position bias needs no gather: a constant
  G[i, d, j] = rel_proj[d, i + 1024 - j] (zero outside 0..1023) and bias = r @ G.
* Positions live on the host: it passes one additive mask per chunk (0 / -30000) for the
  causal window, the 1024 sliding band and not-yet-filled slots.
* Norms divide by the row max first, so fp16 sums of squares cannot overflow or go subnormal.
* Sigmoid is 0.5 + 0.5 tanh(x / 2).
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

D = 2304
H, KV, HD, DREL = 24, 8, 128, 16
G = H // KV
EXTENT = 1024
NEG = -30000.0
import os  # noqa: E402

DEBUG_STOP = os.environ.get("ANE_DEBUG_STOP", "")
VARIANT = set(os.environ.get("ANE_VARIANT", "").split(","))
FUSE_OFF = False


def safe_rms(x, eps):
    """x / sqrt(mean(x^2) + eps) over the last axis, fp16-safe."""
    if "plain_rms" in VARIANT:
        xs = x * (1 / 16)
        return xs * torch.rsqrt((xs * xs).mean(dim=-1, keepdim=True) + eps / 256)
    m = x.abs().amax(dim=-1, keepdim=True) + 1e-4
    xs = x / m
    ms = (xs * xs).mean(dim=-1, keepdim=True)
    return xs * torch.rsqrt(ms + eps / (m * m))


def safe_l2_clip(x, threshold):
    if "plain_l2" in VARIANT:
        xs = x * (1 / 16)
        norm = torch.sqrt((xs * xs).sum(dim=-1, keepdim=True)) * 16
        return x * torch.clamp(threshold / (norm + 1e-6), max=1.0)
    m = x.abs().amax(dim=-1, keepdim=True) + 1e-4
    xs = x / m
    norm = torch.sqrt((xs * xs).sum(dim=-1, keepdim=True)) * m
    return x * torch.clamp(threshold / (norm + 1e-6), max=1.0)


def sigmoid(x):
    if "sig_native" in VARIANT:
        return torch.sigmoid(x)
    return 0.5 + 0.5 * torch.tanh(0.5 * x)


def attn_res(states, query):
    """states [1, n, Q, D], query [m, D] -> [1, m, Q, D] (normalize_values=True)."""
    keys = safe_rms(states, 1e-6)
    scores = torch.matmul(keys, query.t()) * (D ** -0.5)  # [1, n, Q, m]
    w = torch.softmax(scores, dim=1)
    n, m = states.shape[1], query.shape[0]
    outs = []
    for j in range(m):
        acc = keys[:, 0] * w[:, 0, :, j:j + 1]
        for s in range(1, n):
            acc = acc + keys[:, s] * w[:, s, :, j:j + 1]
        outs.append(acc)
    return outs  # list of m tensors [1, Q, D]


class RMSNorm(nn.Module):
    def __init__(self, w, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(w), requires_grad=False)
        self.eps = eps

    def forward(self, x):
        return safe_rms(x, self.eps) * self.weight


class MessageGate(nn.Module):
    def __init__(self, w, eps):
        super().__init__()
        self.down = nn.Linear(2 * D, w["down_proj.weight"].shape[0], bias=False)
        self.down.weight.data = torch.tensor(w["down_proj.weight"])
        self.fg = nn.Linear(self.down.out_features, D, bias=False)
        self.fg.weight.data = torch.tensor(w["forget_gate_proj.weight"])
        self.mg = nn.Linear(self.down.out_features, D, bias=False)
        self.mg.weight.data = torch.tensor(w["message_gate_proj.weight"])
        self.eps = eps

    def forward(self, hidden, msg):
        gi = self.down(torch.cat([safe_rms(hidden, self.eps), safe_rms(msg, self.eps)], dim=-1))
        return (1 - torch.tanh(self.fg(gi))) * hidden + torch.tanh(self.mg(gi)) * msg


class SeparateLinear(nn.Module):
    """Same interface as FusedLinear, one matmul per projection (Q=1: the fused split fails ANECCompile)."""

    def __init__(self, weights):
        super().__init__()
        self.lins = nn.ModuleList()
        for w in weights:
            lin = nn.Linear(D, w.shape[0], bias=False)
            lin.weight.data = torch.tensor(w)
            self.lins.append(lin)

    def forward(self, x):
        return tuple(lin(x) for lin in self.lins)


class FusedLinear(nn.Module):
    """Several projections of the same input as one matmul."""

    def __new__(cls, weights):
        if FUSE_OFF:
            return SeparateLinear(weights)
        return super().__new__(cls)

    def __init__(self, weights):
        super().__init__()
        self.sizes = [w.shape[0] for w in weights]
        self.lin = nn.Linear(D, sum(self.sizes), bias=False)
        self.lin.weight.data = torch.tensor(np.concatenate(weights, 0))

    def forward(self, x):
        return torch.split(self.lin(x), self.sizes, dim=-1)


def short_conv(x, state, taps, q):
    """Causal depthwise K=4 conv + residual, /sqrt(5). x [1,Q,C], state [1,3,C], taps [4,C]."""
    ctx = torch.cat([state, x], dim=1)
    if q == 1:  # one row: window x taps, summed (recipe: slices of the state window fail to compile)
        out = (ctx * taps).sum(dim=1, keepdim=True)
        return (out + x) * (1 / math.sqrt(5)), (ctx[:, :3] if "conv_off0" in VARIANT else ctx[:, 1:])
    if "slice_conv" not in VARIANT:
        c = taps.shape[1]
        y = F.conv2d(ctx.permute(0, 2, 1).unsqueeze(2), taps.t().reshape(c, 1, 1, 4), groups=c)
        out = y.squeeze(2).permute(0, 2, 1)
        return (out + x) * (1 / math.sqrt(5)), ctx[:, q:]
    out = ctx[:, 0:q] * taps[0] + ctx[:, 1:q + 1] * taps[1] + ctx[:, 2:q + 2] * taps[2] + ctx[:, 3:q + 3] * taps[3]
    return (out + x) * (1 / math.sqrt(5)), ctx[:, q:]


SHIFT = None


def _shift_kernel(c):
    w = torch.zeros(4 * c, 1, 1, 4)
    for lag in range(4):
        w[lag::4, 0, 0, 3 - lag] = 1.0
    return w


def dynamic_conv(x, state, static, dyn, q):
    """Inkling DynamicShortConvolution. static [4, C], dyn [1, Q, KV*4] (per head, per lag).
    out[t, c] = sum_lag ctx[t + 3 - lag, c] * (static[lag, c] + dyn[t, head(c), lag]).
    The 4 shifted copies come from a one-hot depthwise conv (state-window slices fail ANECCompile)."""
    c = KV * HD
    ctx = torch.cat([state, x], dim=1)
    if q == 1:  # window row k holds lag 3-k
        w = static.flip(0).reshape(4, KV, HD) + dyn.reshape(KV, 4).t().flip(0).reshape(4, KV, 1)
        out = (ctx.reshape(4, KV, HD) * w).sum(dim=0).reshape(1, 1, c)
        return (out + x) * (1 / math.sqrt(9)), ctx[:, 1:]
    shifts = F.conv2d(ctx.permute(0, 2, 1).unsqueeze(2), _shift_kernel(c), groups=c)  # [1, 4C, 1, Q]
    shifts = shifts.reshape(KV, HD, 4, q)
    w = static.t().reshape(KV, HD, 4, 1) + dyn.reshape(q, KV, 4).permute(1, 2, 0).reshape(KV, 1, 4, q)
    out = (shifts * w).sum(dim=2).reshape(1, c, q).permute(0, 2, 1)
    return (out + x) * (1 / math.sqrt(9)), ctx[:, q:]


def rel_matrix(rel_proj, q):
    """G [Q, DREL, EXTENT + Q]: G[i, d, j] = rel_proj[d, i + EXTENT - j] for 0 <= dist < EXTENT."""
    J = EXTENT + q
    i = np.arange(q)[:, None]
    j = np.arange(J)[None, :]
    dist = i + EXTENT - j
    valid = (dist >= 0) & (dist < EXTENT)
    g = rel_proj[:, np.clip(dist, 0, EXTENT - 1)]  # [DREL, Q, J]
    g = np.where(valid[None], g, 0.0)
    return np.ascontiguousarray(g.transpose(1, 0, 2)).astype(np.float32)


class DenseLayer(nn.Module):
    def __init__(self, cfg, layer_idx, loop_pass, q, w, msg_query=None, s_global=2048, wscale=1.0):
        super().__init__()
        global FUSE_OFF
        FUSE_OFF = "nofuse" in VARIANT
        self.i, self.p, self.q = layer_idx, loop_pass, q
        self.sliding = layer_idx in cfg["local_layer_ids"]
        self.boundary = layer_idx % cfg["n_layers_per_attn_res_block"] == 0
        self.eps = cfg["rms_norm_eps"]
        self.messages = loop_pass > 0
        self.S = EXTENT if self.sliding else s_global
        def t(a):
            return nn.Parameter(torch.tensor(a), requires_grad=False)
        def a(k):
            return w[f"self_attn.{k}"]
        self.attn_query = t(w["attention_input.query"].reshape(-1, D)) if layer_idx else None
        self.mlp_query = t(w["mlp_input.query"].reshape(-1, D))
        if self.sliding:
            self.norms = nn.ModuleList([RMSNorm(w["input_layernorm.weight"], self.eps)] * 3)
        else:
            self.norms = nn.ModuleList([RMSNorm(w[f"{n}_input_layernorm.weight"], self.eps) for n in "qkv"])
        self.post_norm = RMSNorm(w["post_attention_layernorm.weight"], self.eps)
        if self.messages:
            self.msg_query = t(msg_query)  # [4, D]
            self.gates = nn.ModuleList([MessageGate({k[len(n) + 1:]: v for k, v in w.items() if k.startswith(n + ".")}, self.eps)
                                        for n in ("q_message_gate", "k_message_gate", "v_message_gate", "mlp_message_gate")])
        kq = [a("q_proj.weight"), a("r_proj.weight"), a("gate_proj.weight"), a("sink_proj.weight")]
        kk = [a("k_proj.weight")] + ([] if self.sliding else [a("k_sconv.weight_proj.weight")])
        kv = [a("v_proj.weight")] + ([] if self.sliding else [a("v_sconv.weight_proj.weight")])
        self.shared_qkv = not self.messages and (self.sliding or layer_idx == 0)
        # Weights stored x wscale and outputs divided back (ANE small-product precision).
        self.wscale = wscale
        if self.shared_qkv:
            self.proj = FusedLinear([m * wscale for m in kq + kk + kv])
        else:
            self.proj_q = FusedLinear([m * wscale for m in kq])
            self.proj_k = FusedLinear([m * wscale for m in kk])
            self.proj_v = FusedLinear([m * wscale for m in kv])
        self.o_proj = nn.Linear(H * HD, D, bias=False)
        self.o_proj.weight.data = torch.tensor(a("o_proj.weight") * wscale)
        self.q_scale = t(a("q_scale").reshape(1, 1, H, HD))
        self.rel_scale = t(a("rel_scale").reshape(1, 1, H, DREL))
        self.G = t(rel_matrix(a("rel_proj"), q))
        if self.sliding:
            self.k_taps = t(a("k_sconv.conv.weight")[:, :, 0].T.copy())  # [4, C]
            self.v_taps = t(a("v_sconv.conv.weight")[:, :, 0].T.copy())
        else:
            self.k_static = t(a("k_sconv.static_weight"))  # [4, C]
            self.v_static = t(a("v_sconv.static_weight"))
        self.a_taps = t(w["attn_sconv.conv.weight"][:, :, 0].T.copy())
        S = self.S
        self.register_buffer("k_cache", torch.zeros(1, KV, S, HD, dtype=torch.float16))
        self.register_buffer("v_cache", torch.zeros(1, KV, S, HD, dtype=torch.float16))
        self.register_buffer("k_conv", torch.zeros(1, 3, KV * HD, dtype=torch.float16))
        self.register_buffer("v_conv", torch.zeros(1, 3, KV * HD, dtype=torch.float16))
        self.register_buffer("a_conv", torch.zeros(1, 3, D, dtype=torch.float16))

    def _touch(self, *names):
        for n in names:
            b = getattr(self, n)
            b[:] = (b * 1.0).to(torch.float16)

    def state_names(self):
        return ["k_cache", "v_cache", "k_conv", "v_conv", "a_conv"]

    def forward(self, x, mask, blocks=None, msg_states=None):
        """x [1,Q,D]; mask [1,1,G*Q,S+Q]; blocks [1,nb,Q,D] (nb>=1 unless layer 0);
        msg_states [1,2,Q,D] (loop passes >= 1). Returns h [1,Q,D], residual [1,Q,D]."""
        q = self.q
        if self.attn_query is None:
            mixed = [x]
        else:
            mixed = attn_res(torch.cat([blocks, x[:, None]], dim=1), self.attn_query)
        parts = [mixed[min(j, len(mixed) - 1)] for j in range(3)]
        if self.messages:
            msgs = attn_res(msg_states, self.msg_query)
            parts = [self.gates[j](parts[j], msgs[j]) for j in range(3)]
        xq, xk, xv = [self.norms[j](parts[j]) for j in range(3)]
        if DEBUG_STOP == "norm":
            self._touch(*self.state_names())
            return xq, xk
        if self.shared_qkv:
            outs = self.proj(xq)
            pq, pr, pg, ps = outs[:4]
            rest = outs[4:]
            pk, dk = rest[0], (rest[1] if not self.sliding else None)
            pv, dv = rest[-2 if not self.sliding else 1], (rest[-1] if not self.sliding else None)
        else:
            pq, pr, pg, ps = self.proj_q(xq)
            ko, vo = self.proj_k(xk), self.proj_v(xv)
            pk, dk = ko[0], (ko[1] if len(ko) > 1 else None)
            pv, dv = vo[0], (vo[1] if len(vo) > 1 else None)
        inv = 1.0 / self.wscale
        if self.wscale != 1.0:
            pq, pr, pg, ps, pk, pv = (t * inv for t in (pq, pr, pg, ps, pk, pv))
            dk = dk * inv if dk is not None else None
            dv = dv * inv if dv is not None else None
        if self.sliding:
            k, kc = short_conv(pk, self.k_conv, self.k_taps, q)
            v, vc = short_conv(pv, self.v_conv, self.v_taps, q)
        else:
            k, kc = dynamic_conv(pk, self.k_conv, self.k_static, dk, q)
            v, vc = dynamic_conv(pv, self.v_conv, self.v_static, dv, q)
        self.k_conv[:] = kc.to(torch.float16)
        self.v_conv[:] = vc.to(torch.float16)
        if DEBUG_STOP == "conv":
            self._touch("k_cache", "v_cache", "a_conv")
            return k, pq
        qh = safe_l2_clip(pq.reshape(1, q, H, HD) * self.q_scale, 22.627416997969522)
        kh = safe_l2_clip(k.reshape(1, q, KV, HD), 22.627416997969522)
        vh = safe_rms(v.reshape(1, q, KV, HD), self.eps)
        rh = safe_l2_clip(pr.reshape(1, q, H, DREL) * self.rel_scale, 8.0)
        qh = qh.permute(0, 2, 1, 3).reshape(1, KV, G * q, HD)
        kh = kh.permute(0, 2, 1, 3)
        vh = vh.permute(0, 2, 1, 3)
        keys = torch.cat([self.k_cache, kh], dim=2)
        vals = torch.cat([self.v_cache, vh], dim=2)
        off = 0 if "kv_off0" in VARIANT else q
        self.k_cache[:] = keys[:, :, off:off + self.S].to(torch.float16)
        self.v_cache[:] = vals[:, :, off:off + self.S].to(torch.float16)
        scores = torch.matmul(qh, keys.transpose(2, 3)) * (HD ** -0.5)  # [1, KV, G*Q, S+Q]
        # Relative-position bias over the last EXTENT + Q slots: [Q, H, DREL] @ [Q, DREL, J].
        if "bias_ew" in VARIANT:
            rr = rh.reshape(q, H, DREL).permute(1, 0, 2)  # [H, Q, DREL]
            bias = rr[:, :, 0:1] * self.G[:, 0]
            for d in range(1, DREL):
                bias = bias + rr[:, :, d:d + 1] * self.G[:, d]
            bias = bias.reshape(1, KV, G * q, EXTENT + q)
        elif "no_bias" in VARIANT:
            bias = 0
        else:
            bias = torch.matmul(rh.reshape(q, H, DREL), self.G)  # [Q, H, J]
            bias = bias.permute(1, 0, 2).reshape(1, KV, G * q, EXTENT + q)
        sink = ps.permute(0, 2, 1).reshape(1, KV, G * q, 1)
        if self.S > EXTENT and "pad_global" not in VARIANT:
            # Far slots (no relative bias) and near slots (bias) as two score blocks: no pad op.
            far = self.S - EXTENT
            s_far = torch.matmul(qh, keys[:, :, :far].transpose(2, 3)) * (HD ** -0.5) + mask[..., :far]
            s_near = torch.matmul(qh, keys[:, :, far:].transpose(2, 3)) * (HD ** -0.5) + bias + mask[..., far:]
            probs = torch.softmax(torch.cat([s_far, s_near, sink], dim=-1), dim=-1)
            att = torch.matmul(probs[..., :far], vals[:, :, :far]) + torch.matmul(probs[..., far:-1], vals[:, :, far:])
        else:
            if self.S > EXTENT:
                bias = F.pad(bias, (self.S - EXTENT, 0))
            logits = torch.cat([scores + bias + mask, sink], dim=-1)
            probs = torch.softmax(logits, dim=-1)[..., :-1]
            att = torch.matmul(probs, vals)  # [1, KV, G*Q, HD]
        if q == 1 or "gate_heads" in VARIANT:
            # gate in head layout, before the transpose back (Q=1: gating after it fails ANECCompile)
            g = sigmoid(pg).reshape(1, q, H, HD).permute(0, 2, 1, 3).reshape(1, KV, G * q, HD)
            att = (att * g).reshape(1, H, q, HD).permute(0, 2, 1, 3).reshape(1, q, H * HD)
            pg = None
        else:
            att = att.reshape(1, H, q, HD).permute(0, 2, 1, 3).reshape(1, q, H * HD)
        if DEBUG_STOP == "attn":
            self._touch("a_conv")
            return att, att
        if "nogate" not in VARIANT and pg is not None:
            att = att * sigmoid(pg)
        o = self.o_proj(att) * inv
        if DEBUG_STOP == "gate":
            self._touch("a_conv")
            return att, att
        if DEBUG_STOP == "oproj":
            self._touch("a_conv")
            return o, o
        residual, ac = short_conv(o, self.a_conv, self.a_taps, q)
        self.a_conv[:] = ac.to(torch.float16)
        if DEBUG_STOP == "aconv":
            return residual, residual
        if self.boundary:
            states = torch.cat([blocks, x[:, None], residual[:, None]], dim=1) if blocks is not None else torch.cat([x[:, None], residual[:, None]], dim=1)
        else:
            residual = residual + x
            states = torch.cat([blocks, residual[:, None]], dim=1)
        hidden = attn_res(states, self.mlp_query)[0]
        if self.messages:
            hidden = self.gates[3](hidden, msgs[3])
        return self.post_norm(hidden), residual


def host_mask(p0, q, S, sliding):
    """Additive mask [1, 1, G*Q, S+Q] for a chunk whose first row is at position p0."""
    i = np.arange(q)[:, None]
    j = np.arange(S + q)[None, :]
    pos = p0 - S + j
    dist = p0 + i - pos
    ok = (pos >= 0) & (dist >= 0)
    if sliding:
        ok &= dist < EXTENT
    m = np.where(ok, 0.0, NEG).astype(np.float32)
    return np.tile(m, (G, 1))[None, None]
