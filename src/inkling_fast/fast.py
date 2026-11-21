"""Lossless decode-speed patches for Inkling 2 on MLX.

Decode on a base M4 is dispatch-bound, not bandwidth-bound: 48 layer passes
per token, each launching dozens of small kernels. These patches cut kernel
count without changing any dot product:

* Attention: q/k/v/r/gate projections share one input in most layer passes.
  Quantized rows are independent, so concatenating the packed rows (same bits
  and group size) gives one matmul whose rows are bit-identical to the five.
* MoE: gate and up experts are concatenated per expert, so a single
  gather_qmm replaces two.
"""

import os

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.inkling2 import language as L
from mlx_vlm.models.switch_layers import _gather_sort, _scatter_unsort

from .kernels import route_epilogue
from .moe_v4 import moe_v4

_ATTN_ORDER = ("k_proj", "v_proj", "q_proj", "r_proj", "gate_proj")


def _concat_quantized(layers, axis):
    first = layers[0]
    for layer in layers:
        if (
            not isinstance(layer, (nn.QuantizedLinear,))
            and type(layer).__name__ != "QuantizedSwitchLinear"
        ) or (layer.bits, layer.group_size, layer.mode) != (
            first.bits,
            first.group_size,
            first.mode,
        ):
            return None
    parts = {}
    for name in ("weight", "scales", "biases"):
        values = [layer.get(name) for layer in layers]
        if any(v is None for v in values):
            if all(v is None for v in values):
                continue
            return None
        parts[name] = mx.contiguous(mx.concatenate(values, axis=axis))
    # Materialize now: a lazy concat captured by mx.compile is recomputed
    # (every expert matrix copied) on every call.
    mx.eval(list(parts.values()))
    return parts, first.bits, first.group_size, first.mode


class _Packed:
    """A row-sliceable bundle of packed quantized weights."""

    def __init__(self, parts, bits, group_size, mode, splits):
        self.weight = parts["weight"]
        self.scales = parts["scales"]
        self.biases = parts.get("biases")
        self.bits, self.group_size, self.mode = bits, group_size, mode
        self.splits = splits  # row offsets

    def matmul(self, x, start=None, stop=None):
        w, s, b = self.weight, self.scales, self.biases
        if start is not None:
            w, s = w[start:stop], s[start:stop]
            b = b[start:stop] if b is not None else None
        return mx.quantized_matmul(
            x, w, s, b, transpose=True, group_size=self.group_size,
            bits=self.bits, mode=self.mode,
        )


def fuse_attention(attn):
    if attn.sharding_group is not None or "_fast_packed" in attn:
        return False
    layers = [attn[name] for name in _ATTN_ORDER]
    got = _concat_quantized(layers, axis=0)
    if got is None:
        return False
    parts, bits, gs, mode = got
    sizes = [layer.weight.shape[0] for layer in layers]
    offsets = [0]
    for n in sizes:
        offsets.append(offsets[-1] + n)
    # Store outside the module tree so it is not treated as a parameter.
    object.__setattr__(attn, "_fast_packed", _Packed(parts, bits, gs, mode, offsets))
    # Free the originals: the packed copy holds the same rows.
    for name in _ATTN_ORDER:
        attn[name] = _RowView(attn._fast_packed, _ATTN_ORDER.index(name))
    return True


class _RowView(nn.Module):
    """Stands in for a fused projection (used by code paths we did not patch)."""

    def __init__(self, packed, index):
        super().__init__()
        object.__setattr__(self, "_p", packed)
        object.__setattr__(self, "_i", index)

    def __call__(self, x):
        o = self._p.splits
        return self._p.matmul(x, o[self._i], o[self._i + 1])


def _attention_call(self, x, key_input=None, value_input=None, cache=None, conv_mask=None):
    packed = self.__dict__.get("_fast_packed")
    if packed is None:
        return _orig_attention_call(self, x, key_input, value_input, cache, conv_mask)
    b, length, _ = x.shape
    cfg = self.config
    kv, conv = cache if cache is not None else (None, None)
    o = packed.splits
    shared = (key_input is None or key_input is x) and (value_input is None or value_input is x)
    key_input = x if key_input is None else key_input
    value_input = x if value_input is None else value_input
    if shared:
        fused = packed.matmul(x)
        k, v, q, r, g = (fused[..., o[i] : o[i + 1]] for i in range(5))
    else:
        k = packed.matmul(key_input, o[0], o[1])
        v = packed.matmul(value_input, o[1], o[2])
        rest = packed.matmul(x, o[2], o[5])
        q, r, g = (rest[..., o[i] - o[2] : o[i + 1] - o[2]] for i in range(2, 5))
    if self.is_sliding:
        k = self.k_sconv(k, conv, conv_mask)
        v = self.v_sconv(v, conv, conv_mask)
    else:
        k = self.k_sconv(k, key_input, conv, conv_mask)
        v = self.v_sconv(v, value_input, conv, conv_mask)
    q = L.l2_clip(q.reshape(b, length, self.n_heads, self.head_dim), cfg.qk_l2_clip_threshold, self.q_scale)
    k = L.l2_clip(k.reshape(b, length, self.n_kv, self.head_dim), cfg.qk_l2_clip_threshold)
    v = v.reshape(b, length, self.n_kv, self.head_dim)
    if cfg.use_value_norm_no_gamma:
        v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(x.dtype)
    r = L.l2_clip(r.reshape(b, length, self.n_heads, cfg.d_rel), cfg.rel_l2_clip_threshold, self.rel_scale)
    if not self.is_sliding and cfg.log_scaling_n_floor is not None:
        offset = kv.offset if kv is not None else 0
        positions = mx.arange(length) + mx.array(offset)[..., None] + 1
        tau = 1 + cfg.log_scaling_alpha * mx.log(mx.maximum(positions / cfg.log_scaling_n_floor, 1))
        q = (q.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
        r = (r.astype(mx.float32) * tau[..., None, None]).astype(x.dtype)
    q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
    if kv is not None:
        k, v = kv.update_and_fetch(k, v)
        if isinstance(kv, L.RotatingKVCache):
            k, v = kv._temporal_order(k), kv._temporal_order(v)
        elif isinstance(kv, L.BatchRotatingKVCache) and kv.rotated:
            k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
    size = k.shape[2]
    padding = getattr(kv, "left_padding", None)
    mask = L.banded_additive_mask(
        r, self.rel_proj.astype(x.dtype), size - length, size, self.sliding,
        self.rel_extent, left_padding=padding, shape_ref=k,
    )
    sinks = self.sink_proj(x).transpose(0, 2, 1)[..., None]
    mask = mx.concatenate([mask, sinks], axis=-1)
    k = mx.concatenate([k, mx.zeros((b, self.n_kv, 1, self.head_dim), k.dtype)], axis=2)
    v = mx.concatenate([v, mx.zeros((b, self.n_kv, 1, self.head_dim), v.dtype)], axis=2)
    out = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.head_dim**-0.5, mask=mask)
    out = out.transpose(0, 2, 1, 3).reshape(b, length, -1)
    out = (out.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(x.dtype)
    return self.o_proj(out)


_orig_attention_call = L.Attention.__call__


def fuse_moe(moe):
    glu = moe.switch_mlp
    if "_fast_gate_up" in moe.__dict__ or moe.sharding_group is not None:
        return False
    got = _concat_quantized([glu.up_proj, glu.gate_proj], axis=1)
    if got is None:
        return False
    parts, bits, gs, mode = got
    width = glu.up_proj.weight.shape[1]
    object.__setattr__(
        moe, "_fast_gate_up",
        (parts["weight"], parts["scales"], parts.get("biases"), bits, gs, mode, width),
    )
    # The fused copy serves every path; drop the originals (~6 GB at 3-bit).
    for proj in (glu.up_proj, glu.gate_proj):
        proj.weight = mx.zeros((1,), mx.uint32)
        proj.scales = mx.zeros((1,), mx.bfloat16)
        if proj.get("biases") is not None:
            proj.biases = mx.zeros((1,), mx.bfloat16)
    return True


def _moe_call(self, x):
    fused = self.__dict__.get("_fast_gate_up")
    if fused is None or self.training:
        return _orig_moe_call(self, x)
    w, s, b, bits, gs, mode, width = fused
    if x.shape[0] == 1 and x.shape[1] == 1:
        raw = (x.astype(mx.float32) @ self.gate_weight.astype(mx.float32).T).reshape(-1)
        idx, rw = route_epilogue(raw, self.e_score_correction_bias, self.top_k, self.route_scale)
        indices, weights = idx.reshape(1, 1, -1), rw.reshape(1, 1, -1)
    else:
        indices, weights = L._route(
            x, self.gate_weight, self.e_score_correction_bias, self.top_k, self.route_scale
        )
    glu = self.switch_mlp
    dp = glu.down_proj
    if (
        x.shape[0] == 1 and x.shape[1] == 1 and bits == 3 and mode == "affine"
        and gs == 64 and s.dtype == mx.bfloat16 and dp.bits == 3 and dp.mode == "affine"
        and dp.scales.dtype == mx.bfloat16 and x.dtype == mx.bfloat16
    ):
        act = glu.activation
        y = moe_v4(
            x.reshape(-1), (w, s, b), (dp.weight, dp.scales, dp.biases),
            indices.reshape(-1).astype(mx.uint32), act.gate_cap, act.up_cap,
        )
        out = (y.astype(mx.float32) * weights.reshape(-1, 1)).sum(axis=0).astype(x.dtype)
        return out.reshape(x.shape)
    xe = mx.expand_dims(x, (-2, -3))
    do_sort = indices.size >= 64
    idx, inv = indices, None
    if do_sort:
        xe, idx, inv = _gather_sort(xe, indices)
    gu = mx.gather_qmm(
        xe, w, s, b, rhs_indices=idx, transpose=True,
        group_size=gs, bits=bits, mode=mode, sorted_indices=do_sort,
    )
    up, gate = gu[..., :width], gu[..., width:]
    out = glu.down_proj(glu.activation(up, gate), idx, sorted_indices=do_sort)
    if do_sort:
        out = _scatter_unsort(out, inv, indices.shape)
    return L._combine_routes(out.squeeze(-2), weights)


_orig_moe_call = L.SparseMoE.__call__


BF16_SCALES = os.environ.get("INKLING_BF16_SCALES", "1") == "1"


def _scales_to_bf16(lm):
    """Store float32 per-group scales/biases as bf16 (activations are bf16, so the
    fast kernel path needs matching dtypes). The 3/4-bit weight codes are untouched."""
    n = 0

    def conv(obj, names=("scales", "biases")):
        nonlocal n
        for name in names:
            v = obj.get(name) if isinstance(obj, nn.Module) else getattr(obj, name, None)
            if v is not None and v.dtype == mx.float32:
                nv = v.astype(mx.bfloat16)
                mx.eval(nv)
                if isinstance(obj, nn.Module):
                    obj[name] = nv
                else:
                    setattr(obj, name, nv)
                n += 1

    for layer in lm.model.layers:
        moe = layer.mlp
        fused = moe.__dict__.get("_fast_gate_up")
        if fused is not None and fused[1].dtype == mx.float32:
            w, s, b = fused[:3]
            s2, b2 = s.astype(mx.bfloat16), None if b is None else b.astype(mx.bfloat16)
            mx.eval(s2, b2)  # materialize: a lazy astype captured by mx.compile re-runs every call
            object.__setattr__(moe, "_fast_gate_up", (w, s2, b2, *fused[3:]))
            n += 1
        conv(moe.switch_mlp.down_proj)
        conv(layer.self_attn.o_proj)
    # lm_head runs on float32 inputs (exact logits) and the embedding table sets
    # the activation dtype, so both keep their stored scales.
    mx.eval(lm.parameters())
    return n


def apply(model, attention=True, moe=True):
    """Patch a loaded Inkling 2 / Inkling 2 Mini model in place."""
    lm = getattr(model, "language_model", model)
    stats = {"attention": 0, "moe": 0}
    for layer in lm.model.layers:
        if attention and fuse_attention(layer.self_attn):
            stats["attention"] += 1
        if moe and fuse_moe(layer.mlp):
            stats["moe"] += 1
    if BF16_SCALES:
        stats["bf16_scales"] = _scales_to_bf16(lm)
    L.Attention.__call__ = _attention_call
    L.SparseMoE.__call__ = _moe_call
    mx.eval(lm.parameters())
    mx.clear_cache()
    return stats
