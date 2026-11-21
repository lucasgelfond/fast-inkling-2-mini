"""Compiled short-sequence forward for Inkling 2 Mini (inkling2_loop) on MLX.

Decode on a base M4 is latency-bound: one token is a serial chain of 48 layer
passes, each built from many tiny kernels. The reference implementation emits
~12k graph nodes per token.

Each layer pass here is two mx.compile'd pure functions with the conv state
passed explicitly, so MLX fuses the elementwise chains; message gates run as
custom Metal kernels (kernels.py):

  pre(x, blocks, prev, conv_k, conv_v, tau) -> q, k, v, r, gate, sinks, conv ctx
  [eager] conv/KV cache commit (speculation-aware), relative mask, SDPA
  post(attn, gate, x, blocks, conv_a, conv_m, msg) -> out, conv ctx

Handles batch 1 with 1..MAX_TOKENS tokens, including speculative verify with
hidden-state captures (DSpark taps), so drafters run on this path too.
"""

import math
import os  # noqa: E402

import mlx.core as mx
from mlx_vlm.models.base import LanguageModelOutput
from mlx_vlm.models.inkling2 import language as L
from mlx_vlm.models.inkling2.language import (
    BatchRotatingKVCache,
    RotatingKVCache,
    _attention_residual,
    banded_additive_mask,
)

from . import fast
from .fast import fuse_attention, fuse_moe
from .kernels import attention_mix as _kernel_mix
from .kernels import message_gates, ring_mask, sliding_postproj, stack_gates

MAX_TOKENS = int(os.environ.get("INKLING_MAX_TOKENS", "16"))
# Prefill chunk lengths that take the compiled path (each length is traced once).
# Prefill chunks compute vocabulary logits only for the last position (all that
# generation consumes). Set INKLING_PREFILL_LAST_ONLY=0 for full logits (evaluation,
# prompt log-probs).
PREFILL_LAST_ONLY = os.environ.get("INKLING_PREFILL_LAST_ONLY", "1") == "1"
PREFILL_SIZES = {int(v) for v in os.environ.get("INKLING_PREFILL_SIZES", "256,512,1024,2048").split(",") if v}
USE_MIX = os.environ.get("INKLING_MIX", "1") == "1"
USE_POSTPROJ = os.environ.get("INKLING_POSTPROJ", "1") == "1"
USE_RING = os.environ.get("INKLING_RING", "1") == "1"
BF16_HEAD = os.environ.get("INKLING_BF16_HEAD", "1") == "1"


def _static_sconv(state, x, weight, divisor):
    """Depthwise causal conv (K=4) over L tokens in FP32. Returns (out, context)."""
    xf = x.astype(mx.float32)
    w = weight.astype(mx.float32)
    ctx = mx.concatenate([state, xf], axis=1)  # [b, 3 + L, C]
    n = x.shape[1]
    out = ctx[:, 0:n] * w[:, 0] + ctx[:, 1 : n + 1] * w[:, 1] + ctx[:, 2 : n + 2] * w[:, 2] + ctx[:, 3 : n + 3] * w[:, 3]
    out = (out + xf) / divisor
    return out.astype(x.dtype), ctx


def _dynamic_sconv(conv, state, x, dynamic_input):
    b, n, channels = x.shape
    k = conv.kernel_size
    xf = x.astype(mx.float32)
    ctx = mx.concatenate([state, xf], axis=1)  # [b, k - 1 + L, C]
    dynamic = conv.weight_proj(dynamic_input).astype(mx.float32).reshape(b, n, conv.num_heads, 1, k)
    weights = conv.static_weight.T.reshape(conv.num_heads, -1, k) + dynamic
    shape = (b, n, conv.num_heads, channels // conv.num_heads)
    out = mx.zeros(shape, mx.float32)
    for lag in range(k):
        out = out + ctx[:, k - 1 - lag : k - 1 - lag + n].reshape(shape) * weights[..., lag]
    out = (out.reshape(x.shape) + xf) / math.sqrt(2 * k + 1)
    return out.astype(x.dtype), ctx


def _commit_conv(conv, index, ctx, width):
    """Store the next conv state; record the window when a speculation is open."""
    if getattr(conv, "is_speculating", False):
        conv.record_speculative_window(index, ctx, width)
    conv[index] = ctx[:, -width:]


class _Pass:
    """Compiled halves of one layer application (layer index, loop pass)."""

    def __init__(self, model, layer, index, loop_index):
        self.layer = layer
        self.index = index
        self.loop_index = loop_index  # set only for loop passes after the first
        attn = layer.self_attn
        self.cfg = cfg = attn.config
        self.packed = attn.__dict__["_fast_packed"]
        self.message_input = model.message_input
        if loop_index is not None:
            self.qkv_gates = stack_gates([layer.q_message_gate, layer.k_message_gate, layer.v_message_gate])
            self.mlp_gates = stack_gates([layer.mlp_message_gate])
        self.eps = cfg.rms_norm_eps
        self.width = cfg.sconv_kernel_size - 1
        self.sconv_div = math.sqrt(cfg.sconv_kernel_size + 1)
        self.pre = mx.compile(self._pre)
        self.post = mx.compile(self._post)

    # ---- before attention ----
    def _pre(self, x, blocks, prev_blocks, prev_x, conv_k, conv_v, tau):
        layer, attn, cfg, packed = self.layer, self.layer.self_attn, self.cfg, self.packed
        dtype = x.dtype
        b, n, d = x.shape
        nv = self.cfg.attn_res_normalize_values
        msgs = None
        if self.loop_index is not None:
            q_msg = self.message_input.query[self.loop_index]
            msgs = attention_mix([*prev_blocks, prev_x], q_msg, nv)  # [b, n, 4, d] f32
        if layer.is_sliding:
            norms = (layer.input_layernorm,) * 3
        else:
            norms = (layer.q_input_layernorm, layer.k_input_layernorm, layer.v_input_layernorm)
        shared_input = False
        if layer.attention_input is None:
            mixed = x[..., None, :]
            parts = [mixed[..., 0, :]] * 3
            if msgs is None:
                parts = [norm(p).astype(dtype) for norm, p in zip(norms, parts, strict=False)]
                shared_input = layer.is_sliding
        elif msgs is None:
            # Mix and RMSNorm in one kernel.
            uniq = norms[:1] if layer.is_sliding else norms
            nw = mx.stack([nm.weight for nm in uniq])
            mixed = attention_mix([*blocks, x], layer.attention_input.query, nv, nw, self.eps, dtype)
            parts = [mixed[..., 0, :]] * 3 if layer.is_sliding else [mixed[..., i, :] for i in range(3)]
            shared_input = layer.is_sliding
        else:
            mixed = attention_mix([*blocks, x], layer.attention_input.query, nv)
            parts = [mixed[..., 0, :]] * 3 if layer.is_sliding else [mixed[..., i, :] for i in range(3)]
        if msgs is not None:
            # [n, 3, d] -> [3 * n, d]: gate g of token t at row g * n + t.
            h = mx.stack([p.reshape(n, d) for p in parts]).astype(mx.float32).reshape(3 * n, d)
            m = msgs.reshape(n, 4, d)[:, :3].transpose(1, 0, 2).reshape(3 * n, d).astype(mx.float32)
            gated = message_gates(h, m, *self.qkv_gates, self.eps, tokens=n).reshape(3, b, n, d)
            parts = [norm(gated[i]).astype(dtype) for i, norm in enumerate(norms)]
        q_in, k_in, v_in = parts
        o = packed.splits
        if USE_POSTPROJ and attn.is_sliding and n == 1 and b == 1 and dtype == mx.bfloat16:
            if shared_input:
                raw = packed.matmul(q_in).reshape(-1)
            else:
                raw = mx.concatenate([
                    packed.matmul(k_in, o[0], o[1]).reshape(-1),
                    packed.matmul(v_in, o[1], o[2]).reshape(-1),
                    packed.matmul(q_in, o[2], o[5]).reshape(-1),
                ])
            q, k, v, r, ctx_k, ctx_v = sliding_postproj(
                raw, o, attn.q_scale, attn.rel_scale, conv_k, conv_v,
                attn.k_sconv.conv.weight[..., 0], attn.v_sconv.conv.weight[..., 0],
                attn.n_heads, attn.n_kv, attn.head_dim, cfg.d_rel,
                cfg.qk_l2_clip_threshold, cfg.rel_l2_clip_threshold, cfg.rms_norm_eps,
            )
            g = raw[o[4] : o[5]].reshape(b, n, -1)
            sinks = attn.sink_proj(q_in)
            msg_mlp = msgs[..., 3, :] if msgs is not None else None
            return q, k, v, r, g, sinks, ctx_k, ctx_v, msg_mlp
        if shared_input:
            fused = packed.matmul(q_in)
            k, v, q, r, g = (fused[..., o[i] : o[i + 1]] for i in range(5))
        else:
            k = packed.matmul(k_in, o[0], o[1])
            v = packed.matmul(v_in, o[1], o[2])
            rest = packed.matmul(q_in, o[2], o[5])
            q, r, g = (rest[..., o[i] - o[2] : o[i + 1] - o[2]] for i in range(2, 5))
        if attn.is_sliding:
            k, ctx_k = _static_sconv(conv_k, k, attn.k_sconv.conv.weight[..., 0], self.sconv_div)
            v, ctx_v = _static_sconv(conv_v, v, attn.v_sconv.conv.weight[..., 0], self.sconv_div)
        else:
            k, ctx_k = _dynamic_sconv(attn.k_sconv, conv_k, k, k_in)
            v, ctx_v = _dynamic_sconv(attn.v_sconv, conv_v, v, v_in)
        q = L.l2_clip(q.reshape(b, n, attn.n_heads, attn.head_dim), cfg.qk_l2_clip_threshold, attn.q_scale)
        k = L.l2_clip(k.reshape(b, n, attn.n_kv, attn.head_dim), cfg.qk_l2_clip_threshold)
        v = v.reshape(b, n, attn.n_kv, attn.head_dim)
        if cfg.use_value_norm_no_gamma:
            v = mx.fast.rms_norm(v.astype(mx.float32), None, cfg.rms_norm_eps).astype(dtype)
        r = L.l2_clip(r.reshape(b, n, attn.n_heads, cfg.d_rel), cfg.rel_l2_clip_threshold, attn.rel_scale)
        if tau is not None:
            q = (q.astype(mx.float32) * tau[:, None, None]).astype(dtype)
            r = (r.astype(mx.float32) * tau[:, None, None]).astype(dtype)
        sinks = attn.sink_proj(q_in)  # [b, n, H]
        msg_mlp = msgs[..., 3, :] if msgs is not None else None
        q, k, v = (a.transpose(0, 2, 1, 3) for a in (q, k, v))
        return q, k, v, r, g, sinks, ctx_k, ctx_v, msg_mlp

    # ---- after attention ----
    def _post(self, attn_out, g, x, blocks, conv_a, conv_m, msg_mlp):
        layer = self.layer
        dtype = x.dtype
        b, n, d = x.shape
        out = attn_out.transpose(0, 2, 1, 3).reshape(b, n, -1)
        out = (out.astype(mx.float32) * mx.sigmoid(g.astype(mx.float32))).astype(dtype)
        attn = layer.self_attn.o_proj(out)
        residual, ctx_a = _static_sconv(conv_a, attn, layer.attn_sconv.conv.weight[..., 0], self.sconv_div)
        residual = residual.astype(mx.float32)
        if layer.is_block_boundary:
            blocks = [*blocks, x]
        else:
            residual = residual + x.astype(mx.float32)
        nv = self.cfg.attn_res_normalize_values
        if msg_mlp is None:
            normed = attention_mix(
                [*blocks, residual], layer.mlp_input.query, nv,
                layer.post_attention_layernorm.weight[None], self.eps, dtype,
            )[..., 0, :]
        else:
            hidden = attention_mix([*blocks, residual], layer.mlp_input.query, nv)[..., 0, :]
            hidden = message_gates(
                hidden.reshape(n, d), msg_mlp.reshape(n, d).astype(mx.float32),
                *self.mlp_gates, self.eps, tokens=n,
            ).reshape(hidden.shape)
            normed = layer.post_attention_layernorm(hidden).astype(dtype)
        hidden = layer.mlp(normed)
        mlp_out, ctx_m = _static_sconv(conv_m, hidden, layer.mlp_sconv.conv.weight[..., 0], self.sconv_div)
        out = (residual + mlp_out.astype(mx.float32)).astype(dtype)
        return out, ctx_a, ctx_m

    def __call__(self, x, blocks, cache, prev):
        kv, conv = cache
        attn = self.layer.self_attn
        b, n, hidden = x.shape
        w = self.width
        def zeros(c):
            return mx.zeros((b, w, c), mx.float32)
        kv_ch = attn.n_kv * attn.head_dim
        conv_k = conv[0] if conv[0] is not None else zeros(kv_ch)
        conv_v = conv[1] if conv[1] is not None else zeros(kv_ch)
        conv_a = conv[2] if conv[2] is not None else zeros(hidden)
        conv_m = conv[3] if conv[3] is not None else zeros(hidden)
        tau = None
        if not attn.is_sliding and self.cfg.log_scaling_n_floor is not None:
            a, fl = self.cfg.log_scaling_alpha, self.cfg.log_scaling_n_floor
            tau = mx.array(
                [1 + a * math.log(max((kv.offset + 1 + t) / fl, 1)) for t in range(n)], mx.float32
            )
        prev_blocks, prev_x = prev if prev is not None else (None, None)
        q, k, v, r, g, sinks, ctx_k, ctx_v, msg_mlp = self.pre(x, blocks, prev_blocks, prev_x, conv_k, conv_v, tau)
        k, v = kv.update_and_fetch(k, v)
        mask = None
        if isinstance(kv, RotatingKVCache):
            wrapped = kv._idx != k.shape[2] and kv._idx < kv.offset
            if USE_RING and n == 1 and wrapped and kv.keep == 0 and not getattr(conv, "is_speculating", False):
                # Attend in storage order; the mask maps each slot to its age.
                mask = ring_mask(r, attn.rel_proj.astype(x.dtype), k, kv._idx, attn.sliding, attn.rel_extent)
            else:
                k, v = kv._temporal_order(k), kv._temporal_order(v)
        elif isinstance(kv, BatchRotatingKVCache) and kv.rotated:
            k, v = mx.roll(k, -kv._idx, axis=2), mx.roll(v, -kv._idx, axis=2)
        size = k.shape[2]
        if mask is None:
            mask = banded_additive_mask(
                r, attn.rel_proj.astype(x.dtype), size - n, size, attn.sliding, attn.rel_extent,
                left_padding=getattr(kv, "left_padding", None), shape_ref=k,
            )
        scale = attn.head_dim**-0.5
        if n == 1:
            attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask, sinks=sinks.reshape(-1))
        else:
            # Query-dependent sinks: a zero K/V slot with its own logit.
            mask = mx.concatenate([mask, sinks.transpose(0, 2, 1)[..., None]], axis=-1)
            k = mx.concatenate([k, mx.zeros((b, attn.n_kv, 1, attn.head_dim), k.dtype)], axis=2)
            v = mx.concatenate([v, mx.zeros((b, attn.n_kv, 1, attn.head_dim), v.dtype)], axis=2)
            attn_out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        out, ctx_a, ctx_m = self.post(attn_out, g, x, blocks, conv_a, conv_m, msg_mlp)
        for idx, ctx in enumerate((ctx_k, ctx_v, ctx_a, ctx_m)):
            _commit_conv(conv, idx, ctx, w)
        conv.advance(n)
        if self.layer.is_block_boundary:
            blocks.append(x)
        return out


class FastDecoder:
    def __init__(self, lm):
        self.lm = lm
        model = lm.model
        cfg = model.config
        for layer in model.layers:
            fuse_attention(layer.self_attn)
            fuse_moe(layer.mlp)
        self.schedule = model.schedule()
        self.total = cfg.total_num_hidden_layers
        self.start = cfg.loop_layer_index_start
        self.passes = [
            _Pass(model, model.layers[i], i, (i - self.start) if p > 0 else None)
            for (i, _, p) in self.schedule
        ]
        self.final = mx.compile(self._final)

    def _final(self, blocks, x):
        model = self.lm.model
        h = attention_mix([*blocks, x], model.final_input.query, model.config.attn_res_normalize_values)[..., 0, :]
        return h, model.norm(h)

    def eligible(self, inputs, inputs_embeds, cache, kwargs):
        if cache is None or kwargs.get("return_hidden") or kwargs.get("return_shared_kv"):
            return False
        ref = inputs if inputs_embeds is None else inputs_embeds
        if ref is None or ref.shape[0] != 1 or not (1 <= ref.shape[1] <= MAX_TOKENS or ref.shape[1] in PREFILL_SIZES):
            return False
        ids = kwargs.get("capture_layer_ids")
        if ids is not None and any(type(i) is not int or not 0 <= i <= self.total for i in ids):
            return False
        kv, conv = cache[0]
        return (
            getattr(conv, "left_padding", None) is None
            and getattr(conv, "lengths", None) is None
            and getattr(kv, "left_padding", None) is None
        )

    def forward(self, inputs, inputs_embeds, cache, capture_layer_ids=None):
        model = self.lm.model
        requested = set(capture_layer_ids or [])
        captures = {}
        x = model.embed(inputs) if inputs_embeds is None else inputs_embeds
        blocks, pre_loop, previous = [], None, None
        for step, (i, cache_idx, p) in enumerate(self.schedule):
            if i == self.start:
                if p == 0:
                    pre_loop = (x, list(blocks))
                else:
                    previous = (blocks[len(pre_loop[1]) + 1 :], x)
                    x, blocks = pre_loop[0], list(pre_loop[1])
            if step in requested:
                captures[step] = x
            x = self.passes[step](x, blocks, cache[cache_idx], previous if p > 0 else None)
        _, normalized = self.final(blocks, x)
        if self.total in requested:
            captures[self.total] = normalized.astype(x.dtype)
        hidden = [captures[i] for i in capture_layer_ids] if capture_layer_ids is not None else None
        if PREFILL_LAST_ONLY and normalized.shape[1] > MAX_TOKENS and capture_layer_ids is None:
            # Prefill chunk: only the last position's logits are ever consumed
            # (callers take logits[:, -1]); skip the vocabulary head for the rest.
            normalized = normalized[:, -1:]
        return self.logits(normalized), hidden

    def logits(self, normalized):
        hd = self.lm.lm_head
        if not BF16_HEAD or not hasattr(hd, "bits") or hd.scales.dtype != mx.bfloat16:
            return self.lm._logits_from_norm(normalized)
        # bf16 activations against the 4-bit head (weights unchanged); softcap in f32.
        z = mx.quantized_matmul(
            normalized.astype(mx.bfloat16), hd.weight, hd.scales, hd.get("biases"),
            transpose=True, group_size=hd.group_size, bits=hd.bits, mode=hd.mode,
        ).astype(mx.float32)
        cap = self.lm.config.final_logit_softcapping
        if cap:
            z = mx.tanh(z / cap) * cap
        return z[..., : self.lm.config.unpadded_vocab_size]

    def __call__(self, inputs, inputs_embeds, cache, kwargs):
        from mlx_vlm.speculative.cache_state import start_speculative_cache

        transaction = None
        if kwargs.get("speculative_verify"):
            ref = inputs if inputs_embeds is None else inputs_embeds
            transaction = start_speculative_cache(cache, ref.shape[1])
        try:
            logits, hidden = self.forward(inputs, inputs_embeds, cache, kwargs.get("capture_layer_ids"))
        except BaseException:
            if transaction is not None:
                transaction.abort()
            raise
        return LanguageModelOutput(
            logits=None if kwargs.get("skip_logits") else logits,
            hidden_states=hidden,
            shared_kv_states=None,
            gdn_states=transaction,
        )


def install(model):
    """Route eligible short forwards (decode and speculative verify) to the compiled path."""
    lm = getattr(model, "language_model", model)
    fast.apply(model)
    decoder = FastDecoder(lm)
    original = type(lm).__call__

    def call(self, inputs=None, inputs_embeds=None, cache=None, **kwargs):
        if self is lm and decoder.eligible(inputs, inputs_embeds, cache, kwargs):
            return decoder(inputs, inputs_embeds, cache, kwargs)
        return original(self, inputs=inputs, inputs_embeds=inputs_embeds, cache=cache, **kwargs)

    type(lm).__call__ = call
    mx.eval(lm.parameters())
    mx.clear_cache()
    return decoder


def attention_mix(states, query, normalize_values=True, norm_weights=None, eps=1e-6, out_dtype=mx.float32):
    if USE_MIX:
        return _kernel_mix(states, query, normalize_values, norm_weights, eps, out_dtype)
    out = _attention_residual(mx.stack(states, axis=-2), query, normalize_values)
    if norm_weights is not None:
        out = mx.fast.rms_norm(out, None, eps) * norm_weights.astype(mx.float32)
        out = out.astype(out_dtype)
    return out
