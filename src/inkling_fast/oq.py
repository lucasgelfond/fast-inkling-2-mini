"""Load oMLX "oQ" Inkling 2 Mini checkpoints (e.g. mlx-community/Inkling-2-Mini-oQ4e-mtp)
into the mlx-vlm module layout used by inkling_fast, without rewriting files.

oQ keeps TML's routed experts in their native NVFP4 (layers 0-19, with a per-expert
global scale applied after the matmul) and stores attention fused (`attn.in_proj` rows
q|k|v|r|gate|sink on sliding layers; q|r|gate|sink, k|k_dyn, v|v_dyn on global layers).
Quantized rows are independent, so splitting fused rows is exact; the few rows that
mlx-vlm keeps unquantized (attention sinks, dynamic-conv projections) are dequantized.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn

NVFP4_DENOMINATOR = 448.0 * 6.0
_GATE_ORDER = ("q", "k", "v", "mlp")  # oQ loop input order: attn(q), k, v, mlp


def _rows(t, start, stop):
    return {k: mx.contiguous(v[start:stop]) for k, v in t.items() if v is not None}


def _dequant(t, spec):
    return mx.dequantize(
        t["weight"], t["scales"], t.get("biases"), group_size=spec["group_size"],
        bits=spec["bits"], mode=spec.get("mode", "affine"),
    ).astype(mx.bfloat16)


def load_oq(path):
    """Returns (model, processor) with oQ weights mapped into the inkling2_loop model."""
    from mlx_vlm.utils import load_processor

    import inkling_fast  # noqa: F401  (registers inkling2_loop)

    from mlx_vlm.models.inkling2_loop import Model, ModelConfig

    path = Path(path)
    config = json.loads((path / "config.json").read_text())
    quant = config.get("quantization") or {}
    default = {k: quant[k] for k in ("group_size", "bits", "mode") if k in quant}
    text = config["text_config"]
    hidden = text["hidden_size"]
    loop = text["loop_config"]
    start, end = loop["core_start"], loop["core_end"]
    heads, kv, hd, drel = text["num_attention_heads"], text["num_key_value_heads"], text["head_dim"], text["d_rel"]
    ksz = text.get("sconv_kernel_size", 4)
    q_rows, kv_rows, r_rows = heads * hd, kv * hd, heads * drel

    def spec(name):
        s = quant.get(name)
        return dict(s) if isinstance(s, dict) else dict(default)

    # Load every tensor (lazily mapped by MLX).
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    src = {}
    for shard in sorted(set(index.values())):
        src.update(mx.load(str(path / shard)))

    def group(prefix):
        return {s: src.get(f"{prefix}.{s}") for s in ("weight", "scales", "biases")}

    out, qspecs, post_scales, fused_w13 = {}, {}, {}, {}
    lm = "language_model.model."
    for i in range(text["num_hidden_layers"]):
        o = f"model.layers.{i}."
        t = f"{lm}layers.{i}."
        attn = t + "self_attn."
        aspec = spec(o + "attn.o_proj")
        if f"{o}attn.in_proj.weight" in src:  # sliding: q|k|v|r|gate|sink
            ip, ispec = group(o + "attn.in_proj"), spec(o + "attn.in_proj")
            cuts = [0, q_rows, q_rows + kv_rows, q_rows + 2 * kv_rows, q_rows + 2 * kv_rows + r_rows,
                    2 * q_rows + 2 * kv_rows + r_rows, 2 * q_rows + 2 * kv_rows + r_rows + heads]
            for name, a, b in zip(("q_proj", "k_proj", "v_proj", "r_proj", "gate_proj"), cuts, cuts[1:], strict=False):
                for k, v in _rows(ip, a, b).items():
                    out[f"{attn}{name}.{k}"] = v
                qspecs[f"{attn}{name}"] = ispec
            out[f"{attn}sink_proj.weight"] = _dequant(_rows(ip, cuts[5], cuts[6]), ispec)
            out[f"{attn}k_sconv.conv.weight"] = src[o + "attn.k_conv"][..., None]
            out[f"{attn}v_sconv.conv.weight"] = src[o + "attn.v_conv"][..., None]
            out[f"{t}input_layernorm.weight"] = src[o + "attn_norm.weight"]
        else:  # global: q|r|gate|sink, k|k_dyn, v|v_dyn
            qp, qs = group(o + "attn.q_proj"), spec(o + "attn.q_proj")
            cuts = [0, q_rows, q_rows + r_rows, 2 * q_rows + r_rows, 2 * q_rows + r_rows + heads]
            for name, a, b in zip(("q_proj", "r_proj", "gate_proj"), cuts, cuts[1:], strict=False):
                for k, v in _rows(qp, a, b).items():
                    out[f"{attn}{name}.{k}"] = v
                qspecs[f"{attn}{name}"] = qs
            out[f"{attn}sink_proj.weight"] = _dequant(_rows(qp, cuts[3], cuts[4]), qs)
            for n in ("k", "v"):
                g, s = group(f"{o}attn.{n}_proj"), spec(f"{o}attn.{n}_proj")
                for k, v in _rows(g, 0, kv_rows).items():
                    out[f"{attn}{n}_proj.{k}"] = v
                qspecs[f"{attn}{n}_proj"] = s
                out[f"{attn}{n}_sconv.weight_proj.weight"] = _dequant(_rows(g, kv_rows, kv_rows + kv * ksz), s)
                out[f"{attn}{n}_sconv.static_weight"] = src[f"{o}attn.{n}_static"]
            for n in ("q", "k", "v"):
                out[f"{t}{n}_input_layernorm.weight"] = src[f"{o}{n}_input_norm.weight"]
        for k, v in group(o + "attn.o_proj").items():
            if v is not None:
                out[f"{attn}o_proj.{k}"] = v
        qspecs[f"{attn}o_proj"] = aspec
        out[f"{attn}q_scale"] = src[o + "attn.q_scale"]
        out[f"{attn}rel_scale"] = src[o + "attn.r_scale"]
        out[f"{attn}rel_proj"] = src[o + "attn.rel_proj"]
        out[f"{t}attn_sconv.conv.weight"] = src[o + "attn_conv"][..., None]
        out[f"{t}mlp_sconv.conv.weight"] = src[o + "mlp_conv"][..., None]
        out[f"{t}post_attention_layernorm.weight"] = src[o + "mlp_norm.weight"]
        aq = src.get(o + "attn_input.query")
        if aq is not None:
            out[f"{t}attention_input.query"] = aq.reshape(-1) if aq.shape[0] == 1 else aq
        out[f"{t}mlp_input.query"] = src[o + "mlp_input.query"].reshape(-1)
        out[f"{t}mlp.gate_weight"] = src[o + "mlp.router_weight"]
        out[f"{t}mlp.e_score_correction_bias"] = src[o + "mlp.router_bias"]
        # Experts: w13 rows are interleaved [gate0, up0, gate1, up1, ...]; keep them packed
        # as-is (no copies) and let the fused MoE read up/gate as odd/even output columns.
        glu = f"{t}mlp.switch_mlp."
        fused_w13[i] = (group(o + "mlp.experts.w13"), spec(o + "mlp.experts.w13"))
        qspecs[f"{glu}gate_proj"] = qspecs[f"{glu}up_proj"] = fused_w13[i][1]
        for k, v in group(o + "mlp.experts.w2").items():
            if v is not None:
                out[f"{glu}down_proj.{k}"] = v
        qspecs[f"{glu}down_proj"] = spec(o + "mlp.experts.w2")
        g13, g2 = src.get(o + "mlp.experts.w13.global_scale"), src.get(o + "mlp.experts.w2.global_scale")
        if g13 is not None:
            post_scales[i] = (g13 / NVFP4_DENOMINATOR, g2 / NVFP4_DENOMINATOR)
        # Loop message gates and queries.
        if start <= i < end:
            down, gf = src[o + "loop_down"], src[o + "loop_gain_forget"]
            rank = down.shape[-1]
            for m, gname in enumerate(_GATE_ORDER):
                gate = f"{t}{gname}_message_gate."
                out[gate + "down_proj.weight"] = down[m].transpose(2, 0, 1).reshape(rank, 2 * hidden)
                out[gate + "message_gate_proj.weight"] = gf[m][:, :hidden].T
                out[gate + "forget_gate_proj.weight"] = gf[m][:, hidden:].T
    msg = mx.stack([src[f"model.layers.{i}.loop_message.query"] for i in range(start, end)])
    out[f"{lm}message_input.query"] = msg
    out[f"{lm}final_input.query"] = src["model.final_input.query"].reshape(-1)
    for k, v in group("model.embed").items():
        if v is not None:
            out[f"{lm}embed_tokens.{k}"] = v
    qspecs[f"{lm}embed_tokens"] = spec("model.embed")
    out[f"{lm}embed_norm.weight"] = src["model.embed_norm.weight"]
    out[f"{lm}norm.weight"] = src["model.norm.weight"]
    for k, v in group("lm_head").items():
        if v is not None:
            out[f"language_model.lm_head.{k}"] = v
    qspecs["language_model.lm_head"] = spec("lm_head")

    model = Model(ModelConfig.from_dict(config))

    def predicate(p, m):
        s = qspecs.get(p)
        if s is None or not hasattr(m, "to_quantized"):
            return False
        return {"group_size": s["group_size"], "bits": s["bits"], "mode": s.get("mode", "affine")}

    nn.quantize(model, class_predicate=predicate)
    wanted = {k for k, _ in nn.utils.tree_flatten(model.language_model.parameters())}
    have = {k[len("language_model."):] for k in out}
    # gate/up come from the packed w13 (installed below), not from load_weights.
    wanted = {k for k in wanted if not (".switch_mlp.gate_proj." in k or ".switch_mlp.up_proj." in k)}
    missing = sorted(wanted - have)
    extra = sorted(have - wanted)
    if missing or extra:
        raise ValueError(f"oQ mapping mismatch: missing {missing[:8]} (+{len(missing)}), extra {extra[:8]} (+{len(extra)})")
    model.load_weights(list(out.items()), strict=False)
    del out
    lmod = model.language_model
    for i, layer in enumerate(lmod.model.layers):
        moe = layer.mlp
        t13, s13spec = fused_w13[i]
        width = t13["weight"].shape[1] // 2
        object.__setattr__(moe, "_fast_gate_up", (
            t13["weight"], t13["scales"], t13.get("biases"), s13spec["bits"], s13spec["group_size"],
            s13spec.get("mode", "affine"), width, True,  # interleaved gate/up rows
        ))
        for proj in (moe.switch_mlp.up_proj, moe.switch_mlp.gate_proj):
            proj.weight = mx.zeros((1,), mx.uint32)
            proj.scales = mx.zeros((1,), mx.uint8)
            if proj.get("biases") is not None:
                proj.biases = mx.zeros((1,), mx.bfloat16)
        if i in post_scales:
            object.__setattr__(moe, "_oq_post_scales", tuple(v.astype(mx.float32) for v in post_scales[i]))
        # Evaluate one layer at a time: one huge lazy graph times out the GPU.
        mx.eval(layer.parameters(), [v for v in moe.__dict__["_fast_gate_up"][:3] if v is not None],
                moe.__dict__.get("_oq_post_scales", ()))
        mx.clear_cache()
    mx.eval(lmod.parameters())
    src.clear()  # drop the source tensors (the closure above still refers to this dict)
    mx.clear_cache()
    model.eval()
    processor = load_processor(path, add_detokenizer=True)
    return model, processor
