"""LoopCD (Liu et al., Apple, arXiv 2610.02185) on Inkling 2 Mini, teacher-forced.

Each Inkling loop pass restarts from the pre-loop state and reads "messages"
from the previous pass, so the state after pass k (loop output x_k plus the
block states that pass appended) is a complete input for the coda (layers
18-23 + final mixing + head). That gives the aligned weak/strong pair LoopCD
needs: h_1 (one pass) vs h_R.

Forms:
  hidden: coda(h_R + w (h_R - h_1))                         (no extra cost)
  logits: z_R + w (z_R - z_1), z_k = head(coda(h_k))        (one extra coda+head)
  adaptive: w = w_max * (1 - (p1 - p2)) from the unguided top-2 margin (logits form)

Metrics on the true next token: top-1 accuracy and NLL; plus agreement with the
unguided 3-pass model.
"""

import argparse
import math
from pathlib import Path

import mlx.core as mx
from mlx_vlm import load

import inkling_fast  # noqa: F401


def loop_states(lm, ids, passes):
    """Return (pre_blocks_len, [ (x_k, blocks_k) for k = 1..passes ])."""
    model = lm.model
    cfg = model.config
    start, end = cfg.loop_layer_index_start, cfg.loop_layer_index_end
    x = model.embed(ids)
    blocks = []
    for i in range(start):
        x = model.layers[i](x, blocks, None)
    pre_x, pre_blocks = x, list(blocks)
    states, previous = [], None
    for p in range(passes):
        x, blocks = pre_x, list(pre_blocks)
        for i in range(start, end):
            layer = model.layers[i]
            if p == 0:
                x = layer(x, blocks, None)
            else:
                x = layer(x, blocks, None, model.message_input(previous[0], previous[1], i - start))
        previous = (blocks[len(pre_blocks) + 1 :], x)
        states.append((x, list(blocks)))
    return len(pre_blocks), states


def coda_logits(lm, x, blocks):
    model = lm.model
    cfg = model.config
    blocks = list(blocks)
    for i in range(cfg.loop_layer_index_end, cfg.num_hidden_layers):
        x = model.layers[i](x, blocks, None)
    h = model.final_input(blocks, x)[..., 0, :]
    return lm._logits_from_norm(model.norm(h)).astype(mx.float32)


def mix_state(n_pre, strong, weak, w):
    (xs, bs), (xw, bw) = strong, weak
    return xs + w * (xs - xw), bs[:n_pre] + [b + w * (b - b1) for b, b1 in zip(bs[n_pre:], bw[n_pre:])]


def logsoftmax(z):
    return z - mx.logsumexp(z, axis=-1, keepdims=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="weights/inkling-2-mini/mlx-3bit")
    p.add_argument("--text", nargs="+", default=["README.md", "docs/inkling-fast.md", "docs/inkling-ane-prefill.md", "docs/tensorfold-inkling2-mini.md"])
    p.add_argument("--tokens", type=int, default=512)
    a = p.parse_args()
    model, processor = load(a.model)
    lm = model.language_model
    tok = processor.tokenizer
    rows = {}

    def add(name, logits, target, ref_top):
        lp = logsoftmax(logits)
        top = mx.argmax(lp, -1)
        r = rows.setdefault(name, [0.0, 0, 0, 0])
        r[0] += -mx.take_along_axis(lp, target[:, None], -1).sum().item()
        r[1] += (top == target).sum().item()
        r[2] += (top == ref_top).sum().item()
        r[3] += target.size

    for path in a.text:
        ids = tok.encode(Path(path).read_text())[: a.tokens]
        x = mx.array([ids])
        target = mx.array(ids[1:])
        n_pre, st = loop_states(lm, x, 3)
        z = {k: coda_logits(lm, *st[k - 1])[0, :-1] for k in (1, 2, 3)}
        ref_top = mx.argmax(z[3], -1)
        add("3 passes (baseline)", z[3], target, ref_top)
        add("2 passes", z[2], target, ref_top)
        add("1 pass", z[1], target, ref_top)
        for R in (3, 2):
            for w in (0.1, 0.2, 0.3, 0.5):
                add(f"{R}p logits w={w}", z[R] + w * (z[R] - z[1]), target, ref_top)
                add(f"{R}p hidden w={w}", coda_logits(lm, *mix_state(n_pre, st[R - 1], st[0], w))[0, :-1], target, ref_top)
            for wmax in (0.5, 1.0):
                pr = mx.softmax(z[R], -1)
                top2 = mx.topk(pr, 2, axis=-1)
                margin = mx.abs(top2[..., 0] - top2[..., 1])
                w = (wmax * (1 - margin))[:, None]
                add(f"{R}p logits adaptive wmax={wmax}", z[R] + w * (z[R] - z[1]), target, ref_top)
        mx.clear_cache()
        print(f"done {path} ({len(ids)} tokens)", flush=True)

    print(f"\n{'config':34s} {'top1 acc':>9} {'ppl':>8} {'agree w/ 3p':>12}")
    for name, (nll, acc, agree, n) in rows.items():
        print(f"{name:34s} {acc / n:9.4f} {math.exp(nll / n):8.3f} {agree / n:12.3f}")


if __name__ == "__main__":
    main()
