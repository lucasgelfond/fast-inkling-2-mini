"""Check the PyTorch dense-layer restatement (fp32, CPU) against MLX golden data."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from inkling_ane.torch_layer import DenseLayer, host_mask
from inkling_ane.weights import config, layer_weights


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="runtime/ane/golden48")
    p.add_argument("--steps", type=int, nargs="+", default=list(range(48)))
    p.add_argument("--q", type=int, default=0, help="chunk rows (0 = whole prompt)")
    args = p.parse_args()
    cfg, _ = config()
    meta = json.loads((Path(args.golden) / "meta.json").read_text())
    msgq = layer_weights("message_input.")["query"]
    cache = {}
    for step in args.steps:
        i, lp = meta["schedule"][step]
        g = np.load(Path(args.golden) / f"step{step}.npz")
        n = g["x"].shape[0]
        q = args.q or n
        if i not in cache:
            cache = {i: layer_weights(f"layers.{i}.")}
        w = cache[i]
        m = DenseLayer(cfg, i, lp, q, w, msg_query=msgq[i - 6] if lp else None, s_global=max(n, 1024)).eval()
        hs, rs = [], []
        with torch.no_grad():
            for s in range(0, n, q):
                x = torch.tensor(g["x"][s:s + q])[None]
                blocks = torch.tensor(g["blocks"][:, s:s + q])[None] if len(g["blocks"]) else None
                msgs = torch.tensor(g["msg_states"][:, s:s + q])[None] if lp else None
                mask = torch.tensor(host_mask(s, q, m.S, m.sliding))
                for b in m.buffers():
                    b.data = b.data.float() if s == 0 else b.data
                h, r = m(x, mask, blocks, msgs)
                hs.append(h[0].numpy()); rs.append(r[0].numpy())
        h, r = np.concatenate(hs), np.concatenate(rs)
        print(f"step {step:2d} L{i:2d} p{lp} {'slide' if m.sliding else 'glob '}: cos h {cos(h, g['h']):.6f}  residual {cos(r, g['residual']):.6f}", flush=True)


if __name__ == "__main__":
    main()
