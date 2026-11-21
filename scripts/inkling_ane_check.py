"""Per-layer cosine of the Core ML dense slices ON THE ANE vs MLX golden data (chunked, stateful).

PYTHONPATH=src .venv/bin/python scripts/inkling_ane_check.py --golden runtime/ane/golden128 --q 64 --steps 2 5 19 23 [--int8]
"""
import argparse
import json
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import coremltools as ct
import numpy as np

from inkling_ane.convert import build, convert, n_blocks
from inkling_ane.torch_layer import host_mask


def cos(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def model_path(out, i, lp, q, tag):
    return Path(out) / f"L{i}p{lp}_q{q}_{tag}.mlpackage"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--golden", default="runtime/ane/golden128")
    p.add_argument("--steps", type=int, nargs="+", default=list(range(48)))
    p.add_argument("--q", type=int, default=64)
    p.add_argument("--s-global", type=int, default=2048)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--pal", type=int, default=0)
    p.add_argument("--units", default="CPU_AND_NE")
    p.add_argument("--out", default="runtime/ane/models")
    args = p.parse_args()
    tag = "int8" if args.int8 else (f"pal{args.pal}" if args.pal else "fp16")
    meta = json.loads((Path(args.golden) / "meta.json").read_text())
    worst = 1.0
    for step in args.steps:
        i, lp = meta["schedule"][step]
        path = model_path(args.out, i, lp, args.q, tag)
        if not path.exists():
            convert(build(i, lp, args.q, args.s_global), path, args.int8, args.pal)
        m = ct.models.MLModel(str(path), compute_units=getattr(ct.ComputeUnit, args.units))
        st = m.make_state()
        g = np.load(Path(args.golden) / f"step{step}.npz")
        n, q = g["x"].shape[0], args.q
        sliding = i in json.loads((Path("weights/inkling-2-mini/mlx-3bit/config.json")).read_text())["text_config"]["local_layer_ids"]
        S = 1024 if sliding else args.s_global
        hs, rs = [], []
        for s in range(0, n, q):
            feed = {"x": g["x"][s:s + q][None].astype(np.float16), "mask": host_mask(s, q, S, sliding).astype(np.float16)}
            if n_blocks(i):
                feed["blocks"] = g["blocks"][:, s:s + q][None].astype(np.float16)
            if lp:
                feed["msg_states"] = g["msg_states"][:, s:s + q][None].astype(np.float16)
            o = m.predict(feed, state=st)
            hs.append(o["h"][0]); rs.append(o["residual"][0])
        h, r = np.concatenate(hs).astype(np.float32), np.concatenate(rs).astype(np.float32)
        ch, cr = cos(h, g["h"]), cos(r, g["residual"])
        worst = min(worst, ch, cr)
        print(f"step {step:2d} L{i:2d} p{lp}: cos h {ch:.6f} residual {cr:.6f}  (2nd chunk h {cos(h[q:], g['h'][q:]):.6f})", flush=True)
    print(f"worst {worst:.6f}")


if __name__ == "__main__":
    main()
