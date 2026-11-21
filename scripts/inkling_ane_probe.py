"""Bisect ANE compile failures: convert small component modules, report compile + placement."""
import sys
import warnings

warnings.filterwarnings("ignore")
import coremltools as ct
import numpy as np
import torch
import torch.nn as nn

from inkling_ane import torch_layer as T
from inkling_ane.convert import placement

Q = 64


class StateMod(nn.Module):
    def __init__(self, fn, bufs):
        super().__init__()
        self.fn = fn
        for n, s in bufs.items():
            self.register_buffer(n, torch.zeros(*s, dtype=torch.float16))
        self.names = list(bufs)

    def forward(self, *a):
        return self.fn(self, *a)


def try_convert(name, mod, inputs):
    mod.eval()
    with torch.no_grad():
        tr = torch.jit.trace(mod, tuple(inputs))
    states = [ct.StateType(ct.TensorType(shape=tuple(getattr(mod, n).shape), dtype=np.float16), name=n)
              for n in getattr(mod, "names", [])]
    mlm = ct.convert(tr, inputs=[ct.TensorType(name=f"i{k}", shape=tuple(v.shape), dtype=np.float16) for k, v in enumerate(inputs)],
                     states=states or None, minimum_deployment_target=ct.target.macOS15,
                     compute_units=ct.ComputeUnit.CPU_AND_NE)
    path = f"runtime/ane/probe/{name}.mlpackage"
    mlm.save(path)
    try:
        ane, tot, other = placement(path)
        print(f"{name:12s} OK  ANE {ane}/{tot}  other {dict(other)}", flush=True)
    except Exception as e:
        print(f"{name:12s} FAIL {str(e)[:80]}", flush=True)


x = torch.randn(1, Q, 2304)
taps = torch.randn(4, 2304) * 0.1


def conv_fn(self, x):
    out, st = T.short_conv(x, self.c, taps, Q)
    self.c[:] = st.to(torch.float16)
    return out


def rms_fn(self, x):
    return T.safe_rms(x, 1e-6) * 2


def attnres_fn(self, blocks, x):
    return T.attn_res(torch.cat([blocks, x[:, None]], 1), torch.randn(3, 2304))[0]


Gm = torch.randn(Q, 16, 1024 + Q)


def bias_fn(self, r):
    b = torch.matmul(r.reshape(Q, 24, 16), Gm)
    return b.permute(1, 0, 2).reshape(1, 8, 3 * Q, 1024 + Q)


def bias2_fn(self, r):
    # elementwise sum over d instead of batched matmul
    rr = r.reshape(Q, 24, 16).permute(1, 0, 2)  # [24, Q, 16]
    acc = rr[:, :, 0:1] * Gm[:, 0]
    for d in range(1, 16):
        acc = acc + rr[:, :, d:d + 1] * Gm[:, d]
    return acc.reshape(1, 8, 3 * Q, 1024 + Q)


def att_fn(self, q, k, v, mask, sink):
    keys = torch.cat([self.kc, k], 2)
    vals = torch.cat([self.vc, v], 2)
    self.kc[:] = keys[:, :, Q:].to(torch.float16)
    self.vc[:] = vals[:, :, Q:].to(torch.float16)
    s = torch.matmul(q, keys.transpose(2, 3)) * 0.088
    p = torch.softmax(torch.cat([s + mask, sink], -1), -1)[..., :-1]
    return torch.matmul(p, vals)


def att_nosink_fn(self, q, k, v, mask, sink):
    keys = torch.cat([self.kc, k], 2)
    vals = torch.cat([self.vc, v], 2)
    self.kc[:] = keys[:, :, Q:].to(torch.float16)
    self.vc[:] = vals[:, :, Q:].to(torch.float16)
    s = torch.matmul(q, keys.transpose(2, 3)) * 0.088
    p = torch.softmax(s + mask, -1)
    return torch.matmul(p, vals)


def l2_fn(self, x):
    return T.safe_l2_clip(x.reshape(1, Q, 24, 96), 22.6).reshape(1, Q, 2304)


which = sys.argv[1:] or ["conv", "rms", "attnres", "bias", "bias2", "att", "attnosink", "l2"]
S = 1024
for w in which:
    if w == "conv":
        try_convert(w, StateMod(conv_fn, {"c": (1, 3, 2304)}), [x])
    if w == "rms":
        try_convert(w, StateMod(rms_fn, {}), [x])
    if w == "attnres":
        try_convert(w, StateMod(attnres_fn, {}), [torch.randn(1, 2, Q, 2304), x])
    if w == "bias":
        try_convert(w, StateMod(bias_fn, {}), [torch.randn(1, Q, 384)])
    if w == "bias2":
        try_convert(w, StateMod(bias2_fn, {}), [torch.randn(1, Q, 384)])
    if w in ("att", "attnosink"):
        fn = att_fn if w == "att" else att_nosink_fn
        try_convert(w, StateMod(fn, {"kc": (1, 8, S, 128), "vc": (1, 8, S, 128)}),
                    [torch.randn(1, 8, 3 * Q, 128), torch.randn(1, 8, Q, 128), torch.randn(1, 8, Q, 128),
                     torch.zeros(1, 1, 3 * Q, S + Q), torch.randn(1, 8, 3 * Q, 1)])
    if w == "l2":
        try_convert(w, StateMod(l2_fn, {}), [x])
