"""Can MLX GPU work and a Core ML ANE call overlap from one Python process?"""
import threading
import time
import warnings

warnings.filterwarnings("ignore")
import coremltools as ct
import mlx.core as mx
import mlx_vlm.models.inkling2.language as L
import numpy as np

from inkling_ane.mlx_layer import build_layer, load_config
from inkling_fast import fast

cfg, quant = load_config()
layer = build_layer(7, cfg, quant)
fast.fuse_moe(layer.mlp)
L.SparseMoE.__call__ = fast._moe_call
x = (mx.random.normal((1, 512, 2304)) * 0.5).astype(mx.bfloat16)
mx.eval(x)
m = ct.models.MLModel("runtime/ane/q128_fp16_s2048/L7p1_q128_fp16.mlpackage", compute_units=ct.ComputeUnit.CPU_AND_NE)
st = m.make_state()
feed = {i.name: np.zeros(tuple(i.type.multiArrayType.shape), np.float16) for i in m.get_spec().description.input}


def ane(k=4):
    for _ in range(k):
        m.predict(feed, state=st)


def moe():
    return layer.mlp(x)


def t(f, n=10):
    f()
    s = time.perf_counter()
    for _ in range(n):
        f()
    return (time.perf_counter() - s) / n * 1e3


ta = t(ane)
tm = t(lambda: mx.eval(moe()))


def seq_async():
    y = moe(); mx.async_eval(y); ane(); mx.eval(y)


def threaded():
    y = moe()
    th = threading.Thread(target=ane); th.start(); mx.eval(y); th.join()


def threaded_ane_bg():
    th = threading.Thread(target=ane); th.start(); mx.eval(moe()); th.join()


print(f"ANE 4 calls {ta:.1f} ms, MoE C=512 {tm:.1f} ms, sum {ta + tm:.1f}, max {max(ta, tm):.1f}")
print(f"async_eval + predict:     {t(seq_async):.1f} ms")
print(f"predict thread + eval:    {t(threaded):.1f} ms")
print(f"thread first, then build: {t(threaded_ane_bg):.1f} ms")
