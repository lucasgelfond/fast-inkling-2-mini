"""Teacher-forced quality (top-1 accuracy, perplexity) of the current inkling_fast config."""
import math
import os
import sys
from pathlib import Path

import mlx.core as mx

from inkling_fast import fast

os.environ.setdefault("INKLING_PREFILL_LAST_ONLY", "0")  # teacher forcing needs every position
import os as _os

from mlx_vlm import load

WEIGHTS = _os.environ.get("INKLING_WEIGHTS", "weights/inkling-2-mini/mlx-3bit")
if "oq" in WEIGHTS:
    from inkling_fast.oq import load_oq
    model, processor = load_oq(WEIGHTS)
else:
    model, processor = load(WEIGHTS)
fast.apply(model)
if os.environ.get("INKLING_COMPILED"):
    from inkling_fast import decode
    decode.install(model)
lm, tok = model.language_model, processor.tokenizer
docs = ["README.md", "docs/inkling-fast.md", "docs/inkling-ane-prefill.md", "docs/tensorfold-inkling2-mini.md"]
nll = acc = n = 0
for d in docs:
    ids = tok.encode(Path(d).read_text())[:512]
    logits = lm(mx.array([ids]), cache=lm.make_cache()).logits[0, :-1].astype(mx.float32)
    lp = logits - mx.logsumexp(logits, -1, keepdims=True)
    t = mx.array(ids[1:])
    nll += -mx.take_along_axis(lp, t[:, None], -1).sum().item()
    acc += (mx.argmax(lp, -1) == t).sum().item(); n += t.size
print(f"{sys.argv[1] if len(sys.argv)>1 else ''}: top1 {acc/n:.4f}  ppl {math.exp(nll/n):.3f}  ({n} tokens)")
