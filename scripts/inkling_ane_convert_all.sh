#!/bin/zsh
# Convert every dense slice Inkling 2 Mini needs (36 = 12 plain + 12 loop x {pass0, pass>=1}).
# CPU only (no ANE compile): check placement separately, under the GPU lock.
# Usage: scripts/inkling_ane_convert_all.sh <q> <tag: fp16|int8> [s_global]
set -u
cd ${0:A:h:h}
q=$1; tag=$2; sg=${3:-2048}
out=runtime/ane/q${q}_${tag}_s${sg}
mkdir -p $out
flag=""; [[ $tag == int8 ]] && flag="--int8"
for i in {0..23}; do
  passes=(0); (( i >= 6 && i < 18 )) && passes=(0 1)
  for p in $passes; do
    [[ -e $out/L${i}p${p}_q${q}_${tag}.mlpackage ]] && continue
    PYTHONPATH=src .venv/bin/python -c "
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import shutil
from inkling_ane.convert import build, convert
path = Path('$out/L${i}p${p}_q${q}_${tag}.mlpackage')
tmp = Path('$out/tmp_L${i}p${p}.mlpackage')
convert(build($i, $p, $q, $sg), tmp, '$tag' == 'int8')
tmp.rename(path)
print('L$i p$p converted', flush=True)
" 2>&1 | grep -E "^L[0-9]|Error|FAILED\.T"
  done
done
echo done
