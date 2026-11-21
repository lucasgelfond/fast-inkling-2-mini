#!/bin/zsh
# ANE vs GPU contention: each side alone, then both at once. Run under the GPU lock.
cd ${0:A:h:h}
M=${1:-runtime/ane/q128_fp16_s2048/L7p1_q128_fp16.mlpackage}
A="env PYTHONPATH=src .venv/bin/python scripts/inkling_ane_contention.py"
echo "-- alone"; eval "$A ane --model $M --seconds 5" 2>&1 | grep ANE
eval "$A moe --c 512 --seconds 5" 2>&1 | grep MoE
echo "-- together"
eval "$A moe --c 512 --seconds 12 --windows" 2>&1 | grep MoE &
sleep 4; eval "$A ane --model $M --seconds 5" 2>&1 | grep ANE
wait
