#!/bin/zsh
# Run under the GPU lock. Each side alone, then together.
cd ${0:A:h:h}
P="env PYTHONPATH=src .venv/bin/python scripts/inkling_ane_bandwidth.py"
for k in int8 pal4 fp16; do eval "$P ane --kind $k --placement --seconds 4" 2>&1 | grep -E "^ANE|^placement|FAILED\.T"; done
eval "$P gpu --seconds 4" 2>&1 | grep "^GPU"
for k in int8 fp16; do
  echo "-- together ($k)"
  eval "$P gpu --seconds 10" 2>&1 | grep GPU &
  sleep 3; eval "$P ane --kind $k --seconds 5" 2>&1 | grep "^ANE"
  wait
done
