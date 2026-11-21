#!/bin/zsh
# End-to-end server benchmark: decode tok/s and cold TTFT for a ~2k-token prompt.
# Usage: scripts/inkling_server_bench.sh gpu|ane
set -u
mode=$1; port=8770
cd ${0:A:h:h}
if [ $mode = ane ]; then py=.venv/bin/python; export INKLING_ANE_PREFILL=1; else py=.venv/bin/python; fi
PYTHONPATH=src $py scripts/inkling_serve.py --model weights/inkling-2-mini/mlx-3bit --port $port --max-num-seqs 1 > runtime/logs/server-bench-$mode.log 2>&1 &
pid=$!
trap "kill $pid 2>/dev/null" EXIT
for i in $(seq 1 120); do curl -s 127.0.0.1:$port/v1/models >/dev/null 2>&1 && break; sleep 2; done
# wait until a real completion succeeds (model loaded), then let ANE slices finish loading
for i in $(seq 1 120); do
  curl -sf 127.0.0.1:$port/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"weights/inkling-2-mini/mlx-3bit","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' >/dev/null 2>&1 && break
  sleep 3
done
[ $mode = ane ] && sleep 20
.venv/bin/python scripts/openai_bench.py --url http://127.0.0.1:$port --model weights/inkling-2-mini/mlx-3bit --label server-$mode --extra '{"chat_template_kwargs":{"reasoning_effort":"none"}}'
.venv/bin/python scripts/inkling_ane_ttft.py --url http://127.0.0.1:$port --reps 3 2>&1 | tail -3
