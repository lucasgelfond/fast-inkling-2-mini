#!/bin/zsh
# Server TTFT for a ~2k-token prompt, without and with INKLING_ANE_PREFILL. Run under the GPU lock.
cd ${0:A:h:h}
PORT=${PORT:-8799}
for flag in 0 1; do
  log=runtime/ane/serve_flag$flag.log
  INKLING_ANE_PREFILL=$flag APC_ENABLED=0 PYTHONPATH=src .venv/bin/python scripts/inkling_serve.py \
      --model weights/inkling-2-mini/mlx-3bit --port $PORT > $log 2>&1 &
  pid=$!
  until curl -s localhost:$PORT/v1/models >/dev/null 2>&1; do sleep 2; kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 $log; exit 1; }; done
  # first request warms the model (lazy load in mlx-vlm server)
  until .venv/bin/python scripts/inkling_ane_ttft.py --url http://127.0.0.1:$PORT --reps 1 --chars 400 > /dev/null 2>&1; do
    sleep 5; kill -0 $pid 2>/dev/null || { echo "server died"; tail -20 $log; exit 1; }
  done
  if [[ $flag == 1 ]]; then until grep -q "slices ready" $log; do sleep 2; done; fi
  echo "== INKLING_ANE_PREFILL=$flag"
  .venv/bin/python scripts/inkling_ane_ttft.py --url http://127.0.0.1:$PORT --reps 3
  grep "inkling_ane\]" $log | tail -4
  kill $pid; wait $pid 2>/dev/null
done
