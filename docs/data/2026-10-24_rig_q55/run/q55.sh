#!/bin/bash
# Q55: what tool requests lose without backend sampling. UD-Q4_K_XL, the combined v2 build at its defaults
# (screened head off) with Q108 off (it costs decode time, Q117), real-chat server settings; nine cards, then six.
# Ran 2026-10-24. Usage: [RIG=rig] [OUT=dir] q55.sh   (client.py next to it; writes results_<n>cards.jsonl)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
B=/opt/llama-builds/llama.cpp-20261024-b3052a4f8-pascal-p106-v2-combined/bin
M=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
ssh "$RIG" "mkdir -p /tmp/q55 && pkill -x llama-server; true"
scp -q "$D/client.py" "$RIG":/tmp/q55/client.py
for cards in 0,1,2,3,4,5,6,7,8 0,1,2,3,4,5; do
  n=$(echo $cards | tr ',' '\n' | wc -l)
  echo "== $n cards $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=$cards LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_MMQ_MOE_LEGACY=1 setsid nohup $B/llama-server -m $M -ngl 99 -fa on \
    -c 262144 -b 512 -ub 128 -np 1 -bs --cache-ram 0 --ctx-checkpoints 4 --no-webui --host 127.0.0.1 --port 8089 \
    > /tmp/q55/server_$n.log 2>&1 < /dev/null &"
  for i in $(seq 1 120); do ssh "$RIG" "curl -sf 127.0.0.1:8089/health >/dev/null" && break; sleep 5; done
  ssh "$RIG" "python3 /tmp/q55/client.py 8089 3000" | tee "$OUT/results_${n}cards.jsonl"
  ssh "$RIG" "grep -c 'not compatible with grammar' /tmp/q55/server_$n.log; grep -iE 'backend sampling' /tmp/q55/server_$n.log | sort | uniq -c | head" | tee "$OUT/serverlog_${n}cards.txt"
  ssh "$RIG" "pkill -x llama-server"; sleep 5
done
scp -q -r "$RIG":/tmp/q55 "$OUT/raw"; ssh "$RIG" "rm -rf /tmp/q55"
echo Q55_DONE
