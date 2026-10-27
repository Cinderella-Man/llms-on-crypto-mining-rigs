#!/bin/bash
# Q118 M0 (2026-10-24): the host's share of a served token. UD-Q4_K_XL, nine cards, v2 base build. llama-bench at 1 and 2
# threads, then llama-server -bs at -t 2 and -t 1, one 1,000-token request without and with streaming each, while top -H,
# nvidia-smi (200 ms), vmstat and the process's page-fault counters are logged; the logs land in $OUT/raw.
# Usage: [RIG=rig] [OUT=dir] m0.sh   (client: bench/server_mmap/client.py)
set -u
RIG=${RIG:-rig}     # ssh alias of the rig
OUT=${OUT:-$PWD}   # where the logs and tables go
D=$(cd "$(dirname "$0")" && pwd)
B=/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin
M=/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
ssh "$RIG" "mkdir -p /tmp/host && pkill -x llama-server; true"
scp -q "$D/../../../../bench/server_mmap/client.py" "$RIG":/tmp/host/client.py
echo "== warm-up (discarded)"; ssh "$RIG" "$E $B/llama-bench -m $M -ngl 99 -fa on -p 512 -n 32 -r 1 -o md" > "$OUT/bench_warm.md" 2>&1
echo "== llama-bench tg128 -t 1,2"; ssh "$RIG" "$E $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 3 -t 1,2 -o md" | tee "$OUT/bench_tg128.md"
for T in 2 1; do
  echo "== server -t $T"
  ssh "$RIG" "$E setsid nohup $B/llama-server -m $M -ngl 99 -fa on -c 16384 -np 1 -bs --cache-ram 0 --ctx-checkpoints 0 -t $T --no-webui --host 127.0.0.1 --port 8089 > /tmp/host/server_t$T.log 2>&1 < /dev/null &"
  for i in $(seq 1 120); do ssh "$RIG" "curl -sf 127.0.0.1:8089/health >/dev/null" && break; sleep 5; done
  ssh "$RIG" "python3 /tmp/host/client.py nostream 8089 128" > "$OUT/warm_t$T.json"
  for mode in nostream stream; do
    ssh "$RIG" "P=\$(pgrep -x llama-server); awk '{print \$10, \$12}' /proc/\$P/stat > /tmp/host/faults_before_t${T}_$mode.txt; \
      (top -H -b -d 1 -n 50 -p \$P > /tmp/host/threads_t${T}_$mode.txt 2>/dev/null &); \
      (timeout 50 nvidia-smi --query-gpu=timestamp,index,utilization.gpu --format=csv,noheader -lms 200 > /tmp/host/util_t${T}_$mode.csv 2>/dev/null &); \
      (vmstat 1 50 > /tmp/host/vmstat_t${T}_$mode.txt 2>/dev/null &); \
      python3 /tmp/host/client.py $mode 8089 1000; \
      awk '{print \$10, \$12}' /proc/\$P/stat > /tmp/host/faults_after_t${T}_$mode.txt" | tee "$OUT/run_t${T}_$mode.json"
    sleep 20
  done
  ssh "$RIG" "pkill -x llama-server"; sleep 5
done
scp -q -r "$RIG":/tmp/host "$OUT/raw"
ssh "$RIG" "rm -rf /tmp/host"
echo M0_DONE
