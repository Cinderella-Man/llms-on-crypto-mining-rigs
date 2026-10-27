#!/bin/bash
# Q141 (2026-10-24): the cold-clock effect and PowerMizer "Prefer Maximum Performance", run from a machine that reaches
# the rig over ssh. UD-Q4_K_XL, six cards. Phase A: the current mode (2 = Auto). Phase B: mode 1 on all nine GPUs.
# Each phase: tg128 from idle, tg128 after a 2048-token prompt, and a served 500-token reply after 60 s idle; clocks
# and power logged every 200 ms. The mode goes back to 2 at the end, or on any exit.
#   RIG=<ssh alias> BIN=<rig folder with llama-bench and llama-server> MODEL=<rig path of flint-UD-Q4_K_XL.gguf> \
#   XAUTH=<the rig desktop session's Xauthority file> OUT=<local results folder> bash q141.sh
# The request client is bench/server_mmap/client.py.
set -u
HERE=$(cd "$(dirname "$0")" && pwd)
RIG=${RIG:-rig}
B=${BIN:?folder with llama-bench and llama-server on the rig}
M=${MODEL:?the model file on the rig}
D=${OUT:?local results folder}
CLIENT_PY=${CLIENT_PY:-$HERE/../../../../bench/server_mmap/client.py}
mkdir -p "$D"
X="DISPLAY=:0 XAUTHORITY=${XAUTH:?Xauthority file of the rig desktop session}"
setmode() { ssh "$RIG" "$X; export DISPLAY XAUTHORITY; for g in 0 1 2 3 4 5 6 7 8; do nvidia-settings -a \"[gpu:\$g]/GPUPowerMizerMode=$1\" >/dev/null 2>&1; done; nvidia-settings -q '[gpu:0]/GPUPowerMizerMode' -t; nvidia-settings -q '[gpu:8]/GPUPowerMizerMode' -t"; }
trap 'echo "== restoring mode 2"; setmode 2' EXIT
ssh "$RIG" "mkdir -p /tmp/q141; pkill -x llama-server; true"
scp -q "$CLIENT_PY" "$RIG:/tmp/q141/client.py"
phase() { # $1 label
  local L=$1
  ssh "$RIG" "(timeout 900 nvidia-smi --query-gpu=timestamp,index,clocks.sm,clocks.mem,power.draw,utilization.gpu,pstate --format=csv,noheader -lms 200 > /tmp/q141/clk_$L.csv 2>/dev/null &)"
  echo "== $L idle 60 s, then tg128 from idle $(date +%T)"; sleep 60
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LLAMA_NO_MOE_DOWN_RESCALE=tg $B/llama-bench -m $M -ngl 99 -fa on -p 0 -n 128 -r 3 -o md 2>&1" | grep -E 'tg128|error'
  echo "== $L tg128 after a 2048-token prompt $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LLAMA_NO_MOE_DOWN_RESCALE=tg $B/llama-bench -m $M -ngl 99 -fa on -p 2048 -n 128 -r 3 -o md 2>&1" | grep -E 'pp2048|tg128|error'
  echo "== $L served 500 tokens after 60 s idle $(date +%T)"
  ssh "$RIG" "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 LLAMA_NO_MOE_DOWN_RESCALE=tg setsid nohup $B/llama-server -m $M -ngl 99 -fa on -c 16384 -np 1 -bs --cache-ram 0 --ctx-checkpoints 0 --no-webui --host 127.0.0.1 --port 8089 > /tmp/q141/server_$L.log 2>&1 < /dev/null &"
  for i in $(seq 1 120); do ssh "$RIG" "curl -sf 127.0.0.1:8089/health >/dev/null" && break; sleep 5; done
  sleep 60
  ssh "$RIG" "python3 /tmp/q141/client.py nostream 8089 500"
  ssh "$RIG" "pkill -x llama-server"; sleep 5
}
{
  echo "== mode before: $(ssh "$RIG" "$X; export DISPLAY XAUTHORITY; nvidia-settings -q '[gpu:0]/GPUPowerMizerMode' -t")"
  phase A_mode2
  echo "== setting mode 1 (Prefer Maximum Performance) on all nine GPUs: $(setmode 1 | tr '\n' ' ')"
  phase B_mode1
} | tee "$D/q141.md"
scp -q -r "$RIG:/tmp/q141" "$D/raw"; ssh "$RIG" "rm -rf /tmp/q141"
echo Q141_DONE | tee -a "$D/q141.md"
