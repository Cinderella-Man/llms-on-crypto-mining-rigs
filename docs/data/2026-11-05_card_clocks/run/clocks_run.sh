#!/usr/bin/env bash
# clocks_run.sh (C532, Q1021): the cards' clocks while the release line decodes, run on the rig itself. An nvidia-smi log
# every SMI_MS (100) ms per card runs through three parts, each marked in marks.tsv with bash's $EPOCHREALTIME:
#   idle    IDLE_S (15) s with nothing loaded
#   bench   llama-bench on S35's no-draft decode line (docs/data/2026-11-04_s35_stack): K4_INST (10) instances of a
#           4-token prompt pass (T(4), as docs/data/2026-11-04_kshape/run/rig_tk.sh times T(k)) and tg128, -r REPS (10)
#           each, with --progress; both output streams are stamped line by line, so every repetition's window is known
#   served  S35's llama-server line (the agent check's COMMON plus S35's MTP draft line and environment) on the loopback,
#           SERVED_IDLE_S (10) s idle once it is healthy, then one request REQ (the agent check's first turn, built by
#           bench/agent_check/first_turn_request.sh) sent from the rig itself; the reply's timings split prompt and decode
# Both llama processes run under scripts/ram_watchdog.sh (WD) with MIN_AVAIL_MB=60, as S35 ran them.
# Environment: BIN (the build's bin folder), MODEL (the file's first shard), DRAFT (the MTP draft), REQ, WD, OUT (a
# folder; tmpfs is best, as the log is written every SMI_MS), PORT (8192). Writes OUT/DONE last, with the exit codes.
# Tables afterwards: python3 -I clock_phases.py OUT.
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${DRAFT:?}" "${REQ:?}" "${WD:?}" "${OUT:?}"
export LC_ALL=C
PORT=${PORT:-8192}; SMI_MS=${SMI_MS:-100}; IDLE_S=${IDLE_S:-15}; SERVED_IDLE_S=${SERVED_IDLE_S:-10}
REPS=${REPS:-10}; K4_INST=${K4_INST:-10}
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
COMMON="-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 \
-otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
mkdir -p "$OUT" && cd "$OUT" || exit 1
mark() { printf '%s\t%s\n' "$EPOCHREALTIME" "$1" >> marks.tsv; }
stamp() { while IFS= read -r l; do printf '%s\t%s\n' "$EPOCHREALTIME" "$l"; done; }

{ date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|nvidia-smi' || echo 'no llama or nvidia-smi process'
  nvidia-smi --query-gpu=index,pstate,clocks.sm,clocks.mem,memory.used,temperature.gpu,power.draw,power.limit \
    --format=csv,noheader
  grep -E 'MemAvailable|CmaFree|SwapFree' /proc/meminfo; cat /proc/pressure/memory; uptime; } > quiet.txt 2>&1

nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,clocks_event_reasons.active \
  --format=csv,noheader,nounits -lms "$SMI_MS" > smi.csv 2> smi.err &
SMI=$!
mark smi_start
sleep 2

mark idle_start; sleep "$IDLE_S"; mark idle_end

LOG=$OUT/rw_bench.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash "$WD" > /dev/null 2>&1 < /dev/null &
RW=$!
P4=$(printf '4,%.0s' $(seq "$K4_INST")); P4=${P4%,}
mark bench_start
{ { env $KENV $HOP timeout 1500 "$BIN/llama-bench" -m "$MODEL" -ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 \
      -lm dio -p "$P4" -n 128 -r "$REPS" -o jsonl --progress; echo $? > bench.rc; } | stamp > bench_out.tsv; } 2>&1 \
  | stamp > bench_err.tsv
mark bench_end
kill "$RW" 2> /dev/null
sleep 5

mark server_start
echo "env $KENV $HOP $SL $BIN/llama-server $COMMON $MT -m $MODEL --host 127.0.0.1 --port $PORT" > server_cmd.txt
env $KENV $HOP $SL setsid nohup "$BIN/llama-server" $COMMON $MT -m "$MODEL" --host 127.0.0.1 --port "$PORT" \
  > server.log 2>&1 < /dev/null &
SRV=$!
PID=$SRV LOG=$OUT/rw_server.tsv SERVER_LOG=$OUT/server.log MIN_AVAIL_MB=60 \
  setsid nohup chrt -f 10 bash "$WD" > /dev/null 2>&1 < /dev/null &
health=0
for _ in $(seq 450); do
  curl -sf "127.0.0.1:$PORT/health" > /dev/null 2>&1 && { health=1; break; }
  kill -0 "$SRV" 2> /dev/null || break
  sleep 2
done
curl_rc=none
if [ "$health" = 1 ]; then
  mark server_ready
  sleep "$SERVED_IDLE_S"
  mark request_sent
  curl -s -o reply.json -w '%{http_code} %{time_total}\n' -H 'Content-Type: application/json' \
    --data-binary @"$REQ" "127.0.0.1:$PORT/v1/chat/completions" > curl.txt
  curl_rc=$?
  mark reply_received
  sleep 3
fi
kill "$SRV" 2> /dev/null
for _ in $(seq 60); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
kill -9 "$SRV" 2> /dev/null
mark server_stopped
sleep 3
kill "$SMI" 2> /dev/null
mark smi_end
echo "bench_rc=$(cat bench.rc 2> /dev/null || echo none) health=$health curl_rc=$curl_rc" > DONE
