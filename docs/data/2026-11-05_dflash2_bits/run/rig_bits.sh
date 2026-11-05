#!/usr/bin/env bash
# rig_bits.sh: C528 (Q1017) on the rig, client-free. One llama-server load per drafter on S18's DFlash2 line as C134r ran
# it (docs/data/2026-11-04_adaptive_draft/: Swift Q2_0, -ladder3, GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768, -b 512,
# --spec-draft-n-max 2, the drafter on the last card), the same fixed greedy requests sent from the rig itself to each.
# Runs on the rig, detached (setsid nohup), and needs nothing from another machine while it runs. Per arm, in OUT/<arm>/:
#   quiet.txt                 before the load: other llama processes, every card's memory, uptime
#   phases.txt                epoch seconds of the load's start and end and of each request's start and end
#   start.log, server-<arm>.log, ram_watchdog-<arm>.tsv, gpu.csv, meminfo.log, vmstat.log
#                             bench/agent_check/rig_server.sh's: the server and its monitors, the RAM watchdog
#                             (MemAvailable floor MIN_AVAIL_MB, the other limits its defaults)
#   vram_loaded.csv, vram_after.csv, host_loaded.txt, host_after_<request>.txt, metrics.txt
#   reply_<request>.json      each request's reply (timings carry draft_n and draft_n_accepted)
#   adapt.jsonl               LLAMA_SPEC_ADAPT_LOG with the switch off: one line per drafted step (n_draft, n_acc)
#   host_profile.csv          GGML_CUDA_PROFILE_HOST: per card and graph evaluation the host's entry and exit and the
#                             GPU time between two events (no synchronize), from which bits_tables.py takes the draft
#                             pass's time per step
# Then bits_tables.py writes OUT/tables/, and OUT/DONE marks the end (OUT/run.pid holds this script's pid).
# Environment: BIN (llama-server), MODEL (the main model's first shard), ARMS ("name=drafter.gguf ...", in load order),
# REQ (make_requests.py's folder), OUT, SERVER_SH (bench/agent_check/rig_server.sh on the rig), TABLES
# (bits_tables.py on the rig; empty skips the tables); optional PORT (8092), GAP_S (20: idle seconds before each
# request and after the last, which split the profile by request), MIN_AVAIL_MB (60), ARM_ENV
# (GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768), CTX (65536), WD_ENV (other watchdog limits, "VAR=value ...").
set -uo pipefail
: "${BIN:?}" "${MODEL:?}" "${ARMS:?}" "${REQ:?}" "${OUT:?}" "${SERVER_SH:?}"
PORT=${PORT:-8092}
GAP_S=${GAP_S:-20}
export MIN_AVAIL_MB=${MIN_AVAIL_MB:-60}
ARM_ENV=${ARM_ENV:-GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768}
CTX=${CTX:-65536}
COMMON="-ngl 99 -fa on -c $CTX -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek \
--temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui"
LINE="-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host --spec-type draft-dflash --spec-draft-n-max 2 \
--spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99"
mkdir -p "$OUT"
echo $$ > "$OUT/run.pid"
rm -f "$OUT/DONE"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
mark() { echo "$1 $(date +%s.%N)" >> "$L/phases.txt"; }
field() { local v; v=$(grep -o "\"$1\":[0-9.]*" "$2" | head -n 1); echo "${v#*:}"; }
srv() {  # srv ACTION [VAR=value ...]: rig_server.sh with this arm's folder and port
  local action=$1; shift
  # shellcheck disable=SC2086
  env LOGDIR="$L" PORT="$PORT" ACTION="$action" ${WD_ENV:-} "$@" bash "$SERVER_SH" >> "$L/start.log" 2>&1
}

say "rig_bits: arms $ARMS; requests $(cd "$REQ" && ls *.json | tr '\n' ' ')"
for spec in $ARMS; do
  arm=${spec%%=*} draft=${spec#*=}
  L=$OUT/$arm
  mkdir -p "$L"
  {
    echo "# $(date -Is)"
    uptime
    ps -eo comm | grep -x -e llama-server -e llama-cli -e llama-bench -e llama-completion || echo "no llama process"
    nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
  } > "$L/quiet.txt" 2>&1
  if grep -qx -e llama-server -e llama-cli -e llama-bench -e llama-completion "$L/quiet.txt"; then
    say "$arm: another llama process runs; stopping"
    break
  fi
  say "$arm: loading $draft"
  mark load_start
  srv start ARM="$arm" BIN="$BIN" MODEL="$MODEL" \
    SERVER_ENV="$ARM_ENV LLAMA_SPEC_ADAPT_LOG=$L/adapt.jsonl GGML_CUDA_PROFILE_HOST=$L/host_profile.csv" \
    SERVER_ARGS="$COMMON $LINE -md $draft"
  up=$?
  mark load_end
  say "$arm: $(grep -h 'up after\|did not come up\|exited while loading\|load timeout' "$L/start.log" | tail -n 1) (rc $up)"
  if [ "$up" = 0 ]; then
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$L/vram_loaded.csv"
    { cat /proc/meminfo; cat "/proc/$(cat "$L/server.pid")/status"; } > "$L/host_loaded.txt" 2>&1
    for r in "$REQ"/*.json; do
      b=$(basename "$r" .json) name=${b%.*} ep=${b##*.}
      case $ep in chat) path=/v1/chat/completions ;; completion) path=/completion ;; *) continue ;; esac
      sleep "$GAP_S"
      mark "req_start $name"
      curl -s -m 1800 "127.0.0.1:$PORT$path" -H 'Content-Type: application/json' -d @"$r" > "$L/reply_$name.json"
      mark "req_end $name"
      say "$arm $name: prompt $(field prompt_n "$L/reply_$name.json"), generated $(field predicted_n "$L/reply_$name.json")" \
        "at $(field predicted_per_second "$L/reply_$name.json") tok/s, drafted $(field draft_n "$L/reply_$name.json")," \
        "accepted $(field draft_n_accepted "$L/reply_$name.json")"
      { cat /proc/meminfo; cat "/proc/$(cat "$L/server.pid")/status"; } > "$L/host_after_$name.txt" 2>&1
    done
    sleep "$GAP_S"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader > "$L/vram_after.csv"
    curl -s "127.0.0.1:$PORT/metrics" > "$L/metrics.txt"
  fi
  srv end
  mark end
  say "$arm: stopped; RAM watchdog $(grep -vc '^#' "$L/ram_watchdog-$arm.tsv" 2> /dev/null) samples," \
    "$(grep -c '^# [0-9]* KILL ' "$L/ram_watchdog-$arm.tsv" 2> /dev/null) kills"
  sleep 10
done
if [ -n "${TABLES:-}" ]; then
  python3 -I "$TABLES" --gap-s "$GAP_S" "$OUT" > "$OUT/tables.log" 2>&1
  say "tables: rc $?; $OUT/tables"
fi
say "rig_bits: done"
touch "$OUT/DONE"
