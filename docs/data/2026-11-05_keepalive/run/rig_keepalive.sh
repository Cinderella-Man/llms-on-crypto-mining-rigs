#!/usr/bin/env bash
# C533 (Q1021) on one rig: GGML_CUDA_KEEPALIVE on S35's line (pmix-q2-fast, the k-column settings, the Q8_0 hop cast
# through the staged hop, the MTP draft with the head's shortlist; docs/data/2026-11-04_s35_stack) with Miner Llama's
# -keepalive build. Run from the repository root on a machine that only runs ssh; it takes S35's machine file and ports.
#   smoke  llama-bench -p 0 -n 16 -v once with GGML_CUDA_KEEPALIVE=decode: the blocks start and stop, and the log has a
#          line per card with the SM clock its block measured (ggml_cuda_keepalive_log)
#   tk     llama-bench on S35's no-draft line, -p 1,2,3,4,5 -n 128, five repetitions after a warm-up: k-token passes
#          (T(k), k = 1-5) and plain decoding, the switch off and on alternated over LOADS loads each (default 2),
#          under the RAM watchdog; the -v log of each "on" load keeps the per-card clock lines
#   prep   the agent check's bench files and sandbox image on the client
#   arms   the agent check, one arm a setting: s35ka (S35's s35stack with GGML_CUDA_KEEPALIVE=decode); ARMS_KA may add
#          s35kareq (=request: also every card's block when a request arrives, C558) and s35kaidle (=idle: the blocks
#          kept through the agent's tool calls, C558)
#   report the agent check's report of the arms, the client's folder collected to OUT
# Env: ENVFILE (S12's machine file, docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in, with DRAFT),
#      RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT (as s35_arms.sh), MODEL (the first shard on the
#      rig), BUILD (the -keepalive build's folder under /opt/llama-builds), OUT (a local folder), STEPS, LOADS, ARMS_KA
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
# the caller's rig, ports and folders win over the machine file's
keep=$(for v in RIG RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT RIG_LOGDIR; do
           [ -n "${!v:-}" ] && printf '%s=%q ' "$v" "${!v}"; done)
set -a; . "$ENVFILE"; eval "$keep"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-/tmp/c533-agent-logs}
STEPS=${STEPS:-smoke tk prep arms report}
LOADS=${LOADS:-2}
ARMS_KA=${ARMS_KA:-s35ka}
has() { [[ " $STEPS " == *" $1 "* ]]; }
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
BENCH="-ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 -lm dio"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
# <name> <keep-alive setting or ""> <llama-bench options>: one llama-bench load on the rig under the RAM watchdog
bench_load() {
    ssh "$RIG" "mkdir -p /tmp/c533 && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
    ssh "$RIG" "LOG=/dev/shm/rw_$1.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw.pid
        cd /tmp/c533 && env $KENV $HOP ${2:+GGML_CUDA_KEEPALIVE=$2} timeout 2400 /opt/llama-builds/$T/bin/llama-bench -m $M $BENCH $3 \
            -o jsonl > $1.jsonl 2> $1.err; echo \"$1 rc=\$?\"
        kill \$(cat /dev/shm/rw.pid); mv /dev/shm/rw_$1.tsv /tmp/c533/; rm -f /dev/shm/rw.pid /dev/shm/ram_watchdog.sh"
    for f in "$1.jsonl" "$1.err" "rw_$1.tsv"; do ssh "$RIG" "cat /tmp/c533/$f" > "$OUT/$f"; done
}
if has smoke; then
    bench_load smoke decode "-p 0 -n 16 -r 1 -v"
    grep -a 'ggml_cuda_keepalive' "$OUT/smoke.err" || echo "smoke: no keep-alive line in the log"
fi
if has tk; then
    for i in $(seq "$LOADS"); do
        bench_load "tk-off-$i" "" "-p 1,2,3,4,5 -n 128 -r 5"
        bench_load "tk-on-$i" decode "-p 1,2,3,4,5 -n 128 -r 5 -v"
    done
fi
if has prep; then
    ARMS=s35ka STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
fi
run_arm() {  # <arm> <REL_ENV> <args>
    env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
    echo "arm $1: exit $?"
}
if has arms; then
    for a in $ARMS_KA; do
        case $a in
            s35ka)     run_arm s35ka     "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=decode"  "$MT" ;;
            s35kareq)  run_arm s35kareq  "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=request" "$MT" ;;
            s35kaidle) run_arm s35kaidle "$KENV $HOP $SL GGML_CUDA_KEEPALIVE=idle"    "$MT" ;;
        esac
    done
fi
if has report; then
    ARMS="$ARMS_KA" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
fi
if has smoke || has tk; then
    ssh "$RIG" "rm -rf /tmp/c533"
fi
