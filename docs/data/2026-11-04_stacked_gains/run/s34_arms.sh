#!/usr/bin/env bash
# S34 (Q899): the day's gains stacked on the agent check, on one rig. pascal-mix's pmix-q2-fast on Miner Llama's
# -ladder3 build with GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 on every load; the RAM watchdog's MemAvailable floor at 60 MiB.
#   fit    scripts/fit_probe.sh for the no-draft line and the MTP draft line (seconds each)
#   bench  llama-bench -p 0 -n 128, three repetitions, on the no-draft line (under scripts/ram_watchdog.sh, PROC=llama-bench)
#   arms   the agent check, one arm each (docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh, S12's client folder):
#          s34nd (S12's no-draft line), s34mtp (S33's mtp2d line: S15's MTP draft line with -otd token_embd=CUDA4
#          --spec-draft-n-ubatch 128 --no-host), s34mtps (the same + LLAMA_DRAFT_SHORTLIST=32768,8192,2)
#   report the agent check's report of nodraft, mtp2d, s34nd, s34mtp, s34mtps, the client's folder collected to OUT
# Run from the repository root (it runs only ssh here). ENVFILE is S12's machine file
# (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in, with DRAFT); MODEL the file's first shard on the
# rig; BUILD the -ladder3 build's folder under /opt/llama-builds; OUT a local folder; STEPS picks steps.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
set -a; . "$ENVFILE"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-/tmp/s34/agent-logs}
STEPS=${STEPS:-fit bench arms report}
has() { [[ " $STEPS " == *" $1 "* ]]; }
SMEM=GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768
ND="-b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -fit off -lm dio"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
if has fit; then
  RIG=$RIG MODEL=$M BUILDS=/opt/llama-builds OUT=$OUT/fit_probe.md KEEP=400 \
    CONFIGS="nodraft|$T|65536|${ND% -fit off -lm dio};mtp|$T|65536|$(echo "$MT" | sed 's/ -fit off -lm dio//')" \
    bash scripts/fit_probe.sh > /dev/null
fi
if has bench; then
  ssh "$RIG" "mkdir -p /tmp/s34 && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
  ssh "$RIG" "LOG=/dev/shm/rw_bench.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw_bench.pid
    cd /tmp/s34 && $SMEM timeout 1800 /opt/llama-builds/$T/bin/llama-bench -m $M -ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 \
      -lm dio -p 0 -n 128 -r 3 -o json > bench.json 2> bench.err
    kill \$(cat /dev/shm/rw_bench.pid); rm -f /dev/shm/rw_bench.* /dev/shm/ram_watchdog.sh"
  ssh "$RIG" "cat /tmp/s34/bench.json" > "$OUT/bench.json"
fi
run_arm() {  # <arm> <REL_ENV> <args>
  env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
  echo "arm $1: exit $?"
}
if has arms; then
  run_arm s34nd   "$SMEM"                                     "$ND"
  run_arm s34mtp  "$SMEM"                                     "$MT"
  run_arm s34mtps "$SMEM LLAMA_DRAFT_SHORTLIST=32768,8192,2"  "$MT"
fi
if has report; then
  ARMS="nodraft mtp2d s34nd s34mtp s34mtps" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
fi
ssh "$RIG" 'rm -rf /tmp/s34'
