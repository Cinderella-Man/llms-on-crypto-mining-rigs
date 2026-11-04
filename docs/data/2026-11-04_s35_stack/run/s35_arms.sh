#!/usr/bin/env bash
# S35 (Q899's next step): the whole stack on the agent check, on one rig. pascal-mix's pmix-q2-fast on Miner Llama's
# -kshape build (39c3ab5f1: Q823-Q825 and Q916 on -ladder3) with the k-column kernels' 32 KB budget and the dispatch
# table off (GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0, C350's fastest setting), S33's
# mtp2d draft line with the draft head's shortlist (S34's best arm), and the residual in Q8_0 across the card boundaries
# through the staged hop (LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1: C200k's KLD gate passed). RAM watchdog floor 60 MiB.
#   fit    scripts/fit_probe.sh for the MTP draft line (seconds)
#   bench  llama-bench -p 0 -n 128, three repetitions, on the no-draft line, without and with the hop cast
#   prep   the agent check's bench files and sandbox image on the client (its own folder, so it runs beside other rigs')
#   arms   the agent check, one arm: s35stack (and s35nohop, the same without the hop cast, with ARMS_S35)
#   report the agent check's report of the arm(s), the client's folder collected to OUT
# Run from the repository root (it runs only ssh here). ENVFILE is S12's machine file
# (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in, with DRAFT); set RIG_PORT, RELAY_PORT, CLIENT_PORT,
# PROXY_PORT, CLIENT_DIR and CLIENT_OUT after it to keep this run apart from another rig's; MODEL the file's first shard
# on the rig; BUILD the -kshape build's folder under /opt/llama-builds; OUT a local folder; STEPS picks steps.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
set -a; . "$ENVFILE"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-/tmp/s35/agent-logs}
STEPS=${STEPS:-fit bench prep arms report}
has() { [[ " $STEPS " == *" $1 "* ]]; }
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
if has fit; then
  RIG=$RIG MODEL=$M BUILDS=/opt/llama-builds OUT=$OUT/fit_probe.md KEEP=400 \
    CONFIGS="mtp|$T|65536|$(echo "$MT" | sed 's/ -fit off -lm dio//')" bash scripts/fit_probe.sh > /dev/null
fi
if has bench; then
  ssh "$RIG" "mkdir -p /tmp/s35 && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
  ssh "$RIG" "LOG=/dev/shm/rw_bench.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw_bench.pid
    cd /tmp/s35 && for hop in off on; do e='$KENV'; [ \$hop = on ] && e=\"\$e $HOP\"; env \$e timeout 1800 /opt/llama-builds/$T/bin/llama-bench -m $M -ngl 99 -fa 1 \
      -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 -lm dio -p 0 -n 128 -r 3 -o json > bench-\$hop.json 2> bench-\$hop.err; done
    kill \$(cat /dev/shm/rw_bench.pid); rm -f /dev/shm/rw_bench.* /dev/shm/ram_watchdog.sh"
  for hop in off on; do ssh "$RIG" "cat /tmp/s35/bench-$hop.json" > "$OUT/bench-$hop.json"; done
fi
if has prep; then
  ARMS=s35stack STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
fi
run_arm() {  # <arm> <REL_ENV> <args>
  env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
  echo "arm $1: exit $?"
}
if has arms; then
  for a in ${ARMS_S35:-s35stack}; do
    case $a in
      s35stack) run_arm s35stack "$KENV $HOP $SL" "$MT" ;;
      s35nohop) run_arm s35nohop "$KENV $SL"      "$MT" ;;
    esac
  done
fi
if has report; then
  ARMS="${ARMS_S35:-s35stack}" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
fi
