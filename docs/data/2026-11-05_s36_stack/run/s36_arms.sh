#!/usr/bin/env bash
# S36 (Q687 + Q660 on S35's line): the whole stack on the agent check with the draft length chosen per step
# (LLAMA_SPEC_ADAPT, C134) and the expert kernels skipping the experts the router's tail drops (LLAMA_MOE_TAIL, C120),
# on one rig. Every arm runs S35's s35stack line exactly: pmix-q2-fast, S33's MTP draft line with the draft head's
# shortlist, the k-column kernels' 32 KB budget with the dispatch table off, and the Q8_0 card hops. The arms add:
#   s36adapt  LLAMA_SPEC_ADAPT=1 with a T(k) cost table (COST; verify widths 1/3/4 at n-max 3), every step logged
#   s36skip   LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0 (hist=0: without it every micro-batch reads the kept experts back)
#   s36both   both
#   s36base   neither, on the same build (the control for s36skip and s36both)
#   s36ref    S35's s35stack again (run with BUILD = S35's -kshape build): the spread of repeated sessions
# Steps (STEPS, default "bench prep arms report"):
#   bench   llama-bench -p 0 -n 128 -r 3 on the no-draft line, arms ARMS_BENCH (hop: the Q8_0 hops; skip: + the skip;
#           nohop: neither), files bench-<BT><arm>.json in OUT
#   prep    the agent check's bench files and sandbox image on the client
#   arms    the agent check, the arms in ARMS (default "s36adapt s36skip s36both s36base")
#   report  the report of REF_ARM (default s35stack, already in CLIENT_OUT) and ARMS; the adaptive arms' step logs
#           checked and tabulated (adapt_log.py)
# Run from the repository root (it runs only ssh here). ENVFILE is S12's machine file
# (docs/data/2026-11-03_flash_next_mtp/run/s12.env.example, filled in, with DRAFT); set RIG_PORT, RELAY_PORT,
# CLIENT_PORT, PROXY_PORT, CLIENT_DIR and CLIENT_OUT after it (S36 used S35's client folders, so its report pairs the
# arms with s35stack); MODEL the file's first shard on the rig; BUILD the build's folder under /opt/llama-builds; COST
# the cost table's path on the rig; OUT a local folder.
set -uo pipefail
: "${ENVFILE:?}" "${MODEL:?}" "${BUILD:?}" "${OUT:?}"
M=$MODEL; T=$BUILD
set -a; . "$ENVFILE"; set +a
export MIN_AVAIL_MB=60 MODEL=$M REL_BIN=/opt/llama-builds/$T/bin/llama-server RIG_LOGDIR=${RIG_LOGDIR:-/tmp/s36/agent-logs}
STEPS=${STEPS:-bench prep arms report}
ARMS=${ARMS:-s36adapt s36skip s36both s36base}
ARMS_BENCH=${ARMS_BENCH:-hop skip}; BT=${BT:-}
REF_ARM=${REF_ARM:-s35stack}
has() { [[ " $STEPS " == *" $1 "* ]]; }
KENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0"
HOP="LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1"
SL="LLAMA_DRAFT_SHORTLIST=32768,8192,2"
SKIP="LLAMA_MOE_TAIL=p=0.9,renorm=1,hist=0"
adapt() { echo "LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_COST=${COST:?COST=<the cost table on the rig>} LLAMA_SPEC_ADAPT_LOG=adapt-$1.jsonl"; }
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
mkdir -p "$OUT"
if has bench; then
  ssh "$RIG" "mkdir -p /tmp/s36 && cat > /dev/shm/ram_watchdog.sh" < scripts/ram_watchdog.sh
  ssh "$RIG" "LOG=/dev/shm/rw_bench.tsv PROC=llama-bench MIN_AVAIL_MB=60 setsid nohup chrt -f 10 bash /dev/shm/ram_watchdog.sh > /dev/null 2>&1 < /dev/null & echo \$! > /dev/shm/rw_bench.pid
    cd /tmp/s36 && for arm in $ARMS_BENCH; do e='$KENV $HOP'; [ \$arm = skip ] && e=\"\$e $SKIP\"; [ \$arm = nohop ] && e='$KENV'; env \$e timeout 1800 /opt/llama-builds/$T/bin/llama-bench \
      -m $M -ngl 99 -fa 1 -b 2048 -ub 512 -ts 6/5/5/6/5/5/6/5/6 -lm dio -p 0 -n 128 -r 3 -o json > bench-$BT\$arm.json 2> bench-$BT\$arm.err; done
    kill \$(cat /dev/shm/rw_bench.pid); rm -f /dev/shm/rw_bench.* /dev/shm/ram_watchdog.sh"
  for arm in $ARMS_BENCH; do ssh "$RIG" "cat /tmp/s36/bench-$BT$arm.json" > "$OUT/bench-$BT$arm.json"; done
fi
if has prep; then
  ARMS=s36adapt STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1
fi
run_arm() {  # <arm> <REL_ENV> <args>
  env "ARGS_$1=$3" "BUDGET_MIN_$1=40" REL_ENV="$2" ARMS=$1 STEPS=run bash $RC "$OUT/agent-$1" > "$OUT/agent-$1.console" 2>&1
  echo "arm $1: exit $?"
}
if has arms; then
  for a in $ARMS; do
    case $a in
      s36adapt) run_arm s36adapt "$KENV $HOP $SL $(adapt s36adapt)"      "$MT" ;;
      s36skip)  run_arm s36skip  "$KENV $HOP $SL $SKIP"                  "$MT" ;;
      s36both)  run_arm s36both  "$KENV $HOP $SL $(adapt s36both) $SKIP" "$MT" ;;
      s36base)  run_arm s36base  "$KENV $HOP $SL"                        "$MT" ;;
      s36ref)   run_arm s36ref   "$KENV $HOP $SL"                        "$MT" ;;
    esac
  done
fi
if has report; then
  ARMS="$REF_ARM $ARMS" STEPS="report collect" bash $RC "$OUT/agent-report" > "$OUT/agent-report.console" 2>&1
  logs=""
  for a in $ARMS; do case $a in s36adapt|s36both) logs="$logs $CLIENT_OUT/rig/adapt-$a.jsonl" ;; esac; done
  if [ -n "$logs" ]; then
    AL=docs/data/2026-11-04_adaptive_draft/run/adapt_log.py
    ssh "$CLIENT" "cat > $CLIENT_DIR/adapt_log.py" < $AL
    ssh "$CLIENT" "python3 -I $CLIENT_DIR/adapt_log.py --check $logs" > "$OUT/adapt_check.txt" 2>&1
    ssh "$CLIENT" "python3 -I $CLIENT_DIR/adapt_log.py $logs" > "$OUT/adapt.tsv" 2>&1
  fi
fi
