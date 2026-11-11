#!/usr/bin/env bash
# R90 (Q1322) on one rig: the v4 day build with every switch the rigs proved a gain on 2026-11-11 (gains), the same build
# with no switch (off), and the v3.0 release build (v3), all three on Flash-Next's v3 preset
# (release/flash-next-v3/settings.env) with Swift Q2_0 and the MTP draft. Run from the repository root on a machine
# that only runs ssh. The steps run in this order, whatever STEPS lists:
#   lane    bench/deep_lane at depths 0, 8,192, 16,384, 32,768 and 49,152 with the sampler pass, as R70 ran it
#           (docs/data/2026-11-10_v4_candidate), the loads in LANE_LOADS (v3 and gains first, off last), each arm on its
#           own build (deep_lane.sh's ARM_BIN_<arm>); the console's load lines are copied here while it runs, and the
#           results collected to OUT/lane
#   prep    the agent check's bench files and sandbox image on the client, the bugfix tasks validated
#   agent   the agent check (bench/agent_check, 12 bugfix tasks, a 12-minute cap a task, a 40-minute budget an arm), one
#           arm per build in AGENT_ORDER, each on the server line of v3.0's arm t1swift
#           (docs/data/2026-11-05_release_flashnext/run/rig_t1.sh), gains with its switches added; it starts only when
#           the clock is not past AGENT_BY, so that both arms end before the rig's evening stop
#   report  the agent check's report of both arms; the client's folder collected to OUT/agent
# Arms:
#   v3      REF with the preset as shipped
#   off     NEW with the preset as shipped
#   gains   NEW with the preset, GAINS_ENV and GAINS_ARGS: R70's candidate (the sparse Pascal attention, probabilistic
#           draft verification, four checkpoints on the cards, the micro-batch taper, the draft's micro-batch of 512) and
#           the six switches proved on 2026-11-11 (C722, C690, C691, C704, C100, C217)
# Env: RIG, CLIENT (ssh aliases), MODEL (Swift Q2_0's first shard on the rig), DRAFT (the MTP draft on the rig), NEW, REF
#   (build folders under /opt/llama-builds), TEXT_LOCAL (the recorded agent text here: the release check's
#   proxy.jsonl.bodies.gz), RIGDIR (a folder on the rig for the agent check's logs), OUT (a local folder), STEPS ("lane
#   prep agent report"), LANE_WORK (/tmp/deep-lane on the rig), LANE_LOADS, AGENT_ORDER ("gains v3"), AGENT_BY (19:45),
#   and release_check_s12.sh's RIG_PORT RELAY_PORT CLIENT_PORT PROXY_PORT CLIENT_DIR CLIENT_OUT CPU_SHARES (256).
set -uo pipefail
: "${RIG:?}" "${CLIENT:?}" "${MODEL:?}" "${DRAFT:?}" "${NEW:?}" "${REF:?}" "${OUT:?}"
STEPS=${STEPS:-lane prep agent report}
LANE_WORK=${LANE_WORK:-/tmp/deep-lane}
LANE_LOADS=${LANE_LOADS:-warm:v3 v31:v3 gains1:gains v32:v3 gains2:gains off1:off off2:off}
AGENT_ORDER=${AGENT_ORDER:-gains v3}
AGENT_BY=${AGENT_BY:-19:45}
GAINS_ENV=${GAINS_ENV:-GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1}
GAINS_ARGS=${GAINS_ARGS:---spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper on}
has() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "$(date +%H:%M:%S) $*"; }
# shellcheck disable=SC1091
. release/flash-next-v3/settings.env  # PRESET_ENV, PRESET_ARGS
mkdir -p "$OUT"

if has lane; then
  : "${TEXT_LOCAL:?}"
  lane_env=(RIG="$RIG" WORK="$LANE_WORK" BIN="/opt/llama-builds/$NEW/bin/llama-server"
    ARM_BIN_v3="/opt/llama-builds/$REF/bin/llama-server" MODEL="$MODEL" MODEL_DRAFT="$DRAFT" DRAFT=on TEXT_LOCAL="$TEXT_LOCAL"
    TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 ARMS="v3 gains off" LOADS="$LANE_LOADS"
    ARM_ENV_gains="$GAINS_ENV" ARM_ARGS_gains="$GAINS_ARGS --spec-draft-n-ubatch 512")
  say "lane: dry run"
  env "${lane_env[@]}" DRY=1 bash bench/deep_lane/remote.sh start > "$OUT/lane-dry.console" 2>&1
  cat "$OUT/lane-dry.console"
  grep -q 'DRY=1: nothing started' "$OUT/lane-dry.console" || { say "lane: the dry run failed"; exit 1; }
  # deep_lane.sh's own DRAFT is on|off, this script's the draft file: every arm's line must keep the draft
  [ "$(grep -c -- " -md $DRAFT" "$OUT/lane-dry.console")" = 3 ] || { say "lane: an arm's line has no draft"; exit 1; }
  say "lane: start"
  env "${lane_env[@]}" bash bench/deep_lane/remote.sh start || { say "lane: start failed"; exit 1; }
  # one ssh call that follows the console until the driver exits (its pid file is written after the arms' checks);
  # then remote.sh wait in case that call dropped
  ssh "$RIG" "for _ in \$(seq 120); do [ -s $LANE_WORK/out/driver.pid ] || [ -e $LANE_WORK/out/failed ] && break; sleep 1; done; \
    p=\$(cat $LANE_WORK/out/driver.pid 2> /dev/null) && tail -n +1 --pid=\$p -F $LANE_WORK/console.txt 2> /dev/null" \
    | grep --line-buffered -E 'LOAD|DEEP_|watchdog|not idle'
  RIG=$RIG WORK=$LANE_WORK bash bench/deep_lane/remote.sh wait
  RIG=$RIG WORK=$LANE_WORK OUT="$OUT/lane" bash bench/deep_lane/remote.sh collect
  [ -e "$OUT/lane/done" ] || { say "lane: no done marker"; exit 1; }
  say "lane: done"
fi

# rig_t1.sh's MT: the preset's server line after its common part (release_check_s12.sh's COMMON adds that)
MT="-b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
MT_GAINS="${MT/--spec-draft-n-ubatch 128/--spec-draft-n-ubatch 512} $GAINS_ARGS"
[[ $MT_GAINS == *"--spec-draft-n-ubatch 512"* ]] || { say "the draft's micro-batch was not replaced"; exit 1; }
export BENCH=$PWD/bench/agent_check RIG CLIENT MODEL CTX=65536 TIME_CAP_MIN=12 MIN_AVAIL_MB=60 \
  CPU_SHARES=${CPU_SHARES:-256} RIG_LOGDIR=${RIGDIR:-/tmp/r90}/agent-logs
RC=docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh

if has prep; then
  say "prep"
  ARMS=gains STEPS=prep bash $RC "$OUT/agent-prep" > "$OUT/agent-prep.console" 2>&1 || { say "prep failed"; exit 1; }
  say "prep: $(tail -n 1 "$OUT/agent-prep.console")"
fi

if has agent; then
  if [ "$(date +%H%M)" -gt "${AGENT_BY/:/}" ]; then
    say "AGENT_SKIPPED: past $AGENT_BY, both arms would not end before the rig's evening stop"
  else
    for a in $AGENT_ORDER; do
      case $a in
        gains) b=$NEW; e="$PRESET_ENV $GAINS_ENV"; args=$MT_GAINS ;;
        v3) b=$REF; e=$PRESET_ENV; args=$MT ;;
        *) say "unknown arm $a"; exit 1 ;;
      esac
      say "arm $a: $b"
      env REL_BIN="/opt/llama-builds/$b/bin/llama-server" "ARGS_$a=$args" "BUDGET_MIN_$a=40" REL_ENV="$e" \
        ARMS="$a" STEPS=run bash $RC "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
      say "arm $a: exit $?"
    done
    if has report; then
      say "report"
      ARMS="v3 gains" STEPS="report collect" bash $RC "$OUT/agent" > "$OUT/agent-report.console" 2>&1
      say "report: exit $?"
    fi
  fi
fi

ssh "$RIG" "rmdir ${RIGDIR:-/tmp/r90} 2> /dev/null; date -Is; ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-cli' || echo 'no llama process'; \
  nvidia-smi --query-compute-apps=pid,name --format=csv,noheader; \
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr '\n' ' '; echo" > "$OUT/rig_end.txt" 2>&1
say "R90_DONE: $(tr '\n' ' ' < "$OUT/rig_end.txt")"
