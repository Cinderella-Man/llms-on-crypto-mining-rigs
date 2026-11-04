#!/usr/bin/env bash
# rig_blockver.sh OUT: Q797 and Q802 on the rig. S18's DFlash2 line (Q561: Swift-1.5 GSQ-RCO Q2_0 with Inco AI's
# DFlash2 drafter, -b 512, nine cards) on the agent check with probabilistic draft sampling (Q731), against block
# verification (LLAMA_SPEC_BLOCK_VERIFY=1, Q797) and against the tool-call grammar's forced tokens appended to the
# drafts (LLAMA_SPEC_FORCE=1, Q802), on one build (fork branch pascal-p106-v3-b12076-blockver). Runs only ssh here;
# the rig and the client are S18's (docs/data/2026-11-03_flash_next_dflash2/run/s18.env.example), REL_BIN is this
# build's llama-server on the rig. The agent check samples at temperature 1.0 (top-p 0.95, top-k 20), so every arm
# verifies its sampled drafts by rejection; greedy requests would not use any of the switches.
#
# Arms (all S18's dflash2b line plus --spec-draft-sampling probabilistic --spec-draft-p-min 0):
#   prob          --spec-draft-n-max 2: token-by-token rejection (Q731's arm, the reference for the two below)
#   block         the same with block verification
#   keep<N>k<K>   --spec-draft-n-max N with every draft cut to K drafted tokens (LLAMA_SPEC_DRAFT_KEEP=K) and block
#                 verification: the verify width of block outside tool calls on buffers sized for N (the recurrent
#                 snapshots and the outputs per step), the control for force<N>k<K>
#   force<N>k<K>  keep<N>k<K> plus LLAMA_SPEC_FORCE=1: inside a tool call the grammar's forced tokens fill the width up to
#                 N drafted (verify width up to N + 1)
#   Default ARMS="prob block keep4k2 force4k2". At --spec-draft-n-max 4 S18's smoke test met the RAM watchdog with
#   -b 2048 (smoke-n4); this line has -b 512. If the n-max-4 arms fail their smoke tests: ARMS="force3k2 keep3k2"
#   (STEPS="smoke arms report"), or force2k1 (forced tokens replace the second drafted token inside tool calls).
# Steps (STEPS, default "smoke arms report"):
#   smoke   per arm: one load, the agent's first-turn request and Q561's two smoke requests (S18's smoke_dflash.sh
#           with the arm's environment); an arm whose smoke test fails is listed in OUT/smoke_failed and left out
#   arms    per arm that passed: release_check_s12.sh with STEPS=run, ARMS=<arm> and the arm's REL_ENV, BUDGET_MIN
#           minutes each (default 40, as S18)
#   report  release_check_s12.sh's report and collect over REF_ARMS (default "nodraft dflash2b", Q226's and S18's runs
#           in the same client folder) and the arms run; OUT/forced.tsv, the per-task "forced tokens" lines of the
#           force arms' server logs (steps, appended, accepted), and OUT/accept.tsv, every arm's per-task draft
#           acceptance lines
# The host guard: when bench/agent_check/rig_server.sh has no RAM watchdog of its own and GUARD (S15's mem_guard.sh)
# is set, the guard runs on the rig around each timed arm, as S18's arm runner did (OUT/mem_guard_<arm>.log).
# Usage, from the repository root with S18's values filled in:
#   set -a; . s18.env; set +a
#   REL_BIN=/opt/llama-builds/<the blockver tarball>/bin/llama-server [ARMS="prob block keep4k2 force4k2"] \
#     [STEPS=...] [BUDGET_MIN=40] [SMOKE_LOGROOT=<a folder on the rig>] \
#     bash docs/data/2026-11-04_block_verify/run/rig_blockver.sh OUT
# Needs: no command on the rig while it runs (a timed task); the client's agent-check image (S18's prep step).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: rig_blockver.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${RIG:?}" "${CLIENT:?}" "${REL_BIN:?}" "${MODEL:?}" "${DFLASH:?}" "${CLIENT_DIR:?}" "${CLIENT_OUT:?}" "${RIG_LOGDIR:?}"
export BENCH=${BENCH:-$REPO/bench/agent_check}
ARMS=${ARMS:-prob block keep4k2 force4k2}
REF_ARMS=${REF_ARMS:-nodraft dflash2b}
STEPS=${STEPS:-smoke arms report}
BUDGET=${BUDGET_MIN:-40}
SMOKE_LOGROOT=${SMOKE_LOGROOT:-$(dirname "$RIG_LOGDIR")}
S18=$REPO/docs/data/2026-11-03_flash_next_dflash2
REL_CHECK=$REPO/docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
# S18's dflash2b line with probabilistic draft sampling; the arms change its --spec-draft-n-max
BASE_ARGS="-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host -md $DFLASH --spec-type draft-dflash \
--spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99 --spec-draft-sampling probabilistic --spec-draft-p-min 0"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

arm() {  # sets A_ARGS and A_ENV for arm $1
  local n=2
  case $1 in
    prob)  A_ENV="";;
    block) A_ENV="LLAMA_SPEC_BLOCK_VERIFY=1";;
    keep[0-9]*k[0-9]*)
      n=${1#keep}; n=${n%k*}
      A_ENV="LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_DRAFT_KEEP=${1##*k}";;
    force[0-9]*k[0-9]*)
      n=${1#force}; n=${n%k*}
      A_ENV="LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_FORCE=1 LLAMA_SPEC_DRAFT_KEEP=${1##*k}";;
    *) say "unknown arm $1 (prob, block, keep<N>k<K>, force<N>k<K>)"; return 1;;
  esac
  [[ $n =~ ^[0-9]+$ ]] || { say "unknown arm $1"; return 1; }
  A_ARGS=${BASE_ARGS/--spec-draft-n-max 2/--spec-draft-n-max $n}
}

say "rig_blockver: arms $ARMS, steps $STEPS, REL_BIN $REL_BIN"
# s18.env names S18's own build: the arms need one with the switches (ANY_BIN=1 to use another name)
[[ $REL_BIN == *blockver* ]] || [ "${ANY_BIN:-0}" = 1 ] || { say "REL_BIN is not a blockver build; stopping"; exit 1; }
touch "$OUT/smoke_failed"

if has smoke; then
  for a in $ARMS; do
    arm "$a" || exit 1
    say "smoke $a: ${A_ENV} ... ${A_ARGS}"
    ARGS="$A_ARGS" SMOKE_ENV="$A_ENV" SMOKE_LOGDIR="$SMOKE_LOGROOT/blockver-smoke-$a" \
      REQUESTS="$S18/run/smoke_request.json $S18/run/smoke_greedy.json" \
      bash "$S18/run/smoke_dflash.sh" "$OUT/smoke-$a" > "$OUT/smoke-$a.console" 2>&1
    rc=$?
    say "smoke $a: rc $rc; $(grep -h 'RAM watchdog\|did not come up' "$OUT/smoke-$a/smoke.log" | tail -n 1)"
    if [ $rc != 0 ]; then
      grep -qx "$a" "$OUT/smoke_failed" || echo "$a" >> "$OUT/smoke_failed"
    else
      sed -i "/^$a\$/d" "$OUT/smoke_failed"
    fi
  done
fi

RUN_ARMS=
for a in $ARMS; do
  grep -qx "$a" "$OUT/smoke_failed" || RUN_ARMS="$RUN_ARMS $a"
done
RUN_ARMS=${RUN_ARMS# }

guard_up() {
  [ -n "${GUARD:-}" ] && ! grep -q ram_watchdog "$BENCH/rig_server.sh" || return 0
  ssh "$RIG" "cat > /dev/shm/mem_guard.sh" < "$GUARD"
  ssh "$RIG" 'LOG=/dev/shm/mem_guard_blockver.log setsid nohup chrt -f 10 bash /dev/shm/mem_guard.sh > /dev/null 2>&1 \
    < /dev/null & sleep 1; echo "guard $(cat /dev/shm/mem_guard_blockver.log.pid)"' | tee -a "$OUT/run.log"
}
guard_down() {
  [ -n "${GUARD:-}" ] && ! grep -q ram_watchdog "$BENCH/rig_server.sh" || return 0
  ssh "$RIG" 'kill $(cat /dev/shm/mem_guard_blockver.log.pid); sleep 1; cat /dev/shm/mem_guard_blockver.log; \
    rm -f /dev/shm/mem_guard.sh /dev/shm/mem_guard_blockver.log /dev/shm/mem_guard_blockver.log.pid' > "$OUT/mem_guard_$1.log"
}

if has arms; then
  for a in $RUN_ARMS; do
    arm "$a" || exit 1
    say "arm $a: ${A_ENV} ... ${A_ARGS}"
    guard_up
    env "ARGS_$a=$A_ARGS" "BUDGET_MIN_$a=$BUDGET" REL_ENV="$A_ENV" STEPS=run ARMS="$a" \
      bash "$REL_CHECK" "$OUT/agent-$a" > "$OUT/agent-$a.console" 2>&1
    rc=$?
    guard_down "$a"
    say "arm $a: rc $rc; $(tail -n 1 "$OUT/agent-$a/run.log" 2> /dev/null)"
  done
fi

if has report; then
  envs=()
  for a in $RUN_ARMS; do
    arm "$a" && envs+=("ARGS_$a=$A_ARGS")
  done
  env "${envs[@]}" STEPS="report collect" ARMS="$REF_ARMS $RUN_ARMS" bash "$REL_CHECK" "$OUT/agent-report" \
    > "$OUT/agent-report.console" 2>&1
  say "report: rc $?; $OUT/agent-report"
  # the server logs' per-task lines (the collect step copies them into OUT/agent-report)
  printf 'arm\tsteps\tappended\taccepted\n' > "$OUT/forced.tsv"
  printf 'arm\tline\n' > "$OUT/accept.tsv"
  for a in $RUN_ARMS; do
    for f in $(find "$OUT/agent-report" "$OUT/agent-$a" -name "server-$a*.log" 2> /dev/null | sort -u); do
      awk -v arm="$a" '/forced tokens =/ { for (i = 1; i <= NF; i++) { if ($i == "=") s = $(i+1); if ($i == "appended") p = $(i-5); if ($i == "accepted") c = $(i-1) }
        printf "%s\t%s\t%s\t%s\n", arm, s, p, c }' "$f" >> "$OUT/forced.tsv"
      grep -h "draft acceptance =" "$f" | sed "s/^/$a\t/" >> "$OUT/accept.tsv"
    done
  done
  say "tables: $OUT/forced.tsv ($(($(wc -l < "$OUT/forced.tsv") - 1)) tasks), $OUT/accept.tsv"
fi
say "rig_blockver: done"
