#!/usr/bin/env bash
# rig_adraft.sh OUT: Q687 on the rig. S18's DFlash2 line (Q561: Swift-1.5 GSQ-RCO Q2_0 with Inco AI's DFlash2
# drafter, -b 512, nine cards) on the agent check, with the draft length chosen per step (LLAMA_SPEC_ADAPT=1, fork
# branch pascal-p106-v3-b12076-adraft) against fixed draft lengths on the same build. Runs only ssh here; the rig and
# the client are S18's (docs/data/2026-11-03_flash_next_dflash2/run/s18.env.example), REL_BIN is this build's
# llama-server on the rig.
#
# Arms (names fix<N> and adapt<N>; N = --spec-draft-n-max; every arm writes LLAMA_SPEC_ADAPT_LOG into its rig log
# folder, so the fixed arms also record what the rule would have kept):
#   fix<N>    S18's dflash2b line at --spec-draft-n-max N, the switch off
#   adapt<N>  the same with LLAMA_SPEC_ADAPT=1 and ADAPT_ENV (default none: Q583's cost table, verify widths 1, 3, 5, 9
#             and 17 clamped to N + 1, the running speed-up); at N = 5 the widths are 1, 3, 5 and 6
# Steps (STEPS, default "smoke arms report"):
#   smoke   per arm: one load, the agent's first-turn request and Q561's two smoke requests (S18's smoke_dflash.sh
#           with the arm's environment, SMOKE_ENV); an arm whose smoke test fails (the watchdog, no reply) is listed in
#           OUT/smoke_failed and left out of the timed arms
#   arms    per arm that passed: release_check_s12.sh with STEPS=run, ARMS=<arm> and the arm's REL_ENV, BUDGET_MIN
#           minutes each (default 40, as S18)
#   report  release_check_s12.sh's report and collect over REF_ARMS (default "nodraft dflash2b", Q226's and S18's
#           runs in the same client folder) and the arms run, then adapt_log.py on every arm's log on the client:
#           OUT/adapt.tsv (one row per arm and run), OUT/adapt_check.txt, and the smoke logs' rows in
#           OUT/smoke-<arm>/adapt.tsv
# The host guard: when bench/agent_check/rig_server.sh has no RAM watchdog of its own and GUARD (S15's mem_guard.sh)
# is set, the guard runs on the rig around each timed arm, as S18's arm runner did (OUT/mem_guard_<arm>.log).
# Usage, from the repository root with S18's values filled in:
#   set -a; . s18.env; set +a
#   REL_BIN=/opt/llama-builds/<the adraft tarball>/bin/llama-server [ARMS="fix2 fix3 fix5 adapt5"] [ADAPT_ENV=...] \
#     [STEPS=...] [BUDGET_MIN=40] [SMOKE_LOGROOT=<a folder on the rig>] \
#     bash docs/data/2026-11-04_adaptive_draft/run/rig_adraft.sh OUT
# Needs: no command on the rig while it runs (a timed task); the client's agent-check image (S18's prep step).
set -uo pipefail
OUT=$(realpath -m "${1:?usage: rig_adraft.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
: "${RIG:?}" "${CLIENT:?}" "${REL_BIN:?}" "${MODEL:?}" "${DFLASH:?}" "${CLIENT_DIR:?}" "${CLIENT_OUT:?}" "${RIG_LOGDIR:?}"
export BENCH=${BENCH:-$REPO/bench/agent_check}
ARMS=${ARMS:-fix2 fix3 fix5 adapt5}
REF_ARMS=${REF_ARMS:-nodraft dflash2b}
STEPS=${STEPS:-smoke arms report}
ADAPT_ENV=${ADAPT_ENV:-}
BUDGET=${BUDGET_MIN:-40}
SMOKE_LOGROOT=${SMOKE_LOGROOT:-$(dirname "$RIG_LOGDIR")}
S18=$REPO/docs/data/2026-11-03_flash_next_dflash2
REL_CHECK=$REPO/docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh
# S18's dflash2b line; the arms change its --spec-draft-n-max
BASE_ARGS="-b 512 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host -md $DFLASH --spec-type draft-dflash \
--spec-draft-n-max 2 --spec-draft-n-ubatch 64 -devd CUDA8 -ngld 99"
mkdir -p "$OUT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/run.log"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }

arm() {  # sets A_ARGS and A_ENV for arm $1
  local n
  case $1 in
    fix[0-9]*)   n=${1#fix};   A_ENV="LLAMA_SPEC_ADAPT_LOG=adapt-$1.jsonl";;
    adapt[0-9]*) n=${1#adapt}; A_ENV="LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_LOG=adapt-$1.jsonl${ADAPT_ENV:+ $ADAPT_ENV}";;
    *) say "unknown arm $1 (fix<N> or adapt<N>)"; return 1;;
  esac
  [[ $n =~ ^[0-9]+$ ]] || { say "unknown arm $1"; return 1; }
  A_ARGS=${BASE_ARGS/--spec-draft-n-max 2/--spec-draft-n-max $n}
}

say "rig_adraft: arms $ARMS, steps $STEPS, REL_BIN $REL_BIN${ADAPT_ENV:+, ADAPT_ENV $ADAPT_ENV}"
# s18.env names S18's own build: the arms need one with LLAMA_SPEC_ADAPT (ANY_BIN=1 to use another name)
[[ $REL_BIN == *adraft* ]] || [ "${ANY_BIN:-0}" = 1 ] || { say "REL_BIN is not an adraft build; stopping"; exit 1; }
touch "$OUT/smoke_failed"

if has smoke; then
  for a in $ARMS; do
    arm "$a" || exit 1
    say "smoke $a: ${A_ENV} ... ${A_ARGS}"
    ARGS="$A_ARGS" SMOKE_ENV="$A_ENV" SMOKE_LOGDIR="$SMOKE_LOGROOT/adraft-smoke-$a" \
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

# S15's memory guard around a timed arm (GUARD, as S18's arm runner did), when rig_server.sh has no RAM watchdog of
# its own
guard_up() {
  [ -n "${GUARD:-}" ] && ! grep -q ram_watchdog "$BENCH/rig_server.sh" || return 0
  ssh "$RIG" "cat > /dev/shm/mem_guard.sh" < "$GUARD"
  ssh "$RIG" 'LOG=/dev/shm/mem_guard_adraft.log setsid nohup chrt -f 10 bash /dev/shm/mem_guard.sh > /dev/null 2>&1 \
    < /dev/null & sleep 1; echo "guard $(cat /dev/shm/mem_guard_adraft.log.pid)"' | tee -a "$OUT/run.log"
}
guard_down() {
  [ -n "${GUARD:-}" ] && ! grep -q ram_watchdog "$BENCH/rig_server.sh" || return 0
  ssh "$RIG" 'kill $(cat /dev/shm/mem_guard_adraft.log.pid); sleep 1; cat /dev/shm/mem_guard_adraft.log; \
    rm -f /dev/shm/mem_guard.sh /dev/shm/mem_guard_adraft.log /dev/shm/mem_guard_adraft.log.pid' > "$OUT/mem_guard_$1.log"
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
  # the report reads every arm's ARGS_<arm> only to label it; the timed arms' are set here
  envs=()
  for a in $RUN_ARMS; do
    arm "$a" && envs+=("ARGS_$a=$A_ARGS")
  done
  env "${envs[@]}" STEPS="report collect" ARMS="$REF_ARMS $RUN_ARMS" bash "$REL_CHECK" "$OUT/agent-report" \
    > "$OUT/agent-report.console" 2>&1
  say "report: rc $?; $OUT/agent-report"
  # adapt_log.py on the client, where the arms' logs are (CLIENT_OUT/rig/adapt-<arm>.jsonl)
  ssh "$CLIENT" "mkdir -p $CLIENT_DIR && cat > $CLIENT_DIR/adapt_log.py" < "$HERE/adapt_log.py"
  logs=
  for a in $RUN_ARMS; do logs="$logs $CLIENT_OUT/rig/adapt-$a.jsonl"; done
  if [ -n "$logs" ]; then
    ssh "$CLIENT" "python3 -I $CLIENT_DIR/adapt_log.py --check $logs" > "$OUT/adapt_check.txt" 2>&1
    say "adapt logs: $(tail -n 1 "$OUT/adapt_check.txt")"
    ssh "$CLIENT" "python3 -I $CLIENT_DIR/adapt_log.py $logs" > "$OUT/adapt.tsv" 2>&1
  fi
  for a in $ARMS; do
    f=$(ls "$OUT/smoke-$a"/adapt-*.jsonl 2> /dev/null | head -n 1)
    [ -n "$f" ] && ssh "$CLIENT" "python3 -I $CLIENT_DIR/adapt_log.py --check /dev/stdin" < "$f" > "$OUT/smoke-$a/adapt.tsv" 2>&1
  done
  say "adapt tables: $OUT/adapt.tsv, $OUT/smoke-*/adapt.tsv"
fi
say "rig_adraft: done"
