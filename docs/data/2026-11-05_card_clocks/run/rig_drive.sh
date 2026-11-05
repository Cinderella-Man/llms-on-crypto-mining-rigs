#!/usr/bin/env bash
# rig_drive.sh (C532): runs clocks_run.sh on a rig from a machine that only runs ssh, in steps (STEPS, default all):
#   request  the agent check's first-turn request, built on the rig itself (bench/agent_check/first_turn_request.sh with
#            BUILD_SSH="ssh $RIG"; LONG_MAX_TOKENS, default 1024, caps the reply) into OUT/request.json; not to be committed
#   send     clocks_run.sh, clock_phases.py, scripts/ram_watchdog.sh and the request into RD on the rig
#   start    clocks_run.sh detached on the rig (its pid in RD/run.pid)
#   wait     one ssh session that sleeps on the rig until clocks_run.sh exits (tail --pid, checked every 30 s), then DONE
#   tables   clock_phases.py on the rig
#   collect  RD's files to OUT
#   clean    RD removed on the rig; any llama or nvidia-smi process left is named
# Run from the repository root. Environment: RIG (the rig's ssh alias), BIN, MODEL, DRAFT (paths on the rig), OUT (a
# local folder), RD (the rig's work folder, default /dev/shm/c532).
set -uo pipefail
: "${RIG:?}" "${OUT:?}"
RD=${RD:-/dev/shm/c532}
STEPS=${STEPS:-request send start wait tables collect clean}
HERE=docs/data/2026-11-05_card_clocks/run
has() { [[ " $STEPS " == *" $1 "* ]]; }
mkdir -p "$OUT"
if has request; then
  BUILD_SSH="ssh $RIG" LONG_MAX_TOKENS=${LONG_MAX_TOKENS:-1024} \
    bash bench/agent_check/first_turn_request.sh "$OUT/request.json" || exit 1
fi
if has send; then
  ssh "$RIG" "rm -rf $RD && mkdir -p $RD/out" || exit 1
  for f in $HERE/clocks_run.sh $HERE/clock_phases.py scripts/ram_watchdog.sh "$OUT/request.json"; do
    ssh "$RIG" "cat > $RD/$(basename "$f")" < "$f" || exit 1
  done
fi
if has start; then
  : "${BIN:?}" "${MODEL:?}" "${DRAFT:?}"
  timeout 30 ssh "$RIG" "BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT REQ=$RD/request.json WD=$RD/ram_watchdog.sh OUT=$RD/out \
    setsid nohup bash $RD/clocks_run.sh > $RD/console.log 2>&1 < /dev/null & echo \$! > $RD/run.pid"
  echo "started $(date -Is)"
fi
if has wait; then
  ssh -o ServerAliveInterval=30 "$RIG" "tail -s 30 --pid=\$(cat $RD/run.pid) -f /dev/null; cat $RD/out/DONE"
  echo "ended $(date -Is)"
fi
if has tables; then
  ssh "$RIG" "python3 -I $RD/clock_phases.py $RD/out" > /dev/null || exit 1
fi
if has collect; then
  ssh "$RIG" "tar -C $RD -cf - ." | tar -C "$OUT" -xf -
fi
if has clean; then
  ssh "$RIG" "rm -rf $RD; ps -eo comm | grep -x -E 'llama-server|llama-bench|nvidia-smi' || echo 'no llama or nvidia-smi process'"
fi
