#!/usr/bin/env bash
# Q684 on one rig: bench/persistent_pipe's run_rig.sh on every card (~15 minutes), detached on the rig so a dropped ssh
# connection does not stop it, then its output back to OUT and the rig cleaned. Nothing else may run on the rig
# meanwhile; the wait below is one ssh session sleeping on the rig until the run's DONE marker.
# Env: RIG: ssh alias of the rig (root); BIN: the binary (bench/persistent_pipe/build.sh), on this machine or, with
#      FROM set, on the ssh host FROM (ssh arguments: options and the host); MD5_BIN: its md5, checked on the rig;
#      SCRIPT: run_rig.sh (default: this repository's); OUT: where the output goes; STEPS, CARDS, EXTRA: passed to
#      run_rig.sh; STEP: "all" (default), "start", "wait" or "collect" (to resume after a dropped connection).
set -euo pipefail
: "${RIG:?}" "${BIN:?}" "${MD5_BIN:?}" "${OUT:?}"
here=$(cd "$(dirname "$0")" && pwd)
SCRIPT=${SCRIPT:-$here/../../../../bench/persistent_pipe/run_rig.sh}
STEP=${STEP:-all}
D=/tmp/persistent_pipe_q684
mkdir -p "$OUT"
# FROM may carry ssh options, so it is split into words
# shellcheck disable=SC2086
fetch() { if [ -n "${FROM:-}" ]; then ssh $FROM "cat '$1'"; else cat "$1"; fi; }

if [ "$STEP" = all ] || [ "$STEP" = start ]; then
    ssh "$RIG" "mkdir -p $D"
    fetch "$BIN" | ssh "$RIG" "cat > $D/persistent_pipe && chmod +x $D/persistent_pipe"
    ssh "$RIG" "cat > $D/run_rig.sh && chmod +x $D/run_rig.sh" < "$SCRIPT"
    ssh "$RIG" "cd $D && echo '$MD5_BIN  persistent_pipe' > md5 && md5sum -c md5"
    timeout 30 ssh "$RIG" "cd $D && STEPS='${STEPS:-}' CARDS='${CARDS:-}' EXTRA='${EXTRA:-}' \
        setsid nohup ./run_rig.sh > console.log 2>&1 < /dev/null &" || true
    echo "started on $RIG: $D"
fi
if [ "$STEP" = all ] || [ "$STEP" = wait ]; then
    ssh "$RIG" "for i in \$(seq 1 360); do [ -e $D/out/DONE ] && exit 0; sleep 10; done; exit 1" \
        || { echo "no DONE after an hour; collect with STEP=collect"; exit 1; }
fi
if [ "$STEP" = all ] || [ "$STEP" = collect ]; then
    ssh "$RIG" "cd $D && tar -cf - console.log out" | tar -xf - -C "$OUT"
    ssh "$RIG" "rm -rf $D"
    cat "$OUT/out/run.log"
    grep -q 'done, 0 failed' "$OUT/out/run.log" && echo "rig run: PASS" || { echo "rig run: a step FAILED"; exit 1; }
fi
