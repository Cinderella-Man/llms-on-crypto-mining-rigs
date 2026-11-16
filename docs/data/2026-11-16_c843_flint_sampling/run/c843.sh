#!/usr/bin/env bash
# c843.sh start|wait|status|collect: C843 (Q1488) from the operator's machine, which runs only ssh and tar; the run
# (c843_rig.sh) is detached on the rig.
#   start    copies run/ (this folder), the two release settings files, scripts/ram_watchdog.sh and REQUESTS (the
#            picked request bodies, c843_pick.py) to WORK on the rig and starts c843_rig.sh there (console WORK/console.txt)
#   wait     one ssh call that returns when the run has ended (done, failed or stopped)
#   status   the console's last lines
#   collect  WORK/out and the console to OUT here, then removes WORK on the rig once the run has ended
# Env: RIG (the rig's ssh alias), WORK (the run folder on the rig), BIN_V3 and BIN_V4 (llama-server of v3.0 and of
# v4-rc2 on the rig), MODEL (Flint Pascal Q8 GGUF on the rig), REQUESTS (here), OUT (here, for collect); N_PREDICT,
# SEEDS, REPEAT, ARMS, PORT are passed on (c843_rig.sh). Run from the repository's root:
#   RIG=<alias> WORK=<folder> BIN_V3=<v3.0>/bin/llama-server BIN_V4=<v4-rc2>/bin/llama-server MODEL=<model> \
#     REQUESTS=requests.jsonl bash docs/data/2026-11-16_c843_flint_sampling/run/c843.sh start
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${RIG:?}" "${WORK:?}"
rig() { ssh "$RIG" "$@"; }
ended="[ -e $WORK/out/done ] || [ -e $WORK/out/failed ] || ! kill -0 \$(cat $WORK/out/driver.pid 2> /dev/null) 2> /dev/null"
case ${1:-start} in
start)
  : "${BIN_V3:?}" "${BIN_V4:?}" "${MODEL:?}" "${REQUESTS:?}"
  if rig "kill -0 \$(cat $WORK/out/driver.pid 2> /dev/null) 2> /dev/null"; then echo "a run is going in $WORK"; exit 1; fi
  tar -C "$HERE" -cf - . | rig "mkdir -p $WORK/run && tar -C $WORK/run -xf -" || exit 1
  rig "cat > $WORK/run/v3.env" < release/flint-v3/settings.env
  rig "cat > $WORK/run/v4.env" < release/flint-v4/settings.env
  rig "cat > $WORK/run/ram_watchdog.sh" < scripts/ram_watchdog.sh
  rig "cat > $WORK/run/requests.jsonl" < "$REQUESTS"
  { for v in WORK BIN_V3 BIN_V4 MODEL N_PREDICT SEEDS REPEAT ARMS PORT QUIET_MAX_MIB; do
      [ -n "${!v+x}" ] && printf 'export %s=%q\n' "$v" "${!v}"; done; } | rig "cat > $WORK/run.env"
  timeout 30 ssh "$RIG" "cd $WORK && bash -c '. ./run.env && setsid -f nohup bash run/c843_rig.sh > console.txt 2>&1 < /dev/null'"
  sleep 3
  rig "cat $WORK/console.txt"
  ;;
wait) rig "until $ended; do sleep 20; done; tail -n 3 $WORK/console.txt" ;;
status) rig "tail -n 15 $WORK/console.txt; if $ended; then echo 'the run has ended'; else echo 'the run is going'; fi" ;;
collect)
  : "${OUT:?}"
  mkdir -p "$OUT"
  rig "tar -C $WORK/out -cf - ." | tar -C "$OUT" -xf - || { echo "collect failed"; exit 1; }
  rig "cat $WORK/console.txt" > "$OUT/console.txt"
  if rig "$ended"; then rig "[ -f $WORK/run.env ] && [ -d $WORK/run ] && rm -rf $WORK" && echo "removed $WORK on the rig"; fi
  ;;
esac
