#!/usr/bin/env bash
# remote.sh start|wait|status|collect|stop: deep_lane.sh on a rig over ssh, detached there, so the run does not depend
# on this machine, which only runs ssh.
#   start    copies this folder, the preset and scripts/ram_watchdog.sh to WORK on the rig, and the files of
#            TEXT_LOCAL to WORK/text (they become the run's TEXT; a proxy.jsonl beside a .bodies.gz goes with it), writes
#            the run's settings to WORK/run.env and starts deep_lane.sh there: its console in WORK/console.txt, its
#            results in WORK/out
#   wait     one ssh call that returns when the run has ended (done, failed or stopped); nothing else runs on the rig
#   status   the console's last lines and whether the run has ended; not while a timed load runs
#   collect  WORK/out and the console to OUT here; once the run has ended WORK is removed on the rig (CLEAN=0 keeps it)
#   stop     ends the run: the driver stops its server and monitors
# Env: RIG (the rig's ssh alias, non-interactive), SSH (the ssh command, default "ssh"), WORK (on the rig, default
# /tmp/deep-lane), OUT (here, for collect), TEXT_LOCAL (recorded text here), PRESET (here, default
# release/flash-next-v3/settings.env), and deep_lane.sh's own variables, passed on as they are: BIN, MODELS, MODEL,
# MODEL_DRAFT and TEXT are paths on the rig.
#   RIG=rig BIN=<llama-server on the rig> MODELS=<model folder on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz> \
#     ARMS="dense sparse" ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0 ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
#     bash remote.sh start
#   RIG=rig bash remote.sh wait && RIG=rig OUT=<folder> bash remote.sh collect
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${1:?usage: remote.sh start|wait|status|collect|stop}
: "${RIG:?RIG=<ssh alias of the rig>}"
SSH=${SSH:-ssh}
WORK=${WORK:-/tmp/deep-lane}
PASS="BIN MODELS MODEL MODEL_DRAFT TEXT TAG TS CTX UB DRAFT DRAFT_FLAGS ARGS_DROP ARGS_SET ENV_DROP ENV_ADD RUNNER
STOP_CMD ARMS REPS LOADS WARM WARM_DEPTHS DEPTHS N_PREDICT CACHE SAMPLER_PASS TAIL_IDS TAIL_RANK TAIL_TEXT IGNORE_EOS
SEED QUIET QUIET_MAX_MIB QUIET_WAIT_S WATCHDOG WD_ENV MONITOR PORT LOAD_TIMEOUT_S DRY"
rig() { $SSH "$RIG" "$@"; }
ended="[ -e $WORK/out/done ] || [ -e $WORK/out/failed ] || ! kill -0 \$(cat $WORK/out/driver.pid 2> /dev/null) 2> /dev/null"

case $STEP in
start)
  PRESET=${PRESET:-$HERE/../../release/flash-next-v3/settings.env}
  [ -r "$PRESET" ] || { echo "no preset at $PRESET"; exit 1; }
  if rig "kill -0 \$(cat $WORK/out/driver.pid 2> /dev/null) 2> /dev/null"; then echo "a run is going in $RIG:$WORK"; exit 1; fi
  tar -C "$HERE" --exclude=__pycache__ -cf - . | rig "mkdir -p $WORK/deep_lane $WORK/text && tar -C $WORK/deep_lane -xf -" \
    || { echo "copy to $RIG:$WORK failed"; exit 1; }
  rig "cat > $WORK/deep_lane/settings.env" < "$PRESET"
  wd=$HERE/../../scripts/ram_watchdog.sh
  [ -r "$wd" ] && rig "cat > $WORK/deep_lane/ram_watchdog.sh" < "$wd"
  for f in ${TEXT_LOCAL:-}; do
    [ -r "$f" ] || { echo "no recorded text at $f"; exit 1; }
    rig "cat > $WORK/text/$(basename "$f")" < "$f"
    TEXT="${TEXT:+$TEXT }$WORK/text/$(basename "$f")"
    log=${f%.bodies.gz}
    if [ "$log" != "$f" ] && [ -r "$log" ]; then rig "cat > $WORK/text/$(basename "$log")" < "$log"; fi
  done
  { for v in $PASS $(compgen -v | grep '^ARM_'); do [ -n "${!v+x}" ] && printf 'export %s=%q\n' "$v" "${!v}"; done
    printf 'export OUT=%q\n' "$WORK/out"; } | rig "cat > $WORK/run.env"
  # the job is its own session with no terminal; timeout in case the ssh call still waits for it
  timeout 30 $SSH "$RIG" "cd $WORK && bash -c '. ./run.env && setsid -f nohup bash deep_lane/deep_lane.sh > console.txt 2>&1 < /dev/null'"
  sleep 3
  rig "cat $WORK/console.txt"
  ;;
wait)
  rig "until $ended; do sleep 20; done; tail -n 3 $WORK/console.txt"
  ;;
status)
  rig "tail -n 15 $WORK/console.txt; if $ended; then echo 'the run has ended'; else echo 'the run is going'; fi"
  ;;
collect)
  : "${OUT:?OUT=<folder here>}"
  mkdir -p "$OUT"
  rig "tar -C $WORK/out -cf - ." | tar -C "$OUT" -xf - || { echo "collect failed"; exit 1; }
  rig "cat $WORK/console.txt" > "$OUT/console.txt"
  echo "results in $OUT"
  if [ "${CLEAN:-1}" = 1 ] && rig "$ended"; then
    rig "[ -f $WORK/run.env ] && [ -d $WORK/deep_lane ] && rm -rf $WORK" && echo "removed $RIG:$WORK"
  fi
  ;;
stop)
  rig "kill \$(cat $WORK/out/driver.pid) && echo 'stopping'"
  ;;
*)
  echo "usage: remote.sh start|wait|status|collect|stop"; exit 1
  ;;
esac
