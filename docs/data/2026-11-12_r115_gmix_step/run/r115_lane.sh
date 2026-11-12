#!/usr/bin/env bash
# R115 (Q1382), part 3a: the step pair, gmix-q2 against pmix-q2-fast on one server line (line.env on the Flash-Next v3
# preset, the MTP draft), bench/deep_lane at 0 and 8,192 tokens with the sampler pass, R80's recorded text
# (TAG='^t1swift/'), loads warm pmix1 gmix1 pmix2 gmix2 alternated. The model file is the arm's only difference: the
# deep lane's MODEL is pmix-q2-fast's first shard, and gmix's ARM_ARGS end in "-m <gmix-q2's first shard>", which
# llama-server takes as the last -m given (it warns "specified multiple times ... only last value will be used"); each
# server log's model loader line names the file it read.
# Run from the repository root on a machine that only runs ssh:
#   RIG=<ssh alias> WORK=<folder on the rig> BIN=<llama-server on the rig> MODEL=<pmix-q2-fast shard 1 on the rig> \
#   GMIX=<gmix-q2 shard 1 on the rig> DRAFT=<the MTP draft on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz here> \
#   OUT=<folder here> bash r115_lane.sh dry|start|wait|collect
# dry prints both arms' lines and checks that they differ only in the -m; start starts the run detached on the rig
# (remote.sh); wait returns when it has ended; collect copies the results to OUT and removes WORK on the rig.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
STEP=${1:?usage: r115_lane.sh dry|start|wait|collect}
: "${RIG:?}" "${WORK:?}"
# shellcheck disable=SC1091
. "$HERE/line.env"  # LINE_ENV, LINE_ARGS
RS=$HERE/../../../../bench/deep_lane/remote.sh
case $STEP in
dry | start)
  : "${BIN:?}" "${MODEL:?}" "${GMIX:?}" "${DRAFT:?}" "${TEXT_LOCAL:?}"
  lane_env=(RIG="$RIG" WORK="$WORK" BIN="$BIN" MODEL="$MODEL" MODEL_DRAFT="$DRAFT" DRAFT=on TEXT_LOCAL="$TEXT_LOCAL"
    TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 SAMPLER_PASS=1 DEPTHS="0 8192" ARMS="pmix gmix"
    LOADS="warm:pmix pmix1:pmix gmix1:gmix pmix2:pmix gmix2:gmix"
    ARM_ENV_pmix="$LINE_ENV" ARM_ARGS_pmix="$LINE_ARGS"
    ARM_ENV_gmix="$LINE_ENV" ARM_ARGS_gmix="$LINE_ARGS -m $GMIX")
  if [ "$STEP" = dry ]; then
    out=$(env "${lane_env[@]}" DRY=1 bash "$RS" start 2>&1)
    echo "$out"
    grep -q 'DRY=1: nothing started' <<< "$out" || { echo "R115: the dry run failed"; exit 1; }
    p=$(grep -A1 ' arm pmix:' <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    g=$(grep -A1 ' arm gmix:' <<< "$out" | tail -n 1 | sed 's/^[0-9:]* *//')
    [ "$p -m $GMIX" = "$g" ] && echo "R115: the arms differ only in -m $GMIX" || { echo "R115: the arms differ in more than -m"; exit 1; }
  else
    env "${lane_env[@]}" bash "$RS" start
  fi
  ;;
wait) RIG=$RIG WORK=$WORK bash "$RS" wait ;;
collect) : "${OUT:?}"; RIG=$RIG WORK=$WORK OUT=$OUT bash "$RS" collect ;;
*) echo "usage: r115_lane.sh dry|start|wait|collect"; exit 1 ;;
esac
