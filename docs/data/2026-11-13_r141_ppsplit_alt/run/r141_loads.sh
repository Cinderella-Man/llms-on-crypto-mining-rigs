#!/usr/bin/env bash
# r141_loads.sh: R141 (Q1437), LLAMA_PP_MULTI_SPLIT=1's prompt time in alternated loads on one build. Runs R135's
# driver (../../2026-11-12_c812_draft_pinned/run/r135_load.sh) once per rung in ORDER, one load after the other, on the
# rig. Before each load it records the host (uptime, free -m, /proc/meminfo, llama processes, each card's memory) in
# $PRE/pre_<name>.txt; after each load's DONE it renames the driver's r135_<rung> folder and its .out and .launch to
# r141_<n>_<rung>, so the next load of the same rung does not append to it. Run from a machine that only runs ssh.
# Env: RIG, WORK (the rig folder with r43_run.sh, rig_taper.sh, taper_replay.py, turns.csv, settings.env), BUILDS,
#      BUILD, MODEL, DRAFT (as r135_load.sh), PRE (a local folder for the host records), ORDER (default
#      "embd pp1 pp1 embd"), N0 (loads already run, default 0: the first load of ORDER is numbered N0+1), WAIT=1
#      (first wait for a load the driver already started, r135_<the first rung of ORDER>, then rename it to
#      r141_<N0>_<rung> and go on with the rest of ORDER: for resuming after the driver itself stopped).
set -euo pipefail
: "${RIG:?}" "${WORK:?}" "${BUILDS:?}" "${BUILD:?}" "${MODEL:?}" "${DRAFT:?}" "${PRE:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
DRV=$HERE/../../2026-11-12_c812_draft_pinned/run/r135_load.sh
ORDER=${ORDER:-"embd pp1 pp1 embd"}
mkdir -p "$PRE"
n=${N0:-0}
if [ "${WAIT:-0}" = 1 ]; then
  set -- $ORDER; rung=$1; shift; ORDER="$*"; name=r141_${n}_$rung
  ssh "$RIG" "while [ ! -e $WORK/r135_$rung/DONE ]; do sleep 30; done"
  ssh "$RIG" "cd $WORK && mv r135_$rung $name && mv r135_$rung.out $name.out && mv r135_$rung.launch $name.launch"
  echo "load $n ($rung) done $(date -Is)"
fi
for rung in $ORDER; do
  n=$((n + 1)); name=r141_${n}_$rung
  # never append to an earlier load's folder
  ssh "$RIG" "test ! -e $WORK/r135_$rung && test ! -e $WORK/$name"
  ssh "$RIG" 'date -Is; uptime; free -m; grep -E "^(MemTotal|MemFree|MemAvailable|Buffers|Cached|SwapCached|Active\(file\)|Inactive\(file\)|SwapTotal|SwapFree|Shmem|CmaFree):" /proc/meminfo; ps -eo comm | grep -x llama-server || echo "no llama-server"; nvidia-smi --query-gpu=memory.used --format=csv,noheader' > "$PRE/pre_$name.txt"
  echo "load $n ($rung) starting $(date -Is)"
  RUNG=$rung bash "$DRV"
  # the one wait per load: nothing else touches the rig until DONE
  ssh "$RIG" "while [ ! -e $WORK/r135_$rung/DONE ]; do sleep 30; done"
  ssh "$RIG" "cd $WORK && mv r135_$rung $name && mv r135_$rung.out $name.out && mv r135_$rung.launch $name.launch"
  echo "load $n ($rung) done $(date -Is)"
done
echo "all loads done $(date -Is)"
