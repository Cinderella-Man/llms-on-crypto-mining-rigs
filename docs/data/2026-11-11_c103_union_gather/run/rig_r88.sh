#!/usr/bin/env bash
# R88 part 1 (Q1315) on a P106 rig: the qsa-union replay on one card, then fasp_check --gpu and --bench on it.
# Env: DIR (holds qsa-union/, fasp_check, replay_checks.sh; output goes there), BIN (bin/ of the rig tarball),
#   DEV (the card, default 0). Writes DIR/out/ (the replay), DIR/fasp_gpu.txt, DIR/fasp_bench.txt, DIR/steps.txt,
#   DIR/done. Run it detached: setsid nohup bash rig_r88.sh > DIR/driver.log 2>&1 < /dev/null &
set -uo pipefail
DIR=${DIR:?DIR=<folder>}
BIN=${BIN:?BIN=<bin/ of the rig tarball>}
DEV=${DEV:-0}
cd "$DIR" || exit 1
rm -f done
step() { echo "$(date -Is) $*" >> steps.txt; }
step "start; $(nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader -i "$DEV")"
SET=$DIR/qsa-union BIN=$BIN OUT=$DIR/out DEV=$DEV CPU_CHECK=200 bash "$DIR/replay_checks.sh" > replay.log 2>&1
step "replay exit $?"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV ./fasp_check --gpu --nsm 5 --threads 2 > fasp_gpu.txt 2>&1
step "fasp_check --gpu exit $?"
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$DEV ./fasp_check --bench --nsm 5 --threads 2 > fasp_bench.txt 2>&1
step "fasp_check --bench exit $?"
step "end; $(nvidia-smi --query-gpu=index,memory.used,temperature.gpu --format=csv,noheader -i "$DEV")"
touch done
