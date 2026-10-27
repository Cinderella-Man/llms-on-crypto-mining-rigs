#!/usr/bin/env bash
# Q41 main runs (2026-10-23/24), the job each slice ran on the quant machine. WORK: work dir with harness-main/
# (bench/ at 5e26522), venv/, bfcl-v4/, lcb/LiveCodeBench, e1_tasks.jsonl, e3_tasks.jsonl, template.jinja, the ID
# lists, raw/main/, logs/; BIN: llama-server under WORK; MODELS: the folder holding MODEL; GPUS: e.g. 0,1,2;
# STOP_AT: no job starts after it (2026-10-24 12:00 in these runs).
# main.sh LANE TAG MODEL ATTEMPT PART CTX [NP=n] [ENV=VAL ...]: one slice of Q41's main runs on this worker's GPUs
# ($GPUS). LANE e1 = BFCL v4 single-turn (3,477 tasks), e2 = multi-turn (799 tasks), e3 = LiveCodeBench (450 problems,
# 32,768-token budget); attempt ATTEMPT samples with seed ATTEMPT; PART (I/N) is a slice of the tasks in shuffle
# order 1, so that the files alternate slice by slice. NP = server slots and requests in flight (default 8);
# IDS = a file of task IDs to draw the slices from (e2 only).
# llama-server gets the same flags for every file on a GPU set. A job that would start after STOP_AT does
# nothing. The server log keeps only warnings and errors: no timing lines (speed is measured on the rig only).
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${MODELS:?}" "${GPUS:?}"
STOP_AT=${STOP_AT:-2026-10-24 12:00}
LANE=$1; TAG=$2; MODEL=$3; A=$4; PART=$5; CTX=$6; shift 6
if [ "$(date +%s)" -ge "$(date -d "$STOP_AT" +%s)" ]; then echo "after $STOP_AT: not started"; exit 0; fi
V=$WORK
case "$GPUS" in 3,4) PORT=18190;; 5,6) PORT=18192;; *) PORT=18191;; esac
NAME=pq-bfclv4-srv-${GPUS//,/}
SLICE=$LANE-$TAG-a$A-p${PART/\//of}
NP=8; IDS=""; envs=()
for e in "$@"; do case "$e" in NP=*) NP=${e#NP=};; IDS=*) IDS=${e#IDS=};; *) envs+=(-e "$e");; esac; done
docker rm -f $NAME >/dev/null 2>&1
docker run -d --name $NAME --label pascal-quants=1 --network host --gpus "\"device=$GPUS\"" \
  -e CUDA_DEVICE_ORDER=PCI_BUS_ID "${envs[@]}" \
  -v "$WORK:$WORK:ro" -v "$MODELS:$MODELS:ro" \
  nvidia/cuda:12.9.2-devel-ubuntu24.04 "$BIN" \
  -m "$MODEL" -ngl 99 -fa on -np $NP -kvu -c "$CTX" -b 2048 -ub 512 --host 127.0.0.1 --port $PORT >/dev/null
cleanup() {
  docker logs $NAME 2>&1 | grep -iE "error|warn|nan|fail|abort" | grep -viE "eval time|tokens per second|t/s" \
    > $V/logs/server-main-$SLICE.txt
  docker rm -f $NAME >/dev/null 2>&1
}
trap cleanup EXIT
until curl -sf 127.0.0.1:$PORT/health >/dev/null; do
  sleep 10
  [ "$(docker inspect -f '{{.State.Status}}' $NAME 2>/dev/null)" = running ] || { echo "server died"; exit 1; }
done
client() { dir=$1; shift; docker run --rm --label pascal-quants=1 --network host -e BFCL_ROOT=/v4/bfcl-v4 \
  -e LCB_ROOT=/v4/lcb/LiveCodeBench -v $V:/v4 -w /v4/harness-main/$dir python:3.12-slim /v4/venv/bin/python "$@"; }
OUT=/v4/raw/main/$SLICE.jsonl
case "$LANE" in
  e1) client bfcl_flint run.py http://127.0.0.1:$PORT /v4/e1_tasks.jsonl $OUT --attempts $A --shuffle 1 \
        --part $PART --conc $NP;;
  e2) client bfcl_flint multi_turn.py run http://127.0.0.1:$PORT /v4/template.jinja $OUT --attempts $A \
        --shuffle 1 --part $PART --conc $NP ${IDS:+--ids $IDS};;
  e3) client livecodebench_flint run.py http://127.0.0.1:$PORT /v4/e3_tasks.jsonl $OUT --attempts $A --shuffle 1 \
        --part $PART --conc $NP;;
esac
