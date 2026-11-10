#!/usr/bin/env bash
# run_exactness.sh: why the release line's greedy reply with the MTP draft differs from the one without it (registry
# Q1175). Runs on a machine with newer cards, where the build's compute 6.1 PTX is compiled by the driver and no speed
# is read. Every load is one llama-server in a container on the line of release/flash-next-v3/settings.env with the
# same explicit card split (TS) in every arm; exact_client.py sends the requests on the loopback.
#   WORK=<work folder> REPO=<checkout of this repository> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     PROMPTS=<prompts.json of bench/deep_lane> GPUS=<docker device list> TS=<split, e.g. 1,2,2> \
#     MOUNTS="<folders the container reads>" \
#     setsid -f nohup nice -n 19 bash run_exactness.sh > console.txt 2>&1 < /dev/null
# STEPS (default "split:nodraft split:draft gen:draft score:nodraft"), each <kind>:<arm>:
#   split:<arm>  the arm's line at -lv 5 until the main model's layer-to-card lines are printed, then stopped:
#                split_<arm>.txt
#   gen:<arm>    the arm's replies at every depth (exact_client.py, phase 1): replies_<arm>.json
#   score:<arm>  the arm's replies, then the probe and the teacher forcing against SCORE_AGAINST's replies (default
#                draft): scored_<arm>.json
# Arms: nodraft (the line without -md and the --spec-* flags), draft (the preset's draft), draft<N> (the draft with
# --spec-draft-n-max N). ENV_<arm>="NAME=value ..." adds variables to one arm's line.
# Env: REF_CSV (an earlier deep_lane run's requests.csv to check the reply hashes against), DEPTHS, NAME (the
# container's name), IMAGE, DOCKER_ARGS, PORT (8097), JIT (the compiled kernels' folder, WORK/jit), LOAD_TIMEOUT_S.
# Writes WORK/chain.done or WORK/chain.failed; prints CHAIN_DONE or CHAIN_FAILED as its last line. A file WORK/stop
# ends the teacher forcing after the request in flight.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${BIN:?BIN=<llama-server>}"
: "${MODEL:?MODEL=<first shard>}" "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${PROMPTS:?PROMPTS=<prompts.json>}"
: "${GPUS:?GPUS=<docker device list, e.g. 0,1,2>}" "${TS:?TS=<card split, e.g. 1,2,2>}"
HERE=$(cd "$(dirname "$0")" && pwd)
STEPS=${STEPS:-split:nodraft split:draft gen:draft score:nodraft}
SCORE_AGAINST=${SCORE_AGAINST:-draft}
DEPTHS=${DEPTHS:-0,8192,16384,32768,49152}
NAME=${NAME:-draft-exactness}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PORT=${PORT:-8097}
JIT=${JIT:-$WORK/jit}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-900}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mounts=''
for m in ${MOUNTS:-$WORK}; do mounts="$mounts -v $m:$m"; done
mkdir -p "$WORK" "$JIT"
rm -f "$WORK/chain.done" "$WORK/chain.failed"
echo $$ > "$WORK/chain.pid"

say() { echo "$(date +%H:%M:%S) $*"; }
stop_server() { docker rm -f "$NAME" > /dev/null 2>&1; wait 2> /dev/null; }
fail() {
  stop_server
  touch "$WORK/chain.failed"; rm -f "$WORK/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}

line_for() {  # line_for ARM [extra flags...]: the arm's variables in ENVS and its flags in LINE
  local arm=$1 words i drop nmax='' v
  shift
  drop='^(-ts|-otd)$'  # the preset's nine-card split and its fifth card
  case $arm in
    nodraft) drop='^(-ts|-otd|--spec-type|--spec-draft-n-max|--spec-draft-n-ubatch)$' ;;
    draft) ;;
    draft[0-9]*) nmax=${arm#draft} ;;
    *) fail "arm $arm: nodraft, draft or draft<N>" ;;
  esac
  read -r -a words <<< "$PRESET_ARGS"
  LINE=()
  for ((i = 0; i < ${#words[@]}; i++)); do
    if [[ ${words[i]} =~ $drop ]]; then i=$((i + 1)); continue; fi
    LINE+=("${words[i]}")
    if [ -n "$nmax" ] && [ "${words[i]}" = --spec-draft-n-max ]; then i=$((i + 1)); LINE+=("$nmax"); fi
  done
  LINE+=(-ts "$TS" "$@")
  [ "$arm" = nodraft ] || LINE+=(-md "$MODEL_DRAFT")
  ENVS=()
  for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done  # no clock to hold on these cards
  ENVS+=("CUDA_CACHE_PATH=$JIT")
  v=ENV_$arm
  # shellcheck disable=SC2206
  ENVS+=(${!v:-})
}

start_server() {  # start_server STEP ARM [extra flags...]: the server in its container, its output in server-STEP.log
  local step=$1 arm=$2
  shift 2
  line_for "$arm" "$@"
  LOG=$WORK/server-$step.log
  # shellcheck disable=SC2086
  set -- docker run --rm --init --name "$NAME" ${DOCKER_ARGS:-} --network host --gpus "\"device=$GPUS\"" $mounts "$IMAGE" \
    env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT"
  echo "=== $(date -Is) $step: $*" > "$LOG"
  printf '%s\t%s\t%s\t%s\n' "$step" "$arm" "$(date -Is)" "${ENVS[*]} ${LINE[*]}" >> "$WORK/loads.tsv"
  "$@" >> "$LOG" 2>&1 < /dev/null &
  SRV=$!
}

wait_ready() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" 2> /dev/null | grep -q '"ok"'; do
    kill -0 "$SRV" 2> /dev/null || fail "the server of $1 ended while loading: $(tail -n 3 "$LOG" | tr '\n' ' ')"
    [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT_S" ] || fail "the server of $1 did not come up in $LOAD_TIMEOUT_S s"
    sleep 2
  done
  { date -Is; nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader; } > "$WORK/fit_$1.txt" 2>&1
}

for spec in $STEPS; do
  kind=${spec%%:*}; arm=${spec#*:}; step=$kind-$arm
  [ -e "$WORK/step-$step.done" ] && { say "STEP $step: done before"; continue; }
  say "STEP $step"
  case $kind in
    split)
      start_server "$step" "$arm" -lv 5
      n=0; same=0; t0=$SECONDS
      while [ "$same" -lt 3 ]; do  # the layer lines come before the weights are read: wait until they stop growing
        sleep 2
        kill -0 "$SRV" 2> /dev/null || break
        [ $((SECONDS - t0)) -lt 300 ] || break
        m=$(grep -c 'assigned to device' "$LOG")
        if [ "$m" -gt 0 ] && [ "$m" = "$n" ]; then same=$((same + 1)); else same=0; fi
        n=$m
      done
      stop_server
      grep -a -E 'assigned to device|offload|tensor split|using device' "$LOG" | sed 's/^[0-9.]* [A-Z] //' > "$WORK/split_$arm.txt"
      [ "$n" -gt 0 ] || say "STEP $step: no layer lines in $LOG (the run goes on)"
      ;;
    gen|score)
      start_server "$step" "$arm"
      wait_ready "$step"
      extra=()
      [ "$kind" = score ] && extra=(--score "$WORK/replies_$SCORE_AGAINST.json" --stop-file "$WORK/stop")
      [ -n "${REF_CSV:-}" ] && extra+=(--ref-csv "$REF_CSV" --ref-arm "${arm%%[0-9]*}")
      python3 -I "$HERE/exact_client.py" --port "$PORT" --prompts "$PROMPTS" --arm "$arm" --out "$WORK" \
        --depths "$DEPTHS" "${extra[@]}" 2>&1 | tee "$WORK/client-$step.txt"
      rc=${PIPESTATUS[0]}
      stop_server
      [ "$rc" = 0 ] || fail "the client of $step ended with $rc"
      ;;
    *) fail "step $spec: split, gen or score" ;;
  esac
  touch "$WORK/step-$step.done"
done
touch "$WORK/chain.done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
