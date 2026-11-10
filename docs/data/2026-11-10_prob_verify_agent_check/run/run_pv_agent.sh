#!/usr/bin/env bash
# run_pv_agent.sh: probabilistic verification on the agent check, on newer cards (registry Q1183). Every run is one
# llama-server in a container on the line of release/flash-next-v3/settings.env, and the agent check's client on the
# same machine (bench/agent_check/client_arm.sh: the logging proxy in a container, pi and the tests in one container
# per attempt), the bugfix set's 12 tasks, one slot, one task at a time, with the release check's limits. Only counts
# are read from the result (passes, turns, tokens, tool calls, draft counters): no speed.
#   WORK=<work folder> REPO=<checkout of this repository> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     GPUS=<docker device list, e.g. 0,1,2> MOUNTS="<folders the server container reads>" \
#     setsid -f nohup nice -n 19 bash run_pv_agent.sh > console.txt 2>&1 < /dev/null
# RUNS (default "pv0 pv1 pv0 pv1"): the arms in run order; the n-th run of an arm is labelled <arm>-<n>. An arm is
# the preset plus ARGS_<arm> (flags) and ENV_<arm> (variables). Defaults:
#   pv0  the preset
#   pv1  ARGS_pv1="--spec-draft-sampling probabilistic --spec-draft-p-min 0"
# Every arm leaves out the preset's -ts and -otd (its nine-card split) and GGML_CUDA_KEEPALIVE (no clock to hold on
# these cards) and adds EXTRA_ARGS (default "--metrics --no-webui", as the release check's server line has them).
# bench/agent_check/quant_run.sh is not used: its server line is fixed (six slots, 262,144 tokens, no draft) and it
# runs the polyglot set, so this script starts the server itself and calls the release check's client part.
# Env: TASKS (all), TIME_CAP_MIN (12), TURN_CAP (40), BUDGET_MIN (40), CTX (65536): the release check's limits;
# NAME (the server container's name), PREFIX (the client's containers are PREFIX-<label>-*), IMAGE (the server's
# image), SANDBOX_IMAGE (agent-check:0.87.1; it must exist: bench/agent_check/docker/Dockerfile), LABEL
# (pascal-quants=1), CPU_SHARES (256), DOCKER_ARGS (the server container's other options), PORT (8097), PROXY_PORT
# (PORT+1), JIT (the compiled kernels' folder, WORK/jit), LOAD_TIMEOUT_S (900), FREE_MIB (a card counts as free
# below this much used memory, 500), PAIR_START_BY (HH:MM, optional: an odd-numbered run, the first of a pair, starts
# only before this time; the runs left are listed in WORK/not_started).
# Results: WORK/out (res/<label>/<task>/a1/, proxy.jsonl with its bodies), WORK/server-<label>.log,
# WORK/client-<label>.log, WORK/runs.tsv (the lines as run). A finished run leaves WORK/run-<label>.done and is
# skipped by a second start. Writes WORK/done or WORK/failed; prints CHAIN_DONE or CHAIN_FAILED as its last line.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${BIN:?BIN=<llama-server>}"
: "${MODEL:?MODEL=<first shard>}" "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${GPUS:?GPUS=<docker device list, e.g. 0,1,2>}"
RUNS=${RUNS:-pv0 pv1 pv0 pv1}
[ -n "${ARGS_pv1+set}" ] || ARGS_pv1='--spec-draft-sampling probabilistic --spec-draft-p-min 0'
EXTRA_ARGS=${EXTRA_ARGS---metrics --no-webui}
TASKS=${TASKS:-all}; TIME_CAP_MIN=${TIME_CAP_MIN:-12}; TURN_CAP=${TURN_CAP:-40}; BUDGET_MIN=${BUDGET_MIN-40}
CTX=${CTX:-65536}
NAME=${NAME:-pv-agent-srv}
PREFIX=${PREFIX:-pvagent}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
SANDBOX_IMAGE=${SANDBOX_IMAGE:-agent-check:0.87.1}
LABEL=${LABEL:-pascal-quants=1}
CPU_SHARES=${CPU_SHARES:-256}
PORT=${PORT:-8097}
PROXY_PORT=${PROXY_PORT:-$((PORT + 1))}
JIT=${JIT:-$WORK/jit}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-900}
FREE_MIB=${FREE_MIB:-500}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
AC=$REPO/bench/agent_check
OUT=$WORK/out
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mounts=''
for m in ${MOUNTS:-$WORK}; do mounts="$mounts -v $m:$m"; done
mkdir -p "$WORK" "$JIT" "$OUT"
rm -f "$WORK/done" "$WORK/failed" "$WORK/not_started"
echo $$ > "$WORK/chain.pid"

say() { echo "$(date +%H:%M:%S) $*"; }
stop_server() { docker rm -f "$NAME" > /dev/null 2>&1; wait 2> /dev/null; }
fail() {
  stop_server
  touch "$WORK/failed"; rm -f "$WORK/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}

cards_free() {  # every card of GPUS below FREE_MIB of used memory
  local used
  for used in $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits); do
    [ "$used" -lt "$FREE_MIB" ] || return 1
  done
}

start_server() {  # start_server LABEL ARM: the server in its container, its output in server-LABEL.log
  local label=$1 arm=$2 words i v
  read -r -a words <<< "$PRESET_ARGS"
  LINE=()
  for ((i = 0; i < ${#words[@]}; i++)); do
    if [[ ${words[i]} =~ ^(-ts|-otd)$ ]]; then i=$((i + 1)); continue; fi
    LINE+=("${words[i]}")
  done
  v=ARGS_$arm
  # shellcheck disable=SC2206
  LINE+=($EXTRA_ARGS ${!v:-} -md "$MODEL_DRAFT")
  ENVS=()
  for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done
  ENVS+=("CUDA_CACHE_PATH=$JIT")
  v=ENV_$arm
  # shellcheck disable=SC2206
  ENVS+=(${!v:-})
  SLOG=$WORK/server-$label.log
  # shellcheck disable=SC2086
  set -- docker run --rm --init --name "$NAME" --label "$LABEL" --cpu-shares "$CPU_SHARES" ${DOCKER_ARGS:-} \
    --network host --gpus "\"device=$GPUS\"" $mounts "$IMAGE" \
    env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT"
  echo "=== $(date -Is) $label: $*" > "$SLOG"
  printf '%s\t%s\t%s\t%s\n' "$label" "$arm" "$(date -Is)" "${ENVS[*]} ${LINE[*]}" >> "$WORK/runs.tsv"
  "$@" >> "$SLOG" 2>&1 < /dev/null &
  SRV=$!
}

wait_ready() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" 2> /dev/null | grep -q '"ok"'; do
    kill -0 "$SRV" 2> /dev/null || fail "the server of $1 ended while loading: $(tail -n 3 "$SLOG" | tr '\n' ' ')"
    [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT_S" ] || fail "the server of $1 did not come up in $LOAD_TIMEOUT_S s"
    sleep 2
  done
  { date -Is; nvidia-smi -i "$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader; } > "$WORK/fit_$1.txt" 2>&1
}

docker image inspect "$SANDBOX_IMAGE" > /dev/null 2>&1 || fail "the sandbox image $SANDBOX_IMAGE is missing"
if [ ! -e "$OUT/validation.ok" ]; then
  say "validating the bugfix tasks in $SANDBOX_IMAGE"
  (cd "$AC" && python3 validate.py --task-set bugfix --sandbox docker --image "$SANDBOX_IMAGE" --docker-label "$LABEL" \
    --docker-prefix "$PREFIX-val" --docker-cpu-shares "$CPU_SHARES" --jobs 2 --out "$OUT/validation.json") \
    > "$OUT/validation.txt" 2>&1 || fail "task validation failed: $(tail -n 2 "$OUT/validation.txt" | tr '\n' ' ')"
  tail -n 1 "$OUT/validation.txt"
  touch "$OUT/validation.ok"
fi

declare -A nth
pos=0
for arm in $RUNS; do
  pos=$((pos + 1))
  nth[$arm]=$((${nth[$arm]:-0} + 1))
  label=$arm-${nth[$arm]}
  [ -e "$WORK/run-$label.done" ] && { say "RUN $label: done before"; continue; }
  if [ -e "$WORK/not_started" ] || { [ -n "${PAIR_START_BY:-}" ] && [ $((pos % 2)) = 1 ] && [[ ! $(date +%H:%M) < $PAIR_START_BY ]]; }; then
    say "RUN $label: not started ($(date +%H:%M), a pair starts before ${PAIR_START_BY:-})"
    echo "$label" >> "$WORK/not_started"
    continue
  fi
  cards_free || fail "a card of $GPUS holds $FREE_MIB MiB or more before $label"
  say "RUN $label"
  start_server "$label" "$arm"
  wait_ready "$label"
  say "RUN $label: server ready"
  OUT=$OUT ARM=$label UPSTREAM_PORT=$PORT PROXY_PORT=$PROXY_PORT TASK_SET=bugfix TASKS=$TASKS \
    TIME_CAP_MIN=$TIME_CAP_MIN TURN_CAP=$TURN_CAP BUDGET_MIN=$BUDGET_MIN CTX=$CTX CPU_SHARES=$CPU_SHARES \
    IMAGE=$SANDBOX_IMAGE LABEL=$LABEL PREFIX=$PREFIX bash "$AC/client_arm.sh" > "$WORK/client-$label.log" 2>&1
  rc=$?
  stop_server
  [ "$rc" = 0 ] || fail "the client of $label ended with $rc: $(tail -n 2 "$WORK/client-$label.log" | tr '\n' ' ')"
  python3 -I - "$OUT/res/$label" << 'EOF' || fail "the run $label is not whole"
import json, pathlib, sys
rows = [json.loads(p.read_text()) for p in sorted(pathlib.Path(sys.argv[1]).glob("*/a1/result.json"))]
scored = [r for r in rows if r.get("status") == "scored"]
print(f"{len(rows)} attempts, {len(scored)} scored, {sum(r['success'] for r in scored)} passed")
sys.exit(0 if rows and len(scored) == len(rows) else 1)
EOF
  touch "$WORK/run-$label.done"
  say "RUN $label: done"
done
touch "$WORK/done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
