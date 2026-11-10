#!/usr/bin/env bash
# run_prob_verify.sh: probabilistic verification and kept draws on the MTP release line, counted on newer cards
# (registry Q1173). Every load is one llama-server in a container on the line of release/flash-next-v3/settings.env;
# bench/agent_check/replay.py re-sends the recorded requests of one agent-check arm to it on the loopback, with the
# same prompts, seeds and reply lengths in every load. Only counters are read from the result: no speed.
#   WORK=<work folder> REPO=<checkout of this repository> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     LOG=<the recorded proxy.jsonl, its .bodies.gz next to it> GPUS=<docker device list, e.g. 0,1,2> \
#     MOUNTS="<folders the container reads>" \
#     setsid -f nohup nice -n 19 bash run_prob_verify.sh > console.txt 2>&1 < /dev/null
# LOADS (default "pv0 pv1 kd1 pv0 pv1 kd1"): the arms in load order; the n-th load of an arm is labelled <arm>-<n>.
# An arm is the preset plus ARGS_<arm> (flags) and ENV_<arm> (variables). Defaults:
#   pv0  the preset
#   pv1  ARGS_pv1="--spec-draft-sampling probabilistic --spec-draft-p-min 0"
#   kd1  ENV_kd1="LLAMA_SPEC_KEEP_DRAWS=1"
# Every arm leaves out the preset's -ts and -otd (its nine-card split) and GGML_CUDA_KEEPALIVE (no clock to hold on
# these cards), and sets LLAMA_TRACE=1: one log line per verify step ("accepted a/d draft tokens", with "(restore
# checkpoint)" when the step rolled back through a checkpoint), which changes no computation.
# Env: TAG (the recorded arm's tag pattern, default "^t1swift/"), NAME (the container's name), IMAGE, DOCKER_ARGS,
# PORT (8097), JIT (the compiled kernels' folder, WORK/jit), LOAD_TIMEOUT_S (900), FREE_MIB (a card counts as free
# below this much used memory, 500).
# A finished load leaves WORK/load-<label>.done and is skipped by a second start. Writes WORK/done or WORK/failed;
# prints CHAIN_DONE or CHAIN_FAILED as its last line.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${BIN:?BIN=<llama-server>}"
: "${MODEL:?MODEL=<first shard>}" "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${LOG:?LOG=<recorded proxy.jsonl>}"
: "${GPUS:?GPUS=<docker device list, e.g. 0,1,2>}"
LOADS=${LOADS:-pv0 pv1 kd1 pv0 pv1 kd1}
[ -n "${ARGS_pv1+set}" ] || ARGS_pv1='--spec-draft-sampling probabilistic --spec-draft-p-min 0'
[ -n "${ENV_kd1+set}" ] || ENV_kd1=LLAMA_SPEC_KEEP_DRAWS=1
TAG=${TAG:-^t1swift/}
NAME=${NAME:-prob-verify}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
PORT=${PORT:-8097}
JIT=${JIT:-$WORK/jit}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-900}
FREE_MIB=${FREE_MIB:-500}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mounts=''
for m in ${MOUNTS:-$WORK}; do mounts="$mounts -v $m:$m"; done
mkdir -p "$WORK" "$JIT"
rm -f "$WORK/done" "$WORK/failed"
echo $$ > "$WORK/chain.pid"

say() { echo "$(date +%H:%M:%S) $*"; }
stop_server() { docker rm -f "$NAME" > /dev/null 2>&1; wait 2> /dev/null; }
fail() {
  stop_server
  touch "$WORK/failed"; rm -f "$WORK/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}

# the recorded arm's traces in the order they ran, and its number of requests
TRACES=$(python3 -I - "$LOG" "$TAG" << 'EOF'
import json, re, sys
seen, n = [], 0
for l in open(sys.argv[1]):
    if not l.strip():
        continue
    m = re.match(r"^(.+/a\d+/t\d+)/\d+$", json.loads(l).get("tag") or "")
    if m and re.search(sys.argv[2], m.group(1)):
        n += 1
        if m.group(1) not in seen:
            seen.append(m.group(1))
print(n, ",".join(seen))
EOF
) || fail "could not read $LOG"
NREQ=${TRACES%% *}; TRACES=${TRACES#* }
[ "$NREQ" -gt 0 ] || fail "no request tagged $TAG in $LOG"
say "$NREQ requests in $(echo "$TRACES" | tr ',' '\n' | wc -l) traces tagged $TAG"

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
  LINE+=(${!v:-} -md "$MODEL_DRAFT")
  ENVS=()
  for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done
  ENVS+=("CUDA_CACHE_PATH=$JIT" LLAMA_TRACE=1)
  v=ENV_$arm
  # shellcheck disable=SC2206
  ENVS+=(${!v:-})
  SLOG=$WORK/server-$label.log
  # shellcheck disable=SC2086
  set -- docker run --rm --init --name "$NAME" ${DOCKER_ARGS:-} --network host --gpus "\"device=$GPUS\"" $mounts "$IMAGE" \
    env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT"
  echo "=== $(date -Is) $label: $*" > "$SLOG"
  printf '%s\t%s\t%s\t%s\n' "$label" "$arm" "$(date -Is)" "${ENVS[*]} ${LINE[*]}" >> "$WORK/loads.tsv"
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

declare -A nth
for arm in $LOADS; do
  nth[$arm]=$((${nth[$arm]:-0} + 1))
  label=$arm-${nth[$arm]}
  [ -e "$WORK/load-$label.done" ] && { say "LOAD $label: done before"; continue; }
  cards_free || fail "a card of $GPUS holds $FREE_MIB MiB or more before $label"
  say "LOAD $label"
  rm -f "$WORK/replay-$label.jsonl"
  start_server "$label" "$arm"
  wait_ready "$label"
  say "LOAD $label: ready"
  python3 -I "$REPO/bench/agent_check/replay.py" run --log "$LOG" --traces "$TRACES" \
    --server "http://127.0.0.1:$PORT" --label "$label" --out "$WORK/replay-$label.jsonl" > "$WORK/client-$label.txt" 2>&1
  rc=$?
  stop_server
  [ "$rc" = 0 ] || fail "the replay of $label ended with $rc: $(tail -n 2 "$WORK/client-$label.txt" | tr '\n' ' ')"
  python3 -I - "$WORK/replay-$label.jsonl" "$NREQ" << 'EOF' || fail "the replay of $label is not whole"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [r for r in rows if r.get("error") or r.get("status") != 200 or not r.get("predicted_n")]
drafted = sum(r.get("draft_n") or 0 for r in rows)  # the server reports draft_n only when a request drafted
print(f"{len(rows)} rows, {len(bad)} with an error or without a reply, {drafted} drafted tokens")
sys.exit(0 if len(rows) == int(sys.argv[2]) and not bad and drafted > 0 else 1)
EOF
  touch "$WORK/load-$label.done"
  say "LOAD $label: done"
done
touch "$WORK/done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
