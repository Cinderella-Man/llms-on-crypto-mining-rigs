#!/usr/bin/env bash
# race_quant.sh OUT: the first-finish race in the agent check (Q1195) on a GPU machine with Docker. One llama-server of
# the preset on each card set of GPU_SETS, loaded once, then the task set once per run of RUNS, in that order, with one
# agent (client_arm.sh: pi, the proxy and the tests in containers):
#   one    pi -> proxy.py -> one server; the k-th `one` run uses set ((k - 1) mod N) + 1, so the sets take turns
#   raceN  pi -> race_proxy.py -> the first N servers, a seed each; the first whole reply wins and the rest are
#          cancelled (RACE_ARGS adds race_proxy.py options)
#   raceNr the same with race_proxy.py --win-at reasoning: the first sample to end its reasoning wins (Q1271)
# A run is labelled <run>-<k>, its k-th time: the default RUNS "one race2 one race2" gives one-1 race2-1 one-2 race2-2.
# Counts only (tasks passed, turns, tokens per task, the race's samples and the losers' catch-up read): times on a
# shared multi-GPU machine are not speed results.
#
#   REPO=<checkout> BIN=<llama-server in the server image> MODEL=<first shard> MODEL_DRAFT=<MTP draft> \
#     MOUNTS="<folders the server reads>" GPU_SETS="<set 1> <set 2>" PORT=<first port> bash race_quant.sh OUT
#
# Env: GPU_SETS: one Docker device list per server (indices or UUIDs, comma-separated), sets separated by spaces;
#   PORT: server k on PORT+k-1, the proxy on PORT+N. Optional: PRESET (REPO/release/flash-next-v3/settings.env; its
#   -ts and -otd and GGML_CUDA_KEEPALIVE are left out, as run_pv_replies.sh does: the server splits the layers by the
#   cards' free memory), EXTRA_ARGS ("--metrics --no-webui"), EXTRA_ARGS_<k> (server k only), ENV_<k> (variables for
#   server k only), RUNS, RACE_ARGS, TASK_SET (bugfix), TASKS (all), TIME_CAP_MIN (15), TURN_CAP (40), CTX (65536:
#   pi's window, the preset's -c), SAMPLING (default run.py's), SERVER_IMAGE (nvidia/cuda:12.9.2-devel-ubuntu24.04),
#   IMAGE (agent-check:0.87.1), LABEL (pascal-quants=1), CPU_SHARES (256), PREFIX (acrace), JIT (OUT/jit),
#   LOAD_TIMEOUT_S (900), FREE_MIB (500: a card counts as free below this much used memory).
# Dry run without cards or a model: GPU_SETS="- -" (a set "-" gets no --gpus), SERVER_IMAGE=agent-check:0.87.1,
#   BIN="python3 <this folder>/mock_server.py", MODEL=MODEL_DRAFT=/dev/null, MOUNTS=<this folder>, and each mock's speed
#   in EXTRA_ARGS_1, EXTRA_ARGS_2 (e.g. "--decode-ms 2", "--decode-ms 6").
# Writes OUT/compare.md (the runs side by side; client_report.sh, which also writes table.md and gonogo.md),
# OUT/race.md (race_proxy.py summary, the one-server runs as its --one), servers.tsv (each server's line),
# server-<k>.log, fit-<k>.txt (the cards' memory once loaded), <label>.log and every attempt under OUT/res. Resumable:
# finished attempts are skipped. Every container it started is removed at the end, also on failure.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: race_quant.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../.." && pwd)}
: "${BIN:?}" "${MODEL:?}" "${MODEL_DRAFT:?}" "${GPU_SETS:?}" "${PORT:?}"
read -r -a SETS <<< "$GPU_SETS"
N=${#SETS[@]}
RUNS=${RUNS:-one race2 one race2}
EXTRA_ARGS=${EXTRA_ARGS---metrics --no-webui}
TASK_SET=${TASK_SET:-bugfix}; TASKS=${TASKS:-all}; TIME_CAP_MIN=${TIME_CAP_MIN:-15}; TURN_CAP=${TURN_CAP:-40}
CTX=${CTX:-65536}; SERVER_IMAGE=${SERVER_IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
IMAGE=${IMAGE:-agent-check:0.87.1}; LABEL=${LABEL:-pascal-quants=1}; CPU_SHARES=${CPU_SHARES:-256}
PREFIX=${PREFIX:-acrace}; JIT=${JIT:-$OUT/jit}; LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-900}; FREE_MIB=${FREE_MIB:-500}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mkdir -p "$OUT" "$JIT"
say() { echo "$(date +%H:%M:%S) $*" | tee -a "$OUT/race_quant.log"; }
fail() { say "FAILED: $*"; exit 1; }
race_n() { local n=${1#race}; echo "${n%r}"; }  # the servers of a race run: race2 and race2r -> 2
race_args() { if [[ $1 == *r ]]; then echo "${RACE_ARGS:-} --win-at reasoning"; else echo "${RACE_ARGS:-}"; fi; }
for r in $RUNS; do
  case $r in
    one) ;;
    race[0-9]*) [[ $r =~ ^race[0-9]+r?$ ]] && [ "$(race_n "$r")" -ge 2 ] && [ "$(race_n "$r")" -le "$N" ] \
      || fail "run $r needs 2 to $N servers";;
    *) fail "unknown run $r (one, raceN or raceNr)";;
  esac
done

cleanup() {
  local k
  for k in $(seq "$N"); do
    docker logs "$PREFIX-srv$k" >> "$OUT/server-$k.log" 2>&1
    docker rm -f "$PREFIX-srv$k" > /dev/null 2>&1
  done
  docker ps -aq --filter "name=^$PREFIX-" | xargs -r docker rm -f > /dev/null 2>&1
  say "stopped"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

say "building $IMAGE"
docker build -q --label "$LABEL" -t "$IMAGE" -f "$HERE/docker/Dockerfile" "$HERE" >> "$OUT/race_quant.log" 2>&1 \
  || fail "image build"
say "validating the $TASK_SET tasks in the image"
nice -n 19 python3 "$HERE/validate.py" --task-set "$TASK_SET" --sandbox docker --image "$IMAGE" --docker-label "$LABEL" \
  --docker-prefix "$PREFIX-val" --docker-cpu-shares "$CPU_SHARES" --jobs 2 --out "$OUT/validation.json" \
  > "$OUT/validation.txt" 2>&1 || fail "task validation (validation.txt)"
tail -1 "$OUT/validation.txt" | tee -a "$OUT/race_quant.log"

# the preset's line without its nine-card split and its clock keep-alive
read -r -a words <<< "$PRESET_ARGS"
LINE=()
for ((i = 0; i < ${#words[@]}; i++)); do
  if [[ ${words[i]} =~ ^(-ts|-otd)$ ]]; then i=$((i + 1)); continue; fi
  LINE+=("${words[i]}")
done
# shellcheck disable=SC2206
LINE+=(-md "$MODEL_DRAFT" $EXTRA_ARGS)
ENVS=()
for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done
ENVS+=("CUDA_CACHE_PATH=$JIT")
read -r -a BINW <<< "$BIN"
mounts=(-v "$JIT:$JIT")
for m in ${MOUNTS:-}; do mounts+=(-v "$m:$m"); done

for k in $(seq "$N"); do
  set=${SETS[k - 1]}
  gpu=()
  if [ "$set" != - ]; then
    gpu=(--gpus "\"device=$set\"")
    for used in $(nvidia-smi -i "$set" --query-gpu=memory.used --format=csv,noheader,nounits); do
      [ "$used" -lt "$FREE_MIB" ] || fail "a card of set $k ($set) holds $used MiB"
    done
  fi
  ea=EXTRA_ARGS_$k; ev=ENV_$k
  # shellcheck disable=SC2206
  line=("${LINE[@]}" ${!ea:-}) envs=("${ENVS[@]}" ${!ev:-})
  docker rm -f "$PREFIX-srv$k" > /dev/null 2>&1
  docker run -d --init --name "$PREFIX-srv$k" --label "$LABEL" --cpu-shares "$CPU_SHARES" --network host "${gpu[@]}" \
    "${mounts[@]}" "$SERVER_IMAGE" env "${envs[@]}" "${BINW[@]}" -m "$MODEL" "${line[@]}" --host 127.0.0.1 \
    --port $((PORT + k - 1)) > /dev/null || fail "server $k did not start"
  printf '%s\t%s\t%s\t%s\n' "$k" "$set" "$((PORT + k - 1))" "${envs[*]} ${BINW[*]} -m $MODEL ${line[*]}" >> "$OUT/servers.tsv"
  say "server $k on set $set, port $((PORT + k - 1))"
done
for k in $(seq "$N"); do
  t0=$SECONDS
  until curl -sf "http://127.0.0.1:$((PORT + k - 1))/health" 2> /dev/null | grep -q '"ok"'; do
    [ "$(docker inspect -f '{{.State.Running}}' "$PREFIX-srv$k" 2> /dev/null)" = true ] \
      || fail "server $k ended while loading: $(docker logs "$PREFIX-srv$k" 2>&1 | tail -3 | tr '\n' ' ')"
    [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT_S" ] || fail "server $k did not come up in $LOAD_TIMEOUT_S s"
    sleep 3
  done
  [ "${SETS[k - 1]}" = - ] || nvidia-smi -i "${SETS[k - 1]}" --query-gpu=index,memory.used,memory.total \
    --format=csv,noheader > "$OUT/fit-$k.txt" 2>&1
  say "server $k ready"
done

declare -A seen
LABELS=() ONE=()
for r in $RUNS; do
  seen[$r]=$((${seen[$r]:-0} + 1))
  label=$r-${seen[$r]}
  LABELS+=("$label")
  up='' ups=''
  if [ "$r" = one ]; then
    k=$(((seen[one] - 1) % N + 1))
    up=$((PORT + k - 1))
    ONE+=("$label")
    say "run $label: one server, set $k"
  else
    ups=$(for k in $(seq "$(race_n "$r")"); do printf '%s ' $((PORT + k - 1)); done)
    ra=$(race_args "$r")
    say "run $label: race of $(race_n "$r") servers, ports ${ups% }${ra:+ ($ra)}"
  fi
  OUT=$OUT ARM=$label UPSTREAM_PORT=$up UPSTREAM_PORTS=$ups PROXY_PORT=$((PORT + N)) TASK_SET=$TASK_SET \
    TASKS=$TASKS TIME_CAP_MIN=$TIME_CAP_MIN TURN_CAP=$TURN_CAP CTX=$CTX CPU_SHARES=$CPU_SHARES IMAGE=$IMAGE \
    LABEL=$LABEL PREFIX=$PREFIX RACE_ARGS=$(race_args "$r") SAMPLING=${SAMPLING:-} \
    bash "$HERE/client_arm.sh" > "$OUT/$label.log" 2>&1
  say "run $label: run.py exit $(cat "$OUT/$label.rc" 2> /dev/null); $(tail -1 "$OUT/$label.log")"
done

say "reports"
nice -n 19 bash "$HERE/client_report.sh" "$OUT" "$(IFS=,; echo "${LABELS[*]}")" > /dev/null 2>&1 \
  || say "client_report.sh failed"
ONE_ARGS=()
[ "${#ONE[@]}" -gt 0 ] && ONE_ARGS=(--one "$OUT/proxy.jsonl")
if [ -s "$OUT/proxy.samples.jsonl" ]; then
  nice -n 19 python3 -I "$HERE/race_proxy.py" summary "$OUT/proxy.samples.jsonl" "${ONE_ARGS[@]}" > "$OUT/race.md" \
    || say "race_proxy.py summary failed"
fi
say "done: $OUT/compare.md, $OUT/race.md"
