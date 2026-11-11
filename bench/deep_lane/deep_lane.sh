#!/usr/bin/env bash
# deep_lane.sh: generation and prompt speed of one llama-server line at fixed context depths, arm against arm, loads
# alternated. Runs ON the machine that holds the cards, detached (remote.sh starts it on a rig over ssh), and needs
# bash, curl and python3 there. Per load: one llama-server on the line of a release preset (PRESET: its PRESET_ENV,
# PRESET_ARGS, model and draft) with the arm's changes, then deep_client.py on the loopback: prompts of exactly DEPTHS
# tokens cut from recorded agent text (TEXT), N_PREDICT tokens at temperature 0 twice per depth, one CSV line per
# request. At the end deep_table.py writes the tables. README.md has the method and every output file.
#   BIN=<llama-server> MODELS=<folder of the model files> TEXT=<recorded agent text> OUT=<results folder> \
#     ARMS="dense sparse" ARM_ENV_dense=GGML_CUDA_FATTN_SPARSE_PASCAL=0 ARM_ENV_sparse=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
#     setsid -f nohup bash deep_lane.sh > console.txt 2>&1 < /dev/null
# Env, the files:
#   BIN       llama-server (the binary, or the folder that holds it)
#   PRESET    the preset file (default settings.env beside this script, else release/flash-next-v3/settings.env)
#   MODELS    the folder with the preset's model and draft; or MODEL (the first shard) and MODEL_DRAFT
#   TEXT      recorded agent text, one or more files (request bodies or plain text; deep_client.py); TAG keeps the
#             requests whose proxy tag matches (a regular expression, e.g. '^t1swift/')
#   OUT       the results folder
# Env, the line of every arm (applied to PRESET_ARGS and PRESET_ENV in this order):
#   TS        the card split, replacing the preset's -ts (e.g. 1,1,1 on three equal cards)
#   CTX, UB   the context size and micro-batch, replacing -c and -ub (131072 needs UB=256 on nine 6 GB cards)
#   DRAFT     on|off (default on when the preset names a draft): off drops -md and the draft's flags (DRAFT_FLAGS)
#   ARGS_DROP flags to remove, each with its value (e.g. "-otd" where the preset's device does not exist)
#   ARGS_SET  "flag value" pairs that replace a flag's value or are added (e.g. "-otd token_embd=CUDA2")
#   ENV_DROP  names removed from PRESET_ENV (e.g. GGML_CUDA_KEEPALIVE); ENV_ADD "NAME=value ..." added
#   RUNNER    a command the server line is handed to (e.g. "docker run --rm --init --network host --gpus ... IMAGE");
#             STOP_CMD runs after every server stop (e.g. "docker rm -f NAME")
# Env, the arms: ARMS (names of letters, digits and _; default "base", the preset as it is). Per arm:
#   ARM_ENV_<arm> "NAME=value ..."; ARM_DRAFT_<arm> on|off; ARM_CTX_<arm>, ARM_UB_<arm>; ARM_ARGS_<arm> (as ARGS_SET);
#   ARM_BIN_<arm> the arm's own llama-server in place of BIN (binary or folder: two builds in one run)
# Env, the run:
#   REPS (2) loads per arm, alternated: warm, then every arm once per repetition. LOADS ("name:arm ...") sets
#   another order. WARM (1): one untimed load of the first arm first, at WARM_DEPTHS (0). A load whose
#   OUT/load-<name>.done exists is skipped, so an interrupted run resumes.
#   DEPTHS ("0 8192 16384 32768 49152"), N_PREDICT (256), CACHE (1: each depth extends the cache; 0: every request
#   reads its prompt fresh), SAMPLER_PASS (0; 1 adds a pass with the server line's sampler), TAIL_IDS (64), TAIL_RANK
#   (1), TAIL_TEXT, IGNORE_EOS (1), SEED (1), PAUSE (0: seconds of idle before every request, so that a profiler's
#   trace, GGML_CUDA_PROFILE in an arm's ARM_ENV, cuts into one piece per request)
#   QUIET (1): before every load the machine must hold no llama process and at most QUIET_MAX_MIB (100) MiB on any
#   card, waited for up to QUIET_WAIT_S (60), written to OUT/quiet_<load>.txt. QUIET=0 skips the checks and marks
#   every request valid=0: its times are not measurements (a machine where speed is not measured).
#   WATCHDOG  scripts/ram_watchdog.sh for the server (default ram_watchdog.sh beside this script, else the
#             repository's; none with a RUNNER); a host under 8 GB needs it or WATCHDOG=off; WD_ENV its limits
#             (e.g. MIN_AVAIL_MB=60)
#   MONITOR (1) per-card temperature, power, clock, memory and utilization every 10 s to OUT/gpu.csv
#   PORT (8097), LOAD_TIMEOUT_S (1800), DRY (0; 1 prints each arm's line and stops)
# Prints "LOAD <name> done" per load, then DEEP_DONE (and writes OUT/done), or DEEP_FAILED with the reason (OUT/failed).
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${BIN:?BIN=<llama-server>}" "${OUT:?OUT=<results folder>}" "${TEXT:?TEXT=<recorded agent text>}"
if [ -z "${PRESET:-}" ]; then
  PRESET=$HERE/settings.env
  [ -r "$PRESET" ] || PRESET=$HERE/../../release/flash-next-v3/settings.env
fi
[ -r "$PRESET" ] || { echo "DEEP_FAILED: no preset at $PRESET"; exit 1; }
PRESET_MODEL='' PRESET_DRAFT='' PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
MODEL=${MODEL:-${MODELS:?MODELS=<folder of the model files>, or MODEL=<first shard>}/$PRESET_MODEL}
[ -n "${MODEL_DRAFT:-}" ] || { [ -n "$PRESET_DRAFT" ] && [ -n "${MODELS:-}" ] && MODEL_DRAFT=$MODELS/$PRESET_DRAFT; }
MODEL_DRAFT=${MODEL_DRAFT:-}
ARMS=${ARMS:-base}
REPS=${REPS:-2}
WARM=${WARM:-1}
WARM_DEPTHS=${WARM_DEPTHS:-0}
DEPTHS=${DEPTHS:-0 8192 16384 32768 49152}
N_PREDICT=${N_PREDICT:-256}
CACHE=${CACHE:-1}
QUIET=${QUIET:-1}
QUIET_MAX_MIB=${QUIET_MAX_MIB:-100}
QUIET_WAIT_S=${QUIET_WAIT_S:-60}
DRAFT=${DRAFT:-$([ -n "$MODEL_DRAFT$PRESET_DRAFT" ] && echo on || echo off)}
DRAFT_FLAGS=${DRAFT_FLAGS:-^(-md|--model-draft|-otd|--override-tensor-draft|-ngld|--gpu-layers-draft|-cd|--ctx-size-draft|-devd|--device-draft|--spec-.*|--draft-.*)$}
RUNNER=${RUNNER:-}
PORT=${PORT:-8097}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-1800}
[ -d "$BIN" ] && BIN=$BIN/llama-server
if [ -z "${WATCHDOG:-}" ]; then
  WATCHDOG=none
  if [ -z "$RUNNER" ]; then
    for w in "$HERE/ram_watchdog.sh" "$HERE/../../scripts/ram_watchdog.sh"; do [ -r "$w" ] && { WATCHDOG=$w; break; }; done
  fi
fi
mkdir -p "$OUT"
SRV='' WD='' MON=''

say() { echo "$(date +%H:%M:%S) $*"; }
fail() { echo "DEEP_FAILED: $* $(date -Is)"; rm -f "$OUT/done"; echo "$*" > "$OUT/failed"; exit 1; }
refuse() { echo "DEEP_FAILED: $* $(date -Is)"; exit 1; }  # before the run starts: OUT is left as it is

# The server line as the array LINE: flags and their values, one word each
is_flag() { [[ $1 =~ ^--?[A-Za-z] ]]; }
get_flag() {  # get_flag FLAG: its value in LINE
  local i
  for ((i = 0; i < ${#LINE[@]} - 1; i++)); do [ "${LINE[i]}" = "$1" ] && { echo "${LINE[i + 1]}"; return; }; done
}
set_flags() {  # set_flags WORD...: "flag value" replaces the flag's value in LINE or is added; a flag alone is added once
  local i found
  while [ $# -gt 0 ]; do
    found=''
    for ((i = 0; i < ${#LINE[@]}; i++)); do [ "${LINE[i]}" = "$1" ] && { found=$i; break; }; done
    if [ $# -gt 1 ] && ! is_flag "$2"; then
      if [ -n "$found" ]; then LINE[found + 1]=$2; else LINE+=("$1" "$2"); fi
      shift 2
    else
      [ -n "$found" ] || LINE+=("$1")
      shift
    fi
  done
}
drop_flags() {  # drop_flags REGEX: removes the flags that match from LINE, each with its value
  local out=() i=0 n=${#LINE[@]}
  while [ "$i" -lt "$n" ]; do
    if [[ ${LINE[i]} =~ $1 ]]; then
      i=$((i + 1))
      if [ "$i" -lt "$n" ] && ! is_flag "${LINE[i]}"; then i=$((i + 1)); fi
      continue
    fi
    out+=("${LINE[i]}")
    i=$((i + 1))
  done
  LINE=("${out[@]}")
}

arm_config() {  # arm_config ARM: the arm's environment (ENVS), server line (LINE), A_BIN, A_ENV, A_ARGS, A_DRAFT, A_CTX, A_UB
  local v w ub
  v=ARM_ENV_$1; A_ENV=${!v:-}
  v=ARM_DRAFT_$1; A_DRAFT=${!v:-$DRAFT}
  v=ARM_CTX_$1; A_CTX=${!v:-${CTX:-}}
  v=ARM_UB_$1; ub=${!v:-${UB:-}}
  v=ARM_ARGS_$1; A_ARGS=${!v:-}
  v=ARM_BIN_$1; A_BIN=${!v:-$BIN}
  [ -d "$A_BIN" ] && A_BIN=$A_BIN/llama-server
  ENVS=()
  for w in $PRESET_ENV; do [[ " ${ENV_DROP:-} " == *" ${w%%=*} "* ]] || ENVS+=("$w"); done
  # shellcheck disable=SC2206
  ENVS+=(${ENV_ADD:-} $A_ENV)
  read -ra LINE <<< "$PRESET_ARGS"
  drop_flags '^(-md|--model-draft)$'
  if [ "$A_DRAFT" = on ]; then
    [ -n "$MODEL_DRAFT" ] || refuse "arm $1 has the draft on and no draft file is known: set MODELS or MODEL_DRAFT, or DRAFT=off"
    LINE+=(-md "$MODEL_DRAFT")
  else
    drop_flags "$DRAFT_FLAGS"
  fi
  [ -z "${TS:-}" ] || set_flags -ts "$TS"
  [ -z "$A_CTX" ] || set_flags -c "$A_CTX"
  [ -z "$ub" ] || set_flags -ub "$ub"
  [ -z "${ARGS_DROP:-}" ] || drop_flags "^(${ARGS_DROP// /|})\$"
  # shellcheck disable=SC2086
  set_flags ${ARGS_SET:-} $A_ARGS
  A_CTX=$(get_flag -c)
  A_UB=$(get_flag -ub)
}

quiet() {  # quiet NAME: OUT/quiet_NAME.txt (llama processes, per-card memory, temperature and clock, host memory); 0 when idle
  local f=$OUT/quiet_$1.txt gpu
  gpu=$(nvidia-smi --query-gpu=index,memory.used,temperature.gpu,clocks.sm --format=csv,noheader,nounits 2>&1) \
    || gpu="no nvidia-smi: $gpu"
  { date -Is
    ps -eo comm | grep -x -E 'llama-server|llama-bench|llama-perplexity|llama-cli' || echo "no llama process"
    echo "$gpu"
    grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$f" 2>&1
  grep -q "no llama process" "$f" && echo "$gpu" | awk -F', *' -v m="$QUIET_MAX_MIB" '$2 ~ /^[0-9]+$/ && $2 + 0 > m { bad = 1 } END { exit bad }'
}

wd_start() {  # wd_start LOAD: the RAM watchdog on the server, at real-time priority where chrt may, its table on tmpfs
  [ "$WATCHDOG" = off ] && return 0
  local pre='' n=0
  WD_TSV=/dev/shm/deep_lane-$PORT.tsv
  rm -f "$WD_TSV"
  chrt -f 10 true 2> /dev/null && pre="chrt -f 10"
  # shellcheck disable=SC2086
  env ${WD_ENV:-} PID="$SRV" LOG="$WD_TSV" SERVER_LOG="$OUT/server-$1.log" setsid nohup $pre bash "$WATCHDOG" \
    > "$OUT/watchdog.out" 2>&1 < /dev/null &
  WD=$!
  until grep -qs '^[0-9]' "$WD_TSV"; do  # its first sample
    n=$((n + 1))
    if [ "$n" -gt 12 ] || ! kill -0 "$WD" 2> /dev/null; then say "the RAM watchdog did not come up:"; cat "$OUT/watchdog.out"; return 1; fi
    sleep 0.5
  done
  say "$(grep -m 1 'ram_watchdog:' "$WD_TSV" | sed 's/^# [0-9]* /watchdog up: /')"
}

stop_server() {  # the server, then its watchdog; the watchdog's table goes to OUT/ram_watchdog-<load>.tsv
  if [ -n "$SRV" ]; then
    kill "$SRV" 2> /dev/null
    for _ in $(seq 30); do kill -0 "$SRV" 2> /dev/null || break; sleep 1; done
    kill -9 "$SRV" 2> /dev/null
    wait "$SRV" 2> /dev/null
    # shellcheck disable=SC2086
    [ -z "${STOP_CMD:-}" ] || $STOP_CMD > /dev/null 2>&1
    SRV=''
  fi
  if [ -n "$WD" ]; then
    kill "$WD" 2> /dev/null
    for _ in 1 2 3 4 5 6; do kill -0 "$WD" 2> /dev/null || break; sleep 0.5; done
    kill -9 "$WD" 2> /dev/null
    [ -f "$WD_TSV" ] && mv "$WD_TSV" "$OUT/ram_watchdog-$CUR.tsv"
    [ -s "$OUT/watchdog.out" ] || rm -f "$OUT/watchdog.out"
    WD=''
  fi
}

start_server() {  # start_server LOAD: the server on ENVS and LINE; 0 when /health answers
  local log=$OUT/server-$1.log t0
  if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; then say "a server already answers on port $PORT"; return 1; fi
  echo "=== $(date -Is) $1: ${RUNNER:+$RUNNER }env ${ENVS[*]} $A_BIN -m $MODEL ${LINE[*]} --host 127.0.0.1 --port $PORT" >> "$log"
  # shellcheck disable=SC2086
  $RUNNER env "${ENVS[@]}" "$A_BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT" >> "$log" 2>&1 < /dev/null &
  SRV=$!
  wd_start "$1" || return 1
  t0=$(date +%s)
  until curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null; do
    if ! kill -0 "$SRV" 2> /dev/null; then say "the server exited while loading:"; tail -n 20 "$log"; return 1; fi
    if [ $(($(date +%s) - t0)) -gt "$LOAD_TIMEOUT_S" ]; then say "load timeout after $LOAD_TIMEOUT_S s"; return 1; fi
    sleep 2
  done
  LOAD_S=$(($(date +%s) - t0))
  { date -Is
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits 2>&1
    grep -E 'MemAvailable|SwapFree' /proc/meminfo; } > "$OUT/fit_$1.txt" 2>&1
  say "server $1 up after $LOAD_S s"
}

combine() {  # OUT/requests.csv: every finished or running load's requests, in the loads' order
  local first=1 spec f
  : > "$OUT/requests.csv.tmp"
  for spec in $LOADS; do
    f=$OUT/requests-${spec%%:*}.csv
    [ -s "$f" ] || continue
    if [ $first = 1 ]; then cat "$f"; first=0; else tail -n +2 "$f"; fi >> "$OUT/requests.csv.tmp"
  done
  mv "$OUT/requests.csv.tmp" "$OUT/requests.csv"
}

cleanup() {
  stop_server
  [ -z "$MON" ] || kill "$MON" 2> /dev/null
  rm -f "$OUT/driver.pid"
}
trap cleanup EXIT
trap 'exit 130' INT TERM

# The loads: a warm-up load of the first arm, then the arms alternated
first_arm=${ARMS%% *}
if [ -z "${LOADS:-}" ]; then
  LOADS=''
  for rep in $(seq "$REPS"); do for arm in $ARMS; do LOADS="$LOADS $arm$rep:$arm"; done; done
  [ "$WARM" = 1 ] && LOADS="warm:$first_arm$LOADS"
fi
LOADS=$(echo $LOADS)
max_depth=0
for d in $DEPTHS; do [ "$d" -gt "$max_depth" ] && max_depth=$d; done
VALID=$([ "$QUIET" = 1 ] && echo 1 || echo 0)

say "deep lane: model $(basename "$MODEL"); preset $(basename "$(dirname "$PRESET")")/$(basename "$PRESET"); loads $LOADS; depths $DEPTHS; $N_PREDICT tokens; cache $CACHE; quiet checks $QUIET"
settings="$MODEL|$TEXT|${TAG:-}|$DEPTHS|$N_PREDICT|$CACHE|${SAMPLER_PASS:-0}|${TAIL_IDS:-64}|${TAIL_RANK:-1}|${TAIL_TEXT:-}|${IGNORE_EOS:-1}|${SEED:-1}|$QUIET"
drafts=0
for arm in $ARMS; do
  [[ $arm =~ ^[A-Za-z0-9_]+$ ]] || refuse "arm name '$arm': letters, digits and _ only"
  arm_config "$arm"
  [ "$A_DRAFT" = on ] && drafts=1
  say "arm $arm: draft $A_DRAFT, context $A_CTX, -ub $A_UB${A_ENV:+, $A_ENV}${A_ARGS:+, $A_ARGS}"
  say "  ${RUNNER:+$RUNNER }env ${ENVS[*]} $A_BIN -m $MODEL ${LINE[*]}"
  settings="$settings|$arm: ${ENVS[*]} ${LINE[*]}"
  [ "$A_BIN" = "$BIN" ] || settings="$settings bin $A_BIN"
  OWN_BINS="${OWN_BINS:-} $arm=$A_BIN"
  [ -n "$A_CTX" ] && [ $((max_depth + N_PREDICT)) -gt "$A_CTX" ] \
    && refuse "arm $arm: depth $max_depth + $N_PREDICT tokens does not fit its context of $A_CTX"
done
[ "${DRY:-0}" = 1 ] && { say "DRY=1: nothing started"; exit 0; }
# a run resumes only itself: finished loads in OUT must come from these settings
settings=$(echo "$settings" | md5sum | cut -c1-32)
if ls "$OUT"/load-*.done > /dev/null 2>&1 && [ "$(cat "$OUT/settings.md5" 2> /dev/null)" != "$settings" ]; then
  refuse "$OUT holds finished loads of a run with other settings: give this run its own OUT"
fi
echo "$settings" > "$OUT/settings.md5"
rm -f "$OUT/done" "$OUT/failed"
for f in $TEXT; do [ -r "$f" ] || fail "no recorded text at $f"; done
if [ -z "$RUNNER" ]; then  # with a RUNNER the paths are the runner's own
  [ -x "$BIN" ] || fail "no llama-server at $BIN"
  [ -r "$MODEL" ] || fail "no model file at $MODEL"
  [ "$drafts" = 0 ] || [ -r "$MODEL_DRAFT" ] || fail "no draft file at $MODEL_DRAFT"
  "$BIN" --version > "$OUT/build.txt" 2>&1 < /dev/null || true
  for ab in ${OWN_BINS:-}; do  # an arm with its own build: build-<arm>.txt
    [ "${ab#*=}" = "$BIN" ] && continue
    [ -x "${ab#*=}" ] || fail "no llama-server at ${ab#*=} (arm ${ab%%=*})"
    "${ab#*=}" --version > "$OUT/build-${ab%%=*}.txt" 2>&1 < /dev/null || true
  done
fi
if [ "$WATCHDOG" = none ]; then  # none found: fine with a RUNNER or on a host with memory to spare
  mem_kb=$(awk '/^MemTotal/ { print $2 }' /proc/meminfo)
  [ -n "$RUNNER" ] || [ "$mem_kb" -ge 8000000 ] \
    || fail "this host has $((mem_kb / 1024)) MiB and no RAM watchdog script was found: copy scripts/ram_watchdog.sh beside this script, or set WATCHDOG=off"
  WATCHDOG=off
fi
[ "$WATCHDOG" = off ] || [ -r "$WATCHDOG" ] || fail "no RAM watchdog script at $WATCHDOG"
pending=0
for spec in $LOADS; do [ "${spec%%:*}" = warm ] || [ -f "$OUT/load-${spec%%:*}.done" ] || pending=$((pending + 1)); done
RUN=$LOADS
[ "$pending" -gt 0 ] || RUN=${LOADS#warm:* }  # every timed load is done: no warm-up either, only the tables
echo $$ > "$OUT/driver.pid"
[ -s "$OUT/loads.tsv" ] || printf 'load\tarm\ttimed\tvalid\tdraft\tctx\tub\tcache\tn_predict\tstarted\tload_s\tmodel\tarm_env\tarm_args\tline\tbin\n' > "$OUT/loads.tsv"
if [ "${MONITOR:-1}" = 1 ] && command -v nvidia-smi > /dev/null; then
  nvidia-smi --query-gpu=timestamp,index,temperature.gpu,power.draw,clocks.sm,memory.used,utilization.gpu \
    --format=csv,noheader -l 10 >> "$OUT/gpu.csv" 2>&1 < /dev/null &
  MON=$!
fi

for spec in $RUN; do
  name=${spec%%:*}; arm=${spec#*:}; CUR=$name
  timed=1; depths=$DEPTHS
  if [ "$name" = warm ]; then timed=0; depths=$WARM_DEPTHS; fi
  if [ -f "$OUT/load-$name.done" ] && [ "$timed" = 1 ]; then say "LOAD $name: done in an earlier run, skipped"; continue; fi
  arm_config "$arm"
  if [ "$QUIET" = 1 ]; then
    waited=0
    until quiet "$name"; do
      [ "$waited" -ge "$QUIET_WAIT_S" ] && { cat "$OUT/quiet_$name.txt"; fail "the machine is not idle before load $name (a llama process, or more than $QUIET_MAX_MIB MiB on a card)"; }
      sleep 5; waited=$((waited + 5))
    done
  fi
  say "LOAD $name ($arm)"
  started=$(date -Is)
  if ! start_server "$name"; then
    grep -q '^watchdog:' "$OUT/server-$name.log" && fail "load $name: the RAM watchdog stopped the server (server-$name.log)"
    fail "load $name: the server did not start (server-$name.log)"
  fi
  line="${ENVS[*]} ${LINE[*]}"  # as it ran, the draft named by its file alone
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$arm" "$timed" "$VALID" "$A_DRAFT" "$A_CTX" \
    "$A_UB" "$CACHE" "$N_PREDICT" "$started" "$LOAD_S" "$(basename "$MODEL")" "$A_ENV" "$A_ARGS" \
    "${line//$MODEL_DRAFT/$(basename "$MODEL_DRAFT")}" "$A_BIN" >> "$OUT/loads.tsv"
  # shellcheck disable=SC2086
  python3 -I "$HERE/deep_client.py" --port "$PORT" --arm "$arm" --load "$name" --depths "${depths// /,}" \
    --need-depth "$max_depth" --n-predict "$N_PREDICT" --text $TEXT "--tag=${TAG:-}" --tail-ids "${TAIL_IDS:-64}" \
    --tail-rank "${TAIL_RANK:-1}" "--tail-text=${TAIL_TEXT:-}" --cache "$CACHE" --sampler-pass "${SAMPLER_PASS:-0}" \
    --ignore-eos "${IGNORE_EOS:-1}" --seed "${SEED:-1}" --pause "${PAUSE:-0}" --valid "$VALID" --prompts "$OUT/prompts.json" \
    --csv "$OUT/requests-$name.csv" --jsonl "$OUT/$name.jsonl"
  rc=$?
  combine
  if [ "$rc" != 0 ]; then
    grep -q '^watchdog:' "$OUT/server-$name.log" && fail "load $name: the RAM watchdog stopped the server (server-$name.log)"
    fail "load $name: the client exited with $rc"
  fi
  stop_server
  touch "$OUT/load-$name.done"
  say "LOAD $name done"
done
[ "$QUIET" = 1 ] && { quiet end || say "not idle after the last load (quiet_end.txt)"; }
python3 -I "$HERE/deep_table.py" "$OUT" --out "$OUT/table.md" --tsv "$OUT/classes.tsv" > /dev/null || fail "deep_table.py failed"
touch "$OUT/done"
echo "DEEP_DONE $(date -Is): $OUT/table.md"
