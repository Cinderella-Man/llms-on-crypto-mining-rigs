#!/usr/bin/env bash
# run_draft_count.sh: what a draft of 3 to 7 tokens yields, counted by draft position (registry Q1395). One chain on one
# set of cards. Every load is one llama-server in a container on the line of a preset (default
# release/flash-next-v3/settings.env) with the draft length of its arm; bench/agent_check/replay.py re-sends the
# recorded requests of one agent-check arm to it on the loopback, with the same prompts, seeds and reply limits in every
# load. Only counters are read: no speed. Per load it keeps
#   replay-<label>.jsonl   one row per request with the server's timings counters (draft_n, draft_n_accepted, ...)
#   metrics-<label>.txt    the server's /metrics after the last request: accepted tokens per draft position
#                          (llamacpp:spec_decode_num_accepted_tokens_per_pos_total) and the verify steps
#   server-<label>.log     with LLAMA_TRACE=1: one line per verify step, "accepted a/d draft tokens" ("(restore
#                          checkpoint)" when the step rolled back through a checkpoint)
#   dump-<label>.jsonl     with DUMP=1 (the default): LLAMA_SPEC_DUMP, one JSON line per verify step with the draft, the
#                          target's tokens and the drafter's DUMP_K most probable candidates per drafted position, the
#                          input of bench/tree_replay
# bench/draft_count/draft_table.py reads them.
#   WORK=<work folder> REPO=<checkout of this repository> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     LOG=<the recorded proxy.jsonl, its .bodies.gz next to it> GPUS=<docker device list, or cpu> \
#     MOUNTS="<folders the container reads>" \
#     setsid -f nohup nice -n 19 bash run_draft_count.sh > console.txt 2>&1 < /dev/null
# LOADS (default "d3 d4 d5 d6 d7 d7 d6 d5 d4 d3"): the arms in load order; the n-th load of an arm is labelled
# <arm>-<n>. An arm named d<N>[suffix] drafts N tokens (--spec-draft-n-max N in place of the preset's). Every arm is the
# preset plus COMMON_ARGS (default "--spec-draft-sampling probabilistic --spec-draft-p-min 0": probabilistic
# verification) and COMMON_ENV (default "LLAMA_SPEC_REJECT_STOP=1": C675's stop), then ARGS_<arm> and ENV_<arm>.
# An ENV_<arm> of "LLAMA_SPEC_DUMP=" turns the dump off for that arm.
# Every arm leaves out the preset's -ts and -otd (its nine-card split; TS=<split> sets -ts, faster cards first, as CUDA
# lists them) and GGML_CUDA_KEEPALIVE (no clock to hold on these cards), and adds EXTRA_ARGS (default
# "--metrics --no-webui") and LLAMA_TRACE=1, which only logs.
# GPUS=cpu runs the containers without cards (the CPU dry run: a CPU build of llama-server and a preset for it).
# Env: TAG (the recorded arm's tag pattern, default "^t1swift/"), TRACES_N (only the first N traces), REQUESTS (only
# these request positions of each trace, e.g. 0,1; default all), REPEAT (1), REPLY_LIMIT (recorded), PRESET, DUMP (1),
# DUMP_K (3), DUMP_TGT_K (0), DUMP_NGRAM (16), NAME, IMAGE, LABEL (pascal-quants=1), CPU_SHARES (256), DOCKER_ARGS,
# PORT (8098), JIT (the compiled kernels' folder, WORK/jit), LOAD_TIMEOUT_S (900), FREE_MIB (a card counts as free below
# this much used memory, 500), START_BY (HH:MM: a load starts only before this time; the loads left are listed in
# WORK/not_started).
# A finished load leaves WORK/load-<label>.done and is skipped by a second start. Writes WORK/done (the chain ended; see
# not_started) or WORK/failed; prints CHAIN_DONE or CHAIN_FAILED as its last line.
# Since C804 (Q1408, 2026-11-12): every load also counts the target graph's reallocations per request from its server
# log (print_timing's `graph reallocs = S / D`, as scripts/realloc_counts.py counts them), WORK/reallocs_<label>.tsv
# and a line "LOAD <label>: graph reallocations: W of N requests (...)"; SCHED_DEBUG_REALLOC=1 (default 0) runs every
# server with GGML_SCHED_DEBUG_REALLOC=1, which aborts at a reallocation of an unchanged graph size: off by default,
# since the MTP draft's own graph reallocates so 2-3 times a request on every line (C803, Q1407). The loads and the
# requests are the same as before.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${BIN:?BIN=<llama-server>}"
: "${MODEL:?MODEL=<first shard>}" "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${LOG:?LOG=<recorded proxy.jsonl>}"
: "${GPUS:?GPUS=<docker device list, e.g. 0,1,2, or cpu>}"
LOADS=${LOADS:-d3 d4 d5 d6 d7 d7 d6 d5 d4 d3}
COMMON_ARGS=${COMMON_ARGS---spec-draft-sampling probabilistic --spec-draft-p-min 0}
COMMON_ENV=${COMMON_ENV-LLAMA_SPEC_REJECT_STOP=1}
EXTRA_ARGS=${EXTRA_ARGS---metrics --no-webui}
TAG=${TAG:-^t1swift/}
TRACES_N=${TRACES_N:-}
REQUESTS=${REQUESTS:-}
REPEAT=${REPEAT:-1}
REPLY_LIMIT=${REPLY_LIMIT:-recorded}
DUMP=${DUMP:-1}
DUMP_K=${DUMP_K:-3}
DUMP_TGT_K=${DUMP_TGT_K:-0}
DUMP_NGRAM=${DUMP_NGRAM:-16}
NAME=${NAME:-draft-count}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1}
CPU_SHARES=${CPU_SHARES:-256}
PORT=${PORT:-8098}
JIT=${JIT:-$WORK/jit}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-900}
FREE_MIB=${FREE_MIB:-500}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mounts="-v $WORK:$WORK"
for m in ${MOUNTS:-}; do mounts="$mounts -v $m:$m"; done
mkdir -p "$WORK" "$JIT"
rm -f "$WORK/done" "$WORK/failed" "$WORK/not_started"
echo $$ > "$WORK/chain.pid"

say() { echo "$(date +%H:%M:%S) $*"; }
stop_server() { docker rm -f "$NAME" > /dev/null 2>&1; wait "${SRV:-}" 2> /dev/null; }
fail() {
  stop_server
  touch "$WORK/failed"; rm -f "$WORK/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}

# the recorded arm's traces in the order they ran (the first TRACES_N), and how many requests they send
TRACES=$(python3 -I - "$LOG" "$TAG" "$REQUESTS" "$TRACES_N" << 'EOF'
import json, re, sys
seen, n = [], {}
for l in open(sys.argv[1]):
    if not l.strip():
        continue
    m = re.match(r"^(.+/a\d+/t\d+)/\d+$", json.loads(l).get("tag") or "")
    if m and re.search(sys.argv[2], m.group(1)):
        n[m.group(1)] = n.get(m.group(1), 0) + 1
        if m.group(1) not in seen:
            seen.append(m.group(1))
if sys.argv[4]:
    seen = seen[:int(sys.argv[4])]
want = [int(x) for x in sys.argv[3].split(",")] if sys.argv[3] else None
print(sum(n[t] if want is None else sum(1 for i in want if i < n[t]) for t in seen), ",".join(seen))
EOF
) || fail "could not read $LOG"
NREQ=${TRACES%% *}; TRACES=${TRACES#* }
[ "$NREQ" -gt 0 ] || fail "no request tagged $TAG in $LOG"
NROWS=$((NREQ * REPEAT))
say "$NREQ requests in $(echo "$TRACES" | tr ',' '\n' | wc -l) traces tagged $TAG, $REPEAT each: $NROWS replies a load"

cards_free() {  # every card of GPUS below FREE_MIB of used memory, waiting up to a minute for a stopped server's memory
  local used t0=$SECONDS ok
  [ "$GPUS" = cpu ] && return 0
  while :; do
    ok=1
    for used in $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader,nounits); do
      [ "$used" -lt "$FREE_MIB" ] || ok=0
    done
    [ "$ok" = 1 ] && return 0
    [ $((SECONDS - t0)) -lt 60 ] || return 1
    sleep 5
  done
}

start_server() {  # start_server LABEL ARM: the server in its container, its output in server-LABEL.log
  local label=$1 arm=$2 words i v n_max='' gpu_args=()
  [[ $arm =~ ^d([0-9]+) ]] && n_max=${BASH_REMATCH[1]}
  read -r -a words <<< "$PRESET_ARGS"
  LINE=()
  for ((i = 0; i < ${#words[@]}; i++)); do
    if [[ ${words[i]} =~ ^(-ts|-otd)$ ]]; then i=$((i + 1)); continue; fi
    if [ -n "$n_max" ] && [ "${words[i]}" = --spec-draft-n-max ]; then i=$((i + 1)); continue; fi
    LINE+=("${words[i]}")
  done
  [ -n "${TS:-}" ] && LINE+=(-ts "$TS")
  [ -n "$n_max" ] && LINE+=(--spec-draft-n-max "$n_max")
  v=ARGS_$arm
  # shellcheck disable=SC2206
  LINE+=($EXTRA_ARGS $COMMON_ARGS ${!v:-} -md "$MODEL_DRAFT")
  ENVS=()
  for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done
  ENVS+=("CUDA_CACHE_PATH=$JIT" LLAMA_TRACE=1)
  if [ "$DUMP" = 1 ]; then
    rm -f "$WORK/dump-$label.jsonl"
    ENVS+=("LLAMA_SPEC_DUMP=$WORK/dump-$label.jsonl" "LLAMA_SPEC_DUMP_K=$DUMP_K" "LLAMA_SPEC_DUMP_TGT_K=$DUMP_TGT_K"
      "LLAMA_SPEC_DUMP_NGRAM=$DUMP_NGRAM")
  fi
  # shellcheck disable=SC2206
  ENVS+=($COMMON_ENV)
  v=ENV_$arm
  # shellcheck disable=SC2206
  ENVS+=(${!v:-})
  # C804: GGML_SCHED_DEBUG_REALLOC aborts the server at a reallocation of an unchanged graph size; off by default here,
  # since every load holds the MTP draft, whose own graph reallocates so 2-3 times a request on every line (C803)
  [ "${SCHED_DEBUG_REALLOC:-0}" = 0 ] || ENVS+=("GGML_SCHED_DEBUG_REALLOC=$SCHED_DEBUG_REALLOC")
  [ "$GPUS" = cpu ] || gpu_args=(--gpus "\"device=$GPUS\"")
  SLOG=$WORK/server-$label.log
  # shellcheck disable=SC2086
  set -- docker run --rm --init --name "$NAME" --label "$LABEL" --cpu-shares "$CPU_SHARES" ${DOCKER_ARGS:-} \
    --network host "${gpu_args[@]}" $mounts "$IMAGE" \
    env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT"
  echo "=== $(date -Is) $label: $*" > "$SLOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$arm" "${n_max:-preset}" "$(date -Is)" "${ENVS[*]} ${LINE[*]}" >> "$WORK/loads.tsv"
  "$@" >> "$SLOG" 2>&1 < /dev/null &
  SRV=$!
}

# realloc_count SERVER_LOG TSV (C804): the target graph's reallocations per request from print_timing's running count
# (`graph reallocs = S / D` since the load, after the request's `prompt eval time = X ms / N tokens`); writes one row
# per request to TSV and prints the summary. The MTP draft's own reallocations are not in it (C803).
realloc_count() {
  sed 's/\x1b\[[0-9;]*m//g' "$1" 2> /dev/null | awk -v out="$2" '
    BEGIN { print "req\tprompt_tokens\tprompt_ms\treallocs_same\treallocs_changed" > out }
    /prompt eval time =/ { t = $0; sub(/.*prompt eval time = */, "", t); split(t, p, " "); ms = p[1]; n = p[4]; cur = 1; next }
    cur && /graph reallocs =/ { t = $0; sub(/.*graph reallocs = */, "", t); split(t, q, " ")
      s = q[1] - ps; d = q[3] - pd; ps = q[1]; pd = q[3]; cur = 0
      printf "%d\t%s\t%s\t%d\t%d\n", k, n, ms, s, d > out
      k++; ts += s; td += d; if (s + d > 0) w++ }
    END { printf "graph reallocations: %d of %d requests (%d at the same size, %d changed)\n", w, k, ts, td }'
}

wait_ready() {
  local t0=$SECONDS
  until curl -sf "http://127.0.0.1:$PORT/health" 2> /dev/null | grep -q '"ok"'; do
    kill -0 "$SRV" 2> /dev/null || fail "the server of $1 ended while loading: $(tail -n 3 "$SLOG" | tr '\n' ' ')"
    [ $((SECONDS - t0)) -lt "$LOAD_TIMEOUT_S" ] || fail "the server of $1 did not come up in $LOAD_TIMEOUT_S s"
    sleep 2
  done
  if [ "$GPUS" != cpu ]; then
    { date -Is; nvidia-smi -i "$GPUS" --query-gpu=index,memory.used,memory.total --format=csv,noheader; } > "$WORK/fit_$1.txt" 2>&1
  fi
}

declare -A NTH=()
for arm in $LOADS; do
  NTH[$arm]=$(( ${NTH[$arm]:-0} + 1 ))
  label=$arm-${NTH[$arm]}
  [ -e "$WORK/load-$label.done" ] && { say "LOAD $label: done before"; continue; }
  if [ -e "$WORK/not_started" ] || { [ -n "${START_BY:-}" ] && [[ ! $(date +%H:%M) < $START_BY ]]; }; then
    say "LOAD $label: not started ($(date +%H:%M), a load starts before ${START_BY:-})"
    echo "$label" >> "$WORK/not_started"
    continue
  fi
  cards_free || fail "a card of $GPUS holds $FREE_MIB MiB or more before $label"
  say "LOAD $label"
  rm -f "$WORK/replay-$label.jsonl" "$WORK/metrics-$label.txt"
  start_server "$label" "$arm"
  wait_ready "$label"
  say "LOAD $label: ready"
  rq=()
  [ -n "$REQUESTS" ] && rq=(--requests "$REQUESTS")
  python3 -I "$REPO/bench/agent_check/replay.py" run --log "$LOG" --traces "$TRACES" "${rq[@]}" --repeat "$REPEAT" \
    --reply-limit "$REPLY_LIMIT" --server "http://127.0.0.1:$PORT" --label "$label" \
    --out "$WORK/replay-$label.jsonl" > "$WORK/client-$label.txt" 2>&1
  rc=$?
  # the counters of the whole load, before the server goes
  curl -sf "http://127.0.0.1:$PORT/metrics" > "$WORK/metrics-$label.txt" 2> /dev/null || rm -f "$WORK/metrics-$label.txt"
  stop_server
  say "LOAD $label: $(realloc_count "$SLOG" "$WORK/reallocs_$label.tsv")"
  [ "$rc" = 0 ] || fail "the replay of $label ended with $rc: $(tail -n 2 "$WORK/client-$label.txt" | tr '\n' ' ')"
  [ -s "$WORK/metrics-$label.txt" ] || fail "no /metrics from the server of $label"
  python3 -I - "$WORK/replay-$label.jsonl" "$NROWS" << 'EOF' || fail "the replay of $label is not whole"
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
bad = [r for r in rows if r.get("error") or r.get("status") != 200 or not r.get("predicted_n")]
drafted = sum(r.get("draft_n") or 0 for r in rows)  # the server reports draft_n only when a request drafted
print(f"{len(rows)} rows, {len(bad)} with an error or without a reply, {drafted} drafted tokens, "
      f"{sum(r['predicted_n'] or 0 for r in rows)} generated")
sys.exit(0 if len(rows) == int(sys.argv[2]) and not bad and drafted > 0 else 1)
EOF
  touch "$WORK/load-$label.done"
  say "LOAD $label: done"
done
touch "$WORK/done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
