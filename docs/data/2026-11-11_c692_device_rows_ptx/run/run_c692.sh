#!/usr/bin/env bash
# run_c692.sh: C692 (registry Q1225). Upstream llama.cpp #31927, in Miner Llama's v4 line and on by default
# (LLAMA_MTP_H_DEV=1), keeps the target's hidden rows that Flash-Next's MTP draft reads in a buffer on the target's
# output card; v3.0 copies them to host memory and the draft uploads them again. A CPU build has no output card, so only
# a run on cards checks the device path. This one checks it with the fork's draft switches on the whole model: Swift
# Qwen3.8-Flash-Next Q2_0 (the v3 preset's file) with its MTP draft, on three newer NVIDIA cards through the rig builds'
# compute 6.1 PTX, the v4 build against v3.0. Correctness only: no time is read.
# Every load is one llama-server in a container on the release preset's line (release/flash-next-v3/settings.env) with
# a three-card split and the PTX-mode settings; c692_client.py sends its requests on the loopback; c692_compare.py makes
# the tables at the end.
#   WORK=<work folder> REPO=<checkout of this repository> MODEL=<first shard> MODEL_DRAFT=<draft> \
#     BIN_day=<llama-server of the v4 build> BIN_rel=<llama-server of v3.0> GPUS=<docker device list> \
#     MOUNTS="<folders the container reads>" setsid -f nohup nice -n 19 bash run_c692.sh > console.txt 2>&1 < /dev/null
# LOADS (default "rel:pre day:pre day:hdev0 rel:cache day:cache rel:carry day:carry"), each <build>:<arm>, labelled
# <build>-<arm>; the build's server is BIN_<build>. An arm is the preset (with its LLAMA_DRAFT_SHORTLIST: the draft's
# head over a shortlist) plus ENV_<arm>:
#   pre    nothing
#   hdev0  LLAMA_MTP_H_DEV=0 (the v4 build's switch for #31927: the rows through host memory, as v3.0)
#   cache  LLAMA_DRAFT_EMBD_CACHE=8192 (the draft's token rows read from a cache on its layer's card, Q961)
#   carry  LLAMA_DRAFT_H_CARRY=1 (the draft's hidden rows kept on its layer's card between a step's passes, Q961)
# The preset's nine-card -ts and -otd token_embd=CUDA4 become TS (default 1,1,1) and OTD (default token_embd=CUDA1: the
# draft's token table on a card between the first card and the output card, as card 4 of nine on a rig);
# GGML_CUDA_KEEPALIVE is left out (no clock to hold). Every load adds -lv 4 (the load lines that name each buffer's card,
# the h_nextn buffer and whether the draft reads the rows on the device), PTX_ENV, the build's JIT cache and
# LLAMA_SPEC_DUMP=WORK/dump_<label>.jsonl (one line per verify step: the drafts, their candidates and the target's).
# PTX_ENV (default: the operator checks' PTX-mode settings): the fork's Pascal matrix-vector kernels forced on (=2) with
# a P106-090's 5 SMs and 96 KiB of shared memory, so the card runs the kernels a rig runs. CUDA_FORCE_PTX_JIT stays 0:
# both builds hold sm_61 machine code and compute_61 PTX only, so a newer card JIT-compiles the PTX anyway, and 1 would
# also force cuBLAS onto PTX, which cuBLAS does not ship for these cards (Q876). Each build's JIT cache (JIT_<build>,
# default WORK/jit-<build>) is emptied before the first load; its files after each load show the card compiled that
# build's PTX.
# Env: IMAGE, LABEL ("pascal-quants=1 c692=1"), CPU_SHARES (256), NAME (container, c692), PORT (18792), LOAD_TIMEOUT_S
# (1200), FREE_MIB (a card counts as free below this much used memory, 500), N_PREDICT (512), REPEAT (2), LONG_FILE
# (the script the long prompt asks to review; default REPO/docs/data/2026-11-10_draft_line_exactness/run/exact_client.py).
# A finished load leaves WORK/load-<label>.done and is skipped by a second start. Writes WORK/loads.tsv,
# WORK/archs_<build>.txt, WORK/server-<label>.log, WORK/fit_<label>.txt, WORK/jit_<label>.txt,
# WORK/requests_<label>.jsonl, WORK/dump_<label>.jsonl, WORK/tables/, then WORK/done or WORK/failed; prints CHAIN_DONE
# or CHAIN_FAILED as its last line.
# Since C804 (Q1408, 2026-11-12): every load also counts the target graph's reallocations per request from its server
# log (print_timing's `graph reallocs = S / D`, as scripts/realloc_counts.py counts them), WORK/reallocs_<label>.tsv
# and a line "LOAD <label>: graph reallocations: W of N requests (...)"; SCHED_DEBUG_REALLOC=1 (default 0) runs every
# server with GGML_SCHED_DEBUG_REALLOC=1, which aborts at a reallocation of an unchanged graph size: off by default,
# since the MTP draft's own graph reallocates so 2-3 times a request on every line (C803, Q1407). The loads and the
# requests are the same as before.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<checkout of this repository>}" "${MODEL:?MODEL=<first shard>}"
: "${MODEL_DRAFT:?MODEL_DRAFT=<draft>}" "${GPUS:?GPUS=<docker device list>}"
HERE=$(cd "$(dirname "$0")" && pwd)
LOADS=${LOADS:-rel:pre day:pre day:hdev0 rel:cache day:cache rel:carry day:carry}
TS=${TS:-1,1,1}
OTD=${OTD:-token_embd=CUDA1}
[ -n "${ENV_pre+set}" ] || ENV_pre=''
[ -n "${ENV_hdev0+set}" ] || ENV_hdev0='LLAMA_MTP_H_DEV=0'
[ -n "${ENV_cache+set}" ] || ENV_cache='LLAMA_DRAFT_EMBD_CACHE=8192'
[ -n "${ENV_carry+set}" ] || ENV_carry='LLAMA_DRAFT_H_CARRY=1'
PTX_ENV=${PTX_ENV:-CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
LABEL=${LABEL:-pascal-quants=1 c692=1}
CPU_SHARES=${CPU_SHARES:-256}
NAME=${NAME:-c692}
PORT=${PORT:-18792}
LOAD_TIMEOUT_S=${LOAD_TIMEOUT_S:-1200}
FREE_MIB=${FREE_MIB:-500}
N_PREDICT=${N_PREDICT:-512}
REPEAT=${REPEAT:-2}
LONG_FILE=${LONG_FILE:-$REPO/docs/data/2026-11-10_draft_line_exactness/run/exact_client.py}
PRESET=${PRESET:-$REPO/release/flash-next-v3/settings.env}
PRESET_ENV='' PRESET_ARGS=''
# shellcheck disable=SC1090
. "$PRESET"
mounts=''
for m in ${MOUNTS:-$WORK}; do mounts="$mounts -v $m:$m"; done
labels=''
for l in $LABEL; do labels="$labels --label $l"; done
mkdir -p "$WORK"
rm -f "$WORK/done" "$WORK/failed"
echo $$ > "$WORK/chain.pid"

say() { echo "$(date +%H:%M:%S) $*"; }
stop_server() { docker rm -f "$NAME" > /dev/null 2>&1; wait "${SRV:-}" 2> /dev/null; }
fail() {
  stop_server
  touch "$WORK/failed"; rm -f "$WORK/chain.pid"
  echo "CHAIN_FAILED: $* $(date -Is)"
  exit 1
}
trap 'fail "stopped by a signal"' INT TERM

cards_free() {  # every card of GPUS below FREE_MIB of used memory, waiting up to a minute for a stopped server's memory
  local used t0=$SECONDS ok
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

archs() {  # archs BUILD BIN: the GPU code in the build's server, as cuobjdump in the image lists it
  # shellcheck disable=SC2086
  docker run --rm $labels --cpu-shares "$CPU_SHARES" $mounts "$IMAGE" bash -c \
    "echo \"md5 \$(md5sum < '$2' | cut -d' ' -f1)\"; \
     echo \"machine code: \$(cuobjdump --list-elf '$2' | grep -o 'sm_[0-9]*\.cubin' | sort | uniq -c | tr -s ' ' | tr '\n' ';')\"; \
     echo \"PTX: \$(cuobjdump --list-ptx '$2' | grep -o 'sm_[0-9]*\.ptx' | sort | uniq -c | tr -s ' ' | tr '\n' ';')\"" \
    > "$WORK/archs_$1.txt" 2>&1
  say "archs $1: $(tr '\n' ' ' < "$WORK/archs_$1.txt")"
}

start_server() {  # start_server LABEL BUILD ARM: the server in its container, its output in server-LABEL.log
  local label=$1 b=$2 arm=$3 words i v
  BIN=BIN_$b; BIN=${!BIN:-}
  [ -x "$BIN" ] || fail "BIN_$b ('$BIN') is not a program"
  v=ENV_$arm
  [ -n "${!v+set}" ] || fail "arm $arm: no ENV_$arm"
  read -r -a words <<< "$PRESET_ARGS"
  LINE=()
  for ((i = 0; i < ${#words[@]}; i++)); do
    if [[ ${words[i]} =~ ^(-ts|-otd)$ ]]; then i=$((i + 1)); continue; fi
    LINE+=("${words[i]}")
  done
  LINE+=(-ts "$TS" -otd "$OTD" -lv 4 -md "$MODEL_DRAFT")
  ENVS=()
  for v in $PRESET_ENV; do [ "${v%%=*}" = GGML_CUDA_KEEPALIVE ] || ENVS+=("$v"); done
  JITDIR=JIT_$b; JITDIR=${!JITDIR:-$WORK/jit-$b}
  rm -f "$WORK/dump_$label.jsonl"
  # shellcheck disable=SC2206
  ENVS+=($PTX_ENV "CUDA_CACHE_PATH=$JITDIR" CUDA_CACHE_MAXSIZE=4294967296 "LLAMA_SPEC_DUMP=$WORK/dump_$label.jsonl")
  v=ENV_$arm
  # shellcheck disable=SC2206
  ENVS+=(${!v})
  # C804: GGML_SCHED_DEBUG_REALLOC aborts the server at a reallocation of an unchanged graph size; off by default here,
  # since every load holds the MTP draft, whose own graph reallocates so 2-3 times a request on every line (C803)
  [ "${SCHED_DEBUG_REALLOC:-0}" = 0 ] || ENVS+=("GGML_SCHED_DEBUG_REALLOC=$SCHED_DEBUG_REALLOC")
  SLOG=$WORK/server-$label.log
  # shellcheck disable=SC2086
  set -- docker run --rm --init --name "$NAME" $labels --cpu-shares "$CPU_SHARES" ${DOCKER_ARGS:-} \
    --network host --gpus "\"device=$GPUS\"" $mounts "$IMAGE" \
    env "${ENVS[@]}" "$BIN" -m "$MODEL" "${LINE[@]}" --host 127.0.0.1 --port "$PORT"
  echo "=== $(date -Is) $label: $*" > "$SLOG"
  printf '%s\t%s\t%s\t%s\t%s\n' "$label" "$b" "$arm" "$(date -Is)" "${ENVS[*]} $BIN -m $MODEL ${LINE[*]}" >> "$WORK/loads.tsv"
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
  { date -Is; nvidia-smi -i "$GPUS" --query-gpu=index,pci.bus_id,name,memory.used,memory.total --format=csv,noheader; } \
    > "$WORK/fit_$1.txt" 2>&1
}

# a fresh run starts from empty JIT caches, so that each cache's files come from this run's loads
if ! ls "$WORK"/load-*.done > /dev/null 2>&1; then
  for spec in $LOADS; do
    b=${spec%%:*}; JITDIR=JIT_$b; JITDIR=${!JITDIR:-$WORK/jit-$b}
    rm -rf "$JITDIR"; mkdir -p "$JITDIR"
  done
  printf 'label\tbuild\tarm\tstarted\tline\n' > "$WORK/loads.tsv"
fi
say "loads: $LOADS; split -ts $TS -otd $OTD; PTX_ENV: $PTX_ENV"
say "cards: $(nvidia-smi -i "$GPUS" --query-gpu=index,pci.bus_id,name,memory.used --format=csv,noheader | tr '\n' ';')"

for spec in $LOADS; do
  b=${spec%%:*}; arm=${spec#*:}; label=$b-$arm
  [ -e "$WORK/load-$label.done" ] && { say "LOAD $label: done before"; continue; }
  if [ ! -s "$WORK/archs_$b.txt" ]; then BIN=BIN_$b; archs "$b" "${!BIN:-}"; fi
  cards_free || fail "the cards are not free before $label: $(nvidia-smi -i "$GPUS" --query-gpu=memory.used --format=csv,noheader | tr '\n' ' ')"
  say "LOAD $label: start"
  start_server "$label" "$b" "$arm"
  wait_ready "$label"
  say "LOAD $label: ready; cards $(tail -n +2 "$WORK/fit_$label.txt" | tr '\n' ';')"
  python3 -I "$HERE/c692_client.py" --port "$PORT" --label "$label" --out "$WORK" --long-file "$LONG_FILE" \
    --n-predict "$N_PREDICT" --repeat "$REPEAT" 2>&1 | sed "s/^/  /"
  rc=${PIPESTATUS[0]}
  stop_server
  echo "$(find "$JITDIR" -type f | wc -l) files, $(du -sb "$JITDIR" | cut -f1) B" > "$WORK/jit_$label.txt"
  say "LOAD $label: client exit $rc; JIT cache $JITDIR: $(cat "$WORK/jit_$label.txt"); dump $(wc -l < "$WORK/dump_$label.jsonl" 2> /dev/null || echo 0) steps"
  say "LOAD $label: $(realloc_count "$SLOG" "$WORK/reallocs_$label.tsv")"
  [ "$rc" = 0 ] || fail "the client of $label ended with $rc"
  touch "$WORK/load-$label.done"
done

python3 -I "$HERE/c692_compare.py" "$WORK" 2>&1 | sed "s/^/  /"
[ "${PIPESTATUS[0]}" = 0 ] || fail "c692_compare.py failed"
touch "$WORK/done"
rm -f "$WORK/chain.pid"
echo "CHAIN_DONE $(date -Is)"
