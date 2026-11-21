#!/usr/bin/env bash
# C950 (Q1687): the GPU half of C901's check (Flint's pipe slots planned again over the whole memory after a re-plan,
# fork branch pascal-p106-v5-pipe-slot-plan 19048bb6d, switch LLAMA_PIPE_SLOT_PLAN), on a build machine with three newer
# NVIDIA cards that run the rig build's compute 6.1 PTX. With pipelined speculation (--pipe-depth 4 in Flint's preset)
# each generated token is computed in one of 10 pipe slots, each with a scheduler of its own; split over several cards
# those schedulers re-plan their memory after a change of device and then each time the context outgrows the plan of the
# moment, which one processor does not show. Correctness and counts only, nothing timed.
#
# Flint Pascal Q8 GGUF with its v5 preset (release/flint-v5/settings.env; --pipe-depth 4 kept), the layer split
# changed to the three cards (-ts 1,1,1), plus LLAMA_UBATCH_TRACE=1 (every re-plan of a scheduler as a sched-replan
# line) and -lv 4 (libllama's info lines are verbosity 4 in this fork). One llama-server per arm, in a container:
#   sp0   NEW, LLAMA_PIPE_SLOT_PLAN unset (the slots plan as before; the server counts their re-plans)
#   sp1   NEW, LLAMA_PIPE_SLOT_PLAN=1 (a slot that re-plans is planned again over the whole memory)
#   base  REF, the merged v5 engine, the control (no pipe-slot count line; the trace still shows the re-plans)
# Each runs run/session.py's session ROUNDS times (2), then the memory per card is read.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/flint-v5/settings.env and this folder's run/session.py)
#   FLINT     Flint Pascal Q8 GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first load as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18950), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), ARMS ("sp0 sp1 base"),
#   ROUNDS (2), LV (4: the server's -lv), RUN (out: the output folder's name), LOAD_TIMEOUT (1800 s), NAME (c950: the containers' name prefix)
# Output: WORK/RUN/<arm>/ with command.txt, server.log, load.txt, session.tsv and the replies, mem.csv (MiB per card
#   after the session); WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${FLINT:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18950}
ARMS=${ARMS:-sp0 sp1 base}
ROUNDS=${ROUNDS:-2}
LV=${LV:-4}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c950}
PREFLIGHT=${PREFLIGHT:-}
SESSION=$REPO/docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every load (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache" "$WORK/$RUN"
rm -f "$WORK/DONE"
md5sum "$NEW/llama-server" "$REF/llama-server" "$FLINT.sha256" "$SESSION" > "$WORK/md5.txt" 2>&1

# the preset, the split changed to the three cards
unset PRESET_MODEL PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1091
. "$REPO/release/flint-v5/settings.env"
ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
[[ "$ARGS" == *"-ts $TS"* ]] || { say "no -ts in the preset to change"; exit 1; }
say "start: arms '$ARMS', cards $GPUS, NEW=$NEW REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }

arm() {  # <arm> <bin folder> <switches>
    local a=$1 bin=$2 sw=$3 o=$WORK/$RUN/$1 c=$NAME-$1 up=0 t0 r
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s LLAMA_UBATCH_TRACE=1\nargs: -m %s %s -lv %s\n' "$bin" "$PRESET_ENV" "$sw" "$FLINT" "$ARGS" "$LV" \
        > "$o/command.txt"
    say "$a: load ($sw)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $sw LLAMA_UBATCH_TRACE=1 $bin/llama-server -m $FLINT $ARGS -lv $LV \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up" > "$o/load.txt"
    if [ $up = 1 ]; then
        for r in $(seq 1 "$ROUNDS"); do
            python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" > "$o/session-r$r.out" 2>&1
            say "  $a round $r: $(awk -F'\t' -v r="r$r" '$1 == r' "$o/session.tsv" | wc -l) requests, $(awk -F'\t' -v r="r$r" '$1 == r && $5 > 0' "$o/session.tsv" | wc -l) with tokens"
        done
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$a: $(cat "$o/load.txt"), pipeline parallelism line $(grep -ac 'pipeline parallelism enabled' "$o/server.log"), sched-replan lines $(grep -ac 'sched-replan:' "$o/server.log"), whole-memory plan lines $(grep -ac 'pipe_slot_plan_full:' "$o/server.log"), last count: $(grep -a 'pipe reallocs =' "$o/server.log" | tail -n 1 | sed 's/.*pipe reallocs = *//')"
}

for a in $ARMS; do
    case $a in
        sp0) arm sp0 "$NEW" "" ;;
        sp1) arm sp1 "$NEW" "LLAMA_PIPE_SLOT_PLAN=1" ;;
        base) arm base "$REF" "" ;;
        *) say "unknown arm $a" ;;
    esac
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
