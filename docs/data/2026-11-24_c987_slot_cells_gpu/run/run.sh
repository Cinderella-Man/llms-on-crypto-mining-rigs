#!/usr/bin/env bash
# C987 (Q1720, LLMRig point 17): does one slot's greedy reply depend on what the other slot did before it? On a build
# machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# With two slots in one unified KV cache (-np 2 -kvu) a slot's tokens go into whatever cells the other slot's history
# left free, and attention sums the keys in cell order over every cell up to the highest used one (the other slot's
# masked out), so the same prompt can be summed in another order. With one stream of cells per slot (-np 2 alone) each
# slot's cells are its own. The build is pascal-p106-v5 3dfad876f (BIN). The model is Ornith-1.5-122B Pascal N2P GGUF
# with its v5 preset (release/ornith-v5/settings.env), the layer split changed to the three cards (-ts 1,1,1) and -np 1
# changed per mode:
#   one       -np 1, as the preset: the fixed request on slot 0
#   kvu       -np 2 -kvu: two slots, one unified KV cache of the preset's 49,152 cells; the fixed request on slot 1
#   streams   -np 2: two slots, one stream of 24,576 cells each; the fixed request on slot 1
# and per history on slot 0 before it (run/slots.py): none, long (a 6,000-token conversation), short (that
# conversation replaced by a 1,000-token one). Each arm is a mode and a history, loaded REPS times (2); each load
# answers slots.py's requests once.
# LLAMA_KV_CACHE_DEBUG=1 with -lv 5 makes the cache print, before every placement of a micro-batch, the head (where the
# search for free cells starts), the used cells, the highest used cell + 1 and each slot's lowest and highest position
# (upstream's debug line), which shows where the fixed request's cells go. The fork's Pascal matrix-vector kernels are
# forced on with a P106's launch shapes (PASCAL=1; a newer card runs them only so), as the release runs them on a P106.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   BIN       bin/ folder of the rig build
#   REPO      a copy of this repository (release/ornith-v5/settings.env, docs/switches.md and this run/)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the build, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first load as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18987), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), ARMS ("one-none kvu-none
#   kvu-long kvu-short streams-none streams-long streams-short"), REPS (2), LV (5), KVDEBUG (1), PASCAL (1), SLOTS_ARGS
#   ("6000 5000 200 1000 256": DEPTH0 DEPTH1 START1 SHORT NPRED, see slots.py), RUN (out), LOAD_TIMEOUT (1800 s),
#   NAME (c987: the containers' name prefix)
# Output: WORK/RUN/<mode>-<history>-r<rep>/ with command.txt, server.log, server.rc, load.txt, session.tsv and the
#   replies; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${BIN:?}" "${REPO:?}" "${ORNITH:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18987}
ARMS=${ARMS:-one-none kvu-none kvu-long kvu-short streams-none streams-long streams-short}
REPS=${REPS:-2}
LV=${LV:-5}
KVDEBUG=${KVDEBUG:-1}
PASCAL=${PASCAL:-1}
SLOTS_ARGS=${SLOTS_ARGS:-6000 5000 200 1000 256}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c987}
PREFLIGHT=${PREFLIGHT:-}
SLOTS=$REPO/docs/data/2026-11-24_c987_slot_cells_gpu/run/slots.py
TEXT=$REPO/docs/switches.md
EXTRA=""
[ "$KVDEBUG" = 1 ] && EXTRA="$EXTRA LLAMA_KV_CACHE_DEBUG=1"
[ "$PASCAL" = 1 ] && EXTRA="$EXTRA GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 \
GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"

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
md5sum "$BIN/llama-server" "$SLOTS" "$TEXT" "$REPO/release/ornith-v5/settings.env" > "$WORK/md5.txt" 2>&1

unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1090
. "$REPO/release/ornith-v5/settings.env"
BASE=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
[[ "$BASE" == *"-ts $TS"* && "$BASE" == *"-np 1 "* ]] || { say "no -ts or -np 1 in the preset to change"; exit 1; }

say "start: arms '$ARMS', reps $REPS, cards $GPUS, BIN=$BIN"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; fi

arm() {  # <mode> <history> <rep>
    local mode=$1 hist=$2 rep=$3 args slot o c up=0 t0
    case $mode in
        one) args=$BASE slot=0 ;;
        kvu) args=${BASE/-np 1 /-np 2 -kvu } slot=1 ;;
        streams) args=${BASE/-np 1 /-np 2 } slot=1 ;;
        *) say "unknown mode $mode"; return ;;
    esac
    [ "$mode" = one ] && [ "$hist" != none ] && { say "one slot takes no history"; return; }
    o=$WORK/$RUN/$mode-$hist-r$rep c=$NAME-$mode-$hist-r$rep
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\nhistory: %s, fixed request on slot %s\n' "$BIN" "$PRESET_ENV" \
        "$EXTRA" "$ORNITH" "$args" "$LV" "$hist" "$slot" > "$o/command.txt"
    say "$mode-$hist-r$rep: load"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $EXTRA $BIN/llama-server -m $ORNITH $args -lv $LV \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up" > "$o/load.txt"
    if [ $up = 1 ]; then
        # shellcheck disable=SC2086
        timeout 3600 python3 -I "$SLOTS" "http://127.0.0.1:$PORT" "$o" "$mode-$hist-r$rep" "$TEXT" "$hist" "$slot" \
            $SLOTS_ARGS > "$o/session.out" 2>&1
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$mode-$hist-r$rep: $(cat "$o/load.txt") $(cat "$o/server.rc" 2>/dev/null), requests $(wc -l < "$o/session.tsv" 2>/dev/null), fixed $(awk -F'\t' '$2 ~ /^fix/ {printf "%s:%s ", $2, substr($7, 1, 8)}' "$o/session.tsv" 2>/dev/null), placements logged $(grep -ac 'find_slot: stream' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT' "$o/server.log"), log $(du -h "$o/server.log" | cut -f1)"
}

for rep in $(seq 1 "$REPS"); do
    for a in $ARMS; do
        arm "${a%%-*}" "${a#*-}" "$rep"
    done
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
