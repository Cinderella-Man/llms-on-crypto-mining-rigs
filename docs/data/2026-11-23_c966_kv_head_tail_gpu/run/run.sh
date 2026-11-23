#!/usr/bin/env bash
# C966 (Q1696, LLMRig point 13): LLAMA_KV_HEAD_TAIL's two head rules on a real file where they differ. C881's fork
# branch pascal-p106-v5-kv-seq-rm 003bf4023 (llama_kv_cache::seq_rm of one sequence on upstream llama.cpp #32380's walk
# over the sequence's (position, cell) index, the fork's head rule kept) against the base pascal-p106-v5 3dfad876f, on a
# build machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# After a removal of a sequence's tail the fork's rule (LLAMA_KV_HEAD_TAIL, on by default) puts the next token into the
# cell of the first removed position; upstream's (LLAMA_KV_HEAD_TAIL=0) moves the head only down, to the lowest freed
# cell. With one sequence the cells stay in position order and both pick the same cell (C952). Here two conversations
# share one stream of cells: Ornith-1.5-122B Pascal N2P GGUF with its v5 preset (release/ornith-v5/settings.env), the
# layer split changed to the three cards (-ts 1,1,1), -np 1 changed to -np 2 -kvu (two slots, one unified KV cache),
# both slots drafting three tokens a step with the MTP layer in the file (--spec-type draft-mtp --spec-draft-n-max 3),
# so every rejected draft is a removal too; run/session2.py sends each round's requests to the slots in turn.
# LLAMA_KV_CACHE_DEBUG=1 with -lv 5 makes the cache print, before every placement of a micro-batch, the head, the used
# cells and each slot's lowest and highest position (upstream's debug line, the same in both builds), which shows where
# the two rules put the head differently. The fork's Pascal matrix-vector kernels are forced on with a P106's launch
# shapes (PASCAL=1; a newer card runs them only so), as the release runs them on a P106.
# One llama-server per arm, in a container; each arm is a build and a setting:
#   builds    base   REF, pascal-p106-v5 3dfad876f
#             new    NEW, the branch
#   settings  def    every switch as the preset sets it (LLAMA_KV_HEAD_TAIL at its default, on)
#             tail0  LLAMA_KV_HEAD_TAIL=0
#             mask   LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1: each card builds the attention mask from its copy of
#                    the cells, kept up to date from the cache's changed-cell list (which the branch fills before
#                    upstream's walk), and every mask is compared with the host's
#             mask0  mask with LLAMA_KV_HEAD_TAIL=0
# Each loaded arm answers session2.py's 16 requests ROUNDS times (2).
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/ornith-v5/settings.env, docs/switches.md and this run/)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first load as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18966), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), SETTINGS ("def tail0 mask
#   mask0"), BUILDS ("base new"), ROUNDS (2), LV (5), KVDEBUG (1), PASCAL (1), SESSION_ARGS ("6000 5000 200 1000 24
#   16": DEPTH0 DEPTH1 START1 SHORT CANCEL_AT N, see session2.py), RUN (out), LOAD_TIMEOUT (1800 s), NAME (c966: the
#   containers' name prefix)
# Output: WORK/RUN/ornmtp2-<setting>-<build>/ with command.txt, server.log, server.rc, load.txt, session.tsv and the
#   replies; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${ORNITH:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18966}
SETTINGS=${SETTINGS:-def tail0 mask mask0}
BUILDS=${BUILDS:-base new}
ROUNDS=${ROUNDS:-2}
LV=${LV:-5}
KVDEBUG=${KVDEBUG:-1}
PASCAL=${PASCAL:-1}
SESSION_ARGS=${SESSION_ARGS:-6000 5000 200 1000 24 16}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c966}
PREFLIGHT=${PREFLIGHT:-}
SESSION=$REPO/docs/data/2026-11-23_c966_kv_head_tail_gpu/run/session2.py
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
md5sum "$NEW/llama-server" "$REF/llama-server" "$SESSION" "$TEXT" "$REPO/release/ornith-v5/settings.env" > "$WORK/md5.txt" 2>&1

unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1090
. "$REPO/release/ornith-v5/settings.env"
ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/; s/-np 1 /-np 2 -kvu /" <<< "$PRESET_ARGS")
[[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"-np 2 -kvu "* ]] || { say "no -ts or -np 1 in the preset to change"; exit 1; }
ARGS="$ARGS --spec-type draft-mtp --spec-draft-n-max 3"

say "start: settings '$SETTINGS', builds '$BUILDS', cards $GPUS, NEW=$NEW REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
if [ -n "$PREFLIGHT" ]; then $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; fi

arm() {  # <setting> <build> <bin folder> <switches>
    local m=ornmtp2 s=$1 b=$2 bin=$3 sw="$4 $EXTRA" o c up=0 t0 r
    o=$WORK/$RUN/$m-$s-$b c=$NAME-$m-$s-$b
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\n' "$bin" "$PRESET_ENV" "$sw" "$ORNITH" "$ARGS" "$LV" \
        > "$o/command.txt"
    say "$m-$s-$b: load ($4)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $sw $bin/llama-server -m $ORNITH $ARGS -lv $LV \
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
            # shellcheck disable=SC2086
            timeout 3600 python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" "$TEXT" $SESSION_ARGS > "$o/session-r$r.out" 2>&1
            say "  $m-$s-$b round $r: $(awk -F'\t' -v r="r$r" '$1 == r' "$o/session.tsv" | wc -l) requests, $(awk -F'\t' -v r="r$r" '$1 == r && $5 > 0' "$o/session.tsv" | wc -l) with tokens"
        done
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$m-$s-$b: $(cat "$o/load.txt") $(cat "$o/server.rc" 2>/dev/null), restores $(grep -ac 'restored context checkpoint' "$o/server.log"), full re-reads $(grep -ac 'forcing full prompt' "$o/server.log"), placements logged $(grep -ac 'find_slot: stream' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT' "$o/server.log"), log $(du -h "$o/server.log" | cut -f1)"
}

for s in $SETTINGS; do
    case $s in
        def) sw="" ;;
        tail0) sw="LLAMA_KV_HEAD_TAIL=0" ;;
        mask) sw="LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1" ;;
        mask0) sw="LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1 LLAMA_KV_HEAD_TAIL=0" ;;
        *) say "unknown setting $s"; continue ;;
    esac
    for b in $BUILDS; do
        case $b in
            base) arm "$s" "$b" "$REF" "$sw" ;;
            new) arm "$s" "$b" "$NEW" "$sw" ;;
            *) say "unknown build $b" ;;
        esac
    done
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
