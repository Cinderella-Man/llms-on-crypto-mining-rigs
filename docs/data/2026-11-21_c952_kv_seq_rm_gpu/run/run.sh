#!/usr/bin/env bash
# C952 (Q1687 point 6): the GPU half of C881's check (fork branch pascal-p106-v5-kv-seq-rm 003bf4023, two commits on
# pascal-p106-v5 6644c2bc6: llama_kv_cache::seq_rm of one sequence on upstream llama.cpp #32380's walk over the
# sequence's (position, cell) index, the fork's LLAMA_KV_HEAD_TAIL rule kept), on a build machine with three newer
# NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# Two release files with their v5 presets (release/ornith-v5/settings.env; release/flash-next-v5/settings.env with its
# MTP draft), the layer split changed to the three cards (-ts 1,1,1), -lv 4 (the server's checkpoint lines are
# verbosity 4). Model "ornmtp" is Ornith's preset with the MTP layer in its own file drafting three tokens a step
# (--spec-type draft-mtp --spec-draft-n-max 3): its rejected drafts remove a few cells at a time from a cache whose
# attention mask the cards can build (Flash-Next's attention layers are all sparse-attention layers with masks of their
# own, so the cache's mask and its changed-cell list are not used there). One llama-server per arm, in a container;
# each arm is a build and a setting:
#   builds    base  REF, pascal-p106-v5 3dfad876f (6644c2bc6 and one commit that reads LLAMA_PIPE_GRAMMAR=0 as off;
#                   neither preset runs the pipe)
#             new   NEW, the branch
#   settings  def   every switch as the preset sets it (LLAMA_KV_HEAD_TAIL at its default, on)
#             tail0 LLAMA_KV_HEAD_TAIL=0 (upstream's head rule after a tail removal)
#             mask  LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1: each card builds the attention mask from its copy of
#                   the cells, updated from the cache's changed-cell list (which the branch now fills before upstream's
#                   walk), and every mask is compared with the host's
# Each loaded arm answers run/session.py's requests ROUNDS times (2): a fresh prompt of DEPTH tokens, a turn on it, a
# changed history (a checkpoint restore), a streamed reply cancelled after CANCEL_AT chunks, the request after the
# cancel, and a return to the document's end. Flash-Next drafts with its MTP head on every greedy reply, so every
# rejected draft is a removal too.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/{ornith,flash-next}-v5/settings.env, docs/switches.md and this run/)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF; FN Flash-Next's preset file (first shard, the others beside it);
#   DRAFT     the MTP draft for FN
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the models)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first load as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18952), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), MODELS_RUN ("ornith fn";
#   C952 ran it, then MODELS_RUN=ornmtp SETTINGS="def mask"),
#   SETTINGS ("def tail0 mask"), BUILDS ("base new"), ROUNDS (2), LV (4), DEPTH (6000), CANCEL_AT (24), RUN (out),
#   LOAD_TIMEOUT (1800 s), NAME (c952: the containers' name prefix)
# Output: WORK/RUN/<model>-<setting>-<build>/ with command.txt, server.log, server.rc, load.txt, session.tsv and the
#   replies; WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${ORNITH:?}" "${FN:?}" "${DRAFT:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18952}
MODELS_RUN=${MODELS_RUN:-ornith fn}
SETTINGS=${SETTINGS:-def tail0 mask}
BUILDS=${BUILDS:-base new}
ROUNDS=${ROUNDS:-2}
LV=${LV:-4}
DEPTH=${DEPTH:-6000}
CANCEL_AT=${CANCEL_AT:-24}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c952}
PREFLIGHT=${PREFLIGHT:-}
SESSION=$REPO/docs/data/2026-11-21_c952_kv_seq_rm_gpu/run/session.py
TEXT=$REPO/docs/switches.md

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
md5sum "$NEW/llama-server" "$REF/llama-server" "$DRAFT" "$SESSION" "$TEXT" > "$WORK/md5.txt" 2>&1

# preset <name>: PRESET_ENV and ARGS of release/<name>/settings.env, the split changed to the three cards
preset() {
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1090
    . "$REPO/release/$1/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* ]] || { say "no -ts in $1's preset to change"; exit 1; }
}

say "start: models '$MODELS_RUN', settings '$SETTINGS', builds '$BUILDS', cards $GPUS, NEW=$NEW REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }

arm() {  # <model> <setting> <build> <bin folder> <model file> <draft args> <switches>
    local m=$1 s=$2 b=$3 bin=$4 file=$5 md=$6 sw=$7 o=$WORK/$RUN/$1-$2-$3 c=$NAME-$1-$2-$3 up=0 t0 r
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s %s -lv %s\n' "$bin" "$PRESET_ENV" "$sw" "$file" "$md" "$ARGS" "$LV" \
        > "$o/command.txt"
    say "$m-$s-$b: load ($sw)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $sw $bin/llama-server -m $file $md $ARGS -lv $LV \
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
            python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" "$TEXT" "$DEPTH" "$CANCEL_AT" > "$o/session-r$r.out" 2>&1
            say "  $m-$s-$b round $r: $(awk -F'\t' -v r="r$r" '$1 == r' "$o/session.tsv" | wc -l) requests, $(awk -F'\t' -v r="r$r" '$1 == r && $5 > 0' "$o/session.tsv" | wc -l) with tokens"
        done
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$m-$s-$b: $(cat "$o/load.txt") $(cat "$o/server.rc" 2>/dev/null), restores $(grep -ac 'restored context checkpoint' "$o/server.log"), full re-reads $(grep -ac 'forcing full prompt' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT' "$o/server.log")"
}

for m in $MODELS_RUN; do
    case $m in
        ornith) preset ornith-v5; file=$ORNITH; md="" ;;
        fn) preset flash-next-v5; file=$FN; md="-md $DRAFT" ;;
        ornmtp) preset ornith-v5; ARGS="$ARGS --spec-type draft-mtp --spec-draft-n-max 3"; file=$ORNITH; md="" ;;
        *) say "unknown model $m"; continue ;;
    esac
    for s in $SETTINGS; do
        case $s in
            def) sw="" ;;
            tail0) sw="LLAMA_KV_HEAD_TAIL=0" ;;
            mask) sw="LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1" ;;
            *) say "unknown setting $s"; continue ;;
        esac
        for b in $BUILDS; do
            case $b in
                base) arm "$m" "$s" "$b" "$REF" "$file" "$md" "$sw" ;;
                new) arm "$m" "$s" "$b" "$NEW" "$file" "$md" "$sw" ;;
                *) say "unknown build $b" ;;
            esac
        done
    done
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
