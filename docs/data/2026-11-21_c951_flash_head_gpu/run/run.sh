#!/usr/bin/env bash
# C951 (Q1687): the GPU half of C898's check (fork branch pascal-p106-v5-mtp-plain-attn c9c3134d4, one commit on
# pascal-p106-v5 602691153: the MTP draft gets the sparse attention's indexer cache and k-pool inputs only when its own
# block is a sparse-attention block, compress ratio above 0; a block at ratio 0 attends to every cell), on a build
# machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# Qwen3.8-Flash-Next with its v5 preset (release/flash-next-v5/settings.env, --spec-type draft-mtp), the layer split
# changed to the three cards (-ts 1,1,1), -lv 4 (libllama's info lines, such as each "creating indexer KV cache", are
# verbosity 4 in this fork). One llama-server per arm, in a container:
#   own-new    NEW, the MTP head OWN (a head whose block is a sparse-attention block, as the preset's)
#   own-ref    REF (602691153), the same head: the unchanged path must give the same replies and drafts
#   dist-new   NEW, the published distilled head DIST (no token table or output head of its own, its block at ratio 0)
#              with LLAMA_DRAFT_SHARE=embd,head (it reads the target's)
#   plain-new  NEW, no draft (every --spec-* flag and -md removed): plain decoding, the reference for greedy replies
#   dist-ref   REF with DIST, as FN115 ran it on the P106 rig: expected to stop on GGML_ASSERT(buffer) in the draft's
#              first pass (a k-pool input nothing reads); LOAD_TIMEOUT_FAIL caps its wait
# Each loaded arm answers run/session.py's requests ROUNDS times (2), then the memory per card is read.
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/flash-next-v5/settings.env, docs/switches.md and this folder's run/)
#   OWN, DIST the two MTP heads
#   MODEL_maker, MODEL_pmix  first shards of the target files (the others beside them), as MODELS_RUN names them
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the models)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the first load as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18951), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), MODELS_RUN ("maker pmix"),
#   ARMS ("own-new own-ref dist-new plain-new dist-ref"), ROUNDS (2), LV (4), DEPTH (8192), RUN (out),
#   LOAD_TIMEOUT (1800 s), LOAD_TIMEOUT_FAIL (900 s, dist-ref), NAME (c951: the containers' name prefix)
# Output: WORK/RUN/<model>-<arm>/ with command.txt, server.log, server.rc, load.txt, session.tsv and the replies,
#   mem-load.csv and mem.csv (MiB per card after the load and after the requests); WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${OWN:?}" "${DIST:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18951}
MODELS_RUN=${MODELS_RUN:-maker pmix}
ARMS=${ARMS:-own-new own-ref dist-new plain-new dist-ref}
ROUNDS=${ROUNDS:-2}
LV=${LV:-4}
DEPTH=${DEPTH:-8192}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
LOAD_TIMEOUT_FAIL=${LOAD_TIMEOUT_FAIL:-900}
NAME=${NAME:-c951}
PREFLIGHT=${PREFLIGHT:-}
SESSION=$REPO/docs/data/2026-11-21_c951_flash_head_gpu/run/session.py
TEXT=$REPO/docs/switches.md
SHARE=LLAMA_DRAFT_SHARE=embd,head

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
md5sum "$NEW/llama-server" "$REF/llama-server" "$OWN" "$DIST" "$SESSION" "$TEXT" > "$WORK/md5.txt" 2>&1

# the preset, the split changed to the three cards; the plain arm without the draft's flags
unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1091
. "$REPO/release/flash-next-v5/settings.env"
ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
[[ "$ARGS" == *"-ts $TS"* ]] || { say "no -ts in the preset to change"; exit 1; }
PLAIN_ARGS=$(sed -E 's/ --spec-[a-z-]+ [^ ]+//g' <<< " $ARGS")
[[ "$PLAIN_ARGS" != *"--spec-"* ]] || { say "a --spec- flag is left in the plain arm"; exit 1; }
say "start: models '$MODELS_RUN', arms '$ARMS', cards $GPUS, NEW=$NEW REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }

mem() { nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$1"; }

arm() {  # <model> <arm> <bin folder> <switches> <draft args> <server args> <load timeout>
    local m=$1 a=$2 bin=$3 sw=$4 md=$5 args=$6 lt=$7 o=$WORK/$RUN/$1-$2 c=$NAME-$1-$2 up=0 t0 r file
    file=MODEL_$m; file=${!file:?MODEL_$m is not set}
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s %s -lv %s\n' "$bin" "$PRESET_ENV" "$sw" "$file" "$md" "$args" "$LV" \
        > "$o/command.txt"
    say "$m-$a: load ($sw $md)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $sw $bin/llama-server -m $file $md $args -lv $LV \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$lt" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up" > "$o/load.txt"
    if [ $up = 1 ]; then
        mem "$o/mem-load.csv"
        for r in $(seq 1 "$ROUNDS"); do
            python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" "r$r" "$TEXT" "$DEPTH" > "$o/session-r$r.out" 2>&1
            say "  $m-$a round $r: $(awk -F'\t' -v r="r$r" '$1 == r' "$o/session.tsv" | wc -l) requests, $(awk -F'\t' -v r="r$r" '$1 == r && $4 > 0' "$o/session.tsv" | wc -l) with tokens"
        done
        mem "$o/mem.csv"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$m-$a: $(cat "$o/load.txt") $(cat "$o/server.rc" 2>/dev/null), indexer cache lines $(grep -ac 'creating indexer KV cache' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT' "$o/server.log")"
}

for m in $MODELS_RUN; do
    for a in $ARMS; do
        case $a in
            own-new) arm "$m" "$a" "$NEW" "" "-md $OWN" "$ARGS" "$LOAD_TIMEOUT" ;;
            own-ref) arm "$m" "$a" "$REF" "" "-md $OWN" "$ARGS" "$LOAD_TIMEOUT" ;;
            dist-new) arm "$m" "$a" "$NEW" "$SHARE" "-md $DIST" "$ARGS" "$LOAD_TIMEOUT" ;;
            plain-new) arm "$m" "$a" "$NEW" "" "" "$PLAIN_ARGS" "$LOAD_TIMEOUT" ;;
            dist-ref) arm "$m" "$a" "$REF" "$SHARE" "-md $DIST" "$ARGS" "$LOAD_TIMEOUT_FAIL" ;;
            *) say "unknown arm $a" ;;
        esac
    done
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
