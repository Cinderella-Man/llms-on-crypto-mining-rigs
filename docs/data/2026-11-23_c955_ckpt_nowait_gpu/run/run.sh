#!/usr/bin/env bash
# C955 (Q1687 point 9): the GPU half of C907's check (fork branch pascal-p106-v5-ckpt-stall 0aebc4a57, one commit on
# pascal-p106-v5 3dfad876f: LLAMA_SERVER_CKPT_NOWAIT=1 fills a prompt's checkpoint that is taken between two of its
# calls into its device store without waiting for the cards: each row is copied on its card's stream behind the work
# already queued there, an event recorded after it, and a restore or a host copy waits for those events), on a build
# machine with three newer NVIDIA cards that run the rig build's compute 6.1 PTX. A processor runs every copy at once,
# so C907's own check could not reach the queued copies; the cards do. Correctness only, nothing timed.
#
# Each arm is one llama-server of Ornith-1.5-122B Pascal N2P GGUF with its v5 preset (release/ornith-v5/settings.env),
# the layer split changed to the three cards (-ts 1,1,1) and the log at verbosity 4 (-lv 4: each checkpoint's line,
# "no wait" where the switch filled it so). Arms (ARMS):
#   base-off        the base (3dfad876f, C908's rig tarball), the switch unset
#   head-off        the branch, the switch unset
#   head-on         the branch, LLAMA_SERVER_CKPT_NOWAIT=1
#   head-off-spill  the branch, the switch unset, --ctx-checkpoints 8 --ctx-checkpoints-vram 2 instead of the preset's
#                   4 and 4: older checkpoints are moved from the cards to the host, so a restore reads a host copy
#   head-on-spill   the same with LLAMA_SERVER_CKPT_NOWAIT=1: a host copy of a checkpoint filled without a wait
# In each arm client.py sends R183's prompts (8,077, 16,075, 32,025 and 47,012 tokens, greedy, cache_prompt false),
# then two restore pairs (a whole read, then the same messages with one user message changed, so the server restores
# the checkpoint taken between two calls near 8,200 or 41,000 tokens). compare.py then compares the arms.
#
# The replies run on the fork's Pascal matrix-vector kernels: they turn themselves on only on compute 6.1 cards, so
# they are forced on with a P106's launch shapes (5 SMs, 96 KB of shared memory per SM), as C953 ran them. The
# checkpoints' copies do not depend on them:
ON="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/ornith-v5/settings.env, this run/, R183's prompts/)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the loads as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18955), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), ARMS ("base-off head-off
#   head-on head-off-spill head-on-spill"), MAX_TOKENS (64), LOAD_TIMEOUT (2400 s), CLIENT_TIMEOUT (3600 s),
#   NAME (c955: the containers' name prefix)
# Output: WORK/load/<arm>/ (server.log, client.jsonl, client.out, command.txt, cards.tsv: every card's MiB at each
# phase), WORK/compare/ (compare.py's tables and server.tsv), WORK/md5.txt, WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${ORNITH:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18955}
ARMS=${ARMS:-base-off head-off head-on head-off-spill head-on-spill}
MAX_TOKENS=${MAX_TOKENS:-64}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-2400}
CLIENT_TIMEOUT=${CLIENT_TIMEOUT:-3600}
NAME=${NAME:-c955}
PREFLIGHT=${PREFLIGHT:-}
D=$REPO/docs/data/2026-11-23_c955_ckpt_nowait_gpu/run
PROMPTS=$REPO/docs/data/2026-11-20_r183_ornith_recall/prompts
URL=http://127.0.0.1:$PORT

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
md5sum "$NEW/llama-server" "$REF/llama-server" "$D"/* "$PROMPTS"/* "$REPO/release/ornith-v5/settings.env" > "$WORK/md5.txt" 2>&1
say "start: arms '$ARMS', NEW=$NEW REF=$REF, forced: $ON"

smi() {  # every card's memory use, "index:used/total" in MiB
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" |
        awk -F', *' '{printf "%s%s:%s/%s", (NR > 1 ? " " : ""), $1, $2, $3}'
}
mark() { printf '%s\t%s\n' "$1" "$(smi)" >> "$o/cards.tsv"; }
up() { curl -sf -m 5 "$URL/health" > /dev/null 2>&1; }
alive() { docker ps -q --filter "name=^$c\$" | grep -q .; }

preset() {  # PRESET_ENV and PRESET_ARGS of release/ornith-v5/settings.env: the split changed to the three cards, -lv 4
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1090
    . "$REPO/release/ornith-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--ctx-checkpoints 4 "* && "$ARGS" == *"--ctx-checkpoints-vram 4"* ]] ||
        { say "the preset's -ts or checkpoint switches not found"; exit 1; }
    ARGS="$ARGS -lv 4"
    SPILL=$(sed -E "s/--ctx-checkpoints 4 /--ctx-checkpoints 8 /; s/--ctx-checkpoints-vram 4/--ctx-checkpoints-vram 2/" <<< "$ARGS")
}

arm() {  # <arm> <bin folder> <server arguments> <extra environment>
    local a=$1 bin=$2 args=$3 env=$4 t0 up1=0
    o=$WORK/load/$a
    c=$NAME-$a
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s %s\nargs: -m %s %s\n' "$bin" "$PRESET_ENV" "$ON" "$env" "$ORNITH" "$args" > "$o/command.txt"
    mark before
    say "$a: start ($env)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $ON $env $bin/llama-server -m $ORNITH $args \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        up && { up1=1; break; }
        alive || break
        sleep 1
    done
    if [ $up1 = 1 ]; then
        mark up
        timeout "$CLIENT_TIMEOUT" python3 -I "$D/client.py" "$URL" "$PROMPTS" "$o/client.jsonl" "$MAX_TOKENS" > "$o/client.out" 2>&1
        say "$a: client rc=$?"
        mark replied
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    mark stopped
    say "$a: up=$up1 $(cat "$o/server.rc" 2>/dev/null), requests $(wc -l < "$o/client.jsonl" 2>/dev/null)," \
        "failed $(grep -c '"error"' "$o/client.jsonl" 2>/dev/null), checkpoints $(grep -ac 'created context checkpoint' "$o/server.log")," \
        "no wait $(grep -ac ', no wait,' "$o/server.log"), restored $(grep -ac 'restored context checkpoint' "$o/server.log")," \
        "moved $(grep -ac 'moved to the host' "$o/server.log"), asserts $(grep -ac 'GGML_ASSERT\|GGML_ABORT' "$o/server.log")"
}

[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }
preset
for a in $ARMS; do
    case $a in
        base-off)       arm "$a" "$REF" "$ARGS" "" ;;
        head-off)       arm "$a" "$NEW" "$ARGS" "" ;;
        head-on)        arm "$a" "$NEW" "$ARGS" "LLAMA_SERVER_CKPT_NOWAIT=1" ;;
        head-off-spill) arm "$a" "$NEW" "$SPILL" "" ;;
        head-on-spill)  arm "$a" "$NEW" "$SPILL" "LLAMA_SERVER_CKPT_NOWAIT=1" ;;
        *) say "unknown arm $a" ;;
    esac
done

mkdir -p "$WORK/compare"
python3 -I "$D/compare.py" "$WORK/load" "$WORK/compare" >"$WORK/compare/server.tsv" 2> "$WORK/compare/compare.err" ||
    say "compare rc=$?"
grep -a '^verdict' "$WORK/compare/server.tsv"
say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(smi)"
date -Is > "$WORK/DONE"
