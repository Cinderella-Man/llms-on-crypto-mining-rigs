#!/usr/bin/env bash
# C947 (Q1687): the GPU half of the merged v5 engine's check (`pascal-p106-v5` 3dfad876f, C908), on a build machine with
# three newer NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed. Three steps:
#   mbox   test-hop-mailbox on all three cards: the test's nine settings with the hop gate unset, then four gate arms
#          (as pull request 4's review ran them)
#   load   each release file with its v5 preset, every switch the preset does not set at its default, the layer split
#          changed to the three cards (-ts 1,1,1): one llama-server per build (NEW, then REF), memory per card 20 s
#          after it answered, two greedy requests (/completion, temperature 0, top-k 1, 256 tokens, no prompt cache,
#          the tokens returned): C854b's coding request and prose paragraph
#   bops   test-backend-ops test -b CUDA0, every case except the repeated-expert-id ones (dup=1), split over the three
#          cards by operator: MUL_MAT, MUL_MAT_ID, and every other operator the support listing names
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the merged engine, REF = 602691153)
#   HOPTEST   NEW's test-hop-mailbox binary
#   REPO      a copy of this repository (release/*-v5/settings.env)
#   FN        Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's first shard; DRAFT its MTP draft
#   FLINT_Q8  Flint Pascal Q8 GGUF; ORNITH Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the models)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before each GPU step as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), TS (1,1,1), PORT (18947), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS ("mbox load bops"),
#   MODELS_RUN ("flashnext flintq8 ornith"), LOAD_TIMEOUT (1800 s), NAME (c947: the containers' name prefix)
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${HOPTEST:?}" "${REPO:?}" "${FN:?}" "${DRAFT:?}" "${FLINT_Q8:?}" "${ORNITH:?}"
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18947}
STEPS=${STEPS:-mbox load bops}
MODELS_RUN=${MODELS_RUN:-flashnext flintq8 ornith}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c947}
PREFLIGHT=${PREFLIGHT:-}
IFS=, read -ra G <<< "$GPUS"
[ ${#G[@]} = 3 ] || { echo "GPUS must name three cards"; exit 1; }

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
strip() { sed 's/\x1b\[[0-9;]*m//g' "$1"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every step (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
md5sum "$NEW/llama-server" "$NEW/test-backend-ops" "$HOPTEST" "$REF/llama-server" > "$WORK/md5.txt"
say "start: steps '$STEPS', cards $GPUS"

if has mbox; then
    o=$WORK/mbox; mkdir -p "$o"; : > "$o/rc.txt"
    [ -n "$PREFLIGHT" ] && $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    for arm in off g16-b8192 g16-b4096 g1-b64 g64-b4096-r480; do
        e="X=1"; r=96
        case $arm in
            g16-b8192) e="GGML_CUDA_HOP_MAILBOX_GATE_KB=16 GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB=8192" ;;
            g16-b4096) e="GGML_CUDA_HOP_MAILBOX_GATE_KB=16" ;;
            g1-b64) e="GGML_CUDA_HOP_MAILBOX_GATE_KB=1 GGML_CUDA_HOP_MAILBOX_GATE_BUF_KB=64" ;;
            g64-b4096-r480) e="GGML_CUDA_HOP_MAILBOX_GATE_KB=64"; r=480 ;;
        esac
        "${DR[@]}" --gpus "\"device=$GPUS\"" "$IMG" \
            bash -c "env $e nice -n 19 timeout 1800 $HOPTEST --rounds $r > $o/$arm.log 2>&1; echo $arm rc=\$? >> $o/rc.txt" \
            > /dev/null 2>&1
        say "mbox $arm: $(tail -n 1 "$o/rc.txt")"
    done
fi

P_CODE='Write a Python function that parses an ISO 8601 date string into a datetime object, then three pytest tests for it.'
P_PROSE='Explain in one paragraph how a hash map handles collisions, then give one example in Python.'

model() {  # <name>: sets M_FILE M_EXTRA M_ENV M_ARGS from the model's v5 preset, the split changed to $TS
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    case $1 in
        # shellcheck disable=SC1091
        flashnext) . "$REPO/release/flash-next-v5/settings.env"; M_FILE=$FN; M_EXTRA="-md $DRAFT" ;;
        # shellcheck disable=SC1091
        flintq8) . "$REPO/release/flint-v5/settings.env"; M_FILE=$FLINT_Q8; M_EXTRA="" ;;
        # shellcheck disable=SC1091
        ornith) . "$REPO/release/ornith-v5/settings.env"; M_FILE=$ORNITH; M_EXTRA="" ;;
        *) say "unknown model $1"; return 1 ;;
    esac
    M_ENV=$PRESET_ENV
    M_ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$M_ARGS" == *"-ts $TS"* ]] || { say "$1: no -ts in the preset to change"; return 1; }
}

load() {  # <model> <label> <bin folder>
    local o=$WORK/load/$1-$2 c=$NAME-$1-$2 up=0 t0 p body
    mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'env: %s\nargs: -m %s %s %s\n' "$M_ENV" "$M_FILE" "$M_EXTRA" "$M_ARGS" > "$o/command.txt"
    say "load $1-$2: $3"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $M_ENV $3/llama-server -m $M_FILE $M_EXTRA $M_ARGS --host 0.0.0.0 --port $PORT --metrics --no-webui \
            > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up after $(( $(date +%s) - t0 )) s" > "$o/load.txt"
    if [ $up = 1 ]; then
        sleep 20
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
        for p in code prose; do
            if [ $p = code ]; then body=$P_CODE; else body=$P_PROSE; fi
            printf '{"prompt": "%s", "n_predict": 256, "temperature": 0, "top_k": 1, "seed": 1, "cache_prompt": false, "return_tokens": true}' "$body" |
                curl -s -m 900 -H 'Content-Type: application/json' -d @- "http://127.0.0.1:$PORT/completion" \
                > "$o/reply-$p.json" 2> "$o/reply-$p.err"
            say "  $1-$2 $p: $(wc -c < "$o/reply-$p.json") B of JSON"
        done
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem-after.csv"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "load $1-$2: $(cat "$o/load.txt")"
}

if has load; then
    [ -n "$PREFLIGHT" ] && $PREFLIGHT -g "$GPUS" > "$WORK/load-preflight.txt" 2>&1
    for m in $MODELS_RUN; do
        model "$m" || continue
        load "$m" new "$NEW"
        load "$m" ref "$REF"
    done
fi

if has bops; then
    o=$WORK/bops; mkdir -p "$o"; : > "$o/rc.txt"
    F='^(?!.*dup=1)'
    [ -n "$PREFLIGHT" ] && $PREFLIGHT -g "$GPUS" > "$o/preflight.txt" 2>&1
    "${DR[@]}" --gpus "\"device=${G[2]}\"" "$IMG" \
        bash -c "nice -n 19 timeout 900 $NEW/test-backend-ops support -b CUDA0 -p '$F' > $o/support.txt 2>&1" > /dev/null 2>&1
    strip "$o/support.txt" | sed -n 's/^  \([A-Za-z0-9_]*\)(.*/\1/p' | sort -u > "$o/ops.txt"
    strip "$o/support.txt" | grep -c ': SUPPORTED$' > "$o/n_supported.txt"
    REST=$(grep -vx -e MUL_MAT -e MUL_MAT_ID "$o/ops.txt" | paste -sd,)
    say "bops: $(wc -l < "$o/ops.txt") operators, $(cat "$o/n_supported.txt") supported cases"
    run_bops() {  # <card> <tag> <operators>
        "${DR[@]}" --gpus "\"device=$1\"" "$IMG" \
            bash -c "nice -n 19 timeout 10800 $NEW/test-backend-ops test -b CUDA0 -o '$3' -p '$F' > $o/$2.log 2>&1; echo $2 rc=\$? >> $o/rc.txt" \
            > /dev/null 2>&1
        say "bops $2 done: $(grep "^$2 " "$o/rc.txt")"
    }
    run_bops "${G[0]}" mul_mat MUL_MAT &
    run_bops "${G[1]}" mul_mat_id MUL_MAT_ID &
    run_bops "${G[2]}" rest "$REST" &
    wait
    : > "$o/summary.txt"
    for t in mul_mat mul_mat_id rest; do
        f=$o/$t.log
        echo "$t: $(grep "^$t " "$o/rc.txt"); $(strip "$f" | grep -a -m1 'tests passed' | sed 's/^ *//'); OK $(strip "$f" | grep -a -c ': OK$'); FAIL $(strip "$f" | grep -a -c ': FAIL'); not supported $(strip "$f" | grep -a -c 'not supported'); illegal $(strip "$f" | grep -a -c 'illegal memory access')" >> "$o/summary.txt"
        strip "$f" | grep -a ': FAIL' | grep -av 'Backend' | sed 's/^ */    /' | cut -c1-220 >> "$o/summary.txt"
    done
fi

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
