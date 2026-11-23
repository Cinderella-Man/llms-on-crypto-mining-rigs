#!/usr/bin/env bash
# C962 (Q1692): why a sampled reply under pipelined speculation differs between loads with a fixed seed. In C950 Flint
# Pascal Q8 GGUF with its v5 preset (--pipe-depth 4) on three newer NVIDIA cards gave one of two 256-token replies to
# the same plain-text request at temperature 0.7, top-k 8, seed 7 (R14 of C950's session), on every build. Correctness
# only, nothing timed. Three steps:
#
# STEP=loads: one llama-server per arm, the base build (REF), Flint's v5 preset with the split changed to the three
#   cards (-ts 1,1,1); each load runs C950's session once (R14 in place, after R1-R13 with the prompt cache, as C950
#   sent it), then run/requests.py's plan on R14's prompt (run/r14_prompt.txt), every request fresh:
#     pipe-a, pipe-b, pipe-c   the preset as released (--pipe-depth 4), three loads
#     nopipe-a, nopipe-b       --pipe-depth 0 (batched decoding, the same backend sampling chain), two loads
#     p106k                    the preset with the fork's Pascal matrix-vector kernels forced on with a P106's launch
#                              shapes (they turn themselves on only on compute 6.1 cards), one load
# STEP=session: the same loads, each running run/session_r14.py once per entry of VARIANTS: C950's session from R1 (no
#   prompt cache) to R13, then R14 in place in one variant (pipe: as C950 sent it; probs: with n_probs 8 and
#   post_sampling_probs, so the dist step's probabilities in the session's own context; greedy: temperature 0, n_probs 8;
#   p95: top-p 0.95; k0: top-k 0)
#     spipe-a, spipe-b         the preset as released (--pipe-depth 4)
#     snopipe-a                --pipe-depth 0
#   and with ARMS="spipe-c spipe-d" VARIANTS="pipe p95 k0 pipe p95 k0 pipe p95 k0 pipe p95 k0": R14 as sent, with top-p
#   0.95 (sorted candidates) and with top-k 0 (no top-k), alternated on two more loads
# STEP=probe: run/topk_order_probe.cu built in a CUDA 12.9 container (PTX for compute 6.1 only, as the rig build) and run
#   on each card, in each of MODES, on every input file in $WORK/probe/*.in (written from the loads' n_probs replies by run/c962_tables.py
#   --probe-inputs): the card's top-k of one row, ITERS times, its candidate orders counted.
#
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver-$STEP.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   REF       bin/ folder of the rig build (the base, 3dfad876f)
#   REPO      a copy of this repository (release/flint-v5/settings.env, this run/, C950's run/session.py)
#   FLINT     Flint Pascal Q8 GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the build, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before the cards are used, as "$PREFLIGHT -g <ids>" (optional)
#   LOCK      a lock file the probe's compile holds (optional; flock)
#   STEP (loads), ARMS ("pipe-a pipe-b pipe-c nopipe-a nopipe-b p106k"; STEP=session: "spipe-a spipe-b snopipe-a"),
#   PLAN (requests.py's default), VARIANTS ("pipe probs greedy probs pipe probs"), GPUS (0,1,2),
#   TS (1,1,1), PORT (18962), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04), LV (4), RUN (out), LOAD_TIMEOUT (1800 s),
#   NAME (c962: the containers' name prefix), ITERS (2000: probe calls per card, input and mode), MODES ("plain rewrite
#   busy": the probe's modes)
# Output: WORK/RUN/<arm>/ with command.txt, server.log, load.txt, session.tsv and its replies (s1-R*.json),
#   requests.tsv and its replies (<kind>-<n>.json), mem.csv; with STEP=session session_r14.tsv and its replies
#   (v<N>-R*.json); WORK/probe/ with the probe's build log and
#   <input>.<mode>.gpu<N>.txt; WORK/DONE-<step> at the end.
set -uo pipefail
: "${WORK:?}" "${REF:?}" "${REPO:?}" "${FLINT:?}"
STEP=${STEP:-loads}
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
TS=${TS:-1,1,1}
PORT=${PORT:-18962}
if [ "$STEP" = session ]; then ARMS=${ARMS:-spipe-a spipe-b snopipe-a}; else ARMS=${ARMS:-pipe-a pipe-b pipe-c nopipe-a nopipe-b p106k}; fi
VARIANTS=${VARIANTS:-pipe probs greedy probs pipe probs}
PLAN=${PLAN:-}
LV=${LV:-4}
RUN=${RUN:-out}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-1800}
NAME=${NAME:-c962}
ITERS=${ITERS:-2000}
MODES=${MODES:-plain rewrite busy}
PREFLIGHT=${PREFLIGHT:-}
LOCK=${LOCK:-}
D=$REPO/docs/data/2026-11-23_c962_sampled_reply_gpu/run
SESSION=$REPO/docs/data/2026-11-21_c950_flint_slot_plan_gpu/run/session.py
# the fork's Pascal matrix-vector kernels forced on with a P106's launch shapes (5 SMs, 96 KB shared memory per SM)
P106K="GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304"

say() { echo "$(date +%H:%M:%S) $*"; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every load (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache" "$WORK/$RUN" "$WORK/probe"
rm -f "$WORK/DONE-$STEP"

if [ "$STEP" = probe ]; then
    say "probe: build"
    md5sum "$D/topk_order_probe.cu" > "$WORK/probe/md5.txt"
    # shellcheck disable=SC2086
    ${LOCK:+flock "$LOCK"} "${DR[@]}" "$IMG" bash -c "cd $WORK/probe && \
        nvcc -O2 -std=c++17 -gencode arch=compute_61,code=compute_61 -o topk_order_probe $D/topk_order_probe.cu" \
        > "$WORK/probe/build.log" 2>&1
    say "probe: build rc=$? ($(wc -l < "$WORK/probe/build.log") log lines)"
    [ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/probe/preflight.txt" 2>&1; say "preflight rc=$?"; }
    shopt -s nullglob
    for in in "$WORK"/probe/*.in; do
        b=$(basename "$in" .in)
        for m in $MODES; do
            for g in ${GPUS//,/ }; do
                "${DR[@]}" --gpus "\"device=$g\"" "$IMG" timeout 600 "$WORK/probe/topk_order_probe" "$in" "$ITERS" 0 "$m" \
                    > "$WORK/probe/$b.$m.gpu$g.txt" 2>&1
                say "probe $b $m on card $g: rc=$? $(grep -a 'distinct orders' "$WORK/probe/$b.$m.gpu$g.txt")"
            done
        done
    done
    date -Is > "$WORK/DONE-$STEP"
    exit 0
fi

md5sum "$REF/llama-server" "$FLINT.sha256" "$SESSION" "$D/requests.py" "$D/session_r14.py" "$D/r14_prompt.txt" > "$WORK/md5-$STEP.txt" 2>&1
# the preset, the split changed to the three cards
unset PRESET_MODEL PRESET_ENV PRESET_ARGS
# shellcheck disable=SC1091
. "$REPO/release/flint-v5/settings.env"
ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
[[ "$ARGS" == *"-ts $TS"* && "$ARGS" == *"--pipe-depth 4"* ]] || { say "no -ts or --pipe-depth 4 in the preset"; exit 1; }
say "start: arms '$ARMS', cards $GPUS, REF=$REF"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader -i "$GPUS" | tr '\n' ' '; echo
[ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/preflight.txt" 2>&1; say "preflight rc=$?"; }

arm() {  # <arm> <extra env> <args>
    local a=$1 sw=$2 args=$3 o=$WORK/$RUN/$1 c=$NAME-$1 up=0 t0
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s\nargs: -m %s %s -lv %s\n' "$REF" "$PRESET_ENV" "$sw" "$FLINT" "$args" "$LV" > "$o/command.txt"
    say "$a: load ($sw | ${args##*--pipe-depth })"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $sw $REF/llama-server -m $FLINT $args -lv $LV \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up" > "$o/load.txt"
    if [ $up = 1 ] && [ "$STEP" = session ]; then
        local i=0 v
        for v in $VARIANTS; do
            i=$((i + 1))
            timeout 1800 python3 -I "$D/session_r14.py" "http://127.0.0.1:$PORT" "$SESSION" "$o" "v$i" "$v" >> "$o/session_r14.out" 2>&1
            say "  $a v$i $v: R14 $(awk -F'\t' -v t="v$i" '$1 == t && $3 == "R14" {print $7}' "$o/session_r14.tsv")"
        done
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
    elif [ $up = 1 ]; then
        timeout 1800 python3 -I "$SESSION" "http://127.0.0.1:$PORT" "$o" s1 > "$o/session-s1.out" 2>&1
        say "  $a session: $(wc -l < "$o/session.tsv") requests, R14 $(awk -F'\t' '$2 == "R14" {print $8}' "$o/session.tsv")"
        # shellcheck disable=SC2086
        timeout 3600 python3 -I "$D/requests.py" "http://127.0.0.1:$PORT" "$D/r14_prompt.txt" "$o" $PLAN \
            > "$o/requests.out" 2>&1
        say "  $a requests: $(wc -l < "$o/requests.tsv") sent; k8 lists: $(awk -F'\t' '$1 == "k8" {print $6}' "$o/requests.tsv" | sort | uniq -c | tr '\n' ' ')"
        nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits -i "$GPUS" > "$o/mem.csv"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$a: $(cat "$o/load.txt"), pipe line: $(grep -a -m1 'pipelined speculation: up to' "$o/server.log" | sed 's/.*pipelined speculation: //'), top-k rows line $(grep -ac 'one top-k over the output rows' "$o/server.log")"
}

NOPIPE=${ARGS/--pipe-depth 4/--pipe-depth 0}
for a in $ARMS; do
    case $a in
        pipe-*|spipe-*)     arm "$a" "" "$ARGS" ;;
        nopipe-*|snopipe-*) arm "$a" "" "$NOPIPE" ;;
        p106k)    arm "$a" "$P106K" "$ARGS" ;;
        *) say "unknown arm $a" ;;
    esac
done

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE-$STEP"
