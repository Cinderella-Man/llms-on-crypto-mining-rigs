#!/usr/bin/env bash
# C953 (Q1687 point 7): the GPU half of C899's check (fork branch pascal-p106-v5-kcol-ornith 37c05b76a, one commit on
# pascal-p106-v5 3dfad876f: GGML_CUDA_MMVQ_PASCAL_MT=2 sends only the plain Q5_0 and Q8_0 products of 2 to 4 tokens
# that a P106 ran faster on the read-once kernel mul_mat_vec_q_pascal_mt to it), on a build machine with three newer
# NVIDIA cards that run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
#
# The fork's Pascal matrix-vector kernels (mul_mat_vec_q_pascal_mc, _mt and the k-column kernels) turn themselves on
# only on compute 6.1 cards; GGML_CUDA_MMVQ_PASCAL=2 turns them on for any NVIDIA card, and the switch under test acts
# only inside them. So every step runs with them forced on and a P106's launch shapes (5 SMs, 96 KB of shared memory per
# SM), from scripts/pascal_forced.sh (sourced below): FORCED=all (the default) is PASCAL_FORCED, every compute-6.1-only
# path of the fork (also the attention block, the persistent MoE kernel and the Pascal flash-attention tile kernel, as a
# P106 runs them); FORCED=mmv is PASCAL_FORCED_MMV, the matrix-vector kernels alone, the set C953's run used.
# Two steps (STEPS, default "bops load"):
#   bops   test-backend-ops on one card (BOPS_GPU), two parts:
#            suite   the built-in plain MUL_MAT cases of Q5_0 and Q8_0 at 1 to 5 tokens (-p "$SUITE_P")
#            ornith  run/shapes.py's file: Ornith-1.5-122B Pascal N2P GGUF's 9 dense Q5_0/Q8_0 shapes and 7 shapes at
#                    the bounds of the rule, each at 1 to 5 tokens
#          Each part: (1) a record (--record: the inputs and the CPU reference written, and the live check run with the
#          switch unset), (2) the record replayed with the switch unset and with =2 (--replay, --replay-log: per case
#          the verdict and the normalized squared error against the CPU), so both settings compute the same inputs and
#          their errors compare case by case, (3) a live run with =2 (fresh inputs). A case whose kernel the switch
#          changes shows another error with =2; a case it leaves alone the same error.
#   load   llama-server with Ornith's v5 preset (release/ornith-v5/settings.env, the layer split changed to the three
#          cards, -ts 1,1,1) on GPUS, one load per arm; models:
#            ornmtp  the preset with the MTP layer in Ornith's file drafting three tokens a step (--spec-type draft-mtp
#                    --spec-draft-n-max 3): each step verifies 2 to 4 tokens, the products the switch is about
#            ornith  the preset as it is (one token a step; 2 to 4 tokens only where a prompt's last micro-batch is
#                    that short)
#          each with builds/settings: new-unset, new-mt2 (GGML_CUDA_MMVQ_PASCAL_MT=2), base-unset (3dfad876f, C908's
#          rig tarball). run/replies.py's requests on each load.
#   kernels  (after bops, not in the default STEPS) bops' two records replayed on BOPS_GPU with three more settings, so
#          each case's error shows which kernel family computed it in the unset and =2 runs:
#            mtk0   GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL=0: every plain Q5_0 and Q8_0 product of 2 to 4
#                   tokens on the read-once kernel (=1 alone leaves the k-column kernels' products to them)
#            kcol0  GGML_CUDA_MMVQ_PASCAL_KCOL=0: no k-column kernel (the products of 2 to 4 tokens on
#                   mul_mat_vec_q_pascal_mc)
#            kcol2  GGML_CUDA_MMVQ_PASCAL_KCOL=2: every product of 1 to 8 tokens of these types on the k-column kernels
#                   (no table)
# Run it on the build machine itself, detached: setsid nohup bash run.sh > $WORK/driver.log 2>&1 < /dev/null &
# Env:
#   WORK      a folder for the outputs (mounted read-write into every container)
#   NEW, REF  bin/ folders of the two rig builds (NEW = the branch, REF = the base)
#   REPO      a copy of this repository (release/ornith-v5/settings.env and this run/)
#   ORNITH    Ornith-1.5-122B Pascal N2P GGUF
#   MOUNTS    folders the containers mount read-only at the same paths (the builds, the model)
#   DOCKER_ARGS  extra `docker run` arguments (labels, --cpu-shares) the machine's policy asks for
#   PREFLIGHT    a CUDA check run before each step as "$PREFLIGHT -g <ids>" (optional)
#   GPUS (0,1,2), BOPS_GPU (0), TS (1,1,1), PORT (18953), IMG (nvidia/cuda:12.9.2-devel-ubuntu24.04),
#   STEPS ("bops load"), MODELS_RUN ("ornmtp ornith"), ARMS ("new-unset new-mt2 base-unset"), ROUNDS (2), FORCED (all),
#   LOAD_TIMEOUT (2400 s), NAME (c953: the containers' name prefix), WAIT_LABEL (a container label: before the load
#   step, wait up to 2 h while containers carrying it run; C953 waited for a code review's containers on one card)
# Output: WORK/bops/ (logs, replay logs, rc.txt), WORK/load/<model>-<arm>/ (command.txt, server.log, server.rc,
#   load.txt, replies.tsv and the replies), WORK/DONE at the end.
set -uo pipefail
: "${WORK:?}" "${NEW:?}" "${REF:?}" "${REPO:?}" "${ORNITH:?}"
# shellcheck source=../../../../scripts/pascal_forced.sh
. "$REPO/scripts/pascal_forced.sh"
case ${FORCED:-all} in
    all) ON=$PASCAL_FORCED ;;
    mmv) ON=$PASCAL_FORCED_MMV ;;
    *) echo "FORCED must be all or mmv"; exit 1 ;;
esac
[ -n "$ON" ] || { echo "no forced set in $REPO/scripts/pascal_forced.sh"; exit 1; }
IMG=${IMG:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
GPUS=${GPUS:-0,1,2}
BOPS_GPU=${BOPS_GPU:-0}
TS=${TS:-1,1,1}
PORT=${PORT:-18953}
STEPS=${STEPS:-bops load}
MODELS_RUN=${MODELS_RUN:-ornmtp ornith}
ARMS=${ARMS:-new-unset new-mt2 base-unset}
ROUNDS=${ROUNDS:-2}
LOAD_TIMEOUT=${LOAD_TIMEOUT:-2400}
NAME=${NAME:-c953}
PREFLIGHT=${PREFLIGHT:-}
WAIT_LABEL=${WAIT_LABEL:-}
D=$REPO/docs/data/2026-11-23_c953_ornith_kcol_gpu/run
SUITE_P='^type_a=q[58]_0,type_b=f32,m=[0-9]+,n=[1-5],'

say() { echo "$(date +%H:%M:%S) $*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
VOL=()
for m in ${MOUNTS:-}; do VOL+=(-v "$m:$m:ro"); done
VOL+=(-v "$WORK:$WORK")
# the PTX JIT's cache, shared by every container (the cards compile the build's compute 6.1 PTX once)
JIT=(-e "CUDA_CACHE_PATH=$WORK/jitcache" -e CUDA_CACHE_MAXSIZE=4294967296)
# shellcheck disable=SC2206
DR=(docker run --rm ${DOCKER_ARGS:-} "${VOL[@]}" "${JIT[@]}")
mkdir -p "$WORK/jitcache"
rm -f "$WORK/DONE"
md5sum "$NEW/llama-server" "$NEW/test-backend-ops" "$REF/llama-server" "$D"/* > "$WORK/md5.txt" 2>&1
say "start: steps '$STEPS', NEW=$NEW REF=$REF, forced: $ON"

tbo() {  # <tag> <switch (K=V or "")> <test-backend-ops arguments>: one run in a container on BOPS_GPU
    local tag=$1 sw=$2
    shift 2
    printf 'env: %s %s\nargs: test -b CUDA0 %s\n' "$ON" "$sw" "$*" > "$o/$tag.cmd"
    # shellcheck disable=SC2086
    "${DR[@]}" --name "$NAME-bops-$tag" --gpus "\"device=$BOPS_GPU\"" "$IMG" \
        bash -c "env $ON $sw timeout 3600 $NEW/test-backend-ops test -b CUDA0 $* > $o/$tag.log 2>&1; echo $tag rc=\$? >> $o/rc.txt" \
        > /dev/null 2>&1
    say "  bops $tag: $(tail -n 1 "$o/rc.txt"); $(grep -a -E 'tests passed' "$o/$tag.log" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g')"
}

if has bops; then
    o=$WORK/bops
    rm -rf "$o"; mkdir -p "$o"
    python3 -I "$D/shapes.py" > "$o/shapes.txt"
    [ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$BOPS_GPU" > "$o/preflight.txt" 2>&1; say "bops preflight rc=$?"; }
    for part in suite ornith; do
        if [ $part = suite ]; then
            sel="-o MUL_MAT -p '$SUITE_P'"
        else
            sel="-o MUL_MAT --test-file $o/shapes.txt"
        fi
        tbo "$part-record" "" "$sel --record $o/rec-$part"
        tbo "$part-replay-unset" "" "$sel --replay $o/rec-$part --replay-log $o/$part-unset.tsv"
        tbo "$part-replay-mt2" "GGML_CUDA_MMVQ_PASCAL_MT=2" "$sel --replay $o/rec-$part --replay-log $o/$part-mt2.tsv"
        tbo "$part-live-mt2" "GGML_CUDA_MMVQ_PASCAL_MT=2" "$sel"
    done
    du -sh "$o"/rec-* > "$o/rec-size.txt" 2>&1
fi

if has kernels; then
    o=$WORK/bops
    [ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$BOPS_GPU" > "$o/preflight-kernels.txt" 2>&1; say "kernels preflight rc=$?"; }
    for part in suite ornith; do
        if [ $part = suite ]; then
            sel="-o MUL_MAT -p '$SUITE_P'"
        else
            sel="-o MUL_MAT --test-file $o/shapes.txt"
        fi
        for s in "mtk0|GGML_CUDA_MMVQ_PASCAL_MT=1 GGML_CUDA_MMVQ_PASCAL_KCOL=0" "kcol0|GGML_CUDA_MMVQ_PASCAL_KCOL=0" \
                 "kcol2|GGML_CUDA_MMVQ_PASCAL_KCOL=2"; do
            tbo "$part-replay-${s%%|*}" "${s#*|}" "$sel --replay $o/rec-$part --replay-log $o/$part-${s%%|*}.tsv"
        done
    done
fi

preset() {  # PRESET_ENV and ARGS of release/ornith-v5/settings.env, the split changed to the three cards
    unset PRESET_MODEL PRESET_DRAFT PRESET_ENV PRESET_ARGS
    # shellcheck disable=SC1090
    . "$REPO/release/ornith-v5/settings.env"
    ARGS=$(sed -E "s/-ts [0-9,]+/-ts $TS/" <<< "$PRESET_ARGS")
    [[ "$ARGS" == *"-ts $TS"* ]] || { say "no -ts in the preset to change"; exit 1; }
}

arm() {  # <model> <arm> <bin folder> <switch>
    local m=$1 a=$2 bin=$3 sw=$4 o=$WORK/load/$1-$2 c=$NAME-$1-$2 up=0 t0
    rm -rf "$o"; mkdir -p "$o"
    docker rm -f "$c" > /dev/null 2>&1
    printf 'bin: %s\nenv: %s %s %s\nargs: -m %s %s\n' "$bin" "$PRESET_ENV" "$ON" "$sw" "$ORNITH" "$ARGS" > "$o/command.txt"
    say "$m-$a: load ($sw)"
    # shellcheck disable=SC2086
    "${DR[@]}" -d --name "$c" --gpus "\"device=$GPUS\"" -p "127.0.0.1:$PORT:$PORT" "$IMG" \
        bash -c "env $PRESET_ENV $ON $sw $bin/llama-server -m $ORNITH $ARGS \
            --host 0.0.0.0 --port $PORT --no-webui > $o/server.log 2>&1; echo rc=\$? > $o/server.rc" > /dev/null
    t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$LOAD_TIMEOUT" ]; do
        if curl -sf -m 5 "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; then up=1; break; fi
        docker ps -q --filter "name=^$c\$" | grep -q . || break
        sleep 5
    done
    echo "up=$up" > "$o/load.txt"
    if [ $up = 1 ]; then
        python3 -I "$D/replies.py" "http://127.0.0.1:$PORT" "$o" "$ROUNDS" > "$o/replies.out" 2>&1
        say "  $m-$a: $(wc -l < "$o/replies.tsv") requests, $(awk -F'\t' '$3 > 0' "$o/replies.tsv" | wc -l) with tokens"
    fi
    docker stop -t 60 "$c" > /dev/null 2>&1
    docker rm -f "$c" > /dev/null 2>&1
    say "$m-$a: $(cat "$o/load.txt") $(cat "$o/server.rc" 2>/dev/null), asserts $(grep -ac 'GGML_ASSERT' "$o/server.log")"
}

if has load; then
    if [ -n "$WAIT_LABEL" ]; then  # another job's containers on the cards: wait for them to end, never stop them
        t0=$(date +%s)
        while docker ps -q --filter "label=$WAIT_LABEL" | grep -q . && [ $(( $(date +%s) - t0 )) -lt 7200 ]; do sleep 20; done
        say "waited $(( $(date +%s) - t0 )) s for containers labelled $WAIT_LABEL; left: $(docker ps -q --filter "label=$WAIT_LABEL" | wc -l)"
    fi
    [ -n "$PREFLIGHT" ] && { $PREFLIGHT -g "$GPUS" > "$WORK/load-preflight.txt" 2>&1; say "load preflight rc=$?"; }
    for m in $MODELS_RUN; do
        preset
        case $m in
            ornith) ;;
            ornmtp) ARGS="$ARGS --spec-type draft-mtp --spec-draft-n-max 3" ;;
            *) say "unknown model $m"; continue ;;
        esac
        for a in $ARMS; do
            case $a in
                new-unset) arm "$m" "$a" "$NEW" "" ;;
                new-mt2) arm "$m" "$a" "$NEW" "GGML_CUDA_MMVQ_PASCAL_MT=2" ;;
                base-unset) arm "$m" "$a" "$REF" "" ;;
                *) say "unknown arm $a" ;;
            esac
        done
    done
fi

say "done: our containers left: $(docker ps -q --filter "name=^$NAME-" | wc -l); card MiB: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPUS" | paste -sd' ')"
date -Is > "$WORK/DONE"
