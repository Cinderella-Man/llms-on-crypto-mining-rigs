#!/usr/bin/env bash
# Q660 (C120) on a CPU, no GPU needed: the routed experts' tail as one op (GGML_OP_MOE_TAIL, whose -1 ids MUL_MAT_ID
# skips) against the graph-level ops it replaces (LLAMA_MOE_TAIL ...,graph=1), from a CPU build of the branch.
#   build   a native CPU build of WORK/src in Docker: test-bytes-switches, test-llama-archs, llama-quantize and
#           test-backend-ops (which runs its cases on a GPU, gpu_checks.sh; here it is only built)
#   switch  test-bytes-switches on random qwen4exp models (64 routed experts, 10 used, an F32 head and a Q5_K one, the
#           routers scaled by 30): every arm (p = 1, 0.9, 0.8, 0.6, a 7.5 % floor; renormalized or not) with the switch
#           as the graph-level ops, as the op with -1 ids up to 8 tokens, with the last kept expert up to 8 tokens
#           (repeat), and with every slot's own expert (skip 0),
#           in micro-batches of 64, 4 and 1 tokens: the op's logits and histogram must equal the graph's bit for bit
#   archs   test-llama-archs -b CPU over every architecture: the graph-level cut writes each one's logits
#           (LLAMA_ARCHS_LOGITS), the op must give the same (skip 8: -1 ids up to 8 tokens; skip 0: every slot's own
#           expert; repeat=1: the last kept expert up to 8 tokens); at p = 0.8 renormalized and with a 5 % floor
# Env: WORK (src/ = the branch's tree; build/, bin/, logs/, out/ are written there), JOBS (8), THREADS (16),
#      LOCK (a flock file shared by the machine's builds), DOCKER_ARGS (e.g. '--cpu-shares=256 --label ...'),
#      IMAGE (nvidia/cuda:12.9.2-devel-ubuntu24.04), STEPS ("build switch archs")
# Output: WORK/out/{switch,archs}/*.log, WORK/out/verdicts.tsv (step, verdict, detail), WORK/out/DONE
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<folder with src/>}")
JOBS=${JOBS:-8}
THREADS=${THREADS:-16}
STEPS=${STEPS:-build switch archs}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D=${DOCKER_ARGS:-}
O=$WORK/out
mkdir -p "$WORK/logs" "$O/switch" "$O/archs"
rm -f "$O/DONE"
step() { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$O/verdicts.tsv"; }
# runs a command in the build image with WORK at /work
in_image() {
    # shellcheck disable=SC2086
    docker run --rm $D -v "$WORK:/work" -e THREADS="$THREADS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -c "$1"
}

if step build; then
    echo "=== build $(date -Is)"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        cmake -S /work/src -B /work/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cmake.log 2>&1
        T="test-bytes-switches test-llama-archs llama-quantize test-backend-ops"
        rc=0
        nice -n 19 cmake --build /work/build -j "$JOBS" --target $T > /work/logs/build.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf /work/bin && mkdir -p /work/bin
            for t in $T; do cp /work/build/bin/$t /work/bin/; done
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/build /work/logs /work/ccache
        [ -d /work/bin ] && chown -R "$HOST_UID:$HOST_GID" /work/bin
        exit $rc'
    rc=$?
    verdict build "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/build.log") error lines"
    [ $rc = 0 ] || grep -a -E "error" "$WORK/logs/build.log" | head -30
fi

if step switch; then
    echo "=== switch $(date -Is)"
    in_image '
        M=/work/out/switch
        if [ ! -f $M/qwen4exp-moe.gguf ]; then
            LLAMA_TEST_ARCHS_N_VOCAB=4096 LLAMA_TEST_ARCHS_N_EXPERT=64,10 /work/bin/test-llama-archs -a "^qwen4exp$" -o $M \
                > $M/archs-model.log 2>&1
        fi
        [ -f $M/qwen4exp-moe-q5k.gguf ] || /work/bin/llama-quantize --allow-requantize --output-tensor-type q5_k \
            $M/qwen4exp-moe.gguf $M/qwen4exp-moe-q5k.gguf F32 $THREADS > $M/quantize.log 2>&1
        for m in qwen4exp-moe qwen4exp-moe-q5k; do
            nice -n 19 /work/bin/test-bytes-switches -m $M/$m.gguf -t $THREADS -n 24 -g 8 -k 1024 --router-scale 30 \
                > $M/$m.log 2>&1
            echo "$m rc=$? $(tail -n 1 $M/$m.log)"
        done
        rm -f $M/*.gguf
        chown -R "$HOST_UID:$HOST_GID" $M' > "$O/switch/console.txt" 2>&1
    cat "$O/switch/console.txt"
    nok=$(grep -c 'rc=0 .* 0 failed' "$O/switch/console.txt")
    ndiff=$(cat "$O"/switch/qwen4exp-moe*.log | grep -c 'DIFFER')
    verdict switch "$([ "$nok" = 2 ] && [ "$ndiff" = 0 ] && echo OK || echo FAIL)" \
        "$nok/2 models pass; $ndiff op arms differ from the graph; $(cat "$O"/switch/qwen4exp-moe*.log | grep -c "the graph's logits") op arms equal it"
fi

if step archs; then
    echo "=== archs $(date -Is)"
    in_image '
        K=/work/out/archs
        rm -f $K/*.f32
        # an architecture whose dense and MoE models share a name compares its second model with the first within a run
        # (the graph run shows them): an op run passes when its differing comparisons are exactly those of the graph run
        diffs() { grep "logits against" $1 | grep -v ", 0 of " | sed "s/: logits against [^:]*:/:/" | sort; }
        run() { # <name> <LLAMA_MOE_TAIL> <logits prefix> [<graph run>]
            LLAMA_MOE_TAIL=$2 LLAMA_ARCHS_LOGITS=$3 nice -n 19 /work/bin/test-llama-archs -b CPU -s 1 2>&1 \
                | sed "s/\x1b\[[0-9;]*m//g" > $K/$1.log
            local rc=${PIPESTATUS[0]} same=""
            [ -n "${4:-}" ] && { [ "$(diffs $K/$1.log)" = "$(diffs $K/$4.log)" ] && same="; = graph" || same="; DIFFERS from graph"; }
            echo "$1 rc=$rc $(grep -c "logits against" $K/$1.log) compared, $(diffs $K/$1.log | wc -l) differ$same"
        }
        for cut in "p=0.8,renorm=1" "w=0.05,renorm=1"; do
            tag=$(echo $cut | tr -c "a-z0-9\n" _)
            run graph-$tag "$cut,hist=0,graph=1"  $K/$tag-
            run op-$tag    "$cut,hist=0"          $K/$tag- graph-$tag
            run op0-$tag   "$cut,hist=0,skip=0"   $K/$tag- graph-$tag
            run oprep-$tag "$cut,hist=0,repeat=1" $K/$tag- graph-$tag
        done
        rm -f $K/*.f32
        chown -R "$HOST_UID:$HOST_GID" $K' > "$O/archs/console.txt" 2>&1
    cat "$O/archs/console.txt"
    nbad=$(grep -E '^op' "$O/archs/console.txt" | grep -c -v '= graph$')
    verdict archs "$([ "$nbad" = 0 ] && echo OK || echo FAIL)" "$(grep -E '^(graph|op)' "$O/archs/console.txt" | tr '\n' ';')"
fi

echo "$(date -Is)" > "$O/DONE"
cat "$O/verdicts.tsv"
