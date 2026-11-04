#!/usr/bin/env bash
# Q860's checks on a CPU, no GPU needed: route once per verify block (LLAMA_MOE_ROUTE_ONCE) on the CPU backend, every
# binary in a container of the build image.
#   build     a CPU build of SRC (a checkout of the -routeonce branch): test-route-once, test-llama-archs,
#             llama-perplexity, llama-quantize
#   unit      test-route-once on random qwen4exp models from test-llama-archs (4,096 tokens, 32 routed experts, 8 used),
#             with an F32 head and a Q5_K one, the routers as they are and scaled by 30
#   identity  test-llama-archs -b CPU on every architecture: logits written by the base build (BASE_BIN, the build of
#             the branch's parent) and compared by this build with the switch unset, at 0, at k = 1 (with and without
#             the histogram), in a dry run (k = 5) and with max = 1 (no micro-batch qualifies): all bit for bit; at
#             k = 5 only architectures with routed experts may differ
#   real      llama-perplexity on a real model (MODEL, e.g. Swift Q2_0 of Qwen3.8-Flash-Next) over the first CHUNKS
#             chunks of 512 tokens of TEXT: off writes its logits; k = 1, dry runs at k = 2-5 and the switch on at
#             k = 2-5 give the KL divergence against them (the switch's own divergence) and the histogram (the distinct
#             experts per block of k, read and under independent routing); ~1 minute an arm on 12 cores
# Env: WORK, SRC, BASE_BIN; MODEL, TEXT, CHUNKS (4) for real; THREADS (12), JOBS (6), LOCK (a file to flock around
#      the build), STEPS (default "build unit identity"), IMAGE, MOUNTS (more folders to mount, at their own paths),
#      DOCKER_ARGS, REAL_ARMS (a subset of the real step's arms)
# Writes $WORK/cpu/bin, $WORK/random, $WORK/checks/<step>/, one verdict line per check in $WORK/summary.tsv.
set -uo pipefail
: "${WORK:?}" "${SRC:?}"
THREADS=${THREADS:-12}
JOBS=${JOBS:-6}
STEPS=${STEPS:-build unit identity}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
BIN=$WORK/cpu/bin
D="--rm --cpu-shares 256 --label pascal-quants=1 ${DOCKER_ARGS:-}"
mounts="-v $WORK:$WORK -v $SRC:$SRC"
for d in ${BASE_BIN:-} ${MOUNTS:-}; do mounts="$mounts -v $d:$d"; done
mkdir -p "$WORK/logs" "$WORK/checks"
touch "$WORK/summary.tsv"

log()     { printf '%s %s\n' "$(date '+%F %T')" "$*"; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
verdict() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
# a command in a container of the image, the folders mounted at their own paths
box()     { docker run $D $mounts "$@"; }

if step build; then
    log "build"
    locked docker run $D $mounts -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" -e JOBS="$JOBS" -e SRC="$SRC" \
        -e WORK="$WORK" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=$WORK/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory "$SRC"
        T="test-route-once test-llama-archs llama-perplexity llama-quantize"
        cmake -S "$SRC" -B "$WORK/cpu/build" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON \
            -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > "$WORK/logs/cmake-cpu.log" 2>&1
        rc=0
        nice -n 19 cmake --build "$WORK/cpu/build" -j "$JOBS" --target $T > "$WORK/logs/cpubuild.log" 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            mkdir -p "$WORK/cpu/bin"
            for t in $T; do cp "$WORK/cpu/build/bin/$t" "$WORK/cpu/bin/"; done
            git -C "$SRC" log -1 --format="%H %s" > "$WORK/cpu/bin/COMMIT"
            git -C "$SRC" status --short >> "$WORK/cpu/bin/COMMIT"
        fi
        chown -R "$HOST_UID:$HOST_GID" "$WORK/cpu" "$WORK/logs" "$WORK/ccache"
        exit $rc' > "$WORK/logs/build.out" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
        grep -E "error|Error" "$WORK/logs/cpubuild.log" | head -40
        verdict build FAIL "exit $rc, $WORK/logs/cpubuild.log"
        exit 1
    fi
    verdict build OK "$(head -n 1 "$BIN/COMMIT")"
fi

if step unit; then
    log "unit"
    M=$WORK/random K=$WORK/checks/unit
    mkdir -p "$M" "$K"
    if [ ! -f "$M/qwen4exp-moe.gguf" ]; then
        box -e LLAMA_TEST_ARCHS_N_VOCAB=4096 -e LLAMA_TEST_ARCHS_N_EXPERT=32,8 "$IMAGE" \
            "$BIN/test-llama-archs" -a qwen4exp -o "$M" > "$M/archs.log" 2>&1
    fi
    if [ ! -f "$M/qwen4exp-moe-q5k.gguf" ]; then
        box "$IMAGE" "$BIN/llama-quantize" --allow-requantize --output-tensor-type q5_k \
            "$M/qwen4exp-moe.gguf" "$M/qwen4exp-moe-q5k.gguf" F32 "$THREADS" > "$M/quantize.log" 2>&1
    fi
    for m in qwen4exp-moe qwen4exp-moe-q5k; do
        for s in 1 30; do
            lg=$K/$m-s$s.log
            # the checks' lines on stdout, llama's log apart
            box "$IMAGE" nice -n 19 "$BIN/test-route-once" -m "$M/$m.gguf" -t "$THREADS" -n 16 --router-scale $s \
                > "$lg" 2> "${lg%.log}.err"
            rc=$?
            verdict "unit $m router x$s" "$([ $rc -eq 0 ] && echo OK || echo FAIL)" "$(tail -n 1 "$lg")"
        done
    done
fi

if step identity; then
    log "identity"
    : "${BASE_BIN:?identity needs BASE_BIN}"
    K=$WORK/checks/identity
    rm -rf "$K" && mkdir -p "$K"
    printf 'experts\tarm\tLLAMA_MOE_ROUTE_ONCE\texit\tcompared\tdiffer\tarchs_that_differ\n' > "$K/identity.tsv"
    # <round> <LLAMA_TEST_ARCHS_N_EXPERT, empty: the test's own (most MoE architectures: 2 experts, both used, so
    # routing once changes nothing; clover_moe 4)> <arms>: the base build writes the logits, this build compares
    round() {
        local r=$1 ne=$2 arms=$3
        # <name> <bin dir> [LLAMA_MOE_ROUTE_ONCE]
        archs() {
            box ${3:+-e LLAMA_MOE_ROUTE_ONCE=$3} ${ne:+-e LLAMA_TEST_ARCHS_N_EXPERT=$ne} \
                -e LLAMA_ARCHS_LOGITS="$K/logits-$r-" "$IMAGE" \
                nice -n 19 "$2/test-llama-archs" -b CPU -s 1 2>&1 | sed 's/\x1b\[[0-9;]*m//g' > "$K/$r-$1.log"
            echo "${PIPESTATUS[0]}" > "$K/$r-$1.rc"
        }
        archs base "$BASE_BIN"
        # architectures that share a logits file compare against each other already in the base run: those lines,
        # repeated word for word, are the base's own and not a difference
        grep 'logits against' "$K/$r-base.log" | grep -v ', 0 of ' > "$K/$r-base_self.txt"
        for x in $arms; do
            a=${x%%:*} e=${x#*:}
            archs "$a" "$BIN" "$e"
            n=$(grep -c 'logits against' "$K/$r-$a.log")
            bad=$(grep 'logits against' "$K/$r-$a.log" | grep -v ', 0 of ' | grep -vxFf "$K/$r-base_self.txt" \
                | cut -d: -f1 | tr '\n' ' ')
            nb=$(echo "$bad" | wc -w)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ne:-test}" "$a" "${e:-unset}" "$(cat "$K/$r-$a.rc")" "$n" "$nb" \
                "${bad:--}" >> "$K/identity.tsv"
            if [ "$a" = k5 ]; then
                # the switch on: the architectures with more routed experts than they use change, the others stay
                verdict "identity $r $a" "$([ "$nb" -gt 0 ] && echo REVIEW || echo FAIL)" "$nb of $n archs differ: ${bad:-none}"
            else
                verdict "identity $r $a" "$([ "$n" -gt 0 ] && [ "$nb" -eq 0 ] && echo OK || echo FAIL)" "$nb of $n archs differ"
            fi
        done
        rm -f "$K"/logits-$r-*.f32
    }
    round test "" "unset: zero:0 k1:1 k1hist:1,hist=1 dry5:5,dry=1,hist=1 max1:5,max=1 k5:5"
    round e8 8,2 "unset: k1:1 dry5:5,dry=1,hist=1 k5:5"
fi

if step real; then
    log "real"
    : "${MODEL:?real needs MODEL}" "${TEXT:?real needs TEXT}"
    K=$WORK/checks/real
    mkdir -p "$K"
    arm_env() {
        case $1 in
            off) echo "" ;;
            k1)  echo "1,hist=1" ;;
            dry[2-9]) echo "${1#dry},dry=1,hist=1" ;;
            on[2-9])  echo "${1#on},hist=1" ;;
            *) return 1 ;;
        esac
    }
    # off writes the switch-off logits (llama-perplexity --kl-divergence-base); every other arm is compared with them
    base=$K/off.kld
    for a in ${REAL_ARMS:-off k1 dry2 dry3 dry4 dry5 on2 on3 on4 on5}; do
        e=$(arm_env "$a") || { log "unknown arm $a"; continue; }
        lg=$K/$a.log
        if [ "$a" = off ]; then
            grep -q "Final estimate" "$lg" 2> /dev/null && [ -f "$base" ] && continue
            box "$IMAGE" nice -n 19 "$BIN/llama-perplexity" -m "$MODEL" -f "$TEXT" -c 512 -b 512 -ub 512 \
                --chunks "${CHUNKS:-4}" -t "$THREADS" -tb "$THREADS" --kl-divergence-base "$base" -lv 4 > "$lg" 2>&1
            verdict "real $a" "$(grep -q "Final estimate" "$lg" && [ -f "$base" ] && echo OK || echo FAIL)" \
                "$(grep -o 'Final estimate: PPL = [0-9.]* +/- [0-9.]*' "$lg")"
            continue
        fi
        grep -q "^Mean    KLD" "$lg" 2> /dev/null && continue
        box ${e:+-e LLAMA_MOE_ROUTE_ONCE=$e} "$IMAGE" nice -n 19 "$BIN/llama-perplexity" -m "$MODEL" -f "$TEXT" \
            -c 512 -b 512 -ub 512 -t "$THREADS" -tb "$THREADS" --kl-divergence-base "$base" --kl-divergence -lv 4 \
            > "$lg" 2>&1
        verdict "real $a" "$(grep -q "^Mean    KLD" "$lg" && echo OK || echo FAIL)" \
            "$(grep -E "^(Mean    KLD|Same top p)" "$lg" | tr -s ' ' | tr '\n' ' ') $(grep -o 'read mean [0-9.]*, independent routing mean [0-9.]*' "$lg" | tail -n 1)"
    done
fi
log "done"
