#!/usr/bin/env bash
# C536 (Q1025): fork branch pascal-p106-v3-b12076-flags (-ladder4 with an environment switch for every fork behaviour
# that had none, each defaulting to today's behaviour) on a build machine, CPU only, every binary in a container of the
# build image:
#   basecpu   a static native CPU build of the base (-ladder4, BASE) from $WORK/src: CPU_TARGETS into $WORK/base-cpu/bin
#   build     the rig tarball and the extra tests (the -ladder4 merge's run/checks.sh, step build)
#   cpubuild  the branch's CPU build into $WORK/cpu/bin (the same script, step cpubuild)
#   identity  test-llama-archs -b CPU on every architecture, the base's logits against the branch's, for the rig builds
#             (BASE_BUILD against this tarball's tree; the CUDA backend finds no device) and the CPU builds (the same
#             script, step identity): every switch at its default, so every architecture must match bit for bit
#   flips     the branch's CPU build with each graph- and cache-side switch at its other value (FLIPS), and once with
#             every CUDA-side switch at its other value together (CUDA_FLIPS: no CUDA code runs, so nothing may change),
#             against the base's logits: an architecture differs only where the switch's path runs on a CPU
#   context   test-llama-context on random models (CTX_ARCHS, written by the branch's test-llama-archs -o): the base and
#             the branch at their defaults must give the same verdicts; then the branch with each cache switch flipped,
#             reported per case (a flip may fail the cases that check its rule)
#   server    llama-server on a random sliding-window model (SWA_ARCH), five greedy requests with checkpoint restores
#             (as the C22 server check), one server per arm: the default, each server-side switch flipped, and the
#             base build; the default must give the base's tokens, LLAMA_STATE_NOINIT=0 the default's tokens and restore
#             points; the checkpoint arms report their restore points, checkpoint sizes and tokens
# Usage: WORK=<dir> REPO=<this repository> BASE_BUILD=<-ladder4's unpacked rig tarball, extras in bin/> \
#   [BRANCH=pascal-p106-v3-b12076-flags] [BASE=e3b30303d] [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>] \
#   [DOCKER_ARGS=...] [MOUNTS='<dirs>'] [STEPS='basecpu build cpubuild identity flips context server'] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz}, $WORK/extra-<sha>.tar.gz, $WORK/md5.txt, $WORK/cpu/bin/,
#   $WORK/base-cpu/bin/, $WORK/checks/<step>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when
#   $WORK/FLAGS_DONE exists (the -ladder4 script's steps also write $WORK/DONE)
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-flags}
BASE=${BASE:-e3b30303d}
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-basecpu build cpubuild identity flips context server}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
CPU_TARGETS=${CPU_TARGETS:-llama-server test-llama-archs test-llama-context test-backend-ops}
FLIPS=${FLIPS:-LLAMA_ATTN_K_LAST=0 LLAMA_KV_HEAD_EMPTY=0 LLAMA_KV_HEAD_TAIL=0 LLAMA_KV_HEAD_FREE=0 \
LLAMA_KV_RESTORE_CANONICAL=0 LLAMA_GPU_KQ_CELLS=0 LLAMA_QSA_PAD_LINEAR=0 GGML_SCHED_CTX_PER_GRAPH=0}
CUDA_FLIPS=${CUDA_FLIPS:-GGML_CUDA_MMVQ_2WARPS=0 GGML_CUDA_MMVQ_FP32_SUMS=0 GGML_CUDA_GLU_ALIAS_FUSE=0 \
GGML_CUDA_MMQ_MOE_LEGACY=1 GGML_CUDA_MMVF_PASCAL_RULE=0 GGML_CUDA_MMVF_PASCAL_MAX_COLS=4 \
GGML_CUDA_MMVQ_PASCAL_MTOK=1,0,0,0,2 GGML_CUDA_SKIP_IDS_ANY=0 GGML_CUDA_CPY_HOP_TYPES=0}
CTX_FLIPS=${CTX_FLIPS:-LLAMA_KV_HEAD_EMPTY=0 LLAMA_KV_HEAD_TAIL=0 LLAMA_KV_HEAD_FREE=0 LLAMA_KV_RESTORE_CANONICAL=0 \
LLAMA_ATTN_K_LAST=0 LLAMA_QSA_PAD_LINEAR=0 GGML_SCHED_CTX_PER_GRAPH=0}
CTX_ARCHS=${CTX_ARCHS:-llama gemma3 qwen35 qwen4exp mamba}
SWA_ARCH=${SWA_ARCH:-gemma3}
L4=$(dirname "$(realpath "$0")")/../../2026-11-04_ladder4_merge/run/checks.sh
D="--cpu-shares=256 ${CPUS:+--cpuset-cpus $CPUS} ${DOCKER_ARGS:-}"
S=$WORK/src
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/FLAGS_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
mounts="-v $WORK:$WORK -v $REPO:$REPO"
for d in ${BASE_BUILD:+"$BASE_BUILD"} ${MOUNTS:-}; do
    mounts="$mounts -v $d:$d"
done
# shellcheck disable=SC2086
box() { docker run --rm $D $mounts "$@"; }
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
# the -ladder4 merge's checks.sh with this branch: no merged branches of its own (union is not run)
l4() { WORK=$WORK REPO=$REPO BRANCH=$BRANCH REMOTE=$REMOTE BASE=$BASE BRANCHES=- MERGE_OWN=- JOBS=$JOBS THREADS=$THREADS \
       CPU_TARGETS=$CPU_TARGETS BASE_BUILD=${BASE_BUILD:-} BASE_CPU=$WORK/base-cpu/bin DOCKER_ARGS="${DOCKER_ARGS:-}" \
       MOUNTS="${MOUNTS:-}" STEPS=$1 bash "$L4"; }

git -C "$S" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$S" rev-parse "$REMOTE/$BRANCH")
echo "=== checks started $(date -Is), $BRANCH at ${TIP:0:9} on ${BASE:0:9}, steps: $STEPS"

if step basecpu; then
    echo "=== basecpu $(date -Is)"
    git -C "$S" checkout -q --force "$BASE" && git -C "$S" clean -qfdx
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e T="$CPU_TARGETS" -e HOST_UID="$(id -u)" \
        -e HOST_GID="$(id -g)" "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cmake -S /work/src -B /work/base-cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/base-cpu-cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/base-cpu/build -j "$JOBS" --target $T > /work/logs/base-cpu-build.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf /work/base-cpu/bin && mkdir -p /work/base-cpu/bin
            for t in $T; do cp /work/base-cpu/build/bin/$t /work/base-cpu/bin/; done
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/base-cpu /work/logs /work/ccache
        exit $rc'
    rc=$?
    verdict basecpu "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $CPU_TARGETS; commit $(git -C "$S" rev-parse --short=9 HEAD)"
fi

for s in build cpubuild identity; do
    step $s && l4 $s
done

# the base's logits of every architecture (CPU build), written once into $1
base_logits() {
    [ -s "$1/n_base.txt" ] && return 0
    mkdir -p "$1"
    box -v "$1:/k" "$IMAGE" bash -c '
        env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 '"$WORK"'/base-cpu/bin/test-llama-archs -b CPU -s 1 2>&1 \
            | sed "s/\x1b\[[0-9;]*m//g" > /k/base.log
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        chown -R '"$(id -u):$(id -g)"' /k'
    grep 'logits against' "$1/base.log" | grep -v ', 0 of ' > "$1/base_self.txt"
}

if step flips; then
    echo "=== flips $(date -Is)"
    K=$WORK/checks/flips
    base_logits "$K/base"
    # <name> <env assignments ...>: the branch's CPU build compares its logits with the base's
    flip() {
        local n=$1; shift
        local e="" a
        for a in "$@"; do e="$e -e $a"; done
        # shellcheck disable=SC2086
        box $e -e LLAMA_ARCHS_LOGITS=/k/logits- -v "$K/base:/k" "$IMAGE" nice -n 19 "$WORK/cpu/bin/test-llama-archs" -b CPU -s 1 2>&1 \
            | sed 's/\x1b\[[0-9;]*m//g' > "$K/$n.log"
        local nc bad sb sm
        nc=$(grep -c 'logits against' "$K/$n.log")
        bad=$(grep 'logits against' "$K/$n.log" | grep -v ', 0 of ' | grep -vxFf "$K/base/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
        sb=$(grep -m 1 '^Summary' "$K/base/base.log") sm=$(grep -m 1 '^Summary' "$K/$n.log")
        verdict "flip $n" "$([ "$nc" -gt 0 ] && [ -z "$bad" ] && [ "$sb" = "$sm" ] && echo SAME || echo DIFFERS)" \
            "$*: $nc compared, differ: ${bad:-none}; base: $sb; flipped: $sm"
    }
    for f in $FLIPS; do
        flip "${f%%=*}" "$f"
    done
    # shellcheck disable=SC2086
    flip cuda-all $CUDA_FLIPS
fi

if step context || step server; then
    M=$WORK/checks/models
    if [ ! -s "$M/done.txt" ]; then
        mkdir -p "$M"
        for a in $CTX_ARCHS $SWA_ARCH; do
            [ -d "$M/$a" ] && continue
            mkdir -p "$M/$a"
            box -e LLAMA_ARCHS_N_CTX=4096 "$IMAGE" nice -n 19 "$WORK/cpu/bin/test-llama-archs" -b CPU -s 1 -a "^$a\$" \
                -o "$M/$a" > "$M/$a.log" 2>&1
        done
        find "$M" -name '*.gguf' | sort > "$M/done.txt"
    fi
fi

if step context; then
    echo "=== context $(date -Is)"
    K=$WORK/checks/context
    rm -rf "$K" && mkdir -p "$K"
    # <name> <bin dir> <env assignments ...>: test-llama-context on every random model, one line per case and model
    ctx() {
        local n=$1 b=$2; shift 2
        local e="" a
        for a in "$@"; do e="$e -e $a"; done
        : > "$K/$n.log"
        for m in $(cat "$WORK/checks/models/done.txt"); do
            # shellcheck disable=SC2086
            box $e "$IMAGE" nice -n 19 "$b/test-llama-context" -m "$m" -t "$THREADS" 2>&1 | sed 's/\x1b\[[0-9;]*m//g' \
                > "$K/$n.$(basename "$(dirname "$m")").log"
            awk -v a="$(basename "$(dirname "$m")")" 'NF >= 2 && ($2 == "OK" || $2 == "FAIL" || $2 == "SKIP") { print a, $1, $2 }' \
                "$K/$n.$(basename "$(dirname "$m")").log" >> "$K/$n.log"
        done
        printf '%s' "$(grep -cw OK "$K/$n.log") OK, $(grep -cw SKIP "$K/$n.log") SKIP, $(grep -cw FAIL "$K/$n.log") FAIL"
    }
    cb=$(ctx base "$WORK/base-cpu/bin") cd_=$(ctx default "$WORK/cpu/bin")
    same=$(cmp -s "$K/base.log" "$K/default.log" && echo yes || echo no)
    verdict "context default" "$([ "$same" = yes ] && echo OK || echo FAIL)" "base: $cb; branch: $cd_; per-case verdicts equal: $same"
    for f in $CTX_FLIPS; do
        r=$(ctx "${f%%=*}" "$WORK/cpu/bin" "$f")
        verdict "context ${f%%=*}=${f#*=}" "$(cmp -s "$K/default.log" "$K/${f%%=*}.log" && echo SAME || echo DIFFERS)" \
            "$r; cases that differ from the default: $(diff "$K/default.log" "$K/${f%%=*}.log" | grep '^>' | cut -c3- | tr '\n' ';' | cut -c1-300)"
    done
fi

if step server; then
    echo "=== server $(date -Is)"
    K=$WORK/checks/server
    rm -rf "$K" && mkdir -p "$K"
    MODEL=$(grep "/$SWA_ARCH/" "$WORK/checks/models/done.txt" | head -n 1)
    # <arm> <bin dir> <env assignments ...>
    arms=(
        "default $WORK/cpu/bin"
        "base $WORK/base-cpu/bin"
        "ring0 $WORK/cpu/bin LLAMA_SERVER_CKPT_RING=0"
        "past0 $WORK/cpu/bin LLAMA_SERVER_CKPT_RING_PAST=0"
        "resume0 $WORK/cpu/bin LLAMA_SERVER_CKPT_RESUME_NEXT=0"
        "noinit0 $WORK/cpu/bin LLAMA_STATE_NOINIT=0"
        "kv0 $WORK/cpu/bin LLAMA_KV_HEAD_EMPTY=0 LLAMA_KV_HEAD_TAIL=0 LLAMA_KV_HEAD_FREE=0 LLAMA_KV_RESTORE_CANONICAL=0"
    )
    printf '%s\n' "${arms[@]}" > "$K/arms.txt"
    box -e MODEL="$MODEL" -e OUT="$K" -e THREADS="$THREADS" "$IMAGE" bash -c '
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq && apt-get install -y -qq curl jq gawk > /dev/null
        bash '"$(dirname "$(realpath "$0")")"'/server_arms.sh
        chown -R '"$(id -u):$(id -g)"' "$OUT"' > "$K/console.txt" 2>&1
    while IFS=$'\t' read -r c v d; do verdict "$c" "$v" "$d"; done < "$K/verdicts.tsv"
fi

date -Is > "$WORK/FLAGS_DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
