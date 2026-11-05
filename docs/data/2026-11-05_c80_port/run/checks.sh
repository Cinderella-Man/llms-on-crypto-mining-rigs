#!/usr/bin/env bash
# C80 (Q573): the five branches of the old v3 base ported onto pascal-p106-v3-b12076-ladder4 (-textreuse, -promptpath,
# -decodemask, -cheapguess, -mtpk), checked on a build machine, CPU only, one branch after the other in one work folder,
# every binary in a container of the build image:
#   build     the rig tarball (scripts/build_llamacpp_docker.sh fork/<branch>, TESTS=1: sm_61 code and compute_61 PTX,
#             no AVX) and test-llama-archs from its build tree, copied into the tarball tree's bin/
#   identity  test-llama-archs -b CPU on every architecture: -ladder4's rig build (BASE_BUILD) writes the logits, the
#             branch's rig build compares (the CUDA backend finds no device); with every switch unset, all must match
#             bit for bit
#   cpubuild  a CPU build (no CUDA, no llamafile GEMM, AVX2: 2026-10-28_reply_rereads/run/build_cpu.sh's flags) of
#             llama-server, test-llama-archs and llama-pipe-drafter-check into $WORK/cpu-<name>/bin
#   own       the branch's own checks with its switches on, each the original branch's published check:
#             textreuse   2026-10-28_reply_rereads/run/check_server.sh: LLAMA_SERVER_TEXT_REUSE=1 on a tiny random Flint
#                         (TINY: Flint's tokenizer and template), every warm answer equal to a cold server's
#             decodemask  2026-10-28_reply_rereads/run/mask_check_cpu.sh: test-llama-archs with the host mask, the
#                         card-built mask for micro-batches of up to 16 tokens and for every micro-batch, each card mask
#                         compared with the host mask (LLAMA_GPU_KQ_MASK_CHECK=1); test-llama-archs runs 32-128-token
#                         micro-batches, so the decode arm builds no card mask on a CPU (its bound at work)
#             promptpath  both, and check_server.sh with LLAMA_SERVER_TAIL_CKPT_MIN=4096 as well
#             cheapguess  llama-pipe-drafter-check --cpu on CASE (a random-weight head and drafter.Drafter.forward's
#                         logits: 2026-10-27_cheap_guess/check.txt) for every option; LLAMA_DRAFT_LOAD_CHUNK=0 gives the
#                         default's step lines, with and without the Q8_0 head; with no option set, the branch's rig
#                         build gives -ladder4's step lines (both rig builds' CPU backend)
#             mtpk        nothing on a CPU: its kernels run on a GPU only, so the build is its check here
# Usage: WORK=<dir> REPO=<this repository> BASE_BUILD=<-ladder4's unpacked rig tarball, test-llama-archs in bin/> \
#   TINY=<tiny Flint gguf> CASE=<folder with drafter.gguf and case.bin> [BRANCHES='textreuse promptpath ...'] \
#   [PREFIX=pascal-p106-v3-b12076-] [REMOTE=fork] [SPLIT=6,6,5,5,6,5,5,6,5] [JOBS=8] [LOCK=<flock file>] \
#   [DOCKER_ARGS=...] [PY_IMAGE=python:3.12-slim] [STEPS='build identity cpubuild own'] [BUILD_OF=<name>] \
#   [RIG_TREE=<dir>] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL
#   BUILD_OF  one build that holds every branch of BRANCHES (a merge, checked out in $WORK/src): the cpubuild step builds
#             it once into $WORK/cpu-<BUILD_OF>/bin, and every branch's own checks run on it (STEPS='cpubuild own')
#   RIG_TREE  the rig tree the drafter's default check compares with BASE_BUILD's (default: the branch's tarball tree)
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>{,.tar.gz}, $WORK/md5.txt, $WORK/cpu-<name>/bin/, $WORK/checks/<name>/,
#   $WORK/logs/, $WORK/summary.tsv (branch, check, verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
: "${BASE_BUILD:?BASE_BUILD=<-ladder4 rig tree>}" "${TINY:?TINY=<tiny Flint gguf>}" "${CASE:?CASE=<drafter check case>}"
BRANCHES=${BRANCHES:-textreuse promptpath decodemask cheapguess mtpk}
PREFIX=${PREFIX:-pascal-p106-v3-b12076-}
REMOTE=${REMOTE:-fork}
SPLIT=${SPLIT:-6,6,5,5,6,5,5,6,5}
JOBS=${JOBS:-8}
STEPS=${STEPS:-build identity cpubuild own}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
RR=$REPO/docs/data/2026-10-28_reply_rereads/run
mkdir -p "$WORK/logs" "$WORK/checks"
rm -f "$WORK/DONE"
[ -f "$WORK/summary.tsv" ] || printf 'branch\tcheck\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
mounts="-v $WORK:$WORK -v $BASE_BUILD:$BASE_BUILD -v $CASE:$CASE${RIG_TREE:+ -v $RIG_TREE:$RIG_TREE}"
# shellcheck disable=SC2086
box() { docker run --rm $D $mounts "$@"; }
# a container without a GPU whose CUDA backend loads against the stub driver (the rig build's binaries)
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
cpuname() { echo "${BUILD_OF:-$1}"; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$1".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
echo "=== checks started $(date -Is), branches: $BRANCHES, steps: $STEPS"

build() {
    local n=$1 b=$PREFIX$1 rc T
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" \
        "$REMOTE/$b" > "$WORK/logs/rig-$n.log" 2>&1
    rc=$?
    T=$(tarball_tree "$b")
    if [ $rc = 0 ] && [ -n "$T" ]; then
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        verdict "$n" build-rig OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict "$n" build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/rig-$n.log") error lines in logs/rig-$n.log"
        return 1
    fi
    # test-llama-archs from the same build tree (the CUDA build of the branch)
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs > /work/logs/extra-build.log 2>&1 \
            || { tail -40 /work/logs/extra-build.log; exit 1; }
        mkdir -p /work/extra && cp build/bin/test-llama-archs /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs' > "$WORK/logs/extra-$n.log" 2>&1
    rc=$?
    if [ $rc = 0 ]; then
        cp "$WORK/extra/test-llama-archs" "$T/bin/"
        verdict "$n" build-extra OK "test-llama-archs in $(basename "$T")/bin"
    else
        verdict "$n" build-extra FAIL "rc=$rc; logs/extra-$n.log"
    fi
}

identity() {
    local n=$1 T K nb nc bad sb sm
    T=$(tarball_tree "$PREFIX$n")
    K=$WORK/checks/$n/identity
    rm -rf "$K" && mkdir -p "$K"
    if [ -z "$T" ] || [ ! -x "$T/bin/test-llama-archs" ]; then
        verdict "$n" identity FAIL "no rig tree with test-llama-archs"; return 1
    fi
    box -v "$K:/k" "$IMAGE" bash -c "$STUB"'
        run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        run base '"$BASE_BUILD/bin"'
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run port '"$T/bin"'
        rm -f /k/*.f32
        chown -R '"$(id -u):$(id -g)"' /k' > "$K/rc.txt" 2>&1
    # architectures whose models share a logits file compare against each other already in the base run
    grep 'logits against' "$K/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
    nb=$(cat "$K/n_base.txt" 2>/dev/null || echo 0)
    nc=$(grep -c 'logits against' "$K/port.log")
    bad=$(grep 'logits against' "$K/port.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
    sb=$(grep -m 1 '^Summary' "$K/base.log") sm=$(grep -m 1 '^Summary' "$K/port.log")
    verdict "$n" identity "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && [ -z "$bad" ] && [ "$sb" = "$sm" ] && echo OK || echo FAIL)" \
        "-ladder4 wrote $nb logits files; $nc compared, differ: ${bad:-none}; -ladder4: $sb; port: $sm; $(tr '\n' ' ' < "$K/rc.txt")"
}

cpubuild() {
    local n=$1 rc
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e N="$n" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
        "$IMAGE" bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache-cpu
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        T="llama-server test-llama-archs llama-pipe-drafter-check"
        O=/work/cpu-$N
        cmake -S /work/src -B $O/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF \
            -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON -DGGML_LLAMAFILE=OFF -DLLAMA_CURL=OFF \
            -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cpu-cmake-$N.log 2>&1
        rc=0
        nice -n 19 cmake --build $O/build -j "$JOBS" --target $T > /work/logs/cpu-build-$N.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf $O/bin && mkdir -p $O/bin
            for t in $T; do cp $O/build/bin/$t $O/bin/; done
        fi
        chown -R "$HOST_UID:$HOST_GID" $O /work/logs /work/ccache-cpu
        exit $rc'
    rc=$?
    verdict "$n" cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; commit $(git -C "$WORK/src" rev-parse --short=9 HEAD)"
}

# <name> <tag> <extra variables of the "on" server>: check_server.sh, the warm answers against a cold server's
own_server() {
    [ -x "$WORK/cpu-$(cpuname "$1")/bin/llama-server" ] || { verdict "$1" "server$2" FAIL "no CPU build in cpu-$(cpuname "$1")/bin"; return 1; }
    local n=$1 tag=$2 K=$WORK/checks/$1/server$2 bin=$WORK/cpu-$(cpuname "$1")/bin rc
    mkdir -p "$K/run" && cp "$RR/textreuse_check.py" "$K/run/"
    # (BIN before WORK: a later prefix assignment would expand $WORK as K)
    BIN=$bin WORK=$K MODEL=$TINY ON_EXTRA="$3" TAG=$tag PY_IMAGE=${PY_IMAGE:-python:3.12-slim} \
        nice -n 19 bash "$RR/check_server.sh" > "$K/console.txt" 2>&1
    rc=$?
    verdict "$n" "server$tag" "$([ $rc = 0 ] && grep -q '^# PASS' "$K/out/textreuse_check$tag.tsv" && echo OK || echo FAIL)" \
        "LLAMA_SERVER_TEXT_REUSE=1 $3 rc=$rc: $(grep -a '^#' "$K/out/textreuse_check$tag.tsv" 2>/dev/null | tr '\n' ' ' | cut -c1-300)"
}

own_mask() {
    local n=$1 K=$WORK/checks/$1/mask bin=$WORK/cpu-$(cpuname "$1")/bin ok=OK
    mkdir -p "$K"
    BIN=$bin WORK=$K nice -n 19 bash "$RR/mask_check_cpu.sh" > "$K/console.txt" 2>&1
    # every arm exits 0 with the host arm's summary; the all-batches arm compares card masks (an abort on any difference);
    # the decode arm builds none over 16 tokens (test-llama-archs' micro-batches are 32-128 tokens, so on a CPU it builds
    # none at all: the bound is what keeps them on the host mask)
    local s0 a
    s0=$(grep -a -m 1 'Summary' "$K/out/mask_cpu_host.log")
    for a in host decode all; do
        grep -q "^$a: exit 0;" "$K/console.txt" || ok=FAIL
        [ "$(grep -a -m 1 'Summary' "$K/out/mask_cpu_$a.log")" = "$s0" ] || ok=FAIL
    done
    [ "$(grep -a -c 'identical to the host mask' "$K/out/mask_cpu_all.log")" -gt 0 ] || ok=FAIL
    [ -z "$(grep -a -o 'copy [0-9]*: [0-9]* tokens' "$K/out/mask_cpu_decode.log" | awk '$3 > 16')" ] || ok=FAIL
    verdict "$n" mask "$ok" "$(tr '\n' ' ' < "$K/console.txt" | cut -c1-600)"
}

own_drafter() {
    local n=$1 K=$WORK/checks/$1/drafter T ok=OK i
    T=${RIG_TREE:-$(tarball_tree "$PREFIX$n")}
    mkdir -p "$K"
    # <tag> <bin dir> <variables...>: one run of the reference case on the CPU
    dcheck() {
        local tag=$1 bin=$2; shift 2
        box "$IMAGE" bash -c "$STUB"'; env '"$*"' nice -n 19 '"$bin"'/llama-pipe-drafter-check '"$CASE/drafter.gguf $CASE/case.bin"' \
            --split '"$SPLIT"' --cpu --host-table' > "$K/$tag.log" 2>&1
        grep -a '^step \|^DRAFTER_CHECK' "$K/$tag.log" > "$K/$tag.steps"
    }
    local opts=("" "LLAMA_DRAFT_PRUNE=1" "LLAMA_DRAFT_ANC=dense" "LLAMA_DRAFT_SPLIT=4" "LLAMA_DRAFT_SPLIT=4 LLAMA_DRAFT_PRUNE=1"
                "LLAMA_DRAFT_ANC=dense LLAMA_DRAFT_HEAD=q8_0" "LLAMA_DRAFT_LOAD_CHUNK=0"
                "LLAMA_DRAFT_HEAD=q8_0" "LLAMA_DRAFT_HEAD=q8_0 LLAMA_DRAFT_LOAD_CHUNK=0")
    for i in "${!opts[@]}"; do
        dcheck "opt$i" "$WORK/cpu-$(cpuname "$n")/bin" "${opts[$i]}"
        local line; line=$(grep -a '^DRAFTER_CHECK' "$K/opt$i.steps")
        [[ "$line" == *" bad=0 "* ]] || ok=FAIL
        printf '%s\t%s\n' "${opts[$i]:-(defaults)}" "${line:-no DRAFTER_CHECK line}" >> "$K/options.tsv"
    done
    cmp -s "$K/opt0.steps" "$K/opt6.steps" || ok=FAIL   # LLAMA_DRAFT_LOAD_CHUNK=0: the same weights
    cmp -s "$K/opt7.steps" "$K/opt8.steps" || ok=FAIL   # ... with the Q8_0 head converted at once
    verdict "$n" drafter "$ok" "CPU build, ${#opts[@]} option sets (checks/$n/drafter/options.tsv); chunk 0 = default: $(cmp -s "$K/opt0.steps" "$K/opt6.steps" && echo yes || echo no), with q8_0: $(cmp -s "$K/opt7.steps" "$K/opt8.steps" && echo yes || echo no)"
    # the defaults on both rig builds (their CPU backend): the port's staircase drafter computes what -ladder4's does
    if [ -n "$T" ]; then
        dcheck rig-ladder4 "$BASE_BUILD/bin"
        dcheck rig-port "$T/bin"
        verdict "$n" drafter-identity "$(cmp -s "$K/rig-ladder4.steps" "$K/rig-port.steps" && [ -s "$K/rig-port.steps" ] && echo OK || echo FAIL)" \
            "rig builds, no option set: -ladder4 $(grep -a '^DRAFTER_CHECK' "$K/rig-ladder4.steps" | cut -c1-80); port $(grep -a '^DRAFTER_CHECK' "$K/rig-port.steps" | cut -c1-80)"
    fi
}

for n in $BRANCHES; do
    echo "=== $n $(date -Is)"
    if step build; then build "$n" || continue; fi
    step identity && identity "$n"
    if step cpubuild && [ "$n" != mtpk ] && [ ! -x "$WORK/cpu-${BUILD_OF:-.}/bin/llama-server" ]; then cpubuild "$(cpuname "$n")"; fi
    if step own; then
        case $n in
            textreuse)  own_server "$n" "" "" ;;
            decodemask) own_mask "$n" ;;
            promptpath) own_server "$n" "" ""; own_server "$n" _tail "LLAMA_SERVER_TAIL_CKPT_MIN=4096"; own_mask "$n" ;;
            cheapguess) own_drafter "$n" ;;
            mtpk)       verdict "$n" own SKIP "its kernels need a GPU (test-backend-ops with GGML_CUDA_MMVQ_PASCAL_MT=1)" ;;
        esac
    fi
done

date -Is > "$WORK/DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
