#!/usr/bin/env bash
# C86 / C79 (Q551, Q591, Q603) on a build machine, CPU only: the rig build of the merged branch
# pascal-p106-v3-b12076-ladder2 (sm_61 + compute_61 PTX, no AVX, test-backend-ops), the extra test binaries from the same
# tree, then every merged branch's own CPU checks on the merged tree, in containers without a GPU (the CUDA toolkit's
# stub libcuda.so stands in for the driver, so the CUDA backend finds no device):
#   1. kcol    C70's decoding check of the k-column kernels (bench/mmvq_pascal/verify/kcol_unit_check.sh) after checking
#              that the bench's copy of the kernels' header is the merged tree's
#   2. sass    per-kernel SASS identity of the merged test-backend-ops against the source branches' builds
#              (sass_identity.sh): every kernel a branch changed must come out of the merge with the same machine code,
#              which is R7's and C79's check without a GPU (the arithmetic of the Q2_0, grid-type and multi-token kernels
#              is that of the builds whose GPU checks run tomorrow)
#   3. gdn     C19/C20's check: test-llama-archs' random qwen4exp on the CPU, fused graph (LLAMA_GDN_FUSE=1) against
#              unfused, without and with rollback snapshots (LLAMA_ARCHS_N_RS_SEQ=0, 3)
#   4. ctx     C22/C28's checks (docs/data/2026-11-04_ckpt_cards/run/checks.sh): test-llama-context on random qwen4exp,
#              qwen35 and mamba models (the store and store_p cases among them) and llama-server's checkpoint arms
#   5. draft   Q564/C71/C68's unit test test-draft-mem
#   6. archs   test-llama-archs on the CPU backend, every architecture; on a failure, the same run with the binary of
#              ARCHS_REF (a build of the base without these branches) tells a merge fault from a base one
# Usage: WORK=<dir> REPO=<this repository> SASS_REFS='<base tag>=<build dir> <tag>[:<cut-from tag>]=<build dir> ...' \
#   [BRANCH=pascal-p106-v3-b12076-ladder2] [JOBS=6] [THREADS=8] [LOCK=<file>] [ARCHS_REF=<dir with test-llama-archs>] \
#   [DOCKER_ARGS=...] bash cpu_checks.sh
#   WORK   the work folder of scripts/build_llamacpp_docker.sh (its llama.cpp clone in $WORK/src needs the fork as remote
#          "fork", with an https fetch URL)
#   LOCK   a file to flock around the build, when several sessions share the build machine
# Output: $WORK/llama.cpp-<date>-<sha>-<branch>.tar.gz (+ tree, $WORK/md5.txt), $WORK/extra/ (+ extra-<sha>.tar.gz),
#   $WORK/checks/<check>/, $WORK/logs/, $WORK/summary.tsv (check, verdict, detail); done when $WORK/DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
SASS_REFS=${SASS_REFS:?SASS_REFS=<base tag>=<build dir> ...}
BRANCH=${BRANCH:-pascal-p106-v3-b12076-ladder2}
JOBS=${JOBS:-6}
THREADS=${THREADS:-8}
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04
D="--cpu-shares 256 ${DOCKER_ARGS:-}"
HERE=$(dirname "$(realpath "$0")")
mkdir -p "$WORK/logs" "$WORK/extra" "$WORK/checks"
rm -f "$WORK/DONE"
printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 10800 "$LOCK" "$@"; else "$@"; fi; }
# a container without a GPU whose CUDA backend loads against the stub driver
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
echo "=== chain started $(date -Is)"

build() {
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$REPO/scripts/build_llamacpp_docker.sh" "fork/$BRANCH" \
        > "$WORK/logs/build.log" 2>&1 || { grep -E "error|Error" "$WORK/logs/build.log" | head -40; return 1; }
    # the test binaries the rig build leaves out, from the same build tree
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cd /work/src
        cmake -B build -DLLAMA_BUILD_TESTS=ON > /dev/null
        nice -n 19 cmake --build build -j "$JOBS" --target test-llama-archs test-llama-context test-draft-mem \
            > /work/logs/extra.log 2>&1 || { tail -40 /work/logs/extra.log; exit 1; }
        cp build/bin/test-llama-archs build/bin/test-llama-context build/bin/test-draft-mem /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs'
}
export -f build
export WORK BRANCH JOBS IMAGE D REPO

echo "=== build started $(date -Is)"
locked bash -c build
rc=$?
echo "=== build ended $(date -Is) rc=$rc"
if [ $rc != 0 ]; then verdict build FAIL "see logs/build.log, logs/extra.log"; date -Is > "$WORK/DONE"; exit 1; fi
t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz | head -1)
T=$WORK/$(basename "$t" .tar.gz)
sha=$(git -C "$WORK/src" rev-parse --short=9 HEAD)
(cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
(cd "$WORK" && tar czf "extra-$sha.tar.gz" extra && md5sum "extra-$sha.tar.gz") | tee -a "$WORK/md5.txt"
cp "$WORK"/extra/test-* "$T/bin/"
verdict build OK "$(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ';')"

echo "=== 1. kcol $(date -Is)"
mkdir -p "$WORK/checks/kcol"
if diff <(tail -n +4 "$REPO/bench/mmvq_pascal/src/kcol_mm.cuh") "$WORK/src/ggml/src/ggml-cuda/mmvq-pascal-kcol-impl.cuh" \
        > "$WORK/checks/kcol/header.diff"; then
    DOCKER_ARGS="$D" nice -n 19 "$REPO/bench/mmvq_pascal/verify/kcol_unit_check.sh" 40 > "$WORK/checks/kcol/unit.log" 2>&1
    rc=$?
    verdict kcol "$([ $rc = 0 ] && echo OK || echo FAIL)" "header = tree; $(tail -n 3 "$WORK/checks/kcol/unit.log" | tr '\n' ' ')"
else
    verdict kcol FAIL "the bench's kcol_mm.cuh is not the merged tree's header (checks/kcol/header.diff)"
fi

echo "=== 2. sass $(date -Is)"
# shellcheck disable=SC2086
WORK=$WORK/checks DOCKER_ARGS="${DOCKER_ARGS:-}" bash "$HERE/sass_identity.sh" "$T" $SASS_REFS > "$WORK/logs/sass.log" 2>&1
rc=$?
verdict sass "$(grep -q '^verdict: every kernel' "$WORK/checks/sass/summary.txt" 2>/dev/null && echo OK || echo REVIEW)" \
    "rc $rc; $(grep -E '^verdict|carried' "$WORK/checks/sass/summary.txt" 2>/dev/null | tr '\n' ' ')"

echo "=== 3. gdn $(date -Is)"
mkdir -p "$WORK/checks/gdn"
docker run --rm $D -v "$T/bin:/x:ro" -v "$WORK/checks/gdn:/k" "$IMAGE" bash -c "$STUB"'
    rm -f /k/*.f32
    for rs in 0 3; do
        for fuse in 0 1; do
            echo "--- n_rs_seq=$rs LLAMA_GDN_FUSE=$fuse"
            LLAMA_ARCHS_N_RS_SEQ=$rs LLAMA_GDN_FUSE=$fuse LLAMA_ARCHS_LOGITS=/k/rs$rs- \
                nice -n 19 /x/test-llama-archs -a "^qwen4exp$" -b CPU -s 1 > /k/archs-rs$rs-fuse$fuse.log 2>&1
            echo "rc=$?"
            grep -E "logits against|one batch against|wrote|FAIL|OK" /k/archs-rs$rs-fuse$fuse.log | sed "s/\x1b\[[0-9;]*m//g"
        done
    done
    chown -R '"$(id -u):$(id -g)"' /k' > "$WORK/checks/gdn/console.txt" 2>&1
cat "$WORK/checks/gdn/console.txt"
nrc=$(grep -c '^rc=0' "$WORK/checks/gdn/console.txt")
nfail=$(grep -c 'FAIL' "$WORK/checks/gdn/console.txt")
verdict gdn "$([ "$nrc" = 4 ] && [ "$nfail" = 0 ] && echo OK || echo FAIL)" \
    "$nrc/4 runs rc 0, $nfail FAIL lines; $(grep -E 'logits against' "$WORK/checks/gdn/console.txt" | tr -s ' ' | tr '\n' ';')"

echo "=== 4. ctx $(date -Is)"
RUN=$REPO/docs/data/2026-11-04_ckpt_cards/run
mkdir -p "$WORK/checks/ctx"
docker run --rm $D -v "$T/bin:/x:ro" -v "$RUN:/ckrun:ro" -v "$WORK/checks/ctx:/k" -e THREADS="$THREADS" "$IMAGE" bash -c '
    export DEBIAN_FRONTEND=noninteractive
    apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
    '"$STUB"'
    cd /k && BIN=/x RUN=/ckrun OUT=/k THREADS=$THREADS nice -n 19 bash /ckrun/checks.sh
    rc=$?
    chown -R '"$(id -u):$(id -g)"' /k
    exit $rc' > "$WORK/checks/ctx/checks.log" 2>&1
rc=$?
tail -n 20 "$WORK/checks/ctx/checks.log"
st=$(awk -F'\t' 'NR > 1 && $2 ~ /^store/ { print $1 ":" $2 "=" $3 }' "$WORK/checks/ctx/context.tsv" 2>/dev/null | tr '\n' ' ')
tally=$(awk -F'\t' 'NR > 1 { n[$3]++ } END { for (v in n) printf "%s %d; ", v, n[v] }' "$WORK/checks/ctx/context.tsv" 2>/dev/null)
verdict ctx "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; cases: $tally store cases: $st; server: $(head -n 1 "$WORK/checks/ctx/server/verdict.txt" 2>/dev/null)"

echo "=== 5. draft $(date -Is)"
mkdir -p "$WORK/checks/draft"
docker run --rm $D -v "$T/bin:/x:ro" "$IMAGE" bash -c "$STUB"' && nice -n 19 /x/test-draft-mem' \
    > "$WORK/checks/draft/test-draft-mem.log" 2>&1
rc=$?
verdict draft "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc $rc; $(tail -n 2 "$WORK/checks/draft/test-draft-mem.log" | tr '\n' ' ')"

echo "=== 6. archs $(date -Is)"
mkdir -p "$WORK/checks/archs"
archs() { # <dir with test-llama-archs> <log>
    docker run --rm $D -v "$1:/x:ro" "$IMAGE" bash -c "$STUB"' && nice -n 19 /x/test-llama-archs -b CPU -s 1' 2>&1 \
        | sed 's/\x1b\[[0-9;]*m//g' > "$2"
    return "${PIPESTATUS[0]}"
}
archs "$T/bin" "$WORK/checks/archs/merged.log"
rc=$?
# one table row per architecture and configuration: | arch | device | config | nmse | roundtrip | mixed batch |
nf=$(grep -c '^|.*FAIL' "$WORK/checks/archs/merged.log")
detail="rc $rc, $(grep -m 1 '^Summary' "$WORK/checks/archs/merged.log"); rows with a FAIL: $nf"
if [ "$nf" != 0 ] && [ -n "${ARCHS_REF:-}" ]; then
    archs "$(realpath "$ARCHS_REF")" "$WORK/checks/archs/ref.log"
    detail="$detail; ARCHS_REF: $(grep -m 1 '^Summary' "$WORK/checks/archs/ref.log")"
    diff <(grep '^|.*FAIL' "$WORK/checks/archs/ref.log" | cut -d'|' -f2-4) \
        <(grep '^|.*FAIL' "$WORK/checks/archs/merged.log" | cut -d'|' -f2-4) > "$WORK/checks/archs/fail.diff" \
        && detail="$detail, failing in the same rows"
fi
verdict archs "$([ "$rc" = 0 ] && [ "$nf" = 0 ] && echo OK || echo REVIEW)" "$detail"

cat "$WORK/md5.txt"
date -Is > "$WORK/DONE"
echo "=== chain ended $(date -Is)"
