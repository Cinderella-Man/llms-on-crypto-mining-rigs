#!/usr/bin/env bash
# GGML_CUDA_KEEPALIVE (C533) on a build machine, CPU only: fork branch pascal-p106-v3-b12076-keepalive built for the rig
# and checked without a GPU, every binary in a container of the build image, and bench/persistent_pipe built with its
# --keepalive.
#   build      the rig tarball (TESTS=1) and the extra tests (EXTRA) from the same tree: the -ladder4 merge's checks.sh
#   cpubuild   the branch without CUDA (static, native): CPU_TARGETS into $WORK/cpu/bin (the same checks.sh)
#   identity   test-llama-archs -b CPU on every architecture, the logits written by -ladder4 and compared by the branch,
#              for the rig builds (BASE_BUILD) and the CPU builds (BASE_CPU); the switch is off, so every architecture
#              must match bit for bit (the same checks.sh)
#   switchon   the same comparison with GGML_CUDA_KEEPALIVE=decode set for the branch's rig build: a run without a
#              CUDA device never starts a block, and the logits still match
#   unit       test-cuda-keepalive from the CPU build, UNIT_RUNS times: the settings, the poll decision and the
#              controller, with a thread standing in for each card's block
#   bench      bench/persistent_pipe/build.sh (sm_61, compute_61 PTX and sm_86; no AVX) into $REPO/bench/persistent_pipe/build
# Usage: WORK=<dir> REPO=<this repository> BASE_BUILD=<-ladder4's unpacked rig tarball, extras in bin/> \
#   BASE_CPU=<bin/ of -ladder4's static native CPU build> [BRANCH=pascal-p106-v3-b12076-keepalive] [REMOTE=fork] \
#   [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [UNIT_RUNS=3] [STEPS='build cpubuild identity switchon unit bench'] \
#   bash checks.sh
#   WORK has the fork's llama.cpp clone in $WORK/src (remote REMOTE with an https fetch URL) and $WORK/ccache.
# Output: as the -ladder4 checks.sh ($WORK/summary.tsv: check, verdict, detail; $WORK/md5.txt), $WORK/checks/unit/,
#   $WORK/checks/switchon/; done when $WORK/KEEPALIVE_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
REPO=$(realpath "${REPO:?REPO=<this repository>}")
BRANCH=${BRANCH:-pascal-p106-v3-b12076-keepalive}
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
UNIT_RUNS=${UNIT_RUNS:-3}
STEPS=${STEPS:-build cpubuild identity switchon unit bench}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
rm -f "$WORK/KEEPALIVE_DONE"
mkdir -p "$WORK/logs" "$WORK/checks"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
echo "=== keepalive checks started $(date -Is), $BRANCH at $(git -C "$WORK/src" rev-parse --short=9 "$REMOTE/$BRANCH"), steps: $STEPS"

# build, cpubuild and identity: the -ladder4 merge's checks.sh, on this branch
l4=""
for s in build cpubuild identity; do
    step $s && l4="$l4 $s"
done
if [ -n "$l4" ]; then
    WORK=$WORK REPO=$REPO BRANCH=$BRANCH REMOTE=$REMOTE BASE_BUILD=${BASE_BUILD:-} BASE_CPU=${BASE_CPU:-} JOBS=$JOBS \
        LOCK=${LOCK:-} DOCKER_ARGS="${DOCKER_ARGS:-}" STEPS="${l4# }" \
        EXTRA="test-llama-archs test-cuda-keepalive" \
        CPU_TARGETS="llama-server test-llama-archs test-backend-ops test-cuda-keepalive" \
        bash "$REPO/docs/data/2026-11-04_ladder4_merge/run/checks.sh" > "$WORK/logs/ladder4-checks.out" 2>&1
fi

if step switchon; then
    echo "=== switchon $(date -Is)"
    K=$WORK/checks/switchon
    rm -rf "$K" && mkdir -p "$K"
    T=$(tarball_tree)
    if [ -z "${BASE_BUILD:-}" ] || [ -z "$T" ]; then
        verdict switchon FAIL "no BASE_BUILD or no tarball tree"
    else
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:$WORK" -v "$BASE_BUILD:$BASE_BUILD" -v "$K:/k" "$IMAGE" bash -c "$STUB"'
            run() { env LLAMA_ARCHS_LOGITS=/k/logits- $3 nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                        | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
            run base '"$BASE_BUILD/bin"' ""
            ls /k/logits-*.f32 | wc -l > /k/n_base.txt
            run on '"$T/bin"' GGML_CUDA_KEEPALIVE=decode
            rm -f /k/*.f32
            chown -R '"$(id -u):$(id -g)"' /k' > "$K/rc.txt" 2>&1
        grep 'logits against' "$K/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
        nb=$(cat "$K/n_base.txt" 2>/dev/null || echo 0)
        nc=$(grep -c 'logits against' "$K/on.log")
        bad=$(grep 'logits against' "$K/on.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" | cut -d: -f1 | tr '\n' ' ')
        sb=$(grep -m 1 '^Summary' "$K/base.log") so=$(grep -m 1 '^Summary' "$K/on.log")
        verdict switchon "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && [ -z "$bad" ] && [ "$sb" = "$so" ] && echo OK || echo FAIL)" \
            "GGML_CUDA_KEEPALIVE=decode, rig build, no device: base wrote $nb logits files; $nc compared, differ: ${bad:-none}; base: $sb; on: $so; $(tr '\n' ' ' < "$K/rc.txt")"
    fi
fi

if step unit; then
    echo "=== unit $(date -Is)"
    K=$WORK/checks/unit
    rm -rf "$K" && mkdir -p "$K"
    ok=0
    for i in $(seq "$UNIT_RUNS"); do
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:$WORK" "$IMAGE" nice -n 19 "$WORK/cpu/bin/test-cuda-keepalive" > "$K/run$i.log" 2>&1 && ok=$((ok + 1))
    done
    verdict unit "$([ $ok = "$UNIT_RUNS" ] && echo OK || echo FAIL)" \
        "test-cuda-keepalive: $ok of $UNIT_RUNS runs pass; $(tail -n 1 "$K/run1.log")"
fi

if step bench; then
    echo "=== bench $(date -Is)"
    DOCKER_ARGS="$D" locked nice -n 19 bash "$REPO/bench/persistent_pipe/build.sh" > "$WORK/logs/bench-build.log" 2>&1
    rc=$?
    B=$REPO/bench/persistent_pipe/build
    if [ $rc = 0 ] && [ -x "$B/persistent_pipe" ]; then
        verdict bench OK "$(grep -E '^(built|avx|highest|md5)' "$B/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict bench FAIL "rc=$rc; logs/bench-build.log"
    fi
fi

date -Is > "$WORK/KEEPALIVE_DONE"
echo "=== done $(date -Is)"
cat "$WORK/summary.tsv"
