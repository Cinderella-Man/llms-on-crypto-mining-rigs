#!/usr/bin/env bash
# C705 (Q1246) on a build machine, CPU only, no GPU: the fork branch's rig tarball with test-backend-ops; the sparse
# attention's CPU check with the prompt rule (bench/fattn_sparse/check.sh); the branch's GPU code against the base
# build's (cuobjdump -sass and -ptx of llama-server and test-backend-ops); the replay set for a P106
# (scripts/record_checks.sh, suite qsa-prompt). Every step takes LOCK (flock), so on a shared machine one build or
# check runs at a time, and runs at a low CPU share (--cpu-shares=256, nice 19).
# Env: WORK (work folder: src/ with the fork as remote "fork", ccache/, the tarballs, logs/, sets/), REPO (this
#   repository), BRANCH (the fork branch), LOCK (the lock file), BASE (the base build's tarball, for the GPU code),
#   STEPS (default "build check code record"), JOBS (8), DOCKER_ARGS (extra docker run arguments, e.g. a label)
# Prints one "C705_STEP <step> ..." line per step and C705_DONE at the end; the logs are in WORK/logs/.
set -uo pipefail
: "${WORK:?WORK=<work folder>}" "${REPO:?REPO=<this repository>}" "${BRANCH:?BRANCH=<fork branch>}" "${LOCK:?LOCK=<lock file>}"
STEPS=${STEPS:-build check code record}
JOBS=${JOBS:-8}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
DA="--cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-}"
L=$WORK/logs
mkdir -p "$L"
step() { [[ " $STEPS " == *" $1 "* ]]; }
say() { echo "C705_STEP $* ($(date -Is))"; }
locked() { flock "$LOCK" nice -n 19 "$@"; }
tree() { ls -td "$WORK"/llama.cpp-*-"$BRANCH" 2> /dev/null | grep -v '\.tar\.gz$' | head -n 1; }

if step build; then
    locked env TESTS=1 JOBS="$JOBS" WORK="$WORK" DOCKER_ARGS="$DA" bash "$REPO/scripts/build_llamacpp_docker.sh" \
        "fork/$BRANCH" > "$L/build.log" 2>&1
    rc=$?
    T=$(tree)
    [ "$rc" = 0 ] && [ -x "$T/bin/test-backend-ops" ] || { say "build FAILED rc=$rc: $L/build.log"; exit 1; }
    say "build done: $(basename "$T").tar.gz $(stat -c %s "$T.tar.gz") B md5 $(md5sum < "$T.tar.gz" | cut -d' ' -f1)"
fi
T=$(tree)

if step check; then
    # the tree the build checked out is the source of the check
    locked env SRC="$WORK/src" JOBS="$JOBS" THREADS="$JOBS" DOCKER_ARGS="$DA" bash "$REPO/bench/fattn_sparse/check.sh" \
        > "$L/check.log" 2>&1
    rc=$?
    cp "$REPO/bench/fattn_sparse/build/check.txt" "$L/check.txt" 2> /dev/null
    say "check rc=$rc: $(grep -c '^ok ' "$L/check.log") ok, $(grep -c '^WRONG ' "$L/check.log") WRONG"
fi

if step code; then
    # the GPU code of the branch against the base build's: the change is host code, so every kernel must be the same
    : "${BASE:?BASE=<tarball of the base build>}"
    mkdir -p "$WORK/base"
    b=$(tar tzf "$BASE" | head -n 1 | cut -d/ -f1)
    [ -x "$WORK/base/$b/bin/test-backend-ops" ] || tar xzf "$BASE" -C "$WORK/base" "$b/bin/llama-server" "$b/bin/test-backend-ops"
    # shellcheck disable=SC2086
    locked docker run --rm $DA -v "$WORK:$WORK" "$IMAGE" bash -c '
        for x in llama-server test-backend-ops; do
            for w in sass ptx; do
                for d in "$1/base/$2" "$3"; do
                    nice -n 19 cuobjdump -$w "$d/bin/$x" | grep -v "^[[:space:]]*$" > "$d.$x.$w"
                    echo "$x $w $(md5sum < "$d.$x.$w" | cut -d" " -f1) $(wc -l < "$d.$x.$w") lines $(basename "$d")"
                done
                # the listings are kept only where they differ
                if cmp -s "$1/base/$2.$x.$w" "$3.$x.$w"; then
                    echo "$x $w SAME"
                    rm -f "$1/base/$2.$x.$w" "$3.$x.$w"
                else
                    echo "$x $w DIFFERS"
                fi
            done
        done' sh "$WORK" "$b" "$T" > "$L/code.log" 2>&1
    say "code: $(grep -c ' SAME$' "$L/code.log") SAME, $(grep -c ' DIFFERS$' "$L/code.log") DIFFERS (of 4)"
fi

if step record; then
    locked env STEPS="record tar" OUT="$WORK/sets" SUITE=qsa-prompt BIN="$T/bin" JOBS="$JOBS" DOCKER_ARGS="${DOCKER_ARGS:-}" \
        bash "$REPO/scripts/record_checks.sh" > "$L/record.log" 2>&1
    rc=$?
    say "record rc=$rc: $(grep 'manifest:' "$WORK/sets/qsa-prompt.log" | tail -n 1 | sed 's/^[^ ]* //'); $(grep 'tar:' "$WORK/sets/qsa-prompt.log" | tail -n 1 | sed 's/^[^ ]* //')"
fi
echo C705_DONE
