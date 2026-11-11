#!/usr/bin/env bash
# C703 (Q1242): the MTP draft's prompt catch-up queued behind the next call (LLAMA_MTP_CATCHUP_OVERLAP) of a fork
# branch, checked on a build machine, CPU only. Made from C620's checks.sh (docs/data/2026-11-11_c620_draft_kv_catchup).
# Steps:
#   build     the rig tarball of the branch (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code, no AVX), then
#             test-llama-archs from the same CUDA build tree into the tarball's bin/
#   cpubuild  the branch without CUDA (static, native): test-llama-archs into $WORK/cpu/bin; llama-server is compiled
#             too (its draft hooks), not run
#   overlap   test-llama-archs --mtp-catchup-overlap with each build's binary there is (the rig build's CPU code has no
#             AVX) at each seed of SEEDS: on the CPU backend the target and the draft-mtp driver in the server's order,
#             the draft's sequence state and the driver's at a checkpoint, after the prompt and at the end, and every
#             draft and its probabilities, with the switch off against on
#   kvonly    test-llama-archs --mtp-kv-only, the same way: C620's check, unchanged by this branch
#   off       test-llama-archs -b CPU -s 1 on every architecture, the logits written by the base's CPU build (BASE_CPU)
#             and compared by the branch's: with the switch off the branch computes what the base computes
# Usage: WORK=<dir> BUILD=<scripts/build_llamacpp_docker.sh> BRANCH=<fork branch> BASE_CPU=<bin/ of the base's static
#   native CPU build> [REMOTE=fork] [JOBS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [SEEDS='1 2 3']
#   [STEPS='cpubuild overlap kvonly off build'] bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL; $WORK/ccache
#   LOCK      one build or test at a time across the sessions that share the machine: every step waits for it
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/md5.txt, logs in $WORK/logs and $WORK/checks; done when
#   $WORK/C703_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BUILD=$(realpath "${BUILD:?BUILD=<scripts/build_llamacpp_docker.sh>}")
: "${BRANCH:?BRANCH=<fork branch>}" "${BASE_CPU:?BASE_CPU=<bin/ of the base CPU build>}"
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-1 2 3}
STEPS=${STEPS:-cpubuild overlap kvonly off build}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs" "$WORK/checks" "$WORK/cpu"
rm -f "$WORK/C703_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
locked()  { if [ -n "${LOCK:-}" ]; then flock -w 43200 "$LOCK" "$@"; else "$@"; fi; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }
tarball_tree() { local t; t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1); [ -n "$t" ] && echo "$WORK/$(basename "$t" .tar.gz)"; }
# a container without a GPU whose CUDA backend loads against the stub driver, so that the rig build runs on the CPU
STUB='mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub'
git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
echo "=== c703 checks started $(date -Is), $BRANCH at ${TIP:0:9}, steps: $STEPS"

if step cpubuild; then
    echo "=== cpubuild $(date -Is)"
    # the branch's tip in the clone (the build step checks it out again in its container)
    git -C "$WORK/src" checkout -q --force --detach "$TIP"
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory /work/src
        cmake -S /work/src -B /work/cpu/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
            -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
            -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > /work/logs/cpu-cmake.log 2>&1
        rc=0
        nice -n 19 cmake --build /work/cpu/build -j "$JOBS" --target test-llama-archs llama-server > /work/logs/cpu-build.log 2>&1 || rc=$?
        if [ $rc -eq 0 ]; then
            rm -rf /work/cpu/bin && mkdir -p /work/cpu/bin && cp /work/cpu/build/bin/test-llama-archs /work/cpu/bin/
        fi
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs /work/ccache
        exit $rc'
    rc=$?
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; commit $(git -C "$WORK/src" rev-parse --short=9 HEAD)"
fi

# one test-llama-archs check ($1: overlap or kvonly, $2: its flag) with each build's binary at each seed; the table has
# one row per build, seed and configuration
archs_check() {
    local name=$1 flag=$2 K=$WORK/checks/$1 T b bin s rc sum
    rm -rf "$K" && mkdir -p "$K"
    T=$(tarball_tree)
    for b in rig cpu; do
        if [ $b = rig ]; then bin=${T:+$T/bin}; else bin=$WORK/cpu/bin; fi
        if [ -z "$bin" ] || [ ! -x "$bin/test-llama-archs" ]; then
            verdict "$name-$b" SKIP "no test-llama-archs in ${bin:-a tarball tree} yet"
            continue
        fi
        for s in $SEEDS; do
            # shellcheck disable=SC2086
            locked docker run --rm $D -v "$WORK:$WORK" "$IMAGE" bash -c "$STUB"'
                nice -n 19 '"$bin"'/test-llama-archs '"$flag"' -s '"$s"' 2>&1 | sed "s/\x1b\[[0-9;]*m//g" \
                    > '"$K/$b-s$s.log"'; echo "rc=${PIPESTATUS[0]}"; chown '"$(id -u):$(id -g)"' '"$K/$b-s$s.log" \
                > "$K/$b-s$s.rc" 2>&1
            rc=$(sed -n 's/^rc=//p' "$K/$b-s$s.rc")
            sum=$(grep -m 1 '^Summary' "$K/$b-s$s.log")
            verdict "$name-$b-s$s" "$([ "$rc" = 0 ] && [[ "$sum" == *"all "*" passed"* ]] && echo OK || echo FAIL)" \
                "rc=$rc; $sum"
        done
    done
    { printf 'build\tseed\tconfig'; grep -m 1 '^| *Model arch' "$(ls "$K"/*-s*.log | head -n 1)" \
          | awk -F'|' '{ for (i = 4; i < NF; i++) { gsub(/^ +| +$/, "", $i); printf "\t%s", tolower($i) } printf "\tstatus\n" }'
      for f in "$K"/*-s*.log; do
          n=$(basename "$f" .log)
          grep '^| *qwen4exp|' "$f" | awk -F'|' -v b="${n%-s*}" -v s="${n##*-s}" 'BEGIN { OFS = "\t" }
              { printf "%s\t%s", b, s; for (i = 3; i <= NF; i++) { gsub(/^ +| +$/, "", $i); printf "\t%s", $i } printf "\n" }'
      done; } > "$K/$name.tsv"
}

if step overlap; then
    echo "=== overlap $(date -Is)"
    archs_check overlap --mtp-catchup-overlap
fi

if step kvonly; then
    echo "=== kvonly $(date -Is)"
    archs_check kvonly --mtp-kv-only
fi

if step off; then
    echo "=== off $(date -Is)"
    K=$WORK/checks/off
    rm -rf "$K" && mkdir -p "$K/k"
    bb=$(realpath "$BASE_CPU")
    mb=$WORK/cpu/bin
    # shellcheck disable=SC2086
    locked docker run --rm $D -v "$WORK:$WORK" -v "$bb:$bb:ro" -v "$K/k:/k" "$IMAGE" bash -c '
        run() { env LLAMA_ARCHS_LOGITS=/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > /k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        run base '"$bb"'
        ls /k/logits-*.f32 | wc -l > /k/n_base.txt
        run branch '"$mb"'
        rm -f /k/*.f32
        chown -R '"$(id -u):$(id -g)"' /k' > "$K/rc.txt" 2>&1
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep 'logits against' "$K/k/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
    nb=$(cat "$K/k/n_base.txt" 2>/dev/null || echo 0)
    nc=$(grep -c 'logits against' "$K/k/branch.log")
    grep 'logits against' "$K/k/branch.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" > "$K/differ.txt"
    bad=$(cut -d: -f1 "$K/differ.txt" | tr '\n' ' ')
    sb=$(grep -m 1 '^Summary' "$K/k/base.log") sm=$(grep -m 1 '^Summary' "$K/k/branch.log")
    verdict off "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && { [ -z "$bad" ] && echo OK || echo REVIEW; } || echo FAIL)" \
        "base wrote $nb logits files; $nc compared, $(wc -l < "$K/differ.txt") differ: ${bad:-none}; base: $sb; branch: $sm; $(grep -o -E '(base|branch) rc=[0-9]+' "$K/rc.txt" | tr '\n' ' ')"
fi

if step build; then
    echo "=== build $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" locked nice -n 19 "$BUILD" "$REMOTE/$BRANCH" > "$WORK/logs/rig.log" 2>&1
    rc=$?
    T=$(tarball_tree)
    if [ $rc = 0 ] && [ -n "$T" ]; then
        (cd "$WORK" && md5sum "$(basename "$T").tar.gz" && stat -c '%n %s bytes' "$(basename "$T").tar.gz") | tee -a "$WORK/md5.txt"
        verdict build-rig OK "$(basename "$T"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/rig.log") error lines in logs/rig.log"
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
        rm -rf /work/extra && mkdir -p /work/extra && cp build/bin/test-llama-archs /work/extra/
        chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs /work/ccache' > "$WORK/logs/extra.log" 2>&1
    rc=$?
    if [ $rc = 0 ] && [ -n "$T" ]; then
        cp "$WORK/extra/test-llama-archs" "$T/bin/"
        verdict build-extra OK "test-llama-archs in $(basename "$T")/bin"
    else
        verdict build-extra FAIL "rc=$rc; logs/extra.log"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C703_DONE"
