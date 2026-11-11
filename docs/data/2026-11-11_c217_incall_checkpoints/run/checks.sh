#!/usr/bin/env bash
# C217 (Q770): context checkpoints inside a prompt's decode call (--ctx-checkpoints-incall) of a fork branch, checked
# on a build machine, CPU only. Steps:
#   build     the rig tarball of the branch (scripts/build_llamacpp_docker.sh, TESTS=1: sm_61 code, no AVX), then
#             test-llama-context and test-llama-archs from the same CUDA build tree into the tarball's tree
#   cpubuild  the branch and its base without CUDA (static, native): test-llama-archs, test-llama-context and
#             llama-server of the branch into $WORK/cpu/bin, test-llama-archs and llama-server of BASE_REF into
#             $WORK/cpu/base
#   context   random-weight models from test-llama-archs (qwen4exp, Flash-Next's architecture; qwen35; mamba), then
#             test-llama-context on each: the new store_at and store_at_p cases (a state stored inside one call at a
#             cut is the state of the calls cut there, byte for byte, with and without rollback snapshots) and every
#             other case
#   control   the store_at cases again with their logs: whether one call without the cuts would pass as well
#   off       test-llama-archs -b CPU -s 1 on every architecture, the logits written by the base's build and compared
#             by the branch's: with the switch off the branch computes what the base computes
#   server    server_check.sh on the qwen4exp model: the base's llama-server against the branch's, off and on
# Usage: WORK=<dir> BUILD=<scripts/build_llamacpp_docker.sh> BRANCH=<fork branch> [BASE_REF=<base commit>]
#   [REMOTE=fork] [JOBS=8] [THREADS=8] [LOCK=<flock file>] [DOCKER_ARGS=...] [STEPS='build cpubuild context control off server']
#   bash checks.sh
#   WORK      the work folder of scripts/build_llamacpp_docker.sh: its llama.cpp clone in $WORK/src has the fork as
#             remote REMOTE with an https fetch URL; $WORK/ccache
#   LOCK      one build or test at a time across the jobs that share the machine: the steps run under it, taken once
# Output: $WORK/summary.tsv (check, verdict, detail), $WORK/md5.txt, logs in $WORK/logs, results in $WORK/checks; done
#   when $WORK/C217_DONE exists
set -uo pipefail
WORK=$(realpath "${WORK:?WORK=<work dir>}")
BUILD=$(realpath "${BUILD:?BUILD=<scripts/build_llamacpp_docker.sh>}")
RUN=$(dirname "$(realpath "$0")")
: "${BRANCH:?BRANCH=<fork branch>}"
BASE_REF=${BASE_REF:-84f7bd3d2}
REMOTE=${REMOTE:-fork}
JOBS=${JOBS:-8}
THREADS=${THREADS:-8}
STEPS=${STEPS:-build cpubuild context control off server}
IMAGE=${IMAGE:-nvidia/cuda:12.9.2-devel-ubuntu24.04}
D="--cpu-shares=256 ${DOCKER_ARGS:-}"
mkdir -p "$WORK/logs" "$WORK/checks" "$WORK/cpu"
rm -f "$WORK/C217_DONE"
[ -f "$WORK/summary.tsv" ] || printf 'check\tverdict\tdetail\n' > "$WORK/summary.tsv"
verdict() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" | tee -a "$WORK/summary.tsv"; }
step()    { [[ " $STEPS " == *" $1 "* ]]; }

# the whole run under the lock, once
if [ -n "${LOCK:-}" ] && [ -z "${C217_LOCKED:-}" ]; then
    echo "=== waiting for $LOCK $(date -Is)"
    exec env C217_LOCKED=1 flock -w 43200 "$LOCK" bash "$0"
fi

git -C "$WORK/src" fetch -q "$REMOTE" "$BRANCH:refs/remotes/$REMOTE/$BRANCH"
TIP=$(git -C "$WORK/src" rev-parse "$REMOTE/$BRANCH")
echo "=== c217 checks started $(date -Is), $BRANCH at ${TIP:0:9}, base $BASE_REF, steps: $STEPS"

if step build; then
    echo "=== build $(date -Is)"
    TESTS=1 JOBS=$JOBS WORK=$WORK DOCKER_ARGS="$D" nice -n 19 "$BUILD" "$REMOTE/$BRANCH" > "$WORK/logs/rig.log" 2>&1
    rc=$?
    t=$(ls -t "$WORK"/llama.cpp-*-"$BRANCH".tar.gz 2>/dev/null | head -n 1)
    if [ $rc = 0 ] && [ -n "$t" ]; then
        T="$WORK/$(basename "$t" .tar.gz)"
        (cd "$WORK" && md5sum "$(basename "$t")" && stat -c '%n %s bytes' "$(basename "$t")") | tee -a "$WORK/md5.txt"
        verdict build-rig OK "$(basename "$t"); $(grep -E '^(commit|avx|highest)' "$T/BUILDINFO.txt" | tr '\n' ' ')"
    else
        verdict build-rig FAIL "rc=$rc; $(grep -a -c -E ' error|error:' "$WORK/logs/rig.log") error lines in logs/rig.log"
    fi
    # test-llama-context and test-llama-archs from the same CUDA build tree, into the tarball's tree (not the tarball):
    # the same cases on one card later
    if [ $rc = 0 ] && [ -n "$t" ]; then
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
            bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
            apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
            git config --global --add safe.directory "*"
            cd /work/src
            nice -n 19 cmake --build build -j "$JOBS" --target test-llama-context test-llama-archs > /work/logs/extra-build.log 2>&1 \
                || { tail -n 40 /work/logs/extra-build.log; exit 1; }
            rm -rf /work/extra && mkdir -p /work/extra && cp build/bin/test-llama-context build/bin/test-llama-archs /work/extra/
            chown -R "$HOST_UID:$HOST_GID" /work/extra /work/logs /work/ccache' > "$WORK/logs/extra.log" 2>&1
        rc=$?
        [ $rc = 0 ] && cp "$WORK"/extra/test-llama-* "$T/bin/"
        verdict build-extra "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; test-llama-context, test-llama-archs in $(basename "$T")/bin"
    fi
fi

if step cpubuild; then
    echo "=== cpubuild $(date -Is)"
    # the branch at its tip and the base in a worktree of the same clone, each in its own build folder
    git -C "$WORK/src" worktree remove --force "$WORK/cpu/base-src" 2>/dev/null
    git -C "$WORK/src" worktree add -q --detach "$WORK/cpu/base-src" "$BASE_REF"
    git -C "$WORK/src" checkout -q --detach "$TIP"
    # shellcheck disable=SC2086
    docker run --rm $D -v "$WORK:/work" -e JOBS="$JOBS" -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" "$IMAGE" \
        bash -euo pipefail -c '
        export DEBIAN_FRONTEND=noninteractive CCACHE_DIR=/work/ccache
        apt-get update -qq && apt-get install -y -qq git cmake build-essential ccache > /dev/null
        git config --global --add safe.directory "*"
        build() { # <source> <build dir> <bin dir> <targets...>
            local src=$1 bld=$2 bin=$3; shift 3
            cmake -S "$src" -B "$bld" -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF \
                -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DCMAKE_C_COMPILER_LAUNCHER=ccache \
                -DCMAKE_CXX_COMPILER_LAUNCHER=ccache > "$bld.cmake.log" 2>&1
            nice -n 19 cmake --build "$bld" -j "$JOBS" --target "$@" > "$bld.build.log" 2>&1 || return 1
            rm -rf "$bin" && mkdir -p "$bin"
            for t in "$@"; do cp "$bld/bin/$t" "$bin/"; done
        }
        rc=0
        build /work/src /work/cpu/build /work/cpu/bin test-llama-archs test-llama-context llama-server || rc=1
        build /work/cpu/base-src /work/cpu/base-build /work/cpu/base test-llama-archs llama-server || rc=$((rc + 2))
        chown -R "$HOST_UID:$HOST_GID" /work/cpu /work/logs /work/ccache
        exit $rc' > "$WORK/logs/cpubuild.log" 2>&1
    rc=$?
    verdict cpubuild "$([ $rc = 0 ] && echo OK || echo FAIL)" \
        "rc=$rc (1 = branch, 2 = base); branch $(git -C "$WORK/src" rev-parse --short=9 HEAD), base $(git -C "$WORK/cpu/base-src" rev-parse --short=9 HEAD)"
fi

# a container without a GPU for the CPU builds' binaries
cpu_run() { # <log> <command>: runs the command in the image with WORK mounted
    # shellcheck disable=SC2086
    docker run --rm $D -v "$WORK:$WORK" -w "$WORK" "$IMAGE" bash -c "$2"' ; echo "rc=$?"' > "$1" 2>&1
}

if step context; then
    echo "=== context $(date -Is)"
    K=$WORK/checks/context
    rm -rf "$K" && mkdir -p "$K/models"
    for a in qwen4exp qwen35 mamba; do
        cpu_run "$K/models/save-$a.log" "nice -n 19 $WORK/cpu/bin/test-llama-archs -a '^$a\$' -s 1 -o $K/models"
    done
    printf 'model\tcase\tverdict\treason\n' > "$K/context.tsv"
    for m in "$K"/models/*.gguf; do
        n=$(basename "$m" .gguf)
        cpu_run "$K/context-$n.log" "cd $K && nice -n 19 $WORK/cpu/bin/test-llama-context -m $m -t $THREADS"
        # one verdict line per case and run: "<case>  OK|SKIP|FAIL  [reason]" (the recurrent cases twice, the second
        # time against a cache pre-filled with a pattern); a log line of another thread may leave its timestamp in front
        sed 's/\x1b\[[0-9;]*m//g; s/^[0-9][0-9]*\.[0-9][0-9]\.[0-9][0-9][0-9]\.[0-9][0-9][0-9] //' "$K/context-$n.log" | \
            awk -v m="$n" '$2 ~ /^(OK|SKIP|FAIL)$/ { r = $3; for (i = 4; i <= NF; i++) r = r " " $i; print m "\t" $1 "\t" $2 "\t" r }' \
            >> "$K/context.tsv"
        grep -h 'store_at' "$K/context-$n.log" | grep -i 'n_rs_seq' | sed 's/\x1b\[[0-9;]*m//g' >> "$K/store_at.txt"
    done
    nf=$(awk -F'\t' '$3 == "FAIL"' "$K/context.tsv" | wc -l)
    ns=$(awk -F'\t' '$2 ~ /^store_at/ && $3 == "OK"' "$K/context.tsv" | wc -l)
    nm=$(ls "$K"/models/*.gguf 2>/dev/null | wc -l)
    verdict context "$([ "$nm" -gt 0 ] && [ "$nf" = 0 ] && [ "$ns" -gt 0 ] && echo OK || echo FAIL)" \
        "$nm models; $(awk -F'\t' 'NR > 1' "$K/context.tsv" | wc -l) case runs, $nf FAIL; store_at/store_at_p OK in $ns runs"
fi

if step control; then
    echo "=== control $(date -Is)"
    # the store_at cases again with their logs kept: each reports whether one call without the cuts, and with rollback
    # snapshots the cuts without their rule, end in the cut calls' state too (where they do, the case cannot tell that
    # split from the right one)
    K=$WORK/checks/control
    rm -rf "$K" && mkdir -p "$K"
    cpu_run "$K/run.log" "cd $K && nice -n 19 $WORK/cpu/bin/test-llama-context --models $WORK/checks/context/models --logs $K/logs --test '^store_at' -t $THREADS"
    grep -rh 'n_rs_seq 0:' "$K/logs" | sed 's/\x1b\[[0-9;]*m//g' > "$K/store_at.txt"
    n=$(wc -l < "$K/store_at.txt")
    ne=$(grep -o 'an uncut call ends elsewhere' "$K/store_at.txt" | wc -l)
    nr=$(grep -o 'cuts without the rule end elsewhere' "$K/store_at.txt" | wc -l)
    verdict control "$([ "$n" -gt 0 ] && echo OK || echo FAIL)" \
        "$n case runs report; one call without cuts ends elsewhere than the cut calls in $ne of $(grep -o 'an uncut call ends' "$K/store_at.txt" | wc -l) (model, flags, n_rs_seq), the cuts without the rule of the rollback snapshots in $nr of $(grep -o 'cuts without the rule end' "$K/store_at.txt" | wc -l)"
fi

if step off; then
    echo "=== off $(date -Is)"
    K=$WORK/checks/off
    rm -rf "$K" && mkdir -p "$K/k"
    cpu_run "$K/run.log" '
        run() { env LLAMA_ARCHS_LOGITS='"$K"'/k/logits- nice -n 19 $2/test-llama-archs -b CPU -s 1 2>&1 \
                    | sed "s/\x1b\[[0-9;]*m//g" > '"$K"'/k/$1.log; echo "$1 rc=${PIPESTATUS[0]}"; }
        run base '"$WORK"'/cpu/base
        ls '"$K"'/k/logits-*.f32 | wc -l > '"$K"'/k/n_base.txt
        run branch '"$WORK"'/cpu/bin
        rm -f '"$K"'/k/*.f32'
    # a base run compares its own later passes with the file its first pass wrote: what differs there is the base's own
    # spread, not the branch's
    grep 'logits against' "$K/k/base.log" | grep -v ', 0 of ' > "$K/base_self.txt"
    nb=$(cat "$K/k/n_base.txt" 2>/dev/null || echo 0)
    nc=$(grep -c 'logits against' "$K/k/branch.log")
    grep 'logits against' "$K/k/branch.log" | grep -v ', 0 of ' | grep -vxFf "$K/base_self.txt" > "$K/differ.txt"
    bad=$(cut -d: -f1 "$K/differ.txt" | tr '\n' ' ')
    sb=$(grep -m 1 '^Summary' "$K/k/base.log") sm=$(grep -m 1 '^Summary' "$K/k/branch.log")
    verdict off "$([ "$nb" -gt 0 ] && [ "$nc" -gt 0 ] && { [ -z "$bad" ] && echo OK || echo REVIEW; } || echo FAIL)" \
        "base wrote $nb logits files; $nc compared, $(wc -l < "$K/differ.txt") differ: ${bad:-none}; base: $sb; branch: $sm; $(grep -o -E '(base|branch) rc=[0-9]+' "$K/run.log" | tr '\n' ' ')"
fi

if step server; then
    echo "=== server $(date -Is)"
    K=$WORK/checks/server
    q=$(ls "$WORK"/checks/context/models/*qwen4exp*.gguf 2>/dev/null | head -n 1)
    rm -rf "$K" && mkdir -p "$K"
    if [ -n "$q" ]; then
        cp "$RUN/server_check.sh" "$K/"
        # shellcheck disable=SC2086
        docker run --rm $D -v "$WORK:$WORK" -w "$WORK" "$IMAGE" bash -c '
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq && apt-get install -y -qq curl jq > /dev/null
            BASE='"$WORK"'/cpu/base BIN='"$WORK"'/cpu/bin MODEL='"$q"' OUT='"$K"' THREADS='"$THREADS"' \
                SARGS="--override-kv qwen4exp.context_length=int:16384" nice -n 19 bash '"$K"'/server_check.sh' \
            > "$K/console.log" 2>&1
        verdict server "$(tail -n 1 "$K/verdict.txt" 2>/dev/null | grep -q '^PASS$' && echo OK || echo FAIL)" \
            "$(grep -v -e '^PASS$' -e '^FAIL$' "$K/verdict.txt" 2>/dev/null | tr '\n' ';')"
    else
        verdict server FAIL "no qwen4exp model"
    fi
fi

echo "=== done $(date -Is)"
touch "$WORK/C217_DONE"
