#!/usr/bin/env bash
# C983 (2026-11-24): llama-quantize's default rules and qwen4exp's ple_norm_key/query/conv tensors, on a processor only.
# Two trees of Miner Llama:
#   base  TEST: pascal-p106-v5 plus the new test only (test-llama-archs --quantize and its ctest), the rules as they were
#   head  COMMIT: TEST plus the rule that keeps ple_norm_* unquantized
# Steps (STEPS, in this order):
#   src     a local clone of SEED, the fork's BRANCH fetched and checked against COMMIT; TEST must be COMMIT's parent and
#           BASE TEST's; the compiler cache hard-linked from CPU_CCACHE, its counters zeroed
#   cpu     cpu_tests.sh (REPO) on base and head@base, static: a CPU build of every target under LOCK, ctest without
#           network, test-llama-archs (plain, write, compare) and head's logits against base's (ident); then ccache's
#           counters
#   bins    for each tree: llama-quantize, llama-completion and test-llama-archs, built with the cpu step's options and
#           compiler cache in $W/bin-<tree>, under LOCK
#   models  random models from test-llama-archs -s 1: every architecture's (head's build, as ctest's
#           test-generate-models writes them) in models/all; qwen4exp, qwen35moe and flint with a 4,096-token
#           vocabulary from each tree's build in models/v4k-<tree> (the two trees' files must be the same)
#   types   every model of models/all and models/v4k-head quantized to Q8_0 with the default rules (llama-quantize
#           <in> <out> Q8_0 4) by each tree's build; each output's tensor types read from its header (model_types.py,
#           REPO); checks/types.tsv: per model, the tensors whose type differs between the trees. models/all and its
#           Q8_0 copies are deleted afterwards; the v4k copies are kept for the cli step
#   cli     the v4k models' Q8_0 copies run as a reader would: llama-completion, one token, each tree's copy with its
#           own build; base's llama-quantize with --tensor-type ple_norm=f32 (C957's exception) against head's default
#           file, md5
#   sweep   each tree's test-llama-archs --quantize -s 1, one architecture per process (an abort stops only its own),
#           SWEEP_JOBS at a time: which architectures' models decode a token once the default rules quantized them
#   ctest   ctest test by test for base and head ($W/checks/ctest.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone> REPO=<folder with cpu_tests.sh and model_types.py> (under W)
#   [TEST=<sha>] [COMMIT=<sha>] [BASE=<sha>] [CPU_CCACHE=<dir>] [LOCK=<flock file>]
#   [STEPS='src cpu bins models types cli sweep ctest'] [JOBS=8] [SWEEP_JOBS=6] bash run.sh
# Every container: labels pascal-quants=1 and c983=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh and model_types.py>}
TEST=${TEST:-9320d0e054729ef041a894c25e848c340b902756}
COMMIT=${COMMIT:-91eead67a09aa961689cd86c4f865ed9a6c132e0}
BASE=${BASE:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5-quantize-ple-norm}
STEPS=${STEPS:-src cpu bins models types cli sweep ctest}
JOBS=${JOBS:-8}
SWEEP_JOBS=${SWEEP_JOBS:-6}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c983=1"
MEM="--memory=16g --memory-swap=16g"
D="--rm $LBL --cpu-shares=256 $MEM --network none -v $W:$W"
CMAKE_OPTS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF -DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF -DLLAMA_CURL=OFF -DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
ARCHS3='^(qwen4exp|qwen35moe|flint)$'
CC=$W/cpu/ccache
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
locked() { if [ -n "${LOCK:-}" ]; then flock -w 21600 "$LOCK" "$@"; else "$@"; fi; }
finish() { docker ps -q --filter label=c983=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
box()    { docker run $D -w /tmp "$CPU_IMAGE" "$@"; }
cc()     { docker run $D -e CCACHE_DIR="$CC" "$CPU_IMAGE" ccache "$@"; }
export LC_ALL=C
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
echo "=== start $(date -Is): base $TEST (on $BASE), head $COMMIT; steps: $STEPS"

S=$W/src
if step src; then
    [ -d "$S/.git" ] || git clone -q --no-checkout "$SEED" "$S" || finish 1
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/fork/$BRANCH")
    echo "fork/$BRANCH = $got"
    [ "$got" = "$(git -C "$S" rev-parse "$COMMIT")" ] || { echo "$BRANCH is not at $COMMIT: stop"; finish 1; }
    [ "$(git -C "$S" rev-parse "$COMMIT^")" = "$(git -C "$S" rev-parse "$TEST")" ] || { echo "$TEST is not $COMMIT's parent: stop"; finish 1; }
    [ "$(git -C "$S" rev-parse "$TEST^")" = "$(git -C "$S" rev-parse "$BASE")" ] || { echo "$BASE is not $TEST's parent: stop"; finish 1; }
    git -C "$S" diff --stat "$BASE" "$TEST" | tail -n 1
    git -C "$S" diff --stat "$TEST" "$COMMIT" | tail -n 1
    mkdir -p "$W/cpu"
    [ -d "$CC" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$CC"; }
    [ -d "$CC" ] && cc -z > /dev/null
    echo "=== sources ready $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$S" TREES="base=$TEST head=$COMMIT@base" STEPS="tree build ctest archs ident" SHARED=OFF \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" \
        IMAGE="$CPU_IMAGE" bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu tests rc=$?"
    cat "$W/cpu/summary.tsv"
    cc -s > "$W/checks/ccache-cpu.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-cpu.txt"
    echo "=== cpu done $(date -Is)"
fi

if step bins; then
    for t in base head; do
        R=$W/bin-$t
        c=$TEST; [ $t = head ] && c=$COMMIT
        rm -rf "$R" && mkdir -p "$R/src"
        git -C "$S" archive "$c" | tar -x -C "$R/src"
        echo "commit $(git -C "$S" rev-parse "$c")" > "$R/tree.txt"
        s=$(date +%s)
        locked docker run $D -e CCACHE_DIR="$CC" -e CCACHE_BASEDIR="$R" "$CPU_IMAGE" bash -c "
            cmake -S $R/src -B $R/build $CMAKE_OPTS > $R/cmake.log 2>&1 &&
            nice -n 19 cmake --build $R/build -j $JOBS --target llama-quantize llama-completion test-llama-archs > $R/build.log 2>&1"
        echo "bin-$t: build rc=$? s=$(( $(date +%s) - s )); $(cat "$R/tree.txt")" | tee "$R/RC"
    done
    cc -s > "$W/checks/ccache-bins.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-bins.txt"
    echo "=== bins done $(date -Is)"
fi

M=$W/models
if step models; then
    rm -rf "$M" && mkdir -p "$M/all" "$M/v4k-base" "$M/v4k-head"
    box bash -c "nice -n 19 $W/bin-head/build/bin/test-llama-archs -s 1 -o $M/all > $M/all.log 2>&1"
    echo "all: rc=$? $(ls "$M/all" | wc -l) files, $(du -sh "$M/all" | cut -f1)"
    for t in base head; do
        box bash -c "LLAMA_TEST_ARCHS_N_VOCAB=4096 nice -n 19 $W/bin-$t/build/bin/test-llama-archs -a '$ARCHS3' -s 1 \
            -o $M/v4k-$t > $M/v4k-$t.log 2>&1"
        echo "v4k-$t: rc=$? $(ls "$M/v4k-$t" | tr '\n' ' ')"
    done
    (cd "$M/v4k-base" && md5sum ./*.gguf) > "$W/checks/v4k-base.md5"
    (cd "$M/v4k-head" && md5sum ./*.gguf) > "$W/checks/v4k-head.md5"
    if cmp -s "$W/checks/v4k-base.md5" "$W/checks/v4k-head.md5"; then echo "v4k models: the same from both trees"
    else echo "v4k models: DIFFER between the trees"; diff "$W/checks/v4k-base.md5" "$W/checks/v4k-head.md5"; fi
    echo "=== models done $(date -Is)"
fi

if step types; then
    rm -rf "$W/types" && mkdir -p "$W/types"
    for t in base head; do
        mkdir -p "$W/types/$t" "$W/q8-$t/all" "$W/q8-$t/v4k"
        # shellcheck disable=SC2016
        box bash -c '
            Q='"$W/bin-$t/build/bin/llama-quantize"' O='"$W/q8-$t"' T='"$W/types/$t"'
            for f in '"$M"'/all/*.gguf '"$M"'/v4k-head/*.gguf; do
                set=all; case $f in */v4k-head/*) set=v4k ;; esac
                n=$(basename "$f" .gguf)
                nice -n 19 timeout 600 "$Q" "$f" "$O/$set/$n.gguf" Q8_0 4 > "$T/$set-$n.quant.log" 2>&1
                rc=$?
                if [ $rc = 0 ]; then
                    python3 -I '"$REPO"'/model_types.py -v "$O/$set/$n.gguf" | grep -aP "^  " | sort > "$T/$set-$n.types"
                else
                    : > "$T/$set-$n.types"
                fi
                echo "$set	$n	$rc"
                [ $set = all ] && rm -f "$O/$set/$n.gguf"
            done' > "$W/types/$t.rc.tsv" 2> "$W/types/$t.err"
        echo "types $t: $(wc -l < "$W/types/$t.rc.tsv") files, $(awk -F'\t' '$3 != 0' "$W/types/$t.rc.tsv" | wc -l) not quantized (rc != 0)"
    done
    {
        printf 'set\tmodel\trc_base\trc_head\ttensors\tquantized_base\tquantized_head\tdiffer\ttensors_that_differ\n'
        while IFS=$'\t' read -r set n rcb; do
            rch=$(awk -F'\t' -v s="$set" -v m="$n" '$1 == s && $2 == m {print $3}' "$W/types/head.rc.tsv")
            b=$W/types/base/$set-$n.types h=$W/types/head/$set-$n.types
            nt=$(wc -l < "$h")
            qb=$(awk -F'\t' '$2 != "F32" && $2 != "F16" && $2 != "BF16" && $2 != "I32"' "$b" | wc -l)
            qh=$(awk -F'\t' '$2 != "F32" && $2 != "F16" && $2 != "BF16" && $2 != "I32"' "$h" | wc -l)
            dl=$(diff <(cut -f1,2 "$b") <(cut -f1,2 "$h") | grep -a '^>' | sed -E 's/^> +//' | tr '\t' ':' | paste -sd' ')
            nd=0; [ -n "$dl" ] && nd=$(echo "$dl" | wc -w)
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$set" "$n" "$rcb" "${rch:-missing}" "$nt" "$qb" "$qh" "$nd" "$dl"
        done < "$W/types/base.rc.tsv"
    } > "$W/checks/types.tsv"
    echo "types: $(tail -n +2 "$W/checks/types.tsv" | wc -l) models; with a tensor type that differs between the trees:"
    tail -n +2 "$W/checks/types.tsv" | awk -F'\t' '$8 != 0 || $3 != $4 {print "  " $1 "/" $2 ": rc " $3 "/" $4 ", " $8 " differ: " substr($9, 1, 200)}'
    rm -rf "$M/all" "$W/q8-base/all" "$W/q8-head/all"
    echo "=== types done $(date -Is)"
fi

if step cli; then
    rm -rf "$W/cli" && mkdir -p "$W/cli"
    for t in base head; do
        for f in "$W/q8-$t/v4k/"*.gguf; do
            n=$(basename "$f" .gguf)
            box bash -c "nice -n 19 timeout 300 $W/bin-$t/build/bin/llama-completion -m $f -p 'The quick brown fox' -n 1 \
                -c 256 -t 4 -no-cnv --temp 0 > $W/cli/$t-$n.log 2>&1"
            rc=$?
            why=$(grep -a -m1 -oE '(binary_op: unsupported types.*|GGML_ASSERT.*|error: .*)' "$W/cli/$t-$n.log" | cut -c1-160)
            printf '%s\t%s\t%s\t%s\n' "$t" "$n" "$rc" "$why"
        done
    done | tee "$W/checks/cli.tsv"
    # C957's hand exception on base's rules against head's default rules: the same file?
    mkdir -p "$W/q8-override"
    for f in "$M/v4k-head/"*.gguf; do
        n=$(basename "$f" .gguf)
        box bash -c "nice -n 19 timeout 600 $W/bin-base/build/bin/llama-quantize --tensor-type ple_norm=f32 $f \
            $W/q8-override/$n.gguf Q8_0 4 > $W/cli/override-$n.quant.log 2>&1"
        printf '%s\t%s\t%s\t%s\n' "$n" "$?" "$(md5sum < "$W/q8-override/$n.gguf" | cut -c1-32)" "$(md5sum < "$W/q8-head/v4k/$n.gguf" | cut -c1-32)"
    done | awk -F'\t' 'BEGIN { OFS = "\t"; print "model", "rc_override", "md5_base_override", "md5_head_default", "same" }
                       { print $1, $2, $3, $4, ($3 == $4 ? "yes" : "no") }' | tee "$W/checks/override.tsv"
    echo "=== cli done $(date -Is)"
fi

if step sweep; then
    rm -rf "$W/sweep" && mkdir -p "$W/sweep"
    sed -n '/LLM_ARCH_NAMES = {/,/^};/p' "$W/bin-head/src/src/llama-arch.cpp" | grep -oE '^ *\{ *LLM_ARCH_[A-Z0-9_]+, *"[^"]+"' \
        | sed -E 's/.*"([^"]+)"$/\1/' | grep -v '^(unknown)$' > "$W/sweep/archs.txt"
    echo "sweep: $(wc -l < "$W/sweep/archs.txt") architectures, $SWEEP_JOBS at a time"
    for t in base head; do
        mkdir -p "$W/sweep/$t"
        # shellcheck disable=SC2016
        box bash -c '
            one() {
                cd /tmp && TMPDIR=/tmp nice -n 19 timeout 600 '"$W/bin-$t/build/bin/test-llama-archs"' --quantize -s 1 -a "^$1\$" \
                    > '"$W/sweep/$t"'/"$1".log 2>&1
                echo "$?" > '"$W/sweep/$t"'/"$1".rc
            }
            export -f one
            xargs -P '"$SWEEP_JOBS"' -I{} bash -c "one {}" < '"$W/sweep/archs.txt"
        {
            printf 'arch\trc\trows\n'
            while read -r a; do
                rows=$(sed 's/\x1b\[[0-9;]*m//g' "$W/sweep/$t/$a.log" | grep -a -E "^\| *$a *\|" | sed -E 's/ +\|/|/g; s/\| +/|/g' | paste -sd' ')
                why=$(grep -a -m1 -oE '(binary_op: unsupported types.*|GGML_ASSERT.*|GGML_ABORT.*|terminate called.*)' "$W/sweep/$t/$a.log" | cut -c1-160)
                printf '%s\t%s\t%s\n' "$a" "$(cat "$W/sweep/$t/$a.rc")" "$rows${why:+ ABORT: $why}"
            done < "$W/sweep/archs.txt"
        } > "$W/checks/sweep-$t.tsv"
        echo "sweep $t: rc 0 $(awk -F'\t' 'NR > 1 && $2 == 0' "$W/checks/sweep-$t.tsv" | wc -l), other $(awk -F'\t' 'NR > 1 && $2 != 0' "$W/checks/sweep-$t.tsv" | wc -l)"
    done
    join -t "$(printf '\t')" -j 1 <(tail -n +2 "$W/checks/sweep-base.tsv" | cut -f1,2 | sort) <(tail -n +2 "$W/checks/sweep-head.tsv" | cut -f1,2 | sort) \
        | awk -F'\t' '$2 != $3 {print "  " $1 ": base rc " $2 ", head rc " $3}' | tee "$W/checks/sweep-changed.txt"
    echo "=== sweep done $(date -Is)"
fi

if step ctest; then
    ctest_tsv "$W/cpu/base/ctest.log" > "$W/checks/ctest-base.tsv"
    ctest_tsv "$W/cpu/head/ctest.log" > "$W/checks/ctest-head.tsv"
    join -t "$(printf '\t')" -a 1 -a 2 -e missing -o 0,1.2,2.2 "$W/checks/ctest-base.tsv" "$W/checks/ctest-head.tsv" \
        | awk -F'\t' 'BEGIN { OFS = "\t"; print "test", "base", "head", "same" } { print $1, $2, $3, ($2 == $3 ? "yes" : "no") }' \
        > "$W/checks/ctest.tsv"
    echo "ctest: $(tail -n +2 "$W/checks/ctest.tsv" | wc -l) tests; not the same in base and head:"
    tail -n +2 "$W/checks/ctest.tsv" | awk -F'\t' '$4 == "no"'
    echo "passed: base $(grep -c 'Passed' "$W/checks/ctest-base.tsv"), head $(grep -c 'Passed' "$W/checks/ctest-head.tsv")"
fi
finish 0
