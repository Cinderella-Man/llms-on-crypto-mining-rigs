#!/usr/bin/env bash
# C914 (2026-11-24): one CPU test run of Miner Llama's cumulative branch pascal-p106-v5 with cpu_tests.sh after its
# change (the link $WORK/src -> tree/src, so test-jinja and test-jinja-py find the files beside their source), compared
# test by test with an earlier run of the same commit by the script before the change. Processor only, no GPU.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding COMMIT), the fork's branch fetched and checked against
#           COMMIT; the compiler cache hard-linked from CPU_CCACHE (another work folder's, so it hits only through
#           ccache's base folder), its counters zeroed; the earlier run's logits files copied to $W/cpu/before/
#   cpu     cpu_tests.sh (REPO) on the tree c914=COMMIT@before: a CPU build of every target under LOCK, ctest without
#           network, test-llama-archs (plain, write, compare), its logits against the earlier run's (ident), the
#           hand-run tests; then ccache's counters for this build
#   compare ctest test by test against BEFORE/BEFORE_TAG (ctest.tsv), the known failures (failed or not run) of both,
#           the other checks' verdicts (summary.tsv) of both
# Usage: W=<work dir> SEED=<llama.cpp clone with COMMIT> REPO=<folder with cpu_tests.sh> BEFORE=<earlier run's WORK>
#   [BEFORE_TAG=land] [CPU_CCACHE=<dir>] [COMMIT=<sha>] [LOCK=<flock file>] [STEPS='src cpu compare'] [JOBS=8]
#   bash run.sh
# Every container: labels pascal-quants=1 and c914=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with COMMIT>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh>}
BEFORE=${BEFORE:?BEFORE=<WORK folder of the earlier run>}
BEFORE_TAG=${BEFORE_TAG:-land}
COMMIT=${COMMIT:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5}
STEPS=${STEPS:-src cpu compare}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c914=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/cpu" "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c914=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$W/cpu/ccache" "$CPU_IMAGE" ccache "$@"; }
export LC_ALL=C # sort, join and comm in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
# the tests ctest left out ("not run: ^(a|b)$" in the ctest verdict), one name a line
not_run() { grep -P "^$2\tctest\t" "$1" | sed -E 's/.*not run: \^\((.*)\)\$.*/\1/' | tr '|' '\n' | sort; }
echo "=== start $(date -Is): $BRANCH at $COMMIT; steps: $STEPS"

if step src; then
    S=$W/src
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    git -C "$S" fetch -q fork "$BRANCH:refs/remotes/fork/$BRANCH" || finish 1
    got=$(git -C "$S" rev-parse "refs/remotes/fork/$BRANCH")
    echo "fork/$BRANCH = $got"
    [ "$got" = "$COMMIT" ] || { echo "$BRANCH is not at $COMMIT: stop"; finish 1; }
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    [ -d "$W/cpu/ccache" ] && cc -z
    rm -rf "$W/cpu/before" && mkdir -p "$W/cpu/before/archs"
    cp -r "$BEFORE/$BEFORE_TAG/archs/logits" "$W/cpu/before/archs/" || finish 1
    echo "the earlier run's logits files: $(ls "$W/cpu/before/archs/logits" | wc -l)"
    echo "=== sources ready $(date -Is)"
fi

if step cpu; then
    WORK="$W/cpu" SRC="$W/src" TREES="c914=$COMMIT@before" STEPS="tree build ctest archs ident hand" JOBS="$JOBS" \
        CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" KEEP_BUILD=1 \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "cpu tests rc=$?"
    grep -P '^c914\t' "$W/cpu/summary.tsv"
    cc -s > "$W/checks/ccache.txt" 2>&1
    cat "$W/checks/ccache.txt"
    echo "=== cpu done $(date -Is)"
fi

if step compare; then
    C=$W/checks
    ctest_tsv "$BEFORE/$BEFORE_TAG/ctest.log" > "$C/ctest-before.tsv"
    ctest_tsv "$W/cpu/c914/ctest.log" > "$C/ctest-after.tsv"
    { printf 'test\tbefore\tafter\tsame\n'
      join -t $'\t' -a 1 -a 2 -e missing -o 0,1.2,2.2 "$C/ctest-before.tsv" "$C/ctest-after.tsv" \
        | awk -F'\t' -v OFS='\t' '{print $1, $2, $3, ($2 == $3 ? "yes" : "no")}'
    } > "$C/ctest.tsv"
    echo "ctest: $(tail -n +2 "$C/ctest.tsv" | wc -l) tests, $(tail -n +2 "$C/ctest.tsv" | awk -F'\t' '$4 == "no"' | wc -l) differ:"
    tail -n +2 "$C/ctest.tsv" | awk -F'\t' '$4 == "no"'
    # the known failures: failed in ctest, or left out of it
    for x in before after; do
        if [ $x = before ]; then sum=$BEFORE/summary.tsv tag=$BEFORE_TAG; else sum=$W/cpu/summary.tsv tag=c914; fi
        { awk -F'\t' -v OFS='\t' '$2 != "Passed" {print $1, "failed (" $2 ")"}' "$C/ctest-$x.tsv"
          not_run "$sum" "$tag" | awk -v OFS='\t' 'NF {print $1, "not run in ctest"}'
        } | sort > "$C/known-$x.tsv"
        echo "known failures $x: $(wc -l < "$C/known-$x.tsv")"
        cat "$C/known-$x.tsv"
    done
    # the other checks: the same verdict, detail as written (the hand-run tests' random models are new each run)
    { printf 'check\tbefore\tafter\tsame\n'
      join -t $'\t' -a 1 -a 2 -e missing -o 0,1.2,2.2 \
        <(grep -P "^$BEFORE_TAG\t" "$BEFORE/summary.tsv" | cut -f2,3 | grep -vP '^(ctest|ident vs )' | sort) \
        <(grep -P '^c914\t' "$W/cpu/summary.tsv" | cut -f2,3 | grep -vP '^(ctest|ident vs )' | sort) \
        | awk -F'\t' -v OFS='\t' '{print $1, $2, $3, ($2 == $3 ? "yes" : "no")}'
    } > "$C/checks.tsv"
    cat "$C/checks.tsv"
    grep -P '^c914\tident vs before\t' "$W/cpu/summary.tsv"
    echo "=== compare done $(date -Is)"
fi
finish 0
