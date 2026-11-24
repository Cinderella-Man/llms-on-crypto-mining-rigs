#!/usr/bin/env bash
# C933 (2026-11-24): cpu_tests.sh's hand step on fixed random models. Two CPU test runs, each of a base tree and a head
# tree, with cpu_tests.sh after its change (the hand step's models made with test-llama-archs -s HAND_SEED), then the
# hand logs compared line by line (compare_hand.py): each tree against itself across the two runs, and the base against
# the head. Processor only, no GPU.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding BASE and COMMIT), the fork's branch fetched and checked
#           against COMMIT; the compiler cache hard-linked from CPU_CCACHE (another work folder's, so it hits only
#           through ccache's base folder), its counters zeroed
#   run1    cpu_tests.sh (REPO) in $W/run1 on the trees base=BASE and head=COMMIT@base: a CPU build of every target
#           under LOCK, ctest without network, test-llama-archs (plain, write, compare), the head's logits against the
#           base's (ident), the hand-run tests; then ccache's counters
#   run2    the same in $W/run2, its compiler cache hard-linked from run1's
#   compare the hand logs (compare_hand.py, REPO): BEFORE_A against BEFORE_B (two runs of one commit by the script
#           before the change, if given), base and head each against itself across the runs, base against head in
#           each run; ctest test by test and the other checks' verdicts (summary.tsv) of the four trees and BEFORE
# Usage: W=<work dir> SEED=<llama.cpp clone with BASE and COMMIT> REPO=<folder with cpu_tests.sh and compare_hand.py>
#   [BASE=<sha>] [COMMIT=<sha>] [BEFORE=<an earlier run's tree dir, $WORK/<tag>>] [BEFORE_A=<tree dir> BEFORE_B=<tree dir>]
#   [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS='src run1 run2 compare'] [JOBS=8] bash run.sh
# Every container: labels pascal-quants=1 and c933=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with BASE and COMMIT>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh and compare_hand.py>}
BASE=${BASE:-3dfad876f48a41ab2792b23152f5ddd3418ba953}
COMMIT=${COMMIT:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5}
STEPS=${STEPS:-src run1 run2 compare}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c933=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c933=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$1" "$CPU_IMAGE" ccache "${@:2}"; }
export LC_ALL=C # sort and join in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
echo "=== start $(date -Is): base $BASE, head $BRANCH at $COMMIT; steps: $STEPS"

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
    git -C "$S" merge-base --is-ancestor "$BASE" "$COMMIT" || { echo "$BASE is not in $BRANCH: stop"; finish 1; }
    mkdir -p "$W/run1"
    [ -d "$W/run1/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/run1/ccache"; }
    [ -d "$W/run1/ccache" ] && cc "$W/run1/ccache" -z
    echo "=== sources ready $(date -Is)"
fi

for r in run1 run2; do
    step $r || continue
    mkdir -p "$W/$r"
    if [ $r = run2 ] && [ ! -d "$W/run2/ccache" ]; then cp -al "$W/run1/ccache" "$W/run2/ccache" && cc "$W/run2/ccache" -z; fi
    WORK="$W/$r" SRC="$W/src" TREES="base=$BASE head=$COMMIT@base" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/$r.log" 2>&1
    echo "$r: cpu tests rc=$?"
    cat "$W/$r/summary.tsv"
    cc "$W/$r/ccache" -s > "$W/checks/ccache-$r.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-$r.txt"
    echo "=== $r done $(date -Is)"
done

if step compare; then
    C=$W/checks
    rcs=$C/compare_rc.tsv
    : > "$rcs"
    pair() { # <name> <tree dir a> <tree dir b>
        python3 -I "$REPO/compare_hand.py" "$2" "$3" "$C/hand-$1.tsv" > "$C/hand-$1.txt" 2>&1
        local rc=$?
        printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$rc" >> "$rcs"
        echo "--- hand $1 (rc=$rc)"
        cat "$C/hand-$1.txt"
    }
    [ -n "${BEFORE_A:-}" ] && [ -n "${BEFORE_B:-}" ] && pair old-script-twice "$BEFORE_A" "$BEFORE_B"
    pair base-twice "$W/run1/base" "$W/run2/base"
    pair head-twice "$W/run1/head" "$W/run2/head"
    pair base-head-run1 "$W/run1/base" "$W/run1/head"
    pair base-head-run2 "$W/run2/base" "$W/run2/head"

    # ctest, test by test: the earlier run (BEFORE) and the four trees
    cols=() files=()
    if [ -n "${BEFORE:-}" ]; then ctest_tsv "$BEFORE/ctest.log" > "$C/ctest-before.tsv"; cols+=(before); files+=("$C/ctest-before.tsv"); fi
    for r in run1 run2; do for t in base head; do
        ctest_tsv "$W/$r/$t/ctest.log" > "$C/ctest-$r-$t.tsv"; cols+=("$r-$t"); files+=("$C/ctest-$r-$t.tsv")
    done; done
    python3 -I - "$C/ctest.tsv" "${cols[@]}" -- "${files[@]}" << 'EOF'
import sys
out, rest = sys.argv[1], sys.argv[2:]
cut = rest.index("--")
cols, files = rest[:cut], rest[cut + 1:]
tables = [dict(l.split("\t", 1) for l in open(f).read().splitlines() if l) for f in files]
names = sorted(set().union(*tables))
with open(out, "w") as f:
    f.write("test\t" + "\t".join(cols) + "\tsame\n")
    for n in names:
        v = [t.get(n, "missing").strip() for t in tables]
        f.write(n + "\t" + "\t".join(v) + "\t" + ("yes" if len(set(v)) == 1 else "no") + "\n")
EOF
    echo "ctest: $(tail -n +2 "$C/ctest.tsv" | wc -l) tests, $(tail -n +2 "$C/ctest.tsv" | awk -F'\t' '$NF == "no"' | wc -l) not the same in every column:"
    head -n 1 "$C/ctest.tsv"
    tail -n +2 "$C/ctest.tsv" | awk -F'\t' '$NF == "no"'
    awk -F'\t' 'NR == 1 {for (i = 2; i < NF; i++) h[i] = $i; next} {for (i = 2; i < NF; i++) if ($i == "Passed") p[i]++}
        END {for (i in p) printf "%s: %d passed\n", h[i], p[i]}' "$C/ctest.tsv" | sort

    # the other checks' verdicts (the details hold counts and paths): the earlier run's and the four trees'
    { printf 'check'; [ -n "${BEFORE:-}" ] && printf '\tbefore'; printf '\trun1-base\trun1-head\trun2-base\trun2-head\n'
      python3 -I - "${BEFORE:-}" "$W" << 'EOF'
import sys
from pathlib import Path
before, w = sys.argv[1], Path(sys.argv[2])
def rows(summary, tag):
    out = {}
    for l in Path(summary).read_text().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 3 and f[0] == tag:
            out[f[1]] = f[2]
    return out
tabs = []
if before:
    b = Path(before)
    tabs.append(rows(b.parent / "summary.tsv", b.name))
for r in ("run1", "run2"):
    for t in ("base", "head"):
        tabs.append(rows(w / r / "summary.tsv", t))
for k in sorted(set().union(*tabs)):
    print(k + "\t" + "\t".join(t.get(k, "-") for t in tabs))
EOF
    } > "$C/checks.tsv"
    cat "$C/checks.tsv"
    grep -hP '^head\tident vs base\t' "$W/run1/summary.tsv" "$W/run2/summary.tsv"
    # the eval-callback sums, as each run printed them
    for r in run1 run2; do for t in base head; do
        printf '%s\t%s\t%s\n' "$r-$t" "$(grep -ac 'sum = ' "$W/$r/$t/hand/eval-callback.log")" \
            "$(grep -a 'sum = ' "$W/$r/$t/hand/eval-callback.log" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)"
    done; done > "$C/eval_sums.tsv"
    for x in BEFORE_A BEFORE_B; do
        d=${!x:-}
        [ -n "$d" ] && printf '%s\t%s\t%s\n' "$x" "$(grep -ac 'sum = ' "$d/hand/eval-callback.log")" \
            "$(grep -a 'sum = ' "$d/hand/eval-callback.log" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)" >> "$C/eval_sums.tsv"
    done
    echo "eval-callback sums (tree, count, md5 of the list):"
    cat "$C/eval_sums.tsv"
    echo "=== compare done $(date -Is)"
fi
finish 0
