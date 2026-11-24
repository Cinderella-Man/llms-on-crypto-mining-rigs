#!/usr/bin/env bash
# C935 (2026-11-24): the pipe step of cpu_tests.sh (pipelined decoding against plain decoding on random models of every
# architecture llama_pipe_init accepts, and flint) in one CPU test run of two trees, processor only, no GPU:
#   tip  pascal-p106-v5 at COMMIT as it is: it still carries C932's fault (Flash-Next's fused conv window writing every
#        pipe slot's history to state group 0), so the step must give qwen4exp DIFFERS and every other model OK;
#   fix  COMMIT with C932's fix FIX (fork branch pascal-p106-v5-pipe-logits) applied: every model OK, and its
#        test-llama-archs logits the tip's (plain decoding keeps group 0, so its graph is unchanged).
# The run lists its steps as the routine drivers do (tree build ctest archs ident hand), without naming pipe: the hand
# step brings it. Then every other check is compared with BEFORE, an earlier run of COMMIT by cpu_tests.sh before the
# pipe step (same seeds, so the hand logs compare line by line).
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding COMMIT), the fork's BRANCH and FIX_BRANCH fetched and checked
#           against COMMIT and FIX; the compiler cache hard-linked from CPU_CCACHE (another work folder's, so it hits
#           only through ccache's base folder), its counters zeroed
#   run     cpu_tests.sh (REPO, with the scripts of bench/pipespec in REPO/pipespec) in $W/cpu on the trees tip=COMMIT
#           and fix=COMMIT+FIX@tip; then ccache's counters
#   compare the pipe step's verdicts of both trees (pipe.tsv); every other check of tip and fix against BEFORE: the
#           verdicts (checks.tsv), ctest test by test (ctest.tsv) and the hand logs line by line (compare_hand.py, REPO)
# Usage: W=<work dir> SEED=<llama.cpp clone with COMMIT> REPO=<folder with cpu_tests.sh, compare_hand.py and pipespec/>
#   BEFORE=<an earlier run's tree dir of COMMIT, $WORK/<tag>> [COMMIT=<sha>] [FIX=<sha>] [CPU_CCACHE=<dir>]
#   [LOCK=<flock file>] [STEPS='src run compare'] [JOBS=8] bash run.sh
# Every container: labels pascal-quants=1 and c935=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with COMMIT>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh, compare_hand.py and pipespec/>}
BEFORE=${BEFORE:?BEFORE=<an earlier run tree dir of COMMIT>}
COMMIT=${COMMIT:-8a9629942a4635ea0b2711b72e59556b96a63f86}
FIX=${FIX:-80d83fa1a0e0ca18d20970b1e023ae8e6ea91e45}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5}
FIX_BRANCH=${FIX_BRANCH:-pascal-p106-v5-pipe-logits}
STEPS=${STEPS:-src run compare}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c935=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c935=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$1" "$CPU_IMAGE" ccache "${@:2}"; }
export LC_ALL=C # sort and join in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
echo "=== start $(date -Is): $BRANCH at $COMMIT, fix $FIX ($FIX_BRANCH); before: $BEFORE; steps: $STEPS"

if step src; then
    S=$W/src
    if [ ! -d "$S/.git" ]; then
        git clone -q --no-checkout "$SEED" "$S" || finish 1
    fi
    git -C "$S" remote get-url fork > /dev/null 2>&1 || git -C "$S" remote add fork "$FORK_URL"
    for b in "$BRANCH" "$FIX_BRANCH"; do git -C "$S" fetch -q fork "$b:refs/remotes/fork/$b" || finish 1; done
    got=$(git -C "$S" rev-parse "refs/remotes/fork/$BRANCH")
    gotfix=$(git -C "$S" rev-parse "refs/remotes/fork/$FIX_BRANCH")
    echo "fork/$BRANCH = $got; fork/$FIX_BRANCH = $gotfix"
    [ "$got" = "$COMMIT" ] || { echo "$BRANCH is not at $COMMIT: stop"; finish 1; }
    [ "$gotfix" = "$FIX" ] || { echo "$FIX_BRANCH is not at $FIX: stop"; finish 1; }
    if git -C "$S" merge-base --is-ancestor "$FIX" "$COMMIT"; then echo "the fix is already in $BRANCH"; else echo "the fix is not in $BRANCH"; fi
    mkdir -p "$W/cpu"
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    [ -d "$W/cpu/ccache" ] && cc "$W/cpu/ccache" -z
    echo "=== sources ready $(date -Is)"
fi

if step run; then
    WORK="$W/cpu" SRC="$W/src" TREES="tip=$COMMIT fix=$COMMIT+$FIX@tip" STEPS="tree build ctest archs ident hand" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "run: cpu tests rc=$?"
    cat "$W/cpu/summary.tsv"
    cc "$W/cpu/ccache" -s > "$W/checks/ccache.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache.txt"
    echo "=== run done $(date -Is)"
fi

if step compare; then
    C=$W/checks
    # the pipe step's verdicts, both trees
    { printf 'tree\tarch\tcheck\tverdict\tdetail\n'
      awk -F'\t' '$2 ~ /^pipe/ { split($2, p, " "); printf "%s\t%s\t%s\t%s\t%s\n", $1, p[2], p[3], $3, $4 }' "$W/cpu/summary.tsv"
    } > "$C/pipe.tsv"
    cat "$C/pipe.tsv"
    for t in tip fix; do cp "$W/cpu/$t/pipe/models.md5" "$C/models-$t.md5" 2> /dev/null; done

    # every other check's verdict: BEFORE, tip, fix (the details hold counts and paths, so only verdicts are compared)
    python3 -I - "$BEFORE" "$W/cpu" "$C/checks.tsv" << 'EOF'
import sys
from pathlib import Path
before, w, out = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
def rows(summary, tag):
    r = {}
    for l in Path(summary).read_text().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 3 and f[0] == tag and not f[1].startswith("pipe"):
            # the identity check names its base, which is another tag in each run
            r["ident" if f[1].startswith("ident vs ") else f[1]] = f[2]
    return r
tabs = [rows(before.parent / "summary.tsv", before.name), rows(w / "summary.tsv", "tip"), rows(w / "summary.tsv", "fix")]
with open(out, "w") as f:
    f.write("check\tbefore\ttip\tfix\tsame\n")
    for k in sorted(set().union(*tabs)):
        v = [t.get(k, "-") for t in tabs]
        # tip runs no identity check (no base tree) and fix's base is tip, so ident compares where it ran
        ran = [x for x in v if x != "-"]
        f.write(k + "\t" + "\t".join(v) + "\t" + ("yes" if len(set(ran)) == 1 else "no") + "\n")
EOF
    cat "$C/checks.tsv"
    grep -hP '^fix\tident vs tip\t' "$W/cpu/summary.tsv"

    # ctest, test by test: BEFORE, tip, fix
    ctest_tsv "$BEFORE/ctest.log" > "$C/ctest-before.tsv"
    for t in tip fix; do ctest_tsv "$W/cpu/$t/ctest.log" > "$C/ctest-$t.tsv"; done
    python3 -I - "$C/ctest.tsv" "$C/ctest-before.tsv" "$C/ctest-tip.tsv" "$C/ctest-fix.tsv" << 'EOF'
import sys
out, files = sys.argv[1], sys.argv[2:]
tables = [dict(l.split("\t", 1) for l in open(f).read().splitlines() if l) for f in files]
names = sorted(set().union(*tables))
with open(out, "w") as f:
    f.write("test\tbefore\ttip\tfix\tsame\n")
    for n in names:
        v = [t.get(n, "missing").strip() for t in tables]
        f.write(n + "\t" + "\t".join(v) + "\t" + ("yes" if len(set(v)) == 1 else "no") + "\n")
EOF
    echo "ctest: $(tail -n +2 "$C/ctest.tsv" | wc -l) tests, $(awk -F'\t' 'NR > 1 && $NF == "no"' "$C/ctest.tsv" | wc -l) not the same in every column:"
    awk -F'\t' 'NR == 1 || $NF == "no"' "$C/ctest.tsv"
    for i in 2 3 4; do awk -F'\t' -v i=$i 'NR == 1 {h = $i; next} $i == "Passed" {p++} END {printf "%s: %d passed\n", h, p}' "$C/ctest.tsv"; done

    # the hand logs, line by line
    rcs=$C/compare_rc.tsv
    : > "$rcs"
    pair() { # <name> <tree dir a> <tree dir b>
        python3 -I "$REPO/compare_hand.py" "$2" "$3" "$C/hand-$1.tsv" > "$C/hand-$1.txt" 2>&1
        local rc=$?
        printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$rc" >> "$rcs"
        echo "--- hand $1 (rc=$rc)"
        cat "$C/hand-$1.txt"
    }
    pair before-tip "$BEFORE" "$W/cpu/tip"
    pair tip-fix "$W/cpu/tip" "$W/cpu/fix"
    echo "=== compare done $(date -Is)"
fi
finish 0
