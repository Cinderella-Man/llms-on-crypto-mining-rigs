#!/usr/bin/env bash
# C1007 (2026-11-24): llama-eval-callback and test-thread-safety flush the common logger before main returns. Two CPU
# test runs of a base tree and a head tree (cpu_tests.sh, C565, with C933's seeded hand step), their hand logs compared
# line by line (compare_hand.py); then each tree's two programs run REPEAT times more on the same random model, to count
# how often each log loses its end. Processor only, no GPU.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding BASE), the fork's BRANCH fetched and checked against COMMIT;
#           the compiler cache hard-linked from CPU_CCACHE (another work folder's, so it hits only through ccache's base
#           folder), its counters zeroed
#   run1    cpu_tests.sh (REPO) in $W/run1 on the trees base=BASE and head=COMMIT@base, steps tree build ctest archs
#           ident hand (the pipe step left out: PIPE_ARCHS=''; neither program is in it): a CPU build of every target
#           under LOCK, ctest without network, test-llama-archs (plain, write, compare), the head's logits against the
#           base's (ident), the hand-run tests; then ccache's counters
#   run2    the same in $W/run2, its compiler cache hard-linked from run1's
#   repeat  per tree (base, then head): cpu_tests.sh's tree and build steps in $W/repeat/build (the build kept), then in
#           one container without network llama-eval-callback REPEAT times and test-thread-safety REPEAT times, with the
#           hand step's arguments, on run1's dense random model (the same file in every tree: hand/models.md5); one log
#           per run in $W/repeat/<tree>/
#   compare the hand logs (compare_hand.py, REPO): BEFORE against run1's base (an earlier run of BASE, if given), base
#           and head each against itself across the runs, base against head in each run; ctest test by test and the
#           other checks' verdicts (summary.tsv) of BEFORE and the four trees; eval-callback's sums per tree; the repeat
#           step's logs, each run whole, cut short or different (repeat.tsv, repeat_runs.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone with BASE> REPO=<folder with cpu_tests.sh and compare_hand.py>
#   [BASE=<sha>] [COMMIT=<sha>] [BRANCH=<fork branch at COMMIT>] [BEFORE=<an earlier run's tree dir of BASE>]
#   [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS='src run1 run2 repeat compare'] [JOBS=8] [REPEAT=20] bash run.sh
# Every container: labels pascal-quants=1 and c1007=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with BASE>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh and compare_hand.py>}
BASE=${BASE:-af00bb13b0fd59612bcab545964b61978f04b8d6}
COMMIT=${COMMIT:-4d8fe365adaa011ee70f02cbb678ac78d0ac76c4}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5-evalcb-flush}
STEPS=${STEPS:-src run1 run2 repeat compare}
JOBS=${JOBS:-8}
REPEAT=${REPEAT:-20}
HAND_SEED=${HAND_SEED:-1}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c1007=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c1007=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$1" "$CPU_IMAGE" ccache "${@:2}"; }
export LC_ALL=C # sort and join in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
cpu_tests() { # <work> <trees> <steps> [KEEP_BUILD]
    WORK="$1" SRC="$W/src" TREES="$2" STEPS="$3" KEEP_BUILD="${4:-}" PIPE_ARCHS='' HAND_SEED="$HAND_SEED" \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh"
}
echo "=== start $(date -Is): base $BASE, head $BRANCH at $COMMIT; steps: $STEPS; repeat $REPEAT"

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
    git -C "$S" diff --stat "$BASE" "$COMMIT" | cat
    mkdir -p "$W/run1"
    [ -d "$W/run1/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/run1/ccache"; }
    [ -d "$W/run1/ccache" ] && cc "$W/run1/ccache" -z
    echo "=== sources ready $(date -Is)"
fi

for r in run1 run2; do
    step $r || continue
    mkdir -p "$W/$r"
    if [ $r = run2 ] && [ ! -d "$W/run2/ccache" ]; then cp -al "$W/run1/ccache" "$W/run2/ccache" && cc "$W/run2/ccache" -z; fi
    cpu_tests "$W/$r" "base=$BASE head=$COMMIT@base" "tree build ctest archs ident hand" > "$W/logs/$r.log" 2>&1
    echo "$r: cpu tests rc=$?"
    cat "$W/$r/summary.tsv"
    cc "$W/$r/ccache" -s > "$W/checks/ccache-$r.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-$r.txt"
    echo "=== $r done $(date -Is)"
done

if step repeat; then
    RP=$W/repeat
    L=$W/run1/base/hand/random/llama-dense.gguf
    [ -f "$L" ] || { echo "no $L: run run1 first"; finish 1; }
    mkdir -p "$RP/build"
    [ -d "$RP/build/ccache" ] || cp -al "$W/run2/ccache" "$RP/build/ccache"
    cc "$RP/build/ccache" -z
    for t in base head; do
        c=$BASE; [ $t = head ] && c=$COMMIT
        cpu_tests "$RP/build" "$t=$c" "tree build" 1 > "$W/logs/repeat-build-$t.log" 2>&1
        echo "repeat $t: build rc=$?; $(grep -P "^$t\tbuild\t" "$RP/build/summary.tsv" | tail -n 1)"
        rm -rf "${RP:?}/$t" && mkdir -p "$RP/$t"
        # each run's stdout and stderr into one file from inside the container, as the hand step writes its logs
        # shellcheck disable=SC2016
        docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" bash -c '
            B=$1 L=$2 O=$3 N=$4 S=$5
            for i in $(seq 1 "$N"); do
                nice -n 19 "$B/bin/llama-eval-callback" -m "$L" --prompt hello --seed 42 -ngl 0 > "$O/eval-callback-$i.log" 2>&1
                echo "eval-callback $i rc=$?"
            done
            for i in $(seq 1 "$N"); do
                nice -n 19 "$B/bin/test-thread-safety" -m "$L" -ngl 99 -p "The meaning of life is" -n 128 -c 256 -ub 32 \
                    -np 4 -t 2 --seed "$S" > "$O/thread-safety-$i.log" 2>&1
                echo "thread-safety $i rc=$?"
            done' _ "$RP/build/tree/build" "$L" "$RP/$t" "$REPEAT" "$HAND_SEED" > "$RP/$t/rc.txt" 2>&1
        echo "repeat $t: $(grep -c 'rc=0$' "$RP/$t/rc.txt") of $((2 * REPEAT)) runs rc=0 $(date -Is)"
    done
    rm -rf "${RP:?}/build/tree/build"
    cc "$RP/build/ccache" -s > "$W/checks/ccache-repeat.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-repeat.txt"
    echo "=== repeat done $(date -Is)"
fi

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
    [ -n "${BEFORE:-}" ] && pair before-base-run1 "$BEFORE" "$W/run1/base"
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
    # the eval-callback sums and thread-safety's last line, as each test run's hand step printed them
    for r in run1 run2; do for t in base head; do
        f=$W/$r/$t/hand/eval-callback.log
        printf '%s\t%s\t%s\t%s\t%s\n' "$r-$t" "$(grep -ac 'sum = ' "$f")" "$(wc -l < "$f")" \
            "$(grep -a 'sum = ' "$f" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)" \
            "$(grep -av '^ *$' "$W/$r/$t/hand/thread-safety.log" | tail -n 1 | sed -E 's/^[0-9.]+ [A-Z] //')"
    done; done > "$C/eval_sums.tsv"
    [ -n "${BEFORE:-}" ] && printf '%s\t%s\t%s\t%s\t%s\n' before "$(grep -ac 'sum = ' "$BEFORE/hand/eval-callback.log")" \
        "$(wc -l < "$BEFORE/hand/eval-callback.log")" \
        "$(grep -a 'sum = ' "$BEFORE/hand/eval-callback.log" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)" \
        "$(grep -av '^ *$' "$BEFORE/hand/thread-safety.log" | tail -n 1 | sed -E 's/^[0-9.]+ [A-Z] //')" >> "$C/eval_sums.tsv"
    echo "eval-callback per test run (tree, sums, lines, md5 of the sums, thread-safety's last line):"
    cat "$C/eval_sums.tsv"

    # the repeat step: each run's log whole, cut short or different, against the longest log of either tree
    if [ -d "$W/repeat/base" ]; then
        python3 -I - "$W/repeat" "$C/repeat_runs.tsv" "$C/repeat.tsv" << 'EOF'
import re
import sys
from collections import Counter
from pathlib import Path
rp, runs_out, sum_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
STAMP = re.compile(r"^\d+\.\d+\.\d+\.\d+ [A-Z] ?")
ANY_STAMP = re.compile(r"\d+\.\d{2}\.\d{3}\.\d{3} ")
LEVEL = re.compile(r"^[IWED] ")
SUM = re.compile(r"^\s*sum = (\S+)")
END = "All threads finished without errors."
def lines(p):
    return [STAMP.sub("", x).rstrip() for x in p.read_text(errors="replace").splitlines()]
def rcs(tree):
    out = {}
    for x in (rp / tree / "rc.txt").read_text().splitlines():
        m = re.match(r"(\S+) (\d+) rc=(\d+)$", x)
        if m:
            out[(m.group(1), int(m.group(2)))] = int(m.group(3))
    return out
trees = [t for t in ("base", "head") if (rp / t).is_dir()]
rows = []
for prog in ("eval-callback", "thread-safety"):
    logs = {(t, int(p.stem.rsplit("-", 1)[1])): lines(p) for t in trees for p in (rp / t).glob(f"{prog}-*.log")}
    if prog == "eval-callback":
        keyed = {k: [m.group(1) for m in map(SUM.match, v) if m] for k, v in logs.items()}
    else:
        # several threads write this log, so its lines are compared as a sorted list with every time stamp and a
        # line-start level letter dropped (compare_hand.py's rule)
        keyed = {k: sorted(LEVEL.sub("", ANY_STAMP.sub("", x)) for x in v) for k, v in logs.items()}
    full_len = max(len(v) for v in keyed.values())
    full = Counter(tuple(v) for v in keyed.values() if len(v) == full_len).most_common(1)[0][0]
    # a whole log's last line: the most common last line among the logs that hold the whole list
    end = Counter(next((x for x in reversed(logs[k]) if x.strip()), "")
                  for k, v in keyed.items() if tuple(v) == full).most_common(1)[0][0]
    for t in trees:
        rc = rcs(t)
        for (tt, i), v in sorted(keyed.items()):
            if tt != t:
                continue
            raw = logs[(t, i)]
            last = next((x for x in reversed(raw) if x.strip()), "")
            ok = tuple(v) == full and last == end
            if prog == "eval-callback":
                short = not ok and tuple(v) == full[:len(v)]
                n = f"{len(v)} sums"
            else:
                short = not ok and set(v) <= set(full)
                n = f"{len(raw)} lines"
                ok = ok and last == END
            verdict = "whole" if ok else ("cut short" if short else "DIFFERS")
            rows.append((t, prog, i, rc.get((prog, i), "-"), n, len(raw), verdict, last[:80]))
with runs_out.open("w") as f:
    f.write("tree\tprogram\trun\trc\tcount\tlines\tverdict\tlast line\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
with sum_out.open("w") as f:
    f.write("tree\tprogram\truns\trc 0\twhole\tcut short\tdiffers\tcounts when cut short\n")
    for t in trees:
        for prog in ("eval-callback", "thread-safety"):
            rr = [r for r in rows if r[0] == t and r[1] == prog]
            cut = [r[4] for r in rr if r[6] == "cut short"]
            f.write(f"{t}\t{prog}\t{len(rr)}\t{sum(1 for r in rr if r[3] == 0)}\t"
                    f"{sum(1 for r in rr if r[6] == 'whole')}\t{len(cut)}\t{sum(1 for r in rr if r[6] == 'DIFFERS')}\t"
                    f"{', '.join(cut) or '-'}\n")
print(sum_out.read_text(), end="")
EOF
    fi
    echo "=== compare done $(date -Is)"
fi
finish 0
