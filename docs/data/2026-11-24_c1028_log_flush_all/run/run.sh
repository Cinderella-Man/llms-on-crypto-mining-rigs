#!/usr/bin/env bash
# C1028 (2026-11-24): the common logger flushed when main returns or exits in 38 more programs (a flush guard in
# common/log.h declared at the top of each main, and a flush before each exit() outside a signal handler). One CPU test
# run of a base tree and a head tree (cpu_tests.sh, C565, with C933's seeded hand step and the pipe step, which runs two
# of the changed programs), the hand logs compared line by line (compare_hand.py); then two more changed programs
# (llama-completion, llama-batched-bench) run REPEAT times per tree on the same random model, to count how often each
# log loses its end. Processor only, no GPU.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding BASE), the fork's BRANCH fetched and checked against COMMIT;
#           the compiler cache hard-linked from CPU_CCACHE (another work folder's, so it hits only through ccache's base
#           folder), its counters zeroed
#   run1    cpu_tests.sh (REPO, with pipespec/ beside it) in $W/run1 on the trees base=BASE and head=COMMIT@base, steps
#           tree build ctest archs ident hand (the pipe step comes with hand: llama-pipespec and
#           llama-pipe-recurrent-check, both changed, against plain decoding on five random models): a CPU build of
#           every target under LOCK, ctest without network, test-llama-archs (plain, write, compare), the head's logits
#           against the base's (ident), the hand-run tests; then ccache's counters
#   repeat  per tree (base, then head): cpu_tests.sh's tree and build steps in $W/$ROUND/build (the build kept), then in
#           one container without network each program of PROGS REPEAT times on run1's dense random model (the same file
#           in every tree: hand/models.md5); one log per run in $W/$ROUND/<tree>/. PROGS: completion (llama-completion,
#           16 tokens), batched-bench (llama-batched-bench, 16 prompt and 8 generated tokens at 1 and 2 sequences),
#           embedding (llama-embedding on one prompt: it logs each of the embedding's values as its own log entry, a
#           burst just before main returns), completion-nomodel and tokenize-nomodel (llama-completion and
#           llama-tokenize given a model file that does not exist: the library logs why the load failed, the program
#           logs its own error line and returns 1 at once)
#   compare the hand logs (compare_hand.py, REPO): BEFORE against run1's base (an earlier run of BASE, if given), base
#           against head; ctest test by test and the other checks' verdicts (summary.tsv) of BEFORE and the two trees;
#           eval-callback's sums per tree; every repeat round's logs ($W/repeat*), each run whole, cut short or different,
#           with every number masked, since the timings change from run to run (repeat.tsv, repeat_runs.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone with BASE> REPO=<folder with cpu_tests.sh, compare_hand.py, pipespec/>
#   [BASE=<sha>] [COMMIT=<sha>] [BRANCH=<fork branch at COMMIT>] [BEFORE=<an earlier run's tree dir of BASE>]
#   [CPU_CCACHE=<dir>] [LOCK=<flock file>] [STEPS='src run1 repeat compare'] [JOBS=8] [THREADS=6] [REPEAT=20]
#   [ROUND=repeat] [PROGS='completion batched-bench'] bash run.sh
# The record ran it three times: the defaults, then STEPS='repeat compare' with ROUND=repeat2 PROGS=embedding, and
# with ROUND=repeat3 PROGS='completion-nomodel tokenize-nomodel'.
# Every container: labels pascal-quants=1 and c1028=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with BASE>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh, compare_hand.py and pipespec/>}
BASE=${BASE:-af00bb13b0fd59612bcab545964b61978f04b8d6}
COMMIT=${COMMIT:-cf093e11ca2631f6e72a548313f7c5da0513c39f}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5-log-flush-all}
STEPS=${STEPS:-src run1 repeat compare}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
REPEAT=${REPEAT:-20}
ROUND=${ROUND:-repeat}
PROGS=${PROGS:-completion batched-bench}
HAND_SEED=${HAND_SEED:-1}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
LBL="--label pascal-quants=1 --label c1028=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c1028=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$1" "$CPU_IMAGE" ccache "${@:2}"; }
export LC_ALL=C # sort and join in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
cpu_tests() { # <work> <trees> <steps> [KEEP_BUILD] [PIPE_ARCHS]
    WORK="$1" SRC="$W/src" TREES="$2" STEPS="$3" KEEP_BUILD="${4:-}" PIPE_ARCHS="${5-qwen4exp qwen35 qwen35moe qwen3next flint}" \
        HAND_SEED="$HAND_SEED" JOBS="$JOBS" CTEST_JOBS=6 THREADS="$THREADS" LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" \
        IMAGE="$CPU_IMAGE" bash "$REPO/cpu_tests.sh"
}
echo "=== start $(date -Is): base $BASE, head $BRANCH at $COMMIT; steps: $STEPS; repeat $REPEAT ($ROUND: $PROGS)"

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
    git -C "$S" diff --stat "$BASE" "$COMMIT" | tail -n 1
    mkdir -p "$W/run1"
    [ -d "$W/run1/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/run1/ccache"; }
    [ -d "$W/run1/ccache" ] && cc "$W/run1/ccache" -z
    echo "=== sources ready $(date -Is)"
fi

if step run1; then
    mkdir -p "$W/run1"
    cpu_tests "$W/run1" "base=$BASE head=$COMMIT@base" "tree build ctest archs ident hand" > "$W/logs/run1.log" 2>&1
    echo "run1: cpu tests rc=$?"
    cat "$W/run1/summary.tsv"
    cc "$W/run1/ccache" -s > "$W/checks/ccache-run1.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-run1.txt"
    echo "=== run1 done $(date -Is)"
fi

if step repeat; then
    RP=$W/$ROUND
    L=$W/run1/base/hand/random/llama-dense.gguf
    [ -f "$L" ] || { echo "no $L: run run1 first"; finish 1; }
    mkdir -p "$RP/build"
    [ -d "$RP/build/ccache" ] || cp -al "$W/run1/ccache" "$RP/build/ccache"
    cc "$RP/build/ccache" -z
    for t in base head; do
        c=$BASE; [ $t = head ] && c=$COMMIT
        cpu_tests "$RP/build" "$t=$c" "tree build" 1 '' > "$W/logs/repeat-build-$t.log" 2>&1
        echo "$ROUND $t: build rc=$?; $(grep -P "^$t\tbuild\t" "$RP/build/summary.tsv" | tail -n 1)"
        rm -rf "${RP:?}/$t" && mkdir -p "$RP/$t"
        # each run's stdout and stderr into one file from inside the container, as the hand step writes its logs; stdin
        # closed so that nothing waits for input
        # shellcheck disable=SC2016
        docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" bash -c '
            B=$1 L=$2 O=$3 N=$4; shift 4
            for p in "$@"; do
                for i in $(seq 1 "$N"); do
                    case $p in
                    completion)    nice -n 19 "$B/bin/llama-completion" -m "$L" -p "The meaning of life is" -n 16 --seed 42 \
                                       -ngl 0 -t 2 -no-cnv ;;
                    batched-bench) nice -n 19 "$B/bin/llama-batched-bench" -m "$L" -c 256 -b 64 -ub 64 -npp 16 -ntg 8 -npl 1,2 \
                                       -ngl 0 -t 2 ;;
                    embedding)     nice -n 19 "$B/bin/llama-embedding" -m "$L" -p "The meaning of life is" --pooling mean \
                                       -ngl 0 -t 2 ;;
                    completion-nomodel) nice -n 19 "$B/bin/llama-completion" -m "$L.missing" -p "The meaning of life is" \
                                       -n 1 -ngl 0 -t 2 -no-cnv ;;
                    tokenize-nomodel)   nice -n 19 "$B/bin/llama-tokenize" -m "$L.missing" -p "The meaning of life is" ;;
                    *)             echo "unknown program $p"; false ;;
                    esac < /dev/null > "$O/$p-$i.log" 2>&1
                    echo "$p $i rc=$?"
                done
            done' _ "$RP/build/tree/build" "$L" "$RP/$t" "$REPEAT" $PROGS > "$RP/$t/rc.txt" 2>&1
        echo "$ROUND $t: $(grep -c 'rc=0$' "$RP/$t/rc.txt") of $(($(wc -w <<< "$PROGS") * REPEAT)) runs rc=0 $(date -Is)"
    done
    rm -rf "${RP:?}/build/tree/build"
    cc "$RP/build/ccache" -s > "$W/checks/ccache-$ROUND.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-$ROUND.txt"
    echo "=== $ROUND done $(date -Is)"
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
    [ -n "${BEFORE:-}" ] && pair before-base "$BEFORE" "$W/run1/base"
    pair base-head "$W/run1/base" "$W/run1/head"

    # ctest, test by test: the earlier run (BEFORE) and the two trees
    cols=() files=()
    if [ -n "${BEFORE:-}" ]; then ctest_tsv "$BEFORE/ctest.log" > "$C/ctest-before.tsv"; cols+=(before); files+=("$C/ctest-before.tsv"); fi
    for t in base head; do
        ctest_tsv "$W/run1/$t/ctest.log" > "$C/ctest-$t.tsv"; cols+=("$t"); files+=("$C/ctest-$t.tsv")
    done
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

    # the other checks' verdicts (the details hold counts and paths): the earlier run's and the two trees'
    { printf 'check'; [ -n "${BEFORE:-}" ] && printf '\tbefore'; printf '\tbase\thead\n'
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
for t in ("base", "head"):
    tabs.append(rows(w / "run1" / "summary.tsv", t))
for k in sorted(set().union(*tabs)):
    print(k + "\t" + "\t".join(t.get(k, "-") for t in tabs))
EOF
    } > "$C/checks.tsv"
    cat "$C/checks.tsv"
    grep -hP '^head\tident vs base\t' "$W/run1/summary.tsv"
    # the eval-callback sums and thread-safety's last line, as each tree's hand step printed them
    for t in base head; do
        f=$W/run1/$t/hand/eval-callback.log
        printf '%s\t%s\t%s\t%s\t%s\n' "$t" "$(grep -ac 'sum = ' "$f")" "$(wc -l < "$f")" \
            "$(grep -a 'sum = ' "$f" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)" \
            "$(grep -av '^ *$' "$W/run1/$t/hand/thread-safety.log" | tail -n 1 | sed -E 's/^[0-9.]+ [A-Z] //')"
    done > "$C/eval_sums.tsv"
    [ -n "${BEFORE:-}" ] && printf '%s\t%s\t%s\t%s\t%s\n' before "$(grep -ac 'sum = ' "$BEFORE/hand/eval-callback.log")" \
        "$(wc -l < "$BEFORE/hand/eval-callback.log")" \
        "$(grep -a 'sum = ' "$BEFORE/hand/eval-callback.log" | sed -E 's/.*sum = //' | md5sum | cut -c1-12)" \
        "$(grep -av '^ *$' "$BEFORE/hand/thread-safety.log" | tail -n 1 | sed -E 's/^[0-9.]+ [A-Z] //')" >> "$C/eval_sums.tsv"
    echo "eval-callback per tree (tree, sums, lines, md5 of the sums, thread-safety's last line):"
    cat "$C/eval_sums.tsv"

    # every repeat round: each run's log whole, cut short or different, against the most common longest log of both
    # trees, with every number masked (the timings change from run to run) and runs of spaces made one
    if [ -d "$W/repeat/base" ]; then
        python3 -I - "$W" "$C/repeat_runs.tsv" "$C/repeat.tsv" << 'EOF'
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
w, runs_out, sum_out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
NUM = re.compile(r"[-+]?\d+(\.\d+)?(e[-+]?\d+)?")
SPACE = re.compile(r"\s+")
def masked(p):
    out = []
    for x in p.read_text(errors="replace").splitlines():
        x = SPACE.sub(" ", NUM.sub("N", x)).strip()
        if x:
            out.append(x)
    return out
def rcs(rp, tree):
    out = {}
    for x in (rp / tree / "rc.txt").read_text().splitlines():
        m = re.match(r"(\S+) (\d+) rc=(\d+)$", x)
        if m:
            out[(m.group(1), int(m.group(2)))] = int(m.group(3))
    return out
rows, progs = [], []
for rp in sorted(d for d in w.glob("repeat*") if (d / "base").is_dir()):
    trees = [t for t in ("base", "head") if (rp / t).is_dir()]
    names = sorted({p.stem.rsplit("-", 1)[0] for t in trees for p in (rp / t).glob("*-*.log")})
    for prog in names:
        progs.append((rp.name, prog))
        logs = {(t, int(p.stem.rsplit("-", 1)[1])): masked(p) for t in trees for p in (rp / t).glob(f"{prog}-*.log")}
        full_len = max(len(v) for v in logs.values())
        full = Counter(tuple(v) for v in logs.values() if len(v) == full_len).most_common(1)[0][0]
        for t in trees:
            rc = rcs(rp, t)
            for (tt, i), v in sorted(logs.items()):
                if tt != t:
                    continue
                verdict = "whole" if tuple(v) == full else ("cut short" if tuple(v) == full[:len(v)] else "DIFFERS")
                md5 = hashlib.md5("\n".join(v).encode()).hexdigest()[:12]
                rows.append((rp.name, t, prog, i, rc.get((prog, i), "-"), len(v), full_len, verdict, md5,
                             (v[-1] if v else "")[:80]))
with runs_out.open("w") as f:
    f.write("round\ttree\tprogram\trun\trc\tlines\tlines when whole\tverdict\tmd5 masked\tlast line (masked)\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
with sum_out.open("w") as f:
    f.write("round\ttree\tprogram\truns\trc 0\twhole\tcut short\tdiffers\tlines when cut short (of the whole log's)\n")
    for rnd, prog in progs:
        for t in ("base", "head"):
            rr = [r for r in rows if r[0] == rnd and r[1] == t and r[2] == prog]
            if not rr:
                continue
            cut = [f"{r[5]} of {r[6]}" for r in rr if r[7] == "cut short"]
            f.write(f"{rnd}\t{t}\t{prog}\t{len(rr)}\t{sum(1 for r in rr if r[4] == 0)}\t"
                    f"{sum(1 for r in rr if r[7] == 'whole')}\t{len(cut)}\t{sum(1 for r in rr if r[7] == 'DIFFERS')}\t"
                    f"{', '.join(cut) or '-'}\n")
print(sum_out.read_text(), end="")
EOF
    fi
    echo "=== compare done $(date -Is)"
fi
finish 0
