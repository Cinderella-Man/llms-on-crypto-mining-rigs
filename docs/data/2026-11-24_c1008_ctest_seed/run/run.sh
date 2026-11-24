#!/usr/bin/env bash
# C1008 (2026-11-24): ctest's test-generate-models with a fixed seed (-s 1). Two ctest runs of a base tree (the line
# without a seed, upstream's) and a head tree (with -s 1), one tree at a time (cpu_tests.sh, C565: steps tree build
# ctest, the build kept); after each tree's ctest, the checksums of every random model ctest's generating tests wrote,
# and the test-llama-context sweep run once more on ctest's own models, its whole log kept. Processor only, no GPU.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding BASE), the fork's BRANCH fetched and checked against COMMIT;
#           the compiler cache hard-linked from CPU_CCACHE (another work folder's: it hits through ccache's base folder),
#           its counters zeroed
#   run1    per tree (base=BASE, then head=COMMIT): cpu_tests.sh (REPO) in $W/run1, steps tree build ctest, the build
#           kept (a CPU build of every target under LOCK; ctest without network, cpu_tests.sh's default exclusions);
#           then, in containers without network: md5sum of every .gguf under the build's tests/ folder (test-models/,
#           which test-generate-models writes, and the folders of the two fixtures that already pass a seed,
#           test-models-fp8/ and test-speculative-models/) into $W/run1/<tree>/models.md5; ctest's LastTest.log copied
#           to $W/run1/<tree>/ (its test-generate-models section prints the seed the test used); the sweep
#           test-llama-context --models tests/test-models/ with GGML_SCHED_DEBUG_REALLOC=1 and the build's tests/ folder
#           as its working folder, as ctest runs it, its stdout and stderr into $W/run1/<tree>/context.log from inside
#           the container; then the build removed; then ccache's counters
#   run2    the same in $W/run2, its compiler cache hard-linked from run1's
#   compare models.tsv (each model's md5 in the four trees, and in HAND_MD5, an earlier hand step's models.md5, for the
#           files it has), seeds.tsv (each tree's generating tests: their arguments and the seed each printed), ctest.tsv
#           (test by test: BEFORE's run and the four trees), context.tsv (each tree's sweep log: lines, verdict counts,
#           md5 of its lines with time stamps and colours dropped, in order and sorted) and context_pairs.tsv (the
#           pairs), checks.tsv (cpu_tests.sh's verdicts)
# Usage: W=<work dir> SEED=<llama.cpp clone with BASE> REPO=<folder with cpu_tests.sh>
#   [BASE=<sha>] [COMMIT=<sha>] [BRANCH=<fork branch at COMMIT>] [BEFORE=<an earlier run's tree dir of BASE>]
#   [HAND_MD5=<an earlier hand step's models.md5>] [CPU_CCACHE=<dir>] [LOCK=<flock file>]
#   [STEPS='src run1 run2 compare'] [JOBS=8] bash run.sh
# Every container: labels pascal-quants=1 and c1008=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone with BASE>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh>}
BASE=${BASE:-af00bb13b0fd59612bcab545964b61978f04b8d6}
COMMIT=${COMMIT:-55e14cb79cecaaafff379ce9a762ef55e72a5ec4}
FORK_URL=${FORK_URL:-https://github.com/Cinderella-Man/miner-llama}
BRANCH=${BRANCH:-pascal-p106-v5-ctest-seed}
STEPS=${STEPS:-src run1 run2 compare}
JOBS=${JOBS:-8}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
CONTEXT_TIMEOUT=${CONTEXT_TIMEOUT:-900}
LBL="--label pascal-quants=1 --label c1008=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c1008=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
# shellcheck disable=SC2086
box()    { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" "$CPU_IMAGE" "$@"; }
cc()     { box env CCACHE_DIR="$1" ccache "${@:2}"; }
export LC_ALL=C # sort and join in one collation
# ctest results of a run's log, one "name<TAB>result" line per test, sorted by name
ctest_tsv() {
    grep -aE '^ *[0-9]+/[0-9]+ Test +#[0-9]+: ' "$1" \
        | sed -E 's/^ *[0-9]+\/[0-9]+ Test +#[0-9]+: ([^ ]+) [. ]*(\*{0,3}[A-Za-z][A-Za-z *]*[A-Za-z]).*$/\1\t\2/' | sort
}
cpu_tests() { # <work> <trees> <steps> [KEEP_BUILD]
    WORK="$1" SRC="$W/src" TREES="$2" STEPS="$3" KEEP_BUILD="${4:-}" PIPE_ARCHS='' \
        JOBS="$JOBS" CTEST_JOBS=6 THREADS=6 LOCK="${LOCK:-}" DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" \
        bash "$REPO/cpu_tests.sh"
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
    B=$W/$r/tree/build
    for t in base head; do
        c=$BASE; [ $t = head ] && c=$COMMIT
        cpu_tests "$W/$r" "$t=$c" "tree build ctest" 1 > "$W/logs/$r-$t.log" 2>&1
        echo "$r $t: cpu tests rc=$?; $(grep -P "^$t\tctest\t" "$W/$r/summary.tsv" | tail -n 1 | cut -f3,4 | cut -c1-160)"
        R=$W/$r/$t
        [ -d "$B/tests" ] || { echo "$r $t: no build folder $B/tests"; continue; }
        # every random model ctest wrote, path relative to the build's tests/ folder
        # shellcheck disable=SC2016
        box bash -c 'cd "$1" && find . -name "*.gguf" -type f | sort | xargs -r md5sum' _ "$B/tests" > "$R/models.md5" 2>&1
        echo "$r $t: $(wc -l < "$R/models.md5") model files, $(grep -c ' ./test-models/' "$R/models.md5") in test-models/"
        cp "$B/Testing/Temporary/LastTest.log" "$R/LastTest.log" 2> /dev/null || echo "$r $t: no LastTest.log"
        # the sweep as ctest runs it (tests/CMakeLists.txt: ARGS --models <test-models>, GGML_SCHED_DEBUG_REALLOC=1)
        # shellcheck disable=SC2016
        box bash -c 'cd "$1/tests" && GGML_SCHED_DEBUG_REALLOC=1 nice -n 19 timeout "$3" "$1/bin/test-llama-context" \
            --models "$1/tests/test-models/" > "$2" 2>&1; echo "context rc=$?"' _ "$B" "$R/context.log" "$CONTEXT_TIMEOUT" \
            > "$R/context.rc" 2>&1
        echo "$r $t: sweep $(cat "$R/context.rc"), $(wc -l < "$R/context.log") lines $(date -Is)"
        rm -rf "${B:?}"
    done
    cc "$W/$r/ccache" -s > "$W/checks/ccache-$r.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache-$r.txt"
    echo "=== $r done $(date -Is)"
done

if step compare; then
    C=$W/checks
    TREES4="run1/base run2/base run1/head run2/head"
    # the models: one row per file, its md5 (12 characters) in each tree, and in HAND_MD5 for the files it names
    python3 -I - "$W" "$C/models.tsv" "${HAND_MD5:-}" $TREES4 << 'EOF'
import sys
from pathlib import Path
w, out, hand, trees = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4:]
def md5s(p):
    d = {}
    if Path(p).is_file():
        for l in Path(p).read_text().splitlines():
            f = l.split(None, 1)
            if len(f) == 2:
                d[f[1].strip().lstrip("*")] = f[0][:12]
    return d
tabs = [md5s(w / t / "models.md5") for t in trees]
handt = {}
if hand:
    for k, v in md5s(hand).items():
        handt["./test-models/" + Path(k).name] = v
names = sorted(set().union(*tabs))
cols = [t.replace("/", "-") for t in trees]
with out.open("w") as f:
    f.write("model\t" + "\t".join(cols) + ("\thand step" if hand else "") + "\tbase twice\thead twice\n")
    for n in names:
        v = [t.get(n, "missing") for t in tabs]
        row = [n] + v + ([handt.get(n, "-")] if hand else [])
        row += ["same" if v[0] == v[1] else "differs", "same" if v[2] == v[3] else "differs"]
        f.write("\t".join(row) + "\n")
for folder in ("./test-models/", "./test-models-fp8/", "./test-speculative-models/"):
    rows = [n for n in names if n.startswith(folder)]
    if not rows:
        continue
    b = sum(1 for n in rows if tabs[0].get(n) == tabs[1].get(n))
    h = sum(1 for n in rows if tabs[2].get(n) == tabs[3].get(n))
    bh = sum(1 for n in rows if tabs[2].get(n) == tabs[0].get(n))
    print(f"{folder}: {len(rows)} files; base the same in both runs {b} of {len(rows)}; head the same in both runs "
          f"{h} of {len(rows)}; head run1 as base run1 {bh} of {len(rows)}")
for n, v in sorted(handt.items()):
    if n in tabs[2]:
        print(f"hand step's {n}: {v}; head run1 {tabs[2][n]}, run2 {tabs[3][n]}: "
              f"{'the same' if v == tabs[2][n] == tabs[3][n] else 'differs'}")
EOF

    # the seed each generating test printed, from ctest's LastTest.log (its arguments with the build folder cut off)
    for t in $TREES4; do
        awk -v tree="${t/\//-}" '
            /^[0-9]+\/[0-9]+ Test: / { name = $3 }
            /^Command: / && name ~ /^test-generate/ { cmd[name] = $0 }
            /using seed [0-9]+/ && name ~ /^test-generate/ { match($0, /using seed [0-9]+/); seed[name] = substr($0, RSTART + 11, RLENGTH - 11) }
            END { for (n in seed) { c = cmd[n]; sub(/^Command: "[^"]*\/bin\//, "", c); gsub(/"[^"]*\/tests\//, "\"<tests>/", c); gsub(/"/, "", c); print tree "\t" n "\t" seed[n] "\t" c } }' \
            "$W/$t/LastTest.log"
    done | sort -t $'\t' -k2,2 -k1,1 > "$C/seeds.tsv"
    sed -i '1i tree\ttest\tseed\tcommand' "$C/seeds.tsv"
    echo "seeds:"; cat "$C/seeds.tsv"

    # ctest, test by test: the earlier run (BEFORE) and the four trees
    cols=() files=()
    if [ -n "${BEFORE:-}" ]; then ctest_tsv "$BEFORE/ctest.log" > "$C/ctest-before.tsv"; cols+=(before); files+=("$C/ctest-before.tsv"); fi
    for t in $TREES4; do
        ctest_tsv "$W/$t/ctest.log" > "$C/ctest-${t/\//-}.tsv"; cols+=("${t/\//-}"); files+=("$C/ctest-${t/\//-}.tsv")
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

    # the sweep's logs: per tree, and each pair (the same tree twice, base against head in each run)
    python3 -I - "$W" "$C/context.tsv" "$C/context_pairs.tsv" $TREES4 << 'EOF'
import hashlib
import re
import sys
from collections import Counter
from pathlib import Path
w, out, pairs_out, trees = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4:]
ANSI = re.compile(r"\x1b\[[0-9;]*m")
STAMP = re.compile(r"\d+\.\d{2}\.\d{3}\.\d{3} [IWED]? ?")
def lines(t):
    p = w / t / "context.log"
    raw = p.read_text(errors="replace").splitlines() if p.is_file() else []
    return [STAMP.sub("", ANSI.sub("", x)).rstrip() for x in raw]
def md5(xs):
    return hashlib.md5("\n".join(xs).encode()).hexdigest()[:12]
logs = {t: lines(t) for t in trees}
# the sweep prints one table row per model ("<file>.gguf" then one verdict per case) and a summary line per case
with out.open("w") as f:
    f.write("tree\trc\tlines\tmodel rows\tcases per row\tOK\tSKIP\tFAIL\tmodels line\tmd5 in order\tmd5 sorted\n")
    for t in trees:
        v = logs[t]
        rc = (w / t / "context.rc").read_text().strip() if (w / t / "context.rc").is_file() else "-"
        rows = [x.split()[1:] for x in v if re.match(r"^\S+\.gguf\s", x)]
        cells = Counter(c for r in rows for c in r)
        fails = sum(n for k, n in cells.items() if k.startswith("FAIL"))
        per_row = ",".join(sorted({str(len(r)) for r in rows})) or "-"
        models = next((x.split("models:", 1)[1].strip() for x in reversed(v) if "main: models:" in x), "-")
        f.write(f"{t.replace('/', '-')}\t{rc}\t{len(v)}\t{len(rows)}\t{per_row}\t{cells['OK']}\t{cells['SKIP']}\t"
                f"{fails}\t{models}\t{md5(v)}\t{md5(sorted(v))}\n")
def pair(name, a, b):
    x, y = logs[a], logs[b]
    ca, cb = Counter(x), Counter(y)
    only = sum((ca - cb).values()) + sum((cb - ca).values())
    return (name, a.replace("/", "-"), b.replace("/", "-"), "yes" if x == y else "no",
            "yes" if sorted(x) == sorted(y) else "no", str(only))
rows = [pair("base twice", "run1/base", "run2/base"), pair("head twice", "run1/head", "run2/head"),
        pair("base-head run1", "run1/base", "run1/head"), pair("base-head run2", "run2/base", "run2/head")]
with pairs_out.open("w") as f:
    f.write("pair\ta\tb\tsame in order\tsame sorted\tlines in one only\n")
    for r in rows:
        f.write("\t".join(r) + "\n")
print(out.read_text(), end="")
print(pairs_out.read_text(), end="")
EOF

    # cpu_tests.sh's verdicts of the four trees
    { printf 'check\trun1-base\trun1-head\trun2-base\trun2-head\n'
      for k in tree build ctest; do
          printf '%s' "$k"
          for t in run1/base run1/head run2/base run2/head; do
              printf '\t%s' "$(grep -P "^${t#*/}\t$k\t" "$W/${t%/*}/summary.tsv" | tail -n 1 | cut -f3)"
          done
          printf '\n'
      done; } > "$C/checks.tsv"
    cat "$C/checks.tsv"
    echo "=== compare done $(date -Is)"
fi
finish 0
