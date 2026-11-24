#!/usr/bin/env bash
# C1020 (2026-11-24): the streams arm of cpu_tests.sh's pipe step (bench/pipespec/check_streams.sh: two conversations in
# one context, seeded sampling, 64 tokens, prompt chunks of 16) on the trees the v5 landing of 2026-11-24 checked,
# processor only, no GPU:
#   base  pascal-p106-v5 at BASE_SHA (8a9629942): it carries C932's fault (Flash-Next's fused conv window writing every
#         pipe slot's history to state group 0), which every other arm flags on qwen4exp;
#   land  the landing's head HEAD_SHA (af00bb13b), with C932's fix: every arm must pass.
# Each tree builds only the pipe step's three tools (llama-pipespec, llama-pipe-recurrent-check, test-llama-archs) with
# the landing's CMake options and compiler cache, then runs the pipe step with all five arms.
# Steps (STEPS, in this order):
#   src     a local clone of SEED (a llama.cpp clone holding both commits), both commits checked; the compiler cache
#           hard-linked from CPU_CCACHE, its counters zeroed
#   run     cpu_tests.sh (REPO, with bench/pipespec's scripts and prompts in REPO/pipespec) in $W/cpu, STEPS
#           'tree build pipe', trees base=BASE_SHA and land=HEAD_SHA, each with the three targets
#   compare the pipe verdicts of both trees and the landing's (BEFORE, the landing's cpu_tests.sh work folder) in
#           pipe.tsv; each check's seconds (pipe_times.tsv); the models' md5s against the landing's; every hash of the
#           four older arms against the landing's runs (arms.tsv); the streams arm's table, each conversation of each job
#           against its plain reference (streams.tsv)
# Usage: W=<work dir> SEED=<llama.cpp clone> REPO=<folder with cpu_tests.sh and pipespec/> BEFORE=<the landing's
#   cpu_tests.sh work folder with base/ and land/ and summary.tsv> [BASE_SHA=<sha>] [HEAD_SHA=<sha>] [CPU_CCACHE=<dir>]
#   [LOCK=<flock file>] [STEPS='src run compare'] [JOBS=8] [THREADS=6] bash run.sh
# Every container: labels pascal-quants=1 and c1020=1, --cpu-shares=256, --memory=16g, no network. Done when $W/DONE
# exists (it holds the exit code).
set -uo pipefail
W=${W:?W=<work dir>}
SEED=${SEED:?SEED=<llama.cpp clone>}
REPO=${REPO:?REPO=<folder with cpu_tests.sh and pipespec/>}
BEFORE=${BEFORE:?BEFORE=<the landing cpu_tests.sh work folder>}
BASE_SHA=${BASE_SHA:-8a9629942a4635ea0b2711b72e59556b96a63f86}
HEAD_SHA=${HEAD_SHA:-af00bb13b0fd59612bcab545964b61978f04b8d6}
STEPS=${STEPS:-src run compare}
JOBS=${JOBS:-8}
THREADS=${THREADS:-6}
CPU_IMAGE=${CPU_IMAGE:-pq-c565-cpu:24.04}
TGT=llama-pipespec,llama-pipe-recurrent-check,test-llama-archs
LBL="--label pascal-quants=1 --label c1020=1"
MEM="--memory=16g --memory-swap=16g"
mkdir -p "$W/logs" "$W/checks"
rm -f "$W/DONE"
step()   { [[ " $STEPS " == *" $1 "* ]]; }
finish() { docker ps -q --filter label=c1020=1 | xargs -r docker stop > /dev/null; echo "=== end $(date -Is) rc=$1"; echo "$1" > "$W/DONE"; exit "$1"; }
cc()     { docker run --rm $LBL --cpu-shares=256 $MEM --network none -v "$W:$W" -e CCACHE_DIR="$1" "$CPU_IMAGE" ccache "${@:2}"; }
export LC_ALL=C
echo "=== start $(date -Is): base $BASE_SHA, land $HEAD_SHA; before: $BEFORE; steps: $STEPS"

if step src; then
    S=$W/src
    [ -d "$S/.git" ] || git clone -q --no-checkout "$SEED" "$S" || finish 1
    for c in "$BASE_SHA" "$HEAD_SHA"; do
        [ "$(git -C "$S" cat-file -t "$c" 2> /dev/null)" = commit ] || { echo "no commit $c in $S: stop"; finish 1; }
    done
    echo "base $(git -C "$S" log -1 --format='%h %s' "$BASE_SHA")"
    echo "land $(git -C "$S" log -1 --format='%h %s' "$HEAD_SHA")"
    mkdir -p "$W/cpu"
    [ -d "$W/cpu/ccache" ] || { [ -n "${CPU_CCACHE:-}" ] && cp -al "$CPU_CCACHE" "$W/cpu/ccache"; }
    [ -d "$W/cpu/ccache" ] && cc "$W/cpu/ccache" -z
    echo "=== sources ready $(date -Is)"
fi

if step run; then
    WORK="$W/cpu" SRC="$W/src" TREES="base=$BASE_SHA:$TGT land=$HEAD_SHA:$TGT" STEPS="tree build pipe" \
        JOBS="$JOBS" THREADS="$THREADS" LOCK="${LOCK:-}" CMAKE_ARGS="-DGGML_RPC=ON" \
        DOCKER_ARGS="$LBL $MEM" IMAGE="$CPU_IMAGE" bash "$REPO/cpu_tests.sh" > "$W/logs/cpu.log" 2>&1
    echo "run: cpu tests rc=$?"
    cat "$W/cpu/summary.tsv"
    cc "$W/cpu/ccache" -s > "$W/checks/ccache.txt" 2>&1
    grep -E 'Hits|Misses|hit rate' "$W/checks/ccache.txt"
    echo "=== run done $(date -Is)"
fi

if step compare; then
    C=$W/checks
    for t in base land; do
        cp "$W/cpu/$t/pipe/models.md5" "$C/models-$t.md5" 2> /dev/null
        echo "models $t against the landing's: $(cmp -s "$W/cpu/$t/pipe/models.md5" "$BEFORE/$t/pipe/models.md5" && echo same || echo DIFFERENT)"
    done
    python3 -I - "$W/cpu" "$BEFORE" "$C" << 'EOF'
import re, sys
from pathlib import Path
w, before, c = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
trees = ["base", "land"]

def verdicts(summary):
    r = {}
    for l in Path(summary).read_text().splitlines()[1:]:
        f = l.split("\t")
        if len(f) >= 4 and f[1].startswith("pipe "):
            p = f[1].split()
            if len(p) == 3:
                r[(f[0], p[1], p[2])] = (f[2], f[3])
    return r

now, old = verdicts(w / "summary.tsv"), verdicts(before / "summary.tsv")
keys = sorted(set(now) | set(old), key=lambda k: (k[1], k[2], k[0]))
with open(c / "pipe.tsv", "w") as f:
    f.write("tree\tmodel\tarm\tverdict\tlanding's verdict\tdetail\n")
    for k in keys:
        f.write("\t".join([k[0], k[1], k[2], now.get(k, ("-", ""))[0], old.get(k, ("-", ""))[0], now.get(k, ("", "-"))[1]]) + "\n")

with open(c / "pipe_times.tsv", "w") as f:
    f.write("tree\tmodel\tarm\tseconds\n")
    for t in trees:
        for l in (w / t / "pipe" / "times.tsv").read_text().splitlines()[1:]:
            f.write(t + "\t" + l + "\n")

def kv(line):
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)

# the four older arms: every job's hashes (the race's too) against the landing's run of the same tree and model
HK = ["n_gen", "tokens", "logits", "state", "probe", "winner", "race_tokens", "race_hash"]
with open(c / "arms.tsv", "w") as f:
    f.write("tree\tmodel\tarm\tjobs\tsame as the landing's\n")
    for t in trees:
        for res in sorted((w / t / "pipe").glob("*-*/results.txt")):
            key, arm = res.parent.name.rsplit("-", 1)
            if arm == "streams":
                continue
            o = before / t / "pipe" / res.parent.name / "results.txt"
            a = {kv(l)["name"]: [kv(l).get(k) for k in HK] for l in res.read_text().splitlines() if l.startswith("PIPESPEC ")}
            b = {kv(l)["name"]: [kv(l).get(k) for k in HK] for l in o.read_text().splitlines() if l.startswith("PIPESPEC ")} if o.exists() else {}
            same = sum(1 for n in a if b.get(n) == a[n])
            f.write(f"{t}\t{key}\t{arm}\t{len(a)}\t{same}\n")
        for log in sorted((w / t / "pipe").glob("*-recurrent.log")):
            key = log.name[: -len("-recurrent.log")]
            o = before / t / "pipe" / log.name
            get = lambda p: ([l for l in p.read_text(errors="replace").splitlines() if "PIPE-RECURRENT-CHECK" in l] or [""])[-1] if p.exists() else ""
            a, b = get(log), get(o)
            f.write(f"{t}\t{key}\trecurrent\t1\t{1 if a and a == b else 0}\n")

# the streams arm: each conversation of each job against the plain run of its prompt and seed (ref-c<c>)
COLS = ["tokens", "logits", "state", "probe"]
with open(c / "streams.tsv", "w") as f:
    f.write("tree\tmodel\trun\tn_gen\tchunks\tguesses\thits\tmisses\t" + "\t".join(COLS) + "\tverdict\n")
    for t in trees:
        for res in sorted((w / t / "pipe").glob("*-streams/results.txt")):
            key = res.parent.name[: -len("-streams")]
            rows = [kv(l) for l in res.read_text().splitlines() if l.startswith("PIPESPEC ")]
            ref = {r["name"]: r for r in rows if r.get("name", "").startswith("ref-c")}
            for r in rows:
                m = re.fullmatch(r"(.+)-c(\d+)", r.get("name", ""))
                if not m or r["name"].startswith("ref-"):
                    continue
                p = ref.get("ref-c" + m.group(2), {})
                cols = ["same" if r.get(k) == p.get(k) else "DIFF" for k in COLS]
                f.write("\t".join([t, key, r["name"]] + [r.get(k, "-") for k in ["n_gen", "chunks", "guesses", "hits", "misses"]]
                                  + cols + ["DIFF" if "DIFF" in cols else "same"]) + "\n")
EOF
    cat "$C/pipe.tsv" | cut -f1-5
    echo "--- arms against the landing's (tree, model, arm, jobs, same)"
    cat "$C/arms.tsv"
    echo "--- streams, conversations differing per tree and model"
    awk -F'\t' 'NR > 1 { n[$1 " " $2]++; if ($NF == "DIFF") d[$1 " " $2]++ } END { for (k in n) printf "%s: %d of %d\n", k, d[k], n[k] }' "$C/streams.tsv" | sort
    echo "--- times"
    awk -F'\t' 'NR > 1 && ($3 == "streams" || $3 == "total")' "$C/pipe_times.tsv"
    echo "=== compare done $(date -Is)"
fi
finish 0
