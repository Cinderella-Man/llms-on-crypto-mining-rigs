#!/usr/bin/env bash
# dry_run.sh: C798's counting tools end to end on the CPU (registry Q1395), no GPU. Steps:
#   model     a random-weight qwen4exp MoE model with one MTP block and its draft file (test-llama-archs -o --mtp-draft),
#             with the tokenizer and chat template of VOCAB (LLAMA_ARCHS_VOCAB) and a 65,536-token context
#   selftest  bench/draft_count/draft_table.py --selftest and bench/tree_replay/tree_replay.py --selftest
#   chain     bench/draft_count/run_draft_count.sh with GPUS=cpu on run/cpu_preset.env, the recorded requests of LOG,
#             loads "d3 d5 d7 d7x d3": d7x is d7 with the dump off, the second d3 the first again
#   tables    draft_table.py on the chain's folder, and the verdict: every check of the tables passes, the d7 loads
#             with and without the dump wrote the same replies with the same counts, the two d3 loads the same
#   synth     a second chain, "d7s d4s", with the server's synthetic acceptance (--spec-synth-rates SYNTH7 and SYNTH4,
#             the probability that the first i drafted tokens are all accepted) and 1,024-token replies: every check
#             passes, each P_i is its set rate within sampling error (|z| < 4), and d7s cut to 4 tokens gives d4s's
#             own accepted tokens + 1 per verify step within 0.1. A random model accepts almost nothing, so this is the
#             run that reaches the later positions
#   tree      tree_replay.py --check and the analysis (factorized, the shapes) on the d7 load's dump
#   WORK=<folder> REPO=<checkout of this repository> ARCHS=<test-llama-archs> BIN=<llama-server, a CPU build>
#     VOCAB=<a gguf with Flash-Next's tokenizer> LOG=<recorded proxy.jsonl, its .bodies.gz next to it>
#     [IMAGE=<image with the binaries' libraries>] [MOUNTS="<folders the containers read>"] [STEPS="model selftest ..."]
#     bash dry_run.sh
# Every container runs without a GPU, --cpu-shares 256, label pascal-quants=1; the Python steps under nice -n 19.
# Output: WORK/models, WORK/chain (the loads), WORK/tables, WORK/tree, WORK/verdict.txt, WORK/summary.tsv.
set -uo pipefail
: "${WORK:?}" "${REPO:?}" "${ARCHS:?}" "${BIN:?}" "${VOCAB:?}" "${LOG:?}"
IMAGE=${IMAGE:-pq-c735-srv:24.04}
STEPS=${STEPS:-model selftest chain tables synth tree}
SYNTH7=${SYNTH7:-0.9,0.8,0.7,0.6,0.5,0.4,0.3}
SYNTH4=${SYNTH4:-0.9,0.8,0.7,0.6}
RUN=$(dirname "$(realpath "$0")")
mkdir -p "$WORK"
mounts="-v $WORK:$WORK"
for m in ${MOUNTS:-}; do mounts="$mounts -v $m:$m"; done
SUM=$WORK/summary.tsv
[ -e "$SUM" ] || printf 'step\tverdict\tdetail\n' > "$SUM"
note() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$SUM"; echo "$(date +%H:%M:%S) $1 $2 $3"; }
# shellcheck disable=SC2086
box() { docker run --rm --init --label pascal-quants=1 --cpu-shares 256 $mounts "$IMAGE" "$@"; }
M=$WORK/models

for step in $STEPS; do
  case $step in
  model)
    rm -rf "$M" && mkdir -p "$M"
    box env LLAMA_ARCHS_VOCAB="$VOCAB" LLAMA_ARCHS_N_CTX=65536 nice -n 19 "$ARCHS" -a '^qwen4exp$' -s 1 -o "$M" --mtp-draft \
      > "$M/save.log" 2>&1
    rc=$?
    if [ $rc = 0 ] && [ -s "$M/qwen4exp-moe-mtp.gguf" ] && [ -s "$M/qwen4exp-moe-mtp-draft.gguf" ]; then
      note model OK "$(cd "$M" && md5sum qwen4exp-moe-mtp.gguf qwen4exp-moe-mtp-draft.gguf | tr '\n' ' ')"
    else
      note model FAIL "rc=$rc: $(tail -n 3 "$M/save.log" | tr '\n' ' ')"
    fi
    ;;
  selftest)
    for t in "bench/draft_count/draft_table.py" "bench/tree_replay/tree_replay.py"; do
      n=$(basename "$t" .py)
      nice -n 19 python3 -I "$REPO/$t" --selftest > "$WORK/selftest-$n.txt" 2>&1
      rc=$?
      note "selftest-$n" "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(tail -n 1 "$WORK/selftest-$n.txt")"
    done
    ;;
  chain)
    rm -rf "$WORK/chain"
    WORK=$WORK/chain REPO=$REPO BIN=$BIN MODEL=$M/qwen4exp-moe-mtp.gguf MODEL_DRAFT=$M/qwen4exp-moe-mtp-draft.gguf \
      LOG=$LOG GPUS=cpu IMAGE=$IMAGE MOUNTS="${MOUNTS:-}" PRESET=$RUN/cpu_preset.env LOADS="d3 d5 d7 d7x d3" \
      ENV_d7x=LLAMA_SPEC_DUMP= TRACES_N=${TRACES_N:-3} REQUESTS=${REQUESTS:-0,1} REPLY_LIMIT=${REPLY_LIMIT:-128} \
      NAME=${NAME:-c798-dry} PORT=${PORT:-18798} DOCKER_ARGS="--label c798=1" \
      nice -n 19 bash "$REPO/bench/draft_count/run_draft_count.sh" > "$WORK/chain.console" 2>&1
    rc=$?
    note chain "$([ $rc = 0 ] && [ -e "$WORK/chain/done" ] && echo OK || echo FAIL)" "rc=$rc; $(tail -n 1 "$WORK/chain.console")"
    ;;
  tables)
    nice -n 19 python3 -I "$REPO/bench/draft_count/draft_table.py" "$WORK/chain" --out "$WORK/tables" --ref d3 \
      > "$WORK/tables.console" 2>&1
    rc=$?
    note tables "$([ $rc = 0 ] && echo OK || echo FAIL)" "rc=$rc; $(head -n 1 "$WORK/tables.console")"
    nice -n 19 python3 -I - "$WORK/tables" "$WORK/chain" > "$WORK/verdict.txt" 2>&1 << 'EOF'
import csv, sys
from pathlib import Path
t = Path(sys.argv[1])
rd = lambda n: list(csv.DictReader(open(t / f"{n}.tsv"), delimiter="\t"))
checks, loads, same = rd("check"), {r["label"]: r for r in rd("loads")}, rd("same_reply")
bad = [c for c in checks if c["verdict"] != "OK"]
out = [f"checks: {len(checks) - len(bad)} of {len(checks)} OK" + "".join(f"; FAIL {c['label']} {c['check']}: {c['detail']}" for c in bad)]
ok = not bad
def pair(a, b):
    r = [s for s in same if {s["load_a"], s["load_b"]} == {a, b}]
    return (int(r[0]["same_reply"]), int(r[0]["requests"])) if r else (0, 0)
for a, b, what in (("d7-1", "d7x-1", "d7 with and without the dump"), ("d3-1", "d3-2", "the two d3 loads")):
    s, n = pair(a, b)
    cnt = all(loads[a][k] == loads[b][k] for k in ("generated", "drafted", "accepted", "verify_steps", "P1"))
    out.append(f"{what}: {s} of {n} replies the same, counts {'the same' if cnt else 'different'}")
    ok &= s == n > 0 and cnt
dump = Path(sys.argv[2]) / "dump-d7-1.jsonl"
out.append(f"d7's dump: {loads['d7-1'].get('dump_steps')} steps, type {loads['d7-1'].get('dump_type')}, verify "
           f"{loads['d7-1'].get('dump_verify')}, {loads['d7-1'].get('dump_cand_rows')} candidate rows")
ok &= loads["d7-1"].get("dump_type") == "draft-mtp" and loads["d7-1"].get("dump_verify") == "reject"
print("VERDICT " + ("PASS" if ok else "FAIL"))
print("\n".join(out))
sys.exit(0 if ok else 1)
EOF
    rc=$?
    note verdict "$([ $rc = 0 ] && echo PASS || echo FAIL)" "$(tail -n +2 "$WORK/verdict.txt" | tr '\n' ';')"
    ;;
  synth)
    # the server's synthetic acceptance (--spec-synth-rates: the probability that the first i drafted tokens are all
    # accepted, per position) on a 7- and a 4-token draft, so the counts by position have a known answer; the dump
    # skips synthetic steps, so it is off
    rm -rf "$WORK/chain-synth"
    WORK=$WORK/chain-synth REPO=$REPO BIN=$BIN MODEL=$M/qwen4exp-moe-mtp.gguf MODEL_DRAFT=$M/qwen4exp-moe-mtp-draft.gguf \
      LOG=$LOG GPUS=cpu IMAGE=$IMAGE MOUNTS="${MOUNTS:-}" PRESET=$RUN/cpu_preset.env LOADS="d7s d4s" DUMP=0 \
      ARGS_d7s="--spec-synth-rates $SYNTH7" ARGS_d4s="--spec-synth-rates $SYNTH4" \
      TRACES_N=${TRACES_N:-3} REQUESTS=${REQUESTS:-0,1} REPLY_LIMIT=${SYNTH_REPLY_LIMIT:-1024} \
      NAME=${NAME:-c798-dry} PORT=${PORT:-18798} DOCKER_ARGS="--label c798=1" \
      nice -n 19 bash "$REPO/bench/draft_count/run_draft_count.sh" > "$WORK/chain-synth.console" 2>&1
    rc=$?
    note synth-chain "$([ $rc = 0 ] && [ -e "$WORK/chain-synth/done" ] && echo OK || echo FAIL)" "rc=$rc; $(tail -n 1 "$WORK/chain-synth.console")"
    nice -n 19 python3 -I "$REPO/bench/draft_count/draft_table.py" "$WORK/chain-synth" --out "$WORK/tables-synth" \
      --ref d7s > "$WORK/tables-synth.console" 2>&1
    nice -n 19 python3 -I - "$WORK/tables-synth" "$SYNTH7" "$SYNTH4" > "$WORK/verdict-synth.txt" 2>&1 << 'EOF'
import csv, math, sys
from pathlib import Path
t = Path(sys.argv[1])
rd = lambda n: list(csv.DictReader(open(t / f"{n}.tsv"), delimiter="\t"))
checks, pos, cut, loads = rd("check"), rd("positions"), rd("cut"), {r["label"]: r for r in rd("loads")}
bad = [c for c in checks if c["verdict"] != "OK"]
ok = not bad
out = [f"checks: {len(checks) - len(bad)} of {len(checks)} OK" + "".join(f"; FAIL {c['label']} {c['check']}" for c in bad)]
for label, rates in (("d7s-1", sys.argv[2]), ("d4s-1", sys.argv[3])):
    want = [float(x) for x in rates.split(",")]
    zs = []
    for p in (r for r in pos if r["label"] == label):
        i, n, got = int(p["position"]), int(p["drafted"]), float(p["P"])
        w = want[i - 1]
        z = abs(got - w) / math.sqrt(w * (1 - w) / n)
        zs.append(z)
        out.append(f"{label} P{i} {got:.4f} on {n} steps, set {w} (|z| {z:.2f})")
    ok &= len(zs) == len(want) and max(zs) < 4.0
c7 = next(float(r["tokens_a_verify_step"]) for r in cut if r["label"] == "d7s-1" and r["cut_to"] == "4")
c4 = float(loads["d4s-1"]["accepted_per_verify_step"]) + 1
out.append(f"d7s cut to 4: {c7:.4f} tokens a verify step; d4s's own: {c4:.4f}; 1 + the set rates: "
           f"{1 + sum(float(x) for x in sys.argv[3].split(',')):.4f}")
ok &= abs(c7 - c4) < 0.1
print("VERDICT " + ("PASS" if ok else "FAIL"))
print("\n".join(out))
sys.exit(0 if ok else 1)
EOF
    rc=$?
    note verdict-synth "$([ $rc = 0 ] && echo PASS || echo FAIL)" "$(head -n 1 "$WORK/verdict-synth.txt"); $(tail -n 1 "$WORK/verdict-synth.txt")"
    ;;
  tree)
    D=$WORK/chain/dump-d7-1.jsonl
    nice -n 19 python3 -I "$REPO/bench/tree_replay/tree_replay.py" --check "$D" > "$WORK/tree-check.txt" 2>&1
    rc=$?
    note tree-check "$([ $rc = 0 ] && echo OK || echo FAIL)" "$(head -n 1 "$WORK/tree-check.txt")"
    # shellcheck disable=SC2046
    nice -n 19 python3 -I "$REPO/bench/tree_replay/tree_replay.py" "$D" --out "$WORK/tree" --mode factorized \
      $(nice -n 19 python3 -I "$REPO/bench/tree_replay/gguf_markers.py" "$VOCAB") > "$WORK/tree.console" 2>&1
    rc=$?
    note tree "$([ $rc = 0 ] && [ -s "$WORK/tree/shapes.csv" ] && echo OK || echo FAIL)" \
      "rc=$rc; $(grep -c . "$WORK/tree/shapes.csv" 2> /dev/null) lines in shapes.csv"
    ;;
  esac
done
