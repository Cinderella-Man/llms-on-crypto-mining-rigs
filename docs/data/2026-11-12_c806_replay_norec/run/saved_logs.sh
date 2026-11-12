#!/usr/bin/env bash
# C806 (2026-11-12): scripts/replay_checks.sh with C806 (CHECKS) and before it (OLD_CHECKS) on the saved output of three
# replays on a P106, with no GPU: R112's v4day set on the -kern build (71 runs; its 14 dmt runs met 30 cases the set
# does not hold), the same set's 14 dmt runs on the -day build that recorded it, and R73's qsa-prompt set on -kern (11
# runs, 50 unrecorded cases each). A stub test-backend-ops stands in: it prints the saved log of the run whose per-case
# table replay_checks.sh names (GGML_TEST_REPLAY_LOG=<OUT>/<run>.tsv) without the "exit N" line the script appended,
# copies the saved table there and exits N. Those builds print an unrecorded case as "<case>: not recorded [<meta>]".
#   stub   OLD_CHECKS on the stub gives each saved results.tsv again, byte for byte (the stub replays faithfully)
#   new    CHECKS: in results.tsv every column the saved one's but the verdict of the runs with unrecorded cases and the
#          new last column (their count); R112's 14 dmt runs on -kern ok with 30 unrecorded, the same 30 in each, each
#          an IQ4_XS or Q4_K product at k = 6144 and n = 1-5, none of them among the cases the replay ran (R112's
#          cases.tsv.gz); on -day 0; R73's 11 runs ok with 50 unrecorded, the ids in R73's unrecorded.tsv
#   names  the console names the 30 once (NOREC lines) and the 13 dmt runs after the first as "the same 30 unrecorded
#          cases as dmt-on"
#   fake   made-up failures on the saved logs: one OK case of R112's dmt-on and of R73's qp-stock turned into a
#          failing one (an "ERR = 1 > limit ... FAIL" line, the pass count one lower, the table's verdict FAIL): WRONG
#          with the unrecorded counted; R73's fault run qp-fault2 as it ran: ok; and with its 59 failing cases turned OK
#          (a fault that is not caught, the run still failing through its unrecorded cases): WRONG, where OLD_CHECKS
#          says ok
# Runs inside a container of a Ubuntu 24.04 image with python3 (the rigs' userland: GNU coreutils).
# Env: CHECKS, OLD_CHECKS, W (work folder, emptied first), R112_OUT and R112_DAY_OUT (R112's out/ and out-day/: the
#   per-run .log and .tsv, results.tsv), R112_RUNS (the v4day set's runs.tsv: docs/data/2026-11-11_v4_day_merge/
#   set_runs.tsv), R112_CASES (docs/data/2026-11-12_r112_v4day_replay/cases.tsv.gz), R73_OUT, R73_RUNS
#   (docs/data/2026-11-11_c705_sparse_prompt_rule/runs.tsv), R73_UNREC (docs/data/2026-11-12_r73_qsa_prompt_replay/
#   unrecorded.tsv)
# Output: W/saved_logs.tsv (test, source, verdict, detail), W/<source>/{old,new}/ (replay_checks.sh's output),
#   W/<source>/compare.tsv (run, saved verdict, new verdict, unrecorded); done when W/DONE exists
set -uo pipefail
CHECKS=$(realpath "${CHECKS:?}"); OLD_CHECKS=$(realpath "${OLD_CHECKS:?}")
W=${W:?}
rm -rf "$W" && mkdir -p "$W"
W=$(realpath "$W")
R=$W/saved_logs.tsv
printf 'test\tsource\tverdict\tdetail\n' > "$R"
res() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$R"; }
pf() { if "$@"; then echo PASS; else echo FAIL; fi; }
mkdir -p "$W/stub"
cat > "$W/stub/test-backend-ops" << 'EOF'
#!/usr/bin/env bash
# stands in for test-backend-ops under replay_checks.sh: the saved run whose per-case table the script names
r=$(basename "${GGML_TEST_REPLAY_LOG:?}" .tsv)
L=${STUB_SAVED:?}/$r.log
[ -f "$L" ] || { echo "stub: no saved $L"; exit 99; }
[ -f "$STUB_SAVED/$r.tsv" ] && cp "$STUB_SAVED/$r.tsv" "$GGML_TEST_REPLAY_LOG"
rc=$(tail -n 1 "$L" | sed -n 's/^exit //p')
sed '$d' "$L"
exit "${rc:-99}"
EOF
chmod +x "$W/stub/test-backend-ops"

# <label> <saved out> <runs.tsv> <runs, empty for all>: both scripts on the stub
both() {
    local l=$1 o=$2 rf=$3 runs=$4 c sc
    mkdir -p "$W/$l/set/cases"
    cp "$rf" "$W/$l/set/runs.tsv"
    for c in old new; do
        sc=$([ $c = new ] && echo "$CHECKS" || echo "$OLD_CHECKS")
        STUB_SAVED=$o SET=$W/$l/set BIN=$W/stub OUT=$W/$l/$c RUNS="$runs" bash "$sc" > "$W/$l/$c.out" 2>&1
    done
}
# <label> <saved out>: stub and new checks; prints nothing, writes compare.tsv
check() {
    local l=$1 o=$2 same bad nd
    same=$(cmp -s "$W/$l/old/results.tsv" "$o/results.tsv" && echo yes || echo no)
    res stub "$l" "$(pf test "$same" = yes)" "OLD_CHECKS' results.tsv on the stub the saved one byte for byte: $same ($(($(wc -l < "$o/results.tsv") - 1)) runs)" >&2
    awk -F'\t' 'BEGIN { OFS = "\t"; print "run", "saved", "new", "unrecorded", "other_columns" }
        NR == FNR { s[$1] = $0; next } FNR > 1 {
            split(s[$1], a, "\t"); same = 1
            for (i = 1; i <= 9; i++) if (i != 8 && a[i] != $i) same = 0
            print $1, a[8], $8, $10, same ? "same" : "differ" }' "$o/results.tsv" "$W/$l/new/results.tsv" > "$W/$l/compare.tsv"
    bad=$(awk -F'\t' 'NR > 1 && ($5 != "same" || ($4 == 0 && $2 != $3))' "$W/$l/compare.tsv" | cut -f1 | tr '\n' ' ')
    nd=$(awk -F'\t' 'NR > 1 && $2 != $3' "$W/$l/compare.tsv" | wc -l)
    echo "$bad|$nd"
}

# R112 on -kern
both r112-kern "$R112_OUT" "$R112_RUNS" ""
IFS='|' read -r bad nd <<< "$(check r112-kern "$R112_OUT")"
C=$W/r112-kern/compare.tsv
nu=$(awk -F'\t' 'NR > 1 && $4 > 0' "$C" | wc -l)
nu30=$(awk -F'\t' 'NR > 1 && $4 == 30 && $1 ~ /^dmt-/ && $2 == "WRONG" && $3 == "ok"' "$C" | wc -l)
nok=$(awk -F'\t' 'NR > 1 && $3 == "ok"' "$C" | wc -l)
U=$W/r112-kern/new/unrecorded.tsv
ids=$(cut -f2 "$U" | tail -n +2 | sort -u | wc -l); lines=$(($(wc -l < "$U") - 1))
shape=$(cut -f3 "$U" | tail -n +2 | sort -u | grep -c -v -E '^MUL_MAT\(type_a=(iq4_xs|q4_K),type_b=f32,m=[0-9]+,n=[1-5],k=6144,')
ran=$(zcat "${R112_CASES:?}" | cut -f2 | sort -u > "$W/r112-ran-ids.txt"; cut -f2 "$U" | tail -n +2 | sort -u | comm -12 - "$W/r112-ran-ids.txt" | wc -l)
res new r112-kern "$(pf test -z "$bad" -a "$nu" = 14 -a "$nu30" = 14 -a "$nd" = 14 -a "$nok" = 71 -a "$ids" = 30 -a "$lines" = 420 -a "$shape" = 0 -a "$ran" = 0)" \
    "71 runs, $nok ok; $nu with unrecorded cases, $nu30 of them dmt runs with 30 (saved WRONG, now ok); verdicts changed in $nd runs; other columns or verdicts off: ${bad:-none}; unrecorded.tsv $lines lines, $ids distinct ids, $shape not IQ4_XS/Q4_K k = 6144 n = 1-5, $ran among the ids the replay ran"
nl=$(grep -c '^    NOREC ' "$W/r112-kern/new/console.txt"); ns=$(grep -c '^    the same 30 unrecorded cases as dmt-on$' "$W/r112-kern/new/console.txt")
nv=$(grep -c ' ok, 30 unrecorded$' "$W/r112-kern/new/console.txt")
res names r112-kern "$(pf test "$nl" = 30 -a "$ns" = 13 -a "$nv" = 14)" \
    "console: $nv run lines end 'ok, 30 unrecorded'; $nl NOREC name lines; $ns 'the same 30 unrecorded cases as dmt-on' lines; last line: $(tail -n 1 "$W/r112-kern/new/console.txt" | cut -d' ' -f3-)"

# R112's dmt runs on -day
DMT=$(cut -f1 "$R112_RUNS" | grep '^dmt-' | tr '\n' ' ')
both r112-day "$R112_DAY_OUT" "$R112_RUNS" "$DMT"
IFS='|' read -r bad nd <<< "$(check r112-day "$R112_DAY_OUT")"
C=$W/r112-day/compare.tsv
nok=$(awk -F'\t' 'NR > 1 && $3 == "ok" && $4 == 0' "$C" | wc -l)
res new r112-day "$(pf test -z "$bad" -a "$nd" = 0 -a "$nok" = 14)" "14 dmt runs: $nok ok with 0 unrecorded; verdicts changed in $nd; off: ${bad:-none}"

# R73 on -kern
both r73 "$R73_OUT" "$R73_RUNS" ""
IFS='|' read -r bad nd <<< "$(check r73 "$R73_OUT")"
C=$W/r73/compare.tsv
nok=$(awk -F'\t' 'NR > 1 && $3 == "ok" && $4 == 50' "$C" | wc -l)
U=$W/r73/new/unrecorded.tsv
cut -f2 "$U" | tail -n +2 | sort -u > "$W/r73-ids.txt"
tail -n +2 "${R73_UNREC:?}" | cut -f1 | sort -u > "$W/r73-ids-published.txt"
same=$(cmp -s "$W/r73-ids.txt" "$W/r73-ids-published.txt" && echo yes || echo no)
res new r73 "$(pf test -z "$bad" -a "$nd" = 9 -a "$nok" = 11 -a "$same" = yes)" \
    "11 runs: $nok ok with 50 unrecorded (the 9 that must pass were WRONG, now ok; the 2 faults ok as before); verdicts changed in $nd; off: ${bad:-none}; the $(wc -l < "$W/r73-ids.txt") ids are R73's published unrecorded.tsv: $same"

# made-up failures on the saved logs
FK=$W/fake
mkdir -p "$FK/saved" "$FK/set/cases"
python3 - "$R112_OUT" "$R73_OUT" "$FK/saved" << 'EOF'
import re, sys
r112, r73, out = sys.argv[1:4]
OK = re.compile(r"^  ((\w+)\(.*\)): \x1b\[1;32mOK\x1b\[0m$")
BAD = re.compile(r"^\[(\w+)\] ERR = \S+ > \S+   ((\w+)\(.*\)): \x1b\[1;31mFAIL\x1b\[0m$")
PASSED = re.compile(r"^  (\d+)/(\d+) tests passed")

def read(d, r):
    return open(f"{d}/{r}.log", encoding="latin-1").read().split("\n"), open(f"{d}/{r}.tsv").read().split("\n")

def write(r, log, tsv):
    open(f"{out}/{r}.log", "w", encoding="latin-1").write("\n".join(log))
    open(f"{out}/{r}.tsv", "w").write("\n".join(tsv))

def one_fails(d, r, new):
    log, tsv = read(d, r)
    for i, l in enumerate(log):
        m = OK.match(l)
        if m:
            name = m.group(1)
            log[i] = f"[{m.group(2)}] ERR = 1.000000000 > 0.000500000   {name}: \x1b[1;31mFAIL\x1b[0m"
            break
    log = [PASSED.sub(lambda m: f"  {int(m.group(1)) - 1}/{m.group(2)} tests passed", l) for l in log]
    tsv = [("\t".join(v[:2] + ["FAIL"] + v[3:]) if len(v) > 2 and v[1] == name else l) for l in tsv for v in [l.split("\t")]]
    write(new, log, tsv)
    print(new, name)

def none_fail(d, r, new):
    log, tsv = read(d, r)
    n = 0
    for i, l in enumerate(log):
        m = BAD.match(l)
        if m:
            log[i] = f"  {m.group(2)}: \x1b[1;32mOK\x1b[0m"
            n += 1
    log = [PASSED.sub(lambda m: f"  {m.group(2)}/{m.group(2)} tests passed", l) for l in log]
    tsv = [("\t".join(v[:2] + ["OK"] + v[3:]) if len(v) > 2 and v[2] == "FAIL" else l) for l in tsv for v in [l.split("\t")]]
    write(new, log, tsv)
    print(new, n)

one_fails(r112, "dmt-on", "dmt-on-fail")
one_fails(r73, "qp-stock", "qp-stock-fail")
none_fail(r73, "qp-fault2", "qp-fault2-missed")
log, tsv = read(r73, "qp-fault2")
write("qp-fault2", log, tsv)
EOF
{
    awk -F'\t' 'BEGIN { OFS = "\t" } $1 == "dmt-on" { $1 = "dmt-on-fail"; print }' "$R112_RUNS"
    awk -F'\t' 'BEGIN { OFS = "\t" } $1 == "qp-stock" { $1 = "qp-stock-fail"; print }' "$R73_RUNS"
    awk -F'\t' '$1 == "qp-fault2"' "$R73_RUNS"
    awk -F'\t' 'BEGIN { OFS = "\t" } $1 == "qp-fault2" { $1 = "qp-fault2-missed"; print }' "$R73_RUNS"
} > "$FK/set/runs.tsv"
for c in old new; do
    sc=$([ $c = new ] && echo "$CHECKS" || echo "$OLD_CHECKS")
    STUB_SAVED=$FK/saved SET=$FK/set BIN=$W/stub OUT=$FK/$c RUNS="" bash "$sc" > "$FK/$c.out" 2>&1
done
got=$(awk -F'\t' 'NR > 1 { print $1 ":" $8 "/" $10 }' "$FK/new/results.tsv" | tr '\n' ' ')
old=$(awk -F'\t' 'NR > 1 { print $1 ":" $8 }' "$FK/old/results.tsv" | tr '\n' ' ')
res fake saved "$(pf test "$got" = "dmt-on-fail:WRONG/30 qp-stock-fail:WRONG/50 qp-fault2:ok/50 qp-fault2-missed:WRONG/50 ")" \
    "CHECKS: $got; OLD_CHECKS: $old"
date -Is > "$W/DONE"
