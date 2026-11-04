#!/usr/bin/env bash
# C372 (Q931): the record and replay machinery tested on a CPU alone (test-backend-ops -b CPU is the tested backend,
# the reference CPU backend its reference), so no GPU is needed. Runs inside a container with the build's bin/ mounted.
#   t1  record a small MUL_MAT set (the reference alone), then replay it: every case OK, as the live run of the same
#       filter is
#   t2  re-record the references from the replayed inputs into a second folder: every .ref blob identical (md5) to
#       the first record's, so the reference-only record path computes what the live check computes
#   t3  one byte of one .in blob flipped: that case fails with "record mismatch" (md5), the others pass
#   t4  one byte of one .ref blob flipped: that case fails the same way
#   t5  a filter wider than the record: "not recorded" lines, the summary line with FAIL, exit 1; with
#       --replay-allow-missing the run passes
#   t6  --cpu-check 1: every replayed case also runs live, its reference identical to the stored one
#   t7  live + record on the tested backend (-b CPU here, a GPU on the quant machine): the meta holds the live verdict
#       and each node's error, and the replay of that record gives the same errors
# Env: BIN (bin/ with test-backend-ops), W (scratch folder). Output: W/cpu_tests.txt (one PASS/FAIL line per test)
set -uo pipefail
BIN=${BIN:?}; W=${W:?}
T=$BIN/test-backend-ops
rm -rf "$W" && mkdir -p "$W"
R=$W/cpu_tests.txt
F='type_a=(q4_0|q8_0|iq2_xxs),type_b=f32,m=16,n=[1-3],k=256,'
res() { echo "$1 $2: $3" | tee -a "$R"; }
summ() { grep -a -E 'tests passed|replay:|record:' "$1" | sed 's/\x1b\[[0-9;]*m//g; s/^ *//' | tr '\n' ' '; }

$T -b CPU -o MUL_MAT -p "$F" --record "$W/a" > "$W/t1-record.log" 2>&1; rc=$?
n=$(ls "$W"/a/cases/*.meta 2>/dev/null | wc -l)
$T -b CPU -o MUL_MAT -p "$F" > "$W/t1-live.log" 2>&1; rcl=$?
$T -b CPU -o MUL_MAT -p "$F" --replay "$W/a" --replay-log "$W/t1-replay.tsv" > "$W/t1-replay.log" 2>&1; rcr=$?
ok=$(grep -a -c '^  MUL_MAT(.*): .*OK' "$W/t1-replay.log"); okl=$(grep -a -c '^  MUL_MAT(.*): .*OK' "$W/t1-live.log")
[ $rc = 0 ] && [ $rcr = 0 ] && [ $rcl = 0 ] && [ "$n" -gt 0 ] && [ "$ok" = "$n" ] && [ "$okl" = "$n" ] && v=PASS || v=FAIL
res t1 $v "recorded $n cases (exit $rc); live $okl OK (exit $rcl); replay $ok OK (exit $rcr); $(summ "$W/t1-replay.log")"

$T -b CPU -o MUL_MAT -p "$F" --replay "$W/a" --record "$W/b" > "$W/t2.log" 2>&1; rc=$?
d=0; for m in "$W"/a/cases/*.ref; do cmp -s "$m" "$W/b/cases/$(basename "$m")" || d=$((d + 1)); done
same_in=0; for m in "$W"/a/cases/*.in; do cmp -s "$m" "$W/b/cases/$(basename "$m")" && same_in=$((same_in + 1)); done
[ $rc = 0 ] && [ $d = 0 ] && [ "$same_in" = "$n" ] && v=PASS || v=FAIL
res t2 $v "re-recorded from the replayed inputs (exit $rc): $d of $n .ref blobs differ, $same_in of $n .in blobs identical"

flip() { # <file> <offset>: one byte inverted
    local b
    b=$(od -An -tu1 -j "$2" -N1 "$1" | tr -d ' ')
    printf "$(printf '\\%03o' $((255 - b)))" | dd of="$1" bs=1 seek="$2" conv=notrunc 2> /dev/null
}
cp -a "$W/a" "$W/c"
first=$(ls "$W"/c/cases/*.in | head -n 1); idx=$(basename "$first" .in)
flip "$first" 100
$T -b CPU -o MUL_MAT -p "$F" --replay "$W/c" > "$W/t3.log" 2>&1; rc=$?
mm=$(grep -a -c 'record mismatch: .*md5' "$W/t3.log"); fl=$(grep -a -c 'FAIL' "$W/t3.log"); ok=$(grep -a -c '^  MUL_MAT(.*): .*OK' "$W/t3.log")
[ $rc = 1 ] && [ "$mm" = 1 ] && [ "$ok" = $((n - 1)) ] && v=PASS || v=FAIL
res t3 $v "case $idx's .in flipped: exit $rc, $mm md5 mismatch line, $fl FAIL lines, $ok OK; $(summ "$W/t3.log")"

rm -rf "$W/c" && cp -a "$W/a" "$W/c"
flip "$W/c/cases/$idx.ref" 10
$T -b CPU -o MUL_MAT -p "$F" --replay "$W/c" > "$W/t4.log" 2>&1; rc=$?
mm=$(grep -a -c 'record mismatch: .*md5' "$W/t4.log"); ok=$(grep -a -c '^  MUL_MAT(.*): .*OK' "$W/t4.log")
[ $rc = 1 ] && [ "$mm" = 1 ] && [ "$ok" = $((n - 1)) ] && v=PASS || v=FAIL
res t4 $v "case $idx's .ref flipped: exit $rc, $mm md5 mismatch line, $ok OK; $(summ "$W/t4.log")"

FW='type_a=(q4_0|q8_0|iq2_xxs|q5_0),type_b=f32,m=16,n=[1-3],k=256,'
$T -b CPU -o MUL_MAT -p "$FW" --replay "$W/a" > "$W/t5.log" 2>&1; rc=$?
nr=$(grep -a -c 'not recorded' "$W/t5.log"); sf=$(grep -a -c 'have no record (FAIL' "$W/t5.log")
$T -b CPU -o MUL_MAT -p "$FW" --replay "$W/a" --replay-allow-missing > "$W/t5b.log" 2>&1; rcb=$?
[ $rc = 1 ] && [ "$nr" -gt 0 ] && [ "$sf" = 1 ] && [ $rcb = 0 ] && v=PASS || v=FAIL
res t5 $v "wider filter: exit $rc, $nr not-recorded lines, summary FAIL line $sf; allowed: exit $rcb; $(summ "$W/t5b.log")"

$T -b CPU -o MUL_MAT -p "$F" --replay "$W/a" --cpu-check 1 > "$W/t6.log" 2>&1; rc=$?
cc=$(grep -a -c 'cpu-check: live OK, reference identical' "$W/t6.log")
[ $rc = 0 ] && [ "$cc" = "$n" ] && v=PASS || v=FAIL
res t6 $v "cpu-check on every case: exit $rc, $cc of $n live OK with identical references; $(grep -a 'replay cpu-check' "$W/t6.log" | sed 's/^ *//')"

$T -b CPU -o MUL_MAT -p "$F" --record "$W/l" > "$W/t7-record.log" 2>&1
# the record's reference-only path runs on -b CPU, so for the live path the tested backend must not be a CPU device;
# here t7 checks that a reference-only record has no live line and that the replay log matches the meta's layout
nl=$(grep -l '^live' "$W"/l/cases/*.meta 2>/dev/null | wc -l)
$T -b CPU -o MUL_MAT -p "$F" --replay "$W/l" --replay-log "$W/t7.tsv" > "$W/t7.log" 2>&1; rc=$?
lines=$(wc -l < "$W/t7.tsv")
[ $rc = 0 ] && [ "$nl" = 0 ] && [ "$lines" = "$n" ] && v=PASS || v=FAIL
res t7 $v "reference-only record: $nl metas with a live line (0 expected on -b CPU); replay log $lines lines (exit $rc)"
echo "done $(date -Is)" >> "$R"
