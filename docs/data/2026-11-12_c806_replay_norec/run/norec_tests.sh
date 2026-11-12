#!/usr/bin/env bash
# C806 (2026-11-12): test-backend-ops' replay naming the cases a record does not hold (verdict NOREC), and
# scripts/replay_checks.sh's verdicts with them, on a CPU alone: test-backend-ops -b CPU is the tested backend, so no
# GPU is needed (C372's method, docs/data/2026-11-04_replay_checks/run/cpu_tests.sh). NEW is the build with C806, OLD
# the base it branches from; CHECKS is replay_checks.sh with C806, OLD_CHECKS the one before it.
#   c372  C372's seven tests on NEW as they stand: t1-t7 PASS (t5 counts "not recorded" lines: NOREC lines keep the words)
#   n0    set A: a record (-b CPU --record) of the filter F, MUL_MAT of q4_0, q8_0 and iq2_xxs at m = 16, n = 1-3,
#         k = 256; FW adds q5_0, whose cases A does not hold (their count: FW's live run less A's cases)
#   n1    A replayed with FW: NEW prints each unrecorded case as "<case>: NOREC (not recorded) [<meta>]" and gives it a
#         NOREC line in the replay log, every one a q5_0 case and none of A's; A's cases OK; the summary's FAIL line and
#         exit 1 as before. OLD prints "not recorded" lines and no log line for them
#   n2    the same with --replay-allow-missing: NEW still names them (output and log), exit 0, no FAIL; OLD names none
#   n3    set B: A with one q8_0 case's references doubled plus one (its md5 updated in the meta), so that the case
#         fails as a value error ("ERR = ... > ..."), not as a record mismatch: replayed with F, one FAIL case line, exit 1
#   n4    replay_checks.sh on the 15 runs of RUNLIST (sets A and B, expect pass and fail, a known failure, with and
#         without GGML_TEST_REPLAY_ALLOW_MISSING=1) through a shim that runs "-b CUDA0" as "-b CPU", for each of
#         CHECKS and OLD_CHECKS on each of NEW and OLD: CHECKS gives each run the verdict and unrecorded count in
#         RUNLIST on both builds, NEW naming every unrecorded case and OLD every one but those under ALLOW_MISSING;
#         OLD_CHECKS gives CHECKS' verdict to every run without unrecorded cases (the verdicts for recorded cases are
#         unchanged), and the runs where the two differ are listed
#   n5    CHECKS with NOREC=wrong (a replay on the recording build): a run with unrecorded cases is WRONG, one without
#         keeps its verdict
# Runs inside a container of a Ubuntu 24.04 image with python3 (the rigs' userland: GNU coreutils), nothing else.
# Env: NEW and OLD (bin/ folders holding test-backend-ops), CHECKS, OLD_CHECKS, C372 (the 2026-11-04 cpu_tests.sh),
#   W (work folder, emptied first)
# Output: W/norec_tests.tsv (test, build, verdict, detail), W/n4/verdicts.tsv (run, expect, unrecorded, each script x
#   build's verdict and count), W/n4/<script>-<build>-<set>/ (replay_checks.sh's output), the logs; done when W/DONE
#   exists
set -uo pipefail
NEW=$(realpath "${NEW:?}"); OLD=$(realpath "${OLD:?}")
CHECKS=$(realpath "${CHECKS:?}"); OLD_CHECKS=$(realpath "${OLD_CHECKS:?}"); C372=$(realpath "${C372:?}")
W=${W:?}
rm -rf "$W" && mkdir -p "$W"
W=$(realpath "$W")
R=$W/norec_tests.tsv
printf 'test\tbuild\tverdict\tdetail\n' > "$R"
res() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" | tee -a "$R"; }
pf() { if "$@"; then echo PASS; else echo FAIL; fi; }
bin() { [ "$1" = new ] && echo "$NEW" || echo "$OLD"; }
F='type_a=(q4_0|q8_0|iq2_xxs),type_b=f32,m=16,n=[1-3],k=256,'
FW='type_a=(q4_0|q8_0|iq2_xxs|q5_0),type_b=f32,m=16,n=[1-3],k=256,'
FQ='type_a=q5_0,type_b=f32,m=16,n=[1-3],k=256,'
nolines() { grep -a -c -E '^  MUL_MAT\(.*\): NOREC \(not recorded\) \[' "$1"; }
nrlines() { grep -a -c -E '^  MUL_MAT\(.*\): not recorded \[' "$1"; }
nlog() { [ -f "$1" ] && awk -F'\t' -v v="$2" '$3 == v' "$1" | wc -l || echo 0; }

# c372
BIN=$NEW W=$W/c372 bash "$C372" > "$W/c372.out" 2>&1
np=$(grep -c -E '^t[0-9] PASS:' "$W/c372/cpu_tests.txt" 2> /dev/null)
res c372 new "$([ "$np" = 7 ] && echo PASS || echo FAIL)" "$np of 7 PASS; $(grep -E '^t5 ' "$W/c372/cpu_tests.txt" | cut -c1-200)"

# n0
"$NEW/test-backend-ops" -b CPU -o MUL_MAT -p "$F" --record "$W/A" > "$W/n0-record.log" 2>&1; rc=$?
n=$(ls "$W"/A/cases/*.meta 2> /dev/null | wc -l)
"$NEW/test-backend-ops" -b CPU -o MUL_MAT -p "$FW" > "$W/n0-live-wide.log" 2>&1
nw=$(grep -a -c -E '^  MUL_MAT\(.*\): .*OK' "$W/n0-live-wide.log")
nq=$(grep -a -E '^  MUL_MAT\(.*\): .*OK' "$W/n0-live-wide.log" | grep -c 'type_a=q5_0')
u=$((nw - n))
grep -h '^name' "$W"/A/cases/*.meta | cut -f2 | sort > "$W/A-names.txt"
res n0 new "$(pf test "$rc" = 0 -a "$n" -gt 0 -a "$u" -gt 0 -a "$u" = "$nq")" \
    "A holds $n cases (exit $rc); FW selects $nw on a live run, $nq of them q5_0: $u unrecorded expected"

# n1, n2
for b in new old; do
    T=$(bin $b)/test-backend-ops
    "$T" -b CPU -o MUL_MAT -p "$FW" --replay "$W/A" --replay-log "$W/n1-$b.tsv" > "$W/n1-$b.log" 2>&1; rc=$?
    "$T" -b CPU -o MUL_MAT -p "$FW" --replay "$W/A" --replay-log "$W/n2-$b.tsv" --replay-allow-missing > "$W/n2-$b.log" 2>&1; rc2=$?
    for t in n1 n2; do
        L=$W/$t-$b.log
        sed -n -E 's/^  (MUL_MAT\(.*\)): (NOREC \(not recorded\)|not recorded) \[.*$/\1/p' "$L" | sort > "$W/$t-$b-names.txt"
        awk -F'\t' '$3 == "NOREC" { print $2 }' "$W/$t-$b.tsv" 2> /dev/null | sort > "$W/$t-$b-lognames.txt"
    done
    a1=$(nolines "$W/n1-$b.log"); o1=$(nrlines "$W/n1-$b.log"); l1=$(nlog "$W/n1-$b.tsv" NOREC); k1=$(nlog "$W/n1-$b.tsv" OK)
    s1=$(grep -a -c 'have no record (FAIL' "$W/n1-$b.log")
    q1=$(grep -c -v 'type_a=q5_0' "$W/n1-$b-names.txt"); in1=$(comm -12 "$W/n1-$b-names.txt" "$W/A-names.txt" | wc -l)
    a2=$(nolines "$W/n2-$b.log"); o2=$(nrlines "$W/n2-$b.log"); l2=$(nlog "$W/n2-$b.tsv" NOREC); k2=$(nlog "$W/n2-$b.tsv" OK)
    s2=$(grep -a -c 'FAIL' "$W/n2-$b.log")
    if [ $b = new ]; then
        v1=$(pf test "$rc" = 1 -a "$a1" = "$u" -a "$o1" = 0 -a "$l1" = "$u" -a "$k1" = "$n" -a "$s1" = 1 -a "$q1" = 0 -a "$in1" = 0)
        cmp -s "$W/n1-new-names.txt" "$W/n1-new-lognames.txt" || v1=FAIL
        v2=$(pf test "$rc2" = 0 -a "$a2" = "$u" -a "$l2" = "$u" -a "$k2" = "$n" -a "$s2" = 0)
        cmp -s "$W/n1-new-names.txt" "$W/n2-new-names.txt" || v2=FAIL
    else
        v1=$(pf test "$rc" = 1 -a "$a1" = 0 -a "$o1" = "$u" -a "$l1" = 0 -a "$k1" = "$n" -a "$s1" = 1 -a "$q1" = 0 -a "$in1" = 0)
        v2=$(pf test "$rc2" = 0 -a "$a2" = 0 -a "$o2" = 0 -a "$l2" = 0 -a "$k2" = "$n" -a "$s2" = 0)
    fi
    res n1 $b "$v1" "FW on A: exit $rc; $a1 NOREC lines, $o1 'not recorded' lines (of $u unrecorded), $q1 not q5_0, $in1 in A; replay log $k1 OK, $l1 NOREC; summary FAIL lines $s1"
    res n2 $b "$v2" "FW on A, --replay-allow-missing: exit $rc2; $a2 NOREC lines, $o2 'not recorded' lines; replay log $k2 OK, $l2 NOREC; FAIL lines $s2"
done

# n3: set B, one value failure
cp -a "$W/A" "$W/B"
bad=$(grep -l -E '^name	MUL_MAT\(type_a=q8_0,' "$W"/B/cases/*.meta | sort | head -n 1)
bid=$(basename "$bad" .meta)
bname=$(grep '^name' "$bad" | cut -f2)
python3 - "$W/B/cases/$bid.ref" "$bad" << 'EOF'
import hashlib, struct, sys
ref, meta = sys.argv[1], sys.argv[2]
lines = open(meta).read().split("\n")
b = bytearray(open(ref, "rb").read())
off = 0
for l in lines:
    v = l.split("\t")
    if v[0] == "node":
        raw, n = int(v[5]), int(v[6])
        off += raw
        for i in range(n):
            (x,) = struct.unpack_from("<f", b, off + 4 * i)
            struct.pack_into("<f", b, off + 4 * i, 2 * x + 1)
        off += 4 * n
assert off == len(b), (off, len(b))
open(ref, "wb").write(b)
h = hashlib.md5(b).hexdigest()
out = []
for l in lines:
    v = l.split("\t")
    if v[0] == "ref":
        v[2] = h
    out.append("\t".join(v))
open(meta, "w").write("\n".join(out))
EOF
"$NEW/test-backend-ops" -b CPU -o MUL_MAT -p "$F" --replay "$W/B" > "$W/n3.log" 2>&1; rc=$?
ef=$(grep -a -E 'ERR = .* > ' "$W/n3.log" | grep -a -c 'FAIL'); fl=$(grep -a -c 'FAIL' "$W/n3.log")
mm=$(grep -a -c 'record mismatch' "$W/n3.log")
res n3 new "$(pf test "$rc" = 1 -a "$ef" = 1 -a "$fl" = 3 -a "$mm" = 0)" \
    "case $bid ($bname) references doubled plus one: exit $rc, $ef value FAIL line, $fl FAIL lines, $mm record mismatches"

# n4: replay_checks.sh, both scripts on both builds
N4=$W/n4
mkdir -p "$N4"
for b in new old; do
    mkdir -p "$N4/shim-$b"
    printf '#!/usr/bin/env bash\n# replay_checks.sh calls test-backend-ops -b CUDA0: here the CPU backend stands in\nargs=()\nfor a in "$@"; do [ "$a" = CUDA0 ] && a=CPU; args+=("$a"); done\necho "shim: -b CUDA0 runs as -b CPU"\nexec %s/test-backend-ops "${args[@]}"\n' \
        "$(bin $b)" > "$N4/shim-$b/test-backend-ops"
    chmod +x "$N4/shim-$b/test-backend-ops"
done
K=$(printf '%s' "$bname" | sed 's/[][()|.*+?^$\\{}]/\\&/g')
AM=GGML_TEST_REPLAY_ALLOW_MISSING=1
# run, set, switches, filter, expect, known, CHECKS' verdict, unrecorded cases
cat > "$N4/runlist.tsv" << EOF
a-same	A	-	F	pass	-	ok	0
a-wide	A	-	FW	pass	-	ok	$u
a-wide-am	A	$AM	FW	pass	-	ok	$u
a-only	A	-	FQ	pass	-	WRONG	$u
a-only-am	A	$AM	FQ	pass	-	WRONG	$u
a-fault	A	-	FW	fail	-	WRONG	$u
a-fault-same	A	-	F	fail	-	WRONG	0
b-same	B	-	F	pass	-	WRONG	0
b-wide	B	-	FW	pass	-	WRONG	$u
b-wide-am	B	$AM	FW	pass	-	WRONG	$u
b-fault	B	-	FW	fail	-	ok	$u
b-fault-same	B	-	F	fail	-	ok	0
b-known	B	-	FW	pass	K	ok	$u
b-known-same	B	-	F	pass	K	ok	0
b-known-am	B	$AM	FW	pass	K	ok	$u
EOF
for s in A B; do
    awk -F'\t' -v s=$s -v F="$F" -v FW="$FW" -v FQ="$FQ" -v K="$K" 'BEGIN { OFS = "\t" } $2 == s {
        p = ($4 == "F") ? F : ($4 == "FW") ? FW : FQ
        print $1, $3, "MUL_MAT", p, $5, "-", ($6 == "K") ? K : "-" }' "$N4/runlist.tsv" > "$N4/runs-$s.tsv"
done
for c in new old; do
    sc=$([ $c = new ] && echo "$CHECKS" || echo "$OLD_CHECKS")
    for b in new old; do
        for s in A B; do
            SET=$W/$s RUNS_FILE=$N4/runs-$s.tsv BIN=$N4/shim-$b OUT=$N4/$c-$b-$s TIMEOUT=600 bash "$sc" > "$N4/$c-$b-$s.out" 2>&1
        done
    done
done
SET=$W/A RUNS_FILE=$N4/runs-A.tsv BIN=$N4/shim-new OUT=$N4/strict RUNS='a-same a-wide a-fault-same' NOREC=wrong \
    TIMEOUT=600 bash "$CHECKS" > "$N4/strict.out" 2>&1
SET=$W/B RUNS_FILE=$N4/runs-B.tsv BIN=$N4/shim-new OUT=$N4/strict-b RUNS='b-known b-known-same' NOREC=wrong \
    TIMEOUT=600 bash "$CHECKS" > "$N4/strict-b.out" 2>&1
# verdicts.tsv: run, expect, CHECKS' verdict and count wanted, then each script x build's verdict (and count for CHECKS)
vd() { awk -F'\t' -v r="$2" '$1 == r { print $8 (NF >= 10 ? "/" $10 : "") }' "$N4/$1/results.tsv"; }
printf 'run\texpect\twanted\tnew_script_new_build\tnew_script_old_build\told_script_new_build\told_script_old_build\n' > "$N4/verdicts.tsv"
while IFS=$'\t' read -r run s sw fl ex kn want nu; do
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "$ex" "$want/$nu" "$(vd new-new-$s "$run")" "$(vd new-old-$s "$run")" \
        "$(vd old-new-$s "$run")" "$(vd old-old-$s "$run")" >> "$N4/verdicts.tsv"
done < "$N4/runlist.tsv"
bad_new=$(awk -F'\t' 'NR > 1 && ($4 != $3 || $5 != $3)' "$N4/verdicts.tsv" | cut -f1 | tr '\n' ' ')
bad_old=$(awk -F'\t' 'NR > 1 { split($3, w, "/"); if (w[2] == 0 && ($6 != w[1] || $7 != w[1])) print $1 }' "$N4/verdicts.tsv" | tr '\n' ' ')
differ=$(awk -F'\t' 'NR > 1 { split($3, w, "/"); if ($6 != w[1]) print $1 " (" $6 " -> " w[1] ")" }' "$N4/verdicts.tsv" | tr '\n' ' ')
res n4 both "$(pf test -z "$bad_new" -a -z "$bad_old")" \
    "CHECKS: every run as wanted on both builds${bad_new:+ but $bad_new}; OLD_CHECKS: every run without unrecorded cases CHECKS' verdict${bad_old:+ but $bad_old}; the two scripts differ on: ${differ:-none}"
# the names: NEW names every unrecorded case of each run (the q5_0 cases of n1), OLD every one but under ALLOW_MISSING
nn_new=$(awk -F'\t' 'FNR > 1' "$N4"/new-new-?/unrecorded.tsv | wc -l)
nn_old=$(awk -F'\t' 'FNR > 1' "$N4"/new-old-?/unrecorded.tsv | wc -l)
runs_u=$(awk -F'\t' '$8 > 0' "$N4/runlist.tsv" | wc -l); runs_am=$(awk -F'\t' '$8 > 0 && $3 != "-"' "$N4/runlist.tsv" | wc -l)
# by case id: two cases may share one case string (the id counts the earlier cases with the same class and string)
awk -F'\t' 'FNR > 1 { print $2 }' "$N4"/new-new-?/unrecorded.tsv | sort -u > "$N4/ids-new.txt"
awk -F'\t' '$3 == "NOREC" { print $1 }' "$W/n1-new.tsv" | sort -u > "$W/n1-new-ids.txt"
awk -F'\t' 'FNR > 1 { print $3 }' "$N4"/new-new-?/unrecorded.tsv > "$N4/names-new.txt"
nm=$(grep -c -v 'type_a=q5_0' "$N4/names-new.txt")
nsame=$([ -s "$N4/ids-new.txt" ] && cmp -s "$N4/ids-new.txt" "$W/n1-new-ids.txt" && echo yes || echo no)
ns=$(grep -c 'not named by this build' "$N4"/new-old-?/console.txt | awk -F: '{ s += $2 } END { print s + 0 }')
res n4 names "$(pf test "$nn_new" = $((runs_u * u)) -a "$nn_old" = $(((runs_u - runs_am) * u)) -a "$nm" = 0 -a "$nsame" = yes -a "$ns" = "$runs_am")" \
    "unrecorded.tsv: NEW $nn_new lines ($runs_u runs x $u), OLD $nn_old ($((runs_u - runs_am)) runs x $u; $ns runs 'not named' under ALLOW_MISSING); the $(wc -l < "$N4/ids-new.txt") ids are n1's NOREC ids: $nsame; $nm names not q5_0"
s1=$(awk -F'\t' 'FNR > 1 { print $1 ":" $8 "/" $10 }' "$N4/strict/results.tsv" "$N4/strict-b/results.tsv" | tr '\n' ' ')
res n5 new "$(pf test "$s1" = "a-same:ok/0 a-wide:WRONG/$u a-fault-same:WRONG/0 b-known:WRONG/$u b-known-same:ok/0 ")" "NOREC=wrong: $s1"
date -Is > "$W/DONE"
