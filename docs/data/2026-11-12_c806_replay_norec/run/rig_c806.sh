#!/usr/bin/env bash
# C806 on a P106: the v4day set (R112's, recorded with pascal-p106-v4-b12246-day) replayed on one card with a build that
# has C806, through scripts/replay_checks.sh with C806, on the 16 runs of rig_runs.tsv: R112's 14 dmt runs, whose filter
# selects 30 cases the set does not hold (C728's K-tile cases, added after it was recorded), and dmt-q3f-pass and
# dmt-q3f-fail, the dmt filter joined with the q3kmt suite's under GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT=1 (the same 30
# unrecorded beside failing recorded cases).
# Passing: the 14 dmt runs read "ok, 30 unrecorded" (315 of 315 recorded cases pass in each), the 30 named under the
#   first and "the same 30 unrecorded cases as dmt-on" under the rest; dmt-q3f-pass reads "WRONG, 30 unrecorded" (its
#   q3_K cases fail as in R112's q3kmt-fault: 103 of 170 there); dmt-q3f-fail reads "ok, 30 unrecorded". The console's
#   last line: "15 runs as expected, 1 WRONG" (the 1 is dmt-q3f-pass, which must be WRONG).
# Needs: the rig idle; nothing else runs on it. About 5 minutes (R112's 14 dmt runs took 3 min 21 s on -day).
# Env: SET (the unpacked v4day set: cases/, runs.tsv), BIN (bin/ of the C806 tarball), OUT, CHECKS (scripts/
#   replay_checks.sh with C806), RUNS_FILE (default rig_runs.tsv beside this script), DEV (the card, default 0)
# Output: OUT/before.txt and OUT/after.txt (test-backend-ops' and the script's md5, the card, Xid lines),
#   OUT/replay/ (replay_checks.sh's output), OUT/status.txt (the console's last line), OUT/PID; done when OUT/DONE exists
set -uo pipefail
SET=${SET:?SET=<the unpacked v4day set>}; BIN=${BIN:?BIN=<bin/ of the C806 tarball>}; OUT=${OUT:?OUT=<dir>}
CHECKS=${CHECKS:?CHECKS=<scripts/replay_checks.sh with C806>}
RUNS_FILE=${RUNS_FILE:-$(dirname "$(realpath "$0")")/rig_runs.tsv}
DEV=${DEV:-0}
mkdir -p "$OUT"
rm -f "$OUT/DONE"
echo $$ > "$OUT/PID"
state() {
    date -Is
    md5sum "$BIN/test-backend-ops" "$CHECKS" "$RUNS_FILE"
    nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader
    echo "Xid lines in the kernel log: $(dmesg 2> /dev/null | grep -c 'NVRM: Xid')"
}
state > "$OUT/before.txt" 2>&1
SET=$SET BIN=$BIN OUT=$OUT/replay RUNS_FILE=$RUNS_FILE DEV=$DEV bash "$CHECKS" > "$OUT/replay.out" 2>&1
state > "$OUT/after.txt" 2>&1
tail -n 1 "$OUT/replay/console.txt" > "$OUT/status.txt"
date -Is > "$OUT/DONE"
