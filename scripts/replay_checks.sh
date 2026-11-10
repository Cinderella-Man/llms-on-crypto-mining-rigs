#!/usr/bin/env bash
# Replays a recorded test-backend-ops suite on a mining rig's GPU (C372, Q931): the counterpart of
# scripts/record_checks.sh. Each run of the suite's runs.tsv is one test-backend-ops call with its switches and
# GGML_TEST_REPLAY set to the suite's folder, so the cards compute and the stored CPU references judge them: no CPU
# quantization and no CPU op on the rig. Every replayed case is checked against its record first (the case string,
# the tensor layout, an md5 per blob); a record that does not match fails the case.
#
# The scripts that call test-backend-ops directly (each branch's gpu_checks.sh) replay unchanged as well: run them with
# GGML_TEST_REPLAY=<suite folder> in the environment.
#
# Env: SET (a suite's folder: cases/, runs.tsv), BIN (bin/ of a rig tarball of the replay branch or a branch merged
#   from it), OUT, RUNS (default: every run of the run list, in its order), RUNS_FILE (the run list, default
#   SET/runs.tsv; SET/runs-ptx.tsv, where a suite has one, is the list for a newer card through the compute_61 PTX),
#   DEV (the card, CUDA_DEVICE_ORDER=PCI_BUS_ID, default 0), CPU_CHECK (N: about one case in N with inputs under
#   32 MB also runs the live check against this host's CPU, default 0 = none; it is CPU work), CPU_CHECK_RUNS (the
#   runs that do it, default the first run: every run would check the same cases again), CS (compute-sanitizer for
#   the memcheck runs, default /usr/local/cuda/bin/compute-sanitizer), TIMEOUT (seconds a run, default 7200), ENV_ALL
#   (switches added to every run, e.g. the fork's PTX-mode settings when a newer card replays through the compute_61
#   PTX)
# Output: OUT/console.txt (one line per run: run, switches, test-backend-ops' summary, FAIL lines, exit status, what
#   passing means, ok or WRONG), OUT/results.tsv (the same, one tab-separated line per run: run, switches, expect,
#   cases passed, FAIL lines, exit status, memcheck errors, verdict, known cases that failed), OUT/<run>.log,
#   OUT/<run>.tsv (per case: id, case, verdict, each node's error), OUT/done
# A run whose list line names known failures (a seventh field: a regex on the case string of cases that may fail until
# a named fix) is ok when every FAIL line is one of those cases' or one of the two lines a failing run ends with.
# Usage: SET=<dir>/kshape BIN=<tarball>/bin OUT=<dir> [RUNS='kcol-all mtok-iq'] bash scripts/replay_checks.sh
set -uo pipefail
SET=$(realpath "${SET:?SET=<a recorded suite folder>}")
BIN=$(realpath "${BIN:?BIN=<bin/ of the rig tarball>}")
OUT=${OUT:?OUT=<dir>}
DEV=${DEV:-0}
CS=${CS:-/usr/local/cuda/bin/compute-sanitizer}
TIMEOUT=${TIMEOUT:-7200}
RUNS_FILE=${RUNS_FILE:-$SET/runs.tsv}
[ -f "$RUNS_FILE" ] && [ -d "$SET/cases" ] || { echo "no $RUNS_FILE, or $SET has no cases/" >&2; exit 1; }
mkdir -p "$OUT"
OUT=$(realpath "$OUT")
rm -f "$OUT/done"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
C=$OUT/console.txt
T=$OUT/results.tsv
printf 'run\tswitches\texpect\tcases\tfail_lines\texit\tmemcheck_errors\tverdict\tknown_fail\n' > "$T"
say() { echo "$*" | tee -a "$C"; }
wanted() { [ -z "${RUNS:-}" ] || [[ " $RUNS " == *" $1 "* ]]; }
summary() { # test-backend-ops' own summary lines
    grep -a -E 'tests passed|replay:|replay cpu-check:' "$1" | grep -a -v 'tensors and references from' \
        | sed 's/\x1b\[[0-9;]*m//g' | sed -E 's/^ +//; s/^replay: [^:]*: //' \
        | tr '\n' ';' | cut -c1-220
}
say "== $(date -Is) replay_checks SET=$SET BIN=$BIN DEV=$DEV runs: ${RUNS:-all} of $(basename "$RUNS_FILE")"
say "   $(grep -E '^(suite|recorded|name|commit)' "$SET/record.txt" 2>/dev/null | tr '\n' ';')"
say "   $(nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader -i "$DEV" 2>/dev/null)"
n_ok=0; n_wrong=0
first_run=""
# the run list: run, switches, -o, -p, expect (pass: every case passes; fail: some case fails, the run being a fault),
# wrap (memcheck: under compute-sanitizer), known (a regex on the case string: cases that may fail); "-" for an empty
# field
while IFS=$'\t' read -r run envs ops params expect wrap known; do
    [ -n "$run" ] && wanted "$run" || continue
    [ "$envs" = - ] && envs=""
    [ "$wrap" = - ] && wrap=""
    [ "${known:--}" = - ] && known=""
    [ -n "$first_run" ] || first_run=$run
    cc=0
    [[ " ${CPU_CHECK_RUNS:-$first_run} " == *" $run "* ]] && cc=${CPU_CHECK:-0}
    log=$OUT/$run.log
    rm -f "$OUT/$run.tsv"
    pre=()
    if [ "$wrap" = memcheck ]; then
        [ -x "$CS" ] || { say "$(printf '%-24s SKIPPED: no compute-sanitizer at %s' "$run" "$CS")"; continue; }
        pre=("$CS" --tool memcheck --print-limit 5)
    fi
    # shellcheck disable=SC2086
    env CUDA_VISIBLE_DEVICES="$DEV" ${ENV_ALL:-} $envs GGML_TEST_REPLAY="$SET" GGML_TEST_REPLAY_LOG="$OUT/$run.tsv" \
        GGML_TEST_REPLAY_CPU_CHECK="$cc" timeout "$TIMEOUT" "${pre[@]}" "$BIN/test-backend-ops" -b CUDA0 \
        -o "$ops" -p "$params" > "$log" 2>&1
    rc=$?
    echo "exit $rc" >> "$log"
    fails=$(grep -a -c 'FAIL' "$log")
    warn=""
    grep -a -q 'tests passed' "$log" || warn="NO CASES RAN; "
    grep -a -q 'CUDA0' "$log" || warn="${warn}NO CUDA0; "
    # the known cases that failed: with any, the run exits 1 and ends with two more FAIL lines (the backend's, the
    # run's), and nothing else may say FAIL
    kf=0
    [ -n "$known" ] && kf=$(grep -a 'FAIL' "$log" | grep -a -c -E "$known")
    v=WRONG
    case $expect in
        pass) if [ "$kf" -gt 0 ]; then
                  [ "$rc" = 1 ] && [ "$fails" = $((kf + 2)) ] && [ -z "$warn" ] && v=ok
              else
                  [ "$rc" = 0 ] && [ "$fails" = 0 ] && [ -z "$warn" ] && v=ok
              fi ;;
        fail) [ "$fails" -gt 0 ] && [ -z "$warn" ] && v=ok ;;
    esac
    kn=""
    [ -n "$known" ] && kn="($kf known to fail) "
    if [ "$wrap" = memcheck ]; then
        ms=$(grep -a -o 'ERROR SUMMARY: [0-9]* errors' "$log" | tail -n 1)
        [ "$ms" = "ERROR SUMMARY: 0 errors" ] || v=WRONG
        warn="$warn$ms; "
    fi
    if [ $v = ok ]; then n_ok=$((n_ok + 1)); else n_wrong=$((n_wrong + 1)); fi
    printf '%-24s %-44s %-64s %4s FAIL  exit %-3s expect %-4s %s%s\n' "$run" "${envs:-defaults}" "$warn$(summary "$log")" \
        "$fails" "$rc" "$expect" "$kn" "$v" | tee -a "$C"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$run" "${envs:-defaults}" "$expect" \
        "$(grep -a 'tests passed' "$log" | tail -n 1 | sed 's/\x1b\[[0-9;]*m//g; s/^ *//; s/ tests passed//')" "$fails" "$rc" \
        "$(grep -a -o 'ERROR SUMMARY: [0-9]* errors' "$log" | tail -n 1 | grep -o '[0-9]*')" "$v" "${known:+$kf}" >> "$T"
done < "$RUNS_FILE"
say "== $(date -Is) done: $n_ok runs as expected, $n_wrong WRONG"
date -Is > "$OUT/done"
