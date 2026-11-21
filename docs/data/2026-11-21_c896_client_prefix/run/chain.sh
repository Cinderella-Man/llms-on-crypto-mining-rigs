#!/usr/bin/env bash
# chain.sh ROOT: the three checks of C896, one after another, on a machine with Docker and the sandbox image
# (agent-check:0.87.1). ROOT holds new/ (this repository's bench/agent_check, bench/task_results.py,
# bench/paired_tasks.py, scripts/realloc_counts.py and docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh at
# their paths, with C896) and old/ (the same paths from public v5 before C896, for c896_compare.sh). Outputs go to
# ROOT/runs/; ROOT/runs/chain.times gets each check's exit code and end time.
set -uo pipefail
ROOT=$(cd "${1:?usage: chain.sh ROOT}" && pwd -P)
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$ROOT" || exit 1
mkdir -p runs
date +%T > runs/chain.times
AC_CPU_SHARES=256 nice -n 19 bash new/bench/agent_check/selftest_prefix.sh runs/prefix > runs/prefix.txt 2>&1
echo "prefix $? $(date +%T)" >> runs/chain.times
AC_SANDBOX=docker AC_CPU_SHARES=256 nice -n 19 bash new/bench/agent_check/selftest.sh runs/selftest > runs/selftest.txt 2>&1
echo "selftest $? $(date +%T)" >> runs/chain.times
nice -n 19 bash "$HERE/c896_compare.sh" "$ROOT" > runs/compare.txt 2>&1
echo "compare $? $(date +%T)" >> runs/chain.times
