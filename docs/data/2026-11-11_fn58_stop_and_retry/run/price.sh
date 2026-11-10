#!/usr/bin/env bash
# price.sh: FN58's pricing of the agent check's generated-token mark on the recorded runs (registry Q1200), then the
# agent check's self-test, which exercises the mark against a mock server. CPU only, no model, no server.
#   REPO=<checkout of this repository> DATA=<folder holding the recorded runs' folders as run/runs.tsv names them>
#   OUT=<output folder> [SANDBOX=docker] bash price.sh
# The self-test needs the sandbox of bench/agent_check (README "Setup"; SANDBOX=docker uses the agent-check:0.87.1
# image, built from bench/agent_check/docker/Dockerfile) and ports 18590 to 18592 on 127.0.0.1.
set -euo pipefail
: "${REPO:?}" "${DATA:?}" "${OUT:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
AC=$REPO/bench/agent_check
mkdir -p "$OUT"
cp "$HERE/runs.tsv" "$DATA/runs.tsv"
python3 -I "$AC/token_mark.py" price --runs "$DATA/runs.tsv" --out "$OUT/price" \
  --marks 2048,4096,8192,12288,16384,24576,32768,49152,65536 --tries 2 > "$OUT/price.log"
AC_SANDBOX=${SANDBOX:-docker} AC_CPU_SHARES=256 AC_TEST_PORT=18590 bash "$AC/selftest.sh" "$OUT/selftest" \
  > "$OUT/selftest.log" 2>&1
tail -n 1 "$OUT/selftest.log"
