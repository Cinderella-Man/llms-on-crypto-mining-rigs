#!/usr/bin/env bash
# validate.sh OUT: the checks of this folder, before any rig minute. Runs only ssh and tar here; every script runs on
# CHECK_SSH's machine (any Linux machine that is not a rig, with bash, awk, python3, curl and chrt: a rig has no spare
# second), from a temporary folder there that is removed at the end. It uses no GPU and touches no process it did not start.
#   1. scripts/ram_watchdog_test.sh: the watchdog on synthetic /proc files and on the machine's own for REAL_S seconds, the
#      copies embedded in rig_server.sh and server_bench.sh, and both scripts end to end with stand-ins for llama-server and
#      nvidia-smi (selftest.txt, sample_real_<REAL_S>s.tsv)
#   2. bash -n and py_compile of every script this change touched (syntax.txt)
#   3. the first-turn request: built the way the smoke scripts build it (task folder rebuilt with build_bugfix.py on that
#      machine, request built from it), its size reported (request_summary.txt); the request itself is removed, as it holds
#      the task's text. With VOCAB_MODEL (the first shard of a GGUF on that machine) and LLAMA_TOKENIZE it is also
#      counted with the model's template and vocabulary (token_count.tsv)
#   4. DRY_RUN=1 of smoke_dflash.sh and smoke_mtp.sh: they build the request and say what they would send (smoke_dry.txt)
# Environment: CHECK_SSH (the ssh command, e.g. "ssh quant"), optional REPO (default: this checkout), REAL_S (60), VOCAB_MODEL,
# LLAMA_TOKENIZE.
set -uo pipefail
OUT=$(realpath -m "${1:?usage: validate.sh OUT}")
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=${REPO:-$(cd "$HERE/../../../.." && pwd)}
: "${CHECK_SSH:?CHECK_SSH=<ssh command of the machine that runs the checks>}"
REAL_S=${REAL_S:-60}
mkdir -p "$OUT"
cd "$REPO" || exit 1
SRC="scripts/ram_watchdog.sh scripts/ram_watchdog_test.sh scripts/server_bench.sh bench/agent_check/rig_server.sh
  bench/agent_check/first_turn_request.sh bench/agent_check/first_turn_request.py
  docs/data/2026-11-03_flash_next_dflash2/run/smoke_dflash.sh docs/data/2026-11-03_flash_next_mtp/run/smoke_mtp.sh
  docs/data/2026-11-04_ram_watchdog/run/count_tokens.py docs/data/2026-11-04_ram_watchdog/run/summarize.sh"

echo "== 1. the self-test"
# shellcheck disable=SC2086
tar -cf - $SRC | $CHECK_SSH "d=\$(mktemp -d /tmp/rwt.XXXXXX); tar -xf - -C \"\$d\"; mkdir \"\$d/keep\"; \
  REAL_S=$REAL_S KEEP=\"\$d/keep\" REPO=\"\$d\" bash \"\$d/scripts/ram_watchdog_test.sh\"; rc=\$?; \
  echo '=== sample'; cat \"\$d/keep/real_${REAL_S}s.tsv\"; rm -rf \"\$d\"; exit \$rc" > "$OUT/selftest_raw.txt"
rc=$?
sed -e '/^=== sample$/,$d' -e '/^\/tmp\/rwt\./d' "$OUT/selftest_raw.txt" > "$OUT/selftest.txt"
sed -n '/^=== sample$/,$p' "$OUT/selftest_raw.txt" | tail -n +2 > "$OUT/sample_real_${REAL_S}s.tsv"
rm -f "$OUT/selftest_raw.txt"
tail -n 1 "$OUT/selftest.txt"

echo "== 2. syntax"
# shellcheck disable=SC2086
tar -cf - $SRC | $CHECK_SSH 'd=$(mktemp -d /tmp/rwt.XXXXXX); tar -xf - -C "$d"; cd "$d"; \
  for f in $(find . -name "*.sh"); do bash -n "$f" && echo "bash -n ok: $f"; done; \
  for f in $(find . -name "*.py"); do python3 -I -m py_compile "$f" && echo "py_compile ok: $f"; done; cd /; rm -rf "$d"' \
  | sed 's/ \.\// /' | tee "$OUT/syntax.txt"

echo "== 3. the first-turn request"
BUILD_SSH="$CHECK_SSH" bash bench/agent_check/first_turn_request.sh "$OUT/long_request.json" 2> "$OUT/request_summary.txt" \
  && cat "$OUT/request_summary.txt"
if [ -n "${VOCAB_MODEL:-}" ] && [ -n "${LLAMA_TOKENIZE:-}" ] && [ -s "$OUT/long_request.json" ]; then
  st=$(mktemp -d "$OUT/stage.XXXXXX")
  cp "$HERE/count_tokens.py" "$OUT/long_request.json" "$st/"
  tar -C "$st" -cf - . | $CHECK_SSH "d=\$(mktemp -d /tmp/rwt.XXXXXX); tar -xf - -C \"\$d\"; \
    VOCAB_MODEL='$VOCAB_MODEL' LLAMA_TOKENIZE='$LLAMA_TOKENIZE' python3 -I \"\$d/count_tokens.py\" \"\$d/long_request.json\"; rm -rf \"\$d\"" \
    > "$OUT/token_count.tsv" 2>&1
  rm -rf "$st"
fi
rm -f "$OUT/long_request.json"

echo "== 4. the smoke scripts' dry runs"
D1=docs/data/2026-11-03_flash_next_dflash2
D2=docs/data/2026-11-03_flash_next_mtp
BUILD_SSH="$CHECK_SSH" RIG=rig BENCH=bench/agent_check REL_BIN=/path/llama-server MODEL=/path/model.gguf RIG_PORT=8092 \
  ARGS="-b 2048 -ub 512" SMOKE_LOGDIR=/path/smoke REQUESTS="$D1/run/smoke_request.json $D1/run/smoke_greedy.json" DRY_RUN=1 \
  bash $D1/run/smoke_dflash.sh "$OUT/dry_dflash" > "$OUT/smoke_dry.txt" 2>&1
echo "smoke_dflash.sh dry run exit status $?" >> "$OUT/smoke_dry.txt"
BUILD_SSH="$CHECK_SSH" RIG=rig BENCH=bench/agent_check REL_BIN=/path/llama-server MODEL=/path/model.gguf RIG_PORT=8092 \
  ARGS_mtp="-b 2048 -ub 512" SMOKE_LOGDIR=/path/smoke REQUEST=$D2/run/smoke_request.json DRY_RUN=1 \
  bash $D2/run/smoke_mtp.sh "$OUT/dry_mtp" >> "$OUT/smoke_dry.txt" 2>&1
echo "smoke_mtp.sh dry run exit status $?" >> "$OUT/smoke_dry.txt"
cmp -s "$OUT/dry_dflash/long_request.json" "$OUT/dry_mtp/long_request.json" \
  && echo "both scripts build the same request" >> "$OUT/smoke_dry.txt"
rm -rf "$OUT/dry_dflash" "$OUT/dry_mtp"
sed -i "s#$OUT#<OUT>#g; s/^[0-9:]* //" "$OUT/smoke_dry.txt"
cut -c1-200 "$OUT/smoke_dry.txt"
exit "$rc"
