#!/usr/bin/env bash
# The run behind this folder's tables (2026-11-04). Nothing here uses a GPU or the rig; the machine that runs the
# analysis needs Python 3.10+ and a llama.cpp build with llama-tokenize that knows the `qwen35` pre-tokenizer.
#
#   RIG="ssh <analysis machine>"         the ssh command that reaches it
#   WORK=/path/on/that/machine           work folder
#   TOKENIZE=/path/to/llama-tokenize
#   MODEL=/path/to/model-00001-of-0000N.gguf   any GGUF of the model (only its vocabulary is read)
#   LOGS_*                               the folders holding each run's proxy.jsonl and proxy.jsonl.bodies.gz
#   ACC=/path/to/draft_acceptance.csv    docs/data/2026-11-03_flash_next_dflash2/agent/dflash2b/draft_acceptance.csv
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BENCH=$HERE/../../../../bench/copy_spans
: "${RIG:?}" "${WORK:?}" "${TOKENIZE:?}" "${MODEL:?}" "${ACC:?}"
: "${LOGS_BASELINE:?}" "${LOGS_RETYPE:?}" "${LOGS_DFLASH2:?}" "${LOGS_PIPELINED:?}"

# 1. the transcripts to the analysis machine (md5 sums printed on both sides)
RIG="$RIG" WORK="$WORK" "$BENCH/copy_in.sh" \
  flash_next_baseline="$LOGS_BASELINE" retype_agent="$LOGS_RETYPE" \
  flash_next_dflash2="$LOGS_DFLASH2" flash_next_pipelined="$LOGS_PIPELINED"
cat "$ACC" | $RIG "cat > '$WORK/in/draft_acceptance_dflash2b.csv'"
tar cf - -C "$BENCH" . | $RIG "mkdir -p '$WORK/bench/copy_spans' && tar xf - -C '$WORK/bench/copy_spans'"

# 2. the analysis (self-test, rebuild checks, tables, DFlash2 fit, second pass with the fitted figure)
$RIG "cd '$WORK' && WORK='$WORK' TOKENIZE='$TOKENIZE' MODEL='$MODEL' ACC='$WORK/in/draft_acceptance_dflash2b.csv' \
  FIT_RUN=flash_next_dflash2 FIT_ARM=dflash2b \
  RUNS='flash_next_baseline=$WORK/in/flash_next_baseline retype_agent=$WORK/in/retype_agent flash_next_dflash2=$WORK/in/flash_next_dflash2 flash_next_pipelined=$WORK/in/flash_next_pipelined' \
  bash bench/copy_spans/run.sh"

# 3. the tables back (counts only; the token files and text samples stay on the analysis machine)
mkdir -p "$HERE/../tables"
$RIG "cd '$WORK' && tar cf - tables prep/render_check.csv" | tar xf - -C "$HERE/.."
mv "$HERE/../prep/render_check.csv" "$HERE/../tables/" && rmdir "$HERE/../prep"
