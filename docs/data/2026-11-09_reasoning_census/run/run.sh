#!/usr/bin/env bash
# The reasoning census (Q1152), on a CPU at low priority. No GPU. Run from a checkout of this repository: it uses
# the loop census's reader, arm list and token cache (docs/data/2026-11-09_loop_census/run) and Q94's counter
# (bench/reasoning_length/reflections.py).
#   IN         the agent check's result folders (proxy.jsonl, proxy.jsonl.bodies.gz, res/*/*/a*/result.json)
#   WORK       scratch: the loop census's token cache (filled here when it is empty)
#   OUT        the public tables (counts, task ids, single words)
#   PRIVATE    one row per reply
#   TOKENIZER  llama.cpp's llama-tokenize, a CPU build
#   VOCAB_FLASH_NEXT, VOCAB_FLINT, VOCAB_ORNITH   a GGUF file of each model that the tokenizer's build can open
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/../../../.." && pwd)
LOOP=$REPO/docs/data/2026-11-09_loop_census
REFLECTIONS=$REPO/bench/reasoning_length/reflections.py
: "${IN:?}" "${WORK:?}" "${OUT:?}" "${PRIVATE:?}" "${TOKENIZER:?}"
: "${VOCAB_FLASH_NEXT:?}" "${VOCAB_FLINT:?}" "${VOCAB_ORNITH:?}"
VOCABS=(--vocab "flash_next=$VOCAB_FLASH_NEXT" --vocab "flint=$VOCAB_FLINT" --vocab "ornith=$VOCAB_ORNITH")
PY="nice -n 19 python3 -I"
mkdir -p "$WORK" "$OUT" "$PRIVATE"

$PY "$HERE/reasoning_census.py" selftest
# Q94's own counts from the counter as it is now: Flint's 2.4 per 1,000 on Q51's transcripts, to the last digit
$PY "$REFLECTIONS" "$REPO/release/flint-v2/reproduce/data/q51/gen.jsonl" "$VOCAB_FLINT" --out "$WORK/q94.json" \
    | sed -n '3,7p'
cmp "$WORK/q94.json" "$REPO/docs/data/2026-10-24_q94/q94.json" && echo "Q94's counts: identical"
$PY "$LOOP/run/loop_census.py" tokenize --in "$IN" --arms "$LOOP/run/arms.tsv" --work "$WORK" \
    --tokenizer "$TOKENIZER" "${VOCABS[@]}" -j "${JOBS:-8}"
$PY "$HERE/reasoning_census.py" census --in "$IN" --work "$WORK" --tokenizer "$TOKENIZER" "${VOCABS[@]}" \
    --out "$OUT" --private "$PRIVATE" --loop-by-model "$LOOP/by_model.tsv"
