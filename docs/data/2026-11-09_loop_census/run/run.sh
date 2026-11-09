#!/usr/bin/env bash
# The loop census (Q1138, Q1158), on a CPU at low priority. No GPU.
#   IN         the agent check's result folders (proxy.jsonl, proxy.jsonl.bodies.gz, res/*/*/a*/result.json)
#   WORK       scratch: the token cache
#   OUT        the public tables (counts and task ids)
#   PRIVATE    one row per reply
#   TOKENIZER  llama.cpp's llama-tokenize, a CPU build
#   VOCAB_FLASH_NEXT, VOCAB_FLINT, VOCAB_ORNITH   a GGUF file of each model that the tokenizer's build can open
#   SAME_FLASH_NEXT, SAME_ORNITH                  optional: the release file, when VOCAB_* is another file of the
#                                                 same model; the two files' tokenizers must be identical
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
: "${IN:?}" "${WORK:?}" "${OUT:?}" "${PRIVATE:?}" "${TOKENIZER:?}"
: "${VOCAB_FLASH_NEXT:?}" "${VOCAB_FLINT:?}" "${VOCAB_ORNITH:?}"
VOCABS=(--vocab "flash_next=$VOCAB_FLASH_NEXT" --vocab "flint=$VOCAB_FLINT" --vocab "ornith=$VOCAB_ORNITH")
PY="nice -n 19 python3 -I $HERE/loop_census.py"
mkdir -p "$WORK" "$OUT" "$PRIVATE"

if [ -n "${SAME_FLASH_NEXT:-}" ]; then $PY vocab "$VOCAB_FLASH_NEXT" "$SAME_FLASH_NEXT" | tail -n 1; fi
if [ -n "${SAME_ORNITH:-}" ]; then $PY vocab "$VOCAB_ORNITH" "$SAME_ORNITH" | tail -n 1; fi
$PY selftest --tokenizer "$TOKENIZER" "${VOCABS[@]}" --work "$WORK/selftest"
$PY tokenize --in "$IN" --arms "$HERE/arms.tsv" --work "$WORK" --tokenizer "$TOKENIZER" "${VOCABS[@]}" -j "${JOBS:-8}"
$PY census --in "$IN" --arms "$HERE/arms.tsv" --work "$WORK" --tokenizer "$TOKENIZER" "${VOCABS[@]}" \
    --out "$OUT" --private "$PRIVATE"
