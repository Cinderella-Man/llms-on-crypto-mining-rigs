#!/usr/bin/env bash
# The whole copy-span analysis on one machine: vocabulary file, self-test, tokens, tables, DFlash2 fit.
#
#   WORK=/path/to/work RUNS="baseline=/path/in/baseline retype=/path/in/retype dflash2=/path/in/dflash2" \
#   TOKENIZE=/path/to/llama-tokenize MODEL=/path/to/model-00001-of-0000N.gguf \
#   [ACC=/path/draft_acceptance.csv FIT_RUN=dflash2 FIT_ARM=dflash2b] ./run.sh
#
# Each RUNS folder holds an agent-check run's proxy.jsonl and proxy.jsonl.bodies.gz (copy_in.sh puts them there).
# llama-tokenize needs a build that knows the model's pre-tokenizer (qwen35 for Qwen3.8-Flash-Next); the model's own
# architecture need not be known: make_vocab_gguf.py writes a tokenizer-only GGUF. CPU only; nothing here uses a GPU.
set -euo pipefail
: "${WORK:?}" "${RUNS:?}" "${TOKENIZE:?}"
HERE=$(cd "$(dirname "$0")" && pwd)
VOCAB=${VOCAB:-$WORK/vocab.gguf}
if [ ! -f "$VOCAB" ]; then
  : "${MODEL:?MODEL (a GGUF of the model) or VOCAB (the tokenizer-only file) is needed}"
  python3 -I "$HERE/make_vocab_gguf.py" "$MODEL" "$VOCAB"
fi
export WORK RUNS TOKENIZE VOCAB CUDA_VISIBLE_DEVICES=
nice -n 19 python3 -I "$HERE/selftest.py"
nice -n 19 python3 -I "$HERE/prepare.py"
nice -n 19 python3 -I "$HERE/analyze.py"
if [ -n "${ACC:-}" ]; then
  TABLES=$WORK/tables RUN=${FIT_RUN:-dflash2} ARM=${FIT_ARM:-dflash2b} nice -n 19 python3 -I "$HERE/dflash_fit.py"
  TAU_OUT="$(cat "$WORK/tables/tau_out.txt"),2.56" nice -n 19 python3 -I "$HERE/analyze.py"
fi
echo "tables in $WORK/tables"
