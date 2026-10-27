#!/usr/bin/env bash
# Q56 (2026-10-24) domain check: does a head calibrated on wikitext hold up on Flint's own transcripts (chat, code,
# edit, agent)? 8 of the 38 Q51 transcripts (2 chat, 2 code, 2 edit, 2 agent: ids 8, 9, 18, 19, 26, 27, 33, 37) are
# held out of every H used here. Runs after after_dump.sh. Env: WORK (the head dumps x-*.f32 and the outputs), BF16, F3
# (flint-BF16.gguf, flint-q73-F3.gguf); numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
set -euo pipefail
T=$(cd "$(dirname "$0")/../../../../quants/gptq_head" && pwd); cd "$WORK"
G="python $T/gptq_head.py"
BF=$BF16

HO=8,9,18,19,26,27,33,37
$G hessian H-q51a.npy "x-q51.f32@!$HO"
$G hessian H-mixa.npy x-wiki-cal.f32 "x-q51.f32@!$HO"
q() { echo "== $1"; $G quant $BF head-$1.q5_0 "${@:2}" 2>&1 | grep -v chunks; }
q imat-mixa --mode imat --hessian H-mixa.npy
q gptq-search-q51a --mode gptq --hessian H-q51a.npy --scale search
q gptq-search-mixa --mode gptq --hessian H-mixa.npy --scale search
# held-out transcripts: every row at position >= 256 of the 8 held-out sequences
$G proxy $BF "x-q51.f32@$HO" --rows 100000 q8_0:$F3 head-rtn.q5_0 head-imat-mixa.q5_0 head-gptq-max.q5_0 \
  head-gptq-search-wiki.q5_0 head-gptq-search-q51a.q5_0 head-gptq-search-mixa.q5_0
# the new heads on the held-out wikitext rows too
$G proxy $BF x-wiki-val.f32 head-imat-mixa.q5_0 head-gptq-search-q51a.q5_0 head-gptq-search-mixa.q5_0
echo HOLDOUT_DONE
