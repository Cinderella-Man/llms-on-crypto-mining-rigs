#!/usr/bin/env bash
# Q56 (2026-10-23) after the head dump (quants/gptq_head/run.sh step 1, on F3 with the rescale off; x-wiki-cal.f32,
# x-wiki-val.f32, x-q51.f32): dump check, Hessians, the candidate Q5_0 heads, offline check on held-out wikitext rows.
# head-rtn.q5_0 is run.sh's round-to-nearest head. Env: WORK (the dumps and the outputs), BF16, F3 (flint-BF16.gguf,
# flint-q73-F3.gguf); numpy and gguf-py (PYTHONPATH=<llama.cpp>/gguf-py).
set -euo pipefail
T=$(cd "$(dirname "$0")/../../../../quants/gptq_head" && pwd); cd "$WORK"
G="python $T/gptq_head.py"
BF=$BF16

# the dump must reproduce F3's own logits through F3's head before anything is built on it
$G checkdump $F3 x-wiki-val.f32
$G checkdump $F3 x-q51.f32
$G hessian H-wiki.npy x-wiki-cal.f32
$G hessian H-q51.npy x-q51.f32
$G hessian H-mix.npy x-wiki-cal.f32 x-q51.f32
q() { echo "== $1"; $G quant $BF head-$1.q5_0 "${@:2}" 2>&1 | grep -v chunks; }
q imat --mode imat --hessian H-mix.npy
q gptq-max --mode gptq --hessian H-mix.npy --scale max
q gptq-search --mode gptq --hessian H-mix.npy --scale search
q gptq-search-ao --mode gptq --hessian H-mix.npy --scale search --act-order
q gptq-search-r1 --mode gptq --hessian H-mix.npy --scale search --refit 1
q gptq-search-wiki --mode gptq --hessian H-wiki.npy --scale search
q gptq-search-q51 --mode gptq --hessian H-q51.npy --scale search
$G proxy $BF x-wiki-val.f32 bf16 q8_0:$F3 head-rtn.q5_0 head-imat.q5_0 head-gptq-max.q5_0 \
  head-gptq-search.q5_0 head-gptq-search-ao.q5_0 head-gptq-search-r1.q5_0 head-gptq-search-wiki.q5_0 head-gptq-search-q51.q5_0
echo AFTER_DUMP_DONE
