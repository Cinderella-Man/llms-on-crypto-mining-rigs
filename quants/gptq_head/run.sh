#!/usr/bin/env bash
# GPTQ-rounded Q5_0 output head for a Flint quant, end to end. Run from a llama.cpp tree that has head_dump/ built as
# examples/head-dump (see head_dump/CMakeLists.txt), with numpy and gguf-py available to Python.
#
#   MODEL=flint-q73-F3.gguf BF16=flint-BF16.gguf TRAIN=wiki.train.raw OUT=work/ [SEQS=seqs.txt] [GPU env] run.sh
#
# 1. head inputs of MODEL (the quant whose head is replaced) on 500 chunks of 512 tokens of TRAIN for H, and on 40
#    later chunks for the held-out check; optionally on SEQS (llama-head-dump's sequence format, e.g. Flint's own
#    transcripts) as a second calibration set;
# 2. H = mean of x x^T over those rows;
# 3. the head of BF16 in Q5_0: round-to-nearest (what llama-quantize does) and GPTQ;
# 4. the head alone on held-out rows: KLD and same top token against the BF16 head;
# 5. MODEL with the GPTQ head, as one file.
# Score the result with llama-perplexity --kl-divergence against the BF16 logits, like any other quant.
set -euo pipefail
: "${MODEL:?}" "${BF16:?}" "${TRAIN:?}" "${OUT:?}"
BIN=${BIN:-build-cuda/bin}
HERE=$(cd "$(dirname "$0")" && pwd)
RUN="${LAUNCH:-} $BIN/llama-head-dump -m $MODEL -b 512 -ngl 99 -fa on ${TS:+-ts $TS}"
export LLAMA_NO_MOE_DOWN_RESCALE=${LLAMA_NO_MOE_DOWN_RESCALE:-1}   # the rescale-free Flint files run without it
mkdir -p "$OUT"

HEAD_OUT=$OUT/x-cal.f32 HEAD_TEXT=$TRAIN HEAD_CHUNK0=0 HEAD_CHUNKS=500 $RUN -c 512
HEAD_OUT=$OUT/x-val.f32 HEAD_TEXT=$TRAIN HEAD_CHUNK0=2000 HEAD_CHUNKS=40 $RUN -c 512
CAL=$OUT/x-cal.f32
if [ -n "${SEQS:-}" ]; then
    HEAD_OUT=$OUT/x-seqs.f32 HEAD_SEQS=$SEQS $RUN -c 16384
    CAL="$CAL $OUT/x-seqs.f32"
fi

python3 "$HERE/gptq_head.py" hessian "$OUT/H.npy" $CAL
python3 "$HERE/gptq_head.py" quant "$BF16" "$OUT/head-rtn.q5_0" --mode rtn
python3 "$HERE/gptq_head.py" quant "$BF16" "$OUT/head-gptq.q5_0" --mode gptq --hessian "$OUT/H.npy" ${GPTQ_ARGS:-}
python3 "$HERE/gptq_head.py" proxy "$BF16" "$OUT/x-val.f32" "q8_0:$MODEL" "$OUT/head-rtn.q5_0" "$OUT/head-gptq.q5_0"
python3 "$HERE/gptq_head.py" full "$MODEL" "$OUT/head-gptq.q5_0" "$OUT/$(basename "$MODEL" .gguf)-head-gptq-q5_0.gguf"
