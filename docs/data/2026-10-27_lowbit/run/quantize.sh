#!/usr/bin/env bash
# Ornith-1.5-122B in N2P/N3P mixes, requantized from ornith-ai's Q8_0 GGUF with Unsloth's importance matrix, on the
# quality machine's CPU (Docker, --cpu-shares 256, nice -n 19, 8 threads) with the llama-quantize of run/build.sh KIND=cpu.
# Files c and b: routed experts' gate and up N2P (2.6875 bpw), down N3P (3.25 bpw); file d swaps them. --pure, so no
# per-tensor heuristics; the router, norms, convolution and SSM vectors stay F32 as in the source.
#   FILE=c  Ornith-1.5-122B-N2P-N3P-Q5_0dense.gguf: every other tensor Q5_0, except attention k/v, the output head and
#           the token embedding at Q8_0 (~45.5 GB on the cards without the embedding and the MTP block)
#   FILE=b  Ornith-1.5-122B-N2P-N3P-Q8_0dense.gguf: every other tensor Q8_0, as in AD-IQ3_XXS-IQ2_S, so that a KLD
#           comparison with that file isolates the expert formats
#   FILE=d  Ornith-1.5-122B-N3P-N2P-Q5_0dense.gguf: file c with gate/up N3P and down N2P, the other per-tensor choice
#           (a donor of tensors for mixes that pick a format per tensor and layer group)
#   QM=<ssh alias> WORK=<work dir on QM, as for build.sh> MODELS=<folder on QM holding the ornith-ai__Ornith-1.5-122B-GGUF
#     and unsloth__Ornith-1.5-122B-GGUF folders> OUT=<output folder on QM> FILE=c|b|d bash quantize.sh
set -euo pipefail
QM=${QM:?ssh alias of the quality machine}
WORK=${WORK:?work dir on QM}
MODELS=${MODELS:?model folder on QM}
OUT=${OUT:?output folder on QM}
FILE=${FILE:?c, b or d}
GU=n2p; DN=n3p
case "$FILE" in
    c) BASE=Q5_0; NAME=Ornith-1.5-122B-N2P-N3P-Q5_0dense.gguf
       EXTRA="--output-tensor-type q8_0 --token-embedding-type q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0" ;;
    b) BASE=Q8_0; NAME=Ornith-1.5-122B-N2P-N3P-Q8_0dense.gguf; EXTRA="" ;;
    d) BASE=Q5_0; NAME=Ornith-1.5-122B-N3P-N2P-Q5_0dense.gguf; GU=n3p; DN=n2p
       EXTRA="--output-tensor-type q8_0 --token-embedding-type q8_0 --tensor-type attn_k=q8_0 --tensor-type attn_v=q8_0" ;;
    *) echo "FILE must be c, b or d" >&2; exit 1 ;;
esac
ssh "$QM" "mkdir -p $OUT && docker run -d --rm --name pq-lowbit-quant-$FILE --label pascal-quants=1 --cpu-shares 256 \
    -e LD_LIBRARY_PATH=/w/build-n-cpu/bin -v $WORK:/w:ro -v $MODELS:$MODELS:ro -v $OUT:$OUT nvidia/cuda:12.9.2-devel-ubuntu24.04 \
    bash -c 'nice -n 19 /w/build-n-cpu/bin/llama-quantize --allow-requantize --pure \
        --imatrix $MODELS/unsloth__Ornith-1.5-122B-GGUF/imatrix_unsloth.gguf_file $EXTRA \
        --tensor-type ffn_gate_exps=$GU --tensor-type ffn_up_exps=$GU --tensor-type ffn_down_exps=$DN \
        $MODELS/ornith-ai__Ornith-1.5-122B-GGUF/Ornith-1.5-122B-Q8_0.gguf $OUT/$NAME $BASE 8 > $OUT/quant-$FILE.log 2>&1;
        echo exit \$? >> $OUT/quant-$FILE.log'"
echo "started: $OUT/$NAME (log $OUT/quant-$FILE.log)"
