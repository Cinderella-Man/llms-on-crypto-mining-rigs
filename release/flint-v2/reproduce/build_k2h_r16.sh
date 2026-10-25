#!/usr/bin/env bash
# build_k2h_r16.sh: rebuilds the K2h-r16 model file from Flint's original BF16 weights (release/flint-v2/reproduce/
# model.md). Every step is a public tool; the result is checked tensor by tensor against the shipped file's manifest.
#
#   LLAMA_DIR=<llama.cpp at d77256999, built> FLINT_DIR=<RecursiveSI/flint at 2a247be0c0ae> \
#   IMATRIX=<imatrix_unsloth.gguf> WORK=<dir with ~150 GB free> release/flint-v2/reproduce/build_k2h_r16.sh
#
# Steps (STEPS, default "convert quantize head router header verify"):
#   convert   Flint's safetensors -> BF16 GGUF (llama.cpp's convert_hf_to_gguf.py), 79.6 GB
#   quantize  llama-quantize from the BF16 GGUF with Unsloth's importance matrix, base Q5_0, and one --tensor-type per
#             tensor from k2h-r16.types (output head Q8_0 and the router F32 at this stage), ~31 GB
#   head      the GPTQ-rounded Q5_0 output head: with HEAD_FROM=<a K2h-r16 file> it is copied from there byte for byte;
#             otherwise it is computed (quants/gptq_head/run.sh: needs a CUDA GPU with ~40 GB in total, wikitext-2 train)
#   router    the 47 routers (ffn_gate_inp) F32 -> F16 (scripts/retype_tensors.py)
#   header    the release header (scripts/gguf_release_header.py --preset flint; NAME, QUANTIZED_BY)
#   verify    the tensors against k2h-r16.tensors.tsv: every tensor equal byte for byte; with a computed head, every
#             tensor but output.weight (then check the head with the KLD commands in model.md)
#
# Env [defaults]: LLAMA_DIR (llama-quantize in build/bin, convert_hf_to_gguf.py and gguf-py), FLINT_DIR, IMATRIX,
#   WORK, PYTHON [python3; needs numpy, and torch/safetensors for convert], THREADS [nproc], HEAD_FROM, NAME [Flint
#   K2h-r16], QUANTIZED_BY [llms-on-crypto-mining-rigs], TRAIN (wikitext-2 train text, for a computed head), STEPS,
#   BF16 [WORK/flint-BF16.gguf; point it at an existing conversion to skip convert], QUANTIZE [LLAMA_DIR/build/bin/
#   llama-quantize]
# Exit status: 0 done (verify: all compared tensors equal), 1 a step failed or tensors differ.
set -euo pipefail
here=$(cd "$(dirname "$0")" && pwd)
repo=$(cd "$here/../../.." && pwd)
: "${LLAMA_DIR:?LLAMA_DIR: a llama.cpp checkout at d77256999 with build/bin/llama-quantize}"
: "${WORK:?WORK: a directory with ~150 GB free}"
PYTHON=${PYTHON:-python3}
THREADS=${THREADS:-$(nproc)}
STEPS=${STEPS:-convert quantize head router header verify}
NAME=${NAME:-Flint K2h-r16}
QUANTIZED_BY=${QUANTIZED_BY:-llms-on-crypto-mining-rigs}
export PYTHONPATH="$LLAMA_DIR/gguf-py${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$WORK"
BF16=${BF16:-$WORK/flint-BF16.gguf}
QUANTIZE=${QUANTIZE:-$LLAMA_DIR/build/bin/llama-quantize}
BODY=$WORK/flint-k2-body.gguf
K2H=$WORK/flint-k2h.gguf
RAW=$WORK/flint-k2h-r16-raw.gguf
OUT=$WORK/flint-K2h-r16.gguf
has() { case " $STEPS " in *" $1 "*) return 0;; esac; return 1; }
log() { echo "$(date '+%F %T') $*"; }

if has convert; then
    : "${FLINT_DIR:?FLINT_DIR: RecursiveSI/flint, revision 2a247be0c0aec895ccb0ac7ca92229ec6df37288}"
    log "convert: $FLINT_DIR -> $BF16"
    "$PYTHON" "$LLAMA_DIR/convert_hf_to_gguf.py" "$FLINT_DIR" --outtype bf16 --outfile "$BF16"
fi

if has quantize; then
    : "${IMATRIX:?IMATRIX: imatrix_unsloth.gguf from unsloth/flint-GGUF (sha256 in inputs.sha256)}"
    log "quantize: $BF16 -> $BODY (Q5_0 base, $(grep -vc -E $'\t(F32|F16|BF16)$' "$here/k2h-r16.types") tensor types from k2h-r16.types)"
    mapfile -t targs < <("$PYTHON" "$here/tensor_manifest.py" quantize-args "$here/k2h-r16.types" --skip '^output\.weight$' \
        | sed 's/^--tensor-type /--tensor-type\n/' )
    "$QUANTIZE" --allow-requantize --output-tensor-type q8_0 "${targs[@]}" \
        --imatrix "$IMATRIX" "$BF16" "$BODY" Q5_0 "$THREADS"
fi

if has head; then
    if [ -n "${HEAD_FROM:-}" ]; then
        log "head: output.weight copied from $HEAD_FROM"
        "$PYTHON" "$repo/scripts/compose_gguf.py" "$BODY" "$K2H" "^output\.weight\$=$HEAD_FROM"
    else
        : "${TRAIN:?TRAIN: wikitext-2 train text (wiki.train.raw) to calibrate the head}"
        log "head: GPTQ Q5_0 from the body's own head inputs on wikitext-2 train (quants/gptq_head/run.sh)"
        MODEL=$BODY BF16=$BF16 TRAIN=$TRAIN OUT=$WORK/head GPTQ_ARGS="--scale search" BIN=${HEAD_BIN:-$LLAMA_DIR/build-cuda/bin} \
            bash "$repo/quants/gptq_head/run.sh"
        mv "$WORK/head/$(basename "$BODY" .gguf)-head-gptq-q5_0.gguf" "$K2H"
    fi
fi

if has router; then
    log "router: ffn_gate_inp -> F16"
    "$PYTHON" "$repo/scripts/retype_tensors.py" "$K2H" "$RAW" 'ffn_gate_inp=F16'
fi

if has header; then
    log "header: $NAME, quantized by $QUANTIZED_BY"
    "$PYTHON" "$repo/scripts/gguf_release_header.py" "$RAW" "$OUT" --preset flint --name "$NAME" --quantized-by "$QUANTIZED_BY"
fi

if has verify; then
    f=$OUT; [ -f "$f" ] || f=$RAW
    log "verify: the tensors of $f against k2h-r16.tensors.tsv"
    "$PYTHON" "$here/tensor_manifest.py" hash "$f" > "$WORK/rebuilt.tensors.tsv"
    ign=(); [ -z "${HEAD_FROM:-}" ] && ign=(--ignore '^output\.weight$')
    "$PYTHON" "$here/tensor_manifest.py" compare "$here/k2h-r16.tensors.tsv" "$WORK/rebuilt.tensors.tsv" "${ign[@]}"
fi
log "done"
