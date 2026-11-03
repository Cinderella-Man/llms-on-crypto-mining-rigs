#!/usr/bin/env bash
# End-to-end correctness of the two Q8_1 quantizers that follow upstream's block sum (#31247) on a real Flint file, on one
# NVIDIA GPU through the rig build's compute_61 PTX. llama-perplexity -ub 1 makes every token a one-token step; -ngl puts
# the last NGL layers on the GPU, so a GPU with a few GB free is enough. No timings.
#   1. the persistent MoE kernel's self-check (GGML_CUDA_PERSISTENT_MOE_CHECK=1): per graph evaluation, layers run,
#      compared with the per-op path and differing
#   2. the logits with the persistent kernel off and on, byte for byte (Pascal matrix-vector kernels forced on)
#   3. the logits with the norm's Q8_1 copy (GGML_CUDA_NORM_Q8) off and on, byte for byte, with the stock matrix-vector
#      kernels, which read the Q8_1 block sum (Q4_0, Q4_1, Q5_0): equal only if the norm's blocks are bit-identical to
#      quantize_q8_1's
# Steps 1 and 2 drop the MoE down-projection rescale at one token (LLAMA_NO_MOE_DOWN_RESCALE=tg, as the rig runs), which
# the persistent kernel needs. MODEL: a Flint file with Q5_0/Q8_0 or Q5_K/IQ4_XS expert layers (the kernel's types).
# Steps can run as separate calls (STEPS="2" on one GPU, STEPS="3" on another); each writes its own logs.
# Usage: BIN=<unpacked rig tarball> MODEL=<gguf> TEXT=<wikitext-2 wiki.test.raw> WORK=<dir> [GPU=n] [NGL=8] [CHUNKS=2]
#        [STEPS="1 2 3"] e2e.sh
set -uo pipefail
BIN=$(realpath "${BIN:?BIN=<unpacked rig tarball>}")
MODEL=$(realpath "${MODEL:?MODEL=<gguf>}")
TEXT=$(realpath "${TEXT:?TEXT=<wiki.test.raw>}")
WORK=${WORK:?WORK=<work dir>}
STEPS=${STEPS:-1 2 3}
K=$WORK/e2e
mkdir -p "$K" "$WORK/nvcache"

dock() { # <space-separated VAR=value list or -> <log> <binary> <args...>
    local envs=() e
    [ "$1" != - ] && for e in $1; do envs+=(-e "$e"); done
    local log=$2
    shift 2
    docker run --rm --name "pq-e2e-$log" --cpu-shares=256 --label pascal-quants=1 ${DOCKER_ARGS:-} --gpus "\"device=${GPU:-0}\"" \
        -v "$BIN:$BIN:ro" -v "$K:$K" -v "$(dirname "$MODEL"):$(dirname "$MODEL"):ro" -v "$(dirname "$TEXT"):$(dirname "$TEXT"):ro" \
        -v "$WORK/nvcache:$WORK/nvcache" -e CUDA_CACHE_PATH="$WORK/nvcache" -e CUDA_CACHE_MAXSIZE=4294967296 \
        "${envs[@]}" nvidia/cuda:12.9.2-devel-ubuntu24.04 "$@" > "$K/$log.log" 2>&1
    echo "exit $?" >> "$K/$log.log"
}

same() { # tag-a tag-b
    if cmp -s "$K/$1.kld" "$K/$2.kld"; then
        echo "logits: $2 identical to $1 ($(stat -c %s "$K/$1.kld") bytes)"
    else
        echo "logits: $2 DIFFERS from $1"
    fi
    rm -f "$K/$1.kld" "$K/$2.kld"
}

PPL="$BIN/bin/llama-perplexity -m $MODEL -f $TEXT -c 128 --chunks ${CHUNKS:-2} -b 128 -ub 1 -ngl ${NGL:-8} -fa on -t 8 -lv 4"
PASCAL="GGML_CUDA_MMVQ_PASCAL=2 LLAMA_NO_MOE_DOWN_RESCALE=tg"
STOCK="GGML_CUDA_MMVQ_PASCAL=0 GGML_CUDA_PERSISTENT_MOE=0"

case " $STEPS " in *" 1 "*)
    echo "=== 1. persistent MoE kernel self-check (count: device, layers run, compared, differed)"
    dock "$PASCAL GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1" check $PPL
    grep -a -oE "CUDA[0-9]+: persistent MoE kernel ran [0-9]+ layers, [0-9]+ compared with the per-op path, [0-9]+ differed \(.*\)" "$K/check.log" \
        | sed -E "s/persistent MoE kernel ran //; s/ compared with the per-op path,//" | sort | uniq -c
    grep -a -oE "Final estimate.*" "$K/check.log"; tail -n 1 "$K/check.log"
esac

case " $STEPS " in *" 2 "*)
    echo "=== 2. persistent MoE kernel off and on"
    dock "$PASCAL GGML_CUDA_PERSISTENT_MOE=0" moe-off $PPL --kl-divergence-base "$K/moe-off.kld"
    dock "$PASCAL GGML_CUDA_PERSISTENT_MOE=2" moe-on  $PPL --kl-divergence-base "$K/moe-on.kld"
    for t in moe-off moe-on; do echo "$t: $(grep -a -oE 'Final estimate.*' "$K/$t.log") $(tail -n 1 "$K/$t.log")"; done
    echo "moe-on: $(grep -a -c 'run as one persistent kernel' "$K/moe-on.log") device(s) ran the persistent kernel; moe-off: $(grep -a -c 'run as one persistent kernel' "$K/moe-off.log")"
    same moe-off moe-on
esac

case " $STEPS " in *" 3 "*)
    echo "=== 3. the norm's Q8_1 copy off and on, stock matrix-vector kernels"
    dock "$STOCK GGML_CUDA_NORM_Q8=0" norm-off $PPL --kl-divergence-base "$K/norm-off.kld"
    dock "$STOCK GGML_CUDA_NORM_Q8=1 GGML_CUDA_NORM_Q8_DEBUG=1" norm-on $PPL --kl-divergence-base "$K/norm-on.kld"
    for t in norm-off norm-on; do echo "$t: $(grep -a -oE 'Final estimate.*' "$K/$t.log") $(tail -n 1 "$K/$t.log")"; done
    echo "norm-on, per graph evaluation (count: norms that wrote a Q8_1 row, products that read one):"
    grep -a -oE "[0-9]+ norms wrote a Q8_1 row \([0-9]+ with the residual add\), [0-9]+ single-token products read one, [0-9]+ quantized" "$K/norm-on.log" | sort | uniq -c | sort -rn | head -3
    same norm-off norm-on
esac
