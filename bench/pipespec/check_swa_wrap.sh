#!/usr/bin/env bash
# Regression check: pipelined speculation on a model with sliding-window layers once the window's KV cache wraps.
#
# The window cache is a ring of n_swa + n_ubatch cells (padded to 256). Plain decoding fills it in ring order, and the
# executor keeps the same layout by re-decoding a dropped tail into the cells the tail had (seq_rm leaves the cache head
# on the first freed cell). Until fork commit fbb1b24f3, llama.cpp's slot search restarted from cell 0 whenever head >
# used + 2 × n_tokens. With D ≥ 4 dropped guesses, their freed cells count as unused, so that fired whenever head
# stood on one of the last cells of the ring. The re-decoded token then went to cell 0, attention summed the keys in
# another order, and the output stopped being bit-identical to plain decoding (K2h-r16 on the rig, edit prompt, depths
# 4–8).
#
# Here a small -ub makes the window cache small, so that it wraps within the first generated tokens, and an oracle that
# is always wrong drops D tokens at every step, so that the head reaches every cell of the ring with D freed cells
# behind it. Every pipelined run must equal plain decoding (check_identity.sh's comparison).
# -ub must still be at least the executor's slots (2 × (D + 1) for the deepest job: 18 at D = 8), which a sliding-window
# model needs (llama_pipe_init refuses otherwise); -ub 32 gives a window cache of 768 cells for Flint's 513-token window.
#
# Usage: BIN=<dir with llama-pipespec> MODEL=<gguf with sliding-window layers, e.g. Flint>
#   [ARGS='-ngl 99 -fa on -c 4096 -ub 32'] [DEPTHS='4 8'] [MISSES='1 0.5'] [N=128] bench/pipespec/check_swa_wrap.sh
# The prompt (prompts/rewrite.txt, ~750 tokens with Flint's template) plus N must exceed the window cache's size,
# which the script computes from what llama-pipespec reports (fork commit 74f7c7182 or later); it stops with an error
# otherwise. Other check_identity.sh variables pass through.
set -uo pipefail
: "${BIN:?set BIN}" "${MODEL:?set MODEL}"
HERE=$(cd "$(dirname "$0")" && pwd)
export ARGS=${ARGS:--ngl 99 -fa on -c 4096 -ub 32}
export DEPTHS=${DEPTHS:-4 8}
export MISSES=${MISSES:-1 0.5}
export N=${N:-128}
export OUT=${OUT:-/tmp/pipespec_swa_wrap}

bash "$HERE/check_identity.sh"
rc=$?

# llama-pipespec prints "window cache: n_swa = W, n_ubatch = U, n_ctx = C, ..."; one sequence: ring of W + U cells,
# at most the context, padded to 256
read -r n_swa n_ubatch n_ctx <<< "$(grep -o "window cache: n_swa = [0-9]*, n_ubatch = [0-9]*, n_ctx = [0-9]*" "$OUT/run.log" \
    | head -1 | grep -o "[0-9]*" | tr '\n' ' ')"
n_prompt=$(grep -o "name=plain .*" "$OUT/results.txt" | grep -o " n_prompt=[0-9]*" | cut -d= -f2)
if [ -z "${n_swa:-}" ] || [ "$n_swa" -le 0 ]; then
    echo "the model has no sliding-window layers (or the build does not report them): this check needs them"
    exit 1
fi
swa=$((n_swa + n_ubatch)); [ "$swa" -gt "$n_ctx" ] && swa=$n_ctx
swa=$(( (swa + 255) / 256 * 256 ))
echo "window cache: $swa cells; prompt $n_prompt + $N generated tokens"
if [ -n "$n_prompt" ] && [ $((n_prompt + N)) -le "$swa" ]; then
    echo "the generation does not wrap the window cache: raise N or lower -ub"
    exit 1
fi
exit $rc
