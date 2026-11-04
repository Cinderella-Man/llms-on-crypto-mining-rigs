# Decoding check of the IQ2_XXS, IQ2_XS and IQ3_S kernels (no GPU)

[bench/mmvq_pascal/verify/iq_unit_check.sh](../../../bench/mmvq_pascal/verify/iq_unit_check.sh) on the branch's tree
(`pascal-p106-v3-b12076-iq2xs` `314b7d1e3`; the later commits on the branch change tests only), 200 trials per type,
on the CPU of a build machine in the CUDA 12.9 container: `ggml/src/ggml-cuda/mmvq-pascal-iq.cuh` built for the host,
ggml's dequantization (`ggml_get_type_traits(type)->to_float`) from the same tree's `libggml-base`. Per trial: 4 random
super-blocks (FP16 scales ±[1, 2) · 2⁻⁸), Q8_1 activations from normal values of scale 1.5 (every fourth trial 3000),
one all-zero activation block; each block read as the kernels read it from shared memory, once from an even and once
from an odd half-word, with garbage around it. Tolerance: 1e-5 of the sum of the absolute terms of a double-precision
product.

| Type | Cases | Worst error / tolerance | Result |
|---|---|---|---|
| IQ2_XXS | 400 | 0.0016 | PASS |
| IQ2_XS | 400 | 0.0014 | PASS |
| IQ3_S | 400 | 0.0013 | PASS |

**Mutation** (`MUTATE=1`: a copy of the header with IQ2_XXS's scale read one bit too low, `(aux >> 27) & 0xF` for
`aux >> 28`, and IQ3_S's second grid index of each pair given the first one's ninth bit): IQ2_XXS and IQ3_S fail
(worst 8,982 and 6,904 times the tolerance) and IQ2_XS passes (worst 0.0014), as it must.

What it covers: the bit layout of each block, the grids, the signs (IQ2_XXS's and IQ2_XS's parity bit, IQ3_S's sign
bytes), the scales and the word reads at both half-word alignments. What it leaves to the GPU runs: the staging, the
lane groups, the reductions, the expert and fusion paths (test-backend-ops, [run/gpu_checks.sh](run/gpu_checks.sh)).
