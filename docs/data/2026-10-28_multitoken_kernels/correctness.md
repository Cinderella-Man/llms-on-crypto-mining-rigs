# Correctness of the 2–4-token kernels (fork branch pascal-p106-v3-mtpk, build `f0f2be53b`)

`run/correctness.sh` on the quality machine, 2026-10-28: test-backend-ops against the CPU with the Pascal kernels forced on
(`GGML_CUDA_MMVQ_PASCAL=2`), and `multitoken_bitid` ([bench/mmvq_pascal/verify](../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp)),
which compares every 2–4-token product with the single-token products of its columns, bit for bit.

- **Native:** a V100 (sm_70 code of the build).
- **PTX:** an RTX 3080 through the build's compute_61 PTX, the code a P106 runs; 1–8 tokens, since larger batches take
  MMQ, whose tile configuration the host picks for the card it sees.

Settings: **MT** = `mul_mat_vec_q_pascal_mt` at its defaults; **MT tiles** = `GGML_CUDA_MMVQ_PASCAL_MT_ACT_KB=1`
`_MT_RB=3` (activations staged in tiles of one iteration where a row needs more than 1 KB, three row blocks per block
where it fits), so both loops run on every shape; **MC** = the earlier multi-token kernel (`GGML_CUDA_MMVQ_PASCAL_MT=0`).

Cases passed, test-backend-ops (MUL_MAT 1–8 tokens incl. K = 8192 and an odd row count; MUL_MAT_ID with 8 of 32 experts,
8 of 8 and 8 of 10; the fused gate/up; the shared expert in the routed launch):

| Type | MUL_MAT native / PTX | MUL_MAT_ID native / PTX | fused gate/up native / PTX | shared expert native / PTX |
|---|---|---|---|---|
| IQ2_S | 73/73 / 55/55 | 31/31 / 28/28 | 13/13 / 13/13 | 3/3 / 3/3 |
| IQ3_XXS | 73/73 / 55/55 | 31/31 / 28/28 | 11/11 / 11/11 | 3/3 / 3/3 |
| N2P | 40/40 / 35/35 | 33/33 / 28/28 | 11/11 / 11/11 | 3/3 / 3/3 |
| N3P | 40/40 / 35/35 | 33/33 / 28/28 | 11/11 / 11/11 | 3/3 / 3/3 |
| Q8_0 | 158/158 / 113/113 | 115/115 / — | 124/124 / — | 9/9 / — |
| Q5_0 | 108/108 / 85/85 | 40/40 / 28/28 | 11/11 / 11/11 | 3/3 / 3/3 |

The same counts pass with MT tiles (MUL_MAT, MUL_MAT_ID; native and PTX, Q8_0's MUL_MAT_ID through the PTX aside, as
below) and with MC (native). The Flint types' 2–4-token
cases (Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, IQ4_XS), which keep the earlier kernel, pass natively and through the PTX.

Q8_0's MUL_MAT_ID, fused and shared cases through the PTX ended without a summary: at 5–8 tokens they reach llama.cpp's
few-token MoE kernel, whose launch fails when compute 6.1 code runs on a newer card (the host allows 8 tokens for its
architecture, the compute 6.1 code's launch bounds 4 for Q8_0; the same in builds without these kernels). Their 2–4-token
products are covered by the bit-identity check below, also through the PTX.

Bit identity, every 2–4-token product against its columns' single-token products (`multitoken_bitid ornith`: 6 types,
5 dense shapes, down and fused gate/up at three routings; `flint`: the earlier harness's 7 types at Flint's shapes):

| | MT native | MT tiles native | MC native | MT PTX | MT tiles PTX |
|---|---|---|---|---|---|
| ornith, 198 comparisons | 0 differ | 0 differ | 0 differ | 0 differ | 0 differ |
| flint, 147 comparisons | 0 differ (MT for Q8_0 and Q5_0) | | | 0 differ | |

compute-sanitizer memcheck, native (N2P down at 4 tokens on 8 of 10 experts, IQ2_S down at 3 tokens on 8 of 8, N3P fused
gate/up on 8 of 10, Q5_0 dense K = 8192 at 4 tokens, IQ3_XXS shared expert): 0 errors.
