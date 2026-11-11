# The shared expert at 2–4 tokens: its kernels, their loads, their sums (C731, Q1290, 2026-11-11)

What this is: a read of the code and the sm_61 machine code, with no GPU and no new build. In a four-token pass of
Miner Llama's Flash-Next preset, Qwen3.8-Flash-Next's shared expert takes 4.79 ms of GPU time against 2.56 ms at one
token ([2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). The night's read-once kernels for
the dense products ([C690](../2026-11-11_c690_head_four_rows/README.md),
[C722](../2026-11-11_c722_short_batch_products/README.md)) load each weight once for all tokens and repeat the one-token
kernel's sums. This note checks the same two properties for every product of the shared expert: which kernel takes it
at 1–4 tokens, how often it loads each weight, and whether its sums at 2–4 tokens are the one-token sums. Registry row
[Q1290](../../../quants/experiments.md).

**The answer:**
- **The shared expert is not one type.** The release file (Swift GSQ-RCO Q2_0) gives each of the 47 layers its own
  mix. Gate and up are Q2_0, Q3_K, IQ4_XS, Q4_K, Q5_K or Q6_K, and the two often differ. Down is Q2_0, Q4_0, Q5_0,
  IQ4_NL or Q8_0. Beside them sits the F32 gate `ffn_gate_inp_shexp` (2,560 → 1). That makes 17 kinds of product,
  listed below.
- **At 2–4 tokens every one of them already loads each weight once per call.** The quantized products run on the
  k-column kernel, or on the Q2_0 expert kernel's shared-expert block row (the 6 layers fused with the routed experts).
  The F32 gate runs on llama.cpp's `mul_mat_vec_f` at 2–3 tokens and on cuBLAS at 4. In the SASS of every loop the
  global loads are the same instructions at 1 to 8 tokens, while the shared-memory loads of the activations grow by a
  fixed count per token. So C690's and C722's method has no repeated weight read to remove here, and no new kernel was
  made.
- **Bit for bit at 2–4 tokens:** every Q2_0 product (34 gates and ups on 28 of the 47 layers, and 16 downs); Q3_K, except
  the 6 layers whose Q3_K gate and up are fused at one token; IQ4_NL; Q8_0; and the F32 gate at 2–3 tokens.
- **Not bit for bit:**
  - IQ4_XS and Q4_K (C722's `GGML_CUDA_MMVQ_PASCAL_DMT=1` makes them exact), Q5_K (C690's
    `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1`), Q6_K (C727);
  - those 6 Q3_K pairs;
  - the F32 gate at 4 tokens;
  - Q4_0 and Q5_0, where only blocks whose integer sum reaches 2^13 can differ (a CPU emulation: 0 of 10,240 Q4_0
    outputs, 4 and 79 of 10,240 Q5_0 outputs in two samples).
- **Where the four-token excess goes:** of the 2.24 ms, the largest single part is the F32 gate on cuBLAS (+0.448 ms:
  12.78 µs a call against 3.26). The rest is small products bound by latency, not by their weights: 20 to 60 µs a call
  at four tokens, against est. 3 to 6 µs to read the weights.

## The products and the kernels that take them

Under the preset (`release/flash-next-v3/settings.env`: `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`,
`GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`, none of the v4 line's switches), on the fork's branch
`pascal-p106-v4-b12246-c748` at `5b329df16`. Between it and the v3.0 build that R41 profiled (`1551f5fe1`) these paths
changed only by C690's and C722's hooks, which are off by default.

GPU time per pass summed over the nine cards, from R41's kept trace rows ([times.tsv](times.tsv), by
[run/shexp_times.sh](run/shexp_times.sh)). "Layers" counts the launches a pass; L is the lanes per row:

| Product | Layers | Type | Width → rows | Kernel at 1 token | Kernel at 2–4 tokens | Same sums at 2–4 | ms, 1 / 4 / 5 tokens |
|---|---:|---|---|---|---|---|---|
| gate + up, fused with the routed experts | 6 | Q2_0 | 2,560 → 640 | one-token Pascal kernel, routed + shared launch | Q2_0 expert kernel (`mtok`), routed + shared launch | yes | (in the routed launch) |
| gate | 15 | Q2_0 | 2,560 → 640 | one-token Pascal kernel, L = 4 | k-column, L = 4 | yes | 0.166 / 0.316 / 0.368 |
| up | 7 | Q2_0 | 2,560 → 640 | one-token Pascal kernel, L = 4 | k-column, L = 4 | yes | 0.080 / 0.136 / 0.166 |
| down | 16 | Q2_0 | 640 → 2,560 | one-token Pascal kernel, warp tiles, L = 1 | k-column, L = 1 | yes | 0.179 / 0.323 / 0.370 |
| gate + up of one layer, both Q3_K | 6 | Q3_K | 2,560 → 640 | llama.cpp's `mul_mat_vec_q`, fused gate/up + SWIGLU | k-column, apart, then SWIGLU | **no** | 0.262 / in the next two rows |
| gate | 6 | Q3_K | 2,560 → 640 | k-column, L = 2 | k-column, L = 2 | yes | 0.154 / 0.453 / 0.504 (with the pairs' 6) |
| up | 15 | Q3_K | 2,560 → 640 | k-column, L = 2 | k-column, L = 2 | yes | 0.394 / 0.742 / 0.841 (with the pairs' 6) |
| gate | 6 | IQ4_XS | 2,560 → 640 | one-token Pascal kernel | k-column | **no** (yes with `DMT=1`) | 0.117 / 0.228 / 0.255 |
| up | 5 | IQ4_XS | 2,560 → 640 | one-token Pascal kernel | k-column | **no** (yes with `DMT=1`) | 0.097 / 0.189 / 0.212 |
| gate, up (fused at one token on 2 layers) | 6 + 6 | Q4_K | 2,560 → 640 | one-token Pascal kernel | k-column | **no** (yes with `DMT=1`) | 0.181 / 0.405 / 0.448 |
| gate, up | 1 + 1 | Q5_K | 2,560 → 640 | one-token Pascal kernel | k-column | **no** (yes with `Q5K_MT=1`) | 0.034 / 0.082 / 0.089 |
| gate, up | 1 + 1 | Q6_K | 2,560 → 640 | one-token Pascal kernel | k-column | **no** (C727) | 0.049 / 0.122 / 0.082 |
| down | 16 | Q4_0 | 640 → 2,560 | one-token Pascal kernel, L = 2 | k-column, L = 2 | **no**: only past \|I\| ≥ 2^13 | 0.292 / 0.595 / 0.588 |
| down | 7 | Q5_0 | 640 → 2,560 | one-token Pascal kernel, L = 2 | k-column, L = 2 | **no**: only past \|I\| ≥ 2^13 | 0.134 / 0.285 / 0.287 |
| down | 7 | IQ4_NL | 640 → 2,560 | k-column, L = 2 | k-column, L = 2 | yes | 0.224 / 0.266 / 0.271 |
| down | 1 | Q8_0 | 640 → 2,560 | k-column, L = 2 | k-column, L = 2 | yes | 0.040 / 0.049 / 0.053 |
| `ffn_gate_inp_shexp` | 47 | F32 | 2,560 → 1 | llama.cpp's `mul_mat_vec_f` | `mul_mat_vec_f` at 2–3, **cuBLAS at 4–8** | yes at 2–3, **no** at 4 | 0.153 / 0.601 / 0.602 |
| **the shared expert part** (R41's category) | | | | | | | **2.556 / 4.792 / 5.130** |

The same table as data: [products.tsv](products.tsv). How each kernel is chosen:

- **Plain quantized products.** `ggml_cuda_mul_mat` hands them to `ggml_cuda_mul_mat_vec_q` up to 8 tokens, and that
  asks the k-column hook first (`mmvq-pascal-kcol.cu`). With `KCOL_TABLE=0` the k-column kernel takes every plain
  product of 2–8 tokens. At one token it takes only Q3_K, IQ4_NL, and Q8_0 rows whose 64-weight units are not a
  multiple of 4 (K = 640). The other one-token products go to the one-token Pascal kernel (`mul_mat_vec_q_pascal`).
- **Fusion at one token.** Gate, up and SWIGLU are fused where gate and up have the same type.
  - On 6 layers both are Q2_0, the routed experts' type, so they join the routed experts' launch
    (`ggml_cuda_match_shared_expert`).
  - On 6 layers both are Q3_K. The Pascal kernels have no fused one-token Q3_K path, so llama.cpp's `mul_mat_vec_q`
    takes the pair.
  - On 2 layers both are Q4_K, and the one-token Pascal kernel takes the pair with the same lanes as its plain Q4_K
    product.
- **Fusion at 2–8 tokens.** llama.cpp fuses several tokens only for `MUL_MAT_ID`. So the routed + shared launch stays
  fused (the `mtok` kernel), while the Q3_K and Q4_K pairs run gate and up apart, then a SWIGLU.
- **The F32 gate.** `ggml_cuda_should_use_mmvf` takes F32 weights only up to 3 tokens on NVIDIA cards before Turing.
  Nothing else takes F32 at 4–8 tokens on compute 6.1 (no `mul_mat_f` without tensor cores), so cuBLAS does. The
  router `ffn_gate_inp` (2,560 → 512, F32) follows the same rule: 2.35 ms of a four-token pass against 1.79 at one
  token (R41's router part).
- **Layers by type:**
  - gate: Q2_0 21 (6 of them fused with the routed experts), Q3_K 12, IQ4_XS 6, Q4_K 6, Q5_K 1, Q6_K 1;
  - up: Q2_0 13, Q3_K 21, IQ4_XS 5, Q4_K 6, Q5_K 1, Q6_K 1;
  - down: Q2_0 16, Q4_0 16, Q5_0 7, IQ4_NL 7, Q8_0 1.

## Loads per token, from the SASS

[sass_counts.tsv](sass_counts.tsv) is cuobjdump's sm_61 code of the build's objects, read by
[run/sass_check.sh](run/sass_check.sh), with the objects' md5 in [obj_md5.txt](obj_md5.txt). For each instantiation
it gives the registers, the spills and the main loop's instruction counts. The main loop is the loop with the most
dp4a; for `mul_mat_vec_f` it is the loop with the most FFMA. The k-column kernel is `mul_mat_kcol<type, tokens, R>`.
Below are the shapes the preset takes, by `pick_shape`'s rule under its 32 KB budget: R = 1 at K = 2,560; R = 2 for
the 32-weight types at K = 640, where it leaves 20 blocks.

| Kernel | Type | R | Loop global loads at 1 / 2 / 3 / 4 tokens | Loop shared loads (activations and staged weights) | dp4a |
|---|---|---:|---|---|---|
| k-column | Q2_0 | 1 | 2 / 2 / 2 / 2 | 22 / 34 / 46 / 58 | 32 / 64 / 96 / 128 |
| k-column | Q3_K | 1 | 2 / 2 / 2 / 2 | 57 / 85 / 113 / 144 | 64 / 128 / 192 / 256 |
| k-column | Q4_K | 1 | 2 / 2 / 2 / 2 | 29 / 49 / 69 / 89 | 64 / 128 / 192 / 256 |
| k-column | Q5_K | 1 | 2 / 2 / 2 / 2 | 31 / 51 / 71 / 91 | 64 / 128 / 192 / 256 |
| k-column | Q6_K | 1 | 2 / 2 / 2 / 2 | 82 / 111 / 138 / 167 | 64 / 128 / 192 / 256 |
| k-column | IQ4_XS | 1 | 2 / 2 / 2 / 2 | 41 / 66 / 89 / 114 | 64 / 128 / 192 / 256 |
| k-column | Q4_0 | 2 | 3 / 3 / 3 / 3 | 25 / 30 / 35 / 40 | 32 / 64 / 96 / 128 |
| k-column | Q5_0 | 2 | 3 / 3 / 3 / 3 | 29 / 34 / 39 / 44 | 32 / 64 / 96 / 128 |
| k-column | IQ4_NL | 2 | 3 / 3 / 3 / 3 | 26 / 32 / 38 / 44 | 32 / 64 / 96 / 128 |
| k-column | Q8_0 | 2 | 3 / 3 / 3 / 3 | 42 / 48 / 54 / 60 | 32 / 64 / 96 / 128 |
| `mtok`, routed + shared launch (`<gate, staged segments, up to 4 tokens>`) | Q2_0 | 1 | 6 / 6 / 6 / 6 | 28 / 38 / 48 / 58 | 64 / 128 / 192 / 256 |
| `mul_mat_vec_f<float, float, tokens, 256>` | F32 | | 32 / 48 / 16 / 20, per pair of weights 1 + T | 0 | (FFMA 32 / 64 / 24 / 32) |

- **The weights are loaded once per call at every token count.** In the k-column and `mtok` loops the global loads
  stage the weights, and they are the same 2 or 3 instructions (6 for the `mtok` kernel's two matrices) at 1 to 8
  tokens.
- **The activations come from shared memory.** They are staged once per block before the loop, and their loads grow
  by a fixed count per token: 12 per 128 Q2_0 weights; 20 to 31 per super-block for the 256-weight types.
- **The dp4a count is one row's weights times every token.** Each weight is decoded once and multiplied by each
  token's activations.
- **The `mtok` kernel's shared-expert block row serves all tokens.** It sets its group to all `ncols` tokens, so the
  shared expert's gate and up are read once per call (`mmvq-pascal-mtok.cu`).
- **`mul_mat_vec_f`** loads one pair of weights and each token's pair of activations per step: the loop is unrolled 16
  times at 1–2 tokens and 4 times at 3–4.
- **No stack and no local memory** in any of the 198 instantiations read (160 k-column, 15 `mtok`, 19 one-token, 4
  `mul_mat_vec_f`).

## The sums at 2–4 tokens

**Q2_0** (gate, up and down alone):
- Both kernels give a row the same lanes: L is the largest power of two, up to 32, that divides the row's 128-weight
  chunks.
- Lane `sl` adds chunks `sl`, `sl + L`, … in order, and the lanes are added with the butterfly xor 16, …, 1 (offsets
  below L).
- Per 64 weights both compute `acc = fmaf(d, fmaf((float) i1, d8₁, (float) i0 · d8₀), acc)`. In the SASS that is 4 I2F,
  2 FMUL and 4 FFMA per row and 128 weights in both: `mul_mat_kcol<1, 1, 1>` and `mul_mat_vec_q_pascal<Q2_0, 1, 2, …>`.
- Their int8 activations and FP16 scales are the same. The k-column kernel quantizes as `quantize_q8_1` does, and the
  fused norm's Q8_1 copy is bit-identical to `quantize_q8_1` (`norm.cu`).
- So each token's sum is the one-token sum, by construction.

**The routed + shared launch** at 2–8 tokens (`mtok`): its source states that it forms each token's sums as the
one-token kernel does, and its loops have the one-token kernel's 8 I2F, 4 FMUL and 8 FFMA per token for the gate and
up rows (`mul_mat_vec_q_pascal<Q2_0, 1, 2, staged, gate>` has the same).

**Q3_K, IQ4_NL, Q8_0:** their one-token kernel is the k-column kernel, whose sum order depends on the row width only.
So they are exact (C722's reading for Q3_K). The exception is the 6 Q3_K pairs fused at one token. llama.cpp's
`mul_mat_vec_q` splits each 256-weight super-block over 16 threads and adds its warps through shared memory, while the
k-column kernel gives each of 2 lanes whole super-blocks. Their 2–4-token sums therefore come from another order (read
from the code; not emulated).

**IQ4_XS, Q4_K, Q5_K, Q6_K:** the one-token kernel adds per super-block and the k-column kernel per 32 weights (C722).
`GGML_CUDA_MMVQ_PASCAL_DMT=1` (IQ4_XS, Q4_K) and `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1` (Q5_K) take the shared expert's
2,560-wide rows too: their tests depend on the row width and alignment only. Q6_K has no such kernel yet (C727).

**Q4_0 and Q5_0:** the same lanes and order as the one-token kernel, but another block arithmetic. With
I = Σ (code − offset) · q, the block's integer sum (offset 8 or 16):
- one token: `fmaf(d, fmaf((float) (16 Σ code·q), d8 / 16, −offset · (d8 · Σq)), acc)`, which is `d · round(d8 · I)`
  added to acc;
- k-column: `fmaf((float) I · d, d8, acc)`, which is `round(I · d) · d8` added to acc.

In the SASS the one-token kernel has, per row and block, 1 I2F and 2 FFMA, plus 1 I2F and 3 FMUL per activation block
shared by the rows. The k-column kernel has 1 I2F, 1 FMUL and 1 FFMA. Both equal `d · d8 · I + acc`, rounded once,
wherever `d · I` and `d8 · I` are exact in FP32. That holds while |I| < 2^13 (an FP16 scale has 11 significant bits),
so only blocks past that can give other bits.

[run/order_emul.c](run/order_emul.c) emulates both forms at Flash-Next's shapes on the CPU ([order_emul.tsv](order_emul.tsv)).
The weights are quantized from a normal sample as ggml's reference quantizers do. The activations are standard normal,
then uniform on [−1, 1] (larger sums). Each is quantized as `quantize_q8_1` does, with a fixed seed and FTZ:

| Product | Activations | Outputs | Differ | Median of those, ULP | Blocks with a rounded product |
|---|---|---:|---:|---:|---:|
| Q2_0, 2,560 → 640, L = 4 (control) | normal / uniform | 2,560 | 0 / 0 | | |
| Q2_0, 640 → 2,560, L = 1 (control) | normal / uniform | 10,240 | 0 / 0 | | |
| Q4_0, 640 → 2,560, L = 2 | normal / uniform | 10,240 | 0 / 0 | | 0 / 0 of 204,800 |
| Q5_0, 640 → 2,560, L = 2 | normal / uniform | 10,240 | 4 / 79 | 2 / 2 | 11 / 225 of 204,800 |

So Q4_0 and Q5_0 are not bit for bit by construction, and how often they differ depends on the real activations. On
the model's weights only a P106 can say (C755 below). `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0` moves them to the
multi-column Pascal kernel (`mul_mat_vec_q_pascal_mc`), which runs the one-token kernel's arithmetic for every token.

**The F32 gate:** `mul_mat_vec_f` picks its block size from the row width alone (256 at K = 2,560). Each token's sum
is added in the same order at 1–3 tokens, so it is exact there. At 4 tokens cuBLAS's own order takes over.

**SWIGLU:** the fused epilogue and the separate SWIGLU compute the same formula, `g / (1 + exp(−g)) · u`. They were not
compared in the SASS.

## Where the four-token time goes

The excess of a four-token pass over a one-token pass, by type ([times.tsv](times.tsv)):

| Type | ms, 1 token | ms, 4 tokens | Excess |
|---|---:|---:|---:|
| F32 (`ffn_gate_inp_shexp`) | 0.153 | 0.601 | +0.448 |
| Q3_K (gate, up) | 0.810 | 1.195 | +0.385 |
| Q2_0 (gate, up and down alone) | 0.425 | 0.775 | +0.350 |
| Q4_0 (down) | 0.292 | 0.595 | +0.303 |
| Q4_K (gate, up) | 0.181 | 0.405 | +0.224 |
| IQ4_XS (gate, up) | 0.214 | 0.417 | +0.203 |
| Q5_0 (down) | 0.134 | 0.285 | +0.151 |
| Q6_K, Q5_K, IQ4_NL, Q8_0 | 0.347 | 0.519 | +0.172 |
| **the shared expert part** | **2.556** | **4.792** | **+2.236** |

- **The weights' read is a small part of every product.** At C722's measured 153 GB/s the Q2_0 gate (460,800 bytes)
  reads in est. 3.0 µs and takes 11.10 µs a call at one token, 21.06 at four. The Q4_0 down (921,600 bytes): est. 6.0
  against 18.24 and 37.19. The Q3_K up (704,000 bytes): est. 4.6 against 26.27 and 35.32. The IQ4_XS gate (870,400
  bytes): est. 5.7 against 19.52 and 38.03.
- **The launches are small.** With 640 rows of 2,560, or 2,560 short rows of 640, the k-column kernel runs 10 to 20
  blocks of 4 warps on 5 SMs. Each lane makes about 5 loop passes (est., from `pick_shape`'s rule). So the time is
  latency and per-token work, not bandwidth.
- **The F32 gate is the largest single item:** cuBLAS takes 12.78 µs a call at 4 and 5 tokens against
  `mul_mat_vec_f`'s 3.26 at one, for 10,240 bytes of weights.

## What it decides

- **No read-once kernel is needed for the shared expert.** Every product of 2–4 tokens already reads its weights once
  per call, so part 2 of the row (a new switch, branch and rig row) did not run. Q3_K's speed is C726's: its
  kernel would take the shared expert's 33 Q3_K products a pass too (1.195 ms at four tokens against 0.810).
- **New rows** (registry and `QUEUES`):
  - **Q1330 (C754):** F32 products of 4–8 tokens on `mul_mat_vec_f` instead of cuBLAS on compute 6.1. That is the
    shared expert's gate (+0.448 ms) and the router (+0.56 ms) at four tokens, both exact like their one-token sums.
  - **Q1331 (C755):** the shared expert's Q4_0 and Q5_0 downs on the multi-column Pascal kernel
    (`GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0`), exact by construction. Q824's perf table timed the other kernels
    faster than the k-column kernel there at 2–4 tokens. Timing and bit identity need a P106; 0.880 ms of a four-token
    pass today.
  - **Q1332 (C756):** the 6 Q3_K gate/up pairs of one token in the k-column order: unfused at one token (est. +0.05 ms
    a one-token pass), or a fused k-column gate/up.
  - **Q1333 (C757):** one launch for the shared expert's gate, up and SWIGLU at 2–8 tokens, where 41 of 47 layers now
    run three launches of small products. The types often differ: 33 layers mix them.
- **For Q1179's test** (a verify pass that repeats one-token decoding), with C690's and C722's switches on, the shared
  expert still differs in Q6_K (2 products), the 6 Q3_K pairs, Q4_0 and Q5_0 (rarely), and the F32 gate at 4 tokens.
  The router's F32 product has the same 4-token gap, and it chooses the experts.

## Files

| File | What |
|---|---|
| [products.tsv](products.tsv) | every shared-expert product: layers, type, shape, the kernels at 1 and 2–4 tokens, the weights' reads, the sums, R41's times |
| [times.tsv](times.tsv) | R41's launch groups that name a shared-expert tensor, at 1, 4 and 5 tokens: calls a pass, ms a pass, µs a call |
| [sass_counts.tsv](sass_counts.tsv) | every instantiation read: registers, stack, local memory, the main loop's LDG, LDS, STS, dp4a, I2F, FMUL, FFMA, FADD (the `mtok` kernel: every loop with dp4a) |
| [order_emul.tsv](order_emul.tsv) | the CPU emulation of the Q2_0, Q4_0 and Q5_0 sums in the two kernels' arithmetic |
| [obj_md5.txt](obj_md5.txt) | the md5 of the 14 objects read |
| [run/dump_sass.sh](run/dump_sass.sh) | cuobjdump of the objects in the CUDA devel image, no GPU |
| [run/sass_check.sh](run/sass_check.sh) | reads the dump into `sass_counts.tsv` |
| [run/order_emul.c](run/order_emul.c) | the emulation |
| [run/shexp_times.sh](run/shexp_times.sh) | R41's kept trace rows into `times.tsv` |

## Commands

On any x86-64 machine with Docker (no GPU), against a build tree of the fork at `5b329df16`
(`scripts/build_llamacpp_docker.sh`; `BUILD` is its build directory), and R41's `release_parts.py` output (`RES`):

```bash
BUILD=$BUILD OUT=$PWD/sass bash run/dump_sass.sh
SASS=$PWD/sass bash run/sass_check.sh > sass_counts.tsv
gcc -O2 -ffp-contract=off -o order_emul run/order_emul.c -lm && ./order_emul > order_emul.tsv
RES=$RES bash run/shexp_times.sh > times.tsv
```

The objects are those of the `TESTS=1` rig build of `5b329df16` ([obj_md5.txt](obj_md5.txt); `mmvf-pascal.cu.o`, read
by C730, is not among them). R41's trace is its run at k = 1, 4 and 5 without the draft (its README).

## Provenance and licence

Code read: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c748` at `5b329df16`. The times are R41's published profile. The emulation's inputs are
generated (a fixed seed), not model weights. Everything here is this repository's licence.
