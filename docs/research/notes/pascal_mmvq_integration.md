# Porting the Pascal matrix-vector kernels into llama.cpp's CUDA backend

Design for experiment Q38 (and its follow-up Q39) in [quants/experiments.md](../../../quants/experiments.md): how the
Pascal-tuned single-token kernels of [bench/mmvq_pascal](../../../bench/mmvq_pascal/) become a code path inside
llama.cpp, what they are expected to gain on the file at the quality bar, and in which steps to build and upstream them.

Code references:
- llama.cpp lines refer to upstream commit `d77256999` without this repository's patches. `mmvq.cu`, `vecdotq.cuh`,
  `quantize.cu`, `common.cuh`, `ggml-cuda.cu`, `mmq.cuh` and `mmq-load-tiles.cuh` live in `ggml/src/ggml-cuda/`.
  `ggml-sycl.cpp` and `getrows.cpp` live in `ggml/src/ggml-sycl/`.
- `pascal_mmvq.cuh` and `main.cu` are in [bench/mmvq_pascal/src/](../../../bench/mmvq_pascal/src/).

## Summary

- **What.** A second single-token matrix-vector kernel in the CUDA backend, used on compute capability 6.1
  (`ggml/src/ggml-cuda/mmvq-pascal.cu` on the fork's `pascal-p106` branch). It is the benchmark's variant (f): lane
  groups sized to the row width, several rows per lane group sharing one load of the activations, several weight
  blocks per lane, pointer arithmetic hoisted out of the loop, and each lane group staging its rows' next segment
  through shared memory with coalesced 128-bit loads. It reads the GGUF blocks as they are, so it needs no repacking.
- **Where.** One call in `ggml_cuda_mul_mat_vec_q`, after the activations are quantized and right before the stock
  kernel would be launched. Operator selection, fusion matching and the routing of expert (`MUL_MAT_ID`) products
  stay as they are. Any call the new kernel doesn't cover falls through to the stock kernel unchanged.
- **Which types.** Q8_0, Q4_1, Q4_0, Q5_0, Q5_K and IQ4_XS are ported: all of S4-L0's quantized bytes and 96 % of
  Unsloth's UD-Q4_K_XL (the rest is its BF16 router and a few Q4_K, IQ3_S and Q6_K tensors). Q6_K, the 6-bit rung
  for files that would otherwise keep Q8_0 (Q101), is on the fork's branch `pascal-p106-v2-q6k` (section 15). Q4_1_64
  (a fork-only type) is next.
  - The file at the quality bar reads 66 % of its bytes as Q8_0, so Q8_0 carries the largest share of the gain.
  - Q5_0 uses a cheaper way to apply its fifth bit (section 5).
- **Measured gain.** llama-bench tg128 on 6 P106-090 cards, S4-L0, one binary with the kernels switched off and on:
  **29.2 → 37.4 tok/s (+28 %)**; against the wide-rows build, the best before the port, 28.0 → 37.4 (+33 %). Unsloth's
  UD-Q4_K_XL: 24.5 → 34.1 (+39 %; 17.6 on stock llama.cpp). Unsloth's Q8_0 on 9 cards: 26.9 → 35.9 (+33 %). Prompt processing is unchanged. In the microbenchmark Q8_0 runs at 150 GB/s,
  2.07× the stock kernel.
- **Several tokens per step.** 2–4 tokens (a speculative-decoding verify step, a prompt's last tokens) run a
  multi-token variant on the fork's branch `pascal-p106-v2-multicol`. It reads every weight row once per step, and for
  expert products every distinct expert once, and gives each token the single-token result bit for bit (section 14).
- **No repack needed.** Staged loads reach the speed that row-local or tensor-wide weight planes reach (Q8_0 17.1
  against 18.4 ms per Flint token in the microbenchmark), so the load-time repack of Q39 is no longer needed for decode
  speed.
- **Upstream.** Four separate PRs: the launch shape, the Pascal kernel, one activation quantization per shared input,
  and FP32 activation sums. Q4_1_64 stays in the fork.

## 1. Background

### Where a token's time goes on this card

A P106-090 has 5 SMs and streams 179 GB/s from memory. When llama.cpp generates one token, most of the GPU time goes into
matrix-vector products: every weight matrix is read once and multiplied by one activation vector. llama.cpp does this with
its MMVQ kernel (`mul_mat_vec_q`, `mmvq.cu:621-874`). The kernel first quantizes the activation vector to Q8_1: blocks of
32 int8 values with an FP16 scale `d` and an FP16 sum `s` (`quantize.cu:54-101`). It then takes integer dot products
with `dp4a`, four int8 products per instruction.

On this card the kernel runs out of instruction issue before it runs out of memory bandwidth. The 5 SMs can issue about
6.4 instructions per byte of a 173 GB/s stream. llama.cpp's loop needs 5–9 per byte for the legacy formats, and more with
prologue and reduction at Flint's 2048-wide rows. Most of these instructions compute addresses. Loads come only 16 or 32
bits wide, because 18-, 20-, 22- and 34-byte blocks are only 2- or 4-byte aligned. See
[quants/research.md](../../../quants/research.md), "Where a token's time goes", and
[quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) for the instruction counts per format.

### The standalone kernels and what they measured

[bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) times llama.cpp's kernel and Pascal-tuned variants on the exact
shapes of one Flint token (39.8 B-parameter MoE: 49 attention layers, 47 MoE layers with 6 of 256 experts per token, a
150,144-row output head):

- **Variant (c), `mmvq_rows`** (`pascal_mmvq.cuh:190-281`).
  - A warp is cut into lane groups of `lpr` lanes, a power of two chosen from the row width (`main.cu:629-643`).
  - Each lane group owns `R` rows that share one activation load, and `2R` when gate and up are fused.
  - Each lane takes `C` consecutive blocks per iteration (`ldg_words`, `pascal_mmvq.cuh:54-75`). Fields at
    2-byte-aligned offsets come out with one `PRMT` (`word_at`, `:78-80`).
  - Row pointers are computed once and advanced by a constant (`:209-221`, `:249-254`).
  - 4-bit codes are unpacked with two masks and an unsigned-by-signed `dp4a`; the factor 16 goes into the activation
    scale (`q4_isum16`, `:120-132`).
  - The reduction uses warp shuffles only (`:257-268`), with no shared memory and no `__syncthreads`.
- **Variant (d), `mmvq_soa`** (`:307-444`) reads Q4_1 and Q4_1_64 repacked into a plane of 16-byte code chunks and a plane
  of `half2 (d, m)` scales. Every weight load is a coalesced 128-bit load. The activations use an int8 plane and a
  `float2 (d, s)` plane (`quantize_soa`, `:448-466`).

Every variant matches a double-precision CPU reference, and the copied baseline is bit-identical to llama.cpp. On one
P106-090 the matrix-vector time per Flint token is:

| Type | llama.cpp | 2 warps (PR #25479) | best (c) | best of (c) and (d) |
|---|---|---|---|---|
| Q4_1 | 20.0 ms | 17.3 ms | 11.5 ms (1.75×) | 10.6 ms (1.89×) |
| Q4_1_64 | 21.2 ms | 18.2 ms | 10.6 ms (1.99×) | 10.1 ms (2.10×) |
| Q4_0 | 24.3 ms | 19.7 ms | 12.6 ms (1.93×) | — |
| Q8_0 | 35.3 ms | 31.7 ms | 23.3 ms (1.52×) | — |

Q8_0 gains least. Its 34-byte block is 2-byte aligned, so (c) can't use wide loads without extracting every other
block's words with `PRMT`. Its fused gate/up kernels also run short of registers: at R = 2 or C = 4 they need 254–255
registers, which leaves 2 blocks per SM.

### The file at the quality bar

The quality bar for Flint quants is Unsloth's UD-Q4_K_XL. The fastest file found within noise of it is **S4-L0**
([recipe](../../../quants/recipes/flint-speed-s4-l0.txt), Q36 in [experiments.md](../../../quants/experiments.md)). Its
weights per token, from Flint's shapes:

| Tensors | Format | Matrix-vector calls per token | MB per token |
|---|---|---|---|
| attention q | Q4_1 | 49 | 257 |
| attention k, v | Q8_0 | 98 | 109 |
| attention output, layers 0–32 | Q8_0 | 33 | 294 |
| attention output, layers 33–48 | Q4_1 | 16 | 84 |
| dense FFN, layers 0–1 (gate/up fused, down) | Q8_0 | 4 | 80 |
| routed experts gate/up (6 of 256, fused) | Q5_0 | 47 | 407 |
| routed experts down | Q8_0 | 47 | 314 |
| shared expert gate/up (fused) | Q8_0 | 47 | 209 |
| shared expert down | Q8_0 | 47 | 105 |
| output head | Q8_0 | 1 | 327 |
| **Total** | Q8_0 65.8 %, Q5_0 18.6 %, Q4_1 15.6 % | | **2186** |

The router (F32 or F16, 0.05–0.1 GB) is not a quantized type and doesn't run through MMVQ.

On the rig (llama-bench tg128 on 6 cards, Q34), S4-L0 runs at:
- 23.09 tok/s on the standard build (llama.cpp's kernel and launch shape, CUDA graphs for Pascal);
- 26.29 tok/s with 2 warps per block at every row width;
- 28.02 tok/s with 2 warps only where llama.cpp's small-K rule doesn't apply (the "wide-rows" build).

In a real chat at 262k context on 9 cards, the wide-rows build gives 24.72 tok/s.

## 2. How a single-token product runs today

**Operator selection.**
- `ggml_cuda_mul_mat` (`ggml-cuda.cu:1976-2029`) sends a quantized weight with at most 8 activation columns to MMVQ
  (`ggml_cuda_should_use_mmvq`, `mmvq.cu:318`; call at `ggml-cuda.cu:2020-2023`).
- `ggml_cuda_mul_mat_id` (`ggml-cuda.cu:2064-2086`) sends expert products with at most `get_mmvq_mmid_max_batch` tokens
  to the same entry point, with the expert ids.
- Both end in `ggml_cuda_mul_mat_vec_q`. The older entry `ggml_cuda_op_mul_mat_vec_q` (`mmvq.cu:1589-1618`) has no
  caller in the CUDA backend at this commit.

**Fusion.**
- `ggml_cuda_try_fuse` (`ggml-cuda.cu:3777` onwards) matches graph patterns and calls `ggml_cuda_mul_mat_vec_q` with a
  `ggml_cuda_mm_fusion_args_host` (`common.cuh:1568-1579`). It does so only if `ggml_cuda_should_fuse_mul_mat_vec_q`
  (`ggml-cuda.cu:1895-1922`) agrees.
- That check turns fusion off for compute capability 6.0 and below (`ggml-cuda.cu:1907-1911`), so 6.1 fuses. The patterns:

| Pattern | Where matched | Fusion fields | Used by Flint |
|---|---|---|---|
| gate + up + GLU, `MUL_MAT` or `MUL_MAT_ID`, optional bias | `ggml-cuda.cu:4254-4356` | `gate`, `glu_op`, `glu_limit`, `x_bias`, `gate_bias` | dense FFN, routed and shared experts |
| same with per-lane scales | `ggml-cuda.cu:4070-4252` | `x_scale`, `gate_scale` (NVFP4 only, asserted at `mmvq.cu:1489`) | no |
| matrix-vector + scale (+ bias) | `ggml-cuda.cu:4366-4448` | `x_scale`, `x_bias` | no |
| matrix-vector + add | `ggml-cuda.cu:4454-4509` | `x_bias` | no (Flint has no biases) |
| routed + shared expert gate/up in one launch | `ggml-cuda.cu:3785-3803`, matched by `ggml_cuda_match_shared_expert` (`:1924-1974`) | `shared_up`, `shared_gate`, `shared_dst` | no: Flint's shared expert is 1024 wide, its routed experts 512 (Qwen3.6 matches) |

**Inside `ggml_cuda_mul_mat_vec_q`** (`mmvq.cu:1455-1587`):
1. It checks the fusion arguments (`:1483-1538`). With `shared_up`, the routed and shared weights must have the same type
   and shape (`:1491-1506`).
2. It quantizes the activations to Q8_1, once per call (`:1551-1558`).
3. It maps `MUL_MAT_ID` onto the kernel's channel dimension (`:1571-1578`): one destination column per token, one
   channel per selected expert.
4. It calls `mul_mat_vec_q_switch_type` (`:1303-1453`), which instantiates `mul_mat_vec_q_switch_ncols_dst<type>`
   (`:1111-1302`).

**`mul_mat_vec_q_switch_ncols_dst`** picks the kernel:
- Expert products with more than one token go to the dedicated MoE kernel `mul_mat_vec_q_moe` (`:1201-1209`).
- One column (`case 1`, `:1212-1242`) goes to `mul_mat_vec_q` in one of three shapes:
  - the **small-K rule** (`:1142-1177`): when the whole block covers a row in one iteration, rows per block = warps
    (`:599-601`), so warps aren't idle at K = 512;
  - the GB10 "halve iterations" variant;
  - the default.
- The grid is `(row blocks, channels + 1 if shared_up, samples)` (`:1226`, `calc_launch_params` at `:1034-1044`).

**Inside the kernel** (`mmvq.cu:621-874`):
- With `shared_up`, the last `blockIdx.y` slice switches to the shared expert's weights and output (`:650-656`).
- The expert comes from `ids[channel]`, and the activation from `channel % nchannels_y`, which is the shared input for
  gate/up and one input per expert for down (`:663-664`).
- Each thread calls `vec_dot_q_cuda` on `vdr` 32-bit words of one block per iteration (`:743-784`). For Q8_0 and Q5_0
  the words are assembled from 16-bit loads (`get_int_b2`, `vecdotq.cuh:18-25`).
- Warps reduce through shared memory (`:786-830`).
- The epilogue adds biases and applies the GLU (`:831-864`).

The launch shape comes from a parameter table: `calc_nwarps` (`:452-579`) and `calc_rows_per_block` (`:595-619`),
selected by `get_device_table_id` (device side `:106-122`, host side `:124-144`). Compute 6.1 gets the generic table:
4 warps and 1 row per block at one token.

## 3. Where the compute-6.1 path hooks in

**The decision point.** One call in `ggml_cuda_mul_mat_vec_q`, after the activations are quantized to Q8_1 and right
before `mul_mat_vec_q_switch_type`. It takes the same arguments (strides in blocks, the quantized activations, the
device-side fusion arguments) and returns false for anything it doesn't cover:

```cpp
if (ggml_cuda_mmvq_pascal(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00, ne01, ncols_dst, s01,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst, ne03, ne3, s03, s13, s3, stream)) {
    return;
}
```

Placed here, the port reuses llama.cpp's quantization, its MUL_MAT/MUL_MAT_ID stride mapping and every fusion check,
and the diff to `mmvq.cu` is an include and these five lines. One quantization per shared input (section 8) moves the
call above the quantization; the FP32 activation sums didn't need that, because the kernels compute them from the int8
values.

**What it takes.** The kernel runs a call when all of these hold; otherwise the stock path runs as before:
- the device is compute 6.1, or the force switch below is set;
- the weight type is Q4_0, Q4_1, Q5_0, Q8_0, Q5_K or IQ4_XS;
- one destination column: `ne11 == 1` for `MUL_MAT`, `ne2 == 1` for `MUL_MAT_ID`;
- `x_scale` and `gate_scale` are null (they exist only for NVFP4);
- the blocks per row split into `C`-block chunks; `lpr` is then the largest power of two up to 32 that divides the
  chunks per row, so any row width works;
- staged loads need 16-byte-aligned weights, rows, expert and sample strides and a staged segment (`lpr` chunks) that
  is a multiple of 16 bytes; where that fails the same shape runs with direct loads, which need only the chunk's own
  alignment; where that fails too (views with padded rows, as `test-backend-ops` builds with `k_v`, `m_v`, `pad`), the
  stock kernel runs.

**The launch shape of the stock kernel.** The fork's wide-rows commit (2 warps per block where the small-K rule doesn't
apply) edits llama.cpp's generic parameter table, which is right for an sm_61-only build and wrong for every other GPU.
An upstream PR would add `MMVQ_PARAMETERS_PASCAL_DP4A` to `get_device_table_id` instead (host side and device side,
`__CUDA_ARCH__ == 610`), the id PR #25479 introduces. The stock kernel still runs everything the new kernel doesn't
cover, such as UD-Q4_K_XL's K- and I-quants, which gained 26–27 % from the 2-warp shape.

**A switch for testing.** `GGML_CUDA_MMVQ_PASCAL` = `0` (off), `1` (default: compute 6.1 only) or `2` (any NVIDIA GPU).
With `2`, a modern GPU runs the Pascal kernel natively and, with `CUDA_FORCE_PTX_JIT=1`, from the compute_61 PTX. That
tests correctness without the rig, and `0` gives the stock kernels on the same binary for speed comparisons. The
kernel's template parameters (warps per block, R, C) don't depend on `__CUDA_ARCH__`, so the host and the device always
agree on the block size, including under the PTX JIT.

**Files.** `mmvq-pascal.cuh` and `mmvq-pascal.cu` hold the kernel and its launcher. HIP and MUSA builds compile a stub
that returns false.

## 4. The kernel inside llama.cpp

**Signature.** `mul_mat_vec_q_pascal<type, R, C, staged, fusion>` takes one argument struct (byte strides for the
weights and activations, float strides for the destination, the fastdiv ratios of the stock kernel, `lpr_log2` and
`niter`) and the device-side fusion struct. `fusion` is plain, bias only, or gate + GLU: the gate doubles the weight
loads and accumulators, so the compiler must know. Warps per block are fixed at 4.

**Prologue.** The same as `mul_mat_vec_q`'s:
- the shared-expert slice (`blockIdx.y == gridDim.y - 1` with `shared_up`);
- the expert from `ids[channel_dst]`;
- the activation channel `channel_dst % nchannels_y`;
- channels and samples through `channel_ratio` and `sample_ratio`.

**Main loop and reduction.** Per iteration, each lane group copies the segment its lanes are about to read (lpr chunks
of C blocks per row, contiguous in memory) into its own part of shared memory with 128-bit loads, `__syncwarp()`, and
each lane reads its C blocks from there. Only lanes of one warp share a segment, so no `__syncthreads` is needed. Row
pointers are clamped to the last row, so lane groups at the end of the matrix don't branch; only the store is guarded.
The reduction uses warp shuffles inside the lane group.

**Epilogue.** The stock kernel's result handling (bias, gate bias, SWIGLU, GEGLU, SWIGLU_OAI, SWIGLU_CLAMP) is repeated
in the new file, from lane `i` of each lane group after the reduction. Keeping a copy rather than factoring it out of
`mul_mat_vec_q` leaves the stock kernel's code untouched.

**Launch parameters.** Measured best per type at Flint's shapes (GB/s on the P106-090 in brackets, from the
microbenchmark's variant (f)):

| Type | Unfused | Fused gate/up |
|---|---|---|
| Q8_0 | R = 2, C = 2 (113–165) | R = 1, C = 2 (151–160) |
| Q4_0 | R = 2, C = 2 (89–142) | R = 2, C = 2 (132–145) |
| Q4_1 | R = 2, C = 4 (100–159) | R = 2, C = 2 (140–150) |
| Q5_0 | R = 2, C = 2 (98–144) | R = 1, C = 2 (130–138) |
| Q5_K | R = 1, one super-block per lane (97–151) | the same (110–121) |
| IQ4_XS | R = 1, one super-block per lane (75–133) | the same with direct loads (86–94) |

Staged shared memory per block is 4 warps × 32 lanes × R × C × block bytes (twice with the gate): 17 KB for Q8_0,
45 KB for Q5_K's fused gate/up.

**Q5_K and IQ4_XS** take one 256-weight super-block per lane, so each lane reads 8 activation blocks per iteration.
Q5_K's mins need the sum of the quantized activations; as in `vec_dot_q5_K_q8_1` it is computed with
`dp4a(0x01010101, u)`, once per super-block and shared by all rows. IQ4_XS's 4-bit indices select from a 16-entry table
with `__byte_perm` on four table words held in registers; the activations are permuted once per block into the order
the lookup produces (weights 4k, 4k+16, 4k+1, 4k+17, …), which saves llama.cpp's two re-ordering permutes per word.

The lane groups make the small-K rule unnecessary. At K = 512, Q8_0 with C = 2 has 8 chunks per row, so `lpr` = 8, and
each warp computes 8 rows with every lane busy. The expert down-projection reaches 142 GB/s this way, against 81 for the
stock kernel's small-K shape.

**Constraints that carry over.**
- The kernel runs inside CUDA graphs like the stock one: no host synchronization, and the activation buffer comes from
  the pool.
- `ggml_cuda_pdl_sync` is a no-op before Hopper and is called for uniformity.

## 5. Which types, in which order

S4-L0 reads 65.8 % of its bytes as Q8_0, 18.6 % as Q5_0 (the routed experts' gate/up) and 15.6 % as Q4_1. The port
took them in this order:

1. **Q8_0, Q4_1 and Q4_0 together.** One kernel template covers all three. Q8_0 saves the most on the quality-bar file;
   Q4_1 is the makers' Flint file and attention q in S4-L0; Q4_0 is the base type of the Qwen3.6 mixes
   ([recipes](../../../quants/recipes/README.md)). On S4-L0 these three took tg128 from 29.2 to 35.5 tok/s.
2. **Q5_0**, with the fifth-bit dot product below, first in the benchmark (2.28× the stock kernel staged), then in the
   port: 35.5 → 37.4 tok/s.
3. **Q5_K and IQ4_XS**, the K- and I-quants of Unsloth's UD-Q4_K_XL (29 % of its bytes; the rest is mostly Q8_0), from
   the benchmark's kernels (Q46): 2.44× and 2.90× the stock kernel there.
4. **Q4_1_64** is this project's own type ([patch](../../../patches/q4_1_64-type.diff)). It holds the makers' group-64
   weights exactly and is the fastest type in the benchmark (2.10× with planes), but upstream llama.cpp doesn't have it,
   so it stays in the fork.

**A cheaper Q5_0 dot product.**
- The obstacle: a Q5_0 block is `half d`, 4 bytes `qh` with the fifth bit of each of the 32 weights, then 16 bytes of
  nibbles (`block_q5_0`, `ggml-common.h:230-234`). llama.cpp moves each fifth bit into its byte with a shift, AND and OR
  (`vecdotq.cuh:175-201`), 56 of the 192 instructions per 32 weights.
- The trick: the fifth bits don't have to sit in the nibble's byte. The dot product splits as
  Σ q·a = Σ nibble·a + 16·Σ h·a.
  - `(qh >> j) & 0x01010101` holds the fifth bits of weights j, 8 + j, 16 + j and 24 + j, one per byte, ready for `dp4a`.
  - The activations must be grouped the same way: bytes a[j], a[8+j], a[16+j], a[24+j] in one word. That is two 4 × 4
    byte transposes of the block's eight activation words, 16 `PRMT` per 32 activations. All R rows and the gate share
    them.
  - Per row, the fifth bits then cost 8 shifts, 8 masks and 8 `dp4a`.
  - With the 0x0F/0xF0 nibble masks of `q4_isum16`, the block's integer sum in the ×16 domain is
    `(lo << 4) + hi + (hsum << 8)`. Its magnitude stays below 2.1 M.
  - The result is `d * (isum16 * d8/16 - 16 * s8)`, like Q4_0's `-8 * s8` (`pascal_mmvq.cuh:146-149`).
- About 50 instructions per 32 weights per row (2.4 per byte, against 8.7 for llama.cpp's), so the kernel is limited by
  loads like Q4_1's: in the benchmark it reaches 117 GB/s on the expert gate/up shape with direct loads and 138 staged,
  against 53 for the stock kernel.
- It needs whole blocks per lane. The stock kernel's half-block split (`vdr` = 2) can't form these byte groups, so the
  trick belongs to the Pascal kernel.

## 6. Keeping the fusions and the expert path

**Every fused call arrives as an ordinary `ggml_cuda_mul_mat_vec_q` call** with filled-in fusion fields. The predicate
either takes the whole call or leaves it to the stock kernel. No change in `ggml-cuda.cu` is needed, and every fusion
keeps working throughout the work, at stock speed until the Pascal kernel covers it.

| Fusion or path | How the Pascal kernel keeps it |
|---|---|
| gate + up + GLU (`gate`, `glu_op`, `glu_limit`) | gate instantiation: the gate rows load next to the up rows and share the activation registers; the epilogue applies SWIGLU, GEGLU, SWIGLU_OAI or SWIGLU_CLAMP as `mul_mat_vec_q` does |
| biases (`x_bias`, `gate_bias`, also `ADD_ID` for experts) | read per row in the epilogue, with `channel_bias = ids ? channel_x : channel_dst` as at `mmvq.cu:708` |
| routed + shared gate/up (`shared_up`) | the launcher adds one `blockIdx.y` slice, and the prologue swaps in the shared weights, output and column stride, as at `mmvq.cu:650-656`. `nrows` is the same for both, by the assertions at `mmvq.cu:1491-1506` |
| per-lane scales (`x_scale`, `gate_scale`) | NVFP4 only; the predicate refuses them |
| `MUL_MAT_ID`, one token | channel = expert slot, expert from `ids`, activation per expert or shared via `nchannels_y` |
| `MUL_MAT_ID`, 2–8 tokens | unchanged: `mul_mat_vec_q_moe` (`mmvq.cu:1201-1209`) |
| `MUL_MAT`, 2–8 columns (speculative decoding) | unchanged in Q38; the lane-group design extends to several columns later |
| `GGML_CUDA_DISABLE_FUSION=1` | still turns fusion off (`ggml-cuda.cu:172`); unfused calls take the Pascal path one by one |

In Flint's real graph, llama.cpp fuses the routed experts' gate/up and the first dense layer, but not the shared
expert's gate/up or the second dense layer: the profile (section 10) shows them as two matrix products and a separate
SWIGLU. The fusion matcher needs gate, up and GLU as adjacent nodes, which they are not there. Fusing them would save 47
launches, activation quantizations and SWIGLU kernels per token.

## 7. Separate planes and load-time repacking

**Decode speed needs no repacking.** The plane layouts of variants (d) and (e) give every lane 128-bit loads, which
is what held Q8_0 back in (c): its 34-byte blocks are only 2-byte aligned, so direct loads are 32 bits wide and strided.
Staging the unchanged blocks through shared memory (f) gets the same coalesced loads and is as fast or faster: Q8_0
17.1 ms per Flint token staged, 18.4 with row-local planes, 23.2 with direct loads. Q39 is therefore not needed for the
matrix-vector kernels. The design below stays the reference for a repack, should another reader ever need planes.

**The plane layout itself.** A GGUF stores blocks of interleaved scales and codes, and a new GGUF type is not an option:
it would break every other backend and tool. Planes would have to be produced in GPU memory after loading.

### How llama.cpp's SYCL backend does it

The SYCL backend has done this on Intel GPUs since PR #12035 (Q4_0) and PR #21527 (Q8_0); see
[quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md) for their measured gains.
- **Marking.** `init_tensor` attaches a `ggml_tensor_extra_gpu` to every tensor of a reorderable type
  (`ggml-sycl.cpp:764-782`). The feature is on by default (`GGML_SYCL_ENABLE_OPT`, `:415`) and only for Intel GPUs
  (`:237`).
- **Lazy, in place.** At the first `MUL_MAT` with at most 8 activation columns (`should_reorder_tensor`, `:5092-5099`),
  `opt_for_reorder` (`:5101-5133`) runs the repack and sets `extra->optimized_feature.reorder`.
  - The repack copies the tensor to a temporary device buffer of the same size and scatters it back as two planes: all
    codes of the tensor, then all scales (`reorder_qw_q4_0`, `:4410-4447`; Q8_0 at `:4449`).
  - The tensor keeps its size, so no extra memory remains afterwards. A failed temporary allocation skips the repack
    (`sycl_reorder_temp_buffer`, `:4369-4408`).
- **Experts.** Expert tensors are repacked per expert slice, so `nb[2]` still addresses each expert
  (`reorder_qw`, `:5049-5065`; `reorder_qw_q8_0_moe`, `:4489-4527`; `opt_for_reorder_id`, `:5136-5155`).
- **Every reader knows the layout.**
  - The matrix-vector kernels have reorder variants, and the activations are quantized into a matching per-row layout:
    int8 values, then the `half2 (d, s)` pairs (`quantize_and_reorder_q8_1_soa`, `quantize.hpp:63-92`, chosen at
    `ggml-sycl.cpp:5266-5272`).
  - Dequantization for matrix-matrix products and row lookups have reorder variants (`convert.cpp:155` onwards,
    `getrows.cpp:140`).
  - SYCL has no batched quantized kernel to adapt: `ggml_sycl_supports_mmq` returns false (`ggml-sycl.cpp:4241-4245`).

### What differs on CUDA and on this rig

- **Prompt processing reads the same weights.** On compute 6.1, prompts use MMQ, the `dp4a` matrix-matrix kernel, whose
  tile loaders read the block layout (`mmq-load-tiles.cuh:207`, `:270`, `:333`, `:491` for Q4_0, Q4_1, Q5_0, Q8_0).
  - Falling back to dequantize + cuBLAS is not an option: compute 6.1 has no fast FP16 (`fast_fp16_available`,
    `common.cuh:313-317`), so prompts would run as FP32 SGEMM.
  - A second copy in the old layout doesn't fit either: S4-L0 fills 6 cards of 6 GB.
  - So MMQ's four loaders need plane-aware variants.
- **CUDA graphs.** The rig's builds capture each token's work in a CUDA graph. A repack triggered inside the first
  evaluation could be captured and then replayed every token. It has to run before the graph is evaluated or captured.
- **Uploads come in pieces.** The model loader can write a tensor in 1 MB pieces (`buffer_size`,
  `src/llama-model-loader.cpp:1586`; `ggml_backend_tensor_set_async` per piece, `:1756-1801`), so `set_tensor`
  (`ggml-cuda.cu:844-850`) can't assume it sees a whole tensor.
- **Memory headroom.** A temporary buffer of a whole expert tensor, as SYCL uses, is 285 MB for one layer's Q8_0
  down-projections. That doesn't always fit next to a 5 GB share of the model on a 6 GB card.
- **No CUDA mechanism exists yet.** Mainline has no CUDA "extra" or repack buffer type (the CPU backend has one). The
  open CUDA repack PoCs on large GPUs showed no generation gain; see the survey note for the discussion.

### Proposed design: row-local planes, repacked in place before the first graph

- **Layout.** Each row keeps its byte range, and only the fields inside it are regrouped:
  - Q4_1 `[codes: nb × 16 B][d, m: nb × 4 B]`;
  - Q8_0 `[codes: nb × 32 B][d: nb × 2 B]`;
  - Q5_0 `[codes: nb × 16 B][qh: nb × 4 B][d: nb × 2 B]`;
  - Q4_0 `[codes][d]`;
  - Q4_1_64 `[first halves][second halves][d, m]`.

  Here nb = blocks per row. Rows stay the same size, so the tensor's shape, strides and expert slices are unchanged. With
  K a multiple of 256, every code plane starts 16-byte aligned. Within a row the loads are the same coalesced 128-bit
  loads as variant (d), which uses tensor-wide planes. Step 1 checks that row-local planes are as fast.
- **In place, no temporary buffer.** One thread block per row reads the row into shared memory (at most 6.5 KB at
  K = 6144 for Q8_0), synchronizes, and writes it back regrouped.
- **When.**
  - At the start of `ggml_backend_cuda_graph_compute`, before evaluation or capture, scan the graph.
  - Repack every `MUL_MAT`/`MUL_MAT_ID` weight that lives in a weights buffer on this device, has a covered type, and is
    not yet marked.
  - Record repacked tensors in the backend context. Any `set_tensor`, `memset_tensor` or copy into a tensor clears its
    mark.
  - The first token pays the repack, about one read and one write of the card's weights (~0.1 s).
- **Readers.** All of them check the mark: the planar matrix-vector kernels, the planar MMQ loaders, row lookup, and
  dequantization (`convert.cu`). `get_tensor` undoes the regrouping, so weights read back unchanged.
- **Opt-in.** Behind `GGML_CUDA_PASCAL_REPACK=1`, with `=2` to repack any buffer, which `test-backend-ops` needs because
  its tensors don't live in weights buffers.

**Activations.** In the plane kernels the activations use the benchmark's planar format (an int8 plane and a
`float2 (d, s)` plane). FP32 sums come with it at no extra cost (section 8).

## 8. Activation-side changes

These are independent of the weight kernel. Both use the decision point of section 3.

- **One quantization per shared input, done by the norm (Q100, fork branch `pascal-p106-v2-normq8`).** Every
  `ggml_cuda_mul_mat_vec_q` call quantizes its own activations. In Flint, attention q, k and v read the same vector,
  and so do the routed and the shared gate/up; all of them read the output of an RMS norm.
  - One S4-L0 token on six cards runs 389 activation quantizations (from the rig's event profile, with Q50's shared
    gate/up fusion counted in): 147 before Q/K/V (three per layer of the same vector), 47 + 47 + 2 before the routed,
    shared and dense gate/up, 1 before the output head, and 49 + 47 + 47 + 2 before the attention output and the
    down-projections, whose inputs no norm produces. 48 residual adds run as their own launch.
  - The fork's single-row `RMS_NORM` + `MUL` now also writes the row as Q8_1 blocks into a per-device buffer, and when
    the residual `ADD` comes right before the norm (each layer's `l_out` and the attention norm of the next layer, the
    last layer's and the output norm), it adds them in the same kernel and writes the sum too, since the residual
    stream is read again. `mul_mat_vec_q` uses the buffer whenever its `src1` is that norm's output (directly or
    through a reshape or view of all of it) and skips its quantization. The copy is dropped at the start of each graph
    evaluation and when any node writes over the norm's output (the shared gate/up's GLU output may reuse that memory,
    Q50).
  - The norm keeps `rms_norm_f32`'s layout (one float4 per thread, the same block reduction), and the 8 threads that
    hold a Q8_1 block reduce its maximum and sum in `quantize_q8_1`'s butterfly order, so the normed row and the blocks
    are bit-identical to the unfused kernels', and every product sees the same input. One block reads every input
    before its reduction and writes every output after it, so outputs may overlap inputs (an in-place residual add).
  - Covered: one row, F32, width a multiple of 512 up to 4,096 (Flint: 2,048); anything else runs as before.
    `GGML_CUDA_NORM_Q8=0` switches it off; `GGML_CUDA_NORM_Q8_DEBUG=1` logs per graph evaluation how many norms wrote
    a row and how many products read one or quantized their own input.
  - Per token on six cards it removes 244 quantizations and 43 adds (48 minus the five card boundaries, where the add
    and the norm run on different cards): 287 launches. It helps every GPU at one token, not just Pascal.
  - Checked on an RTX 3090 (fork commit `4d2a419b4`):
    - `test-backend-ops ADD_RMS_NORM_MUL_Q8` (a layer-shaped graph: the residual read again, the normed row feeding
      Q/K-like products, a routed and a shared expert pair and an F32 router; 7 weight types, widths 2,048 and 4,096,
      with and without the add, plus a 1,536-wide and a 2-row case it must leave alone) passes with the Pascal kernels
      forced, with the stock kernels, with the fusion off and through the compute_61 PTX; the debug counts show one
      norm writing a row and four products reading it.
    - The existing RMS_NORM, RMS_NORM_MUL_ADD, RMS_NORM_MUL_ROPE, ADD_RMS_NORM, ADD, MUL, MUL_MAT, MUL_MAT_ID and
      fusion tests pass, and the one-token products through the compute_61 PTX; S4-L0's exported operations that this
      touches pass.
    - G3 on two cards, one token per step (`llama-perplexity -ub 1`, 4 × 128 tokens, Pascal kernels forced): per
      graph evaluation the norms write 50 + 49 rows (22 + 24 with the residual add; both dense layers' adds are already
      fused into their down-projections, and the add at the card boundary stays on its own), 123 + 121 products read
      them, and only the attention output and the down-projections quantize: 244 quantizations fewer per token, as
      counted above. Perplexity and every KLD statistic against a fusion-off base are the same with the fusion on as
      with it off, with and without CUDA graphs: the logits are identical.
  - Rig plan, on one build with the switch, S4-L0 on six cards: tg128 (`llama-bench -p 0 -n 128 -r 3`) with
    `GGML_CUDA_NORM_Q8=0` and without; a `GGML_CUDA_PROFILE` of 32 tokens, where the 43 lone `ADD l_out` groups become
    `ADD+RMS_NORM+MUL` groups (924 → 881 launch groups per token; the quantizations run inside the product groups, so
    the debug counts are the way to see them: `llama-perplexity … -ub 1 -lv 4` with `GGML_CUDA_NORM_Q8_DEBUG=1`); and
    KLD with `-ub 1` against a `GGML_CUDA_NORM_Q8=0` base, which must equal that base's own floor.
  - Expected (est.): 287 fewer launches per token on six cards (284 on nine). Each is a tiny kernel in a CUDA graph,
    ~2.5–3.5 µs with the 1.85 µs per graph node a P106 pays, so 0.7–1.1 ms per token: 26.8 → 25.7–26.1 ms, tg128
    37.4 → 38.4–39.0 tok/s (+2.7 to +4.3 %). The registry's kill line: fewer than 150 launches removed, or below +1.5 %.
- **FP32 activation sums** (fork branch `pascal-p106-fp32sums`). Q8_1 stores each 32-value block's sum `s` in FP16
  (`quantize.cu:100`), next to its scale `d`. Flint's routed expert down-projection inputs reach a per-channel RMS of
  25,600 in layer 46 and 48,400 in layer 48 (Unsloth's importance matrix; 1,100 in layer 47, at most 2,200 for every
  other matrix product's input), so `s` becomes infinite and every dot product that reads it returns inf or NaN. That is
  the overflow llama.cpp's rescale of Flint's MoE down-projections prevents, at 2.4–2.8 ms per token.
  - The Pascal kernels no longer read `s`. When a lane loads an activation block it computes the block's int8 sum with
    eight `dp4a(0x01010101, u)`, as Q5_K already did, and uses `d · Σq` in FP32 for the offsets of Q4_0 and Q5_0 and the
    mins of Q4_1 and Q5_K. The R rows of the lane group (2R with the gate) share it, so it costs no memory traffic and
    8 `dp4a` per activation block. The result is the exact dot product with the quantized activations, where the stock
    kernel combines them with the unquantized sum. `d` stays FP16; it overflows only above an activation of
    127 × 65,504 ≈ 8.3 M.
  - Batches still read FP16 sums: `mul_mat_vec_q` for 2–6 tokens (Q4_0, Q4_1, Q5_0, Q5_1; at one token too for Q5_1 and
    Q4_1_64, which the Pascal kernels don't cover) and MMQ's `DS4` layout (`mmq.cuh:20`; Q4_1, Q5_1, Q4_K, Q5_K, and Q4_0
    on GPUs without tensor cores, compute 6.1 included). The rescale therefore stays for batches, where its ~470 small
    launches are about 0.2 % of a 512-token ubatch (est.). The fork's `LLAMA_NO_MOE_DOWN_RESCALE=tg` drops it only where
    the down-projection has one token; `=1` drops it everywhere, which is safe only for Q8_0, Q6_K, IQ4_XS, IQ4_NL and
    MXFP4 down-projections.
  - The rescale divides by a power of two and multiplies back, which is exact, so with FP32 sums the one-token output is
    the same with and without it. On an RTX 5070 Laptop GPU (Flint's layers 45–48 on the GPU with the kernels forced,
    the rest on the CPU, one 512-token wikitext-2 chunk evaluated a token at a time), the makers' Q4_1 gives PPL 18.2197
    with the rescale, with `tg` and with no rescale at all; the build before the change gives NaN without it. Evaluated
    512 tokens per ubatch without the rescale, it still gives NaN (MMQ), and so does UD-Q4_K_XL (Q5_K down-projection in
    layer 46).

## 9. Verifying correctness

The ported kernel changes the order of floating-point additions, so its results differ from the stock kernel in the last
bits. The integer part of each block's dot product is exact: `(lo << 4) + hi` equals 16 times llama.cpp's `sumi`.
`test-backend-ops` compares every case against the CPU backend with a normalized error limit of 5e-4 (`test_mul_mat`,
`tests/test-backend-ops.cpp:5343-5345`).

1. **Existing cases**, on any NVIDIA GPU with the force switch, natively and through the compute_61 PTX:
   ```bash
   GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION,MUL_MAT_ID_FUSION,MUL_MAT_ID_SHARED,MUL_MAT_ADD
   ```
   ```bash
   CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_ADD -p 'type_a=(q8_0|q4_0|q4_1|q5_0),.*,n=1,'
   ```
   - On an RTX 5070 Laptop GPU all 4,899 cases pass natively, with the kernels forced on and off.
   - Through the PTX JIT the full list can't run: llama.cpp's own batched kernels (an MXFP4 MMQ case, a multi-token
     MoE case) fail under the JIT with the Pascal kernels switched off too. The one-token cases of the ported types
     (`-p` above, and `-p 'type=(q8_0|q4_0|q4_1|q5_0),.*m=1,'` for the fusion tests) all pass.
   - A deliberately wrong kernel (results × 1.1, or Q5_0's fifth bits weighted 8 instead of 16) fails 223 cases across
     `MUL_MAT`, `MUL_MAT_ID`, the GLU fusions and the routed + shared expert launch, so the tests reach the kernel.
2. **Cases at Flint's shapes**, added to `make_test_cases_eval` for Q4_0, Q4_1, Q5_0 and Q8_0: `test_mul_mat` at
   (rows, K) = (4096, 2048), (512, 2048), (2048, 4096), (2048, 1024), (6144, 2048), (2048, 6144), (4097, 2048);
   `test_mul_mat_id` with 32 experts, 6 used, 512 × 2048 with a broadcast input and 2048 × 512 with one input per
   expert; `test_mul_mat_vec_fusion` at 512 × 2048 with ids and 1024 × 2048 without. They pass natively and through
   the PTX JIT.
3. **The model's own operations.** `test-export-graph-ops` reads only a GGUF's metadata, so a sparse file with S4-L0's
   first 64 MB and its full length is enough. Its 24 distinct matrix products (one token and 512 tokens, the 150,144-row
   output head included) pass with the kernels forced on and off, the 12 one-token ones also through the PTX JIT.
   UD-Q4_K_XL's 48 pass the same way, except 3 that fail identically with and without the kernels because the replay's
   random K-quant blocks produce NaN scales (the CPU reference is NaN too).
   Replaying the other exported operations fails on a `GET_ROWS` with random indices, with or without the kernels.
   ```bash
   build/bin/test-export-graph-ops -m flint-s4-l0.gguf -o flint-s4-l0-ops.txt
   grep -E '^(29|30) ' flint-s4-l0-ops.txt > flint-s4-l0-matmul.txt   # GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID
   GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 --test-file flint-s4-l0-matmul.txt
   ```
4. **On a P106-090.** The one-token cases of item 1 pass without the force switch for all six types (Q8_0 30 and 50
   fusion cases, Q4_0 29 and 42, Q4_1 30 and 42, Q5_0 14 and 2, Q5_K 15 and 2, IQ4_XS 13 and 2).
5. **Output of a real model.** S4-L0 on 6 cards, 4 × 512 tokens of wikitext-2 evaluated one token per ubatch
   (`llama-perplexity -b 512 -ub 1`), so that every token goes through the one-token kernels; a plain perplexity run
   uses the batched kernels and never reaches them. The Pascal kernels differ from llama.cpp's kernel exactly as much
   as two launch shapes of llama.cpp's kernel differ from each other, which is pure summation order:

   | Pair | Mean KLD | Same top token | Perplexity |
   |---|---|---|---|
   | stock, 2 warps against stock, 4 warps | 0.067 | 92.9 % | 22.30 against 22.41 |
   | Pascal kernels against stock, 4 warps | 0.062 | 92.8 % | 22.56 against 22.41 |
   | Pascal kernels against stock, 2 warps | 0.075 | 92.1 % | 22.56 against 22.30 |

   UD-Q4_K_XL, with the Q5_K and IQ4_XS kernels, the same way:

   | Pair | Mean KLD | Same top token | Perplexity |
   |---|---|---|---|
   | stock, 2 warps against stock, 4 warps | 0.060 | 92.6 % | 22.67 against 23.26 |
   | Pascal kernels against stock, 4 warps | 0.062 | 92.3 % | 23.26 against 23.26 |
   | Pascal kernels against stock, 2 warps | 0.053 | 92.5 % | 23.26 against 22.67 |

   Flint's outputs move this much from rounding alone, because small differences flip the 6th-ranked expert; the same
   file on two GPU types differs by KLD 0.065 / 91.3 % ([experiments.md](../../../quants/experiments.md), Q21).
6. **Activations past FP16's range** (FP32 activation sums, section 8). `test_mul_mat_large_act` runs one-token
   `MUL_MAT` (2048 rows, K = 1024 and 4096) and `MUL_MAT_ID` (6 of 32 experts, K = 512) with one activation of 250,000
   alone in its block and a whole block at ±10,400 in every row. It checks the result against a double-precision
   product computed on the host, because the CPU backend's own Q8_1 sums (used for Q4_1) overflow too, and it runs only
   where the CUDA backend reports the `MMVQ_PASCAL` feature. Before the change Q4_0, Q4_1 and Q5_0 returned NaN or inf;
   with it all six types pass, natively and through the PTX JIT. With the change, all 4,939 `MUL_MAT*` cases of item 1
   pass natively on the laptop, and the one-token cases and S4-L0's and UD-Q4_K_XL's replayed products pass as before.

## 10. Gain per token

**Measured.** llama-bench (`-fa on -p 512 -n 128 -r 3`) on the fork build, one binary with `GGML_CUDA_MMVQ_PASCAL=0`
(llama.cpp's kernel with the wide-rows launch shape) and `=1` (the Pascal kernels):

| File | Cards | Stock kernel | Pascal kernels | Gain |
|---|---|---|---|---|
| S4-L0 (rescale off) | 6 | 29.20 tok/s | **37.37** | +28 % |
| makers' Q4_1 (rescale on) | 5 | 32.20 | **39.59** | +23 % |
| UD-Q4_K_XL (rescale on) | 6 | 24.49 | **34.06** | +39 % |
| Qwen3.6-35B-A3B Q4_0 (bartowski) | 4 | 22.79 | **29.04** | +27 % |
| Unsloth's Q8_0 (rescale off) | 9 | 26.92 | **35.89** | +33 % |

Prompt processing is unchanged (it runs on MMQ). On S4-L0 the port without Q5_0 gives 35.51 tok/s; UD-Q4_K_XL with only
its Q8_0 part ported gave 27.48, and stock llama.cpp gives it 17.56, so Unsloth's file now runs 1.94× as fast. Its
profiled GPU time per token goes from 55.8 to 40.3 ms: Q8_0 20.5 → 14.6, Q5_K 11.8 → 6.2, IQ4_XS 7.1 → 3.3 ms.

**Where it comes from.** GPU time per S4-L0 token from the fork's event profiler (`GGML_CUDA_PROFILE`, 24 tokens, sum
over the 6 cards). Every launch carries 5–8 µs of event overhead in these numbers, the same for both columns, so the
differences are exact and the totals are higher than the unprofiled token time:

| Tensors | Launches per token | Stock µs/launch | Pascal µs/launch | Stock ms | Pascal ms |
|---|---|---|---|---|---|
| attention q (Q4_1) | 49 | 75.7 | 54.7 | 3.71 | 2.68 |
| attention k, v (Q8_0) | 98 | 25.7 | 21.3 | 2.52 | 2.09 |
| attention output, layers 0–32 (Q8_0, + residual add) | 33 | 114.6 | 81.6 | 3.78 | 2.69 |
| attention output, layers 33–48 (Q4_1, + residual add) | 16 | 71.6 | 58.3 | 1.15 | 0.93 |
| routed experts gate/up (Q5_0, fused) | 47 | 174.4 | 106.6 | 8.20 | 5.01 |
| routed experts down (Q8_0) | 47 | 110.9 | 63.0 | 5.21 | 2.96 |
| shared expert gate, up (Q8_0, unfused) | 94 | 36.6 | 28.3 | 3.44 | 2.66 |
| shared expert down (Q8_0, + add) | 47 | 55.6 | 36.2 | 2.61 | 1.70 |
| dense FFN, layers 0–1 (Q8_0) | 5 | 200.5 | 124.9 | 1.00 | 0.62 |
| output head (Q8_0) | 1 | 3289.6 | 2196.9 | 3.29 | 2.20 |
| router (F16, stock kernel) | 47 | 25.8 | 25.2 | 1.21 | 1.18 |
| **matrix products** | | | | **36.12** | **24.73** |
| everything else (norms, attention, router top-k, adds, copies) | | | | 8.05 | 7.96 |
| **total GPU time per token** | | | | **44.17** | **32.70** |

**Against the benchmark.** The real calls are slower than the benchmark's at the same shapes and speeds, by more than
the event overhead: the ported attention q by ~13 µs, the routed down-projection by ~25 µs, the routed gate/up by
~33 µs, the output head by 11 %. The routed experts lose the most; their weights are spread over 285–370 MB per layer
and card, against the benchmark's few copies.

**Wall time against GPU time.** The kernels cut the profiled GPU time by 11.4 ms per token, but llama-bench's token
time by 7.5 ms (Q49). The host is not the limit within a token: `GGML_CUDA_PROFILE_HOST` shows it launching all six
cards' graphs within ~1.9 ms of the token's start, while each card computes for 4–8 ms. CUDA events themselves change
the timing on these cards (even 12 per token slow S4-L0 from 37.4 to 23.4 tok/s), so event timings compare kernels
with each other but don't give absolute token time; what the unprofiled run overlaps is still open.

**What it changes for the file side.** With the Pascal kernels Q8_0, Q5_0 and Q4_1 decode at about the same bandwidth
(150, 138 and 145 GB/s on the large shapes in the benchmark), so decode time follows bytes: Q8_0 costs 1.7× Q4_1 per
parameter (34 against 20 bytes per 32 weights) and Q5_0 1.1×, where the stock kernel charged Q5_0 nearly Q8_0's price
(Q45). The efficiencies in [scripts/token_cost.py](../../../scripts/token_cost.py) should be refitted to the fork build,
and the per-layer allocations of Q36 and Q45 redone with them.

## 11. Implementation plan

| Step | Work | Done when | Status |
|---|---|---|---|
| 0 | Time every kernel of one real S4-L0 token | the real per-call times are known next to the benchmark's | done: nvprof and Nsight Systems refuse the P106 (a CMP mining card), so the fork times launch groups with CUDA events itself (`GGML_CUDA_PROFILE`, section 10) |
| 1 | Benchmark additions: Q5_0 (stock, 2 warps, tuned with the fifth-bit dot product), row-local planes, Q5_K and IQ4_XS (Q46) | Q5_0 tuned ≥ 1.5× stock on the gate/up shape; row-local planes within 3 % of tensor-wide planes | done: Q5_0 2.2× on the gate/up shape; row-local planes 3 % faster than tensor-wide ones; staged loads (f) added and as fast as planes |
| 2 | Port as `mmvq-pascal.cu` for Q4_0, Q4_1 and Q8_0: decision point, force switch, `MUL_MAT` and one-token `MUL_MAT_ID` | `test-backend-ops` passes with the force switch, native and through the PTX JIT | done |
| 3 | Fusions: gate + GLU, biases, `shared_up` | `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_FUSION`, `MUL_MAT_ID_SHARED`, `MUL_MAT_ADD` pass | done |
| 4 | Tests at Flint's shapes, S4-L0's exported operations | all pass | done; bit-identity against the standalone kernels not checked (both pass the CPU references) |
| 5 | Rig: `test-backend-ops` on a P106; tg128 for S4-L0 and other files; KLD against the stock kernels | S4-L0 ≥ 15 % faster than the wide-rows build at unchanged output | done: S4-L0 +33 % (37.4 against 28.0 tok/s), output within summation-order noise (section 9); other files in section 10 |
| 6 | Q5_0 in the port | tests pass; S4-L0 tg128 measured again | done: 35.5 → 37.4 tok/s |
| 7 | Q5_K and IQ4_XS in the port, from the benchmark's kernels (Q46) | tests pass; UD-Q4_K_XL measured | done (section 10) |
| 8 | Q4_1_64 in the port (fork only) | tests pass; makers' file in Q4_1_64 measured | |
| 9 | One activation quantization per shared input | 145 fewer launches per Flint token; tests and KLD unchanged | done by the norm (Q100, branch `pascal-p106-v2-normq8`, section 8): 287 fewer launches per S4-L0 token on six cards, also the residual adds; tests and logits unchanged on an RTX 3090; rig speed pending |
| 10 | FP32 activation sums in the Pascal kernels; drop the rescale where safe | makers' Q4_1 runs without the rescale, without NaN, at unchanged KLD | Pascal kernels done (branch `pascal-p106-fp32sums`); batches keep the rescale (`LLAMA_NO_MOE_DOWN_RESCALE=tg`), MMQ's `DS4` layout unchanged; on the laptop no NaN and output identical to the rescaled one (section 8); rig speed and KLD pending |
| 11 | 2–4 tokens per step for `MUL_MAT` and `MUL_MAT_ID`, each distinct expert read once (Q78) | tests pass; verify-step cost of S4-L0 measured on the rig | kernels and laptop tests done (branch `pascal-p106-v2-multicol`, section 14); rig pending |
| 12 | Q6_K in the port, 1–4 tokens (Q101) | tests pass; ≥ 130 GB/s on the dense K = 2048 shapes in the microbenchmark; then a Q6_K mix (Q90) on the rig | kernels and tests done (branch `pascal-p106-v2-q6k`, section 15); rig pending |

Steps 9 and 10 are independent of the kernel and can go in any order. The repack of section 7 (Q39) is not planned:
staged loads made it unnecessary for decode speed.

## 12. Upstream PRs

Each of these stands alone and can be reviewed separately:

1. **Launch shape for compute 6.1.** The `MMVQ_PARAMETERS_PASCAL_DP4A` table with 2 warps at one token, keeping 4 warps
   and the small-K shape wherever llama.cpp's small-K rule applies.
   - On S4-L0 this gives 28.02 tok/s, against 26.29 for 2 warps at every width and 23.09 for the standard build; on
     UD-Q4_K_XL 24.58 against 19.53.
   - These are the data PR #25479 lacks. Small, and the most likely to be accepted.
2. **The Pascal kernel** for Q4_0, Q4_1, Q5_0 and Q8_0 (steps 2–4 and 6 without Q4_1_64).
   - A new file, called for compute 6.1, which falls back to the stock kernel for everything else, with the Flint-shape
     test cases. Upstream already carries device-specific MMVQ tables (Turing, RDNA, GB10).
   - On S4-L0 it gives 37.4 against 29.2 tok/s on the same binary.
   - It needs measurements on at least one larger compute 6.1 card (a GTX 10-series card or a P40). Those cards have
     2–4× less bandwidth per SM than the P106-090 and are less limited by instruction issue, so the PR must show no
     regression there, or restrict itself by SM count.
3. **One activation quantization per shared input.** Not specific to any architecture, and continues mainline's
   gate/up de-duplication.
4. **FP32 activation sums.** Closes the overflow that the Flint, Mistral4 and Mellum rescale works around. Inside the
   Pascal kernel it is local (the sums come from the int8 values, section 8); as an upstream change it competes with the
   open BF16 proposal (#22571) and touches every Q8_1 reader, MMQ included.

Q4_1_64 stays in the fork: it is a new GGUF type. In the discussion of an earlier CUDA repack PoC (#20563), the
maintainers asked for a baseline without repacking first; PR 2 is that baseline, and on this card it already reaches the
speed of the plane layouts.

## 13. Risks

- **The benchmark predicts the direction but not the exact size.** In the real token the ported calls take 13–35 µs
  longer than in the benchmark at the same shapes (section 10), and the wall time per token falls less than the GPU
  time (Q49). Kernel work should be judged on llama-bench, not on the benchmark alone.
- **Profiling perturbs the token.** CUDA events slow these cards' tokens markedly (Q49), so kernel savings measured with
  the event profilers overstate the wall-clock gain (11.4 ms GPU time for 7.5 ms on S4-L0).
- **Register pressure.** Fused Q8_0 at R = 2 reached 254 registers with direct loads; the port uses R = 1 for fused
  Q8_0 and Q5_0. Staged shared memory (17 KB per block for Q8_0) limits a P106-090 SM to 5 such blocks.
- **Fallback coverage.** Widths that don't split into whole chunks, unaligned views and more than four tokens keep the
  stock kernel. That is correct but slower; tail handling in the last iteration can widen coverage later.
- **Maintenance.** The kernel lives in its own file, and the hook in `mmvq.cu` is an include and one call, so rebasing
  touches little. The GLU epilogue exists three times (stock kernel, single-token and multi-token Pascal kernels) and
  must follow upstream changes to the GLU ops.

## 14. Several tokens per step (Q78)

A speculative-decoding verify step, the last few tokens of a prompt and two-slot serving multiply 2–8 tokens by every
weight matrix. The kernels of sections 2–13 take one token; wider products fell back to llama.cpp's `mul_mat_vec_q`
(dense) and `mul_mat_vec_q_moe` (experts, one warp per token and expert slot). On S4-L0 (6 cards) a 4-token step then
spent 52 % of its GPU time in the stock MoE kernels, and a verify step of 3 and 4 tokens cost 2.2× and 3.6× a plain
token (Q19). The fork's branch `pascal-p106-v2-multicol` extends the kernels to 2–4 tokens.

### Design

- **The same kernel with a token loop inside.** `mul_mat_vec_q_pascal_mc` keeps the single-token kernel's lane groups,
  launch shapes (`mmvqp_choose`) and staging. In every iteration a lane group stages its weight segment once, then loads
  one token's activations after another and multiplies them by each row's chunk. The compiler keeps the chunks in
  registers across the tokens (the shared-memory loads per iteration are the same at 4 tokens as at 1 in the sm_61
  SASS), so every weight row is read once per step; activations are read once per token and iteration and shared by
  the rows of the lane group.
- **Experts: one read per distinct expert.** For `MUL_MAT_ID` the grid has one block per (token, slot) entry of `ids`,
  plus one for the shared expert in the routed + shared launch. The block whose entry is the first to pick an expert
  serves that entry and every later one that picked the same expert, up to 4 entries in all (a fifth starts a new
  group, which happens only when a token picks an expert more than once); every other block finds an earlier entry
  with its expert and exits before loading anything. Each block works this out from `ids` (24 entries for Flint at 4
  tokens), so there is no extra kernel or host step, and the launch stays capturable in a CUDA graph.
  - The products are bandwidth-bound, so the bytes decide. Flint's 2 and 4 consecutive tokens touch 11.1 and 20.1
    distinct experts per layer (Q51), so the expert bytes of a step fall by 7.5 % and 16 % against reading 6 experts
    per token. When no expert is shared, the kernel does exactly what one launch of single-token products would.
  - A block serves 1.08 (2 tokens) to 1.19 (4 tokens) tokens on average, so the expert products keep the single-token
    instruction cost per byte.
- **The same arithmetic per token.** Each token's dot products run the single-token code with the same lane mapping and
  summation order, so each column of a 2–4-token product equals the single-token product of that column bit for bit.
  A verify step's matrix products therefore reproduce plain decoding's; attention over several query rows and the
  F16 router still run llama.cpp's batched kernels.
- **What carries over.** Gate + GLU fusion, biases, the routed + shared expert launch and the FP32 activation sums of
  section 8 work at 2–4 tokens as at one. `LLAMA_NO_MOE_DOWN_RESCALE=tg` is unchanged: it drops the rescale only for
  one-token batches.
- **Limits.** Up to 4 tokens (`MMVQP_MAX_COLS`): 5–8 tokens still run llama.cpp's kernels, and on compute 6.1 the
  expert products go to MMQ above 4 tokens for Q8_0, above 5 for Q5_K and IQ4_XS and above 6 for Q4_0, Q4_1 and Q5_0
  (`get_mmvq_mmid_max_batch`). llama.cpp fuses dense gate/up products only at one token, so at 2–4 tokens the shared
  expert's gate and up are two products. Widths that don't split into whole chunks keep the stock kernel, as at one
  token.

### Instruction cost

Main loop of each kernel in the sm_61 SASS, lane-instructions per weight byte at the shapes `mmvqp_choose` picks
(4 tokens = all four served), with the issue limit this implies (1.107 T lane-instructions per second, ~83 % usable,
see [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md#expected-results)) against the card's ~173 GB/s:

| Type, product | Registers, single- / multi-token kernel | Instructions per byte, 1 / 4 tokens | Issue limit at 4 tokens |
|---|---|---|---|
| Q8_0, dense | 102 / 113 | 1.2 / 2.6 | ~350 GB/s |
| Q8_0, fused gate/up | 95 / 116 | 1.2 / 2.7 | ~350 GB/s |
| Q4_1, dense | 118 / 176 | 1.9 / 5.1 | ~180 GB/s |
| Q4_0, dense | 86 / 116 | 2.5 / 6.2 | ~150 GB/s |
| Q5_0, dense | 90 / 151 | 3.6 / 8.9 | ~100 GB/s |
| Q5_0, fused gate/up | 84 / 148 | 3.6 / 9.1 | ~100 GB/s |
| Q5_K, dense | 142 / 197 | 3.2 / 7.0 | ~130 GB/s |
| IQ4_XS, dense | 96 / 193 | 5.3 / 9.7 | ~95 GB/s |

- Q8_0 and Q4_1 stay far from the issue limit at 4 tokens. Dense Q5_0, Q5_K and IQ4_XS products reach it, which matters
  little for Flint's files: their Q5_0 is expert gate/up only (1.2 tokens per expert block), and UD-Q4_K_XL's Q5_K and
  IQ4_XS are mostly experts too.
- The other open cost is the activation loads. Every lane loads its own activation blocks for each token at a stride of
  72 bytes (two blocks per lane for Q8_0, Q5_0 and Q4_0), so at 4 tokens the kernel issues as many strided loads per
  iteration as the single-token kernel's direct-load variant (c) issued for its weights, which made Q8_0 36 % slower
  than staged loads in the microbenchmark (23.2 against 17.1 ms per token). Those activation loads hit L1 and L2, which
  may make them cheaper than (c)'s.
- `GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1` selects a variant that copies each token's activation segment into shared memory
  with coalesced 128-bit loads, for every shape where it fits next to the staged weights (all but Q5_K and the staged
  IQ4_XS shape). Q8_0 then issues 30 global loads per iteration instead of 46, at 3.9 instructions per byte instead of
  2.6 and 26 KB of shared memory per block instead of 17 KB (3 blocks per SM instead of 4). Its results are the same bit
  for bit. It is off by default; the rig decides.

### Tests

On an RTX 5070 Laptop GPU with the kernels forced (`GGML_CUDA_MMVQ_PASCAL=2`):

- `test-backend-ops` passes all 5,227 `MUL_MAT`, `MUL_MAT_ID`, `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_FUSION`,
  `MUL_MAT_ID_SHARED` and `MUL_MAT_ADD` cases, with and without staged activations, and all 5,155 with the kernels off
  (the 72 large-activation cases run only where the kernels are active). The Flint-shape products, expert products,
  fused expert gate/up and large-activation products of section 9 now run at 1–4 tokens; the routed + shared expert
  launch and repeated expert ids (mostly expert 0, so one expert serves up to 30 entries of a step) run at 2–4 tokens,
  for the six types.
- Through the compute_61 PTX (`CUDA_FORCE_PTX_JIT=1`) the 1–4-token cases of the six types pass (622 products, 290
  fusions, 22 routed + shared launches), with and without staged activations.
- Mutations reach the new code: results × 1.1 in the multi-token kernel fail 602 cases, every one of them a 2–4-token
  case of the six types (the 10 that still pass have K = 32 or 96, which the kernels leave to llama.cpp). Corrupting
  only the second to fourth token a block serves fails 121 of the 126 two-to-four-token `MUL_MAT_ID` cases, so the
  tests do exercise experts shared between tokens. A wrong lane offset in the staged-activation path fails 475 cases
  with the switch on and none with it off.
- S4-L0's and UD-Q4_K_XL's own products at 2, 3 and 4 tokens per step, exported with `test-export-graph-ops -ub N`
  from the sparse header copies (13 and 24 distinct products: attention, shared expert, dense FFN, router, experts and
  output head), pass with the kernels forced and off; the quantized ones also through the PTX (the F16/BF16 router's
  batched kernel has no compute_61 code, with or without these kernels).
- [bench/mmvq_pascal/verify/multitoken_bitid.cpp](../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp) compares
  every 2–4-token product with the single-token products of its columns (dense products at five shapes, expert
  down-projection, fused expert gate/up + SwiGLU, six types: 126 comparisons). All are identical, natively and through
  the PTX, with and without staged activations; llama.cpp's own kernels differ in 84 of them.
- The single-token kernels compile to the same SASS as on `pascal-p106-v2`.

On an RTX 3090 (sm_86) of the 7-GPU quant machine, with the kernels forced: the same 1–4-token cases of the six types
(622 products, 290 fusions, 22 routed + shared launches), the exported S4-L0 and UD-Q4_K_XL products at 2–4 tokens and
the bit-identity check (126 of 126 identical) all pass, natively and through the compute_61 PTX, with and without
staged activations; with the kernels off the products, fusions and exported products pass as well.

### Expected effect on a verify step

A model, to be replaced by the rig's numbers. GPU time of one step of S4-L0 on 6 cards, summed over the cards, without
the profiler's ~6.5 µs per launch; inputs: the one-token profile of section 10, Q19's profile of a 4-token step with
llama.cpp's kernels, Q51's expert counts, and a per-byte slowdown of the dense products between none and 1.4× at 4
tokens (variant (c)'s penalty, the pessimistic case for the strided activation loads):

| Tokens per step | GPU time with the Pascal kernels | Ratio to one token |
|---|---|---|
| 1 | 26.1 ms | 1.00 |
| 2 | 34–38 ms | 1.30–1.44 |
| 3 | 40–47 ms | 1.55–1.80 |
| 4 | 47–56 ms | 1.79–2.14 |

- With llama.cpp's kernels the 4-token step of Q19's profile took 74.5 ms (2.9× a token).
- At 4 tokens: experts 24.7–28.4 ms (a token's 7.4 ms × 3.35 for the distinct experts × up to 1.15), dense products
  13.4–18.7, the F16 router 2.2 and everything else 6.5. The experts dominate: with 256 experts and little locality,
  4 tokens read 3.35× a token's expert bytes whatever the kernel does.
- Perfect drafts could at most gain n × 26.1 ms / v(n) on the GPU side: 1.4–1.5× at 2 tokens, 1.7–2.0× at 3 and
  1.9–2.2× at 4. The server's step adds host time on top: Q19's steps of 4.05 tokens took 92.1 ms against ~75 ms of GPU
  time for 4 tokens. If those ~15–20 ms stay, a 4-token step takes ~65–75 ms (2.5–2.9× a plain token instead of
  3.55×), and perfect 3-token drafts give at most ~1.4–1.6× instead of 1.14×. Real acceptance is lower; Q20 and Q53
  depend on the measured v(n).

### Measuring it on the rig

Build: `TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v2-multicol`, unpacked to `/opt/llama-builds/` on the
rig. Below, `B` is its `bin` directory and `M` the S4-L0 file; S4-L0 runs with `LLAMA_NO_MOE_DOWN_RESCALE=1`.

1. **Kernel times**, one card, three runs (llama.cpp's kernels, the Pascal kernels, and with staged activations):
   ```bash
   ssh rig "BIN=$B bash -s" < bench/mmvq_pascal/multitoken_perf.sh
   ```
   ```bash
   scp -r rig:/tmp/mmvq_multitoken . && bench/mmvq_pascal/multitoken_perf_table.py mmvq_multitoken
   ```
   Expected: dense Q8_0 products at 4 tokens cost 1.0–1.4× one token ("on x1"), the expert products ~3.3×; the
   variant that wins becomes the default.
2. **Correctness on a P106** (the kernels run without the force switch; the CPU references take a while on the rig's
   two cores):
   ```bash
   ssh rig "$B/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID -p 'type_a=(q8_0|q5_0|q4_1|q5_K|iq4_xs),type_b=f32,(m=(4096|512|2048|4097),n=[24],k=(1024|2048|4096),bs|n_mats=32,n_used=6,b=[01],m=(512|2048),n=[24],)'"
   ```
   and the same with `GGML_CUDA_MMVQ_PASCAL_STAGE_ACT=1`.
3. **Step cost of S4-L0**, 6 cards, with the kernels on and off (`GGML_CUDA_MMVQ_PASCAL=0`), and with staged
   activations if step 1 favours them: llama-bench pp1–pp8 and tg64, and the server with synthetic full acceptance at
   1–4 tokens per step (every step stays within the multi-token kernels' 4 tokens; Q19 measured widths 3 and 5):
   ```bash
   ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M GPUS=0,1,2,3,4,5 PARTS=bench TAG=q78_on bash -s" < bench/eagle3_flint/verify_cost.sh
   ```
   ```bash
   ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 BIN=$B MODEL=$M GPUS=0,1,2,3,4,5 PARTS=synth WIDTHS='1 2 3 4' CTX=8192 TAG=q78_synth_on bash -s" < bench/eagle3_flint/verify_cost.sh
   ```
   Every width reloads the model (~7 minutes from a slow USB stick).
4. **Where the time goes** in a 4-token step, against Q19's profile:
   ```bash
   ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 GGML_CUDA_PROFILE=/tmp/q78-prof-pp4.csv $B/llama-bench -m $M -ngl 99 -fa on -ub 128 -p 4 -n 0 -r 24"
   ```
5. **Output** with 4-token batches on real text: `llama-perplexity -c 512 --chunks 4 -b 512 -ub 4` with the kernels on,
   against a KL-divergence base saved from the same command with the kernels off; expect the summation-order noise of
   section 9 (mean KLD ~0.06–0.08).
6. UD-Q4_K_XL (Q5_K and IQ4_XS kernels) with `PARTS=bench`, kernels on and off.

## 15. Q6_K (Q101)

Q90 found that G3 with every Q8_0 tensor except the head in Q6_K costs +0.014 KLD (0.118 against 0.104) for ~0.26 GB
less per token, so a Q6_K kernel decides whether that saving becomes speed. Without one, Q6_K runs llama.cpp's
kernel, which reads its 2-byte-aligned blocks with 16-bit loads. The fork's branch `pascal-p106-v2-q6k` (on top of
`pascal-p106-v2-multicol`) adds Q6_K to the Pascal kernels for 1–4 tokens, `MUL_MAT` and `MUL_MAT_ID`, including the
gate/GLU fusion and the routed + shared expert launch. The microbenchmark has the same kernel as `q6_K` / `st_r1c1`.

- **Layout.** A super-block of 256 weights is 210 bytes: 128 bytes `ql` (low 4 bits), 64 bytes `qh` (high 2 bits), 16
  int8 scales (one per 16 weights) and an FP16 `d`. It is only 2-byte aligned, so Q6_K has no direct-load shape.
- **Staging.** One super-block per lane, one row per lane group, as Q5_K. The lane group stages its segment with
  16-byte loads where the rows and the segment allow it (K a multiple of 2,048: Flint's attention, shared expert gate/up
  and head) and with 8- or 4-byte loads where they are only 8- or 4-byte aligned (K = 1,024: the shared expert's down;
  K = 512: an expert's down). A lane whose block starts at an odd half-word in shared memory reads the 53 words around
  it and shifts them by 2 bytes, one `PRMT` per word.
- **Fused gate/up.** Staging both matrices of a row would take 54 KB of shared memory per block, over Pascal's 48 KB,
  so the kernel stages the up segment, multiplies, then stages the gate segment into the same buffer: 27 KB per block,
  3 blocks per SM.
- **Arithmetic.** A 6-bit code is a nibble of `ql` OR-ed with a 2-bit field of `qh` moved into bits 4–5 (masks and one
  shift), one dp4a per 4 weights. The offset −32 comes from the activations' half-block sums (32 × the sum of 16 int8
  values, computed once per activation load and shared by all rows) instead of llama.cpp's second dp4a per word with
  `0xE0E0E0E0`. The sums come from the int8 values, so Q6_K is safe from the FP16 overflow of section 8 as well.
- **Cost.** In the sm_61 SASS the main loop needs 3.2 lane-instructions per weight byte for a plain product and 2.9
  for the fused gate/up (Q5_K: 3.1 and 2.8, measured at 120–128 GB/s), 3.5 and 3.9 with 8- and 4-byte staging. The
  single-token kernel takes 132–138 registers (Q5_K 142), without spills. Expected 120–135 GB/s. At 125 GB/s Q6_K
  would save ~0.5 ms per token over Q8_0 at 150 GB/s on G3-6's 0.875 GB, at 140 GB/s ~1.3 ms (the research note's
  estimate).
- **Unchanged.** The other types' kernels compile to the same SASS as on `pascal-p106-v2-multicol` (155 of 155).
- **Tests.** Q6_K joins the Flint-shape cases of sections 9 and 14 (1–4 tokens, expert products, fused gate/up, large
  activations, routed + shared, repeated expert ids).
  - On an RTX 3090 with the kernels forced, natively (sm_86) and through the compute_61 PTX, with staged activations
    off and on: the 1–4-token cases of the seven types pass (697 products, 295 fusions, 25 routed + shared launches),
    as do S4-L0's and UD-Q4_K_XL's exported products at 2–4 tokens (UD-Q4_K_XL's own Q6_K expert down-projections
    included) and the bit-identity check (147 comparisons, Q6_K included).
  - With Q6_K's result scaled by 1.1, 57 of its 75 products fail with the kernels forced (the other 18 have K = 256,
    whose 210-byte rows the kernels leave to llama.cpp), all 8 fused and routed + shared cases fail, and so does
    UD-Q4_K_XL's Q6_K product in each replay; with the kernels off everything passes.
  - In the microbenchmark every Q6_K case matches the CPU reference at all eight shapes.
- **Rig.** The microbenchmark gate first (≥ 130 GB/s on the dense K = 2,048 shapes; command in
  [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md#q6_k-q101)). If it passes, llama-bench tg128 of a G3-6 mix
  (G3 with its non-head Q8_0 tensors in Q6_K) against G3, kernels on, on the same cards.

## 16. The combined v2 test build (Q117)

The fork's branch `pascal-p106-v2-combined` (off `pascal-p106-v2`) carries every v2 engine piece that waits for the rig,
cherry-picked in this order with linear history, so one build measures their total and, piece by piece through the
switches below, each one's share:

| Piece | From branch | Acts on | Switch |
|---|---|---|---|
| Q108: MoE tile list for MMQ (llama.cpp PR #31047) | `pascal-p106-v2-prefill` | prompt micro-batches | off: `GGML_CUDA_MMQ_MOE_LEGACY=1` |
| Q109: graph reuse only with nothing in flight | `pascal-p106-v2-prefill` | prompt micro-batches at `-ub 128`, pipeline parallelism | off: `LLAMA_GRAPH_REUSE_WAIT=1` |
| Q110: attention mask built on each card | `pascal-p106-v2-gpumask` | micro-batches of ≥ 32 tokens | off: `LLAMA_GPU_KQ_MASK=0` |
| Q100: the norm writes the Q8_1 copy (section 8) | `pascal-p106-v2-normq8` | one-token steps | off: `GGML_CUDA_NORM_Q8=0` |
| Q92: screened output head | `pascal-p106-v2-screenhead` | batches with one output row | on: `LLAMA_SCREEN_HEAD=1` and the side file (`LLAMA_SCREEN_HEAD_FILE`) |
| Q78: 2–4 tokens in the Pascal kernels (section 14) | `pascal-p106-v2-multicol` | 2–4-token steps | off: `GGML_CUDA_MMVQ_PASCAL_MC=0` |
| Q87/Q96/Q104: pipelined executor | `pascal-p106-v2-pipespec` | `llama-pipespec` only | — |

- **Merging.** Only multicol and normq8 overlap: in `mmvq.cu` the Pascal call takes multicol's arguments and normq8's
  Q8_1 pointer, which is never more than one row, so 2–4-token steps quantize as before; and the test list keeps both
  pieces' cases. The executor's changes act through `llama_pipe_*` and `llama_sampler_set_pos`, except the KV cache
  placing re-decoded tokens of a removed tail on the cells they had (llama-server's speculative rollbacks), which moves
  cells, not results.
- **The three switches the pieces lacked** are one commit on the branch:
  - `GGML_CUDA_MMQ_MOE_LEGACY=1` runs `MUL_MAT_ID` in MMQ as before PR #31047. The tile width is chosen against all
    tokens again, and the ids helper writes the old tiles: ceil(tokens / width) for every expert from its first
    column, the ones past its last column empty and returning at once.
  - `LLAMA_GRAPH_REUSE_WAIT=1` reuses a graph while a micro-batch is in flight, after the full wait.
  - `GGML_CUDA_MMVQ_PASCAL_MC=0` sends 2–4 tokens to llama.cpp's kernels; one token keeps the Pascal kernels.
  - `GGML_CUDA_MMVQ_PASCAL=0` still turns every Pascal matrix-vector kernel off, one token included.
- **Checked** on a 7-GPU quant machine (RTX 3090s, correctness only):
  - `test-backend-ops`, native sm_86 code: `MUL_MAT_ID` (681 cases, our types plus Q4_K/F16/BF16/F32, every batch
    size) with the tile list and with `GGML_CUDA_MMQ_MOE_LEGACY=1`, `MUL_MAT_ID_FUSION` (13), `MUL_MAT` (1,726),
    `KQ_MASK` (20), `ADD_RMS_NORM_MUL_Q8` (32, with the Pascal kernels forced, with and without the norm's Q8_1 copy),
    the screened head's top 256 of 150,144 and its one-row-expert product, and the 1–4-token cases of section 14 (623
    products, 290 fusions, 22 routed + shared launches) with the Pascal kernels forced and staged activations off and
    on: all pass. The same holds with a build that has only compute_61 PTX (the host then takes the Pascal decisions
    too), Q8_0 `MUL_MAT_ID` without its 5–8-token cases (prefill note, "E3 and E4 built").
  - With the stock kernels (`GGML_CUDA_MMVQ_PASCAL=0`), `ADD_RMS_NORM_MUL_Q8`'s Q4_1 cases exceed the test's error
    limit (5 × 10⁻⁴, up to 1.2 × 10⁻³) in about two runs of five, each on fresh random inputs, natively and through the
    PTX, with the norm's Q8_1 copy on and off alike: llama.cpp's Q4_1 kernel with its FP16 block sums against a tight
    limit, not the combination. The Pascal kernels pass every run.
  - With `GGML_CUDA_MMVQ_PASCAL_MC=0` the same 1–4-token cases pass except the 2–4-token large-activation ones of
    Q4_0, Q4_1 and Q5_0 (27 of 623), which give inf or NaN. That is `pascal-p106-v2` itself: llama.cpp's kernels keep
    the Q8_1 block sums in FP16 (section 8), and only the Pascal kernels sum in FP32. The `MC=0` arm on the rig
    therefore runs with Flint's MoE down rescale on (no `LLAMA_NO_MOE_DOWN_RESCALE=1`), in both of its runs.
  - G3 split by layer over two cards, llama-perplexity over 8 × 512 tokens at `-ub 128` and `-ub 512` (Pascal kernels
    forced): the combined build at its defaults and with every off switch set writes reference logits byte-identical
    to `pascal-p106-v2`'s, and at its defaults its KLD statistics against the base equal the base's own floor (maximum
    6 × 10⁻⁵, the same top token everywhere).
  - Decoding G3 (Pascal kernels forced): greedy llama-completion of 8 prompts × 128 tokens gives the base's text
    byte for byte, at the defaults and with every switch off; llama-screen-check passes (262,144 candidate logits
    bit-identical to the full head's, no prompt diverging); llama-pipespec's pipelined decoding (depth 3) gives plain
    decoding's 96 tokens and logit hashes.
- **Measuring it on the rig.** `CB` is the combined tarball's `bin/`, `V2` the `pascal-p106-v2` tarball's, `MODEL`
  S4-L0 (then G3h/K2h), six cards for decode as in the other engine A/Bs, nine for prompts, `SCREEN` the screen side
  file:

  ```bash
  export LLAMA_NO_MOE_DOWN_RESCALE=1
  OFF="GGML_CUDA_MMQ_MOE_LEGACY=1 LLAMA_GRAPH_REUSE_WAIT=1 LLAMA_GPU_KQ_MASK=0"
  OFF="$OFF GGML_CUDA_NORM_Q8=0 GGML_CUDA_MMVQ_PASCAL_MC=0"
  TG="-m $MODEL -ngl 99 -fa on -p 0 -n 128 -r 3 -o jsonl"
  PP="-m $MODEL -ngl 99 -fa on -p 512,2048,8192 -n 0 -b 8192 -ub 128,256,512 -r 2 -o jsonl"
  # 1. decode, six cards: V2, all off (= V2), all on, screened head on top, then one piece off at a time
  $V2/llama-bench $TG; env $OFF $CB/llama-bench $TG; $CB/llama-bench $TG
  LLAMA_SCREEN_HEAD=1 LLAMA_SCREEN_HEAD_FILE=$SCREEN $CB/llama-bench $TG
  GGML_CUDA_NORM_Q8=0 $CB/llama-bench $TG
  # 2. 1-4-token steps (verify steps, several slots): pp1-pp4 with the multi-token kernels on and off, both with the
  #    MoE down rescale on (llama.cpp's kernels overflow without it, see the checks above)
  for mc in 1 0; do
    env -u LLAMA_NO_MOE_DOWN_RESCALE GGML_CUDA_MMVQ_PASCAL_MC=$mc \
      $CB/llama-bench -m $MODEL -ngl 99 -fa on -p 1,2,3,4 -n 0 -r 5 -o jsonl
  done
  # 3. prompts, nine cards: V2, all off, all on, then Q108, Q109 and Q110 off one at a time
  $V2/llama-bench $PP; env $OFF $CB/llama-bench $PP; $CB/llama-bench $PP
  for sw in GGML_CUDA_MMQ_MOE_LEGACY=1 LLAMA_GRAPH_REUSE_WAIT=1 LLAMA_GPU_KQ_MASK=0; do
    env $sw $CB/llama-bench $PP
  done
  ```

  - "All off" must match `V2` within noise; if it does not, a piece costs time even when switched off.
  - Each piece's share is "all on" against "all on but that piece". Expected (est., from each piece's note): decode
    +3–4 % (Q100) and +6 % with the screened head (Q92); pp8192 at `-ub 128` 1.6–1.8× from Q109 and 1.5–2× more from
    Q108 (prefill note, E3/E4), the mask's gain at 32k and beyond (E5's rig run); 2–4-token steps as in section 14.
  - Q110's deep-context run (llama-batched-bench at `-npp 32768`, prefill note E5) and the executor's runs
    (`llama-pipespec`, [pipelined speculation](pipelined_speculation.md)) use the same binaries.
