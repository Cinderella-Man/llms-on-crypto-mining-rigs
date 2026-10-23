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

- **What.** A second single-token matrix-vector kernel in the CUDA backend, used only on compute capability 6.1. It is
  the benchmark's variant (c): lane groups sized to the row width, several rows per lane group sharing one load of the
  activations, several weight blocks per lane with the widest loads their alignment allows, and pointer arithmetic
  hoisted out of the loop. It reads the GGUF blocks as they are, so it needs no repacking.
- **Where.** One decision in `ggml_cuda_mul_mat_vec_q` (`mmvq.cu:1455`), after the fusion arguments are checked and
  before anything is launched. Operator selection, fusion matching and the routing of expert (`MUL_MAT_ID`) products
  stay as they are. Any call the new kernel doesn't cover falls through to the stock kernel unchanged.
- **Which types.** Q8_0 and Q4_1 first (Q4_0 comes with them), then Q5_0, then Q4_1_64.
  - The file at the quality bar reads 66 % of its bytes as Q8_0, so Q8_0 carries most of the gain.
  - Q4_1 is the type the benchmark's kernel was developed and verified on.
  - Q5_0 has no tuned kernel yet. It gets one in the benchmark first, built on a cheaper way to apply its fifth bit.
  - Q4_1_64 is not an upstream type, so it stays in the fork.
- **Expected gain.** The S4-L0 file goes from 23.1 tok/s (stock build) and 26.3 (2-warp build) to about **32–35 tok/s**.
  Separate weight planes (Q39, mostly for Q8_0) would add a projected 3–4 ms per token, to about 35–40 tok/s.
- **Effort.** About 8 working days for Q38 including Q5_0, and 4–6 more for Q39.
- **Upstream.** Four separate PRs: the launch shape, the Pascal kernel, one activation quantization per shared input,
  and FP32 activation sums. The repack and Q4_1_64 stay in the fork.

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
block's words with `PRMT`. With 4 blocks per lane it also runs short of registers: the fused gate/up kernel needs up to
254 registers at R = 2.

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

The router (F32 or F16, 0.05–0.1 GB) is not a quantized type and doesn't run through MMVQ. On the rig, S4-L0 runs at
23.09 tok/s on a stock build and 26.29 tok/s with the 2-warp launch shape (llama-bench tg128 on 6 cards, Q34).

## 2. How a single-token product runs today

**Operator selection.**
- `ggml_cuda_mul_mat` (`ggml-cuda.cu:1976-2029`) sends a quantized weight with at most 8 activation columns to MMVQ
  (`ggml_cuda_should_use_mmvq`, `mmvq.cu:318`; call at `ggml-cuda.cu:2020-2023`).
- `ggml_cuda_mul_mat_id` (`ggml-cuda.cu:2064-2086`) sends expert products with at most `get_mmvq_mmid_max_batch` tokens
  to the same entry point, with the expert ids.
- On a single-GPU-per-layer split (`-sm layer`, as Flint runs), both end in `ggml_cuda_mul_mat_vec_q`.
  `ggml_cuda_op_mul_mat_vec_q` (`mmvq.cu:1589-1618`) serves only the row-split mode.

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

**The decision point.** A new function, called in `ggml_cuda_mul_mat_vec_q` right after the fusion block (after
`mmvq.cu:1538`), before the padding clear and the quantization:

```cpp
if (ggml_cuda_mmvq_pascal_supported(src0, src1, ids, dst, fusion_local, cc)) {
    ggml_cuda_mmvq_pascal(ctx, src0, src1, ids, dst, fusion_local);   // quantizes src1 itself, then launches
    return;
}
```

Deciding here rather than inside `mul_mat_vec_q_switch_ncols_dst` has three advantages:
- All tensors are still at hand (types, `nb[]` strides, data pointers), so alignment can be checked exactly.
- The activation format can change later (FP32 sums, section 8) without touching the stock path.
- One quantization per shared input (section 8) needs this point too.

The row-split entry `ggml_cuda_op_mul_mat_vec_q` receives activations that the generic multi-GPU code has already
quantized. It keeps the stock kernel.

**The predicate** (`ggml_cuda_mmvq_pascal_supported`) is true when all of these hold; otherwise the stock path runs as
today:
- the device is compute 6.1, or the force switch below is set;
- `src0->type` is one of the ported types;
- one destination column: `ne11 == 1` for `MUL_MAT`, `ne2 == 1` for `MUL_MAT_ID`;
- `x_scale` and `gate_scale` are null;
- the row width K splits into `C`-block chunks that the chosen lane-group size divides. This holds for every Flint and
  Qwen3.6 width (512, 1024, 2048, 4096, 6144).
- `src0->data`, `nb01`, `nb02` and `nb03` are multiples of the alignment the chosen `C` needs: 16 bytes for Q4_1 at
  C = 4, 4 bytes for Q8_0 at C = 2, 8 bytes for Q5_0 at C = 4. Views with padded rows (`test-backend-ops` builds them
  with `k_v`, `m_v`, `pad`) fail this check and fall back.

**A parameter table id.** Add `MMVQ_PARAMETERS_PASCAL_DP4A` to `get_device_table_id` for compute 6.1, on the host side
(`mmvq.cu:124-144`) and the device side (`:106-122`, `__CUDA_ARCH__ == 610`). The id is what PR #25479 introduces for the
2-warp launch shape, so both changes share one switch. The 2-warp shape stays useful for the types the new kernel doesn't
cover: UD-Q4_K_XL, which is K-quants and I-quants, gained 27 % from it. The repository's
[2-warp patch](../../../patches/mmvq-pascal-2warps.diff) edits the generic table and only works in sm_61-only builds.
A separate id fixes that.

**A switch for testing.** `GGML_CUDA_MMVQ_PASCAL` = `0` (off), `1` (default: compute 6.1 only) or `2` (any NVIDIA GPU).
With `2`, a modern GPU runs the Pascal kernel natively and, with `CUDA_FORCE_PTX_JIT=1`, from the compute_61 PTX. That
tests correctness without the rig. The new kernel's template parameters (warps per block, R, C) must not depend on
`__CUDA_ARCH__`, so the host and the device always agree on the block size, including under the PTX JIT.

**New files.** `mmvq-pascal.cuh` and `mmvq-pascal.cu` hold the kernel and its launcher. The diff to `mmvq.cu` is the
predicate call, the table id and the factored-out epilogue (section 6), about 40 lines.

## 4. The kernel inside llama.cpp

**Signature.** `mul_mat_vec_q_pascal<type, R, C, gate, has_fusion>` takes the same arguments as `mul_mat_vec_q`
(`mmvq.cu:623-629`): weights, Q8_1 activations, ids, fusion struct, destination, and the row, column, channel and sample
strides with their fastdiv ratios. It adds `lpr_log2` and `niter`. Strides arrive in blocks, as in the stock kernel. The
launcher converts them to bytes once per call (row bytes = `stride_row_x * sizeof(block)`, the expert stride likewise).
`gate` is a template parameter, unlike in the stock kernel, where `use_gate` is checked at run time: the gate doubles
the weight registers, so the compiler must know. Warps per block are fixed at 4, as in the benchmark (`main.cu:21`).

**Prologue.** Copied from `mmvq.cu:650-667`:
- the shared-expert slice (`blockIdx.y == gridDim.y - 1` with `shared_up`);
- the expert from `ids[channel_dst]`;
- the activation channel `channel_dst % nchannels_y`;
- samples through `sample_ratio`.

This is the only place the expert path differs from a dense product. The benchmark's kernel already reads
`ids[blockIdx.y]` and a per-channel activation stride (`pascal_mmvq.cuh:206-219`).

**Main loop and reduction.** Taken unchanged from `mmvq_rows` (`pascal_mmvq.cuh:231-268`). Row pointers are clamped to
the last row (`:213`), so blocks at the end of the matrix don't branch. Only the store is guarded (`:273`).

**Epilogue.** The stock kernel's result handling (`mmvq.cu:831-864`: bias, gate bias, the four GLU ops) moves into a
`__device__` function that both kernels call. The Pascal kernel calls it from lane `i` of each lane group after the
shuffle reduction. The stock kernel calls it where it does today.

**Launch parameters.** Chosen per type and per fused/unfused from the measured best per shape (GB/s on the P106-090 in
brackets). `lpr` is the largest power of two up to 32 that divides the chunks per row.

| Type | Unfused | Fused gate/up | Where the simple rule loses |
|---|---|---|---|
| Q4_1 | R = 2, C = 4 (101–159) | R = 2, C = 4 (107–126) | nowhere: best or within 3 % on every shape |
| Q8_0 | R = 2, C = 2 (91–128); R = 4 for N ≤ 512 (85) | R = 1, C = 2 (108–118) | R = 2 fused needs 254 registers and drops to 90 GB/s |
| Q4_0 | R = 2, C = 4 (79–132); R = 4, C = 2 for N ≤ 512 and K = 4096 | R = 2, C = 4 (97–114) | moe_down: R = 1, C = 2 is 2 % faster |
| Q4_1_64 | R = 2, C = 4 (73–160) | R = 1, C = 2 (123–137) | attn_kv: R = 4, C = 2 is 9 % faster |
| Q5_0 | to be measured (section 5) | to be measured | — |

The lane groups make the small-K rule unnecessary. At K = 512, Q4_1 with C = 4 has 4 chunks per row, so `lpr` = 4, and
each warp computes 16 rows with every lane busy. The expert down-projection reaches 139 GB/s this way, against 57 for the
stock kernel's small-K shape.

**Constraints that carry over.**
- The kernel runs inside CUDA graphs like the stock one: no host synchronization, and the activation buffer comes from
  the pool.
- `ggml_cuda_pdl_sync` is a no-op before Hopper and can be called for uniformity.
- Register counts in the benchmark range from 40 to 164 (Q4_1 fused at R = 2, C = 4: 164 registers, 3 blocks per SM),
  and the table above already reflects them.

## 5. Which types, in which order

Matrix-vector time per S4-L0 token by format, from section 10's table:

| Format | Share of bytes | Stock kernel | Ported (c) | Saved |
|---|---|---|---|---|
| Q8_0 | 65.8 % | 19.9 ms | 13.5 ms | 6.4 ms |
| Q5_0 | 18.6 % | ~7.1 ms (estimated) | ~3.8 ms (estimated) | ~3.3 ms |
| Q4_1 | 15.6 % | 4.6 ms | 2.4 ms | 2.2 ms |

1. **Q8_0 and Q4_1 together, with Q4_0.** The benchmark's kernel already implements all three, so porting them is one
   piece of work.
   - Q8_0 saves the most on the quality-bar file.
   - Q4_1 is the bring-up type: the benchmark was built around it, and the makers' Flint file is all Q4_1, a second
     end-to-end check. That file saves 8.5 ms per token.
   - Q4_0 is the base type of the Qwen3.6 mixes ([recipes](../../../quants/recipes/README.md)) and gains 1.93×.
2. **Q5_0** is the second-largest share of S4-L0, but no tuned Q5_0 kernel exists. It is built and measured in the
   benchmark first (step 1 of section 11), then ported.
3. **Q4_1_64** is this project's own type ([patch](../../../patches/q4_1_64-type.diff)). It holds the makers' group-64
   weights exactly and reaches the highest speed (2.10×), but upstream llama.cpp doesn't have it. It is a 0.5-day
   addition to the fork.

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
- The estimate: about 50 instructions per 32 weights per row (2.4 per byte, against 8.7 today), so the kernel should be
  limited by bandwidth like Q4_1's.
- It needs whole blocks per lane. The stock kernel's half-block split (`vdr` = 2) can't form these byte groups, so the
  trick belongs to the Pascal kernel.
- The load: C = 4 gives 88-byte chunks read with eleven 64-bit loads.

## 6. Keeping the fusions and the expert path

**Every fused call arrives as an ordinary `ggml_cuda_mul_mat_vec_q` call** with filled-in fusion fields. The predicate
either takes the whole call or leaves it to the stock kernel. No change in `ggml-cuda.cu` is needed, and every fusion
keeps working throughout the work, at stock speed until the Pascal kernel covers it.

| Fusion or path | How the Pascal kernel keeps it |
|---|---|
| gate + up + GLU (`gate`, `glu_op`, `glu_limit`) | `gate = true` instantiation: the gate rows load next to the up rows and share the activation registers, as in the benchmark (`pascal_mmvq.cuh:233-252`); the shared epilogue applies SWIGLU, GEGLU, SWIGLU_OAI or SWIGLU_CLAMP |
| biases (`x_bias`, `gate_bias`, also `ADD_ID` for experts) | read per row in the epilogue, with `channel_bias = ids ? channel_x : channel_dst` as at `mmvq.cu:708` |
| routed + shared gate/up (`shared_up`) | the launcher adds one `blockIdx.y` slice, and the prologue swaps in the shared weights, output and column stride, as at `mmvq.cu:650-656`. `nrows` is the same for both, by the assertions at `mmvq.cu:1491-1506` |
| per-lane scales (`x_scale`, `gate_scale`) | NVFP4 only; the predicate refuses them |
| `MUL_MAT_ID`, one token | channel = expert slot, expert from `ids`, activation per expert or shared via `nchannels_y` |
| `MUL_MAT_ID`, 2–8 tokens | unchanged: `mul_mat_vec_q_moe` (`mmvq.cu:1201-1209`) |
| `MUL_MAT`, 2–8 columns (speculative decoding) | unchanged in Q38; the lane-group design extends to several columns later |
| `GGML_CUDA_DISABLE_FUSION=1` | still turns fusion off (`ggml-cuda.cu:172`); unfused calls take the Pascal path one by one |

## 7. Separate planes and load-time repacking

**Q38 needs no repacking.** Variant (c) reads the GGUF block layout as it is.

**The plane layout of variant (d) does.** A GGUF stores blocks of interleaved scales and codes, and a new GGUF type is
not an option: it would break every other backend and tool. The planes therefore have to be produced in GPU memory after
loading. That repack is experiment Q39. For S4-L0 it matters mainly for Q8_0: Q8_0 has the cheapest arithmetic of all
types (8 `dp4a` per 32 weights, no unpacking), and only its 34-byte alignment keeps (c) at 106 GB/s on average.

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
    int8 values, then the `half2 (d, s)` pairs (`quantize_and_reorder_q8_1_soa`, `quantize.hpp:63-90`, chosen at
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

- **One quantization per shared input.** Every `ggml_cuda_mul_mat_vec_q` call quantizes its own activations
  (`mmvq.cu:1551-1558`). In Flint, attention q, k and v read the same vector, and so do the routed and the shared gate/up
  (mainline already shares one quantization between gate and up).
  - Keeping the last quantized input per device, keyed by the `src1` tensor and cleared after every node that writes
    memory, saves 98 + 47 = 145 launches per token. The survey estimates 0.7–1.0 ms.
  - It helps every GPU at one token, not just Pascal.
- **FP32 activation sums.** Q8_1 stores the block sum `s` in FP16 (`quantize.cu:100`), which overflows on Flint's
  expert down-projection inputs. That overflow is why llama.cpp rescales Flint's MoE down-projections, which costs
  2.4–2.8 ms per token.
  - In the Pascal kernels the activation block can become 40 bytes with `float d, s`. Two such blocks are 80 bytes, read
    as five 128-bit loads.
  - S4-L0 doesn't need it: its down-projections are Q8_0, whose dot product doesn't use `s`, so it already runs without
    the rescale.
  - It matters for the makers' Q4_1 file and for any mix with 4- or 5-bit down-projections. Dropping the rescale also
    needs FP32 sums in MMQ's `DS4` activation layout (`mmq.cuh:20`), because prompts go through MMQ.

## 9. Verifying correctness

The ported kernel changes the order of floating-point additions, so its results differ from the stock kernel in the last
bits. The integer part of each block's dot product is exact: `(lo << 4) + hi` equals 16 times llama.cpp's `sumi`.
`test-backend-ops` compares every case against the CPU backend with a normalized error limit of 5e-4 (`test_mul_mat`,
`tests/test-backend-ops.cpp:5324-5355`).

1. **Existing cases**, on any NVIDIA GPU with the force switch, natively and through the compute_61 PTX:
   ```bash
   GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_ID_FUSION,MUL_MAT_ID_SHARED,MUL_MAT_VEC_FUSION,MUL_MAT_ADD
   ```
   ```bash
   CUDA_FORCE_PTX_JIT=1 GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 -o MUL_MAT,MUL_MAT_ID,MUL_MAT_VEC_FUSION
   ```
   - `-p 'type_a=q8_0'` narrows to one type. `MUL_MAT_VEC_FUSION` (`:7743`) covers gate/GLU/bias for `MUL_MAT` and
     `MUL_MAT_ID` with batch dimensions. `MUL_MAT_ID_SHARED` (`:5810-5840`) covers the routed + shared launch.
   - The existing cases are small (16–32 rows, K = 256), so they mostly test the edges: tails, fallbacks and views.
2. **Cases at Flint's shapes**, added to `make_test_cases_eval`, for Q4_0, Q4_1, Q5_0 and Q8_0:
   - `test_mul_mat` with (rows, K) = (4096, 2048), (512, 2048), (2048, 4096), (2048, 1024), (6144, 2048),
     (2048, 6144), and an odd row count such as (4097, 2048);
   - `test_mul_mat_id` with 32 experts, 6 used, 512 × 2048 with a broadcast input and 2048 × 512 with one input per
     expert;
   - `test_mul_mat_vec_fusion` with `use_id` at 512 × 2048.
3. **The model's own operations.** `test-export-graph-ops` (`tests/test-export-graph-ops.cpp:128`) loads a GGUF on the
   CPU and writes every distinct operation of its prompt and generation graphs, with real shapes and types. Fused
   patterns are not single operations, so items 1 and 2 cover them.
   ```bash
   build/bin/test-export-graph-ops -m flint-s4-l0.gguf -o flint-s4-l0-ops.txt
   ```
   ```bash
   GGML_CUDA_MMVQ_PASCAL=2 build/bin/test-backend-ops -b CUDA0 --test-file flint-s4-l0-ops.txt
   ```
4. **Bit-identity with the benchmark.** [bench/mmvq_pascal/verify](../../../bench/mmvq_pascal/README.md#checking-correctness)
   already links a static ggml build and runs `MUL_MAT`, `MUL_MAT_ID` and the fused path at every benchmark shape. Built
   against the ported tree, it compares the port with the standalone (c) kernels at the same R, C and `lpr`, which must
   agree bit for bit.
5. **On a P106-090.** The same `test-backend-ops` commands without the force switch (the CPU reference is slow on the
   rig's 2-core host, so filter by type). Then a KLD/perplexity run of S4-L0 against the stock build
   ([scripts/quality_kld.sh](../../../scripts/quality_kld.sh)): the difference must stay within run-to-run noise.

## 10. Expected gain per token

Matrix-vector time per S4-L0 token. "Stock" is llama.cpp's kernel as measured by the benchmark; "ported" is the best
variant (c) per shape, with the launch parameters of section 4.

| Tensors | Format | Stock | Ported (c) | Basis |
|---|---|---|---|---|
| attention q | Q4_1 | 3.76 ms | 1.78 ms | measured, 49 × (76.6 → 36.4 µs) |
| attention k, v | Q8_0 | 1.94 | 1.28 | measured, 98 × (19.8 → 13.1 µs) |
| attention output, 0–32 | Q8_0 | 2.95 | 2.86 | measured, 33 × (89.4 → 86.6 µs) |
| attention output, 33–48 | Q4_1 | 0.84 | 0.62 | measured, 16 × (52.7 → 38.6 µs) |
| dense FFN, 0–1 | Q8_0 | ~0.94 | ~0.75 | estimated at 85 and 107 GB/s, the nearest measured Q8_0 shapes |
| routed gate/up | Q5_0 | ~7.1 (6.1–8.2) | ~3.8 (3.3–4.2) | estimated: 1.4–1.75× Q4_1's stock time per weight from the token-cost model and the instruction counts; 100–130 GB/s ported |
| routed down | Q8_0 | 3.89 | 3.00 | measured, 47 × (82.8 → 63.8 µs) |
| shared gate/up | Q8_0 | 2.54 | 1.94 | measured, 47 × (54.0 → 41.2 µs) |
| shared down | Q8_0 | 2.51 | 1.15 | measured, 47 × (53.5 → 24.4 µs) |
| output head | Q8_0 | 5.16 | 2.56 | measured |
| **Total** | | **31.6 ms** | **19.7 ms** | −11.9 ms (−11.4 to −12.6 with the Q5_0 range) |

**From kernel time to tokens per second.** Two ways to carry the saving over:
- **Subtract it from the stock build's measured 43.3 ms per token.** That gives 31.4 ms, **31.8 tok/s**: +38 % over the
  stock build and +21 % over the 2-warp build's 26.3.
- **Calibrate on the 2-warp build.** That build gained 5.3 ms per token end to end on S4-L0 and 4.2 ms on the makers'
  file. The benchmark predicts only 2.4 and 2.7 ms. Most of the gap is in the expert down-projection: the benchmark finds
  2 warps 1.9× slower there for Q8_0 at K = 512, and the end-to-end gain doesn't reflect that. Subtracting the
  benchmark's difference between the 2-warp variant (29.2 ms) and the port (19.7 ms) from the 2-warp build's measured
  38.0 ms gives 28.5 ms, **35.1 tok/s**.

Q38 is therefore expected at **32–35 tok/s on S4-L0**, against 24.8 for UD-Q4_K_XL on its best build. Prompt processing
is unchanged.

**Q39 on top** (a projection: no plane kernel for Q8_0 or Q5_0 has been measured yet):
- Q8_0 planes at the bandwidth Q4_1 planes reach (129–162 GB/s on large shapes, 88–111 on the 1–2 MB ones) save
  2–3 ms.
- Q5_0 planes save 0.5–0.9 ms.
- Q4_1 planes save about 0.05 ms: its blocks are already 4-byte aligned, and (c) is within 3 % of (d) on S4-L0's
  shapes.

That is about 35–40 tok/s. One quantization per shared input adds 0.7–1.0 ms on top.

**The makers' Q4_1 file.** It measures 28.07 tok/s (rescale on). The port saves 8.5 ms of its 20.0 ms matrix-vector time,
for about 37 tok/s. Planes plus group-64 compaction reach 38. FP32 sums remove the rescale's 2.4–2.8 ms, for about
42 tok/s.

**What it changes for the file side.** The token-cost model in [scripts/token_cost.py](../../../scripts/token_cost.py)
prices each format by its decode efficiency, and the port moves those prices:
- Stock: Q8_0 costs 1.75× Q4_1 per parameter on S4-L0's shapes (72 and 74 GB/s).
- Ported (c): 2.3× (106 and 142 GB/s), so Q8_0-heavy mixes gain less than Q4_1-heavy ones.
- With Q8_0 planes: back to about 1.8×.

After step 5 the efficiencies should be refitted to the new build, and the per-layer allocation of Q36 and Q45 redone
with them.

## 11. Implementation plan

| Step | Work | Effort | Done when |
|---|---|---|---|
| 0 | Time every kernel of one real S4-L0 token on the stock and the 2-warp build, with a CUDA trace tool that supports sm_61 | 0.5 day, rig | the per-kernel gap between the benchmark and the real token is located, and the real Q5_0 gate/up time is known |
| 1 | Benchmark additions: Q5_0 (stock, 2 warps, tuned with the fifth-bit dot product of section 5, C = 2 and 4); row-local planes for Q8_0, Q5_0 and Q4_1 against variant (d) | 1–1.5 days + 15 min on the rig | Q5_0 tuned ≥ 1.5× stock on the gate/up shape; row-local planes within 3 % of tensor-wide planes |
| 2 | Port (c) as `mmvq-pascal.cu` for Q4_0, Q4_1 and Q8_0: decision point, predicate, table id, force switch, `MUL_MAT` and one-token `MUL_MAT_ID`; fused calls fall back for now | 1.5–2 days | `test-backend-ops` `MUL_MAT`/`MUL_MAT_ID` pass with the force switch, native and through the PTX JIT |
| 3 | Fusions: `gate = true` kernels, shared epilogue, biases, `shared_up` | 1 day | `MUL_MAT_VEC_FUSION`, `MUL_MAT_ID_FUSION`, `MUL_MAT_ID_SHARED`, `MUL_MAT_ADD` pass |
| 4 | Tests at Flint's shapes, bit-identity against the benchmark, `test-export-graph-ops` files for S4-L0 and a Qwen3.6 mix | 1 day | all pass; the port and the standalone kernels agree bit for bit |
| 5 | Rig: `test-backend-ops` on a P106; tg128 for S4-L0, the makers' Q4_1 and a Qwen3.6 Q4_0 mix; KLD against the stock build | 0.5 day | S4-L0 ≥ 15 % faster than the 2-warp build at unchanged KLD; refit `token_cost.py` |
| 6 | Q5_0 in the port, from step 1's kernel | 1 day | tests pass; S4-L0 tg128 measured again |
| 7 | Q4_1_64 in the port (fork only) | 0.5 day | tests pass; makers' file in Q4_1_64 measured |
| 8 | One activation quantization per shared input | 1–2 days | 145 fewer launches per Flint token; tests and KLD unchanged |
| 9 | FP32 activation sums in the Pascal kernels and MMQ's `DS4` layout; drop the rescale where safe | 1–2 days | makers' Q4_1 runs without the rescale, without NaN, at unchanged KLD |
| 10 | Q39: row-local planes, repack before the first graph, planar matrix-vector kernels, planar MMQ loaders, row lookup, dequantization, `get_tensor` inverse, opt-in switch | 4–6 days | all tests pass with `GGML_CUDA_PASCAL_REPACK=2`; prompt processing no slower; tg gain measured |

Steps 2–7 are Q38 proper (about 6 days, plus 1.5–2 for steps 0–1). Steps 8 and 9 are independent of the kernel and can go
in any order. Step 10 is Q39.

## 12. Upstream PRs

Each of these stands alone and can be reviewed separately:

1. **Launch shape for compute 6.1.** The `MMVQ_PARAMETERS_PASCAL_DP4A` table with 2 warps at one token, keeping 4 warps
   and the small-K rule for rows narrower than 2048. These are the data PR #25479 lacks. Small, and the most likely to be
   accepted.
2. **The Pascal kernel** for Q4_0, Q4_1, Q5_0 and Q8_0 (steps 2–4 and 6 without Q4_1_64).
   - A new file selected through the table id, which falls back to the stock kernel for everything else, with the
     Flint-shape test cases. Upstream already carries device-specific MMVQ tables (Turing, RDNA, GB10).
   - It needs measurements on at least one larger compute 6.1 card (a GTX 10-series card or a P40). Those cards have
     2–4× less bandwidth per SM than the P106-090 and are less limited by instruction issue, so the PR must show no
     regression there, or restrict itself by SM count.
3. **One activation quantization per shared input.** Not specific to any architecture, and continues mainline's
   gate/up de-duplication.
4. **FP32 activation sums.** Closes the overflow that the Flint, Mistral4 and Mellum rescale works around. It competes
   with the open BF16 proposal (#22571) and touches every Q8_1 reader, MMQ included.

The repack (step 10) and Q4_1_64 stay in the fork. In the discussion of an earlier CUDA repack PoC (#20563), the
maintainers asked for a baseline without repacking first; PR 2 would provide that baseline, but the repack touches too
many readers for an architecture this old. Q4_1_64 is a new GGUF type.

## 13. Risks

- **The benchmark doesn't fully predict the real token** (section 10's 2-warp gap). Step 0 locates the difference
  before the port starts. If the real token differs mainly in the expert down-projection, the launch table needs a
  per-shape entry there.
- **The Q5_0 numbers are estimates.** If step 1 finds the fifth-bit dot product short of 1.5×, the alternative is a
  load-time permutation of the 32 `qh` bits (same 22-byte block, so it can be done in place). That makes the fifth-bit
  term one shift and one mask per word, and belongs with step 10.
- **Register pressure.** Fused Q8_0 at R = 2 reached 254 registers. The table uses R = 1 there, and Q5_0's transposed
  activation words should be built per block inside the loop, not held for the whole chunk.
- **Fallback coverage.** Widths that don't split into whole chunks, unaligned views, row split, and more than one
  column all keep the stock kernel. That is correct but slower; tail handling in the last iteration can widen coverage
  later.
- **Maintenance.** The kernel lives in its own file. The hook in `mmvq.cu` is a predicate call and a table id, so
  rebasing touches little. The shared epilogue keeps the GLU handling in one place.
