# Flint's attention at long context on the P106 rig: what llama.cpp runs, why it reads the KV cache at 57–87 GB/s, and a Pascal decode kernel (2026-10-23)

Scope: experiment Q54 in [quants/experiments.md](../../../quants/experiments.md) and section 6 of
[hyperqwen_lessons.md](hyperqwen_lessons.md). Pre-fork, Flint slowed from 25.7 tok/s at 8k of context to 19.5 at 32k,
which puts the extra KV-cache reads at ~57 GB/s, against 120–150 GB/s for the fork's Pascal matrix-vector kernels. This
note traces which flash-attention kernel llama.cpp picks on compute 6.1 for Flint's shapes, models what one decode token
costs at depth, surveys GQA-aware and quantized-KV decode work in llama.cpp, ik_llama.cpp and other engines, and specifies
a Pascal decode attention kernel for the fork, with tests, expected speed, KV memory per card and a measurement plan.

Conventions are those of the other notes:
- "Claim to test" marks anything not measured on this rig.
- "(my calc)" marks this note's own arithmetic.
- "P" and "S" in source lists mean primary (code, PRs, papers, measurements in the repo) and secondary.

Code references:
- llama.cpp lines refer to the fork's base, upstream `d77256999`. `fattn.cu`, `fattn-common.cuh`, `fattn-tile.cuh` and
  `fattn-vec.cuh` are identical on current master (`36e36e6`, 2026-10-23), so everything said about kernel selection holds
  there too. CUDA files live in `ggml/src/ggml-cuda/`.
- Register counts, shared memory and instruction counts come from the fork's sm_61 build (`llama-bench` at `eeaa59660`,
  CUDA 12.9): `cuobjdump --dump-resource-usage` and `cuobjdump -sass`, counting the instructions of each kernel's main
  loop. The fork leaves the attention kernels untouched, so these are the stock kernels.

Flint's attention, from [flint_speedups.md](flint_speedups.md): 49 layers, 32 query heads, 4 KV heads (GQA ratio 8), head
size 128, learned sinks on every layer. 14 full-attention layers (0, 1, 5, 9, …, 45, 48) use no RoPE and keep the whole
context; 35 sliding-window layers keep 513 tokens. One token of context costs 28 KiB of f16 KV on the full-attention
layers (14 × 4 heads × 128 × 2 × 2 B).

## Headline findings

1. **With an f16 cache, Flint's decode runs the TILE kernel, not VEC.** On compute 6.1 the GQA optimization applies
   (ratio 8, mask present, KV length padded to 256, 16-byte strides), so `ggml_cuda_get_best_fattn_kernel` returns TILE
   for f16 K/V at every batch size (`fattn.cu:705-718`). It runs `flash_attn_tile<128,128,1,8>`: one block per KV head
   computes all 8 query heads, so K and V come from DRAM once. A q8_0 cache runs VEC for 1–2 tokens and, for 3–8 tokens,
   converts the whole K/V view to f16 before running TILE. Sinks and sliding-window masks don't change the choice.
2. **The 57 GB/s is the TILE kernel's shared-memory limit (my calc from the SASS).** On sm_61 the kernel has no fast FP16,
   so it stores each 64-key K and V tile as FP32 in shared memory, and each of its 4 warps reads the whole tile. That is
   ~12 B of shared-memory traffic per byte of KV, which caps the kernel at ~87 GB/s before bank conflicts and the
   unhidden load latency of a kernel with 8 warps per SM. The three files measured at depth give 57, 77 and 87 GB/s.
3. **A q8_0 cache is probably much slower than f16 at depth today (claim to test).** The VEC kernel runs one block per
   query head, so it repeats its work 8 times per KV head: 496 warp-instructions per key and KV head, ~19 GB/s of q8_0
   bytes (my calc). A quantized cache also switches on llama.cpp's Hadamard rotation of Q, K, V and the attention output:
   4 extra launches per layer (196 per token), and 160 KB of rotation matrices copied to every card per token. Expected
   at 32k: ~18 tok/s with q8_0 against 23–27 with f16 (my calc).
4. **Upstream PR #30792 doesn't build for sm_61.** Its q8_0 loader asserts FP16 accumulation; compiling the patched
   head-128 TILE instance for sm_61 with CUDA 12.9 stops at that static assertion, while the unpatched file compiles.
   With an FP32 loader it would run at f16 TILE speed: on this card the gain would be memory and prefill, not decode.
5. **A quantized cache saves little VRAM on this rig as llama.cpp stands (my calc).** The graph reserved for prompt
   processing makes the CUDA backend keep an f16 copy of one full-attention layer's K and V on every card: 2 KiB per
   context token per card, 512 MiB at 262,144 tokens. On 9 cards q8_0 then needs 1,056 MiB on the fullest card against
   1,024 MiB for f16. Dropping that scratch (TILE reading q8_0 natively) is what turns q8_0 into a capacity lever.
6. **A GQA-aware Pascal decode kernel should read f16 KV at ~130–145 GB/s and q8_0 at ~85–95 GB/s of q8_0 bytes (my
   calc).** One warp per KV head and key chunk; each lane owns 4 of the 128 dimensions for all 8 query heads, so K and V
   are read and converted once and the query stays in registers. Scores are reduced across lanes by a transpose-reduce,
   probabilities go back through shared memory, the KV range is split over 15 blocks, and the last block merges the
   partials and applies the sinks, in one launch per layer. For q8_0, K·Q runs on `dp4a` with an int8 query, as the
   stock VEC kernel already does.
7. **Expected S4-L0 decode, on the 6-card token time (my calc):** at 32k 23–27 → ~30 tok/s, at 64k 17–21 → ~25, at 131k
   11–14 → ~19–21, at 262k 6–9 → ~12.5–14 (contexts above ~90k need more than 6 cards in f16). At 32k attention is
   29–38 % of a token today and ~20 % with the new kernel. The sliding-window layers and the dropped combine launches add
   ~2 % at short context.
8. **Verifying several tokens costs more as context grows, whatever the kernel.** Attention needs 2,048 FP32 FMAs per key
   per KV head for every query token, so at 32k each verified token costs at least 3.45 ms of attention on Pascal (my
   calc), on top of the ~5–6.5 ms of experts per extra token from [hyperqwen_lessons.md](hyperqwen_lessons.md).
9. **No engine has a Pascal GQA decode kernel to port.** ik_llama.cpp runs `vec_f32` for 1–8 tokens on Pascal. vLLM's
   paged attention v1/v2 gives each query head its own block. FlashInfer's decode kernel shares K/V across the group,
   but through `cp.async` shared-memory pipelines built for sm80+. HyperQwen's int8-KV kernels are sm86 Triton and
   tensor-core code. What carries over: split-KV with a log-sum-exp merge, group-shared K/V loads, int8 scales and a
   split-KV verify kernel.

## 1. The starting point

**Measured** ([Flint benchmark](../../benchmarks/2026-10-22-flint.md), llama.cpp `d77256999`, 5 cards, f16 KV,
`llama-bench -fa on -p 0 -n 64 -d 0,8192,32768 -r 1`):

| File | 8,192 | 32,768 | Extra ms per token | Implied KV read rate (my calc) |
|---|---|---|---|---|
| makers' Q4_1 | 25.7 tok/s | 19.5 tok/s | 12.4 | 57 GB/s |
| imatrix Q5_0 | 19.7 | 17.0 | 8.1 | 87 GB/s |
| UD-Q4_K_XL | 17.1 | 14.8 | 9.1 | 77 GB/s |

The 24,576 extra tokens are 0.70 GB of f16 KV. One repetition each; the depth-0 point (20.4 tok/s) is low for the same
file, so the spread is partly noise.

**On the fork.** S4-L0 decodes 37.4 tok/s in tg128 on 6 cards (26.8 ms per token, 2.24 GB of weights per token). The fork
didn't change the attention kernels, so attention per byte of KV costs what it cost before, but it is now a larger share
of a faster token.

**KV read per generated token (my calc):**

| Context | f16 | q8_0 | q4_0 |
|---|---|---|---|
| 8,192 | 0.235 GB | 0.125 GB | 0.066 GB |
| 32,768 | 0.940 GB | 0.499 GB | 0.264 GB |
| 65,536 | 1.879 GB | 0.998 GB | 0.528 GB |
| 131,072 | 3.758 GB | 1.996 GB | 1.057 GB |
| 262,144 | 7.516 GB | 3.993 GB | 2.114 GB |

The sliding-window layers add a constant ~55 MB per token (35 layers × 768 cells × 2 KiB at `-ub 128`; 1,280 cells at
llama-bench's default `-ub 512`).

## 2. Which kernel llama.cpp runs on compute 6.1

### Selection

`ggml_cuda_get_best_fattn_kernel` (`fattn.cu:540-719`) on a GPU without tensor cores:
- `gqa_opt_applies` (`fattn.cu:560-571`) needs a GQA ratio ≥ 2, a mask, no ALiBi, a KV length divisible by 256 and
  16-byte strides for the non-quantized tensors. Flint's KV views satisfy all of it:
  - llama.cpp pads `n_kv` to 256 (`src/llama-kv-cache.cpp:1290-1295`);
  - K rows are 1,024 B per token (4 heads × 256 B);
  - the mask is `n_kv` × 2 B per row (`src/llama-graph.cpp:30-47`).
- The Turing, Volta and AMD branches don't apply. The generic branch (`fattn.cu:705-718`) returns:
  - for f16 K/V: VEC only for one token *without* the GQA optimization, so TILE for Flint at every batch size;
  - for quantized K/V: VEC for 1–2 tokens, TILE above that.
- `launch_fattn_tile_switch_ncols2` (`fattn-tile.cuh:1236-1320`) takes `ncols2 = 8` (all 8 query heads of a KV head in
  one block): the NVIDIA "GQA ≤ 4" limit at `fattn-tile.cuh:1252` doesn't apply to ratio 8.
- Neither selection nor launch looks at sinks. Both kernels apply them in the first KV split only
  (`fattn-tile.cuh:1052-1080`, `fattn-vec.cuh:381-412`), and the combine kernel folds them into the result.
- A mask with `ne[2] != 1` returns no kernel; llama.cpp's masks are `[n_kv, n_tokens, 1, n_stream]`, so that never happens.

### What runs, by case

| Query tokens | f16 K/V | q8_0 K/V |
|---|---|---|
| 1 (decode) | TILE `<128,128,1,8>`: 128 threads, 64 keys per iteration, 140 registers, 39,424 B shared → 2 blocks per SM | VEC `<128,1,q8_0,q8_0>`: 128 threads, one block per query head, 146 registers, 2,304 B shared → 3 blocks per SM |
| 2 | TILE `<128,128,2,8>` (16 rows, 32 keys per iteration), 140 registers | VEC `<128,2,q8_0,q8_0>`: 250 registers plus a 48-byte spill |
| 3–4 | TILE `<128,128,4,8>`: 256 threads, 32 rows, 128 registers, 41,472 B shared | whole K and V view converted to f16, then TILE `<128,128,4,8>` |
| 5–8 | the same, as two row tiles: K/V read twice | conversion, then two row tiles |
| prompt (`-ub 128`) | TILE, 32-row tiles | conversion, then TILE |

The same applies to the sliding-window layers (768 cells at `-ub 128`) and the full-attention layers (`n_kv` up to the
context).

### Precision on sm_61

- `FAST_FP16_AVAILABLE` is undefined for compute 6.1 (`common.cuh:272-274`), so TILE uses its FP32 configuration table
  (`fattn-tile.cuh:90-156`; ncols 8 is line 129: 128 threads, occupancy 3, 64 keys, 128-wide K slices).
  - It converts each half to float on the way into shared memory (`fattn-tile.cuh:428-476`). The SASS shows one
    `F2F.F32.F16` per element, 128 per thread per iteration.
  - KQ and VKQ use FP32 FMA.
- Copies are at most 8 bytes wide below Volta (`common.cuh:400-410`), so TILE's global loads are 32-bit per lane (64
  `LDG.E.CI` per thread per iteration).
- VEC with q8_0 K (`fattn-vec.cuh:97`, `:150-203`):
  - It quantizes Q to q8_1 inside the kernel and takes K·Q with `dp4a` (32 `IDP.4A` per thread per iteration).
  - It reads the 2-byte-aligned q8_0 words as 16-bit pairs (224 `LDG.E.CI.U16`).
  - It converts V with one `I2F.F32.S8` per element (128) and accumulates in FP32.
  - Every key's dot product ends in a 5-step warp reduction (160 `SHFL.BFLY`).

## 3. What one decode token costs at depth

### TILE with f16 KV (my calc from the SASS and the launch code)

- **Grid.**
  - 4 KV heads, `parallel_blocks` = 5 (the efficiency search at `fattn-common.cuh:1139-1218`, with 2 blocks per SM
    allowed by shared memory): 20 blocks, two waves of 10.
  - Each block walks every fifth 64-key tile.
  - A combine kernel (`flash_attn_combine_results<128>`, `fattn-common.cuh:922-978`, 53 registers) merges the 5 partial
    results.
  - So each layer costs 2 launches, 98 per token.
- **DRAM.** K and V once per KV head, plus 4 mask loads per thread per tile, which hit in cache. The mask-scan kernel
  that trims empty tail tiles runs only for ≥ 1,024 queries or several sequences (`fattn-common.cuh:1117`).
- **Per 64-key iteration and block** (32 KB of f16 K/V): the main loop is 1,883 instructions per thread:
  - 1,026 `FFMA`;
  - 352 shared loads (96 `LDS.128`, 256 `LDS.64`) and 66 `STS.64`;
  - 132 `F2F`;
  - 68 global loads.

  That makes 7,532 warp-instructions per 32 KB, or 7.4 thread-instructions per byte. The 5 SMs issue ~6.4 per byte of
  a 173 GB/s stream, so the issue limit alone would allow ~150 GB/s.
- **Shared memory sets the limit.**
  - The FP32 K and V tiles are 2 × 32 KB, written once and read in full by each of the 4 warps: 256 KB of reads.
  - The query and probability reads are broadcasts: ~640 more warp-loads per tile.
  - Altogether ~400 KB of shared traffic per 32 KB of KV, i.e. ~12 B per byte, against 128 B per clock per SM.
  - The ceiling is ~10 B of KV per clock per SM, ~87 GB/s.
  - With only 8 warps per SM and no prefetch (load, `__syncthreads`, compute, for K and then for V) part of the DRAM
    latency stays exposed, which fits the measured 57–87 GB/s.
- **Changing the launch configuration doesn't help.** Fewer warps per block cut the K/V tile reads but multiply the
  query broadcasts per byte. Half-precision tiles converted at the point of use (ik_llama.cpp PR #2142, below) halve the
  tile traffic but run four conversions per element; that variant tops out near 110–125 GB/s.

### VEC with q8_0 KV (my calc)

- Grid `(1, 5, 32)`: 160 blocks, 15 resident (3 per SM), 11 waves.
- Each of the 8 query-head blocks of a KV head repeats the loads, the dequantization and the reduction:
  - 1,984 instructions per thread per 128 keys, i.e. 62 warp-instructions per key and query head;
  - 496 per key and KV head, 1.8 per byte of q8_0, against the issue budget of 0.2.
- Instruction issue limits it to ~19 GB/s of q8_0 bytes, i.e. ~35 GB/s in f16-equivalent context (claim to test).
- DRAM traffic is also inflated: 3 query heads of the same KV head are resident at a time and the L2 holds 1.5 MB
  against 8.9 MB of q8_0 KV per head and layer at 32k, so each KV byte is fetched ~2.7 times.

### Costs outside the kernel that grow with context

- **The mask is built on the host every token.** `set_input_kq_mask_impl` (`src/llama-kv-cache.cpp:1575-1762`) visits
  every cell. A commented-out timer sits at `:1783`/`:1805`.
  - Its cost on the 2-core Celeron is unmeasured: ~0.13 ms at 32k and ~1 ms at 262k at 4 ns per cell (my calc).
- **The mask is copied to every card every token.** It is a graph input, which the scheduler copies synchronously to
  each card that uses it (`ggml/src/ggml-backend.cpp:1829-1837`).
  - At 262,144 tokens that is 512 KB per card, ~2.7 ms per copy over a PCIe 1.0 x1 link.
  - Only the first card's copy is on the critical path, because later copies overlap earlier cards' compute.
  - Fill plus first copy: ~0.5 ms at 32k, ~3.8 ms at 262k (my calc, claim to test).
- **The rotation for quantized caches.** With a quantized K or V type, the KV cache turns on Hadamard rotations
  (`src/llama-kv-cache.cpp:316-359`): 128 × 128 for K and 64 × 64 for V, for both the full and the sliding-window cache
  (`src/llama-graph.cpp:3818-3822`).
  - Per layer: Q, K, V and the attention output each get a fast Walsh–Hadamard transform kernel
    (`src/llama-graph.cpp:3500-3562`, `fwht.cu:215-222`), 196 launches per token.
  - The four rotation matrices (160 KB) are graph inputs too, so they are copied to every card every token, although
    the transform never reads them.
  - `LLAMA_ATTN_ROT_DISABLE=1` turns the rotation off.
  - Estimated 1–2 ms per token (claim to test).

### Attention's share of a token, and predictions for Q54 (my calc)

S4-L0 on 6 cards, 26.8 ms per token at depth 0, f16 KV at 57–87 GB/s:

| Context | Attention (f16) | tok/s | Attention share |
|---|---|---|---|
| 0 | ~1 ms (sliding-window layers, included in the 26.8) | 37.3 | ~4 % |
| 8,192 | 2.7–4.1 ms | 32.3–33.9 | 9–13 % |
| 32,768 | 10.8–16.5 ms | 23.1–26.6 | 29–38 % |
| 65,536 | 21.6–33.0 ms | 16.7–20.7 | 45–55 % |
| 131,072 | 43–66 ms | 10.8–14.3 | 62–71 % |
| 262,144 | 86–132 ms | 6.3–8.8 | 76–83 % |

With q8_0 (VEC at ~19 GB/s of q8_0 bytes, plus ~1.5 ms of rotation): ~29 tok/s at 8k and ~18 tok/s at 32k (claim to
test). The rows above ~90k need more than 6 cards (section 6). Q54's gate, attention ≥ 30 % of a token at 32k, is expected
to pass.

## 4. Upstream and other engines

**llama.cpp**

| PR / issue | State (2026-10-23) | Content | What it means here |
|---|---|---|---|
| [#30792](https://github.com/ggml-org/llama.cpp/pull/30792) "native q8_0 KV (and f16 K / q8_0 V) in the flash-attn tile kernel" | open since 2026-10-19, head `3c25375492` | Sends q8_0 K/V to TILE when the GQA optimization applies; dequantizes q8_0 to `half2` while copying into shared memory; drops the f16 scratch for those cases. gfx1151: 63k context, q8_0 31.2 → 50.5 tok/s (f16 43.4) | Fails to compile for sm_61: `static_assert(... half2 ..., "q8_0 tile kernel requires fp16 accumulation")` fires for the FP32 tile path. An FP32 loader would make q8_0 decode as fast as f16 TILE on this card (shared-memory bound) and remove the scratch |
| [#30202](https://github.com/ggml-org/llama.cpp/pull/30202) "dequant quantized kv on-the-fly for mma kernel" | open (PoC) | q8_0/q4_0 read inside the tensor-core MMA kernel | Ampere and newer only |
| [#23907](https://github.com/ggml-org/llama.cpp/pull/23907) | merged 2026-06-03 | Reserves the f16 conversion buffer for quantized KV at startup | Source of the per-card scratch of section 6 |
| [#29827](https://github.com/ggml-org/llama.cpp/pull/29827), [#29846](https://github.com/ggml-org/llama.cpp/pull/29846) | closed | Cap the conversion buffer with serial split-KV; fuse dequantization into the MMA kernel | Same problem, not merged; issues [#29371](https://github.com/ggml-org/llama.cpp/issues/29371) (open) and [#29826](https://github.com/ggml-org/llama.cpp/issues/29826) |
| [#22137](https://github.com/ggml-org/llama.cpp/pull/22137) / issue [#22032](https://github.com/ggml-org/llama.cpp/issues/22032) | PR closed unmerged; issue closed as stale | P40s crashed in the first decode step above ~24k context; the workaround runs the KV-length mask scan for every batch size | A risk to watch in the q8_0 depth runs |
| [#17558](https://github.com/ggml-org/llama.cpp/pull/17558), [#22541](https://github.com/ggml-org/llama.cpp/pull/22541) | merged | No FP16 arithmetic in VEC; Pascal shared-memory limit for a TILE variant | Already in the build |
| [#30190](https://github.com/ggml-org/llama.cpp/pull/30190) (Vulkan) | open, WIP | Folds 2–8 verify tokens and the GQA heads into one tile row; −48 % for a 4-token op at 52k | The same idea for the verify path below; coopmat-only |

**ik_llama.cpp** (needs AVX2 on the host, so it doesn't run on the rig; its Pascal kernels are references):
- [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md) (state of
  2026-10-13): on Pascal, 1–8 query tokens at head sizes 64–256 run `vec_f32`, with f16, q8_0, q4_0 and iq4_nl at head
  128. It has no GQA-aware decode kernel.
- [PR #2580](https://github.com/ikawrakow/ik_llama.cpp/pull/2580) (merged 2026-10-02): head-256 prefill to the FP32 tile
  kernel; on a P100, prefill at 16k +95 % with f16 KV and +148 % with q8_0. Decode is unchanged.
- [PR #2647](https://github.com/ikawrakow/ik_llama.cpp/pull/2647) (merged 2026-10-12): head size 512 on the GPU for
  Pascal, from mainline's FP32 tile path; Gemma-4-12B decode at depth 8,192 +30.9 % on a P100 (previously on the CPU).
- [PR #2142](https://github.com/ikawrakow/ik_llama.cpp/pull/2142) (merged 2026-07-17): FP32 tile prefill retiled. K and
  V stay `half2` in shared memory and are converted at use, and the scores stay in registers. The tile's shared
  memory halves, one `__syncthreads` per tile goes, +4–9 % on a P100. This is the cheap fix for the shared-memory limit
  of section 3.

**Other engines**
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) `include/flashinfer/attention/decode.cuh` (main at
  `89009ad445`):
  - The decode block has `bdy` = GQA group size, so one K/V tile in shared memory serves all query heads of the group.
  - `compute_qk` reduces each key's dot product across `bdx` lanes with `shfl_xor`.
  - Loads are multi-stage `cp.async` pipelines, which need sm80; older GPUs get synchronous copies.
  - Paper: [arXiv 2501.01005](https://arxiv.org/abs/2501.01005).
- vLLM paged attention v1/v2 (`csrc/attention/attention_kernels.cuh:150-153` at tag v0.9.2; removed from current main):
  one block per *query* head (`kv_head_idx = head_idx / num_queries_per_kv`), so K/V are read once per query head;
  v2 splits the sequence into 512-token partitions plus a reduce kernel.
- [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html): split the KV length across blocks, merge
  with the log-sum-exp. [Lean Attention](https://arxiv.org/abs/2405.10480) does the same as a stream-K decomposition.
- HyperQwen ([docs/long-context.md](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/long-context.md)),
  measured on sm86:
  - The pth-flash-attn kernel reads an int8 per-token-head cache directly: 44.1 → 86.2 tok/s at 128k on two RTX 3080s.
  - A Triton split-KV verify kernel takes 1.3 ms per layer for 8 query tokens at 128k, against 7.4 ms for vLLM's
    unified attention and 10.1 ms for FlashAttention-2.

  Both are tensor-core or Triton code; the ideas that carry over are one int8 scale per token and head, and split-KV
  for multi-token verify.

## 5. A Pascal decode attention kernel for the fork

### 5.1 Budget on a P106-090 (per SM and clock, ~1.7 GHz under load)

| Resource | Rate | Per byte of a 173 GB/s KV stream |
|---|---|---|
| DRAM | 173 GB/s ≈ 20.4 B | — |
| Instruction issue | 4 warp-instructions | ≤ 0.2 warp-instructions (6.4 thread-instructions) |
| FP32 FMA | 128 lanes | — |
| FP16 arithmetic | 1/64 of FP32 | not usable |
| Conversions (`F2F`, `I2F`), shuffles | 32 lanes (NVIDIA's 6.1 throughput table; claim to test) | — |
| `dp4a` | full rate, 4 int8 products | — |
| Shared memory | 32 banks × 4 B = 128 B; 96 KB per SM, 48 KB per block | — |

Per key and KV head Flint reads 512 B of f16 K+V (272 B of q8_0) and needs 2,048 FMAs (8 heads × 128 dimensions × 2).
f16 sits at 4 FMAs per byte, under the card's 6.3, so FMAs alone don't make the kernel compute-bound. Each byte must
cost under ~0.2 warp-instructions, and must not pass through shared memory several times.

### 5.2 Work split

- **Block:** 4 warps, one per KV head, all covering the same key range. A token row of all 4 heads is contiguous
  (1,024 B in f16, 544 B in q8_0, both 16-byte aligned), which the q8_0 staging needs.
- **Lane mapping:** lane `l` owns dimensions `4l … 4l+3` of K and V for all 8 query heads of its warp's KV head.
  - The query (pre-scaled by `log2(e)/sqrt(128)`) lives in registers: 32 floats for f16, or 8 int8 words plus 8 block
    scales for q8_0.
  - The V accumulator is 32 floats per lane.
  - K and V are read and converted once per key; the 8 query heads reuse them from registers.
- **Split-K:** `S = clamp(n_kv / 512, 5, 15)` blocks per layer call.
  - 15 blocks put 3 on each of the 5 SMs (12 warps per SM at ≤ 160 registers); 5 cover the 768-cell sliding-window
    layers with ~150 keys each.
  - At 32k each block covers ~2,200 keys.
- **Loads:**
  - f16: each lane loads its 8 B of the 256-B head row (`LDG.64`, coalesced), 4 keys ahead.
  - q8_0: token rows are staged through shared memory with 128-bit loads, double-buffered, 16 keys per stage (17 KB per
    block). The fields inside are 2-byte aligned (34-byte blocks), and each lane extracts its word with one `PRMT` from
    two aligned words, as the MMVQ port does ([pascal_mmvq_integration.md](pascal_mmvq_integration.md), section 4).

### 5.3 K·Q

- **f16 (FP32 math):**
  - Per key, each lane does 4 `F2F` and 32 `FFMA`, producing 8 partial scores.
  - Every 4 keys, a transpose-reduce over the 32 lanes (xor 16, 8, 4, 2, 1) leaves each lane one finished score: 31
    shuffle-add pairs plus selects, ~31 instructions per key.
  - The microbenchmark also tries lane groups of 8 per key (16 dimensions per lane, the query read from shared memory
    by multicast). Its reductions are shorter, at the cost of shared reads.
- **q8_0 (`dp4a`):**
  - Q is quantized once per block to int8 with one scale per 32 dimensions, the same quantization the stock VEC kernel
    applies before its `dp4a` (`fattn-vec.cuh:150-203`, `quantize_q8_1_to_shared` at `fattn-common.cuh:331-375`).
  - Per key, each lane does one K word and 8 `dp4a`.
  - An integer transpose-reduce runs over the 8 lanes of each q8_0 block, then one `I2F` and the scale by `d_K·d_Q`.
  - Two shuffles add the 4 blocks of a head.
- **Accuracy.** The f16 path has the stock kernels' FP32 accuracy. The q8_0 path has the stock VEC kernel's accuracy (an
  int8 query against int8 keys). The CPU reference in `test-backend-ops` quantizes Q the same way for q8_0 K, so the
  usual 5e-4 normalized error limit applies.

### 5.4 Softmax, masks, sinks

- **Online softmax, base 2, per 32-key tile:**
  - Each lane holds one score per 4 keys. The tile maximum per head is a local max plus 2 shuffles.
  - The V accumulators are rescaled once per tile.
  - Probabilities go to shared memory (one float per lane per 4 keys) and come back as 2 broadcast `LDS.128` per key.
- **Masks.** The lane holding (key, head) adds `mask[key]`: one 2-byte load per lane per 4 keys, 0.4 % of the KV bytes.
  - A warp skips any 32-key group whose mask is all `-inf` (one ballot). That covers the padding up to the next 256
    cells and the 255 out-of-window cells of the 768-cell sliding-window ring buffer, a third of those layers' work.
  - The kernel needs no full mask scan. The host-side cost of the mask (section 3) is a separate fix, needed only if it
    measures ≥ 3 %.
- **Sinks** (one per query head) are applied once, at the merge:
  - `m* = max(m, s_h·log2 e)`;
  - `l* = l·2^(m−m*) + 2^(s_h·log2 e − m*)`;
  - `O = o·2^(m−m*) / l*`.

  This is the same result as the stock kernels, which add the sink in split 0 before combining.

### 5.5 P·V and the merge

- **f16:** per key, each lane does 4 `F2F` and 32 `FFMA` (8 heads × 4 dimensions).
- **q8_0:** per key, each lane does 4 `I2F.S8`, 4 `FMUL` by `d_V` and 32 `FFMA`.
  - `dp4a` for P·V would need probabilities quantized per head and key, and V transposed by key. q8_0's per-token block
    scales don't factor out of that sum, so it isn't planned.
- **Merge:**
  - Each block writes its partials (8 heads × 128 values plus `m`, `l`, per warp) to a pool buffer: 250 KB per layer
    call at S = 15.
  - It then fences and increments a per-device counter. The last block merges the S partials, applies the sinks,
    writes the F32 output and resets the counter.
  - One launch per layer, no host synchronization, nothing for CUDA graphs to re-capture.
  - The stock path needs 2 launches per layer; this saves 49 per token.
- **Instruction estimate (my calc):**

  | Cache | Warp-instructions per key and KV head | Per byte | Issue limit | Expected |
  |---|---|---|---|---|
  | f16 | ~113 (K 37, reduce 31, softmax and probabilities 6, V 37, loop 2) | 0.22 (7.1 thread-instructions) | ~155 GB/s | DRAM-bound, ~130–145 GB/s |
  | q8_0 | ~98 | 0.36 | ~95 GB/s of q8_0 bytes (≈ 180 GB/s f16-equivalent) | ~85–95 GB/s of q8_0 bytes |

  Conversions and shuffles stay under a third of their pipes' rates in both.

### 5.6 Verify batches (2–8 tokens) and two slots

- **Two slots** (`-np 2`, separate streams): one token per stream; the grid's third dimension covers the streams.
- **2 tokens** (`-kvu`, or one draft token): each warp carries both tokens. That is 64 query registers (f16) or 16
  words (q8_0) and 64 accumulators, ~200 registers, 2 blocks per SM.
- **3–8 tokens:**
  - The staged K/V tile is shared by the block, and warps loop over token pairs, so DRAM traffic stays at one read.
  - Each token reads its own mask row, which carries the causal tail and any other sequence's cells.
- **Floor.** At 32k each verified token needs ≥ 3.45 ms of FP32 FMA, ~2.2 ms with `dp4a` for K·Q on q8_0 (my calc). An
  8-token verify at 32k should drop from ~27 ms of attention today (TILE reading K/V twice) to ~19 ms. This path is
  worth building only if speculation goes ahead (Q19, Q51).

### 5.7 Where it hooks in

- **Decision point.** In `ggml_cuda_flash_attn_ext` (`fattn.cu:757-772`), before the kernel switch:

  ```cpp
  if (ggml_cuda_flash_attn_ext_pascal(ctx, dst)) {
      return;
  }
  ```

- **Predicate.** The new kernel runs a call when all of these hold; anything else falls through to the stock kernels
  unchanged:
  - compute 6.1;
  - Q f32 with head size 128 (256 as a second instance for Qwen3.6, GQA 8);
  - K and V both f16 or both q8_0;
  - GQA ratio 8 (a template parameter; 4 and 16 later);
  - 1–8 query tokens;
  - an F16 mask with `ne[2] == 1`, no ALiBi, no softcap;
  - `K->ne[1] % 256 == 0`, and the stride alignment that `gqa_opt_applies` checks.
- **Switch.** `GGML_CUDA_FATTN_PASCAL` = `0` (off), `1` (default: compute 6.1), `2` (any NVIDIA GPU), as for
  `GGML_CUDA_MMVQ_PASCAL`. One binary then gives both arms, and a modern GPU can run the kernel natively and through the
  compute_61 PTX (`CUDA_FORCE_PTX_JIT=1`).
- **Files.** `fattn-pascal.cuh` and `fattn-pascal.cu`, with a stub for HIP and MUSA. `fattn-common.cuh`, `fattn-tile.cuh`
  and `fattn-vec.cuh` stay untouched.
- **Scratch.** `ggml_cuda_flash_attn_ext_get_alloc_size` (`fattn.cu:721-755`) stays as is: a decode graph never needs
  the f16 scratch. The prompt-size reservation still sizes it for q8_0 until TILE reads q8_0 natively (Q61).
- **Profiling.** The fork's profiler keeps labelling the call `FLASH_ATTN_EXT`.

### 5.8 Tests

1. **Flint-shape cases in `make_test_cases_eval`.** The generic FA loop covers head 128 only with GQA 1, 4 and 12, and
   quantized KV only at head sizes 64 and 72, so Flint's shape is missing:

   ```cpp
   for (ggml_type type_KV : {GGML_TYPE_F16, GGML_TYPE_Q8_0}) {
       for (int kv : {256, 768, 1280, 8192, 32768}) {
           for (int nb : {1, 2, 3, 4, 8}) {
               for (bool sinks : {true, false}) {
                   test_cases.emplace_back(new test_flash_attn_ext(128, 128, 4, {8, 1}, kv, nb, true, sinks, 0, 0,
                       GGML_PREC_F32, type_KV, type_KV, {0, 2, 1, 3}));  // permuted views, as in the KV cache
               }
           }
       }
   }
   ```

   Add unpadded lengths (`kv` 113, 1025), ALiBi and softcap cases, which must fall through to the stock kernels.
2. **Correctness runs.** `GGML_CUDA_FATTN_PASCAL=2 test-backend-ops -b CUDA0 -o FLASH_ATTN_EXT` natively and with
   `CUDA_FORCE_PTX_JIT=1`. Then a deliberately wrong kernel (sinks ignored, the last tile dropped, probabilities × 1.01)
   must fail cases, so the tests are shown to reach it.
3. **Kernel speed on one card.** Flint-shape cases in `make_test_cases_perf` (f16 and q8_0; `kv` 768, 8,192, 32,768,
   131,072, 262,144; `nb` 1, 2, 4, 8):
   `test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nh=4,nr23=\[8,1\]'`, kernel on and off. GB/s is KV bytes over
   µs per call.
4. **Model output.** S4-L0, `llama-perplexity -c 4096 --chunks 2 -b 4096 -ub 1`, kernel on against off. Every token goes
   through the decode path at depths up to 4k. Flint's outputs move by KLD ~0.06 from summation order alone
   ([pascal_mmvq_integration.md](pascal_mmvq_integration.md), section 9), so the bar is "within that noise".
5. **Long context.** A needle at 32k and 128k, greedy, against the stock kernel, and a 32k agent prompt in llama-server.

### 5.9 Expected speed (my calc)

Base: S4-L0 on 6 cards, 26.8 ms per token at depth 0; the new kernel also saves ~0.6 ms per token on the sliding-window
layers and the dropped combine launches. f16 at 140 GB/s; q8_0 at 90 GB/s of q8_0 bytes, without and with the rotation
(~1.5 ms):

| Context | Today, f16 (57–87 GB/s) | New kernel, f16 | New kernel, q8_0 |
|---|---|---|---|
| 8,192 | 32.3–33.9 tok/s | 35.9 | 36.2 / 34.4 |
| 32,768 | 23.1–26.6 | 30.4 | 31.5 / 30.1 |
| 65,536 | 16.7–20.7 | 25.2 | 26.8 / 25.8 |
| 131,072 | 10.8–14.3 | 18.9 | 20.7 / 20.0 |
| 262,144 | 6.3–8.8 | 12.5 | 14.2 / 13.9 |

- Rows above ~90k need 9 cards in f16 (section 6), where the base token is a little slower.
- The host's mask work (section 3) takes another ~5 % at 262k unless it is fixed.
- With the new kernel q8_0 and f16 decode at about the same speed: f16 is limited by DRAM, q8_0 by instruction issue.
  q8_0 pays off as capacity once the f16 scratch is gone.

### 5.10 Upstream path

- The new kernel would be a fourth kernel family for compute 6.1, as `mmvq-pascal.cu` is for matrix-vector products.
- Upstream would need measurements on a larger 6.1 card (GTX 1080, P40) showing no regression there.
- The FP32 q8_0 loader for TILE (Q61) is a small change on top of PR #30792, and it fixes that PR's sm_61 build.
- The TILE retile (ik_llama.cpp PR #2142's idea applied to decode) is the smallest upstream change that helps every Pascal
  user with an f16 cache.

### 5.11 Built and measured (Q151)

The kernel is in the fork as `fattn-pascal.cu` (branch `pascal-p106-v2-q151`, `8d30766c2`; in v2.0 behind
`GGML_CUDA_FATTN_PASCAL`, default off): F16 KV, head 128, 8 query heads per KV head, one token; split keys, last-block
merge, one launch.
- **Correct** (V100 through the compute_61 PTX): `FLASH_ATTN_EXT` 24 of 24 Flint-shape cases; dropping the sinks fails
  exactly the 6 sink cases it takes and dropping the last tile all 18 cases it takes; × 1.01 on the output stays
  inside the 5e-4 NMSE limit. G3, 4 × 512 tokens at `-ub 1`: mean KLD 0.052 ± 0.008 and same top token 91.2 ± 0.9 %
  against the stock kernel, where the summation-order reference (stock attention, Pascal matrix-vector kernels against
  llama.cpp's) gives 0.047 ± 0.007 and 92.7 ± 0.8 %.
- **Slower on the rig** (release candidate, K2h-r16, nine cards): tg128 36.80 with it against 40.25 without (−9 %), and
  after 32k tokens of context 15.0 against 25.4 tok/s, against section 5.9's estimate. v2.0 ships it off
  ([engine features](../../engine_features.md); data: [checks](../../data/2026-10-24_q88_persistent/README.md),
  [rig](../../data/2026-10-25_release_rig/README.md)).

## 6. KV memory per card at long context

Per full-attention layer at 262,144 tokens: f16 512 MiB, q8_0 272 MiB, q4_0 144 MiB. The sliding-window cache adds
52.5 MiB in f16 across all cards (27.9 MiB in q8_0).

With an even split by layer count, the full-attention layers fall like this:

| Cards | Full-attention layers per card |
|---|---|
| 6 | 3, 2, 2, 2, 2, 3 (layers 0/1/5 and 41/45/48 on the end cards) |
| 9 | 2, 2, 1, 2, 1, 1, 2, 1, 2 |

**The f16 scratch for quantized caches.**
- llama.cpp reserves its compute buffers with a prompt-size graph (`-ub` tokens over the whole context). For a quantized
  cache that graph runs TILE with an f16 copy of one layer's K and V (`ggml_cuda_flash_attn_ext_get_alloc_size`,
  `fattn.cu:721-755`, applied through the buffer type at `ggml-cuda.cu:969-986`).
- That costs `n_ctx` × 2 KiB per card, 512 MiB at 262,144 tokens, on every card that holds a full-attention layer. On
  this model that is every card.
- The Qwen3.6 measurement fits this: q8_0 took a four-card copy from 98,304 to 131,072 tokens, a third more rather than
  double ([Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md)).

**KV and scratch on one card at 262,144 tokens (my calc):**

| Cache | 3 layers (6 cards, ends) | 2 layers | 1 layer |
|---|---|---|---|
| f16 | 1,536 MiB | 1,024 MiB | 512 MiB |
| q8_0 with today's scratch | 1,328 MiB | 1,056 MiB | 784 MiB |
| q8_0 without scratch (Q61) | 816 MiB | 544 MiB | 272 MiB |
| q4_0 with scratch | 944 MiB | 800 MiB | 656 MiB |

Plus the compute buffer, ~285 MiB per card at `-ub 128` for Qwen3.6, of the same order for Flint.

**What fits.**
- **S4-L0 (32.5 GB) on 9 cards** leaves ~2.5 GB per card. The full 262,144-token context fits in f16 with room for a second
  262k slot, so a cheaper cache type changes nothing there.
- **On 6 cards** S4-L0 leaves roughly 0.55 GB per card for KV after weights and compute buffers (claim to check against
  the load log). The end card with three full-attention layers sets the limit:

  | Cache | Maximum context (my calc) |
  |---|---|
  | f16 | ~90k |
  | q8_0 as today | ~104k |
  | q8_0 without the scratch | ~169k |
  | q4_0 without the scratch | ~318k |

  `-ts` that moves a layer off the end cards raises each figure by about a third.
- q4_0 KV is not an option for quality: KLD +67 % on a community card for Flint, against +6 % for q8_0
  ([flint_speedups.md](flint_speedups.md), section 4).

## 7. Measurement plan for the rig

All on the fork build with S4-L0. `BIN` is the fork build's `bin/`, `MODEL` the S4-L0 file. Each depth run prefills its
context first (~4–5 min at 32k, ~10 min at 64k, ~30 min at 131k).

**Q54, depth sweep.** Run f16 and q8_0 separately: mixed K/V type pairs aren't compiled by default and fall back to f16
conversion.

```bash
for kv in f16 q8_0; do
  CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 64 -r 1 \
      -d 0,8192,32768 -ctk $kv -ctv $kv -o jsonl
done
# 64k and 131k need the ninth-card headroom
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8 $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 64 -r 1 \
    -d 32768,65536,131072 -o jsonl
```

- `-ub 128` matches the server's sliding-window cache (768 cells).
- Repeat the 32k point once to see run-to-run spread.
- Watch the log for `MUL_MAT failed` / `cudaStreamSynchronize` with q8_0 (the P40 crash of issue #22032).
- [scripts/bench.sh](../../../scripts/bench.sh) runs the same with `BIN=… GPU_SETS="0,1,2,3,4,5" BENCH_ARGS="…"`.

**Attention's share and host time at 32k:**

```bash
GGML_CUDA_PROFILE=/tmp/prof-d32k.csv CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 32 -r 1 -d 32768
python3 scripts/cuda_profile_summary.py /tmp/prof-d32k.csv --tokens 24 --by op
GGML_CUDA_PROFILE_HOST=/tmp/host-d32k.csv CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 \
    $BIN/llama-bench -m $MODEL -ngl 99 -fa on -ub 128 -p 0 -n 32 -r 1 -d 32768
python3 scripts/cuda_profile_summary.py /tmp/host-d32k.csv --tokens 24
```

- The first summary gives `FLASH_ATTN_EXT` (kernel plus combine) per token and, with q8_0, the rotation launches
  (`MUL_MAT attn_inp_k_rot…`).
- Events inflate absolute times on these cards (Q49), so use the profile for shares and per-call times, and llama-bench
  for tok/s.
- The host timeline shows whether wall time per token exceeds the GPU chain at depth: mask fill and copies.
- Repeat both with `-ctk q8_0 -ctv q8_0`, and with `LLAMA_ATTN_ROT_DISABLE=1` added. For the rotation's quality cost,
  run [quality_kld.sh](../../../scripts/quality_kld.sh) with the same switches on the quant machine.

**Kernel only, one card**, after adding the Flint perf cases of section 5.8 (no model load, minutes):

```bash
CUDA_VISIBLE_DEVICES=4 $BIN/test-backend-ops perf -o FLASH_ATTN_EXT -b CUDA0 -p 'nh=4,nr23=\[8,1\]'
```

GB/s = 2,048 B × `kv` (f16; 1,088 B for q8_0) per call over µs per call. This separates the kernel from the mask copies,
the rotation and the clock behaviour of six busy cards.

**Instruction throughputs** that the design assumes (`F2F`, `I2F.S8`, `SHFL`, broadcast and multicast `LDS.128`, `PRMT`)
are timed in the microbenchmark of Q59 on one card before the port.

## 8. Experiments, ranked by value for effort

| Registry | Experiment | Where, effort | Expected (my calc unless measured) | Decides |
|---|---|---|---|---|
| Q54 | Depth sweep 0 / 8k / 32k on 6 cards (64k / 131k on 9), f16 and q8_0 KV, one `GGML_CUDA_PROFILE` and one `GGML_CUDA_PROFILE_HOST` run at 32k | rig, 2–3 h (mostly prefill) | f16: 32–34 tok/s at 8k, 23–27 at 32k, attention 29–38 % at 32k; q8_0: ~29 at 8k, ~18 at 32k | Gate for Q59–Q60 (attention ≥ 30 % at 32k); calibrates section 3's model; whether q8_0 is usable at all today |
| Q62 | Host-side and rotation costs: host timeline at 32k and 131k; q8_0 with and without `LLAMA_ATTN_ROT_DISABLE=1` (speed, KLD) | rig 1 h, quant machine 1 h | rotation 1–2 ms per token; mask ≤ 0.5 ms at 32k, 2–4 ms at 262k | Whether q8_0 runs with rotation; whether a mask-upload change (only changed cells) is worth 1–2 days |
| Q59 | Standalone microbenchmark `bench/fattn_pascal` at Flint's shapes on one card, against a CPU double-precision reference: stock TILE copy, the lane-per-4-dimensions kernel (f16, q8_0), the lane-group variant, the TILE retile | code + one card, 2–3 days | f16 ≥ 120 GB/s, q8_0 ≥ 80 GB/s of q8_0 bytes, retile 95–110 GB/s | Which variant goes into the fork; stop if f16 stays < 110 GB/s |
| Q60 | Port into the fork (`fattn-pascal.cu`): 1 token and two slots, f16 and q8_0, sinks, masked-tile skip, last-block merge; tests of section 5.8; rig | fork, 3–4 days | S4-L0 at 32k ~25 → ~30 tok/s, at 131k ~12 → ~19–21; +2 % at short context | Default for long-context serving |
| Q61 | Native q8_0 in TILE's FP32 path (PR #30792 with a float loader) and no f16 scratch in `get_alloc_size` | fork, 1 day | −512 MiB per card at 262k with q8_0; prefill within ±5 %; fixes #30792's sm_61 build | Whether q8_0 makes 6 cards hold 150k+ tokens |
| Q63 | Verify path for 2–8 tokens in the new kernel | fork, 2–3 days, after Q19 and Q51 | 8-token verify at 32k ~27 → ~19 ms of attention | Only if speculation goes ahead |
| — | TILE retile for decode (`half2` tiles, conversion at use, scores in registers) as an upstream PR | 1–2 days | +30–50 % attention for any Pascal card with an f16 cache | Optional; the upstream-friendly part |
| — | q4_0 KV | — | Same speed as q8_0 in the new kernel; +67 % KLD | Not recommended |

## 9. Prompt kernel for compute 6.1 (Q163)

Reading a long prompt is the other half of long context: on nine cards a 64k-token prompt read at 67.8 tok/s (Q116,
about 16 minutes), against 714 tok/s for an 8k prompt (Q109). Agents hit this with tool outputs, files and long histories.

### 9.1 What the stock kernel does with a prompt micro-batch

- **Selection.** F16 or Q8_0 K/V with 3 or more query tokens run TILE on compute 6.1 (section 2). With 8 query heads per
  KV head the launch puts the 8 heads and 4 tokens in one block from 3 tokens up: `flash_attn_tile<128,128,4,8>` for
  Flint, `<256,256,4,8>` for the Qwen3.6 class (16 query heads, 2 KV heads, head size 256 on 10 attention layers).
- **FP32 configuration** (`fattn-tile.cuh:131`, `:143`): 256 threads, 2 blocks per SM, 64-key tiles at head 128 and
  32-key tiles at head 256, K staged in 64-dimension slices, K and V converted to FP32 in shared memory. A Q8_0 cache is
  first converted into an F16 copy of the whole K and V view on every call.
- **Register reuse.** Per 2-dimension step each lane loads 2 keys and 4 query columns (12 floats) for 16 FMAs; in P·V it
  loads 4 values and 4 probabilities (8 floats) per 16 FMAs.
- **Why that is slow on sm_61 (my calc).** The path from shared memory into registers moves 128 B, 32 floats, per clock and
  SM, against 128 FMA lanes: a loop is FMA-bound only below 0.25 floats loaded per FMA. The stock layout loads 0.75
  (scores) and 0.5 (P·V), which caps it at 33 % and 50 % of the FP32 peak, 40 % for the pair, before barriers and the
  staging.
- **Measured.** The rig's fills put attention at ~0.33 ms per prompt token per 1k of depth (Q116: 0 → 32k in 306 s,
  32k → 64k in 661 s). At 229 MFLOP per 1k of depth that is ~0.7 TFLOPS, ~32 % of a P106's FP32 peak (2.2 TFLOPS at
  ~1.7 GHz), so the kernel runs at about 80 % of its own ceiling.

### 9.2 What the starting points give on this card

- **ik_llama.cpp PR #2142** keeps K and V as F16 in shared memory and converts them at the point of use. On sm_61 a
  conversion runs at a quarter of the FMA rate and each converted value feeds 4 FMAs in that layout, so the conversions
  cost about as much issue time as the halved shared-memory traffic saves. Its +4–9 % on a P100 came partly from a second
  block per SM, which the P100's 64 KB of shared memory needed and a P106 (96 KB) already has; the PR says the same.
- **ik_llama.cpp PR #2580** sends head-256 prompts from ik_llama.cpp's VEC kernel, which reads K and V once per query
  column, to its FP32 tile kernel (+95 % at 16k on a P100). Mainline llama.cpp, this fork's base, already runs TILE for
  head-256 prompts, so that gain is not available here.
- **Upstream PR #30792** dequantizes Q8_0 into `half2` and asserts FP16 accumulation (finding 4); the new kernel has its
  own FP32 Q8_0 loader (Q61).

### 9.3 The kernel

Fork branch `pascal-p106-v2-prefill-attn` (off `pascal-p106-v2-next`): `ggml/src/ggml-cuda/fattn-tile-pascal.cu`, hooked
into the TILE case of `ggml_cuda_flash_attn_ext` and into the F16-copy reservation (`get_alloc_size`).

- **Scope.** FLASH_ATTN_EXT with an F32 query of head size 128 or 256, a multiple of 8 query heads per KV head, F16 or Q8_0
  K and V (both the same type), an F16 mask, optional sinks, no ALiBi or softcap, a KV length padded to 256, from 9
  query tokens (3 with Q8_0, where TILE would otherwise convert the whole cache). Anything else runs the stock kernels.
- **Block.** 4 warps on 32 columns (4 tokens × 8 heads) at head 128, 16 columns (2 tokens × 8 heads) at head 256; 64-key
  softmax tiles at head 128 (the stock tile), 128 at head 256; 30 KiB of shared memory and at most 168 registers, so 3
  blocks (12 warps) fit an SM.
- **Scores.** At head 128 a lane takes 4 keys × the warp's 8 columns over one half of the head dimension (128 FMAs per 48
  floats loaded, 0.375 per FMA, a 67 % ceiling), and lane l and l ^ 16 then swap the partial sums of each other's columns.
  At head 256 (and at head 128 with `GGML_CUDA_FATTN_TILE_PASCAL_QK=1`) a lane takes 4 keys × 4 columns over the whole
  head (0.5 per FMA). All loads are `float4`, conflict-free (padded rows) or broadcast.
- **Softmax.** Each lane's 4 columns are the same in the score and the output layout, so the running maximum (a shuffle
  reduction over the 16 or 32 lanes of a column) and the rescale of the output stay in registers, and each warp's
  probabilities pass through its own slice of shared memory without a block barrier. The maximum offset, the sinks and the
  split-KV output are those of the stock kernel, so launch_fattn's combine step works unchanged.
- **P·V.** A lane owns 4 columns × 8 dimensions (two `float4` at 4l and 4l + D/2): 4 probabilities and 8 values per 32
  FMAs (0.375 per FMA, a 67 % ceiling). Keys are summed in order, as in the stock kernel.
- **Staging.** K comes in chunks of 16 (head 128) or 8 (head 256) dimensions × all tile keys, V in chunks of 8 or 4 keys ×
  all dimensions, each thread loading 8 values of one key one chunk ahead into registers: F16 as one 16-byte load,
  Q8_0 as aligned 32-bit words shifted into place (its 34-byte blocks are only 2-byte aligned) and dequantized in FP32
  (`d × q`, as the CPU does). A Q8_0 cache therefore needs no F16 copy: at 262k tokens that is 512 MiB less per card that
  holds a full-attention layer (section 6).
- **Masked tiles.** A key tile whose mask is -inf for every token of the block is skipped (one vote at the tile's first
  barrier): about half of Flint's sliding-window ring at `-ub 512` (1,280 cells; a block's 4 tokens see ~516 of them, 9–10
  of the 20 tiles) and the causal padding of full layers.
- **Switches.** `GGML_CUDA_FATTN_TILE_PASCAL` 0 off, 1 compute 6.1 only (default), 2 any NVIDIA GPU;
  `GGML_CUDA_FATTN_TILE_PASCAL_QK` 2 split scores at head 128 (default), 1 plain; `GGML_CUDA_FATTN_TILE_PASCAL_MIN` the
  fewest query tokens; `GGML_CUDA_FATTN_TILE_PASCAL_FAULT` 1/2/3 (output × 1.05, sinks dropped, last key tile dropped) to
  check that the tests reach the kernel. A log line names the head size, cache type and score layout on first use.
- **Numerics.** FP32 throughout. At head 128 with the plain layout the scores, maxima, probabilities and P·V sums are those
  of the stock kernel bit for bit; only the softmax denominators are summed in a different order. The split layout adds
  the two halves of each score, and head 256 uses longer softmax tiles than the stock kernel (128 against 32 keys).

### 9.4 Checks (quant machine: a V100 running the compute_61 PTX, kernel forced with `=2`)

- `test-backend-ops -o FLASH_ATTN_EXT`: all 4,172 cases pass with the kernel forced, among them 168 new prompt cases (head
  128 and 256, F16 and Q8_0, KV 256–4,096, 3–512 tokens, sinks, large logits, causal and 513-key window masks, 16 query
  heads per KV head, two sequences; ALiBi, softcap, 4 query heads per KV head and an unpadded KV length fall through).
  The 182 cases at these shapes also pass with the plain score layout and with the kernel off.
- The tests reach the kernel: with the output scaled by 1.05, or the last key tile of each block dropped, exactly the 145
  cases the kernel takes fail (the 37 that pass run the stock kernels: decode, ALiBi, softcap, BF16, unpadded KV, F16 at 3
  tokens); with the sinks dropped, its 66 sink cases fail, all but the large-logit ones, where a sink carries no weight.

### 9.5 Expected speed (my calc) and the rig run

- **SASS (sm_61, CUDA 12.9).** The stock kernel's 64-key tile is one unrolled loop of 3,005 instructions: 2,052 FMAs and
  448 shared loads, ~2.5 bytes from shared memory per FMA, FMAs 68 % of the instructions. The new kernel's tile at head
  128 (split) runs 4,096 FMAs in 8 K and 8 V chunk loops of 256 FMAs and 24 `LDS.128` each, 1.5 bytes per FMA, FMAs 74 % of
  the instructions per tile with the softmax and staging included. 155–168 registers, 30 KiB of shared memory, no spills
  except 40 bytes in the split Q8_0 instance.
- **Kernel.** With both loops at a 67 % ceiling against the stock kernel's 40 %, and the same ~80 % of the ceiling reached,
  attention at depth runs ~1.6× faster (~1.1–1.2 TFLOPS per card). If instruction issue rather than shared memory
  limited the stock kernel, the gain would be smaller (~1.1–1.3×, from the FMA shares); the rig's kernel test tells the
  two apart. The skipped tiles of the sliding-window ring save another ~0.3–0.4 ms per prompt token at any depth.
- **Prompts.** At 64k (Q116's build: ~10.6 of ~14.7 ms per prompt token in attention) 1.6× faster attention gives ~10.7 ms,
  ~93 tok/s instead of 68 (1.4×); the newer build's faster expert products make attention a larger share and the gain
  larger.
- **Rig run** (`T=<tarball>` [rig_q163.sh](../../data/2026-10-24_q163_prefill_attn/run/rig_q163.sh); ~80 minutes with 64k): kernel speed on one card for both score
  layouts against the stock kernel (test-backend-ops perf on the prompt cases), K2h-r16 KLD over one 8k chunk, prompt speed
  at 8k/32k/64k for K2h-r16 and Qwen3.6 Q4_0-noK on nine cards, and a Q8_0 cache's compute buffers at 64k, kernel off
  against on in one binary.
- **Measured on the rig** (the script's step 2, run inside the release candidates' rig script): K2h-r16 on nine cards,
  llama-batched-bench `-b 2048 -ub 512`, 8,192 tokens 178.5 → 204.2 tok/s (+14 %), 32,768 tokens 120.6 → 138.7 (+15 %);
  whole-prompt batches on v2.0 699 → 766 and 447 → 461. The 64k, Qwen3.6 and one-card kernel steps have not run, so the
  1.4× estimated at 64k is unmeasured. Checks and scripts:
  [docs/data/2026-10-24_q163_prefill_attn](../../data/2026-10-24_q163_prefill_attn/README.md).

## Sources

llama.cpp, fork base `d77256999` (identical to master `36e36e6` for the attention kernels; P):
- `ggml/src/ggml-cuda/fattn.cu`: `ggml_cuda_get_best_fattn_kernel` (`:540-719`), `get_alloc_size` (`:721-755`), dispatch
  (`:757-772`).
- `ggml/src/ggml-cuda/fattn-tile.cuh`: FP32 configuration (`:90-156`), half-to-float tile loader (`:428-476`), kernel
  (`:793-1146`), sinks (`:1052-1080`), launch switches (`:1148-1320`).
- `ggml/src/ggml-cuda/fattn-vec.cuh`: kernel (`:19-528`), q8_1 query (`:97`, `:150-203`), sinks (`:381-412`), launch
  (`:533-574`).
- `ggml/src/ggml-cuda/fattn-common.cuh`: `FATTN_KQ_STRIDE` (`:9`), q8_0 dot and dequantization (`:303-329`,
  `:592-623`), KV-length scan (`:669-724`), combine (`:922-978`), `launch_fattn` (`:980-1314`).
- `ggml/src/ggml-cuda/common.cuh` (`:268-274`, `:400-410`), `ggml-cuda.cu` (`:969-986`, `:1980`), `fwht.cu`
  (`:215-222`).
- `src/llama-graph.cpp` (`:30-47`, `:2948-3000`, `:3494-3562`, `:3818-3822`), `src/llama-kv-cache.cpp` (`:316-359`,
  `:1285-1296`, `:1575-1806`), `src/llama-kv-cache-iswa.cpp` (`:73`), `src/llama-impl.h` (`:57-75`),
  `ggml/src/ggml-backend.cpp` (`:1823-1864`), `tests/test-backend-ops.cpp` (`:8557-8687`, `:12005-12066`,
  `:12744-12753`).
- The fork's sm_61 build at `eeaa59660`: register, shared-memory and per-loop instruction counts of `flash_attn_tile`
  `<128,128,1,8>`, `<128,128,2,8>`, `<128,128,4,8>`, `flash_attn_ext_vec` `<128,1|2,f16|q8_0>` and
  `flash_attn_combine_results<128>`.
- PR #30792 at `3c25375492`, applied to the fork base and compiled for sm_61 (head-128 TILE instance, CUDA 12.9): fails
  at its FP16-accumulation static assertion; the unpatched file compiles.

llama.cpp PRs and issues (P): [#30792](https://github.com/ggml-org/llama.cpp/pull/30792),
[#30790](https://github.com/ggml-org/llama.cpp/issues/30790), [#30202](https://github.com/ggml-org/llama.cpp/pull/30202),
[#23907](https://github.com/ggml-org/llama.cpp/pull/23907), [#29827](https://github.com/ggml-org/llama.cpp/pull/29827),
[#29846](https://github.com/ggml-org/llama.cpp/pull/29846), [#29371](https://github.com/ggml-org/llama.cpp/issues/29371),
[#29826](https://github.com/ggml-org/llama.cpp/issues/29826), [#22137](https://github.com/ggml-org/llama.cpp/pull/22137),
[#22032](https://github.com/ggml-org/llama.cpp/issues/22032), [#17558](https://github.com/ggml-org/llama.cpp/pull/17558),
[#22541](https://github.com/ggml-org/llama.cpp/pull/22541), [#30190](https://github.com/ggml-org/llama.cpp/pull/30190).

Other engines (P):
- ik_llama.cpp [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md),
  PRs [#2580](https://github.com/ikawrakow/ik_llama.cpp/pull/2580), [#2647](https://github.com/ikawrakow/ik_llama.cpp/pull/2647),
  [#2142](https://github.com/ikawrakow/ik_llama.cpp/pull/2142), [#1973](https://github.com/ikawrakow/ik_llama.cpp/pull/1973).
- FlashInfer `include/flashinfer/attention/decode.cuh` at `89009ad445`; Ye et al., FlashInfer, [arXiv 2501.01005](https://arxiv.org/abs/2501.01005).
- vLLM `csrc/attention/attention_kernels.cuh` and `paged_attention_v2.cu` at tag v0.9.2.
- Dao et al., [Flash-Decoding for long-context inference](https://crfm.stanford.edu/2023/10/12/flashdecoding.html);
  Sanovar et al., Lean Attention, [arXiv 2405.10480](https://arxiv.org/abs/2405.10480).
- HyperQwen [docs/long-context.md](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/long-context.md)
  at `ce9615c`.

This repository:
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md): depth speeds of three files, 262k memory on nine cards.
- [Qwen baselines](../../benchmarks/2026-10-22-qwen-baselines.md): compute buffer per card, q8_0 against f16 context on
  four cards, q8_0 speed cost.
- [hardware spec](../../mining_rig.md): bandwidth, FP16 rate, PCIe links.
- [pascal_mmvq_integration.md](pascal_mmvq_integration.md): staging, force switch, test method, profiler.
- [hyperqwen_lessons.md](hyperqwen_lessons.md) (section 6), [flint_speedups.md](flint_speedups.md) (sections 0 and 4),
  [dense_27b_speedups.md](dense_27b_speedups.md) (KV cache and flash attention on sm_61; P100 f16 against q8_0 at 75k).
- [quants/experiments.md](../../../quants/experiments.md): Q19, Q49, Q51, Q54.
