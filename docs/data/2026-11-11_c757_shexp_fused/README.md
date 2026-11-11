# The shared expert's gate, up and SWIGLU of 2–8 tokens as one launch (C757, Q1333, 2026-11-11)

What this is: a new CUDA kernel of Miner Llama's v4 line, built and checked without a GPU; its timing on a P106 is the
rig run below. In a speculative verify pass Qwen3.8-Flash-Next's shared expert runs, on 41 of its 47 layers, the gate
and the up product (2,560 → 640 each) as two small launches and their SWIGLU as a third, because llama.cpp fuses gate,
up and GLU of several tokens only for the routed experts ([C731](../2026-11-11_c731_shared_expert/README.md)). At four
tokens each of those products takes 20–60 µs a call against est. 3–6 µs to read its weights: the launches are bound by
latency, not by memory. `GGML_CUDA_MMVQ_PASCAL_GLU_MT=1` runs the three as one launch, with every gate and up sum the
separate product's bit for bit. Registry row [Q1333](../../../quants/experiments.md).

**The answer:**
- **Mixed types do not stop the fusion.** The release file (Swift GSQ-RCO Q2_0) mixes the gate's and the up's types on
  33 of the 41 layers. The kernel's warps work in pairs, one on rows of the gate and one on the same rows of the up, and
  each warp runs its own type's code in that type's one-token summation order. So a pair of types needs no common
  order, and no sum changes.
- **Layers in one launch** at Flash-Next's verify (4 tokens), with the read-once kernels the launch repeats switched on
  (`GGML_CUDA_MMVQ_PASCAL_DMT=1 _DMT_R=1`, `GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1`): **39 of the 41** (31 of the 33 mixed
  ones and all 8 of one type). The 2 left are the two with a Q6_K gate or up, whose separate products have no read-once
  kernel yet ([C727](../../../quants/experiments.md)). Without C722's and C690's switches only the Q2_0 and Q3_K pairs
  are taken: 17 layers. At 5–8 tokens the same 17.
- **Exact by construction, checked where a CPU can check it:** the rig tarball builds; every other object's sm_61 code
  is the base's; the new kernel has no spills and no I2F outside the activation staging; its SWIGLU is the separate GLU
  kernel's instruction sequence (FMUL by −log2 e, RRO, MUFU.EX2, FADD 1, MUFU.RCP, FMUL, FMUL) in all 14
  instantiations; `test-llama-archs` gives every architecture's logits unchanged on the CPU with and without the switch;
  ctest passes. The bit identity on the card (against the products and the SWIGLU run apart) and the speed are the rig
  run's.

## The kernel

`mul_mat_vec_q_pascal_glu<NT, PF>` in `ggml/src/ggml-cuda/mmvq-pascal-glu.cu` of fork branch
`pascal-p106-v4-b12246-c757` ([`1a8c38ea0`](https://github.com/Cinderella-Man/miner-llama/commit/1a8c38ea0)), on
`-c748` `5b329df16`. A hook in `ggml_cuda_try_fuse` (`ggml-cuda.cu`) hands it a `MUL_MAT`, `MUL_MAT`, `GLU` triple
(either product first) whose two products read the same activations, before the generic fusions.

- **One staging of the activations per block.** The activations are quantized once, as `ggml_cuda_mul_mat_vec_q` does
  for the separate products (`quantize_row_q8_1_cuda`). Each block of a persistent grid copies every token's Q8_1
  blocks into shared memory: the int8 words in their own order, d8 per 32 values and token, and only the terms its two
  types need, computed as the separate kernels compute them: S = d8 · Σ int8 for Q4_K and Q5_K (as `mmvqp_load_act`
  and C722's staging), and Q3_K's d8 / f and its two offsets −4 f · Σ of each half (as `kcol::quantize_records<KQ3_K>`
  and C726's staging). Groups of four 32-value blocks are padded so that the lanes of a row read distinct banks.
- **Warp pairs.** Block (32, 2 np): warp 2i takes a row block's gate rows, warp 2i + 1 the same up rows. A row block
  has the larger of the two types' rows per warp pass (32 / L); the other type's warp makes two or more passes. At
  2,560 weights: 16 rows, Q2_0's warp in two passes of 8. Each warp stages its lane groups' rows (L units per
  iteration) into shared memory from the 16-byte boundary below each segment, with C722's register prefetch of the
  next segments. When both warps have their row block's sums in shared memory they meet at a named barrier
  (`bar.sync 1 + i, 64`) and write `ggml_cuda_op_silu_single(gate) * up`, the separate GLU kernel's expression
  (`op_silu(x) * g` in `unary_gated_op_kernel`), on the same two FP32 values. The result buffers are double-buffered
  by row block, so one barrier per row block suffices.
- **Each row's sum is the separate product's.** L lanes per row (the largest power of two, at most 32, that divides
  the row's units), lane sl taking units sl, sl + L, … in order, the lanes' sums added with the same xor shuffles, and
  within a unit each type's own arithmetic:

  | Type | Unit | The order it repeats | Per 32 or 64 weights and token, then per unit |
  |---|---|---|---|
  | Q2_0 | 2 blocks, 128 weights | the k-column kernel's (`kcol::unit_q2_0`), the one-token kernel's too (C731) | acc = fmaf(d, fmaf(i1, d8₁, i0 · d8₀), acc) per 64 weights |
  | Q3_K | super-block | the k-column kernel's (`kcol::unit_q3_K`), its one-token kernel (C726) | acc = fmaf(fmaf(i1, d · (sc1 − 32), i0 · (d · (sc0 − 32))), d8 / f, acc) |
  | Q4_K, Q5_K | super-block | the one-token Pascal kernel's (`mmvqp_dot_q4_K`, `_q5_K`; C722's and C690's kernels) | sumd = fmaf(d8, isum · sc, sumd), summ = fmaf(m, S, summ); acc + d · sumd − dmin · summ |
  | IQ4_XS | super-block | the one-token Pascal kernel's (`mmvqp_dot_iq4_xs`, C722's kernel) | sumf = fmaf(d8, isum · (ls − 32), sumf); fmaf(d, sumf, acc) |

  - The integer sums do not depend on the order in which the int8 products are added. Q2_0's and IQ4_XS's table
    lookups return their weights interleaved, (0, 2, 4, 6), (1, 3, 5, 7) and (4k, 4k + 16, 4k + 1, 4k + 17),
    (4k + 2, …). So the two lookups' words are put back in the activations' order with two `__byte_perm` per pair:
    `0x5140`/`0x7362` for Q2_0, `0x6420`/`0x7531` for IQ4_XS. That is per row, not per token, and one staging serves
    both types.
  - The integer-to-float conversions are made without the quarter-rate I2F, exactly, as in C722's and C726's kernels:
    the dp4a chains start at 0x4B400000 (12,582,912 read as a float), so the result less 12,582,912 is the integer.
    The 6-bit scales and mins pass through `__byte_perm` into the same float.
- **Which products it takes.** A pair is fused only where both separate products run the kernel this one repeats:
  - Q2_0 and Q3_K at 2–8 tokens where the k-column kernels run (`GGML_CUDA_MMVQ_PASCAL_KCOL` not 0, the type not in
    `_KCOL_SKIP` or `GGML_CUDA_MMVQ_PASCAL_SKIP`);
  - IQ4_XS and Q4_K at 2–4 tokens only where `GGML_CUDA_MMVQ_PASCAL_DMT`'s kernel takes the separate product;
  - Q5_K at 2–4 only where `GGML_CUDA_MMVQ_PASCAL_Q5K_MT`'s does.

  Anything else stays three launches: other types (Q6_K), other GLU ops, a swapped GLU, products of other shapes, and
  launches whose shared memory would pass 48 KB. With the switch at 0 the hook does nothing.
- **Settings:** `=2` without the prefetch; `_NP=n` warp pairs per block (1–4; default the most that fit 48 KB);
  `_FAULT=1` stages each token's activations as the next token's, a fault the checks must catch.

The launch shapes at Flash-Next's shared expert ([shapes.tsv](shapes.tsv), by [run/shapes.py](run/shapes.py), the host
code's rule written out again): 16 rows a row block, 40 row blocks, 1–4 warp pairs a block, 33.3–48.5 KB of shared
memory a block (2 blocks a SM by shared memory). At four tokens the activations take 11.9–17.4 KB, a pair's staging
buffers 5.8–11.3 KB. With the prefetch the kernel holds 212–240 registers a thread at 2–4 tokens, so a block of 4
pairs runs alone on its SM; the perf step times 1 pair a block beside the default.

With the k-column dispatch table on (`GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=1`, the default outside the Flash-Next preset), a
Q3_K product whose k-column launch shape is the fallback (rows of 6,144 weights and wider from 4–6 tokens) goes to
llama.cpp's kernel, in another order; a fused launch of such a shape keeps the k-column order. Under the preset
(`_KCOL_TABLE=0`), and at the shared expert's 2,560-wide rows either way, the k-column kernel takes every Q3_K product.

## Checked on a CPU (no GPU)

On the quant machine, in containers without GPUs, at low priority ([run/checks.sh](run/checks.sh); [summary.tsv](summary.tsv)):

| Check | Verdict | Detail |
|---|---|---|
| The rig tarball builds | OK | `llama.cpp-20261111-1a8c38ea0-pascal-p106-v4-b12246-c757.tar.gz`, 662,444,152 B, md5 `3a7992db8bbf0756d86629ed7222955e`; `TESTS=1`, sm_61, 0 AVX, GLIBC_2.38; its `test-backend-ops` md5 `8ba2084f676a842e3e82cf7cbeb3d61d`; no warning from the new file |
| Every other object's sm_61 code | the same | 194 objects of the CUDA backend against `5b329df16`'s rig build, the anonymous namespaces' hashes masked: 193 the same, `mmvq-pascal-glu.cu.o` new ([sass_objects.tsv](sass_objects.tsv)); `ggml-cuda.cu.o`'s device code is unchanged by the hook |
| The new kernel | OK | 14 instantiations (2–8 tokens, with and without the prefetch), 140–240 registers, no stack or local memory; 3 I2F each, all in the activation staging (d8 / f, d8 · Σ), none in the products ([sass_counts.tsv](sass_counts.tsv)) |
| Its SWIGLU | the same | 322 of 322 inlined copies give the separate GLU kernel's sequence ([swiglu.txt](swiglu.txt)) |
| The replay set `glumt` | recorded | 636 cases, 9 runs, 353,648,640 B, md5 `532343f760298f4b989f1b766c10de7f` ([replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv)) |
| `multitoken_bitid` with mode `glu` | builds | md5 `d79bccc3b190beba7446d44db1b6beff`, against the tip's build tree |
| `test-llama-archs -b CPU -s 1` | the same | 146 architectures' logits, `5b329df16`'s CPU build against the tip's: 0 differ with no switch and 0 with `GGML_CUDA_MMVQ_PASCAL_GLU_MT=1` (a CUDA switch; the CPU path is untouched) |
| ctest (static CPU build) | OK | 70 of 70 |

## The rig run (R107)

What only a P106 can say, by [run/rig_c757.sh](run/rig_c757.sh) on a rig with the branch's tarball, the replay set
`glumt` and `multitoken_bitid`:

- **replay:** the recorded test-backend-ops set `glumt` (`scripts/record_checks.sh SUITE=glumt`, recorded on the build
  machine's CPU with the rig tarball's own `test-backend-ops`): the `MUL_MAT_GLU` cases at 1–8 tokens under nine
  settings, each expected to pass but the fault.
- **bitid:** `multitoken_bitid glu` (bench/mmvq_pascal/verify): every fused launch against the two products and the
  SWIGLU run apart, at the same tokens and token by token, bit for bit, at `=1`, `=2`, `_NP=1` and off, then the fault
  (must differ).
- **perf:** `test-backend-ops perf -o MUL_MAT_GLU` at the shared expert's shape for the 12 pairs of Q2_0, Q3_K, Q4_K
  and IQ4_XS (37 of the 41 layers), 1–8 tokens, arms off (the separate products on the read-once kernels, R79's
  baseline), on, without the prefetch, one pair a block; two passes.
- **served:** R49's profile of a served verify pass with the MTP draft, the preset plus the read-once switches
  (`_DMT=1 _DMT_R=1 _Q5K_MT=1`) in every load, the switch off, on, on, off; the shared expert's gate, up and SWIGLU
  per pass by [run/shexp_table.py](run/shexp_table.py), the step from the warm loads' server timings.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check: verdict and detail (md5s, counts, the test runs' summaries) |
| [sass_objects.tsv](sass_objects.tsv) | every object of the CUDA backend: the sm_61 code's md5 at the base and the tip, the anonymous namespaces' hashes masked |
| [sass_counts.tsv](sass_counts.tsv) | each instantiation of the new kernel: registers, stack, local memory, instruction counts, its SWIGLU against the separate kernel's |
| [swiglu.txt](swiglu.txt) | the SWIGLU's instructions in the separate GLU kernel and in each instantiation |
| [shapes.tsv](shapes.tsv) | the launch shape per type pair and token count at 2,560 → 640 |
| [replay_cases.tsv](replay_cases.tsv), [replay_runs.tsv](replay_runs.tsv) | the replay set's cases and runs |
| [run/checks.sh](run/checks.sh) | the CPU-only chain: build, SASS, record, the harness, CPU build, `test-llama-archs`, ctest |
| [run/sass_check.sh](run/sass_check.sh) | reads the SASS into sass_counts.tsv and swiglu.txt |
| [run/shapes.py](run/shapes.py) | the launch shapes |
| [run/rig_c757.sh](run/rig_c757.sh) | the rig run (R107) |
| [run/shexp_table.py](run/shexp_table.py) | the served step's table of the shared expert's gate, up and SWIGLU |

Also changed for this row: [scripts/record_checks.sh](../../../scripts/record_checks.sh) (suite `glumt`),
[bench/mmvq_pascal/verify/multitoken_bitid.cpp](../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp) (mode `glu`),
[docs/switches.md](../../switches.md).

## Commands

On an x86-64 machine with Docker, no GPU (`WORK` a work folder holding a llama.cpp clone in `src/` with the fork as
remote `fork`; `BASE_BUILD` the rig build tree of `5b329df16`; `BASE_CPU` the `bin/` of its static CPU build):

```bash
WORK=$WORK REPO=$PWD BRANCH=pascal-p106-v4-b12246-c757 BASE_BUILD=$BASE_BUILD BASE_CPU=$BASE_CPU JOBS=8 \
    bash docs/data/2026-11-11_c757_shexp_fused/run/checks.sh
python3 -I docs/data/2026-11-11_c757_shexp_fused/run/shapes.py > shapes.tsv
```

On the rig, the rig idle (the paths of the rig's copies in the environment, as the script's header lists them):

```bash
BIN=... OUT=... SET=.../glumt REPLAY=scripts/replay_checks.sh BITID=.../multitoken_bitid MODEL=... DRAFT=... \
    PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
    REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
    VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
    setsid -f nohup bash docs/data/2026-11-11_c757_shexp_fused/run/rig_c757.sh > console.txt 2>&1 < /dev/null
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c757`. The type pairs and layer counts are those of the Swift GSQ-RCO Q2_0 file's shared
experts as C731 read them. The test inputs are generated (fixed seeds), not model weights. Everything here is this
repository's licence.
