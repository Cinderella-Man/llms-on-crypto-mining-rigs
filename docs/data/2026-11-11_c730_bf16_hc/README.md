# The BF16 hyper-connections at 2–4 tokens: the kernel, its loads, its sums (C730, Q1289, 2026-11-11)

What this is: a read of the code and the sm_61 machine code, with no GPU and no new build. In a four-token pass of
Miner Llama's Flash-Next preset, Qwen3.8-Flash-Next's four BF16 hyper-connection products take 10.19 ms of GPU time
against 8.23 ms at one token ([2026-11-10_release_profile](../2026-11-10_release_profile/README.md), R41). The night's
read-once kernels for the quantized products ([C690](../2026-11-11_c690_head_four_rows/README.md),
[C722](../2026-11-11_c722_short_batch_products/README.md)) load each weight once for all tokens and repeat the
one-token kernel's sums. This note checks the same two properties for BF16: which kernel takes these products at 1–4
tokens, how often it loads each weight, and whether its sums at 2–4 tokens are the one-token kernel's.

**The answer:**
- **One kernel takes all four products at 1 to 8 tokens:** Miner Llama's Pascal BF16 kernel
  (`mul_mat_vec_bf16_pascal`, `ggml/src/ggml-cuda/mmvf-pascal*`, Q546).
- **It already loads each weight once per call, for every token.** Its loop issues 2R + 4T 128-bit loads per two
  8-weight chunks, in all 96 instantiations: one load per row and chunk however many tokens, two per token and chunk
  however many rows. It converts each weight once (16R conversions per loop pass at every T).
- **The 320-wide `hc_*_up` products are exact at 2–4 tokens:** they take the one-token lane count at every token count.
- **The 10,240-wide `hc_*_down` products are exact at 2 tokens, not at 3–4.** At 3 and 4 tokens they switch from 128
  to 32 lanes per row (Q611's faster shape), which changes the order of each row's sums. In a CPU emulation of the two
  orders, 2,333 of 2,560 outputs differ, by a median 5 units in the last place.

So C690's method has nothing to remove here: the weights are already read once. What a four-token pass adds over a
one-token pass is the per-token work: 4T activation loads and 16RT FMAs per loop pass, 3TR to 5TR shuffles at the end,
and 96 registers at R = 4, T = 4 (est. 5 blocks a SM, against 9 at one token). That excess is at most 1.96 ms of a
four-token pass (10.19 − 8.23, R41). Registry row [Q1289](../../../quants/experiments.md).

## The products and the kernel that takes them

Under the preset (`release/flash-next-v3/settings.env`: no `GGML_CUDA_MMVF_PASCAL*` variable, so the defaults: the
Pascal kernel on compute 6.1, `RULE=2`, `MAX_COLS=8`), `ggml_cuda_mul_mat` hands every BF16 product of 1 to 8 tokens
whose rows and activation columns are 16-byte aligned to `ggml_cuda_mul_mat_vec_f`. That function calls the Pascal
kernel first (`ggml_cuda_mmvf_pascal`). The hyper-connections' inputs are contiguous F32 tensors (`hc_norm` and the
SiLU of the down product, `src/models/qwen4exp.cpp`), so all four products qualify at 1, 2, 3 and 4 tokens.
`GGML_CUDA_MMVQ_PASCAL*` and the k-column and read-once kernels (C690, C722) take quantized types only.

GPU time per pass summed over the nine cards (R41's `summaries/release-k1.txt`, `release-k4.txt`, `release-k5.txt`;
48 calls each, one per layer). The launch shape is G lanes per row group and R rows per group, on a P106-090's 5 SMs
([shapes.tsv](shapes.tsv)):

| Product | Width → rows | 1 token, ms | 4 tokens | 5 tokens | G, R at 1 / 2 / 3 / 4 tokens | Exact at 2 / 3 / 4 tokens |
|---|---|---:|---:|---:|---|---|
| hc_attn_down | 10,240 → 320 | 2.26 | 2.69 | 4.11 | 128, 4 / 128, 4 / **32, 4** / **32, 4** | yes / **no** / **no** |
| hc_ffn_down | 10,240 → 320 | 2.07 | 2.44 | 3.84 | 128, 4 / 128, 4 / **32, 4** / **32, 4** | yes / **no** / **no** |
| hc_attn_up | 320 → 10,240 | 1.98 | 2.56 | 3.55 | 8, 4 / 8, 4 / 8, 4 / 8, 4 | yes / yes / yes |
| hc_ffn_up | 320 → 10,240 | 1.92 | 2.50 | 3.47 | 8, 4 / 8, 4 / 8, 4 / 8, 4 | yes / yes / yes |
| **the four** | | **8.23** | **10.19** | **14.97** | | |

- **The output's `hc_head_down`** has `hc_*_down`'s shape (one call a pass), so it changes order at 3–4 tokens too.
- **The kernel's other Flash-Next shapes** keep their one-token lanes at 2–4 tokens on 5 SMs: the 10,240 → 4
  injections, the indexer's 2,560 → 512 and the recurrent layers' 2,560 → 48 alpha and beta. Only R changes, for alpha
  and beta (1 → 2), and R does not change a row's sums.
- **On a card with 10 SMs** the rule changes differently: `hc_*_down` keeps 128 lanes at 3–4 tokens, and the indexer
  moves from 32 to 128 ([shapes.tsv](shapes.tsv) lists both).

## Loads per token, from the SASS

[sass_counts.tsv](sass_counts.tsv) is cuobjdump's sm_61 code of the build's `mmvf-pascal.cu.o` (fork
`pascal-p106-v4-b12246-c748` `5b329df16`), read by [run/sass_check.sh](run/sass_check.sh). The benches below timed a
verbatim copy of this kernel (`bench/mmvq_pascal/src/bf16_mv.cuh`, Q611's launch rule): since then the kernel has changed
only by `9e1a8492e`'s launch-bound macros, whose defaults leave its code as it was, and `b4d6d427b` only added switches
that default to the rule (`00d662c3a`, Q611). The main loop is unrolled twice in the source, so one pass of it is two
8-weight chunks of each of a lane's R rows. Flash-Next's shapes:

| G, R | Tokens | Registers | Blocks a SM, est. (registers) | Loop LDG.128 | of which weights | of which activations | Loop FFMA | Loop conversions | SHFL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8, 4 (`hc_*_up`) | 1 | 54 | 9 | 12 | 8 | 4 | 64 | 64 | 12 |
| 8, 4 | 2 | 64 | 8 | 16 | 8 | 8 | 128 | 64 | 24 |
| 8, 4 | 3 | 80 | 6 | 20 | 8 | 12 | 192 | 64 | 36 |
| 8, 4 | 4 | 96 | 5 | 24 | 8 | 16 | 256 | 64 | 48 |
| 128, 4 (`hc_*_down`, 1–2 tokens) | 1 | 48 | 10 | 12 | 8 | 4 | 64 | 64 | 20 |
| 128, 4 | 2 | 64 | 8 | 16 | 8 | 8 | 128 | 64 | 40 |
| 32, 4 (`hc_*_down`, 3–4 tokens) | 3 | 80 | 6 | 20 | 8 | 12 | 192 | 64 | 60 |
| 32, 4 | 4 | 96 | 5 | 24 | 8 | 16 | 256 | 64 | 80 |

- **Every one of the 96 instantiations** (G = 8, 32, 128; R = 1, 2, 4, 8; T = 1 to 8) has exactly 2R + 4T loads and
  16RT FMAs in its loop. The split between weights and activations is that count's: 2R loads that do not grow with T
  (one 16-byte chunk of 8 weights per row and chunk) and 4T that do not grow with R (each chunk's 8 FP32 activations,
  two 16-byte loads, shared by the group's R rows). In the code the weight pointers step by 2·G·16 bytes a pass and the
  activation pointer by 4·G·16.
- **Each weight is converted once:** 8R shifts and 8R masks a pass (a BF16 value is the top half of an FP32 one) at
  every T, 16R in every R = 4 instantiation. At R = 1 and 2 a few more shifts and masks serve addresses (at most 10),
  and none grows with T.
- **No stack and no local memory** in any instantiation.
- Blocks a SM are estimated from the registers (65,536 a SM, 128 threads a block, allocated per warp in units of 256),
  under the 32-block and 64-warp limits. `hc_*_down` at 3–4 tokens has 20 blocks in all, 4 a SM.

## The sums at 2–4 tokens

- **A row's sum depends on G only.** Lane `sl` of a group adds its chunks `sl`, `sl + G`, … one FMA per weight in
  column order. The warp's lanes are added with a butterfly (`__shfl_xor_sync` with offsets 16, 8, 4, 2, 1, or 4, 2, 1
  at G = 8), which leaves the same bits in every lane because each step adds the same two values. At G = 128 warp 0
  then adds warps 1, 2 and 3 in that order.
- **Neither R nor T changes any of this:** each (token, row) has its own accumulator, and the unrolling does not
  reorder a chain. The same G gives the same bits; `hc_*_up` and the 2-token `hc_*_down` are the one-token kernel's
  sums bit for bit, by construction.
- **At 3 and 4 tokens `hc_*_down` runs at G = 32:** each lane adds 40 chunks instead of 10, and there is no cross-warp
  step. That is another order.
- **The emulation** ([run/order_emul.c](run/order_emul.c), [order_emul.tsv](order_emul.tsv)) sums 320 rows of 10,240
  BF16 weights (normal, standard deviation 0.02) against 8 activation columns (standard normal) in both orders, with
  `fmaf` and denormals flushed as the kernel's FTZ instructions do:

| Shape | Orders compared | Outputs | Differ | Median difference of those, ULP | Largest difference / sum of \|terms\| |
|---|---|---:|---:|---:|---:|
| 10,240 → 320 (`hc_*_down`) | G = 128 (1–2 tokens) against G = 32 (3–4 tokens) | 2,560 | 2,333 | 5 | 2.59e-08 |
| 320 → 10,240 (`hc_*_up`) | G = 8 against G = 8 (the control) | 20,480 | 0 | 0 | 0 |

So a four-token verify pass on the rig gives `hc_attn_down`, `hc_ffn_down` and `hc_head_down` other sums than one-token
decoding, even with C690's and C722's switches on. Q1179's test of the draft line against the plain line needs these
products exact as well. Two ways:

- **The existing switch `GGML_CUDA_MMVF_PASCAL_RULE=1`** drops Q611's 3–4-token rule. Every BF16 product at 2–4
  tokens then takes its one-token shape: on 5 SMs only `hc_*_down` changes, to G = 128, R = 4 at 3–4 tokens. In the
  two benches below that shape costs, per call:
  - 51.032 and 51.824 µs at 3 tokens, against 43.016 and 43.440;
  - 63.384 and 62.432 µs at 4 tokens, against 46.528 and 44.728.

  That is est. +1.64 and +1.72 ms of a four-token pass (97 calls), +0.78 and +0.81 at three tokens: fine for an
  exactness test, a loss for speed.
- **A kernel in G = 128's order at G = 32's shape:** each lane of a 32-lane group keeps four accumulators, for the
  virtual lanes `j`, `j + 32`, `j + 64` and `j + 96`. Then come four butterflies and the virtual warps added 0 + 1 + 2 + 3.
  That repeats the one-token sums bit for bit with G = 32's loads (C750, Q1325).

## Where the four-token time goes

The kernel at Flash-Next's shapes, µs per call. Two P106-090 benches of this kernel (cold weights, `bench/mmvq_pascal
--bf16` and `lossless_bench`, 2026-11-04), each at the fork's shape:

- [2026-11-04_kcurve_p106/bf16.csv](../2026-11-04_kcurve_p106/bf16.csv), variant `pascal`;
- [2026-11-04_lossless_bf16/rig/lossless.csv](../2026-11-04_lossless_bf16/rig/lossless.csv), `plain/pascal`.

The read floor is that file's `read/plain` (the same 6,553,600 bytes read and nothing computed):

| Shape | | 1 token | 2 | 3 | 4 | 5 |
|---|---|---:|---:|---:|---:|---:|
| `hc_*_up` (G = 8, R = 4; R = 8 at 5) | kcurve | 38.928 | 39.984 | 44.936 | 49.256 | 62.200 |
| | lossless | 38.816 | 40.016 | 44.768 | 49.024 | 61.744 |
| | read floor | 38.032 | 37.904 | 37.888 | 38.912 | 37.920 |
| `hc_*_down` (G = 128, R = 4; G = 32 at 3–4; R = 8 at 5) | kcurve | 41.880 | 41.784 | 43.016 | 46.528 | 65.736 |
| | lossless | 41.080 | 41.104 | 43.440 | 44.728 | 65.864 |
| | read floor | 37.888 | 38.848 | 37.872 | 37.920 | 38.904 |

- **At one token both shapes run near the read floor:** in the lossless run 38.816 µs against 38.032 for `hc_*_up`,
  41.080 against 37.888 for `hc_*_down`. Going to four tokens adds 10.328 and 10.208 µs a call to `hc_*_up`
  and 4.648 and 3.648 µs to `hc_*_down` (the two benches). In R41's served pass the four products
  add 1.96 ms in all.
- **Why `hc_*_up` grows most, est.:** its lanes do only 5 chunks a row (K = 320, 8 lanes), so each block is short and
  the loads of its few loop passes are exposed. At four tokens its 160 blocks run in est. 6.4 waves of 5 a SM, against
  est. 3.6 waves of 9 at one token. The epilogue's shuffles grow with T × R (48 a warp at four tokens, against 12).
- **The ceiling** for any BF16 four-token kernel is the one-token time: at most 1.96 ms of a four-token pass (2.3 % of
  its 85.36 ms of GPU time). C722's estimate for the quantized projections was up to 8.6 ms.
- **Five tokens** cost the four products 14.97 ms in R41's pass, 4.78 ms more than four. The rule's R = 8 from five
  tokens is slower in both benches than shapes the kernel already has:
  - `hc_*_up` at G = 8, R = 4: 57.048 and 56.488 µs, against 62.200 and 61.744 (the same lanes, so the same sums);
  - `hc_*_down` at G = 32, R = 4: 52.072 and 51.904 µs, against 65.736 and 65.864 (another order).

  This is Q650's question (C110): its sweep is these files.

## What it decides

- **C690's method for BF16 is not needed for the weights:** they are already read once per call. No new kernel or
  fork branch was made.
- **Exactness at 3–4 tokens** (Q1179) has one gap among the BF16 products: `hc_*_down` and `hc_head_down`.
  `GGML_CUDA_MMVF_PASCAL_RULE=1` closes it now, at est. +1.64 and +1.72 ms a four-token pass. C750 (Q1325) would close
  it at G = 32's speed.
- **The four-token excess** (at most 1.96 ms a pass) is per-token work, not the weights' read: C751 (Q1326) lists what
  would cut it without changing a sum. That means a reduce-scatter epilogue (the same additions in the same tree,
  14 shuffles a warp instead of 48 for `hc_*_up` at four tokens) and fewer registers at 3–4 tokens
  (`MMVF_PASCAL_MIN_BLOCKS`, a build define).
- **Five tokens:** Q650 (C110) gets this read's numbers. R = 4 for `hc_*_up` keeps the sums. G = 32 for `hc_*_down`
  does not, and that is the owner's call.

## Files

| File | What |
|---|---|
| [shapes.tsv](shapes.tsv) | the launch shape the kernel's rule picks for Flash-Next's five BF16 shapes, 1–8 tokens, on 5 and 10 SMs, and whether G is the one-token G |
| [sass_counts.tsv](sass_counts.tsv) | every `mul_mat_vec_bf16_pascal<G, R, T, false>`: registers, stack, local memory, the main loop's LDG, FFMA, shifts, masks and other instructions beside 2R + 4T and 16RT, SHFL and STG |
| [order_emul.tsv](order_emul.tsv) | the CPU emulation of the sums in two lane orders |
| [run/sass_check.sh](run/sass_check.sh) | reads cuobjdump's output into `sass_counts.tsv` |
| [run/shapes.py](run/shapes.py) | a copy of `mmvfp_pick_shape` that prints `shapes.tsv` |
| [run/order_emul.c](run/order_emul.c) | the emulation |

## Commands

On any x86-64 machine with CUDA 12.9's `cuobjdump` (no GPU), against a build tree of the fork at `5b329df16`
(`scripts/build_llamacpp_docker.sh`; `BUILD` is its build directory):

```bash
O=$BUILD/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir/mmvf-pascal.cu.o
mkdir -p sass
cuobjdump -res-usage "$O" > sass/mmvf_res.txt
cuobjdump -sass "$O" > sass/mmvf.sass          # 108 functions, sm_61
SASS=$PWD/sass bash run/sass_check.sh > sass_counts.tsv
python3 run/shapes.py 5 10 > shapes.tsv
gcc -O2 -ffp-contract=off -o order_emul run/order_emul.c -lm && ./order_emul > order_emul.tsv
```

The object's md5 was `67bcc49babc65701d070cf27478be04d` (the `TESTS=1` rig build of `5b329df16`). The timings are the
two earlier benches' files, linked above; R41's are its summaries.

## Provenance and licence

Code read: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c748` at `5b329df16`. The emulation's inputs are generated (a fixed seed), not model weights.
Everything here is this repository's licence.
