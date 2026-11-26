# IQ1_M on the batched prompt kernel, checked on a V100 (C1179, Q1914; 2026-11-26)

What this is: the code half of registry row Q1914 in [quants/experiments.md](../../../quants/experiments.md), and its
correctness check. Qwen3.8-Flash-Next's better-quality file `gmix-q2` reads prompts about a third slower than
`pmix-q2-fast` (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF), and [C1139](../2026-11-26_c1139_gmix_prompt_rig1/README.md)
found why: IQ1_M, the weight type of two of its layers' routed experts, was the one type of the file that MMQ (the fork's
batched integer kernel for a quantized matrix times a batch of tokens) did not take. So those products took a generic
path that copies the expert ids to the host, waits for the card twice per product and runs one small product per
expert, and that wait also turns CUDA graphs off for the whole card. With one host thread driving nine cards, the cards
stopped overlapping. This change puts IQ1_M on MMQ behind a switch, `GGML_CUDA_MMQ_IQ1_M`, off by default, on the fork
branch `pascal-p106-v5-iq1m-mmq` (`17ff0948a`, one commit on `pascal-p106-v5` at `2d839da1d`). It was built and checked
on one Tesla V100 32 GB: before Turing, MMQ's integer path is DP4A, the same instructions a P106 runs, and the P106's own
compute-6.1 code was run there too. Nothing here is timed. What it saves on a P106 rig is the rig check's (below).

## Result in plain words

- **The new kernel gives the right answers.** With the switch on, every IQ1_M matrix-product test passes against the
  CPU: 357 of 357 cases, plain products and expert products from 9 to 512 tokens (Flash-Next's 640 × 2,560 expert
  shape with 32 and 512 experts among them), both as the V100's own machine code and as the P106's compute-6.1 code
  compiled by the driver.
- **The switch takes the products off the waiting path.** In a 512-token expert product, the switch off runs a
  dequantization and a cuBLAS product for each of the 32 experts (132 kernel launches) and adds two waits for the card;
  on, one MMQ kernel covers every expert (5 launches, no wait of its own). `1`, the rigs' value, acts only on compute
  6.1 cards, so on the V100 it left the old path, as intended; the checks used `2`.
- **With the switch off, nothing else changes.** Every type's plain-product tests pass on both builds (5,344 of 5,344),
  and the branch's card code is the base's, function by function, apart from a source-line number in stubs that never
  run. The one exception is noise: the compiler gives the Q4_0 and Q4_1 kernels slightly different code on every
  compile, even of the same source.
- **Next is the rig.** Whether `gmix-q2`'s prompts close the gap (est. 95 % of it, +3.1 % a task on the agent check
  instead of +8.7 %) is rig 1 row 6e's (R242) check, on a build with this switch and the IQ fast kernels `gmix-q2`
  already runs ([what it needs](#what-the-rig-check-needs)).

## The change

- **The idea.** Each 8 weights of an IQ1_M block are d·(2s+1)·(g + δ): g ∈ {−1, 0, 1} from the 2,048-entry grid, δ =
  ±1/8 with its sign in a bit per 8 weights, s the 3-bit scale of 16 weights and d the block's FP16 scale. That is
  exactly (d·(2s+1)/8) · (8g ± 1): an int8 in {−9, −7, −1, 1, 7, 9} and a scale per 16 weights. MMQ already has that
  tile kind (int8 values, a float scale per 16 weights): IQ2_XS and IQ2_S use it, read by the "Q8_0_16" dot products
  (DP4A on Pascal and Volta, MMA on Turing and newer). So IQ1_M needs only a tile loader; the dot product, write-back and
  launch code are shared.
- **The loader** (`ggml_cuda_mmq_load_tiles_iq1_m` in `ggml/src/ggml-cuda/mmq-load-tiles.cuh`). Eight threads a row,
  each 32 weights: one 32-bit read of 4 grid indices, the 16-bit word of their high bits and delta signs, a grid lookup
  per 8 weights. The grid word holds g + 1 in each nibble, so the 4 int8 values of one word are
  `((grid << 3) & 0x78787878) + (δ < 0 ? 0x77777777 : 0x79797979)`, XOR `0x80808080` (per byte 8(g+1) + 121 or + 119 is
  at most 137, so no byte carries into the next, and the XOR takes 128 off each byte), and `grid >> 1` for the other 4.
  The two scales are d/8 times 2s+1.
- **Around it.** IQ1_M rows (IQ2_XS's tile shapes) in the Pascal DP4A, Pascal-older and Ampere tile tables (the AMD
  tables leave it out, so a HIP build compiles no kernel for it); a kernel instance file
  `template-instances/mmq-instance-iq1_m.cu` (and `generate_cu_files.py`); IQ1_M's activations in Q8_1 with one scale
  per 32 values (`mmq_get_q8_1_ds_layout`: D4); IQ1_M in the type switch of `mmq.cu`.
- **The switch.** `ggml_cuda_should_use_mmq` takes IQ1_M only when `GGML_CUDA_MMQ_IQ1_M` says so: `0` or unset off,
  `1` on compute 6.1 cards, `2` on any NVIDIA card (for tests; Turing and newer take the MMA tile, which no check here
  ran). Every caller asks that function, so `ggml_cuda_mul_mat_id_needs_sync` follows it and the card keeps its CUDA
  graphs. The first read logs `GGML_CUDA_MMQ_IQ1_M=<n>: IQ1_M prompt products on MMQ (...)` at INFO. IQ1_M has no
  fused gate/up loader, so a gate/up pair runs as two MMQ products.
- **Test cases.** `test-backend-ops` gets IQ1_M cases at 9, 16, 17, 32, 48, 63, 64, 65, 128, 129, 256 and 512 columns:
  `MUL_MAT` at 64, 128 and 1,000 rows of 2,560 (a multiple of 128 and not: the row check) and a broadcast batch;
  `MUL_MAT_ID` at Flash-Next's routed gate/up (2,560 → 640, 10 of 32 experts, with and without a shared input), 1,000
  rows of 1,024, and 10 of 512 experts at 9, 64 and 512 tokens; the gate/up pair with SWIGLU (`MUL_MAT_VEC_FUSION`).

## The checks

One V100 32 GB (driver 580, CUDA 12.9.2, Ubuntu 24.04), native builds (no Docker) with the rig tarball's CPU flags, so
the CPU reference is the rig build's code ([run/build.sh](run/build.sh)):

- **Build A**, `CMAKE_CUDA_ARCHITECTURES="61-virtual;70"`: the V100 runs its own sm_70 code, on the tile table of newer
  cards. On a V100, plain products of 64 or more tokens go to cuBLAS (it has FP16 tensor cores); expert products and
  smaller ones take MMQ.
- **Build B**, `"61-virtual"` with `GGML_CUDA_FORCE_MMQ`: the build holds only the P106's compute-6.1 PTX, so the driver
  JIT-compiles the P106's code, and MMQ takes every product size, as on a P106. Both the host and the card then pick
  the DP4A tile table a P106 uses. (Forcing the JIT on build A with `CUDA_FORCE_PTX_JIT=1` would mix the two tables, and
  the variable also pushes cuBLAS's own kernels through the JIT: two build-B runs started with it aborted when cuBLAS
  started, before any product of theirs failed; they are in [checks.tsv](checks.tsv). Build B needs no such variable.)

| Check | Result |
|---|---|
| Builds | base commit first, then the branch into the same folder; no warning in the 44 objects the change rebuilt; its changed objects compiled alone first, in 36 s |
| A, switch unset, every type: `MUL_MAT` | 5,344 of 5,344 OK (683 not supported), 0 failed: 5,211 in a run of all three ops and the 133 it had not reached when it was stopped |
| A, switch unset, every type: `MUL_MAT_ID`, `MUL_MAT_VEC_FUSION` | 934 and 410 OK of the cases reached, 0 failed; stopped after 51 minutes in other types' `MUL_MAT_ID` cases, which the CPU reference without AVX runs slowly and which this change does not reach |
| A, switch unset, IQ1_M | `MUL_MAT` 161, `MUL_MAT_ID` 126, `MUL_MAT_VEC_FUSION` 70: all OK |
| A, `GGML_CUDA_MMQ_IQ1_M=2`, IQ1_M | 357 of 357 OK (161, 126, 70) |
| B, switch unset, every type: `MUL_MAT` | 5,344 of 5,344 OK (683 not supported), 0 failed |
| B, switch unset, IQ1_M: `MUL_MAT_ID`, `MUL_MAT_VEC_FUSION` | 196 of 196 OK |
| B, `GGML_CUDA_MMQ_IQ1_M=2`, IQ1_M | 357 of 357 OK (161, 126, 70) |
| Kernels of one 512-token expert product | unset and `=1`: 32 dequantizations and cuBLAS products, 132 launches, 2 waits; `=2`: one `mul_mat_q` for IQ1_M, 5 launches ([traces.md](traces.md)) |
| Card code, branch against base | every function the same but for one constant in the stubs and the compiler's noise in Q4_0 and Q4_1 ([below](#the-card-code)) |

Every run is a row of [checks.tsv](checks.tsv).

## The card code

Build A's objects of the base commit and of the branch, built one after the other into the same folder, compared
function by function in both forms build A holds: the V100's sm_70 machine code (`cuobjdump -sass`) and the P106's
compute-6.1 PTX (`cuobjdump -ptx`), which is what a rig build compiles to sm_61 ([run/code_check.sh](run/code_check.sh),
[run/code_compare.py](run/code_compare.py)). Of 10,210 functions in both builds:

| | machine code (sm_70) | PTX (compute 6.1) |
|---|---|---|
| the same | 8,850 | 8,839 |
| differ only in one constant, 1124 → 1138 | 1,344 | 1,344 |
| differ otherwise | 16 | 27 |
| new (IQ1_M's MMQ kernels) | 96 | 96 |

- **The constant** is the source line that `NO_DEVICE_CODE` prints: those 1,344 functions are MMQ's stubs for tile
  shapes the tables do not have, which never run, and the line moved from 1124 to 1138 because 14 lines were added
  above it in `mmq.cuh` ([code_constants.tsv](code_constants.tsv)).
- **The other differences** are all in the Q4_0 and Q4_1 MMQ objects and their 64-row copies
  ([code_changed.tsv](code_changed.tsv)), and they are the compiler's: the branch's own Q4_0 and Q4_1 objects compiled
  twice from the same source with the build's command differ in 2 and 8 machine-code functions and 10 and 14 PTX
  functions, while Q8_0 and IQ2_XS come out the same both times ([run/repro_check.sh](run/repro_check.sh),
  [repro_compare.tsv](repro_compare.tsv), [repro_changed.tsv](repro_changed.tsv)). So a byte-for-byte comparison of
  builds cannot show Q4_0 and Q4_1 unchanged; the tests above do.
- Every object's counts are in [code_compare.tsv](code_compare.tsv).

## What the rig check needs

- **The build.** `gmix-q2`'s arms run C1138's IQ fast kernels (`GGML_CUDA_MTOK_IQ_FAST=2`, fork branch
  `pascal-p106-v5-iq-mtok-fast`, not landed), so the rig needs both changes in one tree: fork branch
  `pascal-p106-v5-iq1m-mmq-mtok` (`b6e99bd11`), a merge of the two with no file in common. Its rig tarball comes from
  `scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-iq1m-mmq-mtok` with `TESTS=1`; that tree was not built here.
- **The switch.** `GGML_CUDA_MMQ_IQ1_M=1` on `gmix-q2`'s arms beside `GGML_CUDA_MTOK_IQ_FAST=2` (on `pmix-q2-fast`,
  which has no IQ1_M tensor, it changes nothing).
- **What to read.** C1139's lanes: cards 1 and 2's host time inside a prompt's evaluations back at `pmix-q2-fast`'s,
  and the 8,128- and 1,028-token gaps; and `gmix-q2`'s replies with the switch on against off. MMQ reads the
  activations as 8-bit integers (Q8_1) where the old path read them as floats, so a reply may move at a near-tie.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every `test-backend-ops` run: build, switch, ops, cases OK, not supported, failed, seconds |
| [traces.md](traces.md) | the kernels and synchronizations of one IQ1_M expert product, switch unset, `1` and `2` |
| [code_compare.tsv](code_compare.tsv) | base against branch, per object and form: functions in both, the same, constant only, otherwise, new |
| [code_constants.tsv](code_constants.tsv) | each differing constant pair and how many functions have it |
| [code_changed.tsv](code_changed.tsv) | the functions that differ otherwise (all Q4_0 and Q4_1) |
| [repro_compare.tsv](repro_compare.tsv), [repro_changed.tsv](repro_changed.tsv) | the same source compiled twice: Q4_0, Q4_1, Q8_0, IQ2_XS |
| [run/build.sh](run/build.sh) | the native build (CUDA architectures, MMQ forced or not) |
| [run/all.sh](run/all.sh) | the sequence: the changed objects alone, build A of the base and of the branch, build B, the checks |
| [run/checks.sh](run/checks.sh) | the `test-backend-ops` runs and the `nsys` kernel traces |
| [run/code_digest.sh](run/code_digest.sh) | an md5 of each object's machine code and PTX (a quick per-object view) |
| [run/code_check.sh](run/code_check.sh), [run/code_compare.py](run/code_compare.py) | the function-by-function comparison of base and branch |
| [run/repro_check.sh](run/repro_check.sh) | the same objects compiled twice |

## Commands

On the GPU box, as an ordinary user, in a work folder `$WORK` holding a clone of the fork with the branch and its
parent (`git clone --depth 2 --branch pascal-p106-v5-iq1m-mmq https://github.com/Cinderella-Man/miner-llama $WORK/src`)
and the `run/` scripts in `$WORK/run`:

```bash
WORK=$WORK CUDA=/usr/local/cuda-12.9 NSYS=/usr/local/cuda-12.9/bin/nsys BASE=2d839da1d8b7ae1dc0098acb8d4e0897d06757e7 \
  BRANCH=17ff0948a8888c77f48622051d79a4abee441a83 timeout 25200 bash $WORK/run/all.sh
```

That ran 16:13 to 17:30; its step 4 was stopped (its build-B runs carried `CUDA_FORCE_PTX_JIT=1`), the all-types run
of step 3 had been stopped at 17:25, and the rest ran with the corrected `checks.sh`:

```bash
WORK=$WORK CUDA=/usr/local/cuda-12.9 BASE=2d839da1d8b7ae1dc0098acb8d4e0897d06757e7 \
  BRANCH=17ff0948a8888c77f48622051d79a4abee441a83 bash $WORK/run/code_check.sh
WORK=$WORK CUDA=/usr/local/cuda-12.9 TYPES="q4_0 q4_1 q8_0 iq2_xs" bash $WORK/run/repro_check.sh
CUDA_CACHE_PATH=$WORK/nvcache CUDA_CACHE_MAXSIZE=4294967296 TMPDIR=$WORK/tmp NSYS=/usr/local/cuda-12.9/bin/nsys \
  BIN_A=$WORK/build-a/bin BIN_B=$WORK/build-b/bin OUT=$WORK/logs/checks STEPS="a_rest b_all b_on trace" \
  bash $WORK/run/checks.sh
```

The build times: build A of the base 14 minutes, the branch into it 7, build B 13 (8 jobs at `nice -n 19`).

## Provenance and licence

Generated here from Miner Llama's sources (MIT, as llama.cpp) and `test-backend-ops`'s random inputs; no model file
was read. The raw logs, the `nsys` reports and the full code dumps are not published; the tables summarize them.
