# The grid types and IQ1_M in a k-token pass: k-column and multi-token Pascal kernels (2026-11-04)

Registry row Q713. A speculative verify, or a pipelined guess, runs every matrix of Qwen3.8-Flash-Next on k = 2–8
tokens at once. For the Swift Q2_0 file, C70's k-column kernels
([2026-11-04_kcol_dense](../2026-11-04_kcol_dense/README.md),
[2026-11-04_kcol_coverage](../2026-11-04_kcol_coverage/README.md)) and R7's multi-token expert kernel ([2026-11-04_mtok_experts](../2026-11-04_mtok_experts/README.md)) decode each weight
once for all k tokens. Swift-1.5's IQ files, which FN2 scored closer to the Q8_0 reference than Q2_0, keep their
routed gate/up projections and part of their attention in the grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) and
IQ1_M ([2026-11-04_iq2xs_kernel](../2026-11-04_iq2xs_kernel/README.md) has the census), which until now decoded every
weight once per token at 2–4 tokens and went to llama.cpp's matrix-matrix kernels at 5–8. This folder adds those six
types to both kernels. Fork branch `pascal-p106-v3-b12076-iqkcol` of
[Miner Llama](https://github.com/Cinderella-Man/llama.cpp): `pascal-p106-v3-b12076-iq2xs` (`6c88b6991`) with
`pascal-p106-v3-b12076-kcol` (`af46311af`, the k-column header the bench copies) merged, then three commits.

## What a k-token pass now decodes once

Per file, the bytes one token reads (from the layouts of
[2026-11-04_iq2xs_kernel](../2026-11-04_iq2xs_kernel/README.md)) that a 2–8-token pass runs on kernels that decode each
weight once for all tokens (the k-column kernels, the multi-token expert kernels, the BF16 kernel to 8 tokens):

| Swift-1.5 file | A token reads | Before (`-iq2xs`) | With these kernels | Left on per-token decoding |
|---|---|---|---|---|
| Q2_0 | 4.16 GB | 93.0 % | 93.7 % | the F32 routers (262 MB) |
| IQ2_XS | 4.24 GB | 78.1 % | **93.8 %** | the F32 routers |
| IQ3_XXS | 4.81 GB | 76.5 % | **91.1 %** | the routers, the routed IQ4_NL down projections (166 MB: Q735) |
| IQ3_S | 5.28 GB | 76.7 % | **88.2 %** | the routers, the routed IQ4_NL down projections (359 MB: Q735) |

(The Q2_0 file gains the shared expert's Q4_0, Q5_0 and IQ4_NL down projections, from the merged `-kcol`.)

## The kernels

Both decode a 32-weight block of these types once into 8 words of signed int8 weights and the integer scales of its
two halves (`mmvqp_iq::decode32` in `ggml/src/ggml-cuda/mmvq-pascal-iq.cuh`, free of ggml's headers), then spend one
dp4a per 4 weights and token. The grid bytes are 1..127, so a negative weight is `(g ^ 0xFF) + 1` per byte with no
carry into its neighbour; the one-token kernels instead spend two dp4a and one AND per 4 weights
(`sum(g * u) - 2 * sum((g & m) * u)`), as many operations as the signed bytes at one token and more at every further
token. IQ1_M (no FP16 scale per block: it packs into the four scale words' top nibbles; values grid ± 1/8) is looked up
with `__byte_perm` from ggml's `iq1s_grid_gpu` straight into `8 (grid + delta)`, which fits an int8 (−9, −7, −1, 1, 7,
9); the lookup returns a group's weights in the order (0, 4, 1, 5), (2, 6, 3, 7), so its activations are staged in that
order.

- **Dense products** (`mmvq-pascal-kcol-impl.cuh`, `mmvq-pascal-kcol-<type>.cu`): the six types as k-column types, one
  super-block per lane unit, 2–8 columns (one column stays with the one-token Pascal kernels; IQ1_M's with llama.cpp's).
  The 66-, 74-, 82-, 98- and 110-byte blocks start at 2-byte boundaries, so the warp stages each row's segment from the
  16-byte boundary below it, as for Q3_K and Q6_K; each block copies its type's grid into shared memory once (2–8 KB,
  counted in the launch shape's budget). The grid types run one row per lane group: at two their decoding takes all 255
  registers and a few kernels spill 8–16 bytes (`GGML_CUDA_MMVQ_PASCAL_KCOL_R=2` still selects it for timing).
- **Routed experts** (`mmvq-pascal-mtok-iq.cuh`, `mmvq-pascal-mtok-<type>.cu`): MUL_MAT_ID of 2–8 tokens with R7's
  groups (every distinct expert of the step read once for all the tokens that picked it), routed + shared expert launch,
  biases and GLU epilogue; each block stages its members' activations once (256 int8 values and 16 bytes of padding per
  super-block, so that the two lanes of a K = 2,560 row read different banks) and copies the grid. Lane groups, segment
  staging (16-, 8- or 4-byte loads; 4 at K = 2,560 for the grid types, 16 for IQ1_M) and the order of every FP32
  operation are the one-token kernel's, so **a grid type's result is bit-identical to the one-token Pascal kernel's**.
  Groups of up to 8 members take up to 255 registers (their shared memory allows 2 blocks per SM either way), up to 4
  members 168.
- **Batch limit:** on compute 6.1 llama.cpp sends MUL_MAT_ID of these types to its matrix-matrix kernels above 4–6
  tokens (`get_mmvq_mmid_max_batch`). Where the multi-token kernels take a type it now returns 8, and a product they
  then decline (more than 128 (token, slot) entries, rows they can't stage) runs in chunks of tokens within llama.cpp's
  own limit, whose `mul_mat_vec_q_moe` launch bounds allow no more.
- **Switches:** `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=<types>` (dense) and `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP=<types>` (experts)
  return the types to the kernels before this branch; `GGML_CUDA_MMVQ_PASCAL_MTOK=E,CAP,RB` as for Q2_0;
  `GGML_CUDA_MMVQ_PASCAL_SKIP` and `GGML_CUDA_MMVQ_PASCAL=0` as before.

## Checks without a GPU

All on a build machine's CPU in the CUDA 12.9 container ([run/cpu_checks.sh](run/cpu_checks.sh)):

| Check | Result |
|---|---|
| Decoding against ggml's dequantization ([decoding_check.md](decoding_check.md)) | all six types: the k-column unit at 1, 3, 8 columns and 1, 2 rows (600 cases each), worst 0.0065 of the tolerance; the multi-token arithmetic (200 cases each), worst 0.0016, and for IQ2_XXS, IQ2_XS and IQ3_S bit-identical to the one-token functions in 200 of 200; ggml's `iq1s_grid_gpu` is the CPU grid plus one as nibbles, as the IQ1_M decoding reads it. The mutation run (three planted faults) fails exactly IQ2_S, IQ3_S and IQ1_M (2,500–12,000 times the tolerance) |
| Rig tarball (`scripts/build_llamacpp_docker.sh`, `TESTS=1`) | `llama.cpp-20261104-0fd6ce5c1-pascal-p106-v3-b12076-iqkcol.tar.gz`, 511,449,917 bytes, md5 `ee3db48ef95fd6e621e91539fadc053d`; compute 6.1, no AVX, glibc up to 2.38; ~150 MB larger than `-iq2xs`'s, the new kernels' code in every binary |
| Registers ([registers.tsv](registers.tsv), `cuobjdump -res-usage` of test-backend-ops) | 204 new kernels (96 k-column, 108 multi-token), 64–255 registers, **none with local memory or a stack frame** on the paths the launch rules pick; the grid types' 2-row k-column kernels (only with `GGML_CUDA_MMVQ_PASCAL_KCOL_R=2`): 4 of 40 with an 8–16-byte stack frame |
| SASS against `-iq2xs` and `-kcol` ([sass_summary.txt](sass_summary.txt)) | 7,587 kernels: all 7,326 of `-iq2xs` identical (the earlier k-column kernels among them), the 51 `-kcol` changed since `e1fa4c573` identical, 210 new, all of Q713 |
| C70's decode check of the ten earlier k-column types on the refreshed header (`verify/kcol_unit_check.sh 40`) | 10 of 10 pass, worst 0.0044 of the tolerance (unchanged) |
| `build/mmvq_kcol` with the IQ types (`build_kcol.sh IQ_TREE=...`) | builds, no AVX, glibc up to 2.38; the IQ kernels without spills but one forced-R = 2 variant (16 bytes) |

So the decoding is right, no kernel spills on its default path, and nothing the two joined branches ran changed: every
earlier kernel compiles to the same machine code.

## Not yet run: on a GPU

Nothing ran on a GPU: the quant machine's cards held the quality chain and the rigs were off. Next, in order:

1. **Correctness through the compute 6.1 PTX on any free quant-machine GPU** ([run/gpu_checks.sh](run/gpu_checks.sh)
   `MODE=ptx`, est. 1–2 h, mostly the CPU's IQ quantization of the reference weights): test-backend-ops of the six
   types at 1–8 tokens, the kernels on, at R = 2 (`kcol_r2`), with groups of one (`cap1`) and one row block per block
   (`rb1`), the kernels before this branch (`off`), llama.cpp's throughout (`skip`), then compute-sanitizer on 30 cases;
   and `CUDA_FORCE_PTX_JIT=1 mmvq_kcol --check-only --nsm 5 --types iq2_xxs,iq2_xs,iq2_s,iq3_xxs,iq3_s,iq1_m`. Every run
   must pass every case.
2. **One P106** (`MODE=native` without the dense shapes of 6,144 rows or K = 6,144, est. ~1.5 h), then the timing:
   [bench/mmvq_pascal/iq_kcol_perf.sh](../../../bench/mmvq_pascal/iq_kcol_perf.sh) (~25 min, the Q713 kernels against
   the kernels before them at 1–8 tokens) and `mmvq_kcol --types ...` (the dense kernels' variants), tables by
   [iq_kcol_perf_table.py](../../../bench/mmvq_pascal/iq_kcol_perf_table.py) with the layouts for the per-pass sums.
   Expected: a routed gate/up of 5 tokens at ≲ 1.3× its one-token time where the kernels before cost ~2–3× (decoding
   once per token at 2–4, llama.cpp's matrix-matrix kernels at 5–8), the dense IQ3_S products alike.
3. **In the model:** S21's k-token pass profile on the IQ2_XS file once it runs on the rig (Q710).

## Files

| File | What |
|---|---|
| `decoding_check.md` | the host decode check and its mutation run |
| `registers.tsv` | registers, stack, shared and local memory of every new kernel (sm_61): 96 k-column, 108 multi-token |
| `sass_summary.txt` | the per-kernel SASS identity of the new build against `-iq2xs` and `-kcol` |
| `run/cpu_checks.sh` | the checks above: decode, build, regs, sass |
| `run/gpu_checks.sh` | the GPU correctness runs (PTX on another GPU, or a P106) |

Tools: [bench/mmvq_pascal/verify/iq_kcol_check.cu](../../../bench/mmvq_pascal/verify/iq_kcol_check.cu) and `.sh` (the
decode check), [bench/mmvq_pascal/iq_kcol_perf.sh](../../../bench/mmvq_pascal/iq_kcol_perf.sh) and
[iq_kcol_perf_table.py](../../../bench/mmvq_pascal/iq_kcol_perf_table.py) (the timing inside ggml),
`bench/mmvq_pascal/build_kcol.sh` with `IQ_TREE=<llama.cpp tree>` (the standalone k-column bench with these types).
Weights in every check are random bytes or test-backend-ops' random tensors; no model file was read.

## Commands

```bash
# on a build machine with Docker (no GPU); WORK, REPO, TREE and the build folders as in run/cpu_checks.sh
WORK=$WORK REPO=$REPO TREE=$TREE REF=fork/pascal-p106-v3-b12076-iqkcol BASE_BUILD=$IQ2XS_BUILD \
    KCOL_BUILD=$KCOL_BUILD KCOL_OLD_BUILD=$KCOL_OLD_BUILD JOBS=6 bash $REPO/docs/data/2026-11-04_iq_kcol/run/cpu_checks.sh
IQ_TREE=$TREE bash $REPO/bench/mmvq_pascal/build_kcol.sh
# correctness through the PTX on a newer GPU G (inside nvidia/cuda:12.9.2-devel-ubuntu24.04 with that GPU)
MODE=ptx BIN=$BUILD/bin OUT=$OUT bash run/gpu_checks.sh
# on a P106
MODE=native BIN=$BUILD/bin OUT=$OUT GPU=0 bash run/gpu_checks.sh
BIN=$BUILD/bin OUT=$OUT GPU=0 bash bench/mmvq_pascal/iq_kcol_perf.sh
python3 -I bench/mmvq_pascal/iq_kcol_perf_table.py $OUT --layout docs/data/2026-11-04_iq2xs_kernel/layout_swift-iq2_xs.txt \
    --layout docs/data/2026-11-04_iq2xs_kernel/layout_swift-iq3_xxs.txt > iq_kcol_perf.csv
```

Licence: the tables and scripts here are this repository's; the grids the checks copy are ggml's (MIT).
