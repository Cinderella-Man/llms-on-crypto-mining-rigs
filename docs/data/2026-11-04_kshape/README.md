# C267–C269 / Q823–Q825, Q916: the k-column launch shapes retuned, a dispatch table, the routed down projection on the multi-token kernel, FP32 activation sums in llama.cpp's kernel (2026-11-04)

[Q591](../2026-11-04_kcurve_p106/README.md) timed whole-model k-token passes of Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 on
nine P106-090s and found three things to fix in the fork's multi-token kernels; [C266's profile](../2026-11-04_kpass_profile/README.md)
put the routed experts (+46.8 ms from one token to five), the attention and recurrent projections (+34.6) and the hops
(+28.9) at the top of the growth. This folder holds the three code changes, the estimates they were chosen from, a fourth change
the checks called for (Q916) and the correctness checks. Fork branch `pascal-p106-v3-b12076-kshape`, three commits on
`pascal-p106-v3-b12076-ladder3` `7bc1552f4`: `5174e7ce0` (Q825), `4ba3f424d` (Q823, Q824) and `39c3ab5f1` (Q916). The rig's timing of the result is its own task (C350).

**Status:** built for the rig and correct through the compute 6.1 PTX on two other cards in every switch setting (below);
on a P106 (C350, 2026-11-04) correct natively at the defaults and **a 5-token pass 25 ms faster than `-ladder3`'s with the
same 32 KB budget** (180.2 against 205.3 ms; [On the rig](#on-the-rig-c350)).

## The changes

**Q825 (C269): the routed down projection on the multi-token Q2_0 kernel.** Flash-Next's experts are 640 wide, so the
down projection's rows are 180 bytes (ten 18-byte Q2_0 blocks). The multi-token kernel's warp-tile check asked for
rows that are a whole number of 16-byte units, which 180-byte rows never are, so `ggml_cuda_mmvq_pascal_mtok` declined
every down projection of 2–8 tokens: they ran per token on the earlier kernels (Q591's `perf_experts.tsv`: off = on at
every k). The check is now the one-token kernel's: each warp's 32 rows (5,760 bytes) start at a 16-byte boundary and
the matrix ends at one. Where tiles still can't be used (rows that are views of wider rows, a matrix that doesn't end
at a 16-byte boundary), the product loads its weights directly instead of being declined. On the P106 bench (Q591's
`q2_0_mtok.csv`, the same kernel) the tile-staged multi-token down projection costs 39.5 µs per token at five tokens
against 57.6 for the per-token kernel, under llama-bench-like random routing; est. −10–13 ms of a 5-token pass.
`GGML_CUDA_MMVQ_PASCAL_MTOK` gains a fifth field, T: 1 (default) tiles where they fit, else direct loads; 0 direct loads
always.

**Q823 (C267): the k-column launch shapes.** Q591's in-model arms: a 32 KB shared-memory budget
(`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`) made every pass 7.5–21.8 ms faster than the defaults (74.9 / 107.7 / 138.5 /
162.8 / 205.3 / 234.7 / 261.7 / 297.3 ms at k = 1–8, against 82.4 … 319.1), one row per lane group less so, and both
together measured as the budget alone. The budget works by fitting a third block on each SM (96 KB of shared memory
per SM on compute 6.1). The bench (Q591's `kcol_dense.csv`, `kcol_iq.csv`) shows where it doesn't: at most token
counts the grid types (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S) take 171–221 registers at one row per lane group (IQ3_S
from 2 columns, IQ2_S from 4, IQ2_XXS from 5, IQ2_XS from 6), which allows only 2 blocks per SM whatever the shared
memory, so there the smaller tile only costs (5–19 % at 3–8 columns), while at 168 registers or fewer it gains (IQ2_XS
at 4–5 columns 21–23 %); grids of at most 2 blocks per SM (the shared expert's 640 rows) gain nothing either. `kcol::pick_shape`'s defaults are
now:

- R = 2 rows per lane group only for 2 or more columns, where that leaves at least 8 blocks per SM, its registers allow
  3 blocks per SM and its shape fits a third of the SM; else R = 1; the grid types and IQ1_M always R = 1;
- the largest activation tile within a third of the SM's shared memory (32 KB on a P106) where the kernel's registers
  allow 3 blocks per SM and the grid has more than 2 blocks per SM, else within 48 KB. The registers are the compiled
  kernel's own (`cudaOccupancyMaxActiveBlocksPerMultiprocessor`, asked once per kernel), so the rule follows the
  compiler;
- where no tile fits, the shape before Q823, marked as a fallback (Q824 sends those products to the other kernels).

Any of `GGML_CUDA_MMVQ_PASCAL_KCOL_R`, `_ITT` and `_SMEM` selects the shape as before Q823 with those values
(`_SMEM=49152` is the old default); `_KCOL_SMSM` sets the shared memory per SM the shape is chosen for (with `_KCOL_NSM`,
a P106's shapes on another card).

Replayed on the bench's measured variants ([shapes_est.tsv](shapes_est.tsv)), the new defaults equal or beat the 32 KB
budget at every k, ms per token of the bench's shapes (its calls per token of the Swift Q2_0 census, the hcq8 retype's
Q8_0 hyper-connection rows included):

| k | before Q823 | `KCOL_SMEM=32768` | Q823 defaults | best variant per shape |
|---:|---:|---:|---:|---:|
| 1 | 69.99 | 47.74 | 46.32 | 46.29 |
| 2 | 76.94 | 54.60 | 54.37 | 53.67 |
| 3 | 82.57 | 63.42 | 63.08 | 63.06 |
| 4 | 88.30 | 70.24 | 69.92 | 69.74 |
| 5 | 97.17 | 78.82 | 78.27 | 78.19 |
| 6 | 103.96 | 85.80 | 85.48 | 85.19 |
| 8 | 127.41 | 110.18 | 108.52 | 104.27 |

(The bench has no k = 7 for the dense types.) The gains over the 32 KB budget come from the shared expert's Q4_0,
Q5_0 and IQ4_NL down projections at one row per lane group (7–42 % faster, 4 blocks per SM at R = 2 before) and the
grid types' full tiles. One shape stays slower than the budget's best: the Q5_K output layer at 8 columns, whose 32 KB
tile is one iteration (9 % slower than 48 KB's four; Q913). Three IQ1_M shapes at 6–8 columns have no measured variant
with the new shape (`?` in the table).
Two shapes measure faster at R = 2 than the rule gives: Q2_0's 640-row shared expert gate and up at 6–8 columns (26 %:
R = 1's 20 blocks need two waves on 5 SMs, R = 2's 10 one) and the retype's hc_down at 8 columns (15 %); the R rule leaves
them at R = 1 (Q914).

**Q824 (C268): a dispatch table.** Inside ggml (Q591's `perf_dense.tsv`, `perf_cov.tsv`, `perf_iq.md`:
test-backend-ops perf, the quantizer and the product as the model calls them, the k-column kernels off and on) the
k-column kernels lose at some types, row widths and token counts. [ggml_est.tsv](ggml_est.tsv) scales each k-column
time by the bench's gain from Q823's shape at the same shape (`us_on × new / old default`) and compares it with the
other kernels' time; where the estimate is slower, the table sends the product to the other kernels
(`mul_mat_vec_q_pascal_mc` at 2–4 tokens, llama.cpp's `mul_mat_vec_q` at 5–8), in mode 1 only:

| Type | Row width K | Columns to the other kernels | Evidence (estimated k-column ÷ other, > 1 = k-column slower) |
|---|---|---|---|
| IQ4_XS | > 4096 (ssm_out 6144 → 2560) | 2, 5, 6, 7, 8 | 1.04, 1.13, 1.33, —, 1.18 (3, 4: 0.99, 0.97 stay) |
| Q4_K | > 4096 (attn_output, ssm_out) | 2, 3, 4, 6, 7 | 1.32, 1.18, 1.02, 1.10, — (5, 8: 0.94, 0.98 stay) |
| Q8_0 | > 4096 (the retype's hc_down 10240 → 320) | 2–8 | 1.02–1.92 |
| IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S | > 4096 | 2–8 | 1.1–1.6 (Q713's perf, unchanged shapes) |
| Q5_K | ≤ 4096 (the output layer) | 2, 3 | 1.29, 1.08 |
| Q6_K | ≤ 4096 (the draft's output layer) | 2–8 | 1.08–1.81 (fallback shapes) |
| Q4_0, Q5_0 | ≤ 4096 (the shared expert's down, 640) | 2, 3, 4 | 1.34–1.51, 1.03–1.23 |
| Q2_0 | ≤ 4096 | 2 | 1.08 |
| IQ2_XXS / IQ3_S / IQ2_S / IQ3_XXS | ≤ 4096 | 2 / 2, 3 / 3 / 3 | 1.12 / 1.2, 1.02 / 1.13 / 1.09 |

A product whose launch shape is pick_shape's fallback (no tile within a third of the SM: the K-quants and IQ4_XS at
K = 6144 from 4–6 columns on, Q6_K's 2560-wide rows) also goes to the other kernels: wherever the perf runs timed such
a shape they were faster (level for Q4_K at 8 columns). Q3_K, IQ4_NL, IQ1_M, the 2560-wide IQ4_XS, Q4_K and Q8_0 rows
and Q2_0 from 3 columns stay on the k-column kernels everywhere. `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` turns the table
off; mode 2 (`GGML_CUDA_MMVQ_PASCAL_KCOL=2`) ignores it. The table is an estimate from timings at the old shapes; C350
times the products again on the new build (its `bench` step) so it can be refitted (Q915).

**Q916: FP32 activation sums in llama.cpp's `mul_mat_vec_q`.** The first build's checks found that with the k-column
kernels off, Q4_0 and Q5_0 at 8 tokens give inf and NaN on the cases whose activation blocks sum past FP16's range
(`test_mul_mat_large_act`, 250,000 in some channels); `-ladder3` gives the same. Those products then run on llama.cpp's
`mul_mat_vec_q`, whose dot products for Q4_0, Q4_1, Q5_0, Q5_1 and IQ1_S read the activation block's sum from
`block_q8_1`'s FP16 field, which is infinite past 65,504 (llama.cpp #23606). The fork's Pascal kernels already take that
sum from the block's int8 values in FP32 (Q76). Since Q916 these five dot products do the same: each thread sums its own
int8 values with dp4a, which gives its share of the block sum exactly (`d4 · d8 · (Σ q4·q8 − 8 Σ q8)` for Q4_0), in a
template flag of `vecdotq.cuh`'s impls that only the matrix-vector callers set. Q4_K and Q5_K's matrix-vector products
already summed the int8 values. The batch kernels (MMQ, more than 8 tokens: prompts) keep the FP16 sums (Q911). On the rig's
line these types reach `mul_mat_vec_q` as Q4_1 at 5–8 tokens, Q5_1 and IQ1_S at every token count (they have no Pascal
kernel), and every one of them with the k-column kernels off; Q824's table sends Q4_0 and Q5_0 at 2–4 tokens to the
fork's Pascal multi-token kernel, which never reads the FP16 sum, and keeps them on the k-column kernels at 5–8.

## Build

`scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-kshape` with `TESTS=1` (sm_61 code and compute_61 PTX,
CUDA 12.9.2, no AVX, highest glibc symbol 2.38), [run/build.sh](run/build.sh):

| File | Bytes | md5 |
|---|---:|---|
| `llama.cpp-20261104-39c3ab5f1-pascal-p106-v3-b12076-kshape.tar.gz` (the tip `d27845b67` adds two test-comment commits only) | 515,471,617 | `ff77255163b30b74b2e73a19c3ce73cf` |
| `extra-39c3ab5f1.tar.gz` (`test-llama-archs`, `test-llama-context`, `test-draft-mem`, `test-hop-cast`) | 150,194,278 | `77780542ded77519aab04f3926fa64f6` |
| `llama.cpp-20261104-4ba3f424d-pascal-p106-v3-b12076-kshape.tar.gz` (before Q916; the first checks ran on it) | 515,394,111 | `570d996f46bca2f8ab421f9daf7db9c5` |
| `extra-4ba3f424d.tar.gz` | 150,175,514 | `a9d0de6089c1e54967a2632f520bcdec` |
| bench `mmvq_kcol` (`build_kcol.sh` with the branch's grid tables) | | `72fb9c53e56a81ba1a9bd73cfa52f404` |

The bench's `src/kcol_mm.cuh` is the branch's `mmvq-pascal-kcol-impl.cuh` after its banner, byte for byte. Kernel
census of `test-backend-ops` against `-ladder3`'s ([kernels_added.tsv](kernels_added.tsv), `cuobjdump -res-usage`):
8,013 kernels against 8,010, every one of `-ladder3`'s with the same registers, stack and local memory; the 3 added are
the multi-token Q2_0 kernel without a gate and with direct loads for 2, 4 and 8 members (68, 84 and 115 registers, no
local memory). The k-column kernels' device code is unchanged: Q823 and Q824 are host code.

## Correctness

test-backend-ops compares each product on the card with the CPU backend ([checks.tsv](checks.tsv), [run/gpu_checks.sh](run/gpu_checks.sh)),
through the build's compute 6.1 PTX on an RTX 3080 and an RTX 3090 with the fork's kernels forced on and a P106's launch
shapes (`GGML_CUDA_MMVQ_PASCAL=2`, `_KCOL_NSM=5`, `_KCOL_SMSM=98304`):

| Build | What | Runs | Result |
|---|---|---|---|
| `4ba3f424d` | the k-column products of 1–8 columns, every k-column type (1,961 cases) | defaults, `KCOL_TABLE=0`, `KCOL=2`, `KCOL=2` with `KCOL_SMEM=49152`, `=32768`, `KCOL_R=1`, `=2`, `KCOL_ITT=1` | 1,961/1,961 each |
| `4ba3f424d` | the same, `KCOL=0` (the other kernels) | 1 | 1,959/1,961: Q4_0 and Q5_0 at 8 columns on the large-activation cases (inf, NaN), the same on `-ladder3`; Q916 |
| `4ba3f424d` | the Q2_0 experts (`MUL_MAT_ID`, the fused gate/up, the routed + shared launch; 340 cases) | defaults, `MTOK=1,0,0,0,0`, `=1,0,0,1`, `=1,1`, `=1,0,1`, `=0` | 340/340 each |
| `4ba3f424d` | the grid types' and IQ1_M's experts (952 cases) | defaults | 952/952 |
| `4ba3f424d` | `mmvq_kcol --check-only`: every launch-shape variant of every type at 1–8 columns against a double-precision reference | 1 | 2,592 checks, all variants match |
| `39c3ab5f1` | the k-column products (1,961 cases) | defaults, `KCOL=2`, `KCOL=0` | 1,961/1,961 each (`KCOL=0`'s 2 cases included) |
| `39c3ab5f1` | every large-activation case (138: 126 and Q916's 12) | defaults; `KCOL=0` with Q4_0, Q4_1, Q5_0 on llama.cpp's kernel at every token count | 138/138 each |
| `39c3ab5f1` | the Q2_0 experts (340 cases) | defaults | 340/340 |

SASS ([run/sass_all.sh](run/sass_all.sh), [sass_q916.tsv](sass_q916.tsv)): of the 7,594 kernel names in `39c3ab5f1`'s
`test-backend-ops`, 7,521 have the same sm_61 code as in `4ba3f424d`; the 73 that differ are every `mul_mat_vec_q` and
`mul_mat_vec_q_moe` kernel of Q4_0, Q4_1, Q5_0, Q5_1 and IQ1_S. The fork's one-token and multi-token Pascal kernels,
the k-column kernels and MMQ are unchanged by Q916. The P106 itself runs the native checks first in C350
([run/rig_tk.sh](run/rig_tk.sh), `STEPS=check`).

## On the rig (C350)

One P106-090 of the first rig for the native checks, then nine for T(k); `run/rig_tk.sh` with `STEPS="check tk"`
(2026-11-04, 16:46–19:10).

**Native checks** (`gpu_checks.sh MODE=native`, the P106 against the rig's CPU): the k-column products at the defaults
1,961/1,961; the routed Q2_0 experts with the down projection on tiles 340/340 and with direct loads 340/340;
`mmvq_kcol --check-only`, every launch-shape variant and type, 2,592 PASS lines, 0 FAIL. `kcol-all` and `mtok-iq` were
left out (the rig's CPU makes each run ~1 h; both passed through the PTX).

**T(k)** of the published Swift Q2_0 (Q591's method: llama-bench `-p 1,...,8 -n 0`, five repetitions, random prompt
tokens; ms per pass, median; standard deviation 2–4 ms; [tk_rig.tsv](tk_rig.tsv)), all `GGML_CUDA_MMVQ_PASCAL_…`:

| k | defaults | `KCOL_TABLE=0` | **`KCOL_SMEM=32768 KCOL_TABLE=0`** | the same, `MTOK=1,0,0,0,0` | `KCOL_SMEM=24576 KCOL_TABLE=0` | `KCOL_SMEM=49152 KCOL_TABLE=0` | `MTOK=0` | `-ladder3`, `KCOL_SMEM=32768` (Q591) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 80.6 | 74.0 | **74.1** | 74.9 | 74.8 | 83.1 | 74.8 | 74.9 |
| 2 | 109.7 | 103.6 | **102.4** | 113.6 | 101.8 | 118.4 | 113.2 | 107.7 |
| 3 | 139.6 | 131.1 | **129.9** | 131.3 | 126.1 | 140.8 | 149.2 | 138.5 |
| 4 | 158.3 | 150.5 | **149.0** | 149.9 | 149.9 | 159.2 | 179.8 | 162.8 |
| 5 | 188.2 | 182.3 | **180.2** | 179.6 | 179.1 | 191.9 | 221.0 | 205.3 |
| 6 | 213.0 | 207.1 | **202.3** | 201.5 | 205.4 | 216.7 | 257.4 | 234.7 |
| 7 | 235.0 | 226.0 | **222.1** | 221.1 | 225.1 | 241.9 | 289.4 | 261.7 |
| 8 | 266.7 | 257.2 | **254.4** | 253.2 | 256.2 | 276.4 | 322.2 | 297.3 |
| T(5) / T(1) | 2.34 | 2.46 | **2.43** | 2.40 | 2.39 | 2.31 | 2.95 | 2.74 |

- **The routed down projection on the multi-token kernel (Q825) is the gain**: with the same 32 KB budget a 5-token pass
  takes 180.2 ms against `-ladder3`'s 205.3 (−25 ms), an 8-token pass 254.4 against 297.3 (−43); one token is unchanged.
  The multi-token experts as a whole save 33 ms at five tokens (`MTOK=0`: 221.0).
- **The defaults lose at one to three tokens**: Q823's shapes with Q824's table take 80.6 / 109.7 / 139.6 ms where the
  32 KB budget with the table off takes 74.1 / 102.4 / 129.9 (Q829). Turning the table off at Q823's shapes
  (`KCOL_TABLE=0`) already gets within 0.1–5 ms of the 32 KB budget, so the table's choices are the loss. The shapes
  before Q823 (`SMEM=49152`) are slower at every k.
- The down projection's direct loads (`MTOK=1,0,0,0,0`) match the tiles except at two tokens (113.6 against 102.4);
  `SMEM=24576` matches 32 KB within the noise.

**The bench on the rig** (`mmvq_kcol --tokens 1,...,8`, card 0, every variant passed its CPU reference;
[kcol_bench_rig.csv](kcol_bench_rig.csv)), ms a token over its Flash-Next shapes (calls per token × µs per call):

| variant | k=1 | k=2 | k=3 | k=4 | k=5 | k=6 | k=7 | k=8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pascal` (this build's shapes) | 47.63 | 54.24 | 62.83 | 69.69 | 78.22 | 85.21 | 97.28 | 108.17 |
| `old` (the shapes before Q823) | 68.77 | 76.90 | 82.33 | 88.01 | 97.39 | 104.93 | 116.11 | 126.61 |
| `s32k` (a 32 KB budget) | 45.97 | 52.33 | 60.83 | 65.39 | 73.39 | 66.79 | 77.14 | 87.11 |

Q823's shapes take 18–31 % less than the old ones at every k; a 32 KB budget is still 2–22 % faster than them. The
bench step's in-ggml perf arms (test-backend-ops perf) were left out: their time goes into preparing weights on the
rig's two-core CPU.

## Files

| File | What it is |
|---|---|
| [shapes_est.tsv](shapes_est.tsv) | per type, bench shape and k: the variant with Q823's shape, its µs, the 32 KB budget's, the old default's, the best variant's; per k the sums per token (the `#` lines) |
| [ggml_est.tsv](ggml_est.tsv) | per Q591 perf row: the other kernels' µs, the k-column µs at the old shape, the bench's ratio new / old, the estimate after Q823, the same for the 32 KB budget |
| [checks.tsv](checks.tsv) | every correctness run: build, card, switches, ops and filter, cases passed |
| [kernels_added.tsv](kernels_added.tsv) | the kernels the build adds to `-ladder3`'s, with their resources |
| [sass_q916.tsv](sass_q916.tsv) | the kernels whose sm_61 code Q916 changed (the rest identical) |
| [run/kshape_est.py](run/kshape_est.py) | the two estimate tables from Q591's bench CSVs and perf tables |
| [run/build.sh](run/build.sh) | the rig tarball, the extras and the bench |
| [run/gpu_checks.sh](run/gpu_checks.sh) | the correctness runs (MODE=ptx on another card, MODE=native on a P106) |
| [run/sass_all.sh](run/sass_all.sh) | per-kernel SASS identity of two builds |
| [run/rig_tk.sh](run/rig_tk.sh) | C350 on the rig: the native checks, T(k) per switch setting, the bench and perf timings |
| [tk_rig.tsv](tk_rig.tsv) | C350's T(k) on the rig per switch setting: mean and median ms, standard deviation, ms per token, T(k) / T(1), the step from k − 1 |
| [kcol_bench_rig.csv](kcol_bench_rig.csv) | C350's `mmvq_kcol` on the rig: per shape, type, k and variant, µs, GB/s and the check |

## Run

```bash
# estimates (any machine with Python 3), from Q591's folder
cd docs/data/2026-11-04_kcurve_p106
python3 -I ../2026-11-04_kshape/run/kshape_est.py shapes 5 kcol_dense.csv kcol_iq.csv > ../2026-11-04_kshape/shapes_est.tsv
python3 -I ../2026-11-04_kshape/run/kshape_est.py ggml 5 kcol_dense.csv kcol_iq.csv -- perf_dense.tsv perf_cov.tsv \
  > ../2026-11-04_kshape/ggml_est.tsv
# build (a build machine with Docker; WORK/src a llama.cpp clone with the fork as remote "fork")
WORK=<dir> REPO=<this repository> JOBS=8 DOCKER_ARGS='--cpu-shares=256' bash docs/data/2026-11-04_kshape/run/build.sh
# correctness through the PTX, one container per card, the runs split between them
docker run --rm --gpus '"device=<card>"' -v <dirs> -e MODE=ptx -e BIN=<tree>/bin -e OUT=<dir> -e DEV=0 \
  -e RUNS='kcol kcol-table0 kcol-all kcol-old kcol-r1' -e KCOL_BENCH=<mmvq_kcol> nvidia/cuda:12.9.2-devel-ubuntu24.04 \
  nice -n 19 bash docs/data/2026-11-04_kshape/run/gpu_checks.sh
# (the second card: RUNS='kcol-s32k kcol-r2 kcol-itt1 kcol-off mtok mtok-direct mtok-gdir mtok-cap1 mtok-rb1 mtok-off mtok-iq bench';
#  the Q916 build: RUNS='stock stock-skip kcol-off kcol kcol-all mtok')
# per-kernel SASS of two builds, in the CUDA container (CPU only)
OUT=<dir> bash docs/data/2026-11-04_kshape/run/sass_all.sh <4ba3f424d tree>/bin/test-backend-ops <39c3ab5f1 tree>/bin/test-backend-ops
# on the rig (C350), idle: native checks, T(k) arms, bench and perf
BIN=<kshape tree>/bin MODEL=<first shard> VC=<folder with verify_cost.sh, mem_guard.sh, smoke_request.json> OUT=<dir> \
  KCOL_BENCH=<mmvq_kcol> KCOL_DENSE=docs/data/2026-11-04_kcol_dense setsid nohup bash run/rig_tk.sh > rig_tk.out 2>&1 &
```

The estimates are from Q591's P106 measurements; this folder adds no timing. Licence: the repository's.
