# Q630, Q611, Q612: the k-token pass's remaining dense products and launch shapes (2026-11-04)

Registry rows Q630 (Q4_0, Q5_0 and IQ4_NL in the k-column kernels), Q611 (the BF16 kernel's launch shapes at 3 and 4
tokens) and Q612 (the one-token Q2_0 gate/up on small grids) in [quants/experiments.md](../../../quants/experiments.md).
They extend [Q591's k-column kernels](../2026-11-04_kcol_dense/README.md) (fork branch `pascal-p106-v3-b12076-kcol`)
to the last dense products of Qwen3.8-Flash-Next's Swift Q2_0 file that a k-token pass (a speculative-decoding verify
of k = 2 to 8 tokens) still ran on kernels that loop over the tokens, and retune two launch shapes from
[R16's timing on a P106](../2026-11-03_kernel_timing/README.md).

## What changed

On the same fork branch, as commits on top of Q591's:

- **Q630, `ggml/src/ggml-cuda/mmvq-pascal-kcol*.cu`:** Q4_0, Q5_0 and IQ4_NL join the k-column kernels at 1 to 8
  columns. In the Swift Q2_0 file they are the shared expert's other down projections (640 → 2,560: Q4_0 ×16,
  Q5_0 ×7, IQ4_NL ×7, 30 calls a token beside Q2_0's 16), which a k-token pass ran on the earlier 1–4-column Pascal
  kernel (Q4_0, Q5_0 at 2–4 tokens) or llama.cpp's (IQ4_NL at every count, every type at 5–8).
  - A lane unit is two 32-weight blocks (36 bytes for Q4_0 and IQ4_NL, 44 for Q5_0), as Q2_0's unit is two of its
    64-weight blocks, so that units stay 4-byte aligned in the staging buffer; rows need a multiple of 64 weights.
  - Q4_0: the nibbles stay in place (weights 4k..4k+3 and 4k+16..4k+19 of code word k), and the offset −8 comes from
    the activation records: a new record kind holds −8 times each block's sum of int8 values, the dp4a chain's initial
    value, in place of the second float.
  - IQ4_NL: the nibbles are looked up in `kvalues_iq4nl` with `__byte_perm`, as IQ4_XS's (the lookup is now one helper
    for both), the records in the lookup's order.
  - Q5_0: the records hold each block's values in the order (m, m+8, m+16, m+24) per word m, so that one
    `__byte_perm` gathers the four nibbles (low and high nibbles of code bytes m and m+8) and one shift and mask moves
    the four fifth bits (bits m, m+8, m+16, m+24 of `qh`) to bit 4 of each byte, as for Q5_K; the offset −16 comes from
    the records as Q4_0's.
  - Per 32 weights, row and column: 8 dp4a and 2 FP32 operations; the decoding once per row: about 0.3 instructions
    per weight for Q4_0, 1.3 for IQ4_NL (the lookups) and 1.6 for Q5_0.
  - IQ4_NL also takes the one-token products (`GGML_CUDA_MMVQ_PASCAL_KCOL=1`, the default), since it has no other
    Pascal kernel, as Q3_K; Q4_0 and Q5_0 keep the one-token Pascal kernels there.
- **Q611, `mmvf-pascal.cu` (`mmvfp_pick_shape`):** at 3 and 4 tokens the lane-group size from the row width stops at
  32 lanes, and a grid with fewer than 4 blocks per SM gets more lanes before fewer rows. In R16's sweep (rows per group
  1 and 2 at these counts), the 10,240-wide `hc_*_down` rows ran 50.7 / 58.5 µs at 32 lanes against 59.4 / 70.0 at 128
  (3 / 4 tokens), and the recurrent layers' 2,560 × 48 alpha and beta 4.4 / 4.9 µs at 128 lanes and 2 rows against
  5.6 / 7.3 at 1 row. On a P106: `hc_*_down` G = 32 (20 blocks), alpha/beta G = 128, R = 2 (24 blocks); the
  320-wide `hc_*_up` (G = 8, R = 4), the indexer (G = 32, R = 4), the injects (G = 128, R = 1) and every shape at 1, 2
  and 5–8 tokens keep their shapes. Rows per group stay as Q591 set them (4 at 3–4 tokens): R16's sweep had only 1 and
  2 rows at these counts, and at 2 tokens 4 rows beat 2 by 9 % on the 10,240-wide rows and by 33 % on the indexer
  (which therefore keeps G = 32, R = 4 rather than R16's G = 128, R = 2, which also lost at 4 tokens: 33.7 against
  29.3 µs). Q591's 1–8-token sweep on a P106 decides between 2 and 4 rows (Q650).
- **Q612, `mmvq-pascal.cu`:** a one-token Q2_0 product with a fused gate takes R = 1 row per lane group where R = 2
  leaves fewer than 4 blocks per SM over the launch (the shared expert's 2,560 → 640 gate/up in its own launch: 10
  blocks on a P106, 7.78 against 10.98 µs at R = 1 in R16). The routed gate/up (100 blocks, 110 with the shared expert
  in the same launch) keeps R = 2. `GGML_CUDA_MMVQ_PASCAL_NSM=n` picks the shape for n SMs (5: the P106's, on any
  GPU); `GGML_CUDA_MMVQ_PASCAL_Q2_0=2` forces R = 2 as before.
- **`tests/test-backend-ops.cpp`:** the k-column eval block has the three types at 1–8 tokens (the shared expert's
  shape, odd row counts, 192-wide rows with one lane per row, rows that are views of wider rows, and rows the kernels
  leave to the others: an odd number of blocks, 2-byte-aligned views), activations whose block sums pass FP16's range,
  and two Q2_0 gate/up grids just at and below 4 blocks per SM of a P106; the perf block has the three types'
  640 → 2,560 product at 1–8 tokens.

Bench: [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) `build/mmvq_kcol` has the three types' `shexp_down`
shapes (its `src/kcol_mm.cuh` is the fork's header, copied verbatim), `--bf16`'s `pascal` rows and `--q2_0`'s
`pascal` rows follow the new rules (`src/bf16_mv.cuh`, `src/q2_0_mv.cuh`).

## Results so far

| Check | Result |
|---|---|
| build for the rig (sm_61 + compute_61 PTX, `TESTS=1`, commit `af46311af`) | compiles without warnings in the changed files; no spills in any kernel of the bench build; the 48 new product kernels take 56–138 registers, no stack or local memory ([registers.tsv](registers.tsv)) |
| `verify/kcol_unit_check` (CPU: each type's decoding against ggml's reference dequantizers) | 10 / 10 types, 240 cases each, worst error 0.0037 of the tolerance ([decoding_check.md](decoding_check.md)) |
| the same with three planted errors (`run/mutation_check.sh`) | fails exactly Q4_0, Q5_0 and IQ4_NL |
| correctness on a GPU (test-backend-ops, mmvq_kcol) | next: Q591's `rig_kcol.sh`, then `run/rig_kcol_cov.sh` on one P106 |
| kernel times | next: the same runs |

### What to expect

The three types' 640 → 2,560 products are 0.92 MB (Q4_0, IQ4_NL) and 1.13 MB (Q5_0) a call, ~7–9 µs at the ~130 GB/s
the tuned kernels reach on a P106. Per 32 weights, row and column the kernels spend about 11–12 lane instructions at
2 rows per lane group (8 dp4a, 2 FP32 operations, the column's three shared loads shared by both rows), and per row
the decoding about 10 (Q4_0), 40 (IQ4_NL) or 50 (Q5_0) per 32 weights; at the card's ~5.7 lane instructions per weight
byte (about 100 per 32 weights of Q4_0 and IQ4_NL, 125 of Q5_0) that keeps Q4_0 at memory speed to about 8 columns,
Q5_0 to about 6 and IQ4_NL to about 5, so a pass's 30 calls est. ~0.25–0.3 ms at k ≤ 5. BF16 at 3–4 tokens: est.
−0.9 / −1.3 ms per step on `hc_*_down` and alpha/beta if 4 rows per group hold R16's gains at 32 lanes. Q2_0: −0.15 ms
per token where the shared expert's gate/up runs in its own launch.

## Files

| File | What it holds |
|---|---|
| [decoding_check.md](decoding_check.md) | `verify/kcol_unit_check` on a CPU for all ten types, and the mutation run |
| [registers.tsv](registers.tsv) | per k-column kernel of the three new types (columns, rows per lane group): registers, stack, local memory, from the sm_61 code of `build/mmvq_kcol` (the same header) |
| [run/build.sh](run/build.sh) | the builds: the rig tarball with test-backend-ops, both bench binaries, the CPU check and the mutation check |
| [run/mutation_check.sh](run/mutation_check.sh) | the CPU check on a copy with one planted error per new type |
| [run/registers.sh](run/registers.sh) | `registers.tsv` from the bench build's `cuobjdump -res-usage` output |
| [run/rig_kcol_cov.sh](run/rig_kcol_cov.sh) | one P106 after Q591's `rig_kcol.sh`: test-backend-ops correctness of the three types in every launch-shape setting, the Q2_0 gate/up with and without the small-grid rule, `--q2_0`, test-backend-ops perf of the three types with the k-column kernels off and on |

## Commands

The builds (a build machine with Docker, CPU only):

```
WORK=<work folder> REPO=<this checkout> LOCK=<shared lock file> JOBS=6 docs/data/2026-11-04_kcol_coverage/run/build.sh
```

Correctness through the compute 6.1 PTX on a newer card (here an RTX 3090 in Docker, with the P106's launch shapes),
after Q591's run of `rig_kcol.sh` on the same build:

```
docker run --rm --gpus device=<gpu> -v <work>:/w -e GGML_CUDA_MMVQ_PASCAL=2 -e CUDA_FORCE_PTX_JIT=1 \
  -e GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 -e GGML_CUDA_MMVQ_PASCAL_NSM=5 -e BIN=/w/<build>/bin -e MMVQ_PASCAL=/w/mmvq_pascal \
  -e OUT=/w/ptx_cov -e PARTS="check q2_0" -e Q2_BENCH_ARGS="--check-only" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash /w/rig_kcol_cov.sh
```

On the rig, one card, nothing else running, after Q591's `rig_kcol.sh` (which, on this build, also times the new
types' `shexp_down` shapes and sweeps the BF16 shapes at 1–8 tokens): ~20 min on the rig's Celeron for the correctness
runs, a few minutes for the rest.

```
ssh $RIG "BIN=<build>/bin MMVQ_PASCAL=<dir>/mmvq_pascal OUT=<dir>/out_cov setsid nohup bash <dir>/rig_kcol_cov.sh \
  > <dir>/rig_kcol_cov.out 2>&1 < /dev/null &"
```

Then, on another machine: Q591's `run/perf_table.py <out_cov> > perf.tsv`; `<out_cov>/q2_0.csv` is the `--q2_0` table
(`shexp_gate_up`'s `pascal` row now runs R = 1; `r2_seg` is the earlier shape).

## Provenance and licence

Built 2026-11-04 from fork branch `pascal-p106-v3-b12076-kcol` with `nvidia/cuda:12.9.2-devel-ubuntu24.04`
(`scripts/build_llamacpp_docker.sh`); the CPU checks ran in the same image. The weights and activations of every check
are random. The reference dequantizers in `bench/mmvq_pascal/src/kcol_ref.h` follow ggml's `ggml-quants.c` (MIT, the
ggml authors). Tables and scripts here are under this repository's licence.
