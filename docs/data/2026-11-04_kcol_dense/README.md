# Q591: k-column products that read and decode each weight once for 1–8 tokens (2026-11-04)

Registry rows Q591 (with Q224, the Pascal Q3_K kernel, and Q604, short Q8_0 rows) in
[quants/experiments.md](../../../quants/experiments.md). A speculative-decoding draft pays only if checking k drafted
tokens in one pass of the main model costs much less than k one-token passes. On Miner Llama v3's base a k-token pass
of Qwen3.8-Flash-Next costs 136 ms plus 50–59 ms per extra token ([Q583](../2026-11-03_verify_cost/README.md)): at
4 tokens the BF16 hyper-connections cost 2.3× a token's and the Q3_K/IQ4_XS projections 2.8×, because the kernels that
run them decode every weight and recompute every activation term once per column. These kernels make the dense
products' cost nearly independent of the column count. The goal: a 5-token pass of these products at most ~1.3× a
1-token pass.

## The kernels

Fork branch `pascal-p106-v3-b12076-kcol` (off `pascal-p106-v3-b12076-ladder`, the base plus the BF16 and Q2_0
kernels):

- **`ggml/src/ggml-cuda/mmvq-pascal-kcol*.cu`**, for Q8_0, Q2_0, Q3_K, Q4_K, Q5_K, Q6_K and IQ4_XS at 1 to 8 columns
  (plain `MUL_MAT`; the experts' `MUL_MAT_ID` is Q604's sibling work in `mmvq-pascal.cu`). The design is at the top of
  `mmvq-pascal-kcol-impl.cuh`:
  - each weight is read once and decoded once into int8 words, then multiplied with dp4a by every column's
    activations; the weight scales become FP32 once per row and serve every column;
  - the activations are quantized straight from FP32 as `quantize_q8_1` does it (32 values per FP16 scale), but into
    records laid out for these kernels: per lane unit and column the int8 values in the order each type's decoding
    returns its weights, the FP32 scales, the K-quants' block sums (for their mins) and Q3_K's and Q6_K's half-block
    offset sums (the dp4a chains' initial values), padded so that the lanes of a quarter warp read different banks;
    each block copies its K tile of records into shared memory once, so a column costs two 128-bit and one 64-bit
    shared load per 32 weights, shared by the lane group's rows;
  - per 32 weights, row and column: 8 dp4a and 3 (Q8_0, IQ4_XS), 4 (Q4_K, Q5_K) or 5 (Q3_K, Q6_K) FP32 operations;
  - weights are staged in shared memory with coalesced 128-bit loads, by each lane group for its own rows where a
    row's segment is whole 16-byte units, else by the whole warp, each row from the 16-byte boundary below it
    (Q3_K's and Q6_K's 2-byte-aligned blocks, short Q8_0 rows such as the 320-wide hyper-connection projections);
  - a row's sum is formed in a fixed order that depends on the row width only, so every column's result is the same
    for any column count or launch shape;
  - switches: `GGML_CUDA_MMVQ_PASCAL_KCOL=0` off, `1` (default) 2 to 8 columns plus one column for Q3_K (no other Pascal
    kernel; Q224) and for Q8_0 rows whose 64-weight chunks aren't a multiple of 4 (Q604's 320-wide rows), `2` every
    product of 1 to 8 columns; `_SKIP=<types>`, `_R=1|2`, `_ITT=<iterations per tile>`, `_SMEM=<bytes>`, `_NSM=<SMs>`.
- **`mmvf-pascal*.cu`** (the BF16 kernel, Q546): a lane may write several outputs, so 4 rows per lane group serve 3 and
  4 tokens too (2 before, which cost 1.2–1.5× at 3–4 tokens in [Q546's timing](../2026-11-03_kernel_timing/README.md);
  at 2 tokens and 4 rows the kernel ran as fast as at one) and 8 rows serve 5 to 8 tokens.
- **`tests/test-backend-ops.cpp`**: correctness cases at 1 to 8 tokens for every type at Flash-Next's dense shapes, odd
  row counts, each lane-group size, both staging modes, rows that are views of wider rows, activations whose block sums
  pass FP16's range; perf cases at the exact shapes (the output layer at 248,320 rows).

Bench: [bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md) `build/mmvq_kcol` (`build_kcol.sh`) times the same
code (`src/kcol_mm.cuh`, a verbatim copy) per shape, column count and launch shape, against a double-precision CPU
reference, with per-pass totals over the Swift Q2_0 file's dense products (its tensor census: 18 Q3_K, 13 IQ4_XS,
4 Q4_K and 1 Q2_0 `attn_qkv`, and so on; the hyper-connections as the Q8_0 retype).

## Results so far

| Check | Result |
|---|---|
| build for the rig (sm_61 + compute_61 PTX, `TESTS=1`) | compiles; no register spills in any of the 112 product kernels (56–244 registers; Q3_K, Q5_K and Q6_K at 2 rows and 5–8 columns 207–244, 2 blocks per SM) or the 108 BF16 kernels (30–254; 8 rows at 7–8 tokens 240–254, 2 blocks per SM; 4 rows at 8 tokens 144) |
| `verify/kcol_unit_check` (CPU: each type's decoding against ggml's reference dequantizers) | 7 / 7 types, 240 cases each (1, 3, 8 columns; 1, 2 rows; units at every staging alignment; activations past FP16's range), worst error 0.0044 of the tolerance |
| the same with two planted errors (the 2-byte shift of misaligned units, Q4_K's high-nibble scale) | fails exactly Q3_K, Q6_K and Q4_K |
| correctness on a GPU (test-backend-ops, mmvq_kcol) | next: `run/rig_kcol.sh` on one P106 |
| kernel times at 1–8 columns | next: `run/rig_kcol.sh` |

[decoding_check.md](decoding_check.md) has the CPU check's output, [registers.tsv](registers.tsv) every kernel's
registers.

### What to expect

A P106-090 issues about 30 G warp instructions per second; at 173 GB/s that leaves 5.7 lane instructions per weight byte:
2.4 per weight for Q3_K, 3.0 for IQ4_XS, 3.2 for Q4_K, 6.0 for Q8_0. With about 0.4 per weight per column, plus the
decoding once per row (0.2 for Q8_0, 0.5 for Q4_K, 1.1–1.4 for Q3_K, Q5_K, Q6_K and IQ4_XS), est. columns at memory
speed: Q8_0 ~14, Q4_K, Q5_K and Q6_K ~6–7, IQ4_XS ~4, Q3_K and Q2_0 ~3; at 5 columns est. Q3_K ~1.4× and IQ4_XS ~1.1×
its one-column time, against 2.8× at 4 tokens now. BF16 needs one FP32 FMA per weight and column: est. at memory speed
up to ~6 columns.

## Files

| File | What it holds |
|---|---|
| [decoding_check.md](decoding_check.md) | `verify/kcol_unit_check` on a CPU: the build's kernels' decoding against ggml's reference, and the mutation run |
| [registers.tsv](registers.tsv) | per product kernel (type, columns, rows per lane group): registers, stack, local memory, from the sm_61 code |
| [run/build.sh](run/build.sh) | the builds: the rig tarball with test-backend-ops, the bench binary, the CPU check |
| [run/rig_kcol.sh](run/rig_kcol.sh) | one P106: test-backend-ops correctness in every launch-shape setting, the k-column bench, the BF16 kernel's launch-shape sweep, test-backend-ops perf with the new kernels off and on (also the PTX correctness run on another GPU) |
| [run/perf_table.py](run/perf_table.py) | the perf logs as a table: µs per call off and on, the speed-up, each arm's cost against its own one-token call |

## Commands

The builds (a build machine with Docker, CPU only):

```
WORK=<work folder> REPO=<this checkout> LOCK=<shared lock file> JOBS=6 docs/data/2026-11-04_kcol_dense/run/build.sh
```

Correctness through the compute 6.1 PTX on a newer card (here an RTX 3090 in Docker, with the P106's launch shapes):

```
docker run --rm --gpus device=<gpu> -v <work>:/w -e GGML_CUDA_MMVQ_PASCAL=2 -e GGML_CUDA_MMVF_PASCAL=2 -e CUDA_FORCE_PTX_JIT=1 \
  -e GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 -e GGML_CUDA_MMVF_PASCAL_NSM=5 -e BIN=/w/<build>/bin -e MMVQ=/w/mmvq_kcol -e OUT=/w/ptx \
  -e PARTS="check bench" -e BENCH_ARGS="--check-only --nsm 5" nvidia/cuda:12.9.2-devel-ubuntu24.04 bash /w/rig_kcol.sh
```

On the rig, one card, nothing else running (the tarball unpacked to `<build>`, `mmvq_kcol` and bench/mmvq_pascal's
`mmvq_pascal` copied next to the script; correctness ~1 h on the rig's Celeron, the benches ~30 min, perf ~15 min):

```
ssh $RIG "BIN=<build>/bin MMVQ=<dir>/mmvq_kcol MMVQ_PASCAL=<dir>/mmvq_pascal OUT=<dir>/out PARTS='check bench bf16 perf' \
  setsid nohup bash <dir>/rig_kcol.sh > <dir>/rig_kcol.out 2>&1 < /dev/null &"
```

Then, on another machine: `run/perf_table.py <out> > perf.tsv`; `<out>/kcol.csv` is the bench's table.

## Provenance and licence

Built 2026-11-03/04 from fork branch `pascal-p106-v3-b12076-kcol` with `nvidia/cuda:12.9.2-devel-ubuntu24.04`
(`scripts/build_llamacpp_docker.sh`); the CPU check ran in the same image. The weights and activations of every check
are random. The reference dequantizers in `bench/mmvq_pascal/src/kcol_ref.h` follow ggml's `ggml-quants.c` (MIT, the
ggml authors). Tables and scripts here are under this repository's licence.
