# Lossless BF16 for Qwen3.8-Flash-Next: the entropy of its BF16 weights, two codes, a P106 bench

Registry row: [Q621](../../../quants/experiments.md) (exact BF16 from fewer bytes), with Q932 and Q935 (the codes
parked here).

Flash-Next keeps its hyper-connection projections, router, indexer and recurrent-layer gates in BF16. Swift Q2_0 and
`pmix-q2-fast` read 1,475 MB of them per decoded token (Swift Q2_0's router is F32 on disk, 252 MB, with BF16 values;
`pmix-q2-fast` stores it as BF16). On a P106, Miner Llama's Pascal BF16 kernel runs the hyper-connection products at
~159 GB/s, about 8.9 ms a token for the five shapes it was timed on
([kernel timing](../2026-11-03_kernel_timing/README.md)). A lossless code keeps every weight bit for bit, so it needs no
KL-divergence check; it pays only if the kernel decodes the coded bytes faster than it would have read the plain ones.

This folder holds the entropy of every unquantized matrix of both files (CPU), the two codes chosen from it, their
round-trip check, the P106 microbench ([bench/mmvq_pascal](../../../bench/mmvq_pascal/README.md), "Lossless BF16
(Q621)") built and checked on the CPU, and the run file for its timing on one P106 (not run yet).

## The entropy (2026-11-04)

Each BF16 weight is sign (1 bit), exponent (8), mantissa (7). Measured over all 737.6 M weights of each file's BF16 matrices
([entropy.md](entropy.md); per tensor: [tensors_q2_0.tsv](tensors_q2_0.tsv), [tensors_pmix_q2_fast.tsv](tensors_pmix_q2_fast.tsv)),
in bits a weight:

| Class | Tensors | Weights (M) | Exponent byte | Sign + mantissa byte | High byte (sign, exponent >> 1) | Low byte (exponent bit 0, mantissa) | All 16 bits | x12 | x11 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `hc_*_down`, `output_hc_down` (10,240 → 320) | 97 | 317.9 | 2.73 | 7.97 | 2.81 | 7.97 | 10.67 | 12.01 | 11.92 |
| `hc_*_up`, `output_hc_up` (320 → 10,240) | 97 | 317.9 | 2.84 | 7.97 | 2.90 | 7.97 | 10.78 | 12.11 | 12.05 |
| router `ffn_gate_inp` (2,560 → 512) | 48 | 62.9 | 2.64 | 7.97 | 2.72 | 7.97 | 10.57 | 12.02 | 11.80 |
| `indexer.q_proj` (2,560 → 512) | 12 | 15.7 | 2.64 | 7.97 | 2.77 | 7.97 | 10.57 | 12.02 | 11.58 |
| `ssm_alpha`, `ssm_beta` (2,560 → 48) | 72 | 8.8 | 2.76 | 7.97 | 2.83 | 7.97 | 10.68 | 12.03 | 11.96 |
| `ple_value` (2,560 → 2,560) | 1 | 6.6 | 3.06 | 7.97 | 3.13 | 7.97 | 11.00 | 12.02 | 13.11 |
| `hc_*_inject` (10,240 → 4) | 96 | 3.9 | 2.95 | 7.97 | 3.00 | 7.97 | 10.83 | 12.02 | 12.02 |
| `indexer.k_proj` (2,560 → 128) | 12 | 3.9 | 2.73 | 7.97 | 2.83 | 7.97 | 10.66 | 12.02 | 11.75 |
| **all, MB a token** | 435 | 737.6 | | | | | **987** (0.67×) | **1,111** (0.753×) | 1,103 (0.748×) |

- **The redundancy is in the exponent alone:** the sign + mantissa byte holds 7.97 of its 8 bits and the low byte the
  same, while the exponent byte holds 2.6–3.1 of 8. The best a lossless code could do is the 16-bit entropy,
  10.6–11.0 bits a weight: 987 MB a token instead of 1,475, 0.67× (DFloat11's ~11 bits).
- **Per row it barely narrows:** the exponent's entropy within each row is 0.02–0.16 bits below the whole tensor's
  (class means; 0.38 at most, in `blk.0.hc_attn_up`), so a table per row buys little over one per tensor; the codes
  below choose their window per row anyway, since it costs one byte of the row word they need for the escapes.
- **A static Huffman code of the high byte** (one per tensor) would take 10.76–11.18 bits a weight, ~1.1 bits below
  x12: at most −92 MB a token below x12 on the hyper-connections (−0.6 ms at 159 GB/s), if its decoder were free (Q935).
- Both files hold the same values: all 435 matrices compare equal (Swift Q2_0's F32 router converts to BF16 exactly,
  `not_bf16` = 0 in every row).

## The codes

Both keep each weight's low byte raw and code its high byte (sign | exponent >> 1) as an index into a small table per
row; weights outside the table are escapes, kept as (column, BF16) in a per-row list and put back in place before the
products. The decoder runs in registers inside the matrix-vector kernel: byte permutes (`prmt`) turn the indices into
high bytes four at a time and interleave them with the low bytes into the exact 16-byte layout the plain kernel loads.
Code and tables: [src/lossless_bf16.h](../../../bench/mmvq_pascal/src/lossless_bf16.h) (encoders, the CPU decoder that
runs the kernels' permutes), [src/lossless_mv.cuh](../../../bench/mmvq_pascal/src/lossless_mv.cuh) (the kernels).

| Code | Index | Table per row | Escapes | Bits a weight (all classes) | Decode per 8 weights |
|---|---|---|---|---|---|
| **x12** | 4 bits: bank << 3 \| sign << 2 \| j | the 8 exponent pairs P … P−7 × 2 signs (16 exponents) | **0.017–0.043 %** | **12.01–12.11** (the 320-wide rows carry a 4-byte row word: 0.1 bit) | 3 shifts/masks, 4 `prmt` and 2 adds for the high bytes, 4 `prmt` to interleave |
| x11 | 3 bits: sign << 2 \| j, packed 32 weights to 3 words | the 4 pairs P … P−3 × 2 signs (8 exponents) | 1.8–6.6 % | 11.58–13.11 (11.0 without escapes) | ~1.4 ops and 2 `prmt` for the high bytes, 4 `prmt` to interleave |

**x12 is the code: 0.753× of BF16's bytes, every weight exact.** Its 16-exponent window covers all but ~1 in 5,000
weights, so the escape path is nearly never taken, and because each lane decodes the plain kernel's own 8-weight
chunks and runs the same FMA sequence, its output equals the BF16 kernel's bit for bit at the same launch shape (the
bench checks this). **x11 is parked (Q932):** a window of 8 exponents leaves 2.5–3 % of the weights out, and at
32 bits an escape that costs ~0.9 bits a weight, so it ends at x12's size with a costlier escape path. It stays in the
bench, whose timing shows what a cheaper decode is worth.

Round trip: every weight of all 435 matrices of both files decodes back bit for bit through the kernels' permute
steps ([entropy.md](entropy.md), last column), and so does the bench's sample
([bench_cpu_only.txt](bench_cpu_only.txt)).

## The microbench

`lossless_bench` (bench/mmvq_pascal, `build_lossless.sh`; sm_61 and compute_61 PTX) times each class's product at
1, 2, 3, 4, 5, 6 and 8 tokens over a sample of the model's own weights (27 tensors, 79 MB, from Swift Q2_0 by
`analyze.sh`): the plain BF16 kernel at the fork's launch shape and at 12 others, x12 at 12 shapes, x11 at up to 16,
and a pure read of each code's bytes per call (the card's floor for that size). Every variant is checked against a
double-precision product; each x12 output is compared bit for bit with the plain kernel's at the same shape. The
summary adds calls per token × the fastest correct variant per class.

**Expected** (until the rig run): if x12 keeps the plain kernel's ~159 GB/s, the four bandwidth-bound classes drop with
their bytes:

| Class | Plain kernel, 1 token (ms a token) | x12 at the same GB/s | Net |
|---|---:|---:|---:|
| `hc_up` | 4.00 | 3.03 | −0.97 |
| `hc_down` | 4.04 | 3.03 | −1.01 |
| router (est. at 150 GB/s) | 0.84 | 0.63 | −0.21 |
| `indexer_q` | 0.21 | 0.16 | −0.05 |
| `hc_inject`, `ssm_ab` (latency-bound at 20 and 62 GB/s) | 0.68 | 0.68 | 0 |
| **all timed classes** | **9.8** | **7.5** | **≈ −2.2** |

At 2 tokens the plain kernel still runs at ~160 GB/s, so the same −2 ms holds. At 3–4 tokens it falls to 108–128
GB/s, held by the activation loads and the instruction issue rather than the bytes, which is where x12's ~13 extra
integer operations per 8 weights (plus the escape check) land: the gain there is between 0 and −1.5 ms. x12 issues
~1.9× the plain kernel's instructions per weight; at 1 token and 159 GB/s of coded bytes that is ~40 % of the 5 SMs'
issue rate, so the decode should hide at 1–2 tokens. `indexer_k` and `ple_value` (~0.1 ms together) were not in the
plain kernel's timing and are left out of the sums. Against the Q8_0
retype (Q218: 0.53× the bytes, 1.8–2.1× the KL floor), x12 is exact and needs no new file: a kernel can repack the
BF16 tensors at load time.

## On one P106 (C378, 2026-11-04)

`run/rig_bench.sh` on card 0 of the second rig (nothing else running), the sample above, every variant checked: **all
checks pass** ([rig/lossless_rig.txt](rig/lossless_rig.txt), [rig/lossless.csv](rig/lossless.csv),
[rig/lossless_summary.csv](rig/lossless_summary.csv)). Milliseconds per decoding step over the timed classes (calls per
token × µs per call, each class at its fastest correct shape):

| Tokens | Plain, the fork's shape | Plain, best shape | x12 | x12 against plain's best | x11 |
|---:|---:|---:|---:|---:|---:|
| 1 | 9.65 | 9.42 | 11.04 | +1.38 (1.14×) | 45.0 |
| 2 | 9.97 | 9.68 | 13.55 | +3.58 (1.36×) | 50.5 |
| 3 | 11.03 | 11.00 | 16.82 | +5.80 (1.53×) | 53.3 |
| 4 | 11.80 | 11.44 | 19.43 | +7.63 (1.65×) | 57.7 |
| 5 | 15.88 | 13.57 | 21.35 | +5.48 (1.35×) | 62.3 |
| 6 | 19.07 | 16.15 | 22.83 | +3.76 (1.20×) | 66.2 |
| 8 | 23.42 | 18.02 | 36.80 | +13.39 (1.57×) | 76.8 |

**The 12-bit code is slower than plain BF16 at every token count on a P106:** +1.4 ms a step at one token instead of
the expected −2.2, 85 % of the plain kernel's best rate at one token and 74 % at two where the decision needed 90 %,
and 1.2–1.65× its time at 3–8 tokens. The decode's extra integer work does not hide behind the reads on five SMs, even at one token. x11 is 4–5×
slower. So Q621's repack at load time does not go ahead (C373 parked); the BF16 hyper-connections stay plain BF16 or
the Q8_0 retype. The plain kernel's own best shape is 0.03–5.4 ms a step faster than the fork's default at 1–8 tokens (2.3–5.4 ms
at 5–8 tokens).

## Files

| File | What |
|---|---|
| [entropy.md](entropy.md) | the table above, per class and file (also Huffman, per-row entropies, escapes, megabytes a token) |
| [tensors_q2_0.tsv](tensors_q2_0.tsv), [tensors_pmix_q2_fast.tsv](tensors_pmix_q2_fast.tsv) | per matrix: shape, zeros, entropies, Huffman, x12 and x11 escapes and bits, round trip, a hash of its BF16 values |
| [census_q2_0.tsv](census_q2_0.tsv), [census_pmix_q2_fast.tsv](census_pmix_q2_fast.tsv) | every F32, F16 and BF16 tensor of each file: shard, type, shape, bytes, offset |
| [bench_cpu_only.txt](bench_cpu_only.txt), [sample.lbb.md5](sample.lbb.md5) | the bench's CPU round trip of its sample, and the sample's md5 |
| [run/census.py](run/census.py) | the census of a split GGUF (standard library) |
| [run/analyze.sh](run/analyze.sh), [run/table.py](run/table.py) | census, `lossless_entropy` over each file, the bench's sample, the table |
| [run/rig_bench.sh](run/rig_bench.sh) | the timing on one P106: copies the bench and sample to the rig, checks their md5s, runs, collects, cleans up |
| [rig/lossless_rig.txt](rig/lossless_rig.txt), [rig/lossless.csv](rig/lossless.csv), [rig/lossless_summary.csv](rig/lossless_summary.csv) | C378: the bench on one P106: every class, code, shape and token count with its check, and the per-step sums |

## Commands

```bash
# build (Docker, CUDA 12.9 image; the CPU tool alone with ONLY=entropy)
DOCKER_ARGS="--cpu-shares=256" bench/mmvq_pascal/build_lossless.sh
# the entropy table and the sample (CPU, nice 19; ~1 min for both files)
WORK=$PWD OUT=$PWD/lossless_out FILES="q2_0=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  pmix_q2_fast=$MODELS/swift_q2_0_pmix_fast/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf" \
  bash docs/data/2026-11-04_lossless_bf16/run/analyze.sh
# the CPU round trip of the sample, any machine
bench/mmvq_pascal/build/lossless_bench --sample lossless_out/sample.lbb --cpu-only
# the timing on one P106 (~10 min)
RIG=rig BIN=bench/mmvq_pascal/build/lossless_bench SAMPLE=lossless_out/sample.lbb \
  MD5_BIN=<md5 of the binary> MD5_SAMPLE=f58658da4439cec5afd638b01889b3e1 OUT=lossless_rig \
  bash docs/data/2026-11-04_lossless_bf16/run/rig_bench.sh
```

On another GPU, `CUDA_FORCE_PTX_JIT=1 lossless_bench --sample … --check-only` checks the compute 6.1 code through the
PTX JIT.

## Provenance and licence

The files are Swift-Qwen3.8-Flash-Next GSQ-RCO Q2_0 as published (the n-gram table moved alone into shard 3 by
[scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py)) and `pmix-q2-fast`
([2026-11-04_pascal_mix](../2026-11-04_pascal_mix/README.md)). The tables here are statistics of their weights and
headers; no weights are redistributed (the bench's sample is rebuilt from the published file by `analyze.sh`). The
models' licences are their own.
