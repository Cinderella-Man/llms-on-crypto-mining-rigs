# Pascal kernels for IQ2_S and IQ3_XXS, rounds 1 and 2 (2026-10-27)

Miner Llama v3's first kernels for the 2- and 3-bit expert formats of Ornith-1.5-122B's files (registry row Q166):
the Pascal matrix-vector kernels of `mmvq-pascal` extended to IQ2_S and IQ3_XXS, on the fork's branch
`pascal-p106-v3-iqmmvq` (off `pascal-p106-v3`, the v2.0 release `37e717ff6`). Round 1 wrote and checked them;
round 2 timed 11 tuning variants on a P106 and made the fastest the default. The result is in
[the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md#what-v3-needs-for-ornith) and the registry row.

## What the kernels do

- One 256-weight super-block per lane, staged through shared memory like Q6_K's (both blocks start at 2-byte
  boundaries: 82 and 98 bytes), for 1 to 4 tokens, plain products, expert products (`MUL_MAT_ID`), and the fused expert
  gate/up with its GLU.
- Each block copies its type's grid (IQ2_S 8 KB, IQ3_XXS 1 KB) and a table of byte masks per sign pattern into shared
  memory. A weight's sign is applied as `sum(g * u) - 2 * sum((g & neg) * u)`: two dp4a and one AND per 4 weights,
  instead of llama.cpp's per-byte negation, which Pascal emulates.
- Scales multiply each 16- or 32-weight integer sum in FP32; llama.cpp truncates with integer division.

## Results

**Correctness** ([correctness.md](correctness.md)): every case of the new types and of Q6_K passes, natively on a V100
and as the compute 6.1 code a P106 runs; compute-sanitizer finds 0 errors.

**Decode on the rig** (nine P106-090, Unsloth's UD-Q2_K_XL, `-c 49152 -b 2048 -ub 512 -ts 8,6,5,5,5,4,5,6,6 -bs`,
[decode_ab.csv](decode_ab.csv)): the same binary with the new kernels on against off
(`GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs`):

| | Short prompts, tok/s | Decode at 16k, tok/s | Fresh 13.7k prompt, tok/s |
|---|---|---|---|
| New kernels (builds A, B) | 11.7, 11.3 / 12.2, 11.1 | 8.8 / 8.7 | 208 |
| Off | 10.8, 10.2 | 8.0 | 208 |

With the new kernels off the output is identical to v2.0's (same hashes); A and B give each other's output exactly.
About 73 % of this file's expert bytes are IQ2_S or IQ3_XXS; its other expert types still run on llama.cpp's kernels.

**Per kernel** (decode under `GGML_CUDA_PROFILE`, ms per token summed over the nine cards,
[kernel_profile_ud.csv](kernel_profile_ud.csv), [decode_profile_ud_iq_on.txt](decode_profile_ud_iq_on.txt),
[decode_profile_ud_iq_off.txt](decode_profile_ud_iq_off.txt)):

| Product | llama.cpp's kernels | New kernels | Speedup | New, GB/s |
|---|---|---|---|---|
| IQ3_XXS down, 25 layers | 14.14 ms | 8.11 ms | 1.74× | 29.7 |
| IQ2_S down, 7 layers | 3.57 ms | 2.00 ms | 1.79× | 28.2 |
| IQ2_S gate+up (fused), 18 layers | 10.55 ms | 8.97 ms | 1.18× | 32.4 |
| IQ3_XXS gate+up (fused), 14 layers | 9.11 ms | 7.46 ms | 1.22× | 36.2 |
| Whole token | 110.84 ms | 98.81 ms | 1.12× | |

The new kernels read 28–36 GB/s, a fifth of the ~150 GB/s v2.0's Q8_0 kernels reach on the same card. The fused
gate/up kernels gain least: they need 194–255 registers per thread, so 2 blocks fit per SM ([registers.md](registers.md)).

## Round 2: tuning on a P106

Eleven variants of four compile-time switches, each a `test-backend-ops` build, timed on one P106 (card 0) with
llama.cpp's kernels as the reference ([tune_round2.csv](tune_round2.csv), [tune_round2.md](tune_round2.md)):
`lut<0|1>`, the sign masks from a shared-memory table (1) or computed in registers (0); `mb<n>`, registers kept for
n blocks per SM (`__launch_bounds__`); `r<n>`, rows per lane group; `-p<n>`, a persistent grid of n blocks per SM. The
cases are Ornith's shapes: the fused expert gate/up (8 of 64 experts of 1024 × 3072), the expert down (8 of 64 of
3072 × 1024) and a dense 8192 × 3072 product, at 1, 2 and 4 tokens, every call reading 8 MB or more (the card's L2 is
1.5 MB). One token, picoseconds per weight read:

| Variant | Gate/up IQ2_S | Gate/up IQ3_XXS | Down IQ2_S | Down IQ3_XXS | UD-Q2_K_XL, ms per token |
|---|---|---|---|---|---|
| **lut0-mb3-r1 (the new default)** | **6.67** | **7.28** | 9.19 | **7.07** | **17.25** |
| lut0-mb4-r1 | 7.55 | 8.34 | **7.48** | 7.05 | 18.47 |
| lut0-mb1-r1 | 8.40 | 8.79 | 9.14 | 7.04 | 19.85 |
| lut1-mb3-r1 | 8.08 | 8.73 | 9.51 | 10.26 | 21.60 |
| lut1-mb1-r1 (round 1) | 8.77 | 9.07 | 9.44 | 10.75 | 22.76 |
| lut0-mb3-r1-p3 | 9.95 | 10.76 | 11.04 | 8.10 | 23.64 |
| lut1-mb1-r2 | 11.66 | 12.72 | 11.31 | 9.62 | 27.57 |
| llama.cpp's kernels | 9.97 | 11.07 | 17.36 | 18.66 | 31.63 |

The last column adds up the one-token times of a decoded token's IQ2_S and IQ3_XXS expert products in Unsloth's
UD-Q2_K_XL (gate/up: 18 IQ2_S and 14 IQ3_XXS layers; down: 7 and 25), one card's kernel after another.

- **Sign masks in registers win** (1.4–1.5× on IQ3_XXS down and dense, 3–9 % on IQ2_S), and with the register limit
  for three blocks per SM the fused gate/up kernels run at 168 registers instead of 194–251: 1.25–1.3× faster than
  round 1, 48–53 GB/s. Fork commit `ea55c7865` makes `lut0-mb3-r1` the default.
- **Rows per lane group (r2) and the persistent grid lose** on the products that matter; four blocks per SM (mb4) helps
  IQ2_S down and dense, and slows the gate/up.
- **At 2–4 tokens the expert kernels are slower than llama.cpp's** (0.5–0.8× its speed for gate/up and down; a 2-token
  expert down costs 3–4× one token where llama.cpp's costs ~1×): MTP verification needs its own kernel work. The dense
  product stays 1.1–1.5× faster at 2–4 tokens.
- **Correct on the P106:** the winning build passes `test-backend-ops` against the CPU on every IQ2_S / IQ3_XXS case
  small enough for the rig's CPU to quantize ([correctness.md](correctness.md)).
- **How the fused gate/up is timed:** `test-backend-ops perf` repeats only a graph's last node, for this case the GLU,
  so a second pass runs each case under `GGML_CUDA_PROFILE` and takes the median per evaluation of the launch groups
  that start with `MUL_MAT_ID` ([run/tune_rig_fused.sh](run/tune_rig_fused.sh)). Weights are random bytes
  (`GGML_TEST_RAW_QUANT=1`, fork commit `82661fcad`): quantizing 400 M random values to IQ2_S per case takes the rig's
  2-core Celeron many minutes, and perf mode does not check values.
- **Same output as round 1 on the rig:** the default build `ea55c7865` gives build B's two identity hashes on
  UD-Q2_K_XL (`3e9d790a7c1e`, `1cbdc446d9ad`; [decode_ab.csv](decode_ab.csv)), at 11.2 and 13.5 tok/s against B's
  12.2 and 11.1 (single short runs).

## Attention at depth (registry Q210)

Build B on UD-Q2_K_XL (`-c 49152 -ub 512 -ts 8,6,5,5,5,4,5,6,6`, f16 KV) under `GGML_CUDA_PROFILE`: a fresh
39,448-token prompt, then 128 decoded tokens after it ([run/attn_depth.sh](run/attn_depth.sh),
[decode_profile_ud_39k.txt](decode_profile_ud_39k.txt)), against the same build's short-context profile
([decode_profile_ud_iq_on.txt](decode_profile_ud_iq_on.txt); both in [attn_depth.csv](attn_depth.csv)):

| GPU time per decoded token, summed over the nine cards | Short context | After 39,448 tokens |
|---|---|---|
| Attention (FLASH_ATTN_EXT, 12 layers) | 1.11 ms (1.1 %) | **45.95 ms (32.2 %)** |
| Expert products (MUL_MAT_ID) | 41.94 ms | 40.69 ms |
| Dense products (MUL_MAT) | 36.25 ms | 36.61 ms |
| Gated DeltaNet (36 layers) | 4.65 ms | 4.44 ms |
| All | 98.81 ms | 142.87 ms |

- **Attention grows by ~1.17 ms per 1,000 tokens of context and is a third of a token at 39k.** Each of the 12
  attention layers reads 81 MB of f16 KV per token there (2 KV heads of 256 for 32 query heads, 2 KiB per position) in
  ~3.8 ms: ~21 GB/s, a ninth of the card's memory bandwidth. Everything else stays where it was.
- **For the agent check it is small:** its turns end at 4.6k tokens (median) to 7.5k (deepest), so attention is ~6 %
  of a token there. Long agent sessions reach 30–50k, where a decode-attention kernel that reads the cache once per KV
  head at ~100 GB/s would take ~10 ms instead of 46 (est. a 1.3× faster token at 40k); q8_0 KV halves the bytes again.
- The run's own speeds, with the profiler's per-group timing on: the prompt read at 45.1 tok/s (877 s), decode at
  6.62 tok/s. The second request restored a context checkpoint instead of reading the prompt again.

## Files

| File | What |
|---|---|
| `correctness.md` | test-backend-ops pass counts per operation and type, native and compute 6.1 code on a V100; sanitizer runs; round 2's default on a P106 |
| `decode_ab.csv` | per line (v2.0, builds A and B, A with the new kernels off, round 2's default with the identity prompts only): fresh prompt tok/s, next-turn first token, decode at 16k, the two identity prompts' output hash (first 12 hex digits of MD5) and tok/s, fullest card's MiB |
| `kernel_profile_ud.csv` | per IQ launch group: layers, weights and bytes per token, ms per token off and on, speedup, GB/s and ps per weight |
| `decode_profile_ud_iq_on.txt`, `decode_profile_ud_iq_off.txt` | [scripts/prof_summary.py](../../../scripts/prof_summary.py) on the two profiled runs (build B, ~384 one-token steps each): per card, per first op, per weight type, per launch group |
| `registers.md` | registers, stack bytes and shared memory per kernel for compute 6.1, builds A and B; round 2's variants |
| `run/build.sh` | the build (Docker, CUDA 12.9, compute 6.1 + sm_70) from a source copy of the branch |
| `run/correctness.sh` | the test-backend-ops and compute-sanitizer runs on one GPU |
| `run/rig_ab.sh` | install, probes and profiled runs on the rig, new kernels on and off, through [scripts/serve_probe.sh](../../../scripts/serve_probe.sh) |
| `tune_round2.csv` | round 2, per variant (and `llama.cpp`), product, shape, type and token count: microseconds per call; GB/s and picoseconds per weight where the weights read per call are known (one token, or a dense product). The `4096x14336` rows are llama.cpp's own perf cases |
| `tune_round2.md` | round 2 per variant: the UD-Q2_K_XL token's IQ expert products in ms, and every case's speedup over llama.cpp's kernels |
| `run/tune_build.sh`, `run/tune_rig.sh` | round 2: tuning variants of test-backend-ops, timed with `test-backend-ops perf` on card 0 (expert down, dense) |
| `run/tune_rig_fused.sh` | round 2: the fused gate/up per variant, under `GGML_CUDA_PROFILE` |
| `run/tune_table.py` | round 2: the two rig logs to `tune_round2.csv` and `tune_round2.md` |
| `attn_depth.csv` | GPU ms per decoded token by first op, short context against 39,448 tokens (build B, UD-Q2_K_XL) |
| `decode_profile_ud_39k.txt` | [scripts/prof_summary.py](../../../scripts/prof_summary.py) on the 39k run (~129 one-token steps): per card, per first op, per weight type, per launch group |
| `run/attn_depth.sh` | attention's share of a decoded token after a ~40k-token prompt, under `GGML_CUDA_PROFILE` |

## How it was made

Branch `pascal-p106-v3-iqmmvq` of [Miner Llama](https://github.com/Cinderella-Man/llama.cpp): build A `6b880e344`,
build B `baf2e8ded` (A plus one change: a fused gate/up's two matrices staged one after the other), tuning switches
`9392a958b`, persistent grid `67098bb3b`, `GGML_TEST_RAW_QUANT` `82661fcad` (the round-2 variants were built from it),
round 2's defaults `ea55c7865`. Built in `nvidia/cuda:12.9.2-devel-ubuntu24.04` for compute 6.1 (SASS and
PTX) and sm_70, static, without AVX. The model: Unsloth's
[UD-Q2_K_XL](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF) (revision `745bd742473a`, 44,533,952,768 bytes,
SHA-256 `93fb32cfb1c31b34df426b4a41d961bdaffcddbbd217d850262136cf3689ba66`), on a USB 3 stick on the rig.

```bash
# source copy and build, on a machine with Docker (CLIENT), in a folder WORK there
git -C <checkout of the branch> archive HEAD | ssh $CLIENT "mkdir -p $WORK/src && tar -x -C $WORK/src"
git -C <checkout> rev-parse HEAD | ssh $CLIENT "cat > $WORK/src/COMMIT"
ssh $CLIENT "WORK=$WORK NAME=llama.cpp-20261027-6b880e344-pascal-p106-v3-iqmmvq bash -s" < run/build.sh

# correctness on one GPU there
ssh $CLIENT "WORK=$WORK NAME=llama.cpp-20261027-6b880e344-pascal-p106-v3-iqmmvq GPUS=3 bash -s" < run/correctness.sh

# the rig: install both builds, probes on/off, profiled runs on/off (the md5s the builds printed)
RIG=rig CLIENT=$CLIENT WORK=$WORK BUILDS=<rig builds folder> MODEL=<rig path>/Ornith-1.5-122B-UD-Q2_K_XL.gguf \
  OUT=iq_ab TS=8,6,5,5,5,4,5,6,6 SPECS="<build A>:<md5> <build B>:<md5>" bash run/rig_ab.sh

# round 2: the variants (test-backend-ops only) from a source copy of 82661fcad, then both rig passes and the tables
V='lut1-mb1-r1 lut0-mb1-r1 lut1-mb3-r1 lut0-mb3-r1 lut1-mb1-r2 lut0-mb3-r2 lut0-mb4-r1 lut1-mb3-r1-p3 lut0-mb3-r1-p3 lut1-mb1-r1-p2 lut1-mb3-r2-p3'
ssh $CLIENT "WORK=$WORK VARIANTS='$V' bash -s" < run/tune_build.sh
KEEP=1 RIG=rig CLIENT=$CLIENT WORK=$WORK BUILDS=<rig builds folder> MD5=<md5 printed> VARIANTS="$V" \
  bash run/tune_rig.sh > tune_rig.log
RIG=rig BUILDS=<rig builds folder> VARIANTS="$V" bash run/tune_rig_fused.sh > tune_rig_fused.log
sed 's/\x1b\[[0-9;]*m//g' tune_rig.log > tune_rig.txt
python3 -I run/tune_table.py tune_rig.txt tune_rig_fused.log tune_round2

# the winner's correctness on the P106 (its test-backend-ops; cases the rig's CPU quantizes in minutes)
ssh rig "cd <rig builds folder>/iq-tune/tune/lut0-mb3-r1 && for op in MUL_MAT MUL_MAT_ID MUL_MAT_VEC_FUSION; do
  CUDA_VISIBLE_DEVICES=0 ./test-backend-ops -b CUDA0 -o \$op \
  -p '^(?!.*(k=3072|k=4096|k=2048|k=14336|n_mats=32|n_mats=64|n_mats=128|m=3072|m=4097|m=8192|m=4096)).*=(iq2_s|iq3_xxs),'; done"
```

The rig runs used slimmed tarballs (`llama-server` and `test-backend-ops` only) of the same builds; the profiled runs
used build B. Server logs and the raw profiler traces are not published; the tables summarize them. The probes'
prompts are CPython standard-library source (PSF). Ornith-1.5-122B is MIT-licensed by its makers. Tables and scripts
of this repository (MIT).
