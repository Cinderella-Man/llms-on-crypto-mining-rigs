# Pascal kernels for IQ2_S and IQ3_XXS, round 1 (2026-10-27)

Miner Llama v3's first kernels for the 2- and 3-bit expert formats of Ornith-1.5-122B's files (registry row Q166):
the Pascal matrix-vector kernels of `mmvq-pascal` extended to IQ2_S and IQ3_XXS, on the fork's branch
`pascal-p106-v3-iqmmvq` (off `pascal-p106-v3`, the v2.0 release `37e717ff6`). The result is in
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
Round 2 times tuning variants (sign masks in registers, a register limit, rows per lane group, a persistent grid) with
`test-backend-ops perf`.

## Files

| File | What |
|---|---|
| `correctness.md` | test-backend-ops pass counts per operation and type, native and compute 6.1 code; sanitizer runs |
| `decode_ab.csv` | per line (v2.0, builds A and B, A with the new kernels off): fresh prompt tok/s, next-turn first token, decode at 16k, the two identity prompts' output hash (first 12 hex digits of MD5) and tok/s, fullest card's MiB |
| `kernel_profile_ud.csv` | per IQ launch group: layers, weights and bytes per token, ms per token off and on, speedup, GB/s and ps per weight |
| `decode_profile_ud_iq_on.txt`, `decode_profile_ud_iq_off.txt` | [scripts/prof_summary.py](../../../scripts/prof_summary.py) on the two profiled runs (build B, ~384 one-token steps each): per card, per first op, per weight type, per launch group |
| `registers.md` | registers, stack bytes and shared memory per kernel for compute 6.1, builds A and B |
| `run/build.sh` | the build (Docker, CUDA 12.9, compute 6.1 + sm_70) from a source copy of the branch |
| `run/correctness.sh` | the test-backend-ops and compute-sanitizer runs on one GPU |
| `run/rig_ab.sh` | install, probes and profiled runs on the rig, new kernels on and off, through [scripts/serve_probe.sh](../../../scripts/serve_probe.sh) |
| `run/tune_build.sh`, `run/tune_rig.sh` | round 2: tuning variants of test-backend-ops, timed with `test-backend-ops perf` on card 0 |

## How it was made

Branch `pascal-p106-v3-iqmmvq` of [Miner Llama](https://github.com/Cinderella-Man/llama.cpp): build A `6b880e344`,
build B `baf2e8ded` (A plus one change: a fused gate/up's two matrices staged one after the other), tuning switches
`9392a958b`, persistent grid `67098bb3b`. Built in `nvidia/cuda:12.9.2-devel-ubuntu24.04` for compute 6.1 (SASS and
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
```

The rig runs used slimmed tarballs (`llama-server` and `test-backend-ops` only) of the same builds; the profiled runs
used build B. Server logs and the raw profiler traces are not published; the tables summarize them. The probes'
prompts are CPython standard-library source (PSF). Ornith-1.5-122B is MIT-licensed by its makers. Tables and scripts
of this repository (MIT).
