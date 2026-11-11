# F32 products of 4–8 tokens on `mul_mat_vec_f` (C754, Q1330, 2026-11-11)

What this is: a new switch and its checks on a build machine's CPU, with no GPU. llama.cpp sends F32 weights to its
matrix-vector kernel `mul_mat_vec_f` only up to 3 tokens on NVIDIA cards; on compute 6.1 nothing else takes them at
4–8 tokens, so cuBLAS does. In a four-token verify pass of Qwen3.8-Flash-Next that is the shared expert's gate
`ffn_gate_inp_shexp` (2,560 → 1, 47 calls) and the router `ffn_gate_inp` (2,560 → 512, 47 calls). R41 timed the gate
at 12.78 µs a call at 4 tokens against 3.26 at one (0.601 against 0.153 ms a pass), and the router at 2.35 against
1.79 ms a pass ([C731](../2026-11-11_c731_shared_expert/README.md),
[R41](../2026-11-10_release_profile/README.md)). cuBLAS also adds in its own order, so the router's scores, which pick
the experts, are not one-token decoding's.
Registry row [Q1330](../../../quants/experiments.md); fork branch `pascal-p106-v4-b12246-c754` (`bca4cc4c6`).

**The answer:**
- **`GGML_CUDA_MMVF_F32_MT=1`** (off by default) sends an F32 weight matrix times 4–8 tokens to `mul_mat_vec_f` on
  compute 6.1, the kernel that already takes it at 1–3 tokens. `=2` does the same on any NVIDIA GPU (for tests).
- **No kernel changes:** `mul_mat_vec_f` already keeps each token's one-token sums. It picks its block size from the
  row width alone, and each token's sum is its own chain: a thread's FMAs over its pairs of weights, then the same
  butterflies. The 4–8-token instantiations are compiled into every build already; the switch only routes to them.
- **Each weight is read once a call** at any token count: one load per pair of weights, and one per pair of each
  token's activations.
- **Checked on the CPU:** an emulation of the kernel's code gives the one-token sums on every output at 2–8 tokens
  (0 of 8,960 at the gate's shape, 0 of 17,920 at the router's, 0 of 143,360 over every even row width of 2 to
  4,096). In the machine code every product is an FFMA and every FADD a butterfly's. Every kernel of the base build
  is the same in the new build, none is new, ctest and `test-llama-archs` give the base's results, and the rig's
  replay set is recorded with a fault run that must fail.
- **Not yet on a P106:** bit identity on the card, the speed against cuBLAS and the served verify pass are R99's.

## The change

`ggml/src/ggml-cuda/mmvf.cu` (the switch, `ggml_cuda_mul_mat_vec_f_f32_mt`), `mmvf.cuh` (its line),
`ggml-cuda.cu` (one call in `ggml_cuda_mul_mat`, after the 1–3-token route), `tests/test-backend-ops.cpp` (cases).

- **Where it applies.** With the switch on, after llama.cpp's own routes have declined a product, it takes:
  - F32 weights and F32 activations and output;
  - 4 to 8 columns (tokens);
  - a 2-D weight matrix outside the graph's compute buffers (a weight, not an activation), times 2-D activations
    whose columns are contiguous and 8-byte aligned;
  - a matrix that `mul_mat_vec_f` takes at one token (an even row width at even strides).

  Everything else goes on as before: cuBLAS for the rest, and nothing changes at 1–3 tokens.
- **`GGML_CUDA_MMVF_F32_MT_FAULT=1`** (checks only) runs the products the switch takes over the first half of each
  row, which every check of them must catch. Without the switch it reaches nothing.
- **Tests:** `test-backend-ops` gets F32 products of 1–8 tokens at the gate's and the router's shapes, at row widths
  that take `mul_mat_vec_f`'s other block sizes (64: 32 threads, 100: 64, 300: 160), a 6,144-wide row, an odd width
  (left to cuBLAS) and rows that are views of wider rows (an even stride, and an odd one, left to cuBLAS). Its perf list
  gets the gate's and the router's shapes at 1–8 tokens.
- With the switch at 0 the dispatch is the base's line for line.

## The sums

[run/order_emul.c](run/order_emul.c) emulates `mul_mat_vec_f` for F32 weights step by step as its code runs. Each
thread takes its pairs of weights in order and adds both of a pair's products, token by token, into that token's sum
with one FMA each. Then come the warp's butterfly (xor 16, 8, 4, 2, 1), the warps' sums into a shared buffer of 32
(zeros past the block's warps) and warp 0's butterfly over it. The block size comes from the host's rule, which reads
only the row width: 256 threads at K = 2,560 and 6,144, 32 at 64, 64 at 100 and 160 at 300. The emulation sums FP32
weights (normal, standard deviation 0.02) against FP32 activations (standard normal), with `fmaf` and with denormals
flushed as the kernel's FTZ instructions do. It compares every output's bits with the same columns' one-token launches
([order_emul.tsv](order_emul.tsv)):

| Rows | Width K, rows | Tokens | Against the one-token launches | Outputs | Differ | Median difference of those, ULP | Largest difference / sum of \|terms\| |
|---|---|---:|---|---:|---:|---:|---:|
| the shared expert's gate (256 draws of a row) | 2,560, 1 | 2, 3, 4, 5, 6, 7, 8 | `mul_mat_vec_f` at that token count | 512, 768, 1,024, 1,280, 1,536, 1,792, 2,048 | **0 at each** | – | 0 |
| the router | 2,560, 512 | 2, 3, 4, 5, 6, 7, 8 | `mul_mat_vec_f` at that token count | 1,024, 1,536, 2,048, 2,560, 3,072, 3,584, 4,096 | **0 at each** | – | 0 |
| every even row width | 2 to 4,096, 2 each | 2, 3, 4, 5, 6, 7, 8 | `mul_mat_vec_f` at that token count | 8,192, 12,288, 16,384, 20,480, 24,576, 28,672, 32,768 | **0 at each** | – | 0 |
| the gate | 2,560, 1 | 4 | a sequential order (the control) | 1,024 | 993 | 10 | 1.70e-07 |
| the router | 2,560, 512 | 4 | a sequential order (the control) | 2,048 | 1,976 | 9 | 2.20e-07 |
| the gate | 2,560, 1 | 4 | the one-token order at 128 threads (the control) | 1,024 | 811 | 2 | 1.48e-08 |
| the router | 2,560, 512 | 4 | the one-token order at 128 threads (the control) | 2,048 | 1,572 | 2 | 1.53e-08 |
| the gate | 2,560, 1 | 4 | the fault (the control) | 1,024 | 1,024 | 26,036,353 | 1.42e-01 |
| the router | 2,560, 512 | 4 | the fault (the control) | 2,048 | 2,048 | 27,595,353 | 1.41e-01 |

- **Every multi-token output repeats the one-token sums:** at the gate's and the router's shapes at 2–8 tokens, and at
  every even row width of 2 to 4,096 (each of the rule's block sizes and every tail of a thread's pairs).
- **The order matters at these widths:** a sequential sum, one order cuBLAS's GEMM can take, differs from the
  one-token sums on 993 of 1,024 gate outputs and 1,976 of 2,048 router outputs. Even the same kernel at 128 threads
  instead of 256 differs on 811 and 1,572. The fault differs on every output.
- **The emulation follows the code; the machine code below shows the compiler kept that order.** The rig's
  `multitoken_bitid f32` checks it on a P106 (R99).

## The machine code

[sass_f32.tsv](sass_f32.tsv) is cuobjdump's sm_61 code of the build's `mmvf.cu.o` (md5
`bedbf202a1534c1311cbaed31e06f00c`), read by [run/sass_f32.sh](run/sass_f32.sh) for all 88 F32 instantiations of
`mul_mat_vec_f`: 1–8 tokens at each of the 8 block sizes, and the one-token kernels for fusion and for MUL_MAT_ID at
each. At 256 threads, the block size of both Flash-Next shapes:

| Tokens | Registers | Stack | FFMA | FMUL | FADD (the butterflies': 10 a token) | SHFL | Loops: LDG | Loops: FFMA |
|---:|---:|---:|---:|---:|---:|---:|---|---|
| 1 | 30 | 0 | 58 | 0 | 10 | 10 | 2, 32 | 2, 32 |
| 2 | 39 | 0 | 116 | 0 | 20 | 20 | 3, 48 | 4, 64 |
| 3 | 40 | 0 | 30 | 0 | 30 | 30 | 4, 16 | 6, 24 |
| 4 | 48 | 0 | 40 | 0 | 40 | 40 | 5, 20 | 8, 32 |
| 5 | 64 | 0 | 50 | 0 | 50 | 50 | 6, 24 | 10, 40 |
| 6 | 56 | 0 | 36 | 0 | 60 | 60 | 14 | 24 |
| 7 | 48 | 0 | 42 | 0 | 70 | 70 | 16, 0 | 28, 0 |
| 8 | 56 | 0 | 48 | 0 | 80 | 80 | 18 | 32 |

- **Every product is an FFMA, and every FADD is a butterfly's:** no FMUL in any unfused instantiation, and exactly 5
  FADD and 5 SHFL per token and warp reduction (10 a token above 32 threads, 5 at 32). Nothing else adds into a sum,
  and FMAs on one accumulator cannot be reordered, so each token's sum is the chain the code writes.
- **One load per pair of weights and per token's pair of activations:** in every loop of every instantiation, FFMA ×
  (1 + T) = 2T × LDG (at 4 tokens a pass of 4 pairs: 4 weight loads, 16 activation loads, 32 FMAs). Each weight is
  loaded once a call at every token count.
- **No stack and no spills** in any of the 88.
- The fused one-token kernel (a bias or a GLU gate in its epilogue) has the same loops, one without a gate and one with
  (3 loads and 4 FMAs a pair). Beside the gate sum's own butterflies, its epilogue adds 11 FMUL and 9 FADD.

## Checks on the build machine

[run/chain.sh](run/chain.sh), CPU only, every step in a container without GPUs at a low CPU share; results in
[summary.tsv](summary.tsv):

| Check | Result |
|---|---|
| emulation | `mul_mat_vec_f` at 2–8 tokens gives the one-token sums on every output (the table above) |
| rig tarball | `llama.cpp-20261111-bca4cc4c6-pascal-p106-v4-b12246-c754.tar.gz`, 656,296,922 bytes, md5 `b1c59870910cf1bf86661c9fb12ab11c`; `TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |
| the kernel's code | the table above: in all 88 F32 instantiations no stack; in the 72 unfused ones no FMUL and only the butterflies' FADDs; in every loop of all 88 one load per pair of weights and per token's pair of activations, two FMAs per token |
| every kernel | `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops` against the base's tarball (`5b329df16`), function by function: every function the same in all four listings, none new (the change is host code) |
| ctest | the branch's static CPU build (every target): 70 of 70 pass (the 7 tests that need a download, the network or torch not run) |
| `test-llama-archs` | the base's static CPU build writes 146 architectures' logits, the branch's compares: 0 differ; 151 of 151 test cases pass in both |
| `multitoken_bitid` | mode `f32`, linked against the rig build's tree: md5 `42408e4a3573c4f862b8b6023e9b52ec` (it runs on the rig) |
| replay set `f32x` | 115 cases (F32's 2-D `MUL_MAT` at 1–8 tokens: this row's 72 and 43 of llama.cpp's own), 6 runs ([replay_runs.tsv](replay_runs.tsv)), recorded on the CPU with the rig build's `test-backend-ops`; tar 369,397,760 bytes, md5 `7ed93810cd3109c7d2cebb48c80b9ee7`. The fault runs select the 37 cases of 4–8 tokens at this row's even widths and at 48 rows of 2,560, all of which the switch takes |
| CPU replay | the set's 115 cases replayed on the CPU with the rig build: 115 of 115 pass (the set's files are whole) |

The branch changes only host code and `test-backend-ops`' list of cases, so the GPU code is the base's, and the CPU
builds compile the base's code: ctest and `test-llama-archs` show the CPU side untouched. The switch off on a card
rests on the dispatch: with it at 0 the new function returns at its first line, before any other test.

## What the rig run must show (R99)

[run/rig_c754.sh](run/rig_c754.sh) on a P106-090 rig, the rig idle, with the branch's tarball, the set and the
harness:

1. **replay:** the set `f32x` through `scripts/replay_checks.sh`; every run ends `ok`, and `f32x-fault` and
   `f32x-fault2` fail every one of their cases (the products the switch takes).
2. **bitid:** `multitoken_bitid f32` with the switch at 1 and at 2 gives `0 with differences`; off (the control),
   cuBLAS takes 4–8 tokens and those products differ.
3. **perf:** `test-backend-ops perf` at 2,560 → 1 and 2,560 → 512, 1–8 tokens, arms `off` and `on`, two passes. The
   verdict at 4–8 tokens, each pass by value: gain, negligible or loss against `off` (cuBLAS, not exact). At 1–3
   tokens both arms run `mul_mat_vec_f` and give the spread.
4. **served:** R79's profile of a served verify pass (R49's
   [verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh)) on the Flash-Next preset with the MTP
   draft, `off on on off`: the router's and the shared expert's time per verify pass and the step time of the warm
   loads.

## Files

| File | What |
|---|---|
| [order_emul.tsv](order_emul.tsv) | the emulation: outputs compared and differing, per shape, token count and pair of orders |
| [sass_f32.tsv](sass_f32.tsv) | every F32 instantiation of `mul_mat_vec_f`: registers, stack, FFMA, FMUL, FADD, SHFL, each loop's loads and FMAs |
| [summary.tsv](summary.tsv) | the build machine's checks, one line each |
| [replay_runs.tsv](replay_runs.tsv) | the set's runs: switches, filter, expected verdict (`runs-ptx.tsv`'s, for a newer card, run `=2` instead of `=1`) |
| [replay_cases.tsv](replay_cases.tsv) | the set's cases: id and case string |
| [run/order_emul.c](run/order_emul.c) | the emulation |
| [run/sass_f32.sh](run/sass_f32.sh) | reads cuobjdump's output into `sass_f32.tsv` |
| [run/chain.sh](run/chain.sh) | the build machine's checks |
| [run/rig_c754.sh](run/rig_c754.sh) | the rig run (R99) |

Also changed for this row: `scripts/record_checks.sh` (suite `f32x`), `bench/mmvq_pascal/verify/multitoken_bitid.cpp`
(mode `f32`), [docs/switches.md](../../switches.md).

## Commands

On an x86-64 build machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image, no GPU (`BASE` is the
base build's rig tarball, of `pascal-p106-v4-b12246-c748` `5b329df16`; `SRC` holds a llama.cpp clone with the fork as
remote `fork`):

```bash
WORK=$WORK PUB=$PWD BASE=$BASE BASE_REF=5b329df16 LOCK=$WORK/../lock JOBS=8 DOCKER_ARGS='--label pascal-quants=1' \
    SEED_SRC=$SRC nice -n 19 bash docs/data/2026-11-11_c754_f32_mmvf/run/chain.sh
# the emulation alone, on any x86-64 machine:
gcc -O2 -ffp-contract=off -o order_emul docs/data/2026-11-11_c754_f32_mmvf/run/order_emul.c -lm && ./order_emul
```

On the rig (the branch's tarball unpacked, the set and the harness copied; `rig_c754.sh`'s header names every
variable):

```bash
BIN=<tarball>/bin OUT=<out> SET=<set>/f32x REPLAY=scripts/replay_checks.sh BITID=<multitoken_bitid> \
    MODEL=<Flash-Next GGUF> DRAFT=<MTP draft GGUF> PRESET=release/flash-next-v3/settings.env \
    RIG_SERVER=bench/agent_check/rig_server.sh REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
    VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
    setsid nohup bash docs/data/2026-11-11_c754_f32_mmvf/run/rig_c754.sh > console.txt 2>&1 < /dev/null &
```

## Provenance and licence

Code: Miner Llama (`github.com/Cinderella-Man/miner-llama`, MIT, llama.cpp's licence), branch
`pascal-p106-v4-b12246-c754` at `bca4cc4c6` (one commit on `pascal-p106-v4-b12246-c748` `5b329df16`). The emulation's
and the tests' inputs are generated (fixed seeds), not model weights. Everything here is this repository's licence.
