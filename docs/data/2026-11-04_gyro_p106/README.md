# Gyro's trellis decode on a P106: agentionai's CUDA kernels built for compute 6.1 (2026-11-04)

Registry rows Q895 (this test) and Q241 (trellis formats on Pascal, parked until a decode under ~5 ps per weight) in
[quants/experiments.md](../../../quants/experiments.md). agentionai's Gyro files store Qwen3.8-Flash-Next's routed
experts in TQ2_T, a trellis code of the QTIP family at 2.125 bits per weight; on the reference chain's agent text
Gyro-Coder (256 of 512 experts per layer, 19.3 GiB on the GPUs) scores KL divergence 0.2815, level with Swift Q2_0
(0.285) at about half the bytes ([gyro KLD](../2026-11-04_gyro_kld/README.md)). Whether that turns into speed on the
rig depends on how fast a P106 decodes TQ2_T. This folder builds agentionai's own CUDA kernels for compute 6.1, checks
them against the CPU and times them on one P106 beside Q2_0 at Flash-Next's expert shapes.

## The format and the kernels

A TQ2_T block is 128 weights in 34 bytes: an FP16 scale and 32 bytes, one per step of 4 weights. Step t's 16-bit state
is `qs[(t + 31) % 32] << 8 | qs[t]` (a tail-biting path; any byte string is a valid one), and its four weights come from
one multiply: `x = (s & 0x7fff) * 0x9e3779b1`, then the codebook points `x >> 21` and `(x >> 10) & 2047` of a fixed
2,048-point 2-D codebook, each giving two weights, times the scale.

agentionai's llama.cpp fork ([gitlab.com/agentionai-group/llama.cpp](https://gitlab.com/agentionai-group/llama.cpp),
`main` at `9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f`, 2026-10-12, tag `main-b11377-9aa51ce4f`) has CUDA kernels for
TQ2_T and its siblings (TQK6, TQK7, TQP5–TQP7) in `ggml/src/ggml-cuda/tq.cuh`, `mmvq.cu`, `mmq.cuh` and `convert.cu`.
On CUDA the codebook is an int8 copy with one global scale (`tq-lut-i8.cuh`, 4 KB, relative RMS error 0.0085 of the
codebook's standard deviation), so a step is one 32-bit multiply, two 2-byte reads from shared memory and one `dp4a`
against the int8 activations. The CPU reference keeps the FP16 codebook, so CUDA and CPU agree within
test-backend-ops' tolerance, not bit for bit.

**It compiles for compute 6.1 as it is.** No trellis path uses tensor cores or `cp.async`; the fork carries MMQ
configurations for Pascal with and without `dp4a` (`mmq-config-pascal-dp4a.cuh`, `mmq-config-pascal-older.cuh`), and
its host code picks kernels by the highest compiled architecture (`ggml_cuda_highest_compiled_arch`), so a newer card
running the 6.1 PTX takes the Pascal paths. The paths a Flash-Next token takes:

| Tokens | Routed experts (`MUL_MAT_ID`) | Dense products (`MUL_MAT`) |
|---|---|---|
| 1 | `mul_mat_vec_q` with the int8 codebook in shared memory: a 128-thread block per row for long rows (gate/up, 2,560 wide), 4 rows per block for short rows (down, 640 wide) and 16 rows per block (`GGML_CUDA_TQ_MMVQ_WIDE`, on by default before Blackwell) | the same kernel |
| 2–8 | `mul_mat_vec_q_moe`, the multi-token MoE kernel (gate/up fused with the GLU when the graph allows) | `mul_mat_vec_q` at 2–8 columns |
| 9 and more | MMQ with a trellis tile loader (`dp4a`), or dequantization and cuBLAS (`GGML_CUDA_TQ_MMQ=0`) | MMQ |

The fork's multi-row layout for long trellis rows at one token (`GGML_CUDA_TQ_MMVQ_ROWS`: one row per warp, the block
sharing its codebook load; RTX 3090 gate/up −41 % in its comment) is chosen only from Turing on, by the card's own
compute capability. On a P106 a 2,560-wide gate/up row therefore runs as one row per 128-thread block, whose 16 lanes of
blocks per pass cover the row's 20 blocks in two passes, the second three-quarters idle. `run/p106rows.diff` lets
`GGML_CUDA_TQ_MMVQ_ROWS=1` turn the layout on before Turing as well (unset or `0`: the fork's choice, unchanged), so the
P106 run times both.

## The build

`scripts/build_llamacpp_docker.sh` (the rig's build: CUDA 12.9, compute 6.1 code and PTX, no AVX, Ubuntu 24.04
userland, ggml linked statically) on the fork at the pinned commit with `TESTS=1` and two patches:

| Patch | What |
|---|---|
| [run/p106perf.diff](run/p106perf.diff) | test-backend-ops perf cases sized for one 6 GB card on a 4 GB host: Flash-Next's routed experts with 128 matrices, 10 used, at 1–8 tokens (gate/up 2,560 → 640 with the activations broadcast, down 640 → 2,560), TQ2_T beside Q2_0, the shapes [Miner Llama's Q2_0 experts are timed at](../2026-11-04_kcurve_p106/README.md); and Q8_0, the type of every dense matrix in the Gyro files, at the attention, recurrent and shared-expert shapes (2,560 → 6,144 / 10,240 / 12,288, 6,144 → 2,560, 320 ↔ 10,240) at 1, 2, 4 and 8 tokens |
| [run/p106rows.diff](run/p106rows.diff) | `GGML_CUDA_TQ_MMVQ_ROWS=1` applies the trellis multi-row layout before Turing (above) |

The fork's tree fetched from GitLab failed with "pack has 1705 unresolved deltas" on 2026-11-04, so the build ran on a
clone already holding the pinned commit. Its binaries hold AVX2 code in one place only: the CPU dot product of the
fork's `rocmfp4` Q4_0 variant, compiled with `__attribute__((target("avx2")))` and called only when
`__builtin_cpu_supports("avx2")` is true, so they run on the rig's Celeron, which has no AVX.

## Correctness

Through the compute 6.1 PTX on an RTX 3090 (`run/ptx_check.sh`: test-backend-ops against the CPU on the fork's own
test list). The newer card's host code picks the multi-row layout for long trellis rows at one token, which a P106
does not, so the one-token cases also ran with `GGML_CUDA_TQ_MMVQ_ROWS=0`, the layout a P106 takes:

| Op | Types | Setting | Cases passed |
|---|---|---|---:|
| `MUL_MAT` | all six trellis types | defaults | 486 / 486 |
| `MUL_MAT_ID` | TQ2_T | defaults | 111 / 111 |
| `GET_ROWS` | TQ2_T | defaults | 8 / 8 |
| `MUL_MAT`, one token | TQ2_T | `GGML_CUDA_TQ_MMVQ_ROWS=0` | 23 / 23 |
| `MUL_MAT_ID`, one token | TQ2_T | `GGML_CUDA_TQ_MMVQ_ROWS=0` | 15 / 15 |

The `MUL_MAT` row ran on a first build without `p106rows.diff` and the rotation's perf cases; the other rows on the
final build, whose kernels differ only when `GGML_CUDA_TQ_MMVQ_ROWS=1` is set on a card before Turing.
`MUL_MAT_ID` of the five other trellis types (TQK6, TQK7, TQP5–TQP7) was stopped after 86 cases, all passing: the CPU
encoder makes each type's ~110 cases take ~10 minutes, and no Gyro file in use stores its experts in them.
`MUL_MAT_ID_FUSION` and `MUL_MAT_VEC_FUSION` have no trellis cases in the fork's list. On the P106 itself: the rig run's
`check` part (below).

## The run-time Hadamard rotation

The Gyro files' routed experts are Hadamard-rotated: their headers carry `prism.hadamard.*` keys (`run/gguf_keys.py`)
naming all 144 expert tensors, with 128-wide blocks of a normalized Sylvester–Walsh–Hadamard transform on the input
axis and explicit signs for the two input widths (640 and 2,560). The fork undoes the rotation on the activations
before each product: a sign multiply and a 128 × 128 `MUL_MAT` marked as a Hadamard matrix, which the CUDA backend runs
as its FWHT kernel (`fwht.cu`), once for the gate/up input (20 blocks a token, shared by gate and up) and once for the
down input (10 experts × 640 = 50 blocks a token). That is four small launches per layer beside the expert products,
so the P106 run times the rotation as well (`MUL_MAT_HADAMARD` at 20, 40, 80, 160 and 50, 100, 200, 400 blocks).

## The P106 run

`run/p106_tq.sh` on one idle P106-090, about 10 minutes, in three parts: a logged quiet check (the card empty, no
llama or test process, at least 2 GiB of host memory available), `check` (the TQ2_T `MUL_MAT` cases and the TQ2_T
`MUL_MAT_ID` cases of up to 8 tokens against the CPU, then the one-token cases with `GGML_CUDA_TQ_MMVQ_ROWS=1`), and
`perf` (the P106 cases above and the rotation; then TQ2_T at one token with `GGML_CUDA_TQ_MMVQ_ROWS=1` and with
`GGML_CUDA_TQ_MMVQ_WIDE=0`). On the quality machine's CPU the perf part took 2 minutes, most of it preparing the
weights; the rig's two cores take longer, est. 15–20 minutes for the whole run. The trellis encoder takes minutes per expert tensor on the rig's two cores, so perf fills
each tensor with 4,096 encoded blocks repeated (`GGML_TEST_QUANT_INIT_BLOCKS`, the fork's switch for this); since any
byte string is a valid trellis path, the decode does the same work on any data. test-backend-ops draws new expert ids
before every timed repetition, so each call reads its 10 experts (4.35 MB in TQ2_T) from memory, not from the 1.5 MB L2.

`run/perf_table.py` turns the perf output into one row per case: microseconds per call, the weights the call reads
(the expected number of distinct experts of n tokens), picoseconds per weight read, GB/s of weight bytes and
picoseconds per weight-token, with Miner Llama's Q2_0 kernels at the same cases from the
[k-curve](../2026-11-04_kcurve_p106/README.md) (`perf_experts.tsv`, its `on` column) as arm `miner-llama`.

The line Q241 set is ~5 ps per weight. TQ2_T's bandwidth floor on a P106 is ~1.5 ps per weight (2.125 bits at
173 GB/s); Miner Llama's Q2_0 one-token expert kernel reads at 135 GB/s, 2.1 ps per weight in its bench and 3.3–3.9 ps
per weight inside ggml, where the call includes the activations' quantization and the launch.

## Results

On one P106-090 of the first rig (card 0, idle; 2026-11-04, C335r), [perf.tsv](perf.tsv): ps per weight read at
Flash-Next's routed expert shapes (10 of 128 experts per token, random routing; inside ggml, the activations'
quantization and the launch included), against Miner Llama's Q2_0 kernels on the same card:

| Product | TQ2_T, 1 token | TQ2_T, 1 token, its best layout | TQ2_T, 2–8 tokens | Q2_0 (Miner Llama), 1 token | Q2_0 (Miner Llama), 2–8 tokens |
|---|---:|---:|---:|---:|---:|
| routed gate/up (2,560 → 640) | 23.8 ps (11.2 GB/s) | 10.0 (`rows1`) | 8.4–10.5 (25–32 GB/s) | 3.3 (84 GB/s) | 2.8–3.6 |
| routed down (640 → 2,560) | 37.8 ps (7.0 GB/s) | 22.3 (`wide0`) | 15.2–19.2 (14–18 GB/s) | 3.9 (72 GB/s) | 5.4–9.6 |

**TQ2_T misses the ~5 ps line at every token count**: 3–10× Miner Llama's Q2_0 per weight read at one token, 2–3× at
2–8. The run-time Hadamard rotation adds 6.7 µs a call. Correctness on the card: `MUL_MAT` 81/81, `MUL_MAT_ID` 53/53, the
P106 one-token layout 38/38. Priced in a token at one token, the routed experts would take est. ~4× Q2_0's ~6 ms, about
+20 ms (~25 %), for Gyro-M's agent KLD of 0.148 against Swift Q2_0's 0.285. That trade decides Q908 (a TQ2_T file of
Swift-1.5, the fine-tune the rig runs, from a calibrated encoder of our own, since the Gyro files are quantized from the
base Qwen3.8-Flash-Next); Q241 stays parked as a kernel matter until a P106 decode reaches the line.

## Files here

| File | What |
|---|---|
| `run/p106perf.diff`, `run/p106rows.diff` | the two patches on the fork (above) |
| `run/p106_tq.sh` | the P106 run: quiet check, correctness, perf |
| `run/ptx_check.sh` | the correctness run through the 6.1 PTX on a newer card, in Docker, and a smoke run of `p106_tq.sh` |
| `run/perf_table.py` | the perf output as a table (standard library) |
| `run/gguf_keys.py` | a GGUF's metadata keys and values from its header (standard library), e.g. the `prism.hadamard.` keys |

## Commands

On a build machine with Docker, `W` a work folder holding a clone of the fork with the pinned commit in `W/src` and the
two patches in `W/patches/`:

```bash
WORK=$W TESTS=1 JOBS=8 PATCHES="patches/p106perf.diff patches/p106rows.diff" DOCKER_ARGS="--cpu-shares=256" \
  bash scripts/build_llamacpp_docker.sh 9aa51ce4fdd3435a91531ffe16a17414ccc6dc6f
B=$W/llama.cpp-<date>-9aa51ce4f-p106perf-p106rows/bin
BIN=$B OUT=$W/ptx GPU=<one GPU id> MOUNT=$W OPS=MUL_MAT bash run/ptx_check.sh
BIN=$B OUT=$W/ptx2 GPU=<id> MOUNT=$W OPS="MUL_MAT_ID GET_ROWS" TYPES=tq2_t SMOKE=0 bash run/ptx_check.sh
BIN=$B OUT=$W/ptx3 GPU=<id> MOUNT=$W OPS="MUL_MAT MUL_MAT_ID" TYPES=tq2_t PARAMS=,n=1, \
  TEST_ENV=GGML_CUDA_TQ_MMVQ_ROWS=0 SMOKE=0 bash run/ptx_check.sh
python3 -I run/gguf_keys.py <Gyro GGUF> prism.
```

On the rig, the tarball unpacked in `/opt/llama-builds/`, nothing else running:

```bash
BIN=/opt/llama-builds/llama.cpp-<date>-9aa51ce4f-p106perf-p106rows/bin OUT=<dir> CARD=0 \
  setsid nohup bash run/p106_tq.sh > p106_tq.out 2>&1 < /dev/null &
python3 -I run/perf_table.py --ours ../2026-11-04_kcurve_p106/perf_experts.tsv \
  default=<dir>/perf_experts.txt dense=<dir>/perf_dense.txt rows1=<dir>/perf_rows1.txt wide0=<dir>/perf_wide0.txt > perf.tsv
```
