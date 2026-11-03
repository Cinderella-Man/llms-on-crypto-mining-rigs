# The Pascal decode attention kernel without local memory (2026-11-04)

Registry row Q675. Miner Llama's one-token decode attention kernel for compute 6.1 (`fattn-pascal.cu`, Q151; the
"FAP" kernel), as v2.0 ships it behind `GGML_CUDA_FATTN_PASCAL=1`, keeps a 32-float array of every thread in local
memory. It is off by default because it decoded slower than the stock kernels on the rig (Flint K2h-r16 on nine
P106-090: tg128 36.8 against 40.3 tok/s, and 15.0 against 25.4 tok/s after a 32k-token context). This folder has the
register census that found the array, the change that keeps it in registers (fork branch
`pascal-p106-v3-b12076-fap`, commit `ff8e1ec24`), the checks that ran without a GPU, and the scripts for the GPU and
P106 runs that decide whether the kernel now beats the stock ones.

## What the census found

`-Xptxas -v` and `cuobjdump -res-usage` of `fattn-pascal.cu` compiled for sm_61 with the rig build's exact flags
([run/census.sh](run/census.sh)), and the instructions of its machine code:

| Kernel `flash_attn_ext_pascal_f16` (sm_61) | Registers | Stack frame | Spill stores / loads | Local stores (STL) | Local loads (LDL) | FFMA | FMUL |
|---|---|---|---|---|---|---|---|
| v2.0 (`98bbd6a2f`, the `-ladder` base) | 134 | 128 B | 0 / 0 | 169 | 336 | 1,060 | 330 |
| Both halves read before the selects only ([run/selects_only.diff](run/selects_only.diff)) | 128 | 128 B | 0 / 0 | 161 | 168 | 1,060 | 330 |
| **The change (`ff8e1ec24`)** | **168** | **0** | 0 / 0 | **0** | **0** | 1,060 | 330 |

Counts of instructions are static (in the code, not executed); full tables in [registers.tsv](registers.tsv) and
[sass_counts.tsv](sass_counts.tsv).

- **The 128 bytes are not a spill.** ptxas reports 0 spill bytes: the stack frame is `v[32]`, the 32 partial dot
  products each lane passes to the transpose-reduce (`fap_transpose_reduce`), which the compiler placed in local memory
  because it is indexed at run time.
- **Why it was indexed at run time:** the reduce looped over its five butterfly steps (`n` = 32, 16, 8, 4, 2) with an
  inner loop to `n/2`. The compiler unrolls inner loops first, when `n/2` is not yet a constant, so it unrolled that
  loop only 4 times and left a loop over `i`: that alone keeps `v` in local memory (the middle row). And
  `upper ? v[i] : v[i + n/2]`, a conditional on two array elements, compiled to predicated loads of one element or
  the other (`@P0 LDL` / `@!P0 LDL` in v2.0's code), a load from a lane-dependent address, which also needs `v` in
  memory. The change fixes both.
- **What it cost:** every 4-key score step (8 per 32-key tile) stored the 32 values to local memory and ran the
  butterfly there: about 100 local loads and stores per step, some 800 per tile and thread, against the tile's 2,048
  FMAs (scores and P·V). Local memory goes through the L1 cache and the load/store unit, with a latency of tens of
  cycles per access.
- **The change** ([commit](https://github.com/Cinderella-Man/llama.cpp/commit/ff8e1ec24)): each butterfly step is a
  template on its width (`fap_transpose_step<n>`), so its loop has a constant trip count and unrolls fully, and both
  halves are read into registers before the selects. `v` stays in registers: 168 registers, the cap that
  `__launch_bounds__(128, 3)` sets for 3 blocks per SM, with no spill, so occupancy is unchanged. Fixing only the
  selects (the middle row) still leaves the loop and the stack frame.
- **The arithmetic is unchanged:** the same 1,060 FFMA and 330 FMUL, so no multiply-add was contracted differently; the
  reduce does the same additions on the same values in the same order (`keep + received`). The extra FADD, SHFL and SEL
  instructions in the code are the loops written out, not extra work. Results should match v2.0's kernel bit for bit;
  [run/fap_gpu_check.sh](run/fap_gpu_check.sh) checks that on a GPU (step 4).
- **The prompt kernel next to it** (`fattn-tile-pascal.cu`, Q163, untouched, the census's control: the same numbers in
  both builds): 155–168 registers, no stack except one variant, head 128 with a Q8_0 cache and the split score layout
  (its default at head 128), which spills 40 bytes at the 168-register cap (registry Q676). Because ptxas counts only
  register spills, a census of every fork kernel for stack frames and local loads and stores is registry Q677.

## Which models the kernel serves

The kernel takes FLASH_ATTN_EXT calls of one token with head size 128, 8 query heads per KV head, F16 K and V and a
KV-head count divisible by 4: Flint's attention (4 KV heads of 8 query heads). Head-256 models with other groupings
decode through llama.cpp's stock TILE kernel on the P106, which this change does not touch:

| Model | Attention | Decode kernel on compute 6.1 |
|---|---|---|
| Flint | head 128, 4 KV heads × 8 query heads | this kernel with `GGML_CUDA_FATTN_PASCAL=1` (default 0: stock) |
| Qwen3.8-Flash-Next (Swift) | head 256, 2 KV heads × 12 | stock TILE (the GQA path; the fork's prompt kernel takes 9 or more tokens at a multiple of 8 query heads per KV head) |
| Ornith-1.5-122B | head 256, 2 KV heads × 16 | stock TILE |

The 45.95 ms of attention per token after 39,448 tokens in
[2026-10-27_iq_kernels](../2026-10-27_iq_kernels/README.md#attention-at-depth-registry-q210) (Ornith) ran in the stock
TILE kernel. A decode kernel of this design for head 256 at 12 and 16 query heads per KV head is registry Q702.

## Checks that ran (no GPU)

- **The census** before and after ([registers.tsv](registers.tsv), [sass_counts.tsv](sass_counts.tsv)), from
  [run/build.sh](run/build.sh).
- **The transpose-reduce on an emulated warp** ([run/reduce_check.cpp](run/reduce_check.cpp),
  [reduce_check.txt](reduce_check.txt)): v2.0's form and the new one on 20,000 random warps (640,000 results, inputs
  scaled from 2⁻¹² to 2¹¹): no result differs in any bit; every lane ends with the sum over the lanes of its value (the
  mapping the scores rely on: lane `l` holds key `4·j4 + l/8` of head `l%8`) within the rounding bound of 5 additions
  (worst 0.18 of it); a mutation (keep and send swapped in the last step) fails all 640,000.
- **The rig tarball builds** (`TESTS=1`, sm_61 + compute_61 PTX, no AVX).
- Not checked: anything on a GPU.

## Tomorrow's runs

1. **On a GPU of the quant machine, through the compute_61 PTX** ([run/fap_gpu_check.sh](run/fap_gpu_check.sh), ~1 h,
   correctness only): every `FLASH_ATTN_EXT` case with the kernel forced (`GGML_CUDA_FATTN_PASCAL=2`); faults 2 (sinks
   dropped) and 3 (each block's last key tile dropped) must fail the kernel's own cases, fault 1 (output × 1.01) stays
   inside the 5·10⁻⁴ NMSE limit as in v2.0; the own cases with the kernel off; then `llama-perplexity -ub 1` on Flint
   G3 with the kernel forced, logits of the `-ladder` build twice and of this build once, compared byte for byte.
   **Gate:** all cases pass, faults 2 and 3 fail cases, before = before and before = after.
2. **On one P106** ([run/fap_p106.sh](run/fap_p106.sh), ~30 min, rig idle): `test-backend-ops perf` of Flint's
   decode attention per layer call at 256 to 131,072 keys (1,024 / 8,192 / 32,768 among them), stock kernels and this
   kernel, `-ladder` build and this build, two passes alternating the four arms. **Read:** µs per call and GB/s of
   K and V at 1k / 8k / 32k for `after-fap` against `after-stock` (the decision) and against `before-fap` (the gain
   of the change). The stock arms are the same code in both builds, a control for drift.
3. **If the kernel beats the stock ones at 8k and beyond:** the agent check's long turns on Flint (UD-Q4_K_XL, nine
   cards), this build with `GGML_CUDA_FATTN_PASCAL=1` against `=0`, as replays of the deepest recorded traces
   ([bench/agent_check](../../../bench/agent_check/README.md#the-proxy-and-the-replay): `release_check.sh` with
   `REPLAY_LOG`, `REPLAY_TRACES`, one run per setting into the same `CLIENT_OUT`). The kernel and the stock ones sum
   in different orders, so replies can part after some tokens; ms per generated token at each turn's context compares
   the two settings even then. If it wins there, it becomes the default on compute 6.1 (Q675).

## Files

| File | What |
|---|---|
| `registers.tsv` | per source tree (`before` = `98bbd6a2f`, `selects` = it with `run/selects_only.diff`, `after` = `ff8e1ec24`) and kernel of `fattn-pascal.cu` and `fattn-tile-pascal.cu` for sm_61: registers, stack bytes, spill stores and loads, shared memory (ptxas); kernel names demangled to their templates |
| `sass_counts.tsv` | the same kernels: instructions in the sm_61 code, local stores and loads, FFMA, FMUL, FADD, SHFL, SEL, BRA, global and shared loads, shared stores |
| `reduce_check.txt` | the output of `run/reduce_check.cpp` |
| `run/build.sh` | the rig tarball, the census of the three source trees and the host check, on a build machine with Docker (`STEPS` picks them) |
| `run/selects_only.diff` | the middle arm: v2.0's reduce with both halves read before the selects, its loops unchanged |
| `run/census.sh` | the census of chosen CUDA files for chosen source trees: the rig build's compile command plus `-Xptxas -v`, `cuobjdump -res-usage` and the machine code; writes the two tables |
| `run/reduce_check.cpp` | the transpose-reduce, both forms, on an emulated warp |
| `run/fap_gpu_check.sh` | tomorrow's correctness run on a GPU newer than Pascal (step 1) |
| `run/fap_p106.sh` | tomorrow's timing on one P106 (step 2) |

## How it was made

Fork branch `pascal-p106-v3-b12076-fap` of [Miner Llama](https://github.com/Cinderella-Man/llama.cpp), one commit
(`ff8e1ec24`) on `pascal-p106-v3-b12076-ladder` (`98bbd6a2f`). Built in `nvidia/cuda:12.9.2-devel-ubuntu24.04` for
compute 6.1 (SASS and PTX), static, without AVX.

```bash
# the tarball, the census of the three trees and the host check, on a machine with Docker (CLIENT), in a folder WORK
# there whose llama.cpp clone ($WORK/src) has the fork as remote "fork"; REPO: a checkout of this repository there
ssh $CLIENT "WORK=$WORK REPO=$REPO LOCK=<shared lock file> JOBS=6 $REPO/docs/data/2026-11-04_fap_spills/run/build.sh"
# this run's tarball: llama.cpp-20261103-ff8e1ec24-pascal-p106-v3-b12076-fap.tar.gz, md5 d437df7da046d99c06245a6a9c99f653
# (a rebuild gives another md5: BUILDINFO.txt inside holds the build time)

# 1. correctness on GPUs there (G3 needs two 24 GB cards); BEFORE: the unpacked -ladder tarball (98bbd6a2f)
ssh $CLIENT "BEFORE=<ladder tree> AFTER=$WORK/llama.cpp-20261103-ff8e1ec24-pascal-p106-v3-b12076-fap \
  MODEL=<flint-q73-G3.gguf> TEXT=<wiki.test.raw> WORK=$WORK/gpu GPU=a,b MOUNTS='<-v for each drive>' \
  bash -s" < run/fap_gpu_check.sh

# 2. one P106 (both tarballs unpacked on the rig)
ssh $RIG "BEFORE=<ladder tree> AFTER=<fap tree> OUT=<dir> CARD=0 bash -s" < run/fap_p106.sh
```

Tables and scripts of this repository (MIT). No model output is published here.
