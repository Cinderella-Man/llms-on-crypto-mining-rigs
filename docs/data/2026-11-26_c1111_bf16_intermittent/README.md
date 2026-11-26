# C1111 (Q1844): the BF16 product that failed once

On 2026-11-25 the landing of eight branches ([2026-11-25_land_v5_batch3](../2026-11-25_land_v5_batch3/README.md))
replayed `test-backend-ops` `MUL_MAT` from the kept record ([2026-11-24_c1023_suite_record](../2026-11-24_c1023_suite_record/README.md))
on an RTX 3090 with `--cpu-check 25`, and one case failed once:

```
[MUL_MAT] ERR = 0.499536458 > 0.000500000   MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[8,1],nr=[4,1],per=[0,2,1,3],...): FAIL
```

It passed in every other run (11 later replays of that build, 12 of the base build, 700 runs of the case alone). This
folder finds the cause. Registry: [Q1844](../../../quants/experiments.md); the fix is Q1890, a side finding Q1891.

## The answer

**The BF16 case ran the CUDA graph that had been captured for its F16 sibling.** The CUDA backend keeps captured
graphs keyed by the first node's host address, and replays one while every node and its sources' data pointers, sizes
and strides stay the same. It does not compare the sources' types (upstream llama.cpp b12674 has the same check).
`--cpu-check` runs the F16 case twice (live, then replayed), so its nodes are captured. When the BF16 case that follows
lands at the same host and card addresses (`test-backend-ops` cases often do, depending on the heap), its product node
passes the check and the F16 kernels run on the BF16 data.

- **The size fits:** the F16 kernels on the BF16 data give an error of 0.4983 in `test-backend-ops`' own measure
  (simulated), against 0.4995 in the failure.
- **Reproduced:** a set of 12 F16 and BF16 sibling cases, every case checked live, fails on the base build in 19 of
  20 runs, always the same BF16 case right after its F16 sibling, always ERR 0.505182503. And in the landing's own way
  (the case's neighbourhood replayed with `--cpu-check 25` on the head, a load on the other card) 2 of 150 runs failed,
  one of them on the landing's case with its very error, 0.499536458.
- **Graphs are the cause:** with `GGML_CUDA_DISABLE_GRAPHS=1` the same set passes 20 of 20 and the neighbourhood 100
  of 100, and Nsight Systems shows the failing BF16 case launching the F16 kernels (the F16 GEMM and conversions 10 times
  instead of 8, the BF16 ones 6 times instead of 8).
- **No fault in the fork's kernels:** on the RTX 3090 cuBLAS computes this case, and on a P106 llama.cpp's own
  `mul_mat_vec_f`; the fork's BF16 kernel never takes it. memcheck, racecheck and synccheck find nothing on the whole
  `MUL_MAT` replay or the case's neighbourhood.
- **A P106 can hit it in the same checks**, since the fork captures graphs on compute 6.1 too and its F16 and BF16
  kernels differ there as well. A model's own graphs keep their types, so inference is not affected (est.).

The fix is code row C1157 (Q1890): the sources' types in the graph's properties. Until it lands, an F16 or BF16 case
that fails with an error near 0.5 right after its sibling's live check is this, and a rerun with
`GGML_CUDA_DISABLE_GRAPHS=1` shows it.

## What computes the case

The rig build (`TESTS=1`, compute 6.1 code only) ran on the RTX 3090 through the driver's PTX JIT, with the fork's
matrix-vector kernels forced at a P106's shapes (`scripts/pascal_forced.sh`'s `PASCAL_FORCED_MMV`), as in the landing.
`run/c1111.sh paths` ran the case and its F16 sibling alone under Nsight Systems (`kernels.tsv`, runs `alone-*`):

| Case | Kernels on the RTX 3090 (one launch each) |
|---|---|
| BF16 (the failing one) | `convert_unary_cont_vec4<float, __nv_bfloat16>` (the activations to BF16), `k_compute_batched_ptrs` (a pointer per batch), cuBLAS `gemvx::kernel<int, int, __nv_bfloat16, float, float, float, ...>` (32 batched matrix-vector products, F32 out) |
| F16 sibling | `convert_unary_cont_vec4<float, __half>`, `k_compute_batched_ptrs`, CUTLASS `cutlass_75_wmma_tensorop_h161616gemm_32x32_32x2_tn_align1`, `convert_unary_cont_vec4<__half, float>` |

The case never reaches the fork's BF16 kernel. `mul_mat_vec_bf16_pascal` takes 2-D weights only
(`ggml_cuda_mmvf_pascal_supports` asks for `src0->ne[2] == 1`), and this case has 8 weight matrices shared by 32
activation columns (`bs=[8,1]`, `nr=[4,1]`), permuted, so the forced switches do not touch it. `ggml_cuda_mul_mat` sends
it to cuBLAS on the 3090: `ggml_cuda_should_use_mmvf` says no for BF16 on a card with BF16 tensor cores when the weights
are batched and larger than 512 rows, and `ggml_cuda_mul_mat_cublas` takes BF16 as its compute type there
(`cublasGemmBatchedEx` because of the broadcast, F32 output written straight into the result).

On a P106 the same routing (compute capability 6.1, no BF16 tensor cores: `bf16_mma_hardware_available` false) returns
`ne11 <= MMVF_MAX_BATCH_SIZE`, so llama.cpp's own `mul_mat_vec_f` computes the case: one kernel, no conversion, no
pointer array, no cuBLAS. The F16 sibling also runs `mul_mat_vec_f` there, with the other element type, so a replayed
F16 graph would be just as wrong.

## The size of the error

`test-backend-ops` measures `sum((out - ref)^2) / sum(out^2)`: its `nmse(a, b)` is `mse(a, b) / mse(a, 0)`, called with
the card's values first, so the error is normalized by the card's output, not by the reference. `run/nmse_sim.py`
(plain Python, seeded) draws the test's inputs (uniform in [-1, 1], the weights rounded to BF16) and computes what a few
faults would report (`nmse_sim.txt`):

| Fault | Error |
|---|---:|
| the F16 sibling's kernels run on the BF16 data (its bits read as F16) | 0.4983 |
| channels 16 to 31 of the output left at zero | 0.9567 |
| rows 528 to 1056 of every channel left at zero | 0.9986 |
| channels 16 to 31 left at the replay's restored output (uniform in [-1, 1]) | 0.9569 |
| the landing's failure | 0.4995 |
| tonight's reproduction (another shape of the same pair, `bs=[4,1]`) | 0.5052 |

Read as F16, a BF16 value in [-1, 1] becomes about ±0.9 to ±1.9 with its sign kept, so every output keeps a strong
correlation with the right one and the error sits near 0.5; a missing half of the output would read near 1.

## The CUDA graph cache

Each node of a replayed or live-checked case runs alone (`ggml_graph_view`), so each node is its own graph, keyed by its
host address. A key's graph is captured on the second call with unchanged properties and replayed after that while the
properties stay the same. The properties are the node's whole `ggml_tensor` (with its type) and each source's data
pointer, sizes and strides, not each source's type.

- **Consecutive cases reuse the same keys.** In the 384-case neighbourhood replay with `GGML_CUDA_GRAPH_DEBUG=1`
  (`paths.tsv`) the log shows 183 warm-ups completed and 182 reset: each case the check runs live has its nodes
  captured, and the next case lands on the same keys with other shapes, which resets them.
- **An F16 case and its BF16 sibling look the same to the check** when the addresses also match: the same sizes and
  strides (both types are 2 bytes), the same output type (F32). Only the weight node and its permuted view carry the
  type, and those two reset; the product node replays.
- **Whether the addresses match depends on the heap.** `run/c1111.sh siblings` (`siblings.tsv`) ran three sets of F16
  cases each followed by its BF16 sibling, with `--cpu-check 25` and `1`, on both builds. One run failed: the base
  build's 12-case set with every case checked live, on `MUL_MAT(type_a=bf16,...,m=1057,n=1,k=128,bs=[4,1],nr=[1,1],per=[0,2,1,3])`,
  ERR 0.505182503 in its live check and in its replay alike. Its graph log shows that case capturing 2 of its 9 node
  graphs and resetting the same 2, where every other case captured all 9: two nodes changed (by their properties the
  weight and its permuted view, the only nodes that carry the type), and the other 7, the product among them, replayed
  the F16 case's graphs.
- **Repeated** (`run/c1111.sh again`, `again.tsv`): that set, 20 runs per build and arm, one process each:

  | Build | CUDA graphs | Runs without a failing case | The failure |
  |---|---|---:|---|
  | base `eee615cdb` | on (default) | 1 of 20 | the same BF16 case, ERR 0.505182503, in each of the 19 |
  | base `eee615cdb` | `GGML_CUDA_DISABLE_GRAPHS=1` | 20 of 20 | none |
  | head `2d839da1d` | on (default) | 20 of 20 | none |
  | head `2d839da1d` | `GGML_CUDA_DISABLE_GRAPHS=1` | 20 of 20 | none |

  The two builds' binaries differ, so their heaps put the cases at different addresses (est.); on the head this set
  never lined up, while the landing's whole replay lined up once for the `bs=[8,1]` pair.
- **The kernels** (`run/c1111.sh kernels`, `kernels.tsv`, runs `graphs-sib-cc1` and `nographs-sib-cc1`): the same set
  under Nsight Systems with each graph's kernels traced, base build:

  | Kernel | Launches, graphs on (fails) | Launches, graphs off (passes) |
  |---|---:|---:|
  | CUTLASS F16 GEMM `h161616gemm` | 10 | 8 |
  | `convert_unary_cont_vec4<float, __half>` and `<__half, float>` | 10 and 10 | 8 and 8 |
  | cuBLAS `gemvx<..., __nv_bfloat16, ...>` | 6 | 8 |
  | `convert_unary_cont_vec4<float, __nv_bfloat16>` | 6 | 8 |

  The failing BF16 case's two products (live check and replay) ran the F16 kernels instead of its own.

## The replay's restore and output copy

Read from `tests/test-backend-ops.cpp` at `2d839da1d` (`rr_restore`, `rr_replay_compare`, `test_case::eval`):

- The replay restores every tensor with its own memory, in context order: the BF16 weights (stored as
  `[128, 8, 1057, 1]`), the activations (`[128, 32, 1, 1]`), the output and the sentinels. Each is checked against the
  record's type, shape and size, the file against its md5, and written with `ggml_backend_tensor_set`, a copy the host
  waits for. The permuted views have no memory of their own, so nothing about the permutation or the broadcast reaches
  the restore.
- The replay computes the graph node by node with `ggml_backend_graph_compute`, which waits for the card's stream, then
  reads the whole output with `ggml_backend_tensor_get`, another copy the host waits for, and compares it with the
  stored reference.
- The restore and the copy are sound for this shape; the wrong values came from the kernels the replayed graph ran.

## Sanitizers and repeats

`run/c1111.sh sanitize` (`sanitize.tsv`): `compute-sanitizer` around `test-backend-ops` on the neighbourhood (the 384
F16, BF16 and F32 cases with m and k 1056/1057 against 128/129, n = 1, in the suite's order) with `--cpu-check 25` (the
failing run's way) and `--cpu-check 1` (every case live, then replayed), and on the whole `MUL_MAT` replay with
`--cpu-check 25`. Each run passed every case.

| Tool | Neighbourhood, base and head, cc25 and cc1 | Whole replay, head (5,458 cases) |
|---|---|---|
| memcheck | 0 errors | 0 errors |
| racecheck (`--racecheck-report all`) | 0 hazards | 0 hazards |
| initcheck | 0 errors | 4,122 reads of never-written memory, all in quantized product kernels (below); base the same, 4,122 |
| synccheck | 0 errors | 0 errors |

memcheck's first pass counted the backend's own handled `cudaGraphExecUpdate` refusals (an error return it expects
and answers by building the executable graph anew) as errors; the rows above pass `--report-api-errors no`, which leaves
the memory checks alone.

**initcheck's reports** (`initcheck_kernels.tsv`, `initcheck_cases.tsv`; every report printed, `--show-backtrace no`):
4,119 of the 4,122 name a kernel, all `mul_mat_vec_q` (3,272), `mul_mat_q` (768) and the k-column kernel (79), in 148
quantized cases (most: `q4_0`, `q8_0` and `mxfp4` at m = k = 2880, n = 32; `iq3_s` and `q6_K` with batched shapes), the
same counts on both builds. None is in an F16, BF16 or F32 case, nor in cuBLAS. Every one of those cases passes and its
`--cpu-check` reference is identical, so the values read never reach a result that is compared; which memory they are is
Q1891. A first initcheck pass with 200 reports printed counted 20,631 on the head.

**The failing command again and again** (`full.tsv`): the whole `MUL_MAT` replay with `--cpu-check 25` on the head, one
process each, on one card while the other card ran `MUL_MAT` and `FLASH_ATTN_EXT` live (`load.tsv`, every round
passed), as the landing's run had beside it:

| Runs | Cases | Without a failing case |
|---|---|---:|
| head, the record as it is now | 5,458 | 2 of 2 |
| head, the landing's case list (`head5125`: the record without the 333 cases it lacked then), second card | 5,125 | 40 of 40 |
| the same on the first card, beside the load (`head5125g0`) | 5,125 | 30 of 30 |

With the landing's 11 whole replays with `--cpu-check 25` (6 of the head, 5 of the base), that is one failure in 83.

**The neighbourhood again and again** (`repeat.tsv`): its 384-case replay with `--cpu-check 25` on the head, one process
each, the load on the other card:

| CUDA graphs | Runs without a failing case | The failures |
|---|---:|---|
| on (default) | 148 of 150 | run 14: `MUL_MAT(type_a=bf16,...,m=1056,n=1,k=129,bs=[8,1],nr=[4,1],per=[0,2,1,3])`, ERR 0.497229637; run 93: the landing's case, ERR 0.499536458 |
| `GGML_CUDA_DISABLE_GRAPHS=1` | 100 of 100 | none |

Run 93 repeats the landing's failure to the last printed digit (the same recorded inputs through the same F16 kernels),
and both failures follow the landing's pattern: the F16 sibling checked live, one other case (its `k_v` neighbour), then
the BF16 case.

## Commands

On a machine with two NVIDIA cards and Docker (the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image), the builds' `bin`
folders and the kept record readable at the same paths inside the containers:

```bash
export W=<work folder> REC=<record>/mul_mat MOUNTS='<folders holding the builds and the record>' \
    DOCKER_ARGS='--label <your label> --cpu-shares=256' NSYS=<nsys binary>
BUILDS="head=<2d839da1d>/bin base=<eee615cdb>/bin" GPU=0 bash run/c1111.sh paths
BUILDS="head=<2d839da1d>/bin" GPU=0 bash run/c1111.sh sanitize          # and BUILDS="base=..." GPU=1 beside it
BUILDS="..." SAN_OPTS='--report-api-errors no' SAN_TAG=-noapi TOOLS=memcheck SMODES='cc25 cc1' bash run/c1111.sh sanitize
BUILDS="head=..." SAN_OPTS='--report-api-errors no' SAN_TAG=-noapi TOOLS='memcheck initcheck synccheck racecheck' SMODES=full25 bash run/c1111.sh sanitize
BUILDS="head=... base=..." SAN_OPTS='--report-api-errors no --show-backtrace no' SAN_TAG=-noapi-all PRINT_LIMIT=0 TOOLS=initcheck SMODES=full25 bash run/c1111.sh sanitize
BUILDS="head=..." LOAD_GPU=0 bash run/c1111.sh load &                    # the load, until 'stop'
BUILDS="head5125=..." REC=<record view without the landing's 333 new cases> GPU=1 N_FULL=40 bash run/c1111.sh full
BUILDS="head5125g0=..." REC=<the same view> GPU=0 N_FULL=30 NAME=c1111b bash run/c1111.sh full   # beside the load
BUILDS="head=..." GPU=1 N_REPEAT=150 bash run/c1111.sh repeat
BUILDS="head-nographs=..." GPU=0 N_REPEAT=100 REPEAT_ENV=GGML_CUDA_DISABLE_GRAPHS=1 NAME=c1111n bash run/c1111.sh repeat
bash run/c1111.sh stop
BUILDS="head=... base=..." GPU=0 bash run/c1111.sh siblings
BUILDS="base=... head=..." GPU=0 AGAIN_SET=sib AGAIN_CC=1 N_AGAIN=20 AGAIN_TAG=graphs bash run/c1111.sh again
BUILDS="base=... head=..." GPU=0 AGAIN_SET=sib AGAIN_CC=1 N_AGAIN=20 AGAIN_TAG=nographs AGAIN_ENV=GGML_CUDA_DISABLE_GRAPHS=1 bash run/c1111.sh again
BUILDS="base=... head=..." GPU=0 bash run/c1111.sh kernels
python3 -I run/nmse_sim.py > nmse_sim.txt
python3 -I run/tables.py "$W" .
```

The record view for the `head5125` runs links the record's files of the 5,125 cases the landing's failing replay found
(its replay log's rows other than `NOREC`), so the cases run in that run's order with its gaps.

## Files

| File | What |
|---|---|
| `paths.tsv` | per build: the neighbourhood replayed with the graph log (verdicts, warm-ups), the BF16 case and its F16 sibling alone under Nsight Systems |
| `kernels.tsv` | the kernels each Nsight Systems run launched: the two cases alone, the sibling set with and without graphs (both builds) |
| `kernels_runs.tsv` | the verdicts of those sibling-set runs |
| `siblings.tsv` | the three sibling sets, `--cpu-check 25` and `1`, both builds: verdicts, graph warm-ups completed and reset |
| `again.tsv` | the 12-case sibling set 20 times per build, with and without CUDA graphs |
| `sanitize.tsv` | every sanitizer run: build, tool, mode, verdicts, the sanitizer's summary, seconds |
| `initcheck_kernels.tsv`, `initcheck_cases.tsv` | initcheck's reports on the whole replay per kernel and per case (est. attribution) |
| `full.tsv`, `repeat.tsv`, `load.tsv` | the whole replays, the neighbourhood repeats and the load's rounds |
| `nmse_sim.txt` | `run/nmse_sim.py`'s output |
| `run/c1111.sh` | the driver (steps `paths`, `sanitize`, `load`, `repeat`, `full`, `siblings`, `again`, `kernels`, `stop`) |
| `run/nmse_sim.py` | the error under a few faults, in `test-backend-ops`' measure |
| `run/tables.py` | `kernels.tsv` and the initcheck tables from the logs |

## Provenance and licence

Builds: the rig tarballs of `pascal-p106-v5` `2d839da1d` (head, `test-backend-ops` md5 `65048f96fa48b8722c7ae9f03638b6b0`)
and `eee615cdb` (base, md5 `7e6731c4d42a98cfd6a90cf4bdeb0cc5`), as in the landing. Record: C1023's, extended by the
landing (`mul_mat`, 5,458 cases for the head). Machine: a build machine with RTX 3090 cards (CUDA 12.9 container,
driver 615.71.09, the landing's driver too), two cards used, correctness only, nothing timed. The tables summarize
`test-backend-ops`, `compute-sanitizer` and Nsight Systems output; this folder's text, tables and scripts are under the
repository's licence.
