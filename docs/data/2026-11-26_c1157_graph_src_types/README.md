# C1157 (Q1890): the CUDA graph cache tells sources of different types apart

The CUDA backend reuses a recorded CUDA graph when a new evaluation looks the same as the last one, and it did not
look at the type of the numbers a node reads. [C1111](../2026-11-26_c1111_bf16_intermittent/README.md) found that this
let a BF16 `test-backend-ops` case, landing at the addresses of the F16 case checked just before it, run the F16
kernels on BF16 numbers (an error near 0.5). This folder holds the fix, a test that shows the fault on demand, and the
checks that the fix changes nothing else. Registry: [Q1890](../../../quants/experiments.md). Correctness only:
nothing here is timed. Run on 2026-11-26 on a build machine's RTX 3090s (compute 8.6), which run the rig build's
compute 6.1 code through the driver's JIT.

## Result in plain words

- **The fault is real and the fix removes it.** A new test changes only a matrix's type under a captured graph:
  without the fix every evaluation after the change is wrong (20 of 32 right, in all three runs), with the fix all 32
  are right in every run, with the graphs on or off and with the fork's Pascal paths forced. On C1111's own failing
  build, its 12-case sibling set failed in 17 of 20 runs; with the fix applied to that build it passed 20 of 20, and
  its graph log shows the fix catching the very case that failed.
- **Nothing else changes.** Every kernel is the same machine code (9,508 kernel names, 10,075 copies). The kept
  card-suite records replay with every case passing on both builds and the processor's references identical; the
  pipe check gives the same verdicts and the same plain-decoding hashes on both builds; and Qwen3.8-Flash-Next with its
  v5 preset on two cards gives the base's tokens, drafts and memory exactly.
- **Not landed and not offered upstream yet:** both wait for the owner's word. Upstream llama.cpp has the same gap;
  the pull request's text is ready in [upstream_offer.md](upstream_offer.md).

## The change

Fork branch [`pascal-p106-v5-graph-src-types`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-graph-src-types),
two commits on `pascal-p106-v5` (`2d839da1d`):

| Commit | What |
|---|---|
| `582cfbfe3` | `tests/test-cuda-graph-src-types.cpp`: the fault on demand (below); a tool run by hand on a card, not a ctest; built by `scripts/build_llamacpp_docker.sh` with `TESTS=1` as one of the check tools |
| `9862e2734` | the fix: `node_properties` keeps each source's type (`ggml/src/ggml-cuda/graph.cuh`, 3 lines) and `ggml_cuda_graph_update_required` fills it (`ggml-cuda.cu`, 1 line) |

`ggml_cuda_graph_update_required` decides whether a captured graph can be replayed. For every node it keeps the whole
`ggml_tensor` and, for each source, the data address, the sizes (`ne`) and the strides (`nb`), and it replays while
all of them match the last evaluation. Two types with the same element or block size give a tensor of one shape the
same strides: F16 and BF16 (2 bytes an element), Q4_0 and IQ4_NL (18 bytes a block of 32). A source of the other type
at the same address then matched, and the graph captured for the first type ran its kernels on the second type's
bytes. With the type kept, such an evaluation counts as a change: the warm-up starts again and the graph is captured
anew. No switch: it is a correctness fix, and the checks below found no cost.

- **Upstream has the same gap.** llama.cpp `master` on 2026-11-25 (`a8ad690ca`, after tag b12756) keeps the same
  properties, and no upstream commit compares the sources' types. The SYCL backend's `ggml_sycl_graph_update_required`
  keeps the same set too (not changed here). With the fork's `GGML_CUDA_GRAPHS_PATCH` (off by default) the node-kind
  check already compared source types; both now agree.
- **Inference does not meet it:** a model's graphs keep their tensors' types from one evaluation to the next. The
  Flash-Next load below gives the base's tokens.
- [upstream.diff](upstream.diff) is the fix against upstream `master`; [upstream_offer.md](upstream_offer.md) is the
  pull request's text.

## The fault on demand: test-cuda-graph-src-types

The test evaluates a 512 × 128 matrix times a vector on one card until the graph is captured and replayed, then
changes only the matrix's type (the same tensor: the same address, shape and strides; its data written anew in the
new type), evaluates three times, changes it back, evaluates twice, and checks each of the eight evaluations against
the host's product of the values the type holds (NMSE limit 5e-4, `test-backend-ops`' for `MUL_MAT`). Four orders: F16
then BF16, BF16 then F16, Q4_0 then IQ4_NL, IQ4_NL then Q4_0. Three runs per build and setting on an RTX 3090
([test.tsv](test.tsv)), with the graph log on (`GGML_CUDA_GRAPH_DEBUG=1`). "Test only" is the test commit alone (the
base with the test); "forced" adds the fork's compute-6.1-only paths at a P106's launch shapes
(`scripts/pascal_forced.sh`), the kernels a P106 runs:

| Build | Setting | Runs passing | Evaluations right, each run | Graph warm-ups completed / reset, each run |
|---|---|---:|---|---|
| test only | graphs on | 0 of 3 | 20 of 32 | 4 / 1 |
| test only | `GGML_CUDA_DISABLE_GRAPHS=1` | 3 of 3 | 32 of 32 | 0 / 0 |
| test only | forced | 0 of 3 | 20 of 32 | 4 / 0 |
| branch | graphs on | 3 of 3 | 32 of 32 | 12 / 9 |
| branch | `GGML_CUDA_DISABLE_GRAPHS=1` | 3 of 3 | 32 of 32 | 0 / 0 |
| branch | forced | 3 of 3 | 32 of 32 | 12 / 9 |

Without the fix the 12 wrong evaluations are exactly the three after each change of type, in all four orders: F16
kernels on BF16 data (NMSE 4.29), BF16 kernels on F16 data (0.995), Q4_0 kernels on IQ4_NL data (0.869), IQ4_NL
kernels on Q4_0 data (191); the evaluations after the change back are right again (each type's own graph). With the
fix every change of type resets the warm-up and the graph is captured anew, so every evaluation is right (NMSE between
6.33e-15 and 2.1e-05).

## C1111's reproduction

C1111's 12-case F16/BF16 sibling set (`run/c1111.sh again`, every case checked live with `--cpu-check 1`), 20 runs
per build and setting, one process each ([again.tsv](again.tsv), [again_runs.tsv](again_runs.tsv)). `eee615cdb` is
C1111's base (`pascal-p106-v5-cand-b12674`), where the set fails; "`eee615cdb` + fix" is that commit with the fix
commit's diff as a patch:

| Build | CUDA graphs | Runs without a failing case | Graph warm-ups completed / reset, per run |
|---|---|---:|---|
| `eee615cdb` | on | 3 of 20 | 101 / 74 in the 17 failing runs, 108 / 63 in the 3 passing |
| `eee615cdb` | off (`GGML_CUDA_DISABLE_GRAPHS=1`) | 20 of 20 | 0 / 0 |
| `eee615cdb` + fix | on | 20 of 20 | 102 / 75 in 18 runs, 108 / 63 in 2 |
| `eee615cdb` + fix | off | 20 of 20 | 0 / 0 |
| base `2d839da1d` | on | 20 of 20 | 108 / 63 (one run 107 / 53) |
| base `2d839da1d` | off | 20 of 20 | 0 / 0 |
| branch | on | 20 of 20 | 108 / 63 (two runs 108 / 54) |
| branch | off | 20 of 20 | 0 / 0 |

Every failure is the same case, `MUL_MAT(type_a=bf16,type_b=f32,m=1057,n=1,k=128,bs=[4,1],nr=[1,1],per=[0,2,1,3])`
right after its F16 sibling, with ERR 0.505182503, C1111's number to the last digit. On `eee615cdb` a failing run
completes 101 graph warm-ups and resets 74, a passing one 108 and 63: when the BF16 case lands at its F16 sibling's
addresses, its nodes replay the F16 case's graph. **With the fix, 18 of the 20 runs complete 102 and reset 75, one of
each more than the failing runs:** the BF16 case lands at the same addresses as before, and the fix sees the new type,
resets the warm-up and captures the case anew instead of replaying, so the case passes. The other 2 runs (108 and 63)
are runs where the case landed elsewhere. The base and the branch pass this set either way: their heap puts the BF16
case elsewhere, as C1111 found for `2d839da1d`, so on them only the test above can show the fix.

## Nothing else changed

| Check | Base `2d839da1d` | Branch | Verdict |
|---|---|---|---|
| Machine code ([sass.txt](sass.txt)): `cuobjdump -sass` of `llama-server`, one SHA-1 per kernel copy | 9,508 kernel names, 10,075 copies | the same | every kernel the same bytes; the change is host code only |
| Build ([binaries.tsv](binaries.tsv)) | 24 binaries, 32 warning lines | 25 (`test-cuda-graph-src-types` added), 32 warning lines | no new warning; 0 AVX instructions, newest glibc symbol GLIBC_2.38 in both |
| `MUL_MAT` from C1023's kept record, the fork's matrix-vector kernels forced ([replays.tsv](replays.tsv)) | 5,458 of 5,458 | 5,458 of 5,458; with `--cpu-check 25` 5,458 of 5,458, 216 references identical | every case passes, none missing |
| `MUL_MAT_ID`, the same | 2,389 of 2,389 | 2,389 of 2,389; with `--cpu-check 25` 2,389 of 2,389, 84 references identical | every case passes, none missing |
| `FLASH_ATTN_EXT` from C1042's kept record, every Pascal attention switch / the tile kernel / no switch | 4,679 of 4,679 under each | 4,679 of 4,679 under each; with `--cpu-check 25` 4,679, 186 references identical | every case passes, none missing |
| the 55 fused-case operators from C1042's record, `PASCAL_FORCED` / `PASCAL_FORCED_MMV` | 5,424 of 5,424 under each | 5,424 of 5,424 under each; with `--cpu-check 25` 5,424, 200 references identical | every case passes, none missing |
| GPU pipe check (`bench/pipespec/gpu_pipe_check.sh`, default arms, one card and split over two; [pipe.tsv](pipe.tsv), [pipe_plain_hashes.tsv](pipe_plain_hashes.tsv)) | 125 OK, 5 INFO | 125 OK, 5 INFO | every verdict OK; the plain run's token and logit hashes the same on both builds in all 100 checks that record them |
| Qwen3.8-Flash-Next, v5 preset, `-ts 1,1`, the fork's paths forced ([flash.tsv](flash.tsv)) | coding request 256 tokens, prose 256 | the same 256 and 256 token ids, the same text | the same drafted and accepted counts (236 and 176; 273 and 163), the same MiB per card (20,181 and 22,371), the same count of reused graphs in the server's log (76 and 164) |

Three replays were cut short at 05:51:17, when the second invocation (the `eee615cdb` + fix build) ended and its
clean-up stopped every container carrying the row's label; `run.sh` now stops only its own invocation's containers
(`NAME`). Every case had passed up to the cut: `MUL_MAT_ID` on the base (245), `FLASH_ATTN_EXT` with no switch on the
branch (1,929), the fused cases under `PASCAL_FORCED_MMV` on the base (446). Each ran again whole in a folder of its
own and passed every case; [replays.tsv](replays.tsv) lists the cut runs and their reruns.

**The Flash-Next load:** UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (its n-gram table as the third shard,
the file the v5 preset names) with Swift-1.5's own MTP head as the draft (as in
[C947](../2026-11-21_c947_v5_merged_gpu/README.md): the preset's draft file was on the rigs), its v5 preset
([release/flash-next-v5](../../../release/flash-next-v5/settings.env)) with the layer split changed from nine cards to
the two (`-ts 1,1`) and every compute-6.1-only path forced as on a P106 (`pascal_force_preset`); `GGML_CUDA_GRAPHS_FORCE=1`
and the preset's other graph switches stay on. One server per build (branch, then base), C854b's coding request and
prose paragraph, greedy (`/completion`, temperature 0, top-k 1, 256 tokens, no prompt cache).

## Commands

On a machine with two NVIDIA cards, Docker (the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image) and these folders
readable at the same paths inside containers: the base's `TESTS=1` rig build and its build log, C1111's base build,
C1023's and C1042's kept records, the Flash-Next files and a llama.cpp clone holding the base (with the fork's refs):

```bash
export W=<work folder on a bulk disk> GW=<the same> REPO=<a copy of this repository> SEED=<llama.cpp clone> \
    BASE=<2d839da1d rig build> BASE_LOG=<its build log> C1111_BASE=<eee615cdb rig build> \
    REC_MM=<C1023 record folder> REC_FA=<C1042 record folder> FN=<Flash-Next shard 1> DRAFT=<its MTP draft> \
    RIG_CCACHE=<the base build's compiler cache> LOCK=<build lock file> PREFLIGHT=<CUDA check, run as $PREFLIGHT -g <ids>> \
    MOUNTS='<folders holding the builds, records and models>' JIT_SEED=<a PTX JIT cache of the base's kernels> \
    JIT_SEED_AGAIN=<C1111's JIT cache> JIT_SEED_FA=<C1042's JIT cache> GPUS=<card>,<card>
# the build, the test-only build, the machine code, then every card step
STEPS='src rig testonly sass' GPU_STEPS='test again mm fa pipe flash' timeout 13500 bash run/run.sh
# C1111's base with the fix, in its own folder (its compiler cache: one holding eee615cdb's objects)
W=<second folder> GW=<second folder> RIG_CCACHE= C1111_CCACHE=<that cache> STEPS='src c1111fix' GPU_STEPS=none \
    timeout 2700 bash run/run.sh
# C1111's set on it (the c1111fix folder copied into W first), appended to the first run's again table
STEPS=none GPU_STEPS=again AGAIN_BUILDS=c1111fix NAME=c1157b bash run/run.sh
# the tables, on any machine, from a copy of the outputs
python3 -I run/tables.py <W copy> <GW copy> <this folder>
```

The three cut runs again, each in its own folder with a copy of the first run's JIT cache, without the cpu-check
(the branch's ran), with the same `MOUNTS` and labels:

```bash
env WORK=<W>/mm-rerun NEW=<branch build>/bin REC=<C1023 record folder> BUILDS='base=<2d839da1d rig build>/bin' \
    SHARES=mul_mat_id STEPS=replay ALLOW_MISSING=1 CPU_CHECK=0 GPU=<card> NAME=c1157r-mm \
    bash ../2026-11-24_c1023_suite_record/run/run.sh
env WORK=<W>/fa-rerun NEW=<branch build>/bin REC=<C1042 record folder> BUILDS='base=<2d839da1d rig build>/bin' \
    SHARES=fused ARMS_FUSED=mmv STEPS=replay ALLOW_MISSING=1 CPU_CHECK=0 GPUS=<card> NAME=c1157r-fa \
    PASCAL_FORCED_SH=<this repository>/scripts/pascal_forced.sh bash ../2026-11-26_c1042_suite_records/run/run.sh
env WORK=<W>/fa-rerun2 NEW=<branch build>/bin REC=<C1042 record folder> BUILDS='branch=<branch build>/bin' \
    SHARES=flash_attn_ext ARMS_FA=plain STEPS=replay ALLOW_MISSING=1 CPU_CHECK=0 GPUS=<card> NAME=c1157r-fa2 \
    PASCAL_FORCED_SH=<this repository>/scripts/pascal_forced.sh bash ../2026-11-26_c1042_suite_records/run/run.sh
```

Wall times on 2026-11-26: the first invocation 05:14–06:30 (the build 2 min with the base's compiler cache, the
test-only build 1 min, the machine code 6 min, the two card lanes 51 min with the replays read from the bulk disk,
the pipe check 11 min, Flash-Next 5 min); the `eee615cdb` + fix build 2 min; C1111's set on it 1 min; the reruns
06:30–06:42.

## Files

| File | What |
|---|---|
| [upstream_offer.md](upstream_offer.md) | the pull request's text for ggml-org/llama.cpp, for the owner's word |
| [upstream.diff](upstream.diff) | the fix against upstream `master` `a8ad690ca` |
| [test.tsv](test.tsv) | `test-cuda-graph-src-types`, one row per run: build, setting, exit code, evaluations right, verdict, graph warm-ups completed and reset |
| [again.tsv](again.tsv), [again_runs.tsv](again_runs.tsv) | C1111's sibling set: per build and setting, and per run |
| [replays.tsv](replays.tsv) | every record replay: cases passed, OK, FAIL, NOREC, not supported, illegal accesses, the cpu-check line |
| [pipe.tsv](pipe.tsv), [pipe_plain_hashes.tsv](pipe_plain_hashes.tsv) | the GPU pipe check's verdicts, and the plain run's hashes base against branch |
| [flash.tsv](flash.tsv) | the Flash-Next load: token ids, text, draft counts and MiB per card, branch against base |
| [binaries.tsv](binaries.tsv) | md5 of every binary of both builds and of the test-only build's test |
| [sass.txt](sass.txt) | the machine-code comparison's summary |
| [run/run.sh](run/run.sh) | the driver (every step above) |
| [run/tables.py](run/tables.py) | the tables from the driver's outputs |

## Provenance and licence

The test cases and their data are `test-backend-ops`' and `test-cuda-graph-src-types`' own random data from the builds
named above (the fork of llama.cpp, MIT); the model file is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next
(Swift Open License), read, not published here. The tables, scripts and diff here are under this repository's
licence.
