# The 64-row prompt tile kept behind a switch on the P106's tile table (C1024, Q1757; 2026-11-24)

What this is: the code half of registry row Q1757 in [quants/experiments.md](../../../quants/experiments.md), and its
correctness check. Upstream llama.cpp's #31352, merged into the v5 candidate with tag b12674
([C912](../2026-11-24_c912_upstream_b12674/README.md)), changed the prompt kernel (`mul_mat_q`, the "MMQ" kernel that
multiplies a quantized weight matrix by a batch of tokens) on the tile table Pascal cards use. That table is the DP4A
table, which compute 6.1 runs. #31352 gave its 64-column tile (J=64) 128 rows instead of 64 for 11 types: Q1_0, Q2_0,
Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, IQ2_XXS, IQ3_XXS, IQ4_NL and IQ4_XS. Upstream measured it on larger Pascal cards. A
P106-090 has 5 SMs, and at a 64-token micro-batch the taller tile gives a launch half as many blocks. The new switch
`GGML_CUDA_MMQ_J64_I64` gives that tile its 64 rows back, so a rig can measure both in one build. It lives on the fork
branch `pascal-p106-v5-mmq-j64-tile` (`ebec6544e`, one commit on the candidate `pascal-p106-v5-cand-b12674` at
`eee615cdb`). It was built and checked on a build machine's RTX 3090s, which run the rig build's compute 6.1 code
through the driver's compiler and take the same DP4A table. Correctness only: nothing here is timed. The P106's prompt
time is [the rig check](#the-rig-check) below.

## Result in plain words

- **The switch works, and every test passes both ways.** With `GGML_CUDA_MMQ_J64_I64=2` on the build machine's cards,
  all 22 kernels it adds launched, each with a 64 × 64 tile. Every `MUL_MAT` case passed with the switch unset and set
  (5,458 of 5,458 each), the 297 new cases for this tile among them, and so did every `MUL_MAT_ID` case of the 11 types
  (968 of 968 each). With `=1` on these cards, which are not compute 6.1, no launch was redirected.
- **The added kernels are the old ones, byte for byte.** Each of the 22 is the same machine code as a build of the
  candidate with only those table rows set back to 64 rows.
- **Unset, the build runs the candidate's kernels.** Every kernel the candidate has is in this build. 1,664 differ only
  in the source line number they print: they are stubs for tile shapes the table does not have, never launched, and
  the line moved because 21 lines were added above it. No warning is new.
- **It reaches more than 64-token passes.** On the P106's tile table 64 columns is the widest tile for these types. So
  the switch, like #31352 itself, changes every prompt pass of 49 or more tokens: the taper's 64-token micro-batches,
  and also Flint's 128-token and Flash-Next's 512-token ones. The rig check reads both kinds.
- **What it does not show:** which tile is faster on a P106. That is [the rig check](#the-rig-check) below, est. 1 h.

## The change

- **Where the tile applies.** On the DP4A table, 64 columns is the widest tile for these types. Every prompt product
  of 49 or more tokens takes it: the micro-batch taper's 64-token passes (`--ubatch-taper short=…:64` in Flint's and
  Flash-Next's presets), Flint's 128-token micro-batches and Flash-Next's 512-token ones. A MoE layer's routed experts
  choose their tile by the tokens each expert gets on average, so at 64 tokens they take a narrow one.
- **A second copy of the kernels.** 11 new source files, `ggml/src/ggml-cuda/template-instances/mmq-instance-j64i64-<type>.cu`
  (written by `generate_cu_files.py`, like upstream's), define `GGML_CUDA_MMQ_J64_I64_TU` before they include
  `mmq.cuh`. In those files only, the DP4A table's J=64 row of the 11 types has 64 rows
  (`ggml_cuda_mmq_get_config_dp4a`), and each builds the J=64 launch of its type, without and with the row check (the
  bounds check a weight whose rows are not a multiple of 128 needs). Every other file reads the table as upstream
  wrote it.
- **The switch only picks the launch.** In `mul_mat_q_switch_J`, a 64-column launch of the 11 types goes to the copy
  when `ggml_cuda_mmq_j64_i64()` says so. `0` or unset: off, every kernel and launch upstream's. `1`: on a compute 6.1
  card whose prompt products take the DP4A table. `2`: on any NVIDIA card that takes that table, which is a compute 6.1
  build on a newer card, for tests. Not on HIP or MUSA. The fused gate/up launch and the K-quant FP32-scale launch
  keep upstream's kernel; neither runs on the DP4A table.
- **It says when it acts.** The first read logs `GGML_CUDA_MMQ_J64_I64=<n>: 64-column prompt tiles keep 64 rows`. The
  first launch of each copy logs its type, whether it has the row check, and the tile it took, for example
  `first launch of q8_0's J=64 kernel (no row check), tile 64 x 64`. All of these are at the library's INFO level.
- **Test cases.** `test-backend-ops` gets 297 `MUL_MAT` cases for this tile: the 11 types at 49, 64, 65, 128 and 512
  columns (one column tile and several) times 64, 128, 192, 1,000 and 4,096 rows (with and without the row check),
  broadcast batches, and an input with a nonzero mean for the types' zero points and minimums. Before them the suite
  reached this tile in 17 cases of the 11 types, one to four a type.

## The checks

Everything ran on the build machine's CPU and its GPUs 0 and 1; every row is in [checks.tsv](checks.tsv).

| Check | Result |
|---|---|
| Rig build (`TESTS=1`, web page b12674) | `llama.cpp-20261124-ebec6544e-pascal-p106-v5-mmq-j64-tile.tar.gz`, 1,416,336,444 B (the candidate's 1,403,235,928 B), md5 `a2e397f480f39f7ce7f86d20a969bcff`; `llama-server` md5 `ac6e5a6d36621a27e7c2ab252768ef2f`; 0 AVX instructions, GLIBC_2.38 |
| Warnings against the candidate's build | 32 lines and 32; none new |
| Card code | the switch's 22 kernels the reference's byte for byte; every candidate kernel kept ([below](#the-card-code)) |
| `MUL_MAT`, switch unset | 5,458 of 5,458 (521 not supported), the 297 new cases all OK, no copy launched |
| `MUL_MAT`, `GGML_CUDA_MMQ_J64_I64=2` | 5,458 of 5,458 (521 not supported), the 297 new cases all OK, all 22 copies launched at 64 × 64 ([launches.tsv](launches.tsv)) |
| `MUL_MAT`'s 297 new cases, `=1` | 297 of 297; on a card that is not compute 6.1 the switch is read and no launch goes to a copy |
| `MUL_MAT_ID`, the 11 types, unset and `=2` | 968 of 968 each; with `=2` the copies of Q8_0 and IQ2_XXS (row check) launched, the MoE cases whose experts get 49 or more tokens on average |

`test-backend-ops` ran with a P106's Pascal paths forced (`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2` and their
5-SM launch shapes), every case but the repeated-expert-id ones, as C912 ran them. A first run of the whole
`MUL_MAT_ID` suite with every type, unset and `=2`, was stopped after 751 and 748 passed cases (none failed): the CPU
reference was slow beside other jobs, and the switch reaches no type but the 11. The 11 types' cases then ran alone.

## The card code

Three builds' `llama-server`, compared kernel copy by kernel copy (`cuobjdump -sass`, the compute 6.1 machine code a
P106 runs) and PTX function by PTX function (`cuobjdump -ptx`, the code a newer card compiles):

- **C**, the candidate `eee615cdb` as C912 built it;
- **H**, this branch's build;
- **R**, a reference: the candidate with the DP4A table's 22 J=64 rows of the 11 types set back to 64 rows, and
  nothing else ([run/run.sh](run/run.sh), step `ref`). The upstream commit itself no longer reverts cleanly, since the
  fork's own types sit next to its lines.

| Comparison | Kernel copies | Same bytes | Differ |
|---|---|---|---|
| C against R | 10,009 and 10,009 | all but 66 names | 66 names, all the 11 types' 64-column kernels: `mul_mat_q` (22), its forced-tiling twin (22, launched only on AMD GCN) and its stream-k fixup (22, never launched on the DP4A table) |
| C against H | 10,009 and 10,075 | 7,778 of 9,508 names | 1,664 names only in one constant, the source line a stub kernel prints (1,103 → 1,124, the 21 lines added above it in `mmq.cuh`); the same 66 names hold one more copy each in H; no candidate copy is missing or changed otherwise |
| H's extra copies against R | 22 switch kernels | 22 of 22 byte for byte | none |

The 22 kernels the switch launches, instructions in the candidate's 128-row tile and in the 64-row tile (H's copy and
R's, the same bytes; [j64_sass.tsv](j64_sass.tsv)):

| Type | No row check, 128 rows | 64 rows | Row check, 128 rows | 64 rows |
|---|---|---|---|---|
| Q1_0 | 1,890 | 1,170 | 2,430 | 1,386 |
| Q2_0 | 1,872 | 1,164 | 2,412 | 1,374 |
| Q4_0 | 1,740 | 1,110 | 2,346 | 1,374 |
| Q4_1 | 1,836 | 1,152 | 2,436 | 1,410 |
| Q5_0 | 2,286 | 1,350 | 2,742 | 1,536 |
| Q5_1 | 2,292 | 1,368 | 2,754 | 1,560 |
| Q8_0 | 1,704 | 1,074 | 2,304 | 1,326 |
| IQ2_XXS | 1,968 | 1,182 | 2,316 | 1,332 |
| IQ3_XXS | 2,106 | 1,242 | 2,418 | 1,392 |
| IQ4_NL | 1,878 | 1,158 | 2,532 | 1,422 |
| IQ4_XS | 1,920 | 1,164 | 2,520 | 1,428 |

In PTX the 44 entries of these types' 64-column `mul_mat_q` that the switch adds (the 22 and their forced-tiling
twins) are R's text once the basic-block labels are renumbered: a label carries the entry's index in its file, which
differs between files ([run/ptx_j64.py](run/ptx_j64.py)). The fused gate/up entries of Q8_0 and IQ4_XS, 30-instruction
stubs that only trap on this table, are the same in C and R and differ in H only in the printed line.

## The test cases

[bops.tsv](bops.tsv) has one row per arm, [j64_cases.tsv](j64_cases.tsv) the 297 new cases in each `MUL_MAT` arm (all
OK in all three) and [launches.tsv](launches.tsv) the first-launch lines. The new block runs in
`make_test_cases_eval` after upstream's `MUL_MAT` blocks, in the test file at `ebec6544e`; its cases match

```
type_a=(q1_0|q2_0|q4_0|q4_1|q5_0|q5_1|q8_0|iq2_xxs|iq3_xxs|iq4_nl|iq4_xs),type_b=f32,m=(64|128|192|256|1000|4096),n=(49|64|65|128|512),k=(512|1024),
```

so `test-backend-ops test -o MUL_MAT -p '<that>'` runs them alone.

## The rig check

Registry row Q1757's other half, queued for a P106 rig that holds both files. The deep lane
([bench/deep_lane](../../../bench/deep_lane/README.md)) on this branch's rig build, two arms on the one build:
`unset` (upstream's 128-row tile, the control) and `j64` (`GGML_CUDA_MMQ_J64_I64=1`). Loads
`warm unset1 j641 unset2 j642`, depths 0, 4,096 and 8,192, 256 tokens a pass, the sampler pass; once on Flint Pascal
Q8 GGUF with [release/flint-v5/settings.env](../../../release/flint-v5/settings.env), once on Flash-Next with
[release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) (its file and MTP draft). Each
depth's first decode pass reads the prompt's 64-token tail in one micro-batch, the case the taller tile halves the
blocks of. The primes read 4,032 and 4,096 tokens in the presets' micro-batches, which take the same tile. What to read:
the tail's prompt time at each depth and the primes' prompt speed, each load by value and classed against the spread,
and the same greedy replies in both arms. From the repository root on a machine that only runs ssh, once per model:

```bash
RIG=<rig> WORK=<folder on the rig> BIN=<this build's llama-server on the rig> MODEL=<model file on the rig> \
  [MODEL_DRAFT=<draft on the rig>] PRESET=release/<flint|flash-next>-v5/settings.env \
  TEXT_LOCAL=<recorded agent text> OUT=<folder here> STEP=start \
  bash docs/data/2026-11-24_c1024_mmq_j64_tile/run/rig_lane.sh
```

then `STEP=wait` and `STEP=collect` with the same `RIG`, `WORK` and `OUT`. `STEP=start` runs the deep lane's dry run
first and starts only when the two server lines are the preset's, with the switch in `j64`'s only.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check and its result: the build, the warnings, the card code, the test arms |
| [j64_sass.tsv](j64_sass.tsv) | per 64-column `mul_mat_q` name of the 11 types: copies and instructions in C, H and R, and whether H holds C's copy and R's |
| [bops.tsv](bops.tsv) | per `test-backend-ops` arm: exit code, cases passed, OK, FAIL, not supported, the C1024 cases, the switch's log line, first-launch lines |
| [j64_cases.tsv](j64_cases.tsv) | the 297 C1024 cases, each arm's verdict |
| [launches.tsv](launches.tsv) | the first-launch lines: arm, type, row check, tile |
| [run/run.sh](run/run.sh) | the run on the build machine: sources, rig build, reference build, card code, test arms, tables |
| [run/j64_sass.py](run/j64_sass.py) | the 64-column kernels of the 11 types in three builds' SASS |
| [run/ptx_j64.py](run/ptx_j64.py) | the same in PTX, block labels renumbered |
| [run/c1024_tables.py](run/c1024_tables.py) | `bops.tsv`, `j64_cases.tsv`, `launches.tsv` from the test logs |
| [run/rig_lane.sh](run/rig_lane.sh) | the rig check: the deep lane, two arms on one build |

## Commands

On the build machine, with a llama.cpp clone that holds `eee615cdb` (`SEED`), the candidate's rig build and its log
(C912's), a copy of this repository (`REPO`), and two free cards:

```bash
W=<work dir> SEED=<clone> CAND_RIG=<eee615cdb's rig build folder> CAND_RIG_LOG=<its build log> REPO=<repo copy> \
  RIG_CCACHE=<compiler cache> JIT_SEED=<a PTX JIT cache> GPUS=0,1 PREFLIGHT=<cuda check> LOCK=<build lock> JOBS=8 \
  bash run.sh
```

That was the first run. Its `MUL_MAT_ID` arms then ran every type and were stopped (above); the second run was the same
command with `STEPS="bops tables" BOPS_ARMS="mul_mat_id-unset mul_mat_id-j64i64"`, the run script as committed here
(`MUL_MAT_ID` filtered to the 11 types by `ID_FILTER`).

The PTX check, on the head's and the reference's `cuobjdump -ptx` output:

```bash
python3 -I run/ptx_j64.py ptx-H.txt ptx-R.txt
```

## Provenance and licence

Fork branch `pascal-p106-v5-mmq-j64-tile` of [Miner Llama](https://github.com/Cinderella-Man/miner-llama)
(`ebec6544e`; MIT, as llama.cpp), on upstream llama.cpp b12674, whose #31352 this switch undoes on request. The
tables summarize the build's `cuobjdump` output and `test-backend-ops` logs, which are not published.
