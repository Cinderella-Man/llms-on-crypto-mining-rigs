# The read-once kernels of 5–8 tokens without their one-warp launches (C809, Q1415, 2026-11-12)

What this is: a host-side rule and its checks on a build machine without a GPU. C729's read-once kernels take
IQ4_XS's, Q4_K's and Q5_K's products of 5–8 tokens under `GGML_CUDA_MMVQ_PASCAL_RO8`
([C729](../2026-11-11_c729_readonce_5to8/README.md)), so that every token's sums are the one-token kernel's. On a
P106-090 they gained in 30 of their 32 `perf` cases at Qwen3.8-Flash-Next's shapes and lost in the two whose launch holds
one warp a block ([R120b](../2026-11-12_r120b_kern_speed/README.md#perf), µs a product, pass 1 / pass 2):

| Product | Tokens | Without `_RO8` | With `_RO8` (one warp a block) |
|---|---:|---:|---:|
| IQ4_XS 2,560 × 6,144 | 6 | 277.57 / 279.01 | 595.98 / 595.90 |
| Q5_K 2,560 × 6,144 | 5 | 309.21 / 308.22 | 520.89 / 519.46 |

The change: with `_RO8`, the two kernels decline a product of 5–8 tokens where only one warp a block fits in 48 KB of
shared memory, unless `GGML_CUDA_MMVQ_PASCAL_DMT_NW=1` (`_Q5K_MT_NW=1` for Q5_K) asks for one warp, which is Q6_K's rule
at 2–4 tokens since C727. A declined product runs where it runs with the switch off. It is host code only, behind
C729's switch ([docs/switches.md](../../switches.md)), on the fork's branch `pascal-p106-v4-b12246-c809` (`215d522c7`,
one commit on `pascal-p106-v4` `d21820340`), until C794's K tiles (Q1389) take those products. Registry row
[Q1415](../../../quants/experiments.md).

## Result

**Every check that runs without a GPU passes.**

- **The GPU code is the base's.** In `bin/llama-server` all 8,540 functions are the base's in SASS and PTX, and the dumps
  are byte for byte the same file. The four objects that hold the change and its callers (`mmvq-pascal-dmt`,
  `mmvq-pascal-q5k`, `mmvq`, `mmvq-pascal-kcol`) keep all 1,186 of their functions.
- **The choices change only where they should.** `run/c809_pick.cu` prints what each tree's launcher picks for
  every product the read-once kernels could take: IQ4_XS, Q4_K and Q6_K at 2–8 tokens, Q5_K at 1–8, rows of 1 to 64
  super-blocks. It ran under 54 settings on both trees.
  - Without `_RO8`, every line is the base's: 1,856 products under each of the 29 settings without it.
  - With `_RO8` (any value but 0), the only lines that change are the base's one-warp launches of 5–8 tokens, and every
    one of them becomes "declined (one warp)". That is 26 row widths, from 3,840 to 7,680 weights (15 of IQ4_XS and Q4_K,
    11 of Q5_K), under each of the 25 settings with it. The exception is a setting that asks for one warp (`_DMT_NW=1`,
    `_Q5K_MT_NW=1`): there that kernel's lines do not move.
  - At Flash-Next's two row widths only R120b's two losses change: IQ4_XS 2,560 × 6,144 at 6 tokens and Q5_K
    2,560 × 6,144 at 5 tokens.
- **Off is the base.** `test-llama-archs -b CPU -s 1` gives the base's logits on 146 of 146 random models with no switch,
  with `_RO8=1` and with `_RO8=2`. ctest passes 70 of 70, static and shared. The merged `pascal-p106-v4` (`d7cdd032a`)
  gives the same: archs 146 of 146 with no switch and with `_RO8=1`, and the same 598 changed choice lines against the
  fork's tip before the merge.
- **The rig tarball** is `llama.cpp-20261112-215d522c7-pascal-p106-v4-b12246-c809.tar.gz`: 678,346,098 B, md5
  `08c93faca074cefa7f7243f4cc4772a0`, `TESTS=1`, sm_61, 0 AVX instructions in `llama-bench`, GLIBC_2.38.
  Its `test-backend-ops` has md5 `839d59c0f4134175c7417bb435e23a78`.
- **The bit-identity harness** `multitoken_bitid` is linked against this build: md5
  `97bc6cced0d39c0077da2ded571acd12`, with the modes `dmt8`, `q5k8`, `dmt` and `q5k`.

The products `_RO8=1` now declines ([one_warp_ro8.tsv](one_warp_ro8.tsv); row widths in weights, at every row count):

| Type | 5 tokens | 6 tokens | 7 tokens | 8 tokens |
|---|---|---|---|---|
| IQ4_XS | 7,168, 7,680 | 6,144 | 5,120, 5,632 | 4,608 |
| Q4_K | 6,400, 6,656, 6,912 | 5,376, 5,632 | 4,608, 4,864 | 4,096, 4,352 |
| Q5_K | 6,144, 6,400, 6,656 | 5,120, 5,376, 5,632 | 4,352, 4,608, 4,864 | 3,840, 4,096 |

Flash-Next's 2,560-wide rows keep 4 to 6 warps at every count from 5 to 8 tokens. Its 6,144-wide rows change only at the
two R120b losses. At 5 tokens IQ4_XS keeps 3 warps and Q4_K 2; the 48 KB rule already left the rest of those rows to
the k-column kernel (Q4_K and Q5_K from 6 tokens, IQ4_XS from 7).

## The declined products are not the one-token sums

A product the kernels now decline gets what it gets with `_RO8` off, so its tokens' sums are no longer the one-token
kernel's: the speed comes back, the bit identity of those products goes. `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT`
([C805](../2026-11-12_c805_kcol_exact/README.md)) does not change that: it keeps the products of Q3_K, IQ4_NL, Q2_0 and
short-row Q8_0, the types whose one-token sums are the k-column kernel's, and IQ4_XS, Q4_K and Q5_K are not among them.
Where the two Flash-Next products go, from C805's `kcol_pick.cu` tables of the k-column kernel's choices:

| Product | Under the preset (`_KCOL_SMEM=32768 _KCOL_TABLE=0`), with or without `_KCOL_EXACT=1` | Without the budget (`_KCOL_TABLE=0`) |
|---|---|---|
| IQ4_XS 2,560 × 6,144, 6 tokens | declined by the 32 KB budget; the Pascal multi-token kernel takes at most 4 tokens, so llama.cpp's own `mul_mat_vec_q` | the k-column kernel, one launch of 33,536 B |
| Q5_K 2,560 × 6,144, 5 tokens | the same (declined from 4 tokens on) | the k-column kernel, one launch of 35,968 B |

In R120b's `mt` arm (the preset with `_DMT=1 _Q5K_MT=1`, `_RO8` off) these were the 277.57 / 279.01 and
309.21 / 308.22 µs above. The bit-identity harness prints such products as declined and does not count them: it asks
the build's own `ggml_cuda_mmvq_pascal_*_takes`. Each token's sums of these products become the one-token ones again
when C794's tiles take them.

## Checks on the build machine (no GPU)

[run/checks.sh](run/checks.sh) ran every build and test in a container at a low CPU share (`--cpu-shares=256`, nice 19),
one at a time under a lock it shares with the machine's other sessions. Its verdicts are in [summary.tsv](summary.tsv).
The base is `pascal-p106-v4` `d21820340`. Its `ggml/` is C803's `bd632176f`'s (C806 between them changed only
`tests/test-backend-ops.cpp`), so C803's checked rig build stands for its GPU code and switch list. Its logits are C802's
static CPU build's: C803's own check found 146 of 146 the same.

| Step | What | Verdict |
|---|---|---|
| pick | `run/c809_pick.cu` built against both trees' `mmvq-pascal-dmt.cu` and `mmvq-pascal-q5k.cu`, 54 settings ([pick_arms.txt](pick_arms.txt)); the eight expressions it repeats found in each tree as expected | OK |
| pickcmp | without `_RO8` every line the base's; with it only the one-warp launches change, all to "declined (one warp)", none where one warp is asked for ([pick_counts.tsv](pick_counts.tsv), [pick_changes.tsv](pick_changes.tsv)) | SAME; OK |
| rig | the rig tarball, `TESTS=1` | OK, md5 `08c93fac…` |
| names | the switch list in the build against the base's | SAME, 861 names |
| sass | `bin/llama-server`, `cuobjdump -sass` and `-ptx`, against the base's | SAME, 8,540 of 8,540 functions, the same dump files |
| sassobj | the objects of `mmvq-pascal-dmt.cu`, `mmvq-pascal-q5k.cu`, `mmvq.cu`, `mmvq-pascal-kcol.cu` against the base build's | SAME, 1,186 functions, 0 of 8 dumps differ |
| cpu, off | a static CPU build; `test-llama-archs -b CPU -s 1` against the base's logits, with no switch, `_RO8=1`, `_RO8=2` | SAME, 146 of 146 each |
| bitid | `multitoken_bitid` linked against this build | OK, md5 `97bc6cce…`, modes `dmt8`, `q5k8`, `dmt`, `q5k` |
| c565 | C565's CPU set, static and shared: ctest, the network test, archs written and compared, the nine hand tests | 70 of 70 each; 1 of 1; 146 of 146; 9 of 9 |
| merged | `pascal-p106-v4` after the merge: a static CPU build, archs with no switch and `_RO8=1`, the pick against the fork's tip before the merge | OK; 146 of 146 each; the same 598 changed lines |

**The merge.** The fork's `pascal-p106-v4` is `d7cdd032a`, the `--no-ff` merge of `-c809` `215d522c7` on C812's
`f91533730`. It adds exactly `-c809`'s diff: none of C807, C811 and C812, merged since `d21820340`, touches the two
files.

## The P106 check (R134)

R120b's `perf` step on this build's rig tarball: `test-backend-ops perf -o MUL_MAT` at Flash-Next's IQ4_XS, Q4_K, Q5_K
and Q6_K shapes, 1–8 tokens, the arms `off` (the preset's k-column switches), `mt` (+ `_DMT=1 _Q5K_MT=1`) and `kern`
(`mt` + `_Q6K_MT=1 _DMT_KT=1 _RO8=1`), two passes alternated, on card 0 of an idle rig
([rig_kern.sh](../2026-11-11_v4_kernel_merge/run/rig_kern.sh) with `STEPS=perf PERF_ARMS="off mt kern"`), then
[perf_table.py](../2026-11-12_r120b_kern_speed/run/perf_table.py). Passing: IQ4_XS 2,560 × 6,144 at 6 tokens and Q5_K
2,560 × 6,144 at 5 tokens negligible in `kern` against `mt` (the same kernel in both arms now), and `_RO8`'s other 30
cases gains as in R120b.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every verdict line of the checks (row, check, verdict, detail), the merged tree's included |
| [one_warp_ro8.tsv](one_warp_ro8.tsv) | the products `_RO8=1` now declines: type, tokens, row widths, count |
| [pick_counts.tsv](pick_counts.tsv) | per setting and program: `_RO8`, whether it asks for one warp, product lines, lines changed, changes not of the expected kind, one-warp launches left |
| [pick_changes.tsv](pick_changes.tsv) | every changed line: setting, program, type, tokens, K, super-blocks, the base's launch (R, prefetch, warps, lanes per row, shared memory, stride, kernel), the new one |
| [pick_arms.txt](pick_arms.txt) | the 54 switch settings of the kernel-choice check |
| [run/checks.sh](run/checks.sh) | the driver: every step above |
| [run/c809_pick.cu](run/c809_pick.cu) | the read-once kernels' choices of one tree, without a GPU |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`, an
unpacked rig tarball whose `ggml/` is the base's, its build objects and switch list, and a static CPU build with the
base's logits:

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> BASE_COMMIT=d218203409166a45cf9420ca3d08c861a64068ef BASE_TREE=<base rig tree> \
BASE_OBJ=<base build>/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir BASE_NAMES=<base build>/common/switch-names.cpp \
BASE_CPU=<base CPU bin/> SEED_SRC=<clone> bash docs/data/2026-11-12_c809_ro8_one_warp/run/checks.sh
# once the branch is merged into pascal-p106-v4 (the fork's tip before the merge as the pick's base):
M=<work folder> REPO=$REPO LOCK=<flock file> BASE_COMMIT=d218203409166a45cf9420ca3d08c861a64068ef BASE_CPU=<base CPU bin/> \
SEED_SRC=<clone> BASE_OBJ=<base build>/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir STEPS=merged \
MERGED_STEPS="build off pick" MERGED_PICK_BASE=f91533730 bash docs/data/2026-11-12_c809_ro8_one_warp/run/checks.sh
```

On a rig (R134), with the tarball unpacked at `$BUILD`, from a folder holding this repository's
`docs/data/2026-11-11_v4_kernel_merge/run/rig_kern.sh`:

```bash
STEPS=perf PERF_ARMS="off mt kern" PASSES=2 BIN=$BUILD/bin OUT=<dir> \
  setsid -f nohup bash docs/data/2026-11-11_v4_kernel_merge/run/rig_kern.sh > console.txt 2>&1 < /dev/null
# then, on a build machine:
nice -n 19 python3 -I docs/data/2026-11-12_r120b_kern_speed/run/perf_table.py <OUT>/perf/us_run.tsv --out <tables> > <tables>/perf.md
```

## Provenance and licence

Code and scripts of this repository and of the fork (MIT, as llama.cpp). No model, data set or external text is used:
`c809_pick.cu` and the checks run on the source trees, the archs check on llama.cpp's own random test models.
