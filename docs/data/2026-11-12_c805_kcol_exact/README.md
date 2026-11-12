# The k-column kernel keeps its declined products (C805, Q1411, 2026-11-12)

What this is: the fix for the one fault [R120a](../2026-11-12_r120a_kern_exact/README.md) found on a P106. With
`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM` set (the Flash-Next preset sets 32768) the k-column kernel declined a product of several
tokens whose one iteration of activations did not fit that budget, and without the switch it declined one past 48 KB.
The declined product then ran on another kernel whose sums are not the one-token sums. Flash-Next's Q3_K
6,144 → 2,560 product differed at 6, 7 and 8 tokens under the preset, and so did the harness's wide Q3_K and Q2_0 rows.
The release preset's verify passes (at most 4 tokens) never ran those products; a draft of 5 or more (C797) would.

The fix is the switch `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT`, off by default, on the fork branch
`pascal-p106-v4-b12246-c805`, merged into `pascal-p106-v4`. It is host code only: no GPU function changes. This folder
holds its design, its checks on a build machine's CPU, and the rig run that checks it on a P106. Registry row
[Q1411](../../../quants/experiments.md); the switch is in [docs/switches.md](../../switches.md).

## The design

The k-column kernel (`ggml/src/ggml-cuda/mmvq-pascal-kcol*.cu`) forms each row's sum per lane in unit order, then over the
lanes of its group with shuffles in a fixed order. The number of lanes per row depends on the row width only. So a
token's sums are the same for every launch shape and every token count: one token alone, or the same token among 8, in
one launch or in several. Only the shape of the launch has to fit the shared memory, not the sums.

`GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1` uses that. For the types whose one-token sums are the k-column kernel's, the kernel
no longer leaves a product to another kernel:

- **Which types.** Q3_K and IQ4_NL, whose one-token products the k-column kernel runs; Q8_0 in rows whose 64-weight
  chunks are not a multiple of 4, the same; and Q2_0, whose one-token Pascal kernel (`mmvq-pascal.cu`) sums the same
  128-weight chunks (two blocks per lane) over the same lanes in the same order, with the same integer and FP32
  operations. The other types (IQ4_XS, Q4_K, Q5_K, Q6_K and the rest) keep their one-token sums in other kernels, and the
  read-once kernels (C722, C690, C727, C729) are their exact path; the switch leaves them as they are.
- **Within the budget's reach.** Where the budget declines such a product, it takes the default rule's 48 KB step
  (`pick_old` at 48 KB, as without the budget). That is Flash-Next's Q3_K 6,144 → 2,560 at 6-8 tokens: one launch of
  35,584 to 41,984 bytes of shared memory instead of 32 KB.
- **Past 48 KB.** Where one iteration of all the tokens' activations and the weight staging pass 48 KB (the harness's
  K = 8,192 rows from 3 tokens on, K = 4,096 and 12,288 from 6), the product runs as several launches of fewer tokens
  each, from one quantizer launch: the n tokens split into k launches of ceil(n / k) tokens, k the smallest whose shape
  fits 48 KB. Each launch reads its tokens' activation records and writes its tokens' columns. A K tile smaller than one
  iteration would leave lanes idle (each lane owns its unit of every iteration), and C726's kernel takes Q3_K only, so
  the token split keeps the one kernel for every type and changes no GPU code.
- **Not by the table.** With `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE` on (the default), the dispatch table of Q824 also leaves
  some of these products (Q2_0, Q8_0, the fallback shapes) to the other kernels for speed; with the switch those stay
  too. The preset sets `_KCOL_TABLE=0`, so there it changes nothing.
- **One token.** The one-token products the k-column kernel runs stay on it the same way, so that a budget smaller than
  one token's needs cannot move the one-token sums themselves to another kernel. Under the preset this already happens:
  Q3_K rows of one or three super-blocks (K = 256, 768: one lane per row, so 32 rows a warp) need 32 KB for the warp
  staging alone, and the preset's budget declined them even at one token. Were only their products of 2-8 tokens
  moved, those would leave the one-token kernel's sums.

`=2` tries launches within the budget first (Flash-Next's 6,144-wide Q3_K at 6 tokens: two launches of 3 tokens within
32 KB, each reading the weights), then as `=1`; it is there to time the split against the 48 KB shape.
`GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT_FAULT=1` makes a split product's later launches read the first launch's activations, a
fault the checks must catch.

The code: `kcol::pick_exact` in `mmvq-pascal-kcol-impl.cuh` plans the launches; `kcol::launch` runs `params::cols` tokens
a launch (0, the default, keeps one launch); `mmvq-pascal-kcol.cu` reads the switch, names the types and calls
`pick_exact` where `pick_shape` declines. With the switch unset, every line of the dispatch runs as before.

Which preset line turns it on: `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1` added to `PRESET_ENV` in
[release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env), beside
`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`, once the rig run below has shown it exact on
a P106 and timed it.

## What the switch changes

From [run/kcol_pick.cu](run/kcol_pick.cu) (below): every plain product of the k-column kernel's 16 types at 15 row widths
(256 to 16,384), 14 row counts and 1-8 tokens, on a P106-090's 5 SMs and 96 KB a SM. Under the Flash-Next preset
(`_KCOL_SMEM=32768 _KCOL_TABLE=0`), `=1` changes 882 of the 26,880 products, all Q3_K and Q2_0 (IQ4_NL's and those Q8_0
rows already fit):

| Type | Row width K | Tokens | Before (the preset) | With `=1` |
|---|---|---|---|---|
| Q3_K | 6,144 (Flash-Next's `ssm_out`), 2,048, 10,240 | 6-8 | declined | one launch, 35,584 to 41,984 B |
| Q3_K | 4,096, 12,288 | 3-8 | declined | one launch at 3-5 tokens (35,584 to 48,384 B), two at 6-8 |
| Q3_K | 8,192, 16,384 | 2-8 | declined | one launch at 2 tokens, then launches of 2 tokens (41,984 B each) |
| Q3_K | 256, 768 (one lane per row) | 1-8 | declined | one launch (33,168 to 35,968 B) |
| Q2_0 | 4,096 to 16,384 | 6-8 | declined | one launch at 6 and 7 tokens (38,400, 44,032 B), two of 4 at 8 |

Every other Flash-Next product is unchanged: its Q3_K at K = 2,560 (`attn_qkv`, `attn_gate`, `attn_q`, the shared
expert) fits at every count, its Q2_0 and IQ4_NL down projections are 640 wide. R120a's products, before and with
`=1` and `=2`, one line per token count: [pick_r120a_products.tsv](pick_r120a_products.tsv). Without the preset
(`=1` on the default rule, `_KCOL_TABLE=1`) the products that change are those the table or its fallback rule gave
away: 781, among them Q2_0's 2-token products at K ≤ 4,096 (mmvq-pascal.cu's kernel of 2-4 tokens had them, with the
same sums) and Q3_K's and Q2_0's fallback shapes.

## The checks

[run/checks.sh](run/checks.sh) ran every step on a build machine's CPU, in Docker containers at `--cpu-shares=256` and
`nice 19`, one build or test at a time under a shared lock, no GPU, on `dd755a09c` (2026-11-12 15:04-15:34). The base
is `pascal-p106-v4` `d097d6d88`: its checked rig tarball (`-kern`), its build objects and its static CPU build are the
v4 kernel merge's ([2026-11-11_v4_kernel_merge](../2026-11-11_v4_kernel_merge/README.md)). Every verdict line is in
[summary.tsv](summary.tsv), in the order the steps ran.

| Check | What ran | Result |
|---|---|---|
| builds | the rig tarball (CUDA 12.9, sm_61, no AVX, `test-backend-ops` in `bin/`) and a static native CPU build of every target | both built, 0 error lines; the tarball has 0 AVX instructions and needs GLIBC 2.38 at most |
| switch list | C770's list of the `LLAMA_` and `GGML_` names the sources read | 860 names: the base's 858, `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT` and `_KCOL_EXACT_FAULT`, both in `llama-server` |
| GPU code | `cuobjdump -sass` and `-ptx` of `llama-server` against the base's, cut into functions (anonymous-namespace hashes masked) with the day merge's [sass_attrib.py](../2026-11-11_v4_day_merge/run/sass_attrib.py) | 8,540 functions, every one the base's, SASS and PTX; the two dumps are the same file |
| GPU code | the 17 k-column objects (`mmvq-pascal-kcol.cu` and the 16 per-type files), SASS and PTX, against the base build's objects | 544 functions in 34 dumps, none differs |
| every switch off | `test-llama-archs -b CPU -s 1` on every architecture against `d097d6d88`'s logits, with no switch, `=1`, `=2`, `=1` with the fault, and `=1` with the preset (CUDA switches: they change nothing on a CPU) | 146 of 146 models the same bit for bit in each (151 of 151 tests pass) |
| every switch off | [C565's CPU test set](../2026-11-05_c565_ladder4fix/README.md), static and shared libraries | ctest 70 of 70, `test-arg-parser` with network, `test-llama-archs` written and compared (146 files, none differ), the nine hand tests, in each |
| kernel choices | [run/kcol_pick.cu](run/kcol_pick.cu) built against the base's and the new tree's own `mmvq-pascal-kcol.cu` and run under 24 switch settings ([pick_arms.txt](pick_arms.txt)) | without the switch (10 settings, `=3` among them: not a mode) every one of the 26,880 product lines is the base's; with it on (14 settings) only the four types' lines change, each from declined (or given away by the table) to taken, none of the four types' products is left to another kernel, every launch fits 48 KB and the launches cover the tokens ([pick_counts.tsv](pick_counts.tsv)); at `=1` a product splits into launches exactly where the base declines it at a 48 KB budget (308 products); the fault changes no choice |
| bit identity | `bench/mmvq_pascal/verify/multitoken_bitid` built against the new tree | it holds the modes `q3k` and `glu` (md5 `f1b99a3a1beb210aef8130c4eb1d7260`) |
| the merge | `pascal-p106-v4` after the merge | `d569b8bc2`, whose tree is `dd755a09c`'s: every check above holds for it |

Two driver slips are in [summary.tsv](summary.tsv) as they happened, each followed by the line that settles it: the
first object comparison ran `python3` inside the CUDA image, which has none, so it compared 34 empty dumps (DIFFERS,
0 functions); the step was split out (`sassobj`) and rerun with the cutting on the host (SAME, 544 functions). The
first comparison of the switch-on choices used `split` as an awk variable, a built-in name (FAIL, the awk errors); it
was moved into its own step (`pickcmp`) and rerun on the same arm files, once with the `pick` step's exit status
missing (both lines then read rc=1 with nothing differing) and once with it (rc=0, as the `pick` step logged).

## The rig tarball

The checked build, `llama.cpp-20261112-dd755a09c-pascal-p106-v4-b12246-c805.tar.gz`: 678,327,506 bytes, md5
`00dd0b7eac7a0530aca4198e15461c5b` (`TESTS=1`, CUDA 12.9 image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61, 0 AVX
instructions, highest glibc symbol GLIBC_2.38, `test-backend-ops` in `bin/`, md5 `9892ee34e95403cd290f721e7e3601b8`).
Every check above ran on it; the merge `d569b8bc2` has the same tree.

## The rig run (R126)

[run/rig_c805.sh](run/rig_c805.sh), on one P106-090 card, with that tarball and the harness above:

- `bitid`: `multitoken_bitid q3k` and `glu` under the preset's k-column switches: the controls without the switch
  (`q3k kcol`, `glu off`: R120a's 25 of 91 and 42 of 668 must differ again), `=1` and `=2`, with C726's Q3_K kernel
  (`_Q3K_MT=1`, and `_R=2`), with C757's launch, with all six verify-pass switches, without the budget, with the default
  table, at a 24 KB budget with `=2` (Flash-Next's 6,144-wide Q3_K in three launches of 2 tokens at 6), and `=1` with
  the fault (the split products must differ). 0 comparisons may differ in the 12 others;
- `perf`: `test-backend-ops perf` of Flash-Next's Q3_K 6,144 → 2,560 at 1-8 tokens, arms `off` (the preset: llama.cpp's
  kernel at 6-8 tokens), `x1` (the 48 KB shape), `x2` (two launches within 32 KB), and the three with C726's kernel on,
  two passes.

It decides: exact on the card or not (any difference outside the controls and the fault is a fault to find first), and
the 48 KB shape's time at 6-8 tokens against the declined path's and against the split, each pass by value.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every verdict line of the checks (row, check, verdict, detail) |
| [pick_counts.tsv](pick_counts.tsv) | per switch-on setting: product lines changed, changes not of the expected kind, the four types' products not taken, launches past 48 KB or not covering the tokens, split products |
| [pick_changes.tsv](pick_changes.tsv) | every changed product line under the preset with `=1` and `=2`, and on the default rule with `=1`: the base's choice and the new one (launches x tokens, R, iterations per tile, shared memory, blocks) |
| [pick_r120a_products.tsv](pick_r120a_products.tsv) | R120a's differing products at 1-8 tokens under the preset: the base's choice, `=1`'s and `=2`'s |
| [pick_arms.txt](pick_arms.txt) | the 24 switch settings of the kernel-choice check |
| [run/checks.sh](run/checks.sh) | the driver: every step above |
| [run/kcol_pick.cu](run/kcol_pick.cu) | the k-column kernel's choices of one tree, without a GPU |
| [run/rig_c805.sh](run/rig_c805.sh) | R126's run on a rig |

## Commands

On the build machine, with this repository at `$REPO`, a llama.cpp clone with the fork as remote `fork` at `SEED_SRC`,
the base's unpacked rig tarball, its build objects and its static CPU build:

```bash
M=<work folder> REPO=$REPO LOCK=<flock file> BASE_TREE=<base rig tree> BASE_OBJ=<base build>/ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir \
BASE_CPU=<base CPU bin/> SEED_SRC=<clone> bash docs/data/2026-11-12_c805_kcol_exact/run/checks.sh
# once BRANCH is merged into pascal-p106-v4:
M=<work folder> REPO=$REPO LOCK=<flock file> SEED_SRC=<clone> STEPS=merged MERGED_STEPS=same bash docs/data/2026-11-12_c805_kcol_exact/run/checks.sh
```

On the rig (R126), with the tarball unpacked at `$BUILD` and the harness at `$BITID`:

```bash
BIN=$BUILD/bin OUT=<dir> BITID=$BITID setsid -f nohup bash docs/data/2026-11-12_c805_kcol_exact/run/rig_c805.sh > console.txt 2>&1 < /dev/null
```

## What it decides

- `pascal-p106-v4` holds the fix behind `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT`, off by default; with nothing set it runs
  `d097d6d88`'s code, and its GPU code is `d097d6d88`'s whatever is set.
- With `=1` under the preset, Flash-Next's Q3_K 6,144 → 2,560 at 6-8 tokens runs on the k-column kernel within 48 KB,
  and so does every product of R120a's harness that differed; whether that makes `multitoken_bitid q3k` and `glu` exact
  on a P106, and what it costs against the declined path, is R126.

Provenance and licence: `test-llama-archs`' random-weight models; the kernel-choice check uses no data. Everything here
is this repository's MIT licence.
