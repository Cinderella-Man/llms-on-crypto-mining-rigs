# The v4 kernel merge on a P106, exactness (R120a, Q1391, 2026-11-12)

What this is: the native check of [the v4 kernel merge](../2026-11-11_v4_kernel_merge/README.md) on one P106-090 card.
`pascal-p106-v4` at `d097d6d88` holds three read-once kernel branches, each behind its own switch, off by default:
C727's Q6_K products of 2-4 tokens (`GGML_CUDA_MMVQ_PASCAL_Q6K_MT`), C728's K tiles (`_DMT_KT`) and C729's products of
5-8 tokens (`_RO8`). The same build holds the day merge's three other verify-pass kernels, never run on a P106 until now:
C726's Q3_K products of 2-8 tokens ([`_Q3K_MT`](../2026-11-11_c726_q3k_multitoken/README.md)), C757's one launch for the
shared expert's gate, up and SWIGLU ([`_GLU_MT`](../2026-11-11_c757_shexp_fused/README.md)) and C754's F32 products of
4-8 tokens on `mul_mat_vec_f` ([`GGML_CUDA_MMVF_F32_MT`](../2026-11-11_c754_f32_mmvf/README.md)). R120b stacks all six in
one served verify pass, so all six are checked here, each alone and together. Registry
[Q1391](../../../quants/experiments.md) (the merge), and Q1285, Q1333 and Q1330 for the other three.

The rig run of [run/rig_kern.sh](../2026-11-11_v4_kernel_merge/run/rig_kern.sh) with `STEPS='replay bitid'`, on card 0
of a nine-card P106-090 rig with nothing else running, with the checked `-kern` rig tarball
(`llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern.tar.gz`, md5 `37a4c3731f76455dce14013210b2fac7`):

- **replay**: the recorded set `v4kern` (696 `test-backend-ops` cases with their CPU references, 47 runs,
  `v4kern.tar` 4,074,956,800 bytes, md5 `902506fe67896f898f0f58c5674546d6`) through
  [scripts/replay_checks.sh](../../../scripts/replay_checks.sh): C727's 13 runs, C728's 12 and C729's 13 as their
  branches recorded them, and 9 runs with the three switches on together; 13:07:27-13:37:17;
- **bitid**: [multitoken_bitid](../../../bench/mmvq_pascal/verify/multitoken_bitid.cpp) built against the merged tree
  (md5 `a37dab2167cdb7426f7e23e5b70585f5`), every product of n tokens against the n one-token products of the same
  columns, bit for bit: 54 arms, 13:37:17-13:42:41. Each branch's own arms, every mode with the three merged switches on
  together, then the `q3k`, `glu` and `f32` modes with the arms of their own rows (R103, R107, R99) and with all six
  switches on, each with its fault;
- **diag** ([run/r120a_diag.sh](run/r120a_diag.sh), 13:45:52-13:46:18): four more `multitoken_bitid` arms that tell where
  the `q3k` and `glu` differences come from (below).

## Result

**C727, C728, C729 and C754 are exact on a P106, alone and together. C726's and C757's kernels are exact on every
product they run. The products that differ are the ones they leave to the k-column kernel, and those differ the same
way with every switch off.** The fault is in the k-column path, older than the six switches: a product of several tokens
whose activations do not fit the k-column kernel's shared-memory budget runs on another kernel, whose sums are not the
one-token sums. Among Flash-Next's own products this hits one: Q3_K's 6,144 → 2,560 product at 6, 7 and 8 tokens, under
the release preset's `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`. The preset's verify passes are at most 4 tokens
(`--spec-draft-n-max 3`), so the release line never runs that product.

### Replay: 47 of 47 runs as expected

| Runs | Cases a run | Result |
|---|---:|---|
| C727's `q6kmt`, 12 passing runs | 139 | 139/139 each |
| C728's `dmtkt`, 11 passing runs | 345 | 345/345 each |
| C729's `ro8`, 12 passing runs | 265 | 265/265 each |
| the three on together, 6 settings (`kern-all`, `-nopf`, `-kt2`, `-m2`, `-r2`, `-defaults`) | 696 | 696/696 each |
| faults alone: `q6kmt-fault`, `dmtkt-fault`, `ro8-fault` | 139, 345, 265 | 42, 17 and 198 cases fail |
| faults with the three on: `kern-q6k-fault`, `kern-kt-fault`, `kern-ro8-fault` | 696 | 42, 17 and 198 cases fail |

Every passing run passed every case, 12,819 case replays in all; each fault fails the same number of cases alone and with
the other two switches on ([replay.tsv](replay.tsv)). About one small case in 20 was also checked against the rig's own
CPU.

### Bit identity: the three merged switches and C754's, 0 differences

| Mode | Products of | Arms that must match | Comparisons | With differences | Faults and controls (must differ) |
|---|---|---|---:|---:|---|
| `q6k` | C727, Q6_K at 2-4 tokens | `on`, `nopf`, `r2`, `nw1`, `kcol`, `all` | 6 × 18 | 0 | `fault` 18 of 18, `all-fault` 18 of 18 |
| `dmtkt` | C728's tiles, IQ4_XS and Q4_K at 2-4 tokens | `dmt`, `kt1`, `kt2`, `kt2m2`, `kt2nopf`, `kt2nw4`, `all`, `all-kt2` | 8 × 78 | 0 | `fault` 11 of 78, `all-fault` 11 of 78 |
| `dmt` | C722, IQ4_XS and Q4_K at 2-4 tokens | `all` | 60 | 0 | |
| `dmt8` | C729, IQ4_XS and Q4_K at 5-8 tokens | `on`, `nopf`, `mt`, `all` (13 products declined each) | 4 × 67 | 0 | `control` 80 of 80, `fault` 67 of 67, `all-fault` 67 of 67 |
| `q5k` | C690, Q5_K at 2-4 tokens | `all` | 21 | 0 | |
| `q5k8` | C729, Q5_K at 5-8 tokens | `on`, `nopf`, `pf`, `mt`, `all` (7 declined each) | 5 × 33 | 0 | `control` 40 of 40, `fault` 33 of 33 |
| `f32` | C754, F32 at 2-8 tokens | `on`, `on2`, `all6` | 3 × 42 | 0 | `control` (cuBLAS) 30 of 42, `all6-fault` 30 of 42 |

28 arms, 1,372 comparisons, 0 with differences; all 11 faults and controls differ and exit 1. `all` is the three merged
switches on with C722's and C690's (`_DMT=1 _Q5K_MT=1`); `all6` adds `_Q3K_MT=1 _GLU_MT=1` and `GGML_CUDA_MMVF_F32_MT=1`.
Every arm and its switches: [bitid.tsv](bitid.tsv).

### Bit identity: `q3k` and `glu`

| Mode | Arm | Switches beyond the preset's k-column ones | With differences |
|---|---|---|---:|
| `q3k` | `kcol` | none: `_Q3K_MT` off | 25 of 91 |
| `q3k` | `on`, `nopf`, `pad0` | `_Q3K_MT=1`, `=2`, `=1 _PAD=0` | 15 of 91 each |
| `q3k` | `r2` | `_Q3K_MT=1 _Q3K_MT_R=2` | 18 of 91 |
| `q3k` | `all6` | all six | 15 of 91 |
| `q3k` | `fault`, `all6-fault` | `_Q3K_MT_FAULT=1` | 83 of 91 each |
| `glu` | `off` | `_DMT=1 _DMT_R=1 _Q5K_MT=1`, `_GLU_MT` off | 42 of 668 |
| `glu` | `on`, `nopf`, `np1` | + `_GLU_MT=1`, `=2`, `=1 _NP=1` | 42 of 668 each |
| `glu` | `all6` | all six | 24 of 668 |
| `glu` | `fault`, `all6-fault` | `_GLU_MT_FAULT=1` | 618 and 614 of 668 |

Every product that differs with C726's or C757's switch on also differs with it off, with the same count of differing
values: all 15, 15, 18, 15 and 15 products of `on`, `nopf`, `r2`, `pad0` and `all6` are among `kcol`'s 25, and all 42,
42, 42 and 24 of `glu`'s `on`, `nopf`, `np1` and `all6` among `off`'s 42 (every product and its counts:
[diffs.tsv](diffs.tsv)). The kernels' own products are exact: C726's kernel leaves a product to the k-column kernel
where fewer than two warps' staging fit 48 KB beside the n tokens' activations, and no product it takes differs; C757's launch matches the one-token products wherever it runs (its `GLU/apart` differences at 2-3 tokens are the
separate products', the reference).

The differing products are wide rows of few tokens:

- `q3k`, switch off (`kcol`): the harness's 64 × 8,192 from 2 tokens, 77 × 4,096 and 96 × 12,288 from 3 tokens,
  129 × 2,048 from 6 tokens, and Flash-Next's 2,560 × 6,144 at 6, 7 and 8 tokens (11,570 of 15,360, 13,753 of 17,920
  and 15,716 of 20,480 values; largest relative difference 4.18e-03, 5.04e-05 and 4.19e-04). With the switch on, the
  products C726's kernel takes drop out of that list. Flash-Next's other Q3_K products (2,560 → 10,240, 6,144, 12,288
  and 640) are exact at 2-8 tokens in every arm but the faults.
- `glu`: only the harness's 33 × 8,192 width, for every pair with Q3_K at 2-8 tokens and Q2_0 with Q2_0 at 6-8 tokens.
  Flash-Next's 15 shared-expert type pairs at 2,560 → 640 are exact in every arm but the faults.

### Where the differences come from: the k-column kernel's budget

The k-column kernel stages one iteration of every token's activations in shared memory. With
`GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM` set, as the Flash-Next preset sets it (32768), it takes the launch shape of before Q823
within that budget, and where one iteration of n tokens does not fit, it declines the product, which then runs on another
kernel. Without the switch, its default rule tries a third of the SM's shared memory (32 KB on a P106) and then the 48 KB
a block may hold, and with `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` (as in the preset) declines only past that. The
diagnostic arms ([diag.tsv](diag.tsv)):

| Arm | Switches | With differences |
|---|---|---:|
| `q3k kcol-nobudget` | `_KCOL_TABLE=0`, no budget | 12 of 91 (25 at 32 KB) |
| `q3k on-nobudget` | `_KCOL_TABLE=0 _Q3K_MT=1` | 9 of 91 (15 at 32 KB) |
| `glu on-nobudget` | `_KCOL_TABLE=0`, the read-once kernels, `_GLU_MT=1` | 31 of 668 (42 at 32 KB) |
| `q3k kcol-48k` | `_KCOL_SMEM=49152 _KCOL_TABLE=0` | 12 of 91 |

Without the budget, Flash-Next's 2,560 × 6,144 is exact at 6-8 tokens, and so are the widths whose 6-8 tokens fit 48 KB
(K = 2,048 and 6,144); the 9 to 12 products left are the harness's widths whose tokens' activations pass 48 KB
(K = 4,096 and 12,288 from 6 tokens, K = 8,192 from 3). Every one of them differs with `_Q3K_MT` off as well
([diffs.tsv](diffs.tsv), run `diag`). So the products that differ are exactly the ones the k-column kernel declines, at
its 32 KB budget under the preset and at 48 KB without it.

## What it decides

- **C727, C728 and C729 are exact on the card, each alone and the three together** (the replay and the bit identity).
  R117's, R118's and R116's replay and bit identity are done here. **C754** is exact in the bit identity, alone and
  with all six on (its own replay set `f32x` is R99's and did not run here).
- **C726 and C757 add no difference of their own**, alone or with all six on. The `q3k` and `glu` arms differ only where
  the products go to the k-column kernel and that kernel declines them. Their own replay sets (`q3kmt`, `glumt`) are
  R103's and R107's and did not run here.
- **At the release preset's verify passes (at most 4 tokens) every Flash-Next product these six switches touch is
  exact on a P106**, so R120b's served pair can stack all six.
- **The fault to fix before verify passes of 6 or more tokens** (a draft of 5 or more, C797): Q3_K's 6,144 → 2,560
  product at 6-8 tokens is not the one-token sums under the preset's 32 KB budget. Registry Q1411: make every k-column
  product of 2-8 tokens give the one-token sums, by keeping the default shape rule's 48 KB step for products the budget
  declines and by staging the activations in K tiles past 48 KB (as C728 does for C722's kernel), or by giving C726's
  kernel such tiles.

## Files

| File | What |
|---|---|
| [replay.tsv](replay.tsv) | each replay run: switches, what passing means, cases passed, FAIL lines, exit status, verdict |
| [bitid.tsv](bitid.tsv) | each of the 54 bit-identity arms: mode, switches, comparisons, with differences, products the read-once kernel declined, exit status, what passing means, verdict |
| [diag.tsv](diag.tsv) | the four diagnostic arms, the same columns |
| [diffs.tsv](diffs.tsv) | every product with differences in every arm of both runs (faults and controls included): types, tokens, rows, width, values that differ, largest relative difference |
| [summary.txt](summary.txt) | the run's summary lines |
| [run/r120a_diag.sh](run/r120a_diag.sh) | the diagnostic arms, on the rig |
| [run/r120a_tables.sh](run/r120a_tables.sh) | the tables from the two runs' output folders |

The rig script is the merge folder's [run/rig_kern.sh](../2026-11-11_v4_kernel_merge/run/rig_kern.sh) (the `q3k`, `glu`
and `f32` arms were added to its bit-identity list for this run).

## Commands

On the rig, with the `-kern` tarball unpacked at `$BUILD`, the set at `$SET` and the harness at `$BITID`:

```bash
STEPS='replay bitid' BIN=$BUILD/bin OUT=$OUT SET=$SET REPLAY=scripts/replay_checks.sh BITID=$BITID \
  setsid -f nohup bash docs/data/2026-11-11_v4_kernel_merge/run/rig_kern.sh > console.txt 2>&1 < /dev/null
BITID=$BITID OUT=$DIAG bash docs/data/2026-11-12_r120a_kern_exact/run/r120a_diag.sh
```

The tables, on any machine, from the two output folders:

```bash
OUT=$OUT DIAG=$DIAG DEST=docs/data/2026-11-12_r120a_kern_exact bash docs/data/2026-11-12_r120a_kern_exact/run/r120a_tables.sh
```

The set and the harness come from the merge's checks ([its commands](../2026-11-11_v4_kernel_merge/README.md#commands));
the harness is built with [build_multitoken_bitid.sh](../../../bench/mmvq_pascal/verify/build_multitoken_bitid.sh)
against the merged rig tree.

Provenance and licence: `test-backend-ops`' random tensors and `multitoken_bitid`'s random weights and inputs from fixed
seeds; no model file. Everything here is this repository's MIT licence.
