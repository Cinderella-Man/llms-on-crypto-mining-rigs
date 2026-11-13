# C728's K tiles at 5 to 8 tokens (C794, Q1389, 2026-11-13)

What this is: the read-once kernels' K tiles (C728) for products of 5 to 8 tokens, behind a new switch
`GGML_CUDA_MMVQ_PASCAL_RO8_KT`, off by default. Verify passes of five or more tokens come from longer and adaptive
drafts (C815, FN82, C797). In Qwen3.8-Flash-Next their 2,560 × 6,144 products (`ssm_out` and `attn_output`, IQ4_XS and
Q4_K) are the ones C729's read-once kernel handles worst:

- at 5 tokens its whole staging of the activations holds 3 warps a block (IQ4_XS) or 2 (Q4_K);
- from 6 tokens it holds one warp or none, and those products go back to the k-column kernel, whose sums are not the
  one-token kernel's.

The tiles hold 6 warps a block at every count from 5 to 8, with each token's sums the one-token kernel's bit for bit.
This folder holds the design, the CPU checks (all passed) and the rig run, R149 (Q1455): on a P106 the tiles are bit
for bit the one-token products and a gain at every count from 5 to 8. The switch is in
[docs/switches.md](../../switches.md). The fork branch is `pascal-p106-v4-b12246-c794`, merged off by default into
`pascal-p106-v4`.

## The change (`ggml/src/ggml-cuda/mmvq-pascal-dmt.cu`)

C728's kernel `mul_mat_vec_q_pascal_dmt_kt<type, NT, M, PF>` stages one K tile at a time: one iteration's L
super-block positions of every token. At 2–4 tokens a block's shared memory holds three things: the tile, a raw copy of
it (each token's `block_q8_1` run, loaded with coalesced 128-bit loads) and the warps' weight staging. At 8 tokens and
K = 6,144 (L = 8) that is too much for 6 warps:

| Shared memory at 8 tokens, K = 6,144 | IQ4_XS | Q4_K |
|---|---|---|
| the tile | 18,592 B | 20,672 B |
| the raw tile | 18,432 B | 18,432 B |
| 6 warps' weight staging | 26,112 B | 27,648 B |
| total with a separate raw tile (2–4 tokens' layout) | 63,136 B | 66,752 B |
| total with the raw tile over the staging (5–8 tokens) | 44,704 B | 48,320 B |

So at 5 to 8 tokens:

- **The raw tile lies over the warps' weight staging.** Both are live at different times: the raw tile only between its
  store and its conversion into the tile, the staging only during a warp's products. One more barrier a tile (3 in
  all) makes every warp finish its products of the last tile before the raw tile is stored over its segments. A block
  then holds 6 warps at every count from 5 to 8 (2 blocks, 12 warps a SM), at both types.
- **A thread loads up to `dmt_kt_eu<NT>()` 16-byte units of the next tile ahead:** 4, 5, 6 and 6 at 5 to 8 tokens,
  enough for 6 warps at L = 8. With fewer warps (`_DMT_KT_NW` below 6) the tiles decline those products.
- **At 7 and 8 tokens the tiles run without the register prefetch of the weights** (`DMT_KT_PF_COLS` = 6). With it,
  five kernels passed the 168 registers that 2 blocks of 6 warps allow and spilled 8 to 40 bytes to the stack: Q4_K at 7
  and 8 tokens at M = 1 and 2, IQ4_XS at 8 tokens and M = 2.
- **The values are unchanged.** Each lane adds the same super-blocks in the same order with the same step functions,
  from the same staged values, and the lanes' sums are added with the same shuffles. So each token's sums are the
  one-token kernel's, as at 2–4 tokens.
- **The 2–4-token kernels are untouched:** the same code as the base in SASS and PTX (below).

The host rule (`dmt_kt_nw`, with `GGML_CUDA_MMVQ_PASCAL_RO8` on):

- `_RO8_KT=1` sends a 5–8-token IQ4_XS or Q4_K product to the tiles where whole staging holds at most 3 warps a block,
  or one warp, or none, and the tiles more. That is C808's limit (`DMT_KT_FULL_MAX`); on a P106 the tiles gained only
  from 3 warps down (R120b). At Flash-Next's widths that means K = 6,144 at every count from 5 to 8; the 2,560-wide rows
  keep whole staging (4–6 warps).
- `_RO8_KT=2` sends them wherever the tiles fit, for the checks.
- `ggml_cuda_mmvq_pascal_dmt_takes` (the k-column kernel's gate) now also takes the products only the tiles run: C809's
  one-warp rule declines them unless the tiles take them. With the switch off it is unchanged.
- `_DMT_KT_M` (default 2), `_DMT_KT_NW` and `_DMT_KT_FAULT` apply; `_DMT_KT` (2–4 tokens) does not.
- `RO8=2` turns the prefetch off here too.

| Flash-Next 2,560 × 6,144 at | 5 tokens | 6 | 7 | 8 |
|---|---|---|---|---|
| IQ4_XS, whole staging (C729) | 3 warps | 1 (left to the k-column kernel, C809) | k-column | k-column |
| IQ4_XS, tiles (`_RO8_KT=1`) | 6 warps | 6 | 6, no prefetch | 6, no prefetch |
| Q4_K, whole staging (C729) | 2 warps | k-column | k-column | k-column |
| Q4_K, tiles (`_RO8_KT=1`) | 6 warps | 6 | 6, no prefetch | 6, no prefetch |

`test-backend-ops`: C728's K-tile cases (333 and 4,097 rows of 6,144, and 129-row views of 6,656-wide rows) now run up
to 8 tokens.

## The checks (build machine, CPU only, no GPU)

Every build and test ran in a container without GPUs, one at a time under the build lock, at `--cpu-shares=256` and
nice 19. The driver is [run/checks.sh](run/checks.sh); its rows are in [summary.tsv](summary.tsv). Base: `pascal-p106-v4`
`2144cf760`. Its rig build is C822's `cc97bb6b6`, the same tree. Its CPU code is the C813 merge `65ac9a47d`'s, whose
static build wrote the reference logits; C822 changed CUDA files and tests only.

| Check | Result |
|---|---|
| `mmvq-pascal-dmt.cu` compiled alone, tip against base (sm_61 SASS and compute_61 PTX) | 97 functions: 73 the base's code, 24 new tile kernels, nothing else |
| the 24 new kernels (registers, stack, local memory) | 112–168 registers, no stack, no local memory; 3 barriers a tile |
| rig tarball | `llama.cpp-20261113-c4e82a856-pascal-p106-v4-b12246-c794.tar.gz`, 682,729,820 B, md5 `e01a333ae37284969076f3b9cea01c58`, 0 AVX, GLIBC_2.38; `llama-server` md5 `4ce942095a768ee00797a3a90bdb5e30` |
| `llama-server` SASS and PTX against the base's | 8,564 functions: 8,540 the base's code, the 24 new tile kernels |
| the rig build's `mmvq-pascal-dmt.cu.o` against the base build's | 73 the same, 24 new, in SASS and PTX; `mmvq-pascal.cu.o` (the one-token kernels) the same SASS |
| replay set `ro8kt` (`scripts/record_checks.sh`) | 197 cases (179 linked from the `v4kern` set, 18 new: the K-tile cases at 6–8 tokens), 16 runs, one the fault; `ro8kt.tar` 695,306,240 B, md5 `6e36bb00a844dff3a8b389825293d556` |
| `multitoken_bitid` (modes `kt8`, `dmtkt`, `dmt8`), linked against the rig build's tree | md5 `5ae2dedf20c1f8a40ad3c3b1353a47c7`; it needs a GPU, so R149 runs it |
| static CPU build of the tip | OK |
| `test-llama-archs -b CPU -s 1` against the base's logits | 146 of 146 the same, with no switch, `_RO8=1 _RO8_KT=1`, `_RO8=1 _RO8_KT=2`, and with `_DMT=1 _DMT_KT=1 _RO8=1 _RO8_KT=1` |
| ctest, static and shared (C565's `cpu_tests.sh`) | 72 of 72 tests passed static and shared (the network test, the 146 archs files and the hand tests too) |
| the merge `80c94fe0d` on `pascal-p106-v4` `2a3d6bf0d` (C793's and C820's merges) | adds exactly `-c794`'s diff; `mmvq-pascal-dmt.cu` the tip's; static CPU build OK; 146 of 146 models the same as the base with no switch and with every tile switch on; pushed after the check |

The register table of every tile kernel, with instruction counts beside C722's kernels, is `obj/sass_counts.tsv`
(`run/checks.sh` step `obj`; the rig build's in `sass_counts.tsv`).

The first run of the chain stopped after its `obj` step: five kernels at 7 and 8 tokens with the prefetch spilled
(Q4_K M = 1 and 2 at 7 and 8 tokens: 8, 32, 24 and 40 bytes of stack; IQ4_XS at 8 tokens, M = 2: 8 bytes). The fork's
second commit leaves the prefetch out there, and every row above is from the second run.

## The rig run (R149)

**The tiles are a gain at every count from 5 to 8 tokens, for both types, and bit for bit the one-token products.** On
`v4-rc2` with `_RO8_KT=1`, a 2,560 × 6,144 IQ4_XS product of 5 tokens takes 144.89 and 146.50 µs (two passes) against
166.79 and 168.47 for C729's whole staging (`kern`). At 8 tokens it takes 231.28 and 233.29 µs against 364.43 and 369.48
µs. For a verify pass's est. 17 IQ4_XS and 15 Q4_K such products, the tiles save est. 0.620 and 0.625 ms at 5 tokens
and est. 5.258 and 5.340 ms at 8. M = 1 (`ktm1`) is slower than the default M = 2 at every count from 5 to 8, except
IQ4_XS at 8 tokens, where the two are within the spread.

R149 (Q1455) ran on a P106 rig on 2026-11-13, card 0, with no model. The rig was idle: no llama process, 0 MiB on
every card before and after. Perf ran 20:25:33–20:27:08 and bitid 20:27:35–20:28:17.

- **Build:** `v4-rc2` (`pascal-p106-v4` `eb82916d3`), which holds C794 (merged as `80c94fe0d`, switch off by default).
  Its tree is C828's tarball's: `bin/llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`, `test-backend-ops`
  `e38809f0b16869abc065483f07c4798a`, checked on the rig before and after the run.
- **Harness:** `multitoken_bitid` built against `v4-rc2`'s tree (C828's, md5 `284b3baa0dcbb8e1649819eea2c37551`,
  94,451,944 B, the md5 equal at both ends of the copy). This branch's own harness (md5
  `5ae2dedf20c1f8a40ad3c3b1353a47c7`) links ggml statically, so it would have tested `c4e82a856`'s kernels instead of the
  release candidate's.
- **No replay:** R153 replayed the set `ro8kt` with `v4-rc1`'s build on 2026-11-13: 16 of 16 runs as expected
  ([../2026-11-13_c831_upstream/](../2026-11-13_c831_upstream/README.md)). `v4-rc2` adds only C828's host code to
  `v4-rc1`, so C794's sm_61 code is the same in both.
- **Commands:** `STEPS=perf`, then `STEPS=bitid`, of [run/rig_c794.sh](run/rig_c794.sh) with `BIN` set to `v4-rc2`'s
  `bin/`; tables by [run/tables.sh](run/tables.sh) and [run/r149_diffs.sh](run/r149_diffs.sh) on the build machine.

**Bit identity** ([rig/bitid.tsv](rig/bitid.tsv)). `multitoken_bitid kt8` makes 104 products of 5 to 8 tokens. 17 of
them are shapes the read-once kernel declines (K = 8,192 and 12,288 at both types, and Q4_K 77 × 4,096 at 8 tokens), so
they are not counted. The other 87 are each compared with their columns' one-token products:

| Arm | Switches (and the preset's k-column ones) | Exit | Compared | With differences |
|---|---|---:|---:|---:|
| `kt1` | `_RO8=1 _RO8_KT=1` | 0 | 87 | 0 |
| `kt2` | `_RO8=1 _RO8_KT=2` | 0 | 87 | 0 |
| `kt2m1` | `_RO8=1 _RO8_KT=2 _DMT_KT_M=1` | 0 | 87 | 0 |
| `kt2nopf` | `_RO8=2 _RO8_KT=2` (no prefetch) | 0 | 87 | 0 |
| `fault` | `_RO8=1 _RO8_KT=2 _DMT_KT_FAULT=1` | 1 | 87 | 56 |
| `control` | `_RO8=1`, no tiles, `BITID_ALL=1` | 1 | 104 | 35 |

The fault is caught on all 56 tile products (K = 2,560 and 6,144). The control's 35 differences all lie on products
the read-once kernel declines without the tiles: there the k-column kernel's sums are not the one-token kernel's.

**Perf** ([rig/perf.tsv](rig/perf.tsv), [rig/diffs.tsv](rig/diffs.tsv), [rig/us_run.tsv](rig/us_run.tsv)): µs a
2,560 × 6,144 product, passes 1 and 2:

| Type | Tokens | `off` | `kern` | `kt` (M = 2) | `ktm1` (M = 1) |
|---|---:|---:|---:|---:|---:|
| IQ4_XS | 4 | 241.05, 226.60 | 112.82, 112.47 | 111.81, 112.05 | 111.41, 111.73 |
| IQ4_XS | 5 | 254.50, 255.33 | 166.79, 168.47 | 144.89, 146.50 | 152.15, 152.82 |
| IQ4_XS | 6 | 273.25, 275.74 | 274.67, 276.44 | 166.18, 167.05 | 175.04, 175.61 |
| IQ4_XS | 7 | 324.06, 324.85 | 318.47, 325.04 | 202.50, 204.34 | 215.23, 215.05 |
| IQ4_XS | 8 | 368.86, 370.79 | 364.43, 369.48 | 231.28, 233.29 | 232.77, 233.03 |
| Q4_K | 4 | 230.75, 232.22 | 140.38, 141.57 | 140.85, 142.25 | 140.87, 141.72 |
| Q4_K | 5 | 265.91, 265.66 | 168.59, 170.14 | 152.08, 153.38 | 161.20, 162.56 |
| Q4_K | 6 | 327.54, 331.85 | 330.44, 334.99 | 174.75, 176.43 | 189.23, 189.27 |
| Q4_K | 7 | 383.09, 386.36 | 382.61, 387.12 | 210.56, 212.51 | 226.04, 226.00 |
| Q4_K | 8 | 440.07, 445.96 | 439.82, 443.88 | 240.16, 242.21 | 247.39, 247.36 |

At 4 tokens the tiles do not run, so `kern`, `kt` and `ktm1` run the same kernel there: the control. At 6 to 8 tokens
`kern` runs the k-column kernel, as `off` does, because whole staging holds one warp or none.

**A verify pass** ([rig/verify.tsv](rig/verify.tsv), [rig/verify_diffs.tsv](rig/verify_diffs.tsv)): est. ms of a
pass's est. 17 IQ4_XS and 15 Q4_K products, passes 1 and 2:

| Tokens | `off` | `kern` | `kt` | `ktm1` | `kt` − `kern` | `ktm1` − `kt` |
|---:|---:|---:|---:|---:|---:|---:|
| 4 | 7.559, 7.335 | 4.024, 4.036 | 4.014, 4.039 | 4.007, 4.025 | −0.010, +0.003 | −0.006, −0.013 |
| 5 | 8.315, 8.326 | 5.364, 5.416 | 4.744, 4.791 | 5.005, 5.036 | −0.620, −0.625 | +0.260, +0.245 |
| 6 | 9.558, 9.665 | 9.626, 9.724 | 5.446, 5.486 | 5.814, 5.824 | −4.180, −4.238 | +0.368, +0.338 |
| 7 | 11.255, 11.318 | 11.153, 11.332 | 6.601, 6.661 | 7.050, 7.046 | −4.552, −4.671 | +0.449, +0.384 |
| 8 | 12.872, 12.993 | 12.793, 12.939 | 7.534, 7.599 | 7.668, 7.672 | −5.258, −5.340 | +0.134, +0.073 |

**Verdicts**, against the spread of each arm's two passes:

- **`kt` against `kern` and `off`: a gain** at 5, 6, 7 and 8 tokens, at both types. `kt` − `kern` is −21.90, −21.97
  µs (IQ4_XS, 5 tokens) and −16.51, −16.76 (Q4_K, 5 tokens). At 6, 7 and 8 tokens it is −108.49, −109.39;
  −115.97, −120.70; −133.15, −136.19 µs (IQ4_XS) and −155.69, −158.56; −172.05, −174.61; −199.66, −201.67 (Q4_K).
  `kern`'s own two passes lie 1.68, 1.77, 6.57 and 5.05 µs apart (IQ4_XS, 5 to 8 tokens) and 1.55, 4.55, 4.51 and
  4.06 (Q4_K).
- **At 4 tokens (the control): negligible**, as expected for one kernel: `kt` − `kern` −1.01, −0.42 µs (IQ4_XS) and
  +0.47, +0.68 (Q4_K); est. −0.010 and +0.003 ms a pass.
- **`ktm1` against `kern`: a gain** at 5 to 8 tokens at both types. **Against `kt`: a loss** at 5, 6 and 7 tokens at both
  types: `ktm1` − `kt` +7.26, +6.32; +8.86, +8.56; +12.73, +10.71 µs (IQ4_XS, 5, 6, 7 tokens) and +9.12, +9.18;
  +14.48, +12.84; +15.48, +13.49 (Q4_K). At 8 tokens Q4_K loses too (+7.23, +5.15); IQ4_XS is negligible (+1.49,
  −0.26; `kt`'s passes 2.01 µs apart). The default M = 2 stays.

The switch is not in the v4 release settings, because drafts of five or more tokens lost on agentic speed on
2026-11-13 (R154a). For a later preset whose verify passes reach five tokens, `_RO8_KT=1` (M = 2) is the setting to
take. Its served pair waits for Q1459, which looks into R133's finding that half of the served M = 2 tile launches at
4 tokens ran at the switch-off time.

### How the run is set up

[run/rig_c794.sh](run/rig_c794.sh), on one P106 card of an idle rig, needs only the tarball, the set and the harness, no
model. Default steps `replay bitid perf`:

- **replay**: the set `ro8kt`. Every run must be `ok`; `ro8kt-fault` must fail cases at K = 6,144.
- **bitid**: `multitoken_bitid kt8` compares every product of 5 to 8 tokens with its columns' one-token products, bit
  for bit, over dmtkt's shapes. It runs at `_RO8_KT=1`, `=2`, `=2` with `_DMT_KT_M=1`, and `=2` without the prefetch,
  each needing `0 with differences`. Two runs must differ: the fault (`_DMT_KT_FAULT=1`) and the control (`_RO8=1`
  without tiles, `BITID_ALL=1`: the k-column kernel's sums).
- **perf**: `test-backend-ops perf` at 2,560 × 6,144, IQ4_XS and Q4_K, 4 to 8 tokens, two passes of the arms:
  - `off` (the Flash-Next preset's k-column switches);
  - `kern` (+ `_DMT=1 _Q5K_MT=1 _RO8=1`, C729: whole staging where it fits);
  - `kt` (`kern` + `_RO8_KT=1`, M = 2 by default);
  - `ktm1` (`kt` with `_DMT_KT_M=1`). [R133](../2026-11-12_c808_ktiles_gain/README.md#the-rig-run-r133) found half of
    the served M = 2 tile launches at 4 tokens running at the switch-off time, not yet explained (Q1459).
- **Tables** (on the build machine, [run/tables.sh](run/tables.sh)):
  - `perf.tsv`: µs a product by value per pass, and `kt` against the faster of `off` and `kern`;
  - `verify.tsv`: est. ms of a verify pass's 2,560 × 6,144 products per token count and arm. It counts est. 17 IQ4_XS
    and 15 Q4_K products a pass, from R120b's served 4-token pass: IQ4_XS `ssm_out` 1.912 ms at 112.60 / 113.33 µs a
    product, Q4_K `ssm_out` and `attn_output` 1.552 + 0.567 ms at 142.29 / 141.75.

Verdict per token count: a gain where `kt` (or `ktm1`) is faster than `off` and `kern` beyond the two passes' spread.
Negligible within the spread. A loss where it is slower; that count then leaves the rule. A served pair waits for Q1459
and for a preset whose verify passes reach five tokens (the Miner Llama v3 preset verifies at most four).

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's row (step, check, verdict, detail), second run |
| [obj/sass_counts.tsv](obj/sass_counts.tsv) | registers, stack, local memory and instruction counts of every read-once kernel and tile kernel of the file compiled alone (tip) |
| [sass_counts.tsv](sass_counts.tsv) | the same from the rig build's object |
| [replay_runs.tsv](replay_runs.tsv) | the set `ro8kt`'s runs: name, switches, ops, filter, expected |
| [run/checks.sh](run/checks.sh) | the check driver |
| [run/rig_c794.sh](run/rig_c794.sh) | R149's run on the rig |
| [run/tables.sh](run/tables.sh) | R149's tables |
| [run/r149_diffs.sh](run/r149_diffs.sh) | R149's verdict tables: `kt` and `ktm1` against `kern`, `ktm1` against `kt`, in µs and est. ms a pass |
| [rig/bitid.tsv](rig/bitid.tsv) | R149: each `kt8` arm's switches, exit, declined, compared and differing products |
| [rig/us_run.tsv](rig/us_run.tsv) | R149: µs a product by arm, pass, type and token count (`rig_c794.sh`'s output) |
| [rig/perf.tsv](rig/perf.tsv), [rig/verify.tsv](rig/verify.tsv) | R149: `tables.sh`'s per-type table and est. ms a verify pass |
| [rig/diffs.tsv](rig/diffs.tsv), [rig/verify_diffs.tsv](rig/verify_diffs.tsv) | R149: `r149_diffs.sh`'s differences and `kern`'s pass spread |

Also changed: `scripts/record_checks.sh` (suite `ro8kt`), `bench/mmvq_pascal/verify/multitoken_bitid.cpp` (mode `kt8`).

## Commands

```bash
# on the build machine, the fork clone and the base's build at hand:
M=<work dir> REPO=<this repository> LOCK=<flock file> BASE_COMMIT=2144cf760 BASE_TREE=<the base's unpacked rig tarball> \
  BASE_OBJ=<its build's ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir> BASE_CPU=<bin/ of a static CPU build of the base> \
  SEED_SRC=<a llama.cpp clone with the fork as remote "fork"> LINK_FROM=<the v4kern set's folder> \
  bash docs/data/2026-11-13_c794_ktiles_5to8/run/checks.sh
# the merge, checked before it was pushed (the merge commit put into $M/main/rig/src as refs/c794/merge):
STEPS=merged FINAL_REF=refs/c794/merge <the same env> bash docs/data/2026-11-13_c794_ktiles_5to8/run/checks.sh
# R149, on the rig (card 0):
BIN=<tarball>/bin OUT=<out> BITID=<multitoken_bitid> SET=<ro8kt folder> REPLAY=scripts/replay_checks.sh \
  setsid -f nohup bash docs/data/2026-11-13_c794_ktiles_5to8/run/rig_c794.sh > console.txt 2>&1 < /dev/null
# R149 as run (v4-rc2's tree; the harness built against it; no replay):
#   STEPS=perf BIN=<tree>/bin OUT=<out-perf> setsid nohup bash docs/data/2026-11-13_c794_ktiles_5to8/run/rig_c794.sh ...
#   STEPS=bitid BIN=<tree>/bin BITID=<multitoken_bitid> OUT=<out-bitid> setsid nohup bash docs/data/2026-11-13_c794_ktiles_5to8/run/rig_c794.sh ...
# its tables, on a build machine:
IN=<out>/perf/us_run.tsv OUT=<tables dir> bash docs/data/2026-11-13_c794_ktiles_5to8/run/tables.sh
IN=<out>/perf/us_run.tsv OUT=<tables dir> bash docs/data/2026-11-13_c794_ktiles_5to8/run/r149_diffs.sh
```

## Provenance and licence

The tables were made by the scripts above from the fork's code (MIT, as llama.cpp) and synthetic test inputs
(`test-backend-ops`, `multitoken_bitid`); no model or dataset is involved. Licence of this folder: the repository's.
