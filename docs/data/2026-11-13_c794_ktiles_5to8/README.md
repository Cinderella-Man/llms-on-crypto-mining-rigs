# C728's K tiles at 5 to 8 tokens (C794, Q1389, 2026-11-13)

What this is: the read-once kernels' K tiles (C728) for products of 5 to 8 tokens, behind a new switch
`GGML_CUDA_MMVQ_PASCAL_RO8_KT`, off by default. Verify passes of five or more tokens come from longer and adaptive
drafts (C815, FN82, C797). In Qwen3.8-Flash-Next their 2,560 × 6,144 products (`ssm_out` and `attn_output`, IQ4_XS and
Q4_K) are the ones C729's read-once kernel handles worst:

- at 5 tokens its whole staging of the activations holds 3 warps a block (IQ4_XS) or 2 (Q4_K);
- from 6 tokens it holds one warp or none, and those products go back to the k-column kernel, whose sums are not the
  one-token kernel's.

The tiles hold 6 warps a block at every count from 5 to 8, with each token's sums the one-token kernel's bit for bit.
This folder holds the design, the CPU checks (all passed) and the rig run, R149 (Q1455), which is not run yet. The
switch is in [docs/switches.md](../../switches.md). The fork branch is `pascal-p106-v4-b12246-c794`, merged off by
default into `pascal-p106-v4`.

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

## The rig run (R149, not run)

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
# its tables, on a build machine:
IN=<out>/perf/us_run.tsv OUT=<tables dir> bash docs/data/2026-11-13_c794_ktiles_5to8/run/tables.sh
```

## Provenance and licence

The tables were made by the scripts above from the fork's code (MIT, as llama.cpp) and synthetic test inputs
(`test-backend-ops`, `multitoken_bitid`); no model or dataset is involved. Licence of this folder: the repository's.
