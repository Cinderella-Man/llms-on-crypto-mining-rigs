# The KV-only draft pass with the k-pool tables on the cards (C748, Q1320, 2026-11-11)

What this is: a fix on the v4 line for two switches that meet in the MTP draft's catch-up passes. C620's
`LLAMA_MTP_KV_ONLY=1` ([2026-11-11_c620_draft_kv_catchup](../2026-11-11_c620_draft_kv_catchup/README.md)) cuts a draft
pass with no output row (the catch-up over a prompt chunk, or over a verify step's accepted tokens) down to the writes
of the draft layer's caches. C100's `LLAMA_QSA_KPOOL_DEV=1` ([2026-11-11_c100_kpool_on_cards](../2026-11-11_c100_kpool_on_cards/README.md))
keeps the QSA k-pool tables on each card and sends each micro-batch only the rows that changed. On the merged build
`pascal-p106-v4-b12246-night` (`c408dabcb`, [2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)) the
cut pass leaves the card's copy of the tables out of its graph. C100 then cannot tell what the card holds, so the pass
after it sends both tables whole. The result is still exact, but at 49,152 tokens that is a 296,448-byte update on the
draft's card after every catch-up, where a step needs 3,072 bytes. C720's `LLAMA_QSA_KPOOL_STAGE=1`
([2026-11-11_c720_kpool_small_inputs](../2026-11-11_c720_kpool_small_inputs/README.md)) did not reach the cut pass
either: it still sent its four small inputs as four copies.

The fix is on the fork branch `pascal-p106-v4-b12246-c748` (`5b329df16`, two commits on `-night`): the cut pass updates
the card's tables the way the full pass does, and with C720's switch it reads its small inputs from that update. With
every switch off it computes what `-night` computes. This page is the change, the CPU check and the rig run still to do
([Q1320](../../../quants/experiments.md)).

## Result

**The KV-only pass now leaves the tables on the cards exactly as the full pass leaves them: 1 full rewrite and 574 rows
uploaded per run with the pass cut, as with the whole pass, where `-night`'s code made 10 full rewrites and 718 rows.
With C720's stage the cut pass sends one copy a micro-batch instead of four. The draft's caches and drafts stay the
same bytes, and every switch off gives `-night`'s logits.** The checks, on a CPU ([summary.tsv](summary.tsv),
[kvonly.tsv](kvonly.tsv), [overlap.tsv](overlap.tsv)):

- **`test-llama-archs --mtp-kv-only`, 7 switch sets × 3 seeds × 5 configurations: 105 of 105 pass** with the branch's
  CPU build, and the 21 runs pass with the rig build's binary on the CPU too. In each run the draft's state after the
  prompt, every draft and the end state are the same bytes with the pass cut and whole. With the tables on the cards the
  test now also requires the same full rewrites and rows uploaded both ways (the table below). It passes with C100's
  `LLAMA_QSA_KPOOL_DEV_CHECK=1`, which compared 12 of 12 micro-batches per run, byte for byte: the inputs, the copies
  against the host's tables before each update, and again after it. It also passes with `=2` (the draft's tables only)
  and with the four QSA switches of the merged build on together (C100, C720, C719's mask on the cards with its check,
  C704's indexer rows).
- **The same test without the fix fails where it should.** Commit `7b9dabf66` (the test alone, on `-night`) fails all
  72 runs with the tables on the cards and a cut pass, on the tables only ("the KV-only passes left the k-pool tables on
  the devices behind"). Its states and drafts are still identical. It passes its 33 other runs: tables on the host, or
  a `dist` sampler, which keeps the whole pass.
- **`test-llama-archs --mtp-catchup-overlap`** (C703's driver with the target, one configuration with the cut pass):
  45 of 45 pass, with no switch, with the tables (C100 and C720, with the check) on both contexts, and with the
  draft's tables only.
- **With every switch off, the branch gives `-night`'s logits** (`test-llama-archs -b CPU -s 1`, `c408dabcb`'s CPU build
  writing): 146 of 146 random models give the same logits, and qwen4exp gives them with `LLAMA_QSA_KPOOL_DEV=1
  LLAMA_QSA_KPOOL_STAGE=1`, with C100's check, and with `=2`. **ctest: 70 of 70 pass** (static CPU build; the tests
  that need a download, the network or torch are left out, as in C565).
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the tree's
  `bin/`): `llama.cpp-20261111-5b329df16-pascal-p106-v4-b12246-c748.tar.gz`, 656,346,233 B, md5
  `ec5a39d26fcb597526a85abb63320b96`.

The counts per run (the test's 100-token prompt in three calls, a draft step of two passes, the step's three accepted
tokens caught up, a last draft pass; micro-batches of 18 tokens, so 9 catch-up micro-batches). The figures are the
same at seeds 1, 2 and 3 and in the four configurations that cut the pass:

| Switches | Build | Full rewrites, whole / cut | Rows uploaded, whole / cut | Copies in the 9 catch-ups, whole / cut | Their bytes, whole / cut |
|---|---|---:|---:|---:|---:|
| `LLAMA_QSA_KPOOL_DEV=1` | `-night` (`7b9dabf66`) | 1 / 10 | 574 / 718 | 63 / 36 | 44,092 / 2,024 |
| `LLAMA_QSA_KPOOL_DEV=1` | `-c748` | 1 / 1 | 574 / 574 | 63 / 45 | 44,092 / 29,672 |
| + `LLAMA_QSA_KPOOL_STAGE=1` | `-night` (`7b9dabf66`) | 1 / 10 | 574 / 718 | 18 / 36 | 43,560 / 2,024 |
| + `LLAMA_QSA_KPOOL_STAGE=1` | `-c748` | 1 / 1 | 574 / 574 | 18 / 9 | 43,560 / 30,376 |
| the four QSA switches | `-night` (`7b9dabf66`) | 1 / 10 | 574 / 718 | 18 / 36 | 31,200 / 2,024 |
| the four QSA switches | `-c748` | 1 / 1 | 574 / 574 | 18 / 9 | 31,200 / 30,376 |

- **Full rewrites and rows** come from the draft memory (`llama_kpool_dev_stats`). On `-night` every micro-batch after
  a cut one starts the tables over: 9 extra full rewrites in the run. A cut pass reads no table, so the update it builds
  goes nowhere; the first whole pass after a stretch of cut ones sends all rows. In this run that happens twice, after
  the prompt and after the step's catch-up, which makes 144 extra rows.
- **Copies and bytes** are the k-pool host inputs that an op of a catch-up micro-batch reads, through views. A card
  receives each one as a copy. On `-night` a cut pass read the four small inputs (`k_idxs`, `new_pool_idxs`,
  `new_pool_rep`, `new_pool_pos`): 4 copies a pass, 2,024 B over the nine. On `-c748` it also reads the tables' update,
  padded to a decode step's 128 rows (3,072 B). Without the stage that makes 5 copies a pass. With it, 1 copy that holds
  the update and the small inputs (2,728 B over the nine passes, `tail_idxs` included, which the cut pass does not
  read). The whole pass, for comparison, takes 7 copies a micro-batch with the tables alone and 2 with the stage (the
  update and the pool mask), as C720 found.
- **At this size the bytes do not show the saving:** a full rewrite of the test's 25 pools (64 padded rows) fits in a
  decode step's 128-row update. At the rig's depths it does not (next section).

## What it changes on the rig (est., from the sizes)

In a speculative step the draft context catches up the accepted tokens with a cut pass, then drafts with whole passes.
On `-night` with both switches on, the step's first draft pass carries both tables whole:
`kpool_pad(n_pool)` rows of 24 bytes (the row's number, its cell and its 4 member cells), with four-token pools.
On `-c748` the catch-up and the draft pass each carry a 128-row update:

| Context depth | Pools | `-night`: the step's first draft pass | `-c748`: catch-up + first draft pass | Est. at 186.5 MB/s (R64's link speed) |
|---:|---:|---:|---:|---:|
| 16,384 | 4,096 | 4,160 rows, 99,840 B | 2 × 3,072 B | 0.54 ms against 0.03 ms |
| 49,152 | 12,288 | 12,352 rows, 296,448 B | 2 × 3,072 B | 1.59 ms against 0.03 ms |

The large update also changes the draft graph's shape every step (an update of 12,352 rows, then 128), so on `-night`
the step's second draft pass cannot reuse the first one's graph and builds its own. The est. leaves out that host
time, and the per-copy fixed cost that the stage saves in the catch-up (3 copies). Only the draft's tables are involved
(the target never runs a cut pass), so `LLAMA_QSA_KPOOL_DEV=2` shows the same effect as `=1`.

## The change

In `src/models/qwen4exp.cpp`, `build_qsa_kv` (the QSA part of the cut pass) now opens as `build_qsa_sel` (the whole
pass) does:

- **`build_kpool_dev`** when the tables are on the cards. The first QSA layer on each card reads the update's stage whole
  (`kpool_stage`, placed on that card) and writes the changed rows into the card's copy with `SET_ROWS`. The graph lists
  the copy as updated, so C100's commit after the pass marks it current. No op of the cut pass reads the tables. They
  are kept up to date so that the next micro-batch sends only its changed rows.
- **`build_kpool_small`**: with `LLAMA_QSA_KPOOL_STAGE` the indexer key's `SET_ROWS`, the new pools' `GET_ROWS`, their
  `ROPE` and the pooled keys' `SET_ROWS` read the casts of the staged inputs instead of the host's tensors; without the
  switch it returns the host's tensors, as before.
- **Without `LLAMA_QSA_KPOOL_DEV`** neither function adds a node, and the cut pass's graph is `-night`'s, node for node.
  The four k-pool inputs that had no name are now named (`kpool_upd`, `kpool_pool_cells`, `kpool_pool_idxs`,
  `kpool_pool_mask`) so the test can follow them; names only.
- **The test** (`tests/test-llama-archs.cpp`, `--mtp-kv-only`) records, with the tables on the cards, the draft
  memory's full rewrites, rows uploaded and checked micro-batches after each run, and through its eval callback the
  `kpool_*` host inputs each catch-up micro-batch's ops read, keyed by the memory's micro-batch count. A run fails when
  the cut run's full rewrites or rows differ from the whole run's. The test went in first (`7b9dabf66`) so that it could
  run on the code without the fix.

The cut pass gains the update's nodes: the test's 9 catch-up micro-batches take 594 graph nodes with the tables
against 531 without (7 more a micro-batch) and 657 with the stage (14 more); whole, they take 1,539 and 1,620.

## What the rig run must show (R91)

The CPU check covers the values and the counts; on the cards it covers neither the update inside the cut pass (C100's
`SET_ROWS` into an I32 copy, first run on a card in R74) nor the time. R91 ([Q1323](../../../quants/experiments.md))
runs one deep-lane run ([bench/deep_lane](../../../bench/deep_lane/README.md)) with the c748 tarball, after C620's and
C100's own rows (R76, R74):

| Load | Arm | What it shows |
|---|---|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` | the control, and the warm-up |
| `kv` | + `LLAMA_MTP_KV_ONLY=1` | C620 alone |
| `dev` | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` | C100 and C720 alone |
| `kvdev` | both | the step and prompt at 16,384 and 49,152 against `kv`'s and `dev`'s: do the two stack |
| `check` | `kvdev` + `LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` | no abort; at exit the draft memory's line (`k-pool tables on 1 devices: … N full`) counts full rewrites at the load's sequence edits only, not one a step |

Right when `check` logs no difference and its full rewrites stay far below its steps, and the depth-0 reply hashes of
all five loads match. The verdict is `kvdev` against `kv` and `dev` at 16,384 and 49,152, classed against the spread
(gain, negligible or loss).

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the c748 tarball's bin/ on the rig; MODEL,
# MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse kv dev kvdev check" \
  ARM_ENV_kv="LLAMA_MTP_KV_ONLY=1" \
  ARM_ENV_dev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1" \
  ARM_ENV_kvdev="LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1" \
  ARM_ENV_check="LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  LOADS="warm:sparse kv:kv dev:dev kvdev:kvdev check:check" DEPTHS="0 16384 49152" WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect
grep -a 'k-pool tables on' $OUT/server-check.log
```

## Files

| File | What it holds |
|---|---|
| [run/checks.sh](run/checks.sh) | the whole check: the CPU builds, the KV-only and overlap tests, the identity against the base, ctest, the rig tarball (through C620's `checks.sh`) and the rig binary's KV-only runs |
| [summary.tsv](summary.tsv) | one verdict a check (82 lines; the work folder written as `<work>`, the base's binaries as `BASE_CPU`) |
| [kvonly.tsv](kvonly.tsv) | `--mtp-kv-only`, one row per build (`tip`, `before`, `rig`), switch set, seed and configuration: identity, graph nodes, full rewrites, rows, catch-up micro-batches, copies and bytes, status |
| [overlap.tsv](overlap.tsv) | `--mtp-catchup-overlap`, one row per switch set, seed and configuration |

## Commands

On a build machine with Docker, CPU only, every build and test under one lock (here shared with other sessions' checks)
at `nice -n 19` and `--cpu-shares=256`:

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_CPU: bin/ of -night's (c408dabcb) static native CPU
# build of test-llama-archs; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c748 BEFORE=7b9dabf66 BASE_CPU=$BASE_CPU \
  LOCK=$LOCK JOBS=8 DOCKER_ARGS='--label pascal-quants=1' \
  bash docs/data/2026-11-11_c748_kvonly_kpool/run/checks.sh
```

The first run's rig build failed at once: the copy of `scripts/build_llamacpp_docker.sh` on the build machine had lost
its executable bit. Its three failed lines (`build-rig`, `build-extra`, `kvonly-rig`) were set aside, and
`STEPS='build kvrig'` ran again; every other step ran once.

Provenance and licence: everything here was generated by the commands above from the fork's code (MIT, as llama.cpp)
on random models that `test-llama-archs` builds; no model file, dataset or recorded text is involved.
