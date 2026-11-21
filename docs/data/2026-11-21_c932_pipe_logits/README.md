# Pipelined decoding against plain decoding on the CPU: a fault in Flash-Next's fused conv window (C932, Q1670; 2026-11-21)

What this is: a correctness hunt on a build machine's CPU. Pipelined decoding sends guessed tokens through the cards
while earlier ones finish; it must give exactly the tokens and logits of plain decoding.
[C880](../2026-11-21_c880_batch_api/README.md#a-finding-on-the-side) found that on the v5 line it no longer did for a
random Flash-Next-type model (`qwen4exp`). Its tokens matched, but the logits, the KV state and the probe all differed.
This folder finds when that changed and why, and tests the fix. The fix is fork branch `pascal-p106-v5-pipe-logits`
(`80d83fa1a`, one commit on `pascal-p106-v5` `3dfad876f`). Registry row Q1670 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **It was a real fault, not rounding.** Flash-Next's delta-net layers keep a short history of their last inputs (the
  conv window), one copy per state group. In pipelined decoding each pipe slot works in a group of its own. The fused
  conv window (the default since 2026-11-03) always wrote the new history to group 0, the group of plain decoding.
  Each slot then read its own group's history, which nobody had updated, so every pipelined token after the first one
  used a stale history. Turning the fused window off (`LLAMA_GDN_FUSE=0`) gives every hash the same again, on the
  merge where it started and on today's line.
- **When:** the merge of 2026-11-05 that brought pipelined decoding for the delta-net models into the v3 line
  (`e3b30303d`). Each side was right alone. The pipelining branch wrote the history to the slot's group, but only in
  the unfused code. The release line had meanwhile gained the fused window, whose write had no group. Q550's tip
  (`f377294d6`) and the branch as merged (`8c25952e4`) give every hash the same. The merge and every later commit
  checked differ: the v3.0 and v4.0 tags and the v5 line's four merges. No published check after the merge ran
  pipelined decoding on a Flash-Next model until C880.
- **Who it touches:** only Flash-Next-type models (`qwen4exp`) in pipelined decoding (`--pipe-depth`,
  `llama_pipe_*`). No release preset runs Flash-Next pipelined: its preset drafts with its MTP head. Flint's random
  model gives every hash the same at every commit checked; Flint has no recurrent state, and its v5 preset pipelines.
  The other delta-net types (`qwen35`, `qwen35moe` and `qwen3next`: Qwen3.5/3.6, Ornith) give every hash the same.
  The card path takes its write address from the same graph copy, so the P106 build should have the same fault; the
  P106 check confirms it.
- **How large: it broke Flash-Next's pipelined replies.** On the real file (Swift GSQ-RCO Q2_0, on the CPU) the base
  build's pipelined jobs all left plain decoding's reply within its first words. The text then garbled, and six of the
  seven jobs stopped early, after 23, 52, 32, 16, 23 and 23 tokens, where plain decoding wrote 64. The oracle, which
  feeds back plain decoding's own tokens as guesses, had 8 of 63 accepted. With the fix all seven jobs give plain
  decoding's 64 tokens, logits, state and probe. Random models hide the fault: their logits are flat and tiny (at most
  0.228), it moved them by at most 3.027e-05, and no greedy token changed in 195 jobs of 512 tokens.
- **The fix** gives the fused window's copy the slot's group, as the unfused code does (6 lines in
  `src/models/qwen4exp.cpp`). Plain decoding's group is 0, so plain decoding builds the same graph as before. On the
  CPU every pipelined check now gives plain decoding's tokens, logits, state and probe. That holds for the depths, the
  guessers, the dropped guesses, the race and the two streams. The rest of the suite is the base's: ctest test by test
  (71 of 75 pass; the four known failures), the logits of `test-llama-archs`' 148 random models bit for bit, and the
  nine hand-run tests.
- **On a P106:** not run here (the rigs were off). The check is its own row; `run/p106_check.sh` has the steps
  (below).

## When it changed

`check_at.sh` builds a commit's pipelined tools on the CPU and makes random models with that commit's own
`test-llama-archs` (seed 1, 4,096-token vocabulary). It then runs `bench/pipespec/check_identity.sh` (greedy, 32 tokens,
depth 1, 0 and 50 % wrong guesses: seven pipelined jobs against plain) and `llama-pipe-recurrent-check`. Full table:
[checks.tsv](checks.tsv).

| Commit | What it is | Flash-Next type (`qwen4exp`) | Flint type (`flint`) |
|---|---|---|---|
| `f377294d6` | the pipelining branch's tip, as Q550 checked it on 2026-11-03 | same | same |
| `8c25952e4` | that branch as merged into the v3 line (the merge's second parent) | same | same |
| `e3b30303d` | the merge, 2026-11-05 (its first parent has the fused conv window and no pipelined `qwen4exp`) | **DIFF** (7 of 7 jobs; recurrent check from token 2) | same |
| `e3b30303d` with `LLAMA_GDN_FUSE=0` | the merge, unfused conv window | same | – |
| `1551f5fe1` | v3.0 tag | **DIFF** | same |
| `eb82916d3` | v4.0 tag | **DIFF** | same |
| `1f1110c67` | v5 line, upstream b12410 merged | **DIFF** | same |
| `602691153` | v5 line, upstream b12483 merged | **DIFF** | same |
| `6644c2bc6` | v5 line, the five contributions merged | **DIFF** | same |
| `3dfad876f` | `pascal-p106-v5`, the base | **DIFF** | same |
| `3dfad876f` with `LLAMA_GDN_FUSE=0` | the base, unfused conv window | same | – |
| `80d83fa1a` | the fix | same | same |

On the base, the other delta-net types (`qwen35`, `qwen35moe`, `qwen3next`) give every hash the same in both checks.
No bisect was needed. Every point from v3.0 on differs, so the search moved to the merge that brought the pipelining
in. Q550's checks were made on that branch before the merge.

## What it was

`llm_graph_context::build_rs` reads each cell's recurrent rows from the group that holds the cell's newest state.
`llama_memory_recurrent` keeps that group per cell and switches it when a pipe slot writes. So every write must go to
the group of the ubatch being decoded: its pipe slot's group, or 0 for `llama_decode`.

- The delta-net state is written by `build_recurrent_attn`, with the group.
- The unfused conv window (`build_conv_state_at` → `build_conv_state_store`) writes with the group.
- The n-gram embedding's conv history (also `build_conv_state_at`) writes with the group.
- **The fused conv window (`build_gdn_conv_at`, C20/Q438, `LLAMA_GDN_FUSE` unset or 1) wrote at `kv_head` rows, which
  is group 0.**

A pipelined slot so read its own group's history (still the one from before) and wrote group 0's. That fits every
symptom C880 saw.

- Depth 0 already differs: one slot, no guesses.
- The recurrent check's first difference is its second generated token. The first comes from the prompt; the second
  is the first token a slot decodes after another slot's write.
- Pipelined jobs without wrong guesses agree with each other: they read the same stale rows in the same order.
- Jobs that drop guesses differ again: a dropped guess moves the cell back to another group.

The CUDA backend's `gdn_conv` cache fusion takes its destination from the same `ggml_cpy`, so the P106 build writes
the same wrong rows; the P106 check confirms it on the cards.

## How large, and whether it changed tokens

**On random models (the base, `3dfad876f`).** Two measurements, each with the five random Flash-Next models of seeds 1
to 5 and three prompts: 15 runs of each.

- `llama-pipe-recurrent-check` with `run/prc_measure.patch`: 512 greedy tokens through three pipe slots, with the
  forced rollbacks ([prc.tsv](prc.tsv)).
- `check_identity.sh`: 512 tokens, depths 1 and 3, 0 and 50 % wrong guesses, 13 pipelined jobs a run
  ([long.tsv](long.tsv)).

The yardstick for a changed order of sums is the same 512 tokens decoded again as one batch. It splits into
micro-batches, takes other matrix kernels and the delta-net's chunked form, and so gives the size of honest rounding.

| Seed | Largest absolute logit difference, pipelined against plain (rewrite, code, chat) | The yardstick, one batch against one token at a time (largest of the three) | Largest logit |
|---|---|---|---|
| 1 | 2.903e-05, 2.612e-05, 2.139e-05 | 4.470e-08 | 0.213 |
| 2 | 2.980e-08, 2.980e-08, 2.980e-08 | 4.470e-08 | 0.211 |
| 3 | 1.816e-06, 1.784e-06, 2.347e-06 | 4.470e-08 | 0.217 |
| 4 | 1.985e-05, 1.801e-05, 2.321e-05 | 5.960e-08 | 0.228 |
| 5 | 2.579e-05, 2.563e-05, 3.027e-05 | 5.588e-08 | 0.189 |

- In every run 510 of the 512 steps differ, from the second generated token on.
- On four of the five models the difference is far above the yardstick: 649, 52.5, 389 and 542 times its largest
  (seeds 1, 3, 4 and 5). On seed 2 it is below the yardstick: that model's conv kernel passes little of the history on.
- The per-element relative difference (the table's `max rel`) reaches 1.998. It means nothing here: these logits sit
  near 0, so two tiny values of opposite sign give a relative difference near 2. The absolute difference against the
  largest logit is the honest scale: at most 1.6e-04 of it (seed 5).
- **No greedy token changed on a random model.** In the 15 recurrent-check runs (7,680 steps) the greedy choice was
  the same at every step. In the 15 identity runs at the base, all 195 pipelined jobs gave plain decoding's 512 tokens,
  through 7,778 wrong guesses dropped again. These models' logits are flat (the largest in any run 0.228, the smallest
  gap between the two best tokens under 0.00005), and the fault moved them by at most 3.027e-05.
- With the fix, all 15 recurrent-check runs show 0 differing steps and all 195 identity jobs are the same in every
  hash, as are Flint's 195 jobs on the same seeds and prompts.

**On the real file.** Swift GSQ-RCO Q2_0 (the maker's three shards, the file Q550 used), on the build machine's CPU
with 8 threads. `check_identity.sh` with the model's chat template, the `code` prompt (73 tokens), greedy, 64 tokens,
depth 1, 0 and 50 % wrong guesses: plain against seven pipelined jobs, once with the base, once with the fix
([real_file.tsv](real_file.tsv)). Plain decoding gives the same hashes on both builds (tokens `fc5830f6e0f13ec0`,
logits `2e281cae6c9c8bbf`).

| Job | Base: tokens generated, guesses accepted | Base: verdict | Fix: tokens generated, guesses accepted | Fix: verdict |
|---|---|---|---|---|
| pipe-d0 (no guesses) | 23 | DIFF | 64 | same |
| oracle, 0 % wrong | 64, 8 of 63 | DIFF | 64, 62 of 63 | same |
| oracle, 50 % wrong | 52, 2 of 51 | DIFF | 64, 38 of 63 | same |
| copy2 | 32, 0 of 2 | DIFF | 64, 20 of 27 | same |
| cont4 | 16, 1 of 6 | DIFF | 64, 26 of 43 | same |
| ngram-simple | 23, 0 of 0 | DIFF | 64, 16 of 18 | same |
| ngram-map-k | 23, 0 of 0 | DIFF | 64, 16 of 17 | same |

Plain decoding's reply (both builds) and the base's pipe-d0 reply begin:

```text
plain:   We need answer user: "Write a Python class implementing an LRU cache with a per-entry time-to-live, ...
pipe-d0: We need answer user: user wants a)\n- **` — exact')` — need to be <|im_end|>
```

The base's pipelined reply follows plain decoding for its first four words and then turns to noise; the garbled
replies end early at an end token. With the oracle, the stale history makes the model reject 54 of the 63 tokens of
plain decoding it was offered. So every Flash-Next run with `--pipe-depth` since 2026-11-05 decoded with stale
histories. Q550's numbers of 2026-11-03 came before the merge and are not affected.

## The fix's checks

All on the CPU of a build machine, against the base `3dfad876f`, with nothing set. Run by `run/head.sh`.

| Check | Result |
|---|---|
| `check_at.sh`: random Flash-Next and Flint models, identity and recurrent check | same in all four (the base: Flash-Next DIFF) |
| CPU build of every target | 0 errors |
| ctest, 75 tests | 71 pass; fail: `test-kv-cache` (aborted), `test-jinja`, `test-jinja-py`, `test-kv-cache-scales`, the base's four known ones |
| `test-llama-archs -b CPU -s 1` plain, write, compare | 153/153 each; 148 logits files written and compared, all the same |
| the 148 logits files against the base's | 148 of 148 the same bits |
| hand-run tests (9) | all pass |
| C880's `tools.sh` with the fix's build on C880's random models, against C880's base outputs ([tools.tsv](tools.tsv)) | all 35 runs exit 0 (the base: the 5 pipelined checks exit 1, the recurrent check 2). The 6 plain reference lines are the base's; the 50 pipelined result lines now equal plain decoding's; the other 66 comparisons are the base's: 29 exit codes, 29 logs line for line (screen-check, `test-moe-ids`, `test-hop-cast`, the pool tests), 7 stream summary lines and `test-hop-cast`'s table |
| rig build (`TESTS=1`, `test-llama-archs` and `test-cuda-keepalive` added, web UI `b12483`) | built: `llama.cpp-20261121-80d83fa1a-pascal-p106-v5-pipe-logits.tar.gz`, md5 `91569ac26e57b1086c4cf1c2b8747844`, 0 AVX instructions; 38 warning lines, the base's rig build's, none new |

## The P106 check (its own row, not run here)

`run/p106_check.sh` runs on the rig, with the fix's rig build and the base's, and nothing is timed.

1. **Random models on card 0.** Flash-Next and Flint random models (made by the fix's `test-llama-archs`), 96 tokens,
   depths 1 and 3, 0 and 50 % wrong guesses. Three arms: the base, the base with `LLAMA_GDN_FUSE=0` (the control), and
   the fix.
2. **Flint's release file (Flint Pascal Q8 GGUF, `p065`) on all nine cards.** The v5 preset's split and batch sizes, its chat template, greedy, 256
   tokens, depth 4, with both builds.
3. **Swift Flash-Next's file** on all nine cards, as Q550 ran it (128 tokens, depths 1 and 3), with both builds.

Expected:

- Flint the same everywhere, and the fix's plain hashes the base's.
- Flash-Next with the fix and with the control the same as plain.
- Flash-Next with the base: DIFF on the random model, as on the CPU. On the real file it shows whether the fault
  changed the cards' tokens.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check of every commit: stage, commit, what it is, model, check, result, the pipe-d0 job's four columns and the count of differing jobs, or the recurrent check's line |
| [prc.tsv](prc.tsv) | the size of the differences: per build, random model (seed), prompt: the recurrent check's verdict, steps that differ, the first, the largest absolute and relative logit difference, the largest logit, the smallest top-2 margin, greedy tokens that differ; the same for the yardstick (the tokens decoded again as one batch) |
| [long.tsv](long.tsv) | `check_identity.sh` at 512 tokens, depths 1 and 3, 0 and 50 % wrong guesses, per build, model, seed and prompt: pipelined jobs, jobs that differ, jobs with other tokens, wrong guesses |
| [real_file.tsv](real_file.tsv) | Swift GSQ-RCO Q2_0 on the CPU, plain against pipelined, the base and the fix: each job's tokens generated, guesses, accepted, wrong, and its verdicts (from `check_identity.sh`'s two tables) |
| [tools.tsv](tools.tsv) | C880's `compare.py` of C880's base outputs against the fix's |
| [cpu_summary.tsv](cpu_summary.tsv) | `cpu_tests.sh`'s verdicts for the fix: build, ctest, archs, ident against the base, hand-run tests |
| `run/check_at.sh` | one commit: build the pipelined tools, make the random models, run both checks; exit 0, 1 or 125 for `git bisect run` |
| `run/sweep.sh` | the commits in order, then (when a good and a bad point are found) `git bisect run` on first-parent and into a merged branch |
| `run/head.sh` | the fix's checks: `check_at.sh`, `cpu_tests.sh`, C880's `tools.sh` and `compare.py`, the size measurements, the rig build |
| `run/prc_measure.patch` | `llama-pipe-recurrent-check` prints the size of each difference (`PRC-PIPE`), step by step with `PRC_STEPS=1`, and the yardstick with `PRC_BATCH=1` (`PRC-BATCH`); the tool's verdict is unchanged |
| `run/real_file.sh` | the real file on the CPU, plain against pipelined, per build |
| `run/checks_table.py` | `checks.tsv` from `check_at.sh`'s `verdicts.tsv` |
| `run/p106_check.sh` | the P106 check (above) |

The CPU builds use `docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh`; the tool runs use C880's
`run/tools.sh` and `run/compare.py`.

## Commands

On the build machine, each in a container (`--cpu-shares=256 --memory=16g --network none`), builds under one lock with
8 jobs; `SRC` and `SEED` are a llama.cpp clone with the fork as remote `fork`, `REPO` a folder with this folder's
`run/` files, `cpu_tests.sh`, C880's `tools.sh` and `compare.py`, `scripts/build_llamacpp_docker.sh` and
`bench/pipespec/` as `pipespec/`:

```bash
# the commits in order (the sweep), then the pipelining branch and its merge
W=<work> SRC=<clone> REPO=<scripts> LOCK=<lock file> JOBS=8 bash run/sweep.sh
W=<work> SRC=<clone> REPO=<scripts> LOCK=<lock file> STEPS=points POINTS='f377294d6 8c25952e4 e3b30303d' bash run/sweep.sh
# the controls with the unfused conv window
for c in e3b30303d 3dfad876f; do W=<work> SRC=<clone> REPO=<scripts> STAGE=fuse-off CHECK_ENV=LLAMA_GDN_FUSE=0 \
  ARCHS=qwen4exp MISSES='0 0.5' PRC=1 bash run/check_at.sh $c; done
# the other delta-net types on the base
W=<work> SRC=<clone> REPO=<scripts> STAGE=other-hybrids ARCHS='qwen35 qwen35moe qwen3next' MISSES='0 0.5' PRC=1 \
  bash run/check_at.sh 3dfad876f
# the fix: its checks, the size measurements and the rig build
W=<work> SEED=<clone> REPO=<scripts> MD=<C880's models> BASE_LOGITS=<C880's base archs/logits> BASE_TOOLS=<C880's tools/base> \
  LOCK=<lock file> JOBS=8 bash run/head.sh
python3 -I run/checks_table.py <work>/verdicts.tsv checks.tsv
```

```bash
# the real file, plain against pipelined, the base and the fix (each build's bin/<commit>/ from check_at.sh)
W=<work> MODEL=<Swift GSQ-RCO Q2_0 shard 1> REPO=<scripts> BUILDS='base:3dfad876f head:80d83fa1a' bash run/real_file.sh
```

## Provenance and licence

The models are random weights from the fork's `test-llama-archs` (MIT, as llama.cpp), and the Swift GSQ-RCO Q2_0 file
is the maker's. The prompts are the project's own (`bench/pipespec/prompts/`). Tables and scripts: this repository's
licence.
