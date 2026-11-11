# Switches for two defaults upstream b12246 brought (C693, Q1226, 2026-11-11)

What this is: two switches on the v4 line, one for each of the two changes that upstream llama.cpp b12246 made to how
the rig runs Flash-Next, which until now had no way to be turned off. The v4 line merged b12246 in C662
([2026-11-10_ladder6](../2026-11-10_ladder6/README.md#what-upstream-changed-by-default)). Miner Llama keeps or parks
every change by its measured effect on the rig (a gain, negligible, or a loss), and a change that cannot be turned off
cannot be measured on its own.

| Switch | Default (upstream b12246) | Other value (the release's path, before b12246) |
|---|---|---|
| `LLAMA_MTP_H_DEV` | `1`: the target keeps the hidden rows that an MTP draft with its own memory (Flash-Next's) reads in a buffer on its output card, one row per batch position, and the draft's graph reads them there ([#31927](https://github.com/ggml-org/llama.cpp/pull/31927)); est. 80.0 MiB on card 8 for Flash-Next at `-b 2048` | `0`: the rows are copied to host memory after each decode and uploaded again by the draft. `LLAMA_MTP_CATCHUP_OVERLAP` needs the buffer, so it stays off with `0` |
| `GGML_SCHED_PREFETCH` | `1`: before a split runs, the scheduler starts copying to host memory the inputs of the next split that this split's card already holds, so that a CPU split after it waits for those copies and not for the card's whole split ([#30966](https://github.com/ggml-org/llama.cpp/pull/30966)) | `0`: each split's inputs are copied when that split runs |

`LLAMA_MTP_H_DEV=2` is a check value: as `1`, and the CPU also takes the buffer, which it otherwise refuses because it
has no events (the CPU finishes a decode before it returns, so it needs none). This is what lets a CPU-only build run
the device-row path and compare it with the host rows. With `LLAMA_UBATCH_TRACE=1` a scheduler logs its first early
fetch (`sched-trace: prefetch: split …`) and, when it is freed, how many inputs it fetched early. That shows on the rig
whether Flash-Next's graph has a split the early fetch can act on at all.

The switches are on the fork branch `pascal-p106-v4-b12246-c693` (`8d329fa0e`), two commits on
`pascal-p106-v4-b12246-c748` (`5b329df16`): `bcf34f122` the switches (`src/llama-context.cpp`, `ggml/src/ggml-backend.cpp`)
and `8d329fa0e` the test (`test-llama-archs --mtp-h-dev`). With both at their defaults the branch computes what `-c748`
computes. Both switches are read when a context (its scheduler) is made.

## Result

**With both switches at their defaults the branch gives `-c748`'s logits on 146 of 146 random models, and so does
every switch value. The MTP draft's caches, drafts and probabilities are the same bytes with the target's rows on the
host and in the buffer (45 of 45 configurations, and 15 of 15 with the rig build's binary), and llama-server gives the
same replies and draft counts with every value (9 of 9 arms). On a CPU the early fetch never acts, so whether either
switch changes the rig's speed is the rig row R92's question.** The checks, on a build machine's CPU
([summary.tsv](summary.tsv)):

- **`test-llama-archs --mtp-h-dev`, new** ([hdev.tsv](hdev.tsv)): C703's draft-mtp sequence with the target (a 100-token
  prompt in four calls, three draft steps that accept 0, 1 and all of the draft, a second turn of 45 tokens, a last
  draft step) on the random qwen4exp model with an MTP block, in five cache and sampler configurations. The reference
  keeps the rows on the host (`LLAMA_MTP_H_DEV=0`). Five more runs each compare against it: the default, `=2`, the
  default with `LLAMA_MTP_CATCHUP_OVERLAP=1`, `=0` with it, and `=2` with it. Each compares the draft's state after the
  prompt's second call, after the prompt and at the end, the driver's state at those points, and every draft with its
  probabilities, byte for byte. It also checks where the rows were (read back from the buffer only when they are in it)
  and how many prompt calls deferred their catch-up. All 45 configurations (3 switch sets × 3 seeds × 5) give identical
  bytes in every run, with the rows at `h h d d h d` (host for the reference, the default and `=0` with the overlap;
  the buffer for the rest) and the waits at `0 0 0 4 0 4`. On a CPU the default keeps the host rows, as `-c748`
  does. The three switch sets are none, `GGML_SCHED_PREFETCH=0`, and C100's k-pool tables on the cards with C720's
  stage and C100's check. The rig build's own `test-llama-archs`, run on the CPU against the stub driver: 15 of 15.
- **`test-llama-archs --mtp-catchup-overlap`** (C703's check, whose driver the new test extends): 15 of 15.
- **The logits** (`test-llama-archs -b CPU -s 1`, `-c748`'s CPU build writing): **146 of 146 models the same** with no
  switch set, with `LLAMA_MTP_H_DEV=0`, with `=2`, with `GGML_SCHED_PREFETCH=0`, and with both at `0`. This test runs
  no draft, and on a CPU there is no early fetch, so this shows only that the switches change nothing else.
- **llama-server's draft-mtp** ([server_arms.tsv](server_arms.tsv), [server_replies.tsv](server_replies.tsv),
  [run/server_smoke.py](run/server_smoke.py)): the tiny qwen4exp model with its MTP block, drafting three tokens a
  step, prompts read in calls of 64 tokens, three requests (greedy 96 tokens, temperature 1.0 seed 42 for 48 tokens,
  greedy 64 tokens on a longer prompt). `-c748`'s server is the reference. The branch's server runs with no switch,
  `LLAMA_MTP_H_DEV=0`, `=2`, `GGML_SCHED_PREFETCH=0`, both at `0`, and `LLAMA_MTP_CATCHUP_OVERLAP=1` alone, with `=0`
  and with `=2`. **9 of 9 arms give the reference's tokens and drafted and accepted counts on all three requests** (the
  random model's drafts are never accepted: 279, 135 and 183 drafted, 0 accepted, in every arm). In each arm the log
  shows the rows where its switches put them. The buffer line (`CPU h_nextn buffer size = 0.25 MiB`) appears with `=2`
  and with the overlap unless `=0`. The line `set_embeddings_nextn_dev: LLAMA_MTP_H_DEV=0, extracting the h_nextn rows
  to host memory` appears with `=0`. The default on a CPU logs that the device has no events, as `-c748` does.
- **ctest: 70 of 70 pass** (static CPU build; the tests that need a download, the network or torch are left out, as in
  C565).
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the tree's
  `bin/`): `llama.cpp-20261111-8d329fa0e-pascal-p106-v4-b12246-c693.tar.gz`, 656,350,199 B, md5
  `1a6a5002befb698fafa91b3963c0b726`.

**What the CPU does not cover.** The early fetch acts only where a split's backend reads asynchronously and has
events, which means a card. On a CPU build every split is the CPU's, so `GGML_SCHED_PREFETCH` changes nothing there,
and this check shows only that it changes no number. With the buffer, the rows reach the draft through a CUDA copy on
card 8 and are ordered by events. On the CPU they reach it through the same graph ops in a host buffer, with no events.
The CPU therefore shows that the two paths compute the same numbers, not how they behave on the cards. Neither
switch's speed is measured here.

## What the rig run must show (R92)

R92 ([Q1325](../../../quants/experiments.md)) runs one deep-lane run ([bench/deep_lane](../../../bench/deep_lane/README.md))
on Flash-Next's preset with the MTP draft, on this tarball, after R57 (`-ladder6` on a P106, which loads the buffer
first):

| Load | Arm | What it shows |
|---|---|---|
| `warm` | the preset (both switches at their defaults) | the warm-up |
| `base1`, `base2` | the preset | the control and its spread |
| `host1`, `host2` | `LLAMA_MTP_H_DEV=0` | the rows through the host, as the v3 release does |
| `nopf1`, `nopf2` | `GGML_SCHED_PREFETCH=0` | no early fetch |
| `trace` | `LLAMA_UBATCH_TRACE=1`, `-lv 4` | not timed: the load log's `h_nextn buffer size` line on card 8 (or the warning that the rows go through the host), and whether `sched-trace: prefetch:` lines appear at all |

Each arm is classed against the control at 0, 16,384 and 49,152 tokens: a gain, negligible or a loss beyond the
spread. If the trace load logs no early fetch, `GGML_SCHED_PREFETCH` does nothing on Flash-Next's split, and its class
is "no effect" there, whatever its timing shows. The reply hashes of `base`, `host` and `nopf` at depth 0 must match.

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the c693 tarball's bin/ on the rig; MODEL,
# MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 \
  ARMS="base host nopf trace" \
  ARM_ENV_host="LLAMA_MTP_H_DEV=0" \
  ARM_ENV_nopf="GGML_SCHED_PREFETCH=0" \
  ARM_ENV_trace="LLAMA_UBATCH_TRACE=1" ARM_ARGS_trace="-lv 4" \
  LOADS="warm:base base1:base host1:host nopf1:nopf base2:base host2:host nopf2:nopf trace:trace" \
  DEPTHS="0 16384 49152" WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect
grep -a 'h_nextn buffer size\|extracting the h_nextn\|sched-trace: prefetch' $OUT/server-trace.log | head
```

## Files

| File | What it holds |
|---|---|
| [run/checks.sh](run/checks.sh) | the whole check: the CPU build, `--mtp-h-dev`, `--mtp-catchup-overlap`, the logits against the base, the server arms, ctest, the rig tarball (through C620's `checks.sh`) and the rig binary's `--mtp-h-dev` |
| [run/server_smoke.py](run/server_smoke.py) | the server arms: one llama-server start per arm, three requests, the replies against the base's and the log's lines on where the rows went |
| [summary.tsv](summary.tsv) | one verdict a check (26 lines; the work folder written as `<work>`, the base's binaries as `BASE`) |
| [hdev.tsv](hdev.tsv) | `--mtp-h-dev`, one row per build (`tip`, `rig`), switch set, seed and configuration: each run against the reference, where the rows were in the six runs (h: host, d: the buffer) and the deferred catch-ups |
| [server_arms.tsv](server_arms.tsv) | one row per server arm: its switches, replies identical to the base's, the log's lines against what the switches should give, the verdict |
| [server_replies.tsv](server_replies.tsv) | one row per arm and request: tokens, drafted, accepted, against the base |

## Commands

On a build machine with Docker, CPU only, every build and test under one lock (here shared with other sessions' checks)
at `nice -n 19` and `--cpu-shares=256`:

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE: bin/ with -c748's (5b329df16) static native CPU
# test-llama-archs and llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c693 BASE=$BASE \
  LOCK=$LOCK JOBS=8 DOCKER_ARGS='--label pascal-quants=1' \
  bash docs/data/2026-11-11_c693_upstream_switches/run/checks.sh
```

The first run's server step failed on its log lines only, while every arm's replies matched. llama-server shows
libllama's info lines (the buffer's size, the `=0` line) only from verbosity 4, and one marker also matched the
overlap's warning, which names `LLAMA_MTP_H_DEV=0`. The script now starts the servers with `-lv 4` and looks for the
whole line. The failed line was set aside, and `STEPS=server` ran again. Every other step ran once.

Provenance and licence: everything here was generated by the commands above from the fork's code (MIT, as llama.cpp)
on random models that `test-llama-archs` builds; no model file, dataset or recorded text is involved.
