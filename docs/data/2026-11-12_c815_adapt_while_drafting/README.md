# C815: the MTP draft's length decided while drafting (2026-11-12)

Registry: [Q1424](../../../quants/experiments.md) (C815), from Q1402 (FN82,
[data](../2026-11-12_fn82_adaptive_draft/README.md)) and Q687 (`LLAMA_SPEC_ADAPT`). Fork branch
`pascal-p106-v4-b12246-c815`, merged into `pascal-p106-v4` off by default. Switch: `LLAMA_SPEC_ADAPT_ONLINE`
([docs/switches.md](../../switches.md)).

Flash-Next decodes with an MTP draft: the draft guesses a few tokens, one drafting pass each (3.645 ms on nine P106
cards, R86), and the main model checks them in one verify pass. `LLAMA_SPEC_ADAPT=1` picks each step's draft length
from the draft's own probabilities and a table of what a verify pass of k tokens costs: it keeps the longest prefix of
the draft that pays for its verification. FN82 found it beats every fixed length with a draft of up to 6 tokens. But
as built the drafter drafts all 6 tokens and the rule then cuts the draft, so every step pays 6 drafting passes,
including passes whose tokens the rule throws away.

The rule walks the allowed lengths upwards (by default 2, 4 and 6 drafted tokens) and tests each one with the
probabilities up to that length alone. So the drafter can stop as soon as the outcome is settled. This change lets it
do that, and gives exactly the same verified tokens and replies as the switch as built.

## Result

- **`LLAMA_SPEC_ADAPT_ONLINE=2` leaves out 1.5491 and 1.6638 drafting passes a step with the same tokens: est. +0.946
  and +0.989 tok/s over the switch as built** on FN80's 6- and 7-token texts under FN82's est. v4 cost table (95 %
  over requests +0.879 to +1.011 and +0.911 to +1.062). `=1` leaves out 0.9378 and 1.0128: +0.565 and +0.593 (+0.520
  to +0.606 and +0.541 to +0.645). Est. tok/s, a draft of up to 6, the switch's defaults ([price/](price/)):

  | Arm | 6-token text: drafting passes a step | est. tok/s | 7-token text: passes | est. tok/s |
  |---|---:|---:|---:|---:|
  | fixed draft of 3 | 2.9969 | 24.938 | 2.9921 | 24.114 |
  | `LLAMA_SPEC_ADAPT=1` as built | 5.9771 | 25.951 | 5.9661 | 24.815 |
  | `LLAMA_SPEC_ADAPT_ONLINE=1` | 5.0393 | 26.516 | 4.9533 | 25.408 |
  | `LLAMA_SPEC_ADAPT_ONLINE=2` | 4.4280 | 26.897 | 4.3023 | 25.804 |
  | every width (`LLAMA_SPEC_ADAPT_WIDTHS=1,2,3,4,5,6,7`), as built | 5.9771 | 26.162 | 5.9661 | 25.051 |
  | every width, `LLAMA_SPEC_ADAPT_ONLINE=2` | 4.3000 | 27.221 | 4.1638 | 26.162 |

  Every arm verifies the same tokens as its as-built row on every one of the 3,537 and 3,278 steps (it keeps 4.1660
  and 3.9280 tokens a verify step with the default widths, 4.1139 and 3.8749 with every width); only the drafting
  passes change. The time a step is its drafting passes × 3.645 ms plus the verify pass V(k) of FN82's est. v4 table.
  Under FN82's uniform tables `=2` gains +0.827, +0.950 and +1.040 over as built on the 6-token text at a verify
  column of 8, 12 and 16 ms.
- **`=2` is nearly as good as allowing every width, without its seven verify shapes.** FN82's "every width" arm drafts
  4.4080 passes a step; `=2` with the default widths (four verify shapes, so CUDA graphs stay cached) drafts 4.4280.
  Every width with `=2` adds +0.324 and +0.358 more, at seven shapes.
- **What FN82 priced is what `=1` gives:** its "decided while drafting" arm (26.528 and 25.413) charged the rule's
  running speed-up with the passes actually drafted; the build keeps the as-built charge, so that the tokens stay the
  switch's. The two differ by −0.012 and −0.005 est. tok/s (−0.044 to +0.011, −0.018 to +0.012). `=1` stops at FN82's
  stopping point on every step.

## The change

- **`common_spec_adapt_continue`** (common/speculative): given the probabilities drafted so far, whether one more token
  can change what the rule keeps. `false` once an allowed length's test has failed, once the draft reaches the last
  allowed length, and, at `=2`, once the next allowed length cannot pay even if every token still to draft had
  probability 1. That bound adds the path probability so far once for each missing position, with the same rounded
  additions the test makes, so it is never below what any longer draft would give. When it says stop,
  `common_spec_adapt_choose` keeps the same tokens of the shorter draft, with the same S, E and C, as of any longer
  draft that starts with it.
- **draft-mtp** asks it after each drafted token, where `--spec-draft-p-min` would stop the draft. The server's
  `spec_adapt_choose` then cuts the shorter draft as before, to the same tokens, and its running speed-up is charged
  as before.
- **The same random stream.** A probabilistic draft (`--spec-draft-sampling probabilistic`) samples each token, and
  its sampler carries its random state to the next step. Each pass left out would have drawn one number; the drafter
  draws them (it applies the chain to one candidate per pass left out, and `dist` draws once a call), so later drafts
  are the as-built ones. `LLAMA_SPEC_ADAPT_ONLINE_FAULT=1` skips these draws: the check's control.
- **C738's verify pass prepared during the last drafting pass** (`LLAMA_VERIFY_PREPARE`). As built, the server refuses
  the prepare under `LLAMA_SPEC_ADAPT`, since the rule may cut the draft after it. With the rule taken while drafting,
  draft-mtp calls the prepare only for the pass that reaches the last length the rule can keep, and only when the rule
  keeps that pass's token whatever its probability (`common_spec_adapt_keeps_next`). The server then prepares, and
  stops with an error if the rule's cut ever differs from the prepared draft.
- **Where it applies:** `LLAMA_SPEC_ADAPT=1` and `--spec-draft-p-min 0` (every pass adds a token, so the as-built draft
  length is known). Otherwise the server warns at start and the switch does nothing. The drafter closes with
  `C815: <n> drafts under the rule taken while drafting, <n> stopped early, <n> drafting passes left out, <n> sampler
  draws made for them, <n> last passes kept`. `LLAMA_SPEC_ADAPT_LOG`'s `n_draft` and `p` are what was drafted.

## For the rig

C797's rig session takes it as an arm beside FN82's adaptive arm: `LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2
--spec-draft-n-max 6` with the same `LLAMA_SPEC_ADAPT_COST` and `LLAMA_SPEC_ADAPT_DRAFT_MS=21.87`, probabilistic
verification and C675's stop. Its replies are the as-built arm's, so the two arms differ only in time.

## The checks (a build machine's CPU, 2026-11-12)

Every build and test ran in a container without GPUs at the lowest CPU share; builds took turns under a lock with the
other sessions' builds. `run/checks.sh` is the chain; its rows are in [checks.tsv](checks.tsv).

| Check | Verdict |
|---|---|
| CPU build of the branch (static, every target) and of the base `b9a1c1fa7` (llama-server, test-llama-archs) | OK, OK |
| `test-spec-adapt`, twice | 434,271 of 434,271 checks both times, the same output |
| llama-server against the base, 15 arms in 6 groups (`run/server_check.sh`) | PASS: every arm = its reference |
| `test-llama-archs -b CPU -s 1` against the base's logits: no switch; `LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2` | 146 of 146 the same, 146 of 146 the same |
| The switch list in the rig build | `LLAMA_SPEC_ADAPT_ONLINE` and `_FAULT` in it (869 names) |
| The rig tarball (sm_61, no AVX, GLIBC_2.38 at most) of the branch, and the same build of the base | `llama.cpp-20261112-c33c894da-pascal-p106-v4-b12246-c815.tar.gz`, 678,590,643 B, md5 `1a615ccc61567ab512b1f8fb97b6cb23`; the base's md5 `34beab8b5313778728d17428e1fa6160` |
| `llama-server`'s SASS and PTX against the base's (cuobjdump, cut into functions) | the same files, 8,540 functions each: the GPU code is the base's |
| ctest, static and shared (the C565 set: the network test, archs written and compared, the hand tests) | 71 of 71 and 71 of 71; everything else OK in both |
| The merged `pascal-p106-v4` `14710289b` (its tree is the branch's): a static CPU build, `test-spec-adapt`, archs against the base with every switch off and with `LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2`, the server check's d arms | OK; 434,271 of 434,271; 146 of 146 and 146 of 146 the same; PASS (4 arms = the base) |

[checks.tsv](checks.tsv) also holds two earlier runs of the server check (`server`, `server2`), both FAIL on the
check's own coverage, not on the code: the first ran every group under C797's table, where the rule keeps nothing on
this model, so no group had a verify step to compare; the second did not yet allow group v its empty comparison, and
group r's running S drifted to where the rule kept nothing on 405 of 408 steps. In both runs every arm's replies were
its reference's, and every failure came from a group where the rule kept nothing: no verify step to compare, no
prepare, no draft for the control to change.

- **`test-spec-adapt`** adds 20,000 random drafts and cost tables (convex tables and FN82's est. v4 table; S step,
  fixed and run; default, every, and random widths; probabilities with exact 0, 1 and out-of-range values). A drafter
  that stops where `common_spec_adapt_continue` says so keeps exactly the as-built choice (the same length, S, E and
  C, bit for bit); at `=1` it stops at the first failing allowed length or the last allowed one, FN82's "online" stop;
  at `=2` never later. Every prefix where it says stop gives the same choice on four extensions of it (all ones, all
  zeros, two random). `common_spec_adapt_keeps_next` is exactly "a next token of probability 0 is kept" on every
  prefix (130,522), and the 6,047 where it holds keep the next token at probabilities 1, the drafted one, a random one,
  7 and −1. On these drafts a draft of up to n_max drafts 6.5261 passes as built, 3.1759 at `=1`, 2.4505 at `=2`.
- **The server check** runs llama-server on a random-weight qwen4exp model with one MTP block
  (`test-llama-archs -o --mtp`), four requests per arm: two greedy (200- and 600-token prompts, 96 and 128 tokens) and
  two at temperature 1.0 with seeds. Each arm is compared with the base's llama-server under the same settings, on every
  request's tokens, top-3 probabilities and draft counts, on every verify step (`LLAMA_SPEC_DUMP`: the draft, the
  target's samples, the drafted positions' candidates cut to the verified draft), and on every step of the rule
  (`LLAMA_SPEC_ADAPT_LOG`: what it kept, its S, E and C, the accepted count, the next S; and the drafted probabilities a
  prefix of the base's). [server/verdict.txt](server/verdict.txt), [server/passes.tsv](server/passes.tsv) (the counts
  per arm), [server/summary.tsv](server/summary.tsv) (every reply's tokens).
  - The random model's draft gives each drafted token a probability near 0.10, so under C797's table the rule keeps
    nothing on any step (group v). The other groups use cost tables whose thresholds sit at the median of what this
    model's drafts bring to each test (taken from the logs of a first run), so the rule's choice varies: the base kept
    0, 2, 4 and 6 tokens on 207, 68, 2 and 130 of 407 steps with the default widths, and 0 to 4 on 205, 43, 37, 17 and
    105 with every width.
  - Drafted tokens, base against `=1` and `=2` (passes a step): default widths 2,382 against 1,462 and 1,462 (5.8526
    against 3.5921); every width 2,382 against 989 and 954; every width with S run 2,382 against 984 and 949;
    probabilistic drafting 2,388 against 1,184 and 1,184. For every pass left out the drafter drew what the as-built
    pass would have drawn on its sampler (each arm's count in `server/passes.tsv`).
  - With `LLAMA_VERIFY_PREPARE=1` the verify pass was prepared during the last drafting pass on 75, 75 and 51 steps
    (greedy, backend sampler, probabilistic), each with the replies of the base; the switch as built with
    `LLAMA_VERIFY_PREPARE=1` prepared none, as before.
  - The control, `LLAMA_SPEC_ADAPT_ONLINE_FAULT=1` with probabilistic drafting: the greedy requests are the base's,
    the two sampled ones are not. So the check sees a drafter's random stream that drifts.
  - The model accepts almost no drafted token (1 of 924 verified with the default widths): the replies depend on the
    drafts through the target's rejection sampling and its random draws, which the control shows.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | the chain's rows (`run/checks.sh`) |
| [server/](server/) | the server check's verdict, counts per arm and replies |
| [price/price.tsv](price/price.tsv) | per load, cost table and arm: tokens a verify step, verify columns a step, drafting passes a step, est. ms a step and tok/s |
| [price/compare.tsv](price/compare.tsv) | the arms' est. tok/s differences with 95 % bootstrap intervals over requests, paired |
| [price/checks.txt](price/checks.txt) | the pricing's checks: the same tokens on every step, passes in order, FN82's stopping point |
| [run/checks.sh](run/checks.sh) | the check chain (from C779's) |
| [run/server_check.sh](run/server_check.sh) | the server check (from C738's) |
| [run/price_built.py](run/price_built.py), [run/price_run.sh](run/price_run.sh) | the pricing, on FN80's dumps with FN82's replay ([bench/draft_count/adapt_replay.py](../../../bench/draft_count/adapt_replay.py)) |

## Commands

- The checks, on a machine with Docker (paths in the environment; `SEED_SRC` is a llama.cpp clone with the fork as
  remote `fork`):

  ```bash
  M=<work dir> REPO=<this repository> LOCK=<flock file> BASE_COMMIT=b9a1c1fa7 SEED_SRC=<clone> \
    JOBS=8 THREADS=8 bash docs/data/2026-11-12_c815_adapt_while_drafting/run/checks.sh
  ```

  The server check again, alone, with the chain's binaries and model:

  ```bash
  BASE=$M/bsrv/bin BIN=$M/cpu/bin MODEL=$M/server/models/qwen4exp-moe-mtp.gguf OUT=<dir> THREADS=4 \
    SARGS='--override-kv qwen4exp.context_length=int:16384' bash docs/data/2026-11-12_c815_adapt_while_drafting/run/server_check.sh
  ```

- The pricing (3 s on one core): `WORK=<dir> bash run/price_run.sh`, with `WORK/repo` this repository and
  `WORK/dumps/work/` FN80's dumps.

Provenance: the code is fork branch `pascal-p106-v4-b12246-c815` (`c33c894da`); the dumps are FN80's, recorded with
Flash-Next on the rig ([2026-11-11_fn80_fn81_draft_yield](../2026-11-11_fn80_fn81_draft_yield/README.md)); the random
test model is generated by `test-llama-archs`. Licence: the repository's.
