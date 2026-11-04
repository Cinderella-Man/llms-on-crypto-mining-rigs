# Q687: the draft length chosen per step for Flash-Next (2026-11-04)

A speculative step on the rig drafts a fixed number of tokens (`--spec-draft-n-max`) and verifies them all in one pass
of the main model. On nine P106 cards each extra verified token costs 50–59 ms against 136 ms for a one-token pass
([Q583](../2026-11-03_verify_cost/README.md)), so a drafted token pays only when it is likely to be accepted: Q561
chose two drafted tokens for DFlash2 because three did not pay on average
([Q561](../2026-11-03_flash_next_dflash2/README.md)). But the drafter knows, step by step, how sure it is. This
change lets llama-server keep, after each draft, only the prefix that pays for its verification, judged from the
drafter's own probabilities and a table of what a k-token pass costs: the rule of
[moe_speculation_union.md](../../research/notes/moe_speculation_union.md) (sections 3.1 and 3.2, idea 4), "verify a
position only if its path probability exceeds r·S". Speculation stays exact: only the number of drafted tokens
verified changes. Registry rows Q687 (this), Q605 (the fixed count), Q760–Q764 (ideas from it) in
[quants/experiments.md](../../../quants/experiments.md).

**Status:** written behind a switch that is off by default, built for the rig, checked on the CPU (below). The rig
run, S18's DFlash2 line with the switch against fixed lengths on the agent check, is
[run/rig_adraft.sh](run/rig_adraft.sh).

## The rule

With p_i the drafter's probability of drafted token i given the drafted prefix, the path probability
P_j = p_1 · … · p_j is the chance that drafted token j is accepted, and a step that verifies d drafted tokens yields
E(d) = 1 + P_1 + … + P_d tokens for C(d) = D + T(d + 1) ms, where T(k) is the time of a k-token verify pass and D the
rest of a step (the draft pass, sampling). Going from d to the next allowed length d' pays when

    P_(d+1) + … + P_(d')  >  scale · S · (T(d' + 1) − T(d + 1)) / T(1)

that is, when the added positions' mean path probability exceeds r·S, with r the marginal cost of one more verified
token as a fraction of a one-token pass and S the speed-up over one-token decoding that the extra cost has to keep up.
The rule walks the allowed lengths upwards from 0 and stops at the first that does not pay. S is one of:

| `LLAMA_SPEC_ADAPT_S` | S | What it maximizes |
|---|---|---|
| `run` (default) | the running speed-up of the steps so far, from the cost table: T(1) · (average tokens per step) / (average modelled step cost), exponential averages with weight `LLAMA_SPEC_ADAPT_EMA` (1/32), starting at `LLAMA_SPEC_ADAPT_S0` (1.0) | the long-run tokens per ms: a block is verified when it buys tokens at a better rate than the policy's own average |
| `step` | the chain's own, E(d) · T(1) / C(d) | this step's tokens per ms |
| a number | that number | the long-run rate at a known S |

On monotone drafts (every P_j ≤ P_(j−1)) and convex cost tables, stopping at the first length that does not pay finds
the best allowed length: E(d)/C(d) (`step`) and E(d) − λ·C(d) (`run`, fixed) are unimodal in d. The unit test checks
this against an exhaustive search on 4,000 random drafts and cost tables. With every width allowed it is the note's
rule taken literally, one position at a time.

**Verify widths are limited to a few** (default k = 1, 3, 5, 9, 17, each clamped to `--spec-draft-n-max` + 1), so that
CUDA graphs per verify width stay cached (Q701, [2026-11-04_vgraphs](../2026-11-04_vgraphs/README.md)). A draft that
is cut to nothing makes the step a plain one-token step. At `--spec-draft-n-max 5` the widths are 1, 3, 5 and 6.

**The confidences.** DFlash2: the selector's softmax over its 16 candidates at the one taken (the same number that
`--spec-draft-p-min` reads). Draft-simple, EAGLE-3, MTP, DFlash and DSpark: their sampler's probability of the token
over its top 10. Both overstate the chance of acceptance where the drafter is sure of a wrong token (Q734, Q760). The
n-gram drafters give no probabilities, and their drafts stay as they are.

### What the rule keeps with today's costs and after the kernels

The rule's arithmetic for a drafter whose every token has the same confidence c (P_j = c^j), at
`--spec-draft-n-max 5` with today's table (Q583: T(k) = 136, 185.8, 244.6, 305.4, 359.9 ms, then +54.5 ms per token;
D = 20 ms; S = 1.15, the DFlash2 arm's measured gain) and at `--spec-draft-n-max 7` with the note's estimate after
Q591's kernels (section 3.1, "local": T(k) = 90, 98.1, 106.0, 113.4, 120.7, 127.6, 134.5, 141.3 ms; D = 20 ms;
S = 1.6; est.):

| c | today: drafted tokens verified (verify width) | after the kernels (est.) |
|---|---|---|
| 0.5 | 0 (1) | 2 (3) |
| 0.6 | 2 (3) | 4 (5) |
| 0.7 | 2 (3) | 7 (8) |
| 0.8 | 2 (3) | 7 (8) |
| 0.85 | 4 (5) | 7 (8) |
| 0.9 | 5 (6) | 7 (8) |

Today's thresholds: going from 0 to 2 drafted tokens needs P_1 + P_2 > 0.918, from 2 to 4 P_3 + P_4 > 0.975, from 4 to
5 P_5 > 0.461. The note's DFlash2 chain (cumulative 0.65 / 0.42 / 0.30, then ×0.75) keeps two on today's table, as
Q561 chose by hand, and four or more after the kernels. So a fixed count gives away both ends: steps where the
drafter is unsure verify two tokens that will likely be rejected, and steps in copied text (c ≥ 0.9) stop at two.

## What was built

Fork branch `pascal-p106-v3-b12076-adraft` `b52064398`, one commit on `pascal-p106-v3-b12076-ladder2` `a330672c1`:

| File | What |
|---|---|
| `common/speculative.h`, `.cpp` | `common_spec_adapt_params` and its parser (`common_spec_adapt_params_from_env`), the cost table (`common_spec_adapt_cost`, the last slope past its end), the allowed lengths (`common_spec_adapt_lengths`), the rule (`common_spec_adapt_choose`), the running speed-up (`common_spec_adapt_state`); the draft parameters gain `result_p`, the drafter's probability of each drafted token, which draft-simple, EAGLE-3, DFlash/DFlash2/DSpark and MTP fill only when it is set |
| `tools/server/server-context.cpp` | after each new draft, the rule cuts the draft (and its candidates for rejection sampling) before the checkpoints and the verify batch; the step's log line once its accepted length is known; a per-task summary in the server log (`adaptive draft (on) = N drafted steps, … steps by verify width (1: a, 3: b, …), running S …`) |
| `tests/test-spec-adapt.cpp` | the cost table, the lengths, hand cases (the note's chain), the rule against an exhaustive search of its objective, the literal per-position rule, the running speed-up, the parser |

The settings are environment variables, read once at the server's start (a malformed value stops the server there):

| Variable | Default | What |
|---|---|---|
| `LLAMA_SPEC_ADAPT` | unset (off) | `1` applies the rule |
| `LLAMA_SPEC_ADAPT_COST` | `136,185.8,244.6,305.4,359.9` | T(k) in ms for k = 1, 2, …, as a list or a file of numbers (`#` comments); past its end the last slope |
| `LLAMA_SPEC_ADAPT_WIDTHS` | `1,3,5,9,17` | the verify widths allowed (drafted tokens + 1), clamped to n_max + 1 |
| `LLAMA_SPEC_ADAPT_DRAFT_MS` | `20` | D, a step's cost besides the verify pass |
| `LLAMA_SPEC_ADAPT_S` | `run` | `run`, `step` or a number (above) |
| `LLAMA_SPEC_ADAPT_S0`, `LLAMA_SPEC_ADAPT_EMA` | `1.0`, `0.03125` | the running speed-up's start and the weight of the latest step |
| `LLAMA_SPEC_ADAPT_SCALE` | `1.0` | multiplies the threshold (above 1 for a drafter more confident than it is right) |
| `LLAMA_SPEC_ADAPT_LOG` | unset | a file for one JSON line per drafted step, after a header line per server start; it works with the switch off too, so a fixed-length run records what the rule would have kept |

With neither `LLAMA_SPEC_ADAPT` nor `LLAMA_SPEC_ADAPT_LOG` set, no probability is computed and nothing changes.

The defaults are Q583's wall times on v3's base. Q591's measured k-curve (the small-batch kernels timed on a P106)
replaces them through `LLAMA_SPEC_ADAPT_COST`, e.g. `LLAMA_SPEC_ADAPT_COST=/path/t_k.txt` with one T(k) per line.

**The log.** Header: `hdr`, `run` (the server start, µs), `enabled`, `n_max`, `widths`, `cost_ms` (T(1) … T(n_max + 1)),
`draft_ms`, `s_mode`, `s_fixed`, `s0`, `ema`, `scale`. Each step: `run`, `step`, `task`, `slot`, `pos` (the position of
the token the draft continues), `on`, `n_draft` (drafted), `n_rule` (what the rule keeps), `n_keep` (verified),
`p` (the drafter's probabilities), `s` (the S the rule used), `e_tau` and `cost_ms` (E and C at `n_rule`), `n_acc`
(accepted of the verified), `s_next` (the running S after the step). [run/adapt_log.py](run/adapt_log.py) checks a log
against the rule (`--check`: every choice recomputed from its header, the running S replayed) and summarizes it per run:
steps by verify width, drafted, verified and accepted tokens per step, the modelled speed-up, and for a fixed-length log
the rule replayed on the same drafts (a step that kept n_rule tokens would have accepted min(n_acc, n_rule); the next
draft would start elsewhere, so that is an estimate).

## Design: two drafters' agreement as the confidence (not written; Q761)

The cheapest second drafter is the context itself: the longest earlier match of the context's suffix and the tokens
that followed it (llama.cpp's n-gram drafters; Q688's dump records it as `ng`). Where it proposes the same next tokens
as DFlash2, the step is inside copied text, where a 6-token match continues correctly 95–99 % of the time (Q64), far
above what DFlash2's selector says of itself. The design: per step, compute the match beside the DFlash2 draft (the
server chains drafters, so today the n-gram ones run only when DFlash2 drafts nothing); for each position where both
agree, raise p_i to a fixed agreement confidence (default 0.95, a setting); past the end of DFlash2's block, append
the match's further tokens with that confidence while they last, so the rule can reach the widths that pay in copy
spans (9 and 17, with Q617's long chains and a verify kernel to 16 tokens). It is not written here because it needs the
match computed every step beside the model drafter, a change to the implementation chain rather than a few lines, and
because Q688's replay (`copy_gated.csv`) measures first how often the two agree on Flash-Next's agent sessions.

## CPU check (quant machine, Docker, no GPU, 2026-11-04)

`WORK=… REPO=… LOCK=… bash run/cpu_check.sh` ([cpu_check.md](cpu_check.md) has the full summary):

| Check | Result |
|---|---|
| build | CPU backend, static: the branch's `llama-server`, `test-spec-adapt`, `test-llama-archs`, and the base's `llama-server` |
| unit | `test-spec-adapt` 13,507 / 13,507: the rule equals an exhaustive search of its objective on 4,000 random monotone drafts and convex cost tables (and the literal per-position rule on the 1,440 with every width allowed); hand cases; the parser |
| switch off | a random qwen4exp as its own draft-simple drafter: with no variable set, with only the log, and with the switch on at a threshold every token passes, the server returns the base build's tokens and draft counts on all three requests (9 / 9) |
| exact | every greedy reply of the eight drafting arms equals plain decoding's (16 / 16), whatever the rule kept |
| the log | `adapt_log.py --check` recomputes all 828 logged choices and the running S from the headers: 0 problems; the switched-on arms verify at widths 1, 3 and 5 only |
| bad settings | a malformed cost table stops the server at its start |

The random drafter's probabilities sit at 0.10–0.104 (its top 10 are nearly tied) while, being the target, it is
right on the greedy requests, so on the default table the rule verifies nothing there: the fixed-length arm's log
shows the rule turning a modelled 1.25× into 0.87× on those drafts. The rule gains only where a drafter's
probabilities track its acceptance; DFlash2's and the MTP draft's are first logged on the rig (and Q760 calibrates
them online).

## Next: the rig (S18's DFlash2 line, the agent check)

[run/rig_adraft.sh](run/rig_adraft.sh) runs, on the second rig with S18's setup (Swift-1.5 GSQ-RCO Q2_0, Inco AI's
DFlash2 drafter on card 8, `-b 512`), four arms on the same build: `fix2` (S18's `dflash2b`), `fix3`, `fix5` (the
switch off, the log on) and `adapt5` (`--spec-draft-n-max 5` with `LLAMA_SPEC_ADAPT=1`: widths 1, 3, 5, 6). Each gets a
smoke test first (the first-turn request and Q561's two requests); an arm the RAM watchdog stops is left out. Then
the timed agent check per arm (40 minutes each, as S18), the report against Q226's `nodraft` and S18's `dflash2b`, and
`adapt_log.py` on every arm's log. About 4 hours of rig time.

- **What decides:** minutes per task at equal generated tokens (S18's method), `adapt5` against the best fixed arm;
  the expectation (note, idea 4) is +5–12 % over the best fixed length once the costs fall, less today, where the
  rule picks between none and two verified tokens unless the drafter is at 0.85 or more per token.
- **What else it gives:** the fixed arms' logs replay the rule offline on real DFlash2 drafts (`adapt.tsv`'s
  `rule_*` columns), and the logs' p against acceptance is the calibration Q760 needs.
- **Risks:** `--spec-draft-n-max 4` crossed the RAM watchdog's floor at `-b 2048` in Q561 (at `-b 512` five is
  untested), and DFlash2 decodes n_max + 1 block positions every step whatever the rule keeps, so `adapt5` pays
  `fix5`'s draft pass; if both n-max-5 arms fail their smoke test, `ARMS="fix2 fix3 fix4 adapt4"` is the fallback.
  This branch is `-ladder2`, whose kernels are not yet checked on a GPU (C86); its smoke test is the first run.

## On the rig (C134r, 2026-11-04)

The second rig (`kho=off`, the RAM watchdog's `MemAvailable` floor at 60 MiB), S18's DFlash2 line on Swift-1.5 Q2_0,
run on Miner Llama's `-ladder3` (`7bc1552f4`, which holds this branch) with `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768` on
both arms (`ARM_ENV`). The cost table is the one measured on the first rig under the same setting on real agent text
(`docs/data/2026-11-04_kcurve_p106/cost/t_k_q2_0_s32k_real.txt`: 64.9 / 97.9 / 127.9 / 151.6 / 193.9 / 222.9 / 249.9 /
288.5 ms at k = 1–8). Two of the four arms ran (`fix2`, `adapt5`): the rig was switched off for the night before
`fix3` and `fix5`. Both smoke tests passed, `adapt5`'s at `--spec-draft-n-max 5` too.

| Arm | Passed | Minutes, all tasks | Minutes per task (median) | Prompt tok/s | Generation tok/s | Nine shared tasks: tok/s, at equal tokens | Acceptance | Tokens a step | Usable host memory lowest |
|---|---|---|---|---|---|---|---|---|---|
| no draft (S12, v3's base) | 9 / 9 | 43.0 | 4.1 | 59 | 7.8 | 7.8, 1.0× | — | 1 | — |
| DFlash2 `dflash2b` (S18, v3's base) | 11 / 11 | 41.0 | 2.9 | 47 | 9.5 | 9.5, 1.15–1.16× | 0.778 | 2.56 | 95 MiB |
| `fix2` (two drafted, switch off) | 10 / 10 (2 past the budget) | 41.8 | 3.5 | 51 | 12.0 | 12.2, 1.30–1.39× | 0.742 | 2.49 | 229 MiB |
| **`adapt5`** (`LLAMA_SPEC_ADAPT=1`, up to five) | 11 / 12 | 46.4 | 2.8 | 50 | **12.5** | **14.0, 1.52–1.53×** | 0.761 | 3.19 | 173 MiB |

- **Choosing the verified length per step beats a fixed two by ~10–17 % at equal tokens** on the nine shared tasks
  (1.52–1.53× against 1.30–1.39×). The rule verified 1 token on 1,748 steps, 3 on 2,186, 5 on 631 and all 6 on 2,917
  of 7,482 ([rig/adapt.tsv](rig/adapt.tsv)): 3.19 tokens a step at a modelled 175.8 ms against `fix2`'s 2.49 at 264.6.
- `fix2`'s log replays the rule offline on the same drafts: it would have verified 1 token on 1,087 of 8,004 steps and
  modelled 1.31× against the arm's 1.28× (the `rule_*` columns).
- `adapt_log.py`'s check: 15,486 steps in two runs, 0 problems ([rig/adapt_check.txt](rig/adapt_check.txt)).
- Host memory: `adapt5` held usable memory at a median 903 MiB and lowest 173, swap up to 908 MiB, at most 59 MiB
  swapped out in 10 s; `fix2` lowest 229, swap 33 MiB; no stop.

## Files

| File | What |
|---|---|
| `cpu_check.md` | the CPU check's verdicts and the smoke arms' log summaries |
| `smoke_on8.jsonl` | the `on8` smoke arm's log (`--spec-draft-n-max 8`, the switch on): a sample of the format |
| `run/cpu_check.sh` | the CPU check: build (with the base build), unit test, smoke arms on a random-weight qwen4exp |
| `run/adapt_log.py` | checks and summarizes `LLAMA_SPEC_ADAPT_LOG` files (standard library only) |
| `run/rig_adraft.sh` | the rig run: smoke tests, the timed arms, the report and the logs' tables |
| `rig/report.md` | C134r: the agent check's report, per task and per arm, against S12's no-draft and S18's DFlash2 arms |
| `rig/common_tasks.csv` | C134r: the nine tasks every arm scored, per arm sums and the time at the no-draft arm's tokens |
| `rig/adapt.tsv`, `rig/adapt_check.txt` | C134r: per arm the steps by verify width, tokens a step, the modelled speed-up and the rule replayed on the drafts; the log check |
| `rig/session_memory_*.csv`, `rig/draft_acceptance_*.csv` | C134r: each arm's host memory over the session and acceptance per request |

## Commands

On the build machine (the quant machine here), with the branch checked out in `$WORK/src`:

```bash
WORK=<work dir> REPO=<this repository> LOCK=<build lock file> JOBS=6 THREADS=6 \
  DOCKER_ARGS="--label <your label>" bash docs/data/2026-11-04_adaptive_draft/run/cpu_check.sh
# (here the unit and smoke stages ran once more with STEPS="unit smoke" on the same binaries)
# the rig tarball
TESTS=1 JOBS=6 WORK=<work dir> bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-adraft
```

On the machine that drives the rig, from the repository root, with S18's values
([../2026-11-03_flash_next_dflash2/run/s18.env.example](../2026-11-03_flash_next_dflash2/run/s18.env.example)):

```bash
set -a; . s18.env; set +a
REL_BIN=/opt/llama-builds/<the adraft tarball>/bin/llama-server bash docs/data/2026-11-04_adaptive_draft/run/rig_adraft.sh out/adraft
```

## Provenance and licence

The code is Miner Llama's fork of llama.cpp (MIT). The CPU check uses a random-weight model generated by llama.cpp's
`test-llama-archs`; no model was downloaded. The tables here hold verdicts, counts and probabilities from that model.
The rig run's model and drafter are S18's (Swift Open License 1.0 and qwen-community-1.0; Apache-2.0).
