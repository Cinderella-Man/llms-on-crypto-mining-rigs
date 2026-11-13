# The agent check of Flash-Next's final v4 settings `v4r` on `v4-rc2`, and a second v3.0 run (R156, Q1474; R158, Q1476; 2026-11-13)

R156 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on `v4r`, Flash-Next's final v4 settings, on the release candidate `v4-rc2` (`pascal-p106-v4` `eb82916d3`, the
v4 line on upstream llama.cpp b12360: [2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md)). The model is
the maker's Swift GSQ-RCO Q2_0 with the MTP draft, on a mining rig of nine P106-090 cards. `v4r` is `v4c3`
([2026-11-13_r142_agent_v4c3](../2026-11-13_r142_agent_v4c3/README.md)) with the results of 2026-11-13's rig rows
applied. Two runs, one arm each: R156a (`v4r`, on rig 2) and R156b (`v4r2`, the same line again, on rig 1). The two
rigs are the same model of machine. Registry row Q1474 in [quants/experiments.md](../../../quants/experiments.md).
R158 (registry row Q1476) added a second v3.0 run on rig 1 the same evening, arm `v3b`: R90's `v3` line exactly, so
that v3.0 and v4 each stand on two runs ([Both v3.0 runs](#both-v30-runs-r158)).

**Flash-Next's v4 numbers (`v4r` on `v4-rc2`, both runs passed 12 of 12 tasks):**

| Run | Generation, all 12 tasks | Minutes a task at equal work | Without `complex-numbers` |
|---|---|---|---|
| R156b, rig 1 | **27.61 tok/s** (+33.6 % against `v3`) | **2.37** (−23.1 %) | 27.58 tok/s, 2.46 min |
| R156a, rig 2 | **26.67 tok/s** (+29.1 %) | **2.43** (−21.1 %) | 27.59 tok/s, 2.46 min |
| R90's `v3` (v3.0), rig 1 | 20.66 tok/s | 3.08 | 20.92 tok/s, 3.13 min |
| R158's `v3b` (v3.0), rig 1 | 18.92 tok/s | 3.19 | 20.82 tok/s, 3.09 min |

**The release table's Flash-Next line** (Swift GSQ-RCO Q2_0 with the MTP draft, the agent check's 12 tasks, minutes a
task at equal work and generation speed, each run by value with its rig):
- **v3.0:** 3.08 minutes and 20.66 tok/s (R90, rig 1); 3.19 minutes and 18.92 tok/s (R158, rig 1). Each run passed 11
  of 12.
- **v4:** 2.37 minutes and 27.61 tok/s (R156b, rig 1); 2.43 minutes and 26.67 tok/s (R156a, rig 2). Each run passed 12
  of 12.

**Against `v4c3`, `v4r` is negligible.** `v4c3`'s two runs on rig 1 took 2.35 and 2.44 minutes a task at 28.03 and
26.16 tok/s.
- On all 12 tasks and on R90's 11, `v4r`'s two runs fall inside `v4c3`'s two: 2.37 and 2.43 minutes, and 1.51 and
  1.59 on R90's 11 (1.49 and 1.61 for `v4c3`).
- Without `complex-numbers`, both `v4r` runs take 2.46 minutes against 2.45 and 2.43. Their generation is 27.58 and
  27.59 tok/s against 28.00 and 27.79.
- Each of those gaps to `v4c3`'s slower run is 0.01 minutes and 0.20 or 0.21 tok/s. That is no larger than the gap
  between `v4c3`'s own two runs (0.02 minutes, 0.21 tok/s). So `v4r` is not measurably slower, and not a gain.

The comparison takes in the build too: `v4r` runs on `v4-rc2`, `v4c3` on `65ac9a47d`. Each run had one long task:
- rig 2 took the long path on `complex-numbers` (9.7 minutes in 29 turns), as R142b did;
- rig 1 hit the agent's 12-minute cap on JS `variable-length-quantity` (9 turns). Its tests passed on the code the
  agent left.

Every value below is one run, labelled with its rig. "Equal work" means R90's `v3` work (its prompt and generated
tokens, and its time outside the server) at each arm's own prompt and generation speeds
([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). It compares the engines on the same work; the
minutes for all tasks depend on what each attempt generated.

## The line

`v4r` is `v4c3`'s environment and command (R142a, R142b) on `v4-rc2`'s rig build, with exactly these differences:

| Change | Why |
|---|---|
| `LLAMA_PP_MULTI_SPLIT=1` removed | R141: C812's prompt split costs 1.39 s a pass, beyond the spread (a loss) |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` in place of `=1` | R143: the shared scheduler is planned over the whole memory again whenever it re-plans |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0` added | R150 and C828: the shared expert's Q4_0 and Q5_0 down projections leave the k-column kernel at 1–4 tokens |
| `--spec-draft-n-ubatch 512` kept | `v4c3`'s value. The line already passes the flag that upstream #32078 needs for the MTP draft, and 512 reads prompts faster than 128 (R53) |
| the union gather (C103) stays off | R88 part (2a): a loss in the decode step at every depth with context |

The build is `v4-rc2`'s rig tarball (C828's, `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`), where `v4c3` ran
on C825's tip tarball `65ac9a47d`. The started command (environment, binary and arguments) is R142b's string with the
three switch changes made and the build folder swapped, and with no other difference: the two strings match
byte for byte after those edits (md5 `6b5dac04b78dfe51cfc94cf55bda79fc`). The server's `switches:` line reads
`39 set`. Three warnings in the server log are not errors:

- `LLAMA_SCHED_PER_SIZE_RESERVE` is ignored in one context that has no per-size schedulers. `v4c3`'s two logs show the
  same line.
- Under `=2` the 512-token graph needed a new plan once.
- No line about the draft's graph budget appears (upstream #32078).

R156b's arm `v4r2` is the same line. Its started command (environment, binary and arguments) equals R156a's as a
string, character for character: both rigs keep the model files and the build at the same paths. Its server log has
the same `switches:` line (`39 set`, the same md5 as R156a's) and the same two scheduler lines, and no line about the
draft's graph budget.

## The run

- **R156a** ran on rig 2 on 2026-11-13. The server started at 18:15:30 and was up after 225 s. On rig 1, R142a's server
  took 309 s and R142b's 313 s. The 12 tasks ran 18:19:17–18:52:50, and the report ran 18:52:56–18:53:02.
- **R156b** ran on rig 1 on 2026-11-13. The server started at 18:57:39 and was up after 315 s. The 12 tasks ran
  19:02:57–19:34:02, and the report ran 19:34:32–19:34:44.

The client ran on the build machine's CPU at `--cpu-shares 256`, with no GPU. Both runs wrote into R72's client folder,
beside R72's, R142's and R154's arms.

## Both v3.0 runs (R158)

R158 ran the v3.0 release on rig 1: the tag `miner-llama-v3.0` (`1551f5fe1`, `pascal-p106-v3-b12076-ladder5`; the
tree's `llama-server` md5 `c735e40f184f3de61f16ffd5e2ab5ea5`) with the preset as shipped
([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)), Swift Q2_0 and the MTP draft. The
started command (environment, binary and arguments, 659 bytes) equals R90's `v3` byte for byte. The server started at
19:56:46 and was up after 311 s (R90's `v3`: 313 s). The 12 tasks ran 20:02:01–20:40:24, and the report ran
20:40:46–20:40:50. The client and its folder were R156's.

| | R90 `v3` (rig 1) | **R158 `v3b` (rig 1)** | R156a `v4r` (rig 2) | R156b `v4r2` (rig 1) |
|---|---|---|---|---|
| passed | 11/12 | **11/12** | 12/12 | 12/12 |
| equal work, min a task | 3.08 | **3.19** (+3.6 %) | 2.43 | 2.37 |
| equal work on the 10 tasks neither v3.0 run ran away on | 2.24 | **2.21** (−1.3 %) | 1.52 | 1.52 |
| turns | 66 | **85** | 95 | 69 |
| generation tok/s | 20.66 | **18.92** | 26.67 | 27.61 |
| generation tok/s, the 10 tasks | 20.90 | **20.98** | 27.52 | 27.53 |
| prompt tok/s | 65.5 | **66.3** | 133.9 | 144.6 |
| prompt tok/s at 1,024–1,999, 2,000+ new tokens | 90.5, 93.3 | **86.8, 104.0** | 152.4, 181.7 | 185.5, 177.5 |
| first token median, p90 s | 6.52, 16.17 | **6.51, 16.63** | 2.42, 9.00 | 2.48, 8.65 |
| first token on a task's first turn, median s | 13.54 | **13.84** | 6.68 | 6.65 |
| acceptance, tokens a step | 0.696, 3.09 | **0.671, 3.01** | 0.749, 3.25 | 0.759, 3.28 |
| minutes, all 12 tasks (wall) | 37.0 | **38.2** | 33.4 | 30.6 |

- **Each v3.0 run had one task run to the agent's 12-minute cap, and failed it.** R90's was JS
  `variable-length-quantity` (2 turns); `v3b` passed it in 212 s. `v3b`'s was `complex-numbers`: 26 turns, 25 tool
  calls and 7,923 drafted tokens at 0.542 acceptance ([task_acceptance.tsv](task_acceptance.tsv)); R90's `v3` passed
  it in 154 s. Those long turns at a growing context generate slowly, which is why `v3b`'s 12-task generation is 18.92
  tok/s.
- **On the 10 tasks neither run ran away on ([agent_10.tsv](agent_10.tsv)), the two v3.0 runs agree:** 2.24 and 2.21
  minutes a task at equal work, 20.90 and 20.98 tok/s. `v4r`'s two runs take 1.52 and 1.52 minutes there (−32.1 %
  against R90's `v3`), at 27.52 and 27.53 tok/s.
- **Without `complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv)):** `v3b` 3.09 minutes a task, 20.82 tok/s; R90's `v3`
  3.13 and 20.92. On R90's 11 ([agent_11.tsv](agent_11.tsv)), which keep `v3b`'s runaway: 2.41 against 2.27, 18.81
  against 20.63 tok/s.
- **Draft acceptance by task without `complex-numbers`:** 14,691 of 20,394 drafted tokens (0.720).
- **Per task** against R90's `v3` on the same rig, seconds: `affine-cipher` 200 (248), `space-age` 71 (84), Python
  `phone-number` 95 (142), `triangle` 69 (74), Python `variable-length-quantity` 389 (110), JS `phone-number` 82 (83),
  `list-ops` 109 (161), `grade-school` 66 (86), `scale-generator` 139 (183), `complex-numbers` 720, failed (154),
  `pig-latin` 141 (176), JS `variable-length-quantity` 212 (720, failed).
- **Host** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 209 MiB at 20:02:35, in the first
  task; swap grew from 35 to 628 MiB by 20:07:19 and then stayed flat. The watchdog did not act. R90's `v3`: 66 MiB,
  swap 24 → 963 MiB.
- **Cards** (peak MiB, 19:56:46–20:40:50): 0:5,101 1:5,137 2:5,005 3:5,153 4:4,913 5:5,021 6:4,423 7:5,033 8:4,625.
  Card 8 matches R90's v3.0 loads (4,627 and 4,625 in its deep lane).
- **Server log:** 13 checkpoint restores and 1 erased; one full re-processing at 20:34:30, where `complex-numbers`
  ended and `pig-latin` began (1,553 tokens shared with the cache, no checkpoint below them), as in R156a. No NaN or
  inf, no watchdog kill, no error line.
- **The report** ([v3b_gonogo.md](v3b_gonogo.md), over every arm in the folder): GO on infrastructure (120 of 120
  attempts), tool calls (0 malformed of 729), first token against prefill and cache. NO-GO on proxy rows: `v3b`'s
  `complex-numbers` (26 rows, 25 replies) is one of the attempts listed, the cap having cut off its last reply. NO-GO on
  memory: it reads `meminfo.log` over every arm and both rigs (215 MB at its lowest); `v3b`'s own lowest is the
  209 MiB above.

## All 12 tasks ([agent.tsv](agent.tsv))

| | R90 `v3` | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** | **R156b `v4r2` (rig 1)** |
|---|---|---|---|---|---|
| passed | 11/12 | 12/12 | 12/12 | **12/12** | **12/12** |
| equal work, min a task | 3.08 | 2.35 | 2.44 | **2.43** (−21.1 % against `v3`) | **2.37** (−23.1 %) |
| turns | 66 | 66 | 98 | **95** | **69** |
| generation tok/s | 20.66 | 28.03 | 26.16 | **26.67** | **27.61** |
| prompt tok/s | 65.5 | 145.5 | 136.6 | **133.9** | **144.6** |
| first token median, p90 s | 6.52, 16.17 | 2.45, 8.39 | 2.45, 9.77 | **2.42, 9.00** | **2.48, 8.65** |
| first token on a task's first turn, median s | 13.54 | 6.64 | 6.62 | **6.68** | **6.65** |
| acceptance | 0.696 | 0.765 | 0.723 | **0.749** | **0.759** |
| tokens a step | 3.09 | 3.29 | 3.17 | **3.25** | **3.28** |
| minutes, all 12 tasks (wall) | 37.0 | 16.4 | 32.2 | **33.4** | **30.6** |

## The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

These leave out `bugfix-javascript-variable-length-quantity`, which R90's `v3` did not pass. At equal work against
`v3`'s 2.27 minutes a task:
- `v4r` takes **1.59 minutes a task** on rig 2 (−30.0 %) and **1.51** on rig 1 (−33.5 %);
- `v4c3` took 1.49 and 1.61.

Generation is 26.13 tok/s on rig 2 and 27.57 on rig 1, against `v4c3`'s 28.01 and 25.84.

## Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

| | R90 `v3` | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** | **R156b `v4r2` (rig 1)** |
|---|---|---|---|---|---|
| equal work, min a task | 3.13 | 2.45 | 2.43 | **2.46** | **2.46** |
| turns | 59 | 61 | 61 | **66** | **64** |
| generation tok/s | 20.92 | 28.00 | 27.79 | **27.59** | **27.58** |
| prompt tok/s | 66.6 | 141.6 | 152.0 | **140.7** | **140.8** |
| first token median, p90 s | 6.52, 16.46 | 2.45, 8.26 | 2.50, 8.81 | **2.46, 8.80** | **2.48, 8.30** |
| acceptance | 0.696 | 0.765 | 0.723 | **0.749** (0.775 by task, [task_acceptance.tsv](task_acceptance.tsv)) | **0.759** (0.757 by task) |

## Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

| New tokens | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** | **R156b `v4r2` (rig 1)** |
|---|---|---|---|---|
| 0–511 | 94.0 tok/s (42 turns) | 93.0 (60) | **93.0** (66) | **96.4** (44) |
| 512–1,023 | 124.4 (10) | 111.2 (17) | **112.3** (11) | **123.0** (10) |
| 1,024–1,999 | 186.5 (9) | 162.1 (14) | **152.4** (11) | **185.5** (9) |
| 2,000+ | 180.6 (5) | 181.0 (7) | **181.7** (7) | **177.5** (5) |
| first token median at 2,000+, s | 9.38 | 12.50 | **12.79** | **9.61** |

## The draft's acceptance ([task_acceptance.tsv](task_acceptance.tsv))

- **`v4r` on rig 2:** 28,900 of 38,592 drafted tokens accepted (0.749). Without `complex-numbers`, 21,634 of 27,927
  (0.775). In `complex-numbers` alone, 7,266 of 10,665 (0.681).
- **`v4r` on rig 1:** 17,940 of 23,655 (0.758). Without `complex-numbers`, 17,323 of 22,890 (0.757).
- **`v4c3`** without `complex-numbers`: 0.763 and 0.759.

## Graph reallocations (C804)

`v4r`: 0 of 95 requests (rig 2) and 0 of 68 (rig 1) reallocated the target's graph. `v4c3`: 0 of 66 and 0 of 98.
The counts come from the report's reading of the server logs ([v4r_compare.md](v4r_compare.md),
[v4r2_compare.md](v4r2_compare.md)).

## The host and the cards (rig 1, R156b)

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 419 MiB, at 19:04:24 at the end of
  the first task. Swap grew from 36 to 404 MiB between 19:04:30 and 19:09:05 (tasks 2–5) and then stayed flat. 708 MiB
  were available at the end, and the watchdog did not act. On the same rig, `v4c3` read 478 and 376 MiB, with swap
  5 → 475 and 8 → 8 MiB.
- **Cards** (peak MiB, 18:57:34–19:34:45): 0:5,099 1:5,145 2:5,001 3:5,161 4:4,259 5:5,017 6:4,417 7:5,029
  **8:5,619**. Cards 0–7 equal R142a's on the same rig. Card 8, which holds the draft, peaked at 5,535 MiB in R142a
  and 5,701 in R142b, and at 5,623 in R156a on rig 2.
- **Server log:** 12 checkpoint restores, none erased, no full re-processing, no NaN or inf, no watchdog kill. One task
  was cancelled at the agent's cap (below, "Per task").
- **The report** ([v4r2_gonogo.md](v4r2_gonogo.md), over every arm in the folder):
  - GO: infrastructure (108 of 108 attempts), tool calls (0 malformed of 649; `v4r2` made 65 calls, 5 returned errors),
    first token against prefill, and cache.
  - NO-GO, proxy rows: R72b's runaway task, and `v4r2`'s JS `variable-length-quantity` (9 rows, 8 replies), whose last
    reply the 12-minute cap cut off.
  - NO-GO, memory: it reads `meminfo.log` over every arm and both rigs (276 MB at its lowest), so it is not `v4r2`'s;
    `v4r2`'s own lowest is the 419 MiB above.

## The host and the cards (rig 2, R156a)

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 312 MiB, at 18:52:33 in the last
  task. Swap stayed at 21 MiB from start to end, and the watchdog did not act. On rig 1, `v4c3` read 478 and 376 MiB,
  with swap 5 → 475 and 8 → 8 MiB.
- **Cards** (peak MiB, 18:15:30–18:53:02): 0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045
  **8:5,623**. On rig 1, R142b peaked at 0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045 8:5,701, so
  cards 0–7 match and card 8 peaks 78 MiB lower. R142a's card 8 peaked at 5,535 MiB.
- **Server log:** 12 checkpoint restores and 1 old checkpoint erased. One full re-processing came at 18:39:32, where
  `complex-numbers` ended and `pig-latin` began (1,553 tokens shared with the cache, no checkpoint below them), as in
  R142b. No NaN or inf, no watchdog kill.
- **The report** ([v4r_gonogo.md](v4r_gonogo.md), over every arm in the folder):
  - GO: infrastructure (96 of 96 attempts), tool calls (0 malformed of 584; `v4r` made 90 calls, 3 returned errors),
    first token against prefill, and cache.
  - NO-GO, proxy rows: only R72b's runaway task.
  - NO-GO, memory: it reads the rig's `meminfo.log` over every arm in the folder. That file now holds rows from both
    rigs, so its lowest point is not `v4r`'s; `v4r`'s own lowest is the 312 MiB above.

## Per task ([v4r_compare.md](v4r_compare.md), [v4r2_compare.md](v4r2_compare.md))

**R156b (`v4r2`, rig 1)**, minutes against R142a and R142b on the same rig:
- **Quicker than both in 1 task:** `affine-cipher` 1.7.
- **Between the two runs or equal to one in 7:** `space-age` 0.8, Python `variable-length-quantity` 4.4, JS
  `phone-number` 0.9, `list-ops` 1.6, `grade-school` 0.9, `pig-latin` 2.2, and `complex-numbers` 1.0 in 5 turns (the
  short path, as in R142a).
- **Slower than both in 4:** Python `phone-number` 1.5, `triangle` 1.1 in 6 turns, `scale-generator` 2.6, and JS
  `variable-length-quantity` 12.0. That last task ran to the agent's 12-minute cap in 9 turns (4 tool calls returned
  errors, 3,486 drafted tokens), and its tests passed on the code the agent left.

**R156a (`v4r`, rig 2)**, minutes against R142a and R142b (rig 1):
- **Quicker than both in 5 tasks:** Python `phone-number` 0.6, `space-age` 0.7, Python `variable-length-quantity`
  1.0 (against 2.2 and 4.4), `list-ops` 1.1 and `scale-generator` 1.4.
- **Between the two runs or equal to one in 4:** `triangle` 0.7, JS `phone-number` 1.3, `grade-school` 0.9 and
  `complex-numbers` 9.7 in 29 turns (against 0.9 and 10.8).
- **Slower than both in 3:** `affine-cipher` 2.7 in 11 turns, `pig-latin` 3.7 (4,569 drafted tokens against 2,373 in
  R142a), and JS `variable-length-quantity` 9.6 in 5 turns (13,197 drafted tokens against 1,815 and 4,815). That task
  wrote long replies, which adds wall-clock minutes; equal work leaves those minutes out.

## Commands

R156a, from the repository root on a machine that only runs ssh. It used R72's client folders, whose bench files R142a
had brought up to date, so no prep step ran. It had its own ports, so that R156b can run on the other rig at the same
time:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4r V4R_DRAFT_UB=512 \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad v4r" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r156_agent_v4r/run/r156.sh > r156a.log 2>&1 < /dev/null &
```

R156b ran the same command on the other rig, with R142's ports and these changes:
`AGENT_ORDER=v4r2 REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad v4r v4r2"`.

The report rewrites the client folder's `res/turns.csv` with exactly the arms it is given. So `REPORT_ARMS` names every
arm in R72's folder that the tables read; R90's `v3` comes from R90's own folder. The tables were made on the client's
CPU in an empty folder with R142's [tables.sh](../2026-11-13_r142_agent_v4c3/run/tables.sh) (`nice -n 19`), followed
by the cards' peaks. These are R156b's; R156a's used `NEWARMS="v4c3 v4c32 v4r"` and its window, 18:15:30–18:53:02.
R156b's tables add `v4r2`'s rows and leave every other row as R156a wrote it:

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> NEWARMS="v4c3 v4c32 v4r v4r2" \
  bash <this repository>/docs/data/2026-11-13_r142_agent_v4c3/run/tables.sh
awk -F', ' -v a='2026/11/13 18:57:34' -v b='2026/11/13 19:34:45' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

R158 ran [run/r158.sh](run/r158.sh), R156's driver with the arm `v3b` (R90's `v3` line on `REF`), on rig 1 with R156b's
ports and client folders:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 REF=llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5 \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v3b \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad v4r v4r2 v3b" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r156_agent_v4r/run/r158.sh > r158.log 2>&1 < /dev/null &
```

Its tables add `v3b`'s rows to R156b's and leave every other row as it was; `agent_10.tsv` leaves out both tasks a
v3.0 run ran to the cap:

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> NEWARMS="v4c3 v4c32 v4r v4r2 v3b" \
  bash <this repository>/docs/data/2026-11-13_r142_agent_v4c3/run/tables.sh
SKIP_TASKS=bugfix-javascript-variable-length-quantity,bugfix-javascript-complex-numbers OUT_TAG=_10 \
  nice -n 19 python3 -I <this repository>/docs/data/2026-11-05_keepalive/run/agent_tables.py merged \
  r90v3 r90gains gains v4c2 v4c22 v4c2g v4c3 v4c32 v4r v4r2 v3b
awk -F', ' -v a='2026/11/13 19:56:46' -v b='2026/11/13 20:40:50' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv) | each arm against R90's `v3`: all 12 tasks, R90's 11, and the 12 without `complex-numbers` |
| [agent_10.tsv](agent_10.tsv) | the same without both tasks a v3.0 run ran to the cap (JS `variable-length-quantity` and `complex-numbers`) |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt speed and first token by a turn's new prompt |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's acceptance by task |
| [host.tsv](host.tsv) | lowest MemAvailable and swap of `v4c3`, `v4c32`, `v4r`, `v4r2` and `v3b` (each on its own rig) |
| [v4r_compare.md](v4r_compare.md), [v4r_report_table.md](v4r_report_table.md), [v4r_gonogo.md](v4r_gonogo.md) | the agent check's report over R72's, R142's, R154's and R156a's arms |
| [v4r2_compare.md](v4r2_compare.md), [v4r2_report_table.md](v4r2_report_table.md), [v4r2_gonogo.md](v4r2_gonogo.md) | the same report with R156b's arm `v4r2` added |
| [v3b_compare.md](v3b_compare.md), [v3b_report_table.md](v3b_report_table.md), [v3b_gonogo.md](v3b_gonogo.md) | the same report with R158's arm `v3b` added |
| [run/r156.sh](run/r156.sh) | the driver: R142's with the arms `v4r` and `v4r2` |
| [run/r158.sh](run/r158.sh) | R156's driver with the arm `v3b` (R90's `v3` line) |

The tasks are the agent check's bug-fix set ([bench/agent_check](../../../bench/agent_check/README.md)); the model files
are the maker's (Swift GSQ-RCO Q2_0) and the MTP draft, as in R142.
