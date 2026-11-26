# C912: Flash-Next's coding-agent check on upstream b12674 merged into v5 (2026-11-24)

Registry row [Q1649](../../../quants/experiments.md), the merge of upstream llama.cpp's tag b12674 into the v5 engine
([2026-11-24_c912_upstream_b12674](../2026-11-24_c912_upstream_b12674/README.md)). The merge (the candidate branch
`pascal-p106-v5-cand-b12674`, `eee615cdb`) passed its checks on the build machine and its P106 check
([2026-11-24_c912_p106_check](../2026-11-24_c912_p106_check/README.md)), and Ornith's coding-agent run on it
([2026-11-24_c912_ornith_agent](../2026-11-24_c912_ornith_agent/README.md)). Upstream changed the server and the
scheduler, so before `pascal-p106-v5` moves to the candidate each model's coding-agent check runs on it. This folder is
Flash-Next's: one run on the candidate, then one control run of the same launcher on `pascal-p106-v5` `af00bb13b` on the
same rig and boot, the engine the only difference.

**b12674 loses nothing on Flash-Next.** On the same rig and boot the candidate passed 12 of 12 tasks and the control 11
of 12. The control's JavaScript `complex-numbers` ran to the turn cap (41 turns), a path this task has taken on the v4
line before. No measure is a loss by the class rule ([class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv)),
with or without that task:
- **Minutes and generation:** without `complex-numbers` (both runs 11 of 11, 55 turns each) a task took 2.42 minutes
  at equal work against 2.45, and the model generated 28.58 tok/s against 27.56.
- **Decode step:** 117.06 against 117.35 ms over all requests. So the faster generation comes from the draft's
  acceptance (0.781 against 0.742), which follows the text the model sampled, not from a faster engine.
- **Prompts:** the 13 requests both runs read on the same prompt and cached prefix took 93.71 s against 92.75 s
  (1.0 % more), inside the 1.6 % between R227's two runs of the same command on the 15 such requests they share.
  Eight of the 13 were faster on the candidate; two first turns were slower (by 19.4 % and 7.3 %).
- **No regressions:** no server error, no reply read again, no graph reallocation, the same peak memory on every card.

With the P106 check and Ornith's agent run, all three checks for moving `pascal-p106-v5` to `eee615cdb` have passed.
One change shows in the server log ([grammar.tsv](grammar.tsv), registry Q1813): the candidate counted every token it
sampled as sampled under the tool-call grammar (32,088 of 32,088), where the control counted 6,715 of 26,635 and R227's
runs 3,947 of 31,373 and 3,561 of 18,343. Ornith's candidate run shows the same. It cost neither model a task, a
malformed call or decode speed.

## What runs

Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks, Python and
JavaScript alternating, a 12-minute cap a task, a 40-minute budget): the maker's Swift GSQ-RCO Q2_0
(`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram table in the third shard) with the MTP draft
(`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090 cards, Flash-Next's v5 preset as it is
([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env): the sparse ratio 0.75 and
`--ubatch-taper short=1400:64`). The rig is the first one at stock (75 W, 1911/4004 MHz maximum, no clock offsets), on a
fresh boot. The machine that drives the rig is the agent's client.

| # | Run | Engine | `llama-server` md5 | Status |
|---|---|---|---|---|
| 1 | `f5cand` | the candidate `pascal-p106-v5-cand-b12674` `eee615cdb` (upstream b12674 merged) | `2727faf28b5b34dfe42399a9d431da67` | done 20:15–20:45 |
| 2 | `f5ref` | `pascal-p106-v5` `af00bb13b` (the control) | `f5cca34ea6bd4f3712cd6193bf0605be` | done 20:51–21:21 |

**How the preset is started.** R227's [run/flash_taper.sh](../2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh)
with `ARM=t1400a`: C854's launcher ([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver
[r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm `v4r`, then the report) with
`--ubatch-taper short=1400:64`, the line the v5 preset took from R227's verdict. The candidate's started command (1,717
bytes, md5 `e88afbaa6d856ae489378b407852abc3`) is R227's `t1400a` command byte for byte with the engine's path replaced;
the control's (1,717 bytes, md5 `0e698b0cd0d0924550115c340f60e4ff`) differs from the candidate's in the engine's path
only. Each was printed before its run by R227's [run/dry_line.sh](../2026-11-20_r227_flash_taper1400_agent/run/dry_line.sh)
and equals the started command byte for byte, and [run/preset_check.sh](run/preset_check.sh) finds both equal to
`settings.env` (the 40 switches byte for byte, each flag with its value). Both servers read `short=1400:64` at their
load and logged the same 40 switches as R227's runs (only the build's count of the names it reads differs: 960 on the
candidate, 905 on the control).

## The two runs

| | **`f5cand`** (candidate `eee615cdb`) | **`f5ref`** (control `af00bb13b`) | context: R227 `t1400a` | context: R227 `t1400b` |
|---|---|---|---|---|
| passed | **12/12** | **11/12** (JavaScript `complex-numbers` at the turn cap, 41 turns) | 12/12 | 12/12 |
| minutes a task at equal work (R90 `v3`'s work) | **2.34** | **2.40** | 2.25 | 2.24 |
| minutes, all tasks; turns | **23.2; 62** | **24.0; 96** | 21.5; 66 | 14.6; 67 |
| generation tok/s over the session | **28.59** | **26.78** | 30.71 | 30.87 |
| ms a decode step, all requests; at 4,000 to 8,000 tokens of context | **117.02; 116.88** | **118.80; 117.19** | 107.18; 106.71 | 105.95; 106.70 |
| ms a decode step at the two runs' shared mix of contexts | **116.86** | **118.25** | | |
| the draft's acceptance; tokens a step | **0.781; 3.34** | **0.724; 3.17** | 0.762; 3.29 | 0.753; 3.26 |
| prompt tok/s; prompt s a task (all) | **144.7; 20.27 (243.20)** | **144.3; 32.60 (391.15)** | 157.9; 20.37 (244.47) | 158.8; 20.80 (249.63) |
| the 13 requests read on the same prompt and cached prefix in both runs, prompt s | **93.71** | **92.75** | | |
| the 12 read the same in all four runs, prompt s; of them the 11 first turns | **85.76; 77.44** | **84.74; 75.97** | 79.07; 71.29 | 77.83; 70.01 |
| first token median; p90; a task's first turn median, s | **2.51; 8.25; 5.11** | **2.59; 8.34; 5.15** | 2.32; 7.88; 4.67 | 2.83; 7.90; 4.62 |
| longest first-token wait s (task, turn) | **28.58** (`affine-cipher`, 1) | **28.52** (the same) | 27.37 (the same) | 27.53 (the same) |
| prompt tokens read a task; tokens generated a task | **2,933; 2,673** | **4,704; 2,218** | 3,216; 2,613 | 3,304; 1,527 |
| reply tokens re-read (turns) | **0 (0)** | **0 (0)** | 0 (0) | 215 (1) |
| checkpoints restored / created / erased; full re-processing; graph reallocations | **11 / 0 / 0; 0; 0** | **10 / 0 / 2; 1; 0** | | |
| tool calls / malformed | **58 / 0** | **92 / 0** | | |
| tokens sampled under the tool-call grammar, of all sampled | **32,088 of 32,088** | **6,715 of 26,635** | 3,947 of 31,373 | 3,561 of 18,343 |
| card peak MiB, cards 0–8 | **5,192 5,236 5,094 5,250 4,350 5,108 4,504 5,120 5,698** | **5,192 5,236 5,094 5,250 4,350 5,110 4,504 5,122 5,700** | | |
| lowest MemAvailable MiB; swap highest MiB | **322; 623** | **416; 476** | | |
| busy SM clock, mean of the cards' medians, MHz | **1,879.8** | **1,878.3** | | |
| server load s | **312** | **313** | 226 | 227 |

Each value is one run. **The context columns are not a comparison of engines:** R227's two runs of the same started
command ran on the second rig with its soaked clock profile (R219's memory and core offsets on every card) on
`pascal-p106-v5` `602691153`, so their faster decode step (about 10 ms less) and prompts are the clocks and the rig as
much as the engine. The class uses them for one thing only, the gap between two runs of the same command.

**Without `complex-numbers`** ([agent_nocap.tsv](agent_nocap.tsv), [work_nocap.tsv](work_nocap.tsv),
[steps_nocap.md](steps_nocap.md)), every run doing the same 11 tasks:

| | **`f5cand`** | **`f5ref`** | R227 `t1400a` | R227 `t1400b` |
|---|---|---|---|---|
| passed; turns | **11/11; 55** | **11/11; 55** | 11/11; 60 | 11/11; 62 |
| minutes a task at equal work | **2.42** | **2.45** | 2.34 | 2.34 |
| generation tok/s | **28.58** | **27.56** | 30.73 | 30.86 |
| ms a decode step, all requests; the draft's acceptance; tokens a step | **117.06; 0.781; 3.35** | **117.35; 0.742; 3.23** | 107.17; 0.762; 3.29 | 105.91; 0.752; 3.27 |
| ms a decode step by context: 0 to 4,000; 4,000 to 8,000; 8,000 to 16,000 (requests) | **113.23 (13); 116.89 (33); 117.47 (9)** | **116.57 (17); 117.05 (34); 120.76 (4)** | | |
| prompt tok/s | **146.4** | **147.2** | 155.0 | 153.9 |
| first token median; p90; a task's first turn median, s | **2.58; 8.38; 5.09** | **2.81; 8.38; 5.06** | 2.32; 7.88; 4.58 | 2.83; 7.86; 4.56 |
| prompt tokens read a task | **2,950** | **3,112** | 3,013 | 3,178 |

## The class

The rule ([run/class.py](run/class.py), fixed and pushed before the control's tasks began): one run of each engine
on the same rig and boot gives neither a spread of its own, so the band is the gap between R227's two runs of the same
started command, but never less than 1 % of the control's value. R227's runs agreed to the last digit on some measures,
closer than two runs on one rig have agreed before (R156's two v4.0 runs: 2.43 and 2.37 minutes a task at equal work).
A measure is a gain or a loss only where the candidate differs from the control by more than the band.

| measure | all 12 tasks ([class.tsv](class.tsv)) | without `complex-numbers` ([class_nocap.tsv](class_nocap.tsv)) |
|---|---|---|
| minutes a task at equal work | gain (2.34 against 2.40, −2.5 %) | gain (2.42 against 2.45, −1.2 %; band 0.0245) |
| generation tok/s | gain (28.59 against 26.78, +6.8 %) | gain (28.58 against 27.56, +3.7 %) |
| ms a decode step, all requests | gain (117.02 against 118.80, −1.5 %; band 1.23) | negligible (117.06 against 117.35, −0.2 %) |
| prompt tok/s | negligible (144.7 against 144.3) | negligible (146.4 against 147.2, −0.5 %) |
| first token median | negligible (2.51 against 2.59 s) | negligible (2.58 against 2.81 s; band 0.51) |
| first token p90 | gain (8.25 against 8.34 s) | negligible (8.38 against 8.38 s) |
| first token on a task's first turn | negligible (5.11 against 5.15 s) | negligible (5.09 against 5.06 s) |
| prompt tokens read a task | gain (2,933 against 4,704) | negligible (2,950 against 3,112; band 165.7) |
| reply tokens re-read a task | negligible (0 against 0) | negligible (0 against 0) |
| longest first-token wait | negligible (28.58 against 28.52 s) | negligible (the same) |

- **No loss in either table.** The gains in the full table come mostly from the control's capped task: 41 turns at up
  to 33,111 tokens of context, which add prompt tokens, deeper and slower decode steps (123.82 ms at 16,000 tokens and
  more, a depth the candidate never reached) and a full re-processing of the next task's first turn after two
  checkpoints were erased.
- **Generation:** without that task the decode step is the same (117.06 against 117.35 ms), so the equal-work minutes
  and the generation, still gains by the rule, come from the draft accepting more of the model's text (0.781 against
  0.742). That follows what the model sampled at temperature 1.0, so it is not counted as the engine's gain. At the
  two runs' shared mix of contexts ([steps_mix.md](steps_mix.md)) a step took 116.86 ms against 118.25. Most of that
  gap is at 8,000 to 16,000 tokens (117.47 against 119.85 ms; 12 of the control's 16 requests there are
  `complex-numbers`'), and at 4,000 to 8,000 tokens, where half the shared tokens are, a step took 116.88 against
  117.19 ms.
- **Prompts on the same work: negligible.** 13 requests read the same new tokens on the same cached prefix in both runs
  ([same_prompts_pair.md](same_prompts_pair.md)). They took 93.71 s on the candidate against 92.75 s (+1.0 %); R227's
  two runs took 92.74 and 91.32 s on the 15 such requests they share (1.6 % apart,
  [same_prompts_refs.md](same_prompts_refs.md)). Eight of the 13 were faster on the candidate and three slower (by
  0.2 %, 0.2 % and 1.7 %). Two first turns were slower by more: JavaScript `space-age` (803 new tokens, 6.22 s against 5.21) and
  Python `phone-number` (1,130 new tokens, 7.30 s against 6.81). Without those two the candidate was 0.7 % faster
  (80.19 s against 80.74). Both are read in the taper's 64-token micro-batches, as 10 of the 13 are. Ornith's prompts
  were 3.2 % faster on the candidate; Flash-Next's 64-token micro-batches are where upstream's 128-row prompt tiles
  (#31352) apply, which rig 1's row 12b times on their own (Q1757). Registry Q1814 keeps the two slow turns, parked.

## The tool-call grammar on b12674

The server's sampling line counts the tokens sampled while the tool-call grammar applied ([grammar.tsv](grammar.tsv)).
On `af00bb13b` and `602691153` the grammar is lazy: it starts at a tool call's opening. 11 and 12 of the requests sample
no token under it, and a part of the sampled tokens is counted: 6,715 of 26,635 in the control (25.2 %), 3,947 of
31,373 and 3,561 of 18,343 in R227's runs (12.6 % and 19.4 %). On the candidate all 62 requests count every token (32,088 of 32,088). Ornith's
candidate run on the second rig counts the same: 16,259 of 16,259 tokens, against 3,924 of 18,433 and 3,247 of 23,708
in its two control runs. Two upstream changes in b12674 touch when the grammar starts: #32497 (a pattern trigger on the
opening of a bare JSON tool call) and #29307 (token-aware parsing, which reworks the Qwen3-Coder parser). On both models
no task failed for it and no call was malformed (0 of 58 here, 0 of 59 on Ornith). One token was rejected under the grammar
in each run (none in R227's second), and the decode step was not slower. Only the count changed
([2026-11-26_c1080_grammar_start](../2026-11-26_c1080_grammar_start/README.md), Q1813): since #29307 the Qwen3-Coder
grammar runs from the first token but holds back no text before a call, and greedy replies and the draft's acceptance
are the same with and without it.

## How it ran

- **The rig:** the first rig, booted by the owner at 20:11:04, at stock. Before each run: no llama process, 0 MiB on all
  nine cards, 75 W and 1911/4004 MHz maximum, no GPU Xid in the boot's kernel log. Nothing ran on it between the runs or
  during them but the launcher.
- **`f5cand`:** launched 20:15:47, server up after 312 s, tasks 20:21:09–20:44:32, driver and report exit 0 at 20:44:48.
- **`f5ref`:** launched 20:51:04, server up after 313 s, tasks 20:56:28–21:20:37, driver and report exit 0 at 21:20:53.
  At the end: no llama process, 0 MiB on every card, no Xid, the rig folder removed by the harness.
- **Answers sound in both:** no NaN or inf, no graph reallocation, no server error. The candidate's server log holds
  the same seven warning lines as R227's runs. The control's holds the same seven plus three from its capped task: two
  checkpoints erased at depth and the cap's cancelled request. The harness's report gives NO-GO on memory in both
  (lowest MemAvailable 322 and 417 MB), which every Flash-Next run on a 4 GB host fails. The control's report also gives
  NO-GO on its proxy rows: the cap cut `complex-numbers`' last request, so the task has 41 rows for 40 replies.
- **Text reuse** kept the cached tokens on 12 turns in the candidate and 21 in the control
  ([text_reuse_points.tsv](text_reuse_points.tsv)), so neither run read one of the model's replies again.
- **Clocks:** the busy SM clock's median was 1,847 1,847 1,898 1,885 1,873 1,911 1,873 1,911 1,873 MHz on cards 0 to 8 in
  the candidate and the same in the control except card 4 (1,860); the share of busy samples below 1,800 MHz by card is
  in [clocks.tsv](clocks.tsv).

## Files

| File | What it holds |
|---|---|
| [class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv) | `run/class.py`: the candidate against the control per measure, R227's gap, the band and the class; all 12 tasks, and without the task at its cap |
| [agent.tsv](agent.tsv), [agent_nocap.tsv](agent_nocap.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work); R227's two runs as context; the second without the capped task |
| [work.md](work.md), [work.tsv](work.tsv), [work_nocap.md](work_nocap.md), [work_nocap.tsv](work_nocap.tsv) | per run: prompt tokens read and generated, reply re-reads, prompt seconds, the longest first-token wait |
| [steps.md](steps.md), [steps_nocap.md](steps_nocap.md) | ms a decode step by the context at each request's middle (R203's buckets) |
| [steps_mix.md](steps_mix.md), [steps_mix_nocap.md](steps_mix_nocap.md) | `run/steps_mix.py`: the decode step at a shared mix of contexts, the two runs alone and with R227's |
| [same_prompts_pair.md](same_prompts_pair.md) | the requests the candidate and the control read on the same prompt and cached prefix, prompt ms each |
| [same_prompts.md](same_prompts.md), [same_prompts_refs.md](same_prompts_refs.md) | the same across all four runs, and in R227's two runs alone (the band for the pair) |
| [grammar.tsv](grammar.tsv) | per run: requests, tokens sampled, tokens sampled under the tool-call grammar, requests without it, rejections |
| [prompt_tasks.tsv](prompt_tasks.tsv) | per task and run: prompt seconds, prompt tokens read, turns; the mean a task |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per run: turns, new tokens and prompt seconds by the turn's new tokens (under 600, 600 to 1,023, 1,024 to 1,999, 2,000 and more) |
| [taper_lines.tsv](taper_lines.tsv) | the micro-batch taper each server read at its load |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run, and the median and p90 of the turns after a task's first |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | turns whose cached prefix stops inside the previous reply, per run and per turn (C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py)) |
| [text_reuse_points.tsv](text_reuse_points.tsv) | the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request |
| [clocks.tsv](clocks.tsv) | busy SM clock medians per card over the tasks |
| [host.tsv](host.tsv) | lowest MemAvailable, swap, card memory peaks |
| [reallocs.tsv](reallocs.tsv) | graph reallocations by prompt size |
| [capped.tsv](capped.tsv) | the tasks that ran to a cap (12 minutes or the turn cap) |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `f5cand_report_table.md`, `f5cand_compare.md`, `f5cand_gonogo.md`; the same for `f5ref` | the agent check's own report of each run |
| [run/](run/) | the scripts below |

## Commands

From the repository root on the machine that drives the rig and is the agent's client, one run a session, each with
its own `OUT`, rig folder, client folder and container prefix; only `NEW` and those names change between the two runs:

```bash
export ARM=t1400a PREFIX=acrel-c912f-f5cand RIG=<rig alias> CLIENT=local \
  MODEL=<models>/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 REF=<any build folder, unused by this arm> \
  RIGDIR=<a folder on the rig>/c912f-f5cand OUT=<run folder>/out RIG_PORT=8282 RELAY_PORT=8302 PROXY_PORT=18802 \
  CLIENT_DIR=<client>/bench CLIENT_OUT=<client>/c912f-f5cand
unset GAINS_ENV V4R_ENV V4C2_TAPER
bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh
```

The control: `NEW=llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2` and `f5ref` in place of `f5cand` in the
names. Before a run, the command it will start, and its check against the preset:

```bash
ARM=t1400a WORK=<an empty scratch folder> NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  bash docs/data/2026-11-20_r227_flash_taper1400_agent/run/dry_line.sh > dry_line_f5cand.txt
bash docs/data/2026-11-24_c912_flash_agent/run/preset_check.sh dry_line_f5cand.txt
```

The two runs' dry lines are 1,717 bytes each (md5 `e88afbaa6d856ae489378b407852abc3` for `f5cand`,
`0e698b0cd0d0924550115c340f60e4ff` for `f5ref`), different in the engine's path only.

The tables and the class, with Python 3's standard library only, from the runs' collected logs (the inputs are listed
at the top of [run/tables.sh](run/tables.sh)); the client's `res/` files go into `agent/<run>/` first:

```bash
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client>/c912f-f5cand/res \
  docs/data/2026-11-24_c912_flash_agent/agent/f5cand v4r f5cand
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client>/c912f-f5ref/res \
  docs/data/2026-11-24_c912_flash_agent/agent/f5ref v4r f5ref
bash docs/data/2026-11-24_c912_flash_agent/run/tables.sh <tables folder> f5cand f5ref
```

| Script | What it does |
|---|---|
| [run/preset_check.sh](run/preset_check.sh) | checks a started command against `release/flash-next-v5/settings.env` as it is |
| [run/tables.sh](run/tables.sh) | builds this folder's tables from the runs done so far (R227's `run/tables.sh` with R227's two runs as context rows), and with two runs the class, the shared-mix steps, the pair's same-prompt requests and the grammar counts |
| [run/class.py](run/class.py) | the class of the candidate's run against the control's, with R227's gap (at least 1 %) as the band |
| [run/steps_mix.py](run/steps_mix.py) | the decode step of each named run at one shared mix of contexts |

## Provenance and licence

Measured on the project's own rig on 2026-11-24 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<run>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
