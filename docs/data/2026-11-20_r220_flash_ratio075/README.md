# R220: Flash-Next's sparse attention from 6,153 cells, on the agent check (2026-11-20)

Registry row [Q1635](../../../quants/experiments.md), from R69 ([Q1252](../2026-11-20_r69_sparse_switch/README.md)).
On the deep lane, `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75` in place of the v5 preset's 1.0 moves Flash-Next's
four-token verify pass to the sparse attention kernel from 6,153 cached cells instead of 8,204, which made the steady
decode step 1.71 ms faster at 6,144 tokens and 3.50 ms faster at 7,168; the replies then differ from the preset's. This
folder decides, on the coding-agent check, whether 0.75 goes into Flash-Next's v5 preset.

**Run 1 of 4 done: the control, Flash-Next's v5 preset as it is, passed 12 of 12 tasks.** A task took 2.28 minutes at
equal work, the model generated at 30.28 tok/s over the session, and the prompts took 20.57 s a task (246.9 s in all,
37,117 tokens read). No turn read an earlier reply again; text reuse kept the cached tokens on 7 requests in 3 tasks.
For reference only (another boot, v4.0's preset): R209d on the same rig, engine and clock profile the day before took
2.26 minutes a task at equal work, generated at 30.69 tok/s and read prompts for 20.62 s a task, and the 13 requests
that read the same prompt on the same cached prefix in both runs took 92.59 s here against 92.67 s there. The started
command is R210's text-reuse run's byte for byte and matches `settings.env`; every card held the profile, the watchdog
never tripped and the boot logged no GPU error. No class is given yet: the verdict compares both arms after all four
runs.

**Run 2 of 4 done: with the ratio at 0.75, Flash-Next also passed 12 of 12 tasks.** A task took 2.26 minutes at equal
work (the control: 2.28), the model generated at 30.60 tok/s over the session (30.28), and the prompts took 21.50 s a
task (20.57), 258.0 s in all for 39,887 tokens read (246.9 s for 37,117): this run read more, most of it in one task
that took two more turns. The 14 requests that read the same prompt on the same cached prefix in both runs took 101.65 s
here against 101.57 s. On the 11 requests whose middle lay where 0.75 acts (6,153 to 8,204 cells) a decode step took
108.12 ms against the control's 114.81 ms on 11, but the two runs put different replies there (6,860 tokens against
2,229), and below 6,153 cells, where both arms run the same kernels, the two runs also differ by 1.01 ms a step. The
started command differs from the control's only in the ratio; every card held the profile, the watchdog never tripped
and the boot logged no GPU error. No class yet: the verdict waits for runs 3 and 4.

**Run 3 of 4 done: the control again passed 11 of 12 tasks.** JavaScript `complex-numbers`, which runs 1 and 2 passed
in 6 and 7 turns, ran to its 12-minute cap in 39 turns with 38 of its 40 tests passing, the long path R72b's run of
this model took to the cap in 35 turns on 2026-11-12. Over all 12 tasks a task took 2.32 minutes at equal work, the
model generated at 29.46 tok/s and the prompts took 32.80 s a task (393.6 s, 155.8 s of them in `complex-numbers`).
Without `complex-numbers` the figures are 2.34 minutes, 30.86 tok/s and 21.62 s a task (run 1: 2.36, 30.20, 20.65; run
2 with 0.75: 2.34, 30.71, 20.15). **The two controls differ** by 0.04 minutes a task at equal work over all tasks and
0.02 without `complex-numbers`, by 0.82 and 0.66 tok/s, and by 1.33 ms a decode step on the requests whose middle lay
where 0.75 acts (113.48 ms on 10 requests here, 114.81 on 11 in run 1; run 2 with 0.75: 108.12 on 11). The started
command is run 1's byte for byte; every card held the profile, the watchdog never tripped and the boot logged no GPU
error. No class yet: the verdict waits for run 4.

## What runs

Four runs of Flash-Next's agent check ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget) on one rig, one engine and one boot,
alternated so that each arm has one run in each half of the day:

| # | Run | Arm | `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO` | Status |
|---|---|---|---|---|
| 1 | R220a | `c10a` | 1.0, the v5 preset as it is (the control) | done 11:21–11:39 |
| 2 | R220b | `r075a` | 0.75 | done 11:49–12:10 |
| 3 | R220c | `c10b` | 1.0 (the control) | done 12:25–12:59 |
| 4 | R220d | `r075b` | 0.75 | planned |

Every run: the maker's Swift GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, the n-gram
table in the third shard) with the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`) at 65,536 tokens on nine P106-090
cards, Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)) on
the v5 tree `pascal-p106-v5` `602691153` (`llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`). The rig is the
second one, with its own soaked clock profile set on this boot
([R219](../2026-11-19_r209_oc_v5/r219_apply.md): memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s
and core +256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8, 75 W, the kit's watchdog running), as for
R69. The machine that drives the rig is the agent's client.

**How the preset is started.** [run/flash_ratio.sh](run/flash_ratio.sh) runs C854's launcher
([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh): R156's driver
[r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm `v4r`, then the report) with
`LLAMA_SERVER_TEXT_REUSE=1` added to the driver's `V4R_ENV`. That line is the v5 preset: the same 40 switches in the
same order as `settings.env`'s `PRESET_ENV`, and its `PRESET_ARGS` (R210's text-reuse arm, which became the preset;
[R208/R210](../2026-11-19_r208_r210_flash_switches/README.md)). The 0.75 arm replaces the driver's
`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0` with `=0.75`, so its started command differs from the control's in that
one value. [run/preset_check.sh](run/preset_check.sh) checks a started command against `settings.env` (the
environment byte for byte, each flag with its value), and [run/dry_line.sh](run/dry_line.sh) prints the command an
arm would start without loading anything.

**The reference run** is R209d's `f5oc1` ([R209](../2026-11-19_r209_oc_v5/README.md), 2026-11-19): the same rig, tree
and clock profile on another boot, with v4.0's preset (the v5 preset without text reuse). It is shown beside each run
for reference only; the verdict compares the two arms' runs of this day.

## Run 1, `c10a` (R220a): the control, 11:21–11:39

**The v5 preset ran 12 of 12 tasks in 13.7 minutes.** Its started command (1,715 bytes, md5
`5f0cbde4375a316aafaf49ba4660efcc`) is R210's text-reuse run's (`f5t2`) byte for byte and passes
[run/preset_check.sh](run/preset_check.sh) against `settings.env` (40 switches, `PRESET_ARGS` plus the harness's
`--metrics --no-webui` and the draft). The server started at 11:21:25 and was up after 227 s (R209d: 227 s); the tasks
ran 11:25:15–11:39:05 and the report ended at 11:39:24. Before and after, every card was at the profile's offsets
(75 W), the kit's watchdog was running, and the kernel log held no Xid line.

| | R209d `f5oc1` (2026-11-19, reference) | **`c10a` (R220a)** |
|---|---|---|
| preset | v4.0's (no text reuse) | **v5, ratio 1.0** |
| passed | 12 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | **2.28** |
| the same without JavaScript `complex-numbers` | 2.34 | **2.36** |
| generation tok/s | 30.69 | **30.28** |
| prompt tok/s, the whole session | 152.5 | **150.4** |
| prompt s a task (all prompt s) | 20.62 (247.4) | **20.57 (246.9)** |
| prompt tokens read a task | 3,145 | **3,093** |
| tokens generated a task | 1,855 | **1,367** |
| the 13 requests that read the same prompt on the same cached prefix, prompt s | 92.67 | **92.59** |
| of them the 12 first turns, prompt s | 84.83 | **84.76** |
| first token median s; p90 s | 2.66; 7.91 | **2.30; 7.90** |
| first token on a task's first turn, median s | 6.25 | **6.29** |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.53 | **27.21** |
| draft acceptance; tokens a decode step | 0.767; 3.30 | **0.755; 3.27** |
| ms a decode step, middle context under 4,000 / 4,000 to 8,000 | 104.15 / 108.02 | **105.59 / 108.62** |
| requests in those buckets | 17 / 39 | **17 / 47** |
| ms a decode step, all requests | 107.75 | **108.17** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 1; 941; 6.9 of 247.4 | **0; 0; 0.0 of 246.9** |
| requests on which text reuse kept the cached tokens | — | **7 (3 tasks)** |
| minutes, all 12 tasks; turns | 16.7; 61 | **13.7; 64** |
| tool calls / failed / malformed | 56 / 1 / 0 | **59 / 0 / 0** |
| graph reallocations | 0 in 61 requests | **0 in 64** |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **2113 2138 2151 2100 2100 2100 2151 2151 2126 (2,125.6)** |
| busy memory clock medians, MHz | 4657 4498 4498 4551 4551 4608 4455 4455 4657 | **the same on every card** |
| hottest card; highest power (card log) | 42 °C; 87.80 W | **44 °C (card 0); 83.35 W (card 3)** |
| lowest MemAvailable MiB; swap highest MiB | 384; 589 | **379; 999** |

Each value is one run ([agent.tsv](agent.tsv), [agent_nocx.tsv](agent_nocx.tsv), [work.md](work.md),
[prompt_tasks.tsv](prompt_tasks.tsv), [same_prompts.md](same_prompts.md), [first_token.tsv](first_token.tsv),
[steps.md](steps.md), [rereads_runs.tsv](rereads_runs.tsv), [text_reuse_points.tsv](text_reuse_points.tsv),
[clocks.tsv](clocks.tsv), [cards.md](cards.md), [host.tsv](host.tsv), [reallocs.tsv](reallocs.tsv); the agent check's
own report: [c10a_report_table.md](c10a_report_table.md), [c10a_compare.md](c10a_compare.md),
[c10a_gonogo.md](c10a_gonogo.md)). R209d's figures are this folder's scripts on R209's published tables and its server
log.

- **The prompt seconds of every task** ([prompt_tasks.tsv](prompt_tasks.tsv)), which R220 reads for the slower primes
  R69 saw at 7,168 and 8,192 tokens with the lower ratios: from 15.26 s (`list-ops`) to 40.40 s (`affine-cipher`,
  whose first turn reads 2,780 tokens with nothing cached). The tasks with more turns read more: Python `phone-number`
  27.05 s in 7 turns, `scale-generator` 23.52 s in 7.
- **Where the ratio will act** ([steps_ratio.md](steps_ratio.md)): 11 of the 64 requests had their middle between 6,153
  and 8,204 cells, where 0.75 moves a four-token verify pass to the sparse kernel; they generated 2,229 tokens (33 to
  757 a request) at 114.81 ms a decode step, and the 53 requests below 6,153 at 107.07 ms. R209d's same bucket, 9
  requests at 109.50 ms, is mostly one reply of 5,856 tokens. A step in such a short bucket depends on which replies
  fall in it, so the verdict compares this bucket between the four runs of this day.
- **Text reuse** ([text_reuse_points.tsv](text_reuse_points.tsv)): it kept the cached tokens on JavaScript `triangle`
  turn 5, Python `pig-latin` turns 3 to 5 and JavaScript `variable-length-quantity` turns 3 to 5, each time where the
  client sent back the model's reply, or its tool call's arguments re-serialized, in other tokens. The time from choosing
  the slot to processing each request, where its check runs, was 0.70 ms on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 59 calls malformed, none inside the reasoning, no
  NaN or inf, 11 checkpoints restored, 0 full re-processing. The report's gates are all GO except memory (NO-GO on the
  379 MB low and the swap), as in every Flash-Next run of this preset.
- **Memory:** swap grew from 50 MiB at the start to 999 MiB at 11:38:42, written out at about 1 to 2 MB/s through the
  tasks with almost nothing read back (the rig's vmstat), and was back at 51 MiB once the server stopped; R209d's
  highest was 589 MiB. The RAM watchdog stopped nothing.
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): the hottest card 44 °C, above
  80 W in 1 2 2 3 1 0 0 1 1 single samples on cards 0 to 8, highest 83.35 W on card 3. The kit's watchdog sampled
  1,661 times a card over the same window: 0 trips, never 80 W held for 5 s.

## Run 2, `r075a` (R220b): ratio 0.75, 11:49–12:10

**Ratio 0.75 ran 12 of 12 tasks in 16.1 minutes.** Its started command (1,716 bytes, md5
`f588628a98483a96e8ceb8f2969d27dc`) is its dry line byte for byte, differs from `c10a`'s only in
`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75`, and passes [run/preset_check.sh](run/preset_check.sh) with 0.75 (40
switches, the same 30 flags). The server started at 11:49:47 and was up after 227 s; the tasks ran 11:53:37–12:09:41 and
the report ended at 12:10:00, on the same boot as run 1. Before and after, every card was at the profile's offsets
(75 W), the kit's watchdog was running, and the kernel log held no Xid line.

| | R209d `f5oc1` (2026-11-19, reference) | `c10a` (R220a) | **`r075a` (R220b)** |
|---|---|---|---|
| preset | v4.0's (no text reuse) | v5, ratio 1.0 | **v5, ratio 0.75** |
| passed | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (R90 `v3`'s work) | 2.26 | 2.28 | **2.26** |
| the same without JavaScript `complex-numbers` | 2.34 | 2.36 | **2.34** |
| generation tok/s | 30.69 | 30.28 | **30.60** |
| prompt tok/s, the whole session | 152.5 | 150.4 | **154.6** |
| prompt s a task (all prompt s) | 20.62 (247.4) | 20.57 (246.9) | **21.50 (258.0)** |
| prompt tokens read a task | 3,145 | 3,093 | **3,324** |
| tokens generated a task | 1,855 | 1,367 | **1,699** |
| the 12 first turns, matched in all three runs, prompt s | 84.83 | 84.76 | **84.97** |
| the 14 requests matched in `c10a` and `r075a`, prompt s | — | 101.57 | **101.65** |
| first token median s; p90 s | 2.66; 7.91 | 2.30; 7.90 | **2.28; 7.91** |
| first token on a task's first turn, median s | 6.25 | 6.29 | **6.22** |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.53 | 27.21 | **27.48** |
| draft acceptance; tokens a decode step | 0.767; 3.30 | 0.755; 3.27 | **0.755; 3.27** |
| ms a decode step, middle context under 4,000 / 4,000 to 8,000 / 8,000 to 16,000 | 104.15 / 108.02 / 110.93 | 105.59 / 108.62 / — | **103.63 / 107.44 / 109.78** |
| requests in those buckets | 17 / 39 / 5 | 17 / 47 / 0 | **19 / 38 / 6** |
| ms a decode step, middle context under 6,153 cells (both arms on the same kernels) | 106.62 (47 requests) | 107.07 (53) | **106.06 (46)** |
| ms a decode step, middle 6,153 to 8,204 cells (where 0.75 acts); requests; tokens | 109.50; 9; 7,341 | 114.81; 11; 2,229 | **108.12; 11; 6,860** |
| ms a decode step, middle 8,204 cells and above; requests | 110.93; 5 | —; 0 | **109.78; 6** |
| ms a decode step, all requests | 107.75 | 108.17 | **106.97** |
| reply re-reads: turns; tokens; their prompt s of all prompt s | 1; 941; 6.9 of 247.4 | 0; 0; 0.0 of 246.9 | **0; 0; 0.0 of 258.0** |
| requests on which text reuse kept the cached tokens | — | 7 (3 tasks) | **8 (3 tasks)** |
| minutes, all 12 tasks; turns | 16.7; 61 | 13.7; 64 | **15.9; 63** |
| tool calls / failed / malformed | 56 / 1 / 0 | 59 / 0 / 0 | **58 / 0 / 0** |
| graph reallocations | 0 in 61 requests | 0 in 64 | **0 in 63** |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | 2113 2138 2151 2100 2100 2100 2151 2151 2126 (2,125.6) | **2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0)** |
| busy memory clock medians, MHz | 4657 4498 4498 4551 4551 4608 4455 4455 4657 | the same on every card | **the same on every card** |
| hottest card; highest power (card log) | 42 °C; 87.80 W | 44 °C (card 0); 83.35 W (card 3) | **44 °C (card 0); 84.72 W (card 1)** |
| lowest MemAvailable MiB; swap highest MiB | 384; 589 | 379; 999 | **386; 448** |

Each value is one run (the same tables as run 1; the agent check's own report:
[r075a_report_table.md](r075a_report_table.md), [r075a_compare.md](r075a_compare.md),
[r075a_gonogo.md](r075a_gonogo.md)). The 14 requests matched in the two arms are the 12 first turns plus turn 2 of
JavaScript `grade-school` and of Python `phone-number`, 14,833 tokens read, from C854's script with these two runs only:

```bash
D=docs/data/2026-11-20_r220_flash_ratio075
python3 -I docs/data/2026-11-18_c854b_p106_check/run/same_prompts.py c10a=$D/agent/c10a/turns.csv r075a=$D/agent/r075a/turns.csv
```

- **Where the ratio acts** ([steps_ratio.md](steps_ratio.md)): 11 requests had their middle between 6,153 and 8,204
  cells and ran at 108.12 ms a decode step, against 114.81 ms on the control's 11. The two buckets hold different
  replies: here 6,860 tokens, 56 to 3,666 a request, in six tasks; in the control 2,229 tokens, 33 to 757 a request, in
  four. Below 6,153 cells, where both arms run the same kernels, this run's step is 1.01 ms shorter too (106.06 against
  107.07 ms). This run also had 6 requests past 8,204 cells (109.78 ms), the control none. The verdict reads this bucket
  over all four runs.
- **The prompt seconds of every task** ([prompt_tasks.tsv](prompt_tasks.tsv)): 21.50 s a task against 20.57 s, for
  3,324 tokens read a task against 3,093. Most of the difference is JavaScript `complex-numbers`: 7 turns, 5,852 tokens
  and 36.36 s here, 6 turns, 2,494 tokens and 19.72 s in the control. The requests matched in both arms read at the
  same speed (101.65 s against 101.57 s); none of them lies at 7,168 tokens or deeper, where R69 saw slower primes.
- **Text reuse** ([text_reuse_points.tsv](text_reuse_points.tsv)): it kept the cached tokens on JavaScript
  `grade-school` turns 4 and 5, Python `pig-latin` turns 3 to 5 and JavaScript `variable-length-quantity` turns 3 to 5.
  The time from choosing the slot to processing each request was 0.68 ms on the median ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** every task passed with its hidden tests; 0 of 58 calls malformed, none inside the reasoning, no
  NaN or inf, 11 checkpoints restored, 0 full re-processing. The report's gates are all GO except memory (NO-GO on the
  387 MB low and the swap), as in every Flash-Next run of this preset.
- **Memory:** swap grew from 50 MiB to 448 MiB at 11:58:46 (run 1: 999 MiB) and was back at 50 MiB once the server
  stopped. The lowest MemAvailable was 386 MiB at 11:54:16. The RAM watchdog stopped nothing.
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): the hottest card 44 °C, above
  80 W in 2 3 0 1 4 1 3 1 3 samples on cards 0 to 8, at most two in a row (card 6), highest 84.72 W on card 1. The kit's
  watchdog sampled 1,929 times a card over the same window: 0 trips, above 80 W in 0 1 1 0 0 1 0 1 1 single samples,
  highest 83.50 W on card 7.

## Run 3, `c10b` (R220c): the control again, 12:25–12:59

**The v5 preset ran 11 of 12 tasks in 29.3 minutes.** Its started command (1,715 bytes, md5
`5f0cbde4375a316aafaf49ba4660efcc`) is `c10a`'s byte for byte and passes [run/preset_check.sh](run/preset_check.sh)
(40 switches, the same 30 flags); the server's `switches:` line is `c10a`'s too. The server started at 12:25:47 and was
up after 226 s; the tasks ran 12:29:37–12:59:06 and the report ended at 12:59:17, on the same boot as runs 1 and 2.
Before and after, every card was at the profile's offsets (75 W), the kit's watchdog was running, and the kernel log
held no Xid line.

| | `c10a` (R220a, control) | `r075a` (R220b, 0.75) | **`c10b` (R220c, control)** | **`c10b` − `c10a`** |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **11 / 12** | **JavaScript `complex-numbers` ran to its cap** |
| minutes a task at equal work (R90 `v3`'s work) | 2.28 | 2.26 | **2.32** | **+0.04** |
| the same without JavaScript `complex-numbers` | 2.36 | 2.34 | **2.34** | **−0.02** |
| generation tok/s | 30.28 | 30.60 | **29.46** | **−0.82** |
| the same without `complex-numbers` | 30.20 | 30.71 | **30.86** | **+0.66** |
| prompt tok/s, the whole session | 150.4 | 154.6 | **142.4** | |
| prompt s a task (all prompt s) | 20.57 (246.9) | 21.50 (258.0) | **32.80 (393.6)** | **+12.23** |
| the same without `complex-numbers` | 20.65 (227.1) | 20.15 (221.6) | **21.62 (237.8)** | **+0.97** |
| prompt tokens read a task; without `complex-numbers` | 3,093; 3,148 | 3,324; 3,094 | **4,670; 3,337** | |
| tokens generated a task | 1,367 | 1,699 | **3,129** | |
| the 12 requests matched in all three runs, prompt s | 86.94 | 87.33 | **90.04** | **+3.10** |
| the same without `complex-numbers` turn 1 (11 requests) | 80.60 | 80.97 | **81.36** | **+0.76** |
| first token median s; p90 s | 2.30; 7.90 | 2.28; 7.91 | **2.27; 8.94** | |
| first token on a task's first turn, median s | 6.29 | 6.22 | **6.49** | |
| the longest first-token wait s (`affine-cipher` turn 1, 2,780 tokens, nothing cached) | 27.21 | 27.48 | **27.42** | |
| draft acceptance; tokens a decode step | 0.755; 3.27 | 0.755; 3.27 | **0.741; 3.22** | |
| ms a decode step, middle under 6,153 cells (both arms on the same kernels) | 107.07 (53 requests) | 106.06 (46) | **106.31 (49)** | **−0.76** |
| ms a decode step, middle 6,153 to 8,204 cells (where 0.75 acts); requests; tokens | 114.81; 11; 2,229 | 108.12; 11; 6,860 | **113.48; 10; 3,951** | **−1.33** |
| ms a decode step, middle 8,204 to 16,000 cells; requests | —; 0 | 109.78; 6 | **109.34; 14** | |
| ms a decode step, middle 16,000 cells and above; requests | —; 0 | —; 0 | **112.82; 25** | |
| ms a decode step, all requests | 108.17 | 106.97 | **109.63** | **+1.46** |
| reply re-reads: turns; tokens | 0; 0 | 0; 0 | **0; 0** | |
| requests on which text reuse kept the cached tokens | 7 (3 tasks) | 8 (3 tasks) | **41 (3 tasks, 31 of them in `complex-numbers`)** | |
| minutes, all 12 tasks; turns | 13.7; 64 | 15.9; 63 | **29.3; 99** | |
| tool calls / failed / malformed | 59 / 0 / 0 | 58 / 0 / 0 | **94 / 4 / 0** | |
| graph reallocations | 0 in 64 requests | 0 in 63 | **0 in 98** | |
| busy SM clock medians, cards 0 to 8, MHz (mean) | 2113 2138 2151 2100 2100 2100 2151 2151 2126 (2,125.6) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0)** | |
| busy memory clock medians, MHz | 4657 4498 4498 4551 4551 4608 4455 4455 4657 | the same on every card | **the same on every card** | |
| hottest card; highest power (card log) | 44 °C (card 0); 83.35 W (card 3) | 44 °C (card 0); 84.72 W (card 1) | **44 °C (card 0); 83.81 W (card 7)** | |
| lowest MemAvailable MiB; swap highest MiB | 379; 999 | 386; 448 | **367; 1,055** | |

Each value is one run (the same tables as runs 1 and 2; R209d's reference row stays in them; the agent check's own
report: [c10b_report_table.md](c10b_report_table.md), [c10b_compare.md](c10b_compare.md),
[c10b_gonogo.md](c10b_gonogo.md)). The 12 requests matched in the three runs of this day are the 11 first turns
(`pig-latin`'s first turn drops out, below) and turn 2 of Python `phone-number`, 12,019 tokens read; `c10a` and `c10b`
alone match 13 requests (97.83 s against 94.78 s):

```bash
D=docs/data/2026-11-20_r220_flash_ratio075
python3 -I docs/data/2026-11-18_c854b_p106_check/run/same_prompts.py c10a=$D/agent/c10a/turns.csv r075a=$D/agent/r075a/turns.csv c10b=$D/agent/c10b/turns.csv
python3 -I docs/data/2026-11-18_c854b_p106_check/run/same_prompts.py c10a=$D/agent/c10a/turns.csv c10b=$D/agent/c10b/turns.csv
```

[same_prompts.md](same_prompts.md) now holds the requests matched in all four runs with R209d's: the 11 first turns,
78.69, 78.48, 78.84 and 81.54 s for `f5oc1`, `c10a`, `r075a` and `c10b`. The tables of runs 1 and 2 quote it as it was
before this run, with `pig-latin`'s first turn (12 requests).

- **JavaScript `complex-numbers`** ([prompt_tasks.tsv](prompt_tasks.tsv), `agent/c10b/attempts.csv`): 39 turns, the
  last cut by the 12-minute cap (720 s), 38 tool calls of which 3 failed, 38 of its 40 tests passing at the end; it read
  19,337 prompt tokens in 155.77 s and its context reached 33,530 tokens. Runs 1 and 2 passed it in 6 turns (45.5 s) and
  7 turns (91.8 s). R72b's run of this model on v4's candidate line took it to the cap in 35 turns
  ([R72](../2026-11-12_r72_agent_v4c2/README.md)), which is why every table here has a column without it. 34 of its
  requests had their middle past 8,204 cells, of this run's 39 there (run 2: 6, run 1: none), so the rows "all
  requests" and "8,204 cells and above" compare different work.
- **Where the ratio acts** ([steps_ratio.md](steps_ratio.md)): 10 requests had their middle between 6,153 and 8,204
  cells and ran at 113.48 ms a decode step, against 114.81 ms on run 1's 11; both controls run the dense kernel there.
  The replies differ again: here 3,951 tokens, 148 to 1,358 a request, in six tasks; run 1 2,229 tokens, 33 to 757, in
  four; run 2 (0.75) 6,860 tokens, 56 to 3,666, in six. Below 6,153 cells, where every run uses the same kernels, the
  two controls differ by 0.76 ms a step (106.31 against 107.07 ms); run 2's 106.06 ms lies 0.25 ms under this run and
  1.01 ms under run 1.
- **Two first turns read slower than in runs 1 and 2:** Python `pig-latin`'s first turn, right after `complex-numbers`,
  read all 2,324 tokens in 11.90 s instead of 775 in about 6.2 s, because the long task before it had pushed every
  checkpoint near the start out of the slot's four (the server log: "no checkpoint below 1553"; the report's one full
  re-processing). JavaScript `complex-numbers`' own first turn read its 896 tokens in 8.67 s against 6.34 and 6.36 s,
  after restoring the same 1,549-token checkpoint as runs 1 and 2; the task before it (`scale-generator`, 8 turns) had
  ended at 8,566 tokens. This is not explained yet; in run 2, `pig-latin`'s first turn after a task that ended at
  8,905 tokens read at the usual speed (6.13 s). Without that turn the requests matched in the three runs read within
  0.76 s of each other (80.60, 80.97 and 81.36 s).
- **Text reuse** ([text_reuse_points.tsv](text_reuse_points.tsv)): it kept the cached tokens on Python `affine-cipher`
  turns 2 to 5, Python `scale-generator` turns 3 to 8 and 31 turns of `complex-numbers`, 454,652 cached tokens kept in
  all. The time from choosing the slot to processing each request was 1.19 ms on the median
  ([slot_gap.tsv](slot_gap.tsv)).
- **Answers sound:** the 11 tasks passed with their hidden tests; 0 of 94 calls malformed, none inside the reasoning,
  no NaN or inf, 10 checkpoints restored, 1 full re-processing (`pig-latin`, above), 0 graph reallocations. The report's
  gates are GO except memory (NO-GO on the 368 MB low and the swap, as in every Flash-Next run of this preset) and the
  proxy rows (NO-GO: `complex-numbers` has 39 rows for 38 replies, its last request cut by the cap).
- **Memory:** swap grew from 50 MiB to 1,055 MiB at 12:43:26, during `complex-numbers` (run 1: 999 MiB, run 2: 448),
  and was back at 51 MiB once the server stopped. The lowest MemAvailable was 367 MiB at 12:31:35. The RAM watchdog
  stopped nothing.
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): the hottest card 44 °C, above
  80 W in 1 4 2 2 1 2 1 1 4 samples on cards 0 to 8, highest 83.81 W on card 7. The kit's watchdog sampled 3,538 times
  a card over the same window: 0 trips, above 80 W in 3 1 5 5 3 2 1 1 3 samples, at most two in a row (card 2), highest
  84.15 W on card 0, the offsets the same in every sample.

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv), [agent_nocx.tsv](agent_nocx.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work); the second without JavaScript `complex-numbers` |
| [work.md](work.md), [work.tsv](work.tsv) | per run: prompt tokens read and generated, reply re-reads, prompt seconds, the longest first-token wait |
| [prompt_tasks.tsv](prompt_tasks.tsv) | per task and run: prompt seconds, prompt tokens read, turns; the mean a task |
| [same_prompts.md](same_prompts.md) | the requests that read the same prompt on the same cached prefix in every run named, prompt ms each |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run, and the median and p90 of the turns after a task's first |
| [steps.md](steps.md), [steps_ratio.md](steps_ratio.md) | ms a decode step by the context at each request's middle: R203's buckets, and buckets cut at 6,153 and 8,204 cells, where the ratio moves a four-token verify pass |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | turns whose cached prefix stops inside the previous reply, per run and per turn (C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py); the second is empty while no run re-read a reply) |
| [text_reuse_points.tsv](text_reuse_points.tsv) | the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request (text reuse's check runs there) |
| [clocks.tsv](clocks.tsv) | busy SM clock medians per card over the tasks |
| [cards.md](cards.md) | per card over the tasks: highest temperature and power, samples above 80 W, busy memory clock (R203's [flash_cards.py](../2026-11-18_r203_ornith_offsets/run/flash_cards.py)) |
| [host.tsv](host.tsv) | lowest MemAvailable, swap, card memory peaks |
| [reallocs.tsv](reallocs.tsv) | graph reallocations by prompt size |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report of each run |
| [run/](run/) | the scripts below |

## Commands

From the repository root on the machine that drives the rig and is the agent's client, one run a session, each with
its own `OUT`, rig folder, client folder and container prefix; only `ARM` and those names change between the four runs:

```bash
export ARM=c10a PREFIX=acrel-r220a RIG=<rig alias> CLIENT=local \
  MODEL=<models>/flash_next/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 REF=<v4.0's build folder, unused by this arm> \
  RIGDIR=<a folder on the rig>/r220a OUT=<run folder>/out RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client>/bench CLIENT_OUT=<client>/r220a
unset GAINS_ENV V4R_ENV
bash docs/data/2026-11-20_r220_flash_ratio075/run/flash_ratio.sh
```

The runs' `ARM`, `PREFIX` and folder names: R220a `c10a` `r220a`, R220b `r075a` `r220b`, R220c `c10b` `r220c`, R220d
`r075b` `r220d`. Before a run, the command it will start, and its check against the preset:

```bash
ARM=r075a WORK=<an empty scratch folder> NEW=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 \
  DRAFT=<models>/flash_next/mtp-Qwen3.8-Flash-Next-draft.gguf \
  bash docs/data/2026-11-20_r220_flash_ratio075/run/dry_line.sh > dry_line_r075a.txt
bash docs/data/2026-11-20_r220_flash_ratio075/run/preset_check.sh dry_line_r075a.txt 0.75   # the control: no ratio
```

The tables, with Python 3's standard library only, from the runs' collected logs (the inputs are listed at the top of
[run/tables.sh](run/tables.sh)); the client's `res/` files go into `agent/<run>/` first:

```bash
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <run folder>/out/agent/res \
  docs/data/2026-11-20_r220_flash_ratio075/agent/c10a v4r c10a
bash docs/data/2026-11-20_r220_flash_ratio075/run/tables.sh <tables folder> c10a [r075a c10b r075b]
```

| Script | What it does |
|---|---|
| [run/flash_ratio.sh](run/flash_ratio.sh) | runs one arm: the v5 preset, or the same with the ratio at 0.75 |
| [run/dry_line.sh](run/dry_line.sh) | prints the command an arm would start, without loading anything |
| [run/preset_check.sh](run/preset_check.sh) | checks a started command against `release/flash-next-v5/settings.env` |
| [run/tables.sh](run/tables.sh) | builds this folder's tables from the runs done so far |
| [run/steps_ratio.py](run/steps_ratio.py) | ms a decode step in the buckets where the ratio acts (R203's `flash_steps.py` with other buckets) |

## Provenance and licence

Measured on the project's own rig on 2026-11-20 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<run>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
