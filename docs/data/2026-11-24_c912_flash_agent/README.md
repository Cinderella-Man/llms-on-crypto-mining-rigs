# C912: Flash-Next's coding-agent check on upstream b12674 merged into v5 (2026-11-24)

Registry row [Q1649](../../../quants/experiments.md), the merge of upstream llama.cpp's tag b12674 into the v5 engine
([2026-11-24_c912_upstream_b12674](../2026-11-24_c912_upstream_b12674/README.md)). The merge (the candidate branch
`pascal-p106-v5-cand-b12674`, `eee615cdb`) passed its checks on the build machine and its P106 check
([2026-11-24_c912_p106_check](../2026-11-24_c912_p106_check/README.md)), and Ornith's coding-agent run on it
([2026-11-24_c912_ornith_agent](../2026-11-24_c912_ornith_agent/README.md)). Upstream changed the server and the
scheduler, so before `pascal-p106-v5` moves to the candidate each model's coding-agent check runs on it. This folder is
Flash-Next's: one run on the candidate, then one control run of the same launcher on `pascal-p106-v5` `af00bb13b` on the
same rig and boot, the engine the only difference.

**Run 1 of 2 done: on the candidate, Flash-Next's v5 preset passed 12 of 12 tasks, with no server error.** A task took
2.34 minutes at equal work (R90 `v3`'s work), 62 turns in all; the model generated at 28.59 tok/s over the session, a
decode step took 117.02 ms over all requests, and the prompts read at 144.7 tok/s (20.27 s a task, 243.20 s in all for
35,195 tokens). The first token came after 2.51 s on the median, 8.25 s at the 90th percentile and 5.11 s on the median
of a task's first turn; the longest wait was 28.58 s, Python `affine-cipher`'s first turn (2,780 tokens, nothing cached).
No reply was read again (0 tokens), text reuse kept the cached tokens on 9 turns, the server restored 11 checkpoints,
re-processed nothing in full and reallocated no graph. No task ran to its 12-minute cap. **No class yet:** the control
run on `af00bb13b`, on the same rig and boot with the same launcher, gives the comparison.

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
| 2 | `f5ref` | `pascal-p106-v5` `af00bb13b` (the control) | `f5cca34ea6bd4f3712cd6193bf0605be` | next |

**How the preset is started.** R227's [run/flash_taper.sh](../2026-11-20_r227_flash_taper1400_agent/run/flash_taper.sh)
with `ARM=t1400a`: C854's launcher ([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh), R156's driver
[r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh), the harness's arm `v4r`, then the report) with
`--ubatch-taper short=1400:64`, the line the v5 preset took from R227's verdict. Its started command (1,717 bytes, md5
`e88afbaa6d856ae489378b407852abc3`) is R227's `t1400a` command byte for byte with the engine's path replaced, it was
printed before the run by R227's [run/dry_line.sh](../2026-11-20_r227_flash_taper1400_agent/run/dry_line.sh), and
[run/preset_check.sh](run/preset_check.sh) finds it equal to `settings.env` (the 40 switches byte for byte, each flag
with its value). The server read `short=1400:64` at its load and logged the same 40 switches as R227's runs. The
control's dry line differs in the engine's path only.

## Run 1, `f5cand`: the candidate, 20:15–20:45

| | **`f5cand`** (rig 1, stock, `eee615cdb`) | context: R227 `t1400a` | context: R227 `t1400b` |
|---|---|---|---|
| passed | **12/12** (no task at its cap) | 12/12 | 12/12 |
| minutes a task at equal work (R90 `v3`'s work) | **2.34** | 2.25 | 2.24 |
| minutes, all tasks; turns | **23.2; 62** | 21.5; 66 | 14.6; 67 |
| generation tok/s over the session | **28.59** | 30.71 | 30.87 |
| ms a decode step, all requests; at 4,000 to 8,000 tokens of context | **117.02; 116.88** | 107.18; 106.71 | 105.95; 106.70 |
| the draft's acceptance; tokens a step | **0.781; 3.34** | 0.762; 3.29 | 0.753; 3.26 |
| prompt tok/s; prompt s a task (all) | **144.7; 20.27 (243.20)** | 157.9; 20.37 (244.47) | 158.8; 20.80 (249.63) |
| the 13 requests read on the same prompt and cached prefix in all three, prompt s | **90.84** | 83.79 | 82.47 |
| of them the 12 first turns, prompt s | **82.52** | 76.01 | 74.66 |
| first token median; p90; a task's first turn median, s | **2.51; 8.25; 5.11** | 2.32; 7.88; 4.67 | 2.83; 7.90; 4.62 |
| longest first-token wait s (task, turn) | **28.58** (`affine-cipher`, 1) | 27.37 (the same) | 27.53 (the same) |
| prompt tokens read a task; tokens generated a task | **2,933; 2,673** | 3,216; 2,613 | 3,304; 1,527 |
| reply tokens re-read (turns) | **0 (0)** | 0 (0) | 215 (1) |
| checkpoints restored / created / erased; full re-processing; graph reallocations | **11 / 0 / 0; 0; 0** | | |
| tool calls / malformed | **58 / 0** | | |
| card peak MiB, cards 0–8 | **5,192 5,236 5,094 5,250 4,350 5,108 4,504 5,120 5,698** | | |
| lowest MemAvailable MiB; swap highest MiB | **322; 623** | | |
| busy SM clock, mean of the cards' medians, MHz | **1,879.8** | | |
| server load s | **312** | 226 | 227 |

Each value is one run. **The context columns are not a comparison:** R227's two runs of the same started command ran on
the second rig with its soaked clock profile (R219's memory and core offsets on every card) on `pascal-p106-v5`
`602691153`, so their faster decode step (about 10 ms less) and prompts are the clocks and the rig as much as the
engine. They show only that the run did the same kind of work: the same 12 tasks passed, the same longest wait, a
similar number of turns. The class comes from the control run on this rig and boot.

- **Answers sound:** 0 of 58 tool calls malformed, no NaN or inf, no full re-processing, no graph reallocation; the
  server log holds the same seven warning lines as R227's runs and no error. The harness's report gives GO on every gate
  but memory (lowest MemAvailable 322 MB, swap 0 to 624 MB), which every Flash-Next run on a 4 GB host fails (R227's
  runs 311 and 309 MB; C854's on this rig 387 MB with swap 31 to 464 MB).
- **Text reuse** kept the cached tokens on 9 turns (`affine-cipher`, `variable-length-quantity` in Python and
  JavaScript `complex-numbers`, turns 3 to 5 or 5 to 7 of each; [text_reuse_points.tsv](text_reuse_points.tsv)), so no
  turn read one of the model's replies again.
- **The rig:** every card at 75 W and 1911/4004 MHz maximum before and after, 0 MiB and no llama process at the end, no
  GPU Xid in the boot's kernel log. The busy SM clock's median was 1,847 1,847 1,898 1,885 1,873 1,911 1,873 1,911 1,873 MHz on cards 0 to 8 (the share of busy samples below 1,800 MHz by card in
  [clocks.tsv](clocks.tsv)).

## Files

| File | What it holds |
|---|---|
| [agent.tsv](agent.tsv) | per run: tasks passed, turns, prompt and generation tok/s, first token median and p90, the draft's acceptance, minutes, a task at equal work (R90 `v3`'s work); R227's two runs as context |
| [work.md](work.md), [work.tsv](work.tsv) | per run: prompt tokens read and generated, reply re-reads, prompt seconds, the longest first-token wait |
| [prompt_tasks.tsv](prompt_tasks.tsv) | per task and run: prompt seconds, prompt tokens read, turns; the mean a task |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per run: turns, new tokens and prompt seconds by the turn's new tokens (under 600, 600 to 1,023, 1,024 to 1,999, 2,000 and more) |
| [taper_lines.tsv](taper_lines.tsv) | the micro-batch taper each server read at its load |
| [same_prompts.md](same_prompts.md) | the requests that read the same prompt on the same cached prefix in every run named, prompt ms each |
| [first_token.tsv](first_token.tsv) | the longest first-token wait of each run, and the median and p90 of the turns after a task's first |
| [steps.md](steps.md) | ms a decode step by the context at each request's middle (R203's buckets) |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv) | turns whose cached prefix stops inside the previous reply, per run and per turn (C854's [reply_rereads.py](../2026-11-18_c854b_p106_check/run/reply_rereads.py)) |
| [text_reuse_points.tsv](text_reuse_points.tsv) | the turns on which text reuse kept the cached tokens |
| [slot_gap.tsv](slot_gap.tsv) | the time from choosing the slot to processing each request |
| [clocks.tsv](clocks.tsv) | busy SM clock medians per card over the tasks |
| [host.tsv](host.tsv) | lowest MemAvailable, swap, card memory peaks |
| [reallocs.tsv](reallocs.tsv) | graph reallocations by prompt size |
| [capped.tsv](capped.tsv) | the tasks that ran to their cap (none) |
| [agent/](agent/) | each run's per-turn and per-attempt tables (`turns.csv`, `attempts.csv`), the harness's arm `v4r` renamed to the run |
| `f5cand_report_table.md`, `f5cand_compare.md`, `f5cand_gonogo.md` | the agent check's own report of the run |
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

The tables, with Python 3's standard library only, from the runs' collected logs (the inputs are listed at the top of
[run/tables.sh](run/tables.sh)); the client's `res/` files go into `agent/<run>/` first:

```bash
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <client>/c912f-f5cand/res \
  docs/data/2026-11-24_c912_flash_agent/agent/f5cand v4r f5cand
bash docs/data/2026-11-24_c912_flash_agent/run/tables.sh <tables folder> f5cand [f5ref]
```

| Script | What it does |
|---|---|
| [run/preset_check.sh](run/preset_check.sh) | checks a started command against `release/flash-next-v5/settings.env` as it is |
| [run/tables.sh](run/tables.sh) | builds this folder's tables from the runs done so far (R227's `run/tables.sh` with R227's two runs as context rows) |

## Provenance and licence

Measured on the project's own rig on 2026-11-24 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<run>/attempts.csv`). No
model file or task text is in this folder. The tables and scripts here are under the repository's licence.
