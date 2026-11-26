# Flash-Next's agent check on the 131,072-token line against the v5 preset (R243, Q1921, 2026-11-26)

What this measured: Qwen3.8-Flash-Next on the coding-agent check with its v5 preset as it stands (65,536 tokens of
context, prompt micro-batches of 512) against the same preset with `-c 131072 -ub 256`, the line R124 found to fit
nine P106-090 cards ([R124](../2026-11-26_r124_flash_131k_rig2/README.md)), nothing else changed. Two runs each,
alternated, on one rig, engine and boot. Registry row [Q1921](../../../quants/experiments.md), from Q1403 (R124).

Why: 11 of the sister project's 14 passed real coding cases need up to 131,072 tokens and only 3 fit 65,536
([task-level review](../../research/notes/v5_review_task_level_2026-11.md)). The agent check's tasks fit 65,536 tokens,
so this prices what the larger line costs where most turns are: prompt calls of more than 1,400 tokens read in
micro-batches of 256 instead of 512 (up to 1,400 the preset's taper reads micro-batches of 64 in both lines), and the
bigger caches.

**Result: the 131,072-token line is not measurably slower per task.**
- **The task:** 2.31 and 2.38 minutes at equal work against the preset's 2.31 and 2.32 (+1.3 %, negligible); the
  line's second run had JavaScript `complex-numbers` at its 12-minute cap (30 turns; the other three runs passed it in
  5 turns), the task that capped in C1175's and R220's control runs too. Without that task 2.40 and 2.39 against 2.41
  and 2.41 (−0.6 %, 0.9 s a task, a gain by the class rule because the preset's two runs are equal to the hundredth).
  12/12 and 11/12 tasks against 12/12 and 12/12, 0 malformed tool calls in all four.
- **Generation and the decode step are the same:** 113.30 and 115.04 ms a step against 112.91 and 113.44 (negligible),
  and the same in every bucket of context (below); generation 29.05 and 27.02 tok/s against 29.07 and 28.81
  (negligible; 29.12 and 28.80 against 29.03 and 28.77 without the capped task, whose 12 requests above 16,000 cells
  only that run has).
- **Prompt reading: one gain, one loss, the rest the same.**
  - The first request after loading (2,780 tokens on an empty cache, the same request in every run) took 19,480.4
    and 19,460.9 ms against 27,406.9 and 28,128.4 (−29.9 %, a gain of 8.3 s once per load). Every later call of more
    than 1,400 tokens read on a cached prefix is about as fast: 4.502 and 4.621 ms a token against 4.467 and 4.455
    (+2.3 %, negligible; +1.8 %, a loss, without the capped task), and the one such call that read the same prompt in
    all four runs took 8,255.0 and 8,296.2 ms against 8,324.1 and 8,262.7.
  - Calls of fewer than 600 tokens are slower: the four tasks' first turns that read 403 to 577 new tokens on the same
    cached prefix took 14,002.5 and 14,019.3 ms together against 13,032.9 and 12,939.2 (+7.9 %, about 0.26 s a call, a
    loss), so the median first-token wait of the tasks' first turns is 5.25 and 5.31 s against 5.07 and 5.08 (+4.0 %,
    a loss). The matched turns of 600 to 1,400 tokens are the same (−1.4 %, negligible). Both sizes are read in
    micro-batches of 64 in either line, so this is not the micro-batch of 256; what part of it is the context size is
    rig row R244's question (Q1924).
- **Card 8, the output head and the draft, is the margin:** 5,931 MiB in use after loading and 5,997 and 5,999 at the
  highest (71 and 69 MiB left of the 6,068 a card offers), against 5,545 and 5,623 (445 left) on the preset. R124
  measured 5,931 and 5,999 on the other rig with the preset's previous sparse-attention values. The host is the same as
  the preset's: lowest MemAvailable 267 and 232 MiB against 235 and 245, and no run wrote to swap (22 MiB from start to
  end); the RAM watchdog never acted.

By these figures the 131,072-token line can be Flash-Next's preset: per task it costs nothing measurable and it holds
the long real tasks the 65,536-token line cannot. The runs used the preset as it stood at 15:59 (`d296f5ce`); since
16:08 its taper also has `cuts=1`, not measured with the line. The preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env))
is unchanged until the owner's word; the two lines it would change are both `[order]` (micro-batch size sets the
arithmetic's order; [docs/reply_effects.md](../../reply_effects.md)):

```bash
PRESET_ARGS="$PRESET_ARGS -ngl 99 -fa on -c 131072 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek"  # [order] the run line ...
PRESET_ARGS="$PRESET_ARGS -b 4096 -ub 256"  # [order] the micro-batch size sets the arithmetic's order; prompt calls of 4,096 tokens in micro-batches of 256, the most card 8 holds at 131,072 tokens (R124, R243)
```

## The figures

Each value is one run; the class is the release check's rule (a gain or a loss only beyond the larger of the two
lines' spreads, a spread being the difference between a line's two runs; [class.tsv](class.tsv),
[class_nocap.tsv](class_nocap.tsv), [class_cells.tsv](class_cells.tsv), [class_prompt.tsv](class_prompt.tsv),
[same_sizes.tsv](same_sizes.tsv)).

| | `ca` (preset) | `l131a` | `cb` (preset) | `l131b` | the line against the preset |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | 11 / 12 | |
| minutes a task at equal work (R90 `v3`'s work) | 2.31 | 2.31 | 2.32 | 2.38 | +1.3 %, negligible |
| the same without JavaScript `complex-numbers` | 2.41 | 2.40 | 2.41 | 2.39 | −0.6 %, **gain** |
| generation tok/s | 29.07 | 29.05 | 28.81 | 27.02 | −3.1 %, negligible |
| the same without `complex-numbers` | 29.03 | 29.12 | 28.77 | 28.80 | +0.2 %, negligible |
| prompt tok/s, the whole session | 148.9 | 153.5 | 150.9 | 148.6 | +0.8 %, negligible |
| the same without `complex-numbers` | 145.1 | 149.3 | 146.3 | 154.9 | +4.4 %, **gain** (the first request) |
| ms a decode step, all requests | 112.91 | 113.30 | 113.44 | 115.04 | +0.9 %, negligible |
| the same, middle context under 4,102 cells (dense attention kernels) | 114.36 | 113.47 | 113.98 | 113.95 | −0.4 %, negligible |
| the same, 4,102 to 6,153 cells (the per-token sparse kernel) | 112.95 | 112.80 | 112.93 | 113.40 | +0.1 %, negligible |
| the same, 6,153 to 8,204 cells | 112.14 | 113.95 | 112.79 | 112.67 | +0.8 %, negligible |
| the same, 8,204 to 16,000 cells | 114.95 | 113.66 | 113.98 | 115.27 | 0.0 %, negligible |
| requests in those four buckets | 18 / 33 / 13 / 9 | 18 / 30 / 12 / 4 | 21 / 26 / 10 / 7 | 19 / 26 / 10 / 18 | |
| first token median s, all turns | 2.43 | 2.56 | 2.37 | 3.08 | +17.5 %, negligible |
| first token median s, the tasks' first turns | 5.07 | 5.25 | 5.08 | 5.31 | +4.0 %, loss |
| first token median s; p90 s, later turns | 2.20; 8.41 | 2.30; 8.29 | 2.20; 9.06 | 2.18; 8.29 | +1.8 %; −5.1 %, negligible |
| the first request after loading (2,780 tokens, empty cache), prompt ms | 27,406.9 | 19,480.4 | 28,128.4 | 19,460.9 | −29.9 %, **gain** |
| matched first turns under 600 new tokens (4), prompt ms together | 13,032.9 | 14,002.5 | 12,939.2 | 14,019.3 | +7.9 %, loss |
| matched turns of 600 to 1,400 new tokens (6), prompt ms together | 35,541.7 | 34,814.0 | 33,935.8 | 33,670.9 | −1.4 %, negligible |
| ms a prompt token, calls over 1,400 tokens on a cached prefix | 4.467 | 4.502 | 4.455 | 4.621 | +2.3 %, negligible |
| the same without `complex-numbers` | 4.563 | 4.633 | 4.559 | 4.655 | +1.8 %, loss |
| draft acceptance; tokens a decode step | 0.758; 3.27 | 0.761; 3.28 | 0.754; 3.26 | 0.701; 3.10 | |
| turns; tool calls malformed | 73; 0 | 64; 0 | 64; 0 | 86; 0 | |
| minutes, all 12 tasks | 19.2 | 18.6 | 19.3 | 28.2 | |
| graph reallocations | 0 in 73 requests | 0 in 64 | 0 in 64 | 0 in 85 | |
| card 8 MiB after loading; at the highest; left | 5,545; 5,623; 445 | 5,931; 5,997; 71 | 5,545; 5,623; 445 | 5,931; 5,999; 69 | |
| lowest host MemAvailable MiB; most swap in use MiB | 235; 22 | 267; 22 | 245; 22 | 232; 22 | |

- **Why the task time does not move:** the line differs from the preset in prompt calls only, and there in three
  directions that nearly cancel inside a task: 8.3 s saved once per load on the first request, 1,024.9 ms lost over
  the four matched calls under 600 tokens, and the larger calls the same within 2.3 %. Prompt reading is 279.2, 248.1, 255.4 and
  335.3 s of each run's 19.2, 18.6, 19.3 and 28.2 minutes ([prompt_tasks.tsv](prompt_tasks.tsv),
  [reallocs.tsv](reallocs.tsv)); decoding runs the same kernels at the same speed in both lines.
- **The first request.** Each run's first request reads the system prompt and the first task, 2,780 tokens, on an
  empty cache, in one call; the preset reads it in 6 micro-batches of up to 512, the line in 11 of up to 256. The
  preset's two runs took 27,406.9 and 28,128.4 ms (9.86 and 10.12 ms a token), as every preset run on the agent check
  did (C1175's four runs 26,709.0, 27,005.7, 27,192.2 and 27,267.7 ms on rig 2; C1173's two runs of another file,
  `pmix-q2-fast`, 27,092.0 and 27,171.9 on rig 1), while later calls of 1,401 to 3,368 tokens on a cached prefix take
  4.467 and 4.455 ms a token on the preset and 4.502 and 4.621 on the line. So most of the 27 s is a
  cost the first large micro-batches pay once per load, and the line pays less of it (7.01 and 7.00 ms a token).
  Whether a later request on an empty or unmatched cache pays it again, and what in the engine it is, are R244 and
  code row C1187 (Q1923).
- **The small calls.** Under 600 new tokens each of the four matched first turns is slower on the line in both runs
  (403 tokens: 3,292.6 and 3,306.3 ms against 3,044.0 and 2,990.9; 430: 3,539.1 and 3,431.2 against 3,125.8 and
  3,131.5; 441: 3,352.4 and 3,436.4 against 3,141.6 and 3,117.0; 577: 3,818.4 and 3,845.4 against 3,721.5 and
  3,699.8; [same_prompts.md](same_prompts.md)), while the six of 600 to 1,400 tokens are not. Both are read in
  micro-batches of 64 in either line, so it is a cost of the line that the micro-batch size does not set; the
  bucket of all calls under 600 tokens is +4.8 % (negligible against the line's spread of 0.706 ms a token,
  [class_prompt.tsv](class_prompt.tsv)).
- **Decoding:** the step is the same below 4,102 cells (dense kernels in both lines) and above (the per-token sparse
  kernel in both, the preset's `_NSM=10 _MIN_RATIO=0.5`). `l131b`'s lower generation is its capped task: 12 requests
  above 16,000 cells at 118.75 ms a step that no other run has ([steps_cells.md](steps_cells.md); R203's buckets in
  [steps.md](steps.md)).
- **Answers:** every run's tool calls well formed, 0 graph reallocations, no reply read again on the line (one turn of
  63 tokens in `ca`; [rereads_runs.tsv](rereads_runs.tsv)), every token sampled under the tool-call grammar
  ([grammar.tsv](grammar.tsv)). The report's gates are all GO but memory (NO-GO on the low MemAvailable, as in every
  Flash-Next run of this preset) and, in `l131b`, the proxy rows of the capped task (30 rows for 29 replies: the cap
  cut a reply); the RAM watchdog stopped nothing.
- **Clocks** ([clocks.tsv](clocks.tsv)): rig 1 at stock clocks, busy SM medians 1,847, 1,835, 1,885, 1,885, 1,860,
  1,911, 1,873, 1,911 and 1,873 MHz on cards 0 to 8 in all four runs (card 8 1,860 in `cb` and `l131b`); the boot
  logged no GPU Xid.

## The line

The v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it stood when the runs
started at 15:59, at `d296f5ce`, md5 `3616639032df7218d1a4b744abe5b60e`: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its n-gram table cut into a third shard, the MTP
draft three tokens a step, 40 switches, among them `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10` and `_MIN_RATIO=0.5` since
`d296f5ce`) against the same with `-c 131072` in place of `-c 65536` and `-ub 256` in place of `-ub 512`
(`--spec-draft-n-ubatch 512` kept), R124's line. At 16:08 the preset's taper became `short=1400:64,cuts=1` (`602ba0cf`,
C905, C930: each stretch of a call between its in-call checkpoint cuts planned on its own, the recorded agent prompts
2.66 s a pass faster, negligible); these runs read prompts with `short=1400:64`, so the line with `cuts=1` is not
measured here. The two lines the line changes do not touch the taper's. pi, the agent, gets a context window equal to the server's `-c` in
each line (131,072 on the line); no request came near 65,536 tokens (the longest prompts per run 10,807, 13,358,
13,192 and 28,317 tokens, the last in `l131b`'s capped task).

**How it is started.** [run/flash_131k.sh](run/flash_131k.sh) starts the preset as C1175's arm `r05` did
([flash_nsm.sh](../2026-11-26_c1175_nsm10_rig2/run/flash_nsm.sh); that arm is the preset since `d296f5ce`): C854's
launcher ([flash_agent.sh](../2026-11-18_c854b_p106_check/run/flash_agent.sh)), R156's driver
([r156.sh](../2026-11-13_r156_agent_v4r/run/r156.sh)), the harness's arm `v4r`, then the report. R156's driver fixes
the context (`CTX=65536`, the server's `-c` and pi's window) and the micro-batch (`-ub 512` in its `MT` line), so the
line's runs start a copy of the driver with those two values replaced, each found exactly once, and a copy of the
launcher that runs it; the script checks that the copies differ from the originals in those lines only and prints the
driver's diff.
- The preset's two runs started a 1,717-byte command, md5 `b65c090f6b98f8446cf19f2ece272fba`: C1175's `r05` runs byte
  for byte.
- The line's two runs started a 1,718-byte command, md5 `58fe73f18c315a9c959a7c24eb33c569`, which differs from the
  preset's in `-c 131072` and `-ub 256` only.
- Each started command is its dry line ([run/dry_line.sh](run/dry_line.sh)) byte for byte and passes
  [run/preset_check.sh](run/preset_check.sh) against `settings.env` at `d296f5ce` (`PRESET=` that file; the environment byte for byte, 40 switches; the
  flags the preset's, with the line's two values for `l131`, plus the harness's `--metrics --no-webui` and `-md`, 30
  flags). Each server printed its context (`n_ctx_slot = 65536` or `131072`) and its taper
  (`short=1400:64 (n_ubatch 512)` or `(n_ubatch 256)`; [taper_lines.tsv](taper_lines.tsv)).

## What ran

| # | Run | Line | Launched | Server up after | Tasks | Driver |
|---|---|---|---|---|---|---|
| 1 | `ca` | the v5 preset (`-c 65536 -ub 512`) | 15:59:16 | 308 s | 16:04:34–16:23:58 | exit 0 at 16:24:10 |
| 2 | `l131a` | `-c 131072 -ub 256` | 16:24:11 | 306 s | 16:29:28–16:48:12 | exit 0 at 16:48:19 |
| 3 | `cb` | the v5 preset | 16:48:19 | 306 s | 16:53:36–17:13:06 | exit 0 at 17:13:27 |
| 4 | `l131b` | `-c 131072 -ub 256` | 17:13:27 | 307 s | 17:18:44–17:47:05 | exit 0 at 17:47:21 |

- **Rig:** rig 1, nine P106-090 cards (6,144 MiB each, 76 reserved by the driver) on PCIe 1.0 x1, a 2-core Celeron
  with 4 GB of RAM, driver 580.178.04, at stock clocks (no offsets, the 75 W default power limit), all four runs on
  the boot of 09:21. Before each run no llama process ran and every card held 0 MiB; after each, the same, and the
  run's folder on the rig was removed.
- **Engine:** the `pascal-p106-v5` candidate tree `eee615cdb` (upstream b12674 merged), llama-server md5
  `2727faf28b5b34dfe42399a9d431da67`, checked before each run; the build C1175 and R124 ran the same day.
- **The agent check** ([bench/agent_check](../../../bench/agent_check/README.md)): 12 bug-fix tasks, Python and
  JavaScript alternating, a 12-minute cap a task, a 40-minute budget, the machine that drives the rig as the agent's
  client (its own container prefix, client folder and relay ports, so that another check on the same client could not
  touch this one's containers).

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv), [agent_nocap.tsv](agent_nocap.tsv), [capped.tsv](capped.tsv) | the agent check per run, with and without the task that ran to its cap in a run |
| [class.tsv](class.tsv), [class_nocap.tsv](class_nocap.tsv) | the line's class against the preset on the session's measures, with and without the capped task |
| [class_cells.tsv](class_cells.tsv), [steps_cells.md](steps_cells.md), [steps.md](steps.md) | ms a decode step per bucket of context (C1175's cuts, and R203's buckets) and the first-token wait of first and later turns, classed |
| [class_prompt.tsv](class_prompt.tsv), [prompt_buckets.tsv](prompt_buckets.tsv) | ms a prompt token per size of the new tokens (under 600, 600 to 1,400, 1,401 to 1,999, 2,000 and up, over 1,400, over 1,400 on a cached prefix), classed |
| [same_prompts.md](same_prompts.md), [same_sizes.tsv](same_sizes.tsv) | the requests that read the same prompt on the same cache in all four runs, and those summed by size, classed |
| [prompt_tasks.tsv](prompt_tasks.tsv), [first_token.tsv](first_token.tsv) | prompt seconds per task, and the longest and the later turns' first-token waits |
| [card_memory.tsv](card_memory.tsv) | each card's memory after loading and at its highest per run, on nvidia-smi's scale, and what it leaves of 6,068 MiB |
| [host.tsv](host.tsv) | host MemAvailable and swap per run, and each card's highest reading as NVML counts it (with the driver's 75 MiB) |
| [work.md](work.md), [work.tsv](work.tsv) | the work each run read and generated, priced against R90 `v3`'s |
| [clocks.tsv](clocks.tsv), [reallocs.tsv](reallocs.tsv), [slot_gap.tsv](slot_gap.tsv), [grammar.tsv](grammar.tsv), [taper_lines.tsv](taper_lines.tsv) | busy clocks, graph reallocations, the slot-to-processing gap, the tool-call grammar's samples and the taper each server read |
| [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv), [text_reuse_points.tsv](text_reuse_points.tsv) | reply re-reads and where text reuse kept the cache |
| `<run>_report_table.md`, `<run>_compare.md`, `<run>_gonogo.md` | the agent check's own report for each run |
| [agent/](agent/) | each run's `turns.csv` and `attempts.csv` (the harness's arm renamed to the run) |
| [run/flash_131k.sh](run/flash_131k.sh), [run/dry_line.sh](run/dry_line.sh), [run/preset_check.sh](run/preset_check.sh) | the runs' script, the server line it would start, and its check against `settings.env` |
| [run/tables_agent.sh](run/tables_agent.sh), [run/class_prompt.py](run/class_prompt.py), [run/same_sizes.py](run/same_sizes.py), [run/card_memory.py](run/card_memory.py) | the tables (C1175's `run/tables_agent.sh` with the prompt reading cut at 1,400 tokens, the matched requests by size and the card memory) |

## Commands

```bash
# the agent check, one run at a time, in the order ca, l131a, cb, l131b, each with its own OUT, CLIENT_OUT and WORK;
# from the repository root on the machine that drives the rig; RIG, the paths, PREFIX and the ports are yours
ARM=ca RIG=rig CLIENT=local MODEL=<first shard on the rig> DRAFT=<MTP draft on the rig> NEW=<build folder on the rig> \
  REF=<v4.0 build folder on the rig> RIGDIR=<folder on the rig> OUT=<folder> WORK=<folder> CLIENT_DIR=<bench on the client> \
  CLIENT_OUT=<results on the client> PREFIX=<container prefix> RIG_PORT=8282 RELAY_PORT=<port> PROXY_PORT=<port> \
  bash docs/data/2026-11-26_r243_flash_131k_agent_rig1/run/flash_131k.sh
ARM=l131a WORK=<scratch> NEW=<build folder on the rig> DRAFT=<MTP draft on the rig> \
  bash docs/data/2026-11-26_r243_flash_131k_agent_rig1/run/dry_line.sh > line_l131a.txt
git show d296f5ce:release/flash-next-v5/settings.env > preset_d296f5ce.env
PRESET=preset_d296f5ce.env bash docs/data/2026-11-26_r243_flash_131k_agent_rig1/run/preset_check.sh <started command of l131a> l131   # ca, cb: preset
python3 -I docs/data/2026-11-19_r208_r210_flash_switches/run/rename_arm.py <CLIENT_OUT>/res agent/l131a v4r l131a
bash docs/data/2026-11-26_r243_flash_131k_agent_rig1/run/tables_agent.sh TABLES ca l131a cb l131b
```

`TABLES` holds each run's server log, the rig's `meminfo.log` and card log, the proxy's rows and bodies and the run's
task window, as [run/tables_agent.sh](run/tables_agent.sh) lists; the card memory takes `CARD_TOTAL_MIB` (6,068 by
default).

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The agent check's tasks are the
bench's own ([bench/agent_check](../../../bench/agent_check/README.md)). The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next with its MTP draft, under the makers' licences.
