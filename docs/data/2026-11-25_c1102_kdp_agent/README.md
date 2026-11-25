# Ornith's v5 preset with pull request 9's `_KDP` switch: four alternated agent checks on the second rig (C1102, Q1835; 2026-11-25)

**In plain words.** Since 2026-11-24 Ornith-1.5-122B's v5 preset keeps its attention cache in 8-bit numbers (Q8_0).
That made decoding faster but prompts slower ([R232 (b)](../2026-11-24_r232b_ornith_q8_agent/README.md)). A third
switch from a contributor's pull request 9, `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1`, scores the 8-bit keys directly in
8-bit arithmetic when a prompt is read. On one P106 at Ornith's exact shape it made a 512-token prompt step 39.67 %
faster ([C1037](../2026-11-25_c1037_pr9_lds_kdp/README.md)). This folder checks whether whole coding-agent sessions get
faster: four agent checks on one boot of the second rig, alternating the preset as it is and the preset plus that
switch.

**The verdict: `_KDP=1` is a gain on agentic speed, and slower on no measure.** All 12 tasks passed in each of the four
runs. Two runs with the switch against two without ([class.tsv](class.tsv)):
- **minutes a task at equal work:** 1.79 and 1.80 against 1.82 and 1.84 (1.9 % less);
- **prompt reading:** 107.5 and 106.9 tok/s against 104.4 and 103.6 (3.1 % more). Like for like, the 21 requests read
  on the same prompt in all four runs took 185.30 and 186.07 s against 191.94 and 192.84 s (3.5 % less), and 20 of the
  21 were faster in both runs with the switch than in both without;
- **first token:** median 5.37 and 4.40 s against 6.29 and 6.49; p90 9.09 and 9.15 s against 10.27 and 9.68; on a
  task's first turn 7.59 and 7.63 s against 7.83 and 7.87;
- **unchanged (negligible):** generation, 20.50 and 20.37 tok/s against 20.40 and 20.15 (a decode step at the four runs'
  shared mix of contexts 48.95 and 49.14 ms against 49.10 and 49.03), the prompt tokens read, and the longest
  first-token wait.

The switch changes the numbers inside the prompt kernel (the query is rounded to 8 bits), so Ornith's v5 preset takes
it only after its quality check on the quant machine passes ([Q1836](../../../quants/experiments.md)).

## The runs

The release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at the preset's 49,152 tokens on nine P106-090 cards, on the
v5 engine candidate `eee615cdb` (`pascal-p106-v5` merged with upstream b12674; its Pascal attention file is the same
as `af00bb13b`'s) with Ornith's v5 preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)).
The arms differ only in the one switch, through R232 (b)'s launcher with its `kdp` arm
([ornith_q8.sh](../2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh)):

| Run | Arm | Added to the v5 preset | Status |
|---|---|---|---|
| 1, preset a | `o5q8a` | nothing | done 15:37–16:04, 12/12 |
| 2, `_KDP` a | `o5kdpa` | `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` | done 16:14–16:39, 12/12 |
| 3, preset b | `o5q8b` | nothing | done 16:50–17:16, 12/12 |
| 4, `_KDP` b | `o5kdpb` | as run 2 | done 17:28–17:51, 12/12 |

The class is two runs against two by [class_arms.py](../../../bench/agent_check/class_arms.py): a measure is a gain or a
loss only where the difference of the two arms' means is beyond the larger of their spreads (the highest run minus the
lowest).

| | **`o5q8a`** | **`o5kdpa`** | **`o5q8b`** | **`o5kdpb`** |
|---|---|---|---|---|
| passed | 12/12 | 12/12 | 12/12 | 12/12 |
| minutes a task at equal work (`v3ka`'s work) | 1.82 | 1.79 | 1.84 | 1.80 |
| minutes, all 12 tasks; a task's median; turns | 26.8; 1.59; 60 | 23.9; 1.37; 68 | 25.7; 1.43; 64 | 22.7; 1.54; 66 |
| generation tok/s, session | 20.40 | 20.50 | 20.15 | 20.37 |
| ms a decode step at the four runs' shared mix of contexts; tok/s at it | 49.10; 20.37 | 48.95; 20.43 | 49.03; 20.39 | 49.14; 20.35 |
| prompt tok/s, session | 104.4 | 107.5 | 103.6 | 106.9 |
| the 21 requests read on the same prompt in all four, prompt s | 191.94 | 185.30 | 192.84 | 186.07 |
| of them the 12 first turns, prompt s | 108.54 | 105.88 | 109.14 | 106.25 |
| first token median; p90 s | 6.29; 10.27 | 5.37; 9.09 | 6.49; 9.68 | 4.40; 9.15 |
| first token median; p90 s over the 57 task-turns all four runs have | 6.62; 9.91 | 6.41; 9.35 | 6.82; 9.93 | 5.45; 9.42 |
| first token on a task's first turn, median s | 7.83 | 7.59 | 7.87 | 7.63 |
| longest first-token wait s (task, turn) | 25.65 (Python `affine-cipher`, 1) | 25.72 (the same) | 26.47 (the same) | 25.74 (the same) |
| prompt tokens read (a task) | 35,671 (2,973) | 36,743 (3,062) | 36,722 (3,060) | 35,062 (2,922) |
| tokens generated (a task) | 25,385 (2,115) | 20,995 (1,750) | 23,411 (1,951) | 20,467 (1,706) |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 341.8) | 0 (0; 0.0 of 341.7) | 0 (0; 0.0 of 354.6) | 0 (0; 0.0 of 328.1) |
| tool calls / failed / malformed | 60 / 1 / 0 | 68 / 2 / 0 | 64 / 1 / 0 | 66 / 1 / 0 |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 11 / 0 / 0; 0; 0 in 60 | 11 / 0 / 0; 0; 0 in 68 | 11 / 0 / 0; 0; 0 in 64 | 11 / 0 / 0; 0; 0 in 66 |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2100 2100 2151 2151 2126 (2,125.6) | 2113 2138 2151 2100 2100 2100 2151 2151 2126 (2,125.6) | 2113 2138 2151 2088 2100 2100 2151 2151 2126 (2,124.2) | 2100 2138 2138 2088 2100 2100 2151 2138 2113 (2,118.4) |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | 0; 0 | 0; 0 |
| lowest MemAvailable MiB; swap MiB during the load | 1,200; 54 → 363 | 1,206; 55 → 369 | 1,146; 55 → 368 | 1,167; 56 → 370 |

- **The class** ([class.tsv](class.tsv)): the `_KDP` arm's two runs against the preset's two:
  - **gain:** minutes a task at equal work 1.79 and 1.80 against 1.82 and 1.84 (1.9 % less; spreads 0.01 and 0.02);
    prompt tok/s 107.5 and 106.9 against 104.4 and 103.6 (3.1 % more; spreads 0.6 and 0.8); first token median 5.37
    and 4.40 s against 6.29 and 6.49 (23.6 % less; spreads 0.97 and 0.20), p90 9.09 and 9.15 s against 10.27 and 9.68
    (8.6 % less), on a task's first turn 7.59 and 7.63 s against 7.83 and 7.87 (3.1 % less; spreads 0.04 and 0.04);
  - **negligible:** generation 20.50 and 20.37 tok/s against 20.40 and 20.15 (spreads 0.13 and 0.25); prompt tokens
    read a task 3,062 and 2,922 against 2,973 and 3,060; the longest first-token wait 25.72 and 25.74 s against 25.65
    and 26.47; no reply was read again in any run;
  - **loss:** none.
- **Prompts, like for like** ([same_prompts.md](same_prompts.md)): the 21 requests read on the same prompt in all four
  runs took 185.30 and 186.07 s with the switch against 191.94 and 192.84 s without: 6.71 s (3.5 %) less between the
  means, beyond both arms' spreads (0.77 and 0.90 s). 20 of the 21 were faster in both runs with the switch than in
  both without. The one that was not is the longest prompt with nothing cached (Python `affine-cipher`'s first turn,
  2,742 tokens from depth 0: 24.70 and 24.81 s against 24.79 and 25.34), where attention is the smallest share of the
  work; C1037 measured the switch's gain smallest at small depths (30.76 % at depth 256 against 39.67 % at 8,192). The
  12 first turns took 105.88 and 106.25 s against 108.54 and 109.14 (2.5 % less).
- **Generation does not change:** a decode step at the four runs' shared mix of contexts took 48.95 and 49.14 ms with
  the switch against 49.10 and 49.03 without ([decode.md](decode.md)). Calls of 1 to 4 tokens go to the decode kernel,
  which the switch does not touch.
- **What the first-token median is made of:** it is taken over all of a run's turns, and the runs with the switch
  took more turns (68 and 66 against 60 and 64; 20 and 18 at turn 5 or later, against 12 and 16), most of them short
  late turns. Over the 57 task-turns all four runs have ([common_turns.md](common_turns.md),
  [run/common_turns.py](run/common_turns.py)) the median is 6.41 and 5.45 s with the switch against 6.62 and 6.82 s,
  and the p90 9.35 and 9.42 s against 9.91 and 9.93: lower on the same turns too, by less than the class's 23.6 %. A
  first-token measure on the same prompts for the class is code row C1075 (Q1808).
- **What it adds up to** (est.): prompt reading is 21.6 % and 23.4 % of the two preset runs' prompt and decode seconds
  ([work.md](work.md), [decode.md](decode.md)). At 3.5 % less prompt time and the same decoding, run 1's work takes
  est. 12.0 s less of its 1,586.0 s (0.8 %), run 3's est. 12.4 s less of 1,516.6 s (0.8 %). The minutes at equal work
  (1.9 % less) price `v3ka`'s work at each run's session speeds, and so carry the session prompt rate (3.1 % more) and
  the generation difference within its spread (0.8 %) as well.
- **The work** ([work.md](work.md)): in all four runs every later turn found the previous prompt and reply in the cache
  (no reply read again; median cache hit from turn 2 96.8 % in `o5q8a`, 99.2 % in `o5kdpa`, 98.9 % in `o5q8b` and
  98.7 % in `o5kdpb`, [o5q8a_gonogo.md](o5q8a_gonogo.md), [o5kdpa_gonogo.md](o5kdpa_gonogo.md),
  [o5q8b_gonogo.md](o5q8b_gonogo.md), [o5kdpb_gonogo.md](o5kdpb_gonogo.md)). The tokens generated differ with what the
  agent did (25,385, 20,995, 23,411 and 20,467), so the minutes for all 12 tasks follow the work and are not classed.
- **Answers sound in all four runs:** every task passed, also its hidden tests, test files untouched; 0 of 60, 0 of
  68, 0 of 64 and 0 of 66 calls malformed, none inside the reasoning. The failed calls are edits in Python
  `affine-cipher` whose old text did not match the file, which the tool answered with an error, and the agent's next
  edit went through: one in runs 1, 3 and 4, two in run 2. No NaN or inf, no watchdog kill. The report's gates: all GO
  except memory (NO-GO on the swap growth during the load, as for every Ornith run of this preset).
- **Power and cards** ([cards.md](cards.md), [clocks.tsv](clocks.tsv), the card log every 0.5 s over the tasks): the
  highest single sample 86.62 W (`o5q8a`), 89.79 W (`o5kdpa`), 88.08 W (`o5q8b`) and 84.54 W (`o5kdpb`); the clock
  watchdog (80 W for 5 s) never tripped and the boot logged no GPU Xid. Busy SM clock medians are the same to within
  one 12.5 MHz step on each card.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,200, 1,206, 1,146 and 1,167 MiB. Each card's peak
  memory is the same in all four runs: 5,784 5,756 5,730 5,756 5,408 5,516 5,678 5,706 5,156 MiB on cards 0 to 8. The
  switch needs no memory of its own.
- **Beside yesterday's runs** (context, not a class: another boot and engine, `af00bb13b`): R232 (b)'s two Q8_0 runs
  took 1.84 and 1.84 minutes a task at equal work with 20.39 and 20.45 tok/s generation and 100.5 and 99.2 tok/s
  prompts; today's preset runs on `eee615cdb` 1.82 and 1.84, 20.40 and 20.15, 104.4 and 103.6.

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 10:53 and set to its soaked clock profile by the offset kit's
  `apply` (R237): memory +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231
  +206 +206 +282 +256 +231 MHz on cards 0 to 8, the 75 W power limit, the kit's watchdog running. Before and after
  each run every card read back at those values, no load process, 0 Xid, 0 step-downs. Nothing was set or reset during
  or between the runs, and nothing else ran on the rig.
- **The build:** the `pascal-p106-v5` candidate `eee615cdb` (upstream b12674 merged), the rig's tree
  `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674` (`llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`,
  checked before and after every run), the same for all four runs.
- **The launcher:** R232 (b)'s [ornith_q8.sh](../2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh) with one new arm,
  `ARM=kdp`: the `q8` arm with `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` at the end of the environment. The script checks
  that the preset file holds the line it was written for; since the preset took the Q8_0 cache, it accepts the file
  in either form and builds every arm from the F16 line, so each arm's command is what it was. A dry run gave the `q8`
  arm's command byte for byte as R232 (b)'s `o5q8a` started it (`cmp`, but for the engine's path), and that line is
  the current v5 preset's.
- **The started command** of `o5q8a` (658 bytes with its newline, md5 `8703269db8c96c429d5f3f97cef9e984`) is the
  launcher's dry run byte for byte, and its server logged `switches: 10 set`, the preset's ten:

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1
    GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall -ctk q8_0 -ctv q8_0
  ```

  `o5kdpa` started the same line with `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` after `GGML_CUDA_FATTN_DEC_PASCAL=4` (692
  bytes, md5 `cbdbedbdb37819fa1414dd96f59258ef`, the dry run byte for byte), and its server logged `switches: 11 set`,
  the preset's ten and `_KDP=1`. `o5q8b` started `o5q8a`'s line and `o5kdpb` `o5kdpa`'s, byte for byte (`cmp`), with
  the same `switches:` lines.
- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the bench files R232 (b)'s runs used. Each run has its own client folder, which also names
  its containers.
- **Timeline:** run 1's server started at 15:28:29 and was up after 544 s; the 12 tasks ran 15:37:34–16:04:37. Run 2's
  server was up after 579 s; tasks 16:14:53–16:39:00. Run 3's after 662 s; tasks 16:50:42–17:16:39. Run 4's after
  660 s; tasks 17:28:15–17:51:08. The load is not part of the check's times.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local ARM=q8 RUN=a \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674/bin/llama-server \
  RIG_LOGDIR=<rig folder of this run> RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/c1102-o5q8a OUT=<out>/agent-o5q8a \
  bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# the other runs: ARM=kdp RUN=a, ARM=q8 RUN=b, ARM=kdp RUN=b, each with its own rig folder and client folder
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` (the equal-work reference of every v5 Ornith table, from
[R232 (b)](../2026-11-24_r232b_ornith_q8_agent/README.md#commands)) and each run's rows ([agent/o5q8a/](agent/o5q8a/),
[agent/o5kdpa/](agent/o5kdpa/), [agent/o5q8b/](agent/o5q8b/), [agent/o5kdpb/](agent/o5kdpb/)), each run's server log
as `rig/server-<arm>.log`, and in `rig_<arm>/` its `meminfo.log`, `cards.csv` and `window` (from the driver's log:
`o5q8a`'s `2026-11-25 15:37:34|2026-11-25 16:04:37`, `o5kdpa`'s `2026-11-25 16:14:53|2026-11-25 16:39:00`, `o5q8b`'s
`2026-11-25 16:50:42|2026-11-25 17:16:39`, `o5kdpb`'s `2026-11-25 17:28:15|2026-11-25 17:51:08`):

```bash
bash docs/data/2026-11-25_c1102_kdp_agent/run/tables.sh <tables folder> o5q8a o5kdpa o5q8b o5kdpb
python3 -I bench/agent_check/class_arms.py <tables folder>/agent.tsv o5kdp=o5kdpa,o5kdpb o5q8=o5q8a,o5q8b > class.tsv
D=docs/data/2026-11-25_c1102_kdp_agent
python3 -I $D/run/common_turns.py o5q8a=$D/agent/o5q8a/turns.csv o5kdpa=$D/agent/o5kdpa/turns.csv \
  o5q8b=$D/agent/o5q8b/turns.csv o5kdpb=$D/agent/o5kdpb/turns.csv > common_turns.md
```

## Files

| File | Content |
|---|---|
| [class.tsv](class.tsv) | `class_arms.py`: the `_KDP` arm's two runs against the preset's two, per measure (runs by value, means, spreads, the difference, the class) |
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5q8a`, `o5kdpa`, `o5q8b`, `o5kdpb`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [common_turns.md](common_turns.md) | the first-token median and p90 over the task-turns every run has, with each run's turns and late turns |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on each run: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait |
| [reallocs.tsv](reallocs.tsv), [o5q8a_requests_reallocs.tsv](o5q8a_requests_reallocs.tsv), [o5kdpa_requests_reallocs.tsv](o5kdpa_requests_reallocs.tsv), [o5q8b_requests_reallocs.tsv](o5q8b_requests_reallocs.tsv), [o5kdpb_requests_reallocs.tsv](o5kdpb_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5q8a_compare.md](o5q8a_compare.md), [o5q8a_report_table.md](o5q8a_report_table.md), [o5q8a_gonogo.md](o5q8a_gonogo.md); [o5kdpa_compare.md](o5kdpa_compare.md), [o5kdpa_report_table.md](o5kdpa_report_table.md), [o5kdpa_gonogo.md](o5kdpa_gonogo.md); [o5q8b_compare.md](o5q8b_compare.md), [o5q8b_report_table.md](o5q8b_report_table.md), [o5q8b_gonogo.md](o5q8b_gonogo.md); [o5kdpb_compare.md](o5kdpb_compare.md), [o5kdpb_report_table.md](o5kdpb_report_table.md), [o5kdpb_gonogo.md](o5kdpb_gonogo.md) | the harness's report of each run: per task, per turn and the go/no-go |
| [agent/o5q8a/](agent/o5q8a/), [agent/o5kdpa/](agent/o5kdpa/), [agent/o5q8b/](agent/o5q8b/), [agent/o5kdpb/](agent/o5kdpb/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of each run |
| [run/tables.sh](run/tables.sh), [run/common_turns.py](run/common_turns.py) | the tables; the first token over the common task-turns. The launcher is R232 (b)'s [ornith_q8.sh](../2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh) (`ARM=kdp`) |

## Provenance and licence

Measured on the project's own rig on 2026-11-25 with the repository's own scripts and Miner Llama's v5 engine (MIT, like
llama.cpp, with pull request 9's kernels under the same licence); the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<arm>/attempts.csv`); the
tables hold times, counts and task ids, no text. The tables and scripts here are under the repository's licence.
