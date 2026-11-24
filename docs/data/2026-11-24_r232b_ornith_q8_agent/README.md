# Ornith with a Q8_0 cache and pull request 9's attention kernels: four alternated agent checks on the second rig (R232 (b), Q1706; 2026-11-24)

**In plain words.** Ornith-1.5-122B keeps its attention cache in 16-bit numbers (F16). An 8-bit cache (Q8_0) is half
the size, and a contributor's pull request 9 added a decode kernel for it on Pascal cards. On one card at Ornith's
exact shape that kernel made a one-token attention call 1.85 times faster than the F16 cache at 8,192 tokens, but
prompts were read slower ([R232 (a)](../2026-11-24_r232a_ornith_q8_kernels/README.md)). This folder decides whether a
whole coding-agent session gets faster: four agent checks on one boot of the second rig, every card at its soaked clock
profile, alternating Ornith's v5 preset as it is (F16 cache, the control) and the same with a Q8_0 cache and the two
switches that route its one-token calls to the new kernel.

**Run 1, the control (`o5f16a`, 2026-11-24 16:14–16:45): 12 of 12 tasks passed, 1.88 minutes a task at equal work,
19.42 tok/s generation, and no reply was read again.** As context only, R17's two runs of the same preset on the same
rig and clock profile, on another boot and an older v5 engine (`602691153`): 1.89 and 1.90 minutes a task, 19.41 and
19.01 tok/s.

**Run 2, the Q8_0 cache (`o5q8a`, 16:52–17:26): 12 of 12 tasks passed, 1.84 minutes a task at equal work, 20.39 tok/s
generation, and no reply was read again.** Like for like against run 1, a decode step took 48.76 ms against 51.60 at
the four runs' shared mix of contexts (5.5 % less), and the 19 requests read on the same prompt took 184.21 s against
179.86 (2.4 % more): faster generation, slower prompts, as the kernel times said. No class before the fourth run.

## The runs

The release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at the preset's 49,152 tokens on nine P106-090 cards, on the
v5 engine (`pascal-p106-v5` at `af00bb13b`, which holds pull request 9's kernels) with Ornith's v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)). The arms differ only in the cache and the
two switches ([run/ornith_q8.sh](run/ornith_q8.sh)):

| Run | Arm | Added to the v5 preset | Status |
|---|---|---|---|
| 1, F16 a | `o5f16a` | nothing | done 16:14–16:45, 12/12 |
| 2, Q8_0 a | `o5q8a` | `-ctk q8_0 -ctv q8_0`; `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` | done 16:52–17:26, 12/12 |
| 3, F16 b | `o5f16b` | nothing | |
| 4, Q8_0 b | `o5q8b` | as run 2 | |

The class will be two runs against two by [class_arms.py](../../../bench/agent_check/class_arms.py): a measure is a
gain or a loss only where the difference of the two arms' means is beyond the larger of their spreads. Beside them, as
context, R17's text-reuse runs `o5t1a` and `o5t1b`
([R17](../2026-11-19_r17_ornith_textreuse/README.md#text-reuse-run-b-o5t1b-r17f-2026-11-19)): the same preset,
rig and clock profile and the same started command but for the engine's path, on 2026-11-19's boot with the older v5
engine `602691153`. They are not a control: the engine and the boot differ.

## Runs 1 and 2: `o5f16a` (F16 a) and `o5q8a` (Q8_0 a), 2026-11-24

The shared mix of contexts and the requests read on the same prompt are taken over every run in the table, so they are
recomputed as runs are added.

| | R17d `o5t1a` (context) | R17f `o5t1b` (context) | **`o5f16a`** | **`o5q8a`** |
|---|---|---|---|---|
| passed | 12/12 | 11/12 (Python `scale-generator` at the 12-minute cap) | **12/12** | **12/12** |
| minutes a task at equal work (`v3ka`'s work) | 1.89 | 1.90 | **1.88** | **1.84** |
| minutes, all 12 tasks; a task's median; turns | 21.4; 1.33; 64 | 34.8; 1.52; 55 | **21.9; 1.68; 59** | **25.0; 1.76; 65** |
| generation tok/s, session | 19.41 | 19.01 | **19.42** | **20.39** |
| ms a decode step at the four runs' shared mix of contexts; tok/s at it | 51.39; 19.46 | 51.31; 19.49 | **51.60; 19.38** | **48.76; 20.51** |
| prompt tok/s, session | 102.5 | 106.0 | **104.3** | **100.5** |
| the 19 requests read on the same prompt in all four, prompt s | 180.29 | 180.80 | **179.86** | **184.21** |
| of them the 12 first turns, prompt s | 110.39 | 110.77 | **110.01** | **111.86** |
| first token median; p90 s | 6.13; 9.58 | 6.94; 10.31 | **6.59; 10.27** | **6.16; 9.95** |
| first token on a task's first turn, median s | 8.00 | 7.98 | **8.01** | **8.16** |
| longest first-token wait s (task, turn) | 26.70 (Python `affine-cipher`, 1) | 26.89 (the same) | **26.28 (the same)** | **26.48 (the same)** |
| prompt tokens read (a task) | 36,956 (3,080) | 34,628 (2,886) | **35,247 (2,937)** | **36,712 (3,059)** |
| tokens generated (a task) | 17,458 (1,455) | 21,478 (1,790) | **18,433 (1,536)** | **22,559 (1,880)** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 360.4) | 0 (0; 0.0 of 326.8) | **0 (0; 0.0 of 337.8)** | **0 (0; 0.0 of 365.5)** |
| tool calls / failed / malformed | 64 / 2 / 0 | 55 / 1 / 0 | **59 / 1 / 0** | **65 / 1 / 0** |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 11 / 0 / 0; 0; 0 in 64 | 11 / 0 / 0; 0; 0 in 54 | **11 / 0 / 0; 0; 0 in 59** | **11 / 0 / 0; 0; 0 in 65** |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8) | 2113 2138 2138 2088 2100 2088 2151 2151 2113 (2,120.0) | **2113 2138 2138 2088 2100 2100 2151 2138 2126 (2,121.3)** | **2113 2138 2138 2088 2100 2088 2151 2138 2126 (2,120.0)** |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | **0; 0** | **0; 0** |
| lowest MemAvailable MiB; swap MiB during the load | 1,189; 53 → 363 | 1,227; 53 → 363 | **1,223; 50 → 362** | **1,240; 52 → 362** |

- **The work** ([work.md](work.md)): in both runs every later turn found the previous prompt and reply in the cache (no
  reply read again; median cache hit from turn 2 95.0 % in `o5f16a` and 98.5 % in `o5q8a`,
  [o5f16a_gonogo.md](o5f16a_gonogo.md), [o5q8a_gonogo.md](o5q8a_gonogo.md)). The longest wait is Python
  `affine-cipher`'s first turn, a 2,742-token prompt with nothing cached: 26.28 s and 26.48 s.
- **Run 1 runs at R17's speed:** a decode step at the four runs' shared mix 51.60 ms against 51.39 and 51.31
  ([decode.md](decode.md)); the 19 requests every run read on the same prompt 179.86 s against 180.29 and 180.80
  ([same_prompts.md](same_prompts.md)). These compare two engines on two boots, so they are context, not a class.
- **Run 2 generates faster and reads prompts slower, as the kernel times said** ([R232 (a)](../2026-11-24_r232a_ornith_q8_kernels/README.md)):
  - a decode step at the shared mix took 48.76 ms against `o5f16a`'s 51.60: 2.84 ms (5.5 %) less, 20.51 tok/s against
    19.38;
  - it was faster in every context bucket, and the gap grows with the context ([decode.md](decode.md)): 47.45 against
    48.34 ms under 3,000 tokens, 47.91 against 49.98 at 3,000 to 4,500, 48.64 against 51.54 at 4,500 to 6,000, and
    49.68 against 53.18 at 6,000 to 8,000 (R232 (a)'s kernel times: est. 3.42 ms less at 8,192 tokens);
  - the 19 same-prompt requests took 184.21 s against 179.86 (4.36 s, 2.4 % more), of them the 12 first turns 111.86
    against 110.01.
- **Run 2 did more work:** it generated 22,559 tokens against run 1's 18,433, mostly in Python `scale-generator`
  (7,557 tokens against 3,462: 6.8 minutes against 3.6) and `affine-cipher` (12 turns against 6), so its 12 tasks took
  25.0 minutes against 21.9. At equal work (`v3ka`'s) it took 1.84 minutes a task against 1.88. One run of each arm so
  far: the class waits for runs 3 and 4.
- **Answers sound in both runs:** every task passed, also its hidden tests, test files untouched; 0 of 59 and 0 of 65
  calls malformed, none inside the reasoning. Each run's one failed call is an edit in Python `affine-cipher` whose old
  text did not match the file, which the tool answered with an error; the agent's next edit went through. No NaN or
  inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load, as for
  every Ornith run of this preset).
- **Power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): above 80 W in 0 4 4 0 1 0 0 1 4 single
  samples on cards 0 to 8 in `o5f16a` (highest 87.81 W on card 1) and 2 3 3 2 1 1 1 1 3 in `o5q8a` (highest 84.50 W
  on card 1); the clock watchdog (80 W for 5 s) never tripped and the boot logged no GPU Xid.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,223 MiB (16:42:47) in `o5f16a` and 1,240 MiB
  (17:24:40) in `o5q8a`; each card's peak memory 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB on cards
  0 to 8 in `o5f16a`, and 5,784 5,756 5,730 5,756 5,408 5,516 5,678 5,706 5,156 MiB in `o5q8a`, which is lower by
  24 72 24 72 28 28 28 76 32 MiB (the Q8_0 cache holds half the bytes of the F16 one).

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 13:43 and set to its soaked clock profile between 14:02 and
  14:19 by the offset kit's `apply` ([R234](../2026-11-19_r209_oc_v5/r234_apply.md)): memory +1,300 +1,000 +1,000 +1,100 +1,100
  +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8 (R17's values),
  the 75 W power limit, the kit's watchdog running. Before and after each run every card read back at those values, no
  load process, 0 Xid, 0 step-downs. Nothing was set or reset during or between the runs. R232 (a) ran on card 0 of the
  same boot before them (14:41–16:01).
- **The build:** `pascal-p106-v5` `af00bb13b`, the rig's tarball `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2`
  (`llama-server` md5 `f5cca34ea6bd4f3712cd6193bf0605be`, checked before and after).
- **The started command** of `o5f16a` (575 bytes with its newline, md5 `1da8fb3fcf808af67ebcf41f008df06d`) is the
  launcher's dry run byte for byte (`cmp`), and R17's text-reuse runs' line byte for byte but for the engine's path; the
  server logged `switches: 8 set`, the v5 preset's eight:

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  ```

  The Q8_0 arm's line is the same with `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` after
  `LLAMA_SERVER_TEXT_REUSE=1` and `-ctk q8_0 -ctv q8_0` at the end: `o5q8a` started it as 658 bytes with its newline
  (md5 `070c3b31b29cd8289fcd3f048c616b8a`), `cmp`-equal to `o5f16a`'s line with those two edits, and its server logged
  `switches: 10 set`, the preset's eight and the two attention switches.
- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the bench files R17's runs used. Each run has its own client folder, which also names its
  containers, so another check on the same client at the same time cannot remove them. The work tables come from v5's
  [report.py](../../../bench/agent_check/report.py) `work` on the published rows.
- **Timeline:** run 1's server started at 16:14:08 and was up after 542 s (R17's runs: 658, 653, 660 and 662 s); the
  client reached it at 16:23:11, the 12 tasks ran 16:23:11–16:45:14, and the rig's logs went to the client at 16:45:35.
  Run 2's server started at 16:52:14 and was up after 539 s; the client reached it at 17:01:14, the 12 tasks ran
  17:01:14–17:26:27, and the rig's logs went to the client at 17:26:46. Nothing else ran on the rig.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local ARM=f16 RUN=a \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2/bin/llama-server \
  RIG_LOGDIR=<rig folder of this run> RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r232b-o5f16a OUT=<out>/agent-o5f16a \
  bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/ornith_q8.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# the other runs: ARM=q8 RUN=a, ARM=f16 RUN=b, ARM=q8 RUN=b, each with its own rig folder and client folder
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` (the equal-work reference of every v5 Ornith table, as
[R17](../2026-11-19_r17_ornith_textreuse/README.md#commands) used them), `o5t1a` and `o5t1b`
([R17's agent/](../2026-11-19_r17_ornith_textreuse/agent/)) and each run's rows ([agent/o5f16a/](agent/o5f16a/),
[agent/o5q8a/](agent/o5q8a/)), each run's server log as `rig/server-<arm>.log`, and in `rig_<arm>/` its
`meminfo.log`, `cards.csv` and `window` (from the driver's log; `o5f16a`'s `2026-11-24 16:23:11|2026-11-24 16:45:14`,
`o5q8a`'s `2026-11-24 17:01:14|2026-11-24 17:26:27`):

```bash
bash docs/data/2026-11-24_r232b_ornith_q8_agent/run/tables.sh <tables folder> o5f16a o5q8a
```

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5t1a`, `o5t1b`, `o5f16a`, `o5q8a`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `o5t1a`, `o5t1b` and each run: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [reallocs.tsv](reallocs.tsv), [o5f16a_requests_reallocs.tsv](o5f16a_requests_reallocs.tsv), [o5q8a_requests_reallocs.tsv](o5q8a_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5f16a_compare.md](o5f16a_compare.md), [o5f16a_report_table.md](o5f16a_report_table.md), [o5f16a_gonogo.md](o5f16a_gonogo.md); [o5q8a_compare.md](o5q8a_compare.md), [o5q8a_report_table.md](o5q8a_report_table.md), [o5q8a_gonogo.md](o5q8a_gonogo.md) | the harness's report of each run: per task, per turn and the go/no-go |
| [agent/o5f16a/](agent/o5f16a/), [agent/o5q8a/](agent/o5q8a/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of each run |
| [run/ornith_q8.sh](run/ornith_q8.sh), [run/tables.sh](run/tables.sh) | the launcher of the four runs (`ARM=f16\|q8`, `RUN=a\|b`) and the tables |

## Provenance and licence

Measured on the project's own rig on 2026-11-24 with the repository's own scripts and Miner Llama's v5 engine (MIT, like
llama.cpp, with pull request 9's kernels under the same licence); the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/<arm>/attempts.csv`); the
tables hold times, counts and task ids, no text. The tables and scripts here are under the repository's licence.
