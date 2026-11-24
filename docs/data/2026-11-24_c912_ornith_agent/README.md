# Ornith's agent check on the upstream b12674 candidate (C912, Q1649; 2026-11-24)

**In plain words.** The fork's next engine is upstream llama.cpp's newest release (tag b12674) merged into the v5 line,
the candidate `pascal-p106-v5-cand-b12674` `eee615cdb` ([the merge](../2026-11-24_c912_upstream_b12674/README.md)).
On a P106 it already gives the same replies, memory and kernel launches as the current v5 engine `af00bb13b`
([the P106 check](../2026-11-24_c912_p106_check/README.md)). Before `pascal-p106-v5` moves to it, one coding-agent run
per model checks upstream's server and scheduler changes in real use. This is Ornith's: Ornith-1.5-122B Pascal N2P
GGUF with its v5 preset as it is (F16 cache), on the second rig at its soaked clock profile, the same launcher and
started command as [R232 (b)](../2026-11-24_r232b_ornith_q8_agent/README.md)'s control runs of the same preset on
`af00bb13b`, on the same boot, with only the engine changed.

**The candidate (`o5cand`, 2026-11-24 17:52–18:22): 12 of 12 tasks passed, no server error, 1.87 minutes a task at
equal work against the control run's 1.88, and no reply was read again.** At the two runs' shared mix of contexts a
decode step took 51.37 ms against `o5f16a`'s 51.46. Prompts were read faster: the 19 requests read on the same prompt
took 176.72 s against 182.57 (3.2 % less), and each of the 19 was faster. Each card's peak memory was the same to the
MiB. One run of each engine so far: R232 (b)'s run 3 (`o5f16b`, `af00bb13b` again) gives the control's second reading
before any verdict.

## The run

The release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix tasks,
Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on Ornith-1.5-122B Pascal N2P GGUF
(valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at the preset's 49,152 tokens on nine P106-090 cards, with
Ornith's v5 preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)). The run script
([run/ornith_cand.sh](run/ornith_cand.sh)) is R232 (b)'s with its F16 line as it is; it differs only in the arm's name.

| Run | Arm | Engine | Status |
|---|---|---|---|
| R232 (b) run 1 (the control) | `o5f16a` | `pascal-p106-v5` `af00bb13b` | done 16:14–16:45, 12/12 |
| this run | `o5cand` | the candidate `pascal-p106-v5-cand-b12674` `eee615cdb` | done 17:52–18:22, 12/12 |
| R232 (b) run 3 (the control again) | `o5f16b` | `af00bb13b` | next on the same boot |

The control is R232 (b)'s run 1: the same rig, boot and clock profile, the same model file, launcher, harness, client
and started command but for the engine's path. Between the two, R232 (b)'s run 2 (`o5q8a`, a Q8_0 cache) ran on the
same rig. Beside them, as context only, R17's text-reuse runs `o5t1a` and `o5t1b` (the same preset, rig and kind of
clock profile, on 2026-11-19's boot with the older v5 engine `602691153`): they are not a control.

## `o5cand` against `o5f16a`

The shared mix of contexts and the requests read on the same prompt are taken over the four runs in this table
(`o5t1a`, `o5t1b`, `o5f16a`, `o5cand`), so `o5f16a`'s values here differ slightly from R232 (b)'s table, which takes
them over that folder's runs.

| | R17d `o5t1a` (context) | R17f `o5t1b` (context) | **`o5f16a`** (`af00bb13b`) | **`o5cand`** (`eee615cdb`) |
|---|---|---|---|---|
| passed | 12/12 | 11/12 (Python `scale-generator` at the 12-minute cap) | **12/12** | **12/12** |
| minutes a task at equal work (`v3ka`'s work) | 1.89 | 1.90 | **1.88** | **1.87** |
| minutes, all 12 tasks; a task's median; turns | 21.4; 1.33; 64 | 34.8; 1.52; 55 | **21.9; 1.68; 59** | **20.0; 1.62; 62** |
| generation tok/s, session | 19.41 | 19.01 | **19.42** | **19.49** |
| ms a decode step at the four runs' shared mix of contexts; tok/s at it | 51.22; 19.52 | 51.18; 19.54 | **51.46; 19.43** | **51.37; 19.47** |
| prompt tok/s, session | 102.5 | 106.0 | **104.3** | **105.5** |
| the 19 requests read on the same prompt in all four, prompt s | 182.95 | 183.47 | **182.57** | **176.72** |
| of them the 12 first turns, prompt s | 110.39 | 110.77 | **110.01** | **106.38** |
| first token median; p90 s | 6.13; 9.58 | 6.94; 10.31 | **6.59; 10.27** | **6.06; 9.38** |
| first token on a task's first turn, median s | 8.00 | 7.98 | **8.01** | **7.71** |
| longest first-token wait s (task, turn) | 26.70 (Python `affine-cipher`, 1) | 26.89 (the same) | **26.28 (the same)** | **25.55 (the same)** |
| prompt tokens read (a task) | 36,956 (3,080) | 34,628 (2,886) | **35,247 (2,937)** | **35,815 (2,985)** |
| tokens generated (a task) | 17,458 (1,455) | 21,478 (1,790) | **18,433 (1,536)** | **16,259 (1,355)** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 0 (0; 0.0 of 360.4) | 0 (0; 0.0 of 326.8) | **0 (0; 0.0 of 337.8)** | **0 (0; 0.0 of 339.4)** |
| tool calls / failed / malformed | 64 / 2 / 0 | 55 / 1 / 0 | **59 / 1 / 0** | **62 / 1 / 0** |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 11 / 0 / 0; 0; 0 in 64 | 11 / 0 / 0; 0; 0 in 54 | **11 / 0 / 0; 0; 0 in 59** | **11 / 0 / 0; 0; 0 in 62** |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8) | 2113 2138 2138 2088 2100 2088 2151 2151 2113 (2,120.0) | **2113 2138 2138 2088 2100 2100 2151 2138 2126 (2,121.3)** | **2113 2126 2138 2088 2100 2088 2151 2138 2113 (2,117.2)** |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | **0; 0** | **0; 0** |
| lowest MemAvailable MiB; swap MiB during the load | 1,189; 53 → 363 | 1,227; 53 → 363 | **1,223; 50 → 362** | **1,193; 52 → 365** |
| server load, s | | | **542** | **546** |

What this one run shows:
- **The candidate runs the agent sessions soundly.** Every task passed, also its hidden tests, test files untouched;
  0 of 62 calls malformed, none inside the reasoning. The one failed call is an edit in Python `affine-cipher` whose old
  text did not match the file, which the tool answered with an error, as in `o5f16a`; the agent's next edit went
  through. The server log has no error line and the same set of warning lines as `o5f16a`'s; no NaN or inf, no
  watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load, as for every
  Ornith run of this preset; [o5cand_gonogo.md](o5cand_gonogo.md)).
- **Upstream's new prompt cache reuse changes nothing here:** every later turn found the previous prompt and reply in
  the cache (no reply read again, median cache hit from turn 2 96.0 %; 11 checkpoint restores in both runs, none
  created or erased, no full re-processing, no graph reallocation; [work.md](work.md), [reallocs.tsv](reallocs.tsv)).
- **Generation at the control's speed:** a decode step at the shared mix 51.37 ms against 51.46 (0.09 ms, 0.2 %, less),
  and within 0.19 ms of `o5f16a` in every context bucket ([decode.md](decode.md)): 48.38 against 48.34 ms under 3,000
  tokens, 49.89 against 49.98 at 3,000 to 4,500, 51.35 against 51.54 at 4,500 to 6,000, 53.31 against 53.18 at 6,000
  to 8,000.
- **Prompts read faster:** the 19 same-prompt requests took 176.72 s against 182.57 (5.85 s, 3.2 % less), and every one
  of the 19 took less ([same_prompts.md](same_prompts.md)): the smallest difference 0.7 % (JavaScript `grade-school`'s
  first turn, 6,095.3 against 6,140.2 ms), the largest 6.2 % (JavaScript `phone-number`'s first turn, 7,518.4 against
  8,016.9 ms); the 12 first turns 106.38 against 110.01 s. The two context runs on the older engine read the same requests in 182.95 and 183.47 s. Upstream b12674 changes
  the sm_61 code only in prompt processing, and one of its two changes, the 128-row `mul_mat_q` tiles for 64-column
  passes (#31352), reaches every prompt pass of 49 or more tokens of Ornith's Q5_0 and Q8_0 products
  ([C1024](../2026-11-24_c1024_mmq_j64_tile/README.md)); whether it is the cause is registry Q1791.
- **Less work in this run:** it generated 16,259 tokens against `o5f16a`'s 18,433 (JavaScript
  `variable-length-quantity` 1,133 against 2,372, Python `variable-length-quantity` 1,951 against 2,867), with three
  more turns, so its 12 tasks took 20.0 minutes against 21.9. At equal work (`v3ka`'s) it took 1.87 minutes a task
  against 1.88.
- **Memory the same:** each card's peak 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB on cards 0 to 8 in
  both runs ([host.tsv](host.tsv)); lowest MemAvailable 1,193 MiB (18:18:13) against 1,223.
- **Power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): above 80 W in 1 5 3 2 2 3 3 4 1 single
  samples on cards 0 to 8 (highest 87.01 W on card 1; `o5f16a`: 0 4 4 0 1 0 0 1 4, highest 87.81 W on card 1); the
  clock watchdog (80 W for 5 s) never tripped and the boot logged no GPU Xid.

**No class yet.** A measure is a gain or a loss only where the difference is beyond the spread of repeated runs
([class_arms.py](../../../bench/agent_check/class_arms.py)); the control's second run (`o5f16b`) gives that spread. Until then, one run against one: the same generation speed, prompts 3.2 %
faster, the same answers' soundness and memory.

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 13:43 and set to its soaked clock profile between 14:02 and
  14:19 by the offset kit's `apply` ([R234](../2026-11-19_r209_oc_v5/r234_apply.md)): memory +1,300 +1,000 +1,000 +1,100
  +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206 +282 +256 +231 MHz on cards 0 to 8, the 75 W
  power limit, the kit's watchdog running. Before and after the run every card read back at those values, no load
  process, 0 Xid, 0 step-downs. Nothing was set or reset. R232 (b)'s runs 1 and 2 ran before it on the same boot.
- **The build:** the candidate's rig tarball `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz`
  (1,403,235,928 B, md5 `575ef3c702f69f14d7d3022c1287c38f` at both ends of the copy), `llama-server` md5
  `2727faf28b5b34dfe42399a9d431da67`, checked before and after. The control's: `af00bb13b`, `llama-server`
  `f5cca34ea6bd4f3712cd6193bf0605be`.
- **The started command** (575 bytes with its newline) is `o5f16a`'s byte for byte with the engine's path replaced
  (`cmp`): the v5 preset's eight switches, and the server logged `switches: 8 set` (the candidate reads 960 switch
  names, the control 905):

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SERVER_TEXT_REUSE=1
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  ```

- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the same bench files and harness as R232 (b) (`bench/agent_check` and the release check
  script identical to that folder's copy). The work tables come from v5's
  [report.py](../../../bench/agent_check/report.py) `work` on the published rows.
- **Timeline:** the server started at 17:52:50 and was up after 546 s; the client reached it at 18:01:57, the 12 tasks
  ran 18:01:57–18:22:09, and the rig's logs went to the client at 18:22:21. Nothing else ran on the rig.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local NAME=o5cand \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674/bin/llama-server \
  RIG_LOGDIR=<rig folder of this run> RIG_PORT=8282 RELAY_PORT=8298 PROXY_PORT=18798 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/c912-o5cand OUT=<out>/agent-o5cand \
  bash docs/data/2026-11-24_c912_ornith_agent/run/ornith_cand.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` (the equal-work reference of every v5 Ornith table), `o5t1a` and `o5t1b`
([R17's agent/](../2026-11-19_r17_ornith_textreuse/agent/)), `o5f16a`
([R232 (b)'s agent/o5f16a/](../2026-11-24_r232b_ornith_q8_agent/agent/o5f16a/)) and `o5cand`
([agent/o5cand/](agent/o5cand/)), each run's server log as `rig/server-<arm>.log`, and in `rig_<arm>/` its
`meminfo.log`, `cards.csv` and `window` (from the driver's log; `o5f16a`'s `2026-11-24 16:23:11|2026-11-24 16:45:14`,
`o5cand`'s `2026-11-24 18:01:57|2026-11-24 18:22:09`):

```bash
bash docs/data/2026-11-24_c912_ornith_agent/run/tables.sh <tables folder> o5f16a o5cand
# after R232 (b)'s run 3, with its rows and logs added the same way:
bash docs/data/2026-11-24_c912_ornith_agent/run/tables.sh <tables folder> o5f16a o5cand o5f16b
```

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5t1a`, `o5t1b`, `o5f16a`, `o5cand`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `o5t1a`, `o5t1b`, `o5f16a` and `o5cand`: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again (none) |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [reallocs.tsv](reallocs.tsv), [o5f16a_requests_reallocs.tsv](o5f16a_requests_reallocs.tsv), [o5cand_requests_reallocs.tsv](o5cand_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5cand_compare.md](o5cand_compare.md), [o5cand_report_table.md](o5cand_report_table.md), [o5cand_gonogo.md](o5cand_gonogo.md) | the harness's report of the run: per task, per turn and the go/no-go (`o5f16a`'s are in [R232 (b)'s folder](../2026-11-24_r232b_ornith_q8_agent/)) |
| [agent/o5cand/](agent/o5cand/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of the run |
| [run/ornith_cand.sh](run/ornith_cand.sh), [run/tables.sh](run/tables.sh) | the run script (R232 (b)'s F16 line with the arm's name as a setting) and the tables |

## Provenance and licence

Measured on the project's own rig on 2026-11-24 with the repository's own scripts and Miner Llama's candidate engine
(MIT, like llama.cpp); the agent check's tasks are [bench/agent_check](../../../bench/agent_check/README.md)'s (their
task IDs are in `agent/o5cand/attempts.csv`); the tables hold times, counts and task ids, no text. The tables and
scripts here are under the repository's licence.
