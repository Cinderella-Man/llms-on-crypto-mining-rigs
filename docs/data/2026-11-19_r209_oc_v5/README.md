# R209: the full overclock on the v5 engine, Ornith and Flash-Next on one boot (Q1603, 2026-11-19)

**In plain words.** Two separate gains landed on 2026-11-18. The second rig's full clock profile (a memory and a core
offset on every card, [R202b](../2026-11-18_r202b_core_soak/README.md)) made Ornith's coding agent about 8 % faster in
a first look on the v4.0 engine ([R203e](../2026-11-18_r203_ornith_offsets/README.md#the-full-profile-first-look-o4oc2-2026-11-18)),
and the v5 engine with Ornith's v5 preset made each decode step 6.3 % shorter at stock
([R204](../2026-11-18_r204_ornith_glu_agent/README.md)). This row asks whether they add up. On one fresh boot of the
second rig it runs five steps in order: Ornith at stock, Flash-Next at stock, the kit's `apply` of the full profile
with Ornith's whole-memory check, Flash-Next with the offsets, Ornith with the offsets; then every offset back to 0.
The comparison of the row is each model with the offsets against the same model at stock on this boot.

**Run 1, Ornith at stock on the v5 engine: 12 of 12 tasks passed, 2.00 minutes a task at equal work, generation
17.87 tok/s over the session and 17.84 tok/s at R204's mix of contexts.** This is the stock number the offset run on
the same boot will be compared with. As context only (another engine, or another rig): on the same rig at stock with
v4.0's engine and preset ([R203a](../2026-11-18_r203_ornith_offsets/README.md#the-stock-half-o4oc0-2026-11-18), a
day earlier) a decode step at the same mix of contexts took 6.33 % longer; on the first rig with this engine and preset
(R204's two runs) it took 0.64 % and 0.99 % longer.

Registry row Q1603 in [quants/experiments.md](../../../quants/experiments.md). The verdict of the row comes after the
offset runs.

## The method

The agent check of [R166](../2026-11-16_r166_ornith_stack_no_taper/README.md) and
[R204](../2026-11-18_r204_ornith_glu_agent/README.md) ([bench/agent_check](../../../bench/agent_check/README.md):
12 bug-fix tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget, the makers' sampling
at temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on the second rig's nine P106-090 cards, one load a run:

- **The engine:** the v5 engine, `pascal-p106-v5` at `602691153` (upstream's b12483 merged); the rig tree's
  `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`, checked before the run.
- **The preset:** Ornith's v5 preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)): v4.0's
  preset plus `GGML_CUDA_MMID_GLU_UNFUSE=2`, through R204's harness call
  ([ornith_agent_glu.sh](../2026-11-18_r204_ornith_glu_agent/run/ornith_agent_glu.sh)). The started command is R204's
  `o4g2a` byte for byte (549 bytes with its newline, md5 `a0a97e752b2acbfef50bbaf86250e1e5`), and the server's
  `switches:` line is the same: `7 set: GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode
  GGML_CUDA_KEEPALIVE_EAGER=1 GGML_CUDA_MMID_GLU_UNFUSE=2 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`, 889 names
  read.
- **Flash-Next:** R203c's run ([run/flash_run.sh](../2026-11-18_r203_ornith_offsets/run/flash_run.sh): R156's driver,
  the released v4.0 preset) with the v5 engine's `llama-server`.
- **The offsets:** R202b's full profile ([use-profile.json](../2026-11-18_r202b_core_soak/use-profile.json): a memory
  and a core offset on every card), set by the offset kit's `apply`.

Arms: `o5oc0` (Ornith at stock), `f5oc0` (Flash-Next at stock), `f5oc1` and `o5oc1` (with the offsets). Beside them,
as context only: R203a's `o4oc0` (the same rig at stock on v4.0's engine and preset, 2026-11-18) and R204's `o4g2a` and
`o4g2b` (this engine and preset on the first rig, 2026-11-18).

## Ornith at stock: `o5oc0` (2026-11-19)

The rig was booted at 09:18:21. Before the run the offset kit's read-only start check passed: every card at 0 offset,
75 W, 1,911 / 4,004 MHz maximum, P8, 0 MiB, no load process, no GPU Xid, no step-down this boot, 3,168 MiB of host
memory available. The server started at 09:41:22 and was up after 585 s (R203a on this rig: 642 s; R204 on the first
rig: 683 and 681 s); the tasks ran 09:51:09–10:11:21. After the run the same check passed again, every offset still 0.

| | R203a `o4oc0` (rig 2, v4.0's engine and preset, 2026-11-18) | R204a `o4g2a` (rig 1, this engine and preset, 2026-11-18) | R204b `o4g2b` (rig 1, the same) | **R209a `o5oc0`, stock** |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.13 | 2.04 | 2.05 | **2.00** |
| generation tok/s, the whole session | 16.42 | 17.66 | 17.28 | **17.87** |
| ms a decode step at R204's mix of contexts (four buckets) | 59.85 | 56.42 | 56.62 | **56.06** |
| generation tok/s at that mix | 16.71 | 17.72 | 17.66 | **17.84** |
| prompt tok/s, the whole session | 102.2 | 98.5 | 102.8 | **101.9** |
| first token median s; p90 s | 7.07; 11.51 | 6.99; 11.18 | 6.97; 11.74 | **6.78; 11.30** |
| first token on a task's first turn, median s | 8.48 | 8.53 | 8.51 | **8.43** |
| the 20 requests read on the same prompt in all four runs, prompt s | 203.81 | 206.32 | 206.08 | **201.77** |
| of them, the 12 first turns, prompt s | 116.87 | 118.71 | 118.68 | **115.49** |
| prompt tokens read a task | 3,725 | 3,235 | 3,322 | **3,213** |
| tokens generated a task | 1,980 | 1,376 | 1,701 | **1,188** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 8,860 (5; 65.6 of 437.3) | 3,406 (3; 31.2 of 394.3) | 4,592 (2; 31.4 of 387.5) | **3,431 (2; 22.8 of 378.4)** |
| minutes, all 12 tasks; turns | 31.9; 67 | 22.6; 62 | 26.6; 59 | **20.0; 60** |
| tool calls / failed / malformed | 67 / 2 / 0 | 62 / 1 / 0 | 59 / 0 / 0 | **60 / 2 / 0** |
| checkpoints restored / created / erased; full re-processing | 16 / 0 / 0; 0 | 14 / 0 / 0; 0 | 13 / 0 / 0; 0 | **13 / 0 / 0; 0** |
| graph reallocations | 0 in 67 requests | 0 in 62 | 0 in 59 | **0 in 60** |
| busy SM clock, mean of the nine cards' medians, MHz | 1,883.9 | 1,879.8 | 1,881.2 | **1,886.8** |
| lowest MemAvailable MiB; swap highest MiB | 1,207; 334 | 1,223; 343 | 1,205; 342 | **1,248; 336** |

Each value is one run ([agent.tsv](agent.tsv), [decode_r204_mix.md](decode_r204_mix.md),
[same_prompts.md](same_prompts.md), [work.md](work.md), [clocks.tsv](clocks.tsv), [host.tsv](host.tsv),
[reallocs.tsv](reallocs.tsv), [o5oc0_gonogo.md](o5oc0_gonogo.md); the other runs' tool calls and checkpoints from their
folders). Minutes a task at equal work price `ka`'s work (v3.0's 2026-11-06 run on the second rig: 2.52) at each run's
own prompt and generation speeds, so they compare the engine and the cards on the same work. The minutes for all tasks
follow what the agent wrote at temperature 0.6: this run's replies were the shortest of the four (1,188 tokens a task,
none past 8,000 tokens of context; the largest context any task reached was 7,155 tokens).

- **Decode by context** ([decode_r204_mix.md](decode_r204_mix.md)): this run had no request whose middle context
  reached 8,000 tokens, so it cannot be read at R204's five-bucket mix. The table uses R204's four runs' tokens over the
  four buckets every run here decoded in (under 3,000 5.0 %, 3,000 to 4,500 27.4 %, 4,500 to 6,000 49.9 %, 6,000 to
  8,000 17.7 %), and R204's runs are recomputed on that mix (their five-bucket figures in R204's
  [r204b_decode.md](../2026-11-18_r204_ornith_glu_agent/r204b_decode.md) are 57.22 and 57.70 ms). Bucket by bucket, ms
  a token, this run against `o4oc0`: 52.76 against 56.66 under 3,000 tokens (−6.88 %), 54.73 against 58.48 at 3,000 to
  4,500 (−6.41 %), 56.30 against 60.22 at 4,500 to 6,000 (−6.51 %), 58.35 against 61.86 at 6,000 to 8,000 (−5.67 %).
  Against `o4g2a` and `o4g2b`: −0.57 % and −0.68 %, −0.35 % and −0.71 %, −0.81 % and −1.09 %, −0.66 % and −1.19 %.
- **The same prompts** ([same_prompts.md](same_prompts.md)): 20 requests read the same number of new tokens on the same
  cached prefix in all four runs (every task's first turn and eight later turns). They took 201.77 s here, against
  203.81 s in `o4oc0` (−1.00 %) and 206.32 and 206.08 s in `o4g2a` and `o4g2b` (−2.21 %, −2.09 %). With the cached
  tokens matched within one ([same_prompts_tol1.md](same_prompts_tol1.md)) 22 requests: 219.14 s against 221.23,
  224.03 and 223.88 s.
- **The cards' busy clocks** ([clocks.tsv](clocks.tsv), from the card log `scripts/card_log.py` writes every 0.5 s,
  busy = utilization above 0 %, 09:51:09–10:11:21): medians of 1,860 1,885 1,885 1,873 1,911 1,898 1,873 1,898 1,898
  MHz on cards 0 to 8 (`o4oc0` on this rig: 1,860 1,885 1,885 1,860 1,898 1,898 1,873 1,898 1,898); busy samples below
  1,800 MHz 3.50 3.17 3.29 2.49 3.94 3.87 5.38 3.76 2.83 %.
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,248 MiB at 10:10:12, during the last task; swap
  0 MiB at the start and 336 MiB at its highest at 09:50:41, during the load. The RAM watchdog stopped nothing. Card
  peaks from the card log, 5,808 5,828 5,754 5,828 5,436 5,544 5,706 5,782 5,188 MiB, are `o4g2a`'s card for card. The
  report's memory gate is NO-GO on the swap, as for every Ornith run of this preset; every other gate is GO
  ([o5oc0_gonogo.md](o5oc0_gonogo.md)). Every task also passed its hidden tests; no NaN or inf.
- **The client:** the machine that drives the rig was also the agent's client; no other agent check used it as a
  client during the run.
- **The two failed tool calls** were two edits in Python `affine-cipher` whose old text had the wrong indentation;
  the agent read the file's exact line and redid the edit, and the task passed.

## Flash-Next at stock: `f5oc0`

Run 2 of the row; not run yet.

## The full profile's `apply` and Ornith's whole-memory check

Run 3 of the row; not run yet.

## Flash-Next with the offsets: `f5oc1`

Run 4 of the row; not run yet.

## Ornith with the offsets: `o5oc1`

Run 5 of the row; not run yet. It is compared with `o5oc0` above: the decode steps at the same context and the
identical prompts are the like-for-like measures.

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv) | per arm (`ka`'s reference `v3ka`, `o4oc0`, `o4g2a`, `o4g2b`, `o5oc0`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes for all tasks, a task's median, minutes a task at equal work against `v3ka` |
| [decode.md](decode.md) | `o4oc0`, `o4g2a`, `o4g2b` and `o5oc0` by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their own shared mix of contexts |
| [decode_r204_mix.md](decode_r204_mix.md) | `o5oc0` and `o4oc0` by context, and every run at R204's mix (R204's four runs' tokens) over the buckets all of them decoded in |
| [same_prompts.md](same_prompts.md), [same_prompts_tol1.md](same_prompts_tol1.md) | the requests read on the same prompt in `o4oc0`, `o4g2a`, `o4g2b` and `o5oc0` (task, turn, new tokens, cached tokens, each run's prompt ms), exactly and with the cached tokens within one; each run's generation over all its requests |
| [work.md](work.md), [work.tsv](work.tsv) | the work of each run: prompt tokens read, tokens generated, reply tokens re-read and the turns that re-read a reply |
| [clocks.tsv](clocks.tsv) | each card's busy SM clock median, their mean and each card's share of busy samples below 1,800 MHz, over each run's tasks |
| [host.tsv](host.tsv) | lowest MemAvailable and its time, swap at the start and highest, MemAvailable at the log's end, each card's peak memory from the card log |
| [reallocs.tsv](reallocs.tsv), [o5oc0_requests_reallocs.tsv](o5oc0_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size (the four runs), and per request (`o5oc0`) |
| [o5oc0_compare.md](o5oc0_compare.md), [o5oc0_report_table.md](o5oc0_report_table.md), [o5oc0_gonogo.md](o5oc0_gonogo.md) | the agent check's report on `o5oc0`: per task and first token per turn; per arm over the client's attempts; the gates and the server log's counts |
| [agent/o5oc0/](agent/o5oc0/) | `turns.csv` (one row a request) and `attempts.csv` (one row a task) |
| [run/ornith_run.sh](run/ornith_run.sh) | one Ornith run: the rig's and the kit's read-only state, then R204's harness call, the arm and the machine-specific values from the environment |
| [run/tables.sh](run/tables.sh) | Ornith's tables, from a folder holding the rows, server logs, card logs and host logs it lists |
| [run/decode_at_mix.py](run/decode_at_mix.py) | decode speed at a fixed mix of contexts taken from earlier runs, so a new run sits beside an earlier table without moving it |

## Commands

From the repository root on the machine that drives the rig (here also the client), with the rig freshly booted, no
llama process, 0 MiB on every card, and the offset kit's read-only start check passing
(`python3 climb.py state && python3 climb.py start-check` in the kit's folder: every offset 0, no step-down this
boot). The bench files must be on the client in `CLIENT_DIR` (the check's `prep` step copies them).

```bash
# Ornith at stock (~10 min load, 12 tasks), then its report; another check may share the local client, so other ports
RIG=$RIG CLIENT=local ARM=o5oc0 MODEL=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<the v5 tree>/bin/llama-server KIT=<the offset kit's folder> RIG_LOGDIR=<rig folder>/r209a/agent-logs \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r209 RELAY_PORT=8296 PROXY_PORT=18793 \
  OUT=<out> bash docs/data/2026-11-19_r209_oc_v5/run/ornith_run.sh
RIG=$RIG CLIENT=local ARM=o5oc0 MODEL=<the same> BIN=<the same> KIT=<the same> RIG_LOGDIR=<the same> \
  CLIENT_DIR=<the same> CLIENT_OUT=<the same> RELAY_PORT=8296 PROXY_PORT=18793 OUT=<out> STEPS="report collect" \
  bash docs/data/2026-11-19_r209_oc_v5/run/ornith_run.sh

# the tables, on a folder holding the rows and logs run/tables.sh lists; o5oc0's window (rig_o5oc0/window):
# 2026-11-19 09:51:09|2026-11-19 10:11:21 (the driver's log; its clock and the rig's agreed)
bash docs/data/2026-11-19_r209_oc_v5/run/tables.sh <tables folder> o5oc0
```

`agent/o5oc0/` is the client's `res/turns.csv` rows of `o5oc0` and its `res/attempts.csv` rows whose file is `o5oc0`.
`res/` in the tables folder holds those rows plus the rows of `v3ka` and `o4oc0`
([R203](../2026-11-18_r203_ornith_offsets/README.md)) and of `o4g2a` and `o4g2b`
([R204](../2026-11-18_r204_ornith_glu_agent/README.md)); their server, host and card logs are the ones those folders'
tables used. Ornith with the offsets runs the same two commands with `ARM=o5oc1` and its own rig folder, after the
kit's `apply`; then `run/tables.sh <tables folder> o5oc0 o5oc1`.

## Provenance and licence

Measured on the project's own rig on 2026-11-19 with the repository's own scripts; the agent check's tasks are
[bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are in `agent/o5oc0/attempts.csv`). The
tables and scripts here are under the repository's licence.
