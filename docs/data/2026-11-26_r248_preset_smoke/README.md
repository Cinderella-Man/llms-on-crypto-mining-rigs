# Flash-Next's v5 preset as it stands, one agent run through preset_agent.sh (R248, Q1938, 2026-11-26)

What this measured: one coding-agent run (bench/agent_check, the 12 bug-fix tasks) of Qwen3.8-Flash-Next's v5 preset
exactly as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) stands on public `v5` at
`fa31e1b8`, on nine P106-090 cards, started by the new driver
[bench/agent_check/preset_agent.sh](../../../bench/agent_check/preset_agent.sh), which reads the server line, the
context and the micro-batch from the preset file. Registry row [Q1938](../../../quants/experiments.md). The preset
changed twice that evening: a 131,072-token context with micro-batches of 256 (`-c 131072 -ub 256`,
[R243](../2026-11-26_r243_flash_131k_agent_rig1/README.md)) and `GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.45`
([R245](../2026-11-26_r245_flash_ratio2/README.md)). The driver had not run on a rig before. One run is no class: this is
a smoke test of the driver and the preset together, and the preset's first agent figures as it stands.

**Result: the driver works and the preset runs clean.** The started command equals the driver's dry line byte for byte
and passes a word-by-word check against the preset file; all 12 tasks passed, none at a cap; the card that holds the
output head stays inside its memory with 75 MiB left; the host never ran short or swapped. The driver had one bug, fixed
here: its `files` step copied the bench files with `ssh "$CLIENT"` even when `CLIENT=local`, so the first start stopped
at once, before anything reached the rig.

| | `pre` (this run) |
|---|---|
| tasks passed | 12/12, no task at its 12-minute cap |
| minutes a task at equal work | 2.33 |
| all 12 tasks, minutes (first task's start to last task's end) | 15.2 |
| generation tok/s | 28.22 |
| ms a decode step, all requests | 115.88 |
| first request after loading (2,780 tokens, empty cache), ms | 19,801.1 |
| first token median s, the tasks' first turns | 4.99 |
| card 8 MiB after loading; highest; left of 6,068 | 5,931; 5,993; 75 |
| host lowest MemAvailable MiB; swap used at the start and its highest, MiB | 289; 27 and 27 |
| load: server start to the client's first answer | 3 min 46 s (20:47:07 to 20:50:53) |

## The checks of the line

- **The dry line** (`DRY=1`, nothing started): the preset's 40 switches and its arguments with the harness's
  `--metrics --no-webui` and `-md <draft>` after `-lm dio`; 1,726 bytes. [run/preset_check.sh](run/preset_check.sh)
  (R243's word-by-word check without its 65,536-token arms) passed: the environment equals `PRESET_ENV` byte for byte,
  and the 30 flags are `PRESET_ARGS` plus those three, each with its value. Against R244's started 131,072-token line
  ([lines/l131.txt](../2026-11-26_r244_flash_ctx_ub/lines/l131.txt)) it differs in one word,
  `_MIN_RATIO=0.45` for `0.5`.
- **The started line:** the harness's `starting pre on the rig:` line equals the dry line byte for byte
  ([lines/pre.txt](lines/pre.txt), with the build and draft paths as `<build>` and `<draft>`).
- **Read back from `/proc` on the rig** once the server answered: its command line is the dry line's binary and
  arguments (55 words), followed by `-m <model> --host 127.0.0.1 --port <port>`, which
  [rig_server.sh](../../../bench/agent_check/rig_server.sh) adds to every arm; its `GGML_*`, `LLAMA_*` and `CUDA_*`
  variables are exactly the dry line's 40 switches with the same values.
- The server log's taper line reads `short=1400:64,cuts=1 (n_ubatch 256)` ([taper_lines.tsv](taper_lines.tsv)); no
  graph reallocation in any request ([reallocs.tsv](reallocs.tsv)); every sampled token under the tool-call grammar,
  none rejected ([grammar.tsv](grammar.tsv)); no reply read again ([rereads_runs.tsv](rereads_runs.tsv)).

## Next to earlier runs (no class)

One run on rig 2 at stock clocks. The nearest runs differ in rig or clocks, so the values below are listed side by
side, not classed:
- **R243's `l131a` and `l131b`:** the same context and micro-batch, before 0.45 (`_MIN_RATIO=0.5`), and the taper
  without `cuts=1`; rig 1 at stock clocks (busy SM clock 1,875.6 and 1,874.1 MHz, mean of the cards).
- **C1175's runs:** `-c 65536 -ub 512`; rig 2 at its soaked clock profile (busy SM clock 2,128.4, 2,128.4, 2,127.0 and
  2,124.2 MHz), so their decode steps are not this run's: `ca` and `cb` the preset of that morning (`_NSM=5`,
  `_MIN_RATIO=0.75`), `r05a` and `r05b` the arm that became the preset (`_NSM=10`, `_MIN_RATIO=0.5`).
- This run's busy SM clock: 1,876.8 MHz ([clocks.tsv](clocks.tsv)).

| | `pre` | R243 `l131a` | R243 `l131b` | C1175 `ca` | C1175 `cb` | C1175 `r05a` | C1175 `r05b` |
|---|---|---|---|---|---|---|---|
| tasks passed | 12/12 | 12/12 | 11/12 | 11/12 | 12/12 | 12/12 | 12/12 |
| minutes a task at equal work | 2.33 | 2.31 | 2.38 | 2.29 | 2.23 | 2.23 | 2.21 |
| generation tok/s | 28.22 | 29.05 | 27.02 | 29.05 | 31.56 | 31.54 | 32.04 |
| ms a decode step, all requests | 115.88 | 113.30 | 115.04 | 109.42 | 107.10 | 104.05 | 104.26 |
| first request after loading, ms | 19,801.1 | 19,480.4 | 19,460.9 | | | | |
| card 8 highest, nvidia-smi MiB (with the 75 MiB the card log reads idle) | 6,068 | 6,072 | 6,074 | 5,698 | 5,698 | 5,694 | 5,700 |
| card 8 left of 6,068 at its highest, MiB | 75 | 71 | 69 | | | | |
| host lowest MemAvailable MiB | 289 | 267 | 232 | 294 | 228 | 316 | 224 |
| swap at the start; highest, MiB | 27; 27 | 22; 22 | 22; 22 | 51; 997 | 52; 52 | 52; 398 | 52; 52 |

R243's figures are from its [agent.tsv](../2026-11-26_r243_flash_131k_agent_rig1/agent.tsv),
[steps_cells.md](../2026-11-26_r243_flash_131k_agent_rig1/steps_cells.md),
[card_memory.tsv](../2026-11-26_r243_flash_131k_agent_rig1/card_memory.tsv) and
[host.tsv](../2026-11-26_r243_flash_131k_agent_rig1/host.tsv); C1175's from its
[agent.tsv](../2026-11-26_c1175_nsm10_rig2/agent.tsv), [steps_cells.md](../2026-11-26_c1175_nsm10_rig2/steps_cells.md)
and [host.tsv](../2026-11-26_c1175_nsm10_rig2/host.tsv).

### The decode step by context

In C1175's buckets ([steps_cells.md](steps_cells.md); a request counts in the bucket of its middle context), ms a step:

| context at the request's middle | `pre` | R243 `l131a` | R243 `l131b` | C1175 `r05a` | C1175 `r05b` |
|---|---|---|---|---|---|
| 0 to 4,102 | 115.17 (22 requests) | 113.47 | 113.95 | 104.59 | 103.93 |
| 4,102 to 6,153 | 115.70 (26) | 112.80 | 113.40 | 103.62 | 104.05 |
| 6,153 to 8,204 | 117.20 (17) | 113.95 | 112.67 | 105.64 | 103.48 |
| 8,204 to 16,000 | (no request) | 113.66 | 115.27 | 105.39 | 106.19 |

Cut where 0.45 acts ([steps_views.md](steps_views.md), [run/steps_views.py](run/steps_views.py)): by R245's view rule a
four-token verify runs the sparse kernel from 3,585 cells at 0.45 and from 4,097 at 0.5, so only the bucket from 3,585 to
4,096 differs between this run's preset and R243's line (each bucket below includes its lower end, not its upper):

| context at the request's middle | `pre` | R243 `l131a` | R243 `l131b` |
|---|---|---|---|
| 0 to 3,585 (both on the dense kernels) | 115.04 (16 requests) | 114.18 (13) | 113.78 (14) |
| 3,585 to 4,097 (only 0.45 on the sparse kernel) | 115.27 (6) | 113.18 (5) | 114.08 (5) |
| 4,097 to 6,153 (both on the sparse kernel) | 115.70 (26) | 112.80 (30) | 113.40 (26) |

The step is longer than on rig 1 also below 3,585 cells, where 0.45 changes nothing (115.04 against 114.18 and 113.78),
so this run cannot tell the switch from the rig. Row Q1935's agent pair (0.45 against 0.5 on one rig and boot) is the
measurement of 0.45.

## Files

| File | What it holds |
|---|---|
| [agent/pre/](agent/pre/) | the run's `turns.csv` and `attempts.csv` (the client's `res/`; the harness's arm is the run's own name) |
| [agent.tsv](agent.tsv) | the agent table: passed, turns, prompt and generation tok/s, first token, acceptance, minutes, minutes a task at equal work (priced on R90's v3, as in R243 and C1175) |
| [pre_report_table.md](pre_report_table.md), [pre_compare.md](pre_compare.md), [pre_gonogo.md](pre_gonogo.md) | the harness's own report |
| [steps.md](steps.md), [steps_cells.md](steps_cells.md), [steps_views.md](steps_views.md) | ms a decode step by context: R203's buckets, C1175's, and cut at 3,585 and 4,097 cells (with R243's `l131a` and `l131b`) |
| [card_memory.tsv](card_memory.tsv) | each card's MiB after loading and at its highest, without the card log's idle reading, and what is left of 6,068 |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, each card's raw highest |
| [prompt_buckets.tsv](prompt_buckets.tsv), [prompt_tasks.tsv](prompt_tasks.tsv), [first_token.tsv](first_token.tsv) | prompt reading by new tokens and by task, the first-token waits |
| [work.md](work.md), [work.tsv](work.tsv), [same_prompts.md](same_prompts.md) | the work read and generated |
| [clocks.tsv](clocks.tsv), [reallocs.tsv](reallocs.tsv), [grammar.tsv](grammar.tsv), [slot_gap.tsv](slot_gap.tsv), [taper_lines.tsv](taper_lines.tsv), [text_reuse_points.tsv](text_reuse_points.tsv), [rereads_runs.tsv](rereads_runs.tsv), [rereads_turns.tsv](rereads_turns.tsv), [capped.tsv](capped.tsv) | busy clocks, graph reallocations, grammar sampling, slot gaps, the taper line, text reuse, reply re-reads, capped tasks (none) |
| [lines/pre.txt](lines/pre.txt) | the started server line |
| [run/r248.sh](run/r248.sh) | the run as it ran, with the rig, model folder, build and work folder from the environment |
| [run/preset_check.sh](run/preset_check.sh) | the word-by-word check of a line against the preset file |
| [run/tables_agent.sh](run/tables_agent.sh), [run/card_memory.py](run/card_memory.py), [run/steps_views.py](run/steps_views.py) | the tables (R243's scripts with this folder; the view-cut step table) |

## Commands

From the repository root at `fa31e1b8` with `preset_agent.sh`'s `files` fix (this folder's commit), the machine that
runs it being the client (Docker, `agent-check:0.87.1`):

```bash
DRY=1 RIG=<rig> MODELS=<rig's Flash-Next folder> NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 WORK=<folder> \
  bash docs/data/2026-11-26_r248_preset_smoke/run/r248.sh
bash docs/data/2026-11-26_r248_preset_smoke/run/preset_check.sh <folder>/out/pre/dry_line.txt
RIG=<rig> MODELS=<rig's Flash-Next folder> NEW=llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674 WORK=<folder> \
  bash docs/data/2026-11-26_r248_preset_smoke/run/r248.sh
bash docs/data/2026-11-26_r248_preset_smoke/run/tables_agent.sh <tables> pre
```

`<tables>` holds the client's `rig/server-pre.log`, `rig/meminfo.log` and `rig/cards.csv` (as `rig/` and `rig_pre/`),
`proxy.jsonl` with its bodies (`proxy_pre/`), the task window (`window_pre`) and R90's server log, as R243's
[run/tables_agent.sh](../2026-11-26_r243_flash_131k_agent_rig1/run/tables_agent.sh) lists them.

## Provenance

- **Rig:** nine P106-090 6 GB cards (6,144 MiB each, 76 MiB reserved by the driver: 6,068 usable), stock clocks (reset
  20:32), boot 09:22, driver 580; no GPU Xid before or after; at the end no llama process, 0 MiB on every card, the rig's
  log folder removed.
- **Engine:** Miner Llama's v5 candidate tree [`eee615cdb`](https://github.com/Cinderella-Man/miner-llama/commit/eee615cdb)
  (llama-server md5 `2727faf28b5b34dfe42399a9d431da67`, checked before the run).
- **Model:** Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's base file, Swift GSQ-RCO Q2_0 (three shards), with its MTP draft
  `mtp-Qwen3.8-Flash-Next-draft.gguf`; the preset's own files.
- **Run:** 2026-11-26, server started 20:47:07, tasks 20:50:55 to 21:06:17, driver exit 0 at 21:06:31.
- **Licence:** the tables and scripts are this repository's; the task set is the agent check's bugfix set (task IDs only).
