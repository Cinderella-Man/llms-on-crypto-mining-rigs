# Flint's release agent check: the v4 preset against v3.0 on Flint Pascal Q8 GGUF (R159, Q1478; 2026-11-14)

R159 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 40-minute budget an arm) on Flint Pascal Q8 GGUF
(p065, `flint-near-q8-p065.gguf`, 40,431,236,544 bytes), the headline Flint file, at 262,144 tokens on nine P106-090
cards. It compares two arms, each run twice, one load a run, alternated: **R159a `f4a`, R159b `f3a`, R159c `f4b`,
R159d `f3b`**.
- `f4*` is the v4 preset ([release/flint-v4/settings.env](../../../release/flint-v4/settings.env): v3.0's line and
  three switches) on the release candidate `v4-rc2` (`pascal-p106-v4` `eb82916d3`, upstream b12360; the rig tree's
  `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`).
- `f3*` is the v3.0 release (`miner-llama-v3.0`, `1551f5fe1`, `pascal-p106-v3-b12076-ladder5`; md5
  `c735e40f184f3de61f16ffd5e2ab5ea5`) with [release/flint-v3/settings.env](../../../release/flint-v3/settings.env) as
  shipped (no switch).

Registry row Q1478 in [quants/experiments.md](../../../quants/experiments.md). This folder fills as the runs land; the
report over the four runs and the verdict come after R159d. Until then each run stands alone and is not classed.

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R159a | `f4a` | rig 1 | 12:09:16, up after 586 s | 12:19:05–12:50:32 | 12 / 12 | 401 bytes, md5 `46991da4e7acc094a91dec01c16252d3` |
| R159b | `f3a` | rig 1 | 14:05:56, up after 586 s | 14:15:45–14:40:37 | 11 / 12 | 331 bytes, md5 `fc324d8e91ff8ea133f6f7a6b9ffc05a` |
| R159c | `f4b` | rig 1 | 14:51:45, up after 587 s | 15:01:35–15:30:55 | 12 / 12 | 401 bytes, md5 `46991da4e7acc094a91dec01c16252d3` |

## R159a: `f4a`, the v4 preset (2026-11-14)

- **The started command** (the harness's `COMMON` line, then the arm's flags, with the preset's three switches in
  front):

  ```
  LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 <v4-rc2>/bin/llama-server -ngl 99 -fa on
  -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95
  --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4
  ```

  Every flag of the settings file's `PRESET_ARGS` is in it. The server's `switches:` line, read during the load,
  says `3 set: LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`.
- **The client** ran on the machine that drives the rig (`CLIENT=local`, Docker, CPU only, `--cpu-shares 256`; the
  sandbox image `agent-check:0.87.1`, prepared and validated 12 of 12 by C842, Q1485).

**Numbers** (one run; the reference column is v3.0's earlier run on the same file and line, arm `v3base` of
[2026-11-05_release_regression](../2026-11-05_release_regression/README.md), 2026-11-06, on `1551f5fe1`; R159's own
v3.0 runs are R159b and R159d):

| | v3.0, `v3base` (2026-11-06) | **R159a `f4a`** |
|---|---|---|
| passed | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`v3base`'s work) | 1.86 | **1.85** |
| minutes, all 12 tasks (wall) | 22.3 | **31.3** |
| minutes a task, median | 1.03 | **1.21** |
| turns | 78 | **73** |
| generation tok/s | 47.38 | **47.45** |
| prompt tok/s | 175.2 | **176.9** |
| first token median, p90 s | 2.66, 6.75 | **2.85, 7.84** |
| first token on a task's first turn, median s | 4.23 | **4.29** |
| tool calls / failed / malformed | 74 / 2 / 0 | **70 / 1 / 0** |
| graph reallocations (target) | not counted | **323 with the graph's size unchanged, 0 changed, in 71 of 73 requests** |
| lowest MemAvailable, swap | | **885 MiB at 12:40:35; swap 30 → 295 MiB** |

- "Equal work" means `v3base`'s work (its prompt and generated tokens, and its time outside the server) at each arm's
  own prompt and generation speeds ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py), [agent.tsv](agent.tsv)).
  It compares the engines on the same work. The wall minutes for all tasks depend on what each attempt generated:
  `f4a`'s Python `scale-generator` took 11.5 minutes in 9 turns, 28 s under the cap, and passed; Python `pig-latin`
  took 4.4 minutes in 11 turns.
- **Graph reallocations** ([f4a_reallocs.tsv](f4a_reallocs.tsv), per request in
  [f4a_requests_reallocs.tsv](f4a_requests_reallocs.tsv)): under 512 prompt tokens 67 in 44 of 45 requests;
  512–1,023, 46 in 12 of 12; 1,024–1,999, 53 in 9 of 9; 2,000 and more, 157 in 6 of 7. Both builds log the count
  (`graph reallocs = S / D`), v3.0's included (R159b, below). R161's deep lane on the same preset counted none
  ([2026-11-14_r161_flint_release_deep](../2026-11-14_r161_flint_release_deep/README.md)).
  `release/flint-v4/README.md`'s candidate `LLAMA_SCHED_PER_SIZE_RESERVE=2` was to be tried only if an agent report
  counts reallocations above 0, and this one does.
- **Host and cards** ([host.tsv](host.tsv)): swap grew from 30 to 295 MiB during the load (its highest at 12:18:53,
  before the first task) and stayed flat; MemAvailable was lowest at 885 MiB (12:40:35) and 1,775 MiB at the end. The
  report's memory gate says NO-GO on that swap growth (it allows 16 MiB); MemAvailable stayed above its 600 MB floor.
  The RAM watchdog did not stop the server. Peak MiB per card: 5,413 5,951 5,107 5,617 5,951 5,107 5,571 5,321 4,833.
- **Server log:** 16 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([f4a_gonogo.md](f4a_gonogo.md)).
- The report's files: [f4a_compare.md](f4a_compare.md), [f4a_report_table.md](f4a_report_table.md),
  [f4a_gonogo.md](f4a_gonogo.md); the per-request and per-attempt rows: [agent/f4a/](agent/f4a/).

## R159b: `f3a`, v3.0 (2026-11-14)

- **The started command** (no switch in front: `release/flint-v3/settings.env` sets `PRESET_ENV=""`):

  ```
  <v3.0>/bin/llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
  --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512
  -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4
  ```

  Every flag of the settings file's `PRESET_ARGS` is in it. v3.0's build prints no `switches:` line, so the running
  server's environment was read during the load instead: no `LLAMA_*` or `GGML_*` variable set.
- This arm is the same build, file and line as the reference column `v3base`, run eight days later.

**Numbers** (one run; the same reference column as R159a):

| | v3.0, `v3base` (2026-11-06) | R159a `f4a` | **R159b `f3a`** |
|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **11 / 12** |
| minutes a task at equal work (`v3base`'s work) | 1.86 | 1.85 | **1.81** |
| minutes, all 12 tasks (wall) | 22.3 | 31.3 | **24.7** |
| minutes a task, median | 1.03 | 1.21 | **1.17** |
| turns | 78 | 73 | **66** |
| generation tok/s | 47.38 | 47.45 | **48.55** |
| prompt tok/s | 175.2 | 176.9 | **179.2** |
| first token median, p90 s | 2.66, 6.75 | 2.85, 7.84 | **2.65, 7.20** |
| first token on a task's first turn, median s | 4.23 | 4.29 | **4.17** |
| tool calls / failed / malformed | 74 / 2 / 0 | 70 / 1 / 0 | **57 / 0 / 0** |
| graph reallocations (target) | not counted | 323 same size, 0 changed, in 71 of 73 requests | **188 same size, 0 changed, in 64 of 65 requests** |
| lowest MemAvailable, swap | | 885 MiB at 12:40:35; swap 30 → 295 MiB | **948 MiB at 14:39:09; swap 31 → 293 MiB** |

- **The failed task:** Python `scale-generator` reached the 12-minute cap in 4 turns (the report's "runaway (time)").
  Its fourth reply started after 3.0 s and was still generating when the cap stopped the task, 652.9 s into the
  request. The server never logged that request's timing, so it counts in none of the speeds (est. ~31,000 tokens at
  48.55 tok/s). In R159a the same task passed at 11.5 minutes. Every other task passed: Python `pig-latin` 2.3
  minutes, `list-ops` 1.7, `affine-cipher` 1.7; the rest under 1.5 minutes each.
- **Graph reallocations** ([f3a_reallocs.tsv](f3a_reallocs.tsv), per request in
  [f3a_requests_reallocs.tsv](f3a_requests_reallocs.tsv)): under 512 prompt tokens 61 in 41 of 41 requests;
  512–1,023, 44 in 11 of 11; 1,024–1,999, 70 in 11 of 11; 2,000 and more, 13 in 1 of 2. Bucket by bucket
  against `f4a`: under 512 tokens 61 in 41 requests against 67 in 45, 512–1,023 44 in 11 against 46 in 12,
  1,024–1,999 70 in 11 against 53 in 9, 2,000 and more 13 in 2 against 157 in 7.
- **Host and cards** ([host.tsv](host.tsv), made by [run/host_summary.py](run/host_summary.py)): swap grew from 31 to
  293 MiB during the load (its highest at 14:15:31, before the first task at 14:15:45);
  MemAvailable was lowest at 948 MiB (14:39:09) and 1,426 MiB at the end. Peak MiB per card: 5,413 5,953 5,109 5,619
  5,953 5,109 5,573 5,323 4,837. The RAM watchdog did not stop the server.
- **The report's go/no-go lines** ([f3a_gonogo.md](f3a_gonogo.md)) read the client folder that R159a–d share, where
  the host logs of each run are appended and the attempts of each run are kept: its harness and cache lines count both
  runs' 24 attempts, and its memory line (885 MB, swap 30 → 295 MB) is R159a's run. This run's own host numbers are
  the ones above, and its swap growth is over the gate's 16 MiB too. The proxy-rows line says NO-GO for the capped
  attempt alone (4 proxy rows against 3 replies; its last row has no timings).
- **Server log:** 13 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill.
- The report's files: [f3a_compare.md](f3a_compare.md), [f3a_report_table.md](f3a_report_table.md) (its table lists
  `f4a` beside `f3a`), [f3a_gonogo.md](f3a_gonogo.md); the per-request and per-attempt rows: [agent/f3a/](agent/f3a/).

## R159c: `f4b`, the v4 preset's second run (2026-11-14)

- **The started command** is R159a's, byte for byte (401 bytes, the same md5): the preset's three switches, then the
  harness's `COMMON` line and the arm's flags. The server's `switches:` line, read during the load, says
  `3 set: LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SERVER_YIELD=2`.

**Numbers** (one run; from the harness's own report and the rig's memory log):

| | R159a `f4a` | R159b `f3a` | **R159c `f4b`** |
|---|---|---|---|
| passed | 12 / 12 | 11 / 12 | **12 / 12** |
| minutes a task at equal work (`v3base`'s work) | 1.85 | 1.81 | **not yet tabled** |
| minutes, all 12 tasks (wall) | 31.3 | 24.7 | **29.2** |
| turns | 73 | 66 | **80** |
| generation tok/s (the report's rounding) | 47.4 | 48.5 | **48.2** |
| prompt tok/s (the report's rounding) | 177 | 179 | **179** |
| first token median, p90 s | 2.85, 7.84 | 2.65, 7.20 | **2.56, 6.76** |
| first token on a task's first turn, median s | 4.29 | 4.17 | **4.16** |
| tool calls / failed / malformed | 70 / 1 / 0 | 57 / 0 / 0 | **72 / 0 / 0** |
| graph reallocations (target) | 323 same size, 0 changed, in 71 of 73 requests | 188 same size, 0 changed, in 64 of 65 requests | **308 same size, 0 changed, in 79 of 80 requests** |
| lowest MemAvailable, swap | 885 MiB at 12:40:35; swap 30 → 295 MiB | 948 MiB at 14:39:09; swap 31 → 293 MiB | **873 MiB at 15:30:47; swap 32 → 296 MiB** |

- The report prints generation tok/s to one decimal and prompt tok/s to whole numbers; R159a's and R159b's columns
  above are the same report's figures for them (their two-decimal figures, 47.45 and 48.55, are in
  [agent.tsv](agent.tsv)). This run's two-decimal speeds, its minutes a task at equal work, its reallocations per
  bucket of prompt tokens and its card peaks come from the tables of [How to run it](#how-to-run-it)
  (`agent_tables.py`, `realloc_counts.py`, `host_summary.py`), which were not run with this run; they are added to
  [agent.tsv](agent.tsv) and [host.tsv](host.tsv) with them.
- **Long tasks:** JavaScript `variable-length-quantity` 8.7 minutes in 13 turns, Python `scale-generator` 7.8 minutes
  in 9 turns (it passed; R159b's `f3a` reached the cap on it, R159a's `f4a` passed at 11.5 minutes), Python
  `pig-latin` 4.0 minutes in 10 turns, Python `list-ops` 2.1 minutes. Every other task took under 2 minutes.
- **Host** (the rig's `meminfo.log`, this run's lines only): swap grew from 32 to 296 MiB during the load (its highest
  at 15:01:32, before the first task at 15:01:35); MemAvailable was lowest at 873 MiB (15:30:47, in the last task)
  and 1,388 MiB at the end. The RAM watchdog did not stop the server.
- **The report's go/no-go lines** ([f4b_gonogo.md](f4b_gonogo.md)) read the shared client folder, as for R159b: its
  harness and cache lines count the three runs' 36 attempts, the proxy-rows NO-GO is R159b's capped attempt, and its
  memory line's 873 MB is this run's low (below R159a's and R159b's) with R159a's starting swap of 30 MB.
- **Server log:** 14 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill.
- The report's files: [f4b_compare.md](f4b_compare.md), [f4b_report_table.md](f4b_report_table.md) (its table lists
  `f3a` and `f4a` beside `f4b`), [f4b_gonogo.md](f4b_gonogo.md); the per-request and per-attempt rows:
  [agent/f4b/](agent/f4b/).

## How to run it

[run/r159.sh](run/r159.sh) runs one arm. It reads the settings file of the arm's version and stops if its
`PRESET_ARGS` is not v3.0's Flint line. Run it from the repository's root on the machine that drives the rig; the
machine-specific values go in the environment:

```bash
ARM=f4a RIG=<rig alias> CLIENT=local MODEL=<rig path>/flint-near-q8-p065.gguf \
  BIN_V4=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  BIN_V3=<rig path>/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r159 OUT=<out>/agent-f4a \
  bash docs/data/2026-11-14_r159_flint_release_agent/run/r159.sh
```

R159a was started with the same call written out (the harness
[release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) with `STEPS=run`, `ARGS_f4a="--alias
flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4"`, `BUDGET_MIN_f4a=40`, `REL_ENV` the preset's
`PRESET_ENV`). The run's own report came from the same environment with `STEPS="report collect"`. The equal-work
table: `agent_tables.py OUT v3p065 f4a` over a folder holding `v3base`'s `turns.csv` and `attempts.csv` (its arm
`rel` renamed `v3p065`) beside `f4a`'s, and `f4a`'s server log; the reallocations: `scripts/realloc_counts.py
f4a=server-f4a.log` (default and `--requests`). Both ran in the sandbox image.

R159b ran the same call with arm `f3a` (`. release/flint-v3/settings.env`, `REL_BIN` the v3.0 tree's
`llama-server`, `ARGS_f3a` and `BUDGET_MIN_f3a=40` as above), then `STEPS="report collect"` with `ARMS=f3a`. Its
tables: `agent_tables.py OUT v3p065 f4a f3a` over the same folder with `f3a`'s rows added (so [agent.tsv](agent.tsv)
holds all three arms at `v3base`'s work), `realloc_counts.py f3a=server-f3a.log`, and
`host_summary.py RIGDIR R159b f3a` ([run/host_summary.py](run/host_summary.py): one [host.tsv](host.tsv) row from the
rig's `meminfo.log` and `gpu.csv`, reading only the last run where the shared client folder appended several; on
R159a's logs it gives R159a's row exactly), all three in the sandbox image.

R159c ran R159a's call with arm `f4b` (`. release/flint-v4/settings.env`, `REL_BIN` the `v4-rc2` tree's
`llama-server`, `ARGS_f4b` and `BUDGET_MIN_f4b=40` as above), then `STEPS="report collect"` with `ARMS=f4b`. Its
tables are `agent_tables.py OUT v3p065 f4a f3a f4b` over the same folder with `f4b`'s rows added,
`realloc_counts.py f4b=server-f4b.log` (default and `--requests`) and `host_summary.py RIGDIR R159c f4b`.
