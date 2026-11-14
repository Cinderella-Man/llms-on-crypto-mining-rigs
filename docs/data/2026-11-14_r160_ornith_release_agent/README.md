# Ornith's release agent check: the v4 preset, the v4 stack and v3.0 on Ornith-1.5-122B Pascal N2P GGUF (R160, Q1479; 2026-11-14)

R160 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md): 12 bug-fix
tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget an arm, the makers' sampling at
temperature 0.6) on Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`,
48,196,722,592 bytes) at 49,152 tokens on nine P106-090 cards. It compares three arms, each run twice, one load a
run, alternated: **R160a `o4a`, R160b `o4sa`, R160c `o3a`, R160d `o4b`, R160e `o4sb`, R160f `o3b`**.
- `o4*` is the v4 preset ([release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env): v3.0's line, the
  keep-alive and three switches) on the release candidate `v4-rc2` (`pascal-p106-v4` `eb82916d3`, upstream b12360;
  the rig tree's `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`).
- `o4s*` is `o4` plus the candidate stack of R162's arm `v4s`
  ([2026-11-14_r162_ornith_release_deep](../2026-11-14_r162_ornith_release_deep/README.md)): `GGML_CUDA_HOP_STAGED=1
  GGML_CUDA_GRAPHS_FORCE=1`, `--ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64` and
  `-b 4096` in place of `-b 2048`.
- `o3*` is the v3.0 release (`miner-llama-v3.0`, `1551f5fe1`, `pascal-p106-v3-b12076-ladder5`) with
  [release/ornith-v3/settings.env](../../../release/ornith-v3/settings.env) as shipped (the keep-alive).

Registry row Q1479 in [quants/experiments.md](../../../quants/experiments.md). This folder fills as the runs land; the
report over the six runs and the verdict come after R160f. Until then each run stands alone and is not classed.

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R160a | `o4a` | rig 2 | 14:05:46, up after 651 s | 14:16:40–14:46:29 | 12 / 12 | 423 bytes, md5 `7b16f13d060c8b1fde8f9af58493226a` |
| R160b | `o4sa` | rig 2 | 14:55:39, up after 651 s | 15:06:32–15:34:18 | 12 / 12 | 549 bytes, md5 `cccb6a6fb839fc14b24f91c63dae8e17` |
| R160c | `o3a` | rig 2 | 16:22:21, up after 651 s | 16:33:16–17:09:13 | 12 / 12 | 349 bytes, md5 `d9e48849bb95aea7d055a6f0ed35c34a` |
| R160d | `o4b` | rig 2 | 17:17:31, up after 651 s | 17:28:25–17:59:33 | 12 / 12 | 423 bytes, md5 `7b16f13d060c8b1fde8f9af58493226a` |

## R160a: `o4a`, the v4 preset (2026-11-14)

- **The started command** (the harness's `COMMON` line, then the arm's flags, with the preset's four switches in
  front; the arm's `--temp 0.6` comes after `COMMON`'s `--temp 1.0` and wins, and every request carries temperature
  0.6, top-p 0.95, top-k 20, min-p 0):

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
  <v4-rc2>/bin/llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
  --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6
  -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off
  ```

  Every flag of the settings file's `PRESET_ARGS` is in it. The server's `switches:` line, read during the load,
  says `4 set: GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1
  LLAMA_SERVER_YIELD=2`.
- **The client** ran on the machine that drives the rig (`CLIENT=local`, Docker, CPU only, `--cpu-shares 256`; the
  sandbox image `agent-check:0.87.1`, prepared and validated 12 of 12 by C842, Q1485). Flint's R159 ran its own client
  on the same machine at the same time, with its own ports, arm names and results folder.

**Numbers** (one run; the reference column is v3.0's earlier run on the same rig, file and line, arm `ka` of
[2026-11-06_ornith_mtp_rig2](../2026-11-06_ornith_mtp_rig2/README.md), on `1551f5fe1` with the keep-alive, renamed
`v3ka` in [agent.tsv](agent.tsv); R160's own v3.0 runs are R160c and R160f):

| | v3.0, `ka` (2026-11-06, the same rig) | **R160a `o4a`** |
|---|---|---|
| passed | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.52 | **2.57** |
| minutes, all 12 tasks (wall) | 30.2 | **29.7** |
| minutes a task, median | 2.11 | **2.07** |
| turns | 63 | **62** |
| generation tok/s | 16.42 | **16.53** |
| prompt tok/s | 60.1 | **56.6** |
| first token median, p90 s | 9.34, 19.86 | **8.65, 19.22** |
| first token on a task's first turn, median s | 16.39 | **16.32** |
| tool calls / failed / malformed | 63 / 1 / 0 | **62 / 0 / 0** |
| graph reallocations (target) | not counted (its server log is not in its folder) | **0 in 62 requests** |
| lowest MemAvailable, swap | | **803 MiB at 14:24:35; swap 29 → 336 MiB** |

- "Equal work" means `ka`'s work (its prompt and generated tokens, and its time outside the server) at each arm's own
  prompt and generation speeds ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py), [agent.tsv](agent.tsv)).
  It compares the engines on the same work. `o4a` generates faster than `ka` (16.53 against 16.42 tok/s) and reads
  prompts slower (56.6 against 60.1), so `ka`'s work takes 2.57 minutes a task against 2.52. The wall minutes for all
  tasks depend on what each attempt generated: `o4a`'s Python `scale-generator` took 5.6 minutes in 5 turns, Python
  `pig-latin` 4.5 minutes in 7 turns and Python `affine-cipher` 3.3 minutes in 7 turns; every other task took under 3
  minutes.
- **Graph reallocations** ([o4a_reallocs.tsv](o4a_reallocs.tsv), per request in
  [o4a_requests_reallocs.tsv](o4a_requests_reallocs.tsv)): none in any prompt-size bucket (38 requests under 512
  prompt tokens, 11 at 512–1,023, 10 at 1,024–1,999, 3 at 2,000 and more). Flint's R159a on the same build counted
  323 in 71 of 73 requests. v3.0's build logs the count as well (`graph reallocs = S / D`,
  [R159b](../2026-11-14_r159_flint_release_agent/README.md)), so R160c and R160f count `o3*`'s.
  `release/ornith-v4/README.md`'s candidate `LLAMA_SCHED_PER_SIZE_RESERVE=2` is to be tried
  only if an agent report counts reallocations above 0, and this one does not.
- **Host and cards** ([host.tsv](host.tsv), from the rig's `meminfo.log` and `gpu.csv` by [run/host.py](run/host.py)):
  swap grew from 29 to 336 MiB during the load (its highest at 14:10:07, before the first task) and was 50 MiB when the
  log ended, after the server stopped. MemAvailable was lowest at 803 MiB (14:24:35). The report's memory gate says
  NO-GO on that swap growth (it allows 16 MiB; R159a's Flint run says the same); MemAvailable stayed above its 600 MB
  floor. The RAM watchdog did not stop the server. Peak MiB per card: 5,665 5,697 5,611 5,697 5,307 5,415 5,575 5,663
  5,069.
- **Server log:** 13 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4a_gonogo.md](o4a_gonogo.md)).
- The report's files: [o4a_compare.md](o4a_compare.md), [o4a_report_table.md](o4a_report_table.md),
  [o4a_gonogo.md](o4a_gonogo.md); the per-request and per-attempt rows: [agent/o4a/](agent/o4a/).

## R160b: `o4sa`, the v4 preset plus the candidate stack (2026-11-14)

- **The started command** (as R160a's, with the stack's two switches after the preset's four, `-b 4096` in place of
  `-b 2048`, and the stack's three flags at the end):

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
  GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1
  <v4-rc2>/bin/llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
  --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6
  -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  --ubatch-taper short=600:64
  ```

  It holds one `-b`, 4096. The server's `switches:` line, read during the load, says `6 set: GGML_CUDA_GRAPHS_FORCE=1
  GGML_CUDA_HOP_STAGED=1 GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1
  LLAMA_SERVER_YIELD=2`. This is the line of R162's arm `v4s`, whose fit load kept the whole stack.

**Numbers** (one run each; the reference is v3.0's run `ka` as for R160a):

| | v3.0, `ka` (2026-11-06, the same rig) | R160a `o4a` | **R160b `o4sa`** |
|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.52 | 2.57 | **2.08** |
| minutes, all 12 tasks (wall) | 30.2 | 29.7 | **27.6** |
| minutes a task, median | 2.11 | 2.07 | **1.94** |
| turns | 63 | 62 | **63** |
| generation tok/s | 16.42 | 16.53 | **16.30** |
| prompt tok/s | 60.1 | 56.6 | **114.5** |
| first token median, p90 s | 9.34, 19.86 | 8.65, 19.22 | **3.05, 10.97** |
| first token on a task's first turn, median s | 16.39 | 16.32 | **8.45** |
| tool calls / failed / malformed | 63 / 1 / 0 | 62 / 0 / 0 | **64 / 0 / 0** |
| graph reallocations (target) | not counted | 0 in 62 requests | **0 in 63 requests** |
| lowest MemAvailable, swap | | 803 MiB at 14:24:35; swap 29 → 336 MiB | **1,252 MiB at 15:28:56; swap 30 → 335 MiB** |

- **Equal work:** `ka`'s work at `o4sa`'s speeds takes 2.08 minutes a task, against 2.57 at `o4a`'s (−19.1 %) and
  2.52 at `ka`'s (−17.5 %). `o4sa` read prompts at 114.5 tok/s against `o4a`'s 56.6 (+102.3 %) and generated at
  16.30 against 16.53 tok/s (−1.4 %). An agent's prompts are short: a task's first two turns process a median 789 and
  1,452 new tokens, and later turns 128 or fewer. R162's deep lane measured the stack's largest prompt gain on its
  shortest prompt (+65.1 % at 64 tokens, +26.4 % at 4,096).
- **Wall minutes** depend on what each attempt generated: `o4sa`'s attempts generated 20,914 tokens in all, `o4a`'s
  18,526. Python `scale-generator` took 8.6 minutes in 7 turns (`o4a`: 5.6 in 5) and Python `affine-cipher` 3.7 in 9;
  every other task took 2.2 minutes or less.
- **Graph reallocations** ([o4sa_reallocs.tsv](o4sa_reallocs.tsv), per request in
  [o4sa_requests_reallocs.tsv](o4sa_requests_reallocs.tsv)): none in any bucket (39 requests under 512 prompt tokens,
  9 at 512–1,023, 11 at 1,024–1,999, 4 at 2,000 and more): `-b 4096` and the taper add none.
- **Host and cards** ([host.tsv](host.tsv), R160b's rows of the rig's `meminfo.log` and `gpu.csv` from 14:55:40):
  swap grew from 30 to 335 MiB during the load (its highest at 15:02:36, before the first task). MemAvailable was
  lowest at 1,252 MiB (15:28:56), against `o4a`'s 803: the checkpoints sit on the cards (R162's `v4s`: 1,238 and
  1,254 MiB against `v4`'s 651 and 655). Peak MiB per card: 5,757 5,773 5,703 5,773 5,381 5,489 5,649 5,721 5,129, of
  6,144. The RAM watchdog did not stop the server. The report's memory gate in [o4sa_gonogo.md](o4sa_gonogo.md) reads
  the client's `meminfo.log`, which holds R160a's rows before R160b's, so its "lowest MemAvailable 804 MB; swap used
  29 → 337 MB" is R160a's load; on R160b's rows alone the swap growth of 305 MiB is also above the gate's 16 MiB.
- **Server log:** 12 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4sa_gonogo.md](o4sa_gonogo.md)).
- The report's files: [o4sa_compare.md](o4sa_compare.md), [o4sa_report_table.md](o4sa_report_table.md) (both runs'
  attempts), [o4sa_gonogo.md](o4sa_gonogo.md); the per-request and per-attempt rows: [agent/o4sa/](agent/o4sa/).

## R160c: `o3a`, the v3.0 release on its preset (2026-11-14)

- **The started command** (the harness's `COMMON` line, then the arm's flags, with the preset's one switch in front):

  ```
  GGML_CUDA_KEEPALIVE=decode
  <v3.0>/bin/llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja
  --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6
  -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off
  ```

  Every flag of the settings file's `PRESET_ARGS` is in it. The v3.0 build prints no `switches:` line (v4 added it),
  so the one switch is seen in the started command only.

**Numbers** (one run each). `o3a`'s are the run's own report ([o3a_compare.md](o3a_compare.md),
[o3a_report_table.md](o3a_report_table.md)), which gives generation to one decimal, prompts in whole tok/s and the
median minutes to one decimal; in that rounding `o4a` read 16.5 and 57 tok/s and `o4sa` 16.3 and 115. The table
scripts that give the other runs' finer figures, and the minutes at equal work, were not run for `o3a` (see below):

| | v3.0, `ka` (2026-11-06, the same rig) | R160a `o4a` | R160b `o4sa` | **R160c `o3a`** |
|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.52 | 2.57 | 2.08 | **not tabled** |
| minutes, all 12 tasks (wall) | 30.2 | 29.7 | 27.6 | **35.8** |
| minutes a task, median | 2.11 | 2.07 | 1.94 | **2.5** |
| turns | 63 | 62 | 63 | **71** |
| generation tok/s | 16.42 | 16.53 | 16.30 | **16.2** |
| prompt tok/s | 60.1 | 56.6 | 114.5 | **61** |
| first token median, p90 s | 9.34, 19.86 | 8.65, 19.22 | 3.05, 10.97 | **7.26, 19.56** |
| first token on a task's first turn, median s | 16.39 | 16.32 | 8.45 | **16.34** |
| tool calls / failed / malformed | 63 / 1 / 0 | 62 / 0 / 0 | 64 / 0 / 0 | **71 / 2 / 0** |
| graph reallocations (target) | not counted | 0 in 62 requests | 0 in 63 requests | **0 in 71 requests** |
| lowest MemAvailable, swap | | 803 MiB at 14:24:35; swap 29 → 336 MiB | 1,252 MiB at 15:28:56; swap 30 → 335 MiB | **732 MB, swap up to 341 MB (the report's gate)** |

- **Where the wall minutes went** ([o3a_report_table.md](o3a_report_table.md), the three runs' attempts): `o3a`'s
  attempts spent a mean 117 s a task generating (`o4a` 93, `o4sa` 107) and 59 s reading prompts (53, 29), and
  processed 3,618 prompt tokens a task (2,991, 3,316). Python `scale-generator` took 9.2 minutes in 7 turns (`o4a` 5.6
  in 5, `o4sa` 8.6 in 7) and Python `affine-cipher` 4.7 in 13 turns; every other task took 3.6 minutes or less. No
  task came near the 12-minute cap.
- **Graph reallocations:** none in 71 requests (the report's count from the server log, whose v3.0 build logs
  `graph reallocs` per request as v4's does). The per-bucket and per-request files were not made.
- **Host:** the report's memory gate ([o3a_gonogo.md](o3a_gonogo.md)) reads the client's `meminfo.log`, which holds
  R160a's and R160b's rows before R160c's. Its lowest MemAvailable, 732 MB, is below both earlier runs' own lows (the
  same gate gave R160a's as 804 MB; R160b's was 1,252 MiB), and its swap high, 341 MB, is above both runs' (336 and
  335 MiB), so both are R160c's. R160c's rows start at 30 MiB of swap (16:22:21). NO-GO on the swap growth, as
  for R160a and R160b. The RAM watchdog did not stop the server. The times of the low and the high and the card peaks
  were not tabled.
- **Server log:** 15 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o3a_gonogo.md](o3a_gonogo.md)).
- **Not tabled:** the machine that drives the rig refused the container that runs the table scripts. So `o3a` has no
  row in [agent.tsv](agent.tsv) (minutes a task at equal work, generation to two decimals, prompts to one) or in
  [host.tsv](host.tsv), and no `o3a_reallocs.tsv` or `o3a_requests_reallocs.tsv`. The inputs are kept, and the command
  is below. R160d's table run later gave `o3a`'s row in [agent.tsv](agent.tsv) (see R160d).
- The report's files: [o3a_compare.md](o3a_compare.md), [o3a_report_table.md](o3a_report_table.md) (the three runs'
  attempts), [o3a_gonogo.md](o3a_gonogo.md); the per-request and per-attempt rows: [agent/o3a/](agent/o3a/).

## R160d: `o4b`, the v4 preset's second run (2026-11-14)

- **The started command** is R160a's, byte for byte (423 bytes, md5 `7b16f13d060c8b1fde8f9af58493226a`): the
  preset's four switches, the harness's `COMMON` line, then `--temp 0.6 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit
  off`. Every flag of the settings file's `PRESET_ARGS` is in it. The server's `switches:` line, read during the load,
  says `4 set: GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SAMPLING_TOP_K_ROWS=1
  LLAMA_SERVER_YIELD=2`.

**Numbers** (one run each; [agent.tsv](agent.tsv), [host.tsv](host.tsv)). R160d's table run also covered `o3a`, so
`o3a` now has its row in [agent.tsv](agent.tsv) with the finer figures (its host row and reallocation files are still
not made):

| | v3.0, `ka` (2026-11-06, the same rig) | R160a `o4a` | R160b `o4sa` | R160c `o3a` | **R160d `o4b`** |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 12 / 12 | 12 / 12 | **12 / 12** |
| minutes a task at equal work (`ka`'s work) | 2.52 | 2.57 | 2.08 | 2.52 | **2.45** |
| minutes, all 12 tasks (wall) | 30.2 | 29.7 | 27.6 | 35.8 | **31.0** |
| minutes a task, median | 2.11 | 2.07 | 1.94 | 2.49 | **1.97** |
| turns | 63 | 62 | 63 | 71 | **57** |
| generation tok/s | 16.42 | 16.53 | 16.30 | 16.22 | **16.37** |
| prompt tok/s | 60.1 | 56.6 | 114.5 | 61.1 | **65.4** |
| first token median, p90 s | 9.34, 19.86 | 8.65, 19.22 | 3.05, 10.97 | 7.26, 19.56 | **9.58, 20.29** |
| first token on a task's first turn, median s | 16.39 | 16.32 | 8.45 | 16.34 | **16.26** |
| tool calls / failed / malformed | 63 / 1 / 0 | 62 / 0 / 0 | 64 / 0 / 0 | 71 / 2 / 0 | **57 / 0 / 0** |
| graph reallocations (target) | not counted | 0 in 62 requests | 0 in 63 requests | 0 in 71 requests | **0 in 57 requests** |
| lowest MemAvailable, swap | | 803 MiB at 14:24:35; swap 29 → 336 MiB | 1,252 MiB at 15:28:56; swap 30 → 335 MiB | 732 MB, swap up to 341 MB (the report's gate) | **718 MiB at 17:51:05; swap 31 → 335 MiB** |

- **The same arm twice:** `o4a` and `o4b` ran the same command on the same rig. `o4b` took 2.45 minutes a task at
  equal work against `o4a`'s 2.57, generated at 16.37 tok/s against 16.53 and read prompts at 65.4 tok/s against
  56.6. These are two runs of six; the verdict comes from the report over all six.
- **Where the wall minutes went** ([o4b_report_table.md](o4b_report_table.md), the four runs' attempts): `o4b`'s
  attempts spent a mean 99 s a task generating (`o4a` 93, `o4sa` 107, `o3a` 117) and 54 s reading prompts (53, 29,
  59), and processed 3,506 prompt tokens a task (2,991, 3,316, 3,618). Python `scale-generator` took 8.4 minutes in 5
  turns (`o4a` 5.6 in 5, `o4sa` 8.6 in 7, `o3a` 9.2 in 7) and Python `pig-latin` 4.1 in 7 turns; every other task
  took 2.8 minutes or less. No task came near the 12-minute cap.
- **Graph reallocations** ([o4b_reallocs.tsv](o4b_reallocs.tsv), [o4b_requests_reallocs.tsv](o4b_requests_reallocs.tsv)):
  0 in every bucket (34 requests under 512 tokens, 8 at 512–1,023, 10 at 1,024–1,999, 5 at 2,000+).
- **Host** ([host.tsv](host.tsv), R160d's own lines of `meminfo.log` and `gpu.csv`, from 17:17:31): lowest
  MemAvailable 718 MiB at 17:51:05 (during Python `scale-generator`), 1,553 MiB at the end; swap 31 MiB at the start,
  335 MiB at its highest at 17:26:16, during the load. The report's memory gate ([o4b_gonogo.md](o4b_gonogo.md)) reads
  the shared log: its 718 MB low is R160d's, its 341 MB swap high R160c's. NO-GO on the swap growth, as for the
  earlier runs. The RAM watchdog did not stop the server. Card peaks MiB: 5,667 5,699 5,613 5,699 5,307 5,415 5,577
  5,665 5,071.
- **Server log:** 13 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill ([o4b_gonogo.md](o4b_gonogo.md)).
- The report's files: [o4b_compare.md](o4b_compare.md), [o4b_report_table.md](o4b_report_table.md) (the four runs'
  attempts), [o4b_gonogo.md](o4b_gonogo.md); the per-request and per-attempt rows: [agent/o4b/](agent/o4b/).

## How to run it

[run/r160.sh](run/r160.sh) runs one arm. It reads the settings file of the arm's version and stops if its
`PRESET_ARGS` is not v3.0's Ornith line. Run it from the repository's root on the machine that drives the rig; the
machine-specific values go in the environment:

```bash
ARM=o4a RIG=<rig alias> CLIENT=local MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN_V4=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  BIN_V3=<rig path>/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r160 OUT=<out>/agent-o4a \
  bash docs/data/2026-11-14_r160_ornith_release_agent/run/r160.sh
```

R160a was started with the same call written out (the harness
[release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh) with `STEPS=run`, `ARGS_o4a="--temp 0.6
-b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"`, `BUDGET_MIN_o4a=50`, `CTX=49152`, `TIME_CAP_MIN=12`,
`SAMPLING='{"temperature":0.6,"top_p":0.95,"top_k":20,"min_p":0}'`, `REL_ENV` the preset's `PRESET_ENV`). The run's
own report came from the same environment with `STEPS="report collect"`. The equal-work table: `agent_tables.py OUT
v3ka o4a` over a folder holding `ka`'s `turns.csv` and `attempts.csv` (its arm `rel` renamed `v3ka`) beside `o4a`'s,
and `o4a`'s server log; the reallocations: `scripts/realloc_counts.py o4a=server-o4a.log` (default and
`--requests`); the host table: `run/host.py <folder with meminfo.log and gpu.csv> R160a o4a`. All three ran in the
sandbox image.

R160b was started the same way with arm `o4sa`: `REL_ENV="$PRESET_ENV GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1"`,
`ARGS_o4sa="--temp 0.6 -b 4096 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
--ubatch-taper short=600:64"`, `BUDGET_MIN_o4sa=50`, the rest as R160a's (with `run/r160.sh`: `ARM=o4sa`). Its
report: `STEPS="report collect"` with `ARMS=o4sa`. The client's results folder is shared by the six runs and the
rig's logs are appended to it, so the tables used R160b's own rows: `agent_tables.py OUT v3ka o4a o4sa` over `ka`'s and
`o4a`'s rows above plus `o4sa`'s, `realloc_counts.py o4sa=server-o4sa.log`, and `run/host.py` over the lines of
`meminfo.log` and `gpu.csv` that R160b added (from 14:55:40). All three ran in the sandbox image.

R160c was started the same way with arm `o3a` on the v3.0 tree: `. release/ornith-v3/settings.env`, `REL_BIN` the
v3.0 tree's `llama-server`, `REL_ENV="$PRESET_ENV"`, `ARGS_o3a="--temp 0.6 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit
off"`, `BUDGET_MIN_o3a=50`, the rest as R160a's (with `run/r160.sh`: `ARM=o3a`). Its report: `STEPS="report collect"`
with `ARMS=o3a`. Its tables are to be made as R160b's, in the sandbox image, from a folder holding R160b's table rows
(`v3ka`, `o4a`, `o4sa`) plus `o3a`'s rows of `res/turns.csv` and `res/attempts.csv`, the three runs' server logs, and the
lines of `meminfo.log` and `gpu.csv` that R160c added (from 16:22:21): `agent_tables.py OUT v3ka o4a o4sa o3a`,
`realloc_counts.py o3a=rig/server-o3a.log` (default and `--requests`), `run/host.py <those lines> R160c o3a`.

R160d was started as R160a with arm `o4b` (with `run/r160.sh`: `ARM=o4b`). Its report: `STEPS="report collect"` with
`ARMS=o4b`. Its tables ran in the sandbox image from a folder holding R160c's table rows (`v3ka`, `o4a`, `o4sa`,
`o3a`) plus `o4b`'s rows of `res/turns.csv` and of `res/attempts.csv` (the collected `attempts.csv` holds every run's
rows, so only those whose `tag` starts with `o4b/`), the four runs' server logs, and the lines of `meminfo.log` and
`gpu.csv` that R160d added (from 17:17:31): `agent_tables.py OUT v3ka o4a o4sa o3a o4b`, `realloc_counts.py
o4b=rig/server-o4b.log` (default and `--requests`), `run/host.py <those lines> R160d o4b`.
