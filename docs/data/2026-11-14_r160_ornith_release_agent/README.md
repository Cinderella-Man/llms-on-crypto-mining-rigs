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
| graph reallocations (target) | not logged by v3.0's build | **0 in 62 requests** |
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
  323 in 71 of 73 requests. `release/ornith-v4/README.md`'s candidate `LLAMA_SCHED_PER_SIZE_RESERVE=2` is to be tried
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
