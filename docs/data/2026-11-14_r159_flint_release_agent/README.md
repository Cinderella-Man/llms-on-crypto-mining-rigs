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
- `f4t*` (R159e `f4ta` and R159f `f4tb`, 2026-11-16) is the tuned v4 preset: `f4` plus the four candidates that the
  deep lane's split ([2026-11-16_r165_flint_stack_split](../2026-11-16_r165_flint_stack_split/README.md)) does not
  class as a loss over all depths, `GGML_CUDA_GRAPHS_FORCE=1` and `LLAMA_SCHED_PER_SIZE_RESERVE=2` in the environment,
  `--ctx-checkpoints-vram 4` and `--ubatch-taper short=600:64` on the line. It leaves out the keep-alive (that split's
  loss) and staged hops (`GGML_CUDA_HOP_STAGED=1`, a loss on the 8,192-token prompt chunk there). Its two runs are
  classed against `f4a` and `f4b` and against v3.0 in [the report over the six runs](#the-report-over-the-six-runs-the-tuned-preset-f4t)
  below.

Registry row Q1478 in [quants/experiments.md](../../../quants/experiments.md); the speed table is in
[release/flint-v4/README.md](../../../release/flint-v4/README.md). **The verdict: v4 against v3.0 is negligible** on
minutes a task at equal work (1.85 and 1.82 against 1.81 and 1.82), and on generation and first token; **prompt
tok/s alone is a loss** (−1.3 %: 176.9 and 178.9 against 179.2 and 181.3), as the
[report over the four runs](#the-report-over-the-four-runs) below gives. **The tuned preset `f4t` against the v4
preset** ([the report over the six runs](#the-report-over-the-six-runs-the-tuned-preset-f4t)): prompts a gain
(+19.1 %), first token median a gain (−34.2 %), first token on a task's first turn a gain (−24.0 %); minutes a task
at equal work (−3.8 %), generation (+0.7 %) and first token p90 (−11.2 %) negligible. Against v3.0 the same, except
that p90 is a gain too (−14.6 %).

## The runs

| Run | Arm | Rig | Server started, up | Tasks | Passed | Started command |
|---|---|---|---|---|---|---|
| R159a | `f4a` | rig 1 | 12:09:16, up after 586 s | 12:19:05–12:50:32 | 12 / 12 | 401 bytes, md5 `46991da4e7acc094a91dec01c16252d3` |
| R159b | `f3a` | rig 1 | 14:05:56, up after 586 s | 14:15:45–14:40:37 | 11 / 12 | 331 bytes, md5 `fc324d8e91ff8ea133f6f7a6b9ffc05a` |
| R159c | `f4b` | rig 1 | 14:51:45, up after 587 s | 15:01:35–15:30:55 | 12 / 12 | 401 bytes, md5 `46991da4e7acc094a91dec01c16252d3` |
| R159d | `f3b` | rig 1 | 15:44:46, up after 584 s | 15:54:35–16:14:24 | 12 / 12 | 331 bytes, md5 `fc324d8e91ff8ea133f6f7a6b9ffc05a` |
| R159e (2026-11-16) | `f4ta` | rig 1 | 15:56:50, up after 587 s | 16:06:40–16:35:36 | 11 / 12 | 509 bytes, md5 `85b1ebffdbee2705ba57d648df2bded0` |
| R159f (2026-11-16) | `f4tb` | rig 1 | 16:46:19, up after 587 s | 16:56:09–17:19:56 | 12 / 12 | 509 bytes, md5 `85b1ebffdbee2705ba57d648df2bded0` |

## The report over the six runs: the tuned preset `f4t`

The harness's report with all six arms ([r159_six_compare.md](r159_six_compare.md),
[r159_six_report_table.md](r159_six_report_table.md), [r159_six_gonogo.md](r159_six_gonogo.md)), the tables over
them ([agent.tsv](agent.tsv), [host.tsv](host.tsv), [prompt_by_size.tsv](prompt_by_size.tsv)) and the class
([class.tsv](class.tsv), made by [run/class_arms.py](run/class_arms.py) from `agent.tsv`). Each arm is two runs, given
by value:

| | `f4t` (`f4ta`, `f4tb`) | v4 preset (`f4a`, `f4b`) | v3.0 (`f3a`, `f3b`) | `f4t` against the v4 preset | `f4t` against v3.0 |
|---|---|---|---|---|---|
| passed | 11 / 12, 12 / 12 | 12 / 12, 12 / 12 | 11 / 12, 12 / 12 | | |
| minutes a task at equal work (`v3base`'s work) | 1.72, 1.81 | 1.85, 1.82 | 1.81, 1.82 | negligible (−3.8 %) | negligible (−2.8 %) |
| generation tok/s | 49.32, 46.99 | 47.45, 48.19 | 48.55, 48.02 | negligible (+0.7 %) | negligible (−0.3 %) |
| prompt tok/s | 219.7, 204.1 | 176.9, 178.9 | 179.2, 181.3 | **gain (+19.1 %)** | **gain (+17.6 %)** |
| first token median s | 1.74, 1.82 | 2.85, 2.56 | 2.65, 2.61 | **gain (−34.2 %)** | **gain (−32.3 %)** |
| first token p90 s | 6.61, 6.35 | 7.84, 6.76 | 7.20, 7.97 | negligible (−11.2 %) | **gain (−14.6 %)** |
| first token on a task's first turn, median s | 3.24, 3.18 | 4.29, 4.16 | 4.17, 4.18 | **gain (−24.0 %)** | **gain (−23.1 %)** |
| Python `scale-generator` against the 12-minute cap | capped (5 turns); passed in 10.2 min (6 turns) | passed in 11.5 min (9 turns); passed in 7.8 min (9 turns) | capped (4 turns); passed in 5.7 min (6 turns) | | |

- **The rule** (the release check's): a gain or a loss only where the difference of the two arms' means is beyond the
  larger of their two spreads (the difference between an arm's two runs), else negligible; the percentage is the
  difference over the base's mean. The same script on `f4` against v3.0 gives the
  [report over the four runs](#the-report-over-the-four-runs)' classes.
- **Prompts:** means 211.9 against 177.9 and 180.25 tok/s, spreads 15.6, 2.0 and 2.1; the differences, 34.0 and
  31.65, are beyond `f4t`'s own spread, and `f4t`'s slower run (204.1) is faster than every other run. Without
  Python `scale-generator` (in `f4tb` its timed requests reached 26,171 tokens of context; in `f4ta` they stayed at
  4,088 or less before the cap stopped the fifth reply) its two runs read prompts at 218.8 and 221.2 tok/s.
- **First token:** median 1.78 against 2.705 and 2.63 s (spreads 0.08, 0.29, 0.04), on a task's first turn 3.21
  against 4.225 and 4.175 s (spreads 0.06, 0.13, 0.01): gains against both. p90 6.48 against 7.30 and 7.585 s
  (spreads 0.26, 1.08, 0.77): the difference to the v4 preset, 0.82 s, is within its spread of 1.08, so negligible;
  to v3.0, 1.105 s, beyond 0.77, so a gain.
- **Minutes a task at equal work:** means 1.765 against 1.835 and 1.815, spreads 0.09, 0.03 and 0.01: the
  differences, 0.07 and 0.05, are within `f4t`'s own spread, so negligible. `v3base`'s work is mostly generation
  (85 s of decode against 23 s of prefill a task), so this measure follows generation speed, and `f4tb` generated at
  46.99 tok/s, the slowest of the six runs, against `f4ta`'s 49.32, the fastest.
- **Generation:** means 48.155 against 47.82 and 48.285 tok/s, `f4t`'s spread 2.33: negligible. Without Python
  `scale-generator` (its long replies at deep context count only where they were timed) the six runs generated at
  49.35 and 47.80 (`f4t`), 48.74 and 48.99 (v4 preset), 48.54 and 48.69 tok/s (v3.0), so `f4tb`'s lower speed is not
  that one task alone.
- **Python `scale-generator` and the 12-minute cap:** two of the six runs reached the cap on it, `f3a` (v3.0) in its
  fourth reply and `f4ta` (`f4t`) in its fifth; in each, the reply still generating when the cap stopped the task
  never logged a timing. `f4tb` passed it in 10.2 minutes (613 s, 107 s under the cap), `f4a` in 11.5 (28 s under),
  `f4b` in 7.8 and `f3b` in 5.7. Every other task passed in all six runs.
- **Graph reallocations** (the target, the same size in every case, none changed): 250 in 71 of 72 requests and 290
  in 70 of 71 (`f4t`) against 323 in 71 of 73 and 308 in 79 of 80 (v4 preset) and 188 in 64 of 65 and 237 in 66 of 68
  (v3.0). `LLAMA_SCHED_PER_SIZE_RESERVE=2` does not bring them to 0.
- **Host:** lowest MemAvailable 1,084 and 1,100 MiB (`f4t`) against 885 and 873 (v4 preset) and 948 and 947 MiB
  (v3.0); swap grew to 296 and 299 MiB during the load, as in every run.
- **Go/no-go** ([r159_six_gonogo.md](r159_six_gonogo.md), over the shared client folder's 72 attempts):
  infrastructure GO (72 of 72), tool calls GO (0 malformed of 396), cache GO (median hit 94.5 % from the second
  turn), first token against prefill GO; proxy rows NO-GO for the two capped attempts alone; memory NO-GO on the swap
  growth during each load (its line reads R159c's low of 873 MB and the highest swap of the six runs, `f4tb`'s 299 MB), with
  MemAvailable above the 600 MB floor in every run.

## The report over the four runs

The harness's report with all four arms ([r159_compare.md](r159_compare.md), [r159_report_table.md](r159_report_table.md),
[r159_gonogo.md](r159_gonogo.md)) and the tables over them ([agent.tsv](agent.tsv): `agent_tables.py` at `v3base`'s
work, [host.tsv](host.tsv), `<arm>_reallocs.tsv`). Each value is one run:

| | v4 `f4a` | v4 `f4b` | v3.0 `f3a` | v3.0 `f3b` | v4 against v3.0 |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 11 / 12 | 12 / 12 | |
| minutes a task at equal work (`v3base`'s work) | 1.85 | 1.82 | 1.81 | 1.82 | negligible (+1.1 %) |
| generation tok/s | 47.45 | 48.19 | 48.55 | 48.02 | negligible (−1.0 %) |
| prompt tok/s | 176.9 | 178.9 | 179.2 | 181.3 | loss (−1.3 %) |
| first token median s | 2.85 | 2.56 | 2.65 | 2.61 | negligible (+2.9 %) |
| first token p90 s | 7.84 | 6.76 | 7.20 | 7.97 | negligible (−3.8 %) |
| first token on a task's first turn, median s | 4.29 | 4.16 | 4.17 | 4.18 | negligible (+1.2 %) |
| graph reallocations (target) | 323 in 71 of 73 requests | 308 in 79 of 80 | 188 in 64 of 65 | 237 in 66 of 68 | |
| lowest MemAvailable, swap during the load | 885 MiB; 30 → 295 MiB | 873 MiB; 32 → 296 MiB | 948 MiB; 31 → 293 MiB | 947 MiB; 32 → 295 MiB | |
| minutes, all 12 tasks (wall) | 31.3 | 29.2 | 24.7 | 19.7 | (different work) |
| turns | 73 | 80 | 66 | 68 | |
| decode seconds a task, mean | 120 | 112 | 49 | 73 | |
| prompt tokens processed a task | 6,007 | 5,560 | 3,111 | 4,099 | |

- **The class** (the release check's rule: a gain or a loss only beyond the larger of the two arms' spreads between
  their runs, else negligible): minutes a task at equal work 1.85 and 1.82 against 1.81 and 1.82 (means 1.835 and
  1.815, spreads 0.03 and 0.01); generation 47.45 and 48.19 tok/s against 48.55 and 48.02 (means 47.82 and 48.285,
  spreads 0.74 and 0.53); first token median 2.85 and 2.56 s against 2.65 and 2.61 (spreads 0.29 and 0.04), p90 7.84
  and 6.76 against 7.20 and 7.97 (spreads 1.08 and 0.77): each difference within the larger spread. Prompts 176.9 and
  178.9 tok/s against 179.2 and 181.3 (means 177.9 and 180.25, spreads 2.0 and 2.1): the difference, 2.35, is beyond
  the larger spread, and v4's faster run is below v3.0's slower one, so a loss. Each run reads its own prompts here
  (v4's attempts read more, below); the deep lane on the same file and line, where both arms read the same prompts,
  measured v4 as a gain: decode +0.7 %, prompts +3.1 %
  ([2026-11-14_r161_flint_release_deep](../2026-11-14_r161_flint_release_deep/README.md)).
- **The wall minutes differ because the work differed.** v4's two runs took 31.3 and 29.2 minutes for all tasks
  against v3.0's 24.7 and 19.7, at nearly the same speeds: their attempts wrote more (decode 120 and 112 s a task
  against 49 and 73), read more (6,007 and 5,560 prompt tokens a task against 3,111 and 4,099) and took more turns
  (73 and 80 against 66 and 68). Most of it is three tasks: JavaScript `variable-length-quantity` (4.5 and 8.7 minutes
  against 1.3 and 2.0), Python `pig-latin` (4.4 and 4.0 against 2.3 and 2.2) and Python `list-ops` (3.5 and 2.1
  against 1.7 and 1.7). Whether that is chance in sampled replies at temperature 1.0 over two runs, or one of the
  preset's switches changing what the sampler picks, is registry row Q1488.
- **Graph reallocations by prompt size** (`<arm>_reallocs.tsv`, per request in `<arm>_requests_reallocs.tsv`), the
  same size in every case, none changed: under 512 prompt tokens `f4a` 67 in 44 of 45 requests, `f4b` 85 in 56 of 56,
  `f3a` 61 in 41 of 41, `f3b` 64 in 43 of 44; 512–1,023 46 in 12 of 12, 46 in 12 of 12, 44 in 11 of 11, 35 in 9 of 9;
  1,024–1,999 53 in 9 of 9, 41 in 7 of 7, 70 in 11 of 11, 59 in 10 of 10; 2,000 and more 157 in 6 of 7, 136 in 4 of 5,
  13 in 1 of 2, 79 in 4 of 5. Both builds reallocate at every size.
- **Go/no-go** ([r159_gonogo.md](r159_gonogo.md)): infrastructure GO (48 of 48 attempts), tool calls GO (0 malformed
  of 260), cache GO (median hit 94.4 % from the second turn), first token against prefill GO; proxy rows NO-GO for
  R159b's capped attempt alone; memory NO-GO on the swap growth during each load (the gate allows 16 MiB), with
  MemAvailable above its 600 MB floor in every run. Its memory line reads the shared client folder's appended logs;
  each run's own low is in the table above.

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
| minutes a task at equal work (`v3base`'s work) | 1.85 | 1.81 | **1.82** |
| minutes, all 12 tasks (wall) | 31.3 | 24.7 | **29.2** |
| turns | 73 | 66 | **80** |
| generation tok/s | 47.45 | 48.55 | **48.19** |
| prompt tok/s | 176.9 | 179.2 | **178.9** |
| first token median, p90 s | 2.85, 7.84 | 2.65, 7.20 | **2.56, 6.76** |
| first token on a task's first turn, median s | 4.29 | 4.17 | **4.16** |
| tool calls / failed / malformed | 70 / 1 / 0 | 57 / 0 / 0 | **72 / 0 / 0** |
| graph reallocations (target) | 323 same size, 0 changed, in 71 of 73 requests | 188 same size, 0 changed, in 64 of 65 requests | **308 same size, 0 changed, in 79 of 80 requests** |
| lowest MemAvailable, swap | 885 MiB at 12:40:35; swap 30 → 295 MiB | 948 MiB at 14:39:09; swap 31 → 293 MiB | **873 MiB at 15:30:47; swap 32 → 296 MiB** |

- The speeds and minutes at equal work come from [agent.tsv](agent.tsv) (`agent_tables.py`), made with the report
  over the four runs; the harness's own report for this run rounds them to 48.2 and 179.
- **Graph reallocations** ([f4b_reallocs.tsv](f4b_reallocs.tsv), per request in
  [f4b_requests_reallocs.tsv](f4b_requests_reallocs.tsv)): under 512 prompt tokens 85 in 56 of 56 requests;
  512–1,023, 46 in 12 of 12; 1,024–1,999, 41 in 7 of 7; 2,000 and more, 136 in 4 of 5.
- **Long tasks:** JavaScript `variable-length-quantity` 8.7 minutes in 13 turns, Python `scale-generator` 7.8 minutes
  in 9 turns (it passed; R159b's `f3a` reached the cap on it, R159a's `f4a` passed at 11.5 minutes), Python
  `pig-latin` 4.0 minutes in 10 turns, Python `list-ops` 2.1 minutes. Every other task took under 2 minutes.
- **Host** (the rig's `meminfo.log`, this run's lines only): swap grew from 32 to 296 MiB during the load (its highest
  at 15:01:32, before the first task at 15:01:35); MemAvailable was lowest at 873 MiB (15:30:47, in the last task)
  and 1,388 MiB at the end. The RAM watchdog did not stop the server. Peak MiB per card ([host.tsv](host.tsv)): 5,413 5,951
  5,107 5,617 5,951 5,107 5,571 5,321 4,833.
- **The report's go/no-go lines** ([f4b_gonogo.md](f4b_gonogo.md)) read the shared client folder, as for R159b: its
  harness and cache lines count the three runs' 36 attempts, the proxy-rows NO-GO is R159b's capped attempt, and its
  memory line's 873 MB is this run's low (below R159a's and R159b's) with R159a's starting swap of 30 MB.
- **Server log:** 14 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill.
- The report's files: [f4b_compare.md](f4b_compare.md), [f4b_report_table.md](f4b_report_table.md) (its table lists
  `f3a` and `f4a` beside `f4b`), [f4b_gonogo.md](f4b_gonogo.md); the per-request and per-attempt rows:
  [agent/f4b/](agent/f4b/).

## R159d: `f3b`, v3.0's second run (2026-11-14)

- **The started command** is R159b's, byte for byte (331 bytes, the same md5): no switch in front, the harness's
  `COMMON` line and the arm's flags. v3.0's build prints no `switches:` line; the running server's environment, read
  during the load, held no `LLAMA_*` or `GGML_*` variable.
- **Numbers** (one run, from the report over the four runs and the rig's memory log): 12 of 12 passed, 19.7 minutes
  for all tasks, 68 turns, a task's median 1.2 minutes; generation 48.02 tok/s and prompts 181.3 tok/s; first token median 2.61 s, p90 7.97 s, on a task's first turn 4.18 s; tool calls 61 / 0 failed / 0
  malformed; graph reallocations 237 with the graph's size unchanged, 0 changed, in 66 of 68 requests. Minutes a task
  at equal work 1.82 ([agent.tsv](agent.tsv)). Reallocations by prompt size ([f3b_reallocs.tsv](f3b_reallocs.tsv), per
  request in [f3b_requests_reallocs.tsv](f3b_requests_reallocs.tsv)): under 512 tokens 64 in 43 of 44 requests,
  512–1,023 35 in 9 of 9, 1,024–1,999 59 in 10 of 10, 2,000 and more 79 in 4 of 5.
- **Tasks:** Python `scale-generator` passed in 5.7 minutes in 6 turns (R159b's `f3a` reached the cap on it), Python
  `affine-cipher` 2.2, Python `pig-latin` 2.2, JavaScript `variable-length-quantity` 2.0, Python `list-ops` 1.7; every
  other task under 1.6 minutes.
- **Host** (the rig's `meminfo.log`, this run's lines only): swap grew from 32 to 295 MiB during the load (its highest
  at 15:54:32, before the first task at 15:54:35); MemAvailable was lowest at 947 MiB (16:10:04) and 1,392 MiB at the
  end. The RAM watchdog did not stop the server. Peak MiB per card ([host.tsv](host.tsv)): 5,411 5,951 5,107 5,617 5,951
  5,107 5,571 5,321 4,837.
- **Server log:** 14 checkpoint restores, none created or erased, no full re-processing, no NaN or inf, no watchdog
  kill.
- Its per-request and per-attempt rows: [agent/f3b/](agent/f3b/); its report is the report over the four runs.

## R159e: `f4ta`, the tuned preset `f4t`'s first run (2026-11-16)

- **The started command** is R159c's plus exactly the four additions (509 bytes): the preset's three switches, the
  two new ones, then the harness's `COMMON` line and the arm's flags with the two new flags at the end:

  ```
  LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1 GGML_CUDA_GRAPHS_FORCE=1
  LLAMA_SCHED_PER_SIZE_RESERVE=2 <v4-rc2>/bin/llama-server -ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4
  --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui
  --alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64
  ```

  The server's `switches:` line, read during the load, says `5 set: GGML_CUDA_GRAPHS_FORCE=1 LLAMA_KV_SEQ_RM_BOUND=1
  LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_YIELD=2`. At the end of the load it printed
  both plans: `context checkpoints on the devices: the newest 4 of each slot, older ones move to the host in the
  background` and `micro-batch taper of prompt calls: short=600:64 (n_ubatch 128)`. The file fitted on the first load.

**Numbers** (one run, beside R159's four runs, every arm at `v3base`'s work in [agent.tsv](agent.tsv); classed with
R159f in [the report over the six runs](#the-report-over-the-six-runs-the-tuned-preset-f4t)):

| | v4 `f4a` | v4 `f4b` | v3.0 `f3a` | v3.0 `f3b` | **`f4t` `f4ta`** |
|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 11 / 12 | 12 / 12 | **11 / 12** |
| minutes a task at equal work (`v3base`'s work) | 1.85 | 1.82 | 1.81 | 1.82 | **1.72** |
| minutes, all 12 tasks (wall) | 31.3 | 29.2 | 24.7 | 19.7 | **28.8** |
| minutes a task, median | 1.21 | 0.95 | 1.17 | 1.25 | **1.10** |
| turns | 73 | 80 | 66 | 68 | **73** |
| generation tok/s | 47.45 | 48.19 | 48.55 | 48.02 | **49.32** |
| prompt tok/s | 176.9 | 178.9 | 179.2 | 181.3 | **219.7** |
| first token median s | 2.85 | 2.56 | 2.65 | 2.61 | **1.74** |
| first token p90 s | 7.84 | 6.76 | 7.20 | 7.97 | **6.61** |
| first token on a task's first turn, median s | 4.29 | 4.16 | 4.17 | 4.18 | **3.24** |
| tool calls / failed / malformed | 70 / 1 / 0 | 72 / 0 / 0 | 57 / 0 / 0 | 61 / 0 / 0 | **68 / 3 / 0** |
| graph reallocations (target) | 323 in 71 of 73 requests | 308 in 79 of 80 | 188 in 64 of 65 | 237 in 66 of 68 | **250 in 71 of 72** |
| lowest MemAvailable, swap during the load | 885 MiB; 30 → 295 MiB | 873 MiB; 32 → 296 MiB | 948 MiB; 31 → 293 MiB | 947 MiB; 32 → 295 MiB | **1,084 MiB at 16:35:19; 30 → 296 MiB** |
| decode seconds a task, mean | 120 | 112 | 49 | 73 | **64** |
| prompt tokens processed a task | 6,007 | 5,560 | 3,111 | 4,099 | **4,076** |

- **The capped task:** Python `scale-generator` reached the 12-minute cap in 5 turns (the report's "runaway
  (time)"). Its fifth reply's first token came after 3.25 s, and it was still generating when the cap stopped the
  task, 705.2 s into the request. The server never logged that request's timing, so it counts in none of the speeds
  (73 turns, 72 timed requests; est. ~34,600 tokens at 49.32 tok/s). R159b's `f3a` reached the cap on the same task;
  `f4a`, `f4b` and `f3b` passed it in 11.5, 7.8 and 5.7 minutes. The other long tasks: JavaScript
  `variable-length-quantity` 3.7 minutes in 8 turns, Python `pig-latin` 3.1 in 8, `affine-cipher` 2.9 in 5,
  `list-ops` 1.8 in 6; every other task took 1.2 minutes or less.
- **Prompt speed by prompt size** ([prompt_by_size.tsv](prompt_by_size.tsv), from each arm's timed turns in
  `agent/<arm>/turns.csv`; [run/prompt_by_size.py](run/prompt_by_size.py)). Each run reads its own prompts, so the
  buckets hold different requests:

  | prompt tokens | `f4a` | `f4b` | `f3a` | `f3b` | **`f4ta`** |
  |---|---|---|---|---|---|
  | under 512 | 123.5 (45) | 121.5 (56) | 124.5 (41) | 107.9 (44) | **172.7 (48)** |
  | 512–1,023 | 175.6 (12) | 181.0 (12) | 177.5 (11) | 171.3 (9) | **220.3 (10)** |
  | 1,024–1,999 | 215.5 (9) | 208.3 (7) | 211.7 (11) | 216.2 (10) | **236.9 (9)** |
  | 2,000 and more | 184.8 (7) | 200.5 (5) | 235.4 (2) | 209.3 (5) | **236.2 (5)** |

  Tokens a second over the bucket's requests (their number in parentheses).
- **Checkpoints:** 14 restores, all from the cards (52.5 MiB on the devices) in 2.0, 2.1, 2.2 or 2.3 ms each; `f4b`'s
  14 came from host memory in 292.0, 292.1, 292.2 or 294.6 ms. None created on the host or erased, no full
  re-processing, no NaN or inf, no watchdog kill.
- **Graph reallocations** ([f4ta_reallocs.tsv](f4ta_reallocs.tsv), per request in
  [f4ta_requests_reallocs.tsv](f4ta_requests_reallocs.tsv)), the same size in every case, none changed: under 512
  prompt tokens 82 in 48 of 48 requests; 512–1,023, 42 in 10 of 10; 1,024–1,999, 52 in 9 of 9; 2,000 and more, 74 in
  4 of 5. With `LLAMA_SCHED_PER_SIZE_RESERVE=2` the target still reallocates at every size.
- **Host and cards** ([host.tsv](host.tsv)): swap grew from 30 to 296 MiB during the load (its highest at 16:06:29,
  before the first task at 16:06:40); MemAvailable was lowest at 1,084 MiB (16:35:19, in the last task) and 1,915 MiB
  at the end. The RAM watchdog did not stop the server. Peak MiB per card: 5,433 5,979 5,129 5,639 5,979 5,129 5,593
  5,349 4,853 (`f4b`: 5,413 5,951 5,107 5,617 5,951 5,107 5,571 5,321 4,833).
- **The report's go/no-go lines** ([f4ta_gonogo.md](f4ta_gonogo.md)) read the shared client folder, now with five
  runs' 60 attempts: the proxy-rows NO-GO is the two capped attempts (`f3a`'s and this one's), and the memory line's
  873 MB is R159c's low. This run's own host numbers are the ones above.
- The report's files: [f4ta_compare.md](f4ta_compare.md), [f4ta_report_table.md](f4ta_report_table.md) (its table
  lists the four earlier arms beside `f4ta`), [f4ta_gonogo.md](f4ta_gonogo.md); the per-request and per-attempt rows:
  [agent/f4ta/](agent/f4ta/).

## R159f: `f4tb`, the tuned preset `f4t`'s second run (2026-11-16)

- **The started command** is R159e's, byte for byte (509 bytes, the same md5). The server's `switches:` line, read
  during the load, says `5 set: GGML_CUDA_GRAPHS_FORCE=1 LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SAMPLING_TOP_K_ROWS=1
  LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_YIELD=2`, and at the end of the load it printed both plans, as in
  R159e. The file fitted on the first load.

**Numbers** (one run, beside R159's five earlier runs, every arm at `v3base`'s work in [agent.tsv](agent.tsv)):

| | v4 `f4a` | v4 `f4b` | v3.0 `f3a` | v3.0 `f3b` | `f4t` `f4ta` | **`f4t` `f4tb`** |
|---|---|---|---|---|---|---|
| passed | 12 / 12 | 12 / 12 | 11 / 12 | 12 / 12 | 11 / 12 | **12 / 12** |
| minutes a task at equal work (`v3base`'s work) | 1.85 | 1.82 | 1.81 | 1.82 | 1.72 | **1.81** |
| minutes, all 12 tasks (wall) | 31.3 | 29.2 | 24.7 | 19.7 | 28.8 | **23.3** |
| minutes a task, median | 1.21 | 0.95 | 1.17 | 1.25 | 1.10 | **1.13** |
| turns | 73 | 80 | 66 | 68 | 73 | **71** |
| generation tok/s | 47.45 | 48.19 | 48.55 | 48.02 | 49.32 | **46.99** |
| prompt tok/s | 176.9 | 178.9 | 179.2 | 181.3 | 219.7 | **204.1** |
| first token median s | 2.85 | 2.56 | 2.65 | 2.61 | 1.74 | **1.82** |
| first token p90 s | 7.84 | 6.76 | 7.20 | 7.97 | 6.61 | **6.35** |
| first token on a task's first turn, median s | 4.29 | 4.16 | 4.17 | 4.18 | 3.24 | **3.18** |
| tool calls / failed / malformed | 70 / 1 / 0 | 72 / 0 / 0 | 57 / 0 / 0 | 61 / 0 / 0 | 68 / 3 / 0 | **68 / 1 / 0** |
| graph reallocations (target) | 323 in 71 of 73 requests | 308 in 79 of 80 | 188 in 64 of 65 | 237 in 66 of 68 | 250 in 71 of 72 | **290 in 70 of 71** |
| lowest MemAvailable, swap during the load | 885 MiB; 30 → 295 MiB | 873 MiB; 32 → 296 MiB | 948 MiB; 31 → 293 MiB | 947 MiB; 32 → 295 MiB | 1,084 MiB; 30 → 296 MiB | **1,100 MiB at 17:15:51; 30 → 299 MiB** |
| decode seconds a task, mean | 120 | 112 | 49 | 73 | 64 | **86** |
| prompt tokens processed a task | 6,007 | 5,560 | 3,111 | 4,099 | 4,076 | **5,093** |

- **Every task passed.** Python `scale-generator` took 10.2 minutes (613 s, 107 s under the 12-minute cap) in 6
  turns, with 478 s of decode; its six requests generated 22,006 tokens at 46.05 tok/s and reached 26,171 tokens of
  context. R159b's `f3a` and R159e's `f4ta` reached the cap on it. The other long tasks: JavaScript
  `variable-length-quantity` 2.2 minutes in 6 turns, Python `pig-latin` 1.7 in 5, `affine-cipher` 1.7 in 6,
  `phone-number` 1.6 in 8; every other task took 1.2 minutes or less. All 71 requests were timed.
- **Prompt speed by prompt size** ([prompt_by_size.tsv](prompt_by_size.tsv)): under 512 tokens 183.8 tok/s (43
  requests), 512–1,023 217.5 (13), 1,024–1,999 230.7 (10), 2,000 and more 196.0 (5); `f4ta` 172.7, 220.3, 236.9 and
  236.2 in the same buckets, the four earlier runs in [R159e](#r159e-f4ta-the-tuned-preset-f4ts-first-run-2026-11-16).
- **Checkpoints:** 15 restores, all from the cards (52.5 MiB on the devices): the first, 11 min 50 s after the
  server started, in 162.7 ms, the other 14 in 2.1, 2.2 or 2.6 ms. None created on the host or erased, no full
  re-processing, no NaN or inf, no watchdog kill.
- **Graph reallocations** ([f4tb_reallocs.tsv](f4tb_reallocs.tsv), per request in
  [f4tb_requests_reallocs.tsv](f4tb_requests_reallocs.tsv)), the same size in every case, none changed: under 512
  prompt tokens 74 in 43 of 43 requests; 512–1,023, 48 in 13 of 13; 1,024–1,999, 59 in 10 of 10; 2,000 and more, 109
  in 4 of 5.
- **Host and cards** ([host.tsv](host.tsv)): swap grew from 30 to 299 MiB during the load (its highest at 16:56:00,
  before the first task at 16:56:09); MemAvailable was lowest at 1,100 MiB (17:15:51, in JavaScript
  `complex-numbers`) and 2,817 MiB at the end. The RAM watchdog did not stop the server. Peak MiB per card: 5,439
  5,985 5,133 5,643 5,985 5,133 5,597 5,355 4,859 (`f4ta`: 5,433 5,979 5,129 5,639 5,979 5,129 5,593 5,349 4,853).
- The run's own report: [f4tb_compare.md](f4tb_compare.md). Its report table and go/no-go lines read the shared
  client folder and came out identical to the six-run report's, [r159_six_report_table.md](r159_six_report_table.md)
  and [r159_six_gonogo.md](r159_six_gonogo.md). The per-request and per-attempt rows: [agent/f4tb/](agent/f4tb/).

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
They ran with R159d's report, over all four runs.

R159d ran R159b's call with arm `f3b` (`. release/flint-v3/settings.env`, `REL_BIN` the v3.0 tree's `llama-server`,
`ARGS_f3b` and `BUDGET_MIN_f3b=40` as above). The report over the four runs is the same environment with
`ARMS="f3a f4a f3b f4b" STEPS="report collect"` and an `ARGS_<arm>` for each (with `run/r159.sh`: `ARM=f3b
ARMS="f3a f4a f3b f4b" STEPS="report collect"`). The tables for all four, in the sandbox image: `agent_tables.py OUT v3p065 f4a
f3a f4b f3b` over a folder holding `v3base`'s rows (arm `v3p065`) beside the report's `res/turns.csv` and
`res/attempts.csv` and the four server logs in `rig/`; `realloc_counts.py <arm>=rig/server-<arm>.log` (default and
`--requests`) for `f4b` and `f3b`; `host_summary.py RIGDIR R159c f4b` on R159c's collected rig logs and
`host_summary.py RIGDIR R159d f3b` on R159d's, each in the sandbox image.

R159e ran the harness directly ([run/r159.sh](run/r159.sh) takes only the preset's own switches and flags), with
`f4b`'s call and the four additions, from the repository's root on the machine that drives the rig:

```bash
. release/flint-v4/settings.env
BENCH=$PWD/bench/agent_check RIG=<rig alias> CLIENT=local CTX=262144 TIME_CAP_MIN=12 CPU_SHARES=256 \
  MODEL=<rig path>/flint-near-q8-p065.gguf RIG_LOGDIR=<rig folder>/agent-logs \
  RIG_PORT=8272 RELAY_PORT=8276 PROXY_PORT=18773 CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r159 \
  REL_BIN=<rig path>/llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828/bin/llama-server \
  REL_ENV="$PRESET_ENV GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2" \
  ARGS_f4ta="--alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64" \
  BUDGET_MIN_f4ta=40 ARMS=f4ta STEPS=run \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <out>/agent-f4ta
```

Its report is the same environment with `STEPS="report collect"`. Its tables, in the sandbox image:
`agent_tables.py OUT v3p065 f4a f3a f4b f3b f4ta` over R159d's table folder with `f4ta`'s 73 turns and 12 attempts
added (the collected `res/attempts.csv` holds every arm of the shared client folder, so only the rows whose `tag`
starts with `f4ta/` were added; the five earlier rows of [agent.tsv](agent.tsv) came out unchanged),
`realloc_counts.py f4ta=rig/server-f4ta.log` (default and `--requests`), `host_summary.py RIGDIR R159e f4ta` on the
collected rig logs, and `run/prompt_by_size.py . f4a f3a f4b f3b f4ta` in this folder.

R159f ran R159e's call with the arm renamed `f4tb` (`ARGS_f4tb`, `BUDGET_MIN_f4tb=40`, `ARMS=f4tb`); its started
command is R159e's byte for byte. Its own report is the same environment with `STEPS="report collect"`. The report
over the six runs is the harness's report step alone with `ARMS="f3a f4a f3b f4b f4ta f4tb"` and
`STEPS="report collect"` ([r159_six_compare.md](r159_six_compare.md), [r159_six_report_table.md](r159_six_report_table.md),
[r159_six_gonogo.md](r159_six_gonogo.md)). Its tables, in the sandbox image: `agent_tables.py OUT v3p065 f4a f3a f4b
f3b f4ta f4tb` over R159e's table folder with `f4tb`'s turns and attempts added (only the rows whose `tag` starts with
`f4tb/`; the six earlier rows of [agent.tsv](agent.tsv) came out unchanged), `realloc_counts.py
f4tb=rig/server-f4tb.log` (default and `--requests`), `host_summary.py RIGDIR R159f f4tb` on the collected rig logs,
`run/prompt_by_size.py . f4a f3a f4b f3b f4ta f4tb` in this folder, and the class
([class.tsv](class.tsv), [run/class_arms.py](run/class_arms.py)):

```bash
python3 run/class_arms.py agent.tsv f4t=f4ta,f4tb f4=f4a,f4b v3.0=f3a,f3b
```
