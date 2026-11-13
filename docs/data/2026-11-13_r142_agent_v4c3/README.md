# The agent check of v4's candidate line `v4c3` (R142, Q1446; 2026-11-13)

R142 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on `v4c3`, v4's candidate line after the code freeze, on one rig of nine P106-090 cards with Flash-Next's v3
preset (Swift Q2_0 and the MTP draft). `v4c3` is `v4c2` (R72b and R72d, [2026-11-12_r72_agent_v4c2](../2026-11-12_r72_agent_v4c2/README.md))
with six switches added, each of which passed its own check on a P106 rig. Two runs, one arm each: R142a (`v4c3`) and
R142b (`v4c32`, the same line again). Registry row Q1446 in [quants/experiments.md](../../../quants/experiments.md).

**On two runs each, `v4c3` is negligible against `v4c2` at equal work, with a small gain in generation.** At equal work
`v4c3`'s runs take 2.35 and 2.44 minutes a task over all 12 tasks against `v4c2`'s 2.44 and 2.37, 1.49 and 1.61 over
R90's 11 against 1.61 and 1.51, and 2.45 and 2.43 without `bugfix-javascript-complex-numbers` against 2.45 and 2.46:
the same spread, not slower on any task set. Without that task, generation is faster in both `v4c3` runs than in both
`v4c2` runs (28.00 and 27.79 tok/s against 27.48 and 27.52), and the median first token is lower (2.45 and 2.50 s
against 2.99 and 2.89). All four runs passed 12 of 12 except R72b (11 of 12), and none reallocated the target's graph.
The six switches stay in `v4c3`.

**R142b: `v4c32` takes 2.44 minutes a task at equal work with 12 of 12 passed.** It spent 10.8 minutes and 37 turns on
`bugfix-javascript-complex-numbers` (passed), the long path R72b's `v4c2` took to its 12-minute cap in 35 turns, so each
line took that path in one of its two runs; that task alone moves R142b's whole-session figures (98 turns, generation
26.16 tok/s, acceptance 0.723, 32.2 minutes for all 12). Without it R142b takes 2.43 minutes a task, generates at 27.79
tok/s and accepts 0.759 of the draft.

**R142a: `v4c3` takes 2.35 minutes a task at equal work with 12 of 12 passed, against `v4c2`'s 2.44 (R72b, 11 of 12)
and 2.37 (R72d, 12 of 12).** Without `bugfix-javascript-complex-numbers`, the task R72b ran to its cap, the three runs
take 2.45, 2.45 and 2.46 minutes a task. Generation is faster than in both `v4c2` runs (28.03 tok/s against 26.12 and
27.53; without that task 28.00 against 27.48 and 27.52), the draft's acceptance is 0.765 (0.726 and 0.753), and no
request reallocated the target's graph. The prompt side reads between the two `v4c2` runs at 512 new tokens and more, so at
equal work `v4c3` stays within `v4c2`'s run-to-run spread: **negligible** against `v4c2` on one run, not slower on any
task set, with a gain in generation.

Every value below is one run; `v4c2`'s and `v4c3`'s two runs are listed each by its own values. "Equal work" is R90's `v3` work (its
prompt and generated tokens, and its time outside the server) at each arm's own prompt and generation speeds
([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)): it compares the engines on the same work, while the
minutes for all tasks depend on what each attempt generated.

## The line

`v4c2`'s line exactly (its 33 switches, `-b 4096`, the command and the model files, as R72b and R72d ran it), on C825's
tarball of `pascal-p106-v4` at `65ac9a47d` ([2026-11-13_c825_tip_tarball](../2026-11-13_c825_tip_tarball/README.md)),
with these six added:

| Switch | Change | Its P106 check |
|---|---|---|
| `LLAMA_SCHED_PER_SIZE_RESERVE=1` | C803: the scheduler reserves its buffers per graph size | R130 |
| `GGML_CUDA_MMVF_F32_MT=1` | C807: the f32 matrix-vector kernel on Pascal | R132 |
| `LLAMA_PP_MULTI_SPLIT=1` | C812: the prompt's split across the cards | R135 |
| `GGML_CUDA_KEEPALIVE_EAGER=1` | C811 with C817's fix: the keep-alive block starts at once; its timeout stays at the default 1,000 ms (R137) | R140 |
| `LLAMA_MTP_HROW_LAZY=1` | C813: the draft's hidden row kept lazily | R139 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1` | C805: exact k-columns in the Pascal MMVQ kernels | R126 |

Not in it: `GGML_CUDA_GRAPHS_PATCH` (R136 patched nothing), C808's `_DMT_KT` (R133: a loss), `LLAMA_SPEC_ADAPT_ONLINE`
(its own arm later) and `GGML_CUDA_KEEPALIVE_MS` (default).

## The run

- **Build:** `llama.cpp-20261113-65ac9a47d-pascal-p106-v4.tar.gz` (807,543,830 B, md5
  `c6c9496354a3fb25e0857e15cce3b987`, checked on both ends of the copy); `bin/llama-server` md5
  `9cbea9859c17e3355c9eb51d14a4a863`, as C825 built it.
- **Driver:** [run/r142.sh](run/r142.sh) (R72d's driver with the arm `v4c3` and the step `files`) with
  `STEPS="files agent report"`, `AGENT_ORDER=v4c3`, in R72's client folders, so that the report sets `v4c3` beside R72's
  arms. The step `files` copied the bench files to the client, which R72a had prepared before C804 step (a), so that
  the report counts graph reallocations; the sandbox image and the validated tasks stayed as R72a left them, and the
  agent's own files (`run.py`, `client_arm.sh`, `rig_server.sh`) were byte-identical before and after. The client ran on
  a build machine's CPU at `--cpu-shares 256`, no GPU.
- **Checked before the first task:** the started command is R72d's character for character with the six switches
  added before the binary and the build's folder in place of `-c802`'s; the server's `switches:` line reads `39 set`,
  R72d's 33 and these six, none missing (`this build reads 870 LLAMA_ and GGML_ names`); `load_model` names Swift Q2_0's
  first shard.
- **Timeline:** server started 10:38:50, up after 309 s (R72b 313 s, R72d 311 s); 12 tasks 10:44:03–11:00:45; report
  11:00:57–11:01:00. The rig was clean afterwards: no llama process, 0 MiB on all nine cards.
- **R142b** ran the same build's tree, still unpacked on the rig (`bin/llama-server` md5 checked again), with the
  driver's arm `v4c32` (`v4c3`'s line, a second name for a second run), `STEPS="agent report"` (the client's bench files
  were current after R142a's `files`) and `REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32"`. Before the first task: the
  started command, environment included, is R142a's character for character (1,657 characters each), and the server's
  `switches:` line is R142a's byte for byte (`39 set`). Server started 11:34:12, up after 313 s; 12 tasks
  11:39:29–12:11:56; report 12:12:12–12:12:15; the rig clean afterwards.

## All 12 tasks ([agent.tsv](agent.tsv))

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` (`v4c2` again) | **R142a `v4c3`** | **R142b `v4c32`** (`v4c3` again) |
|---|---|---|---|---|---|
| passed | 11/12 | 11/12 | 12/12 | **12/12** | **12/12** |
| equal work, min a task | 3.08 | 2.44 (−20.8 %) | 2.37 (−23.1 %) | **2.35 (−23.7 %)** | **2.44 (−20.8 %)** |
| turns | 66 | 92 | 65 | **66** | **98** |
| generation tok/s | 20.66 | 26.12 | 27.53 | **28.03** | **26.16** |
| prompt tok/s | 65.5 | 136.0 | 147.6 | **145.5** | **136.6** |
| first token median, p90 s | 6.52, 16.17 | 2.42, 9.29 | 2.81, 8.58 | **2.45, 8.39** | **2.45, 9.77** |
| first turn's first token, median s | 13.54 | 6.65 | 6.67 | **6.64** | **6.62** |
| acceptance | 0.696 | 0.726 | 0.753 | **0.765** | **0.723** |
| tokens a step | 3.09 | 3.18 | 3.26 | **3.29** | **3.17** |
| minutes, all 12 tasks (wall) | 37.0 | 31.3 | 19.9 | **16.4** | **32.2** |

## The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

R90's `v3` ran `bugfix-javascript-variable-length-quantity` out of its budget, so R90 compared the other 11.

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** | **R142b `v4c32`** |
|---|---|---|---|---|---|
| equal work, min a task | 2.27 | 1.61 (−29.1 %) | 1.51 (−33.5 %) | **1.49 (−34.4 %)** | **1.61 (−29.1 %)** |
| generation tok/s | 20.63 | 25.72 | 27.42 | **28.01** | **25.84** |
| prompt tok/s | 65.6 | 133.8 | 145.9 | **143.9** | **130.8** |

## Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

R72b ran this task to its 12-minute cap in 35 turns; R142b passed it in 10.8 minutes and 37 turns; R72d passed it in
1.0 minute and R142a in 0.9, each in 5 turns.

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** | **R142b `v4c32`** |
|---|---|---|---|---|---|
| equal work, min a task | 3.13 | 2.45 | 2.46 | **2.45** | **2.43** |
| turns | 59 | 57 | 60 | **61** | **61** |
| generation tok/s | 20.92 | 27.48 | 27.52 | **28.00** | **27.79** |
| prompt tok/s | 66.6 | 148.7 | 144.0 | **141.6** | **152.0** |
| first token median, p90 s | | 2.99, 8.72 | 2.89, 8.58 | **2.45, 8.26** | **2.50, 8.81** |

## Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

Turns, prompt tok/s and median first token (s):

| new tokens | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** | **R142b `v4c32`** |
|---|---|---|---|---|
| 0–511 | 56, 92.8, 1.68 | 41, 89.3, 1.70 | **42, 94.0, 1.83** | **60, 93.0, 1.96** |
| 512–1,023 | 15, 112.8, 6.60 | 10, 126.6, 6.57 | **10, 124.4, 6.64** | **17, 111.2, 6.69** |
| 1,024–1,999 | 13, 163.5, 7.97 | 8, 187.7, 7.91 | **9, 186.5, 7.91** | **14, 162.1, 7.98** |
| 2,000+ | 7, 169.8, 11.69 | 6, 185.2, 9.48 | **5, 180.6, 9.38** | **7, 181.0, 12.50** |

From 512 new tokens up `v4c3` reads prompts between `v4c2`'s two runs in R142a, and below 512 above both
(94.0 against 92.8 and 89.3). Its whole-session prompt speed without
`complex-numbers` (141.6) is below both runs because fewer of its turns had 2,000+ new tokens. R142b, with
more turns in every bucket from its long `complex-numbers` session (as R72b had from its runaway), reads the buckets
close to R72b (93.0, 111.2, 162.1, 181.0 against 92.8, 112.8, 163.5, 169.8), and its prompt speed without that task is
the highest of the four runs (152.0 against 148.7, 144.0 and 141.6). The two lines read prompts within the same spread.

## The draft's acceptance ([task_acceptance.tsv](task_acceptance.tsv))

`v4c3` accepted 12,574 of 16,446 drafted tokens (0.765), 12,130 of 15,894 without `complex-numbers` (0.763); `v4c2`:
0.723 and 0.761 (R72b), 0.753 and 0.751 (R72d), by the same per-task count. `bugfix-javascript-variable-length-quantity`
took 1.6 minutes and 1,815 drafted tokens, against 5.0 and 5.1 minutes and 6,726 and 6,900 drafted tokens in `v4c2`'s
runs: the agent took a shorter path through that task, which lowers the wall minutes (16.4 against 31.3 and 19.9) but
not the equal-work number.

`v4c32` (R142b) accepted 22,748 of 31,479 drafted tokens (0.723), 15,666 of 20,649 without `complex-numbers` (0.759);
`complex-numbers` alone 7,082 of 10,830 (0.654) over its 37 turns. Without that task the two lines accept within the
same spread: `v4c3` 0.763 and 0.759, `v4c2` 0.761 and 0.751.

## Graph reallocations (C804)

`v4c3`: 0 of 66 (R142a) and 0 of 98 (R142b) requests reallocated the target's graph (0 at an unchanged size, 0 at a
changed size). `v4c2`: 0 of 91
(R72b) and 0 of 65 (R72d); R72a's `gains` line, for contrast: 69 of 69 (329 same size, 137 changed). From the server
logs, by the report's own count ([v4c3_compare.md](v4c3_compare.md), [v4c3_gonogo.md](v4c3_gonogo.md),
[v4c32_compare.md](v4c32_compare.md), [v4c32_gonogo.md](v4c32_gonogo.md)).

## The host and the cards

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 478 MiB at 10:45:52, the end of the
  first task (R72b 392 MiB, R72d 365 MiB). Swap grew from 5 to 475 MiB between 10:47:00 and 10:53:34 (tasks 3 to 7)
  and stayed there; MemAvailable was 862 MiB at the end. R72b's swap grew 19 → 472 MiB in its first four tasks, R72d's
  stayed at 20 MiB. The watchdog did not act. R142b: lowest MemAvailable 376 MiB at 12:05:23, in the last minute of
  its long `complex-numbers` session; swap stayed at 8 MiB from start to end; the watchdog did not act. So the two
  lines' lowest MemAvailable reads 478 and 376 MiB (`v4c3`) against 392 and 365 MiB (`v4c2`), and in each line one
  run's swap grew (R142a 5 → 475 MiB, R72b 19 → 472 MiB) and the other's did not (R142b 8 → 8, R72d 20 → 20).
- **Cards** (peak MiB in the run): 0:5,099 1:5,145 2:5,001 3:5,161 4:4,259 5:5,017 6:4,417 7:5,029 8:5,535; R72d
  0:5,115 1:5,159 2:5,017 3:5,175 4:4,273 5:5,033 6:4,427 7:5,045 8:5,783. Card 8, where the draft's buffers sit,
  peaks 248 MiB lower; cards 0, 2, 5 and 7 16 MiB lower, cards 1, 3 and 4 14, card 6 10. R142b (11:34:07–12:12:15):
  0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045 8:5,701, cards 0–7 within 2 MiB of R72d and
  card 8 82 MiB below it.
- **Server log:** 13 checkpoint restores, none erased, no full re-processing, no NaN or inf, no watchdog kill. R142b:
  18 restores, 2 erased, one full re-processing at 12:05:40, where `complex-numbers` ended and `pig-latin` began
  (1,553 tokens shared with the cache and no checkpoint below them), no NaN or inf, no watchdog kill; R72b's `v4c2`
  had 1 erased and one full re-processing in its runaway session.
- **The report** ([v4c3_gonogo.md](v4c3_gonogo.md)): infrastructure GO (60 of 60 attempts over R72's and R142a's arms),
  tool calls GO (0 malformed of 336; `v4c3`: 60 calls, 2 returned errors), first token against prefill GO. Proxy rows
  NO-GO only for R72b's runaway task. Memory NO-GO reads the rig's `meminfo.log` over every arm in the folder, so its
  338 MB and 18 → 713 MB are R72c's figures, not `v4c3`'s. R142b's report over the six arms
  ([v4c32_gonogo.md](v4c32_gonogo.md)) reads the same: infrastructure GO (72 of 72), tool calls GO (0 malformed of
  429; `v4c32`: 93 calls, 2 returned errors), first token against prefill GO, proxy rows NO-GO only for R72b's
  runaway, memory NO-GO with R72c's figures.

## Per task ([v4c3_compare.md](v4c3_compare.md))

Minutes against R72b and R72d: quicker than both in 4 tasks (`affine-cipher` 1.8 against 2.5 and 2.4, Python
`phone-number` 0.9 against 1.1 and 1.3, `complex-numbers` 0.9 against 12.0 and 1.0, JS `variable-length-quantity` 1.6
against 5.0 and 5.1), between the two runs or equal to one in 5, slower than both in 3 (Python `variable-length-quantity` 2.2 against 1.7 and 1.1, JS `phone-number` 1.5 in 7 turns
against 0.9 and 1.1, `grade-school` 0.8 against 0.7 and 0.7).

R142b ([v4c32_compare.md](v4c32_compare.md)), against R72b and R72d: quicker than both in 3 tasks (`affine-cipher` 2.2,
Python `phone-number` 1.0, JS `variable-length-quantity` 4.0 against 5.0 and 5.1), between the two runs or equal to
one in 6 (`complex-numbers` 10.8 in 37 turns against 12.0 and 1.0), slower than both in 3 (Python
`variable-length-quantity` 4.4 against 1.7 and 1.1, `grade-school` 0.9 in 7 turns against 0.7 and 0.7,
`scale-generator` 2.3 against 2.0 and 1.9).

## Commands

R142a, from the repository root on a machine that only runs ssh, with R72's client folders (R72a's prep):

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-65ac9a47d-pascal-p106-v4 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="files agent report" AGENT_ORDER=v4c3 REPORT_ARMS="v4c2 v4c22 v4c3" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r142_agent_v4c3/run/r142.sh > r142a.log 2>&1 < /dev/null &
```

The report rewrites the client folder's `res/turns.csv` with the arms it is given, so the report was run once more over
R72's arms and `v4c3` before the tables (the same environment, no rig command):

```bash
ARMS="gains v4c2 v4c22 v4c2g v4c3" STEPS="report collect" \
  bash docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh <local folder>
```

R142b, the same build's tree still on the rig and the same client folders, with the arm `v4c32` and a report over
every arm in those folders (so no second report was needed):

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-65ac9a47d-pascal-p106-v4 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4c32 \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r142_agent_v4c3/run/r142.sh > r142b.log 2>&1 < /dev/null &
```

The tables, on the client's CPU in an empty folder ([run/tables.sh](run/tables.sh); `nice -n 19`; R142b's with
`NEWARMS="v4c3 v4c32"`, which adds `v4c32`'s rows and leaves the others as R142a wrote them), and the cards' peaks
(R142b's window: 11:34:07 to 12:12:15):

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> NEWARMS=v4c3 \
  bash <this repository>/docs/data/2026-11-13_r142_agent_v4c3/run/tables.sh
awk -F', ' -v a='2026/11/13 10:38:50' -v b='2026/11/13 11:01:00' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv) | each arm against R90's `v3`, all 12 tasks: passed, turns, prompt and generation tok/s, first token, acceptance, minutes, equal work |
| [agent_11.tsv](agent_11.tsv) | the same over R90's 11 tasks |
| [agent_nocx.tsv](agent_nocx.tsv) | the same without `bugfix-javascript-complex-numbers` |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt speed and first token by a turn's new prompt |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's acceptance per task |
| [host.tsv](host.tsv) | `v4c3`'s and `v4c32`'s lowest MemAvailable (with its Unix time) and swap, from the rig's RAM watchdog |
| [v4c3_compare.md](v4c3_compare.md), [v4c3_report_table.md](v4c3_report_table.md), [v4c3_gonogo.md](v4c3_gonogo.md) | the agent check's own report of R142a's run, `v4c2`, `v4c22` and `v4c3` side by side, with the graph reallocations |
| [v4c32_compare.md](v4c32_compare.md), [v4c32_report_table.md](v4c32_report_table.md), [v4c32_gonogo.md](v4c32_gonogo.md) | the agent check's own report of R142b's run: `gains`, `v4c2`, `v4c22`, `v4c2g`, `v4c3` and `v4c32` side by side, with the graph reallocations |
| [run/r142.sh](run/r142.sh) | R142's run: R72d's driver with the arms `v4c3` and `v4c32` (the same line, one name a run) and the step `files` |
| [run/tables.sh](run/tables.sh) | the tables above, from the client's folders |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the build and files named above; R90's and R72's rows
are their own measurements. The prompts are the agent sessions of bench/agent_check's bugfix set; no prompt text is in
this folder. The tables are this repository's measurements, under its licence.
