# The agent check of v4's candidate line `v4c3` (R142, Q1446; 2026-11-13)

R142 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on `v4c3`, v4's candidate line after the code freeze, on one rig of nine P106-090 cards with Flash-Next's v3
preset (Swift Q2_0 and the MTP draft). `v4c3` is `v4c2` (R72b and R72d, [2026-11-12_r72_agent_v4c2](../2026-11-12_r72_agent_v4c2/README.md))
with six switches added, each of which passed its own check on a P106 rig. Two runs, one arm each: R142a (this page so
far) and R142b. Registry row Q1446 in [quants/experiments.md](../../../quants/experiments.md).

**R142a: `v4c3` takes 2.35 minutes a task at equal work with 12 of 12 passed, against `v4c2`'s 2.44 (R72b, 11 of 12)
and 2.37 (R72d, 12 of 12).** Without `bugfix-javascript-complex-numbers`, the task R72b ran to its cap, the three runs
take 2.45, 2.45 and 2.46 minutes a task. Generation is faster than in both `v4c2` runs (28.03 tok/s against 26.12 and
27.53; without that task 28.00 against 27.48 and 27.52), the draft's acceptance is 0.765 (0.726 and 0.753), and no
request reallocated the target's graph. The prompt side reads between the two `v4c2` runs at 512 new tokens and more, so at
equal work `v4c3` stays within `v4c2`'s run-to-run spread: **negligible** against `v4c2` on one run, not slower on any
task set, with a gain in generation. R142b, the second run, follows.

Every value below is one run; `v4c2`'s two runs are listed each by its own values. "Equal work" is R90's `v3` work (its
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

## All 12 tasks ([agent.tsv](agent.tsv))

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` (`v4c2` again) | **R142a `v4c3`** |
|---|---|---|---|---|
| passed | 11/12 | 11/12 | 12/12 | **12/12** |
| equal work, min a task | 3.08 | 2.44 (−20.8 %) | 2.37 (−23.1 %) | **2.35 (−23.7 %)** |
| turns | 66 | 92 | 65 | **66** |
| generation tok/s | 20.66 | 26.12 | 27.53 | **28.03** |
| prompt tok/s | 65.5 | 136.0 | 147.6 | **145.5** |
| first token median, p90 s | 6.52, 16.17 | 2.42, 9.29 | 2.81, 8.58 | **2.45, 8.39** |
| first turn's first token, median s | 13.54 | 6.65 | 6.67 | **6.64** |
| acceptance | 0.696 | 0.726 | 0.753 | **0.765** |
| tokens a step | 3.09 | 3.18 | 3.26 | **3.29** |
| minutes, all 12 tasks (wall) | 37.0 | 31.3 | 19.9 | **16.4** |

## The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

R90's `v3` ran `bugfix-javascript-variable-length-quantity` out of its budget, so R90 compared the other 11.

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** |
|---|---|---|---|---|
| equal work, min a task | 2.27 | 1.61 (−29.1 %) | 1.51 (−33.5 %) | **1.49 (−34.4 %)** |
| generation tok/s | 20.63 | 25.72 | 27.42 | **28.01** |
| prompt tok/s | 65.6 | 133.8 | 145.9 | **143.9** |

## Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

R72b ran this task to its 12-minute cap in 35 turns; R72d passed it in 1.0 minute and R142a in 0.9, each in 5 turns.

| | R90 `v3` | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** |
|---|---|---|---|---|
| equal work, min a task | 3.13 | 2.45 | 2.46 | **2.45** |
| turns | 59 | 57 | 60 | **61** |
| generation tok/s | 20.92 | 27.48 | 27.52 | **28.00** |
| prompt tok/s | 66.6 | 148.7 | 144.0 | **141.6** |
| first token median, p90 s | | 2.99, 8.72 | 2.89, 8.58 | **2.45, 8.26** |

## Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

Turns, prompt tok/s and median first token (s):

| new tokens | R72b `v4c2` | R72d `v4c22` | **R142a `v4c3`** |
|---|---|---|---|
| 0–511 | 56, 92.8, 1.68 | 41, 89.3, 1.70 | **42, 94.0, 1.83** |
| 512–1,023 | 15, 112.8, 6.60 | 10, 126.6, 6.57 | **10, 124.4, 6.64** |
| 1,024–1,999 | 13, 163.5, 7.97 | 8, 187.7, 7.91 | **9, 186.5, 7.91** |
| 2,000+ | 7, 169.8, 11.69 | 6, 185.2, 9.48 | **5, 180.6, 9.38** |

From 512 new tokens up `v4c3` reads prompts between `v4c2`'s two runs, and below 512 above both
(94.0 against 92.8 and 89.3). Its whole-session prompt speed without
`complex-numbers` (141.6) is below both runs because fewer of its turns had 2,000+ new tokens.

## The draft's acceptance ([task_acceptance.tsv](task_acceptance.tsv))

`v4c3` accepted 12,574 of 16,446 drafted tokens (0.765), 12,130 of 15,894 without `complex-numbers` (0.763); `v4c2`:
0.723 and 0.761 (R72b), 0.753 and 0.751 (R72d), by the same per-task count. `bugfix-javascript-variable-length-quantity`
took 1.6 minutes and 1,815 drafted tokens, against 5.0 and 5.1 minutes and 6,726 and 6,900 drafted tokens in `v4c2`'s
runs: the agent took a shorter path through that task, which lowers the wall minutes (16.4 against 31.3 and 19.9) but
not the equal-work number.

## Graph reallocations (C804)

`v4c3`: 0 of 66 requests reallocated the target's graph (0 at an unchanged size, 0 at a changed size). `v4c2`: 0 of 91
(R72b) and 0 of 65 (R72d); R72a's `gains` line, for contrast: 69 of 69 (329 same size, 137 changed). From the server
logs, by the report's own count ([v4c3_compare.md](v4c3_compare.md), [v4c3_gonogo.md](v4c3_gonogo.md)).

## The host and the cards

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 478 MiB at 10:45:52, the end of the
  first task (R72b 392 MiB, R72d 365 MiB). Swap grew from 5 to 475 MiB between 10:47:00 and 10:53:34 (tasks 3 to 7)
  and stayed there; MemAvailable was 862 MiB at the end. R72b's swap grew 19 → 472 MiB in its first four tasks, R72d's
  stayed at 20 MiB. The watchdog did not act.
- **Cards** (peak MiB in the run): 0:5,099 1:5,145 2:5,001 3:5,161 4:4,259 5:5,017 6:4,417 7:5,029 8:5,535; R72d
  0:5,115 1:5,159 2:5,017 3:5,175 4:4,273 5:5,033 6:4,427 7:5,045 8:5,783. Card 8, where the draft's buffers sit,
  peaks 248 MiB lower; cards 0, 2, 5 and 7 16 MiB lower, cards 1, 3 and 4 14, card 6 10.
- **Server log:** 13 checkpoint restores, none erased, no full re-processing, no NaN or inf, no watchdog kill.
- **The report** ([v4c3_gonogo.md](v4c3_gonogo.md)): infrastructure GO (60 of 60 attempts over R72's and R142a's arms),
  tool calls GO (0 malformed of 336; `v4c3`: 60 calls, 2 returned errors), first token against prefill GO. Proxy rows
  NO-GO only for R72b's runaway task. Memory NO-GO reads the rig's `meminfo.log` over every arm in the folder, so its
  338 MB and 18 → 713 MB are R72c's figures, not `v4c3`'s.

## Per task ([v4c3_compare.md](v4c3_compare.md))

Minutes against R72b and R72d: quicker than both in 4 tasks (`affine-cipher` 1.8 against 2.5 and 2.4, Python
`phone-number` 0.9 against 1.1 and 1.3, `complex-numbers` 0.9 against 12.0 and 1.0, JS `variable-length-quantity` 1.6
against 5.0 and 5.1), between the two runs or equal to one in 5, slower than both in 3 (Python `variable-length-quantity` 2.2 against 1.7 and 1.1, JS `phone-number` 1.5 in 7 turns
against 0.9 and 1.1, `grade-school` 0.8 against 0.7 and 0.7).

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

The tables, on the client's CPU in an empty folder ([run/tables.sh](run/tables.sh); `nice -n 19`), and the cards' peaks:

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
| [host.tsv](host.tsv) | `v4c3`'s lowest MemAvailable (with its Unix time) and swap, from the rig's RAM watchdog |
| [v4c3_compare.md](v4c3_compare.md), [v4c3_report_table.md](v4c3_report_table.md), [v4c3_gonogo.md](v4c3_gonogo.md) | the agent check's own report of R142a's run, `v4c2`, `v4c22` and `v4c3` side by side, with the graph reallocations |
| [run/r142.sh](run/r142.sh) | R142's run: R72d's driver with the arm `v4c3` and the step `files` |
| [run/tables.sh](run/tables.sh) | the tables above, from the client's folders |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the build and files named above; R90's and R72's rows
are their own measurements. The prompts are the agent sessions of bench/agent_check's bugfix set; no prompt text is in
this folder. The tables are this repository's measurements, under its licence.
