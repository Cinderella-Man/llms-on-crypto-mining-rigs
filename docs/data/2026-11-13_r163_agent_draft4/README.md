# Flash-Next's final v4 settings with drafts of 4 tokens (R163, Q1483; 2026-11-13)

One question: does Flash-Next do an agent task faster when its MTP draft guesses 4 tokens a step in place of 3, now that
5-token verify passes have exact read-once kernels? This folder answers it with two agent-check runs of the arm `v4rd4`:
R163 on one P106 rig, and R163b, a second run of the same command on the other rig (arm `v4rd42`). The result is the
registry row Q1483 in [quants/experiments.md](../../../quants/experiments.md).

**Answer on these two runs: no.** With drafts of 4 tokens, each step yields more tokens (3.7305 and 3.7760 against
3.2786 and 3.2461), as FN80 estimated (3.7028). But each step costs more than that gain allows: est. 152.44 and
151.11 ms against 118.74 and 121.71. So the model generates at 24.47 and 24.99 tok/s against `v4r`'s 27.61 and 26.67,
and an agent task takes 2.48 and 2.47 minutes at equal work against 2.37 and 2.43. Both runs passed 12 of 12 tasks.

## The arm

`v4rd4` is `v4r`, Flash-Next's final v4 settings ([release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env);
measured in [2026-11-13_r156_agent_v4r](../2026-11-13_r156_agent_v4r/README.md)), on the same build `v4-rc2`
(`pascal-p106-v4` `eb82916d3`), model files (Swift GSQ-RCO Q2_0 and the MTP draft) and tasks, with three changes:

| Change | What it does | Its P106 check |
|---|---|---|
| `--spec-draft-n-max 4` in place of `3` | the draft guesses 4 tokens a step, so each verify pass reads 5 tokens | |
| `GGML_CUDA_MMVQ_PASCAL_RO8=1` | IQ4_XS, Q4_K and Q5_K products of 5–8 tokens read each weight once (C729 with C809) | exact, R134 ([data](../2026-11-12_c809_ro8_one_warp/README.md)) |
| `GGML_CUDA_MMVQ_PASCAL_RO8_KT=1` | the 6,144-wide IQ4_XS and Q4_K products of 5–8 tokens on C794's K tiles | exact and a gain, R149 ([data](../2026-11-13_c794_ktiles_5to8/README.md)) |

Everything else is `v4r`'s: the draft's micro-batch of 512, probabilistic verification and C675's stop at an accepted
end of turn. Before the first task, the started command and environment were checked against R156b's `v4r2` (`v4r`
byte for byte): the same text with `--spec-draft-n-max 4` and the two switches added, nothing else (`cmp`). The
server listed 41 switches set, `v4r`'s 39 and these two.

`_RO8_KT=1` is the setting R149 measured as a gain and names for verify passes of 5 tokens; `=2` sends the tiles
wherever they fit and serves the correctness checks.

## Why it was tried

FN80 (Q1396) measured the MTP draft's yield on recorded agent requests: est. 3.7028 tokens a step at a draft of 4
against 3.2401 at 3. At `v4r`'s step that pays only if the extra drafting pass (est. 3.645 ms) and the wider verify pass
together cost less than est. 14.77 ms. Earlier on 2026-11-13, a draft of up to 6 tokens decided while drafting (C815,
arm `v4c3ad`, R154a) lost: 4.049 tokens a step at est. 160.3 ms against 3.29 at 117.4. That run had no exact kernels
for 5–8-token verify passes; this one has both.

## Results

`v4rd4` (R163, rig 1) and `v4rd42` (R163b, rig 2) against `v4r`'s two runs (R156b on rig 1, R156a on rig 2):

| | `v4r2` (rig 1) | `v4r` (rig 2) | **`v4rd4` (rig 1)** | **`v4rd42` (rig 2)** |
|---|---|---|---|---|
| passed | 12/12 | 12/12 | **12/12** | **12/12** |
| equal work, min a task (12 tasks) | 2.37 | 2.43 | **2.48** | **2.47** |
| equal work, R90's 11 tasks | 1.51 | 1.59 | **1.66** | **1.62** |
| equal work without `complex-numbers` | 2.46 | 2.46 | **2.57** | **2.56** |
| equal work, the 10 tasks no v3.0 run ran away on | 1.52 | 1.52 | **1.67** | **1.64** |
| generation tok/s | 27.61 | 26.67 | **24.47** | **24.99** |
| prompt tok/s | 144.6 | 133.9 | **146.5** | **144.1** |
| first token median, p90 s | 2.48, 8.65 | 2.42, 9.00 | **2.37, 8.43** | **2.51, 8.35** |
| acceptance (accepted / drafted) | 0.7595 | 0.7489 | **0.6825** | **0.6941** |
| drafted tokens a step | 2.994 | 2.994 | **3.992** | **3.990** |
| tokens a step | 3.2786 | 3.2461 | **3.7305** | **3.7760** |
| est. ms a step | 118.74 | 121.71 | **152.44** | **151.11** |
| graph reallocations | 0 of 68 | 0 of 95 | **0 of 67** | **0 of 63** |

- **Equal work** ([agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv),
  [agent_10.tsv](agent_10.tsv)): `v4rd4` takes 2.48 minutes a task, 0.11 more than `v4r2` on the same rig (+4.6 %) and
  0.05 more than `v4r` (+2.1 %). On every subset of tasks both 4-token runs are slower than both 3-token runs.
- **Generation:** 24.47 tok/s, 3.14 below `v4r2` (−11.4 %) and 2.20 below `v4r` (−8.2 %).
- **Prompts:** 146.5 tok/s, against 144.6 and 133.9 ([prompt_buckets.tsv](prompt_buckets.tsv)). The prompt reading
  does not change with the draft's length; R156's two runs already differed by more than this.
- **The step** ([steps.tsv](steps.tsv), from the server logs): each reply's steps are its generated tokens over the
  server's "mean len" for that reply, and ms a step is the replies' eval time over those steps (est.: it holds the
  drafting passes, the verify pass and the sampling). A step of `v4rd4` yields 0.4519 tokens more than `v4r2`'s
  (+13.8 %) and costs est. 33.70 ms more (+28.4 %). At `v4r2`'s 118.74 ms, the 13.8 % more tokens would pay for at
  most est. 16.37 ms more; the step costs about twice that. Against `v4r` the figures are +0.4844 tokens and est.
  +30.73 ms.
- **Acceptance** falls from 0.7595 and 0.7489 to 0.6825 and 0.6941, as expected: the fourth guess is accepted less
  often than the first three. The tokens a step still match FN80's estimate.
- **The agent tables' `tokens_per_step` column** is 1 + 3 × acceptance, which assumes a draft of 3 tokens: for the
  4-token arms it reads 3.05 and 3.08, which is wrong. Use [steps.tsv](steps.tsv) for these arms.

Not classed as a gain or a loss: one run on each rig. Both runs point the same way, against both 3-token runs.

## The host and the cards (rig 1, R163)

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 343 MiB, at 21:24:22 during the
  last task. Swap stayed at 35 MiB from start to end; the watchdog did not act. `v4r2` on the same rig: 419 MiB, swap
  36 → 404 MiB.
- **Cards** (peak MiB, 20:59:51–21:29:17): 0:5,135 1:5,177 2:5,037 3:5,191 4:4,291 5:5,051 6:4,441 7:5,063
  **8:5,789**. Against `v4r2` on the same rig (0:5,099 1:5,145 2:5,001 3:5,161 4:4,259 5:5,017 6:4,417 7:5,029
  8:5,619), cards 0–7 hold 24 to 36 MiB more (the 5-token verify pass's buffers) and card 8, which holds the draft,
  170 MiB more.
- **Server log:** 13 checkpoint restores, none erased, no full re-processing, no NaN or inf, no watchdog kill, no graph
  reallocation.
- **The report** ([v4rd4_gonogo.md](v4rd4_gonogo.md), over every arm in the client folder, both 4-token arms
  included): GO on infrastructure, tool calls (0 malformed of 850; `v4rd4` made 63 calls), first token against prefill
  and cache. The NO-GO on proxy rows lists earlier arms' attempts, none of `v4rd4`'s. The NO-GO on memory reads the
  client's `meminfo.log` over every arm and both rigs, so its lowest point is not `v4rd4`'s; `v4rd4`'s own is the
  343 MiB above.

## The host and the cards (rig 2, R163b)

From the same client folder: lowest MemAvailable 385 MiB, swap 21 → 895 MiB ([host.tsv](host.tsv)); card peaks
(21:00:47–21:24:01) 0:5,119 1:5,161 2:5,021 3:5,177 4:4,277 5:5,033 6:4,431 7:5,045 8:5,623; 14 checkpoint restores,
none erased, no full re-processing, no NaN, no watchdog kill. R163b's own record adds its run's details.

## Per task ([v4rd4_compare.md](v4rd4_compare.md))

`v4rd4`'s minutes (pi's start to its exit), against `v4r2` on the same rig: `affine-cipher` 2.6 (1.7), `space-age` 0.7
(0.8), Python `phone-number` 0.8 (1.5), `triangle` 0.7 (1.1), Python `variable-length-quantity` 1.9 (4.4), JS
`phone-number` 1.1 (0.9), `list-ops` 1.9 (1.6), `grade-school` 1.0 (0.9), `scale-generator` 5.6 (2.6),
`complex-numbers` 1.1 (1.0, the short path in 7 turns), `pig-latin` 1.7 (2.2), JS `variable-length-quantity` 5.0
(12.0, `v4r2`'s ran to the cap). Wall-clock minutes follow how long each reply is; equal work leaves that out.

## Commands

R163, from the repository root on a machine that only runs ssh. It used R72's client folders (R158's ports), so no
prep step ran:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4rd4 V4R_DRAFT_UB=512 AGENT_BY=21:15 \
  V4RD4_ENV="GGML_CUDA_MMVQ_PASCAL_RO8=1 GGML_CUDA_MMVQ_PASCAL_RO8_KT=1" \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad v4r v4r2 v3b v4rd4" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r163_agent_draft4/run/r163.sh > r163.log 2>&1 < /dev/null &
```

The report rewrites the client folder's `res/turns.csv` with exactly the arms it is given, so `REPORT_ARMS` names every
arm in R72's folder that the tables read (R90's `v3` comes from R90's own folder). R163b ran the same driver with
`AGENT_ORDER=v4rd42` on the other rig.

The tables, on the client's CPU in an empty folder, with R142's
[tables.sh](../2026-11-13_r142_agent_v4c3/run/tables.sh) and R158's 10-task table; then [run/steps.sh](run/steps.sh)
over the server logs, and the cards' peaks over each rig's block of the client's `gpu.csv` (the two rigs' logs are
appended to one file as each run ends, so each block is taken by its lines, not by time):

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> \
  NEWARMS="v4c3 v4c32 v4r v4r2 v3b v4rd4 v4rd42" bash <this repository>/docs/data/2026-11-13_r142_agent_v4c3/run/tables.sh
SKIP_TASKS=bugfix-javascript-variable-length-quantity,bugfix-javascript-complex-numbers OUT_TAG=_10 \
  nice -n 19 python3 -I <this repository>/docs/data/2026-11-05_keepalive/run/agent_tables.py merged \
  r90v3 r90gains gains v4c2 v4c22 v4c2g v4c3 v4c32 v4r v4r2 v3b v4rd4 v4rd42
bash <this repository>/docs/data/2026-11-13_r163_agent_draft4/run/steps.sh <R72's client results folder>/rig \
  v4r v4r2 v4rd4 v4rd42 > steps.tsv
awk -F', ' -v a=<first line> -v b=<last line> 'NR >= a && NR <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv), [agent_10.tsv](agent_10.tsv) | each arm against R90's `v3`: all 12 tasks, R90's 11, the 12 without `complex-numbers`, and the 10 no v3.0 run ran away on; the rows before `v4rd4` are R156's and R158's, unchanged |
| [steps.tsv](steps.tsv) | per arm: replies, generated tokens, eval ms, steps, tokens a step, est. ms a step, accepted and drafted tokens |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt tok/s by the size of a turn's new prompt |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's acceptance per task |
| [host.tsv](host.tsv) | per arm: lowest MemAvailable, when, and swap |
| [v4rd4_report_table.md](v4rd4_report_table.md), [v4rd4_compare.md](v4rd4_compare.md), [v4rd4_gonogo.md](v4rd4_gonogo.md) | the agent check's report after R163's run |
| [run/r163.sh](run/r163.sh) | the driver: R156's with the arms `v4rd4` and `v4rd42` |
| [run/steps.sh](run/steps.sh) | tokens and ms a step from the server logs |

Provenance: the agent check's 12 bugfix tasks come from Exercism's Python and JavaScript tracks (MIT; see
[bench/agent_check](../../../bench/agent_check/README.md)). The tables are this repository's measurements, under its
licence.
