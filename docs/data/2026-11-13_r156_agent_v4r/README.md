# The agent check of Flash-Next's final v4 settings `v4r` on `v4-rc2` (R156, Q1474; 2026-11-13)

R156 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on `v4r`, Flash-Next's final v4 settings, on the release candidate `v4-rc2` (`pascal-p106-v4` `eb82916d3`, the
v4 line on upstream llama.cpp b12360: [2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md)). The model is
the maker's Swift GSQ-RCO Q2_0 with the MTP draft, on a mining rig of nine P106-090 cards. `v4r` is `v4c3`
([2026-11-13_r142_agent_v4c3](../2026-11-13_r142_agent_v4c3/README.md)) with the results of 2026-11-13's rig rows
applied. Two runs, one arm each: R156a (`v4r`, on rig 2) and R156b (`v4r2`, the same line again, on rig 1). The two
rigs are the same model of machine. Registry row Q1474 in [quants/experiments.md](../../../quants/experiments.md).

**R156a (rig 2): `v4r` generates at 26.67 tok/s over all 12 tasks and takes 2.43 minutes a task at equal work, with
12 of 12 passed.** Without `bugfix-javascript-complex-numbers` it generates at 27.59 tok/s and takes 2.46 minutes a
task. `v4c3`'s two runs on rig 1 took 2.35 and 2.44 minutes a task (2.45 and 2.43 without that task) at 28.03 and 26.16
tok/s (28.00 and 27.79). R90's `v3` took 3.08 minutes at 20.66 tok/s, so `v4r` on rig 2 is 21.1 % quicker at equal
work and generates 29.1 % faster. Like R142b, this run took the long path on `complex-numbers` (9.7 minutes in 29
turns, passed), so its whole-session figures sit near R142b's. The line is classed after R156b.

Every value below is one run, labelled with its rig. "Equal work" means R90's `v3` work (its prompt and generated
tokens, and its time outside the server) at each arm's own prompt and generation speeds
([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). It compares the engines on the same work; the
minutes for all tasks depend on what each attempt generated.

## The line

`v4r` is `v4c3`'s environment and command (R142a, R142b) on `v4-rc2`'s rig build, with exactly these differences:

| Change | Why |
|---|---|
| `LLAMA_PP_MULTI_SPLIT=1` removed | R141: C812's prompt split costs 1.39 s a pass, beyond the spread (a loss) |
| `LLAMA_SCHED_PER_SIZE_RESERVE=2` in place of `=1` | R143: the shared scheduler is planned over the whole memory again whenever it re-plans |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0` added | R150 and C828: the shared expert's Q4_0 and Q5_0 down projections leave the k-column kernel at 1–4 tokens |
| `--spec-draft-n-ubatch 512` kept | `v4c3`'s value. The line already passes the flag that upstream #32078 needs for the MTP draft, and 512 reads prompts faster than 128 (R53) |
| the union gather (C103) stays off | R88 part (2a): a loss in the decode step at every depth with context |

The build is `v4-rc2`'s rig tarball (C828's, `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`), where `v4c3` ran
on C825's tip tarball `65ac9a47d`. The started command (environment, binary and arguments) is R142b's string with the
three switch changes made and the build folder swapped, and with no other difference: the two strings match
byte for byte after those edits (md5 `6b5dac04b78dfe51cfc94cf55bda79fc`). The server's `switches:` line reads
`39 set`. Three warnings in the server log are not errors:

- `LLAMA_SCHED_PER_SIZE_RESERVE` is ignored in one context that has no per-size schedulers. `v4c3`'s two logs show the
  same line.
- Under `=2` the 512-token graph needed a new plan once.
- No line about the draft's graph budget appears (upstream #32078).

## The run

R156a: rig 2, 2026-11-13. The server started at 18:15:30 and was up after 225 s; on rig 1, R142a's server took 309 s
and R142b's 313 s. The 12 tasks ran 18:19:17–18:52:50, and the report ran 18:52:56–18:53:02. The client ran on the
build machine's CPU at `--cpu-shares 256`, with no GPU. It wrote into R72's client folder beside R72's, R142's and
R154's arms.

## All 12 tasks ([agent.tsv](agent.tsv))

| | R90 `v3` | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** |
|---|---|---|---|---|
| passed | 11/12 | 12/12 | 12/12 | **12/12** |
| equal work, min a task | 3.08 | 2.35 | 2.44 | **2.43** (−21.1 % against `v3`) |
| turns | 66 | 66 | 98 | **95** |
| generation tok/s | 20.66 | 28.03 | 26.16 | **26.67** |
| prompt tok/s | 65.5 | 145.5 | 136.6 | **133.9** |
| first token median, p90 s | 6.52, 16.17 | 2.45, 8.39 | 2.45, 9.77 | **2.42, 9.00** |
| first token on a task's first turn, median s | 13.54 | 6.64 | 6.62 | **6.68** |
| acceptance | 0.696 | 0.765 | 0.723 | **0.749** |
| tokens a step | 3.09 | 3.29 | 3.17 | **3.25** |
| minutes, all 12 tasks (wall) | 37.0 | 16.4 | 32.2 | **33.4** |

## The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

Without `bugfix-javascript-variable-length-quantity`, which R90's `v3` did not pass: `v4r` takes **1.59 minutes a task**
at equal work (−30.0 % against `v3`'s 2.27), against `v4c3`'s 1.49 and 1.61, at 26.13 tok/s of generation (28.01 and
25.84).

## Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

| | R90 `v3` | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** |
|---|---|---|---|---|
| equal work, min a task | 3.13 | 2.45 | 2.43 | **2.46** |
| turns | 59 | 61 | 61 | **66** |
| generation tok/s | 20.92 | 28.00 | 27.79 | **27.59** |
| prompt tok/s | 66.6 | 141.6 | 152.0 | **140.7** |
| first token median, p90 s | 6.52, 16.46 | 2.45, 8.26 | 2.50, 8.81 | **2.46, 8.80** |
| acceptance | 0.696 | 0.765 | 0.723 | **0.749** (0.775 by task, [task_acceptance.tsv](task_acceptance.tsv)) |

## Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

| New tokens | R142a `v4c3` (rig 1) | R142b `v4c32` (rig 1) | **R156a `v4r` (rig 2)** |
|---|---|---|---|
| 0–511 | 94.0 tok/s (42 turns) | 93.0 (60) | **93.0** (66) |
| 512–1,023 | 124.4 (10) | 111.2 (17) | **112.3** (11) |
| 1,024–1,999 | 186.5 (9) | 162.1 (14) | **152.4** (11) |
| 2,000+ | 180.6 (5) | 181.0 (7) | **181.7** (7) |
| first token median at 2,000+, s | 9.38 | 12.50 | **12.79** |

## The draft's acceptance ([task_acceptance.tsv](task_acceptance.tsv))

`v4r`: 28,900 of 38,592 drafted tokens accepted (0.749); 21,634 of 27,927 without `complex-numbers` (0.775), against
`v4c3`'s 0.763 and 0.759. In `complex-numbers` 7,266 of 10,665 (0.681).

## Graph reallocations (C804)

`v4r`: 0 of 95 requests reallocated the target's graph. `v4c3`: 0 of 66 and 0 of 98. The counts come from the
report's reading of the server logs ([v4r_compare.md](v4r_compare.md)).

## The host and the cards (rig 2)

- **RAM** ([host.tsv](host.tsv), from the rig's RAM watchdog): lowest MemAvailable 312 MiB, at 18:52:33 in the last
  task. Swap stayed at 21 MiB from start to end, and the watchdog did not act. On rig 1, `v4c3` read 478 and 376 MiB,
  with swap 5 → 475 and 8 → 8 MiB.
- **Cards** (peak MiB, 18:15:30–18:53:02): 0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045
  **8:5,623**. On rig 1, R142b peaked at 0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045 8:5,701, so
  cards 0–7 match and card 8 peaks 78 MiB lower. R142a's card 8 peaked at 5,535 MiB.
- **Server log:** 12 checkpoint restores and 1 old checkpoint erased. One full re-processing came at 18:39:32, where
  `complex-numbers` ended and `pig-latin` began (1,553 tokens shared with the cache, no checkpoint below them), as in
  R142b. No NaN or inf, no watchdog kill.
- **The report** ([v4r_gonogo.md](v4r_gonogo.md), over every arm in the folder):
  - GO: infrastructure (96 of 96 attempts), tool calls (0 malformed of 584; `v4r` made 90 calls, 3 returned errors),
    first token against prefill, and cache.
  - NO-GO, proxy rows: only R72b's runaway task.
  - NO-GO, memory: it reads the rig's `meminfo.log` over every arm in the folder. That file now holds rows from both
    rigs, so its lowest point is not `v4r`'s; `v4r`'s own lowest is the 312 MiB above.

## Per task ([v4r_compare.md](v4r_compare.md))

Minutes against R142a and R142b (rig 1):
- **Quicker than both in 5 tasks:** Python `phone-number` 0.6, `space-age` 0.7, Python `variable-length-quantity`
  1.0 (against 2.2 and 4.4), `list-ops` 1.1 and `scale-generator` 1.4.
- **Between the two runs or equal to one in 4:** `triangle` 0.7, JS `phone-number` 1.3, `grade-school` 0.9 and
  `complex-numbers` 9.7 in 29 turns (against 0.9 and 10.8).
- **Slower than both in 3:** `affine-cipher` 2.7 in 11 turns, `pig-latin` 3.7 (4,569 drafted tokens against 2,373 in
  R142a), and JS `variable-length-quantity` 9.6 in 5 turns (13,197 drafted tokens against 1,815 and 4,815). That task
  wrote long replies, which adds wall-clock minutes; equal work leaves those minutes out.

## Commands

R156a, from the repository root on a machine that only runs ssh. It used R72's client folders, whose bench files R142a
had brought up to date, so no prep step ran. It had its own ports, so that R156b can run on the other rig at the same
time:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4r V4R_DRAFT_UB=512 \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad v4r" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r156_agent_v4r/run/r156.sh > r156a.log 2>&1 < /dev/null &
```

The report rewrites the client folder's `res/turns.csv` with exactly the arms it is given. So `REPORT_ARMS` names every
arm in R72's folder that the tables read; R90's `v3` comes from R90's own folder. The tables were made on the client's
CPU in an empty folder with R142's [tables.sh](../2026-11-13_r142_agent_v4c3/run/tables.sh) (`nice -n 19`), followed
by the cards' peaks:

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> NEWARMS="v4c3 v4c32 v4r" \
  bash <this repository>/docs/data/2026-11-13_r142_agent_v4c3/run/tables.sh
awk -F', ' -v a='2026/11/13 18:15:30' -v b='2026/11/13 18:53:02' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv) | each arm against R90's `v3`: all 12 tasks, R90's 11, and the 12 without `complex-numbers` |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt speed and first token by a turn's new prompt |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's acceptance by task |
| [host.tsv](host.tsv) | lowest MemAvailable and swap of `v4c3`, `v4c32` and `v4r` (each on its own rig) |
| [v4r_compare.md](v4r_compare.md), [v4r_report_table.md](v4r_report_table.md), [v4r_gonogo.md](v4r_gonogo.md) | the agent check's report over R72's, R142's, R154's and R156a's arms |
| [run/r156.sh](run/r156.sh) | the driver: R142's with the arms `v4r` and `v4r2` |

The tasks are the agent check's bug-fix set ([bench/agent_check](../../../bench/agent_check/README.md)); the model files
are the maker's (Swift GSQ-RCO Q2_0) and the MTP draft, as in R142.
