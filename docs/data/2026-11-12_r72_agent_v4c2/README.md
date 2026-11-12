# The agent check of v4's lines on the kernel-merge build (R72, Q1261; 2026-11-12)

R72 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on v4's lines, one arm per run, on one rig of nine P106-090 cards with Flash-Next's v3 preset (Swift Q2_0 and
the MTP draft). Registry row Q1261 in [quants/experiments.md](../../../quants/experiments.md). Every value below is one
run.

- **R72a, `gains`:** R90's line exactly ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md)), now on the
  kernel-merge build `d097d6d88` (`pascal-p106-v4`, tarball `llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern`)
  instead of the day build `18d0fa309`. It is the reference for R72's later arms and a second run of R90's `gains`.

## R72a: `gains` on the kernel-merge build

**At equal work `gains` takes 2.80 minutes a task on the kernel-merge build, against 2.79 in R90 on the day build and
3.08 for v3.0 (R90's `v3`, −9.1 %), with 12 of 12 tasks passed.** On the 11 tasks R90 compared (without
`bugfix-javascript-variable-length-quantity`, which v3.0 ran to its cap) it takes 1.96 minutes a task, as R90's `gains`
did (1.96), against 2.27 for `v3` (−13.7 %). Generation 24.75 tok/s against R90's 24.85, prompts 73.2 against 73.9,
draft acceptance 0.762 against 0.757. The `gains` line has two runs now: 2.79 and 2.80 minutes a task at equal work.

### The run

- **Server line:** Flash-Next's v3 preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env))
  on the server line of v3.0's arm `t1swift`, with R90's `GAINS_ENV` (`GGML_CUDA_FATTN_SPARSE_PASCAL=1
  GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1
  GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1`) and
  `GAINS_ARGS` (`--spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4
  --ctx-checkpoints-incall --ubatch-taper on`, and `--spec-draft-n-ubatch 512` in place of 128). The line as started
  is R90's `gains` line character for character, the binary's folder aside.
- **Timeline:** prep 17:18:02–17:18:36 (12 of 12 tasks valid in the sandbox image); the server started at 17:18:41 and
  answered after 313 s; the 12 tasks 17:23:55–17:50:34; the rig without a llama process and with 0 MiB on every card
  at 17:50:42.
- **Settings:** one attempt a task, a 12-minute cap a task (no task reached it), a 40-minute budget for the arm (not
  reached), the RAM watchdog at 60 MiB. The client (pi, the logging proxy, the tasks' tests) ran in containers on
  another machine's CPU at `--cpu-shares 256`, with no GPU.

### All 12 tasks ([agent.tsv](agent.tsv))

Equal work times one fixed amount of work for every arm: R90's `v3` prompt tokens at the arm's prompt speed, plus its
generated tokens at the arm's generation speed, plus its time outside the server, over its tasks
([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)). `r90v3` and `r90gains` are R90's arms `v3` and
`gains` (rig 1, 2026-11-11) under other names, so that both `gains` runs fit one table
([run/merge_named.py](run/merge_named.py)); their values are R90's.

| Arm | Build | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Turn 1, median s | Draft acceptance | Tokens a step | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `r90v3` | v3.0 `1551f5fe1` | 11/12 | 66 | 65.5 | 20.66 | 6.52, 16.17 | 13.54 | 0.696 | 3.09 | 37.0 | 3.08 | — |
| `r90gains` | day `18d0fa309` | 12/12 | 66 | 73.9 | 24.85 | 4.55, 24.48 | 10.68 | 0.757 | 3.27 | 28.7 | 2.79 | −9.4 % |
| `gains` (R72a) | kernel merge `d097d6d88` | 12/12 | 69 | 73.2 | 24.75 | 4.23, 24.54 | 10.75 | 0.762 | 3.29 | 26.1 | 2.80 | −9.1 % |

### The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 11/11 | 64 | 65.6 | 20.63 | 6.47, 16.15 | 25.0 | 2.27 | — |
| `r90gains` | 11/11 | 61 | 73.8 | 24.85 | 4.38, 22.82 | 24.1 | 1.96 | −13.7 % |
| `gains` (R72a) | 11/11 | 63 | 73.0 | 24.88 | 4.11, 21.06 | 23.5 | 1.96 | −13.7 % |

### Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

| New prompt tokens | `r90v3`: turns, tok/s, median first token s | `r90gains` | `gains` (R72a) |
|---|---|---|---|
| 0–511 | 42, 39.4, 3.57 | 42, 64.4, 2.12 | 44, 61.1, 2.39 |
| 512–1,023 | 11, 61.3, 11.45 | 9, 71.2, 10.78 | 9, 72.5, 10.61 |
| 1,024–1,999 | 10, 90.5, 16.02 | 8, 80.1, 18.26 | 9, 79.0, 18.91 |
| 2,000 and more | 2, 93.3, 25.73 | 7, 76.4, 27.67 | 7, 75.8, 26.73 |

The `gains` line reads turns of 1,024 new tokens or more slower than v3.0 in this run too (79.0 and 75.8 tok/s
against 90.5 and 93.3). That loss is the prompt graph's reallocations under `LLAMA_SAMPLING_TOP_K_ROWS=1` (Q1235,
Q1405); the reference keeps R90's line, that switch included.

### The host and the report

- **Host memory:** the lowest MemAvailable over the arm is 479 MiB (the RAM watchdog's own samples: 478 MiB), and swap
  in use went from 18 to 72 MiB. The watchdog did not act. R90's `gains` reached 415 MiB with no swap growth (24 → 24).
- **The report's lines** ([gains_gonogo.md](gains_gonogo.md), [gains_compare.md](gains_compare.md),
  [gains_report_table.md](gains_report_table.md)): every harness line GO (12 of 12 attempts without an infrastructure
  failure, 69 requests logged, the proxy's first token 0.05 s after the server's prompt time at the median); tool calls
  0 malformed of 63 (1 returned an error, in `bugfix-python-affine-cipher`); median cache hit 96.2 % from the second
  turn on; the server log has no NaN, no full re-processing and 16 checkpoint restores. The memory line is NO-GO by
  the report's rule (at least 600 MB available and at most 16 MB of swap growth), so the overall line is NO-GO, as
  R90's report was.
- **Per task** ([gains_compare.md](gains_compare.md)): every task passed; the slowest was `bugfix-python-affine-cipher`
  (4.8 minutes, 9 turns), the quickest `bugfix-javascript-space-age` (1.0 minutes, 5 turns).

## Commands

The run, from the repository root on a machine that only runs ssh (aliases, paths and ports are the user's own):

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="prep agent report" AGENT_ORDER=gains REPORT_ARMS=gains \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-12_r72_agent_v4c2/run/r72a.sh > r72a.log 2>&1 < /dev/null &
```

The tables, on the client's CPU (`nice -n 19`), from R90's collected agent folder and this run's (`CLIENT_OUT`):

```bash
python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/merge_named.py merged \
  <R90's agent folder>:v3:r90v3 <R90's agent folder>:gains:r90gains <R72's agent folder>:gains:gains
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains
SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py merged/res/turns.csv r90v3 r90gains gains \
  > prompt_buckets.tsv
# the host over the arm: rig/meminfo.log holds epoch, MemAvailable kB, swap used kB
awk 'NF >= 3 && $1 ~ /^[0-9]+$/ { if (m == "" || $2 < m) m = $2; if (s0 == "") s0 = $3; if ($3 > s) s = $3 }
  END { printf "lowest MemAvailable %.0f MiB; swap %.0f -> %.0f MiB\n", m / 1024, s0 / 1024, s / 1024 }' \
  <R72's agent folder>/rig/meminfo.log
```

R72's later arms use the same client folders (`CLIENT_DIR`, `CLIENT_OUT`) with `STEPS="agent report"` and their own
arm names; their tables add their arms to `merge_named.py`'s list, with `r90v3` kept as the reference.

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv) | the agent check per arm, all 12 tasks and the 11 |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt reading by the size of a turn's new prompt |
| [gains_compare.md](gains_compare.md), [gains_report_table.md](gains_report_table.md), [gains_gonogo.md](gains_gonogo.md) | the agent check's own report of R72a's arm (`bench/agent_check/client_report.sh`) |
| [run/r72a.sh](run/r72a.sh) | the run: R90's `run/r90.sh` without its deep-lane step, the report over `REPORT_ARMS` |
| [run/merge_named.py](run/merge_named.py) | several checks' arms in one folder, each under its own name |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the builds and files named above; R90's rows are
R90's own measurements. The prompts are the agent sessions of bench/agent_check's bugfix set; no prompt text is in
this folder. The tables are this repository's measurements, under its licence.
