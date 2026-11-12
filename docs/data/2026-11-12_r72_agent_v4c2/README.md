# The agent check of v4's lines on the kernel-merge build (R72, Q1261; 2026-11-12)

R72 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on v4's lines, one arm per run, on one rig of nine P106-090 cards with Flash-Next's v3 preset (Swift Q2_0, or in
R72c the file `gmix-q2`, and the MTP draft). Registry row Q1261 in [quants/experiments.md](../../../quants/experiments.md).
Every value below is one run; the candidate line `v4c2` has two runs (R72b and R72d), listed each by its own values.

**What the four arms say:** v4's candidate line `v4c2` does R90's `v3` work in 2.44 minutes a task in R72b and 2.37 in
R72d, against 2.80 for the `gains` line and 3.08 for v3.0. Without `bugfix-javascript-complex-numbers`, which R72b ran to
its 12-minute cap and R72d passed in 1.0 minute, its two runs take 2.45 and 2.46 minutes a task. With the better-scoring
model file `gmix-q2` the same line takes 2.61 minutes a task, 7.0 % more than R72b's run, 10.1 % more than R72d's and
15.3 % less than v3.0, and it passed all 12 tasks (`v4c2` 11 of 12 in R72b, 12 of 12 in R72d).

- **R72a, `gains`:** R90's line exactly ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md)), now on the
  kernel-merge build `d097d6d88` (`pascal-p106-v4`, tarball `llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern`)
  instead of the day build `18d0fa309`. It is the reference for R72's later arms and a second run of R90's `gains`.
- **R72b, `v4c2`:** v4's candidate line on the C802 build `8a55934f3` (tarball
  `llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802`, four commits on the kernel merge `d097d6d88`): the `gains`
  line with R60's taper, C691's switch kept now that C802 keeps its graph fixed, and the switches 2026-11-12's rig rows
  kept ([below](#r72b-v4c2-the-v4-candidate-line)).
- **R72c, `v4c2g`:** `v4c2`'s line exactly, on the same build, with the model file `gmix-q2` in place of Swift Q2_0
  (`pmix-q2-fast`): R115's file pick, better on the agent text's KLD (0.236062 against 0.279228) and 16.75 % slower a
  step on v4's line ([2026-11-12_r115_gmix_step](../2026-11-12_r115_gmix_step/README.md)). It gives that cost on whole
  agent tasks ([below](#r72c-v4c2g-the-candidate-line-with-gmix-q2)).
- **R72d, `v4c22`:** `v4c2`'s line exactly, a second run of it on the same build with the same model file and draft;
  only the arm's name differs, so that its results sit beside R72b's ([below](#r72d-v4c22-a-second-run-of-v4c2)).

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

## R72b: `v4c2`, the v4 candidate line

**At equal work `v4c2` takes 2.44 minutes a task, against 2.80 for `gains` (R72a) and 3.08 for v3.0 (R90's `v3`,
−20.8 %), with 11 of 12 tasks passed.** Prompts read at 136.0 tok/s against `gains`' 73.2, generation runs at
26.12 tok/s against 24.75, and the first token of a turn comes after 2.42 s at the median and 9.29 s at p90 (`gains`
4.23 and 24.54). On R90's 11 tasks it takes 1.61 minutes a task, against 1.96 for `gains` and 2.27 for `v3` (−29.1 %).
One task failed: `bugfix-javascript-complex-numbers`, which `gains` passed in 2.0 minutes and 5 turns, ran to its
12-minute cap in 35 turns. Because of that task the arm's wall time is longer than `gains`' (31.3 minutes for the 12
tasks against 26.1); without it, `v4c2` does the other 11 in 19.3 minutes against 24.1.

### The line

- **Build:** `pascal-p106-v4-b12246-c802` `8a55934f3` (`llama-server` md5 `61fb0827bdc0715fc0396153ea0e4501`).
- **Environment:** the v3 preset's six switches, R90's `GAINS_ENV` (`LLAMA_SAMPLING_TOP_K_ROWS=1` included), and
  - C720's and C719's k-pool switches: `LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_POOL_MASK_DEV=1`;
  - R122's eight host switches: `LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1
    LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1`;
  - R123a's three: `GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1`;
  - R120b's three verify-pass kernels: `GGML_CUDA_MMVQ_PASCAL_Q3K_MT=1 GGML_CUDA_MMVQ_PASCAL_GLU_MT=1
    GGML_CUDA_MMVQ_PASCAL_Q6K_MT=1`;
  - FN66's stop on a rejected draft: `LLAMA_SPEC_REJECT_STOP=1`;
  - R123b's draft token table on its layer's card and the sparse prompt rule: `LLAMA_DRAFT_EMBD_DEV=layer
    GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1`.
- **Arguments:** R90's `gains` arguments with `--ubatch-taper short=600:64` in place of `on`, `-b 4096` in place of
  `-b 2048`, and without `-otd token_embd=CUDA4`.
- **Checked at the start:** the server's `switches:` line reads `33 set`, the 33 names and values above, none unknown
  to the build. The line as started is `run/r72b.sh`'s `v4c2` arm: the 33 variables, then `llama-server` with the
  common part, `-b 4096 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md <draft> --spec-type draft-mtp
  --spec-draft-n-max 3 --spec-draft-n-ubatch 512 --no-host --spec-draft-sampling probabilistic --spec-draft-p-min 0
  --ctx-checkpoints-vram 4 --ctx-checkpoints-incall --ubatch-taper short=600:64`.

### The run

- **Timeline:** the server started at 18:12:37 and answered after 313 s; the 12 tasks 18:17:58–18:49:35; the report
  18:49:47–18:49:51; the rig without a llama process and with 0 MiB on every card at 18:49:51.
- **Settings:** R72a's: one attempt a task, a 12-minute cap a task (one task reached it), a 40-minute budget for the
  arm (not reached), the RAM watchdog at 60 MiB, the client on another machine's CPU at `--cpu-shares 256` with no GPU.
  R72a's prep (bench files, sandbox image, the 12 tasks validated) was reused.

### All 12 tasks ([agent.tsv](agent.tsv))

| Arm | Build | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Turn 1, median s | Draft acceptance | Tokens a step | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `r90v3` | v3.0 `1551f5fe1` | 11/12 | 66 | 65.5 | 20.66 | 6.52, 16.17 | 13.54 | 0.696 | 3.09 | 37.0 | 3.08 | — |
| `r90gains` | day `18d0fa309` | 12/12 | 66 | 73.9 | 24.85 | 4.55, 24.48 | 10.68 | 0.757 | 3.27 | 28.7 | 2.79 | −9.4 % |
| `gains` (R72a) | kernel merge `d097d6d88` | 12/12 | 69 | 73.2 | 24.75 | 4.23, 24.54 | 10.75 | 0.762 | 3.29 | 26.1 | 2.80 | −9.1 % |
| `v4c2` (R72b) | C802 `8a55934f3` | 11/12 | 92 | 136.0 | 26.12 | 2.42, 9.29 | 6.65 | 0.726 | 3.18 | 31.3 | 2.44 | −20.8 % |

Equal work prices R90's `v3` work (its prompt tokens, generated tokens and time outside the server) at each arm's
speeds, so the runaway task's 35 turns count only through the speeds they were read and written at. 35 of `v4c2`'s 92
turns are that task's.

### The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 11/11 | 64 | 65.6 | 20.63 | 6.47, 16.15 | 25.0 | 2.27 | — |
| `r90gains` | 11/11 | 61 | 73.8 | 24.85 | 4.38, 22.82 | 24.1 | 1.96 | −13.7 % |
| `gains` (R72a) | 11/11 | 63 | 73.0 | 24.88 | 4.11, 21.06 | 23.5 | 1.96 | −13.7 % |
| `v4c2` (R72b) | 10/11 | 87 | 133.8 | 25.72 | 2.40, 9.49 | 26.3 | 1.61 | −29.1 % |

### Without the runaway task ([agent_nocx.tsv](agent_nocx.tsv))

The same table without `bugfix-javascript-complex-numbers` in any arm (R90's `v3` passed 10 of these 11):

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 10/11 | 59 | 66.6 | 20.92 | 6.52, 16.46 | 34.4 | 3.13 | — |
| `r90gains` | 11/11 | 61 | 74.1 | 24.87 | 4.38, 20.90 | 26.0 | 2.87 | −8.3 % |
| `gains` (R72a) | 11/11 | 64 | 73.2 | 24.80 | 4.17, 21.82 | 24.1 | 2.88 | −8.0 % |
| `v4c2` (R72b) | 11/11 | 57 | 148.7 | 27.48 | 2.99, 8.72 | 19.3 | 2.45 | −21.7 % |

The acceptance and tokens-a-step columns of [agent_nocx.tsv](agent_nocx.tsv) are the whole arm's (the server log's
count over the session), so they do not change when a task is left out; the split per task is below.

### Prompts by the size of a turn's new prompt ([prompt_buckets.tsv](prompt_buckets.tsv))

| New prompt tokens | `r90v3`: turns, tok/s, median first token s | `r90gains` | `gains` (R72a) | `v4c2` (R72b) |
|---|---|---|---|---|
| 0–511 | 42, 39.4, 3.57 | 42, 64.4, 2.12 | 44, 61.1, 2.39 | 56, 92.8, 1.68 |
| 512–1,023 | 11, 61.3, 11.45 | 9, 71.2, 10.78 | 9, 72.5, 10.61 | 15, 112.8, 6.60 |
| 1,024–1,999 | 10, 90.5, 16.02 | 8, 80.1, 18.26 | 9, 79.0, 18.91 | 13, 163.5, 7.97 |
| 2,000 and more | 2, 93.3, 25.73 | 7, 76.4, 27.67 | 7, 75.8, 26.73 | 7, 169.8, 11.69 |

`v4c2` reads every size faster than `v3` and `gains`, and the long-prompt loss of the `gains` line is gone: turns of
1,024–1,999 new tokens at 163.5 tok/s and of 2,000 or more at 169.8, against `v3`'s 90.5 and 93.3 and `gains`' 79.0 and
75.8. (`v4c2` has 91 timed turns: the runaway task's last request was cut by its cap.)

### The runaway task and the draft's acceptance ([task_acceptance.tsv](task_acceptance.tsv))

- **What the agent did:** in `bugfix-javascript-complex-numbers` its 34 tool calls are all well formed (2 returned an
  error), every turn ends in a tool call, and the server log has no NaN. At the cap the task's tests stood at 38 of 40
  passed, 2 failed in dividing a complex number by a real one; the agent's last turns read the test framework's own
  matcher code and wrote a scratch test to trace that comparison. One run cannot tell whether the line or the sampling
  (temperature 1.0) sent it there; R90's `v3` lost a different task to its cap the same way.
- **Acceptance per task**, from the server's metrics before and after each task ([run/task_acceptance.py](run/task_acceptance.py)):
  the runaway task accepted 8,215 of 12,426 drafted tokens (0.661), 37 % of the arm's drafts. The other 11 tasks
  accepted 15,840 of 20,823 (0.761), against 15,272 of 20,010 (0.763) for the same 11 in `gains`. The draft accepts as
  much on `v4c2` as on `gains`; the arm's 0.726 is that one task.

### The host and the report

- **Host memory:** the lowest MemAvailable in the RAM watchdog's samples (`ram_watchdog-v4c2.tsv`) is 392 MiB, at
  18:18:28 during the first task (`rig/meminfo.log` over the arm: 435 MiB). Swap in use went from 19 to 472 MiB, all of
  it between 18:18:28 and 18:26:48 (the first four tasks), and stayed there; MemAvailable was 698 MiB at 18:47:48. The
  watchdog did not act. `gains` (R72a): 479 MiB, swap 18 → 72 MiB.
- **The report's lines** ([v4c2_gonogo.md](v4c2_gonogo.md), [v4c2_compare.md](v4c2_compare.md),
  [v4c2_report_table.md](v4c2_report_table.md), both arms): infrastructure GO (24 of 24 attempts); proxy rows NO-GO
  only for the runaway task (35 rows, 34 replies, one row without timings: its last request, cut by the cap); first
  token against prefill GO; tool calls 0 malformed of 150; median cache hit 97.0 % from the second turn on. The server
  log of `v4c2` has 17 checkpoint restores, 1 checkpoint erased, no NaN and one full re-processing: the first request
  of `bugfix-python-pig-latin`, right after the runaway task's context of more than 25,000 tokens, shared 1,553 tokens
  with it and found no checkpoint below that, so it read all 2,324 tokens (in `gains` 775 new and 1,549 from the
  cache). The memory line is NO-GO by the report's rule (at least 600 MB available, at most
  16 MB of swap growth), so the overall line is NO-GO, as in R72a and R90.
- **Per task** ([v4c2_compare.md](v4c2_compare.md)): 9 of the 11 passed tasks were quicker than in `gains`
  (`bugfix-python-affine-cipher` 2.5 minutes against 4.8); `bugfix-python-list-ops` (2.3 against 1.9) and
  `bugfix-javascript-variable-length-quantity` (5.0 against 2.6, 5 turns each, 6,726 drafted tokens against 2,004)
  were slower.

## R72c: `v4c2g`, the candidate line with `gmix-q2`

**At equal work `v4c2g` takes 2.61 minutes a task, against 2.44 for `v4c2` (R72b, +7.0 %), 2.80 for `gains` (R72a) and
3.08 for v3.0 (R90's `v3`, −15.3 %), with 12 of 12 tasks passed.** The file `gmix-q2` costs on both sides of a turn:
prompts read at 105.6 tok/s against `v4c2`'s 136.0, generation runs at 24.63 tok/s against 26.12, and the first token
of a turn comes after 4.42 s at the median and 10.84 s at p90 (`v4c2` 2.42 and 9.29). On R90's 11 tasks it takes 1.77
minutes a task, against 1.61 for `v4c2` (+9.9 %) and 2.27 for `v3`; without `bugfix-javascript-complex-numbers` 2.70
against 2.45 (+10.2 %). From its step pair R115 had put the cost at est. +14.7 % of an agent task.

`bugfix-javascript-complex-numbers`, which `v4c2` ran to its 12-minute cap in 35 turns, passed in 1.5 minutes and 5
turns. The arm took 65 turns (`v4c2` 92, 35 of them that task's; 60 against 57 without it) and 23.0 minutes for the 12
tasks, the shortest wall time of R72's arms. One run cannot tell whether the file's better KLD or the sampling passed
that task.

### The line

- **Build:** `pascal-p106-v4-b12246-c802` `8a55934f3` (`llama-server` md5 `61fb0827bdc0715fc0396153ea0e4501`), as R72b.
- **Model:** `gmix-q2`'s first shard, `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf`
  (the server log's `load_model` names it), with the MTP draft as in R72a and R72b.
- **Environment and arguments:** `v4c2`'s, unchanged ([above](#the-line)). The arm `v4c2g` in
  [run/r72c.sh](run/r72c.sh) is `v4c2`'s case; the only difference is `MODEL`.
- **Checked at the start:** the server's `switches:` line is R72b's character for character (`33 set`), and the
  started command is R72b's apart from the time and the arm's name.

### The run

- **Timeline:** the server started at 19:03:22 and answered after 351 s (`v4c2` 313 s); the 12 tasks
  19:09:17–19:32:39; the report 19:32:47–19:32:51; the rig without a llama process and with 0 MiB on every card at
  19:32:51.
- **Settings:** R72a's and R72b's (one attempt a task, a 12-minute cap a task, reached by none, a 40-minute budget for
  the arm, not reached, the RAM watchdog at 60 MiB, the client on another machine's CPU at `--cpu-shares 256` with no
  GPU, R72a's prep reused).
- **Fit:** the cards' peaks over the arm are `v4c2`'s within 220 MiB: card 3 5,211 MiB (`v4c2` 5,175), card 4 4,493
  (4,275), card 1 4,955 (5,159), card 8 5,779 (5,783), the rest within 70 MiB.

### All 12 tasks ([agent.tsv](agent.tsv))

| Arm | Build | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Turn 1, median s | Draft acceptance | Tokens a step | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `r90v3` | v3.0 `1551f5fe1` | 11/12 | 66 | 65.5 | 20.66 | 6.52, 16.17 | 13.54 | 0.696 | 3.09 | 37.0 | 3.08 | — |
| `r90gains` | day `18d0fa309` | 12/12 | 66 | 73.9 | 24.85 | 4.55, 24.48 | 10.68 | 0.757 | 3.27 | 28.7 | 2.79 | −9.4 % |
| `gains` (R72a) | kernel merge `d097d6d88` | 12/12 | 69 | 73.2 | 24.75 | 4.23, 24.54 | 10.75 | 0.762 | 3.29 | 26.1 | 2.80 | −9.1 % |
| `v4c2` (R72b) | C802 `8a55934f3` | 11/12 | 92 | 136.0 | 26.12 | 2.42, 9.29 | 6.65 | 0.726 | 3.18 | 31.3 | 2.44 | −20.8 % |
| `v4c2g` (R72c, `gmix-q2`) | C802 `8a55934f3` | 12/12 | 65 | 105.6 | 24.63 | 4.42, 10.84 | 8.03 | 0.747 | 3.24 | 23.0 | 2.61 | −15.3 % |

### The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 11/11 | 64 | 65.6 | 20.63 | 6.47, 16.15 | 25.0 | 2.27 | — |
| `r90gains` | 11/11 | 61 | 73.8 | 24.85 | 4.38, 22.82 | 24.1 | 1.96 | −13.7 % |
| `gains` (R72a) | 11/11 | 63 | 73.0 | 24.88 | 4.11, 21.06 | 23.5 | 1.96 | −13.7 % |
| `v4c2` (R72b) | 10/11 | 87 | 133.8 | 25.72 | 2.40, 9.49 | 26.3 | 1.61 | −29.1 % |
| `v4c2g` (R72c) | 11/11 | 60 | 104.6 | 24.55 | 4.41, 10.84 | 21.2 | 1.77 | −22.0 % |

### Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 10/11 | 59 | 66.6 | 20.92 | 6.52, 16.46 | 34.4 | 3.13 | — |
| `r90gains` | 11/11 | 61 | 74.1 | 24.87 | 4.38, 20.90 | 26.0 | 2.87 | −8.3 % |
| `gains` (R72a) | 11/11 | 64 | 73.2 | 24.80 | 4.17, 21.82 | 24.1 | 2.88 | −8.0 % |
| `v4c2` (R72b) | 11/11 | 57 | 148.7 | 27.48 | 2.99, 8.72 | 19.3 | 2.45 | −21.7 % |
| `v4c2g` (R72c) | 11/11 | 60 | 102.1 | 24.70 | 4.51, 10.84 | 21.5 | 2.70 | −13.7 % |

Over the same 11 tasks `v4c2g` generates at 24.70 tok/s against `v4c2`'s 27.48 (−10.1 %). R115 measured the step
+16.75 % on v4's line; the draft's acceptance here is 0.750 against 0.761 ([below](#r72cs-prompts-and-drafts)), and
the tokens a step of the two arms' whole sessions are 3.24 and 3.18.

### R72c's prompts and drafts

**Prompts by the size of a turn's new prompt** ([prompt_buckets.tsv](prompt_buckets.tsv)): `gmix-q2` reads every size
slower than Swift Q2_0 on the same line, and faster than `gains` and v3.0.

| New prompt tokens | `r90v3`: turns, tok/s, median first token s | `gains` (R72a) | `v4c2` (R72b) | `v4c2g` (R72c) | `v4c2g` against `v4c2`, tok/s |
|---|---|---|---|---|---:|
| 0–511 | 42, 39.4, 3.57 | 44, 61.1, 2.39 | 56, 92.8, 1.68 | 41, 73.0, 1.96 | −21.3 % |
| 512–1,023 | 11, 61.3, 11.45 | 9, 72.5, 10.61 | 15, 112.8, 6.60 | 11, 97.4, 7.48 | −13.7 % |
| 1,024–1,999 | 10, 90.5, 16.02 | 9, 79.0, 18.91 | 13, 163.5, 7.97 | 9, 129.3, 10.11 | −20.9 % |
| 2,000 and more | 2, 93.3, 25.73 | 7, 75.8, 26.73 | 7, 169.8, 11.69 | 4, 131.6, 16.02 | −22.5 % |

R115 read an 8,128-token prompt 17.43 % slower with `gmix-q2` on v4's line.

**The draft's acceptance per task** ([task_acceptance.tsv](task_acceptance.tsv)): `v4c2g` accepted 15,599 of 20,871
drafted tokens (0.747); without `bugfix-javascript-complex-numbers` 14,971 of 19,959 (0.750), against 15,840 of 20,823
(0.761) for `v4c2` and 15,272 of 20,010 (0.763) for `gains` over the same 11 tasks. Its own `complex-numbers` run
accepted 628 of 912 (0.689).

### The host and the report

- **Host memory:** the lowest MemAvailable in the RAM watchdog's samples (`ram_watchdog-v4c2g.tsv`) is 337 MiB, at
  19:09:53 during the first task (`rig/meminfo.log` over the arm: 338 MiB). Swap in use went from 19 to 713 MiB: past
  100 MiB at 19:11:45 (the second task) and at 713 MiB from 19:28:24 (the eighth) to the end; MemAvailable was
  980 MiB at the last sample. The watchdog did not act. `v4c2` (R72b): 392 MiB, swap 19 → 472 MiB.
- **The report's lines** ([v4c2g_gonogo.md](v4c2g_gonogo.md), [v4c2g_compare.md](v4c2g_compare.md),
  [v4c2g_report_table.md](v4c2g_report_table.md), the three arms `gains`, `v4c2` and `v4c2g`): infrastructure GO (36 of
  36 attempts); proxy rows NO-GO only for `v4c2`'s runaway task, as in R72b; first token against prefill GO; tool calls
  0 malformed of 215; median cache hit 96.7 % from the second turn on. The server log of `v4c2g` has 15 checkpoint
  restores, none erased, no full re-processing and no NaN. The memory line is NO-GO by the report's rule (at least
  600 MB available, at most 16 MB of swap growth), so the overall line is NO-GO, as in R72a, R72b and R90.
- **Per task against `v4c2`** ([v4c2g_compare.md](v4c2g_compare.md)): 4 of the 12 were quicker
  (`bugfix-javascript-complex-numbers` 1.5 minutes against 12.0, `bugfix-javascript-variable-length-quantity` 1.8
  against 5.0, `bugfix-python-list-ops` 1.4 against 2.3, `bugfix-python-affine-cipher` 2.4 against 2.5) and 8 slower,
  the most `bugfix-python-scale-generator` (5.4 against 2.0, 6 turns against 5, 6,537 drafted tokens against 2,493:
  this run wrote more) and `bugfix-javascript-phone-number` (1.8 against 0.9, 7 turns against 5, 1 tool call returned
  an error).

## R72d: `v4c22`, a second run of `v4c2`

**In its second run v4's candidate line takes 2.37 minutes a task at equal work, with 12 of 12 tasks passed; its first
run (R72b) took 2.44, with 11 of 12.** `bugfix-javascript-complex-numbers`, which R72b ran to its 12-minute cap in 35
turns, passed in 1.0 minute and 5 turns. Without that task the two runs take 2.45 (R72b) and 2.46 (R72d) minutes a task
at equal work, against 2.88 for `gains` and 3.13 for v3.0. On R90's 11 tasks R72d takes 1.51 minutes a task (R72b 1.61,
`v3` 2.27, −33.5 %). Prompts read at 147.6 tok/s (R72b 136.0), generation runs at 27.53 tok/s (26.12), and the first
token of a turn comes after 2.81 s at the median and 8.58 s at p90 (2.42 and 9.29).

### The two runs of `v4c2`, each by its values

| | R72b `v4c2` | R72d `v4c22` |
|---|---|---|
| Passed | 11/12 | 12/12 |
| Equal work, min a task, all 12 tasks | 2.44 | 2.37 |
| Equal work, min a task, R90's 11 tasks | 1.61 | 1.51 |
| Equal work, min a task, without `bugfix-javascript-complex-numbers` | 2.45 | 2.46 |
| Turns (without `complex-numbers`) | 92 (57) | 65 (60) |
| Prompt tok/s (without `complex-numbers`) | 136.0 (148.7) | 147.6 (144.0) |
| Generation tok/s (without `complex-numbers`) | 26.12 (27.48) | 27.53 (27.52) |
| First token per turn, median and p90, s | 2.42, 9.29 | 2.81, 8.58 |
| Draft acceptance (without `complex-numbers`) | 0.726 (0.761) | 0.753 (0.751) |
| Tokens a step | 3.18 | 3.26 |
| Minutes, all 12 tasks (without `complex-numbers`) | 31.3 (19.3) | 19.9 (18.8) |
| `bugfix-javascript-complex-numbers` | ran to its cap: 12.0 min, 35 turns | passed: 1.0 min, 5 turns |
| Lowest MemAvailable; swap in use | 392 MiB; 19 → 472 MiB | 365 MiB; 20 → 20 MiB |

Where neither run has a runaway task the line repeats: 2.45 and 2.46 minutes a task at equal work, generation at 27.48
and 27.52 tok/s. The 12-task numbers differ more (2.44 and 2.37), because R72b's speeds include the runaway task's 35
turns.

### The line

- **Build, model and draft:** R72b's: `pascal-p106-v4-b12246-c802` `8a55934f3` (`llama-server` md5
  `61fb0827bdc0715fc0396153ea0e4501`), Swift Q2_0's first shard (`pmix-q2-fast`; the server log's `load_model` names it)
  and the MTP draft.
- **Environment and arguments:** `v4c2`'s, unchanged ([above](#the-line)). The arm `v4c22` in
  [run/r72d.sh](run/r72d.sh) is `v4c2`'s case; only the arm's name differs.
- **Checked at the start:** the server's `switches:` line is R72b's character for character (`33 set`), and the
  started command is R72b's apart from the time and the arm's name.

### The run

- **Timeline:** the server started at 19:42:58 and answered after 311 s (R72b 313 s); the 12 tasks 19:48:15–20:08:34;
  the report 20:08:44–20:08:47; the rig without a llama process and with 0 MiB on every card at 20:08:47.
- **Settings:** R72a's to R72c's (one attempt a task, a 12-minute cap a task, reached by none, a 40-minute budget for
  the arm, not reached, the RAM watchdog at 60 MiB, the client on another machine's CPU at `--cpu-shares 256` with no
  GPU, R72a's prep reused). That machine ran other jobs at the same time (load average 12.08 at 19:42); equal work does
  not use the arm's own time outside the server.
- **Fit:** the cards' peaks over the arm are R72b's within 2 MiB: 0:5,115 1:5,159 2:5,017 3:5,175 4:4,273 5:5,033
  6:4,427 7:5,045 8:5,783 MiB.

### All 12 tasks ([agent.tsv](agent.tsv))

| Arm | Build | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Turn 1, median s | Draft acceptance | Tokens a step | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| `r90v3` | v3.0 `1551f5fe1` | 11/12 | 66 | 65.5 | 20.66 | 6.52, 16.17 | 13.54 | 0.696 | 3.09 | 37.0 | 3.08 | — |
| `r90gains` | day `18d0fa309` | 12/12 | 66 | 73.9 | 24.85 | 4.55, 24.48 | 10.68 | 0.757 | 3.27 | 28.7 | 2.79 | −9.4 % |
| `gains` (R72a) | kernel merge `d097d6d88` | 12/12 | 69 | 73.2 | 24.75 | 4.23, 24.54 | 10.75 | 0.762 | 3.29 | 26.1 | 2.80 | −9.1 % |
| `v4c2` (R72b) | C802 `8a55934f3` | 11/12 | 92 | 136.0 | 26.12 | 2.42, 9.29 | 6.65 | 0.726 | 3.18 | 31.3 | 2.44 | −20.8 % |
| `v4c22` (R72d, `v4c2` again) | C802 `8a55934f3` | 12/12 | 65 | 147.6 | 27.53 | 2.81, 8.58 | 6.67 | 0.753 | 3.26 | 19.9 | 2.37 | −23.1 % |
| `v4c2g` (R72c, `gmix-q2`) | C802 `8a55934f3` | 12/12 | 65 | 105.6 | 24.63 | 4.42, 10.84 | 8.03 | 0.747 | 3.24 | 23.0 | 2.61 | −15.3 % |

### The 11 tasks R90 compared ([agent_11.tsv](agent_11.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 11/11 | 64 | 65.6 | 20.63 | 6.47, 16.15 | 25.0 | 2.27 | — |
| `r90gains` | 11/11 | 61 | 73.8 | 24.85 | 4.38, 22.82 | 24.1 | 1.96 | −13.7 % |
| `gains` (R72a) | 11/11 | 63 | 73.0 | 24.88 | 4.11, 21.06 | 23.5 | 1.96 | −13.7 % |
| `v4c2` (R72b) | 10/11 | 87 | 133.8 | 25.72 | 2.40, 9.49 | 26.3 | 1.61 | −29.1 % |
| `v4c22` (R72d) | 11/11 | 59 | 145.9 | 27.42 | 2.81, 8.58 | 14.8 | 1.51 | −33.5 % |
| `v4c2g` (R72c) | 11/11 | 60 | 104.6 | 24.55 | 4.41, 10.84 | 21.2 | 1.77 | −22.0 % |

### Without `bugfix-javascript-complex-numbers` ([agent_nocx.tsv](agent_nocx.tsv))

| Arm | Passed | Turns | Prompt tok/s | Generation tok/s | First token per turn, median and p90, s | Minutes, all tasks | Equal work, min a task | Against `r90v3` |
|---|---|---:|---:|---:|---|---:|---:|---:|
| `r90v3` | 10/11 | 59 | 66.6 | 20.92 | 6.52, 16.46 | 34.4 | 3.13 | — |
| `r90gains` | 11/11 | 61 | 74.1 | 24.87 | 4.38, 20.90 | 26.0 | 2.87 | −8.3 % |
| `gains` (R72a) | 11/11 | 64 | 73.2 | 24.80 | 4.17, 21.82 | 24.1 | 2.88 | −8.0 % |
| `v4c2` (R72b) | 11/11 | 57 | 148.7 | 27.48 | 2.99, 8.72 | 19.3 | 2.45 | −21.7 % |
| `v4c22` (R72d) | 11/11 | 60 | 144.0 | 27.52 | 2.89, 8.58 | 18.8 | 2.46 | −21.4 % |
| `v4c2g` (R72c) | 11/11 | 60 | 102.1 | 24.70 | 4.51, 10.84 | 21.5 | 2.70 | −13.7 % |

Against R72d's run `v4c2g` (`gmix-q2`) takes 10.1 % more at equal work on all 12 tasks (2.61 against 2.37), 17.2 % more
on R90's 11 (1.77 against 1.51) and 9.8 % more without `complex-numbers` (2.70 against 2.46).

### `bugfix-javascript-complex-numbers`

R72d passed it in 1.0 minute: 5 turns, 5 tool calls, none returned an error, the draft accepting 618 of 783 drafted
tokens (0.789). In R72b the same build, model, switches and command ran it to its cap in 35 turns. R72's other arms
passed it too, each in 5 turns (`gains` 2.0 minutes, `v4c2g` 1.5). So the line does not send this task into a loop
every time; at temperature 1.0 its two runs took different paths through it.

### R72d's prompts and drafts

**Prompts by the size of a turn's new prompt** ([prompt_buckets.tsv](prompt_buckets.tsv)):

| New prompt tokens | `r90v3`: turns, tok/s, median first token s | `gains` (R72a) | `v4c2` (R72b) | `v4c22` (R72d) |
|---|---|---|---|---|
| 0–511 | 42, 39.4, 3.57 | 44, 61.1, 2.39 | 56, 92.8, 1.68 | 41, 89.3, 1.70 |
| 512–1,023 | 11, 61.3, 11.45 | 9, 72.5, 10.61 | 15, 112.8, 6.60 | 10, 126.6, 6.57 |
| 1,024–1,999 | 10, 90.5, 16.02 | 9, 79.0, 18.91 | 13, 163.5, 7.97 | 8, 187.7, 7.91 |
| 2,000 and more | 2, 93.3, 25.73 | 7, 75.8, 26.73 | 7, 169.8, 11.69 | 6, 185.2, 9.48 |

Both runs of the line read every size faster than `gains` and v3.0: 1,024–1,999 new tokens at 163.5 and 187.7 tok/s,
2,000 or more at 169.8 and 185.2, against `gains`' 79.0 and 75.8. R72b's long turns include the runaway task's, read
into a context that grew past 25,000 tokens.

**The draft's acceptance per task** ([task_acceptance.tsv](task_acceptance.tsv)): `v4c22` accepted 15,890 of 21,108
drafted tokens (0.753); without `bugfix-javascript-complex-numbers` 15,272 of 20,325 (0.751), against 15,840 of 20,823
(0.761) for R72b's run and 15,272 of 20,010 (0.763) for `gains` over the same 11 tasks.

### The host and the report

- **Host memory:** the lowest MemAvailable in the RAM watchdog's samples (`ram_watchdog-v4c22.tsv`) is 365 MiB, at
  20:08:16 in the last task (`rig/meminfo.log` over the arm: 366 MiB). MemAvailable fell from 1,473 MiB at 19:47:40,
  near the load's end, to 366 MiB at 19:48:51 in the first task, and from 19:49:00 to the end it stayed between 365 and
  399 MiB. Swap in use stayed at 20 MiB from the start to the end. The watchdog did not act. R72b's run: 392 MiB,
  swap 19 → 472 MiB.
- **The report's lines** ([v4c22_gonogo.md](v4c22_gonogo.md), [v4c22_compare.md](v4c22_compare.md),
  [v4c22_report_table.md](v4c22_report_table.md), the four arms `gains`, `v4c2`, `v4c22` and `v4c2g`): infrastructure
  GO (48 of 48 attempts); proxy rows NO-GO only for R72b's runaway task; first token against prefill GO; tool calls 0
  malformed of 276 (`v4c22`: 61, 2 returned an error); median cache hit 96.8 % from the second turn on. The server log
  of `v4c22` has 13 checkpoint restores, none erased, no full re-processing and no NaN. The memory line reads
  `rig/meminfo.log` over all four arms, so its 338 MB and 18 → 713 MB are R72c's arm; it is NO-GO by the report's rule
  (at least 600 MB available, at most 16 MB of swap growth), and so is the overall line, as in R72a to R72c and R90.
- **Per task against R72b's run** ([v4c22_compare.md](v4c22_compare.md)): 5 of the 12 were quicker
  (`bugfix-javascript-complex-numbers` 1.0 minute against 12.0, `bugfix-python-list-ops` 1.3 against 2.3,
  `bugfix-python-variable-length-quantity` 1.1 against 1.7, `bugfix-python-affine-cipher` 2.4 against 2.5,
  `bugfix-python-scale-generator` 1.9 against 2.0), 3 took the same time to a tenth of a minute (`space-age`,
  `triangle`, `grade-school`), and 4 were slower: `bugfix-python-pig-latin` 2.3 against 1.4 (9 turns against 5, one
  tool call returned an error), `bugfix-python-phone-number` 1.3 against 1.1, `bugfix-javascript-phone-number` 1.1
  against 0.9, and `bugfix-javascript-variable-length-quantity` 5.1 against 5.0. That last task is slow in both runs of
  the line, with 6,900 and 6,726 drafted tokens against 2,004 in `gains` and 1,725 in `v4c2g`.

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

R72b, the arm `v4c2` in R72a's client folders (the same ports and `CLIENT_DIR`, `CLIENT_OUT`; prep not repeated):

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4c2 REPORT_ARMS="gains v4c2" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-12_r72_agent_v4c2/run/r72b.sh > r72b.log 2>&1 < /dev/null &
```

Its tables, on the client's CPU (`nice -n 19`):

```bash
python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/merge_named.py merged \
  <R90's agent folder>:v3:r90v3 <R90's agent folder>:gains:r90gains \
  <R72's agent folder>:gains:gains <R72's agent folder>:v4c2:v4c2
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2
SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2
SKIP_TASKS=bugfix-javascript-complex-numbers OUT_TAG=_nocx \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py merged/res/turns.csv r90v3 r90gains gains v4c2 \
  > prompt_buckets.tsv
SKIP_TASKS=bugfix-javascript-complex-numbers python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/task_acceptance.py \
  <R72's agent folder>/res gains v4c2 > task_acceptance.tsv
# the lowest MemAvailable and swap in the RAM watchdog's samples (epoch, avail_mib, cma_free_mib, usable_mib, swap_used_mib, ...)
awk -F'\t' '!/^#/ && NF >= 5 { if (m == "" || $2 < m) m = $2; if (s0 == "") s0 = $5; if ($5 > s) s = $5 }
  END { print "lowest MemAvailable " m " MiB; swap " s0 " -> " s " MiB" }' <R72's agent folder>/rig/ram_watchdog-v4c2.tsv
```

R72c, the arm `v4c2g` in the same client folders: R72b's call with `run/r72c.sh`, `gmix-q2`'s first shard as `MODEL`
and the three arms in the report:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-gmix-q2-00001-of-00003.gguf \
  DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4c2g REPORT_ARMS="gains v4c2 v4c2g" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-12_r72_agent_v4c2/run/r72c.sh > r72c.log 2>&1 < /dev/null &
```

Its tables, on the client's CPU (`nice -n 19`): R72b's commands with `v4c2g` added to every arm list, and to
`merge_named.py`'s as `<R72's agent folder>:v4c2g:v4c2g`:

```bash
python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/merge_named.py merged \
  <R90's agent folder>:v3:r90v3 <R90's agent folder>:gains:r90gains \
  <R72's agent folder>:gains:gains <R72's agent folder>:v4c2:v4c2 <R72's agent folder>:v4c2g:v4c2g
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2 v4c2g
SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2 v4c2g
SKIP_TASKS=bugfix-javascript-complex-numbers OUT_TAG=_nocx \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged r90v3 r90gains gains v4c2 v4c2g
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py merged/res/turns.csv r90v3 r90gains gains v4c2 v4c2g \
  > prompt_buckets.tsv
SKIP_TASKS=bugfix-javascript-complex-numbers python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/task_acceptance.py \
  <R72's agent folder>/res gains v4c2 v4c2g > task_acceptance.tsv
awk -F'\t' '!/^#/ && NF >= 5 { if (m == "" || $2 < m) m = $2; if (s0 == "") s0 = $5; if ($5 > s) s = $5 }
  END { print "lowest MemAvailable " m " MiB; swap " s0 " -> " s " MiB" }' <R72's agent folder>/rig/ram_watchdog-v4c2g.tsv
# the cards' peaks over an arm: gpu.csv holds time, card, temperature, power, clock, memory used, load
awk -F', ' -v a='2026/11/12 19:03:22' -v b='2026/11/12 19:32:51' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's agent folder>/rig/gpu.csv
```

R72d, the arm `v4c22` (a second run of `v4c2`) in the same client folders: R72b's call with `run/r72d.sh` and the four
arms in the report:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261112-8a55934f3-pascal-p106-v4-b12246-c802 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4c22 REPORT_ARMS="gains v4c2 v4c22 v4c2g" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-12_r72_agent_v4c2/run/r72d.sh > r72d.log 2>&1 < /dev/null &
```

Its tables, on the client's CPU (`nice -n 19`): R72c's commands with `v4c22` added after `v4c2` in every arm list, and
to `merge_named.py`'s as `<R72's agent folder>:v4c22:v4c22`:

```bash
A="r90v3 r90gains gains v4c2 v4c22 v4c2g"
python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/merge_named.py merged \
  <R90's agent folder>:v3:r90v3 <R90's agent folder>:gains:r90gains <R72's agent folder>:gains:gains \
  <R72's agent folder>:v4c2:v4c2 <R72's agent folder>:v4c22:v4c22 <R72's agent folder>:v4c2g:v4c2g
python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged $A
SKIP_TASKS=bugfix-javascript-variable-length-quantity OUT_TAG=_11 \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged $A
SKIP_TASKS=bugfix-javascript-complex-numbers OUT_TAG=_nocx \
  python3 -I docs/data/2026-11-05_keepalive/run/agent_tables.py merged $A
python3 -I docs/data/2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py merged/res/turns.csv $A > prompt_buckets.tsv
SKIP_TASKS=bugfix-javascript-complex-numbers python3 -I docs/data/2026-11-12_r72_agent_v4c2/run/task_acceptance.py \
  <R72's agent folder>/res gains v4c2 v4c22 v4c2g > task_acceptance.tsv
awk -F'\t' '!/^#/ && NF >= 5 { if (m == "" || $2 < m) m = $2; if (s0 == "") s0 = $5; if ($5 > s) s = $5 }
  END { print "lowest MemAvailable " m " MiB; swap " s0 " -> " s " MiB" }' <R72's agent folder>/rig/ram_watchdog-v4c22.tsv
awk -F', ' -v a='2026/11/12 19:42:58' -v b='2026/11/12 20:08:47' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's agent folder>/rig/gpu.csv
```

## Files

| File | Content |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv) | the agent check per arm (`r90v3`, `r90gains`, `gains`, `v4c2`, `v4c22`, `v4c2g`), all 12 tasks and R90's 11 |
| [agent_nocx.tsv](agent_nocx.tsv) | the same without `bugfix-javascript-complex-numbers` (R72b's runaway task) |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt reading by the size of a turn's new prompt, the six arms |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's accepted and drafted tokens per task, `gains`, `v4c2`, `v4c22` and `v4c2g`, with each arm's total and its total without the runaway task |
| [gains_compare.md](gains_compare.md), [gains_report_table.md](gains_report_table.md), [gains_gonogo.md](gains_gonogo.md) | the agent check's own report of R72a's arm (`bench/agent_check/client_report.sh`) |
| [v4c2_compare.md](v4c2_compare.md), [v4c2_report_table.md](v4c2_report_table.md), [v4c2_gonogo.md](v4c2_gonogo.md) | the agent check's own report of R72b's run, `gains` and `v4c2` side by side |
| [v4c2g_compare.md](v4c2g_compare.md), [v4c2g_report_table.md](v4c2g_report_table.md), [v4c2g_gonogo.md](v4c2g_gonogo.md) | the agent check's own report of R72c's run, `gains`, `v4c2` and `v4c2g` side by side |
| [v4c22_compare.md](v4c22_compare.md), [v4c22_report_table.md](v4c22_report_table.md), [v4c22_gonogo.md](v4c22_gonogo.md) | the agent check's own report of R72d's run, `gains`, `v4c2`, `v4c22` and `v4c2g` side by side |
| [run/r72a.sh](run/r72a.sh) | R72a's run: R90's `run/r90.sh` without its deep-lane step, the report over `REPORT_ARMS` |
| [run/r72b.sh](run/r72b.sh) | R72b's run: `r72a.sh` with the arm `v4c2` |
| [run/r72c.sh](run/r72c.sh) | R72c's run: `r72b.sh` with the arm `v4c2g` (`v4c2`'s line; the model file through `MODEL`) |
| [run/r72d.sh](run/r72d.sh) | R72d's run: `r72b.sh` with the arm `v4c22` (`v4c2`'s line again, under its own name) |
| [run/merge_named.py](run/merge_named.py) | several checks' arms in one folder, each under its own name |
| [run/task_acceptance.py](run/task_acceptance.py) | the draft's acceptance per task from the server's metrics before and after each task |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the builds and files named above; R90's rows are
R90's own measurements. The prompts are the agent sessions of bench/agent_check's bugfix set; no prompt text is in
this folder. The tables are this repository's measurements, under its licence.
