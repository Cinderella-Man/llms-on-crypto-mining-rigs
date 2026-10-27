# Data and method behind every experiment

Every experiment in this repository from 2026-10-22 on has a folder name, `<date>_<experiment>`, and a row below. The
row says what it measured, where its result is written up, and what of its data and method is published:

- **Result page:** the benchmark page, registry row in [quants/experiments.md](../../quants/experiments.md) or research
  note that states the outcome.
- **Data:** a folder here, `docs/data/<date>_<experiment>/`, with curated tables (CSV, TSV, Markdown), a README giving
  the exact commands and provenance, and the cleaned scripts that drove the run in `run/`. Reusable tools live in
  [scripts/](../../scripts/) and [bench/](../../bench/) instead.
- Raw command output (server and build logs, consoles, GPU telemetry, profiler dumps, full agent transcripts) is not
  published; the tables summarize it. Benchmark task text (BFCL, LiveCodeBench) is not republished either: each
  folder gives the task IDs and the command that rebuilds the task file.

Run scripts reach the rig through the SSH alias in `RIG` (default `rig`) and take model, build and work paths from
`MODEL`, `MODELS`, `BIN`, `WORK` and `OUT`. Quality work (quantization, KLD, task checks, training) ran on a separate
machine with 3 × RTX 3090, 2 × V100 32 GB and 2 × RTX 3080; speed is only ever measured on the rig.

## Task-level quality

| Date | Folder | Registry | What it measured | Result | What is here |
|---|---|---|---|---|---|
| 2026-10-23 | [2026-10-23_bfcl](2026-10-23_bfcl/README.md) | Q30 | BFCL v3, 300 tasks, greedy: BF16 and five quants, plus five repeat runs | [quants/README.md](../../quants/README.md); Q30 | the write-up (repeatability, per category, the makers-fast rescale finding), per-task verdicts and decoded calls of 11 runs, task IDs, run scripts |
| 2026-10-23 | [2026-10-23_lcb_replay](2026-10-23_lcb_replay/README.md) | Q71 | LiveCodeBench's R1-0528 and o4-mini (high) leaderboard runs re-graded by the harness, under load and then idle | [bench/livecodebench_flint](../../bench/livecodebench_flint/README.md) | per-sample verdicts of both passes, disagreement lists, summaries |
| 2026-10-23 | [2026-10-23_q41_pilot](2026-10-23_q41_pilot/README.md) | Q41, Q93 | BFCL v4 pilot: UD-Q4_K_XL E1 376 × 5; E2 100 tasks with G3 | [bench/bfcl_flint](../../bench/bfcl_flint/README.md); Q41, Q93 | per-attempt CSVs, paired and race tables, task IDs, manifests, run script |
| 2026-10-24 | [2026-10-24_q41](2026-10-24_q41/README.md) | Q41 | G3 and K2h against UD-Q4_K_XL: BFCL v4 E1 (3,477 tasks), E2 (458), the pre-registered miss_param test, one eighth of LiveCodeBench v6 | Q41; [bench/bfcl_flint](../../bench/bfcl_flint/README.md), [bench/livecodebench_flint](../../bench/livecodebench_flint/README.md) | per-attempt CSVs (E1 gzipped), paired, race and ask tables, the pre-registration, ID lists, slice manifests, run scripts |
| 2026-10-24 | [2026-10-24_q123](2026-10-24_q123/README.md) | Q123 | log p of `</think>` and of hesitation markers per quant, from Q106's dumps | [bench/long_context_kld](../../bench/long_context_kld/README.md); Q123 | three full tables, run and regression scripts |
| 2026-10-24 | [2026-10-24_q94](2026-10-24_q94/README.md) | Q94 | reflection words in Flint's reasoning (Q51 transcripts, BFCL E1 answers) | [bench/reasoning_length](../../bench/reasoning_length/README.md); Q94 | the three JSON outputs of `reflections.py` |

## Drafter, guessers and the answer store

| Date | Folder | Registry | What it measured | Result | What is here |
|---|---|---|---|---|---|
| 2026-10-23 | [2026-10-23_q51](2026-10-23_q51/README.md) | Q51 | distinct experts per layer for 2–16 consecutive Flint tokens (38 transcripts, Q8_0) | [bench/expert_overlap](../../bench/expert_overlap/README.md); Q51 | per-layer `q51.json`, summary, answer lengths, run scripts (prompts and answers: [release/flint-v2/reproduce/data/q51](../../release/flint-v2/reproduce/data/q51/README.md)) |
| 2026-10-23 | [2026-10-23_q64](2026-10-23_q64/README.md) | Q64 | offline replay of n-gram guessers on Q51 at modelled and measured verify costs | [bench/ngram_replay](../../bench/ngram_replay/README.md); Q64 | both full reports |
| 2026-10-24 | [2026-10-24_q96](2026-10-24_q96/README.md) | Q96 | a continuation table behind the copy lookup; Gumbel-coupled guesses | [unconventional_ideas.md](../research/notes/unconventional_ideas.md); Q96 | the full grid, the coupled and coupling-study tables (+ JSON), the logit-dump build and job |
| 2026-10-24 | [2026-10-24_q20_drafter](2026-10-24_q20_drafter/README.md) | Q20 | the staircase drafter's first data round and its licence audit | [docs/drafter.md](../drafter.md) | the audit scripts and results, storage figures; `q51_sources.py` and `rec_weights.py` are in [bench/staircase_drafter](../../bench/staircase_drafter/README.md) |
| 2026-10-24 | [2026-10-24_q20_d3](2026-10-24_q20_d3/README.md) | Q20 | the staircase head in the engine: exactness set, served rig run | [docs/drafter.md](../drafter.md) §6 | per-request guess counts of three runs, the 15 exactness jobs, three rig tests |
| 2026-10-24 | [2026-10-24_q161_datastore](2026-10-24_q161_datastore/README.md) | Q161 | a store of Flint's earlier answers behind the copy lookup, and its licence variants (stores built from the first data round, since deleted) | [pipelined_speculation.md](../research/notes/pipelined_speculation.md); Q161 | score tables (+ JSON), the per-package licence ruling, allow lists, the store's builder, checker and variant scripts |
| 2026-10-25 | — (`2026-10-25_vllm_gen`) | Q20 | vLLM as the drafter's answer writer | [docs/drafter.md](../drafter.md) step 2 | nothing: its numbers are in the drafter page; the answers it wrote were deleted with the first data round |

## Agent checks and the v2 release

| Date | Folder | Registry | What it measured | Result | What is here |
|---|---|---|---|---|---|
| 2026-10-24 | — (agent-check plan, research log, rig runbook) | Q131, Q87–Q157 | nothing measured: the agent check's machine setup, the log of the hourly research rounds, a day's rig plan | [bench/agent_check](../../bench/agent_check/README.md), [research notes](../research/notes/README.md) | nothing: their technical content is in those pages |
| 2026-10-25 | — (`2026-10-25_agent_check_release`) | Q131 | the release check's dry runs against a mock server (no model) | [bench/agent_check](../../bench/agent_check/README.md) | nothing (synthetic output); the machine values as [release.env.example](../../bench/agent_check/release.env.example) |
| 2026-10-25 | [2026-10-25_release_rig](2026-10-25_release_rig/README.md) | Q117, Q150–Q152, Q110, Q157, Q163, Q67 | the release engine on the rig: each v2 switch off alone and together, decode after 32k tokens, the prompt-speed bisect, the card-built mask A/B, Q8_0 pipelined identity, tool-request sampling, GPU time per operation | [engine_features.md](../engine_features.md), [results.md](../../release/flint-v2/results.md) | `llama_bench.tsv` (98 rows), two pipelined tables, `tool_request_q67.tsv`, `op_profile_rc3.tsv`, five run scripts |
| 2026-10-26 | [2026-10-26_release_setup](2026-10-26_release_setup/README.md) | Q131, Q111, Q140 | the release command line chosen on agentic speed: 15 agent-check runs (161 attempts), the 262k fit grid, the server's prompt reading | [results.md](../../release/flint-v2/results.md) | `arms.tsv`, per run the comparison, table, verdicts, attempts and turns; the agent's 161 patches; `fit_prefill.tsv`, `serve_prefill.tsv`; run scripts; the tool [scripts/serve_prefill.sh](../../scripts/serve_prefill.sh) |
