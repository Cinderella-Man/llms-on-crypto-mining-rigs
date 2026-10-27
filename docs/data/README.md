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
