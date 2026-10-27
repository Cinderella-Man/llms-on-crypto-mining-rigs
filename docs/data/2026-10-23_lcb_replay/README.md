# Q71: LiveCodeBench's own leaderboard runs re-graded by the Flint harness

The check behind the LiveCodeBench harness's "Validation without a model"
([bench/livecodebench_flint/README.md](../../../bench/livecodebench_flint/README.md); registry row Q71 in
[quants/experiments.md](../../../quants/experiments.md)). `evaluate.py --replay` re-graded two of LiveCodeBench's
published leaderboard runs, DeepSeek-R1-0528 (10 samples per problem) and o4-mini (high) (1 sample), over the 454
problems from 2024-08-01 to 2025-04-30: the harness's code extraction must give the submitted code, and its verdicts
must match LiveCodeBench's. These per-sample verdicts are the evidence for the 39 / 10 disagreement count and for
the four problems in [excluded_problems.txt](../../../bench/livecodebench_flint/excluded_problems.txt).

- **First pass,** all 4,994 samples at 12 processes on a 32-core workstation under a load average of 15–40 from other
  jobs: of the two files' 11,605 answers, 11,600 give the submitted code. For R1-0528 the harness fails 205
  samples that LiveCodeBench passed and passes 10 it failed; for o4-mini (high) it fails 10 that LiveCodeBench
  passed. 186 of these 215
  false fails are time limits, which are wall-clock. pass@1 here: R1-0528 68.8 %, o4-mini (high) 78.0 % (454
  problems).
- **Second pass,** the 225 disagreeing samples only, with Python 3.11 at 4 processes on an idle machine: 153 of the
  215 false fails now pass. 72 disagreements remain, all R1-0528's: 62 LiveCodeBench pass / harness fail and 10 the
  other way. 23 of the 62 are float-print cases on the four excluded problems, whose answers count within an error
  tolerance but are graded as exact decimals; that leaves 39 and 10 on the 450 problems.

| File | What |
|---|---|
| `e3_replay/per_attempt.csv` | first pass: one row per problem, submission and sample, 4,994 rows (columns of [task_results.py](../../../bench/task_results.py) plus platform, contest date and LiveCodeBench's error code) |
| `e3_replay/replay_disagreements.tsv` | first pass: the 225 samples where the verdicts differ (submission, problem, sample, LiveCodeBench's verdict, the harness's verdict and error label) |
| `e3_replay/summary.md` | first pass: pass@1 per difficulty (all problems and from 2025-02-01), extraction count, the agreement table |
| `e3_replay311/per_attempt.csv` | second pass over the 225 samples |
| `e3_replay311/replay_disagreements.tsv` | the 72 that still differ |
| `e3_replay311/summary.md` | second pass: the same tables |

In every file the error columns (`error_type`, `lcb_error`, `our_error`) hold only the grader's label: Wrong Answer,
Time Limit Exceeded, Runtime Error, Error during testing, or no code. The grader's message, which quotes expected test
outputs, is left out; `lcb_error_code` is LiveCodeBench's code (−2 wrong answer, −3 time limit, −4 runtime error).
The second pass reads the samples from a file cut down to the disagreeing ones, so its sample numbers count within
that file and do not match the first pass's.

Made on 2026-10-23 with [evaluate.py](../../../bench/livecodebench_flint/evaluate.py)'s `--replay` mode inside a
bubblewrap sandbox without network ([sandbox.sh](../../../bench/livecodebench_flint/sandbox.sh)), LiveCodeBench at
`28fef95` and the problems of `code_generation_lite` release_v6 as written by `fetch.py`:

```bash
export LCB_ROOT=$PWD/LiveCodeBench
python -I fetch.py problems.jsonl                                           # 454 problems
# first pass: each leaderboard run's *_eval_all.json as LiveCodeBench publishes it
python evaluate.py problems.jsonl e3_replay --replay R1-0528=R1-0528.json --replay o4-mini-high=o4-mini-high.json --procs 12
# second pass: the same submissions cut to the disagreeing samples, Python 3.11, idle machine
python evaluate.py problems.jsonl e3_replay311 --replay R1-0528=r1-0528-disagree.json \
  --replay o4-mini-high=o4mini-high-disagree.json --procs 4
```

The problems are LiveCodeBench's (from LeetCode, AtCoder and Codeforces) and the graded code is DeepSeek-R1-0528's and
o4-mini's as LiveCodeBench published it. Only problem IDs, dates, difficulties, verdicts and error labels are included:
no problem statements, test data or code.
