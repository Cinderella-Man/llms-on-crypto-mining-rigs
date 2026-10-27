# Q41: G3 and K2h against UD-Q4_K_XL on BFCL v4 and LiveCodeBench v6

Q41's main runs: does a quant of my own lose task quality against Unsloth's UD-Q4_K_XL, the bar? G3 ran against
UD-Q4_K_XL on BFCL v4 single-turn (E1, 3,477 tasks at two attempts), multi-turn (E2, attempt 1) and one eighth of
LiveCodeBench v6 (E3, 56 problems); K2h ran E1 at one attempt. The decisions are in the registry row Q41
([quants/experiments.md](../../../quants/experiments.md)) and in the harness pages
([BFCL](../../../bench/bfcl_flint/README.md), [LiveCodeBench](../../../bench/livecodebench_flint/README.md)); the
pilot that set K is in [2026-10-23_q41_pilot](../2026-10-23_q41_pilot/README.md). Every per-task verdict is here, so
a new candidate can be paired against UD-Q4_K_XL, G3 or K2h without running them again.

| Lane | Pair | Tasks × attempts | Δ pass@1, points (90 % interval) | Result | Table |
|---|---|---|---|---|---|
| E1 single-turn | G3 − UD-Q4_K_XL | 3,477 × 2 | +0.35 (−0.40 to +1.09) | non-inferior at the 1.5-point margin; format gate passes | `paired.md` |
| E1 single-turn | K2h − UD-Q4_K_XL | 3,477 × 1 | +0.14 (−0.87 to +1.16) | non-inferior; format gate passes | `paired.md` |
| E2 multi-turn, all | G3 − UD-Q4_K_XL | 458 × 1 | +0.87 (−3.23 to +4.98) | open: the lower bound misses the 3-point margin by 0.23; format gate fails (Δ +1.31, upper +4.32) | `paired_e2_combined.md` |
| E2 multi-turn, new scenarios | G3 − UD-Q4_K_XL | 358 × 1 | +3.07 (−1.59 to +7.73) | inside the 3-point margin, but the format gate fails (Δ +1.40, upper +4.81) | `paired.md` |
| E2 miss_param, pre-registered | does G3 pass fewer? | 139 × 1 | +8.6 (+1.8 to +15.4) | no: UD-Q4_K_XL 34, G3 46, one-sided exact McNemar p 0.988 | `missparam_confirmatory/` |
| E3 LiveCodeBench, 32,768-token budget | G3 − UD-Q4_K_XL | 56 × 1 | −5.36 (−10.43 to −0.28) | 3 discordant problems, all hard and all at the budget; exact McNemar p 0.25 | `paired_e3.md` |

- **New scenarios** are the 358 multi-turn tasks outside the pilot's 100. They lean towards miss_param (161 of 358),
  because the pre-registered test drew 139 of them.
- **Ask or act.** In the turn whose ground truth is empty (the user leaves out a value the calls need), "asked" means
  the turn holds no decodable tool call ([ask_rate.py](../../../bench/bfcl_flint/ask_rate.py)). The exploratory 47
  miss_param tasks, picked after seeing the pilot, showed a gap after asking that did not replicate on the 139 fresh
  tasks:

  | Set | File | Conversations | Asked | Question mark | Pass if asked | Pass if acted |
  |---|---|---|---|---|---|---|
  | exploratory, 47 tasks | UD-Q4_K_XL | 46 | 43.5 % (20) | 56.5 % | 75.0 % (15/20) | 34.6 % (9/26) |
  | exploratory, 47 tasks | G3 | 47 | 44.7 % (21) | 51.1 % | 38.1 % (8/21) | 23.1 % (6/26) |
  | pre-registered, 139 tasks | UD-Q4_K_XL | 139 (141 turns) | 36.9 % (52) | 46.1 % | 44.2 % (23/52) | 12.4 % (11/89) |
  | pre-registered, 139 tasks | G3 | 138 (140 turns) | 33.6 % (47) | 48.6 % | 46.8 % (22/47) | 25.8 % (24/93) |

| File | What |
|---|---|
| `e1/per_attempt.csv.gz` | E1: one row per task, file and attempt, 17,385 rows: UD-Q4_K_XL and G3 attempts 1 and 3, K2h attempt 1. Verdict, BFCL's error type, prompt and output tokens, reasoning characters, stop type, format flags (columns of [task_results.py](../../../bench/task_results.py)); gzip, 2.1 MB unpacked |
| `e1/summary.md` | E1 per file: pass@1, malformed and runaway rates, mean output tokens, pass@1 per category |
| `e2/per_attempt.csv` | E2 attempt 1 on the 358 new-scenario tasks, UD-Q4_K_XL and G3 (716 rows); adds steps, forced stops and the largest context |
| `e2/summary.md` | E2 new scenarios per file and category |
| `e2_combined.csv` | the pilot's E2 rows (its file label `ud-q4kxl` renamed `ud`) and the new-scenario rows: 458 tasks, 1,016 rows |
| `e3/per_attempt.csv` | E3 eighth 1: 56 problems × UD-Q4_K_XL and G3, with platform, contest date and LiveCodeBench's error code. `error_type` and `lcb_error` hold only the grader's label (Wrong Answer, Time Limit Exceeded, no code): its message quotes test outputs |
| `e3/summary.md` | E3 per file and difficulty, all problems and those from 2025-02-01 |
| `paired.md` | [paired_tasks.py](../../../bench/paired_tasks.py): E1 (G3 at K = 2, K2h at K = 1) and E2 new scenarios against UD-Q4_K_XL |
| `paired_e2_combined.md` | the same on all 458 E2 tasks |
| `paired_e3.md` | the same on E3 |
| `race.md` | [race_first_finish.py](../../../bench/race_first_finish.py) at k = 2 on E1 |
| `ask_exploratory/` | `ask_rate.md` and the per-turn `ask_turns.csv` on the 47 exploratory miss_param tasks (pilot 25, new slices 1–2: 22) |
| `missparam_confirmatory/` | the same on the 139 pre-registered tasks, with the test |
| `PREREG_missparam.md` | the pre-registration, written before any part of the test ran (file names adapted to this folder) |
| `e2_new_ids.txt` | the 699 multi-turn tasks outside the pilot (799 minus the pilot's 100) |
| `e2_missparam_ids.txt` | the test's 139 miss_param tasks (sha256 `0f3be5e1fc323374…`, as pre-registered) |
| `e2_rest_ids.txt` | the 411 new-scenario tasks that are neither miss_param nor in slices 1–3, for the overall tally |
| `manifests.tsv` | one row per slice (56): the harness's manifest flattened, the model path cut to the file name, dates in UTC |
| `run/main.sh` | the job each slice ran |
| `run/jobs.txt` | every slice as a `main.sh` line, grouped by lane |

Made on 2026-10-23/24 with the harnesses in `bench/` at commit `5e26522` (the first slice at `8623f65`, whose BFCL
client is the same) on the 7-GPU quant machine, in three GPU sets: the V100 pair (`-np 8 -c 262144`), an RTX 3090
with two RTX 3080s (`-np 8 -c 131072` for E1; `-np 4 -c 139264` for E2 and E3), and two RTX 3090s for K2h
(`-np 8 -c 131072`). Files alternate slice by slice on one set, so whatever has finished is a paired sample; pairs that
cross GPU sets show no extra disagreement (13.2 % against 14.3 % on the same set). The server is llama.cpp `d77256999`
with the `LLAMA_NO_MOE_DOWN_RESCALE` switch (build `b11820`), `-ngl 99 -fa on -kvu -b 2048 -ub 512`; G3 and K2h run
with `LLAMA_NO_MOE_DOWN_RESCALE=1`. Sampling is the makers' (temperature 1.0, top-p 0.95, top-k 20, min-p 0), attempt n
at seed n, up to 4,096 tokens per BFCL answer or step and 32,768 per LiveCodeBench sample. No timings were recorded.
Slice 3 of the new scenarios stopped before its first task finished, and UD-Q4_K_XL's E3 eighth 2 has no G3 pair, so
neither is scored.

```bash
export BFCL_ROOT=$PWD/bfcl-v4 LCB_ROOT=$PWD/LiveCodeBench      # bfcl-eval 2026.3.23; LiveCodeBench at 28fef95
cd bench/bfcl_flint
python prepare.py template.jinja e1_tasks.jsonl                  # 3,477 tasks
python ../livecodebench_flint/prepare.py template.jinja problems.jsonl e3_tasks.jsonl   # 450 problems
# the slices: run/jobs.txt, each through run/main.sh (WORK holds the harness, task files and ID lists)

python score.py e1_tasks.jsonl out/e1 ud=raw/e1-ud-a1-p1of2.jsonl ud=raw/e1-ud-a1-p2of2.jsonl ... g3=... k2h=...
python multi_turn.py score out/e2 ud=raw/e2-ud-a1-p1of14.jsonl ... g3=raw/e2-g3-a1-p1of14.jsonl ...
python ../paired_tasks.py --ref ud --bootstrap 2000 out/e1/per_attempt.csv out/e2/per_attempt.csv > out/paired.md
python ../race_first_finish.py --k 2,3 --bootstrap 5000 out/e1/per_attempt.csv out/e2/per_attempt.csv > out/race.md
{ cat PILOT/e2/per_attempt.csv; tail -n +2 out/e2/per_attempt.csv; } | sed 's/,ud-q4kxl,/,ud,/' > out/e2_combined.csv
python ../paired_tasks.py --ref ud --bootstrap 2000 out/e2_combined.csv > out/paired_e2_combined.md
python ask_rate.py out/missparam_confirmatory ud=raw/e2-ud-a1-p1of12.jsonl ... g3=raw/e2-g3-a1-p12of12.jsonl \
  --ids e2_missparam_ids.txt --attempt 1 --pair ud,g3
# E3, graded in a container with --network none
python ../livecodebench_flint/evaluate.py problems.jsonl out/e3 ud=raw/e3-ud-a1-p1of8.jsonl g3=raw/e3-g3-a1-p1of8.jsonl
python ../paired_tasks.py --ref ud --bootstrap 2000 out/e3/per_attempt.csv > out/paired_e3.md
```

`PILOT` is [2026-10-23_q41_pilot](../2026-10-23_q41_pilot/README.md). The exploratory table ran `ask_rate.py` the same
way on the pilot's and new slices 1–2's attempt-1 outputs, without `--ids`.

The BFCL tasks come from the Berkeley Function Calling Leaderboard (BFCL v4, `bfcl-eval` 2026.3.23, Gorilla project,
Apache-2.0) and the E3 problems from LiveCodeBench (`code_generation_lite` release_v6, problems from LeetCode,
AtCoder and Codeforces). Neither's text is included, only task IDs, verdicts and counts; `prepare.py` rebuilds the task
files, and the manifests' `tasks_sha256` checks each slice. Flint's outputs are Apache-2.0; the raw outputs are not
included.
