# Q41 pilot: BFCL v4 at five sampled attempts per task

The pilot of Q41's task-level checks (registry row Q41 in [quants/experiments.md](../../../quants/experiments.md),
method in the [BFCL harness's pilot plan](../../../bench/bfcl_flint/README.md#pilot-plan-q41)). UD-Q4_K_XL ran a
stratified set of 376 BFCL v4 single-turn tasks (E1) at five sampled attempts each, and 100 multi-turn tasks (E2) at two
attempts; G3 ran the same 100 multi-turn tasks once. The pilot set K for the main runs
([2026-10-24_q41](../2026-10-24_q41/README.md)), and it is the only data with five attempts per task, the basis of Q93's
first-finish race at k = 5.

- **E1, UD-Q4_K_XL, 376 tasks × 5:** pass@1 75.69 %; 1.17 % malformed answers; 1 of 1,880 answers at the 4,096-token
  limit; E[p(1−p)] 0.094, and attempts 1 and 2 disagree on 17.8 % of tasks. The set takes 40 tasks per category, so
  the small categories weigh more than in the full pool.
- **E2, 100 tasks:** UD-Q4_K_XL 38.0 % over two attempts (39 and 37 %), G3 32.0 % at one attempt. G3 − UD −7.0 points,
  90 % interval −15.3 to +1.3: inconclusive at this size. UD's two attempts give E[p(1−p)] 0.100 and disagree on 20 %
  of tasks.
- **First-finish race (Q93), E1:** keeping the attempt with the shortest reasoning among k changes pass@1 by +0.51
  (k = 2), +1.02 (k = 3) and +3.16 points (k = 5) against a random attempt, with 0.72, 0.64 and 0.56 of the mean
  reasoning (`race.md`, task-bootstrap intervals).

| File | What |
|---|---|
| `e1/per_attempt.csv` | E1: one row per task and attempt, 1,880 rows. Verdict, BFCL's error type, prompt and output tokens, reasoning characters, stop type and the format flags (columns of [task_results.py](../../../bench/task_results.py)) |
| `e1/summary.md` | E1 per file: pass@1, malformed and runaway rates, mean output tokens, pass@1 per category |
| `e2/per_attempt.csv` | E2: one row per conversation and attempt, 300 rows; adds steps, forced stops and the largest context |
| `e2/summary.md` | E2 per file and category |
| `paired.md` | [paired_tasks.py](../../../bench/paired_tasks.py): G3 against UD-Q4_K_XL, noise per lane |
| `race.md` | [race_first_finish.py](../../../bench/race_first_finish.py) at k = 2, 3 and 5 |
| `e1_pilot_task_ids.txt` | the 376 E1 task IDs |
| `e2_pilot_task_ids.txt` | the 100 E2 task IDs (25 scenarios × 4 categories) |
| `manifests/*.manifest.json` | the harness's record of each raw output: sampling, server build, slots and context, chat-template hashes, task-list hash, attempts; the model path is cut to the file name. A manifest records the latest run into its file, so `e2-ud-q4kxl`'s shows attempt 2 |
| `run/pilot.sh` | the job each file ran |

Made on 2026-10-23/24 with [bench/bfcl_flint](../../../bench/bfcl_flint/README.md) at commit `39ecdd2` on the 7-GPU
quant machine's V100 pair (2 × V100 32 GB). The server is llama.cpp `d77256999` with the `LLAMA_NO_MOE_DOWN_RESCALE`
switch (build `b11820`), `-ngl 99 -fa on -np 8 -kvu -c 262144 -b 2048 -ub 512`; G3 runs with
`LLAMA_NO_MOE_DOWN_RESCALE=1`, UD-Q4_K_XL without it. Sampling is the makers': temperature 1.0, top-p 0.95, top-k 20,
min-p 0, with attempt n sampled at seed n and up to 4,096 tokens per answer or step. No timings were recorded.

```bash
pip download --no-deps bfcl-eval==2026.3.23 && python -m zipfile -e bfcl_eval-2026.3.23-py3-none-any.whl bfcl-v4
export BFCL_ROOT=$PWD/bfcl-v4
cd bench/bfcl_flint
GGUF_PY=/path/to/llama.cpp/gguf-py python gguf_meta.py flint.gguf > template.jinja
python prepare.py template.jinja e1_pilot.jsonl --per-category 40 --seed 0   # 376 tasks; tasks_sha256 06327ae3… in the manifest

# UD-Q4_K_XL: E1 attempts 1-5, then E2 (multi_turn.py run --scenarios 25), stopped once E2 attempt 1 covered all 100 tasks
WORK=... BIN=... MODELS=... GPUS=3,4 bash run/pilot.sh ud-q4kxl "$MODELS/flint-UD-Q4_K_XL.gguf"
# then single-attempt E2 steps with the same server flags: UD-Q4_K_XL attempt 2, and G3 attempt 1 on a G3 server
# started with LLAMA_NO_MOE_DOWN_RESCALE=1 (multi_turn.py run ... --scenarios 25 --attempts N --conc 8)

python score.py e1_pilot.jsonl out/e1 ud-q4kxl=raw/e1-ud-q4kxl.jsonl
python multi_turn.py score out/e2 ud-q4kxl=raw/e2-ud-q4kxl.jsonl g3=raw/e2-g3.jsonl
python ../paired_tasks.py --ref ud-q4kxl --bootstrap 2000 out/e1/per_attempt.csv out/e2/per_attempt.csv > out/paired.md
python ../race_first_finish.py --bootstrap 5000 out/e1/per_attempt.csv out/e2/per_attempt.csv > out/race.md
```

The tasks come from the Berkeley Function Calling Leaderboard (BFCL v4, `bfcl-eval` 2026.3.23, Gorilla project,
Apache-2.0). Their text is not included: `prepare.py` rebuilds it from the wheel, and the manifests' `tasks_sha256`
(a hash of every task's ID and rendered prompt) checks the rebuild. The verdicts and counts describe Flint's outputs
(Flint is Apache-2.0); the raw outputs are not included.
