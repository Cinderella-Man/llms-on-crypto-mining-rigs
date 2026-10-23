# LiveCodeBench check for Flint

A harness for checking whether a quantized Flint file still writes working code, run the way Flint's makers report
LiveCodeBench (lane E3 of [Q41](../../quants/experiments.md)):

- **Problems.** LiveCodeBench v6 code generation, the 454 problems dated 2024-08-01 to 2025-04-30, minus 4 whose
  answers LiveCodeBench's grader cannot judge ([excluded_problems.txt](excluded_problems.txt)). That leaves 450:
  109 easy, 139 medium, 202 hard, from AtCoder (283) and LeetCode (167). The 130 problems from 2025-02-01 are also
  reported apart; it is the window Qwen calls "v6".
- **Sampling.** K sampled solutions per problem at the makers' settings, with a 32,768-token budget per sample.
- **Grading.** LiveCodeBench's own test runner, unmodified, on the public and hidden tests (2 to 52 per problem,
  median 42; 6 s each), offline in a sandbox without network.
- **Score.** pass@1 per problem is the share of its K samples that pass every test.

Flint's model card gives BF16 80.3 and NVFP4 80.8 on LiveCodeBench v6: average pass@1 over 5 attempts at
temperature 1.0, top-p 0.95, top-k 20. It does not say which problems or prompt. The protocol is in
[task_level_quality_checks.md](../../docs/research/notes/task_level_quality_checks.md).

## How it works

- **Data** ([fetch.py](fetch.py)).
  - It streams `livecodebench/code_generation_lite` at a pinned revision (release_v6) and keeps the problems in the
    window as they are: 1.9 GB, hidden tests still compressed. The data stays outside the repository.
  - The hidden tests are a pickled string. [lcb_common.py](lcb_common.py) unpickles them with an unpickler that
    refuses to import anything, so the data cannot run code.
  - Decoded, the tests of the 454 problems take 4.0 GB: median 68 KB per problem, three problems over 150 MB.
- **Prompts** ([prepare.py](prepare.py)).
  - LiveCodeBench's prompt for an OpenAI-compatible chat model (`LMStyle.OpenAIChat`): its generic system message,
    then the question with the LeetCode starter code or the stdin/stdout instruction.
  - It is rendered with Flint's chat template from the GGUF (`../bfcl_flint/gguf_meta.py`). The generation prompt
    is `<|assistant|>`, and Flint opens its `<think>` block itself. The longest prompt is 4,433 characters.
  - The 4 excluded problems accept any answer within an error tolerance (10⁻⁶ to 10⁻⁹). LiveCodeBench's grader
    compares output lines as exact decimals, so on them it grades the print format: correct answers printed with
    other digits fail.
- **Generation** ([run.py](run.py)).
  - Requests go to llama-server's native `/completion`, as for the BFCL harness ([flint_client.py](../flint_client.py)).
  - Every request carries the makers' sampling: temperature 1.0, top-p 0.95, top-k 20, min-p 0, the other samplers
    neutral, and the chain order pinned.
  - `n_predict` is 32,768, for the reasoning and the answer together. The run refuses to start if a server slot
    cannot hold the longest prompt plus that budget, so that every file gets the same budget.
  - Attempt n samples with seed n for every problem and file. Attempts run in order and resume after a stop.
- **Extraction.** The answer is the text after the last `</think>`, which is what rec's API returns as content.
  LiveCodeBench's `extract_code` takes its last fenced block. A sample that never closes its reasoning has no answer.
- **Grading** ([evaluate.py](evaluate.py)).
  - It runs LiveCodeBench's `evaluate_generations` (commit `28fef95`) with its defaults: 6 s per test, and a fail at
    the first wrong test.
  - LeetCode problems call the `Solution` method; AtCoder problems run the program on stdin.
  - It refuses to run while the internet is reachable.
  - The time limits are wall-clock, so every file is graded on the same machine with the same number of processes.

**Output.** `per_attempt.csv` has the shared columns of [task_results.py](../task_results.py), plus platform, contest
date and LiveCodeBench's error code and message (wrong answer, time limit, runtime error, compile error). In it:

- `cluster` is the contest;
- `malformed` means no code block in the answer;
- `runaway` means the sample hit 32,768 tokens or never closed its reasoning.

[paired_tasks.py](../paired_tasks.py) compares files (margin 3 points). Q41's gate also caps the share of samples cut
off at 32,768 tokens at UD-Q4_K_XL's plus 2 points; the `runaway` column carries it.

## Validation without a model

LiveCodeBench publishes its leaderboard runs with the raw answers, the extracted code and its own verdict per
sample. [evaluate.py](evaluate.py) `--replay` re-grades such a file offline:

- our extraction must give the submitted code from the raw answers;
- our verdicts must match LiveCodeBench's.

The replay covers the 450 problems, 4,950 answers, from two runs:

- DeepSeek-R1-0528: 10 samples per problem at temperature 0.6;
- o4-mini (high): 1 sample per problem.

The correct samples serve as reference solutions.

- **Extraction.** Our extraction gives the submitted code for 4,945 of the 4,950 answers. In the other 5,
  R1-0528 wrote a block after its Python code: a sample input 3 times, a C++ version twice. The submission took the
  Python block; LiveCodeBench's `extract_code` at this commit takes the last block, as this harness does.
- **Verdicts.**

  | Submission | Samples | Same verdict | LCB pass, ours fail | LCB fail, ours pass |
  |---|---|---|---|---|
  | o4-mini (high) | 450 | 450 (100 %) | 0 | 0 |
  | DeepSeek-R1-0528 | 4,500 | 4,451 (98.9 %) | 39 | 10 |

  - Of R1-0528's 39: 33 hit the 6 s limit on our machine (hard problems, solutions near the limit), 2 raised a
    runtime error, and 4 print a wrong grid on `abc375_c`'s public sample, which LiveCodeBench graded against other
    test data.
  - The 10 the other way failed on LiveCodeBench's machine with a runtime error or time limit on a late test.
  - pass@1 on the 450 problems: R1-0528 72.6 % here against 73.2 % in LiveCodeBench's grades; o4-mini 80.9 % in
    both.
- **Load.** The time limits are wall-clock. A first pass at 12 processes on a 32-core workstation, under a load
  average of 15–40 from other jobs, failed 186 correct samples on time. A second pass over the disagreeing samples,
  at 4 processes on an idle machine with Python 3.11, passed 153 of them; the table holds its verdicts.

  Grade every file on the same otherwise idle machine, with the same process count (half the cores or fewer).

## Running it

Needs Python 3.11 (LiveCodeBench's own version) with numpy, tqdm, jinja2 and requests, and a LiveCodeBench checkout.
The checkout is only imported, not installed: its `pyproject` pulls in vLLM and the API SDKs.

```bash
git clone https://github.com/LiveCodeBench/LiveCodeBench && git -C LiveCodeBench checkout 28fef95
export LCB_ROOT=$PWD/LiveCodeBench
python -I fetch.py /data/lcb/problems.jsonl                            # 454 problems, 1.9 GB
python prepare.py template.jinja /data/lcb/problems.jsonl tasks.jsonl  # 450 problems, sha256 c48319a4…
llama-server -m flint.gguf -ngl 99 -fa on -np 8 -kvu -c 278528 -b 2048 -ub 512 --cache-ram 0 &
python run.py http://127.0.0.1:8080 tasks.jsonl raw/myfile.jsonl --attempts 1-5 --conc 8
```

Grade after generation, with no network. On a workstation, [sandbox.sh](sandbox.sh) uses bubblewrap. The command
gets its own network namespace, the system read-only, an empty `$HOME` and `/tmp`, read-only access to the paths
listed and write access to the output folder only. Each process is limited to 8 GB of address space. List the
Python environment too when it lives under `$HOME`.

```bash
./sandbox.sh out/ $PWD/.. $LCB_ROOT /data/lcb raw/ $VIRTUAL_ENV ~/.local/share/uv/python -- \
  $VIRTUAL_ENV/bin/python $PWD/evaluate.py /data/lcb/problems.jsonl out/ myfile=raw/myfile.jsonl ref=raw/ref.jsonl
python ../paired_tasks.py --ref ref out/per_attempt.csv
```

On a GPU server, grade in a container instead: `docker run --rm --network none --memory 48g --cpus 12` with the
same paths mounted read-only and the output folder writable, in any Python image with numpy and tqdm.

## Pilot plan (Q41)

**Pilot.** UD-Q4_K_XL, `prepare.py --sample 100` (seed 0, sha256 `6f88f52e…`), 100 problems × 5 attempts = 500
samples on one 8-slot server: ~3.7 server-hours if Flint averages 8k tokens per sample, ~7 at 15k. It measures:

1. **Tokens per sample** (Q71's open question, assumed 8k): the E3 cost below scales with it.
2. **The share of samples cut off at 32,768 tokens.** The makers' card asks for at least 32,768. If more than 5 %
   hit the limit, raise the budget for every file; that costs time, not comparability.
3. **E[p(1−p)]**, which sets K. N·K must reach 12.4 · E[p(1−p)] / m² for 80 % power at margin m = 3 points when the
   files are equal. With N = 454 that is K = 3 at E[p(1−p)] 0.08, K = 4 at 0.12 and K = 5 at 0.16. K = 5, the
   makers' count, stays the default.
4. **Grading time** for one file's samples in the sandbox, on a workstation. The replay graded ~5,000 programs in
   about an hour at 12 processes.

**Full run, per file at K = 5 (est.).** 2,250 samples:

- at 8k tokens per sample, ~18 M output tokens, ~17 server-hours;
- at 15k tokens, ~34 M tokens, ~31 server-hours.

The assumption is ~300 tokens/s aggregate decode per 8-slot server at these lengths (the note's §7.5). Prompts
(~1k tokens) are negligible. These are job times for scheduling on the quant machine, not speed results. The whole
of Q41, all three lanes and every file, is sized in [the BFCL harness's pilot plan](../bfcl_flint/README.md#pilot-plan-q41).
