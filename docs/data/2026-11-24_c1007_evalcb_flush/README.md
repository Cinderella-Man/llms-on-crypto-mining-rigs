# llama-eval-callback prints its whole log (C1007, Q1740; 2026-11-24)

What this is: a two-program fix in Miner Llama, my llama.cpp fork, and its check. llama.cpp's common logger prints from
its own thread and is never destroyed on purpose (upstream pull request 22273, for a hang at exit on Windows), and
`common/log.h` asks programs to call `common_log_flush()` before they exit. `llama-eval-callback` returned from `main`
without it, so the process sometimes ended while the logger still held its last lines, and those lines were never
printed. The CPU test script's hand step ([cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh)) runs
`llama-eval-callback` and `test-thread-safety`, and [C933](../2026-11-24_c933_hand_tests_seed/README.md) found the
eval-callback log cut short in 6 of 20 runs. The fix flushes the logger before every return of `main` in both programs.
Registry row Q1740 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **With the fix, eval-callback's log is whole every time.** Run 20 times on the same random model, the base lost the
  end of its log 7 times (95, 96, 96, 96, 96, 97 and 98 of the 99 sums printed); the fix printed all 99 sums 20 times
  out of 20, and every one of those 20 logs is the same as the base's 13 whole ones.
- **Nothing else changes.** In two CPU test runs, base and fix give the same output line for line: 24 of 24 hand-log
  comparisons the same, base against itself, fix against itself and base against fix, in both runs. ctest gives the
  same result test by test in all four trees (76 of 77 pass; `test-kv-cache-scales`, a Python test, fails in every
  tree, as it does in the base's earlier runs), and the fix's 148 `test-llama-archs` logits files are the base's, bit
  for bit.
- **test-thread-safety had the same gap** (its last line was "All threads finished without errors." with nothing
  after it to push it out) and gets the same fix. Its log was whole in all 40 runs here, base and fix alike: its end
  comes after the threads join, so the logger is usually idle by then.
- **38 more programs share the gap** (below). They are named for a later change (Q1761), not fixed here.

## The change

Fork branch `pascal-p106-v5-evalcb-flush`, commit `4d8fe365a`, one commit on `pascal-p106-v5` `af00bb13b`:

- `examples/eval-callback/eval-callback.cpp`: `common_log_flush(common_log_main());` before each of `main`'s four
  returns (argument error, model load error, decode error, success).
- `tests/test-thread-safety.cpp`: the same before each of its four returns.

Six lines added in one program, five in the other; nothing else is touched, and no switch is needed (the flush only prints
what the program already logged).

## The check

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; each build under the machine's
build lock with `-j 8` at a low CPU share, from the compiler cache of the base's earlier CPU test run (645, 647 and 647 of
664 compiles hit, [ccache.tsv](ccache.tsv)). The driver is [run/run.sh](run/run.sh), 14:05 to 14:15:

1. **Two CPU test runs** (`run1`, `run2`): `cpu_tests.sh` on the trees `base` (`af00bb13b`) and `head` (`4d8fe365a`),
   steps tree, build, ctest, archs, ident, hand. The pipe step is left out: it runs neither program.
2. **The hand logs compared** with [compare_hand.py](../2026-11-05_c565_ladder4fix/run/compare_hand.py): each tree
   against itself across the two runs, and base against head in each run. In these four runs eval-callback's log
   happened to be whole every time, base included, so two runs alone could not show the difference; hence step 3.
3. **Twenty more runs of each program per tree** (`repeat`): each tree built again with the build kept, then
   `llama-eval-callback` 20 times and `test-thread-safety` 20 times in one container, with the hand step's arguments,
   on the same random dense model (`test-llama-archs -a llama -s 1`; `hand/models.md5` gives the same md5 in every
   tree). A run is **whole** when its sums (eval-callback) or sorted lines (thread-safety) are the full list and its
   last line is the whole logs' last line, **cut short** when they are the beginning of the full list.
4. The base's earlier run (the v5 landing's CPU test run of `af00bb13b`) is compared with run 1's base as well. That
   run was built with `-DGGML_RPC=ON`, which registers an RPC backend, so the model-loading lines differ ("offloading
   output layer to GPU" instead of the CPU buffer line, no "no usable GPU found" warning) and six logs differ for that
   reason only. Its eval-callback sums are the same 99, and its eval-callback log is 892 of 895 lines: cut short.

| Check | Result |
|---|---|
| eval-callback, 20 runs per tree ([repeat.tsv](repeat.tsv)) | base 13 whole, 7 cut short (95, 96, 96, 96, 96, 97 and 98 of 99 sums); fix 20 whole, 0 cut short |
| thread-safety, 20 runs per tree | base 20 whole, fix 20 whole |
| The 33 whole eval-callback logs ([repeat_runs.tsv](repeat_runs.tsv)) | one md5 (time stamps dropped), base's and fix's alike; the 40 thread-safety logs one md5 as sorted lines |
| Hand logs, base twice, fix twice, base against fix in each run ([hand_pairs.tsv](hand_pairs.tsv)) | 24 of 24 the same in each of the four pairs |
| Hand logs, the landing's run against run 1's base | 17 same, 6 differ, 1 cut short: the RPC build option (point 4), not the code |
| eval-callback sums in the test runs ([eval_sums.tsv](eval_sums.tsv)) | 99 sums, one list (md5 `cf742168eb51`) in all five trees |
| ctest test by test ([ctest.tsv](ctest.tsv)) | 76 of 77 pass in all four trees, the same tests; the landing's run 79 of 80 (its three RPC tests on top) |
| Other verdicts ([checks.tsv](checks.tsv)) | all as the base's; ident: the fix's 148 logits files the base's |

## The 38 other programs with the same gap

They use the common logger (they call `common_init()`, which also routes the library's own messages through it, or
print with `LOG`) and return from `main` without `common_log_flush()`. In Miner Llama at `af00bb13b`; all but the four
marked "fork" have the same code upstream at `af5c58347`:

- tools: `llama-cli` (`cli.cpp`), `llama-completion`, `llama-perplexity`, `llama-batched-bench`, `llama-imatrix`,
  `llama-mtmd-cli`, `llama-mtmd-debug`, `llama-tts`, `llama-tokenize`, `llama-cvector-generator`, `llama-export-lora`,
  `llama-results`;
- examples: `batched`, `convert-llama2c-to-ggml`, `debug`, `diffusion-cli`, `embedding`, `idle`, `lookahead`, `lookup`,
  `lookup-create`, `lookup-stats`, `parallel`, `passkey`, `retrieval`, `speculative`, `speculative-simple`, `finetune`,
  `pipespec` (fork), `pipe-recurrent-check` (fork), `screen-check` (fork);
- tests: `test-chat-analysis`, `test-chat-auto-parser`, `test-export-graph-ops`, `test-fp8`, `test-fusion`,
  `test-speculative-rejection-sampling`, `test-spec-reject-stop` (fork).

Already flushing: `llama-server`, `llama-fit-params`, `test-llama-context`, `test-llama-archs`, `test-chat`,
`test-log`. The list comes from a search of every `main` (and the `llama_<tool>` functions the tools' `main.cpp`
calls) in `tools/`, `examples/` and `tests/` for `common_init()` or `LOG` calls and no `common_log_flush`.

## Files

| File | What |
|---|---|
| [hand_pairs.tsv](hand_pairs.tsv) | per pair of trees: compare_hand.py's comparisons, how many same, cut short, different, and which |
| [hand_items.tsv](hand_items.tsv) | per compared item (models, verdicts, logs, sums): its verdict in each pair |
| [repeat.tsv](repeat.tsv) | the 20-run repeat per tree and program: runs, exit 0, whole, cut short, different, the counts when cut short |
| [repeat_runs.tsv](repeat_runs.tsv) | each of the 80 repeat runs: exit code, sums or lines, verdict, the md5 of its log without time stamps, its last line |
| [eval_sums.tsv](eval_sums.tsv) | the test runs' eval-callback sums (count, lines, md5 of the list) and thread-safety's last line, per tree |
| [ctest.tsv](ctest.tsv) | ctest, test by test, in the landing's run and the four trees |
| [checks.tsv](checks.tsv) | every other check's verdict in the landing's run and the four trees |
| [ccache.tsv](ccache.tsv) | the compiler cache's hits per build round |
| [run/run.sh](run/run.sh) | the driver: sources, the two test runs, the repeat, the comparisons |
| [run/tables.py](run/tables.py) | these tables from the driver's work folder |

## Commands

```bash
# on the build machine: REPO holds cpu_tests.sh and compare_hand.py (docs/data/2026-11-05_c565_ladder4fix/run/) and
# run.sh; SEED is a llama.cpp clone holding af00bb13b; CPU_CCACHE and BEFORE are optional (an earlier CPU test run's
# compiler cache and its base tree)
W=$WORK/c1007 SEED=$WORK/llama.cpp REPO=$WORK/c1007/repo CPU_CCACHE=$WORK/land/cpu/ccache BEFORE=$WORK/land/cpu/land \
  LOCK=$WORK/locks/build.lock JOBS=8 bash $WORK/c1007/repo/run.sh > $WORK/c1007/logs/run.out 2>&1
# then, on a copy of $W's logs, checks, run1/, run2/ and repeat/ folders
python3 -I run/tables.py <copy> docs/data/2026-11-24_c1007_evalcb_flush
```

Defaults in `run.sh`: `BASE=af00bb13b0fd59612bcab545964b61978f04b8d6`,
`COMMIT=4d8fe365adaa011ee70f02cbb678ac78d0ac76c4`, `BRANCH=pascal-p106-v5-evalcb-flush`, `REPEAT=20`, `HAND_SEED=1`,
the CPU image `pq-c565-cpu:24.04` (`cpu_tests.sh` builds it from `ubuntu:24.04` if missing). Every container carries
`--cpu-shares=256 --memory=16g --network none`.

Provenance: the fork's code at the two commits above, the random models `test-llama-archs` generates with seed 1; no
downloaded data. Licence: the tables and scripts are this repository's; llama.cpp is MIT.
