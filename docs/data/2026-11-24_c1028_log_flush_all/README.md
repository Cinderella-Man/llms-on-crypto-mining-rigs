# Every program flushes llama.cpp's common logger before it ends (C1028, Q1761; 2026-11-24)

What this is: a fix in Miner Llama, my llama.cpp fork, for 38 programs, and its check. llama.cpp's common logger prints
from its own thread and is never destroyed on purpose (upstream pull request 22273, for a hang at exit on Windows), and
`common/log.h` asks programs to call `common_log_flush()` before they exit. [C1007](../2026-11-24_c1007_evalcb_flush/README.md)
found `llama-eval-callback` losing its last lines at random because it ended without that call, fixed it and
`test-thread-safety`, and listed 38 more programs with the same gap. This change fixes those 38. Registry row Q1761 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The 38 programs now flush the logger on every return and before every `exit()`, and nothing else changes.** In a
  CPU test run of base and fix, every hand-run log is the same line for line (24 of 24), ctest gives the same result
  test by test (76 of 77 pass in both; `test-kv-cache-scales`, a Python test, fails in every tree as before), the
  fix's 148 `test-llama-archs` logits files are the base's bit for bit, and the pipe step, which runs two of the
  changed programs (`llama-pipespec`, `llama-pipe-recurrent-check`) on five random models, passes all 25 of its
  checks in both trees.
- **The unfixed programs did not lose lines in these runs.** Four of the changed programs in five settings, 20 runs
  each per tree (`llama-completion`, `llama-batched-bench`, `llama-embedding`, and `llama-completion` and
  `llama-tokenize` on a missing model file, an error path), gave whole logs all 200 times, base included, and each
  setting's log is the same in both trees. Their last log lines are few, and the logger's thread printed them before
  the process was gone; the loss showed in `llama-eval-callback` (C1007: 7 of 20 runs), which logs most of its 895
  lines in one burst during its only decode, just before it returns. So for these programs the change makes the end
  of the log certain; it does not fix a loss seen here.
- **Why one shared helper declared once per program, rather than an edit before every return or a hook in `common`:**
  below.

## The change

Fork branch `pascal-p106-v5-log-flush-all`, commit `cf093e11c`, one commit on C1007's branch
`pascal-p106-v5-evalcb-flush` (`4d8fe365a`, itself one commit on `pascal-p106-v5` `af00bb13b`), so the two land as one
line. 39 files, 77 lines added and 1 changed:

- **`common/log.h`: `common_log_flush_guard`**, a small object whose destructor flushes the main logger. A program
  declares one as the first statement of `main`; when `main` returns, by any of its `return` statements, the guard is
  destroyed last, after `main`'s other locals, and the lines still queued are printed.
- **Each of the 38 programs declares the guard** as the first line of `main` (for the four tools whose `main.cpp`
  calls a `llama_<tool>` function, in that function): one line per program, plus `#include "log.h"` in three that did
  not include it.
- **The 23 `exit()` calls in those programs** (outside a signal handler) get `common_log_flush(common_log_main());`
  on the line before: an error message logged just before `exit(1)` could be lost the same way. `llama-cli`'s Ctrl-C
  handler is left as it is: it avoids the logger on purpose, since flushing from a signal handler can deadlock.

**Why a guard in each program, not one exit hook in `common`.** The shared way would be a hook inside `common` that
runs when the process exits (an `atexit` handler or a static object's destructor), with no edit to any program. On
Windows `common` is a DLL in the default shared build, and such a hook then runs during the DLL's teardown, after the
logger's thread has been killed: joining that thread is exactly the hang (upstream issue 22142) that made upstream
stop destroying the logger, and upstream tried an explicit free in that pull request and reverted it. A guard in
`main` runs while `main` is still on the stack, at the same moment as an explicit `common_log_flush()` before
`return`, so it cannot bring that hang back. And one line at the top of `main` replaces a flush before every
`return` (`pipespec`'s `main` has 12), cannot miss a return added later, and also flushes what the other locals log
while they are destroyed. The four upstream programs that already flush (`llama-server`, `llama-fit-params`,
`test-llama-context`, `test-llama-archs`), `test-chat`, `test-log` and C1007's two keep their explicit calls.

No switch: the change only prints lines the programs already logged.

### The 38 programs

All but the four marked "fork" are upstream's code as well (upstream `33aeb17f8`, 2026-11-24):

- tools (12): `llama-cli`, `llama-completion`, `llama-perplexity`, `llama-batched-bench`, `llama-imatrix` (8 exits),
  `llama-mtmd-cli` (4 exits), `llama-mtmd-debug` (1 exit), `llama-tts`, `llama-tokenize`, `llama-cvector-generator`
  (1 exit), `llama-export-lora` (1 exit), `llama-results`;
- examples (19): `batched`, `convert-llama2c-to-ggml` (4 exits), `debug`, `diffusion-cli`, `embedding`, `idle`,
  `lookahead`, `lookup` (1 exit), `lookup-create`, `lookup-stats` (1 exit), `parallel`, `passkey`, `retrieval`,
  `speculative`, `speculative-simple`, `finetune`, `pipespec` (fork), `pipe-recurrent-check` (fork), `screen-check`
  (fork);
- tests (7): `test-chat-analysis`, `test-chat-auto-parser`, `test-export-graph-ops`, `test-fp8`, `test-fusion`
  (2 exits), `test-speculative-rejection-sampling`, `test-spec-reject-stop` (fork).

Three of `llama-imatrix`'s eight exits are in the fork's own token-class code, so upstream's copy of the change has 20
flushed exits instead of 23. It applies to upstream `33aeb17f8` (34 programs and `log.h`, 70 lines).

## The check

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; each build under the machine's
build lock with `-j 8` at a low CPU share, from the compiler cache of C1007's second CPU test run (547 of 664 compiles
hit in the first build round: the fix changes `log.h`, which most of the tree includes; [ccache.tsv](ccache.tsv)).
The driver is [run/run.sh](run/run.sh), 21:37 to 21:48:

1. **One CPU test run** (`run1`): `cpu_tests.sh` on the trees `base` (`af00bb13b`) and `head` (`cf093e11c`), steps
   tree, build, ctest, archs, ident, hand, and the pipe step that comes with hand (pipelined decoding against plain
   with `llama-pipespec` and `llama-pipe-recurrent-check` on random `qwen4exp`, `qwen35`, `qwen35moe`, `qwen3next` and
   `flint` models, five arms each).
2. **The hand logs compared** with [compare_hand.py](../2026-11-05_c565_ladder4fix/run/compare_hand.py): base against
   head, and C1007's run of the same base (`af00bb13b`, run 2) against this run's base.
3. **Five changed programs repeated** (three rounds of the driver's repeat step): each tree built again, then each
   program 20 times in one container on the hand step's seeded dense random model (`hand/models.md5` gives the same
   md5 in every tree). A run is **whole** when its log, with every number masked (the timings change from run to run)
   and runs of spaces made one, is the most common longest log of both trees, **cut short** when it is that log's
   beginning, and **different** otherwise.

| Check | Result |
|---|---|
| Hand logs, base against head ([hand_pairs.tsv](hand_pairs.tsv)) | 24 of 24 the same |
| Hand logs, C1007's run 2 base against this base | 24 of 24 the same |
| eval-callback's sums ([eval_sums.tsv](eval_sums.tsv)) | 99 sums, 895 lines, one list (md5 `cf742168eb51`) in all three trees |
| ctest test by test ([ctest.tsv](ctest.tsv)) | 76 of 77 pass in all three trees, the same tests |
| ident ([checks.tsv](checks.tsv)) | the fix's 148 logits files the base's |
| The pipe step ([pipe.tsv](pipe.tsv)) | 25 of 25 OK in both trees (no pipelined run's tokens, logits, KV state or probes differ from plain decoding's) |
| Repeat: `llama-completion`, 16 tokens ([repeat.tsv](repeat.tsv)) | base 20 whole, head 20 whole; one masked log (md5 `dc5b3fdc5c84`) |
| Repeat: `llama-batched-bench`, 16 prompt and 8 generated tokens at 1 and 2 sequences | base 20 whole, head 20 whole; one masked log (`5fc3eb2dc841`) |
| Repeat: `llama-embedding` on one prompt | base 20 whole, head 20 whole; one masked log (`ba398e583881`) |
| Repeat: `llama-completion`, a missing model file (exit code 1) | base 20 whole, head 20 whole; one masked log (`ac39d9476c49`) |
| Repeat: `llama-tokenize`, a missing model file (exit code 1) | base 20 whole, head 20 whole; one masked log (`46f4b76f0049`) |
| Every other verdict ([checks.tsv](checks.tsv)) | the base's |

## Files

| File | What |
|---|---|
| [programs.tsv](programs.tsv) | the 38 changed files: kind, program, guard added, exits flushed, whether upstream has the file and whether upstream's copy is the base's |
| [hand_pairs.tsv](hand_pairs.tsv) | per pair of trees: compare_hand.py's comparisons, how many same, cut short, different |
| [hand_items.tsv](hand_items.tsv) | per compared item (models, verdicts, logs, sums): its verdict in each pair |
| [repeat.tsv](repeat.tsv) | the repeat rounds per tree and program: runs, exit 0, whole, cut short, different |
| [repeat_runs.tsv](repeat_runs.tsv) | each of the 200 repeat runs: exit code, lines, verdict, the md5 of its masked log, its last line masked |
| [eval_sums.tsv](eval_sums.tsv) | eval-callback's sums (count, lines, md5 of the list) and thread-safety's last line, per tree |
| [ctest.tsv](ctest.tsv) | ctest, test by test, in C1007's run 2 base and the two trees |
| [checks.tsv](checks.tsv) | every other check's verdict in C1007's run 2 base and the two trees |
| [pipe.tsv](pipe.tsv) | the pipe step's 25 verdicts per tree, with the head's detail line |
| [ccache.tsv](ccache.tsv) | the compiler cache's hits per build round |
| [run/run.sh](run/run.sh) | the driver: sources, the test run, the repeat rounds, the comparisons |
| [run/tables.py](run/tables.py) | these tables from the driver's work folder |
| [run/programs.py](run/programs.py) | programs.tsv from a llama.cpp clone (git objects only) |

## Commands

```bash
# on the build machine: REPO holds cpu_tests.sh and compare_hand.py (docs/data/2026-11-05_c565_ladder4fix/run/), run.sh,
# and pipespec/ (check_identity.sh, check_streams.sh and prompts/ from bench/pipespec/); SEED is a llama.cpp clone holding
# af00bb13b; CPU_CCACHE and BEFORE are optional (an earlier CPU test run's compiler cache and its base tree)
W=$WORK/c1028 SEED=$WORK/llama.cpp REPO=$WORK/c1028/repo CPU_CCACHE=$WORK/c1007/run2/ccache BEFORE=$WORK/c1007/run2/base \
  LOCK=$WORK/locks/build.lock JOBS=8 THREADS=6 bash $WORK/c1028/repo/run.sh > $WORK/c1028/logs/run.out 2>&1
# the second and third repeat rounds (the same variables)
... STEPS='repeat compare' ROUND=repeat2 PROGS=embedding bash $WORK/c1028/repo/run.sh
... STEPS='repeat compare' ROUND=repeat3 PROGS='completion-nomodel tokenize-nomodel' bash $WORK/c1028/repo/run.sh
# then, on a copy of $W's logs, checks, run1/ and repeat*/ folders
python3 -I run/tables.py <copy> docs/data/2026-11-24_c1028_log_flush_all
# the program list, from a llama.cpp clone holding the fork's commits and upstream's master
python3 -I run/programs.py <clone> 4d8fe365a cf093e11c 33aeb17f8 > docs/data/2026-11-24_c1028_log_flush_all/programs.tsv
```

Defaults in `run.sh`: `BASE=af00bb13b0fd59612bcab545964b61978f04b8d6`,
`COMMIT=cf093e11ca2631f6e72a548313f7c5da0513c39f`, `BRANCH=pascal-p106-v5-log-flush-all`, `REPEAT=20`, `HAND_SEED=1`,
the pipe step's five random models (`qwen4exp qwen35 qwen35moe qwen3next flint`), the CPU image `pq-c565-cpu:24.04`
(`cpu_tests.sh` builds it from `ubuntu:24.04` if missing). Every container carries
`--cpu-shares=256 --memory=16g --network none`.

Provenance: the fork's code at the commits above, the random models `test-llama-archs` generates with seed 1; no
downloaded data. Licence: the tables and scripts are this repository's; llama.cpp is MIT.
