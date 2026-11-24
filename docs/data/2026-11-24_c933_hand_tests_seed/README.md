# The CPU test runs' hand-run tests on fixed random models (C933, Q1671; 2026-11-24)

What this is: a fix to the CPU test script [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) (C565), which
builds every fork branch and merge on a build machine's processor and runs upstream's and the fork's tests on it, base
against head. Its hand step runs nine tests that the build only compiles (route-once, bytes-switches, moe-ids,
thread-safety, context-model, eval-callback and others) on random models it generates. Those models had a new random
seed on every run, so two runs of the same code wrote different logs, and a base's and a head's logs could only be
compared by pass or fail. Now two runs of the same code write the same logs, and a new tool,
[compare_hand.py](../2026-11-05_c565_ladder4fix/run/compare_hand.py), compares two trees' hand logs line by line.
Registry row Q1671 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **Two runs of the same code now give the same hand logs.** Every model, every log line and every eval-callback sum is
  the same, in both of two runs of each of two commits. One exception: the eval-callback log sometimes loses its last
  few lines. That is the program's fault, not the script's (below), and the comparison shows it as "cut short": every
  line and sum it did print matches.
- **A base and a head now compare line by line.** Miner Llama's `pascal-p106-v5` at `8a9629942` against `3dfad876f`
  shows no hand-log line different, in both runs. The code between them (C880's batch-API move of `test-hop-cast` and
  `test-moe-ids`, a CUDA kernel and server changes these tests do not run) leaves these tests' output as it was, as
  C880's own 129 comparisons found.
- **Every test's result is as before:** ctest passes 73 of 75 in all four trees, test by test the same as C914's run of
  `8a9629942`; the other checks give the same verdicts; the head's 148 `test-llama-archs` logits files are the base's,
  bit for bit.

## Three reasons the logs differed, and the changes

1. **The random models had a new seed every run.** The hand step made its qwen4exp MoE model with
   `test-llama-archs -a qwen4exp -o …` without `-s`, and `test-llama-archs` then seeds from `std::random_device`. The
   dense model that thread-safety, context-model and eval-callback read was the one ctest's `test-generate-models` writes,
   also without a seed; and test-thread-safety's sampler drew its own new seed (no `--seed`). **Change:** the hand step
   makes both models itself with `-s HAND_SEED` (default 1): qwen4exp as before, and `test-llama-archs -a llama -s 1`,
   which writes the same dense model `test-generate-models` writes (and a MoE one beside it), only seeded. thread-safety gets `--seed 1`.
   `hand/models.md5` lists the models' checksums.
2. **Docker interleaved stdout and stderr differently on every run.** The hand step ran each test as
   `docker run … > log 2>&1`. Docker carries a container's stdout and stderr as two streams, and its client writes both
   into the file as they arrive, so where a 4 KiB block of stdout lands among the stderr lines changes from run to run,
   and lines get spliced part way. After the seed fix, `test-bytes-switches`' two logs still differed in every pair for
   this reason alone (round 1 below). **Change:** each hand test writes its log from inside its container
   (`bash -c '"$@" > log 2>&1'`), where both streams share one file and keep the order the program wrote them in.
3. **Two of the tests write from two threads.** thread-safety's contexts run in threads and print in the order they
   finish. context-model's own lines and the library's (which go through llama.cpp's common logger and its thread) both
   go to stderr, and the logger writes a line's time stamp, level and text in separate writes, so the other thread's
   line can land between them. **compare_hand.py** compares these two logs as sorted lists of lines, with every time
   stamp and line-start level letter dropped. All other logs it compares in order.

**The cut-short eval-callback log is the program's.** llama.cpp's common logger prints from its own thread and is
deliberately never destroyed, and its header asks programs to call `common_log_flush()` before they exit.
`llama-eval-callback` returns from `main` without it (the fork's and upstream's at `af5c58347`), so the thread
sometimes loses the last entries: a run printed 99 sums and 895 lines when whole, and 98, 97, 94 or 90 sums when not.
It lost its tail when writing to its own files inside the container too: 1 of 4 runs with stdout and stderr in two
files (the stream test below), and 2 of 4 runs in round 2 with one file. No way of catching the output fixes it; the
program needs the flush. That is a new row
(Q1740).

## The runs

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; each build under the machine's
build lock with `-j 8` at a low CPU share, from a compiler cache copied from C914's folder (643 of 660 compiles hit).
Each round is two CPU test runs, each of the base `3dfad876f` and the head `8a9629942` (`pascal-p106-v5`'s tip),
compared in four pairs: each tree against itself across the runs, and base against head within each run.
**Before C933** is the pair of the two earlier runs of `8a9629942` by the script before this change (C880's landing
and C914).

| Round | Script | When | Pairs: comparisons the same / cut short / different |
|---|---|---|---|
| before C933 | no seeds, logs through docker | 2026-11-24 01:35–01:49 and 05:24–05:27 | 12 / 0 / **9**: no model checksums (the seeds differ), bytes-switches ×2, route-once ×2, thread-safety, eval-callback and its sums, the model-making log |
| round 1 | seeds; logs through docker | 05:43:39–05:53:08 | 20 / 2 / **2** in three pairs, 22 / 0 / 2 in one: the 2 are `test-bytes-switches`' two logs (reason 2) |
| round 2 | seeds; logs from inside the containers | 06:04:21–06:13:35 | **22 / 2 / 0** in every pair: the 2 are eval-callback's log and its sums, cut short |

Round 2, item by item ([hand_round2.tsv](hand_round2.tsv); the last column is before C933):

| Item | base twice | head twice | base–head run 1 | base–head run 2 | before C933 |
|---|---|---|---|---|---|
| the three models' md5 | same | same | same | same | differ (seeds 1470167595 and 2192038968) |
| the nine hand verdicts and exit codes | same | same | same | same | same |
| route-once ×1, ×30 (2,792 and 2,793 lines) | same | same | same | same | differ |
| bytes-switches, zero-tail (24,176 and 24,086 lines) | same | same | same | same | differ |
| moe-ids (655), hop-cast (31) | same | same | same | same | same |
| thread-safety (29, sorted) | same | same | same | same | differ |
| context-model (38, sorted) | same | same | same | same | same |
| eval-callback log | cut short (895 / 884) | cut short (891 / 895) | cut short (895 / 891) | cut short (884 / 895) | differ |
| eval-callback sums | cut short (99 / 97) | cut short (98 / 99) | cut short (99 / 98) | cut short (97 / 99) | differ (first sum 0.936041 / -1.717601) |
| the two model-making logs | same | same | same | same | qwen4exp's differs (its seed); the dense one is new |

The eval-callback runs whole printed 895 lines and 99 sums, the last -1.064878, the same list in both commits and both
rounds ([eval_sums.tsv](eval_sums.tsv)); the models' md5 are the same in all eight trees of both rounds
([models.tsv](models.tsv)).

The other checks, round 2 ([ctest.tsv](ctest.tsv), [checks.tsv](checks.tsv)); "before" is C914's run of `8a9629942`:

| Check | Before | run 1 base, head | run 2 base, head |
|---|---|---|---|
| ctest, 75 tests | 73 pass | 73, 73 | 73, 73: every test the same in all five columns; `test-kv-cache` and `test-kv-cache-scales` fail as before |
| `test-llama-archs -b CPU -s 1`: plain, writing, comparing; 148 files | OK | OK, OK | OK, OK |
| the head's logits against the base's (ident) | — | 148 of 148 the same | 148 of 148 the same |
| the nine hand-run tests | 9 pass | 9, 9 | 9, 9 |

### The stream test

[run/streams.sh](run/streams.sh) ran three of the hand tests four times in each of three ways, with C914's build of
`8a9629942` and round 1's models ([streams.tsv](streams.tsv): lines and md5 of each reduced log, and of its sorted
lines):

| Test | through docker (`> log 2>&1` on the host) | one file inside the container | stdout and stderr in two files inside |
|---|---|---|---|
| `test-bytes-switches` | 4 different files | 4 identical | 4 identical, both files |
| `llama-eval-callback` | 1 of 4 cut short (862 lines, 94 sums) | 4 whole (895, 99) | 1 of 4 cut short (881 stdout lines, 98 sums) |
| `test-llama-context -m` | the order changes between runs (two threads) | the order changes | the order changes on stderr |

streams.sh's sorted md5 drops time stamps only at a line's start, so a context-model file whose stamp was spliced away
from its line shows a sorted md5 of its own there; compare_hand.py drops them wherever they stand, and its context-model
comparisons all read "same".

## Files

| File | What |
|---|---|
| [hand_pairs.tsv](hand_pairs.tsv) | every pair of both rounds and before C933: comparisons the same, cut short and different, and which |
| [hand_round2.tsv](hand_round2.tsv) | round 2 item by item, and before C933 |
| [hand_round1.tsv](hand_round1.tsv) | round 1 item by item (the seeds only) |
| [eval_sums.tsv](eval_sums.tsv) | eval-callback's sums per run: how many printed, the log's lines, the md5 of the list, the last |
| [models.tsv](models.tsv) | the three random models' md5 in each tree of both rounds |
| [ctest.tsv](ctest.tsv) | every ctest test's result in C914's run and the four trees of round 2 |
| [checks.tsv](checks.tsv) | the script's other verdicts in C914's run and the four trees of round 2 |
| [streams.tsv](streams.tsv) | the stream test |
| [run/run.sh](run/run.sh) | one round: the clone, the compiler cache, two runs of `cpu_tests.sh` on the base and head trees, then the four hand comparisons (and before C933's), ctest and the verdicts across the trees, the eval-callback sums |
| [run/streams.sh](run/streams.sh) | the stream test |
| [run/tables.py](run/tables.py) | the tables of this folder from the two rounds' work folders |
| [../2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) | the CPU test script with the change (the seeded models, thread-safety's seed, the logs written inside the containers, `hand/models.md5`) |
| [../2026-11-05_c565_ladder4fix/run/compare_hand.py](../2026-11-05_c565_ladder4fix/run/compare_hand.py) | the hand-log comparison of two trees |

## Commands

```bash
# on the build machine; SEED a llama.cpp clone holding both commits, REPO a folder with cpu_tests.sh, compare_hand.py
# and run.sh; BEFORE an earlier run's tree folder ($WORK/<tag>) for the ctest and verdict columns; BEFORE_A and BEFORE_B
# two earlier runs' tree folders of one commit, for the before-C933 pair
W=<work dir> SEED=<llama.cpp clone> REPO=<scripts> CPU_CCACHE=<an earlier WORK>/ccache BEFORE=<tree dir> \
  BEFORE_A=<tree dir> BEFORE_B=<tree dir> LOCK=<build lock file> JOBS=8 bash run.sh
# two trees' hand logs, any time after (exit 1 if a line differs)
python3 -I compare_hand.py <WORK a>/<tag a> <WORK b>/<tag b> hand.tsv
# the stream test
BIN=<build>/bin DENSE=<WORK>/<tag>/hand/random/llama-dense.gguf MOE=<WORK>/<tag>/hand/random/qwen4exp-moe.gguf \
  OUT=<dir> N=4 MOUNTS='-v <dir>:<dir>:ro ...' bash streams.sh
# the tables, from the two rounds' folders copied back without builds, models and logits
python3 -I tables.py <round 1 W> <round 2 W> <this folder>
```

`run.sh` runs `cpu_tests.sh` twice, in `$W/run1` and `$W/run2`, as `TREES='base=3dfad876f48a41ab2792b23152f5ddd3418ba953
head=8a9629942a4635ea0b2711b72e59556b96a63f86@base' STEPS='tree build ctest archs ident hand' CTEST_JOBS=6 THREADS=6`,
with labels and `--memory=16g` in `DOCKER_ARGS`; the image is `cpu_tests.sh`'s own (Ubuntu 24.04 with the compiler,
CMake, ccache, Python 3 and Jinja2). Round 1 ran `cpu_tests.sh` with the seeds only (its comparisons redone with the
final `compare_hand.py`); round 2 with the whole change.

Provenance and licence: every table here was made by these scripts from their own runs; llama.cpp and its tests are
MIT-licensed.
