# ctest's random test models with a fixed seed (C1008, Q1741; 2026-11-24)

What this is: a one-line change to the test list of Miner Llama, my llama.cpp fork, and its check. ctest's
`test-generate-models` runs `test-llama-archs -o tests/test-models/` to write a small random model for every
architecture, and `test-llama-context` then runs its context-state sweep on those models. The line passed no seed
(`-s`), so `test-llama-archs` seeded itself from `std::random_device` and every ctest run tested new random weights: a
failure that depends on one draw might not come back on the next run. The two other fixtures that write random models,
`test-generate-fp8-models` and `test-generate-speculative-target`, already pass `-s 1` and `-s 999`. The line is
upstream's too. [C933](../2026-11-24_c933_hand_tests_seed/README.md) found it while fixing the same problem in the CPU test
script's hand-run tests. Registry row Q1741 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **With the seed, ctest writes the same models every run.** In two ctest runs of the change, all 147 random models of
  `tests/test-models/` had the same md5 both times. In two runs without it, all 147 differed: the generator drew the
  seeds 2960570594 and 1120871947, and printed them only in its own output, which ctest does not show on screen for a
  test that passes (it stays in the build's `Testing/Temporary/LastTest.log`).
- **Nothing else changes.** ctest gives the same result test by test in all four trees and in the base's earlier run
  (76 of 77 pass; `test-kv-cache-scales`, a Python test, fails in every tree, as before). The sweep's verdict table is
  the same in all four trees, line for line: 147 models, 26 cases each, 2,162 passed, 1,660 skipped, 0 failed.
- **The two random draws of the base also passed,** so this check found no failure that depends on the draw. What the
  seed gives is that a failure, when one comes, comes back on the next run and on anyone else's machine.
- **ctest's llama models are now the same files as the hand step's.** With `-s 1`, `test-models/llama-dense.gguf` and
  `llama-moe.gguf` are byte for byte the models the CPU test script's hand step makes (`test-llama-archs -a llama -s 1`)
  and the fp8 fixture's copies (`-s 1` too): md5 `c57717ea92ba` and `86effb74ecc2`.
- **Two other draws in ctest have no fixed seed** (below): `test-gguf`, which does not even print its seed, and the
  sampler of `test-thread-safety`. They are named for later changes (Q1767, Q1768), not changed here.

## The change

Fork branch `pascal-p106-v5-ctest-seed`, commit `55e14cb79`, one commit on `pascal-p106-v5` `af00bb13b`, in
`tests/CMakeLists.txt`:

```diff
 llama_test(
     test-llama-archs
     NAME test-generate-models
     LABEL "main behavior backend"
-    ARGS -o "${MODEL_DIR}"
+    ARGS -s 1 -o "${MODEL_DIR}"
 )
```

No engine code changes and no switch is needed. A run by hand still draws a new seed unless `-s` is given, so other
draws can still be tested on purpose (`test-llama-archs -s <n> -o <dir>`, then `test-llama-context --models <dir>`).

## The check

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; each build under the machine's
build lock with `-j 8` at a low CPU share, from the compiler cache of C1007's runs of the same base (647 of 664 compiles
hit, each run). The driver is [run/run.sh](run/run.sh), 14:31 to 14:36, with
[cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh):

1. **Two runs** (`run1`, `run2`), each of the trees `base` (`af00bb13b`, the line without a seed) and `head`
   (`55e14cb79`), one tree at a time: `cpu_tests.sh` steps tree, build, ctest (its default exclusions: the tests that
   need a downloaded model or network, and the torch test), with the build kept.
2. After each tree's ctest: the md5 of every `.gguf` ctest wrote under the build's `tests/` folder (147 in
   `test-models/`, 2 in `test-models-fp8/`, 2 in `test-speculative-models/target/`); ctest's `LastTest.log`, whose
   generating tests' output names the seed each used; and the sweep run once more as ctest runs it
   (`test-llama-context --models tests/test-models/`, `GGML_SCHED_DEBUG_REALLOC=1`), its log written to its own file.
3. The comparison: each model's md5 in the four trees and in C1007's hand step; ctest test by test against C1007's run
   of the same base (the `before` column); the sweep's logs with time stamps and colours dropped, in order and sorted.

| Check | Result |
|---|---|
| `test-models/`, base twice ([models.tsv](models.tsv)) | 0 of 147 files the same: seeds 2960570594 and 1120871947 ([seeds.tsv](seeds.tsv)) |
| `test-models/`, head twice | 147 of 147 the same (seed 1) |
| `test-models/`, head against base | 0 of 147 the same (other seeds) |
| `test-models-fp8/` and `test-speculative-models/`, all four trees | 4 of 4 the same (seeds 1 and 999 before and after) |
| head's `llama-dense.gguf`, `llama-moe.gguf` against the hand step's | the same (`c57717ea92ba`, `86effb74ecc2`); its `qwen4exp-moe.gguf` differs because the hand step makes it with a 4,096-token vocabulary and 32 experts |
| ctest test by test ([ctest.tsv](ctest.tsv)) | 76 of 77 pass in all four trees and in the earlier base run, the same tests |
| The sweep's log ([context.tsv](context.tsv), [context_pairs.tsv](context_pairs.tsv)) | 165 lines, the same in all four trees and in every pair: 147 model rows of 26 cases, 2,162 OK, 1,660 SKIP, 0 failed |
| cpu_tests.sh's verdicts ([checks.tsv](checks.tsv)) | tree and build OK in every tree; ctest FAIL in every tree, for `test-kv-cache-scales` only |

## Other draws in ctest without a fixed seed

Read in the fork at `af00bb13b` and upstream at `40fe9164f`, whose lines are the same:

- **`test-gguf`** runs with no argument, so it draws its seed from `std::random_device`, and it never prints it: a
  failing draw of its random GGUF files (handcrafted files, round trips in four read modes, `gguf_set_kv`) cannot be
  repeated. Q1767: print the seed, and give ctest a fixed one.
- **`test-thread-safety`** gets no `--seed`, so its sampler draws a new seed every run and the four sequences' text
  changes (its model is the downloaded stories15M, the same file each time). The CPU test script's hand step already
  passes `--seed 1`. Q1768: pass `--seed 1` in ctest's line.
- Not draws: `test-llama-context` makes its random prompt tokens with the default sampling seed, used as a fixed number;
  `test-fp8` uses `std::random_device` only to name a temporary folder; `test-tiled-mulmat` seeds with a constant;
  `test-backend-ops`, which seeds from `std::random_device`, is built but not registered with ctest.

## Files

| File | What |
|---|---|
| [models.tsv](models.tsv) | each of the 151 random model files: its md5 (12 characters) in the four trees and in the hand step, and whether each tree's two runs match |
| [seeds.tsv](seeds.tsv) | the three generating tests in each tree: the seed each printed and its arguments |
| [ctest.tsv](ctest.tsv) | every ctest test's result in C1007's run of `af00bb13b` (`before`) and the four trees |
| [context.tsv](context.tsv) | each tree's sweep log: lines, model rows, verdict counts, the summary line, md5 of its lines in order and sorted |
| [context_pairs.tsv](context_pairs.tsv) | the sweep logs compared in pairs: base twice, head twice, base against head in each run |
| [checks.tsv](checks.tsv) | `cpu_tests.sh`'s tree, build and ctest verdicts of the four trees |
| [run/run.sh](run/run.sh) | the driver: the clone, the two runs, the checksums, the sweep, the comparison |

## Commands

On the build machine, with a llama.cpp clone holding `af00bb13b` in `$SEED`, `cpu_tests.sh` and `run.sh` in `$REPO`:

```bash
W=<work dir> SEED=<llama.cpp clone> REPO=<folder with cpu_tests.sh and run.sh> \
  CPU_CCACHE=<an earlier CPU run's ccache, optional> BEFORE=<an earlier cpu_tests.sh tree dir of af00bb13b, optional> \
  HAND_MD5=<an earlier hand step's models.md5, optional> LOCK=<flock file, optional> JOBS=8 \
  bash run.sh
```

It fetches `pascal-p106-v5-ctest-seed` from `https://github.com/Cinderella-Man/miner-llama` and stops if the branch is
not at `55e14cb79`. The tables are `$W/checks/*.tsv`. Every container carries the labels `pascal-quants=1` and
`c1008=1`, runs at `--cpu-shares=256` and `--memory=16g`, without network.

Provenance: the fork's own code and ctest's own random models; no downloaded model or data. Licence: the tables and the
script are this repository's (MIT).
