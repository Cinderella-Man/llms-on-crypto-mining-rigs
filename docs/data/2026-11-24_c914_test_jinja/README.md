# test-jinja and test-jinja-py counted in the CPU test runs (C914, Q1651; 2026-11-24)

What this is: a fix to the CPU test script [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) (C565), which
builds every fork branch and merge on a build machine's processor and runs upstream's and the fork's tests on it. Two of
upstream's tests, `test-jinja` and `test-jinja-py`, had failed in every run since the script was written, because of how
the script builds, not because of the code, so they sat among the known failures and checked nothing. One CPU test run
of Miner Llama's cumulative branch `pascal-p106-v5` at its tip `8a9629942` with the changed script shows them passing,
and every other check as before. Registry row Q1651 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **Both tests now pass and count.** ctest passes 73 of 75 tests instead of 71; the two that changed are `test-jinja`
  and `test-jinja-py`, from failed to passed.
- **The known failures went from 11 to 9.** Two tests still fail: `test-kv-cache` (upstream's test against the fork's
  own `seq_rm`, C881) and `test-kv-cache-scales` (it needs `torch`, which the test image does not have). Seven are left
  out of ctest on purpose: five need upstream's downloaded model or the network, one needs `torch` and `safetensors`
  (`test-conversion-fp8`), and `test-arg-parser` needs the network. Three of the seven (`test-llama-context-model`,
  `test-thread-safety`, `test-eval-callback`) run in the script's hand step on a generated model instead, and pass.
- **Nothing else changed:** the other 73 tests give the same result, test by test; `test-llama-archs`' 148 random models
  give the same logits bit for bit as the earlier run of the same commit; the nine hand-run tests pass as before.

## Why the two tests failed, and the change

Some cases of `test-jinja` (two TTS chat templates, `models/templates/Zyphra-ZONOS3-Mini-tts.jinja` and
`MiniMaxAI-MiniMax-Music3-tts.jinja`) and of `test-jinja-py` (also `scripts/chat_tts_norm.py`, the Python reference of
the TTS text normalizer) open files at a path made from `__FILE__`, the test's own source path as the compiler saw it.
The script compiles through ccache with a base folder (`CCACHE_BASEDIR`), and ccache then passes the compiler every
source path relative to the compile folder, so in `build/tests/` the test's `__FILE__` is `../../src/tests/test-jinja.cpp`.
ctest runs both tests from the build folder one level up, where `../../src` points beside the tree, so the files were
not found: 2 of 505 cases of `test-jinja` and 24 of 442 of `test-jinja-py` failed.

The script now makes the link `$WORK/src -> tree/src`, which is where `../../src` points from the build folder
(`$WORK/tree/build`). The build itself is as it was. Of the two ways the task named, this is the smaller and safer one:

- **Building without the base folder** would make `__FILE__` absolute, but the base folder is what lets a compiler cache
  copied from another work folder hit: every recent build-and-check script starts from another run's cache
  (`CPU_CCACHE`, hard-linked). Without the base folder ccache hashes the absolute paths, and each such run would build
  every target cold. This run's build took 313 of its 330 compiles from the cache of the earlier run in another folder
  (ccache's counters, below), which only works with the base folder.
- **Running the two tests from `build/tests/`** would mean taking them out of the ctest step and running them on their
  own, so the ctest table would no longer hold every test.
- **The link** is one line, changes nothing in the build, the binaries or any test's working folder, and leaves the
  cache shared between work folders as it was. Tree tags named `src`, `tree` or `ccache` (folders of the script's own)
  are now skipped with a message.

## The checks

On a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU; the build under the machine's
build lock with `-j 8` at a low CPU share. **Before** is the CPU test run of the same commit `8a9629942` by the script
before the change, from [C880's landing](../2026-11-24_c880_land/README.md) (tree `land`); **after** is this run,
05:24:21–05:26:51 (2 min 30 s, the build from the cache).

| Check | Before | After | Verdict |
|---|---|---|---|
| ctest, 75 tests ([ctest.tsv](ctest.tsv)) | 71 pass | 73 pass | `test-jinja` and `test-jinja-py` from failed to passed; the other 73 the same, test by test |
| Known failures ([known_failures.tsv](known_failures.tsv)) | 11 | 9 | the two above gone; the other 9 the same |
| `test-jinja` cases ([jinja_cases.tsv](jinja_cases.tsv)) | 505, 2 failed | 535, 0 failed | the two TTS templates open: 29 and 3 assertions run and pass |
| `test-jinja-py` cases | 442, 24 failed | 472, 0 failed | the two templates and the normalizer's 15 and dictionary's 8 cases pass |
| `test-llama-archs -b CPU -s 1`: plain, writing, comparing | 153 of 153 pass, 148 files the same | 153 of 153, 148 files the same | the same ([checks.tsv](checks.tsv)) |
| Logits against the earlier run's files (ident) | — | 148 of 148 compared, none differ | the same, bit for bit |
| Hand-run tests (hop-cast, route-once ×2, bytes-switches ×2, moe-ids, thread-safety, context-model, eval-callback) | 9 pass | 9 pass | the same |
| Compiler cache (ccache from the earlier run's folder, counters zeroed first) | — | 313 of 330 compiles hit | the cache works across folders as before |

The known failures, after:

| Test | Why |
|---|---|
| `test-kv-cache` | fails: upstream's test against the fork's own `seq_rm` (C881) |
| `test-kv-cache-scales` | fails: needs `torch` |
| `test-conversion-fp8` | not run: needs `torch` and `safetensors` |
| `test-arg-parser` | not run: its last case needs the network (the script's `net` step runs it with network) |
| `test-download-model`, `test-eval-callback-download-model` | not run: download upstream's model |
| `test-llama-context-model`, `test-thread-safety`, `test-eval-callback` | not run in ctest: need the downloaded model; the hand step runs them on a generated model, and they pass |

The case counts after the change come from running both tests by hand from the build folder once the run had ended,
the link in place, as ctest runs them (ctest prints a test's output only when it fails).

## Files

| File | What |
|---|---|
| [ctest.tsv](ctest.tsv) | every ctest test's result, before and after, and whether they are the same |
| [known_failures.tsv](known_failures.tsv) | the 11 known failures before and what each is after |
| [checks.tsv](checks.tsv) | the script's other verdicts (tree, build, archs, hand-run tests), before and after |
| [jinja_cases.tsv](jinja_cases.tsv) | the two tests' own counts of cases, assertions and failures, before and after |
| [run/run.sh](run/run.sh) | the run: a clone of the seed with the fork's branch checked against the commit, the compiler cache copied, `cpu_tests.sh` on one tree with ident against the earlier run's logits, then the comparison tables |
| [../2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) | the CPU test script with the change (the link, and the skipped reserved tags) |

## Commands

```bash
# on the build machine; SEED a llama.cpp clone holding 8a9629942, BEFORE the work folder of the earlier CPU test run
# of the same commit (its tree "land"), REPO a folder with cpu_tests.sh and run.sh
W=<work dir> SEED=<llama.cpp clone> REPO=<scripts> BEFORE=<earlier WORK> BEFORE_TAG=land \
  CPU_CCACHE=<earlier WORK>/ccache LOCK=<build lock file> JOBS=8 bash run.sh
# the two tests by hand, as ctest runs them (from the build folder)
docker run --rm --cpu-shares=256 --memory=16g --network none -v $W:$W -w $W/cpu/tree/build pq-c565-cpu:24.04 \
  bash -c "bin/test-jinja; bin/test-jinja -py"
```

`run.sh` runs `cpu_tests.sh` as `WORK=$W/cpu SRC=$W/src TREES='c914=8a9629942a4635ea0b2711b72e59556b96a63f86@before'
STEPS='tree build ctest archs ident hand' CTEST_JOBS=6 THREADS=6 KEEP_BUILD=1`, with labels and `--memory=16g` in
`DOCKER_ARGS`; the image is `cpu_tests.sh`'s own (Ubuntu 24.04 with the compiler, CMake, ccache, Python 3 and Jinja2).

Provenance and licence: every table here was made by these scripts from their own runs; llama.cpp and its tests are
MIT-licensed.
