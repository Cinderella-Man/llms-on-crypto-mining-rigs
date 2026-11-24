# The batch-API clean-up landed in the v5 engine (C880, Q1589; 2026-11-24)

What this is: the check before the fork's own tools and tests, moved to upstream's new batch calls
([C880](../2026-11-21_c880_batch_api/README.md), checked on the cards in
[C957](../2026-11-23_c957_batch_api_gpu/README.md)), went into Miner Llama's cumulative v5 branch. The branch
`pascal-p106-v5-batch-api` (`aebff903e`) was merged into `pascal-p106-v5` at `14d679604` (the contributor's pull
requests 8 to 10 and the decode kernel's mask fix), giving merge commit `8a9629942` on `pascal-p106-v5-land-batch-api`.
It was built and checked on a build machine's processor against `14d679604`'s build from the
[pull request 9 and 10 review](../2026-11-23_pr9_pr10_review/README.md), and `pascal-p106-v5` was then
fast-forwarded to `8a9629942`. Registry row Q1589 in [quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **It landed, and nothing that runs a model changed.** The merge touches the same eight files as the branch (three
  example tools, five tests) and nothing in `src/`, `ggml/`, `include/`, `common/` or `tools/`; its diff against
  `14d679604` is the branch's own diff, line for line.
- **The card code is the same, byte for byte.** The whole SASS and PTX dumps of the new rig build's `llama-server`
  equal those of `14d679604`'s build (9,518 kernel copies).
- **The deprecation warnings are gone and nothing new came:** the rig build has 32 warning lines instead of 38 and a
  CPU build of every target 2 instead of 26; every line that went is a call of one of the three deprecated functions.
- **Everything else gives the base's results:** the logits of `test-llama-archs`' 148 random models bit for bit (rig
  builds and CPU builds), ctest test by test (71 of 75 pass on both, the four known failures), the nine hand-run tests,
  and the eight changed tools and tests in all 129 of C880's comparisons.
- **The new rig tarball** is `llama.cpp-20261124-8a9629942-pascal-p106-v5-land-batch-api.tar.gz` (842,627,482 bytes,
  md5 `fd50fb9c9a4634b9d20ce9d14b6c6185`). Its programs other than `llama-pipespec` and `llama-screen-check` (the two
  changed tools in it) differ from the base's only in what the build stamps into them (below). No rig run is needed:
  the engine is `14d679604`'s.

## The checks

All on a build machine's processor (Ryzen 9 5900X), in containers without network, no GPU. Base `14d679604`, head
`8a9629942`. [checks.tsv](checks.tsv) has the same table as a file.

| Check | Base `14d679604` | Head `8a9629942` | Verdict |
|---|---|---|---|
| Rig build (CUDA 12.9, sm_61, `TESTS=1`, `test-llama-archs` and `test-cuda-keepalive` added): warning lines | 38, of them 6 deprecated batch calls | 32, of them 0 | none new ([warnings.tsv](warnings.tsv)) |
| CPU build of every target: warning lines | 26, of them 24 deprecated batch calls | 2, of them 0 | none new |
| Card code: `cuobjdump -sass` of `llama-server` | 12,473,975 lines | 12,473,975 lines | identical, byte for byte |
| Card code: `cuobjdump -ptx` of `llama-server` | 11,366,980 lines | 11,366,980 lines | identical, byte for byte |
| Card code: one SHA-1 per kernel copy | 9,518 | 9,518 | identical ([sass_summary.txt](sass_summary.txt)) |
| `test-llama-archs -b CPU -s 1` of the rig builds | writes 148 logits files, 153 of 153 pass | 153 of 153 pass, plain and comparing | 148 of 148 files the base's, bit for bit |
| ctest, 75 tests, no network (CPU build) | 71 pass; fail: `test-kv-cache` (aborted), `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` | the same | the same, test by test ([ctest.tsv](ctest.tsv)) |
| `test-llama-archs -b CPU -s 1` of the CPU builds | 153 of 153 plain, write, compare; 148 files | the same | 148 of 148 files the base's, bit for bit |
| Hand-run tests (`test-hop-cast --unit-only`, `test-route-once` x2, `test-bytes-switches` x2, `test-moe-ids`, `test-thread-safety`, `test-llama-context`, `llama-eval-callback`) | 9 of 9 pass | 9 of 9 pass | the same verdicts |
| The eight changed tools and tests, C880's `tools.sh` with each CPU build on the same random models | | | 129 of 129 comparisons the same ([tools.tsv](tools.tsv)) |

ctest leaves out the seven tests every run of `cpu_tests.sh` leaves out (three need a download, one needs PyTorch,
three run in the hand step on a generated model instead); C880's README lists them. The four failures are the base's
known set, the same as in C908's and the review's runs; the head's table is also the review build `82d9eb5ef`'s,
test by test (that build is `747d81040`'s tree, and `14d679604` changes only one card kernel).

The hand-run tests' random models are new on every run, so their verdicts are compared, not their logs (row C933,
Q1671). `tools.sh` uses fixed models (`test-llama-archs -s 1`): their three files came out byte for byte the same as in
C880's run of 2026-11-21.

What `tools.sh` covers is in [C880's README](../2026-11-21_c880_batch_api/README.md): `llama-pipespec` through
`bench/pipespec/check_identity.sh` and `check_streams.sh` (63 result lines in every field but timings),
`llama-pipe-recurrent-check`, `llama-screen-check` greedy and sampled on a model with a screened head, `test-moe-ids`,
`test-hop-cast` with its table, and the 24 arms of the three pool tests; 35 exit codes, 30 logs line for line.
As in C880 and C957, the pipespec checks exit 1 and `llama-pipe-recurrent-check` exits 2 with both builds: pipelined
decoding's logits differ from plain decoding's on this engine (C932, Q1670; its fix `pascal-p106-v5-pipe-logits` is
not in `pascal-p106-v5` yet). The exit codes and every line are the same in both builds.

### The rig binaries

[binaries.tsv](binaries.tsv) lists every program of both tarballs with its md5 and size. `llama-pipespec` and
`llama-screen-check` changed (their code did). Each of the other eleven has the base's size and differs in 28 bytes
(4 programs), 29 (5), 38 (`llama-pipe-drafter-check`) or 39 (`test-llama-archs`). In `llama-server` the 29 are the
build ID note (20 bytes), the commit id the build writes in (`14d6796041` to `8a9629942a`, 8 characters differ) and
the build number (12,988 to 12,990, one byte). In the two with 38 and 39, the rest are references to a one-letter
string `"a"`, which the linker now places at the end of the new commit id, `8a9629942a`, instead of at the end of
another string.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check: base, head, verdict |
| [tools.tsv](tools.tsv) | every comparison of the eight tools and tests between the two CPU builds (C880's `compare.py`): item, kind, the two sides, verdict, detail |
| [ctest.tsv](ctest.tsv) | ctest, test by test, both builds |
| [warnings.tsv](warnings.tsv) | every warning line of the rig and CPU builds (each once, without line numbers; the CPU build's tree path shown as `<tree>`), and whether it is in each build |
| [binaries.tsv](binaries.tsv) | the rig tarballs' programs: md5, size and how many bytes differ |
| [sass_summary.txt](sass_summary.txt) | the card-code comparison's summary; its last line is `sass_compare.py compare`, whose 176 "different" are its own pairing (it compares every kernel against the other build's kernel of that name with the `kq_fp32` flag off, so the 176 kernels with the flag on meet a different kernel); the dumps themselves are identical |
| [run/run.sh](run/run.sh) | the driver: the steps below |

## Commands

On a build machine with Docker, from a folder `REPO` that holds [run/run.sh](run/run.sh),
`scripts/build_llamacpp_docker.sh`, [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh),
[sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py), C880's
[tools.sh](../2026-11-21_c880_batch_api/run/tools.sh), [add_screen_head.py](../2026-11-21_c880_batch_api/run/add_screen_head.py)
and [compare.py](../2026-11-21_c880_batch_api/run/compare.py), and `bench/pipespec/` (as `REPO/pipespec`, with
`check_identity.sh`, `check_streams.sh` and `prompts/`):

```bash
W=<work dir> SEED=<llama.cpp clone holding 14d679604> REPO=<that folder> \
  BASE_RIG=<14d679604's rig build folder> BASE_RIG_LOG=<its build log> BASE_CTEST=<a ctest table of the base engine> \
  RIG_CCACHE=<ccache dir> CPU_CCACHE=<ccache dir> LOCK=<flock file> JOBS=8 bash "$REPO/run.sh"
```

Steps (`STEPS`, default all, in this order): `src` (a clone, the merge branch fetched and checked against its commit),
`rig` (the rig tarball, its md5s and warnings), `archs` (the rig builds' logits), `sass` (the card code), `base` and
`head` (`cpu_tests.sh`: a CPU build of every target, ctest without network, `test-llama-archs` written and compared,
the hand-run tests; the head's logits against the base's), `models` (the random models), `tbase` and `thead`
(`tools.sh` with each CPU build), `compare`. The base's rig build is the one the pull request 9 and 10 review made with
the same script and settings (`TESTS=1`, the same two extra targets, the web page of b12483). Images:
`nvidia/cuda:12.9.2-devel-ubuntu24.04` for the rig build, the logits and `cuobjdump`; `ubuntu:24.04` with
`git cmake build-essential ccache python3 python3-jinja2` for the CPU steps. The run took 15 minutes (01:34 to 01:49):
the rig build 2 minutes from a warm compiler cache, the card-code dumps 6.

Provenance: everything here comes from the fork's code and random-weight models that `test-llama-archs` makes; the
prompts are the repository's own `bench/pipespec/prompts/`. No model file, download or private data.
