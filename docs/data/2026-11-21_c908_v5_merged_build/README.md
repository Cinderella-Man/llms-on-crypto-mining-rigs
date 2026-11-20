# v5 after the five contributions: the merged branch built and checked (C908, C909; 2026-11-21)

Registry rows [Q1645](../../../quants/experiments.md) (C908) and [Q1646](../../../quants/experiments.md) (C909).

**Passed.** Miner Llama's cumulative branch `pascal-p106-v5` with the five contributed pull requests merged and the
`LLAMA_PIPE_GRAMMAR` fix on top builds cleanly, with no compiler warning that `602691153` (v5 before the
contributions) does not have. Its test suite gives the same result test by test. Every random test model gives the
same logits bit for bit, in a native CPU build and in the rig build. Its compiled card code is `602691153`'s in 8,833
of 8,835 kernels; the two others and one new kernel come from pull request 4, and nothing else changed.
`LLAMA_PIPE_GRAMMAR` now turns the pipeline's grammar mode on only for a value other than `0` or empty. So
`pascal-p106-v5` was fast-forwarded to the checked commit, `3dfad876f`. Nothing here ran on a GPU: the check on a P106
rig with every switch unset is the next step.

## What was built

| | Commit | What it is |
|---|---|---|
| base | `602691153` | `pascal-p106-v5` before the contributions: upstream b12483 with #32434 carried ([its build](../2026-11-17_c854b_upstream_newest/README.md), [its P106 check](../2026-11-18_c854b_p106_check/README.md)) |
| merged | `6644c2bc6` | the base with pull requests [#3](https://github.com/Cinderella-Man/miner-llama/pull/3) (`39668e199`, a temperature-0 speculation replay accepted as it is), [#5](https://github.com/Cinderella-Man/miner-llama/pull/5) (`66612a5fe`, `--spec-ckpt-vram`), [#4](https://github.com/Cinderella-Man/miner-llama/pull/4) (`079c5ac59`, `GGML_CUDA_HOP_MAILBOX_GATE_KB`) and [#7](https://github.com/Cinderella-Man/miner-llama/pull/7) with #6's commit (`6644c2bc6`: `--mtp-device`, `--pipe-guess mtp`, `--pipe-mtp-walk`, `LLAMA_PIPE_GRAMMAR`, and the draft shortlist in Ornith's kind of MTP head) |
| head | **`3dfad876f`** | the merged branch plus C909: `LLAMA_PIPE_GRAMMAR` unset, empty or `0` is off. Branch `pascal-p106-v5-pipe-grammar-off`; `pascal-p106-v5` now points here |

Every new switch is off by default. Between base and head, 13 files changed; the only card code among them is
`ggml/src/ggml-cuda/hop-mailbox.cu` (pull request 4).

**C909, the fix.** As merged, `pipe_grammar_on()` in `tools/server/server-context.cpp` read the switch as
`getenv("LLAMA_PIPE_GRAMMAR") != nullptr`, so any value turned the mode on, `0` included. It now reads
`e != nullptr && atoi(e) != 0`, the way the fork's other off-by-default switches read (for example
`LLAMA_DRAFT_PREFETCH`). That function is the variable's only reader; the other mentions are comments. Like every
other fork switch read with `atoi`, a word such as `true` counts as off: the documented value is `1`.

## Builds

Both on a build machine's CPU in Docker (`--cpu-shares=256`, 8 compile jobs).

| Build | Result |
|---|---|
| Rig tarball: [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) `fork/pascal-p106-v5-pipe-grammar-off` with `TESTS=1`, `EXTRA_TARGETS="test-llama-archs test-cuda-keepalive"` and the web UI archive pinned (`HF_UI_VERSION=b12483`) | rc 0; **`llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, 841,414,022 B, md5 `0b5b983e22d7296ff44fe363d55f61bb`**; sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, 0 AVX instructions, highest glibc symbol GLIBC_2.38 |
| Native CPU build of every target, base and head ([cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), shared libraries) | both rc 0, 0 errors |

The rig build's binaries (md5): `llama-server` `5b330eccef892b7dd9f4faf1027319c3`, `test-backend-ops`
`526d0231d933867a14e51f11050c1b79`, `test-llama-archs` `d3368d24ec3621f0ebf6c604dd971287`, `test-cuda-keepalive`
`b413207e3962a64031620ae8dbf27c7f`, `llama-bench` `3dff85038f459d7b8c9772799fdf08c6`, `llama-completion`
`f72d03aa56d7ed2f2795c2d15e362ccd`.

## Checks

| Check | Base `602691153` | Head `3dfad876f` |
|---|---|---|
| Compiler warnings, rig build ([warnings.tsv](warnings.tsv)) | 42 lines, 38 distinct (the base's own tarball log) | **42 lines, the same 38**, none new |
| Compiler warnings, native CPU build | 31 lines, 26 distinct | **31 lines, the same 26**, none new |
| ctest, 75 tests without network ([ctest.tsv](ctest.tsv)) | 71 pass; 4 fail | **the same, test by test** |
| `test-llama-archs -b CPU -s 1`, native: plain, write, compare ([cpu_summary.tsv](cpu_summary.tsv)) | 153 of 153 pass each time; 148 logit files | the same |
| The head's logits against the base's, native CPU | — | **148 compared, none differ** |
| The head's rig build against the base's rig build (`test-llama-archs -b CPU`, the CUDA backend loaded against the driver stub) | 153 of 153 pass, 148 files written | **153 of 153 pass, 148 compared, none differ** |
| Hand-run tests: `test-hop-cast`, `test-route-once` (×2), `test-bytes-switches` (×2), `test-moe-ids`, `test-thread-safety`, `test-llama-context`, `llama-eval-callback` | all pass | all pass |
| Server: `LLAMA_PIPE_GRAMMAR` ([server.tsv](server.tsv)) | — | **unset, empty, `0` and `true` off; `1` on** |
| Card code, `cuobjdump -sass` of `llama-server` ([sass_summary.txt](sass_summary.txt)) | 8,835 kernel names | **8,833 identical**, 2 differ, 1 new, none missing; all from pull request 4 |

**The ctest failures and the tests ctest does not run** are the base's own, as on every v5 build since b12483:

- fail on both: `test-kv-cache` (aborts at `tests/test-kv-cache.cpp:164`, `GGML_ASSERT(add(0, 6, 1) == 0)`:
  upstream's test, added with b12483, expects freed cells to be taken in cell order, which the fork's own
  `llama_kv_cache::seq_rm` does not do; registry Q1590); `test-jinja` and `test-jinja-py` (some cases read files from
  the source tree, two TTS chat templates and `scripts/chat_tts_norm.py`, at paths the compiler cache's base folder made
  relative, so they are not found from the build folder: the build setup, not the code); `test-kv-cache-scales` (a
  Python test; the build image has no `torch`);
- not run in ctest: `test-download-model`, `test-eval-callback-download-model`, `test-arg-parser` (they download a
  model or reach the network) and `test-conversion-fp8` (Python with torch); `test-llama-context-model`,
  `test-thread-safety` and `test-eval-callback`, which need that downloaded model, run in the hand step on a generated
  model instead, and pass.

Those are the 11 tests that do not pass in a build container without network on the base as on the head.

**The server check.** Both rig builds' `llama-server` (`-bs`, one slot, 6 threads), on a random dense llama model
written by `test-llama-archs -s 1` with a real tokenizer (the vocabulary-only GGUF pull request 7's review used), five
requests each ([client.py](run/client.py)):
greedy text, sampled text at seed 7, a JSON schema greedy and sampled, greedy text again. The CUDA backend loads against
the driver stub and finds no card, so everything runs on the CPU.

| Run | `LLAMA_PIPE_GRAMMAR` | Pipelined requests | Guesses | Right | Wrong | Grammar rejections | Tokens | Grammar mode |
|---|---|---:|---:|---:|---:|---:|---|---|
| base, `-bs` | unset | 0 | 0 | 0 | 0 | — | — | no pipe |
| head, `-bs` | unset | 0 | 0 | 0 | 0 | — | 5 of 5 the base's | no pipe |
| base, `--pipe-depth 3` | unset | 5 | 370 | 37 | 113 | (not printed) | — | off |
| head, `--pipe-depth 3` | unset | 5 | 370 | 37 | 113 | 0 | 5 of 5 the base's | off |
| head, `--pipe-depth 3` | empty | 5 | 370 | 37 | 113 | 0 | 5 of 5 the base's | off |
| head, `--pipe-depth 3` | `0` | 5 | 370 | 37 | 113 | 0 | 5 of 5 the base's | **off** |
| head, `--pipe-depth 3` | `true` | 5 | 370 | 37 | 113 | 0 | 5 of 5 the base's | off |
| head, `--pipe-depth 3` | `1` | 5 | 480 | 45 | 144 | 15 | 5 of 5 the base's | **on** |

With the mode on, the pipe also runs inside the two JSON-schema requests (110 more guesses, 15 tokens drawn on the
device and rejected by the grammar on the host), and every reply keeps the base's tokens. Before the fix, `0` gave the
`1` row: pull request 7's review saw 480 guesses and 15 rejections with `=0` on the same model and requests.

**The card code.** [sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py) hashes each kernel's SASS
(addresses and encoding padding removed); the base's 176 `kq_fp32` kernels are compared apart (identical) and left out
of the name match, as in [C854 step two](../2026-11-17_c854b_upstream_newest/README.md).

| Kernel | Instructions, base / head | What differs | From |
|---|---|---|---|
| `k_hop_mb_send` | 222 / 222 | four constant-bank offsets (`c[0x4][…]`) 8 bytes lower and one `MOV32I 0x0` → `0x80`: the addresses of its printf format strings, which the new code in the same file moved; no instruction changed ([sass_mailbox.diff](sass_mailbox.diff)) | pull request 4 |
| `k_hop_mb_recv` | 252 / 252 | the same: four offsets 8 bytes lower, one `MOV32I 0x0` → `0x80` | pull request 4 |
| `k_hop_mb_gate_copy` | — / 42 | new: the gate's copy kernel, launched only with `GGML_CUDA_HOP_MAILBOX=1` and `GGML_CUDA_HOP_MAILBOX_GATE_KB` set | pull request 4 |

This is the result pull request 4's own review reported for its branch alone; pull requests 3, 5, 6 and 7 changed no
card code.

## Files

| File | What |
|---|---|
| [cpu_summary.tsv](cpu_summary.tsv) | `cpu_tests.sh`'s verdicts for base and head: tree, build, ctest, `test-llama-archs` (plain, write, compare, files), the head against the base's logits, the hand-run tests |
| [ctest.tsv](ctest.tsv) | every ctest test's outcome on base and head |
| [warnings.tsv](warnings.tsv) | every distinct warning of both builds (rig, native CPU), line and column numbers removed, and whether base and head print it |
| [server.tsv](server.tsv) | the server check per run: switch value, the pipeline's counts from the server log, tokens against the base, grammar mode (written by `run/compare.py`) |
| [sass_summary.txt](sass_summary.txt) | the kernel comparison's summary, the `kq_fp32` comparison, the line classification and the kernels' instruction counts |
| [sass_differing.tsv](sass_differing.tsv) | the kernels whose SASS differs |
| [sass_mailbox.diff](sass_mailbox.diff) | the two differing kernels' SASS lines, base against head |
| [run/run.sh](run/run.sh) | the driver: sources, rig build, CPU test set, the rig builds' logits, the server check, the card code |
| [run/client.py](run/client.py), [run/compare.py](run/compare.py) | the server check's requests and its table |

## Commands

On the build machine, with this repository's scripts in one folder (`REPO`: `scripts/build_llamacpp_docker.sh`,
`docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh`, `docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py`,
`docs/data/2026-11-17_c854b_upstream_newest/run/sass_classify.py` and this folder's `run/`):

```bash
W=<work folder> SEED=<llama.cpp clone holding 602691153> \
BASE_RIG=<the base's rig build folder, llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483> \
BASE_RIG_LOG=<that build's log> VOCAB=<a GGUF with a tokenizer> REPO=<scripts folder> \
RIG_CCACHE=<a compiler cache to seed from> CPU_CCACHE=<another> LOCK=<flock file> JOBS=8 \
  setsid nohup bash $REPO/run.sh > $W/logs/run.out 2>&1 < /dev/null &
```

The run took 17 min 56 s: the rig build 2 min 11 s (from a warm compiler cache), the CPU test set 6 min 49 s, the rig
builds' logits 1 min 1 s, the server check 1 min 13 s and the card code 6 min 41 s. Every container carries
`--cpu-shares=256`; every one that runs the code also `--network none` and `--memory=16g`. Only the rig build has
network, to fetch the sources and the web UI archive.

## Provenance

The fork's commits are public on [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama). The test
models are random (`test-llama-archs`). The base's rig build and its build log are those of
[C854 step two](../2026-11-17_c854b_upstream_newest/README.md). Raw logs are not published.
