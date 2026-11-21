# The fork's own tools and tests on upstream's new batch API (C880, Q1589; 2026-11-21)

What this is: a clean-up of the v5 engine's own tools and tests, built and checked on a build machine's CPU. Upstream
llama.cpp marked its old batch calls (`llama_batch_init`, `llama_batch_free`, `llama_decode`) deprecated in #32306,
which reached the v5 line with tag b12483, in favour of a new batch object (`llama_batch_ext`) and one call to run it
(`llama_process`). Upstream's own code and the fork's server already use the new calls; eight of the fork's own files
did not, and they made 24 of the 26 warning lines of a CPU build. Fork branch `pascal-p106-v5-batch-api`
(`aebff903e`, one commit on `pascal-p106-v5` `3dfad876f`). Registry row Q1589 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **The deprecation warnings are gone and nothing new came.** A CPU build of every target gives 2 warning lines
  instead of 26: the 24 that went were the eight files' calls of the three deprecated functions (three lines a file),
  and the 2 left are the base's own (a missing initializer in `common/sampling.cpp`, a CMake note about the web UI
  archive). The rig build gives 32 lines instead of the base's 38: the 6 that went are `llama-pipespec`'s and
  `llama-screen-check`'s (the only two of the eight it compiles). No warning line is in the new builds that is not in
  the old.
- **Every tool and test gives the same output as before.** Run on the same random models with the old binaries and
  the new ones, all 129 comparisons match: every exit code, every result line of `llama-pipespec` (63 lines: the
  tokens, the logits of every token, the KV state and the probe, as hashes), `llama-pipe-recurrent-check`'s result,
  `llama-screen-check`'s counts, and the logs of the five tests line for line, `test-hop-cast`'s table included.
- **The rest of the suite is the base's.** ctest test by test (71 of 75 pass on both, the same four known failures),
  the logits of `test-llama-archs`' 148 random models bit for bit, and the nine hand-run tests' verdicts.
- **No P106 is needed for what changed.** The changed lines only fill a batch on the host; which backend then runs
  it does not enter. The three k-pool and pool-mask tests run on the CPU backend by design and `test-moe-ids` is a CPU
  check. `test-hop-cast`'s card parts (each card's round trip against the CPU's, its arms across cards) need cards, and
  feed their model through the same changed lines as the CPU run here. With no engine change, no rig tarball comes from
  this row and nothing is to be run on a rig.
- **A finding on the side, older than this change:** on the CPU, pipelined decoding no longer gives plain decoding's
  logits (the tokens still agree). Both builds give the same hashes, so the change made no difference; it is its own
  row (C932, Q1670), below.

## What changed in the code

The commit touches only `examples/` and `tests/`: nothing in `src/`, `ggml/` or `tools/server/`, so the engine and
the server are the same code, and no rig tarball comes from this row.

| File | Change |
|---|---|
| `examples/pipespec/pipespec.cpp` | `decode_one` (one token of one sequence, the plain mode's reference decoding and the probe after generation) and the prompt, decoded in calls of `--pipe-chunk` tokens (`n_batch` without one), fill a `llama_batch_ext` and run it with `llama_process` |
| `examples/pipe-recurrent-check/pipe-recurrent-check.cpp` | the same for its `decode_one` and its prompt |
| `examples/screen-check/screen-check.cpp` | `decode_tokens` the same |
| `tests/test-moe-ids.cpp`, `tests/test-hop-cast.cpp` | each test's batch the same; one batch reused across the test's calls, as before |
| `tests/test-kpool-dev.cpp`, `tests/test-pool-mask-dev.cpp`, `tests/test-kpool-stage.cpp` | these decode the same entries on three or four contexts. A `llama_batch_ext` is made for one context and checks its positions against that context's memory, so the entries are drawn once, in the same order as before, and each context gets its own batch |

How each call maps: `llama_batch_init` becomes `llama_batch_ext_init(ctx)` (a batch holds up to the context's
`n_batch` entries) and `llama_batch_free` becomes `llama_batch_ext_free` (here through `llama_batch_ext_ptr`); each
entry is `llama_batch_ext_add_token(batch, seq, token)`, then `llama_batch_ext_set_pos` and
`llama_batch_ext_set_output_logits`; `llama_decode(ctx, batch)` becomes
`llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)`. This is upstream's `examples/simple` pattern. The old
`llama_decode` already turned its argument into a `llama_batch_ext` with the same entries (upstream's
`llama_batch_compat`) and ran the same decode, so the engine sees the same input as before. Two things differ only on
failure: a call with more entries than `n_batch` now fails with a message instead of the decode's assertion, and
`llama_process` does not print `llama_decode`'s "failed to decode" line (each tool prints its own).

## The checks

All on a build machine's CPU (Ryzen 9 5900X), in containers without network, no GPU. The base is `pascal-p106-v5`
`3dfad876f`, the head `aebff903e`; both built from the same compiler cache at the same paths.

| Check | Base `3dfad876f` | Head `aebff903e` |
|---|---|---|
| Fail-fast build of the eight changed targets | | built, 0 errors, 0 deprecation warnings |
| CPU build of every target: warning lines (each once, without line numbers) | 26, of them 24 deprecated batch calls | 2, none new |
| Rig build (CUDA 12.9, sm_61, `TESTS=1`): warning lines | 38 (C908's build of this commit), of them 6 deprecated batch calls | 32, none new |
| ctest, 75 tests, no network | 71 pass; fail: `test-kv-cache` (aborted), `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` | the same, test by test |
| `test-llama-archs -b CPU -s 1`: plain, write, compare | 153 of 153 each; 148 logits files | the same; 148 of 148 files the base's bit for bit |
| Hand-run tests (`test-hop-cast --unit-only`, `test-route-once` x2, `test-bytes-switches` x2, `test-moe-ids`, `test-thread-safety`, `test-llama-context`, `llama-eval-callback`) | 9 of 9 pass | 9 of 9 pass |
| [tools.tsv](tools.tsv): `tools.sh` with each build on the same models | | 129 of 129 comparisons the same |

ctest leaves out seven tests, as in every run of the script: three need a download (`test-download-model`,
`test-eval-callback-download-model` and `test-arg-parser`'s web case), one needs PyTorch (`test-conversion-fp8`), and
three run in the hand step on a generated model instead (`test-llama-context-model`, `test-thread-safety`,
`test-eval-callback`). The hand step's models are random on every run (`test-llama-archs` seeds from the system's
random device without `-s`), so its outputs are compared by verdict only; the same commit printed `llama-eval-callback`
sums of 0.124211 and -2.255645 in two runs. Giving that step a fixed seed is row C933 (Q1671). `tools.sh` uses fixed
models (`-s 1`), whose files came out byte for byte the same in both runs of the night.

What `tools.sh` ran, each with both builds ([tools.tsv](tools.tsv) has every line):

| Tool or test | Runs | Result (the same with both builds) |
|---|---|---|
| `llama-pipespec` via `bench/pipespec/check_identity.sh` | greedy (14 jobs, depths 1 and 3), the prompt in calls of 7 tokens (7 jobs), seeded sampling (12 jobs), a race of two samples (7 jobs); 96 tokens (64 in the race) on a random qwen4exp | every job's PIPESPEC line the same in both builds; the plain job of each the same tokens as every pipelined job |
| `llama-pipespec` via `bench/pipespec/check_streams.sh` | two conversations, their plain references in sequences 0 and 1 with the prompt in calls of 16 tokens, seven streams jobs | every line the same in both builds |
| `llama-pipe-recurrent-check` | 96 greedy tokens after a 550-token prompt, three pipe slots, 32 in-place submits, 24 guesses, 32 dropped tokens | the same result line |
| `llama-screen-check` | three prompts, 32 tokens each, greedy and seeded sampling, on the model with a screened head (64 of 4,096 candidates) | passed in both: 96 steps, 6,144 candidate logits bit for bit, the same tokens |
| `test-moe-ids` | the MoE model with 32 experts, 8 used | 24 nodes, 146 token columns, 13 checks, 0 failed |
| `test-hop-cast` | `--unit-only`; `-m` with the arms off and q8_0 | 54 checks; 64 checks, the same table (q8_0: 10 hops, max logit difference 0.00103594, top-1 1.000000) |
| `test-kpool-dev`, `test-pool-mask-dev`, `test-kpool-stage` | 8 arms each (per-sequence and unified KV, top-k 0 and 32, micro-batches of 16 and 512), seed 1, 240 steps | every arm passes; 887, 1,172, and 4,064 (16) or 3,564 (512) checks a run, 0 failed; e.g. 239 decodes and 1,020 outputs compared a run of `test-kpool-stage` |

The pipespec scripts exit 1 and `llama-pipe-recurrent-check` exits 2 with both builds: their own check, pipelined
against plain decoding, fails on the CPU. That is the finding below, not this change.

## A finding on the side

On this engine's CPU build, pipelined decoding no longer gives plain decoding's logits. In every pipelined job above
the tokens are the plain job's, but the hashes of the logits, of the KV state after generation and of the probe
(tokens decoded afterwards, which read every KV cell still visible) differ, from the pipeline without guesses
(`pipe-d0`) on. The pipelined jobs with no wrong guess agree with each other (greedy: logits hash `910013568c6e0000`
for `pipe-d0`, `oracle-d1-m0`, `copy2`, `simple`, `mapk`, against plain's `274fa13a0ec5bf0c`), and those that drop wrong
guesses (`oracle-*-m0.5`, `cont4`) differ again. `llama-pipe-recurrent-check` gives the same tokens and logits that
differ from the second generated token on. It holds with flash attention on and off, with the default vocabulary and
with 4,096 tokens, at `-c 2048` and `-c 4096`. The 2026-11-03 check of pipelined speculation
([Q550](../2026-11-03_flash_next_pipelined/README.md)) found every hash the same on the CPU. The old and the new binaries
here give identical hashes, so the cause is older than this change. Which commit, and whether a P106 shows it too, is
row C932 (Q1670).

## Files

| File | What |
|---|---|
| [tools.tsv](tools.tsv) | every comparison of the tools and tests between the base build and the head build: item, kind (exit code, PIPESPEC line, log, TSV), the two sides, verdict and detail |
| [run/run.sh](run/run.sh) | the driver: the steps below, in containers without network on the build machine's CPU |
| [run/tools.sh](run/tools.sh) | the tools and tests the commit changes, run with one build's binaries on fixed random models |
| [run/add_screen_head.py](run/add_screen_head.py) | copies a GGUF and adds a screened output head (`output_screen.weight`, a copy of the output head, and `<arch>.output_screen.top_m`), so that `llama-screen-check` runs on a random model |
| [run/compare.py](run/compare.py) | compares the two `tools.sh` outputs and writes `tools.tsv` |

## Commands

On a build machine with Docker, from a folder `REPO` that holds `run.sh`, `tools.sh`, `add_screen_head.py`,
`compare.py`, `scripts/build_llamacpp_docker.sh`, `docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh` and
`bench/pipespec/` (as `REPO/pipespec`, with `check_identity.sh`, `check_streams.sh` and `prompts/`):

```bash
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> REPO=<that folder> BASE_RIG_LOG=<the base's rig build log> \
  CPU_CCACHE=<ccache dir> RIG_CCACHE=<ccache dir> LOCK=<flock file> JOBS=8 bash "$REPO/run.sh"
```

Steps (`STEPS`, default all, in this order): `src` (a clone, the fork's branch fetched and checked against its
commit), `quick` (only the eight changed targets of the head, fail-fast), `base` and `head` (`cpu_tests.sh`: a CPU
build of every target, ctest without network, `test-llama-archs` written and compared, the hand-run tests; the head's
logits against the base's), `models` (the random models, from the base's `test-llama-archs`), `tbase` and `thead`
(`tools.sh` with each build), `rig` (the rig build, for its warnings), `compare`. The images: `ubuntu:24.04` with
`git cmake build-essential ccache python3 python3-jinja2` for the CPU steps, `nvidia/cuda:12.9.2-devel-ubuntu24.04`
for the rig build; the two Python scripts need only the standard library.

Provenance: everything here comes from the fork's code and random-weight models that `test-llama-archs` makes; the
prompts are the repository's own `bench/pipespec/prompts/`. No model file, download or private data.
