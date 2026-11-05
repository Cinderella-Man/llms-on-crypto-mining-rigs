# C565 (Q1050): `-ladder4`'s failing CPU tests, traced and fixed (2026-11-05)

The switch audit (Q1025) found `test-llama-archs` failing 7 of its 148 cases on the cumulative engine branch
`pascal-p106-v3-b12076-ladder4` (`e3b30303d`) with every switch at its default, the same on the branch and on its base.
This folder traces those failures and everything else the whole CPU test set finds, fixes them on the fork branch
`pascal-p106-v3-b12076-ladder4fix`, and runs the whole set before and after, on `-ladder4` and on its variant on upstream
b12127, `pascal-p106-v3-b12127-ladder4` (`a042afb15`). The script is [run/cpu_tests.sh](run/cpu_tests.sh). The registry
row is Q1050 in [quants/experiments.md](../../../quants/experiments.md).

**Result:** two causes, one in a test and one in the code, both older than the day's merges or outside what the rig
builds exercise. Fixed by two commits on `-ladder4`: fork branch `pascal-p106-v3-b12076-ladder4fix` at **`fdd2e565d`**.
With them, every CPU test that needs no download passes on `-ladder4fix` (shared and static builds) and on
`-b12127-ladder4` with the same two commits. Six upstream tests need a download (a 19 MB model, or torch for a Python
test) and did not run (Q1051): two only fetch the model, and the hand step runs the three that use it on a generated model instead.

## The failures

| # | What fails | Where | Cause | Wrong | Fix |
|---|---|---|---|---|---|
| 1 | `test-llama-archs` with `LLAMA_ARCHS_LOGITS` set: 7 of 148 cases, the MoE models of `llama`, `refact`, `minicpm`, `granite`, `granitemoe`, `mistral3` and `k2-horizon`; without the variable all 148 pass | every build since `455cc53b4`: `-ladder3` `7bc1552f4`, `-kshape` `d27845b67`, `-ladder4` `e3b30303d`, `-b12127-ladder4` `a042afb15` | `455cc53b4` (2026-11-03, `-gdnfuse`'s test tool, merged into `-ladder2` at `3db80fa67`) named the logits file after the architecture only. These seven architectures run a Dense and a MoE model, so the MoE run found the Dense model's file and compared its logits with another model's | the test | `fd8276bbc`: the MoE model of an architecture that also has a Dense one writes `<prefix><arch>-moe.f32`; every other file keeps its name, so files written by earlier builds still compare |
| 2 | the build with shared libraries (`BUILD_SHARED_LIBS=ON`, upstream's default): `test-backend-ops` does not link (`undefined reference to ggml_graph_view`), so `cmake --build` stops | `-ladder4`, `-b12127-ladder4`; static builds link (the rig's tarballs, `checks.sh`'s `cpubuild`, `-replay`'s own checks) | `-replay`'s record and replay (`76683f4e4`, merged into `-ladder4` at `b861d7b58`) compute the replayed graph node by node through `ggml_graph_view`, which `ggml-impl.h` declares without `GGML_API`, so `libggml-base.so` hides it | the code (a missing export) | `fdd2e565d`: `GGML_API` on the declaration, as `ggml_op_can_inplace` beside it has |

**Where each came from.** Failure 1 is the same 7 cases on a CPU build of `-ladder3` from 2026-11-04, on `-kshape` and on
`-ladder4`, and `git log -L` shows the file naming unchanged since `455cc53b4`, the commit that added the variable: none
of `-ladder4`'s merges is involved (the audit's "`-ladder4` is the cause" was its base). Failure 2 is the only error of a
keep-going shared build of `e3b30303d` (two undefined references, from `rr_reference` and `rr_replay_compare`), and the
calls came with `76683f4e4`, which `-kshape` does not contain.

**What failure 1 hid.** The identity checks of the merges since `455cc53b4` (`checks.sh`'s `identity` step and its
predecessors) run `test-llama-archs` with `LLAMA_ARCHS_LOGITS`, so they never compared those seven MoE models between
two builds: in both runs the MoE model met the Dense model's file, and the step set those lines aside. With the fix,
`-ladder4fix` against `-kshape` with the same commit gives all 143 files the same, the seven MoE models included.
[`checks.sh`](../2026-11-04_ladder4_merge/run/checks.sh)'s `identity` step now also passes a merge with the fix against
a base from before it (the base still fails the seven, and the merge writes their files itself).

## The CPU test set

Each tree is built on the CPU (`GGML_CUDA=OFF`, `GGML_NATIVE=ON`, `LLAMA_BUILD_TESTS=ON`, every target, a keep-going
build so the log lists every error) in a container without network (Ubuntu 24.04, GCC 13), then:

| Step | What runs | Passes when |
|---|---|---|
| `ctest` | every test the build registers (72 in these trees) but seven: the five that download upstream's `stories15M-q4_0.gguf` (`test-download-model`, `test-eval-callback-download-model` and their users `test-llama-context-model`, `test-thread-safety`, `test-eval-callback`), `test-conversion-fp8` (Python with torch and safetensors), and `test-arg-parser` (the `net` step) | 65 of 65 pass |
| `net` | `test-arg-parser` with network: its last case requests `http://ggml.ai/` and a missing page there | it passes |
| `archs` | `test-llama-archs -b CPU -s 1` three times: without `LLAMA_ARCHS_LOGITS`, then with it twice (write, then compare) | all 148 cases each time; 143 files written, all 143 the same |
| `ident` | the tree's `test-llama-archs` against the logits another tree wrote | every compared file the same (informational across the upstream merge) |
| `hand` | the tests the build only compiles: `test-hop-cast --unit-only`; `test-route-once` (routers ×1, ×30), `test-bytes-switches` (head as it is, `--zero-tail`) and `test-moe-ids` on a random qwen4exp MoE model (4,096 tokens, 32 experts, 8 used, as their own checks make it); `test-thread-safety`, `test-llama-context -m` and `llama-eval-callback` with ctest's arguments on the generated `llama-dense.gguf` in place of the downloaded model | exit status 0 |

## Results

Trees: `ks` = `-kshape` `d27845b67`, `ks+fix` = it with `fd8276bbc` (test-llama-archs only), `l4` = `-ladder4`
`e3b30303d`, `l4fix` = `-ladder4fix` `fdd2e565d`, `u4` = `-b12127-ladder4` `a042afb15`, `u4fix` = it with `fd8276bbc` and
`fdd2e565d`; `l4fix` static = the same commit with `BUILD_SHARED_LIBS=OFF`. Every verdict line is in
[results.tsv](results.tsv); ctest's per-test status in [ctest_tests.tsv](ctest_tests.tsv).

| Check | `-ladder3` | `ks` | `l4` | `u4` | `ks+fix` | `l4fix` | `l4fix` static | `u4fix` |
|---|---|---|---|---|---|---|---|---|
| build, every target | — | — | **fails**: `ggml_graph_view` | **fails**: the same | — | OK | OK | OK |
| ctest, no network | — | — | 65 of 68 (\*) | 65 of 68 (\*) | — | **65 of 65** | **65 of 65** | **65 of 65** |
| `test-arg-parser`, network | — | — | — | — | — | OK | OK | OK |
| `test-llama-archs`, plain | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 | 148/148 |
| `test-llama-archs`, write then compare | **141/148** | **141/148** | **141/148** | **141/148** | 148/148 | 148/148 | 148/148 | 148/148 |
| logits files, all the same on compare | 136 | 136 (7 shared) | 136 (7 shared) | 136 (7 shared) | 143 | 143 | 143 | 143 |
| identity | — | — | — | — | — | = `ks+fix`, 143/143 | — | against `l4fix`: 140/143 (below) |
| by-hand tests | — | — | 8/8 | 8/8 | — | 9/9 | 9/9 | 9/9 |

(\*) The unfixed trees ran ctest without network and without the `net` step: their three failures are
`test-arg-parser` at its URL case ("cannot make GET request"; every earlier case passes) and
`test-eval-callback-download-model` with its user `test-eval-callback`, a download of `stories15M-q4_0.gguf` that the
later passes leave out. They are not failures of the code: `test-arg-parser` passes with network on every fixed tree,
and the eval-callback test's C++ code passes on a generated model (`llama-eval-callback`, the ninth by-hand test, which
the unfixed pass did not have yet). On the fixed trees nothing fails.

## Identity across the upstream merge (for the cumulative build)

`u4fix` against `l4fix`: 140 of 143 files the same; `gpt2` and `qwen2` differ (nmse 0.21 and 0.080) because upstream
b12127 gave both new optional tensors, an output bias in `gpt2` (#30993) and per-tensor FFN scales in `qwen2` (#31657),
and `test-llama-archs` fills every optional tensor of its random models; a file without them builds the graph it did.
`gemma4` differs by rounding (nmse 1.7e-7, largest difference 2.7e-5), its graph reordered upstream (#31356, #31635).
An identity check across an upstream merge expects these three on random models.

## Files

| File | What |
|---|---|
| [run/cpu_tests.sh](run/cpu_tests.sh) | the whole set per tree: `tree`, `build`, `ctest`, `net`, `archs`, `ident`, `hand` (usage in its header) |
| [results.tsv](results.tsv) | every verdict (pass, tree, check, verdict, detail): pass 1 `-kshape` without and with the fix, and `-ladder3`'s existing CPU build by hand; pass 2 the unfixed `-ladder4` and `-b12127-ladder4`; pass 3 the fixed tips, shared and static |
| [ctest_tests.tsv](ctest_tests.tsv) | each of the 72 registered tests' status per tree (`test-arg-parser` from the `net` step on the fixed trees) |

## Commands

On a build machine with Docker and a llama.cpp clone holding the fork's commits (`SRC`), CPU only:

```bash
# the unfixed trees and the first fixed one (logits of -kshape with the fix for the identity check)
WORK=$W SRC=$SRC LOCK=$LOCK DOCKER_ARGS='--label pascal-quants=1' \
  TREES='ks=d27845b67:test-llama-archs ks4fix=d27845b67+fd8276bbc:test-llama-archs l4=e3b30303d u4=a042afb15' \
  bash run/cpu_tests.sh
# the fixed tips, shared libraries, then -ladder4fix static
WORK=$W SRC=$SRC LOCK=$LOCK TREES='l4fix=fdd2e565d@ks4fix u4fix=a042afb15+fd8276bbc+fdd2e565d@l4fix' bash run/cpu_tests.sh
WORK=$W SRC=$SRC LOCK=$LOCK SHARED=OFF TREES='l4fixs=fdd2e565d' STEPS='tree build ctest net archs hand' bash run/cpu_tests.sh
# the six download tests (Q1051): MODEL15M=<stories15M-q4_0.gguf> PIP='torch safetensors numpy' (a new IMAGE tag)
```

`-ladder3`'s point came from its existing CPU build (`7bc1552f4`): the same `archs` runs, by hand.

## Provenance

Fork `github.com/Cinderella-Man/llama.cpp`, branches `pascal-p106-v3-b12076-ladder4fix`, `-ladder4`, `-kshape`,
`pascal-p106-v3-b12127-ladder4`; upstream llama.cpp (MIT). The test models are the random models `test-llama-archs`
generates; nothing was downloaded. Run on 2026-11-05 on the 7-GPU quant machine's CPU (Ryzen 9 5900X, `--cpu-shares=256`,
`-j 8`), no GPU.
