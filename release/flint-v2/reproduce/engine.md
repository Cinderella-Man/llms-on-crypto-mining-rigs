# The engine: source, build and switches

The v2 engine is a llama.cpp fork for GPUs of compute capability 6.1. It is upstream llama.cpp at `d77256999`
(2026-10-22) plus the commits listed on the fork's release branch; every change has its own switch, so any piece can be
turned off and compared with stock behaviour.

## Source

| | |
|---|---|
| Fork | [Cinderella-Man/llama.cpp](https://github.com/Cinderella-Man/llama.cpp) |
| Release commit | `2d53179aa` (rc4: rc3 + the staircase drafter, its store order, the fix of a crash in the drafter path, and an allocator fix for pipelined decoding), branch `pascal-p106-v2-rc4`. OWNER-TBD: pushed to the fork, the release tag and branch name |
| Without the drafter | rc3 `7e6ec762f`, the same engine with the drafter code left out |
| v1, for comparison | `eeaa59660`, branch `pascal-p106` |
| Upstream base | `d77256999` |

Each piece came from a feature branch that was checked on its own (operator tests natively and through the compute_61
PTX, logits byte for byte against the base with every switch off, pipelined output identical to plain decoding) and on
the P106s before it joined the release branch. The branches and their checks are listed in the
[registry](../../../quants/experiments.md#registry): Q38/Q46 (Pascal kernels), Q76 (FP32 sums), Q50 (shared-expert
fusion), Q108–Q110 and Q116 (prefill and the card-built mask), Q88 (persistent MoE kernel), Q87, Q104, Q120, Q121,
Q153–Q156 (pipelined speculation and the server path), Q150–Q152 (host and launch slimming), Q149/Q101 (Q4_K and Q6_K
kernels), Q151 (decode attention), Q163 (prefill attention), Q67 (GPU sampling for tool requests), Q111 (agent prompt
cache), Q161 (store), Q20/D3 (drafter).

## Build

On any x86-64 Linux PC with Docker (CUDA 12.9 inside the container, code for compute 6.1 only, no AVX, Ubuntu 24.04
userland):

```bash
git -C ~/llama-builds/src remote add fork https://github.com/Cinderella-Man/llama.cpp     # once
TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v2-rc4
```

It writes `llama.cpp-<date>-<commit>-pascal-p106-v2-rc4.tar.gz` with `bin/` and `BUILDINFO.txt` (commit, image, AVX
count, highest glibc symbol). `SCHED_COPIES=8` builds the variant with eight scheduler copies for prefill.

On the rig itself (slow on two cores):

```bash
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build llama.cpp/build -j 2 --target llama-server
```

## Release assets

`scripts/package_release.sh` packs a build for GitHub Releases: the binaries with RPATH `$ORIGIN`, NVIDIA's
redistributable CUDA runtime libraries as a second archive, `test-backend-ops` as a third, `SHA256SUMS`, and the
release notes from [release/notes-v2.0.md](../../notes-v2.0.md) (`NOTES_VARIANT=rc4`; `DRAFT=<side file>` adds the
drafter):

```bash
BUILD_DIR=<unpacked build> NOTES_VARIANT=rc4 DRAFTS="<nine-card file>=<its export> <general file>=<its export>" scripts/package_release.sh - v2.0 2d53179aa
```

It checks that the binaries contain no AVX instructions, need glibc 2.38 at most, and find every library in the
unpacked folder with the system's CUDA libraries removed. The hashes of the published archives are in
[CHECKSUMS](../CHECKSUMS).

## Switches

Defaults are what the engine does on compute 6.1 without any variable set.

**Speed pieces** (set to 0 to turn one off):

| Variable | Default | What it switches |
|---|---|---|
| `GGML_CUDA_MMVQ_PASCAL` | on | the Pascal one-token matrix-vector kernels |
| `GGML_CUDA_MMVQ_PASCAL_SKIP` | none | weight types left to llama.cpp's kernels, e.g. `q4_K,q6_K` |
| `GGML_CUDA_MMVQ_PASCAL_MC` | on | the Pascal kernels for steps of 2–4 tokens |
| `GGML_CUDA_PERSISTENT_MOE` | 1 | one persistent kernel per MoE layer at one token (needs the rescale off, the default here) |
| `GGML_CUDA_FATTN_PASCAL` | 1 | the decode-attention kernel (head size 128, 8 query heads per KV head) |
| `GGML_CUDA_FATTN_TILE_PASCAL` | 1 | the prefill-attention kernel (head size 128 or 256), also for a Q8_0 cache |
| `GGML_CUDA_ATTN_BLOCK` | 1 | the attention block of one token in two launches (2 forces it on other GPUs) |
| `GGML_CUDA_NORM_Q8` | on | the norm writes the Q8_1 activations and absorbs the residual add |
| `GGML_CUDA_MMQ_MOE_LEGACY` | 0 | 1 restores llama.cpp's MoE prefill tiles (without PR #31047) |
| `LLAMA_GPU_KQ_MASK` | 1 | the attention mask built on each card; 0 builds it on the host |
| `LLAMA_GRAPH_REUSE_WAIT` | unset | 1 switches off the graph-reuse change for pipelined prefill micro-batches (Q109) |
| `LLAMA_INPUT_ON_GPU` | on (6.1) | the embedding row looked up on the first card; 0 on the host (needed for a drafter on a spare card) |
| `LLAMA_PIPE_UPLOAD`, `LLAMA_INPUT_UPLOAD` | packed | one packed asynchronous upload per card and token; `LLAMA_PIPE_UPLOAD=0` or `1` and `LLAMA_INPUT_UPLOAD=0` restore the separate copies |
| `LLAMA_GRAMMAR_HOST_SAMPLING` | 0 | 1 samples requests with a grammar (tool calls) on the host, as upstream does |

**Numerics:**

| Variable | Default | Meaning |
|---|---|---|
| `LLAMA_NO_MOE_DOWN_RESCALE` | unset = `tg` on compute 6.1 | `tg`: drop llama.cpp's MoE down-projection rescale for one-token steps, which the Pascal kernels sum in FP32; `0`: keep it everywhere; any other value drops it everywhere, which overflows to NaN on files with Q4/Q5 expert downs |

**Server flags:** `--pipe-depth N`, `--pipe-guess none|copy|cont|staircase`, `--pipe-copy-min N`, `--pipe-floor N`,
`--pipe-store FILE`, `--pipe-draft FILE` (with `--pipe-draft-dev`, `-ctx`, `-alpha`, `-kind`, `-stair`), `--prefix-cache
DIR` (with `--prefix-cache-min`, `-max`), `--checkpoint-gen` (with `--checkpoint-gen-at`, `-min`). Each one's help text:
`llama-server --help`; what they do: the [Flint page](../../../models/flint/README.md#command-lines) and the
[release notes](../../notes-v2.0.md).

**Diagnostics:** `GGML_CUDA_PERSISTENT_MOE_CHECK=1` (each persistent layer compared with the per-op path),
`GGML_CUDA_PROFILE=<file>` and `GGML_CUDA_PROFILE_HOST=<file>` (CUDA-event and host-side profiles that work on mining
cards, which NVIDIA's profilers refuse), `LLAMA_GPU_KQ_MASK_CHECK`, and the `_FAULT` variables that break a kernel on
purpose to show that its tests catch it.

**Not in the release defaults:** the screened output head (`LLAMA_SCREEN_HEAD=1` with `LLAMA_SCREEN_HEAD_FILE`), built
and exact, but no screen file ships with this release.

## Checks anyone can run

- `test-backend-ops` from the tests archive, on the user's own cards: the operator tests with the Pascal kernels.
- `bench/pipespec/check_identity.sh` and `check_server.sh`: pipelined decoding and the server's `--pipe-depth` give
  plain decoding's tokens ([bench/pipespec](../../../bench/pipespec/README.md)).
- `GGML_CUDA_PERSISTENT_MOE_CHECK=1 llama-perplexity … -ub 1`: every persistent MoE layer equals the per-op path.
- The drafter's checks: [drafter.md](drafter.md), step 8.
