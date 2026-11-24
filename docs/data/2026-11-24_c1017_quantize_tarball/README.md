# `llama-quantize` and the other hand-run tests in every `TESTS=1` tarball (C1017, C1012; 2026-11-24)

Result: registry [Q1750](../../../quants/experiments.md) (C1017) and [Q1745](../../../quants/experiments.md) (C1012),
from Q1742/C1009 and Q1717/C984.

The pipe step on the cards ([bench/pipespec/gpu_pipe_check.sh](../../../bench/pipespec/gpu_pipe_check.sh), C1009) makes
quantized copies of its random models, because the fork's Pascal matrix-vector kernels take quantized weights only.
The rig tarballs held no `llama-quantize`, so every check passed a separate one (`QUANT`), a static processor build from
another branch. Three hand-run tests of the fork were also left out of [C984's list](../2026-11-24_c984_tests_build_tools/README.md):
`test-bytes-switches` and `test-route-once`, run on a model file, and `test-top-k-sort`, run on a card; GPU checks
passed `test-cuda-keepalive` and `test-top-k-sort` by hand in `EXTRA_TARGETS`.

- **[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1` now builds thirteen
  check tools**: C984's eight, `test-bytes-switches`, `test-route-once`, `test-top-k-sort`, `test-cuda-keepalive` and
  `llama-quantize`, each only when the configured tree defines it (the same guard as C984's), all left out with
  `CHECK_TOOLS=0`.
- **The pipe check uses the tarball's own `llama-quantize` when `QUANT` is unset**, as its default already read
  (`QUANT` falls back to the build's `bin/llama-quantize`); it now says so in its header and prints a line when a build
  has none. A tarball built before this change still needs `QUANT`.
- **What it costs:** each new tool is a static binary with every CUDA kernel, 96 to 103 MB unpacked. Against the
  landing's tarball of the same commit, which already held `test-cuda-keepalive` and `test-top-k-sort`, the three
  others add 205,553,064 bytes (1,590,824,185 against 1,385,271,121), 68.5 MB each.

## The build

One rig build of Miner Llama's `pascal-p106-v5` at `af00bb13b` (the landing of 2026-11-24) on a build machine's
processor: CUDA 12.9.2 image, compute 6.1 code, no AVX, `-j 8`, the web page pinned to b12483 as the landing's build
had it, no `EXTRA_TARGETS`. Its clone and compiler cache were copied from the landing's build, so it took 113 s.

- **Tarball:** `llama.cpp-20261124-af00bb13b-pascal-p106-v5.tar.gz`, 1,590,824,185 bytes, md5
  `1c9e1db8900ee45c7c3b48bc9bf1e5c1`; `BUILDINFO.txt` lists all thirteen check tools and no missing one
  ([buildinfo.txt](buildinfo.txt)).
- **The landing's 21 programs are bit for bit the same:** every file of the landing's tarball, `llama-server`
  (`f5cca34e…`), `test-backend-ops` (`65fd4d6b…`), the eight check tools and the two tests it took through
  `EXTRA_TARGETS` among them, has its md5 here. New are `llama-quantize` (`98ca59ec…`), `test-bytes-switches` and
  `test-route-once` ([binaries.tsv](binaries.tsv): every file's bytes and md5 beside the landing's).
- **Warnings:** 34 lines with "warning" in the build log, as in the landing's.

## The pipe check with no `QUANT`

`gpu_pipe_check.sh` on three RTX 3090s, with `QUANT` unset and the new tarball's `bin/` as both base and head, at its
defaults: five architectures, arms `f32-stock` (one card and split over three), `q8-p106` (both) and `mix-fn-c4` (one
card), checks host, sampled, race, fed and recurrent, 256 tokens a run, model seed 1. The run passed `JOBS=8` through;
with five containers a build that is the default, all side by side. Another job used the same cards, so a build took
396 and 288 s (C1019 alone: 165 and 171 s); nothing here is timed.

| What | Result |
|---|---|
| The models and copies, from the tarball's own `test-llama-archs` and `llama-quantize` | 15 files per build: five random models, their Q8_0 copies and the `mix` copies with Q2_0, Q4_K, Q5_K, IQ4_XS and Q6_K by tensor role ([models.tsv](models.tsv)) |
| The same files from the earlier outside `llama-quantize` (C1019's run on the landing's tarball, same seed) | **15 of 15 the same md5**: the tarball's `llama-quantize` writes the copies byte for byte as the separate tool did |
| Verdicts, base and head ([verdicts.tsv](verdicts.tsv)) | **250 OK** (125 a build), 10 INFO (the kernels line), no DIFFERS, no FAIL |
| Plain decoding's tokens and logits hashes against C1019's tip run | the same in all 130 rows of a build |
| Containers ([jobs.tsv](jobs.tsv)) | all exit code 0 |

So a GPU check of a branch on `af00bb13b` needs nothing outside the tarballs: this tarball serves as its base, with the
landing's engine bit for bit and its own `llama-quantize`.

## Files

| File | What |
|---|---|
| [binaries.tsv](binaries.tsv) | every file of the new tarball's `bin/`: bytes, md5, the landing tarball's md5, whether it is a check tool |
| [models.tsv](models.tsv) | the 15 model files' md5s here and in C1019's run, and each copy's tensor types |
| [verdicts.tsv](verdicts.tsv) | the pipe check's `summary.tsv`: every verdict with its detail and plain decoding's hashes |
| [jobs.tsv](jobs.tsv) | the pipe check's containers: cards, exit code, seconds |
| [buildinfo.txt](buildinfo.txt) | the head of the tarball's `BUILDINFO.txt` |
| [run/run.sh](run/run.sh) | the driver: clone and cache, the build, the md5 comparison, the pipe check, the model comparison |

## Commands

On the build machine, from a copy of this repository at `$REPO` (with `scripts/`, `bench/pipespec/` and
`release/flash-next-v5/settings.env`):

```bash
W=<work dir> SEED=<llama.cpp clone holding af00bb13b> REPO=<repo copy> \
REF_BIN=<the landing tarball's bin/> RIG_CCACHE=<the landing build's compiler cache> \
REF_MODELS=<C1019's head models.md5> JIT_SEED=<a PTX JIT cache> LOCK=<flock file> \
PREFLIGHT=<cuda_preflight.sh> GPUS=0,1,2 JOBS=8 bash run.sh
```

The run used `gpu_pipe_check.sh` as committed (md5 `d3d5f858…`) and `build_llamacpp_docker.sh` before its header named
the new tools and their size (md5 `5d129957…`; the committed one differs in comment lines only).
The build alone is `TESTS=1 JOBS=8 HF_UI_VERSION=b12483 scripts/build_llamacpp_docker.sh fork/pascal-p106-v5`. The pipe
check alone is `WORK=<dir> BUILDS='base=<bin> head=<bin>' GPUS=0,1,2 MOUNTS=<folder holding the builds>
bash bench/pipespec/gpu_pipe_check.sh`, with no `QUANT`.

Provenance: Miner Llama (MIT, like llama.cpp), commit `af00bb13b0fd59612bcab545964b61978f04b8d6`; the models are
random weights written by `test-llama-archs`, nothing downloaded.
