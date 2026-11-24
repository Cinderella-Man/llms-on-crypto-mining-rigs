# Fourteen branches landed in pascal-p106-v5 (2026-11-24)

Miner Llama's v5 engine collects in one fork branch, `pascal-p106-v5`, and every finished change lands there behind its
own switch, off by default. On 2026-11-24 fourteen branches built and checked over the three nights before were merged
into it at once: ten that had passed their checks on a build machine's processor and on three RTX 3090s, and four from
the night before. One of them fixes a fault in pipelined Flash-Next decoding that `pascal-p106-v5` carried until then.

This folder records the merge and the checks of the combined build against the branch before it. Correctness only:
nothing here is timed. Each change's speed is measured on a P106 rig, on this combined build, with its switch set
against the same build with it unset (the rows are in each branch's registry entry).

**Result: all fourteen landed.** Every merge was clean. With no switch set, the combined build computes what the
base computes everywhere the checks look, except where a fix changes a result on purpose:

- pipelined Flash-Next-type decoding now gives plain decoding's tokens, logits and state on the processor
  (C932's fix; the base differs in every pipelined run);
- `test-kv-cache` passes (C881's change; it aborted on the base), and three new tests pass;
- the server prints a `pipe reallocs` line after a request with pipe slots (C901).

The release files Flint Pascal Q8 GGUF (`p065`) and Ornith-1.5-122B Pascal N2P GGUF (`valid1`) give the same greedy
reply token for token and use the same memory on every card.

`pascal-p106-v5` was then fast-forwarded to the landing merge `af00bb13b`.

## What was merged

`pascal-p106-v5` at `8a9629942`, plus one `--no-ff` merge per branch, in this order (the fix first), on the branch
`pascal-p106-v5-land-batch2`: [merges.tsv](merges.tsv) (branch, its head, the merge commit, code row, registry row, the
switch it brings and its default). No merge had a conflict, and no two branches change the same function (the shared
files are `src/llama-context.cpp`, `include/llama.h`, `tools/server/server-context.cpp`, `src/llama-model.cpp`,
`src/models/qwen4exp.cpp`, `ggml/src/ggml-cuda/ggml-cuda.cu`, `tests/test-llama-archs.cpp` and
`tests/CMakeLists.txt`, each touched in other functions).

The switches, each off by default, have their rows in [docs/switches.md](../../switches.md):
`LLAMA_SCHED_PER_SIZE_PP` (read only under `LLAMA_SCHED_PER_SIZE=1`, itself off by default), `LLAMA_PIPE_SLOT_PLAN`,
`GGML_CUDA_MMVQ_PASCAL_MT=2`, `LLAMA_SERVER_CKPT_NOWAIT`, the taper key `cuts=1`, `GGML_CUDA_KEEPALIVE_TRACE`,
`GGML_CUDA_TOP_K_SORT` and `GGML_CUDA_PROFILE_HOST_NOWAIT`. Six changes have no switch; they are listed under
"What has no switch" there.

## The checks

Run on a build machine, 09:53–10:46: one detached driver ([run/run.sh](run/run.sh)), the processor steps beside the card steps; the card pipe step ran a second time alone, 10:34–10:38, after the driver's first try passed a wrong seed to it (fixed in the driver). Every check in [checks.tsv](checks.tsv):

| Check (head `af00bb13b` against base `8a9629942`) | Result |
|---|---|
| Rig build (`TESTS=1`, the eight check tools, `test-cuda-keepalive`, `test-top-k-sort`, web page b12483), 09:54–10:00 | `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz`, 1,385,271,121 bytes, **md5 `b51ad2740d5822e7e14af656c413ec22`** |
| Warnings | rig build 32 lines on both, CPU build 2 on both; none new |
| Logits, `test-llama-archs -b CPU -s 1`, the rig builds | 148 of 148 files the base's bit for bit; 153 of 153 cases pass on both |
| CPU trees (`cpu_tests.sh`, `-DGGML_RPC=ON`): ident | 148 of 148 files the base's |
| ctest, test by test | base 75 of 77 pass, head 79 of 80; `test-kv-cache` aborted on the base and passes on the head (C881's change), `test-cuda-mmvq-pascal-pick`, `test-quantize-default-rules` and `test-rpc-graph-cache` are new and pass (C899, C983, C929); `test-kv-cache-scales` fails on both; every other test the same |
| Hand-run tests | 9 of 9 pass on both; logs line by line 23 of 24 the same, `context-model` lists two more cases (C907's new checkpoint cases, skipped on that model: "not a recurrent model") |
| CPU pipe step (C935, C1010): 5 architectures × identity, sampled, race, recurrent | head 20 of 20 OK; base 16 of 20, the four `qwen4exp` arms DIFFERS (C932's fault, which the head fixes) |
| Card pipe step (C1009), one card and split over three, F32 on llama.cpp's kernels and Q8_0 on the fork's forced | head 60 of 60 OK; base 48 of 60, `qwen4exp` host, fed and recurrent DIFFERS in all four arm-layout pairs (C932's fault); plain decoding's tokens and logits hashes the same on both builds in all 20 pairs |
| `test-backend-ops MUL_MAT`, the fork's Pascal kernels forced with a P106's launch shapes | 5,125 of 5,125; the same with `GGML_CUDA_MMVQ_PASCAL_MT=2` (C968: 5,125 of 5,125 on the base's kernels) |
| `test-backend-ops MUL_MAT_ID`, forced | 2,370 of 2,370 (C968: 2,370 of 2,370) |
| `test-backend-ops FLASH_ATTN_EXT` unset, with `GGML_CUDA_FATTN_TILE_PASCAL=2`, and with every Pascal attention switch | 4,679 of 4,679 in each (the pull request 9 review: 4,679 of 4,679) |
| Flint Pascal Q8 GGUF, v5 preset on three cards (`-ts 1,1,1`), the fork's kernels forced, one greedy reply of 256 tokens | the same 256 token ids; MiB per card after the load 15,953 / 17,585 / 13,851 and after the request 16,007 / 17,645 / 13,919 on both builds |
| Ornith-1.5-122B Pascal N2P GGUF, the same | the same 256 token ids; 15,425 / 16,641 / 15,605 and 15,503 / 16,719 / 15,669 MiB on both builds |

No CUDA error, illegal access or timeout anywhere. The card suite's base numbers are those of C968 and the pull request 9
review on `14d679604`, whose engine code is `8a9629942`'s (only the fork's own tools and tests changed between them).

## Files

| File | What it holds |
|---|---|
| [merges.tsv](merges.tsv) | the fourteen merges |
| [checks.tsv](checks.tsv) | every check, head against base, with its verdict |
| [ctest.tsv](ctest.tsv) | ctest per test on both CPU builds |
| [pipe.tsv](pipe.tsv) | the CPU pipe step's verdicts per architecture and arm, both trees |
| [gpu_pipe.tsv](gpu_pipe.tsv) | the card pipe step's verdicts per build, arm, layout, architecture and check |
| [replies.tsv](replies.tsv) | the two release files' greedy replies and memory per card, head and base |
| [bops.tsv](bops.tsv) | each `test-backend-ops` arm: settings, cases passed, failures |
| [binaries.tsv](binaries.tsv) | the rig tarballs and their programs' md5s |
| [run/run.sh](run/run.sh) | the driver: build, CPU steps and GPU steps |

## Commands

On a build machine with Docker, three NVIDIA cards and this repository's scripts copied into one folder
(`build_llamacpp_docker.sh`, `docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh` and `compare_hand.py`,
`bench/pipespec/` as `pipespec/`, `release/flint-v5/settings.env` and `release/ornith-v5/settings.env` under
`release/`):

```bash
W=<work dir> SEED=<llama.cpp clone holding 8a9629942> REPO=<that folder> \
BASE_TGZ=<the base's TESTS=1 tarball> BASE_MD5=59ca8336b87367a6522e63c6fcc1381b BASE_RIG_LOG=<its build log> \
FLINT_Q8=<flint-pascal-Q8.gguf> ORNITH=<Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf> QUANT=<a llama-quantize> \
PREFLIGHT=<a CUDA check, run as $PREFLIGHT -g 0,1,2> MOUNTS='<folders holding the builds and models>' GPUS=0,1,2 \
LOCK=<build lock> JOBS=8 setsid nohup bash run.sh > run.out 2>&1 < /dev/null &
# the card pipe step again, alone (STEPS=none runs only GPU_STEPS)
env <the same> STEPS=none GPU_STEPS=gpupipe setsid nohup bash run.sh > run-gpupipe.out 2>&1 < /dev/null &
```

The base tarball is `scripts/build_llamacpp_docker.sh` of `8a9629942` with `TESTS=1` and `HF_UI_VERSION=b12483`
(1,252,901,378 bytes, md5 `59ca8336b87367a6522e63c6fcc1381b`). The head's is the same command on
`fork/pascal-p106-v5-land-batch2` with `EXTRA_TARGETS="test-cuda-keepalive test-top-k-sort"`. Fork:
<https://github.com/Cinderella-Man/miner-llama>, branches `pascal-p106-v5` and `pascal-p106-v5-land-batch2`.

## Provenance and licence

The random models are written by the build's own `test-llama-archs` with fixed seeds. The release files are this
project's own (their cards on Hugging Face give the licences). The tables are this project's, under the repository's
licence.
