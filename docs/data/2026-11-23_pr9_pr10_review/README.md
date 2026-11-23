# Pull requests 9 and 10 built and checked: Q8_0 attention switches and the pipe with an image projector (2026-11-23)

Two pull requests from a contributor on Miner Llama's `pascal-p106-v5`, merged into a review branch, built for the
P106 rigs and checked against the base on a build machine:
- the CPU test set;
- the random models' logits;
- a server check of the new switch;
- the compiled card code;
- `test-backend-ops`' attention suite on one RTX 3090 through the build's compute 6.1 PTX, with each new switch unset
  and set.

Result pages: registry rows Q1702, Q1703, Q1704 (pull request 9) and Q1705 (pull request 10) in
[quants/experiments.md](../../../quants/experiments.md).

**Result:** with every switch unset both pull requests leave the engine as it was.
- The base's ctest result test by test, every random model's logits bit for bit, and the server's replies and pipeline
  counts with `LLAMA_PIPE_MEDIA` at every value.
- Every one of the base's six Pascal tile kernels compiles to the base's exact code.
- On one RTX 3090 through the compute 6.1 PTX, `test-backend-ops`' attention suite passes 4,679 of 4,679 with nothing
  set and with each new switch and all of them set. The output fault makes 227 cases fail with all switches set: every
  case the base's Pascal kernel takes, the new cases, and the 1–7-row cases `_MIN=1` routes to the Pascal kernels.
- Both were merged into `pascal-p106-v5` the same day (10 as `7b2b683af`, then 9 as `747d81040`, the review branch's
  tree). One fix followed on top: the decode kernel now reads its mask as 8-byte vectors
  ([below](#the-mask-fix-14d679604)). `pascal-p106-v5` is `14d679604`.

## What was built

| | Commit | What it is |
|---|---|---|
| base | `3dfad876f` | `pascal-p106-v5` as checked in [C908](../2026-11-21_c908_v5_merged_build/README.md) (the branch is `c998c6cb4` since pull request [#8](https://github.com/Cinderella-Man/miner-llama/pull/8), a Windows-only test build fix, was merged) |
| pull request [#9](https://github.com/Cinderella-Man/miner-llama/pull/9) | `a01fe66d1` | three switches, off by default, in `ggml/src/ggml-cuda/fattn-tile-pascal.cu` for a Q8_0 K/V cache on compute 6.1: `GGML_CUDA_FATTN_TILE_PASCAL_LDS=1` (Q8_0 loads decoded at store in the tile kernel), `GGML_CUDA_FATTN_DEC_PASCAL=N` (a decode kernel for head 256 and up to N rows; 1 and 2 rows need `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1`), `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` (Q8_0 K scored with `dp4a` in the tile kernel at head 256); a routing check in `fattn.cu`; 20 `FLASH_ATTN_EXT` test cases at head 256 |
| pull request [#10](https://github.com/Cinderella-Man/miner-llama/pull/10) | `debd2adbe` | `LLAMA_PIPE_MEDIA`, off by default: `1` lets text requests use the pipe and in-call checkpoints when an image projector (`--mmproj`) is loaded, `2` also runs the pipe with images in the cache |
| head | **`82d9eb5ef`** | `c998c6cb4` with both merged (`--no-ff`), branch `pascal-p106-v5-contrib2` |
| merged | `747d81040` | `pascal-p106-v5` after the pull requests' own merges (10 as `7b2b683af`, then 9): the same tree as the head |
| fix | **`14d679604`** | the decode kernel's mask read as `uint2`, branch `pascal-p106-v5-dec-mask`; `pascal-p106-v5` was fast-forwarded to it |

## Builds

| Build | Result |
|---|---|
| Rig tarball: [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) `fork/pascal-p106-v5-contrib2` with `TESTS=1`, `EXTRA_TARGETS="test-llama-archs test-cuda-keepalive"`, `HF_UI_VERSION=b12483`, `JOBS=8` | rc 0; `llama.cpp-20261123-82d9eb5ef-pascal-p106-v5-contrib2.tar.gz`, 842,620,911 bytes, md5 `361c7ce15fd232a1c5f859ac341cc0f9`; sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no AVX instructions, highest glibc symbol 2.38 |
| Native CPU build of every target, base and head ([cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), shared libraries) | both rc 0, 0 errors |

## Checks

| Check | Base `3dfad876f` | Head `82d9eb5ef` |
|---|---|---|
| Compiler warnings, rig build ([warnings.tsv](warnings.tsv)) | 42 lines, 38 distinct (the base's own tarball log) | **the same 38**, none new |
| Compiler warnings, native CPU build | 26 distinct | **the same 26**, none new |
| ctest, 75 tests without network ([ctest.tsv](ctest.tsv)) | 71 pass; 4 fail: `test-kv-cache` (upstream's test against the fork's own cell removal), `test-jinja` and `test-jinja-py` (their source paths under the compiler cache's base folder), `test-kv-cache-scales` (no torch in the image) | **the same, test by test** (also the same as C908's run of this base) |
| `test-llama-archs -b CPU -s 1`, native: plain, write, compare ([cpu_summary.tsv](cpu_summary.tsv)) | 153 of 153 pass each time; 148 logit files | the same |
| The head's logits against the base's, native CPU | — | **148 compared, none differ** |
| The head's rig build against the base's rig build (`test-llama-archs -b CPU`, the CUDA backend loaded against the driver stub) | 153 of 153, 148 files written | **153 of 153; 148 compared, none differ** |
| Hand-run tests: `test-hop-cast`, `test-route-once` (×2), `test-bytes-switches` (×2), `test-moe-ids`, `test-thread-safety`, `test-llama-context`, `llama-eval-callback` | all pass | all pass |
| Server: `LLAMA_PIPE_MEDIA` without a projector ([server.tsv](server.tsv)) | — | **unset, `0`, `1`, `2`: 5 of 5 replies the base's tokens, the base's pipeline counts** |
| Card code, `cuobjdump -sass` of `llama-server` ([sass_summary.txt](sass_summary.txt), [sass_differing.tsv](sass_differing.tsv)) | 8,836 kernel names | **8,833 identical**, 3 with one more copy of the same code, 5 new, none missing |
| The Pascal attention kernels ([pascal_kernels.tsv](pascal_kernels.tsv)) | 6 | **the 6 the base's code**, registers and stack; 5 new |
| `test-backend-ops -o FLASH_ATTN_EXT` on an RTX 3090 through the compute 6.1 PTX ([fa.tsv](fa.tsv)) | 4,659 of 4,659; Pascal kernels forced 4,659 of 4,659; fault: 165 fail | **4,679 of 4,679** with nothing set, forced, and each switch and all set; fault: 173, all switches 227 |

### The server check

A random llama model with a real tokenizer (`test-llama-archs -o`), no projector. Each run sends five requests
([run/client.py](run/client.py)): a greedy one, a sampled one (seed 7), greedy and sampled JSON-schema requests, and
the greedy one again. [run/compare.py](run/compare.py) compares every run's token ids with the base's run of the same
mode and its pipeline counts with the base's pipelined run.

| Run | `LLAMA_PIPE_MEDIA` | Pipelined requests | Guesses | Right | Wrong | Tokens | Pipeline counts |
|---|---|---:|---:|---:|---:|---|---|
| base, `-bs` | unset | 0 | 0 | 0 | 0 | — | — |
| head, `-bs` | unset | 0 | 0 | 0 | 0 | 5 of 5 the base's | — |
| base, `--pipe-depth 3` | unset | 5 | 370 | 37 | 113 | — | — |
| head, `--pipe-depth 3` | unset | 5 | 370 | 37 | 113 | 5 of 5 the base's | the base's |
| head, `--pipe-depth 3` | `0` | 5 | 370 | 37 | 113 | 5 of 5 the base's | the base's |
| head, `--pipe-depth 3` | `1` | 5 | 370 | 37 | 113 | 5 of 5 the base's | the base's |
| head, `--pipe-depth 3` | `2` | 5 | 370 | 37 | 113 | 5 of 5 the base's | the base's |

Without a projector, `LLAMA_PIPE_MEDIA` cannot change anything. With no image in the cache, the two lines it changes
outside the switch compute the same numbers as before, because `server_tokens::pos_next()` then equals the token count.

### The card code

The head's tile kernel template has two more flags (`lds`, `kdp`). Before the comparison, its name is cut back to the
base's form where both are false, so each default kernel is compared with the base's kernel of the same shape.

| Kernel | Selected by | The base's SASS | Instructions | `dp4a` | Registers | Stack bytes |
|---|---|---|---:|---:|---:|---:|
| `tile<256, F16, F16, plain>` | default | **yes** | 2,070 | 0 | 155 | 0 |
| `tile<256, Q8_0, Q8_0, plain>` | default | **yes** | 2,256 | 0 | 158 | 0 |
| `tile<128, F16, F16, plain>` | default with `_QK=1` | **yes** | 2,328 | 0 | 159 | 0 |
| `tile<128, Q8_0, Q8_0, plain>` | default with `_QK=1` | **yes** | 2,508 | 0 | 160 | 0 |
| `tile<128, F16, F16, split>` | default | **yes** | 2,406 | 0 | 168 | 0 |
| `tile<128, Q8_0, Q8_0, split>` | default | **yes** | 2,604 | 0 | 168 | 40 |
| `tile<256, Q8_0, Q8_0, plain, lds>` | `_LDS=1` | new | 2,256 | 0 | 159 | 0 |
| `tile<128, Q8_0, Q8_0, plain, lds>` | `_LDS=1` with `_QK=1` | new | 2,502 | 0 | 164 | 0 |
| `tile<128, Q8_0, Q8_0, split, lds>` | `_LDS=1` | new | 2,616 | 0 | 168 | 56 |
| `tile<256, Q8_0, Q8_0, plain, lds, kdp>` | `_KDP=1` | new | 1,908 | 64 | 157 | 0 |
| `dec_pascal<256>` | `_DEC_PASCAL=N` | new | 2,886 | 64 | 168 | 0 (12,672 B shared) |

The three helper kernels that differ ([sass_differing.tsv](sass_differing.tsv)) have the same code in every copy. There
is one more copy of each because the decode kernel's launch instantiates `launch_fattn<256, 1, 8>` once more, in
`fattn-tile-pascal.cu`.

**The decode kernel's mask reads.** In the compute 6.1 PTX, the kernel's two reads of the mask are 16-byte vector
loads (`ld.global.v4.u32`) at addresses that are 8-byte aligned for every other batch of 4 keys. The sm_61 SASS has
them as 8-byte loads (`LDG.E.64`): the compiler dropped the unused upper half. The fix (`14d679604`,
[below](#the-mask-fix-14d679604)) reads the mask as `uint2`, 4 halves for the batch's 4 keys, so the PTX itself asks for
8 aligned bytes inside the slice.

### The attention suite on an RTX 3090

`test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT`, one arm after another on one RTX 3090 (compute 8.6), from 14:28:34
to 15:04:57. The card runs the sm_61 build's compute 6.1 PTX, compiled by the driver. Each arm takes about 3 minutes.
On a card that is not compute 6.1, the Pascal kernels take their cases only with `GGML_CUDA_FATTN_TILE_PASCAL=2`
(`=2` below), as in Q163's checks; on a P106 they do so by default. In every arm 1,405 cases are not supported on
CUDA and skipped, on both builds alike.

| Arm | Build | Switches | Cases | Passed | Failed | Pascal paths that logged |
|---|---|---|---:|---:|---:|---|
| `base-plain` | base | none | 4,659 | 4,659 | 0 | none |
| `head-plain` | head | none | 4,679 | 4,679 | 0 | none |
| `base-forced` | base | `=2` | 4,659 | 4,659 | 0 | the tile kernel: head 128 F16 and Q8_0 (split), head 256 F16 and Q8_0 |
| `head-forced` | head | `=2` | 4,679 | 4,679 | 0 | the same four |
| `head-lds` | head | `=2 _LDS=1` | 4,679 | 4,679 | 0 | the same, with both Q8_0 lines "decoded at store" |
| `head-kdp` | head | `=2 _KDP=1` | 4,679 | 4,679 | 0 | the four, and "q8_0 K scored in int8 with dp4a" |
| `head-dec` | head | `=2 _MIN=1 _DEC_PASCAL=4` | 4,679 | 4,679 | 0 | the four, and the decode kernel "head 256, q8_0 K/V, up to 4 rows" |
| `head-all` | head | `=2 _LDS=1 _KDP=1 _MIN=1 _DEC_PASCAL=4` | 4,679 | 4,679 | 0 | every path: decoded at store, `dp4a` scores, the decode kernel |
| `base-forced-fault` | base | `=2 _FAULT=1` | 4,659 | 4,494 | **165** | the four |
| `head-forced-fault` | head | `=2 _FAULT=1` | 4,679 | 4,506 | **173** | the four |
| `head-all-fault` | head | `head-all`'s and `_FAULT=1` | 4,679 | 4,452 | **227** | every path |

- **Every arm passes.** With nothing set, the head passes as the base does. The 20 new cases are the only difference
  in the count (4,679 against 4,659).
- **The fault arms show where the cases go.** `_FAULT=1` scales the Pascal kernels' output by 1.05, decode kernel
  included, so every case they take fails:
  - **Base, 165.** Those are the cases the stock Pascal kernel takes.
  - **Head, 173.** Every one of the base's 165, plus 8 of the new cases: Q8_0 at 3, 4 and 128 rows and F16 at 128
    rows, with 512 and 4,096 keys.
  - **All switches, 227.** The same 173 plus 54 cases of 1 to 7 rows, which `_MIN=1` sends to the Pascal kernels:
    - the decode kernel: Q8_0 at head 256 with 1 or 2 rows;
    - the tile kernel: F16 at head 128 and 256 with 1 to 7 rows, and Q8_0 at head 128 with 1 row.

    The contributor's own fault count on a GTX 1070 is also 227.
- **The decode kernel ran through the PTX without a fault**, although its two mask reads are 16-byte vector loads at
  8-byte-aligned addresses there (the paragraph above on the mask reads). The driver's compiler drops the unused half
  as ptxas does for sm_61.

### The mask fix (`14d679604`)

`flash_attn_dec_pascal` read each batch's mask as a `uint4` (8 halves), although a batch is 4 keys. The fix reads it
as one `uint2` and builds the two mask words from `.x` and `.y`. A `static_assert` ties the read to the batch's key
count. It was built and checked with the same driver against the review branch's build (`82d9eb5ef`, the tree of
`747d81040`): steps `src rig sass fa`, 15:13–15:35.

| Check | Result |
|---|---|
| Rig build ([fix_sass_summary.txt](fix_sass_summary.txt)) | rc 0; `llama.cpp-20261123-14d679604-pascal-p106-v5-dec-mask.tar.gz`, 842,625,429 bytes, md5 `d8dd415dea36f9d064ce67e65840c218`; the same 38 distinct warning lines, none new |
| Card code against `82d9eb5ef` | **8,840 of 8,841 kernels identical**, none new, none missing. The decode kernel has the same 2,886 instructions, registers (168), stack (0) and shared memory (12,672 bytes). Its 8 differing lines are all zeroing moves (`MOV Rx, RZ`) in another order |
| The decode kernel's mask reads | PTX: 2 × `ld.global.v2.u32` (were `ld.global.v4.u32`). sm_61 SASS: 2 × `LDG.E.64` at the same two places: 8 bytes each, the batch's 4 keys, inside the warp's 32-key slice |
| `FLASH_ATTN_EXT` on an RTX 3090 through the compute 6.1 PTX ([fix_fa.tsv](fix_fa.tsv)) | nothing set: 4,679 of 4,679; `=2 _MIN=1 _DEC_PASCAL=4`: 4,679 of 4,679, the decode kernel logged; all switches: 4,679 of 4,679; all switches with `_FAULT=1`: 227 fail, as before the fix |

## Files

| File | What |
|---|---|
| [ctest.tsv](ctest.tsv) | every ctest test's result, base and head |
| [warnings.tsv](warnings.tsv) | every distinct compiler warning of the rig and the native builds, base and head (line numbers removed) |
| [cpu_summary.tsv](cpu_summary.tsv) | `cpu_tests.sh`'s verdict per tree and check |
| [server.tsv](server.tsv) | the server check, one row per run |
| [sass_summary.txt](sass_summary.txt) | the card-code comparison's counts and the classes of the differing kernels |
| [sass_differing.tsv](sass_differing.tsv) | the three kernels with one more copy, their SHA-1 and why |
| [pascal_kernels.tsv](pascal_kernels.tsv) | the Pascal attention kernels: SHA-1, instruction counts, registers, stack and shared memory, base and head |
| [fa.tsv](fa.tsv) | the attention suite's arms: build, switches, exit code, passed, failed, unsupported, which Pascal kernels logged |
| [fix_sass_summary.txt](fix_sass_summary.txt) | the mask fix against the review branch: card-code counts, the decode kernel's loads in the SASS and the PTX, its resources |
| [fix_fa.tsv](fix_fa.tsv) | the mask fix's attention-suite arms |
| [run/run.sh](run/run.sh) | the driver: steps `src`, `rig`, `cpu`, `archs`, `server`, `sass`, `fa` |
| [run/client.py](run/client.py), [run/compare.py](run/compare.py) | the server check's client and table |

The other scripts it calls are [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh),
[cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh),
[sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py) and
[sass_classify.py](../2026-11-17_c854b_upstream_newest/run/sass_classify.py).

## Commands

The review branch, made with git alone:

```bash
git worktree add -b pascal-p106-v5-contrib2 <dir> c998c6cb4
git -C <dir> merge --no-ff a01fe66d1      # pull request 9
git -C <dir> merge --no-ff debd2adbe      # pull request 10, head 82d9eb5ef
git -C <dir> push fork pascal-p106-v5-contrib2
```

The checks, on the build machine. `REPO` holds the scripts listed under Files, and the base's folders are C908's:

```bash
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> \
BASE_RIG=<C908's rig build folder of 3dfad876f> BASE_RIG_LOG=<its build log> BASE_CTEST=<C908's ctest table of 3dfad876f> \
VOCAB=<a gguf with a tokenizer> REPO=<scripts> RIG_CCACHE=<C908's rig ccache> CPU_CCACHE=<C908's CPU ccache> \
SRV_IMAGE=<the CUDA image with python3 and curl> LOCK=<build lock> JOBS=8 GPU=2 PREFLIGHT=<a CUDA check taking -g> \
STEPS="src rig cpu archs server sass fa" bash run.sh
```

The `fa` step's arms. `GGML_CUDA_FATTN_TILE_PASCAL=2` lets the Pascal kernels take their cases on a card that is not
compute 6.1; on a P106 they take them by default. `_FAULT=1` multiplies their output by 1.05, so every case they take
fails: a check that the cases reach them.

```bash
P=GGML_CUDA_FATTN_TILE_PASCAL
test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT                                   # base-plain, head-plain
env $P=2 test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT                          # base-forced, head-forced
env $P=2 ${P}_LDS=1 test-backend-ops ...                                           # head-lds
env $P=2 ${P}_KDP=1 test-backend-ops ...                                           # head-kdp
env $P=2 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4 test-backend-ops ...              # head-dec
env $P=2 ${P}_LDS=1 ${P}_KDP=1 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4 test-backend-ops ...   # head-all
env $P=2 ${P}_FAULT=1 test-backend-ops ...                                         # base-forced-fault, head-forced-fault
env $P=2 ${P}_LDS=1 ${P}_KDP=1 ${P}_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4 ${P}_FAULT=1 test-backend-ops ...   # head-all-fault
```

Every container ran with `--cpu-shares=256` (CPU ones also with `--network none --memory=16g`). The GPU arms ran one
at a time on one card, with the PTX JIT's cache shared between them. No speed was measured.

The mask fix, the same driver against the review branch's build (the name cut applies to both builds, so a base that
already has the kernel's new flags compares too):

```bash
W=<work dir> SEED=<the review run's clone> REF=fork/pascal-p106-v5-dec-mask \
HEAD_SHA=14d67960410bd3d6ff1f9ac819ce69c9c6f9396f BASE_SHA=82d9eb5ef243c6166c5095ead1d7d8e43eb8a9cf \
BASE_RIG=<the review branch's rig build folder> BASE_RIG_LOG=<its build log> VOCAB=<a gguf with a tokenizer> \
REPO=<scripts> RIG_CCACHE=<the review run's rig ccache> LOCK=<build lock> JOBS=8 GPU=2 PREFLIGHT=<a CUDA check> \
STEPS="src rig sass fa" FA_ARMS="head-plain head-dec head-all head-all-fault" bash run.sh
```

## Provenance and licence

The code is Miner Llama's, a llama.cpp fork (MIT), with both pull requests' code under the same licence. The random
models are generated by `test-llama-archs`; the tokenizer vocabulary is from a public GGUF model. The tables were made
by the scripts above from the runs' logs; the raw logs are not published.
