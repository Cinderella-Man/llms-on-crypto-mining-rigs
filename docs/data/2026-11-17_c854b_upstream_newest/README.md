# The v5 candidate on upstream b12483: `pascal-p106-v5-cand-b12483` (2026-11-17)

**Built on a CPU, not yet run on a rig.** Step two of moving the v5 engine line onto upstream llama.cpp (registry
Q1525): `pascal-p106-v5` (`1f1110c67`, upstream b12410, [checked on a P106](../2026-11-17_v5_cand_p106_check/README.md))
with upstream's newest tag at the time, b12483, merged in, and upstream's still-open fix for recurrent and hybrid
models in the server's new slot scheduler carried as one commit. With every switch unset its GPU code is
`pascal-p106-v5`'s apart from never-run error stubs and one reordered pair of instructions in the biased MoE router.
`pascal-p106-v5` moves here once the candidate passes its checks on a P106 (the kernel suite, the release files'
replies, memory and launches a token, and one agent run each on Ornith and Flash-Next, since the server changed).
Every switch is in [docs/switches.md](../../switches.md).

- Fork branch [`pascal-p106-v5-cand-b12483`](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v5-cand-b12483)
  at **`602691153`**: the `--no-ff` merge of upstream tag b12483 (`08a9b3589`) on `pascal-p106-v5` (`1f1110c67`), commit
  `4fd6356e2`, then upstream [#32434](https://github.com/ggml-org/llama.cpp/pull/32434) as one commit. Its first parent
  line is `pascal-p106-v5`'s tip, so moving `pascal-p106-v5` here is a fast-forward.
- Rig build `llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483.tar.gz` (841,342,598 bytes), md5
  **`4e2cf20dbcb2a144be88d1eee1ec976d`**. CUDA 12.9.2 image, `sm_61`, 0 AVX instructions in `llama-bench`, highest
  GLIBC symbol `GLIBC_2.38`.

## The merge

b12483 is 73 upstream commits after b12410. Among them: the server's slot scheduler and prompt hashes
([#32154](https://github.com/ggml-org/llama.cpp/pull/32154), b12411), the warp-primitive wrappers in the CUDA backend
([#31617](https://github.com/ggml-org/llama.cpp/pull/31617), b12414), an indexed sequence removal in the KV cells
(#32380), the old batch API marked deprecated (#32306), and the web UI moved to `ui/apps/chat` and SvelteKit 3
(#32032, #32463). Eight files conflicted, in nine hunks; each resolution is in [conflicts.tsv](conflicts.tsv). Every
fork switch keeps its default (off), and v4.0's presets mean what they meant:
- **The server.** #32154 moves `server_prompt` and the prompt cache into `server-slot-sched.{h,cpp}` and the reuse of a
  slot's cells into `server_prompt::get_reuse()`, the same search as before moved into a method. The fork's
  `--ctx-checkpoints-share` moves with the structures. The fork's two sliding-window rules
  (`LLAMA_SERVER_CKPT_RING_PAST`, `LLAMA_SERVER_CKPT_RESUME_NEXT`) become two arguments of `get_reuse`, false by
  default (upstream's rule), and are passed only with the sliding-window trim, as before. The restore log and the
  cache statistics the agent check reads (`cache_src`, the restored checkpoint's tokens, restore time) keep their
  values.
- **The KV cache.** #32380's walk over a sequence's position index is what the fork's `LLAMA_KV_SEQ_RM_BOUND` already
  does (v4.0's Flint and Flash-Next lines set it). The fork's loop stays, because it also marks the attention mask's
  changed cells and keeps the cell layout rules (`LLAMA_KV_HEAD_EMPTY`, `LLAMA_KV_HEAD_TAIL`); unset, it scans every
  cell as v4.0 does.
- **CUDA.** #31617's `warp_size` and `ggml_cuda_shfl*` wrappers in the MoE router; #32392's shared-memory limit raised
  only for the launched `mul_mat_q`, with the fork's FP32-scale template argument passed through; #31327's renamed
  locals in the GDN tensor-core kernel (not run on compute 6.1) with the fork's state-row index.

40 files are changed by both sides since b12410. In 35 of them the merged file's lines that differ from upstream's are
exactly the fork's changes, and its lines that differ from `pascal-p106-v5`'s exactly upstream's (one,
`tests/test-llama-archs.cpp`, needs `git diff --minimal` for the two diffs to line up); the other five are the
conflicts above that needed code (`mmq.cuh`, `llama-kv-cache.cpp`, `server-context.cpp`, `server-task.h`,
`server-task.cpp`).

**#32434** ("server: slot_sched: fix lora and recurrent", open on 2026-11-17, one commit `92c4c070` on b12457): its
diff, fitted in three lines where the merge's code sits next to it. The server's reuse rule treats a recurrent or
hybrid model as one without a sliding window, and a cached prompt remembers its LoRA set. On the release files it
changes nothing: Flint is neither recurrent nor hybrid, Ornith and Flash-Next have no sliding window, and no LoRA is
loaded. The commit is dropped when upstream merges the pull request.

## The builds

```bash
WORK=<build folder> TESTS=1 JOBS=8 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" \
  DOCKER_ARGS="--cpu-shares=256 -e HF_UI_VERSION=b12483" \
  scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-cand-b12483
```

- **`HF_UI_VERSION=b12483`**, not b12360 as for the builds on b12410
  ([why a pin](../2026-11-17_v5_merge_switches/README.md#the-build)): from b12483 the tree's own `ui-assets.cmake`
  takes SvelteKit's `service-worker.js`, and the page served is upstream b12483's archive (`build.json`
  `{"version":"b12483"}`, sha256 `f62bb00ef959cbdccde87a52a2b5393b4adea5e34ee4ca21a6c029d4c71d4bdc`).
- A CPU build of every target (static, native, tests on, no network) passed first; its only new warnings against
  step one's CPU build are #32306's deprecations in eight of the fork's own tools and tests (registry Q1589).
- A trial build of the same commit from a local branch, before the push, gave the same files in `bin/` byte for byte.
  The rig build has 42 compiler warning lines: the 34 of `pascal-p106-v5`'s build and six of those deprecations in
  `pipespec.cpp` and `screen-check.cpp`.

| File in `bin/` | md5 |
|---|---|
| `llama-server` | `86607661d5a19d1a90d19ee4db10c90b` |
| `test-backend-ops` | `624c544cb27a89e50ccb321331c8fc72` |
| `test-llama-archs` | `373ee35accf4a4327903ad5a12c39dea` |
| `test-cuda-keepalive` | `9ddf2220fd13bfe29b8929c002688f5f` |
| `llama-bench` | `a134deaf7e51c07385ce7f70940f9fb6` |
| `llama-completion` | `cf3c298e23857d701f9c555509319f1d` |

## The machine code with every switch unset

`run/sass_check.sh` disassembles `pascal-p106-v5`'s `llama-server` (`1f1110c67`, md5 `5b134f43…`) and the candidate's
with `cuobjdump -sass` (in the CUDA image, on a CPU), hashes each kernel's SASS with C861's `sass_compare.py`
([its run folder](../2026-11-17_c861a_fp32_scale/run/)) and classifies each differing kernel with
[run/sass_classify.py](run/sass_classify.py). Full output: [sass_identity.txt](sass_identity.txt).

- **7,924 of 8,835 kernels are byte for byte the same; none is missing and none is new.**
- **899 of the other 911 are error stubs** (at most 30 instructions: the `NO_DEVICE_CODE` branches compute 6.1 never
  runs): 268 of the tensor-core attention kernel and 3 of the GDN tensor-core kernel print another source line
  (#31617 and #31327 added lines above it), and 628 `mul_mat_q` stubs read their message strings' addresses from other
  constant slots.
- **12 are real kernels: the MoE router with a selection bias** (`topk_moe_cuda<N, true>`, N from 1 to 576). Each
  differs in two lines: two `MOV Rx, RZ` (zeroing two registers) in the other order. The routers without a bias are
  the same bytes. Ornith's and Flash-Next's graphs pass no selection bias; Flint's graph passes one when its file
  holds the tensor.
- **C861's 176 FP32-scale kernels** (only with `GGML_CUDA_MMQ_KQ_FP32SCALE` set): every one over 30 instructions is the
  same bytes (56 kernels, up to 2,058 instructions); the 120 that differ are 30-instruction stubs.
- The comparison keeps the base build's FP32-scale kernels apart: `sass_compare.py compare` would count them as extra
  copies of their switched-off twins, which flagged 56 identical K-quant kernels in a first pass.

## Files

| File | What |
|---|---|
| [conflicts.tsv](conflicts.tsv) | each conflicted file: hunks, upstream's change, the fork's, the resolution |
| [sass_identity.txt](sass_identity.txt) | the per-kernel comparison with `pascal-p106-v5`'s build, by family, kind and size |
| [run/sass_check.sh](run/sass_check.sh) | the comparison: `OUT=<folder> MNT=<folder holding the binaries> BASE=<1f1110c67's llama-server> NEW=<this llama-server> bash run/sass_check.sh`, with `sass_compare.py` and `sass_classify.py` copied into `OUT` |
| [run/sass_classify.py](run/sass_classify.py) | one row per differing kernel: family, instructions in each build, the kind of difference (constants only, the same opcodes reordered, or other code), the opcodes whose counts change |

Everything ran on a build machine's CPU in Docker (`--cpu-shares=256`, `-j 8`, one build at a time); nothing ran on a
GPU. Licence: the scripts are this repository's.
