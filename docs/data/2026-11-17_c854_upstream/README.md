# C854 step one (Q1525): Miner Llama's v5 line on upstream llama.cpp b12410 (2026-11-17)

v4.0 runs on upstream llama.cpp b12360 ([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md)). The v5
line (`pascal-p106-v5`, cut at v4.0's commit `eb82916d3`) takes upstream in two steps, as the news sweep of
2026-11-17 advised ([news](../../research/notes/news_2026-11-17.md)):
- **step one, here:** tag b12410 (`58a4a6b49`, 2026-11-15), 50 upstream commits after b12360. It holds the DFlash
  speculation split (#32016, b12377) that the published DFlash2 drafters are named by;
- **step two, later:** the server's rewrite (#32154, b12411) once its fix for recurrent models (#32434) is in a tag,
  with the warp-primitive change (#31617, b12414).

One merge commit on a new branch, `pascal-p106-v5-b12410-up`; every fork switch keeps its default (off) and v4.0's
presets mean the same. `pascal-p106-v5` itself takes the branch after the checks on a P106 (next row). The registry
row is Q1525 in [quants/experiments.md](../../../quants/experiments.md).

**Result:** the merged branch builds: on a CPU every target (tests and tools included), and the rig tarball for
CUDA 12.9 and sm_61 with no AVX instruction. Two upstream API changes outside the conflicts needed fixes. Whether it
computes what v4.0 computes on a P106 is the next row: the kernel suite, the release files' greedy replies and the
launches a token, all against v4.0.

## The branch

`pascal-p106-v5-b12410-up` at **`b3dfb0d89`**, one `--no-ff` merge of upstream tag b12410 on `pascal-p106-v5` `eb82916d3`.
Against `eb82916d3`: 177 files change (+42,411 −9,538 lines). 8 files conflicted, in 15 hunks; the merge commit also
holds the two fixes that upstream's API changes needed in files without a conflict, so the branch builds at its commit.

## The conflicts

Every hunk, both sides and the resolution: [conflicts.tsv](conflicts.tsv). The ones that needed thought:
- **The DFlash driver (#32016).** Upstream split one `draft()` into a class per head: DFlash, DSpark, DFlash2 and
  LiliCorr. The fork's hooks lived inside that one function, so each moved into the class of its head:
  - DFlash and DSpark: the draft records (`LLAMA_SPEC_DUMP`) and the drafted tokens' probabilities (`result_p`);
  - DFlash2: the selector's walk, its lattice record, `result_p` and the dump's full second block;
  - LiliCorr: the scorer's softmax at the taken candidate into `result_p`, computed only when `p_min` or `result_p`
    asks for it, as before.

  A DFlash2 or LiliCorr draft's record now names its own type (`draft-dflash2`, `draft-lilicorr`) where v4.0 wrote
  `draft-dflash`.
- **The MoE cache (#32157)** moved its lookup out of the graph code into `llama_moe_cache::build_lookup` and
  `build_experts`. A layer with a cold tier still skips the cache. The fork's tiered expert build calls
  `build_experts` for the cache's merge of cached and CPU-computed experts: the same operations in the same order.
- **Attention masks (#32217, Inkling 2).** `build_attn` takes an optional mask from the caller. A caller's mask is
  used as it is; the fork's device-built mask (`LLAMA_GPU_KQ_MASK`) replaces only the default one.
- **`test-backend-ops` (#30279)** now builds its case list in one function and exits 1 when the filters select no case,
  when `-b` names no device, or when no case runs on the named backend. The fork's record and replay name each case
  before any filter, as before. The new "nothing ran" failure applies outside record and replay: a resumed record skips
  the cases it already holds, and a replay with `--replay-allow-missing` skips those it has no record of.

## Outside the conflicts

- **#32217 widens `nb32` to 64 bits** in the flash-attention kernel signature (`fattn_kernel_t`), for masks with a head
  dimension. The fork's Pascal tile kernel (`fattn-tile-pascal.cu`) takes the new signature. The fork's three Pascal
  attention kernels refuse a mask with a head dimension and fall back, so per-head masks never reach them.
- **#29557 (grammar left factoring)** keeps a grammar's rules in a `shared_ptr`. The fork's walk of the grammar's
  forced tokens (Q797, Q802) reads them through it.

## Upstream changes for the checks on a P106 to watch

| Change | What it does | Where a P106 run could show it |
|---|---|---|
| #31775 | the scheduler starts a new split at an incompatible weight even in a split without inputs | split and copy counts with weights on the host (`-ot`, the MoE cache) |
| #32135 | RMS norm in BF16; the fused RMS norm + multiply (+ add) checks the multiply's operand instead of asserting F32 | F32 models fuse as before; launches a token show it if not |
| #28982 | the cooperative softmax keeps its columns in registers | softmax cases of the kernel suite |
| #32217 | per-head masks and the `nb32` width in every attention kernel; every `FLASH_ATTN_EXT` test case's name gains `,mask_heads=<n>` | attention cases; a record set made before the merge misses every attention case, so they are recorded again with this build (or aliased, as `alias_base.py` did for b12360's `MUL_MAT` rename) |
| #30279 | `test-backend-ops` exits 1 on an empty selection or nothing run | check scripts that pass `-b` with a filter the card does not support |
| #32016 | `draft-dflash2` and `draft-lilicorr`; `draft-dflash` still runs every DFlash family draft | DFlash records' type names |
| #29557 | grammars are left-factored | the forced tokens a grammar appends to drafts (Q802) |
| #30492, #30367 | the server reuses a connection after a streamed reply; a slot's cache is cleared when the global LoRA set changes | the agent check's server runs |

## Two questions read in the merged code

- **Can it load a published DFlash2 drafter such as Flint's (Q1543)?** Yes, as code; no file was loaded here.
  `--spec-type draft-dflash2` exists (`common/speculative.cpp`, the type table), and `draft-dflash` still runs any
  DFlash family draft. The head is read from the file: a `selector_hidden.weight` tensor makes it DFlash2
  (`src/models/dflash.cpp`, `load_arch_hparams`). The file needs `target_layers` in its metadata. Flint's graph
  hands its layer inputs to the draft (`src/models/flint.cpp`, `res->t_layer_inp[il]`). v4.0 already ran DFlash2
  drafts under `draft-dflash`. The new type only names the head and refuses a file whose head is another.
- **Is Ornith's routed expert product fused, and can a switch turn that off (Q350)?** It is fused, and no switch turns
  off that fusion alone.
  - The gate, up and GLU of the routed experts run as one launch, through the fork's one-token Pascal kernel
    (`ggml_cuda_mmvq_pascal`, which takes N2P and N3P with a fused gate). Upstream's guard "fusion is not universally
    faster on Pascal" refuses only compute capability 6.0 and below; a P106 is 6.1.
  - When the shared expert follows, the graph optimizer moves it beside the routed experts, and both run in one launch.
  - `GGML_CUDA_GLU_ALIAS_FUSE=0` refuses only a fusion whose GLU output reuses the input's memory: a shared expert or a
    dense layer, where the gate and up products are the input's last readers. The routed experts' input is still read
    afterwards, so their fusion stays.
  - `GGML_CUDA_DISABLE_FUSION=1` turns it off together with every other CUDA fusion, which makes it no arm for Q350.
  - Q350's first arm needs a switch of its own: a few lines where the fusion is decided. Upstream's open #32239
    changes that guard for every Pascal card.

## Commands

The merge (a laptop worktree of the fork, git only):

```bash
git -C "$SRC" fetch fork && git -C "$SRC" fetch origin --tags
git -C "$SRC" worktree add "$WT" -b pascal-p106-v5-b12410-up fork/pascal-p106-v5
git -C "$WT" merge --no-ff --no-commit b12410
# resolve as conflicts.tsv says, then: git -C "$WT" commit
```

The builds ran on a build machine's CPU, one at a time, at `--cpu-shares=256` with 8 jobs. The merge commit was built
before it was pushed:
- a static CPU build of every target, tests and tools included (`cmake -DGGML_CUDA=OFF -DGGML_NATIVE=ON
  -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF`). Its first run found the grammar change above; after the fix every
  target builds;
- the rig build of the same commit: [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)
  with `TESTS=1`;
- after the push, the rig tarball:

```bash
WORK=$WORK TESTS=1 JOBS=8 EXTRA_TARGETS="test-cuda-keepalive test-llama-archs" \
    DOCKER_ARGS="--cpu-shares=256" scripts/build_llamacpp_docker.sh fork/pascal-p106-v5-b12410-up
```

| Tarball | Size | md5 | Checks |
|---|---:|---|---|
| `llama.cpp-20261117-b3dfb0d89-pascal-p106-v5-b12410-up.tar.gz` | 835,969,104 B | `87c537aa51bc557cde0f6b8f6355ed85` | CUDA 12.9 (`nvidia/cuda:12.9.2-devel-ubuntu24.04`), sm_61; 0 AVX instructions in `llama-bench`; highest glibc symbol GLIBC_2.38 (the rig has 2.39) |

| Binary | md5 |
|---|---|
| `bin/llama-server` | `99fa3375aeac9a60a8393141b7137407` |
| `bin/test-backend-ops` | `f5c2af6952a12d2dc624e34bdb9bb62c` |
| `bin/test-llama-archs` | `888ba6c08f6a3a58958bbe08ad385964` |
| `bin/test-cuda-keepalive` | `88b3a2993ee0017dc79d61a71bab309a` |

## Next: the checks on a P106

On one rig, against v4.0's own build:
- the kernel suite and `test-llama-archs` on the card, every switch off. Every attention case needs a new record (the
  `mask_heads` rename above);
- the release files' greedy replies, byte for byte, with the scripts of
  [2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md) (`run/r153_load.sh`);
- launch groups a token per release file, against v4.0, with the fork's profiler (`GGML_CUDA_PROFILE_MODE=graph`,
  [scripts/prof_summary.py](../../../scripts/prof_summary.py)): a fusion that stops matching passes every test but adds
  launches.

`pascal-p106-v5` takes the branch when they pass.

## Files

| File | What |
|---|---|
| [conflicts.tsv](conflicts.tsv) | each conflicted file: hunks, upstream's side, the fork's side, the resolution |
