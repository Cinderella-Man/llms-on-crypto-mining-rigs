# Bringing Miner Llama v3 up to current upstream llama.cpp (2026-11-03)

Miner Llama's v3 line (`pascal-p106-v3`) sits on upstream `d77256999` (build 11820, 2026-10-22). Fixes from upstream have
been taken one at a time, most recently #30927 for DFlash drafters on Flash-Next (`pascal-p106-v3-dflash`). This note
measures what a move to current upstream costs, from a trial rebase onto `master` at `20a51f24a` (b12076, 2026-11-03).
The trial is kept as an unpushed local branch, `rebase-probe-2026-11-03`; nothing was compiled.

## Summary

1. **Most of the fork replays cleanly.** 77 of v3's 91 commits apply as they are onto 272 newer upstream commits.
2. **Two places hold the real work:**
   - **The card-built attention mask** has to be ported onto upstream's new attention-mask code.
   - **Upstream changed the Q8_1 activation block sum** ([#31247](https://github.com/ggml-org/llama.cpp/pull/31247)).
     The persistent MoE kernel's own quantizer and the fused norm path still use the old sum, so they must follow, or
     their exact-match checks fail.
3. **Nothing upstream breaks Pascal.** Compute 6.1 and DP4A are still supported, CUDA 12.x still builds, and upstream
   still switches CUDA graphs off below Volta, so the fork's Pascal graphs patch stays.
4. **Upstream gained what Flash-Next speculation needs:**
   - [#31350](https://github.com/ggml-org/llama.cpp/pull/31350): checkpoint restore for drafts longer than the state
     planes;
   - [#30141](https://github.com/ggml-org/llama.cpp/pull/30141): the MTP hidden state saved and restored;
   - [#31506](https://github.com/ggml-org/llama.cpp/pull/31506): DFlash copies its hidden features on the device;
   - [#30663](https://github.com/ggml-org/llama.cpp/pull/30663): probabilistic sampling for DFlash2;
   - [#31368](https://github.com/ggml-org/llama.cpp/pull/31368): no rollback past the last ubatch's snapshots;
   - `qwen4exp` fixes ([#30965](https://github.com/ggml-org/llama.cpp/pull/30965),
     [#31443](https://github.com/ggml-org/llama.cpp/pull/31443), [#30878](https://github.com/ggml-org/llama.cpp/pull/30878)).

   Several of these sit on code that upstream has refactored since, so taking them one by one gets dearer with each
   one.
5. **Recommendation: one rebase now, before more features land on v3.** Put it on a new base branch, since history
   that is already pushed is never rewritten. Then rebase the feature branches onto it, and go back to cherry-picks
   between release candidates. Est. 3–5 days with the checks.

## The new base branch

The rebase is done as `pascal-p106-v3-b12076` (`b05ecc2d1`, 101 commits on `20a51f24a`). It builds for the rig and
passes `test-backend-ops`, `test-llama-archs` and the persistent MoE kernel's self-check through the compute 6.1 PTX
on an RTX 3090, with the Pascal kernels on and off; the method and every result are in
[docs/data/2026-11-03_upstream_rebase/](../../data/2026-11-03_upstream_rebase/README.md).

- **Beyond the trial's ports,** the build and the tests needed six more fixes. One would have broken every model on a
  GPU: the KV cache filled each layer's members by position, and upstream's new K rotation (#31073) took the slot of the
  card-built mask's cell copy. It compiled cleanly. The others: the compute-6.1 attention kernels step aside for
  attention through kv rows, take upstream's new `KV_rows` kernel argument, and devices without a backend registry
  (upstream's tensor-split meta device) report no features; two pre-existing bugs that only full test runs on a newer
  GPU show (the MUL_MAT_ID batch limit under older PTX, and `KQ_MASK`'s test leaving the harness's sentinels unset).
- **The card-built mask stays**, off by default: with `LLAMA_GPU_KQ_MASK_CHECK=1`, 2,429 card-built masks over every
  architecture equal the host's.
- **`Q4_1_64` is retired.** Type id 43 is F8_E4M3 again, as upstream has it; no file in use needs the type.
- **Still to do:** the feature branches onto the new base (`-bf16mv` and `-dflash` apply cleanly, `-hybrid` conflicts
  in three files), then identity and the agent-check A/B on a rig.

## The two sides

| | Commits | Files |
|---|---:|---|
| `pascal-p106-v3` over its base | 91 | 92 files, +15.3k/−0.4k: about 35 CUDA kernels, 29 pipelined speculation, drafter and sampling, 9 server, 9 graph and model, 4 KV cache, 3 scheduler and allocator, 4 tests |
| Feature branches beyond it | `-hybrid` 9, `-bf16mv` 3, `-dflash` 2, `-iqmmvq` 6, `-lowbit` 7, `-mtpk` 16, `-promptpath` 3, `-cheapguess` 4, `-textreuse` 1, `-decodemask` 1 | — |
| Upstream `master` since the base | 272 (b11820 → b12076 and 16 more) | 743 files, +118.7k/−50.7k; 45 of them are files the fork also changes |

## Conflicts in the trial

14 of the 91 commits stopped. A single merge instead would meet 12 files and 48 conflicted hunks at once.

| Files | Difficulty | Why |
|---|---|---|
| `ggml-cuda.cu` | trivial | the CUDA-graph check became `instances.empty()` ([#30804](https://github.com/ggml-org/llama.cpp/pull/30804)) |
| `llama-graph.cpp`, `llama-graph.h`, `test-backend-ops.cpp`, `llama-kv-cache.cpp` | trivial | neighbouring list entries; a removed `n_pad` member ([#31200](https://github.com/ggml-org/llama.cpp/pull/31200)) |
| `ggml.h`, `llama.h`, `constants.py`, `llama-model-loader.cpp` (the fork's `Q4_1_64` type) | moderate | type ID 43 now belongs to F8_E4M3 ([#27512](https://github.com/ggml-org/llama.cpp/pull/27512)); the trial renumbered `Q4_1_64` to 44, which makes any file written with it unreadable. No released file uses it |
| `mmq.cu`, `mmq.cuh` (21 hunks) | replaced | the fork carried two commits of the open upstream PR #31047; its current head is rebased on `master` and merges cleanly |
| `llama-graph.cpp`/`.h`, `llama-kv-cache.cpp`/`.h` (the card-built mask) | hard | ported by hand onto upstream's `llm_attn_mask{raw,rows}` ([#29510](https://github.com/ggml-org/llama.cpp/pull/29510)); upstream also dropped `_cnv` ([#31558](https://github.com/ggml-org/llama.cpp/pull/31558)) and keeps the k/v rotations on the GPU ([#31073](https://github.com/ggml-org/llama.cpp/pull/31073)) |
| `-hybrid`: `llama-graph.cpp`, `llama-memory-recurrent.cpp`/`.h` | moderate | beside [#31368](https://github.com/ggml-org/llama.cpp/pull/31368)'s new rollback bound |
| `-lowbit`, `-mtpk`: `ggml.h`, `constants.py`, `mmq.cuh` | moderate | the MMQ tile loaders need the new fused gate/up signature |

`-bf16mv` rebased cleanly, and the older branches merge cleanly in the same test. Upstream now carries #30927, so the
copies of it on `-dflash`, `-hybrid` and `-bf16mv` drop out. `qwen4exp` now builds its sparse-attention selection
without the regular attention mask ([#30965](https://github.com/ggml-org/llama.cpp/pull/30965)), so on Flash-Next the
card-built mask may buy little (est.).

## Risks to check

- **Numerics** (#31247): the persistent MoE kernel is on by default on Pascal, and its check against the per-operation
  path must pass again after its quantizer follows the new block sum.
- **Speed:** chunked CUDA graphs ([#30804](https://github.com/ggml-org/llama.cpp/pull/30804), 512 nodes by default) and
  graph eviction ([#30662](https://github.com/ggml-org/llama.cpp/pull/30662)) change how graphs launch; an agent-check
  A/B on the rig settles it.
- **Not audited in the trial:** code that merged cleanly as text but whose meaning upstream changed. That covers the
  profiler and the persistent kernel under chunked capture, and the KV-cache changes against #29510's new cell code.
- **Not a host-RAM gain:** the loader now reuses one staging buffer ([#31321](https://github.com/ggml-org/llama.cpp/pull/31321)),
  which loads faster but does not lower the steady-state host memory (est.).

## Options

| Option | Work | Keeps linear history | Cost over time |
|---|---|---|---|
| **One rebase onto a new base branch** | ~1 more laptop session to finish the ports (the Q8_1 sums, the lowbit/mtpk MMQ loaders, the rest of `-hybrid`'s mask); 2–4 build-and-test rounds on the quant machine; rig checks. Est. 3–5 days | yes | low: later fixes cherry-pick onto current code |
| A merge commit | the same conflicts and checks in one step | no | low |
| Cherry-picks only | ~one session per fix, as for #30927 | yes | rising: #31506, #30141 and #31350 sit on refactored code |

**The checks:**
- **On the quant machine:** a `TESTS=1` build; `test-backend-ops` through the compute 6.1 PTX with the Pascal kernels on
  and off; the persistent MoE kernel's check mode; `test-llama-archs`; `llama-pipe-recurrent-check`;
  `LLAMA_GPU_KQ_MASK_CHECK=1`.
- **On a rig:** identity checks against the v3 tarball's greedy outputs, then the agent check A/B.

**The two choices, as made:**
- **The card-built mask:** kept, since `LLAMA_GPU_KQ_MASK_CHECK=1` passes; it stays off by default.
- **`Q4_1_64`:** retired rather than given a new type ID.

`llama-pipe-recurrent-check` lives on the hybrid-model branch, so it runs when `-hybrid` is rebased.

The registry row is Q573 in [quants/experiments.md](../../../quants/experiments.md).
