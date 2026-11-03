# Q573: Miner Llama v3 rebased onto upstream llama.cpp b12076 (2026-11-03)

Miner Llama's v3 line, `pascal-p106-v3`, sits on upstream `d77256999` (b11820, 2026-10-22). This folder holds the
method and results of moving it onto upstream `master` at `20a51f24a` (b12076, 272 commits later), as a new base branch
`pascal-p106-v3-b12076` on the fork. `pascal-p106-v3` stays as it is. What the move costs and why it was done now is in
[upstream_rebase.md](../../research/notes/upstream_rebase.md); the registry row is Q573 in
[quants/experiments.md](../../../quants/experiments.md).

## The branch

`pascal-p106-v3-b12076` at `b05ecc2d1`: 101 commits on upstream `20a51f24a`, 80 files, +15.2k/−0.4k lines.

- **92 commits replay v3's 91.** 77 apply as they are. Upstream PR #31047, which v3 carried as two commits, is replaced
  by its current three-commit head, which is rebased on `master`. The rest were ported by hand, the card-built
  attention mask onto upstream's new mask code (#29510, #31558) among them.
- **Nine commits finish the move:**

| Commit | What | Why |
|---|---|---|
| `39e9423ca` | the persistent MoE kernel's Q8_1 quantizer and the fused norm's Q8_1 copy store d·Σq as the block sum | upstream's `quantize_q8_1` does (#31247); both must give its blocks bit for bit |
| `b8abbf252` | the compute-6.1 decode and prompt attention kernels decline nodes that read K/V through kv rows | upstream's flash attention can gather K/V rows (`src[5]`, #29510); the Pascal kernels index K/V by cell. A P106 never gets such nodes (they need the MMA kernel); a newer GPU with the kernels forced on would |
| `230506b65` | `Q4_1_64` retired (this reverts the commit that added it) | upstream gave its type id 43 to F8_E4M3 (#27512); no file in use needs it |
| `db6ed9f99` | the compute-6.1 prompt attention kernel takes upstream's new `KV_rows` kernel argument, unused | `fattn_kernel_t` gained it; the build failed without |
| `165b306f6` | comments of the mask port | comments only |
| `45fbad874` | the matrix-vector MUL_MAT_ID batch limit from the architecture the device's code was built for | `mul_mat_vec_q_moe` takes its launch bounds from that architecture, while the host limit came from the device's own. A newer GPU running the compute_61 PTX launched more warps than the bounds allow, and the launch failed (v3 fails the same way). Unchanged on a P106 |
| `879f3b86f` | the KV cache sets each layer's members by name | the positional `{ il, k, v, k_stream, v_stream, kq_cells }` put the card-mask cell copy into the K rotation that upstream inserted before it (#31073): every model with attention layers on a GPU aborted building its first graph |
| `15b5de992` | `test-backend-ops` `KQ_MASK` fills the harness's sentinel tensors | left as they were, they held whatever an earlier test left; in a full run that was NaN at times |
| `b05ecc2d1` | a device without a backend registry reports no features | upstream's meta device for tensor split has none; creating a context with `-sm tensor` aborted |

Three of these matter on the rig: `39e9423ca` (exact numerics), `230506b65` (file compatibility) and `879f3b86f`, without
which the rebased engine runs no model on a GPU. Its compiler sees two pointers of the same type, so the merge built
cleanly; `test-llama-archs` and a real Flint file found it.

## Results

All runs used one RTX 3090 of a 7-GPU machine, through the rig build's compute_61 PTX (the code the P106's sm_61 machine
code is compiled from), in the CUDA 12.9 container. No timings were taken; other jobs shared the machine.

| Check | Build | Result |
|---|---|---|
| `test-backend-ops`, every case, the fork's CUDA paths off (Pascal matrix-vector kernels, persistent MoE kernel, both Pascal attention kernels, two-launch attention block, the norm's Q8_1 copy) | `45fbad874` | 20,053/20,054; the one failure is `KQ_MASK`'s sentinel (fixed in `15b5de992`) |
| the same with all of them forced on (`=2`) | `45fbad874` | 20,157/20,158; the same `KQ_MASK` case |
| `KQ_MASK`, paths off and forced on | `b05ecc2d1` | 26/26 and 26/26 |
| `MOE_LAYER` (17 Flint-shaped layers: K2h-r16, G3, S4-L0 and UD-Q4_K_XL types) with the persistent kernel's self-check | `45fbad874` | 17/17 against the CPU; every stage bit for bit equal to the per-op path in all 17 |
| the same with a deliberate routing error (`GGML_CUDA_PERSISTENT_MOE_FAULT=2`) | `45fbad874` | the self-check reports the differing layer in all 17 |
| `ADD_RMS_NORM_MUL_Q8` with the stock matrix-vector kernels | `45fbad874` | 32/32 |
| `test-llama-archs` (random tiny models of every architecture, each device against the CPU: GPU, CPU, tensor split, host experts, MoE cache) | `b05ecc2d1` | all 411 tests passed |
| the same with the fork's CUDA paths forced on | `b05ecc2d1` | all 411 tests passed |
| the same with the card-built mask for every micro-batch and its check (`LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1`) | `b05ecc2d1` | all 411 tests passed; 2,429 card-built masks equal to the host's (1,090 GPU copies, 1,339 CPU copies; micro-batches of 32, 64 and 128 tokens; no window, standard and chunked windows) |
| Flint K2h-r16 (`llama-perplexity -ub 1`, 20 layers on the GPU, 256 one-token steps): the persistent kernel's self-check | `879f3b86f` | 0 differing layers in all 258 graph evaluations (18 or 19 layers each) |
| the same file, 30 layers on the GPU, 128 steps: logits with the persistent kernel off and on | `b05ecc2d1` | byte for byte equal (18.9 MB); PPL 15.668 both |
| the same, stock matrix-vector kernels (Q5_0 reads the block sum): logits with the norm's Q8_1 copy off and on | `b05ecc2d1` | byte for byte equal; the norm wrote 57–59 Q8_1 rows per captured graph that 143–146 products read; PPL 15.406 both |
| the rig tarball as packaged (built from the pushed branch): `MOE_LAYER` with the self-check, `KQ_MASK`, `ADD_RMS_NORM_MUL_Q8`, `MUL_MAT_ID_SHARED`, the Pascal kernels forced on | `b05ecc2d1` | 111/111; 0 differing stages in all 17 layers |

Between `45fbad874` and `b05ecc2d1` only `src/` and `tests/` changed (the last three commits), so the `test-backend-ops`
results hold for the final build; `KQ_MASK`, whose test changed, was run again on it.

**The card-built mask stays.** Its check passes on every architecture, so the chain is kept, off by default as before
(`LLAMA_GPU_KQ_MASK=<min tokens>` turns it on). Its cell copies are still allocated on every card when it is off (8 bytes
per cell and cache; Q585).

**Not covered here:**
- Speed. Upstream changed how CUDA graphs launch (chunked capture, #30804; eviction, #30662), so an agent-check A/B on a
  rig decides whether the new base is as fast as v3.
- Identity on a P106: greedy outputs against the v3 tarball's on the rig.
- `llama-pipe-recurrent-check`: it lives on the hybrid-model branch (`-hybrid`), not on this base.
- The feature branches (`-hybrid`, `-bf16mv`, `-dflash` and the older ones) are not rebased yet. A three-way merge onto
  this base, with `pascal-p106-v3` as the merge base, applies `-bf16mv` and `-dflash` cleanly (`-dflash`'s copy of
  #30927 is upstream now); `-hybrid` conflicts in `llama-graph.cpp` and `llama-memory-recurrent.{cpp,h}`; `-lowbit` and
  `-mtpk` conflict in the type tables (the retired `Q4_1_64` beside their new types) and the MMQ tile loaders.

## Files

| File | What |
|---|---|
| [run/build.sh](run/build.sh) | the rig build: [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1`, then `test-llama-archs` from the same tree |
| [run/check.sh](run/check.sh) | the `test-backend-ops` and `test-llama-archs` runs above, one GPU |
| [run/e2e.sh](run/e2e.sh) | the Flint runs: the self-check and the two byte-for-byte logit comparisons |

## Commands

```bash
# the rig tarball and test-llama-archs (WORK holds the llama.cpp clone with the fork as remote "fork")
WORK=<dir> REF=fork/pascal-p106-v3-b12076 JOBS=8 docs/data/2026-11-03_upstream_rebase/run/build.sh
tar xzf <dir>/llama.cpp-20261103-b05ecc2d1-pascal-p106-v3-b12076.tar.gz -C <dir>/bin
# test-backend-ops and test-llama-archs (two GPUs in parallel: split RUNS between them)
BIN=<dir>/bin/llama.cpp-20261103-b05ecc2d1-pascal-p106-v3-b12076 ARCHS=<dir>/extra/test-llama-archs WORK=<dir> GPU=0 \
    docs/data/2026-11-03_upstream_rebase/run/check.sh
# Flint: the self-check on 20 layers, then the comparisons on 30 layers and one chunk
BIN=... MODEL=<flint-q113-K2h-r16.gguf> TEXT=<wikitext-2-raw/wiki.test.raw> WORK=<dir> GPU=0 NGL=20 STEPS=1 \
    docs/data/2026-11-03_upstream_rebase/run/e2e.sh
BIN=... MODEL=... TEXT=... WORK=<dir> GPU=0 NGL=30 CHUNKS=1 STEPS="2 3" docs/data/2026-11-03_upstream_rebase/run/e2e.sh
```

The `test-backend-ops` runs above used the build of `45fbad874` and the Flint self-check that of `879f3b86f`, made the
same way from the branch at those commits.

## Provenance and licence

The fork is MIT-licensed like llama.cpp. Flint K2h-r16 is Q113's quant of Flint (see the registry); wikitext-2 is
CC BY-SA 3.0. The tables summarize the run logs, which are not published.
