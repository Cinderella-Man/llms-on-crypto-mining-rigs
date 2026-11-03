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
- `llama-pipe-recurrent-check`: it lives on the hybrid-model branch (`-hybrid`), not on this base; its run on
  `pascal-p106-v3-b12076-hybrid` is below.
- The older feature branches (`-cheapguess`, `-promptpath`, `-textreuse`, `-decodemask`, `-lowbit`, `-mtpk`). A
  three-way merge onto this base, with `pascal-p106-v3` as the merge base, applies the first four cleanly; `-lowbit`
  and `-mtpk` conflict in the type tables (the retired `Q4_1_64` beside their new types) and the MMQ tile loaders.

## The feature branches on this base (C66b)

Each branch is a new branch, `pascal-p106-v3-b12076-<name>`, made by cherry-picking its commits from `pascal-p106-v3`
onto `pascal-p106-v3-b12076` (the old branches stay as they are). `-dflash` is not carried over: its fix is upstream's
#30927, already in the base, and its `HC_STREAM_MEAN` test rides on `-bf16mv` and `-hybrid`. Each was built for the rig
by [run/branches.sh](run/branches.sh) (sm_61 code + compute_61 PTX, `TESTS=1`, 0 AVX instructions, newest glibc symbol
GLIBC_2.38), on CPUs only.

| Branch | Commit | Commits on the base | How it moved | Rig tarball, md5 |
|---|---|---|---|---|
| `-bf16mv` (Pascal BF16 matrix-vector kernel, Q546) | `abd60e7c0` | 2 | clean | `llama.cpp-20261103-abd60e7c0-pascal-p106-v3-b12076-bf16mv.tar.gz`, `d349f315ea24a5592f741f93d5d08556` |
| `-q2_0mv` (Pascal Q2_0 expert kernel, Q223) | `d29b0cc56` | 1 | clean | `llama.cpp-20261103-d29b0cc56-pascal-p106-v3-b12076-q2_0mv.tar.gz`, `8055e39db5517253680eaabe8f7fe539` |
| `-iqmmvq` (IQ2_S / IQ3_XXS in the Pascal kernels, Q166) | `76788bf4a` | 6 | clean | `llama.cpp-20261103-76788bf4a-pascal-p106-v3-b12076-iqmmvq.tar.gz`, `5913a87dbb126a16d57ca139b7ebe928` |
| `-hybrid` (pipelined decoding of the delta-net models, Q550) | `8c25952e4` | 8 | 2 commits ported by hand (below) | `llama.cpp-20261103-8c25952e4-pascal-p106-v3-b12076-hybrid.tar.gz`, `57fc7db48572b38e50f70dfd92b98348` |
| `-ladder` (the base + `-bf16mv` + `-q2_0mv`, for the speed ladder Q551) | `98bbd6a2f` | 3 | the two kernels' test cases side by side | `llama.cpp-20261103-98bbd6a2f-pascal-p106-v3-b12076-ladder.tar.gz`, `09bfc0b21d2f7413a28c6cedd3cca392` |

**`-hybrid`'s two ports.** Upstream replaced the attention input's `self_kq_mask` with `self_attn_mask` (a mask plus
optional kv rows, #29510) and no longer sets the K/V rotations as inputs there, so the hybrid memory's attention part
now takes the device-built mask exactly as a plain KV cache does on this base: its inputs when the graph
built it, upstream's `set_input_attn_mask` / `can_reuse_attn_mask` otherwise. In the recurrent memory, upstream keeps
its buffers in `llama_memory_bufs` (which also simulates allocations with `no_alloc`, #31087) and bounds a rollback by
the snapshots the last ubatch wrote (`rs_max`, #31368); the state groups for pipe slots allocate through the new holder,
copy no rows when it only simulates, and keep `rs_max` beside their own fields (pipelining needs `n_rs_seq` 0, where
`rs_max` stays 0). Every other state read and write already went through `s_copy` (whose new `s_copy_main` and
`s_copy_tail` are views of it) or through the writes the branch moves into the slot's group.

**`-iqmmvq` is not in `-ladder`:** with `-q2_0mv` it conflicts in seven places in `mmvq-pascal.cu` and one in
`mmvq-pascal-impl.cuh`, where Q2_0's tile staging and the grid types' lookup tables both change the kernels' shared
memory and their load paths. Merging them is its own task (Q603).

**Checked (CPU only):** all five compile, with their tests. `-hybrid`'s pipelined decoding on a random-weight
`qwen4exp` model (Qwen3.8-Flash-Next's architecture: a Gated DeltaNet layer with the n-gram embedding's conv history,
an attention layer with the sparse-attention indexer, hyper-connections, MoE; written by this build's
`test-llama-archs -a qwen4exp -s 1`) on the CPU, with C55's
[checks.sh](../2026-11-03_flash_next_pipelined/run/checks.sh): 29 pipelined runs (depths 1 to 4; oracle guesses with
0, 10 and 50 % made wrong; `copy`, `cont`, `ngram-simple` and `ngram-map-k` guessers) give the same tokens, logits of
every token, sequence state and probe as plain decoding, and `llama-pipe-recurrent-check` (24 forced rollbacks through
the stash, 24 pipe leaves, 8 decodes) is the same in all four too ([c66b_hybrid_cpu.tsv](c66b_hybrid_cpu.tsv)).
**Not yet:** every kernel on a P106 (`test-backend-ops` on one card for the ops each branch touches), `-hybrid` on the
GPUs (two virtual devices, the real Flash-Next file, the nine cards), and the speed ladder on the rig.

## Files

| File | What |
|---|---|
| [run/build.sh](run/build.sh) | the rig build: [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) with `TESTS=1`, then `test-llama-archs` from the same tree |
| [run/check.sh](run/check.sh) | the `test-backend-ops` and `test-llama-archs` runs above, one GPU |
| [run/e2e.sh](run/e2e.sh) | the Flint runs: the self-check and the two byte-for-byte logit comparisons |
| [run/branches.sh](run/branches.sh) | C66b: the rig builds of the feature branches, one at a time, with md5s; `test-llama-archs`, `llama-pipespec` and `llama-pipe-recurrent-check` for the branches in `EXTRA` |
| [c66b_hybrid_cpu.tsv](c66b_hybrid_cpu.tsv) | C66b: `-hybrid`'s pipelined runs and `llama-pipe-recurrent-check` on the random `qwen4exp` model, CPU: per run the depth, tokens generated, guesses, right and wrong guesses, and same/DIFF for tokens, logits, state and probe |

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
# C66b: the feature branches (each pushed to the fork first), then -hybrid's CPU checks in a container with no GPU
# (the binaries link libcuda.so.1, so the CUDA toolkit's stub stands in for it and CUDA finds no device)
P=pascal-p106-v3-b12076
WORK=<dir> BRANCHES="$P-ladder $P-hybrid $P-bf16mv $P-q2_0mv $P-iqmmvq" EXTRA="$P-hybrid" JOBS=4 \
    docs/data/2026-11-03_upstream_rebase/run/branches.sh
mkdir -p /tmp/stub && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1
LD_LIBRARY_PATH=/usr/local/cuda/lib64:/tmp/stub BIN=<dir>/extra/$P-hybrid PS=bench/pipespec OUT=<out> \
    STEPS="rand-cpu rand-prc-cpu" bash docs/data/2026-11-03_flash_next_pipelined/run/checks.sh
```

The `test-backend-ops` runs above used the build of `45fbad874` and the Flint self-check that of `879f3b86f`, made the
same way from the branch at those commits.

## Provenance and licence

The fork is MIT-licensed like llama.cpp. Flint K2h-r16 is Q113's quant of Flint (see the registry); wikitext-2 is
CC BY-SA 3.0. The tables summarize the run logs, which are not published.
