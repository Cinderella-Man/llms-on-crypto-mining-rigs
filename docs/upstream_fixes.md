# Bugs in upstream llama.cpp found while building Miner Llama, with fixes

Miner Llama is my Pascal-only fork of llama.cpp for mining rigs of nine P106 cards. Testing its v3 release turned up
bugs that are not in my code but in upstream llama.cpp's, and that upstream `master` still has. This page describes
each one well enough to check line by line, with a patch on upstream `master` and a reproduction before and after.

- **Upstream base:** `master` at [`45b6604b2`](https://github.com/ggml-org/llama.cpp/commit/45b6604b20f17b7d7638042306c0cd1b6af5999c)
  (2026-11-05). Every line number and quote below is from that commit.
- **The patches:** one branch per fix in my fork, each cut from that base and holding only its own commits, in
  upstream's style and touching nothing else, each with a pull request inside my fork against a copy of that base
  ([`upstream-master-45b6604b2`](https://github.com/Cinderella-Man/miner-llama/tree/upstream-master-45b6604b2)):
  [Cinderella-Man/miner-llama#1](https://github.com/Cinderella-Man/miner-llama/pull/1) (bug 1) and
  [Cinderella-Man/miner-llama#2](https://github.com/Cinderella-Man/miner-llama/pull/2) (bug 2). Branch
  [`upstream-fixes`](https://github.com/Cinderella-Man/miner-llama/tree/upstream-fixes) holds both; the measurements ran on
  its commits (the data folder has the details).
- **Measured or read:** every claim says which. "Measured" means a run on upstream code (the base or the branch),
  with its numbers and scripts in [data/2026-11-05_upstream_fixes](data/2026-11-05_upstream_fixes/README.md); "by
  reading" means from the source, not run.

| # | What | Where | Status on `master` | Patch | Reproduction |
|---|---|---|---|---|---|
| 1 | A one-token `MUL_MAT_ID` writes rows past each used expert into the next expert's output | CUDA, `mmvq.cu` | present | [`e52e823`](https://github.com/Cinderella-Man/miner-llama/commit/e52e8234622b4615eaa2a63a27a1d691983b6f76) (test), [`44956c7`](https://github.com/Cinderella-Man/miner-llama/commit/44956c778bf973de8beba846d31ce2f2f8e869c8) (fix), [PR #1](https://github.com/Cinderella-Man/miner-llama/pull/1) | measured: the new cases fail 30/30 and 29/30 on `master`, 0/30 each with the fix, 0 of 80 runs in all (RTX 3090) |
| 2 | After a speculative checkpoint restore the sampler gets its used random draws back, so rejection sampling is biased | `llama-server`, `common/sampling.cpp` | present | [`b50c521`](https://github.com/Cinderella-Man/miner-llama/commit/b50c521d7feb74dcfeaa8fbc428a484c0eb58805), [PR #2](https://github.com/Cinderella-Man/miner-llama/pull/2) | measured: p = 6.05e-9 on `master`, 0.23 with the fix (chi-square, 1,500 requests an arm) |
| 3 | `MUL_MAT_ID`'s batch limit on a newer GPU running older code | CUDA, `mmvq.cu` | fixed for real builds by [#31679](https://github.com/ggml-org/llama.cpp/pull/31679); a case under `CUDA_FORCE_PTX_JIT=1` remains | none | measured: `MUL_MAT_ID_SHARED` aborts under `CUDA_FORCE_PTX_JIT=1` on `master` |

## 1. CUDA: a one-token `MUL_MAT_ID` writes past each expert's rows

### What goes wrong

When a `MUL_MAT_ID` with one token goes to `mul_mat_vec_q` and the expert matrices' row count is not a multiple of
the kernel's rows per block, the last block of every used expert writes its extra rows into the first rows of the next
used expert's output (and the last expert's past the end of `dst`), so those rows are wrong whenever that block writes
after the next expert's first block.

### The code and why it is wrong

`mul_mat_vec_q` gives each block `rows_per_cuda_block` consecutive rows
([`mmvq.cu:647`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L647)),
and the grid has `ceil(nrows_x / rpb)` blocks in x
([`mmvq.cu:1041`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L1041)),
so when `nrows_x % rpb != 0` the last block's last rows lie past the matrix. The K loop computes them anyway (it reads
`rows_per_cuda_block` rows from `row0`, line 742 onwards), and the write is meant to drop them:

```cpp
// mmvq.cu:832
            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
                ...
                dst[j*stride_col_dst + i] = result;   // :864, after dst += ... + channel_dst*stride_channel_dst + row0 (:809)
```

The bound is the column stride of `dst`, not the row count. For `MUL_MAT` that is the same number when `dst` is
contiguous (`stride_col_dst = dst->nb[1]/4 = ne0 = nrows_x`). For `MUL_MAT_ID` the host swaps the roles
([`mmvq.cu:1576–1578`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L1576)):

```cpp
    const int64_t stride_col_dst     = ids ? s2   : s1;   // MUL_MAT_ID: dst->nb[2]/4 = m * n_used, the next token
    const int64_t stride_channel_dst = ids ? s1   : s2;   // MUL_MAT_ID: dst->nb[1]/4 = m, the next used expert
```

`dst` of a `MUL_MAT_ID` is `[m, n_used, n_tokens]`: the outputs of a token's used experts lie next to each other,
`m` floats apart. With one token the bound is `m * n_used`, so row `m + r` of slot `s` (for `r < rpb - 1`) passes the
guard and lands on row `r` of slot `s + 1`. The value written there is computed from the memory after slot `s`'s expert
matrix (`kbx_offset` at line 742 is `channel_x*stride_channel_x + row0*stride_row_x`): with the experts stored
contiguously that is the first rows of expert `ids[s] + 1`, times slot `s`'s input, not slot `s + 1`'s product. Slot
`s + 1`'s own first block writes the right value to the same addresses, and nothing orders the two blocks, so the
result is a race. For the last slot the extra rows go past the end of `dst`, and for the last expert (`ids[s] ==
n_as - 1`) the K loop reads past the end of the weight tensor (both by reading; where they land depends on the
allocator's padding).

The bias prefetch of the fused path has the same bound
([`mmvq.cu:711`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L711))
and reads `x_bias`/`gate_bias` rows past the expert's bias row (past the bias tensor for the last expert), whose
values then go into the misplaced rows.

The multi-token kernel `mul_mat_vec_q_moe`, which takes `MUL_MAT_ID` with 2 to 8 tokens, bounds the same rows by
`nrows_x` ([`mmvq.cu:983`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L983)),
so only the one-token path is affected.

**History (by reading):** the guard came with the one-token `MUL_MAT_ID` path in
[#13014](https://github.com/ggml-org/llama.cpp/pull/13014) (2025-04), as `row0 + int(threadIdx.x) < stride_col_dst`,
when a one-column block had one row (`calc_rows_per_block` returned 1 for `ncols_dst == 1`), so it was never reached.
It became reachable when one-column blocks got more rows: [#20635](https://github.com/ggml-org/llama.cpp/pull/20635)
(2026-03, 4 rows a block with a small K) and [#30517](https://github.com/ggml-org/llama.cpp/pull/30517) (2026-10, 2
rows a block for low-bit types). [#25445](https://github.com/ggml-org/llama.cpp/pull/25445) (2026-07) gave the write
guard its current form, and [#16847](https://github.com/ggml-org/llama.cpp/pull/16847) (2025-10) the bias prefetch.

### Who is affected, and when

All of these at once (by reading of `master`):
- the CUDA backend, `MUL_MAT_ID` with **one token** (`dst->ne[2] == 1`, i.e. one-sequence decoding of a MoE model)
  and **at least 2 used experts**, on a quantized type that takes MMVQ;
- **more than one row a block** for `ncols_dst == 1`
  ([`calc_rows_per_block`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L596)):
  4 (`nwarps`) when `should_use_small_k` holds, i.e. the row has fewer than `nwarps * vdr * warp_size / qi` blocks
  (on NVIDIA with 4 warps: K < 2048 for Q4_0 and Q4_K, K < 1024 for Q8_0, K < 4096 for Q2_K) and
  the type is not excluded for that GPU; or 2 on the generic table (Pascal, Ampere and newer) for Q1_0, Q2_K, Q3_K, IQ1_S and IQ1_M at any K;
- **an expert row count `m` that is not a multiple of that number**: for gate and up projections `m` is the expert
  FFN width, for the down projection it is `n_embd`.

No model in Miner Llama's release has such a row count, and I have not surveyed other released models or measured an
affected real one. A row count that is not a multiple of 4 is enough for the 4-row blocks; for the 2-row blocks it
must be odd. AMD builds share the kernel, but their tables differ and I have not checked them.

### How I found it

Miner Llama's release check runs `test-backend-ops` on an RTX 3090 through the build's compute 6.1 PTX
(`CUDA_FORCE_PTX_JIT=1`). One case of a fork-only type, `MUL_MAT_ID(n3p_128, n_mats=32, n_used=10, m=2561, n=1,
k=640)`, failed (NMSE 9.1e-4 > 5e-4). The fork's Pascal kernel declines 2561 rows of that type, so the product goes to
upstream's `mul_mat_vec_q`. Repeating the case 30 times on two builds failed 23 and 19 times; replaying one recorded
input 10 times per build gave NMSE from 7.4e-6 to 1.3e-3 on identical inputs, and compute-sanitizer (memcheck,
initcheck, racecheck) found 0 errors while the case still failed. A result that changes run to run on fixed inputs,
with no out-of-allocation access, is a race between blocks; at K = 640 the kernel runs 4 rows a block, 2561 = 4 × 640 +
1, and the guard above lets rows 2561–2563 through. Those runs are in my fork's
[data/2026-11-05_ladder5](data/2026-11-05_ladder5/README.md) (T0b).

### The fix and why it is right

Pass `nrows_x` to `mul_mat_vec_q` (it is already known at every launch) and bound both the bias prefetch and the write
by it, as `mul_mat_vec_q_moe` does:

```cpp
-            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < stride_col_dst)) {
+            // bound by nrows_x, not stride_col_dst: with ids and one token stride_col_dst spans all used experts
+            if (threadIdx.x == i && (rows_per_cuda_block == 1 || uint32_t(row0 + i) < nrows_x)) {
```

- A block owns rows `row0 .. row0 + rpb - 1` of its slot; row `r` exists exactly when `r < nrows_x`. That is the
  bound for every layout, `MUL_MAT` or `MUL_MAT_ID`, contiguous or not.
- `nrows_x <= stride_col_dst` on every launch (the rows of one output column are a subset of one column stride), and
  rows at or past `nrows_x` exist only when `nrows_x % rpb != 0`. So for every shape that was right before, both guards
  let the same rows through and the output is bit for bit the same; only the rows that should never have been written
  change.
- The fused shared expert (the last `blockIdx.y` with `fusion.shared_up`) uses its own `stride_col_dst`, but the
  fusion requires the shared weight to have the routed weight's row count
  ([`ggml-cuda.cu:2141`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/ggml-cuda.cu#L2141)),
  so `nrows_x` bounds it too.
- The K loop still computes the rows past the matrix for the last block, as it does for `MUL_MAT`; the patch leaves
  that alone, since those values are now never stored.

The test commit adds two cases that `master` lacks (no `MUL_MAT_ID` case on a quantized type has `m` that is not a
multiple of 4): Q4_0 at K = 640 (4 rows a block) and Q2_K at K = 4096 (2 rows a block), both with 32 experts, 10 used,
`m = 129`, one token.

### Reproduction, before and after

Measured on one RTX 3090 (driver 595.99.02, CUDA 12.9.2), with `test-backend-ops` built static for
`61-virtual;86-real`: once at `master` plus the test commit, once with the fix on top. Each run draws new random
inputs; the builds alternate within each repetition
([run/guard_reps.sh](data/2026-11-05_upstream_fixes/run/guard_reps.sh); the limit for `MUL_MAT_ID` is NMSE 5e-4).

| Case (32 experts, 10 used, `m = 129`, one token) | Code the GPU runs | `master` + test: failed | median NMSE of the failures | with the fix: failed |
|---|---|---:|---:|---:|
| Q4_0, K = 640 (4 rows a block) | sm_86 | **30 / 30** | 0.018 (0.0029–1.0) | **0 / 30** |
| Q2_K, K = 4096 (2 rows a block) | sm_86 | **29 / 30** | 0.0026 (0.00063–0.016) | **0 / 30** |
| Q4_0, K = 640 | compute_61 PTX (`CUDA_FORCE_PTX_JIT=1`) | **10 / 10** | 0.012 (0.0011–0.032) | **0 / 10** |
| Q2_K, K = 4096 | compute_61 PTX | **10 / 10** | 0.0049 (0.00071–1.0) | **0 / 10** |

Every run's verdict and error: [guard_sass.tsv](data/2026-11-05_upstream_fixes/guard_sass.tsv),
[guard_jit.tsv](data/2026-11-05_upstream_fixes/guard_jit.tsv). At `m = 129` the race covers 3 rows of every 129 with
4-row blocks (129 = 4 × 32 + 1) and 1 row with 2-row blocks, enough to move the NMSE far over the limit whenever a
race is lost; at `m = 2561`, as in my fork's case, the same 3 rows are diluted 20 times and the case failed about 2
runs in 3.

```bash
# on upstream master with the two commits applied (the test commit alone shows the failure)
cmake -B build -DGGML_CUDA=ON && cmake --build build -j --target test-backend-ops
for i in $(seq 30); do
    ./build/bin/test-backend-ops -b CUDA0 -o MUL_MAT_ID -p 'n_mats=32,n_used=10,b=0,m=129,n=1,'
done 2>&1 | grep -c 'MUL_MAT_ID(.*FAIL'    # failed cases: up to 60 on master, 0 with the fix
```

### The patch

Pull request [Cinderella-Man/miner-llama#1](https://github.com/Cinderella-Man/miner-llama/pull/1) (in my fork).
Branch [`upstream-fix/mmvq-row-guard`](https://github.com/Cinderella-Man/miner-llama/tree/upstream-fix/mmvq-row-guard),
on `master` `45b6604b2`, two commits:
- [`e52e823`](https://github.com/Cinderella-Man/miner-llama/commit/e52e8234622b4615eaa2a63a27a1d691983b6f76)
  `tests : add MUL_MAT_ID cases with one token and rows not a multiple of 4` (`tests/test-backend-ops.cpp`, +6);
- [`44956c7`](https://github.com/Cinderella-Man/miner-llama/commit/44956c778bf973de8beba846d31ce2f2f8e869c8)
  `CUDA: bound the rows of mul_mat_vec_q by nrows_x, not stride_col_dst` (`ggml/src/ggml-cuda/mmvq.cu`, +15 −14: the
  new kernel argument, its 10 call sites, the two guards).

As files for `git am`: [patches/mmvq-row-guard/](data/2026-11-05_upstream_fixes/patches/mmvq-row-guard/).

## 2. Speculative decoding: the random draws come back after a checkpoint restore

### What goes wrong

When `llama-server` rolls a partly accepted draft back by restoring its checkpoint, it also restores the sampler's
random state from before that step, so the next verification draws again the uniforms whose outcome is already in the
output, and the sampled text no longer follows the target model's distribution.

### The code and why it is wrong

`post_decode()` in [`tools/server/server-context.cpp`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/tools/server/server-context.cpp#L4950)
verifies a draft like this:

```cpp
// server-context.cpp:4950
                common_sampler_ptr smpl_save(common_sampler_clone(slot.smpl.get()));
                ...
                } else if (slot.spec_is_replay && slot.use_spec_rejection()) {                         // :4965
                    accepted = server_accept_replay(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft);
                } else if (use_rejection) {
                    accepted = common_sampler_sample_and_accept_n_rejection(slot.smpl.get(), slot.ctx_tgt, spec_i_batch, slot.spec_draft, slot.spec_draft_q);
                ...
                const bool use_ckpt_tgt =                                                              // :4978
                    ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_FULL ||
                    (ctx_tgt_seq_rm_type == COMMON_CONTEXT_SEQ_RM_TYPE_RS && n_draft > llama_n_rs_seq(ctx_tgt));
                if (n_rollback > 0) {
                    if (use_ckpt_tgt) {
                        slot.spec_is_replay = true;                                                    // :4990
                        slot.spec_draft = std::move(accepted);
                        ... // load the checkpoint, drop the positions after it
                        common_sampler_copy(smpl_save.get(), slot.smpl.get());                         // :5006
                        return;
```

- **The verification uses two random streams.** `common_sampler_sample_and_accept_n_rejection`
  ([`common/sampling.cpp:736`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/common/sampling.cpp#L736))
  draws its acceptance uniforms and its residual draw from `gsmpl->rng` (`uni(gsmpl->rng)`, lines 795 and 814), and at
  every position it visits it calls `common_sampler_sample` (line 754), which runs the chain and so draws from the
  `dist` sampler's own `std::mt19937`. Under a grammar that draw also decides which candidates the position has: the
  drawn token is checked against the grammar first, and only on a rejection is the row masked and sampled again.
- **The restore puts both streams back.** `common_sampler_copy`
  ([`sampling.cpp:531–549`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/common/sampling.cpp#L531))
  copies the chain with `llama_sampler_copy` (the `dist` sampler's state is its RNG) and then `dst->rng = src->rng`.
- **The replay does not use `rng`.** On the next call the kept tokens are re-accepted by
  [`server_accept_replay`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/tools/server/server-context.cpp#L71),
  which calls `common_sampler_sample` and `common_sampler_accept` per token and never draws a uniform.

So after a restore, `rng` stands where it stood before the rejected step, and the verification after the replay draws
the same uniforms: `u_1` was below `p/q` at the first position (the token was accepted), `u_{a+1}` above it where the
draft was rejected, and so on. Those values are no longer independent of the text produced so far; they were selected
by it. Rejection sampling is exact only if every uniform is fresh. The chain's `dist` draws that the step used beyond
the positions the replay re-samples (grammar resamples, the rejected position) come back the same way.

**History (by reading):** `server_accept_replay`, the restore of the sampler and `common_sampler_copy` copying `rng`
came with rejection sampling for draft-simple and MTP in [#27694](https://github.com/ggml-org/llama.cpp/pull/27694)
(2026-10-02); [#30663](https://github.com/ggml-org/llama.cpp/pull/30663) added the same verifier for DFlash2 drafts,
and [#31350](https://github.com/ggml-org/llama.cpp/pull/31350) (2026-10-29) made a recurrent target with rollback
snapshots restore the checkpoint too when the draft is longer than `n_rs_seq`.

### Who is affected, and when

All of these at once (by reading):
- `llama-server` with speculative decoding at **temperature above 0**, with a drafter that fills draft distributions,
  so the draft is verified by rejection sampling (`use_spec_rejection()` is `temp > 0`; draft-simple, draft-mtp and
  DFlash2 with `--spec-draft-sampling probabilistic`). N-gram drafts carry no distributions and are verified by
  sample-and-match, which draws only from `dist` and is not affected by the `rng` path;
- a **target that restores its checkpoint** on a partial acceptance: one whose memory cannot remove part of a
  sequence (`COMMON_CONTEXT_SEQ_RM_TYPE_FULL`, e.g. a recurrent or hybrid model without rollback snapshots), or a
  recurrent or hybrid model whose draft is longer than its rollback snapshots (`n_rs_seq`);
- a step after a partly accepted draft, which on such targets is most steps.

Measured: the bias is large under a grammar (the JSON tool call of the test below). Without a grammar my test did not
tell the outputs apart at 4,000 requests (smallest p = 0.0049 against a threshold of 1e-4, on my fork's base, which has
the same path), so on plain text it is smaller than that test can see, not shown to be absent.

### How I found it

I wrote block verification (Sun et al., arXiv:2403.10444) for my fork and checked it with a statistical test: random
-weight models, every speculative arm against plain decoding with two-sample chi-square tests on the outputs' first
tokens and, under a grammar, on a JSON tool call's 30 possible outcomes. The control (temperature 0.8 against 1.0) is
told apart at p = 1.7e-21. Every exact arm passed except the base build's own token-by-token rejection under the
grammar: p = 6e-9, total variation 0.095, with 1,500 requests per arm. Keeping the draws across the restore made the
same arm pass (p = 0.24). Those runs are in [data/2026-11-04_block_verify](data/2026-11-04_block_verify/README.md).

### The fix and why it is right

`common_sampler_copy_keep_rng(src, dst)` copies the sampler as `common_sampler_copy` does and then gives `dst` back its
own `rng` and the state of the chain's `dist` and `xtc` samplers; the restore in `post_decode()` uses it.

- The restore must undo what the rejected step **accepted** (the grammar's state, the penalties' history, `prev`, the
  reasoning budget), because the tokens after the checkpoint are dropped. It must not undo what the step **drew**:
  a uniform that has decided an outcome cannot be drawn again. With the streams kept, every draw after the restore is
  one the output has not seen, which is what plain decoding does.
- `dist` and `xtc` hold nothing but random state (`xtc` has no `accept`; `dist`'s `accept` only advances its stream for
  draws made by backend sampling), so keeping them aside and copying them back undoes nothing the accepted text
  implies.
- Mirostat (v1 and v2) and adaptive-p keep a random stream and a history in one sampler. The patch restores them whole,
  as before; splitting their state is left out of this minimal change.
- The output still depends only on the seed, but it changes for a given seed on setups that restore checkpoints.

### Reproduction, before and after

Measured on CPU, no download needed, with
[run/stat.sh](data/2026-11-05_upstream_fixes/run/stat.sh) (the grammar arm of
[2026-11-04_block_verify](data/2026-11-04_block_verify/README.md)'s test, its models and its checker):

- **Models:** a target and a drafter with random weights, written by `test-llama-archs` for the `qwen4exp`
  architecture (a hybrid with recurrent layers; target weights of stdev 1.0, drafter 0.8, the same seed) with GPT-2's
  vocabulary. The target has no rollback snapshots, so every partly accepted draft restores the checkpoint.
- **Requests:** 1,500 per arm, each with its own seed, temperature 1, top-k 8, a JSON tool call under
  [stat_tool.gbnf](data/2026-11-04_block_verify/run/stat_tool.gbnf) (30 possible outcomes), `llama-server -np 1
  -c 4096`, the draft arms with `-md <drafter> --spec-type draft-simple --spec-draft-n-max 4
  --spec-draft-sampling probabilistic --spec-draft-p-min 0`.
- **Test:** two-sample chi-square of the outcome counts against plain decoding on the same build (cells pooled to an
  expected count of at least 5, 21 cells), failing at p < 1e-4.

| Arm | Build | Restores | p | Total variation |
|---|---|---:|---:|---:|
| plain decoding (the reference) | `master` | 0 | — | — |
| draft-simple, rejection sampling | `master` | 12,109 | **6.05e-9** (χ² 78.9, 20 dof) | 0.095 |
| draft-simple, rejection sampling | `master` + the fix | 12,109 | **0.23** (χ² 24.2, 20 dof) | 0.057 |

In this test the drafter's tokens are never valid under the grammar (48,436 drafted, 0 accepted, in both draft arms),
so every verification rejects at the first position and restores, and the replay re-accepts the one residual token.
With `master` the sampler's `rng` therefore never advances within a request: every residual draw of a reply comes from
the same two uniforms (by reading of the code above). The `master` arm's 1,500 replies are identical to those of the
same arm on my fork's base, on which the bias was first found; plain decoding differs from it in 1 reply of 1,500 and
the fixed arm in 4.

```bash
# from a checkout of this repository; WORK must hold the two llama-server builds and the models
docker run --rm -v "$WORK:/c" -v "$PWD/docs/data/2026-11-04_block_verify/run:/r:ro" \
    -v "$PWD/docs/data/2026-11-05_upstream_fixes/run:/u:ro" -v "$MODELS:/m:ro" \
    -e B_MASTER=/c/bin-cpu-master/llama-server -e B_FIX=/c/bin-cpu-fix/llama-server \
    -e ST=/m/st/qwen4exp-moe.gguf -e SD=/m/sd/qwen4exp-moe.gguf ubuntu:24.04 bash -c '
    apt-get update -qq && apt-get install -y -qq curl jq python3 > /dev/null
    bash /u/stat.sh && python3 -I /r/stat_check.py /c/stat'
```

The models come from my fork's `test-llama-archs` (`LLAMA_ARCHS_VOCAB` copies a vocabulary into them): the commands
are in [2026-11-04_block_verify/run/cpu_check.sh](data/2026-11-04_block_verify/run/cpu_check.sh), step 3.

### The patch

Pull request [Cinderella-Man/miner-llama#2](https://github.com/Cinderella-Man/miner-llama/pull/2) (in my fork).
Branch [`upstream-fix/spec-replay-draws`](https://github.com/Cinderella-Man/miner-llama/tree/upstream-fix/spec-replay-draws),
on `master` `45b6604b2`, one commit:
[`b50c521`](https://github.com/Cinderella-Man/miner-llama/commit/b50c521d7feb74dcfeaa8fbc428a484c0eb58805)
`server : keep the random draws used across a speculative checkpoint restore` (`common/sampling.{h,cpp}` +33,
`tools/server/server-context.cpp` +3 −1). The new function:

```cpp
void common_sampler_copy_keep_rng(const common_sampler * src, common_sampler * dst) {
    if (!src || !dst || src == dst) {
        return;
    }

    // dst's random samplers, kept aside while the copy overwrites them
    // (their state is their random stream only)
    const int32_t n = llama_sampler_chain_n(dst->chain);
    std::vector<llama_sampler *> draws(n, nullptr);
    for (int32_t i = 0; i < n; ++i) {
        const llama_sampler * s = llama_sampler_chain_get(dst->chain, i);
        const char * name = llama_sampler_name(s);
        if (name && (strcmp(name, "dist") == 0 || strcmp(name, "xtc") == 0)) {
            draws[i] = llama_sampler_clone(s);
        }
    }
    const std::mt19937 rng = dst->rng;

    common_sampler_copy(src, dst);

    dst->rng = rng;
    for (int32_t i = 0; i < n; ++i) {
        if (draws[i]) {
            llama_sampler_copy(draws[i], llama_sampler_chain_get(dst->chain, i));
            llama_sampler_free(draws[i]);
        }
    }
}
```

The binary measured above was built from the same change with an older wording of that function's second comment
line (no code difference). As a file for `git am`:
[patches/spec-replay-draws/](data/2026-11-05_upstream_fixes/patches/spec-replay-draws/).

## 3. `MUL_MAT_ID`'s batch limit on a newer GPU running older code (fixed upstream)

`mul_mat_vec_q_moe` takes its launch bounds from `get_mmvq_mmid_max_batch_for_device<type>()`, evaluated for the
architecture the device code was compiled for
([`mmvq.cu:882`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L882)),
while the host decides how many tokens go to it with `get_mmvq_mmid_max_batch(type, cc)`. Until 2026-11-05 the host used
the device's own compute capability, so a newer GPU running older code (a build for compute 6.1 only, on an RTX 3090)
launched more warps than the kernel's bounds allow and the launch failed ("invalid argument"). I fixed that in my fork
on 2026-11-03; upstream fixed it independently on 2026-11-05 in [#31679](https://github.com/ggml-org/llama.cpp/pull/31679),
which makes the host use `ggml_cuda_highest_compiled_arch(cc)` (by reading: that covers a build for older
architectures only)
([`mmvq.cu:288`](https://github.com/ggml-org/llama.cpp/blob/45b6604b20f17b7d7638042306c0cd1b6af5999c/ggml/src/ggml-cuda/mmvq.cu#L288)).

What remains is `CUDA_FORCE_PTX_JIT=1` on a build whose newest PTX for the device is older
than its newest machine code: the driver then compiles the PTX, so the device code is the PTX's architecture while
`ggml_cuda_highest_compiled_arch` names the machine code's. That is a debugging setting, and every host-side choice in
ggml-cuda that uses `ggml_cuda_highest_compiled_arch` (the MMVQ parameter tables among them) makes the same assumption,
so I have no minimal patch for it. My fork's fix asks the driver which PTX version a kernel actually runs
(`cudaFuncGetAttributes(...).ptxVersion`) and takes the smallest of that, the device's capability and the highest
compiled architecture.

**Measured** with the `61-virtual;86-real` build above on the RTX 3090 under `CUDA_FORCE_PTX_JIT=1`:
`test-backend-ops -o MUL_MAT_ID_SHARED` passes the q4_K cases with 1–4 tokens and aborts at 8 tokens with
"CUDA error: invalid argument" in `ggml_cuda_kernel_launch`, on `master` and on my branch alike
([jit_shared.tsv](data/2026-11-05_upstream_fixes/jit_shared.tsv)). By reading: the host takes 8.6 (Turing and newer:
up to 8 tokens for q4_K), the device runs code compiled from compute 6.1 (Pascal and older: launch bounds for 5
warps), so 8 warps are launched against a bound of 5. Without `CUDA_FORCE_PTX_JIT` this build runs its sm_86 code,
where host and device agree (by reading; not run).

## Not included

- **MMQ with repeated expert ids** (an expert used twice by one token): out-of-bounds reads in `quantize_mmq_q8_1`. It
  is upstream's known limitation: its own cases are disabled for CUDA (`TAG_MUL_MAT_ID_DUPLICATE_ID` in
  `tests/test-backend-ops.cpp`), and routers never repeat an expert within a token.
- **Problems my fork's own code caused**, found by the same checks and fixed there: seven `test-llama-archs` failures
  from a logits-comparison option I added to that test, a shared-library link error from a function of mine, and a
  crash in Gemma 4's graph reserve that needs my `LLAMA_GPU_KQ_MASK` switch.
