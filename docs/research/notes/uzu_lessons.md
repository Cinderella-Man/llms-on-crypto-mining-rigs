# uzu (github.com/trymirai/uzu): what Miner Llama can learn from it (2026-10-27)

Read-only study through GitHub web pages, `raw.githubusercontent.com`, the GitHub API, PR/issue pages and the DFlash
paper. Nothing was cloned, built, installed or run. Paths below are relative to
`https://github.com/trymirai/uzu/blob/main/`. "(est.)" marks my own arithmetic. Every uzu number is the PR author's own,
measured on Apple M-series hardware, and nobody has reproduced it.

## 1. What uzu is

| | |
|---|---|
| What | An on-device inference engine by Mirai (trymirai.com) for LLMs (plus classifiers and, earlier, TTS), with Swift/Python/TypeScript bindings, a CLI, an OpenAI-compatible server and a "pi" coding-agent extension (`apps/uzu-pi-extension`) |
| Licence | MIT (GitHub API, `LICENSE`) |
| Language | Rust for the host side, Metal Shading Language for the kernels, a CPU backend for reference ("CPU twins" of the kernels) |
| Targets | Working: Apple Silicon macOS and iOS (`aarch64-apple-darwin`, `-ios`, `-ios-macabi`, `-ios-sim`, `x86_64-apple-darwin`). Linux, Windows and wasm are "in progress" per the README. **There is no CUDA or Vulkan backend**: `crates/uzu-engine/src/backends/` has only `common`, `cpu` and `metal`. Builds need macOS 26 and Metal 4 (issue #841) |
| Model format | Its own: `model.safetensors` + `config.json` (+ `speculator/` with `shapes.json`), made by the separate `lalamo` converter. No GGUF |
| Maturity | v0.6.13 (2026-10-22). Repo created 2025-06-23, 2,642 stars, 152 forks, 26 open issues, several commits a day (last push 2026-10-27). Big rewrite in June 2026 (#545). MoE was **removed** on 2026-10-02 (#886 "Burn MoE") and is being re-added in open drafts (#999, #1000, #1009, #1018, #1020) |
| Layout | `crates/uzu-engine` = the engine (kernels, blocks, decode loop, speculators); `crates/legacy/*` = SDK, registry/downloader, chat-template/tool-call handling (`hanashi`), streaming (`nagare`), CLI/server (`cli`), Apple power telemetry (`keisoku`) |
| Vendor claim (not verified) | Speculative decoding (DFlash + Weaver) gives Qwen3.6-27B 105 tok/s on an M5 Max, "2.9× the fastest MLX speculative decoding" (a Mirai post found by web search) |

## 2. Execution model (from the source)

- **One forward pass = one command buffer, encoded fresh every step.** `src/engine/language_model/stream/stream.rs`
  (~700 lines) calls `create_command_buffer` each step and replaces the allocation pool before each pass. Nothing like
  a CUDA graph is reused.
- **CPU/GPU overlap:** `generate()` encodes and submits pass t+1, then `resolve`s pass t. When every drafted token was
  accepted, the sampled token is copied GPU→GPU into the next pass's input, so there is no host round trip.
- **Prefill:** chunks of at most 1,024 tokens (`let max_batch_size = 1024;`, "required for attention correctness"), all
  chunks in one command buffer; the final token can be split into its own chunk so its row is the one sampled.
- **Decode with a speculator:** each pass runs a flattened token trie (`TrieNode` → `GpuTrieNode`), samples every node
  on the GPU, reads back one `u32` per node, walks the trie on the CPU (`trie.rs::accept`) and then encodes
  `encode_accept(accepted_indices)` for every mixer and for the drafter. Rejected nodes are never committed; there is
  no explicit rollback.
- **Model state** (`src/engine/language_model/state.rs`): token list, PRNG, `TransformerState` (per-layer mixer
  states), optional `DFlashState`. Context capacity is fixed at creation; with Metal sparse buffers the full maximum is
  reserved virtually (`recommended_context_length`), otherwise it is capped at 8,192 (iOS) / 16,384.

## 3. Techniques, by area (source-backed)

### 3.1 Speculative decoding: DFlash drafter, Weaver tree builder, trie verification

- **DFlash drafter** (`src/encodable_block/dflash.rs`, `src/speculators/dflash_tfm.rs`, `src/config/dflash.rs`),
  following DFlash (arXiv 2602.06036, ICML 2026):
  - Its input block is `[last target token, mask_token_id × (d−1)]`, run through its own attention layers in **one
    forward pass**, producing logits for d−1 future positions. It uses the target's embedding and readout (no own
    vocabulary or head).
  - Conditioning: hidden states of the target at `target_layer_ids` (config) are concatenated per token, projected
    (`target_feature_projection`: `model_dim × n_layers → model_dim`), normalized, then `state_kv_projection` turns
    them into **K/V rows appended to every drafter layer's cache**, only for accepted tokens (`encode_accept`).
  - Paper (Qwen3-4B/8B dense, H200): 5 drafter layers, 5 target layers from layer 2 to third-from-last, block 16, τ ≈
    6.5 at T=0 and ≈ 5.5 at T=1, 4–5× speedup; Qwen3-Coder-30B-A3B in SGLang 2.6–3.5× at concurrency 1.
- **Weaver** (`src/encodable_block/weaver.rs`, `weaver_layer.rs`, `src/config/weaver.rs`): a small learned transformer
  of its own (`num_layers`, `num_heads`, `candidate_pool_size`, `max_depth`) that takes the target's hidden state and
  DFlash's draft hidden states and logits, keeps a top-k candidate pool per draft row (`RadixTopKSmall`), and grows a
  tree over `rounds` (round 0 the root, then `expand_per_round` nodes per round, `expand_width` children each), with
  attention over the prefix KV and each node's ancestors (`ancestor_attention`). Everything runs in one command
  buffer; the CPU reads back the packed tree. Default shape (#763): `tree_budget` 16, `max_tree_depth` 16, Weaver
  16 rounds × 4 × 4. Shapes are chosen **per device** from `shapes.json`.
- **Trie** (`src/trie.rs`): preorder flattening with **subtree ranges** (node j descends from i iff
  `start_i ≤ j ≤ end_i`); `prune_to_budget` keeps the `budget` nodes with the highest cumulative log-probability;
  `accept` walks from the root, following the child whose token equals the target's sample at each node
  ("sample, then match"), and returns the matched path plus the bonus token.
- **Lossless stochastic sampling:** every node's seed is `prng.derive(prefix_length + index)`, i.e. a function of the
  absolute position only, so the target's sample at a position is the same whatever was drafted: the output equals
  seeded plain decoding.
- Pipelining ideas tried and closed: #984 "accept behind the draft and submit the verify in chunks of 8 layers"
  (72.8 → 70.8 ms per pass, closed: "will be superseded by megakernel work"); #985 reading only the draft
  vocabulary's rows of the target readout (65.3 → 62.3 ms per pass).

### 3.2 Hybrid models: Gated DeltaNet speculation without state snapshots (the most relevant piece)

- `src/encodable_block/mixer/delta_net.rs` (~650–700 lines): recurrent state in **f32**, `[num_heads, 128, 128]`
  (head dims 128 enforced), conv state `[conv_dim, kernel−1]`. One token: `conv_update` + `delta_net_update` (norm and
  gate fused). Prompt: tiled `conv_pack`/`conv_scan`, then a chunked prefill (`gdn/chunked/*.metal`, Gram matrix and
  triangular inverse) or a plain one.
- **Verify reads the committed state and writes nothing to it.** For a tree batch, `conv_tree_scan` computes per-node
  conv states from parent pointers into scratch, `delta_net_tree_prep` writes per-node normalized q/k, v, β, log-decay
  into scratch, and `tree_verify.metal` (~185 lines) computes every node's output from `h0` in **matrix (WY/chunk)
  form**: one threadgroup per value head, Gram matrices of keys/queries weighted by decay and β, `(I + A)` inverted by
  `invert_lower_triangular_block`, ancestor masking from the subtree ranges, output RMS-norm and SiLU gate fused. At
  most 16 nodes (NB=1) or 32 (NB=2). Only the output is written to memory.
- **Commit = replay of the accepted path** (`encode_accept`): check that the accepted indices form a parent chain, copy
  the last accepted node's conv state, then `state_advance.metal` (~80 lines) loads the committed state into
  registers once, applies for each accepted token `state *= decay; kv_mem = state·k; delta = β(v − kv_mem);
  state += k ⊗ delta`, and writes it back once.
- So uzu needs **no per-token state copies at all**: per in-flight token it keeps only k, v, β, decay and the conv
  window. Caveat from the PR (#609) review: the tree-prep kernel stored normalized K in BF16, so the replayed state
  differed from plain decoding's; whether that was fixed is not visible.

### 3.3 Attention and KV cache

- `src/encodable_block/mixer/attention/state.rs`: the KV buffer is the context plus a fixed **suffix region**
  (`ATTENTION_SUFFIX_CAPACITY = 1024`); drafted nodes are written there and accepted rows are **compacted** to the
  start of the suffix by row copies (`encode_accept`); sliding-window layers use a ring (`offset`, `length`,
  `capacity`).
- INT8 KV when the kernel supports it (#943, merged 2026-10-15): scales in key-tile order, grouped attention that packs
  the GQA query heads with 1–4 suffix rows so each K/V tile is read once, split-K capped at 16 splits, parallel combine.
  #979: grouped BF16 attention on M1–M4, "3–12 % faster kernel wise". No quality numbers.
- Tree verification uses `ancestor_attention.metal` with the same subtree-range mask.

### 3.4 Matrix-vector products and weight formats

- **GEMV** (`backends/metal/kernel/matmul/gemv/gemv.metal`, `common/quant_*.h`, `policy/quantized.rs`):
  - formats: 4-bit with scale+bias or scale+zero-point, 8-bit symmetric, group 32 or 64 (MLX-style affine), bf16/f32;
  - per element: `extract_bits` → `float_with_offset(code) − origin` (the magic-number int→float trick), then f32 FMA
    with the activation; per group, `scale·Σ(q·x) + bias·Σx` (`fold`/`finish`): the scale and bias are applied once per
    group, not per element;
  - **1–8 activation rows per kernel** (`INPUT_ROW_TILE`), so a 2–4-token verify batch decodes each weight once;
  - launch shape (simdgroups, rows per simdgroup, K-split) chosen from **hand-tuned tables** by GPU core count, GPU
    family and buckets of K and N; m 1–4 have tuned tiles, m 5–8 only for small N;
  - epilogue fuses bias, Hadamard output factors, soft-cap and gather indices.
- **Trellis (QTIP-style) weights** (`gemm/common/quantized/trellis.h`, `activation_transform/trellis.metal`,
  PRs #808, #833, #864, #925, #1004): 16-bit states read from a bit tape every T bits, value = a **computed hash**
  (`code_state * HASH_MULTIPLIER + HASH_INCREMENT`) folded into a sum of 2-bit fields (no stored codebook), V = 1/2/4
  weights per state, ~1.66–1.89 bpw at T6/T7 (#925), restart tapes every 64 columns. My count from the V=4 path: ~5–6
  integer operations per weight before the multiply. Used with **incoherence processing**: random ±1 sign vectors and a
  Hadamard transform on the input (and output) of each linear (`config/weight_matrix/hybrid_spec.rs`,
  `encodable_block/linear/rht_wrapper.rs`, `input_rht.rs`).
- **D4S4 embeddings** (`config/weight_matrix/d4s4_spec.rs`): one D4-lattice code per 4 columns and a "ladder" scale per
  64 columns, lookup-only.
- Fused QKV+gate projection for gated attention (#699): +10 % decode on Apple.

### 3.5 Mixture of experts (absent from `main`)

- `src/encodable_block/mlp/mod.rs`: `AnyMLPConfig::MixtureOfExpertsConfig(_) =>
  Err(MlpBlockNewError::UnsupportedMixtureOfExperts)`.
- Open drafts for Qwen3.5/3.6-35B-A3B (M5 Pro): f32 router + top-k (k ≤ 8), an **expert-offset GEMV** (pointers offset
  by `expert_id × stride`), SwiGLU, fixed-order combine (#999: 77.5 tok/s, weights-only ceiling 118); a 4-bit LM head
  (the bf16 head was 43 % of bytes per token: 78.2 → 97.9 tok/s) plus **the shared expert as one extra route of the
  expert GEMV** and **the shared-expert gate row folded into the router matvec** (257 rows) (#1009: → 112.5); router
  top-k across 256 threads (#1018: +2 %; fusing router GEMV and top-k into one threadgroup dropped to 86.7); a fused
  **down projection + weighted combine for M ≥ 2 verify tokens** (#1020). The input is rotated (Hadamard) **once per
  MoE layer and shared by all experts** (#999). Nobody has implemented reading each expert once for all verify tokens
  that share it; a reviewer asked about it (#1000).
- Issue #791 (expert paging, trace-based): with 90 % of expert pages resident only 16.1 % of generated tokens avoid a
  page-in; at 30 % none do. On-demand expert paging stalls almost every token.

### 3.6 Sampling and grammar

- `backends/metal/kernel/sampling/unified_sampling.metal`: one 1,024-thread threadgroup per row, **Gumbel-max with
  Philox** (seed per row), **no sort**: take the Gumbel argmax, check top-k (count of logits above it), top-p (softmax
  mass above it) and min-p, step to the next candidate if one fails, at most 64 iterations; the noise is replayed per
  iteration. The source notes the top-k + top-p combination is not exactly the reference's. Output: one `u32` per row.
- Grammar: the CPU computes a **1-bit-per-token mask** (xgrammar rewritten in Rust, #993), per trie node with grammar
  rollback between siblings (`fill_bitmasks`), uploads it, and the GPU sampler sets masked logits to −∞.

### 3.7 Agent sessions and prefix reuse

- `apps/uzu-pi-extension/src/cache.ts`: one resident model and one native chat session; a turn continues the session
  if the stored history (JSON per message) is a prefix of the new one and system prompt, tools and reasoning settings
  are unchanged; only the new messages are sent, so the model's own sampled tokens are never re-tokenized. Otherwise
  the session is reset and fully re-prefilled. Tools run in pi (`withToolTurnLimit(0)`).
- #805 (closed, superseded by #806): Qwen tool turns lost the whole prefix when a sampled tool call was re-rendered with
  different JSON spacing; with the fix, 184 ms vs 14,365 ms to first token on a 6,021-token history (M3 Ultra,
  Qwen3.6-27B). #826 (closed as "a qwen3.5 specific hack") appended only the rendered suffix to the sampled-token ledger.

### 3.8 Loading, memory and benchmarking

- Loading: `model.safetensors`, `assert_all_tensors_read()`. Draft #980 memory-maps the file and uses weights in place
  (unified memory: 7.8 → 0.66 GB peak). Metal 4 sparse heaps reserve the context virtually.
- `unit/BENCHMARKS.md`, `unit/common/cold_pool.rs`: Criterion kernel benchmarks; GEMV benchmarks **rotate copies of
  the weights over a 256 MiB working set** so kernels are ranked on DRAM reads, not cache-warm data. Every kernel has a
  CPU reference and a parity test.
- `scripts/benchmarks/` (#860): a cross-engine harness (MLX, mlx-serve, uzu, llama.cpp, …) that reports decode tok/s,
  duration and memory per single request. No multi-turn or agentic metric.

## 4. Technique → our rig

"v2" = Miner Llama v2.0 and its branches; "upstream" = llama.cpp at the fork base.

| Technique (uzu source) | Transfers to P106 sm_61, 5 SMs, x1, Celeron? | Already in upstream / v2? | Expected effect on agentic speed | Effort |
|---|---|---|---|---|
| GDN verify from committed state + **replay of the accepted path** (`delta_net.rs`, `gdn/tree_verify/state_advance.metal`) | **Yes.** Plain f32 recurrence; the replay must reuse the decode kernel's arithmetic so states stay bit-identical | **No.** Upstream keeps `n_rs_seq` full snapshot planes (`src/llama-memory-recurrent.cpp:101, 200-218`); the hybrid-model branch (Q162) keeps one state group per pipe slot: 9 × ~144 MiB = 1.3 GB on Ornith at depth 3 | Enabler: 2 state copies + a ring of per-token k/v/β/decay/conv rows (est. ~3 MB per in-flight token) instead of 9 groups frees est. ~1.0 GB (~110 MiB per card), the memory [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md) says pipelined speculation lacks. The speed gain is pipelined speculation's (Flint 1.30–1.55×; Ornith unmeasured) | Medium (1–2 weeks: replay kernel, record ring, Q162 executor, exactness test) |
| Multi-row matvec, 1–8 tokens per weight decode (`gemv.metal` `INPUT_ROW_TILE`, tuned m 1–4); fused MoE down+combine for M ≥ 2 (#1020) | Yes: weights stay bandwidth-bound up to ~8 rows with dp4a (est.) | Upstream MMVQ handles ≤ 8 columns and `mul_mat_vec_q_moe` one token per warp; **v2's Pascal kernels are one-token only**, so MTP verify falls back | Lets the MTP head pay on v2 (stock gains +22 % from it, v2 gains nothing today): est. +15–25 % generation on Ornith with the head | Medium (extend the v2 Pascal kernels; the per-type expert kernels of v3 item 1 should take M from the start) |
| 4-bit LM head (#1009: head 43 % of bytes on 35B-A3B) | Yes, as a file choice | Possible today (Q4_0/Q5_K/Q6_K kernels exist) | Ornith's Q8_0 head (~0.8 GB/token, est. ~12 % of bytes) in Q5_K/Q6_K: est. +2–4 %, frees ~0.25–0.4 GB on the last card (where the MTP layer sits) | Low (requantize, check KLD) |
| Fold the shared-expert gate row into the router matvec (257 rows); shared expert as an extra route (#1009) | Yes | The extra-route part is in upstream's `mul_mat_vec_q_moe` (`shared_expert` block) and v2's fusion; the gate fold is not | est. ≤ 1 % (one tiny launch per layer; CUDA graphs already hide most launch cost) | Low |
| Rotate the MoE input once per layer, shared by all experts (#999, `rht_wrapper.rs`) | Yes: a 3,072-point Hadamard per layer is negligible | v2 has a Hadamard line for Flint files only | Only useful together with a new rotated low-bit expert format (quality per bit for dp4a-friendly linear codes) | High (quantizer + kernel + quality work) |
| Trellis/QTIP codes with computed codebooks, D4 lattice codes | **No.** ~5–6 integer ops per weight (est. from `trellis.h`), and the hash needs a 32-bit multiply (3 XMADs on sm_61); a P106 has a budget of ~1.5–2 instructions per weight at 2–2.5 bpw (est.: ~1 T instr/s against ~0.7 T weights/s) | Already rejected in `docs/research/notes/breakthrough_candidates.md` | Would be compute-bound at est. ≤ 30–40 % of bandwidth | — |
| Float-dequant GEMV (`float_with_offset` + f32 FMA, `scale·Σqx + bias·Σx`) | Partly: the fold is llama.cpp's Q4_1/Q8_1 sum trick; dp4a on Q8_1 activations is better on Pascal than f32 FMA per weight | Yes (MMVQ, v2 kernels) | none | — |
| Per-device tuned launch tables (`policy/quantized.rs`) | Yes | v2's 5-SM launch shape | none new | — |
| DFlash block drafter (one pass drafts d−1 tokens from mask tokens and target features) | As a **guesser for pipelined speculation**, maybe: one pass could feed several pipe slots and amortize the 5.3–7.3 ms per-guess cost that caps the Flint drafter (`docs/drafter.md`). As a 16-token tree verify, no: wide verify batches are what a 5-SM card does worst, and on Ornith a 16-token verify would touch est. ~101 of 256 experts per layer (~15.7 GB) | Upstream has `draft-dflash` (#22105) and DFlash2, using `n_rs_seq` snapshots for hybrids (`dense_27b_speedups.md`) | Unknown; needs an offline replay (Q64-style) of a block head against the staircase head at equal drafter cost | High (training data under the publishable-data rules, teacher dumps) |
| Weaver learned tree builder, trie with subtree ranges, budget pruning by path probability | Low fit: pipelined speculation runs one chain; trees need wide verify batches | No | none expected | — |
| Position-derived seeds + Gumbel-max, sample-then-match acceptance | Already the principle of v2's exact pipelined speculation | Yes (v2) | none | — |
| Sort-free GPU top-k/top-p/min-p via candidate walk | Yes | Upstream backend sampling (`-bs`) and v2's GPU sampling exist | none measurable | — |
| Grammar as an uploaded 1-bit token mask (31 KB/token for 248k vocab) | Upload is cheap over x1 (est. 0.16 ms); computing the mask needs an xgrammar/llguidance-class engine on a 2-core Celeron | v2 samples on the GPU, checks on the host, resamples on rejection (`fork_scope.md` row "GPU sampling with a grammar") | ~0 (v2 already closed Q55's tool-request gap) | — |
| Exact-token session continuation for agents (pi extension `cache.ts`; #805/#826) | Yes | **Not needed now:** our Ornith agent runs show cache paths "exact" 23–35 per run, "reply re-read" ≤ 1, "older checkpoint" 0 ([the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md)) | none now; the fix to reach for if a template or agent change breaks it | — |
| INT8 KV + GQA-packed decode attention reading each K/V tile once for all q heads and 1–4 rows (#943, #979) | Probably: Ornith has 16 q heads per KV head | Upstream has q8_0 KV and GQA handling in its FA kernels; v2's compute-6.1 prompt attention kernel is f16-cache | Attention is 6 % of an Ornith token at 16k and grows linearly with context: check its share at 32–48k first; q8_0 KV would also halve the 24 KiB/token cache | Low to check, medium to build |
| KV suffix region + compaction on accept | Not needed for chain speculation (`seq_rm` suffices) | Equivalent in upstream | none | — |
| Fresh command buffer per step, encode t+1 before reading t, token fed GPU→GPU | Already better: CUDA graphs; device-fed token is Q121 | Yes / planned | none | — |
| Weights used in place from mmap (#980), sparse KV heaps | No: discrete cards need the PCIe copy; the 6-minute Ornith load is a host-side mmap problem uzu doesn't touch | — | none | — |
| Expert paging trace (#791) | As evidence: offloading or paging experts would stall nearly every token | — | rules out a direction | — |
| Cold-cache kernel benchmarks (256 MiB rotating weight copies, `cold_pool.rs`) | Yes. A GP106 has a 1.5 MB L2, and an Ornith expert matrix at 2–3 bits is ~0.4–1.2 MB (est.), so a naive microbenchmark can run from L2 | Not known whether our microbenchmarks rotate | Methodology: keeps the v3 expert-kernel GB/s honest | Low |
| CPU reference twin + parity test per kernel | Yes | `test-backend-ops` does this | none new | — |

## 5. Ranked shortlist for v3

Ornith-1.5-122B first:
1. **Pascal expert kernels that take 1–8 tokens** (items 1, 4 and 5 of [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md) together). uzu has no expert kernel
   we can reuse; its lessons are design ones: decode each weight once for all M rows (`INPUT_ROW_TILE`), fuse down +
   combine for M ≥ 2 (#1020), and pick formats whose decode fits Pascal's ALU budget (not trellis, not lookup-heavy
   codes). Its estimate is ~13 tok/s plain from 8.8, and it would make the MTP head pay on v2. High
   effort, largest gain.
2. **Recurrent rollback by accepted-path replay** (uzu's GDN commit) instead of nine state groups: frees est. ~1 GB
   across the cards so pipelined speculation fits beside the 2–3-bit file. Medium effort; the speed comes from
   pipelined speculation (Flint 1.30–1.55×).
3. **Head in Q5_K/Q6_K** in the Pascal-shaped Ornith file (#1009's lesson): est. +2–4 % and ~0.3 GB on the last card.
   Low effort.
4. **Check attention at 32–48k context**, then decide on q8_0 KV with a GQA-packed Pascal decode-attention path.
   Low to check.
5. **Cold-pool microbenchmarks** for the new expert kernels. Low effort.
6. Router + shared-gate fold: ≤ 1 %, only if it falls out of the persistent-kernel work.

Flint second:
1. **DFlash-style block guesser for pipelined speculation:** simulate offline first (acceptance by distance from the
   newest complete token, at equal drafter cost against the staircase head). Research item for v2.1/v3; training only
   on publishable sources (the DFlash paper used Nemotron post-training data and CodeAlpaca: licences to check).
2. Nothing else in uzu beats what v2 already has for an attention-only MoE: uzu has no MoE on `main`, and its MoE
   drafts trail v2's persistent kernel and shared-expert fusion.

## 6. What does not transfer, and why

- **Apple-specific execution:** MXU/cooperative-vector GEMM, simdgroup MMA, Metal 4 sparse heaps, function-constant
  specialization, unified-memory in-place weights, IOReport power telemetry.
- **ALU-heavy weight codes** (trellis/QTIP with hashed codebooks, D4 lattice): Apple GPUs have ALU to spare per byte;
  a 5-SM P106 is issue-bound at 2–3 bpw and has no fast 32-bit multiply.
- **Float dequantization per weight:** Pascal's dp4a on Q8_1 activations (llama.cpp's MMVQ, v2) does 4 MACs per
  instruction.
- **Wide tree verification (16-node DFlash + Weaver trees):** on MoE every extra verify token reads more experts, and
  wide batches are slow on 5 SMs; v2 uses idle cards (pipelined speculation) instead.
- **Command-buffer-per-step execution:** CUDA graphs are already ahead; uzu's open discussion points to a megakernel.
- **Prefix-cache fixes:** our agent runs already reuse the cache exactly.
- **MoE engine:** not on `main`, top-k ≤ 8, Qwen3.5-35B-A3B shapes only, no expert reuse across tokens.

## 7. Security observations

- No text in the files, PRs and issues I read tries to instruct an AI agent. Text aimed at agents: `scripts/benchmarks/AGENTS.md` asks coding
  agents to keep four benchmark model files (C++, Python, Zig, Rust) in sync; it's harmless. PR threads carry Codex/Copilot bot
  boilerplate ("@codex review"). Issue #523 is an AI-generated kernel audit, closed as not planned.
- Repo code that runs automatically if someone opened the repo locally (I did neither): `.envrc` (direnv loads a Nix flake dev shell
  after `direnv allow`); `.cargo/config.toml` alias `cargo tools` runs the workspace's `cli-tools` crate (the README
  recommends `cargo tools setup`); the wasm test runner is `wasmtime ... --dir /::/`, which gives the guest the whole
  host filesystem; build scripts only call `xcrun metal` (`crates/uzu-engine/build/metal/toolchain.rs`, no downloads
  seen).
- Release CI (`.github/workflows/release-uzu-pi-extension.yml`): actions pinned by mutable tags, `pnpm install` without
  `--ignore-scripts` in the build job, `NPM_TOKEN` only in the publish job. These are supply-chain notes for anyone
  installing `@trymirai/*` packages. They don't affect us.
- The SDK (`crates/legacy/uzu/src/registry/mirai/*`, `helpers/reachability.rs`) talks to Mirai's API and Hugging Face
  for model listing and downloads. I did not read whether it also sends telemetry.

## 8. Not verified

- None of uzu's speed claims (all are author-measured on M3 Ultra/M5 Pro/M5 Max; the 105 tok/s DFlash + Weaver figure
  is a vendor post).
- Whether uzu's replayed GDN state is now bit-identical to plain decoding (the #609 review found a BF16 K mismatch).
- The Weaver's selection kernels (`weaver_selection.metal`, `generated/weaver.h`: not read), the meaning of
  `prune_sigma`/prune noise, and how Mirai trains the DFlash and Weaver heads (that is in `lalamo`, not read).
- Trellis quality per bit for Mirai's XS/S/M models. Only fragments appear in PR bodies.
- The final design of the MoE drafts (#999–#1020 are open), and whether expert reuse across verify tokens ever lands.
- My Pascal estimates (instruction budget per weight, replay memory, attention growth, MTP gain with multi-row kernels)
  are arithmetic, not rig measurements.

## Sources

- Repo and API: https://github.com/trymirai/uzu, https://api.github.com/repos/trymirai/uzu
- Engine files cited above under `crates/uzu-engine/` (raw views at `https://raw.githubusercontent.com/trymirai/uzu/main/...`)
- PRs/issues: #545, #609, #699, #763, #791, #805, #806, #826, #841, #860, #864, #886, #925, #943, #979, #980, #984,
  #985, #999, #1000, #1004, #1009, #1018, #1020 (https://github.com/trymirai/uzu/pull/N or /issues/N)
- DFlash: https://arxiv.org/abs/2602.06036
- This repository: [the Ornith-1.5-122B benchmark](../../benchmarks/2026-10-27-ornith-122b.md),
  [model_branches.md](model_branches.md), [pipelined_speculation.md](pipelined_speculation.md),
  [breakthrough_candidates.md](breakthrough_candidates.md), [dense_27b_speedups.md](dense_27b_speedups.md),
  [fork_scope.md](fork_scope.md), [the drafter process](../../drafter.md),
  [bench/agent_check](../../../bench/agent_check/README.md)
- Miner Llama's fork, branch `pascal-p106-v3`: `src/llama-memory-recurrent.cpp`, `ggml/src/ggml-cuda/mmvq.cu`
