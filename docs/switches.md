# Miner Llama's switches

Every behaviour that Miner Llama adds to upstream llama.cpp (b12076) can be turned on or off on its own, at run time, in
one binary. Each switch's default is the engine's own behaviour: the features that are on stay on, and the experiments
that are off stay off. Setting a switch to its other value gives upstream's path for that one change, or, for a
tuning, the fork's earlier path. This lets each change, and each small gain, be measured on its own on the rig.

How to use them:
- Environment variables are read once, when the process first needs them. Set them on the command line, for example
  `GGML_CUDA_MMVQ_2WARPS=0 llama-server ...`.
- Flags (`--pipe-depth`, `--ctx-checkpoints-vram`, ...) are llama-server or tool arguments.
- "Compute 6.1" is the P106-090's architecture. Kernels marked so run only there unless their switch says `2` (any
  NVIDIA GPU, for tests).

The **Branch** column is the fork branch that brought the change: `base` is `pascal-p106-v3-b12076`, the v1/v2 engine
replayed onto upstream b12076, and the others are its `pascal-p106-v3-b12076-*` feature branches (`-flags` adds the
switches marked *new*). The **Row** column is the registry row in [quants/experiments.md](../quants/experiments.md)
where the change was proposed and measured.

Every switch below is in one build, `pascal-p106-v3-b12076-ladder5` (the v3 release build). With every switch at its
default it gives `-ladder4`'s logits bit for bit on every architecture `test-llama-archs` builds, and the sm_61 code of
every kernel the two share is the same but the persistent MoE kernel's F16 router branch. The checks behind that, and
what each switch does when flipped on a CPU, are in [docs/data/2026-11-05_ladder5/](data/2026-11-05_ladder5/README.md)
and [docs/data/2026-11-05_switches/](data/2026-11-05_switches/README.md).

The v4 line carries every switch at the same default on a newer upstream: `pascal-p106-v4-b12246-ladder6` is the
release tree with upstream b12246 merged in ([docs/data/2026-11-10_ladder6/](data/2026-11-10_ladder6/README.md)). Where
upstream has since merged a change that a switch describes, the row says "v4:" and what the switch's other value now
gives; [what upstream changed by default](#what-upstream-changed-by-default-on-the-v4-line) lists the behaviour that
came with the merge and has no switch.

The switches of the night of 2026-11-10/11, one fork branch each (`pascal-p106-v4-b12246-<row>`, or `-<row>`, in the
**Branch** column), are all in one build, `pascal-p106-v4-b12246-night`, the merge of those 17 branches on
`pascal-p106-v4`; none is in a release build yet. With every switch at its
default it gives `pascal-p106-v4`'s logits bit for bit on every architecture `test-llama-archs` builds, and each
branch's own check passes on it with its switch on
([docs/data/2026-11-11_v4_night_merge/](data/2026-11-11_v4_night_merge/README.md)).

## Which preset sets each switch

Each v3 model has a preset, `release/<model>-v3/settings.env`: the switches (`PRESET_ENV`) and the command line
(`PRESET_ARGS`) its agent check ran with. A switch no preset names stays at its default. These are the switches the
presets set:

| Switch | [Flash-Next](../release/flash-next-v3/settings.env) | [Flint](../release/flint-v3/settings.env) | [Ornith-1.5-122B](../release/ornith-v3/settings.env) |
|---|---|---|---|
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM` | `32768` | — | — |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE` | `0` | — | — |
| `LLAMA_HOP_CAST` | `q8_0` | — (no effect: Flint's graph does not call it, Q1125) | — (no effect: Ornith's graph does not call it, Q1125) |
| `GGML_CUDA_HOP_STAGED` | `1` | — (not run in v3.0) | — |
| `LLAMA_DRAFT_SHORTLIST` | `32768,8192,2` | — | — |
| `GGML_CUDA_KEEPALIVE` | `decode` | — (not run in v3.0) | `decode` (16.3 against 14.9 tok/s, [`valid1-ka`](data/2026-11-05_release_regression/README.md#the-agent-check)) |
| `LLAMA_SERVER_TEXT_REUSE`, `LLAMA_SERVER_TAIL_CKPT_MIN` | — | — (negligible: 2.59 against 2.60 minutes a task at equal work, the first token 2.25 against 2.64 s, median; [`reread`](data/2026-11-06_flint_side_rig2/README.md)) | — |
| `LLAMA_MOE_TAIL` | — (negligible at p = 0.9: −2.9 % a task, inside the spread of two runs, one task failed; [`t1skip`](data/2026-11-05_release_flashnext/README.md#the-agent-check)) | — (KL divergence above noise at p = 0.9, [T5b](data/2026-11-05_night_quant_chain/README.md#results)) | — (KL divergence above noise at p = 0.9, [T5b](data/2026-11-05_night_quant_chain/README.md#results)) |
| The negligible group (switches whose gain is within the spread of repeated loads, tested together) | — (a loss together: 19.05 against 20.75 tok/s on my Pascal mix of the file, +5.3 % a task; [`t1neg`](data/2026-11-05_release_flashnext/README.md#the-agent-check)) | — (not run in v3.0) | — |
| `--pipe-depth` | — | `4` | — |
| `--pipe-store` | — | — (not run in v3.0) | — |
| `-md`, `--spec-type draft-mtp` (upstream's) | the Gyro MTP draft | — | — (its own MTP layer as the draft is a loss: 3.42 against 2.52 minutes a task, [2026-11-06_ornith_mtp_rig2](data/2026-11-06_ornith_mtp_rig2/README.md)) |

Flash-Next's column is its preset, measured on the release build at 20.40 tok/s, 12 of 12
([2026-11-05_release_flashnext](data/2026-11-05_release_flashnext/README.md)).

## Matrix-vector kernels (token generation, speculative verify)

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_MMVQ_PASCAL` | `1` | The fork's matrix-vector kernels on compute 6.1 for 1–4 tokens (Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K, IQ4_XS, Q2_0, the grid types IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, and N2P/N3P), with FP32 activation sums. `0`: llama.cpp's `mul_mat_vec_q`; `2`: any NVIDIA GPU | base | Q76, Q78, Q101, Q149 |
| `GGML_CUDA_MMVQ_PASCAL_SKIP` | empty | Comma-separated ggml type names (`q2_0`, `iq2_s`, `n2p`, ...) that go back to llama.cpp's kernels, at every token count; `f16` also turns `GGML_CUDA_MMVQ_PASCAL_F16` off | base, -f16router | Q149, Q97 |
| `GGML_CUDA_MMVQ_PASCAL_MC` | `1` | `0`: products of 2 or more tokens go to llama.cpp's kernels; one token keeps the fork's | base | Q78 |
| `GGML_CUDA_MMVQ_PASCAL_STAGE_ACT` | `0` | `1`: the 2–4-token kernels stage the activations in shared memory | base | Q78 |
| `GGML_CUDA_MMVQ_PASCAL_Q2_0` | `0,0,1` | Q2_0 launch shape `G,P,T`; `2` forces R = 2 at one token, the rule before Q612 | -ladder, -kcol | Q223, Q612 |
| `GGML_CUDA_MMVQ_PASCAL_NSM` | the device's | SM count the launch shapes are chosen for (to run the P106's shapes on another GPU) | -kcol | Q612 |
| `GGML_CUDA_MMVQ_PASCAL_MTOK` | `1,0,0,0,1` | `E,CAP,RB,G,T` of the multi-token expert kernel (2–8 tokens, each distinct expert read once). `E=0`: off, 2–4 tokens go to the 2–4-token kernels and 5–8 to llama.cpp's. `T=0`: direct loads for the routed Q2_0 down projection; `T=2`: that product declined, as before Q825 | -mtok, -kshape, -flags | Q647, Q825 |
| `GGML_CUDA_MMVQ_PASCAL_MT` (`_MT_RB`, `_MT_ACT_KB`) | `0` | `1`: 2–4-token kernels that read and decode each weight once for every token (IQ2_S, IQ3_XXS, N2P, N3P, Q8_0, Q5_0), for the products the multi-token expert and k-column kernels decline; a skipped expert slot (id -1) gets zeros | -mtpk | Q211 |
| `GGML_CUDA_MMVQ_PASCAL_F16` | `0` | `1`: an F16 matrix (a MoE router kept in F16) at one token through the Pascal F16 kernel, and the persistent MoE kernel's router rows summed in the same order | -f16router | Q97 |
| `GGML_CUDA_MMVQ_MAX_COLS` | unset | `N`: products of more than N tokens leave llama.cpp's `mul_mat_vec_q` for MMQ (to compare the two at 5–8 tokens) | -cheapguess | Q194 |
| `GGML_CUDA_MMVQ_PASCAL_MTOK_SKIP` | empty | Types (grid types, IQ1_M, N2P/N3P) left out of the multi-token expert kernel | -iqkcol | Q713 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL` | `1` | k-column dense kernels for 1–8 tokens (each weight read and decoded once). `0`: off; `2`: on any GPU | -kcol | Q591, Q630, Q713 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` | empty | Types left out of the k-column kernels | -iqkcol | Q713 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_R`, `_ITT`, `_SMEM` | unset | Any of them set: the k-column launch-shape logic before Q823, with the value given (`_SMEM=49152` is the old default) | -kshape | Q823 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE` | `1` | `0`: no (type, row width, tokens) table sending some products to the other kernels, as before Q824 | -kshape | Q824 |
| `GGML_CUDA_MMVQ_PASCAL_Q5K_MT` (`_Q5K_MT_R`, `_Q5K_MT_PF`, `_Q5K_MT_NW`) | `0` | *v4, not yet in a release build.* `1`: Q5_K's plain products of 2–4 tokens (Flash-Next's output layer in a verify) leave the k-column kernel for a kernel that decodes each weight once for every token and gives each token the one-token kernel's sums bit for bit, at one row per lane group and without its register prefetch, the fastest setting on a P106 at 2–4 rows ([R75](data/2026-11-11_r75_c690_p106/README.md)); `2`: the same, with the prefetch off even under `_PF=1`; `_R=2`: two rows per lane group, which share the activation loads (default 1); `_PF=1` (with `1`): the register prefetch of each warp's next weights (default off); `_NW`: warps per block (default the most: 6 at R = 1, 3 at R = 2) ([design](data/2026-11-11_c690_head_four_rows/README.md)). Since `pascal-p106-v4-b12246-c766` ([C766](data/2026-11-11_c766_q5k_default/README.md)); before it, `1` meant R = 2 with the prefetch, the setting now reached with `=1 _R=2 _PF=1`, and `=2 _R=1` (what `1` gives now) keeps its meaning | `pascal-p106-v4-b12246-c690`, defaults `-c766` | Q1222, Q1347 |
| `GGML_CUDA_MMVQ_PASCAL_DMT` (`_DMT_R`, `_DMT_PF`, `_DMT_NW`, `_DMT_SKIP`) | `0` | *v4, not yet in a release build.* `1`: IQ4_XS's and Q4_K's plain products of 2–4 tokens (Flash-Next's attention and recurrent projections in a verify) leave the k-column kernel for a kernel that decodes each weight once for every token and gives each token the one-token kernel's sums bit for bit (C690's method). Since `-c758` it runs one row per lane group with its register prefetch, except IQ4_XS's 2-token products at K = 2,560, which run without the prefetch: on a P106 the fastest setting at every Flash-Next shape, or within the spread of it ([R79](data/2026-11-11_r79_c722_p106/README.md), [C758](data/2026-11-11_c758_dmt_default/README.md)). `2`: the same without the prefetch anywhere; `_R=2`: two rows per lane group, which share the activation loads (slower on a P106); `_R=0`: R chosen per product, the default before `-c758` (2 where its warps hold as many rows as 1's); `_PF=1`: the prefetch for every product, the setting R79 served as `DMT=1 DMT_R=1`; `_NW`: warps per block (default the most that fit: 6 at R = 1, 3 at R = 2); `_SKIP=q4_K` or `iq4_xs`: a type left to the k-column kernel ([design](data/2026-11-11_c722_short_batch_products/README.md)). On `-c722` to `-c748` (the night merge and the branches cut from it) the default is R by product, so there set `DMT=1 DMT_R=1` | `pascal-p106-v4-b12246-c722`; its default since `pascal-p106-v4-b12246-c758` | Q1277, Q1335 |
| `GGML_CUDA_MMVQ_PASCAL_Q3K_MT` (`_Q3K_MT_R`, `_Q3K_MT_NW`, `_Q3K_MT_PAD`, `_Q3K_MT_FAULT`) | `0` | *v4, not yet in a release build.* `1`: Q3_K's plain products of 2–8 tokens (Flash-Next's Q3_K attention and recurrent projections and the shared expert's Q3_K products in a verify) leave the k-column kernel for a kernel that decodes each weight once for every token, stages the tokens' activations once per block and keeps the k-column kernel's order, so each token's sums stay the one-token ones bit for bit (Q3_K's one-token kernel is the k-column kernel; C722's method); `2`: the same without its register prefetch; `_R`: rows per lane group, 1 (default, as R79 found for C722's kernel) or 2; `_NW`: warps per block (default the most that fit: 6 at R = 1, 3 at R = 2; a product where fewer than 2 fit stays on the k-column kernel unless `_NW=1`); `_PAD=0`: no padding of the weight staging's row slots (default: the padding that spreads the lanes' reads over the banks); `_FAULT=1`: each token takes the next token's activations, a fault the checks must catch. It follows `GGML_CUDA_MMVQ_PASCAL_KCOL` and `_KCOL_SKIP`: where the k-column kernel does not take Q3_K, neither does it ([design](data/2026-11-11_c726_q3k_multitoken/README.md)) | `pascal-p106-v4-b12246-c726` | Q1285 |
| `GGML_CUDA_MMVQ_PASCAL_GLU_MT` (`_GLU_MT_NP`, `_GLU_MT_FAULT`) | `0` | *v4, not yet in a release build.* `1`: a gate product, an up product of the same activations and their SWIGLU of 2–8 tokens (Flash-Next's shared expert in a verify: 41 of its 47 layers run them as three launches, and 33 of those mix the two weight types) run as one launch: each block stages the tokens' activations once, warp pairs compute the same rows of gate and up, each in its type's one-token order, and write `silu(gate) * up`, so each gate and up sum is the separate product's bit for bit and the SWIGLU the separate op's. It takes a pair only where both separate products run the kernel it repeats: Q2_0 and Q3_K at 2–8 tokens where the k-column kernels run (it follows `GGML_CUDA_MMVQ_PASCAL_KCOL`, `_KCOL_SKIP` and `GGML_CUDA_MMVQ_PASCAL_SKIP`), IQ4_XS and Q4_K at 2–4 only with `GGML_CUDA_MMVQ_PASCAL_DMT`'s kernel (set `_DMT_R=1`), Q5_K at 2–4 only with `GGML_CUDA_MMVQ_PASCAL_Q5K_MT`'s; other types (Q6_K) stay three launches. `2`: the same without its register prefetch; `_NP`: warp pairs per block, 1–4 (default the most that fit 48 KB); `_FAULT=1`: each token takes the next token's activations, a fault the checks must catch ([design](data/2026-11-11_c757_shexp_fused/README.md)) | `pascal-p106-v4-b12246-c757` | Q1333 |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_NSM`, `_KCOL_SMSM` | the device's | SM count and shared memory the k-column shapes are chosen for (other GPUs) | -kcol, -kshape | Q823 |
| `GGML_CUDA_MMVQ_2WARPS` | `1` | *new.* `0`: llama.cpp's `mul_mat_vec_q` at one token runs compute 6.1 code with the generic 4 warps per block, as upstream; `1`: 2 warps for rows that are not small-K | base, -flags | Q1025 |
| `GGML_CUDA_MMVQ_FP32_SUMS` | `1` | *new.* `0`: llama.cpp's `mul_mat_vec_q` for Q4_0, Q4_1, Q5_0, Q5_1 and IQ1_S reads the activation block sums from block_q8_1's FP16 field, as upstream; that sum overflows past 65,504 (llama.cpp #23606) | -kshape, -flags | Q916, Q1025 |
| `GGML_CUDA_MMVF_PASCAL` | `1` | The BF16 matrix-vector kernel on compute 6.1. `0`: off; `2`: any GPU | -ladder | Q546 |
| `GGML_CUDA_MMVF_PASCAL_RULE` | `2` | *new.* The BF16 kernel's launch-shape rule: `1` the rule before Q611 (no 3–4-token rule), `0` the first kernel's (at most 2 rows per lane group at 3–4 tokens) | -flags | Q611, Q1025 |
| `GGML_CUDA_MMVF_PASCAL_MAX_COLS` | `8` | *new.* The most tokens the BF16 kernel takes; `4` leaves 5–8 tokens to llama.cpp's path, as before C70 | -flags | Q591, Q1025 |
| `GGML_CUDA_MMVF_PASCAL_NSM`, `_SHAPE` | unset | SM count for the shapes; `G,R` forces one launch shape (tests) | -ladder | Q546 |
| `GGML_CUDA_MMVF_PASCAL_EXACT` (`_EXACT_FAULT`) | `0` | *v4, not yet in a release build.* `1`: BF16 products of 2–4 tokens add each row in the one-token order. On a P106-090 the 10,240-wide hyper-connection products (`hc_attn_down`, `hc_ffn_down`, `hc_head_down`) at 3–4 tokens keep the rule's 32 lanes, each lane running the one-token kernel's four 128-lane chains one after another; `2`: the same with the four chains interleaved in one pass (four times the accumulators); where the 3–4-token rule changes the lanes otherwise (cards of 10 SMs), the product takes its one-token shape. `_EXACT_FAULT=1` (checks only) leaves a quarter of each row out of the 32-lane kernel ([design](data/2026-11-11_c750_bf16_exact/README.md)) | `pascal-p106-v4-b12246-c750` | Q1325 |
| `GGML_CUDA_MMVF_F32_MT` (`_FAULT`) | `0` | *v4, not yet in a release build.* `1`: an F32 weight matrix times 4–8 tokens runs on llama.cpp's `mul_mat_vec_f` on compute 6.1, as it does at 1–3 tokens, instead of cuBLAS (Flash-Next's shared-expert gate and router in a verify pass); `mul_mat_vec_f` sums each token in its own chain at a block size set by the row width, so each token gets the one-token sums bit for bit; `2`: on any NVIDIA GPU. `_FAULT=1` (checks only) adds only the first half of each row of the products the switch takes ([design](data/2026-11-11_c754_f32_mmvf/README.md)) | `pascal-p106-v4-b12246-c754` | Q1330 |
| `GGML_CUDA_PERSISTENT_MOE` | `1` | One persistent kernel per MoE layer at one token, for the layer types it covers (Flint's). `0`: off; `2`: any GPU. `_CHECK=1` checks it against the per-op path; `_GRID`, `_FAULT` are test settings | base | Q88 |

## Prompt-processing kernels

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_MMQ_MOE_LEGACY` | `0` | `1`: MUL_MAT_ID in MMQ and MMF as before upstream PR #31047's tile list (tile width from all tokens, a tile column per expert); `2`: that layout only up to 8 tokens (speculative verify, multi-token steps) and the tile list for prompt batches (v2's prefill split). v4: upstream merged #31047, so `0` is upstream's own path and `1` the layout upstream had before b12246 | base, -flags, -prefill-split | Q108 |
| `GGML_CUDA_FATTN_TILE_PASCAL` | `1` | Prompt attention kernel on compute 6.1 (head size 128 or 256, reads a Q8_0 cache directly). `0`: llama.cpp's TILE kernel; `2`: any GPU; `_MIN`: the fewest query tokens it takes | base | Q163 |
| `GGML_CUDA_FATTN_TILE_PASCAL_QK` | `2` | `1`: the head-128 scores without the split over the head's halves | base | Q163 |

## Attention at decode

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_ATTN_BLOCK` | `1` | One token's attention block (Q/K/V products, norms, RoPE, cache writes) in two launches on compute 6.1. `0`: off; `2`: any GPU | base | Q152 |
| `GGML_CUDA_FATTN_PASCAL` | `0` | `1`: the Pascal decode-attention kernel (slower than llama.cpp's on the rig); `2`: any GPU | base, -fap | Q60, Q151 |
| `GGML_CUDA_FATTN_SPARSE_PASCAL` | `0` | `1`: sparse decode attention below Turing over the cells a QSA layer keeps; `2`: any GPU. `_NQ`, `_SPLITS`, `_NSM`, `_MIN_RATIO` tune it | -qsa | Q230, Q441 |
| `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT` | `0` | *new.* `1`: with the sparse attention on, prompt micro-batches (more than `_NQ` and at most `_PROMPT_NQ` query tokens, default 128) go to it too: in mode 1 once the cache holds `_PROMPT_MIN_RATIO` (default 1.0) × the kept cells, whatever the micro-batch's size; decode and verify calls keep `_MIN_RATIO`'s rule. `0`: every call of more than `_NQ` tokens stays on the dense kernels | v4: -c705 | Q1246 |
| `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION` | `0` | *new.* `1`: with the sparse attention on, a call of two or more query tokens (a verify, or a prompt micro-batch with `_PROMPT`) runs the union kernel: a block takes as many consecutive query tokens as 12 warps hold (Flash-Next: 4) and reads the union of their kept cells once, each token scoring with its own mask; mode 1 then takes a verify from `_UNION_MIN_RATIO` (default 1.5) × the kept cells per union list instead of per query token (a 4-token verify from 3,077 cells). The output equals the per-token kernel's to rounding | v4: -c103 | Q643 |
| `LLAMA_ATTN_K_LAST` | `1` | *new.* `0`: a sliding-window cache's layer expands q, k, v (upstream's order) instead of k last, which lets K's norm, RoPE and cache write fuse | base, -flags | Q1025 |
| `LLAMA_GPU_KQ_MASK` | `0` | `n`: micro-batches of at least n tokens build their attention mask on each card; `LLAMA_GPU_KQ_MASK_CHECK=1` compares it with the host's. v4: with it on, gemma4 keeps every row through its KV-shared layers (upstream's row pruning there reads the host masks) | base, -hybrid | Q110, Q116, Q1073 |
| `LLAMA_GPU_KQ_MASK_MAX` | unset | `m`: with `LLAMA_GPU_KQ_MASK`, card-built masks only for micro-batches of at most m tokens (decode steps); longer ones use the host's | -decodemask, -promptpath | Q261 |
| `LLAMA_GPU_KQ_CELLS` | `1` | *new.* `0`: no per-card copy of the cells' positions in the KV buffers (upstream's buffers; the card-built mask then stays off) | base, -flags | Q110, Q1025 |
| `LLAMA_QSA_PAD_LINEAR` | `1` | *new.* `0`: the QSA k-pool's pad cells found by sorting and binary search, as upstream (the same cells, more host time) | -qsa, -flags | Q441, Q1025 |
| `LLAMA_QSA_INPUT_STATS` | `0` | `N`: log the k-pool inputs every N micro-batches | -qsa | Q441 |
| `LLAMA_QSA_KPOOL_DEV` (`_CHECK`) | `0` | `1`: the QSA k-pool tables (`pool_cells`, `pool_idxs`) kept on each card, each micro-batch sending only the rows that changed instead of both tables whole; `2`: the MTP draft's context only. `LLAMA_QSA_KPOOL_DEV_CHECK=1` compares every micro-batch's copies and update with the host's tables and aborts on a difference ([design](data/2026-11-11_c100_kpool_on_cards/README.md)). On nine P106-090 cards exact and a gain at depth: the step 1.59 ms shorter at 16,384 tokens and 5.99 at 49,152 ([rig run](data/2026-11-11_r74_c100_p106/README.md)) | v4: `pascal-p106-v4-b12246-c100` | Q640, Q1273 |
| `LLAMA_QSA_POOL_MASK_DEV` (`_CHECK`) | `0` | `1`: the QSA pool mask built on each card (`ggml_range_mask`) from each token's range of visible pools, 8 bytes a token, instead of the host mask's upload, 2 bytes a pool and token; `2`: the MTP draft's context only. `LLAMA_QSA_POOL_MASK_DEV_CHECK=1` reads every card's mask back after each micro-batch, compares it with the host mask and aborts on a difference. Test switch: `GGML_CUDA_RANGE_MASK_FAULT=1` ([design](data/2026-11-11_c719_pool_mask_on_cards/README.md)) | v4: `pascal-p106-v4-b12246-c719` | Q1274 |
| `LLAMA_QSA_KPOOL_STAGE` | `0` | `1`, with `LLAMA_QSA_KPOOL_DEV`: the five small QSA k-pool inputs (`k_idxs`, `tail_idxs`, `new_pool_idxs`, `new_pool_rep`, `new_pool_pos`) go up inside the tables' update and are cast back on each card, two copies a card and micro-batch instead of seven; the same values ([design](data/2026-11-11_c720_kpool_small_inputs/README.md)) | v4: `pascal-p106-v4-b12246-c720` | Q1275 |
| `LLAMA_QSA_INDEXER_ROWS` | `0` | `1`: the QSA indexer (qwen4exp, and glm5-next with its fused indexer) reads each pool’s key in its cell of the cache by `pool_cells` instead of from a gathered F32 copy of every pooled key; the same scores bit for bit ([design](data/2026-11-11_c704_indexer_gather/README.md)) | v4: `pascal-p106-v4-b12246-c704` | Q1244 |
| `GGML_CUDA_LID_ROWS_FAULT` | `0` | `1` (checks only): an indexer with a list of rows scores rows i mod n_cells instead of the listed ones (0 to n−1 while the list is no longer than the cache), so the recorded check must fail on every case ([C749](data/2026-11-11_c749_lid_rows_fault/README.md)) | v4: `pascal-p106-v4-b12246-c704`; any list since `pascal-p106-v4-b12246-c749` | Q1244, Q1324 |

## Fusions and numerics

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_NORM_Q8` | `1` | The RMS norm of one row writes the Q8_1 activations and absorbs the residual add. `0`: off (v2.0's release candidate hit a CUDA-pool assertion with it off) | base | Q100 |
| `GGML_CUDA_GLU_ALIAS_FUSE` | `1` | *new.* `0`: a gate/up/GLU whose output reuses the activations' memory (a shared expert, a dense FFN) runs unfused, as upstream | base, -flags | Q50, Q1025 |
| `GGML_CUDA_GDN_FUSE` | `1` | The delta-net layer's fusions: the GDN state read in place, the conv window, the gated norm. `0`: off | -gdnfuse | Q437 |
| `LLAMA_GDN_FUSE` | `1` | `0`: qwen4exp's graph without the GDN_CONV op (the unfused conv window and norms). v4: the unfused window stores its history as upstream does since #31660, one write for all rollback slots | -gdnfuse | Q438 |
| `LLAMA_NO_MOE_DOWN_RESCALE` | unset | Unset: the MoE down-projection rescale is dropped at one token on compute 6.1 (FP32 sums make it unnecessary). `0`: kept everywhere (upstream); `tg`: dropped at one token; any other value: dropped everywhere | base | Q76 |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | Upstream's switch; `1` turns CUDA graphs off, which upstream does for compute 6.1 | base | v1 |

## Graphs, scheduler and memory

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `LLAMA_GRAPH_PER_SIZE` | `1` | One graph arena per verify width up to 8 tokens. `0`: the two arenas of before | -vgraphs | Q701 |
| `LLAMA_SCHED_PER_SIZE` | `0` | `1`: a scheduler per verify width | -vgraphs | Q701 |
| `LLAMA_GRAPH_META_CALLOC` | `1` | `0`: the graph metadata buffer touched page by page, as before | -vgraphs | Q701 |
| `GGML_SCHED_CTX_PER_GRAPH` | `1` | *new.* `0`: one scheduler context buffer sized for the largest graph, made at creation, as upstream | -vgraphs, -flags | Q701, Q1025 |
| `GGML_SCHED_PREFETCH` | `1` | *v4, not yet in a release build.* `1`: before a split runs, the scheduler starts copying to host memory the inputs of the next split that this split's card already holds, so that a CPU split after it waits for those copies and not for the card's whole split (upstream #30966). `0`: each split's inputs are copied when that split runs, as before b12246. Acts only on a card (a backend with asynchronous reads and events). With `LLAMA_UBATCH_TRACE` a scheduler logs its first early fetch and, when freed, the count ([checks](data/2026-11-11_c693_upstream_switches/README.md)) | v4: `pascal-p106-v4-b12246-c693` | Q1226 |
| `LLAMA_GRAPH_REUSE_WAIT` | `0` | `1`: with pipeline parallelism, reuse a graph after a full wait instead of only when nothing is in flight | base | Q87 |
| `LLAMA_SAMPLER_RESERVE` | `0` | `1`: a scheduler reserve on every backend-sampler attach and detach (upstream) | base | Q111 |
| `LLAMA_INPUT_ON_GPU` | `0` | `1`: the token-embedding lookup on the first card | base | Q150 |
| `LLAMA_INPUT_UPLOAD` | `0` | `1`: one packed input upload per card and token in plain decoding | base | Q120, Q150 |
| `LLAMA_KV_PAD` | `256` | Padding of the KV cells a step attends to | base | Q153 |
| `LLAMA_KV_HEAD_EMPTY` | `1` | *new.* `0`: a sequence removal that empties a stream keeps upstream's head rule instead of restarting at cell 0 | base, -flags | Q111, Q1025 |
| `LLAMA_KV_HEAD_TAIL` | `1` | *new.* `0`: removing a sequence's tail keeps upstream's head rule instead of putting head on the first removed cell (pipelined speculation needs this rule for exact results) | base, -flags | Q87, Q1025 |
| `LLAMA_KV_HEAD_FREE` | `1` | *new.* `0`: the slot search restarts from cell 0 as upstream, even when head is on a free cell | base, -flags | Q87, Q1025 |
| `LLAMA_KV_RESTORE_CANONICAL` | `1` | *new.* `0`: a state restored into an empty stream is placed by the slot search (upstream) instead of position p in cell p | base, -flags | Q111, Q1025 |
| `LLAMA_KV_SEQ_RM_BOUND` | off | `1`: removing positions of one sequence from a KV cache (`seq_rm`: in every speculative step, the draft's tokens out of the draft context and the rejected ones out of both memories) visits only the cells it can change, read from the sequence's index of (position, cell) in cell order, instead of every cell of the cache (65,536 at the presets' context); removing positions of every sequence visits the used range. The same cells, head, changed-cell list and QSA tables as without it; in llama-server's MTP path on a CPU at 65,536 tokens the two stages of a step that hold the calls take 0.0070 ms instead of 0.8270. Read when each cache is created. `2`: a test hook that leaves the cells unsorted, which `test-llama-archs --kv-seq-rm` must catch ([design](data/2026-11-11_c740_seq_rm_bound/README.md)) | v4: `pascal-p106-v4-b12246-c740` | Q1309, Q1344 |
| `LLAMA_STATE_NOINIT` | `1` | *new.* `0`: state buffers (checkpoints, prompt cache) are zero-filled when allocated, as upstream. v4: upstream merged #30137 and leaves them uninitialized too, so `0` is upstream before b12246 | -ckptrc, -flags | Q704, Q1025 |
| `LLAMA_DIO_STAGING` | `4x64` | Pinned staging of the direct-I/O loader, `<n>x<MiB>` | -draftmem | Q580 |
| `GGML_CUDA_KEEPALIVE` (`_MS`, `_POLL_US`) | off | `decode` (or `1`): one block per card, on a stream of the card's lowest priority, polls a word in mapped host memory so that the driver holds the card at its boost clock while a layer split decodes; it ends itself `_MS` after the card's last graph (default 1,000) and is ended before captures, frees and device-wide synchronizes. `request`: also every card's block when llama-server takes a request; `idle`: a 120 s timeout, through an agent's tool calls | -keepalive | Q1021, Q1045 |
| `GGML_CUDA_GRAPHS_FORCE` | off | `1`: a graph that works on more than 8 tokens (a prompt micro-batch, a draft's catch-up pass) is captured as a CUDA graph whenever its properties change, without the warm-up of two unchanged calls that pipeline parallelism never lets a prompt complete; replayed only where every property equals its capture's. The backend context's first such graph runs eagerly; decode and verify graphs keep the warm-up; nothing changes on a device without VMM, with `GGML_CUDA_GRAPH_OPT=1`, or in a graph with an `ARGSORT` over more than 1,024 columns and several rows ([design](data/2026-11-11_c702_prompt_graphs/README.md)) | v4: `pascal-p106-v4-b12246-c702` | Q1240 |
| `LLAMA_UBATCH_TAPER`, `--ubatch-taper` | off | A taper of prompt micro-batches (`on` or `head=N:S,tail=N:S,short=N:S,min=N`); a request may send `"ubatch_taper"`; with an MTP draft loaded no planned micro-batch holds fewer than `--spec-draft-n-max` + 2 tokens (Q1192) | -taper | Q771 |
| `LLAMA_NGRAM_PREFETCH` (`_MB`, `_STATS`) | off | *new.* `1`: the n-gram table's row reads (the `madvise(MADV_WILLNEED)` that the n-gram input issues on the rows of a table read from the disk, `--lazy-mode`: Flash-Next's 28.8 GB table) of a decode call's later micro-batches are issued by a helper thread while the earlier ones compute, instead of in the input's `set_input` just before card 0's launch; each call's first micro-batch issues its own as before. The helper stays at most `LLAMA_NGRAM_PREFETCH_MB` (default `48`) MiB of row pages ahead of the micro-batch the input has reached. The rows read are the same: only when the reads are issued changes. Planned calls: one sequence at consecutive positions, more than one micro-batch. `LLAMA_NGRAM_PREFETCH_STATS=1`: each call's rows, distinct pages and row digest counted (`llama_ngram_prefetch_stats`), the switch off too ([the checks and the rig row](data/2026-11-11_c761_ngram_prefetch/README.md)) | v4: `pascal-p106-v4-b12246-c761` | Q1339, Q1351 |

## MoE routing and card hops

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `LLAMA_MOE_TAIL` | off | `p=..,w=..,renorm=..` drops the router's low-weight tail of experts; as one op whose dropped slots the expert kernels skip (`graph=1`: the graph-level ops; `skip`, `repeat`, `hist`) | -bytesw, -moeskip | Q618, Q660 |
| `GGML_CUDA_SKIP_IDS_ANY` | `1` | *new.* `0`: only `LLAMA_MOE_TAIL`'s op gets the handling for -1 expert ids; every other ids tensor is read as it is, as upstream | -moeskip, -flags | Q660, Q1025 |
| `LLAMA_MOE_ROUTE_ONCE` | off | `k[,max=n][,hist=..][,dry=..]`: blocks of k positions take their first token's experts | -routeonce | Q860 |
| `LLAMA_HEAD_BOUND` | off | `k[,m]`: the main head over a shortlist with an exact bound and a full-head fallback (`_FILE`) | -bytesw | Q619 |
| `LLAMA_HOP_CAST`, `LLAMA_HOP_CAST_AT` | off | The residual cast at each card boundary of a layer split: `f16`, `bf16`, `q8_0`, `q6_k`, `q5_1`, `q5_0`, `q4_1`, `q4_0`, `iq4_nl` (qwen4exp) | -hops, -hopcast4 | Q231, Q750, Q963 |
| `GGML_CUDA_CPY_HOP_TYPES` | `1` | *new.* `0`: CUDA does not claim the copies IQ4_NL → F32, F32 → Q6_K and Q6_K → F32, so they run on the CPU as upstream | -hopcast4, -flags | Q963, Q1025 |
| `GGML_CUDA_HOP_STAGED` | `0` | `1`: a hop between cards without peer access goes through a pinned host ring (`_CHUNKS`, `_SLOTS`, `_SLOT_KB`, `_STREAM`); `2`: every hop | -hops | Q751 |
| `GGML_CUDA_HOP_MAILBOX` | `0` | `1`: a hop between cards through a mailbox in mapped host memory (`_CHUNKS`, `_SLOTS`, `_SLOT_KB`, `_MAX_KB`, `_STREAM`, `_THREADS`, `_POLL_US`, `_TIMEOUT_MS`); `2`: pairs with peer access too | -mailbox | Q682 |

## Sampling, drafts and speculation

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `LLAMA_GRAMMAR_HOST_SAMPLING` | `0` | `1`: a request with a grammar turns device sampling off, as upstream (default: the device's token is checked on the host) | base | Q67 |
| `LLAMA_GRAMMAR_TEST_REJECT` | `0` | `N`: count every Nth grammar-checked token as rejected (tests) | base | Q67 |
| `LLAMA_SCREEN_HEAD` (`_M`, `_FILE`) | `0` | The screened output head (files with a screen) | base | Q92, Q125 |
| `LLAMA_SCREEN_HEAD_GRAMMAR` | `0` | `1`: a request with a grammar keeps the screened head until the grammar can reject a token (a lazy grammar before its trigger, the reasoning); `0`: any grammar turns the screen off | -toolsample-screen | Q360 |
| `LLAMA_REASONING_BUDGET_UNMASK` | `1` | *new.* `0`: the reasoning budget's forced token keeps a -inf logit, as upstream (the screened head gives -inf to tokens outside its screen) | base, -flags | Q92, Q1025 |
| `--pipe-depth`, `--pipe-guess`, `--pipe-copy-min`, `--pipe-floor`, `--pipe-store`, `--pipe-draft` | off | Pipelined speculation for a slot that generates alone, its guessers, filler floor, continuation store and staircase drafter | base | Q87, Q154, Q160, Q161, Q20 |
| `LLAMA_PIPE_UPLOAD`, `LLAMA_PIPE_CHUNK_STREAM` | packed, off | The pipe slots' packed uploads; their prompt chunks on lower-priority streams | base | Q120, Q155 |
| `LLAMA_DRAFT_HEAD`, `LLAMA_DRAFT_VOCAB`, `LLAMA_DRAFT_ANC`, `LLAMA_DRAFT_SPLIT`, `LLAMA_DRAFT_PRUNE` | unset | The staircase drafter's cheaper guess: its output head converted at load (`q8_0`, `f16`), its vocabulary cut to the first k draft ids, the ancestor selection as one dense product (`dense`), each step's matrix products in groups of k rows, the last block on the output row only (`PRUNE=1`, exact) | -cheapguess | Q194 |
| `LLAMA_DRAFT_LOAD_CHUNK` | `8` | MiB per read when the staircase drafter's file loads and its head converts (the same bytes, a small peak on a 4 GB host); `0`: each tensor at once | -cheapguess | Q194 |
| `LLAMA_PIPE_STAIR_CONT` | `0` | `1`: `--pipe-guess staircase` without a store guesses the request's continuations of 2-token contexts before the drafter | -cheapguess | Q194 |
| `LLAMA_PIPE_DRAFT_TAPS_ONLY` | `0` | `1`: the staircase drafter follows the sequence but never guesses (what following costs) | -cheapguess | Q194 |
| `LLAMA_DRAFT_SHARE` | off | `embd[,head]`: the MTP draft reads the target's token embeddings and head | -draftmem | Q564 |
| `LLAMA_DRAFT_SHORTLIST` (`_FILE`) | off | `k[,m[,r]]`: the MTP draft's head over a token shortlist | -draftmem | Q592, Q635 |
| `LLAMA_DRAFT_EMBD_CACHE` | off | `<rows>[,<max tokens>]`: the draft's token rows cached on its layer's card | -draft1card | Q961 |
| `LLAMA_DRAFT_H_CARRY` | off | `1`: the draft's hidden rows stay on its card between passes. v4: the rows the draft reads from the target stay on the target's output card by default, upstream's #31927 (`LLAMA_MTP_H_DEV`); this switch still covers the draft's own passes | -draft1card | Q961 |
| `LLAMA_MTP_H_DEV` | `1` | *v4, not yet in a release build.* Where the target keeps the hidden rows that an MTP draft with its own memory (Flash-Next's) reads. `1`: in a buffer on the target's output card, one row per batch position, read there by the draft's graph, wherever that card has events (upstream #31927; est. 80.0 MiB for Flash-Next at `-b 2048`). `0`: copied to host memory after each decode, as before b12246 (the v3 release's path); `LLAMA_MTP_CATCHUP_OVERLAP` needs the buffer and stays off. `2`: as `1`, and the CPU takes the buffer too (checks). The draft's caches and drafts are the same bytes with the rows on the host and in the buffer on a CPU ([checks](data/2026-11-11_c693_upstream_switches/README.md)) | v4: `pascal-p106-v4-b12246-c693` | Q1226 |
| `LLAMA_FIT_DRAFT_CTX`, `LLAMA_FIT_NEXTN_SKIP` | `1` | `0`: `--fit` measures a draft alone and leaves skipped NextN layers out, as b12076 (upstream #31610, #31569). v4: upstream merged #31569, so `LLAMA_FIT_NEXTN_SKIP=1` is upstream's own count | -ckptrc | Q706 |
| `LLAMA_SPEC_ADAPT` (`_COST`, `_S`, `_WIDTHS`, `_LOG`, ...) | off | `1`: the draft length chosen per step from the drafter's probabilities and a verify-cost table | -adraft | Q687 |
| `LLAMA_SPEC_BLOCK_VERIFY` | off | `1`: sampled drafts verified as a block | -blockver | Q797 |
| `LLAMA_SPEC_FORCE` (`_CHARS`, `_MAX`), `LLAMA_SPEC_DRAFT_KEEP` | off | The text a grammar forces appended to the draft | -blockver | Q802 |
| `LLAMA_SPEC_KEEP_DRAWS` | off | `1`: token-by-token rejection keeps its draws across a checkpoint replay | -blockver | Q808 |
| `LLAMA_SPEC_REJECT_STOP` | off | `1`: token-by-token rejection (`--spec-draft-sampling probabilistic` above temperature 0) stops after an accepted end-of-generation token that is not the last drafted one, and where the screened head meets a grammar that has become active, as sample-and-match does; `0`: upstream's path, which verifies the drafted tokens behind the end of a turn ([design](data/2026-11-11_c675_end_of_turn_stop/README.md)). With `LLAMA_SPEC_BLOCK_VERIFY=1` it also ends the verified block at its first drafted end-of-generation token before the last drafted one, and a block so cut that is accepted whole takes no next token ([design](data/2026-11-11_c711_block_verify_stop/README.md)) | v4: `pascal-p106-v4-b12246-c675`; the block cut on `-c711` | Q1187, Q1263 |
| `LLAMA_MTP_KV_ONLY` | off | `1`: an MTP draft pass with no output row (the catch-up over a prompt chunk or a verify step's tokens) writes only the draft layer's caches (K and V, and on a QSA layer its indexer and pooled keys) and skips its queries, attention, MoE and head; a pass with a `dist` backend sampler keeps the whole layer. `qwen4exp` (Flash-Next) only; the same caches and drafts as off on the CPU ([design](data/2026-11-11_c620_draft_kv_catchup/README.md)). With `LLAMA_QSA_KPOOL_DEV` the cut pass keeps the k-pool tables on the cards current from `-c748` on; before it, the next whole pass sends both tables whole ([C748](data/2026-11-11_c748_kvonly_kpool/README.md)) | v4: `pascal-p106-v4-b12246-c620`; with the tables on the cards `pascal-p106-v4-b12246-c748` | Q1105, Q1320 |
| `LLAMA_VERIFY_PREPARE` | off | *new.* `1`: llama-server prepares a speculative step's verify pass while draft-mtp's last drafting pass runs (`llama_decode_prepare`: the batch split, the memory's slots applied, the graph chosen, every input that reads no token id set), and the step's decode only puts in the last drafted id, sets the inputs that read ids and launches; when the step's batch is one slot's sampled token and whole draft and nothing before the decode reads or changes the target context. `2`: also copies card 0's graph inputs then, and the evaluation skips each one whose bytes are unchanged. The same replies, draft counts and verify steps as off on the CPU ([design](data/2026-11-11_c738_verify_prepare/README.md)) | v4: `pascal-p106-v4-b12246-c738` | Q1307 |
| `LLAMA_DRAFT_PREFETCH` (`_KB`, `_STATS`) | off | *new.* `1`: the n-gram table's rows (Flash-Next's table read from the disk, `--lazy-mode`) of each token a speculative step's verify pass will read are issued (`llama_draft_prefetch`: `madvise(MADV_WILLNEED)` through `llama_prefetch_rows`) as soon as the drafter has the token: draft-mtp issues `id_last`'s once its first drafting pass is queued, each drafted token's once the next pass is queued and the last one's after the last pass; the other drafters issue a finished draft's. The verify pass's n-gram input then finds them in the page cache instead of reading them just before card 0. The rows read are the same: only when the reads are issued changes. At most `LLAMA_DRAFT_PREFETCH_KB` (default `1024`) KiB of pages are issued between two evaluations of the table's rows. `LLAMA_DRAFT_PREFETCH_STATS=1` (or `LLAMA_UBATCH_TRACE`): a `draft-prefetch:` line per evaluation (its rows, those issued ahead, their pages and how many were in the page cache) and a closing `C615:` line, the switch off too ([design and the rig row](data/2026-11-11_c615_draft_prefetch/README.md)) | v4: `pascal-p106-v4-b12246-c615` | Q1100, Q1354 |
| `LLAMA_SERVER_YIELD` | `0` | *new.* `1`: a generation step of llama-server runs its draft and its catch-up without a yield to the task queue (the hand-over to the thread that answers `/metrics` and `/slots` while the server computes), so a speculative step yields once, around the verify pass, instead of three times; a prompt call's catch-up keeps its yield. `2`: also, a yield wakes that thread only when a read is waiting or comes during it. Completions and slot changes wait for the step's end either way; a read that comes during the draft is answered at the verify pass's yield or the step's end. The same replies, drafts and probabilities as off on the CPU, with one and two slots ([design](data/2026-11-11_c741_server_yields/README.md)) | v4: `pascal-p106-v4-b12246-c741` | Q1310 |
| `LLAMA_SAMPLING_TOP_K_ROWS` | off | `1`: when every backend sampling chain (`-bs`) starts with a top-k (after samplers that do nothing at their settings, as the presets' penalties), one `TOP_K` over all the output rows per k and one gather of their logits replace each chain's own, and the chains read the logits without their padded copy (`PAD`); every row gets the same candidates and logits. Read when a context is made, so the MTP draft's one-row chains read the logits unpadded too. On nine P106-090 cards a gain: the verify pass's chain 0.323, 0.314, 0.317 and 0.316 ms against 0.650, 0.655, 0.651 and 0.651, the warm step 121.05 and 121.26 ms against 121.78 and 121.98 ([R83](data/2026-11-11_r83_c691_p106/README.md); [design](data/2026-11-11_c691_sampling_launches/README.md)) | v4: `pascal-p106-v4-b12246-c691` | Q1223, Q1300 |
| `LLAMA_MTP_CATCHUP_OVERLAP` | off | `1`: llama-server's MTP draft catches up on a prompt call (one without an output row) once the next call is queued on the target, not before it, so that the draft's passes run while the next call starts through the cards; the target keeps each call's hidden rows in a second half of its row buffer meanwhile (80.0 MiB more on the output card in the Flash-Next preset). The draft's caches, its drafts and the server's checkpoints are the ones the immediate catch-up gives (the same bytes on the CPU, [design](data/2026-11-11_c703_catchup_overlap/README.md)); llama-server only. With `--ctx-checkpoints-incall` from `-c735` on: the waiting pass takes its call's cuts and fills, and the checkpoints' draft parts are read after it; before it the server lost its in-call checkpoints ([C735](data/2026-11-11_c735_c217_c703/README.md)) | v4: `pascal-p106-v4-b12246-c703`; with `--ctx-checkpoints-incall` `pascal-p106-v4-b12246-c735` | Q1242, Q1302 |
| `LLAMA_MTP_GRAPH_KEEP` | off | `1`: an MTP draft context keeps one graph on a scheduler of its own for each batch of up to 8 tokens that differs in its tokens, its outputs, its head's shortlist or its hidden-row carry (`LLAMA_SCHED_PER_SIZE`'s schedulers, for that context only), so that a speculative step's catch-up and its first drafting pass stop rebuilding each other's graph; a per-size scheduler waits only for its own graph in flight. The same drafts and replies on the CPU, 0.030 draft graph builds a step instead of 2.035 in llama-server; each scheduler adds compute buffers on the draft's cards ([design](data/2026-11-11_c619_draft_graph_reuse/README.md)) | v4: `pascal-p106-v4-b12246-c619` | Q1104, Q1334 |
| `LLAMA_DRAFT_EMBD_DEV` | off | *new.* `layer` (or `1`): a draft file that holds only its MTP layers (Flash-Next's MTP draft) keeps its token embeddings on the device of its MTP layer, ahead of the input layer's host buffer and of any `-otd` rule for `token_embd` (the v3 preset's `-otd token_embd=CUDA4`), so each draft pass gathers its token rows on that card and only the token ids cross to it, instead of the F32 rows from card 4 through the host (1.31 MB a 128-row catch-up pass). The same rows, drafts and replies as off on a CPU with two devices laid out as cards 4 and 8; the Flash-Next preset at 65,536 tokens: card 8 +644.14 MiB (est. 777 MiB left), card 4 −645.39 MiB; at 196,608 tokens card 8 is est. 51 MiB over ([design](data/2026-11-11_c762_draft_rows/README.md)) | v4: `pascal-p106-v4-b12246-c762` | Q1340, Q1349 |

## Server checkpoints and prompt cache

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `--ctx-checkpoints-vram N`, `--ctx-checkpoints-vram-sync` | `0`, off | A slot's newest N checkpoints in device stores | -ckpt | Q440, Q548 |
| `--ctx-checkpoints-spread N` | `0` | Checkpoints inside long prompts, at most N tokens apart | -ckpt | Q448 |
| `--ctx-checkpoints-incall` | off | With `--ctx-checkpoints-vram`: a checkpoint is taken inside the prompt's decode call instead of cutting the call there, its state copied into the device store behind the micro-batch that ends at it, with no wait for the cards; calls are joined only where the joined call keeps the cut calls' micro-batches, so the checkpoints and every number stay the same ([design](data/2026-11-11_c217_incall_checkpoints/README.md)). With `LLAMA_MTP_CATCHUP_OVERLAP` from `-c735` on ([C735](data/2026-11-11_c735_c217_c703/README.md)) | v4: `pascal-p106-v4-b12246-c217`; with `LLAMA_MTP_CATCHUP_OVERLAP` `pascal-p106-v4-b12246-c735` | Q770, Q1302 |
| `--ctx-checkpoints-share` | off | Prompt-cache entries and child slots share the slot's host checkpoints (upstream #30137). v4: upstream merged #30137 and always shares; Miner Llama keeps the flag and copies without it, as the release does (following upstream's default is Q1227) | -ckptrc | Q704 |
| `--prefix-cache DIR` (`_MAX`, `_MIN`), `--checkpoint-gen` (`_AT`, `_MIN`) | off | Saved shared prefixes and generation checkpoints for agent sessions | base | Q111, Q132 |
| `--checkpoint-tail2` | off | Two tail checkpoints per prompt and the mid-prompt rule, as stock, instead of one ring checkpoint for a sliding-window model | base | Q111 |
| `LLAMA_SERVER_TEXT_REUSE` | `0` | `1`: llama-server keeps the cached tokens where a prompt spells the same text with other tokens (a reply sent back and tokenized its own way) | -textreuse, -promptpath | Q260 |
| `LLAMA_SERVER_TAIL_CKPT_MIN` | `0` | `n`: a prompt's end-of-prompt checkpoint only when it ends at least n tokens past the slot's newest checkpoint | -promptpath | Q263 |
| `LLAMA_SERVER_CKPT_RING` | `1` | *new.* `0`: a sliding-window model's checkpoints hold only the last window, as stock, not the whole ring | base, -flags | Q111, Q1025 |
| `LLAMA_SERVER_CKPT_RING_PAST` | `1` | *new.* `0`: no ring checkpoint taken after the divergence is restored, as stock | base, -flags | Q111, Q1025 |
| `LLAMA_SERVER_CKPT_RESUME_NEXT` | `1` | *new.* `0`: a sliding-window checkpoint restore resumes at its last position, as stock, not after it | base, -flags | Q111, Q1025 |

## Diagnostics

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_PROFILE=<file>` | off | Times every launch group with CUDA events; `GGML_CUDA_PROFILE_MODE` = `group` (default), `layer`, `sample=<N>`, `graph` | base, -lightprof | Q54, Q681 |
| `GGML_CUDA_PROFILE_HOST=<file>` | off | Host and GPU time per graph evaluation | base | Q49 |
| `LLAMA_UBATCH_TRACE` | off | `1`: a log line per decode call, per micro-batch (graph reused or rebuilt and why, the host's parts and its time before the first split reaches a card), per scheduler evaluation (each split's wait, copy and launch time on the host) and per re-plan of the compute buffers with its trigger; llama-server's phase averages every 5 s. `2`: llama-server also waits for the cards after each decode (changes the schedule) ([the lines](data/2026-11-11_c701_microbatch_trace/README.md)). On `-c761` each micro-batch's line also splits `inputs` per graph input (`inputs_by`) and gives the n-gram input's own prefetch (`ngram`), and each call of a model with a mapped n-gram table gets an `ngram-prefetch` line ([the fields](data/2026-11-11_c761_ngram_prefetch/README.md#how-it-works)) | v4: `pascal-p106-v4-b12246-c701`, `-c761` | Q1238, Q1339 |
| `LLAMA_STEP_TRACE=<file>` | off | *new.* llama-server writes one JSON line per step with host timestamps of the server's, the MTP drafter's and both contexts' stages (the draft's passes, the batch, the verify pass's decode stages, every wait for the cards, the catch-up, the acceptance), on the clock `GGML_CUDA_PROFILE_HOST`'s log line names, so the two files go side by side ([design](data/2026-11-11_c405_server_gap/README.md)). From `-c741` on each line also counts the step's yields to the task queue and wakes of its worker (`yld`, `wake`) and marks completions and reads posted during the step and reads answered in a yield (`q.cmpl`, `q.read`, `q.take`; [C741](data/2026-11-11_c741_server_yields/README.md)) | v4: `pascal-p106-v4-b12246-c405`; the counts `-c741` | Q960, Q1310 |
| `LLAMA_SPEC_DUMP=<file>` (`_K`, `_TGT_K`, `_NGRAM`, `_FULL_BLOCK`) | off | One JSON line per speculative verify step | -specdump, -specdump2 | Q688, Q804 |
| `GGML_CUDA_NORM_Q8_DEBUG`, `GGML_CUDA_FATTN_*_FAULT`, `GGML_CUDA_PERSISTENT_MOE_FAULT` | off | Test hooks | base | — |
| `LLAMA_VERIFY_PREPARE_FAULT` | off | Test hook: a decode prepared with `llama_decode_prepare` leaves its stand-in token ids in the KV cells (`test-llama-archs --verify-prepare` must fail with it) | v4: `pascal-p106-v4-b12246-c738` | Q1307 |
| `LLAMA_TENSOR_HASH=<prefix>[,<prefix>...]` | off | Check hook: every graph node whose name starts with a prefix is hashed (FNV-1a over its bytes) once computed and logged at the library's INFO level (`-lv 4` in llama-server) with its type, shape and buffer (`tensor-hash: <v|d> <n> <name> ...`, `d` an MTP draft's context), so two runs' intermediate tensors can be compared whatever device computed them; unused when the caller sets an eval callback, and the scheduler computes each split in pieces ending at those nodes while it is set ([C762](data/2026-11-11_c762_draft_rows/README.md)) | v4: `pascal-p106-v4-b12246-c762` | Q1340 |

Tools: `llama-imatrix --span-mask` and `--span-dump` (Q897), `test-backend-ops --record` and `--replay` (Q931),
`llama-pipespec` and `llama-pipe-recurrent-check`.

## Build-time switches

`-DGGML_CUDA_PASCAL_FLAGS="<nvcc options>"` passes options to the Pascal kernel files only, and the `-D` macros below set
those kernels' compile-time tuning (Q671). The defaults are the engine's:

| Macro | Default | What it sets |
|---|---|---|
| `MMVQP_MIN_BLOCKS`, `MMVQP_MC_MIN_BLOCKS`, `KCOL_MIN_BLOCKS`, `MMVF_PASCAL_MIN_BLOCKS`, `FAP_MIN_BLOCKS`, `FTP_MIN_BLOCKS` | as built | The blocks per SM each Pascal kernel keeps registers for |
| `MMVF_PASCAL_X4_MIN_BLOCKS` | `4` | The same for `GGML_CUDA_MMVF_PASCAL_EXACT`'s 32-lane kernels (at most 128 registers, so a 10,240 × 320 product's 20 blocks fit one wave on 5 SMs) |
| `MMVQP_IQ_MIN_BLOCKS`, `MMVQP_IQ_MC_MIN_BLOCKS` | `3` | The same for the grid types; `1` is the setting before the P106 tuning round |
| `MMVQP_IQ_SIGN_LUT` | `0` | `1`: IQ2_S and IQ3_XXS read their sign masks from a shared-memory table instead of computing them |
| `MMVQP_IQ_R`, `MMVQP_IQ_PERSIST` | `1`, `0` | Rows per lane group and persistent row blocks of the one-token grid-type kernel |

The IQ2_S/IQ3_XXS tuning (sign masks in registers, 3 blocks per SM) has only this build-time switch; a run-time one is
Q1040.

## What has no switch

- **New quant types** (N2P, N3P, N2P-128, N3P-128, Q4_1_64) and **cold-tier tensors** (Q238) act only in files that hold
  them. Q4_1_64 (Q16, v2's group-64 Q4_1) has type id 504 on v3; files written by v2.0 carry its old id and do not
  load as Q4_1_64 (Q1037).
  Their Pascal kernels have the kernel switches above: `GGML_CUDA_MMVQ_PASCAL_SKIP=n2p,n3p,n2p_128,n3p_128` sends them
  to llama.cpp's `mul_mat_vec_q` with the fork's dot products.
- **Fixes whose other value would bring a crash or a wrong result back:** ggml-alloc plans again for a leaf that no node
  reads; MUL_MAT_ID's matrix-vector batch limit comes from the architecture the device's code was built for (v4:
  upstream's #31679 takes the highest architecture the build holds; the fork also bounds it by the architecture of the
  code the device loaded); the prompt cache drops an entry whose state was copied short; the KV layer's members are
  set by name; devices without a backend registry report no features. On v4 also the two recurrent-state fixes of
  Q1158: a new sequence's state is cleared with a fill, so a NaN cannot stay in a cell (upstream's open #31913), and a
  sequence that the slot finder moves to another cell takes its rollback depth along (upstream's #31656).
- **Output only:** the `COMPUTE_6_1` backend feature entry, the server's extra timing fields (`sampled_on_device`,
  `grammar_n`, `cache_src`, `ckpt_*`, ...) and the info level of its checkpoint log lines.
- **The -1 expert-id checks** compiled into llama.cpp's expert kernels: a comparison never taken for valid ids
  (`GGML_CUDA_SKIP_IDS_ANY` switches the host-side handling; what the comparisons cost at five and six tokens is Q1028).
- **Op numbering:** the fork's ops (`GGML_OP_GDN_CONV`, `GGML_OP_KQ_MASK`, `GGML_OP_MOE_TAIL`) change the op enum, so an
  RPC build carries the fork's own protocol version (upstream's major version plus 100: 108 on v3, 109 on v4).

## What upstream changed by default on the v4 line

The merge of upstream b12246 brings 154 upstream commits. These change what the engine does with no switch set, on
the models Miner Llama serves
([the full list and the checks](data/2026-11-10_ladder6/README.md#what-upstream-changed-by-default)). Two of them have
a switch on `pascal-p106-v4-b12246-c693`, named in the table; the others have none:

| Upstream change | What changes with no switch set |
|---|---|
| [#31927](https://github.com/ggml-org/llama.cpp/pull/31927) | An MTP draft with one head and its own memory (Flash-Next's) reads the target's hidden rows from a buffer on the target's output card instead of the host. The buffer holds one row per batch position: 80.0 MiB for Flash-Next at `-b 2048` (est. from the code: 2,049 rows of 10,240 F32 values). If it cannot be allocated the rows go through the host as before. `LLAMA_MTP_H_DEV=0` puts them back on the host |
| [#31660](https://github.com/ggml-org/llama.cpp/pull/31660) | Recurrent models with rollback snapshots store the conv history of all snapshots in one write (the fused GDN path, the default for qwen4exp, already did) |
| [#31419](https://github.com/ggml-org/llama.cpp/pull/31419), [#30966](https://github.com/ggml-org/llama.cpp/pull/30966) | The scheduler does not upload the inputs of nodes that are not computed, and fetches a CPU split's inputs from the previous card before that card's split runs. `GGML_SCHED_PREFETCH=0` turns the early fetch off |
| [#31084](https://github.com/ggml-org/llama.cpp/pull/31084), [#31803](https://github.com/ggml-org/llama.cpp/pull/31803), [#31005](https://github.com/ggml-org/llama.cpp/pull/31005) | CUDA kernels on the decode path: the top-k of device sampling picks tied candidates in a fixed order, the lightning indexer (Flash-Next's sparse attention) lays out its shared memory differently, the fused MoE router replaces a NaN score |
| [#31851](https://github.com/ggml-org/llama.cpp/pull/31851), [#31854](https://github.com/ggml-org/llama.cpp/pull/31854) | Attention without flash attention asks for BF16 accumulators instead of F32; the CPU backend multiplies F16 and BF16 matrices with F32 activations |
| [#31683](https://github.com/ggml-org/llama.cpp/pull/31683) | Context shift moves the KV cells correctly, and each model builds its own RoPE |
| [#31956](https://github.com/ggml-org/llama.cpp/pull/31956), [#28697](https://github.com/ggml-org/llama.cpp/pull/28697) | llama-server: a request with tools and a `response_format` gets a full grammar, not a lazy one; a `json_schema` not nested in its wrapper is accepted |
