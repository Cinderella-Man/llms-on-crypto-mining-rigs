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
| `GGML_CUDA_MMVQ_PASCAL_KCOL_NSM`, `_KCOL_SMSM` | the device's | SM count and shared memory the k-column shapes are chosen for (other GPUs) | -kcol, -kshape | Q823 |
| `GGML_CUDA_MMVQ_2WARPS` | `1` | *new.* `0`: llama.cpp's `mul_mat_vec_q` at one token runs compute 6.1 code with the generic 4 warps per block, as upstream; `1`: 2 warps for rows that are not small-K | base, -flags | Q1025 |
| `GGML_CUDA_MMVQ_FP32_SUMS` | `1` | *new.* `0`: llama.cpp's `mul_mat_vec_q` for Q4_0, Q4_1, Q5_0, Q5_1 and IQ1_S reads the activation block sums from block_q8_1's FP16 field, as upstream; that sum overflows past 65,504 (llama.cpp #23606) | -kshape, -flags | Q916, Q1025 |
| `GGML_CUDA_MMVF_PASCAL` | `1` | The BF16 matrix-vector kernel on compute 6.1. `0`: off; `2`: any GPU | -ladder | Q546 |
| `GGML_CUDA_MMVF_PASCAL_RULE` | `2` | *new.* The BF16 kernel's launch-shape rule: `1` the rule before Q611 (no 3–4-token rule), `0` the first kernel's (at most 2 rows per lane group at 3–4 tokens) | -flags | Q611, Q1025 |
| `GGML_CUDA_MMVF_PASCAL_MAX_COLS` | `8` | *new.* The most tokens the BF16 kernel takes; `4` leaves 5–8 tokens to llama.cpp's path, as before C70 | -flags | Q591, Q1025 |
| `GGML_CUDA_MMVF_PASCAL_NSM`, `_SHAPE` | unset | SM count for the shapes; `G,R` forces one launch shape (tests) | -ladder | Q546 |
| `GGML_CUDA_PERSISTENT_MOE` | `1` | One persistent kernel per MoE layer at one token, for the layer types it covers (Flint's). `0`: off; `2`: any GPU. `_CHECK=1` checks it against the per-op path; `_GRID`, `_FAULT` are test settings | base | Q88 |

## Prompt-processing kernels

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_MMQ_MOE_LEGACY` | `0` | `1`: MUL_MAT_ID in MMQ and MMF as before upstream PR #31047's tile list (tile width from all tokens, a tile column per expert); `2`: that layout only up to 8 tokens (speculative verify, multi-token steps) and the tile list for prompt batches (v2's prefill split) | base, -flags, -prefill-split | Q108 |
| `GGML_CUDA_FATTN_TILE_PASCAL` | `1` | Prompt attention kernel on compute 6.1 (head size 128 or 256, reads a Q8_0 cache directly). `0`: llama.cpp's TILE kernel; `2`: any GPU; `_MIN`: the fewest query tokens it takes | base | Q163 |
| `GGML_CUDA_FATTN_TILE_PASCAL_QK` | `2` | `1`: the head-128 scores without the split over the head's halves | base | Q163 |

## Attention at decode

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_ATTN_BLOCK` | `1` | One token's attention block (Q/K/V products, norms, RoPE, cache writes) in two launches on compute 6.1. `0`: off; `2`: any GPU | base | Q152 |
| `GGML_CUDA_FATTN_PASCAL` | `0` | `1`: the Pascal decode-attention kernel (slower than llama.cpp's on the rig); `2`: any GPU | base, -fap | Q60, Q151 |
| `GGML_CUDA_FATTN_SPARSE_PASCAL` | `0` | `1`: sparse decode attention below Turing over the cells a QSA layer keeps; `2`: any GPU. `_NQ`, `_SPLITS`, `_NSM`, `_MIN_RATIO` tune it | -qsa | Q230, Q441 |
| `LLAMA_ATTN_K_LAST` | `1` | *new.* `0`: a sliding-window cache's layer expands q, k, v (upstream's order) instead of k last, which lets K's norm, RoPE and cache write fuse | base, -flags | Q1025 |
| `LLAMA_GPU_KQ_MASK` | `0` | `n`: micro-batches of at least n tokens build their attention mask on each card; `LLAMA_GPU_KQ_MASK_CHECK=1` compares it with the host's | base, -hybrid | Q110, Q116 |
| `LLAMA_GPU_KQ_MASK_MAX` | unset | `m`: with `LLAMA_GPU_KQ_MASK`, card-built masks only for micro-batches of at most m tokens (decode steps); longer ones use the host's | -decodemask, -promptpath | Q261 |
| `LLAMA_GPU_KQ_CELLS` | `1` | *new.* `0`: no per-card copy of the cells' positions in the KV buffers (upstream's buffers; the card-built mask then stays off) | base, -flags | Q110, Q1025 |
| `LLAMA_QSA_PAD_LINEAR` | `1` | *new.* `0`: the QSA k-pool's pad cells found by sorting and binary search, as upstream (the same cells, more host time) | -qsa, -flags | Q441, Q1025 |
| `LLAMA_QSA_INPUT_STATS` | `0` | `N`: log the k-pool inputs every N micro-batches | -qsa | Q441 |

## Fusions and numerics

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `GGML_CUDA_NORM_Q8` | `1` | The RMS norm of one row writes the Q8_1 activations and absorbs the residual add. `0`: off (v2.0's release candidate hit a CUDA-pool assertion with it off) | base | Q100 |
| `GGML_CUDA_GLU_ALIAS_FUSE` | `1` | *new.* `0`: a gate/up/GLU whose output reuses the activations' memory (a shared expert, a dense FFN) runs unfused, as upstream | base, -flags | Q50, Q1025 |
| `GGML_CUDA_GDN_FUSE` | `1` | The delta-net layer's fusions: the GDN state read in place, the conv window, the gated norm. `0`: off | -gdnfuse | Q437 |
| `LLAMA_GDN_FUSE` | `1` | `0`: qwen4exp's graph without the GDN_CONV op (the unfused conv window and norms) | -gdnfuse | Q438 |
| `LLAMA_NO_MOE_DOWN_RESCALE` | unset | Unset: the MoE down-projection rescale is dropped at one token on compute 6.1 (FP32 sums make it unnecessary). `0`: kept everywhere (upstream); `tg`: dropped at one token; any other value: dropped everywhere | base | Q76 |
| `GGML_CUDA_DISABLE_GRAPHS` | unset | Upstream's switch; `1` turns CUDA graphs off, which upstream does for compute 6.1 | base | v1 |

## Graphs, scheduler and memory

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `LLAMA_GRAPH_PER_SIZE` | `1` | One graph arena per verify width up to 8 tokens. `0`: the two arenas of before | -vgraphs | Q701 |
| `LLAMA_SCHED_PER_SIZE` | `0` | `1`: a scheduler per verify width | -vgraphs | Q701 |
| `LLAMA_GRAPH_META_CALLOC` | `1` | `0`: the graph metadata buffer touched page by page, as before | -vgraphs | Q701 |
| `GGML_SCHED_CTX_PER_GRAPH` | `1` | *new.* `0`: one scheduler context buffer sized for the largest graph, made at creation, as upstream | -vgraphs, -flags | Q701, Q1025 |
| `LLAMA_GRAPH_REUSE_WAIT` | `0` | `1`: with pipeline parallelism, reuse a graph after a full wait instead of only when nothing is in flight | base | Q87 |
| `LLAMA_SAMPLER_RESERVE` | `0` | `1`: a scheduler reserve on every backend-sampler attach and detach (upstream) | base | Q111 |
| `LLAMA_INPUT_ON_GPU` | `0` | `1`: the token-embedding lookup on the first card | base | Q150 |
| `LLAMA_INPUT_UPLOAD` | `0` | `1`: one packed input upload per card and token in plain decoding | base | Q120, Q150 |
| `LLAMA_KV_PAD` | `256` | Padding of the KV cells a step attends to | base | Q153 |
| `LLAMA_KV_HEAD_EMPTY` | `1` | *new.* `0`: a sequence removal that empties a stream keeps upstream's head rule instead of restarting at cell 0 | base, -flags | Q111, Q1025 |
| `LLAMA_KV_HEAD_TAIL` | `1` | *new.* `0`: removing a sequence's tail keeps upstream's head rule instead of putting head on the first removed cell (pipelined speculation needs this rule for exact results) | base, -flags | Q87, Q1025 |
| `LLAMA_KV_HEAD_FREE` | `1` | *new.* `0`: the slot search restarts from cell 0 as upstream, even when head is on a free cell | base, -flags | Q87, Q1025 |
| `LLAMA_KV_RESTORE_CANONICAL` | `1` | *new.* `0`: a state restored into an empty stream is placed by the slot search (upstream) instead of position p in cell p | base, -flags | Q111, Q1025 |
| `LLAMA_STATE_NOINIT` | `1` | *new.* `0`: state buffers (checkpoints, prompt cache) are zero-filled when allocated, as upstream | -ckptrc, -flags | Q704, Q1025 |
| `LLAMA_DIO_STAGING` | `4x64` | Pinned staging of the direct-I/O loader, `<n>x<MiB>` | -draftmem | Q580 |
| `GGML_CUDA_KEEPALIVE` (`_MS`, `_POLL_US`) | off | `decode` (or `1`): one block per card, on a stream of the card's lowest priority, polls a word in mapped host memory so that the driver holds the card at its boost clock while a layer split decodes; it ends itself `_MS` after the card's last graph (default 1,000) and is ended before captures, frees and device-wide synchronizes. `request`: also every card's block when llama-server takes a request; `idle`: a 120 s timeout, through an agent's tool calls | -keepalive | Q1021, Q1045 |
| `LLAMA_UBATCH_TAPER`, `--ubatch-taper` | off | A taper of prompt micro-batches (`on` or `head=N:S,tail=N:S,short=N:S,min=N`); a request may send `"ubatch_taper"` | -taper | Q771 |

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
| `LLAMA_DRAFT_H_CARRY` | off | `1`: the draft's hidden rows stay on its card between passes | -draft1card | Q961 |
| `LLAMA_FIT_DRAFT_CTX`, `LLAMA_FIT_NEXTN_SKIP` | `1` | `0`: `--fit` measures a draft alone and leaves skipped NextN layers out, as b12076 (upstream #31610, #31569) | -ckptrc | Q706 |
| `LLAMA_SPEC_ADAPT` (`_COST`, `_S`, `_WIDTHS`, `_LOG`, ...) | off | `1`: the draft length chosen per step from the drafter's probabilities and a verify-cost table | -adraft | Q687 |
| `LLAMA_SPEC_BLOCK_VERIFY` | off | `1`: sampled drafts verified as a block | -blockver | Q797 |
| `LLAMA_SPEC_FORCE` (`_CHARS`, `_MAX`), `LLAMA_SPEC_DRAFT_KEEP` | off | The text a grammar forces appended to the draft | -blockver | Q802 |
| `LLAMA_SPEC_KEEP_DRAWS` | off | `1`: token-by-token rejection keeps its draws across a checkpoint replay | -blockver | Q808 |

## Server checkpoints and prompt cache

| Switch | Default | What it does | Branch | Row |
|---|---|---|---|---|
| `--ctx-checkpoints-vram N`, `--ctx-checkpoints-vram-sync` | `0`, off | A slot's newest N checkpoints in device stores | -ckpt | Q440, Q548 |
| `--ctx-checkpoints-spread N` | `0` | Checkpoints inside long prompts, at most N tokens apart | -ckpt | Q448 |
| `--ctx-checkpoints-share` | off | Prompt-cache entries and child slots share the slot's host checkpoints (upstream #30137) | -ckptrc | Q704 |
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
| `LLAMA_SPEC_DUMP=<file>` (`_K`, `_TGT_K`, `_NGRAM`, `_FULL_BLOCK`) | off | One JSON line per speculative verify step | -specdump, -specdump2 | Q688, Q804 |
| `GGML_CUDA_NORM_Q8_DEBUG`, `GGML_CUDA_FATTN_*_FAULT`, `GGML_CUDA_PERSISTENT_MOE_FAULT` | off | Test hooks | base | — |

Tools: `llama-imatrix --span-mask` and `--span-dump` (Q897), `test-backend-ops --record` and `--replay` (Q931),
`llama-pipespec` and `llama-pipe-recurrent-check`.

## Build-time switches

`-DGGML_CUDA_PASCAL_FLAGS="<nvcc options>"` passes options to the Pascal kernel files only, and the `-D` macros below set
those kernels' compile-time tuning (Q671). The defaults are the engine's:

| Macro | Default | What it sets |
|---|---|---|
| `MMVQP_MIN_BLOCKS`, `MMVQP_MC_MIN_BLOCKS`, `KCOL_MIN_BLOCKS`, `MMVF_PASCAL_MIN_BLOCKS`, `FAP_MIN_BLOCKS`, `FTP_MIN_BLOCKS` | as built | The blocks per SM each Pascal kernel keeps registers for |
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
  reads; MUL_MAT_ID's matrix-vector batch limit comes from the architecture the device's code was built for; the prompt
  cache drops an entry whose state was copied short; the KV layer's members are set by name; devices without a backend
  registry report no features.
- **Output only:** the `COMPUTE_6_1` backend feature entry, the server's extra timing fields (`sampled_on_device`,
  `grammar_n`, `cache_src`, `ckpt_*`, ...) and the info level of its checkpoint log lines.
- **The -1 expert-id checks** compiled into llama.cpp's expert kernels: a comparison never taken for valid ids
  (`GGML_CUDA_SKIP_IDS_ANY` switches the host-side handling; what the comparisons cost at five and six tokens is Q1028).
- **Op numbering:** the fork's ops (`GGML_OP_GDN_CONV`, `GGML_OP_KQ_MASK`, `GGML_OP_MOE_TAIL`) change the op enum, so an
  RPC build carries the fork's own protocol version.
