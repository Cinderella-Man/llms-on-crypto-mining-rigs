# Fork scope: a Flint llama.cpp, a Pascal llama.cpp, or a llama.cpp for layer splits without P2P? (2026-10-24)

Scope: the owner asked what the project's llama.cpp fork actually is, because the answer decides the public
repository's name and positioning, the fork's branch names, what goes upstream, and which other models and GPUs the
work helps. This note:
- lists every change in the fork (v1 `pascal-p106`, v2 `pascal-p106-v2` and its feature branches) with its commit, files
  and purpose, and classifies it as **generic** (any model, any GPU), **Pascal** (compute capability 6.1), **MoE**
  (any mixture-of-experts model in llama.cpp) or **Flint** (only Flint's architecture), checked against the code: does a
  kernel hard-code Flint's shapes or router, or read them from the model;
- works out which other models (Qwen3.6-35B-A3B, Qwen3-30B-A3B, gpt-oss, GLM-4.5-Air, DeepSeek-style MoE, Ornith,
  dense models) and which other GPUs (the rest of Pascal, Volta, Turing, Ampere) get each speed-up;
- compares three positionings with their names, evidence and maintenance cost, and what a GitHub rename breaks;
- ranks the changes for upstream llama.cpp, with the reviewers' likely concerns, and sketches an upstream-friendly
  split of the fork;
- recommends one positioning, the names, the README's title and first paragraph, and the steps. Nothing was renamed:
  renames are the owner's call.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- Code citations are `branch:path:line` in the fork's worktrees on 2026-10-24. Branches: `v1` = `pascal-p106`
  (`eeaa59660`), `v2` = `pascal-p106-v2` (`032b130f4`), `comb` = `pascal-p106-v2-combined` (`b3052a4f8`), `pers` =
  `pascal-p106-v2-persistent` (`4b2831712`), `pipe2` = `pascal-p106-v2-pipespec2` (`74f7c7182`), `mask2` =
  `pascal-p106-v2-gpumask2` (`6f943f792`), `q6k` = `pascal-p106-v2-q6k` (`03b92cf10`), `tool` =
  `pascal-p106-v2-toolsample` (`f11ac1589`), `tscr` = `pascal-p106-v2-toolsample-screen` (`b6ffd7e6b`). `ggml-cuda/`
  stands for `ggml/src/ggml-cuda/`. The upstream base is `d77256999`.
- Speeds are from the [registry](../../../quants/experiments.md) (Qn) and the [README](../../../README.md); nothing ran
  on the rig or the quant machine for this note.

## Verdict

- **The fork is a Pascal llama.cpp with a generic multi-GPU layer on top and a thin Flint layer.**
  - **Speed:** of v2's decode gain on Unsloth's UD-Q4_K_XL (17.6 → 37.5 tok/s on six cards, 2.13×), the Pascal-only
    changes carry 87 % (CUDA graphs on compute 6.1, the 2-warp launch shape, the Pascal matrix-vector kernels), the FP32
    activation sums inside those kernels 12 %, and a generic fusion fix 1 % (shares of the logarithm, est. from the
    README's step table and Q76/Q50). Not one percent of that is Flint-only.
  - **Code:** of ~9,000 inserted source lines in 50 own commits, ~1,510 are Pascal kernels, ~5,320 generic (of which
    ~1,800 are the `llama-pipespec` tool) and ~2,080 Flint-only (the persistent MoE-layer kernel, ~1,870 including ~360
    lines moved out of the Pascal kernel file, and the Q4_1_64 type). Two more commits are upstream PR #31047, by a
    llama.cpp CUDA maintainer.
  - **The only Flint-only speed piece is the persistent MoE-layer kernel** (Q88: 1.16–1.17× per MoE layer, est. +5 % per
    token, rig timing running). It hard-codes hidden size 2,048, 256 experts, 6 used, width 512, a shared expert of 1,024
    and sigmoid routing (`pers:ggml-cuda/moe-persistent.cu:13-17, 213-220`). Every other kernel reads its shapes from
    the tensors.
- **Who else benefits, on code reading and the measurements in hand:**
  - **Any model on a compute-6.1 card** gets CUDA graphs and the 2-warp shape (#27721: GTX 1080 +6–48 %, P102-100
    +25 %), and the Pascal kernels wherever its tensors are Q4_0, Q4_1, Q5_0, Q8_0, Q5_K or IQ4_XS (Q6_K on a branch).
    Qwen3.6-35B-A3B Q4_0 measured +27 % from the kernels on this rig (Q38).
  - **The kernels miss the most common downloads' main type and gpt-oss:** Q4_K (Q4_K_M files), MXFP4 (gpt-oss's
    experts) and the IQ2/IQ3 types that a 110–122 B model needs to fit 54 GB fall back to llama.cpp's kernels.
  - **Pipelined speculation, the on-card attention mask, the graph-reuse fix, packed uploads and GPU sampling with a
    grammar are generic**: any layer split without P2P on any CUDA GPU, any model with an attention-only KV cache.
    Hybrid recurrent models (Qwen3.6, Ornith, Qwen3.8-27B) are excluded from the mask and from wrong-guess rollback in
    the executor today (`mask2:src/llama-graph.cpp:4249`, `pipe2:src/llama-memory-recurrent.cpp:189-190`).
  - **Other Pascal cards** have 2–3× more SMs per byte of bandwidth than the P106-090, so the kernels' gains there are
    likely smaller (est.; #25479 measured +2.5–5.4 % on a GTX 1060 for what gave +13–27 % here). None was measured.
    The P100 (compute 6.0, no dp4a) gets none of the Pascal pieces.
- **Recommendation: position the project as (b), a llama.cpp fork for Pascal cards and mining rigs that runs any model,
  with Flint as the measured flagship.**
  - It matches the code and the owner's goal ("help anybody with Pascal architecture cards").
  - The generic multi-GPU pieces go upstream rather than becoming the brand (c).
  - The Flint-only kernel stays a switchable extra.
- **Names:**
  - Keep the docs repository name `llms-on-crypto-mining-rigs`, and give the README the title "LLMs on crypto-mining rigs
    and other Pascal GPUs" with a first paragraph that states the scope.
  - Never rename the pushed branches `pascal-p106` and `pascal-p106-v2` (a branch rename breaks `git clone --branch` and
    `git pull` on the old name).
  - Add a release branch `pascal` (made the default branch at v2.0) and tags `pascal-v1.0` (`eeaa59660`) and
    `pascal-v2.0`; name new branches `up/<topic>` (upstream candidates on master), `pascal/<topic>` and `flint/<topic>`.
  - Optionally rename the fork repository to `llama.cpp-pascal`: GitHub redirects web links and git operations after a
    repository rename.
- **Before the README claims "many models":**
  - one rig session with a model matrix: Qwen3.6, Qwen3-30B-A3B, gpt-oss-20b and a dense 7–8 B, stock against the fork;
  - one measurement on a second Pascal card;
  - a Pascal Q4_K kernel, so that the most common downloads get the speed-up.
- **Upstream, in order:**
  1. Rig data posted on #27721 (CUDA graphs on Pascal) and #31047, which costs no PR slot.
  2. The gate/up/GLU fusion fix as the first own PR: small, generic, any GPU.
  3. A Pascal-only MMVQ table.
  4. The Pascal kernel file, as a large addition with the owner as its code owner.
  5. GPU sampling with a grammar, coordinated with #25919.
  6. Later and only after rig evidence: the graph-reuse fix, the on-card mask (a new ggml op) and the pipelined-decoding
     API (an RFC first).

  Q4_1_64, the Hadamard metadata line, the rescale switch, the persistent kernel and the screened head's files stay in
  the fork.

## 1. Inventory and classification

### 1.1 Every change

Each row is one change; cherry-picks of the same change on several branches are listed once. "Pushed" means the branch
is on the fork's GitHub remote.

| Change (commits; branch) | Files | Class | Evidence | Other models and GPUs that benefit |
|---|---|---|---|---|
| **CUDA graphs on compute 6.1** (`fedac87b2`; v1) | `ggml-cuda.cu`, 1 line | Pascal | `v2:ggml-cuda/ggml-cuda.cu:5017`: gate `cc < GGML_CUDA_CC_DP4A` instead of Volta; same as upstream #27721 | Every model on compute 6.1/6.2; the gain follows the node count (#27721: MoE up to +48 %, dense +6–7 %). Not P100 (regressed in #27721). Volta and newer already have graphs upstream |
| **`LLAMA_NO_MOE_DOWN_RESCALE`** (`8534e1b4d`; v1), **`=tg`** (in `3733df1dd`; v2) | `src/llama-graph.cpp` | MoE (three archs) | `v2:src/llama-graph.cpp:2528`: upstream's rescale applies to `MISTRAL4`, `MELLUM`, `FLINT` only; `:1677-1684` the switch | Mistral Small 4 and Mellum too; on any GPU only where the down-projection's kernel does not read FP16 block sums |
| **Hadamard metadata for Flint** (`0f71ddac2`; v1) | `src/llama-model.cpp`, 1 line | Flint | `v2:src/llama-model.cpp:2136` adds Flint to PR #29600's architecture list | None (Qwen3.5/3.6/Next are listed upstream); no published Flint file carries the metadata |
| **Q4_1_64 type** (`b60676d5a`; v1) | 22 files | Flint (a generic format made for the makers' group-64 file) | `v2:ggml/include/ggml.h:433`, type id 43 | None in practice; the makers' Q4_1 tier left the release targets (decision 2026-10-23) |
| **2 warps per block at one token, wide rows** (`9e826151a`; v1) | `ggml-cuda/mmvq.cu`, 4 lines | Pascal (tuned), not gated | `v2:ggml-cuda/mmvq.cu:456-461` edits `MMVQ_PARAMETERS_GENERIC`, which `:118-123` assigns to every NVIDIA GPU except Volta/Turing and GB10 | Any model on Pascal. On Ampere and newer, a fork build made for them changes launch shapes unmeasured |
| **CUDA-event profiler** (`bb7390902`, `a562fca77`, `7d1f070fd`; v1) | `ggml-cuda.cu` | generic | `v2:ggml-cuda/ggml-cuda.cu:4608, 4965` (`GGML_CUDA_PROFILE`, `_HOST`) | Any CUDA GPU; on mining cards the only profiler, since NVIDIA's tools refuse them |
| **Pascal matrix-vector kernels, one token** (Q4_0, Q4_1, Q8_0 `f254fc7f6`; Q5_0 `112cdc597`; Q5_K, IQ4_XS `eeaa59660`; v1) | new `ggml-cuda/mmvq-pascal.cu/.cuh`, a call in `mmvq.cu` | Pascal | `v2:ggml-cuda/mmvq-pascal.cu:610`: on only for cc 6.1 (`GGML_CUDA_MMVQ_PASCAL=2` forces any NVIDIA GPU); `:509-512`: launch shape chosen by type, measured at Flint's K of 512–4,096; `pers:ggml-cuda/mmvq-pascal.cu:669-673`: lanes per row from the row width; rows, strides, expert ids and fusions from the call's arguments | Any model whose tensors use these types, on cc 6.1: Qwen3.6 Q4_0 +27 %, Unsloth's Flint Q8_0 +33 %, makers' Q4_1 +23 % (Q38). Not Q4_K, Q2_K/Q3_K, IQ2/IQ3, MXFP4. Widths that are not a multiple of 2^k blocks get narrow lane groups (gpt-oss's 2,880: one lane per row, speed unknown) |
| **FP32 activation sums in those kernels** (`3733df1dd`; v2) | `mmvq-pascal.cu`, `llama-graph.cpp`, a 143-line test | Pascal kernel; its speed gain MoE (three archs) | `v2:src/llama-graph.cpp:1670-1684`; test `test_mul_mat_large_act` | Correct results for every model with activation blocks past 65,504 on cc 6.1; +9.6 % (Q76) only where the rescale exists (Flint, Mistral4, Mellum) |
| **Gate/up/GLU fusion when the GLU output reuses the activations' memory** (`032b130f4`; v2) | `ggml-cuda.cu`, 15 lines | generic | `v2:ggml-cuda/ggml-cuda.cu:3517` | Any CUDA GPU; shared-expert models whose shared FFN is the last reader of the layer input (Qwen3.6, GLM-4.5-Air, DeepSeek-style, Ornith) and dense FFNs where ggml-alloc reuses the input (est.); Flint +0.7 % (Q50) |
| **Tests at Flint's shapes** (`5d64ad05c` v1; `bfc79be6f`, `3bb9ac3b3` comb) | `tests/test-backend-ops.cpp` | test only | 256 experts, 6 used; 1–4 tokens and prompt-sized batches | Perf and correctness cases for any 256-expert MoE |
| **Upstream PR #31047, MoE tile list for MMQ** (`84dc16eba`, `4a8ab277a`; prefill, comb; pushed) | `mmq.cu/.cuh`, `mmid.cu/.cuh`, `mmf.cu/.cuh` | MoE (upstream) | Authored upstream; off switch `comb:ggml-cuda/mmq.cu:93-98` | Prompt processing of every MoE model on any GPU that runs MMQ: P40 1.40× at ubatch 64 (the PR), this rig +22–29 % (Q117) |
| **Graph reuse only with no micro-batch in flight** (`6b002d284`; prefill, comb; pushed) | `src/llama-context.cpp/.h`, 15 lines | generic | `comb:src/llama-context.cpp:1586-1587` | Any model, any backend, layer split with pipeline parallelism at small `-ub`; on the rig no effect at `-ub 512`, its `-ub 128` target not yet timed (Q117) |
| **Attention mask built on each card** (`GGML_OP_KQ_MASK`: `3ed40053f`; cell copies per card: `6f943f792`; gpumask, gpumask2; pushed) | new op (CPU + CUDA), `llama-graph`, KV cache | generic | `comb:ggml/include/ggml.h:607, 1837`; `mask2:src/llama-graph.cpp:3394, 4016` (plain and sliding-window caches), `:4249` (hybrid memory keeps the host mask) | Models with a plain or sliding-window KV cache (Qwen3-30B, gpt-oss, GLM, DeepSeek-style, dense Llama-class), any layer split; not hybrid recurrent models. +1.3 % at 8k on the rig; the 32k–131k target untimed |
| **The norm writes the Q8_1 copy and absorbs the residual add** (`5a8a3be90`; normq8, comb; pushed) | `norm.cu`, `ggml-cuda.cu`, `mmvq.cu`, `common.cuh` | generic | `comb:ggml-cuda/norm.cu:909-915` (switch only, no architecture gate), `:925-926` (row width a multiple of 512, ≤ 4,096) | Any CUDA GPU; hidden sizes 2,048–4,096 (Qwen3.x MoE, GLM-4.5-Air, Ornith's 3,072, Llama-8B); not gpt-oss (2,880) or Qwen3.8-27B (5,120). No measurable decode effect on the rig (Q117) |
| **Screened output head** (`3e28a186d`, `ae450bfec`, `0e78c7e30`; screenhead, comb; pushed) | `llama-graph`, model loader, `llama-model`, common, server, `llama-screen-check` | generic engine, per-model data | `comb:src/llama-graph.cpp:4390`: any architecture whose logits are the plain head product of the final hidden state (no softcap, scale or bias); `comb:src/llama-model-loader.cpp:678-680`: side file | Any model once a screen is built and validated for it ([quants/gptq_head](../../../quants/gptq_head/) takes `--n-embd`); the head is 10 % of Qwen3-30B-A3B's per-token weights, 12 % of Flint's, 16 % of gpt-oss-20b's and 17 % of Qwen3.6's (est., vocabulary × hidden over active parameters) |
| **2–4 tokens in the Pascal kernels** (`0c24381d6`, `8faeb9fce`; multicol, comb; pushed) | `mmvq-pascal.cu`, tests | Pascal | `comb:ggml-cuda/mmvq-pascal.cu:45` (`MMVQP_MAX_COLS 4`); each distinct expert read once per step from the ids | Verify steps, several slots and prompt tails of any model on cc 6.1 with the covered types; Qwen3.6's MTP was held at 1.04× by the stock multi-token kernels (Q43) |
| **Q6_K in the Pascal kernels** (`03b92cf10`; q6k; pushed) | `mmvq-pascal.cu` | Pascal | `q6k:ggml-cuda/mmvq-pascal.cu:63` | The Q6_K tensors of every Q4_K_M, Q5_K_M and Q6_K file (heads, some down projections) on cc 6.1 |
| **Scheduler without host waits before split inputs** (`b6833432d`; pipespec, comb) | `ggml-backend.cpp/.h` | generic | `comb:ggml/src/ggml-backend.cpp:2156` | Any multi-backend split |
| **KV cache: head placement after tail removal, free-cell search** (`13ac81c6f` comb; `fbb1b24f3` pipe2) | `src/llama-kv-cache.cpp` | generic | `comb:src/llama-kv-cache.cpp:450` | Any model with speculative rollbacks |
| **Pipelined single-token decoding** (`llama_pipe_*` `1024b215f`, backend greedy `2ba5eda76`, position-keyed draws `c6e4289d5`, slots for several sequences `c99db49bd`; pipespec, comb; pushed) | `include/llama.h`, `llama-context`, `llama-sampler` | generic | `pipe2:src/llama-context.cpp:1626-1642`: any causal decoder with a memory module; a sliding-window model needs `n_ubatch` ≥ slots. `pipe2:src/llama-memory-recurrent.cpp:189-190` (upstream code): a recurrent state cannot drop a tail | Any attention-only model on a layer split of ≥ 2 devices, any GPU (correctness checked on two RTX 3090s and on the CPU); 1.3–1.55× on this rig with untrained guesses (Q87). Hybrid recurrent models need a state rollback first |
| **`llama-pipespec`** (`465102164` … `1113087aa`, `74f7c7182`; pipespec, pipe2) | `examples/pipespec` | generic (tool) | guessers, race of k samples, streams mode | Any model the executor supports |
| **Packed asynchronous uploads; device-fed next token** (`f249104ab`; pipe2; pushed) | `ggml-backend`, `llama-context`, `llama-model` | generic | `pipe2:ggml/include/ggml-backend.h:377-381`; `pipe2:src/llama-model.cpp:1699-1705` (`LLAMA_INPUT_ON_GPU`) | Any multi-device split with a slow host |
| **Off switches** (`b3052a4f8`; comb) | `mmid.cu`, `mmq.cu`, `mmvq-pascal.cu`, `llama-context` | mixed | measurement only | — |
| **PR #31047 for prompt batches only** (`511943869`; prefill-split, local) | `mmid.cu`, `mmq.cu` | MoE | parked: the split is not needed (decision 2026-10-24 12:25) | — |
| **Persistent MoE-layer kernel** (`152eb5968`, `2d5e1fbf4`, `4b2831712`; persistent, local) | new `ggml-cuda/moe-persistent.cu/.cuh`, `mmvq-pascal-impl.cuh`, a graph matcher in `ggml-cuda.cu`, tests | **Flint** | `pers:ggml-cuda/moe-persistent.cu:13-17` (shapes as `#define`s), `:213-220` (sigmoid top-k with normalized weights), `:957-995` (exact-shape checks), `:837-855` (the type combinations of Flint files); `pers:ggml-cuda/ggml-cuda.cu:3503-3515` (matcher: sigmoid chain, a shared expert required) | None today. Qwen3.6 is the closest shape (hidden 2,048, 256 × 512), but top-8, a softmax router and a sigmoid-gated shared expert (`v2:src/models/qwen35moe.cpp:502, 520-533`) |
| **GPU sampling with a grammar** (`e2d587398`, `972ccde20`, `f11ac1589`; toolsample, local) and **the screened head kept with a grammar** (`45d243e80`; tscr, local) | `common/sampling`, `llama-context`, server | generic | `tool:common/sampling.cpp:432-433, 690-732`: the device's token is checked on the host and the full row resampled only on a rejection | Tool requests of any model on any GPU with backend sampling |

### 1.2 Hard-coded or read from the model

- **Pascal matrix-vector kernels: read from the model.**
  - Rows, row width, strides, channels, expert ids and fusion operands come from the call
    (`v2:ggml-cuda/mmvq-pascal.cu`, the `ggml_cuda_mmvq_pascal` arguments).
  - The lane count per row is derived from the width: halve until the block count stops dividing, at most 32 lanes
    (`pers:ggml-cuda/mmvq-pascal.cu:669-673`).
  - What is fixed is the tuning: the rows per lane group and blocks per lane per type (`v2:ggml-cuda/mmvq-pascal.cu:509-528`),
    measured on a P106-090 at Flint's K values, 4 warps per block and at most 4 tokens.
  - Unaligned or indivisible shapes return to llama.cpp's kernel instead of failing.
  - The gate is the device, not the model: cc 6.1 only unless forced (`:610`).
- **The 2-warp launch rule: read from the row width** (small-K or not), but placed in the generic table, so it is
  device-ungated (`v2:ggml-cuda/mmvq.cu:456-461`).
- **FP32 activation sums: model-independent.** Only the decision to drop the rescale names architectures, and those are
  upstream's three (`v2:src/llama-graph.cpp:2528`).
- **Norm + Q8_1: generic** within its width rule (`comb:ggml-cuda/norm.cu:925-926`).
- **Screened head: generic in the engine.** It reads `output_screen.weight` and `<arch>.output_screen.top_m` for any
  architecture. The screen itself (a GPTQ Q4_0 copy of the head, validated for exactness on held-out rows) is made per
  model.
- **Device mask, executor, uploads, scheduler, sampling: generic.** They are written against llama.cpp's memory and
  scheduler abstractions. The exclusions are hybrid memory (mask) and recurrent tails (executor).
- **Persistent MoE kernel: hard-coded to Flint.**
  - Fixed: hidden 2,048, 256 experts, top-6, routed width 512, shared width 1,024 (`pers:ggml-cuda/moe-persistent.cu:13-17`).
  - The work list, scratch layout and counter offsets are compile-time functions of these.
  - Routing is sigmoid with normalization and no selection bias or groups (`:213-220`; the matcher at
    `pers:ggml-cuda/ggml-cuda.cu:3503-3515`).
  - A shared expert is required.
  - The type combinations are the twelve that Flint's K2h-r16, G3, S4-L0 and UD-Q4_K_XL files use (`:837-855`).
  - The router type (F16, BF16, F32) is a runtime branch.
  - Any other model fails `ggml_cuda_moe_persistent_supported` (`:957-995`) and runs the per-op path, so it is safe but
    gives nothing.

### 1.3 Totals

| Class | Own commits | Inserted source lines | Test lines | Share of v2's decode gain on UD-Q4_K_XL, 6 cards (est.) | Pending pieces |
|---|---|---|---|---|---|
| Pascal | 9 | ~1,510 | ~190 | 87 % (graphs 13.5 %, 2 warps 30 %, kernels 43.5 %) + the FP32 sums' 12 % | multi-token kernels, Q6_K |
| MoE (incl. the three-arch rescale) | 2 (+2 upstream) | ~40 (+224) | ~10 | inside the FP32 sums' 12 % | — (prompts +22–29 % from #31047) |
| generic | 30 | ~5,320 (~1,800 of it the tool) | ~200 | 1 % (fusion fix) | pipelined decoding 1.3–1.55×, screen est. +3–6 %, mask at depth, GPU sampling for tools est. +9–16 % |
| Flint | 5 | ~2,080 | ~150 | 0 % | persistent kernel est. +5 % |

- **Not in the table:** three test-only commits and the commit with the off switches; 50 own commits in all.
- **With pipelined decoding:** at its measured 1.3–1.55×, a quarter to a third of the total logarithmic speed-up
  becomes generic (est.).
- **With the persistent kernel:** if its +5 % (est.) holds, it would be the only Flint-only share, ~4 % of the total
  (est.).
- **Footprint:** the v2 base touches 26 files with +1,379 lines; the most complete branch (`pers`) 63 files with +7,390.

## 2. Who else benefits

### 2.1 Models

Code reading plus the measurements named; "—" means the piece does not apply.

| Piece | Qwen3.6-35B-A3B (hybrid GDN, 256 × 512, top-8 softmax, gated shared expert) | Qwen3-30B-A3B (128 × 768, top-8 softmax, no shared) | gpt-oss-20b (32 × 2,880, top-4, MXFP4 experts, SWA) | GLM-4.5-Air (128 × 1,408, top-8 sigmoid + bias, shared) | DeepSeek-style (shared experts, sigmoid/softmax + bias, groups, MLA) | Ornith-1.5-122B (Qwen3.5-122B arch, hybrid) | Dense (Llama-class; Qwen3.8-27B hybrid) |
|---|---|---|---|---|---|---|---|
| CUDA graphs (cc 6.1) | yes; Q4_0 18.5 → 19.6 tok/s on this rig | yes | yes | yes | yes | yes | yes; #27721 dense +6–7 % |
| 2 warps, wide rows | yes | yes | yes | yes | yes | yes | yes |
| Pascal 1-token kernels | Q4_0 +27 % (Q38); UD-Q4_K_XL's Q5_K/IQ4_XS | Q4_0/Q8_0 files; not Q4_K | non-expert Q8_0 only (MXFP4 not covered; width 2,880 gives one lane per row) | by type (on 54 GB only ~Q3 fits) | by type | the files that fit 54 GB are IQ2/IQ3: not covered | Q4_0/Q8_0 files (est.); Q4_K_M not |
| FP32 sums + rescale switch | correctness only | correctness only | — (MXFP4 reads no sums) | correctness only | correctness only | correctness only | correctness only; Mistral Small 4 gains the rescale removal |
| Gate/up/GLU fusion fix | likely (shared expert, est.) | — | — | likely (est.) | likely (est.) | likely (est.) | where ggml-alloc reuses the FFN input (est.) |
| PR #31047 (prompts) | yes | yes | yes | yes | yes | yes | — |
| Graph-reuse fix (prompts, layer split) | yes | yes | yes | yes | yes | yes | yes |
| On-card attention mask | — (hybrid) | yes | yes (SWA) | yes | yes (est., plain cache) | — | Llama-class yes; Qwen3.8-27B — |
| Norm writes Q8_1 | yes (2,048) | yes | — (2,880) | yes (4,096) | 2,048-wide yes; V3's 7,168 no | yes (3,072) | ≤ 4,096 yes; 5,120 no |
| Screened head | high value (17 % head, est.) | 10 % head (est.) | 16 % head (est.) | needs a screen | needs a screen | needs a screen | small share on large dense (5 % for Qwen3.8-27B) |
| 2–4-token kernels | MTP verify (Q43 held at 1.04× by stock kernels) | verify steps | non-expert tensors | by type | by type | — | verify steps |
| Pipelined speculation | not until a recurrent-state rollback exists | yes | yes (SWA: `n_ubatch` ≥ slots) | yes | yes | not until rollback | Llama-class yes; Qwen3.8-27B not until rollback |
| GPU sampling with grammar | yes | yes | yes | yes | yes | yes | yes |
| Persistent MoE kernel | needs top-8, softmax, gated shared expert templates (est. days) | needs a no-shared, 128 × 768 variant (est. days) | far (MXFP4, expert biases, OAI SwiGLU) | far (selection bias) | far (bias, groups) | far (3,072 hidden, hybrid) | — |

- **What the table says:** the engine work transfers to any model through the Pascal pieces and the multi-GPU pieces.
  The two biggest gaps are formats and hybrid models:
  - **Formats:** a Pascal Q4_K kernel, Q6_K landing and MXFP4 would reach Q4_K_M files and gpt-oss.
  - **Hybrid recurrent models:** Qwen3.6, Qwen3.5-122B, Ornith and Qwen3.8 need the mask and the executor extended to
    hybrid memory.
- **Fit on 54 GB:** gpt-oss-120b (~63 GB at MXFP4, est.) does not fit on nine P106 cards; GLM-4.5-Air (110 B) and Ornith
  need ~3-bit or lower files to fit.

### 2.2 GPUs

| Card | Compute | SMs, bandwidth (spec unless noted) | Bandwidth per SM | Pascal kernels, graphs, 2 warps | Generic pieces | Evidence |
|---|---|---|---|---|---|---|
| P106-090 (this rig) | 6.1 | 5, 173 GB/s measured | ~35 GB/s | on by default | yes | all measurements here |
| P106-100, GTX 1060 6 GB | 6.1 | 10, 192 GB/s | ~19 GB/s | on | yes | P106-100 reached ~115 GB/s against the P106-090's 73 on the same file ([overview](../../overview.md)), so less issue-bound; #25479 +2.5–5.4 % on a GTX 1060 for the 2-warp idea |
| P104-100, GTX 1070/1080 | 6.1 | 15/15/20, 320/256/320 GB/s | 16–21 GB/s | on | yes; the P104-100's VBIOS also locks PCIe 1.0 | #27721 GTX 1080 graphs +6–48 %; P104 rigs on Gen1 x1 exist ([community](quant_speed_d_community.md)) |
| P102-100, GTX 1080 Ti, Titan Xp | 6.1 | 25–30, ~440–550 GB/s | ~17–18 GB/s | on | yes | #27721 P102-100 graphs +25 % |
| Tesla P40 | 6.1 | 30, 346 GB/s | ~12 GB/s | on | yes | PR #31047 1.40× (its author's P40) |
| Tesla P100 | 6.0 | 56, 732 GB/s | ~13 GB/s | **off** (no dp4a; graphs regressed in #27721) | yes | — |
| Jetson TX2 | 6.2 | — | — | kernels off (`cc == 610`); graphs on, untested | yes | — |
| V100 (Volta), Turing | 7.0, 7.5 | — | — | graphs already upstream; kernels off; 2-warp rule not applied (own table) | yes | executor and kernels checked for correctness on V100s/RTX 3090s, speed not measured |
| Ampere and newer | 8.x+ | — | — | kernels off; the fork's 2-warp generic-table edit applies if built for them (unmeasured) | yes | — |

- **Expected size of the Pascal gains elsewhere (est.).** The P106-090 has 2–3× the bandwidth per SM of the other cards
  above, which is why it runs out of instruction issue long before bandwidth. Larger cards are less issue-bound, so the
  kernels and the 2-warp rule should gain less there, and need a no-regression check. The pattern is visible in the
  2-warp idea: +2.5–5.4 % on a 10-SM GTX 1060 against +13–27 % here.
- **Bigger Pascal cards may want different tuning.** The upstream draft for the kernel PR already says the PR must show
  no regression on a GTX 10-series card or P40, or restrict itself by SM count
  ([kernel design §12](pascal_mmvq_integration.md)).
- **The P106-100 and P104-100 are the closest relatives.** They have the same instruction set and the same PCIe 1.0 x1
  rigs, so every piece applies, and only the size of the kernel gain is open.

### 2.3 Multi-GPU boxes without P2P

- **These pieces assume nothing about Pascal:** pipelined speculation and the device-fed next token (one card busy at a
  time on a layer split, idle cards fed guesses), the packed uploads and the scheduler change (fewer host copies per
  token), the graph-reuse fix (prompt micro-batches overlapping across cards) and the on-card mask (no per-token mask
  copies over slow links).
  - They help any layer split where the per-token hand-off and the host loop cost time: consumer GPUs without NVLink,
    mixed cards, PCIe x1 risers, a weak host.
  - Their size shrinks as links and hosts get faster (est.).
- **How far the evidence reaches:** the executor's correctness was checked on two RTX 3090s and on the CPU
  ([pipelined speculation](pipelined_speculation.md)); its speed exists only for this rig.
- **Where they belong:** they are the strongest upstream candidates in the fork, because every llama.cpp user with
  several GPUs gets them, Pascal or not.

## 3. Positioning and naming

### 3.1 The three options

| | (a) Flint on mining rigs | (b) Pascal / mining-rig llama.cpp, many models, Flint as flagship | (c) llama.cpp for layer splits without P2P |
|---|---|---|---|
| **Serves** | Flint users with a P106 rig: the intersection of one model and one card | Owners of any compute-6.1 card (GTX 10-series, P40, P102/P104/P106 rigs), with any model whose files the kernels cover | Anyone with several GPUs and no NVLink/P2P, any architecture |
| **Evidence today** | Strong: every measurement is Flint on P106 | Flint 2× stock (README); Qwen3.6 Q4_0 +27 % from the kernels (Q38); graphs +6–48 % on other Pascal cards (#27721); no other card measured with the kernels | Pipelined speculation 1.3–1.55× and prompt pieces, all measured on one PCIe 1.0 x1 rig; correctness only on RTX 3090s |
| **What the code says** | Contradicts it: 0 % of v2's decode gain is Flint-only | Matches it: 87 % Pascal, the generic layer on top, one Flint-only kernel behind a switch | Partly: the generic pieces fit, but the Pascal kernels (most of the gain) mean nothing to most of this audience |
| **Maintenance** | Low per change, but the project's value falls with one model's popularity | Moderate: the kernels sit in their own file with a one-call hook in `mmvq.cu` (14 upstream commits there in the last three months); a model matrix per release (est. half a rig day); Q4_K and MXFP4 kernels to close the format gap (est. days each) | Highest: the pieces live in upstream's busiest files: `llama-context.cpp` 52, `llama-graph.cpp` 49, `server-context.cpp` 61, `ggml-cuda.cu` 64 upstream commits in three months (2026-07-23 to 10-23). Carrying them as a product means rebasing them forever |
| **Repo name** | `flint-on-mining-rigs` | keep `llms-on-crypto-mining-rigs` | e.g. `llama-layer-split` |
| **Fork branches** | `flint-p106`, `flint-p106-v2` | `pascal` (release, default), tags `pascal-v1.0`, `pascal-v2.0`; frozen `pascal-p106`, `pascal-p106-v2`; `up/`, `pascal/`, `flint/` topic branches | `pipeline`, `pipeline-v2` |

### 3.2 What the owner's goal implies

- **"Help anybody with Pascal architecture cards"** names the audience by architecture, not by model. That rules out (a)
  as the brand. It also doesn't stretch to Volta or Ampere, so (c) goes beyond the goal.
- **"Thousands of these Manli 9 × P106 rigs exist"** keeps the mining rig as the core reader: the hardware page, the rig
  setup, the PCIe 1.0 x1 constraints and the tarballs for this rig stay first. The P104-100 and P102-100 rigs are
  natural second readers, with the same links and the same instruction set.
- **Who gets the generic pieces:** users of any GPU get them best through upstream llama.cpp, which this project feeds
  but does not compete with.

### 3.3 What a rename breaks

- **Repository rename (GitHub docs):**
  - Web traffic and `git clone`, `git fetch` and `git push` to the old name are redirected, and so are issues, stars and
    followers.
  - Not redirected: GitHub Pages project sites and calls to Actions hosted in the renamed repository. Neither repository
    uses them.
  - The redirect ends if a new repository is created under the old name, so the old name must never be reused.
- **Branch rename (GitHub docs):**
  - Web URLs are redirected; raw file URLs are not.
  - `git pull` on the old name is not redirected.
  - Open pull requests whose head is the renamed branch are closed.
  - So `git clone --branch pascal-p106` in the v1 README, the v1 benchmark page, `patches/README.md`,
    `scripts/build_llamacpp_docker.sh`'s example and every worktree that tracks `fork/pascal-p106-v2` would break.
  - Pushed branches therefore keep their names; new names are added beside them.
- **Links in this repository:**
  - Absolute links to the fork are in [README.md](../../../README.md),
    [the Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md), [patches/README.md](../../../patches/README.md)
    and [release v2](release_v2.md).
  - Code comments in the fork name this repository's URL (`v2:ggml-cuda/mmvq-pascal.cu:510`,
    `pers:ggml-cuda/mmvq-pascal-impl.cuh:342`, `pers:ggml-cuda/moe-persistent.cuh:17`). All of these keep working through the redirect, and should be updated with the rename.
  - Relative links are unaffected.
  - Independently of any rename, eight relative links in [quant_speed_b_formats_kernels.md](quant_speed_b_formats_kernels.md)
    and [quant_speed_d_community.md](quant_speed_d_community.md) are broken today: after `../../../` they repeat the
    folder name `llms-on-crypto-mining-rigs/`.
- **Submodule:** a private checkout that holds this repository as a submodule keeps fetching through the redirect. Its
  `.gitmodules` URL should be updated in the same step.

## 4. Upstreaming

### 4.1 What llama.cpp asks of a pull request

- **From [CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md), read 2026-10-24:**
  - The contributor writes every post. AI may not write PR descriptions or replies, and AI use in code must be
    disclosed.
  - One open PR at a time for a new contributor.
  - Features begin with an issue, not a PR.
  - Bug fixes need a test that fails before and passes after.
  - `test-backend-ops` on two backends for any `ggml` change, and test cases for new or changed operators.
  - `llama-perplexity` and `llama-bench` to show nothing regresses.
  - New public API faces a higher bar.
  - "Niche features" may land only as an example or tool, or stay in a fork.
  - Large additions need a designated owner who maintains them, a CI workflow and the hardware to test on.
- **Owners ([CODEOWNERS](https://github.com/ggml-org/llama.cpp/blob/master/CODEOWNERS) at `0ae0e1976`):**
  - `ggml/src/ggml-cuda/`: the CUDA team (JohannesGaessler, am17an, IMbackK, ORippler);
  - `ggml/include/` and `ggml.c`: ggerganov;
  - `src/llama-graph.*` and `src/llama-model.*`: CISC;
  - `common/`: the llama-common team; the server: the llama-server team;
  - `src/llama-context.*`, the KV cache and `ggml-backend.cpp` have no listed owner.
- **Earlier threads add one rule** ([upstream status](upstream_pascal_patches.md)): no kernel change may alter every
  GPU's path. Launch shapes go into per-architecture tables with a per-type whitelist.

### 4.2 Ranked

| # | Change | Route | Reviewers' likely concerns | Before posting |
|---|---|---|---|---|
| 0 | CUDA graphs on cc 6.1 | Rig data as a comment on #27721; no PR slot | cc 6.2 untested; P100 stays excluded; master's chunked graphs (#30804) changed the code around the gate (the one-line change still applies by hand) | A clean-master A/B on the rig |
| 0 | PR #31047 | Rig data on the PR | — (their own PR) | Q117's numbers are in hand |
| 1 | Gate/up/GLU fusion when the GLU output reuses the input memory | First own PR (small, generic; a merged PR lifts the one-open-PR limit); an issue first if a maintainer asks | That every path the fused op can take reads its activations before writing the output (MMVQ quantizes them into its own buffer first); a test that the fusion fires; a gain shown on more than one GPU and model (Flint +0.7 %) | An A/B on a shared-expert model (Qwen3.6) on the rig, plus `test-backend-ops` |
| 2 | Pascal MMVQ table (2 warps at one token, wide rows) | PR after asking #25479's author; replaces the fork's generic-table edit | Per-type whitelist; data from more than one Pascal card; other GPUs unchanged | Type sweep on a MoE and a dense model ([upstream status](upstream_pascal_patches.md)) |
| 3 | Pascal kernel file (1 token, then 2–4 tokens and Q6_K as follow-ups) | Issue, then PR; the owner as code owner of `mmvq-pascal.cu`, the rig as test hardware | Maintenance of a per-architecture kernel (the GLU epilogue exists three times); regressions on big Pascal cards (perhaps a gate by SM count); FP32 block sums differ numerically from the stock kernel; test coverage | A GTX 10-series or P40 result; the large-activation test |
| 4 | GPU sampling with a grammar (Q67) | Coordinate with #25919 | The exactness argument (sample once, accept if the grammar allows it, else resample from the masked row: the standard rejection argument gives the masked distribution exactly); the reasoning budget; tests without the `LLAMA_GRAMMAR_TEST_REJECT` hook | Rig timing (Q55's gap: 12–16 % on tool requests) |
| 5 | Graph reuse only with nothing in flight | Issue + PR once the rig shows the `-ub 128` gain | Interaction with input copies and CUDA graphs; no listed owner for `llama-context` | Q109's `-ub 128` A/B |
| 6 | On-card attention mask (`GGML_OP_KQ_MASK`) | Issue/RFC: a new ggml op | ggerganov owns `ggml.h`; CPU-first rule (CPU and CUDA exist); other backends fall back; ~300 lines of cell-copy bookkeeping in the KV cache; gain only at long context over slow links | Rig numbers at 32k–131k (E5) |
| 7 | Pipelined single-token decoding (`llama_pipe_*`) | Discussion/RFC; possibly first as an example, per the "niche features" rule | New public API; overlap with `common/speculative`; recurrent models; server integration | The depth ≥ 4 exactness bug fixed; llama-server integration (C7) |
| — | Norm writes Q8_1 | Hold | No measured decode gain on the rig (Q117) | A gain first |
| — | Screened head | Keep in the fork | Per-model screen files and conversion tooling; side-file loading through an environment variable is not upstream style | — |
| — | KV head fixes | Small PRs only if they fix a reproducible upstream problem | Tests | A reproducer on master |
| — | Profiler | Optional small debug PR | Overlap with existing tools | — |

- **Stays in the fork:**
  - Q4_1_64 (a new GGUF type would need a public file and the full quality data, and its use case is gone);
  - the Hadamard metadata line, until a published Flint file needs it;
  - `LLAMA_NO_MOE_DOWN_RESCALE` (an environment switch in the graph builder; upstream's fix is FP32 or BF16 sums);
  - the persistent kernel, until it is generalized and timed;
  - `llama-pipespec`'s experiments (race, streams);
  - the test hooks.

### 4.3 An upstream-friendly split of the fork

- **Order of the release branch, so that each layer can be dropped or upstreamed without touching the others:**
  1. upstream cherry-picks (PR #31047);
  2. upstream candidates, each one self-contained commit with its tests and at most a kill switch;
  3. the Pascal kernels (one file and one hook);
  4. generic experimental pieces (executor, mask, screen, GPU sampling with a grammar);
  5. Flint-only pieces last (persistent kernel, rescale switch, Hadamard line, Q4_1_64).

  Dropping layer 5 leaves a Pascal build that runs every model exactly as before, faster.
- **One `up/<topic>` branch per candidate, based on upstream master:** rebased when it is posted, so that a PR never
  carries fork-only context.
- **The cost of carrying the generic pieces:** today they are interleaved with Pascal commits on `comb`/`pers`. Each one
  that merges upstream removes a slice of the fork's footprint in upstream's busiest files.

## 5. Recommendation

### 5.1 Positioning: (b)

- **The code says Pascal.** 87 % of v2's decode gain on UD-Q4_K_XL is Pascal-only, and none of it is Flint-only. The one
  Flint-only kernel is switchable and worth est. +5 %.
- **Other models already show the transfer.** Qwen3.6 Q4_0 +27 % from the kernels on this rig; CUDA graphs on Pascal
  +6–48 % on other people's cards. The gaps (Q4_K, MXFP4, hybrid models) are named and bounded.
- **The owner's stated goal is Pascal owners.** Mining-rig owners stay the core reader.
- **Maintenance is cheapest this way.** The Pascal kernels live in their own file. The expensive generic pieces leave
  through upstream instead of defining the project.
- **Flint stays the flagship.** It is the best-measured model, the quality bar and the file work live there, and the
  persistent kernel shows what a model-specific extra looks like.

### 5.2 Names

| Item | Today | Recommended |
|---|---|---|
| Docs repository | `llms-on-crypto-mining-rigs` | unchanged; description and topics add `pascal`, `gtx-1080`, `p40`, `p104`, `p106`, `llama-cpp`, `mining-rig` |
| README title | "LLMs on crypto-mining rigs" | "LLMs on crypto-mining rigs and other Pascal GPUs" |
| Fork repository | `llama.cpp` | optional rename to `llama.cpp-pascal` (redirects keep old links and clones working; never recreate `llama.cpp` under the account) |
| Fork default branch | upstream's `master` | `pascal`, created at the v2.0 release commit |
| v1 branch | `pascal-p106` | unchanged, frozen; tag `pascal-v1.0` on `eeaa59660` |
| v2 integration branch | `pascal-p106-v2` | unchanged until v2.0, then frozen; tag `pascal-v2.0` on the release commit; later work on `pascal` |
| Feature branches | `pascal-p106-v2-<topic>` | existing ones unchanged; new ones `up/<topic>`, `pascal/<topic>`, `flint/<topic>` |
| Release tags | none (release v2 proposed `p106-v1.0`, `p106-v2.0`) | `pascal-v1.0`, `pascal-v2.0` (prefixed, so they never collide with upstream's `bNNNN` tags) |

**README first paragraph (draft):**

> A llama.cpp fork for NVIDIA Pascal GPUs (compute capability 6.1: GTX 10-series, Tesla P40 and the P102, P104 and
> P106 mining cards), built and measured on a retired mining rig with nine P106-090 cards, each on one PCIe 1.0 lane,
> behind a 2-core Celeron. Its kernels work with any model whose files use Q4_0, Q4_1, Q5_0, Q8_0, Q5_K or IQ4_XS. The
> flagship is Flint, a 40 B mixture-of-experts model that runs 2× faster here than on stock llama.cpp at the same
> quality. The pieces that help any multi-GPU box without peer-to-peer links go to upstream llama.cpp. This repository
> holds the measurements, model files and build recipes, and lists which cards and models were measured and which were
> not.

### 5.3 Steps (none done; the owner decides)

1. **Decide the positioning.**
2. **Measure before claiming:**
   - one rig session, stock against the fork with a warm-up, six cards, llama-bench tg128/pp512: Qwen3.6-35B-A3B Q4_0
     and UD-Q4_K_XL (Q82 is planned), Qwen3-30B-A3B Q4_0 and Q4_K_M, gpt-oss-20b, and a dense 7–8 B in Q4_0 and Q4_K_M;
   - a second Pascal card, from #27721's GTX 1080 or P102-100 testers or a P106-100/P104-100.
3. **Close the format gap:** land Q6_K (Q101), then a Pascal Q4_K kernel; MXFP4 if gpt-oss users show up.
4. **Make the 2-warp rule Pascal-only in the fork**, as a Pascal table, before anyone builds the fork for newer cards.
   It is the same work as upstream PR 2.
5. **Set up the fork:** description and topics; tags `pascal-v1.0` now and `pascal-v2.0` at release; the `pascal` branch
   as default at release; `up/` branches for the candidates.
6. **Update the docs repository:** README title and first paragraph, a "cards and models" table that separates measured
   from unmeasured; description and topics; fix the eight broken relative links.
7. **If the fork is renamed:** update the links in the four files of §3.3, the build script's example, the private
   checkout's remote and submodule URLs, and the code comments.
8. **Upstream** in the order of §4.2. Each post is written by the owner.
9. **Generalize the persistent kernel only after its rig timing**, and first for the next model people run on these
   cards (a Qwen3.6 template: top-8, softmax, gated shared expert).

## Not verified

- **Other cards and other GPUs:**
  - No Pascal card other than the P106-090 has run the fork's kernels. Their gains on GTX 10-series, P40, P102 and P104
    cards are estimates from bandwidth per SM and #25479's GTX 1060 data, and the SM counts and bandwidths of those cards
    are spec values.
  - The Pascal kernels' speed when forced on Volta, Turing or Ampere, and the 2-warp generic-table edit on Ampere and
    newer, are unmeasured.
- **Other models:**
  - Qwen3-30B-A3B, gpt-oss, GLM-4.5-Air, DeepSeek-style models, Ornith and dense models have not run on the fork. Their
    columns in §2.1 come from code reading, and "likely" marks fusion cases that depend on ggml-alloc's buffer reuse in
    each graph.
  - Whether upstream's rollback for speculative decoding on hybrid models (used by Qwen3.6's MTP, Q43) can serve the
    executor's wrong-guess drop.
  - The screen's value for other models is arithmetic on head size; exactness has to be validated per model.
- **This note's arithmetic:**
  - The decode-gain shares come from successive llama-bench steps in different sessions (README table, Q76, Q50).
  - The line counts are insertions per commit, including moved code and tests counted apart.
- **Upstream and GitHub:**
  - How upstream reviewers will respond is inferred from CONTRIBUTING.md, CODEOWNERS and earlier threads.
  - GitHub's redirect behaviour is taken from its documentation, not tested on these repositories.
- **Pending results:** the round-17 notes on RTX 3090s and V100s were being written at the same time as this one; this
  note does not rely on them.

## Sources

The fork, read-only (P):
- `git log --oneline d77256999..<branch>` and `git show --stat` for `pascal-p106`, `pascal-p106-v2` and the branches
  `-combined`, `-persistent`, `-pipespec`, `-pipespec2`, `-gpumask`, `-gpumask2`, `-prefill`, `-prefill-split`,
  `-screenhead`, `-q6k`, `-normq8`, `-multicol`, `-toolsample`, `-toolsample-screen`, `-combined-split`, and the v1-based
  `pascal-p106-fp32sums` and `pascal-p106-shexp`, whose commits were cherry-picked onto v2.
- Source files at the lines cited in §1.
- Upstream at `origin/master` `0ae0e1976` (2026-10-23): `CODEOWNERS`, the graph gate in `ggml-cuda.cu`, commit counts
  per file from 2026-07-23 to 2026-10-23, and `src/models/qwen35moe.cpp`, `qwen3moe.cpp`, `glm4-moe.cpp`,
  `deepseek2.cpp`, `openai-moe.cpp` for routing.

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): the decision log and Q38, Q39, Q43, Q50, Q76, Q78, Q82, Q87,
  Q88, Q92, Q99–Q101, Q108–Q110, Q116, Q117, Q120, Q121, Q67.
- [README](../../../README.md), [overview](../../overview.md), [hardware spec](../../mining_rig.md).
- Notes: [v2 synthesis](v2_synthesis.md), [kernel design](pascal_mmvq_integration.md) (§12, §16, §17),
  [release v2](release_v2.md), [llama.cpp-family engines](engines_llamacpp_family.md),
  [upstream status](upstream_pascal_patches.md), [rec](rec_engine.md),
  [Pascal forks and quants](pascal_forks_and_quants.md), [pipelined speculation](pipelined_speculation.md),
  [K-quant decode](kquant_decode_pascal.md), [small MoE models](models_small_moe.md),
  [large MoE models](models_large_moe_and_compression.md), [dense 27B](dense_27b_speedups.md),
  [Flint speed-ups](flint_speedups.md), [community reports](quant_speed_d_community.md).

External (P):
- [llama.cpp CONTRIBUTING.md](https://github.com/ggml-org/llama.cpp/blob/master/CONTRIBUTING.md), read 2026-10-24.
- [llama.cpp PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721), [#25479](https://github.com/ggml-org/llama.cpp/pull/25479),
  [#31047](https://github.com/ggml-org/llama.cpp/pull/31047), [#25919](https://github.com/ggml-org/llama.cpp/pull/25919),
  as summarized in [upstream status](upstream_pascal_patches.md), [tool-request sampling](tool_request_sampling.md) and
  [prompt processing](prefill_pascal.md).
- GitHub Docs: [Renaming a repository](https://docs.github.com/en/repositories/creating-and-managing-repositories/renaming-a-repository),
  [Renaming a branch](https://docs.github.com/en/repositories/configuring-branches-and-merges-in-your-repository/managing-branches-in-your-repository/renaming-a-branch),
  read 2026-10-24.
