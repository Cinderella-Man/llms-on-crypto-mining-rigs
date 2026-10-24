# Which of the Flint tricks carry over to NVIDIA V100 cards (Volta, sm_70): one 32 GB card, two over PCIe, NVLink (2026-10-24)

Scope: this project tuned Flint (40B-A2.5B MoE) for nine P106-090 mining cards. Used V100s (Volta, compute capability
7.0, 16 or 32 GB of HBM2 at a nominal 900 GB/s, FP16 tensor cores, no BF16) are cheap now, and many hobbyists run
LLMs on them. This note works out which of the project's speed and quality tricks would make Flint faster or better on
V100s, and by how much, for three setups:
- one V100 32 GB;
- two V100s over PCIe without P2P, as in a mixed box next to RTX 3090s;
- SXM2 boards with NVLink.

It answers five questions in order: does each trick transfer, and what limits Flint there; the V100 specifics (no BF16,
tensor cores and llama.cpp's matrix paths on sm_70, flash attention, why batched MoE generation was slow on V100s
today, HBM2 against what llama.cpp reaches); what fits on the cards; a ranked list for V100 owners with what to
upstream; and the commands a V100 owner can run to confirm each item.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (code, papers, measurements in a repository or PR) and secondary.
- Code citations are `branch:path:line` in the fork's worktrees on 2026-10-24: `v2` = `pascal-p106-v2` (`032b130f4`),
  whose CUDA backend outside the fork's own files is upstream `d77256999`; `comb` = `pascal-p106-v2-combined`
  (`b3052a4f8`); `pers` = `pascal-p106-v2-persistent` (`4b2831712`). `ggml-cuda/` stands for `ggml/src/ggml-cuda/`.
- "UD" is Unsloth's UD-Q4_K_XL, the quality bar. "T" is the time of one plain token. "Stock" is upstream llama.cpp.
- Nothing ran on the rig, the quant machine or a V100 for this note. No Flint decode has been timed on a V100: every
  Flint-on-V100 speed below is an estimate from similar MoE models measured by others.

Starting point:
- **On the rig** ([registry](../../../quants/experiments.md)): v2 decodes UD at 34.7–36.5 tok/s on six to nine P106-090
  cards, 2.2× stock. The gain is 87 % Pascal-only pieces (CUDA graphs on compute 6.1, the 2-warp launch shape, the Pascal
  matrix-vector kernels), 12 % FP32 activation sums with the MoE down rescale off (Q76, +9.6 %) and 1 % a generic fusion
  fix (Q50) ([fork scope](fork_scope.md)). On a P106 ~55 % of a token is weight streaming.
- **V100 hardware** (NVIDIA datasheet; Jia et al.): 80 SMs at up to 1,380 MHz (PCIe, 250 W) or 1,530 MHz (SXM2, 300 W),
  14 / 15.7 TFLOPS FP32, 112 / 125 TFLOPS FP16 tensor, 900 GB/s nominal and ~750 GB/s measured streaming (83 %), 6 MiB
  L2, up to 96 KiB shared memory per SM, an L2 TLB covering ~8 GiB in 2 MiB pages. PCIe Gen3 x16 (~12 GB/s measured
  host↔device); SXM2 adds NVLink 2.0, 25 GB/s per sub-link and direction, ~48 GB/s measured over two sub-links.
- **Similar MoE models on one V100 in llama.cpp** (others' measurements, section 3.2): Qwen3-30B-A3B Q4_K_M 110 tok/s
  at ~1k tokens of context; gemma4 26B-A4B Q4_0 124 tok/s in one-token steps, against 174 on an RTX 3090 with almost the
  same bandwidth.
- **Observed today on the 7-GPU machine used for this project's quality work** (3 × RTX 3090, 2 × V100-PCIE-32GB,
  2 × RTX 3080; no P2P; the V100s sit on x1 and x4 links): the drafter pilot's generation copies ran Flint batched,
  24–32 requests at a time; the V100 pair managed ~150–200 tok/s in total against ~470 on four 3090/3080 cards, and a
  3090 pair alone ~305 tok/s (1.1 M response tokens per hour with Q8_0,
  [staircase drafter](../../../bench/staircase_drafter/README.md)). The feature dump was slow over the narrow links.
  These are operational observations, not benchmarks.

## Verdict

- **On a V100, Flint's decode is bound by kernel count, not by HBM2.** MoE models of Flint's class decode at 25–35 % of
  900 GB/s in llama.cpp (Qwen3-30B-A3B ~220 GB/s, gemma4 26B-A4B ~270 GB/s, est. from their bytes per token), dense
  models at ~68 % (Gemma 4 31B Q4_K_M ~610 GB/s). A Flint token on stock llama.cpp runs ~1,300–1,900 kernels, each with
  a floor of ~2–3 µs on a V100 even inside a CUDA graph, so est. a third of a token streams weights and two thirds are
  fixed per-kernel cost: est. **85–115 tok/s for UD on one V100 32 GB** (8.5–11.5 ms per token; section 1.1). That
  inverts the P106 picture, so the ranking inverts too:
  - **launch-count cuts transfer best**: the MoE down rescale off (~470 launches), the persistent MoE-layer kernel (9 of
    10 launches per MoE layer), the norm writing Q8_1 and the shared-expert fusion;
  - **byte savings transfer weakly** (screened head, GPTQ head, F16 router, byte-priced mixes: 0–2 % each, est.);
  - **the P106 instruction-issue kernels mostly don't**: a V100 issues ~2.5× more instructions per byte of bandwidth
    than a P106-090 (est. 15.7 against 6.3), Volta has its own matrix-vector table, and CUDA graphs are already on for
    Volta upstream.
- **Top five for a V100 owner (est.):**
  1. **The MoE down rescale off: +8–15 % decode.** The fork's graph switch works on Volta without any kernel change:
     `LLAMA_NO_MOE_DOWN_RESCALE=tg` drops it at one token, where decode lives. That is safe for UD (stock K-quant
     one-token dot products compute block sums in FP32) and for the rescale-free files (K2h-r16, G3, S4-L0 and Q8_0 keep
     layers 46–48 in Q8_0); `=1` also drops it for batches, for Q8_0. Proof of safety: a bit-identical one-token
     perplexity.
  2. **MoE prompts on MMQ at every batch size, plus PR #31047: +30–70 % prefill.** Stock Volta leaves MMQ for a
     host-synchronized per-expert fallback at 64+ tokens (`v2:ggml-cuda/mmq.cu:423`); PR #31047's own V100 table shows
     prompts slowing from 812 tok/s at micro-batch 32 to 626 at 64. A V100 fork measured +55–70 % from forcing MMQ on a
     routed MoE.
  3. **Batched speculation pays on a V100: 1.1–1.2× overall with copy drafts, ~1.4× on edits, 1.4–1.65× with a
     trained Flint head.** A 2-token step costs 1.10 T and a 4-token step 1.43 T on a V100 (gemma4 26B-A4B, PR #31047's
     table) against 1.65–3.55 T on the P106s (Q19). Q19's no-go was a P106 result and does not transfer.
  4. **The persistent MoE-layer kernel (Q88): +5–15 %** (`GGML_CUDA_PERSISTENT_MOE=2`). In the fork, bit-exact through the
     compute_61 PTX on two V100s, never timed natively on Volta.
  5. **The combined build's launch fusions (norm writes Q8_1, Q100; shared-expert gate/up/GLU, Q50): +2–6 %.** Both
     are generic and on by default.
  - Together est. ×1.2–1.4 on one-token decode (~100–160 tok/s for UD on one V100), more on copy-heavy work with
    speculation.
- **Why batched MoE generation was slow on the V100s today** (est., in order):
  - Volta has no int8 tensor cores, so MMQ runs the expert tiles on `dp4a` where the 3090/3080s use int8 MMA
    (`v2:ggml-cuda/mmq.cu:408-424`).
  - The build predates PR #31047, so each touched expert in a 24–32-token step was computed as a 32-column tile for
    ~1.4 real tokens (~95 % padding), which costs several times more on `dp4a` than on tensor cores.
  - Every prompt micro-batch of 64+ tokens fell to the per-expert fallback with two host synchronizations per expert
    matrix (each `MUL_MAT_ID` call) and no CUDA graph (`v2:ggml-cuda/ggml-cuda.cu:2097-2177`).
  - A layer split over two cards leaves each card idle half of every step.
  - Fixes: PR #31047 (+15 % at 32 tokens, measured on a V100), forced MMQ for MoE, one copy per card where a copy fits.
    The one-token kernels (Pascal MMVQ, persistent kernel) do not touch 24–32-token steps.
- **No BF16 matters at a few FP16 points, not for the weights.** llama.cpp keeps activations in FP32. FP16 enters at:
  - the Q8_1 activation block sums, which Flint's layer-46/48 down-projection inputs overflow (channel RMS 25,600–48,400
    against FP16's 65,504);
  - the Q4_1/Q5_1 scale products, computed in half on fast-FP16 GPUs such as the V100 (`v2:ggml-cuda/vecdotq.cuh:159-163`,
    upstream issue #30950);
  - FP16 cuBLAS inputs for dense products of 64+ tokens, and F16-weight matrix-vector products accumulated in half2.

  So: keep the rescale for batches, never use `=1` with Q4_x/Q5_x down-projections, and treat FP16 engines (vLLM or
  transformers at `dtype=half`) as unsafe for Flint on V100s (est.). Our FP32-sum work matters more on Volta than on
  Pascal, because Volta also multiplies the scales in half.
- **What fits:**
  - **one 32 GB card:** UD (23.4 GiB) with ~200–250k tokens of f16 context, K2h-r16 (27.4 GiB) with ~64–96k; Q8_0
    (39.4 GiB) needs two cards, or expert offload at roughly half speed (est.);
  - **two 32 GB cards:** Q8_0 with the full 262,144-token context, or one copy of a Q4-class file per card for twice the
    throughput;
  - **16 GB cards:** at least two for UD.
- **Two cards and NVLink.** Decode on a layer split runs at one card's speed. Every per-card gain above applies, the
  host-staged hop costs est. 20–60 µs per token, and NVLink changes nothing for decode. NVLink matters for `-sm tensor`
  (Flint is supported): est. 1.3–1.7× prefill but only 0.9–1.15× decode, because both cards still run every kernel.
  Pipelined speculation (Q87) needs a layer split, so it applies only to two-card setups: est. 1.1–1.25× with copy and
  table guesses, ~1.45× with a drafter at p 0.7 (two stages cap it at 2×).
- **Upstream, in order:**
  1. MMQ for MoE on Volta at every batch size, an `n_experts` rule like RDNA3's in `ggml_cuda_should_use_mmq`.
  2. FP32 activation sums and FP32 scale products in the stock matrix-vector kernels. This fixes #30950 on every
     fast-FP16 GPU and gives the graph's rescale TODO (`v2:src/llama-graph.cpp:2523-2525`) a backend capability to test.
  3. The gate/up/GLU fusion fix.
  4. V100 measurements posted on PR #31047.

  Publish a V100 page with the commands of section 5 and ask owners to report.

## Ranked table

Gains are decode tok/s against stock llama.cpp at short context unless the row says otherwise; all est. except where a
measurement is named. "2 × PCIe" assumes a layer split without P2P (a model or context that needs two cards), "NVLink"
an SXM2 pair.

| # | Trick (registry) | Transfers to Volta? | 1 × V100 32 GB | 2 × PCIe, no P2P | 2 × NVLink | What it needs | Effort | How to verify |
|---|---|---|---|---|---|---|---|---|
| 1 | MoE down rescale off: FP32 activation sums (Q4, Q76) | Yes. The graph switch is backend-independent; UD's K-quant down-projections already sum in FP32 in stock one-token kernels | +8–15 % | same per card: +8–15 % | same | Fork's `LLAMA_NO_MOE_DOWN_RESCALE=tg` (any file at bar quality; `=1` for Q8_0); upstream: an FP32-sum PR | Hours | `-ub 1` perplexity identical with and without; tg128 A/B (5.2) |
| 2 | MoE prompt path: MMQ at every batch size + PR #31047 (Q108) | Partly. On stock Volta PR #31047 only acts below 64 tokens, where MMQ still runs | Prefill +30–70 %; 32-token steps +15 % (PR #31047's gemma4 prompt test at `-ub 32`, measured) | same | same; `-sm tensor` adds est. 1.3–1.7× prefill | `-DGGML_CUDA_FORCE_MMQ=ON` build of `comb`; upstream rule | A build | Prompt grid at `-ub 32…512`, two builds (5.5) |
| 3 | Batched speculation (Q19, Q64 and Q20 reversed) | Yes, unlike on Pascal: multi-token steps are cheap | Copy/n-gram drafts 1.1–1.2× overall (chat ~1.0, edits ~1.4, rewrites 2–3×); trained head 1.4–1.65× | same per copy | same | Upstream `--spec-type ngram-*`; later Q20's full-depth head through llama.cpp's EAGLE-3 path | Flags; weeks for the head | Step costs with llama-bench `-ub 1,2,3,4`; server A/B (5.6) |
| 4 | Persistent MoE-layer kernel (Q88) | Yes in code (`=2` runs on any NVIDIA GPU); not timed on Volta | +5–15 % | same per card | same | `pers` built for `70`, rescale off; covers 42 of UD's 47 MoE layers, all of K2h-r16's | A build | Self-check mode, then tg128 A/B (5.3) |
| 5 | Launch fusions: norm writes Q8_1 (Q100), shared-expert gate/up/GLU (Q50) | Yes, generic, on by default | +2–6 % together | same | same | `comb` | A build | `GGML_CUDA_NORM_Q8=0` A/B; fork against stock (5.2) |
| 6 | Several conversations (Q104) | The streams mode fills idle cards; a V100 batches well instead | `-np 4–8`: est. 2.5–3.5× total (gemma4: 8 tokens per step = 3.65× one) | one copy per card: 2× total | same | Stock llama-server | Flags | llama-batched-bench `-npl 1…64` (5.4) |
| 7 | GPU sampling for tool requests (Q55, Q67) | Yes, generic | Tool requests +3–6 % on x16, +8–13 % on x1/x4 links | same | same | Fork, not built (2–3 days) | Days | Same request with and without tools, ms per token (5.8) |
| 8 | First-finish race, k = 2 (Q93) | Partly: as a batch of two on one card | Per answer 1.1–1.2× (step 1.1 T, reasoning ends at 0.72–0.75 of the mean) | one sample per card: ~1.3× per answer | same | Client-side race on a `-np 2` server | A script | Time to first `</think>` (5.8) |
| 9 | Pipelined speculation + staircase drafter (Q87, Q96, Q20) | Only on a layer split | n/a | 1.1–1.25× (copy + table), ~1.45× at p 0.7 | same | `llama-pipespec` (`pascal-p106-v2-pipespec2`) | A build | `bench/pipespec/rig_run.sh` with two GPUs (5.7) |
| 10 | Screened head (Q91/Q92), GPTQ head (Q56), F16 router (Q113), GPTQ body (Q112) | Quality exactly; speed weakly | 0–2 %, 0.5–1.5 %, < 1 %, 0.5–2 % | same | same | Side file, files | Files | tg128 A/B |
| 11 | File choice from KLD and task checks (Q41, Q73, Q77, Q106) | Quality ranking yes; byte prices no (they are P106 rates) | K2h-r16 = UD ± 3 % speed (est.), KLD 0.105 against 0.120 | same | same | Files | — | Re-price per type with `test-backend-ops perf` |
| 12 | Pascal matrix-vector kernels, 2-warp shape (v1 core, Q38) | Mostly no: Volta is not issue-bound and uses its own table | −5 to +5 % when forced | same | same | `GGML_CUDA_MMVQ_PASCAL=2` | A flag | Op perf at Flint's shapes (5.3) |
| 13 | CUDA graphs on Pascal (#27721) | Already upstream for Volta (PR #25749); ik_llama.cpp keeps them off below Ampere | 0 % on mainline | 0 % | 0 % | Mainline-derived build | — | — |
| 14 | On-card mask (Q110/Q116), graph-reuse fix (Q109), packed uploads and fed tokens (Q120/Q121) | Yes, generic; small on fast hosts | ≤ 1–3 % | prefill at `-ub 128` est. up to 1.3× (Q109), else ≤ 3 % | same | `comb`, `pipespec2` | A build | Prompt grid at `-ub 128` with `LLAMA_GRAPH_REUSE_WAIT` |
| 15 | Hardware tuning (Q133, Q141) | P106-specific findings | not researched | — | — | — | — | — |

## 1. Does each trick transfer? (question 1)

### 1.1 What limits a Flint token on a V100

Est. per token for UD on one V100 32 GB, stock llama.cpp, short context, CUDA graphs on:

| Part | Time (est.) | Basis |
|---|---|---|
| Weight streaming: 2.21 GB at ~680–750 GB/s | 2.9–3.3 ms | UD's bytes per token ([v2 synthesis](v2_synthesis.md)); Jia et al.'s 750 GB/s streaming |
| Per-kernel floor of ~1,300–1,900 kernels (launch in the graph, ramp-up, one-wave tails; small kernels ≥ 2–3 µs) | 5–7 ms | NVIDIA's CUDA Graphs post: a 2.9 µs kernel costs 3.4 µs per step in a graph on a V100; Flint's launch counts ([Flint speed-ups](flint_speedups.md), [kernel note](pascal_mmvq_integration.md) section 8) |
| Of that, the MoE down rescale: 11 graph ops × 47 layers | 0.8–1.5 ms | ~470 launches at ~2–3 µs (P106: 2.6 ms measured, Q76) |
| Attention at ≤ 4k context (FP16 tile kernel, reads K/V once per KV head) | 0.2–0.5 ms | 14 full-attention layers, 28 KiB of KV per context token |
| Host: sample to next launch, on a desktop or server CPU | 0.1–0.3 ms | [host decode loop](host_decode_loop.md) §1.1, scaled from the 2-core Celeron |
| **Total** | **8.5–11.5 ms: 85–115 tok/s** | Qwen3-30B-A3B (48 layers, no shared expert, no rescale, similar bytes) decodes 110 tok/s, gemma4 26B-A4B 124 |

What this says about each kind of limit:
- **Bytes against 900 GB/s** set only the floor: est. ~300–340 tok/s if Flint streamed at Jia's 750 GB/s with no other cost.
  Byte savings buy ~1 % per 30 MB (est.).
- **Launches** set most of the token. Flint has 49 layers, 256 experts with a shared expert, and (on stock) the rescale,
  so its kernel count per token is at the high end for its active size.
- **FP16 overflow without BF16** is handled by the rescale (section 2.1). Removing it is the single largest launch cut.
- **Kernels tuned for Ampere:** on Volta, MMQ uses the Ampere tile table (`v2:ggml-cuda/mmq.cuh:255-257`) without any
  int8 MMA (`v2:ggml-cuda/mmq.cuh:188-201`, Turing and newer only). The matrix-vector kernels share Turing's table
  (`v2:ggml-cuda/mmvq.cu:118-119, 140-142, 536-551`). Volta-specific tuning exists only in forks (PXA, a Qwen V100 fork;
  section 1.2).
- **Clock:** a V100 runs at 1.38–1.53 GHz against an RTX 3090's ~1.7–1.9 GHz, so the same per-kernel overheads cost it
  more. gemma4 26B-A4B's one-token steps take 8.1 ms against 5.8 ms on a 3090 at nearly the same bandwidth (PR #31047's
  table).

### 1.2 Trick by trick

**Kernels and launches**

- **MoE down rescale off / FP32 activation sums (Q4, Q76).** llama.cpp scales Flint's expert down-projection input by a
  power of two for every backend (`v2:src/llama-graph.cpp:2528-2534`), because the Q8_1 activation blocks store their
  sum in FP16.
  - **Limit on Volta:** the same ~470 launches per token, relatively larger on a faster card.
  - **Which kernels read the FP16 sum at one token on Volta:** the stock Q4_0, Q4_1, Q5_0 and Q5_1 dot products
    (`v2:ggml-cuda/vecdotq.cuh:118-137, 143-171, 177-206, 208-236`). The K-quants compute it from the int8 values with
    `dp4a(0x01010101, u)` (Q5_K: `:586-595`), and Q8_0, Q6_K and IQ4_XS have no offset.
    - **UD at one token** is therefore safe without the rescale on stock Volta kernels (est., code reading): its expert
      down-projections are Q5_K, IQ4_XS, Q6_K and Q8_0 ([kernel note](pascal_mmvq_integration.md) section 17).
    - **The rescale-free files** (Q8_0; K2h-r16, G3 and S4-L0 keep layers 46–48 in Q8_0, where the overflow sits) are
      safe at one token too (est.). Their Q5_0 down-projections in layers 2–45 read the FP16 sum on Volta, but their
      inputs reach at most ~2,200 channel RMS, against 25,600–48,400 in layers 46 and 48; the `-ub 1` check of 5.2
      confirms it. `=1` (batches too) is safe by construction only where
      every down-projection is Q8_0, Q6_K, IQ4_XS or IQ4_NL (`v2:src/llama-graph.cpp:1670-1676`): on Volta that is the
      Q8_0 file.
    - **Batches** keep the rescale: MMQ's DS4 layout reads FP16 sums for Q4_0, Q4_1, Q5_1, Q4_K and Q5_K on Volta,
      which has no MMA data layout.
  - **Gain (est.):** +8–15 % on one card, the same per card on two. The fork's Pascal kernels are not needed for UD on
    Volta; for the makers' Q4_1 they are (`GGML_CUDA_MMVQ_PASCAL=2`), or a generic FP32-sum port.
  - **Needs:** the fork's switch (`8534e1b4d`, in v1 and v2; its `tg` value from `3733df1dd`, v2). Upstream has none.
  - **Prior art:** upstream PR #26223 (the rescale) and the TODO pointing to BF16 source precision (#30656), which a V100
    cannot use; issue #30950 (Q4_1 FP16 scale overflow on fast-FP16 GPUs); rec computes every activation sum in FP32 and
    never rescales ([rec note](rec_engine.md)).
- **Persistent MoE-layer kernel (Q88).** One launch for the 35 graph nodes from the FFN norm to `ffn_out`, instead of 10
  launches.
  - **Code:** switch `GGML_CUDA_PERSISTENT_MOE=2` (any NVIDIA GPU, `pers:ggml-cuda/moe-persistent.cu:947-950`). The grid
    is the resident maximum from the occupancy API, ×80 SMs on a V100 (`:886-897`). Combinations for K2h-r16, G3, S4-L0
    and 42 of UD's 47 MoE layers (`:841-853`). It needs the rescale off at one token.
  - **Gain (est.):** on a P106 it saved ~27 µs per MoE layer (1.16×, −1.3 ms per token, ~+5 %). On a V100 the same nine
    removed launches cost est. 1.5–3 µs each plus their ramps and tails, against a layer that streams ~18 MB in ~24 µs:
    est. −0.7–1.3 ms per token, +5–15 %.
  - **Risk on 80 SMs:** 2,944 warp items per layer meet ~1,280 resident warps, so the router → top-6 → expert waits
    weigh more than on 5 SMs.
  - **Prior art:** Cohere's MoE decode megakernel (1.58× vLLM at batch 1 on an H100), Hazy's megakernel, MPK
    ([breakthrough candidates](breakthrough_candidates.md) §2); nothing in llama.cpp or ik_llama.cpp.
- **Norm writes the Q8_1 copy (Q100) and the shared-expert gate/up/GLU fusion (Q50).**
  - Both generic and on by default in `comb`: ~290 fewer launches per token on one card (244 quantizations, ~48 adds) and
    141 fewer (3 per MoE layer).
  - On the rig they measured +1.8 % (within noise, Q117) and +0.7 % (Q50). On a V100, whose token is ~3× shorter, est.
    +2–6 % together.
  - **Prior art:** upstream fused shared experts into MMVQ only where the shared and routed widths match (#29184; Flint's
    are 1,024 and 512); ik_llama.cpp's `-fmoe` and `-muge` fuse up/gate.
- **Pascal matrix-vector kernels and the 2-warp shape (v1).**
  - They attack the P106-090's instruction issue: 6.3 thread-instructions per byte of bandwidth, against est. 15.7 on a
    V100 (80 SMs × 128 lanes × 1.38 GHz / 900 GB/s).
  - Upstream's loop needs 5–9 per byte, so on Volta it is not issue-bound.
  - The 2-warp rule edits the generic table, which Volta does not use (`v2:ggml-cuda/mmvq.cu:118-123`).
  - Forcing the kernels (`GGML_CUDA_MMVQ_PASCAL=2`, `comb:ggml-cuda/mmvq-pascal.cu:954-957`) keeps their coalesced
    128-bit staging and shared activation loads: est. −5 to +5 % on the matrix share (~a third of a token), unmeasured.
  - **Prior art for a Volta table:** PXA bundles four rows per block on V100 into a set worth +7 % plain decode on
    Gemma 4.
- **CUDA graphs on Pascal (#27721).** Upstream enables graphs from Volta on (`cc < GGML_CUDA_CC_VOLTA` disables them,
  upstream `ggml-cuda.cu:4798`; PR #25749 measured 41.9 → 47.7 tok/s on eight V100s in layer split). Nothing to gain on
  mainline. ik_llama.cpp still uses no graphs below Ampere, so a V100 owner on ik loses them.

**Numerics and files**

- **F16 router (Q113).** On fast-FP16 GPUs an F16-weight matrix-vector product with default precision accumulates in
  half2 (`v2:ggml-cuda/mmvf.cu:196-222, 654`), and Flint's router logits carry no precision request
  (`v2:src/llama-graph.cpp:2240-2243`).
  - Q113's "no measurable cost" was scored on RTX 3090s, which take the same half path, so it transfers to Volta as
    measured.
  - Speed: 49 MB less per token, < 1 % (est.).
- **Screened head (Q91/Q92), GPTQ head (Q56), GPTQ body (Q112), byte-priced mixes (Q73/Q77).**
  - Quality transfers exactly: the KLD and task numbers are properties of the file, and the compute-path noise between
    V100 and 3090 kernels is KLD 0.065 for one file (Q21), the routing-flip floor.
  - Speed transfers weakly. The screen saves 0.153 GB (est. 0.2 ms on a V100) but adds a top-256 selection and a
    gather; it gave +1.5–2.3 % on the P106s (Q125): est. 0–2 % on a V100.
  - The mixes' prices (Q8_0 1.7× Q4_1 per weight) are the fork's Pascal rates. On Volta every format runs on upstream's
    kernels at different relative speeds, so re-price with `test-backend-ops perf` before choosing a file for speed.
- **File choice.** UD for long context on one card; K2h-r16 for fidelity at 512 tokens (KLD 0.105 against 0.120) and
  K2h's task-level non-inferiority on single-turn tasks (Q41; K2h-r16 differs only in its F16 router), with ~64–96k of
  context on 32 GB; same speed within est. ±3 %.

**Speculation and the executor**

- **Batched speculation (Q19, Q64, Q43).** The project parked it because a verify step of 2.9 tokens cost 2.22 T and 4.05
  tokens 3.55 T on the P106s.
  - On a V100 multi-token steps are cheap: gemma4 26B-A4B (128 experts, top 8) runs 2 tokens per step in 1.10 T,
    4 in 1.43 T, 8 in 2.19 T (PR #31047's V100 table, prompt steps of one sequence).
  - Flint reads ~28 % of its bytes in routed experts, which adjacent tokens barely share (Q51), so est. 1.1–1.2 T for 2
    tokens and 1.25–1.45 T for 4.
  - **With copy-gated drafts:** 28 % of Flint's own output sits in copies of ≥ 8 tokens, where a 6-token match continues
    95–99 % of the time (Q64). Est. 1.1–1.2× overall: chat ~1.0, code ~1.1, whole-file edits ~1.4, rewrites 2–3×.
  - **A trained head:** Q20's full-depth variant (head U) is an EAGLE-style head. At per-token acceptance 0.6–0.7 with 2–3
    drafted tokens and a ~0.05–0.1 T head step, est. 1.4–1.65×.
  - **Prior art on V100s:** a Qwen V100 fork runs MTP speculation on one V100 and made it +32 % faster at 100k of
    context with Volta attention and weight-reuse kernels (27.6 → 36.5 tok/s); PXA ships a speculation gate for
    Gemma 4 on V100s. Flint has no MTP head.
- **Pipelined speculation (Q87, Q96) and the staircase drafter (Q20).** These need idle cards in a layer split.
  - **One V100:** nothing to fill; batched speculation does the job.
  - **Two cards** (Q8_0, or long context): S_eff ≈ 2, ceiling 2×. With the event model of
    [pipelined speculation](pipelined_speculation.md) (cycle ≈ 0.55 T, D = 1): Q96's guesser right on 40 % of tokens
    gives 1/(1 − 0.40 × 0.45) ≈ 1.22×; chat (~20 %) ~1.1×; edits (~60 %) ~1.37×; a p 0.7 drafter ~1.46× (est.).
  - The executor is generic and was checked bit-identical on RTX 3090s, so it runs on two V100s as built.
- **First-finish race (Q93).** On one V100 two samples run as a batch of two at ~1.1 T per step. Reasoning ends at
  0.72–0.75 of the mean, so est. 1.2–1.27× on the reasoning part and 1.1–1.2× per answer. On two cards with one copy each:
  ~1.3× per answer, at the cost of the second card's capacity. llama-server has no race mode; a client sends the
  request twice with different seeds and keeps the first to close `</think>`.

**Prompts, serving, host**

- **PR #31047 (Q108).** Applies to Volta (`comb` commit `4a8ab277a` drops the RDNA-only condition), but only where MMQ
  runs. On stock Volta that is below 64 tokens per product: +15 % at micro-batch 32 on a V100, 1.00× at 64 and above
  (PR #31047's table). Forced MMQ for MoE makes the rest of the curve reachable (section 2.4).
- **Graph-reuse fix (Q109), on-card mask (Q110, Q116), packed uploads and device-fed tokens (Q120, Q121).**
  - These remove host work that a 2-core Celeron behind PCIe 1.0 x1 links pays per token. On a V100 box with a fast host
    and x16 links the host mask copy at 131k is ~0.25 MB per step at ~12 GB/s, ~20 µs (est.).
  - So ≤ 1–3 % on one card. On two cards at `-ub 128` the reuse fix lets micro-batches overlap: est. up to 1.3× prefill
    there, nothing at `-ub` ≥ 256.
  - On x1/x4 links the mask copies grow ~4–15× and start to matter at depth (est.).
- **GPU sampling for tool requests (Q55, Q67).** Every tool request samples on the host: a 600,576-byte logits copy plus
  the host chain.
  - On x16 Gen3: est. 0.05 ms + 0.2–0.5 ms, 3–6 % of a V100 token.
  - On x1/x4 links: est. 0.2–0.75 ms more, 8–13 %.
  - Q67 is not built.
- **Several conversations (Q104).** The streams mode exists to fill idle cards. On a V100, batching works: gemma4's step
  of 8 tokens costs 2.19 T (3.65× throughput). Flint at `-np 4–8` est. 2.5–3.5× total. On two 32 GB cards one copy per
  card doubles that.

## 2. V100 specifics (question 2)

### 2.1 No BF16

- **Flint is a BF16 model**, and a V100 has no BF16 arithmetic (`fast_bf16_hardware_available` is Ampere and newer,
  `v2:ggml-cuda/common.cuh:334-337`). In llama.cpp the weights are quantized blocks with FP16 scales, activations,
  norms and accumulators are FP32, flash attention accumulates in FP32 (`v2:src/llama-graph.cpp:3014`), and cuBLAS
  writes FP32 outputs on Volta (`v2:ggml-cuda/ggml-cuda.cu:1584`). The KV cache is f16, as on the rig.
- **Where FP16 can still overflow on a V100:**

  | Point | Code | Flint's exposure | What to do |
  |---|---|---|---|
  | Q8_1 activation block sum `s` | `v2:ggml-cuda/quantize.cu` (FP16 `ds`) | Layer 46/48 down-projection inputs, channel RMS 25,600–48,400 | Rescale on for batches; at one token off only for files whose down-projections don't read `s` (row 1) |
  | Q4_1/Q5_1 scale products `__hmul2(dm4, ds8)` under `FAST_FP16_AVAILABLE` | `v2:ggml-cuda/vecdotq.cuh:159-163` | Q4_1 expert files (the makers' Q4_1); not on Pascal, where this runs in FP32 | Rescale on, or FP32 products (the fork's Pascal kernels, or a fix of #30950) |
  | Activations converted to FP16 for cuBLAS (dense products of 64+ tokens; expert products in the fallback) | `v2:ggml-cuda/ggml-cuda.cu:1694-1700` | Inputs with RMS ≤ 2,200 elsewhere; the down-projection is covered by the rescale | Keep the rescale for batches; `LLAMA_NO_MOE_DOWN_RESCALE=1` only with Q8_0/Q6_K/IQ4_XS/IQ4_NL down-projections |
  | F16-weight matrix-vector products in half2 | `v2:ggml-cuda/mmvf.cu:196-222` | An F16 router (Q113) | Measured harmless on RTX 3090s, the same path |

- **FP16 engines.** vLLM, SGLang or transformers on a V100 would run Flint in FP16, and its layer-46/48 SwiGLU outputs
  exceed 65,504 (est. from Unsloth's importance matrix; nothing was run). llama.cpp is the practical engine for Flint on
  Volta.
- **Our FP32-sum work matters more on Volta** than on Pascal: Volta adds the half-precision scale products, and its
  launch cost per rescale op weighs more in a shorter token.
- **Training a drafter on V100s** uses FP16 with loss scaling (`--fp16` in
  [staircase drafter](../../../bench/staircase_drafter/README.md)); the feature dumps come from llama.cpp, which runs on
  V100s.

### 2.2 Tensor cores and llama.cpp's matrix paths on sm_70

Volta's tensor cores do FP16 only (`mma.m8n8k4`). llama.cpp's `VOLTA_MMA_AVAILABLE` is defined only for code compiled
for exactly sm_70 (`v2:ggml-cuda/common.cuh:283-287`), and `volta_mma_available` requires that the highest compiled
architecture at or below the card's is 7.0 (`:361-363`). Paths for a quantized Flint tensor on a V100 built for `70`:

| Product, tokens per step | Kernel on Volta | Code | Note |
|---|---|---|---|
| Dense or expert, 1–8 | MMVQ (`dp4a`), Turing's table; Volta always uses MMVQ for `MUL_MAT_ID` ≤ 8 | `v2:ggml-cuda/mmvq.cu:288-293, 371-385, 536-551` | decode, verify steps, batches ≤ 8 |
| Dense or expert, 9–63 | MMQ on `dp4a` with the Ampere tile table; DS4 layout (FP16 sums) for Q4_0, Q4_1, Q5_1, Q4_K, Q5_K | `v2:ggml-cuda/mmq.cu:408-424`; `mmq.cuh:8, 188-201, 255-257` | no int8 tensor cores; PR #31047 acts here |
| Dense, ≥ 64 | dequantize to FP16 + cuBLAS on tensor cores, FP32 output | `mmq.cu:423`; `ggml-cuda.cu:1584, 1694-1700` | fine for prompts |
| Expert (`MUL_MAT_ID`), ≥ 64 | host-sorted fallback: ids copied to the host and back, two `cudaStreamSynchronize`, one product per used expert (MMVQ/MMQ/cuBLAS by its token count), no CUDA graph | `v2:ggml-cuda/ggml-cuda.cu:2097-2177` | the V100 dip at micro-batch 64 |
| F16/BF16/F32 weights (router, MMF) | MMVF ≤ 8; MMF on Volta MMA for F16/BF16; BF16 above 8 tokens through FP32 cuBLAS | `v2:ggml-cuda/mmf.cu:185`; `ggml-cuda.cu:1700-1707` | small for Flint |

- **ik_llama.cpp differs:** it uses MMQ for every n > 8 on sm_70 and no CUDA graphs below Ampere (its
  `docs/pascal-volta.md`, tested on a Titan V).
- **A compute_61-PTX-only build on a V100** (as in Q88's and Q108's checks) makes the host take every Pascal decision:
  no fast FP16, no Volta MMA in flash attention, Pascal MMQ tiles. It is a correctness tool, not a speed path.

### 2.3 Flash attention on Volta

- **Flint's attention** (head 128, 32 query and 4 KV heads, GQA 8, learned sinks) on a V100:
  - decode: the FP16 tile kernel, because 1 token × GQA 8 ≤ 16 (`v2:ggml-cuda/fattn.cu:672-679`: "On Volta tensor
    cores are only faster for sufficiently large matrices"); it reads K/V once per KV head;
  - prompts: the MMA kernel on Volta's tensor cores with 8 columns per KV head (`:222-245`).
- **It works with Flint:** Flint runs on V100s in llama.cpp (Q21, Q51), with `-fa on` in Q51's and the Q126/Q127 jobs'
  commands.
- **Attention at depth is not Flint's first problem on a V100.** At 32k its share of a token is est. 15–25 %, against
  16 % on the P106s at 32k (Q54); the Pascal attention kernel (Q59/Q60) is not needed (est.).
- **Volta-specific FA work in forks** targets head 256 (a Qwen V100 fork: +27 % prefill at 100k; PXA: +9 % decode); for
  Flint's head 128 est. small.
- **Quantized KV on Volta.** q8_0 KV halves the cache. In one V100 comparison llama.cpp with q8_0 KV decoded Qwen3-30B-A3B
  at 32.6 tok/s at 67k against 63.7 for Ollama with f16 KV (different engines, so est.). Keep f16 where it fits.

### 2.4 Why batched MoE generation was slow on the V100s today

- **The run:** Flint, batched (24–32 requests per step, continuous batching of new prompts); the V100 pair 150–200
  tok/s against ~305 for a 3090 pair and ~470 on four 3090/3080 cards. Causes, est., largest first:

  | Cause | Mechanism | Evidence | Fix | Expected (est.) |
  |---|---|---|---|---|
  | `dp4a` MMQ with padded expert tiles | 24–32 tokens × 6 experts touch ~110–140 of 256 experts per layer, ~1.4 tokens each; before PR #31047 each is a 32-column tile, ~95 % padding, on `dp4a` (~25 TOPS sustained on a V100) where Ampere uses int8 MMA | PR #31047's V100 table: 812 → 931 tok/s at micro-batch 32 with the PR; a V100 MoE PR measured MMQ at ~25 TOPS against ~53 TFLOPS for FP16 cuBLAS | PR #31047 (in `comb`) | +15 % measured on gemma4; est. up to +30 % with Flint's 256 experts |
  | Prompt micro-batches of 64+ tokens on the fallback | Host sort, two synchronizations and up to 256 small products per expert matrix, no graph | V100 prompts 812 tok/s at 32 → 626 at 64 (PR #31047's table) | `-DGGML_CUDA_FORCE_MMQ=ON`, or an upstream `n_experts` rule | prompts +30–70 % (a V100 fork: +55–70 % on a routed MoE) |
  | Layer split over two cards | One step at a time, each card idle half of it; hops staged through the host over x1/x4 | llama.cpp pipelines only prompt micro-batches | One copy per card where it fits (Q4-class files) | up to 2× total |
  | A unified KV cache (`-kvu`, if used) | Every token's attention reads all conversations' cells | [multi-conversation serving](multi_conversation_serving.md) | One KV stream per slot | depends on context |

- **Not the cause:** HBM2 bandwidth. A 32-token step reads ~53 % of all expert bytes, est. ~28 ms at 750 GB/s for
  Q8_0: room for est. ~1,100 tok/s.
- **What the project's one-token kernels can't do here:** the Pascal MMVQ kernels and the persistent kernel cover one
  token, and the multi-token Pascal kernels (Q78) 2–4.
- **Prior art for batched expert products on Volta:**
  - a V100 fork's expert-only forced MMQ (`GGML_CUDA_VOLTA_FORCE_MMQ=moe`, +55 % at 100k);
  - project-maya's hybrid (experts with ≥ 128 rows on FP16 cuBLAS, the rest on MMQ: a 27k prompt 41 → 33 s on one V100,
    KLD 0.015 against MMQ);
  - PXA's compact expert tiles (+10–40 % V100 prefill) and an occupancy setting for legacy types only (+5–7 %; K-quants
    measured 0.3–0.76× with it).

### 2.5 HBM2 against what llama.cpp reaches

| Workload on one V100 | Effective bandwidth | Share of 900 GB/s | Basis |
|---|---|---|---|
| Streaming microbenchmark | 750 GB/s | 83 % | Jia et al., Table 3.1 |
| Dense decode, Gemma 4 31B Q4_K_M (20.4 GB) at 29.9 tok/s | ~610 GB/s | ~68 % | DmNep/llm-bench-v100 (est., file bytes ≈ bytes per token) |
| MoE decode, gemma4 26B-A4B Q4_0 at 124 tok/s (~2.2 GB per token) | ~270 GB/s | ~30 % | PR #31047's table (est. bytes) |
| MoE decode, Qwen3-30B-A3B Q4_K_M at 110 tok/s (~2.0 GB per token) | ~220 GB/s | ~24 % | DmNep/llm-bench-v100 (est. bytes) |
| Flint UD, stock, est. 85–115 tok/s × 2.21 GB | ~190–255 GB/s | 21–28 % | this note (est.) |
| Flint UD with rows 1, 4 and 5 of the ranked table, est. ×1.2–1.4 | ~220–350 GB/s | 25–39 % | this note (est.) |

- **The gap is the MoE shape.** Six 512-row experts per matrix are 1–3 MB products, too small to fill 80 SMs at full
  bandwidth before the next kernel's ramp-up.
- **Fewer, larger kernels** (the persistent layer kernel, later one kernel per token as rec does on Apple GPUs) close it;
  faster products alone don't.
- **Address translation:** Volta's L2 TLB covers ~8 GiB (Jia et al.), and a 23–27 GiB Flint file sweeps random experts
  across all of it. Q99's question (do page walks slow real-token expert calls?) may apply on a 32 GB V100 too
  (unverified).

## 3. Flint on V100s (question 3)

### 3.1 What fits

Usable memory est. ~31.7 GiB on a 32 GB card and ~15.8 GiB on a 16 GB card (Jia et al. list 16,152 MiB). Overhead per
card est. 1.2–1.7 GiB: CUDA context and cuBLAS workspace, compute buffers at `-ub 512` including 293 MiB of logits rows,
the sliding-window cache. f16 KV is 28 KiB per context token in Flint's 14 full-attention layers: 0.9 / 1.75 / 3.5 /
7.0 GiB at 32k / 64k / 131k / 262k.

| Setup | File (GiB) | Context with f16 KV (est.) | Remarks |
|---|---|---|---|
| 1 × 32 GB | UD-Q4_K_XL (23.4) | ~200–250k; the full 262,144 at `-ub 128` or with q8_0 KV | The default: quality bar, long context |
| 1 × 32 GB | K2h-r16 (27.4) | ~64–96k | KLD 0.105; K2h (same body) single-turn non-inferior (Q41); F16 router, GPTQ Q5_0 head |
| 1 × 32 GB | makers' Q4_1 (24.2) | ~200k | below the quality bar (KLD 0.322) |
| 1 × 32 GB | S4-L0 (30.3) | a few thousand tokens | not practical |
| 1 × 32 GB | Q8_0 (39.4) with `--n-cpu-moe 12–13` | ~200k | experts of 12–13 layers on the CPU: est. 40–60 tok/s and slow prompts |
| 2 × 32 GB | Q8_0 (39.4), layer split | full 262,144 | KLD 0.052; one card's decode speed |
| 2 × 32 GB | one UD or K2h-r16 copy per card | as on one card | twice the throughput for two conversations |
| 2 × 16 GB | UD | ~150–190k | layer split |
| 3–4 × 16 GB | Q8_0 | ~100–140k on three, full on four | |

**Expert offload.**
- Each Q8_0 MoE layer's experts are ~0.8 GiB, so fitting Q8_0 on one card moves 12–13 layers' experts to host RAM.
- Decode then reads est. ~0.26 GB per token from host memory on the CPU, est. 7–13 ms: about half speed (est.).
- Prompt micro-batches copy the offloaded experts to the card, est. ~0.8 s per micro-batch at PCIe 3.0 x16.
- On the same V100, Qwen3-Coder-30B-A3B Q8_0 with 8 expert layers offloaded ran 35.7 tok/s and 414–524 tok/s prefill
  (DmNep).
- UD or K2h-r16 fully on the card is the better use of one 32 GB V100 (est.).

### 3.2 Community numbers for Flint and similar MoE models on V100s

No Flint measurement on a V100 was found (web search, 2026-10-24); vLLM's day-0 Flint post reports a GB200 only.

| Model, file | Hardware | Engine | Decode | Prompts | Source |
|---|---|---|---|---|---|
| Qwen3-30B-A3B Q4_K_M | 1 × V100-SXM2 32 GB on a PCIe adapter, driver 580 | llama.cpp, q8_0 KV | 110 tok/s at ~1k, 69.8 at 16k, 32.6 at 67k | 1,177–1,309 tok/s | DmNep/llm-bench-v100 (P) |
| same | same | Ollama 0.34.2, f16 KV | 133 at ~1k, 105 at 16k, 63.7 at 67k | 1,298–1,311 | same |
| Qwen3-30B-A3B UD-Q5_K_XL (20.27 GiB) | 1 × V100-SXM2-32GB | llama.cpp b8149, Windows | tg128 59.0 | pp512 736 | llama.cpp issue #19894 (P) |
| Qwen3.5-35B-A3B UD-Q5_K_XL | same | same | tg128 38.4 | pp512 570 | same |
| gemma4 26B-A4B Q4_0 | 1 × V100-PCIE-32GB | llama.cpp (PR #31047's base) | one-token steps 124 tok/s (RTX 3090: 174) | pp8192 812 at `-ub 32`, 626 at 64, 1,562 at 512 | PR #31047 (P) |
| Gemma 4 26B-A4B, Google QAT Q4_0 | 1 × V100 16 GB | PXA (llama.cpp-derived) | 111–113 tok/s plain decode | — | PXA `docs/LEVERS.md` (P, vendor) |
| Qwen3.6-35B-A3B PXQ4 | 2 × V100 | PXA | 84.3 tok/s, model resident | — | same |
| Qwen3.5-122B-A10B Q8_0 | 8 × V100 SXM2 | llama.cpp, graphs on | layer split 47.7, tensor split 58.3 | — | PR #25749 (P) |
| Ornith (routed MoE) | 1 × V100-SXM2 32 GB | a Volta llama.cpp fork | — | pp512 863 → 1,464 with forced MMQ (+70 %) | jusko-llama-volta-qwen3flash (P) |

The 59 tok/s Windows result against 110 tok/s on Linux for the same model class shows how much the platform and file
move these numbers; compare only within one setup.

## 4. The ranked list for V100 owners (question 4)

| Rank | What | Expected (est.) | Effort | Status |
|---|---|---|---|---|
| 1 | Our fork's `comb` built for `70`, run with `LLAMA_NO_MOE_DOWN_RESCALE=tg` (`=1` for Q8_0) | decode +8–15 %, plus +2–6 % from the default fusions | one build | rig-proven pieces; Volta untimed |
| 2 | Second build with `-DGGML_CUDA_FORCE_MMQ=ON` for prompt-heavy and batched work (agents, many requests) | prefill +30–70 %; batched steps of 24–32 +15 % from PR #31047 | one build | evidence from V100 forks and PR #31047's table |
| 3 | Copy-gated n-gram speculation in llama-server | 1.1–1.2× overall, ~1.4× on edits, 2–3× on rewrites | flags | upstream feature; Flint's step costs on Volta unmeasured |
| 4 | The persistent MoE-layer kernel (`GGML_CUDA_PERSISTENT_MOE=2`, `pers` branch) | +5–15 % | one build + self-check | correct through compute_61 PTX on V100s; native speed unmeasured |
| 5 | Two cards: one copy per card (Q4-class files), or Q8_0 layer split for fidelity; NVLink pairs try `-sm tensor` for prompts | 2× total; tensor split prefill 1.3–1.7× | flags | — |
| 6 | Tool requests: GPU sampling with a host grammar check (Q67) | +3–6 % on x16, +8–13 % on x1/x4 | days (fork) | not built |
| 7 | A trained one-token Flint head as a speculative drafter (Q20's head U through the EAGLE-3 path) | 1.4–1.65× | weeks | pilot training running on the quant machine |
| 8 | First-finish race of two samples | 1.1–1.2× per answer | a script | offline gate passed (Q93) |
| 9 | Pipelined speculation on a two-card layer split | 1.1–1.25×, ~1.45× with a drafter | a build | executor exact; Volta untimed |
| 10 | Byte savings: screened head, GPTQ head, F16 router | 0–2 % each | files | quality exact or near |

**What to upstream** (Volta reasons first; [fork scope](fork_scope.md) ranks the whole fork):
1. **MMQ for MoE on Volta at every batch size.** In `ggml_cuda_should_use_mmq`, return true for NVIDIA cards without
   int8 MMA when `n_experts` is large, as the RDNA3 branch does for `n_experts >= 64` (`v2:ggml-cuda/mmq.cu:445-452`). It
   is a few lines; the evidence is PR #31047's own V100 rows and the forks above; it makes PR #31047 apply to Volta prompt
   batches. Measure first on Flint and gemma4 (section 5.5).
2. **FP32 activation sums and FP32 scale products in the stock dot products** (Q4_0, Q4_1, Q5_0, Q5_1, Q8_1): the Pascal
   kernels' method, `d · Σq` from eight `dp4a` per block, no extra memory traffic.
   - It fixes #30950 on every fast-FP16 GPU.
   - It lets llama.cpp skip Flint's, Mistral4's and Mellum's rescale at one token behind a backend query, which the TODO
     at `v2:src/llama-graph.cpp:2523-2525` asks for.
3. **The gate/up/GLU fusion fix** (Q50, `032b130f4`): small, generic.
4. **V100 results on PR #31047 and #19894-style issues**, once an owner has run section 5. No PR slot needed.
5. **The persistent kernel stays in the fork** until a V100 and a P106 both confirm the gain; it hard-codes Flint's
   shapes.

**What to publish:** a V100 page (or a section of the README) with the build lines, the env switches, the measurement
script of section 5, the fits table, and a request for results with card type, link width, driver and build commit.

## 5. What a V100 owner can measure (question 5)

Conventions for the commands: `M` is the Flint file, `W` wikitext-2's `wiki.test.raw`, CUDA 12.x (12.9 recommended;
CUDA 13 cannot build for sm_70) with a 580-branch driver. Warm the card with one run before timing; run one thing at a
time; report tok/s with card, link width (`nvidia-smi -q | grep -A3 'Link Width'`), clocks and power limit.

### 5.1 Builds

```bash
git clone --branch pascal-p106-v2-combined https://github.com/Cinderella-Man/llama.cpp v2
cmake -S v2 -B v2/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build v2/build -j --target llama-bench llama-server llama-perplexity llama-batched-bench test-backend-ops
cmake -S v2 -B v2/build-mmq -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 -DGGML_CUDA_FORCE_MMQ=ON
cmake --build v2/build-mmq -j --target llama-bench llama-batched-bench
git clone https://github.com/ggml-org/llama.cpp stock && git -C stock checkout d77256999   # the fork's upstream base
cmake -S stock -B stock/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=70 && cmake --build stock/build -j --target llama-bench
# the persistent kernel: the same cmake lines on the fork's branch pascal-p106-v2-persistent, into p/build
```

### 5.2 Rescale and fusions (rows 1 and 5)

```bash
B=v2/build/bin
P="-m $M -ngl 99 -fa on -f $W -c 512 --chunks 4 -ub 1"
$B/llama-perplexity $P --kl-divergence-base /tmp/base.kld
LLAMA_NO_MOE_DOWN_RESCALE=tg $B/llama-perplexity $P --kl-divergence-base /tmp/base.kld --kl-divergence
rm /tmp/base.kld
for e in "" "LLAMA_NO_MOE_DOWN_RESCALE=tg" "LLAMA_NO_MOE_DOWN_RESCALE=tg GGML_CUDA_NORM_Q8=0"; do
  env $e $B/llama-bench -m $M -ngl 99 -fa on -p 512 -n 128 -r 5 -o md
done
stock/build/bin/llama-bench -m $M -ngl 99 -fa on -p 512 -n 128 -r 5 -o md
```

- The second perplexity must equal the first to every printed digit, with KLD at the storage floor and no NaN. For
  Q8_0 with `=1`, add a 512-token-batch perplexity (`-ub 512`) to the check.
- Expected (est.): +8–15 % from the rescale, +2–6 % for the fork's defaults against stock (without `GGML_CUDA_NORM_Q8=0`).

### 5.3 Persistent kernel and Pascal kernels (rows 4 and 12)

```bash
E="LLAMA_NO_MOE_DOWN_RESCALE=tg"
env $E GGML_CUDA_PERSISTENT_MOE=2 GGML_CUDA_PERSISTENT_MOE_CHECK=1 p/build/bin/llama-perplexity $P   # log: 0 layers differ
for k in 0 2; do env $E GGML_CUDA_PERSISTENT_MOE=$k p/build/bin/llama-bench -m $M -ngl 99 -fa on -p 512 -n 128 -r 5 -o md; done
for k in 0 2; do GGML_CUDA_MMVQ_PASCAL=$k $B/test-backend-ops perf -o MUL_MAT_ID; done
for k in 0 2; do env $E GGML_CUDA_MMVQ_PASCAL=$k $B/llama-bench -m $M -ngl 99 -fa on -p 512 -n 128 -r 5 -o md; done
```

Expected (est.): persistent +5–15 %; Pascal kernels within ±5 %.

### 5.4 Step costs and batching (rows 3 and 6)

```bash
$B/llama-bench -m $M -ngl 99 -fa on -p 2048 -n 0 -ub 1,2,3,4,8 -r 2 -o md
$B/llama-batched-bench -m $M -ngl 99 -fa on -c 65536 -npp 512 -ntg 128 -npl 1,2,4,8,16,32,64
v2/build-mmq/bin/llama-batched-bench -m $M -ngl 99 -fa on -c 65536 -npp 512 -ntg 128 -npl 16,32,64
```

- From the first line, time per step = tokens per step / tok/s. Divided by the one-token step, that is the verify cost
  a speculative step pays (est. 1.1–1.2 for 2 tokens, 1.25–1.45 for 4).
- The batched lines give total throughput per number of conversations and show the drop at 64.

### 5.5 Prompt path (row 2)

```bash
for b in v2/build v2/build-mmq; do
  $b/bin/llama-bench -m $M -ngl 99 -fa on -p 512,2048,8192 -n 0 -b 8192 -ub 32,64,128,256,512 -r 2 -o md
done
GGML_CUDA_MMQ_MOE_LEGACY=1 v2/build-mmq/bin/llama-bench -m $M -ngl 99 -fa on -p 8192 -n 0 -b 8192 -ub 32,128,512 -r 2 -o md
```

Expected: `build` dips at `-ub 64`; `build-mmq` keeps rising, est. 1.3–1.7× at 512. The last line is PR #31047 switched
off.

### 5.6 Speculation in the server (row 3)

```bash
$B/llama-server -m $M -ngl 99 -fa on -c 65536 -np 1 --port 8080 \
  --spec-type ngram-map-k --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 2
```

`size-n` is the lookup length and `size-m` the draft length; try `size-m` 2 and 4, since a V100's longer steps are cheap.
Send the same requests (a chat question, a code task, a whole-file edit, a rename) at temperature 0 with and without the
`--spec-*` flags. Compare `timings.predicted_per_second`, and check that the texts are identical.

### 5.7 Two cards (rows 5 and 9)

```bash
CUDA_VISIBLE_DEVICES=0,1 $B/llama-bench -m flint-Q8_0.gguf -ngl 99 -fa on -p 512,2048 -n 128 -sm layer -r 3 -o md
CUDA_VISIBLE_DEVICES=0,1 $B/llama-bench -m flint-Q8_0.gguf -ngl 99 -fa on -p 512,2048 -n 128 -sm tensor -r 3 -o md   # NVLink, NCCL build
```

- For pipelined speculation, build `pascal-p106-v2-pipespec2` and run
  [bench/pipespec/rig_run.sh](../../../bench/pipespec/rig_run.sh) with `GPUS=0,1`.
- `same_as_plain` must read `yes` in every row; compare `cont-d1-*` against `plain-*`.

### 5.8 Tool requests and the race (rows 7 and 8)

- **Tool requests:** run a server with `-bs` and send the same long request with and without a `tools` array. The
  difference in ms per token is what Q67 would recover.
- **The race:** send one reasoning request twice in parallel to a `-np 2` server with different seeds, record when each
  closes `</think>`, and compare the first finish with a single request's time.

## Already covered

| Topic | Where |
|---|---|
| Every fork change classified as Pascal, generic, MoE or Flint, and other GPUs' share | [fork scope](fork_scope.md) |
| The same question for RTX 3090s | the round's RTX 3090 note, `transfer_rtx3090.md` |
| Designs and P106 measurements of each trick | [breakthrough candidates](breakthrough_candidates.md), [pipelined speculation](pipelined_speculation.md), [pipeline drafter](pipeline_drafter.md), [prompt processing](prefill_pascal.md), [host decode loop](host_decode_loop.md), [kernel integration](pascal_mmvq_integration.md), [GPTQ for the body](gptq_body.md), [v2 synthesis](v2_synthesis.md) |
| rec's FP32 sums and persistent kernel | [rec note](rec_engine.md) |
| llama.cpp and ik_llama.cpp split modes, NCCL, P2P, the ik Pascal/Volta document | [llama.cpp-family engines](engines_llamacpp_family.md) |
| V100 tensor-split and graph evidence (PR #25749), P100/CMP comparisons | [multi-GPU Pascal performance](multigpu_pascal_performance.md) |
| PXA, pxq_llama and other Pascal/Volta community engines | [community quant speed](quant_speed_d_community.md) |

## Not verified

- **No Flint run on a V100 has been timed.** The 85–115 tok/s baseline and every gain on Volta are estimates from Qwen3,
  gemma4 and Gemma 4 measurements by others and from this project's P106 results.
- The split of a V100 token into a third streaming and two thirds per-kernel cost; the per-kernel floor of 2–3 µs is
  NVIDIA's single-kernel figure, not a profile of a Flint token.
- That `LLAMA_NO_MOE_DOWN_RESCALE=tg` is safe for UD on stock Volta kernels: code reading of the K-quant dot products and
  of UD's down-projection types; the perplexity identity of 5.2 decides it.
- The persistent kernel natively on sm_70 (checked through the compute_61 PTX on V100s and natively on sm_86 only) and
  its speed with 80 SMs.
- The Pascal kernels' speed on Volta.
- Flint's multi-token step costs on a V100: gemma4 26B-A4B's are the proxy.
- The causes of today's slow batched generation: the run logged neither its full flags (`-kvu`, slots) nor its prompt
  share, so the ranking is code reading plus PR #31047's V100 table.
- Forced MMQ for Flint's prompts: evidence from Ornith, gemma4 and other V100 forks.
- `-sm tensor` for Flint on an NVLink pair; the internal AllReduce exchanges activations in BF16 by default
  (`v2:ggml-cuda/allreduce.cu` init), so its output differs from a layer split in the last bits.
- FP16 overflow in FP16 engines: inferred from the importance matrix's channel RMS; nothing was run.
- TLB effects with a 23–27 GiB file.
- The 16 GB and expert-offload rows of the fits table: arithmetic.
- Third-party numbers: DmNep's repository and PXA are single sources; issue #19894 ran on Windows.

## Sources

NVIDIA and architecture (P)
- NVIDIA Tesla V100 datasheet,
  [images.nvidia.com/…/tesla-volta-v100-datasheet-letter-fnl-web.pdf](https://images.nvidia.com/content/technologies/volta/pdf/tesla-volta-v100-datasheet-letter-fnl-web.pdf)
- Z. Jia, M. Maggioni, B. Staiger, D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking",
  [arXiv 1804.06826](https://arxiv.org/abs/1804.06826): Table 3.1 (bandwidth, TLBs, shared memory), Tables 5.1–5.2 (P2P,
  NVLink, host links)
- NVIDIA, "Getting Started with CUDA Graphs", [developer.nvidia.com/blog/cuda-graphs](https://developer.nvidia.com/blog/cuda-graphs/)
  (per-kernel times on a Tesla V100-SXM2-16G)

llama.cpp and forks (P)
- Fork worktrees `pascal-p106-v2` (`032b130f4`, upstream `d77256999` underneath), `pascal-p106-v2-combined`
  (`b3052a4f8`), `pascal-p106-v2-persistent` (`4b2831712`): `ggml-cuda/common.cuh`, `mmq.cu`, `mmq.cuh`, `mmvq.cu`,
  `mmvf.cu`, `mmf.cu`, `vecdotq.cuh`, `fattn.cu`, `ggml-cuda.cu`, `allreduce.cu`, `mmvq-pascal.cu`, `moe-persistent.cu`;
  `src/llama-graph.cpp`, `src/llama-arch.cpp`, `common/arg.cpp`
- [PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047) "CUDA: linearize MoE data layout for MMQ/MMF" (open; V100-PCIE-32GB and RTX 3090 tables)
- [PR #25749](https://github.com/ggml-org/llama.cpp/pull/25749) (CUDA graphs for Volta and Turing; 8 × V100 results)
- [PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) (CUDA graphs on Pascal)
- [Issue #30950](https://github.com/ggml-org/llama.cpp/issues/30950) (Q4_1 FP16 scale-product overflow on fast-FP16 GPUs), [issue #30951](https://github.com/ggml-org/llama.cpp/issues/30951) (FP16 cuBLAS fallback overflow; Volta already uses FP32 output)
- [Issue #19894](https://github.com/ggml-org/llama.cpp/issues/19894) (Qwen3-30B-A3B and Qwen3.5-35B-A3B on a V100-SXM2-32GB)
- ik_llama.cpp, [docs/pascal-volta.md](https://github.com/ikawrakow/ik_llama.cpp/blob/main/docs/pascal-volta.md) (Volta paths, no CUDA graphs below Ampere, CUDA 12.x)
- [jackjusko/jusko-llama-volta-qwen3flash](https://github.com/jackjusko/jusko-llama-volta-qwen3flash) (V100 fork: forced MMQ for routed MoE, Volta FA, MTP numbers)
- project-maya [PR #254](https://github.com/mw00/project-maya/pull/254) (Volta MoE prompts: MMQ against FP16 cuBLAS per expert)
- PXA, [docs/LEVERS.md](https://github.com/poisonxa16/pxa/blob/main/docs/LEVERS.md) (V100 levers; vendor measurements)

Measurements by others (P)
- [DmNep/llm-bench-v100](https://github.com/DmNep/llm-bench-v100) (one V100 32 GB: llama.cpp and Ollama, MoE and dense)
- vLLM, "Day-0 Support for Flint on vLLM", [vllm.ai/blog/2026-10-21-flint](https://vllm.ai/blog/2026-10-21-flint) (no V100 data)

This project (P)
- [Quant registry](../../../quants/experiments.md): Q4, Q19, Q20, Q21, Q35, Q41, Q43, Q50, Q51, Q54, Q55, Q56, Q64, Q67,
  Q73, Q76, Q77, Q78, Q87, Q88, Q91, Q92, Q93, Q96, Q99, Q100, Q104, Q106, Q108–Q113, Q116, Q117, Q120, Q121, Q125, Q141
- Notes: [fork scope](fork_scope.md), [v2 synthesis](v2_synthesis.md), [breakthrough candidates](breakthrough_candidates.md),
  [pipelined speculation](pipelined_speculation.md), [pipeline drafter](pipeline_drafter.md),
  [prompt processing](prefill_pascal.md), [host decode loop](host_decode_loop.md), [GPTQ for the body](gptq_body.md),
  [rec note](rec_engine.md), [llama.cpp-family engines](engines_llamacpp_family.md),
  [kernel integration](pascal_mmvq_integration.md), [multi-GPU Pascal performance](multigpu_pascal_performance.md),
  [multi-conversation serving](multi_conversation_serving.md), [Flint speed-ups](flint_speedups.md),
  [community quant speed](quant_speed_d_community.md)
- [staircase drafter](../../../bench/staircase_drafter/README.md) (3090-pair generation rate, FP16 training on V100s),
  [EAGLE-3 training plan](../../../bench/eagle3_flint/TRAINING_PLAN.md) (V100 build for 7.0, BF16 limits),
  [expert overlap](../../../bench/expert_overlap/README.md) (Q51 on two V100s)
- The drafter pilot's generation on the 7-GPU machine, 2026-10-24: the V100 pair's batched rate (~150–200 tok/s against
  ~470 on four 3090/3080 cards), an operational observation, not a benchmark
