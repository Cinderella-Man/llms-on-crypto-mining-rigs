# Keeping a k-token verify pass cheap on a hybrid MoE: what other engines do, and what it means for a 2.63× pass on nine P106s (2026-11-04)

Qwen3.8-Flash-Next is a hybrid model: 48 layers (36 Gated DeltaNet, 12 attention with a sparse-attention indexer), four
hyper-connection streams, 512 routed experts with 10 active plus a shared expert. On nine P106-090 cards (compute 6.1,
5 SMs, `dp4a`, no tensor cores, PCIe 1.0 x1, a layer split that runs one card at a time) Miner Llama's `-ladder3` build
(`7bc1552f4`, the night's kernels) decodes one token of the Swift-1.5 Q2_0 file in 82.4 ms. A pass of k tokens, which a
speculative verify needs, costs 82.4 / 122.1 / 149.4 / 173.0 / 217.1 / 249.1 / 280.1 / 319.1 ms at k = 1–8
([k-curve](../../data/2026-11-04_kcurve_p106/README.md)): **T(5)/T(1) = 2.63**, where the target is ≤ ~1.3. The k-column
dense kernels and the multi-token expert kernel ([Q591](../../data/2026-11-04_kcol_dense/README.md),
[R7](../../data/2026-11-04_mtok_experts/README.md)) changed a 5-token pass of the retyped file by only −2 %.

This note asks how vLLM, SGLang, TensorRT-LLM, FlashInfer, flash-linear-attention (FLA) and llama.cpp keep a verify pass
cheap on models of this kind, in five parts: the Gated DeltaNet layers, the routed experts, attention over a draft
block, the operations without weights, and the owner's idea of routing once per verify block (Q860). It then prices
each finding on the rig. It builds on [MoE speculation](moe_speculation_union.md) (the union of experts, trees and forks
on hybrid models), [Flash-Next acceptance](flash_next_acceptance.md), [speculation plumbing](speculation_plumbing.md),
[pipelined speculation](pipelined_speculation.md), [Flash-Next prefill](flash_next_prefill.md) and
[the floor note](flash_next_floor_and_engine.md) (the bandwidth floor of a token and of the k-curve's slope), and does not
repeat them. Sources were read on 2026-11-04; the fork's behaviour is read from `-ladder3`'s source. Nothing was run,
built or downloaded. The k-pass profile (Q822, [kpass_profile](../../data/2026-11-04_kpass_profile/)) was still running
on the rig; section 1 is the bottom-up estimate it replaces.

Conventions:
- "est." marks this note's own arithmetic or inference; the basis is next to it. Numbers without it are measured on the
  rig or quoted from the linked source.
- **T(k)** is the wall time of a k-token pass (llama-bench `-p k -n 0`, logits for the last row only), T1 = T(1).
  **Slope** is the cost of one more token in a pass, in ms. **u(k)** is the number of distinct routed experts per layer
  among k tokens. **τ** is accepted tokens per step, the bonus token included; **S** = τ·T1 / (draft time + T(d + 1)) is
  the speed-up of a step with d drafted tokens over decoding without a draft.
- Q-numbers are rows of the [registry](../../../quants/experiments.md).

## Summary

1. **About half of the slope is not in any part that can be priced from the parts.** The four extra tokens of a
   5-token pass add 134.7 ms. The parts priced bottom-up add est. 63–78 ms: routed experts +23 (llama-bench's random
   token ids touch ~48 distinct experts per layer), dense products +22–27 (the k-column bench on a P106), card hops
   +7–9, router +3.5, operations without weights +4.5, recurrent state +2, attention +1, the n-gram table's SSD reads
   0–8. That leaves **57–72 ms unaccounted for**. Section 1.
2. **The curve has two steps where code paths change.** k = 1 → 2 costs +39.7 ms and k = 4 → 5 +44.1, against +23.6–27.4
   between them. At k ≥ 2 several one-token-only paths drop out (the fused gate/up products, the shared expert fused
   with the routed experts, the one-token Pascal attention kernel); at k = 5 the BF16 and k-column kernels change their
   launch shapes. Together est. up to ~33 ms of the unaccounted part; llama-bench's prompt path (Q589: 20–30 % more for
   the dense products than the server's verify) may hold another est. 15–25.
3. **On GPUs a 4-token verify of an MoE costs about 1.25× a token, and the reason does not transfer.** Cohere measured
   1.25× at batch 1 on its 128-expert MoE (vLLM). The dense parts are flat there because tensor cores give the extra
   columns for free and a large fixed part (attention, communication, launches) is shared. Every engine read for this
   note reads each weight once per pass, routes the draft block's tokens through grouped expert kernels, attends with
   all k queries in one call, and captures the verify in a CUDA graph per width. The rig already has the first two and
   per-size graphs; what it lacks is the arithmetic, and its hops through the host scale with k.
4. **An exact 5-token pass at ≤ 1.3× a token is at the floor of what the rig can do.** The floor of the slope is est.
   6.2–7.6 ms a token with Q8_0 hops (distinct experts 2.3–2.7, hop bytes 0.9, `dp4a` arithmetic ~2, operations without
   weights 1–2), so T(5)/T(1) ≥ 1.30–1.37 at today's T1 (1.42–1.49 with F32 hops). Kernels at twice their floor, as the
   one-token pass runs today, give est. 1.6–1.8×. The ratio also rises whenever T1 falls; the slope in ms, and tok/s,
   are the targets that hold. Section 8.
5. **The Gated DeltaNet layers are not where the slope is.** Every engine verifies k ≤ 8 tokens of a linear-attention
   layer with the recurrence inside one kernel (FLA's `fused_recurrent_gated_delta_rule`, FlashInfer's
   `gated_delta_rule_mtp`, llama.cpp's fused `GATED_DELTA_NET`) or with chunked WY forms that pay off only on tensor
   cores. Their cost is the per-token state snapshots: ReplaySSM measured a snapshot-based verify kernel at 4.85× the
   decode kernel at 6 tokens and its replay at 1.27–1.72×. On the rig the recurrence adds ~2 ms at k = 5, and snapshots
   ~1.8 ms per row in a server verify (est. −5–7 ms at k = 5 with Q686's recorded inputs). No measurement on a GPU
   without tensor cores was found. Section 3.
6. **Upstream's multi-token MoE kernel loses to its tiled kernel from 5 tokens on a P40**, the only Pascal data found:
   llama.cpp #20905 gains 1.13–1.59× at 2–4 tokens and loses (0.74–0.97×) at 5–8 tokens against master, which switches
   to the `dp4a` tile kernel (MMQ) there. The rig's own 5–8-column kernels were never compared with MMQ (Q871).
   Section 4.
7. **Adjacent tokens share only part of a sparse-attention selection.** In DeepSeek-V3.2, consecutive decode steps share
   35–50 % of their top-2,048 cells in layers 20–60 and 1–2 % in layers 0–1 (NVIDIA's GVR paper). A union gather over a
   5-token block (Q643) would read est. 3.0–3.6× one token's cells instead of 5×. The rig's T(k) is measured at an
   empty context, and 2–8 query tokens run on the stock attention kernels, so the attention's slope at agent depth is
   unmeasured (Q873). Section 5.
8. **Routing once per verify block (Q860) is lossy in two ways** and saves est. 14.5–17 ms at k = 5 in a real verify
   (23 ms in llama-bench), 7–11 % of the pass: T(5) → ~194–202 ms, 2.35–2.45×. Tokens 2..k are verified by a different
   model, and the keys, values and recurrent state of every accepted draft are computed with the wrong experts and stay
   in the cache. The cut (34–48 distinct experts → 10, −71 to −79 %) is deeper than any published operating point;
   cuts of a similar share cost 9–15 points on GSM8K in Lynx (batch 32), and Swift's router already changes greedy text
   when only its lowest-weight 14 % of reads is dropped (Q618). Section 7.
9. **What the slope is worth:** at DFlash2's measured acceptance on agent requests (τ 3.11 at three drafted tokens), a
   step gives 16.1 tok/s today, est. 19.2 at a slope of 20 ms, 22.5 at 12 and 24.6 at the exact floor of 8. Section 8.
10. **Ideas, ranked by est. ms saved at k = 5** (section 9): find the unaccounted half with Q822's profile plus a k = 2
    arm (Q870); retune the k-column launch shapes (Q823, est. −19 ms); route once (Q860, −14.5–17, lossy); Q8_0 hops
    (Q750, −12); the one-token-only paths made k-aware (Q870, −5–14); partial route-once with two own experts per token
    (Q874, −10–12, lossy); MMQ at 5–8 columns (Q871, 0–10).

## 1. The pass on the rig, bottom-up

### 1.1 Where the four extra tokens of a 5-token pass go (est.)

T(5) − T(1) = 217.1 − 82.4 = **134.7 ms** (plain Swift Q2_0, `-ladder3` defaults, median of five; standard deviation
2–3 ms a pass).

| Part | + ms at k = 5 (est.) | Basis |
|---|---:|---|
| Routed experts, each distinct expert read once ([R7](../../data/2026-11-04_mtok_experts/README.md)) | +23.1 | u(5) = 48.1 for llama-bench's random ids against 10 at k = 1, at 0.606 ms per distinct expert per layer summed over the layers ([R16](../../data/2026-11-03_kernel_timing/README.md): 6.06 ms for 10). A real verify's tokens share more: u ≈ 34–38 (68–77 % of independent routing), +14.5–17 |
| Dense products: k-column kernels, Pascal BF16 kernel | +22 to +27 | Q591's k-column bench on a P106 summed over Flash-Next's shapes (Q823): 65.1 → 89.0 ms a pass at the default launch shapes, 41.4 → 70.0 at the best ones; less the output layer's +1.7–2.0 (llama-bench computes one row). The bench has the retype's Q8_0 hyper-connection rows; the plain file's BF16 rows add est. the same within 1 ms (8.93 → ~16 ms from one to five tokens, R16) |
| Card hops, F32, eight boundaries | +7 to +9 | a one-token hop 0.36–0.455 ms, a 4-token hop 1.11–1.29 ms ([C140](../../data/2026-11-04_megakernel_rig/README.md), Q820) |
| Router (F32, 2,560 × 512 per layer) | +3.5 | [Q583](../../data/2026-11-03_verify_cost/README.md): 2.5 → 6.0 ms |
| Recurrent state and convolution | +2 | Q583: 4.3 → 6.2 ms; a sequential loop over the tokens (section 3) |
| Operations without weights | +4.5 | Q583: 26.3 → 30.8 ms |
| Attention and indexer at an empty context | +1 | Q583: 0.7 → 1.7 ms |
| N-gram table rows (host, SSD) | 0 to +8 | 11–14 random 4 KiB reads a token ([Q214](../../data/2026-11-03_table_storage/README.md)) at 152 µs: 6.7–8.5 ms for four tokens one read at a time, ~2.4 at queue depth 32, 0 if llama-bench's ids repeat across repetitions so that the rows are cached |
| **Sum of the parts** | **+63 to +78** | |
| **Measured** | **+134.7** | |
| **Not accounted for** | **57–72** | |

- **The kernels did what they were built to do, and the slope did not follow.** Their bench numbers make the dense
  products of five tokens 1.37× those of one at the default shapes, and the routed experts follow the union. The parts
  they do not touch (hops, router, state, host) add only ~17–27 ms more. Half the slope is elsewhere.
- **Where to look, from the curve's shape.** The steps are +39.7, +27.4, +23.6, +44.1, +32.0, +31.0 and +39.1 ms from
  k = 1 to 8. A cost linear in k would give steps of equal size; the two large steps sit where code paths change:
  - **At k = 2 the one-token-only paths drop out** (from `-ladder3`'s source): `ggml_cuda_should_fuse_mul_mat_vec_f/q`
    fuse a gate/up product with its GLU only for one column; the shared-expert fusion beside the routed experts
    (`ggml_cuda_match_shared_expert`) needs a one-row input; `fattn-pascal` takes one query token, and the Pascal tile
    kernel starts at 9 query tokens (3 with a Q8_0 cache), so 2–8 query tokens run on the stock kernels. The excess of
    the first step over the mean of the next three (25.5 ms) is 14.2 ms: est. 5–14 ms of fixed cost that appears at
    every k ≥ 2.
  - **At k = 5 the launch shapes change:** the BF16 kernel takes 4 rows per lane group at 3–4 tokens and 8 at 5–8
    ([C70](../../data/2026-11-04_kcol_dense/README.md)), and the k-column kernels' default shapes lose to other shapes
    by 19–24 ms a pass on the bench at every k (Q823). The k = 4 → 5 step exceeds the same mean by 18.6 ms.
  - **llama-bench's prompt path.** On the old base, llama-bench's 4-token pass cost the dense products 20–30 % more than
    the server's 4-token verify for the same weights and columns (Q589). If that holds on `-ladder3`, est. 15–25 ms of
    the slope is a llama-bench artefact that a real verify does not pay.
- **What settles it:** Q822's profile (graph, layer, sampled and group modes at k = 1 and 5) gives the parts and the time
  outside the cards; a k = 2 arm (Q870) separates the path switches from the per-token growth; a server verify of 4
  and 5 tokens on `-ladder3` (Q583's method) prices the llama-bench artefact.

### 1.2 The floor of the slope

From [the floor note](flash_next_floor_and_engine.md) (its 8–12 ms a token assumes independent routing and F32 hops),
per extra token (est.):

| Term | F32 hops, independent routing | Q8_0 hops, measured-class locality |
|---|---:|---:|
| Distinct routed experts (1.38 MB each per layer, 48 layers, 173 GB/s) | 3.6 (9.5 new a layer) | 2.3–2.7 (6–7 new a layer, u(5) = 34–38) |
| Hop bytes, eight boundaries | 3.3 | 0.9 |
| `dp4a` and FP32 arithmetic of one more column | ~2 | ~2 |
| Operations without weights | 1–2 | 1–2 |
| **Slope floor** | **9.9–10.9** | **6.2–7.6** |
| T(5)/T(1) at T1 = 82.4 ms | 1.48–1.53 | 1.30–1.37 |

Routing once per block removes the first row: est. 3.9–4.9 ms a token, T(5)/T(1) ≥ 1.19–1.24.

## 2. What a verify pass costs elsewhere

| Engine, model, hardware | Pass | Cost against one token | Source |
|---|---|---:|---|
| vLLM, Cohere's MoE (128 experts, top-8, shared experts), batch 1 | 4 tokens (3 drafted) | 1.25× | [Cohere](https://cohere.com/blog/mixture-of-experts-models-get-more-from-speculative-decoding): 20.4 distinct experts per layer against 29.1 uniform; routed experts 30 % of a step, a fixed cost the rest |
| ReplaySSM, Qwen3.5-122B-A10B (Gated DeltaNet hybrid), one B300, batch 128 | GDN verify kernel, 6 tokens | 1.27–1.72× the decode kernel (4.85× with per-token snapshots) | [ReplaySSM](https://tridao.me/blog/2026/replayssm/); the whole decode step 1.20–1.58× faster than with snapshots |
| llama.cpp tree prototype, Qwen3-32B (dense), B200 | 12 nodes | 1.71× (22.4 against 13.1 ms) | [unsloth #224](https://github.com/unslothai/llama.cpp/pull/224), via [MoE speculation](moe_speculation_union.md) |
| TensorRT-LLM, DeepSeek-R1 with three MTP layers, B200, batch 1 | 4 tokens | 2.16× end to end over no MTP (no isolated verify time) | [TensorRT-LLM MTP blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog2_DeepSeek_R1_MTP_Implementation_and_Optimization.html) |
| Miner Llama `-ladder3`, Swift Q2_0, nine P106 | 2 / 4 / 5 / 8 tokens | 1.48 / 2.10 / 2.63 / 3.87× | [k-curve](../../data/2026-11-04_kcurve_p106/README.md) |

- **The GPU numbers are flat for a reason the rig lacks.** A tensor-core GPU computes 4–16 columns of a weight tile in
  the time it reads the tile; the extra columns cost only the extra experts and the extra attention. Cohere's own
  Amdahl estimate (1.46×) overshot its measurement by 17 %, which it attributes partly to batched kernels sharing
  launch overhead.
- **The engines' common recipe:** each weight read once for the block; the routed experts in grouped kernels over the
  block's token–expert pairs; the block's queries in one attention call with a causal mask (TensorRT-LLM feeds K + 1
  input tokens per request to its generation-phase kernels); the verify, and in TensorRT-LLM the draft too, captured as
  one CUDA graph per width. In TensorRT-LLM's words, "both the verification and draft stages can be captured in one
  graph"; vLLM captures its uniform-decode graphs at token counts that are multiples of 1 + the drafted tokens
  ([vLLM cudagraph_utils](https://docs.vllm.ai/en/v0.22.0/api/vllm/v1/worker/gpu/cudagraph_utils/)).

## 3. Gated DeltaNet and linear attention: verifying k tokens

### 3.1 Forms of a k-token update

| Form | What it computes for k tokens | Where | Fits the rig? |
|---|---|---|---|
| Recurrence inside one kernel | the state in registers, the k tokens in a loop: two reductions and a rank-1 update per token | FLA `fused_recurrent_gated_delta_rule` ([vLLM API](https://docs.vllm.ai/api/vllm/model_executor/layers/fla/ops/fused_recurrent.html)); FlashInfer `gated_delta_rule_mtp` (T > 1, SM90 and up, FP32 state; a tensor-core variant at T = 4–8 on SM100; [docs](https://docs.flashinfer.ai/api/gdn_decode.html)); llama.cpp's fused `GATED_DELTA_NET` (`gated_delta_net.cu`) | yes: it is what `-ladder3` runs at k ≥ 2 |
| Chunked WY form | k tokens as one chunk: triangular solves and small matrix products | FLA's chunk kernels; llama.cpp's chunked kernel (enabled on sm_86 and sm_89 only) | no: est. 1.8× the multiply-adds of the recurrence, paid back only by tensor cores ([prefill note](flash_next_prefill.md)) |
| Replay or closed form | outputs from a checkpoint plus the block's inputs, one T × T solve; no state per token | ReplaySSM (FlashInfer `gated_delta_rule_replayssm_commit`, SM100/SM103), TreeWY, Bole ([MoE speculation](moe_speculation_union.md) section 5) | the replay idea yes (Q686); the GEMM form no |

- **`-ladder3`'s path, from its source.** For one token the fused autoregressive op runs; for k ≥ 2 the context's
  probe finds `GATED_DELTA_NET` supported on CUDA and builds the same fused op, never the chunked graph. The kernel walks
  the k tokens in order with the state in registers and, when the server keeps rollback states (`n_rs_seq`), writes a
  full state snapshot per token into its slot.
- **The time is small on the rig.** The recurrent state and convolution cost 4.3 ms at one token and 6.2 at five
  (Q583), +0.5 ms per extra token over 36 layers. The chunked form would add arithmetic without tensor cores.
- **No measurement of a k-token Gated DeltaNet verify on a GPU without tensor cores was found.** Every published number
  is on Hopper, Blackwell, Ampere or Apple silicon.

### 3.2 State rollback

- **Per-position state slots are the default everywhere.** vLLM gives a drafting sequence num_spec + 1 state slots from
  its block table and passes `ssm_state_indices` and `num_accepted_tokens` to the recurrent kernel
  ([gdn_attn](https://docs.vllm.ai/en/latest/api/vllm/v1/attention/backends/gdn_attn/)); SGLang's `extra_buffer`
  strategy keeps extra state slots per request for speculative decoding
  ([SGLang](https://docs.sglang.io/basic_usage/qwen3.html)); llama.cpp keeps `n_rs_seq` planes.
- **Their cost is the writes.** ReplaySSM's baseline verify kernel at 6 tokens costs 4.85× the decode kernel; keeping a
  ring of the block's inputs (k, the corrected v, the decay) and a checkpoint brings it to 1.27–1.72×, and a commit
  replays only the accepted prefix. On the rig a snapshot costs ~1.8 ms per row in the server's verify (Q583's +5.5 ms
  for three rows) and nothing in llama-bench, which keeps no rollback states.
- **For the rig:** Q686 (recorded inputs, the accepted path replayed in FP32 with the decode kernel's arithmetic) is
  worth est. −5–7 ms in a 5-token server verify, and it is what lets 16–32-token blocks fit (112.6 MiB per state plane).
  It does not touch llama-bench's T(k).

## 4. Routed experts at small batch

- **Grouped kernels read each distinct expert once.** Upstream llama.cpp's `mul_mat_vec_q_moe`
  ([#20905](https://github.com/ggml-org/llama.cpp/pull/20905), merged 2026-03-29) raised the multi-token expert limit of
  its vector kernel from 4 to 8 tokens and gave the grid a dimension over the block's tokens. The fork's `-mtok` kernel
  does the same for Q2_0, bit-identical to the one-token kernel, so the experts of a pass cost
  0.606 ms × u(k) (est., [R7](../../data/2026-11-04_mtok_experts/README.md)).
- **On a P40 (compute 6.1, like the P106) the vector kernel loses from 5 tokens.** #20905's P40 rows, against master:
  gpt-oss-20B MXFP4 1.13 / 1.20 / 1.23× at 2–4 tokens, then 0.96 / 0.87 / 0.82 / 0.79× at 5–8; granitemoe-3B in
  IQ1_S to IQ3_S 1.37–1.59× at 2–4 and 0.74–1.11× at 5–8. Master changes kernel at 5 tokens (its vector kernel then stopped at 4): its throughput jumps from 4
  to 5 tokens by 37 % (gpt-oss, 140.6 → 192.8 t/s) and 49 % (IQ1_S, 243.0 → 362.8), the step to the tiled `dp4a`
  kernel (MMQ). These are rows of parallel sequences, not one verify block, but the products are the same shape: on
  Pascal a tile kernel that stages weights and activations in shared memory beats a multi-column vector kernel from
  ~5 columns. The fork's k-column and multi-token kernels at 5–8 columns (56–244 registers, C70) were never timed
  against MMQ on the same shapes (Q871).
- **The union is the price, and exact methods only choose which tokens to verify.** EVICT, EcoSpec and Cascade (exact)
  and MoE-Spec, AcceptMoE, XShare and Lynx (inexact) are compared in [MoE speculation](moe_speculation_union.md)
  section 4; Q690 and Q691 hold the parked rows. Section 7 adds what the inexact ones report against the size of their cut.
- **What the rig pays:** +23.1 ms at k = 5 in llama-bench, est. +14.5–17 in a real verify, 11–17 % of the slope.

## 5. Attention for a draft block

- **The engines attend with all k queries in one call.** TensorRT-LLM verifies MTP drafts by feeding K + 1 tokens per
  request to its generation-phase attention with a causal mask; FlashMLA's sparse decode kernel takes `s_q` query
  tokens per sequence and an index tensor of shape (batch, s_q, top-k), one selection per query token
  ([FlashMLA](https://github.com/deepseek-ai/FlashMLA)). One kernel call reads a dense KV tile once for all of the
  block's queries and heads. FlashInfer's cascade attention (a shared prefix attended once, the per-request suffixes
  separately, the partial states merged) reports up to 31× over vLLM's PagedAttention at batch ≥ 128 and a 32k shared
  prefix ([FlashInfer](https://flashinfer.ai/2024/02/02/cascade-inference.html)); it serves many requests over one
  prefix, which a single agent on the rig does not have.
- **Sparse attention selects per query token, and adjacent tokens overlap partly.** NVIDIA's
  [GVR](https://arxiv.org/abs/2604.22312) measured DeepSeek-V3.2's top-2,048 selections on SWE-bench-derived 64k
  prompts: consecutive decode steps share 35–50 % of their cells in layers 20–60 (at most ~60 %) and 1–2 % in layers 0
  and 1. A draft block is the same relation, consecutive positions. With 50–65 % new cells per extra token a 5-token
  union holds est. 3.0–3.6× one token's selection: Q643's union gather would read 28–40 % fewer cells than five
  separate selections, not ~80 % fewer. Flash-Next's QSA selects pools of cells, where overlap is likely higher (est.);
  layer-to-layer reuse (IndexCache: 70–100 % overlap between adjacent layers,
  [arXiv 2603.12201](https://arxiv.org/html/2603.12201)) is a different axis.
- **On the rig the attention's slope is unmeasured where it matters.** T(k) was measured at an empty context, where
  attention is ~1 ms. At 2–8 query tokens `-ladder3` runs the stock kernels (`fattn-pascal` takes one token; the Pascal
  tile kernel starts at 9, or 3 with a Q8_0 cache); whether they read the KV once for the block or once per query
  decides the slope at 8–32k cells, where one token's dense attention costs est. 6–38 ms
  ([the floor note](flash_next_floor_and_engine.md)). A k-curve at depth (Q873) settles it.

## 6. Operations without weights, host work and the verify graph

- **On the rig they add ~5 ms of GPU time at k = 5** (Q583: 26.3 → 30.8 ms), and the hops est. +7–9 ms. The hops are
  this setup's own term: a layer split over x1 links without peer access moves 40 KB per token per boundary through the
  host. Q750's Q8_0 hop (10.9 KB) is est. −2.4 ms a token and −9.8 ms per 4-token pass, ~−12 ms at k = 5.
- **The one-token fusions do not apply to k tokens** (section 1.1), and the fork's per-size graphs (`-vgraphs`) key a
  captured graph per shape. Whether llama-bench's k-token passes replay a graph or run eagerly is in the light
  profiler's per-card `how` counts (C142r); eager launches would add est. 5–10 ms of fixed cost per pass (~2,900
  kernels).
- **The n-gram table is Flash-Next's own per-token host cost.** Every token makes 11–14 random reads of the 28.8 GB table
  on the SSD (Q214). If a pass reads its tokens' rows one at a time, four extra tokens cost est. 6.7–8.5 ms before the
  first card starts; issued together at queue depth 32 est. ~2.4 (Q872).

## 7. Routing once per verify block (Q860)

**What it is.** In a verify pass of k tokens (the last accepted token and k − 1 drafts), tokens 2..k choose their
experts only among the 10 the first token chose, weighted by their own router scores renormalized over that set. A
pass then reads 10 experts per layer instead of the union.

**It is lossy, in two ways.**
- **The verified distribution changes.** Row 1 (the last accepted token) is exact, so the first drafted token is
  verified against the target; rows 2..k come from a different model, so the later drafts and the bonus token are
  accepted or sampled from it. Speculative decoding stops being exact from the second drafted token.
- **The cache keeps the error.** The keys and values of the attention layers and the Gated DeltaNet and convolution
  states written for each accepted draft were computed with the wrong experts. They stay in the cache and every later
  token reads them; the error does not end with the step. Only rejected drafts are discarded.
- **No exact repair exists.** Using the restricted pass as a draft and verifying exactly needs a second full pass
  through the cards (self-speculation, SpecMoE and DraftExpert in [MoE speculation](moe_speculation_union.md)); a
  token's true experts are almost never all inside the first token's 10 at 512 experts.

**What the cut is.** At k = 5 the union is 34–38 distinct experts per layer in a real verify (est.) and 48.1 for random
ids; route-once keeps 10, a cut of 71–79 %. Each token keeps the share of its own routed weight that falls on the first
token's experts: with ~38 % of adjacent tokens' picks shared (Cohere's MoE) and less for tokens further apart, est.
40–60 % of the weight.

**The published evidence, by size of cut:**

| Method | Setting | Cut | Quality | Speed | Source |
|---|---|---|---|---|---|
| Lynx-Acc (training-free, per layer, high-confidence tokens keep their first expert) | Mixtral-8x7B, decode batch 32 | 8 → 6 / 4 / 2 experts per layer for the batch | GSM8K 59.1 → 58.6 / 55.3 / 44.0; HumanEval 37.8 → 38.4 / 34.8 / 23.2 | 1.3–1.5× at half the experts | [Lynx](https://arxiv.org/html/2411.08982v1) |
| Lynx-Acc | DBRX, batch 32 | 16 → 12 / ~8 / ~4 | GSM8K 71.6 → 73.2 / 72.6 / 62.6; HumanEval 39.6 → 39.4 / 36.2 / 34.1 | 1.7× (16 → 8), 1.9× (16 → 4) at batch 64 | Lynx |
| Lynx-Lat (fixed count, equal votes) | DBRX, batch 32 | 16 → 4 | GSM8K 71.6 → 28.7 | | Lynx |
| MoE-Spec (budget for a 63-node tree, the rest substituted) | Qwen3-30B-A3B / Mixtral / OLMoE | budgets 16–64 of 128 / 2–7 of 8 / 8–56 of 64 | at the chosen budget GSM8K 84 (exact decoding 84, EAGLE-3 87); ~95 % of quality at 2.0× (Qwen3), ~78 % at the top speed (Mixtral), the steepest decline on OLMoE; τ −1.4 % | +10–30 % over EAGLE-3 | [MoE-Spec](https://arxiv.org/html/2602.16052v1) |
| AcceptMoE (a set per verify block sized from commitment probabilities) | three targets, four tasks | 23–34 experts per block | −0.27 points mean accuracy against exact speculation | +29 % over EAGLE-3 | [AcceptMoE](https://arxiv.org/abs/2608.02989) |
| XShare (in-batch selection maximizing total gate score) | batched serving and speculation | up to −30 % expert activations | baseline accuracy kept (ICML 2026 abstract) | +14 % with speculation (v1); 10–37 % end to end | [XShare](https://arxiv.org/abs/2602.07265) |
| Autoregressive selection (keep k − l of the previous token's experts, l own) | 1B MoE, top-4, trained with it | union over a block ≤ k + γ·l | matches the baseline's MCQA accuracy (with three other router changes, not separated) | +21 % tokens/s | [arXiv 2609.22471](https://www.alphaxiv.org/abs/2609.22471) |
| ReMoE (router fine-tuned toward the previous step's experts) | DeepSeek-V2-Lite, Qwen1.5-MoE | reuse +26–27 % | perplexity 3.228 with its anchor to the original router, 3.263 without | +8.4 % offloaded on an RTX 3090 | [ReMoE](https://arxiv.org/html/2605.27081) |
| Q618's tail cut on the rig's own file | Swift-1.5 Q2_0, CPU sample | p = 0.9 keeps 8.62 of 10 (86 % of the reads) | the greedy text changes after a few words | graph level only | Q618 |

- **Nothing published comes near route-once's cut on a fine-grained model.** The closest shares are Lynx's 8 → 2 on
  Mixtral (keep 25 % of the batch's experts: −15 points on GSM8K) and 16 → 4 on DBRX (−9 points with confidence-aware
  selection, −43 without), and they choose the kept set by the whole batch's votes, not by one token. The methods that
  keep quality within a point cut 13–30 % (EcoSpec, XShare) or size the set per block with 23–34 experts (AcceptMoE);
  the trained method keeps k − l of the previous token's experts with l ≥ 1 own. Route-once is l = 0 and untrained.
- **Bytes and time saved at k = 5 (est.):** 24–28 fewer experts per layer in a real verify (u 34–38 → 10), 1.38 MB each
  over 48 layers: **1.59–1.86 GB a pass, 14.5–17.0 ms** at the multi-token kernel's 0.606 ms per distinct expert (9.2–10.7
  at 173 GB/s). In llama-bench (u 48.1): 2.53 GB, 23.1 ms; T(5) 217.1 → ~194 ms, 2.35×. Against 27 ms at k = 1 and 71 at
  k = 5 on the old kernels, route-once would have saved ~44 ms; R7's kernel already took most of that.
- **How to measure the loss so that it includes the cache.** A KLD on FN1's bases where, inside each scored chunk, every
  token at a position not divisible by k uses the experts of the block's first token: the later layers and later
  tokens then read the restricted tokens' keys, values and state, as they would after an accepted chain. Report the KLD
  per position in the block and over the whole chunk, at k = 2–5, beside Q618's thresholds; then acceptance and task
  quality on the agent check.
- **A smaller cut with the same machinery (Q874).** Tokens 2..k keep their own top-l experts and fill the rest from the
  first token's set by their own scores: the union is ≤ 10 + (k − 1)·l, 18 at k = 5 with l = 2 and 14 with l = 1, est.
  −10–12 and −12–14.5 ms. Lossy too, with est. 55–75 % of each token's routed weight kept (l = 2), between route-once and
  exact.

## 8. What it means for the rig

DFlash2 on agent requests accepts τ = 2.56 at two drafted tokens and 3.11 at three
([Q561](../../data/2026-11-03_flash_next_dflash2/README.md)); with a 20 ms draft pass and T1 = 82.4 ms (est., T(k) taken
as the server's verify):

| Slope (ms per extra token) | T(4) ms | T(5)/T(1) | S at 2 drafted | S at 3 drafted | tok/s at 3 drafted |
|---|---:|---:|---:|---:|---:|
| today's curve (measured T(k)) | 173.0 | 2.63 | 1.25 | 1.33 | 16.1 |
| 20 | 142.4 | 1.97 | 1.48 | 1.58 | 19.2 |
| 15 (kernels at ~2× the exact floor) | 127.4 | 1.73 | 1.59 | 1.74 | 21.1 |
| 12 | 118.4 | 1.58 | 1.67 | 1.85 | 22.5 |
| 8 (exact floor, Q8_0 hops) | 106.4 | 1.39 | 1.78 | 2.03 | 24.6 |
| 5 (route-once floor; acceptance unknown) | 97.4 | 1.24 | 1.88 | 2.18 | 26.5 |

- **The first 13–18 ms of slope are worth the most:** from 33 to 15–20 ms a token raises a 3-token draft from 1.33× to
  1.58–1.74×. Below ~12 ms longer drafts start to pay ([MoE speculation](moe_speculation_union.md) section 3.2's rule:
  a node pays when its path probability exceeds the marginal cost times S).
- **The ratio target misleads once T1 moves.** Every saving at k = 1 alone (launches, the logits copy, Q823's shapes)
  raises T(5)/T(1) while it raises tok/s. Report the slope in ms beside the ratio.
- **The lossy lever buys ~2 tok/s at the end of the table and nothing before.** Route-once's 14.5–17 ms equal the
  first 3.5–4 ms of slope per extra token; the unaccounted 57–72 ms are four times that and exact.

## 9. Ideas for the rig

Ranked by est. ms saved on a 5-token pass. Ideas marked **new** got their registry rows with this note.

| # | Idea | Est. ms saved at k = 5 | Exact? | Effort | First measurement that confirms it | Registry |
|---|---|---:|---|---|---|---|
| 1 | Find the unaccounted half of the slope: Q822's profile (k = 1 and 5) plus a k = 2 arm, and a server verify of 4 and 5 tokens on `-ladder3` for the llama-bench gap | up to 57–72 (the unaccounted part; ≤ ~33 at the two path switches, 15–25 if Q589's gap holds) | yes | profile ~35 min + ~10 min for the k = 2 arm; fixes after | the parts at k = 1 / 2 / 5 summing to the wall within 5 %, and the switches' launch counts differing between k = 1 and 2 | Q822, Q589; the k = 2 arm in Q870 (**new**) |
| 2 | Retune the k-column launch shapes (1 row per lane group or a 32 KB activation budget where the bench shows it faster) | −19 (bench sums; −24 at k = 1) | yes | days | the bench's sums with the new `pick_shape`, then T(k) | Q823 |
| 3 | Route once per verify block | −14.5 to −17 (−23 in llama-bench) | **no**, verified distribution and cache | the `route-once` switch, a KLD by k | the block-emulating KLD below Q618's gate; acceptance on agent requests not below exact | Q860 |
| 4 | Q8_0 residual hops for verify passes | ~−12 (−2.4 at k = 1) | near (KLD pending) | code done (`LLAMA_HOP_CAST`, CPU-checked), not yet on a GPU | KLD, then T(k) with the cast | Q750, Q231, Q820 |
| 5 | The one-token-only paths made k-aware: the GLU-fused gate/up products and the shared-expert fusion at 2–8 columns, a Pascal attention kernel for 2–8 query tokens | −5 to −14 at every k ≥ 2 | yes | 1–2 weeks | the k = 2 arm of idea 1: the first step's excess (14.2 ms) and the launches that appear at k = 2 | Q870 (**new**) |
| 6 | Partial route-once: l own experts per token plus the first token's best (union ≤ 10 + (k − 1)·l) | −10 to −12 (l = 2), −12 to −14.5 (l = 1) | **no** | a parameter of Q860's switch | the same KLD at l = 1, 2, 3 beside l = 0 | Q874 (**new**) |
| 7 | MMQ (`dp4a` tiles) for 5–8 columns, the crossover upstream shows on a P40 | 0 to −10 (−10 to −30 at k = 8, est.) | yes | a switch and a bench case | llama.cpp's MMQ on Flash-Next's shapes at 5–8 columns in `bench/mmvq_pascal` beating the k-column and multi-token kernels | Q871 (**new**) |
| 8 | Recorded-input recurrent state instead of a snapshot per row | −5 to −7 in a server verify, 0 in llama-bench | yes | 1–2 weeks | the server verify's state rows (Q583's method) at k = 5 | Q686 |
| 9 | A verify block's n-gram rows read together (one queue-depth-32 batch for all k tokens) | 0 to −6 | yes | days | host time before card 0 at k = 1 against k = 5 (the graph-mode profile's time outside the cards) | Q872 (**new**) |
| 10 | Union gather of the block's sparse-attention selections | 0 at an empty context; est. −1 to −4 at 8–32k cells | yes | written as Q643's design | the selections' overlap between consecutive tokens on Flash-Next (`LLAMA_QSA_INPUT_STATS`) | Q643 |
| 11 | T(k) at agent depth: the k-curve at 8k and 32k cells, sparse attention off and on, before any attention work for drafts | decides; the attention's slope at 32k is unknown (0 to tens of ms) | — | ~30 min on a rig | llama-bench `-p 1..8 -d 8192,32768` against the empty-context curve | Q873 (**new**) |

- **Order of work.** Idea 1 first: it is a measurement on a build that exists and it sizes ideas 2, 5, 7 and 9. Ideas 2
  and 4 are exact and nearly ready. Ideas 3 and 6 need only the switch the `route-once` work is writing and a KLD; they
  decide whether a lossy lever is worth ~2 tok/s once the exact ones are in.
- **Parked and not reopened by this note:** EcoSpec-style node selection (Q690) and capped expert budgets (Q691). Both
  act on the expert union, which is 11–17 % of the slope after R7's kernel.
