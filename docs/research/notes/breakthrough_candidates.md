# Breakthrough candidates for Flint and Qwen3.6-35B-A3B on the P106 rig (2026-10-23)

Scope: ideas that could give a large decode speed-up, or a large quality gain at the same speed, for Flint (40B-A2.5B
MoE) and Qwen3.6-35B-A3B on nine P106-090 cards with this project's llama.cpp fork, and that the
[registry](../../../quants/experiments.md) (Q1–Q86) and the other [research notes](README.md) have not already found
or ruled out. Each candidate was checked against its primary source (paper, code or repository) and priced with the
rig's own numbers. Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs: code and correctness on a laptop GPU (compute 6.1
  kernels forced or run through the compute_61 PTX), quality and training on the 7-GPU quant machine (never speed),
  speed on the rig only.

Starting point, measured ([registry](../../../quants/experiments.md)):
- S4-L0 on six cards, fork build: 37.4 tok/s in llama-bench tg128, 26.8 ms per token (T below); 2.24 GB of weights per
  token. G3 (KLD 0.101 against BF16) reads 2.33 GB. UD-Q4_K_XL reaches 37.3 tok/s with FP32 activation sums (Q76).
- Real chat, nine cards, 262,144-token context, backend sampling: 31.9 tok/s (S4-L0), 28.0 (UD-Q4_K_XL) (Q58).
- Layer split runs the cards one after another: each card computes for 4–8 ms of a 26.8 ms token and waits the rest
  (Q49), so five of six cards are idle at any moment.
- Verify steps run on llama.cpp's stock multi-token kernels: 2.9 tokens cost 2.23 T, 4.05 tokens 3.56 T (Q19). n-gram
  drafting gives 1.03× overall at that cost (Q64), Qwen3.6 MTP 1.04× (Q43).
- Profiled GPU time per S4-L0 token (event profiler, 5–8 µs of overhead per launch included): 24.7 ms of matrix
  products, 8.0 ms of everything else; the Pascal kernels stream 138–150 GB/s on large shapes, while the routed-expert
  calls of a real token run 25–33 µs slower than the same shapes in the microbenchmark
  ([kernel note](pascal_mmvq_integration.md), section 10).

## Verdict

- **The largest untried lever is pipelined speculation: guessed tokens enter the first card while later cards still
  work on the current token, each as an ordinary single-token step.** The project parked speculation because a
  batched verify step costs 2.2–3.6 T on these cards. A layer-split rig doesn't need batched verification: a guess for
  token t+1 can start on card 1 as soon as card 1 has finished token t, and the last card checks it by sample-and-match
  when token t's own sample appears. Every in-flight guess uses the fork's fast single-token kernels and otherwise-idle
  cards, so a wrong guess costs at most one stage of delay. Est. with today's n-gram matches and no training: 1.2×
  overall, 1.5× on Flint's whole-file edits, 3–4.5× on file rewrites, 1.0× on chat. With a trained pipeline drafter
  (per-token acceptance 0.5–0.7): 1.5–2.4× on chat. The same executor runs independent conversations staggered through
  the cards for up to ~5× aggregate throughput. The method is published and measured elsewhere (SPD 2.05–2.21× wall
  clock at temperature 1 on 8-stage pipelines; SpecPipe; PipeInfer, built on llama.cpp); nothing like it exists in
  llama.cpp.
- **Second: a persistent decode kernel per card**, staged from one fused MoE-FFN layer. About 9 ms of a token's GPU time
  is not weight streaming (kernel boundaries, tails, small ops). A 30B-A3.3B MoE decode megakernel runs 1.58× vLLM at
  batch 1 on an H100. Est. +20–35 % here, exact.
- **Byte levers that are new here, each +3–9 % (est.):** a 6-bit simple format (Q6_0, as in ik_llama.cpp) between Q5_0
  and Q8_0, since 63 % of G3's bytes are Q8_0; an output head that computes exact Q8_0 logits only for candidates picked
  by a cheap screen (exact for top-k 20 sampling while recall holds; +8–18 % for Qwen3.6's 248k vocabulary); and
  KL-aware rounding (YAQA, −36 % KL against GPTQ at INT4 group 32) to move the always-read tensors off Q8_0.
- **Prompt processing is limited by the attention mask's memory, not only by kernels.** The scheduler keeps four
  pipeline copies of an n_ctx × n_ubatch f16 mask on the host and on every card (256 MiB each at 262k and `-ub 128`),
  which is what forces `-ub 128`. A mask computed inside the attention kernel from positions frees that memory, permits
  `-ub 512+` at full context (est. 1.3–2× prefill there) and removes the per-step mask copies speculation needs gone.
- **Long context:** query-aware KV page selection (Quest) on the 14 full-attention layers would cut KV reads ~15× at
  131k (est. ~2× decode over the planned Q60 kernel), at a retrieval-quality risk that 512-token KLD cannot see.
- **Weaker but real:** intra-expert activation sparsity (+2–9 %, quality-limited at this bar), an expert-contiguous
  weight layout if the real-token expert slowdown is address translation (+3–8 %, unverified), learned expert skipping
  by self-distillation (+8–14 %, training and quality risk).
- **Rejected by arithmetic:** trellis codes (QTIP/EXL3), vector quantization (AQLM, QuIP#) and learned non-uniform
  4-bit lookups (any4) all cost more instructions per weight than a 5-SM card can issue at 145 GB/s, or more time per
  weight than Q4_1; expert merging and low-rank expert deltas keep bytes per token or need heavy retraining.
- **What it changes in the plan:** speculation should not wait for multi-token kernels (Q78); a Flint drafter (Q20)
  is worth training again, as a single-token pipeline drafter; Q52/Q53's batched-verify plumbing matters less; Q73/Q77's
  format ladder lacks a 6-bit rung; and `-ub 128` at long context is a software limit.

## Ranked candidates

Gains are for single-stream decode against today's fork build unless the row says otherwise. "Bar" is the quality bar
(within noise of UD-Q4_K_XL against BF16).

| # | Idea | Mechanism | Evidence | Expected gain on the rig (est.) | Quality risk vs the bar | Effort | Where to test | What would kill it |
|---|---|---|---|---|---|---|---|---|
| 1 | Pipelined speculation | Guessed token t+1 enters card 1 while cards 2–6 finish t; single-token steps only; sample-and-match at the last card; flush on a miss | [SPD](https://arxiv.org/abs/2605.30852) 2.05–2.21× wall clock at T 1 (n = 8); [SpecPipe](https://arxiv.org/abs/2504.04104) 4.2–5.5× over plain PP (8 × RTX 3090); [PipeInfer](https://arxiv.org/abs/2407.11798) on llama.cpp; [FlowSpec](https://arxiv.org/abs/2507.02620) 1.45–1.70× on Jetsons | n-gram guesses: 1.21× overall, 1.53× edits, 3–4.5× rewrites, 1.01× chat; trained drafter: 1.5–2.4× chat; Qwen3.6 with MTP 1.4–1.8×; staggered sequences up to ~5× aggregate | None: sample-and-match output is bit-identical to plain decoding | Executor 1–2 weeks; drafter 3–5 days of training | Simulator on Q64 transcripts (laptop); executor on the laptop GPU; rig for cycle time | Rig cycle time with all stages busy above 0.3 T (S = 6), or host dispatch above one cycle |
| 2 | Persistent decode kernel per card | One resident kernel per card (or per layer) with counter-based dependencies instead of ~1,300 kernel boundaries per token | [Hazy megakernel](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) 78 % vs ≤ 50 % of H100 bandwidth; [Cohere](https://cohere.com/blog/megakernels) 30B-A3.3B MoE 1.58× vLLM at batch 1; [MPK](https://arxiv.org/abs/2512.22219) up to 1.7×; rec 1.4–2.6× llama.cpp | 26.8 → 20–22 ms per token, +20–35 % | None (summation order only) | First step 1 week; full 3–6 weeks | Fused MoE-FFN layer in `bench/mmvq_pascal` on one rig card | Fused layer < 1.3× the sum of the fork's kernels for that layer |
| 3 | Attention mask computed on the card | Flash attention derives causal and sliding-window masking from positions; no n_ctx × n_ubatch mask input | Scheduler code (4 input copies); [long-context note](long_context_cliff.md) section 3; [setup](../../setup.md) | Prefill at ≥ 100k context 1.3–2× via `-ub 512+`; −256 MiB VRAM per card and −256 MiB pinned host at 262k; no per-step mask copies | None | 3–5 days | Rig, first with a `GGML_SCHED_MAX_COPIES=1` build (no code) | `-ub 512` not ≥ 1.25× faster than `-ub 128` on 2k–8k prompts |
| 4 | Q6_0, a 6-bit simple type | 32 weights, FP16 scale, 4-bit plane + 2-bit plane (6.5 bpw); decodes like Q5_0 | [ik_llama.cpp PR #77](https://github.com/ikawrakow/ik_llama.cpp/pull/77) ("almost as good as Q6_K"); Llama-3-8B KLD Q8_0 0.0014, Q6_K 0.0055, Q5_0 0.022 | G3's 1.46 GB of Q8_0 → 1.12 GB: up to −0.34 GB per token, +3–7 % at equal KLD | +0.005–0.018 KLD if every Q8_0 tensor moves (est.); choose per class | 3–5 days (fork type) after a 1-day proxy test | Quant machine (Q6_K proxy, then Q6_0), rig | Q6_K proxy for head + shared expert + attention output costs > 0.008 KLD |
| 5 | Screened output head | Cheap screen (Q4_1 copy or rank-256 factor) over all rows → top-K candidates → exact Q8_0 logits for K rows → sampler on candidates | [HiRE](https://arxiv.org/abs/2402.09360) 1.47× (softmax + FFN, 1B model, TPU) | Flint −0.13 to −0.28 GB per token, +3–8 %; Qwen3.6 +8–18 % | Exact for greedy and top-k ≤ K while recall holds; tail logits approximate | 3–5 days | Recall offline (quant machine), then fork + rig | Recall of the top-20 / top-p 0.95 set at K = 1,024 below 99.9 % |
| 6 | KL-aware rounding of the always-read tensors | YAQA (full-model KL Hessian, Kronecker sketch) or GPTQ into Q5_0/Q6_0 blocks for attention q/o and the shared expert | [YAQA](https://arxiv.org/abs/2505.22988): INT4 g32 KL 0.033 (LDLQ) → 0.021 | Q8_0 → Q5_0/Q6_0 at equal KLD for 0.78 B params: −0.2 to −0.29 GB, +5–9 % | Aimed at zero by construction | 1–3 weeks | Quant machine (KLD), rig | GPTQ Q5_0 attention output (layers 0–32) recovers < half of its Q8_0 → Q5_0 KLD loss |
| 7 | Query-aware KV page selection (long context) | Per-page min/max keys; read only the top pages per KV head on the 14 full-attention layers | [Quest](https://arxiv.org/abs/2406.10774): 7.03× attention, 2.23× end to end at 30k, 2k budget | +10–15 % at 32k; ~2× at 131k and 2.5–3× at 262k over Q60's kernel | Retrieval at long range; invisible to 512-token KLD | 1–2 weeks after Q60 | Quant machine (masked emulation, long-context tasks), rig | Long-context retrieval drops > 2 points at a 4k budget |
| 8 | Intra-expert activation sparsity | Dense gate; up rows and down columns only for neurons above a threshold (down stored transposed) | [Park et al.](https://arxiv.org/abs/2605.08575): Qwen3.5-35B-A3B 70.9 → 68.2 avg accuracy at 60 %; MoE layer 1.5–2.5× | +2–9 % (20–80 % sparsity on routed experts) | High at this bar; likely only 20–40 % fits | 2–3 weeks | Quant machine (graph-level thresholding), then kernels | +0.01 KLD at 30 % sparsity on G3 |
| 9 | Expert-contiguous weight layout | Place each expert's gate, up and down rows together (strided views), fewer 2 MiB pages touched per call | Real-token routed calls run 25–33 µs slower than the benchmark; [Jia et al.](https://arxiv.org/abs/1804.06826) TLB data | +3–8 % if the gap is address translation | None | 1 day bench, 3–5 days fork | One rig card (bench), rig | Spreading experts over 370 MB in the bench doesn't reproduce the gap |
| 10 | Learned expert skipping | Parameter-free zero experts added to the router, trained by self-distillation, so easy tokens use fewer of their 6 experts | [ZEDA](https://arxiv.org/abs/2605.18643): > 50 % of expert FLOPs removed, 1.20× end to end (Qwen3-30B-A3B) | Routed bytes −30–50 %: +8–14 % | Medium to high; needs task checks, not only KLD | 1–2 weeks of training + zero-expert support in the fork | Quant machine, rig | Adapted BF16 model's KLD against the original > 0.03 |

## Where a token's time goes now

What G3 reads per token (est., from its recipe and Flint's shapes; Q8_0 = 1.0625 B per weight, Q5_0 = 0.6875):

| Tensors | Format | MB per token |
|---|---|---|
| Attention q, layers 0–24 / 25–48 | Q8_0 / Q5_0 | 223 / 138 |
| Attention output, layers 0–32 / 33–48 | Q8_0 / Q5_0 | 294 / 92 |
| Attention k, v | Q8_0 | 109 |
| Routed gate/up, layers 2–8 / 9–48 | Q8_0 / Q5_0 | 94 / 346 |
| Routed down, layers 2–45 / 46–48 | Q5_0 / Q8_0 | 190 / 20 |
| Shared expert | Q8_0 | 314 |
| Dense layers 0–1 | Q8_0 | 80 |
| Router | F32 | 99 |
| Output head | Q8_0 | 327 |
| **Total** | | **2,326 (Q8_0: 1,461, 63 %)** |

On the S4-L0 profile, ~2.24 GB at the 150 GB/s the Pascal kernels reach on large shapes would take ~15 ms; the
unprofiled GPU time is ~24 ms and the wall time 26.8 ms (est.: 32.7 ms profiled minus ~6.5 µs of event overhead on each
of ~1,300 launches). So roughly 9 ms per token goes to kernel boundaries, ramp-up and tails of small kernels, and the
non-matrix work, and ~3 ms to hops and the host. Candidates 1 and 2 attack the idle cards and the 9 ms; 4–6 and 8–10
attack the 15 ms.

## 1. Pipelined speculation

### Mechanism

In layer split, token t passes card 1, then card 2, and so on; the last card samples token t+1. Card 1's part of the KV
cache for token t is complete as soon as card 1 has finished it, so card 1 can already run token t+1, if it knows which
token that is. Pipelined speculation feeds a guess g(t+1) into card 1 right then, g(t+2) one stage later, and so on, up
to S tokens in flight for S cards. When the last card finishes token t it samples the real token t+1:
- if it equals g(t+1), the computation of g(t+1) already in flight is the correct one and continues;
- if not, everything behind token t is dropped, its KV cells are released, and the sampled token enters card 1.

Every in-flight token is an ordinary single-token step, so the fork's Pascal kernels and CUDA graphs run unchanged and
no verify batch exists. With sample-and-match each position is sampled once, in order, from logits computed by the same
kernels as plain decoding, so the output is bit-identical to plain decoding with the same seed. A drafter that supplies
probabilities can instead use rejection sampling at the last card (higher acceptance, distribution-exact), or Gumbel-max
sampling with noise shared between drafter and target, which keeps the output a function of the seed alone and accepts
with probability at least (1 − TV)/(1 + TV) ([Daliri et al.](https://arxiv.org/abs/2408.07978)).

### Why it fits this rig

- The cards are idle 70–85 % of each token (Q49), so in-flight guesses run on time that is wasted today.
- The obstacle the project hit, the cost of multi-token verify steps on 5-SM cards (Q19), does not arise.
- The hop between cards is a few KB, small next to the stage time.
- The 75 W limit applies to busier cards: at full duty the core clock settles at ~1,700–1,760 MHz instead of boosting
  in short bursts. The bandwidth-bound Pascal kernels lose little from that (est. ≤ 10 % per stage, folded into the
  cycle time below).

### Evidence

| Work | Setup | Drafter | Result |
|---|---|---|---|
| [Speculative Pipeline Decoding (SPD)](https://arxiv.org/abs/2605.30852), 2026 | Qwen3.5-4B/9B split into n = 4, 8, 16 stages, one token per stage; 8 × H20 plus one GPU for the drafter | 1–4-layer "pipeline draft module" trained by KD on the target's features from several depths; each in-flight token only sees the depths it has reached | Wall clock at T 1: 2.05× (4B, n = 8), 2.21× (9B) against EAGLE-3's best 1.47× and 1.81×; step acceptance α = 0.34 (n = 8) and 0.58 (n = 4) at T 1, i.e. per-token acceptance ≈ 0.72–0.75 (est., from α = 1/((1 − p)·n + p)); n = 16 falls back to ~1× wall clock from per-cycle overhead |
| [SpecPipe](https://arxiv.org/abs/2504.04104), 2025 | Llama3.1-70B on an 8-stage pipeline of RTX 3090s over PCIe | Llama3.2-1B on its own GPU, a token tree 64 wide per stage | 4.19–5.53× time between tokens over plain pipeline parallelism; misses flush the pipeline |
| [PipeInfer](https://arxiv.org/abs/2407.11798), SC 2024 ([code](https://github.com/AutonomicPerfectionist/PipeInfer)) | llama.cpp with MPI across CPU nodes (Gigabit Ethernet, InfiniBand) | Small draft models | Up to 2.15× over standard speculative inference; KV multibuffering through sequence ids, early cancellation of invalidated runs |
| [FlowSpec](https://arxiv.org/abs/2507.02620), 2025 | 5 × Jetson Orin Nano over a 1 Gb/s LAN | EAGLE-2 tree segments of up to 16 tokens | 1.45–1.70× over plain pipelining |
| [CMP 170HX project](https://github.com/allover326/deepseek-v4-cmp170hx) (S) | 4 × CMP 170HX mining cards, PCIe Gen2 x4, no P2P, vLLM pipeline parallel | DSpark | 50.8 → 98.1 tok/s single stream (1.93×); tensor-core cards, where batched verify is cheap |

SpecPipe and FlowSpec push several tokens per stage, which suits tensor-core GPUs; on the P106 the chain variant (one
token per stage) keeps every step on the fast kernels. SPD is the closest design.

### Expected gain (est.)

Model: speed-up = T / (T − h·(T − c) + w·d), where h is the share of tokens whose guess was right, w the share with a
wrong guess in flight, c the cycle time (the slowest stage with its hop) and d the delay a wrong guess causes (≤ c).
Six cards: T = 26.8 ms, c ≈ 1.15 × T/6 = 5.1 ms (stage imbalance and the power limit), d ≈ 2.5 ms.

n-gram guesses where the context has a ≥ 6-token match, with Q64's measured copy statistics (share of copied tokens,
share of them with a match, share of matches that continue correctly):

| Output (Q64 transcripts) | h | Pipelined (est.) | Batched `ngram-copy`, 2 tokens (Q64, measured cost) |
|---|---|---|---|
| Chat | 1.6 % | 1.01× | 0.99× |
| Code | 14.6 % | 1.13× | 1.02× |
| Whole-file edits | 43 % | 1.53× | 1.08× |
| Agent turns | 8.3 % | 1.07× | 1.00× |
| All 65,520 tokens | 21.5 % | **1.21×** | 1.03× |
| File rewrites | 96 % | 3–4.5× (one token per cycle caps it at 5.3×) | 1.26× |

A wrong guess costs only d, so shorter matches can be tried too; the simulator below decides the threshold.

A trained pipeline drafter that always guesses, per-token acceptance p (w = 1 − p):

| p | Six cards (T 26.8 ms) | Nine cards at 262k (T ≈ 31.3 ms, c ≈ 4.0 ms) |
|---|---|---|
| 0.5 | 1.56× | 1.68× |
| 0.6 | 1.81× | 1.99× |
| 0.7 | 2.17× | 2.45× |

SPD's 0.72–0.75 comes from a dense 4B model with reasoning off; Flint at temperature 1.0 with two thirds of its output in
reasoning (Q64) is placed at 0.5–0.7 here.

### Design in the fork

- **Executor.** A host loop that keeps up to S tokens in flight and launches card j's single-token graph for token i
  only when card j has finished token i − 1 and token i is still valid (event queries, at most one queued graph per
  card). That is the cancellation: CUDA work cannot be withdrawn once queued, and enqueuing all cards at once (what
  `llama_decode` does) would make a miss wait behind every wrong token on every card, doubling the miss cost.
- **Inputs.** One set of input buffers per in-flight token (the scheduler's pipeline copies, `GGML_SCHED_MAX_COPIES`
  raised to S) and one captured CUDA graph per card and copy. Today `llama_context` synchronizes the scheduler before
  writing inputs whenever it reuses a graph with pipeline parallelism on (`llama-context.cpp`, graph reuse path); the
  executor bypasses that.
- **KV cache.** Cells for each in-flight position are allocated in order; a miss at position q releases q+1 onwards
  (`seq_rm`), as speculative decoding does today. The sliding-window cache keeps n_swa + n_ubatch cells padded to 256,
  enough for S ≤ 9 positions.
- **Mask.** Per in-flight token the host builds and copies the KQ mask to every card synchronously; candidate 3 removes
  that.
- **Drafter, phase 1:** the copy-gated n-gram lookup of Q53 (no draft length logic needed: one guess per position).
  **Phase 2:** a pipeline drafter on a spare card, trained like SPD's module on features from several depths of the
  in-flight tokens, reusing [the EAGLE-3 plan](../../../bench/eagle3_flint/TRAINING_PLAN.md)'s feature dump
  (llama.cpp's layer-input API) and a 32k draft vocabulary. It must deliver one guess per cycle: two 2048-wide layers
  and a 32k Q8_0 head read ~0.2 GB, ~1.5–2 ms on a P106 (est.), inside c.

### Qwen3.6 variant

Qwen3.6's MTP layer supplies the guesses: chained MTP drafts from the last verified hidden state feed card 1 one per
cycle. Its Gated DeltaNet layers carry recurrent state, so each card keeps a state snapshot per in-flight token (~2 MB
per layer, est.) to roll back on a miss. Four cards, T = 34.4 ms (29.0 tok/s, Q38): 1.4× at p = 0.5, 1.8× at p = 0.7
(est.), against 1.04× for batched MTP on the fork (Q43).

### Same executor, independent conversations

With S independent sequences staggered by one stage, every card works on some sequence's token at every moment:
aggregate ≈ 1/c ≈ 190–250 tok/s on six to nine cards (est.), five or more times one stream, each conversation at its single-stream
speed. llama-server today batches its slots into one micro-batch that passes all cards together, which uses the stock
multi-token kernels and leaves the other cards idle. This is the throughput mode for agents that run sub-tasks in
parallel, and the simplest first rig test of the executor (no guesses involved).

### First experiments

1. **Simulator (laptop, half a day).** Replay Q51/Q64's transcripts through a pipeline model with S, c and d as
   parameters: n-gram guesses at match lengths 2–12, and the acceptance an oracle drafter would need per workload for
   1.5×. Output: the h/w table above from real positions instead of averages.
2. **Overlap on the rig (half a day).** A standalone CUDA program, or the executor with independent sequences: S cards
   each run their real share of an S4-L0 token, staggered, with host-staged hops. Measure the cycle time, clocks and
   power with all cards busy, and host CPU per token. Gate: cycle ≤ 0.3 T on six cards.
3. **Executor in the fork (1–2 weeks)**, n-gram guesses, bit-identical output against plain decoding as the
   correctness test, then the drafter.

## 2. A persistent decode kernel per card

**Mechanism.** One resident kernel per card runs the card's layers for one token as a list of tasks (norm, Q/K/V,
attention, output, router, top-6, gate/up, down, shared expert, sums), with dependencies tracked by counters in global
memory instead of kernel boundaries. Weight streaming for the next task starts while the current one drains, and the
~25 small kernels per layer stop paying a launch, a ramp-up and a tail each.

**Evidence.**
- Hazy Research's Llama-3.2-1B megakernel uses 78 % of H100 bandwidth where vLLM and SGLang reach at most 50 %; its
  post names tail stalls, launch cost (1.3 µs per kernel even in CUDA graphs on an H100) and load latency after each
  launch as the overheads.
- Cohere's decode megakernel for a 30B MoE with 3.3B active parameters: 292 against 185 tok/s for vLLM at batch 1 on an
  H100 (1.58×, 62 % of speed of light). Counters instead of grid barriers; static task order plus work stealing for
  MoE tiles; real routing leaves more bubbles to fill than uniform routing.
- MPK (OSDI 2026) compiles models into one megakernel: up to 1.7× lower latency than kernel-per-operator systems.
- Flint's makers run a whole token in one persistent kernel on Apple GPUs (rec: 1.4–2.6× llama.cpp, [rec note](rec_engine.md)).

**On the rig (est.).** Streaming a card's share of 2.24 GB at ~140 GB/s through all phases plus ~0.6 ms of counter
waits (49 layers × ~6 dependency points × ~2 µs) would bring the token to ~20–22 ms with hops and host: +20–35 %.
Pascal supports what this needs (resident blocks, global atomics, `__threadfence`); risks are register pressure (the
fused Q8_0 kernel already reached 254 registers) and correctness of counting barriers. The end state can also drop the
host from the loop: each card polls a flag in mapped pinned memory, writes its output to the next card's mailbox, and
the last card samples on the GPU.

**First experiment (one week, one rig card).** In `bench/mmvq_pascal`, one Flint MoE-FFN layer (norm, F16 router,
top-6, fused Q5_0 gate/up + SwiGLU, Q8_0 down, shared expert, weighted sum) as one persistent kernel against the sum of
the fork's kernels for the same layer, with 6 random experts out of 256 per call. Gate: ≥ 1.3×.

**Prototype (2026-10-24): [bench/persistent_pascal](../../../bench/persistent_pascal/).** One MoE layer at one token as
one launch, next to the same layer as the fork's 13 launches in the same binary, for S4-L0's formats and for G3's
layers 9–45:
- **Design.** 2,944 warp-sized work items, numbered so that every dependency points to an earlier item (router rows,
  shared gate/up, routed gate/up expert by expert, shared down, routed down by 8-row group), taken from a counter in
  global memory. Dependencies are counters, not barriers: the warp that finishes the last router item computes the
  top-6; the warp that finishes an expert's last gate/up item quantizes that expert's activations, so each expert is a
  team and its down projection starts without waiting for the other five; the warp that adds the seventh contribution
  to an 8-row output group (six experts and the shared expert) writes those rows. The shared expert's gate/up fills
  the wait for the top-6.
- **No deadlock, no residency assumption.** With no grid barrier, the smallest unfinished item always has its inputs
  and a warp working on it, so the kernel completes at any grid size, including more blocks than fit at once; it needs
  no cooperative launch. A wait past a poll limit sets an error flag and the host falls back to the 13 launches.
- **Same arithmetic.** Every reduction, quantization and product is the fork's device code with the same inputs and
  order, so the output equals the 13-launch path's bit for bit, and a model would give the same logits.
- **Resources on the P106.** 128 registers, 37 or 47 KB of shared memory per 256-thread block: 2 blocks (16 warps) per
  SM, where the fork's product kernels run 20.
- **Checked** on an RTX 3090 natively and through the compute_61 PTX, both formats: identical bit for bit to the
  13-launch path in every intermediate and the output, at every grid size from 1 to 5,248 blocks, through chains of
  16 layers and in the captured timing graphs; every stage within tolerance of a double-precision reference; a fault
  injected into the top-6 is detected and the 13 launches take over with the correct output.
- **Expected on one P106 (est.).** 150–170 µs per S4-L0 layer (23.07 MB at 140–155 GB/s plus a few µs of prologue and
  tail), against ~190 µs for the bench's 13 launches and ~245 µs for the real token's layer (the fork's event profile,
  overhead removed): 0.9–4.5 ms per token over 47 layers, +4 to +20 %. Against the bench's own 13 launches that is
  1.12–1.27×, below the 1.3× gate; against the real token's layer it passes. The rig run settles which baseline the
  bench reproduces; its trace shows where any remaining time goes (waits for the top-6 or an expert's activations, the
  prologue, the tail). Commands: the bench's README, "Running on the rig".
- **Next, if it passes.** Attention adds three layer-wide dependencies per layer (all 32 heads before the output
  projection, all 2,048 residual rows before the FFN norm, and the same before the next layer's norm), each published
  by the warp that completes it as the top-6 is here; per-KV-head attention items with rec's layout; and prefetching
  the next phase's weight rows into L2 while a warp waits. Then one kernel per card per token.

## 3. Attention mask computed on the card

Built as a graph op (Q110, fork branch `pascal-p106-v2-gpumask`): each card expands its masks from 8 bytes per cell for
prompt micro-batches, with logits identical to the host-mask path; rig speed pending ([prefill note](prefill_pascal.md),
E5). Inside the attention kernel and for decode at depth it is still open.

**Mechanism.** Flint's masks are positional: causal for the 14 full-attention layers, causal plus a 513-token window for
the others, one sequence. The flash-attention kernel can compute them from the query's position and the cell positions
(plus a per-cell sequence id for several slots) instead of reading an f16 matrix.

**Why it matters (from the code and the [long-context note](long_context_cliff.md)).**
- The mask is a graph input of n_kv × n_ubatch f16 values. With pipeline parallelism on, the scheduler allocates
  `GGML_SCHED_MAX_COPIES` = 4 copies of every graph input on the host and of every split input on each card
  (`ggml-backend.cpp`, pass 6 of the split). At 262,144 tokens and `-ub 128` that is 256 MiB of pinned host memory and
  256 MiB of VRAM on each card; at `-ub 512`, 1 GiB each. This, not the kernels, is why long-context serving runs
  `-ub 128` ([setup](../../setup.md)).
- Small micro-batches make MoE prefill expensive: a 128-token ubatch selects ~243 of 256 experts per layer, so it reads
  nearly every expert weight for 128 tokens (est.). Served prompts read at ~7 ms per token plus a ~2 s fixed part per
  request (est., fitted to the per-turn table of the [chat benchmark](../../benchmarks/2026-10-23-flint-chat.md)),
  against ~5 ms per token in llama-bench's pp512.
- The host builds and copies the mask for every decode step and every speculative token (~1 ms of fill plus a
  synchronous copy per card at 262k, [attention note](pascal_attention_long_context.md)); Q65 and candidate 1 need it
  gone.

**Expected (est.).** Prefill at ≥ 100k context 1.3–2× from `-ub 512`–`1024`; −256 MiB VRAM per card at 262k (room
for G1-class or Q8_0 files with the full context); no per-step mask copies.

**First experiment (rig, 1 h, no code).** Build with `-DGGML_SCHED_MAX_COPIES=1` (one copy: `-ub 512` costs what
`-ub 128` costs today) and time 2k/8k/32k-token prompts at `-c 262144` with `-ub 128` (four copies, current build)
against `-ub 512` (one copy); read the compute-buffer sizes from the log. If larger ubatches win, the in-kernel mask
(TILE kernel and Q59's kernel, 3–5 days) makes them available with pipeline copies too.

## 4. Q6_0: a 6-bit rung between Q5_0 and Q8_0

**Mechanism.** ik_llama.cpp's Q6_0 block: FP16 scale, 16 bytes of low nibbles, 8 bytes holding two high bits per
weight, 26 bytes per 32 weights (6.5 bpw). Its CUDA dot product restores the high bits with one shift, mask and OR per
four weights, cheaper than Q5_0's fifth bit. A fork-only type, like Q4_1_64.

**Why it fits.** With the Pascal kernels, decode time follows bytes (Q8_0 costs 1.7× Q4_1 per weight, Q5_0 1.1×). The
mixes at the bar keep 63 % of their bytes in Q8_0, because Q5_0 costs too much quality in those tensors (Q73: a Q5_0
head +0.023 KLD; Q45: the shared expert must stay above Q4_1). On Llama-3-8B the KLD ladder is Q8_0 0.0014, Q6_K 0.0055,
Q5_0 0.022 (llama.cpp's perplexity README), and ik reports Q6_0 "almost as good as Q6_K": a 6-bit rung costs about a
quarter of what Q5_0 costs.

**Expected (est.).** All of G3's Q8_0 tensors in Q6_0: 1.46 → 1.12 GB, −0.34 GB per token; at ~138 GB/s for Q6_0 (the
Q5_0 kernel's rate) against 150 for Q8_0 that saves ~1.7 ms (+6–7 %), for an estimated +0.01–0.02 KLD. Per class
(head, shared expert, attention output first) +3–5 % at ≤ +0.01 KLD. The same rung lets the knapsack of Q77 trade
finer.

**First experiment (quant machine, 1 day).** G3 with the head, the shared expert, attention output 0–32 and attention q
0–24 moved one class at a time from Q8_0 to Q6_K, the mainline 6-bit type and a slightly optimistic proxy for Q6_0, KLD
against BF16. Then the Q6_0 type in the fork: CPU reference, quantizer, Pascal one-token kernel from Q5_0's, MMQ from
Q5_0's.

## 5. Screened output head

**Mechanism.** Flint's head is 150,144 × 2048 (327 MB at Q8_0, 2.2 ms per token). A screen computes approximate logits
for every row from a cheap copy: the head in Q4_1 (192 MB) or a rank-256 factorization (41 MB at Q8_0). A GPU top-K
picks K = 256–1,024 candidates, and exact Q8_0 logits are computed only for those rows (~2 MB). The sampler runs on
the candidates. Greedy decoding and top-k sampling with k ≤ K (the makers' setting is top-k 20, top-p 0.95) are exact as
long as the true top-k lies among the candidates; perplexity and KLD see approximate logits only for tokens outside the
candidate set.

**Evidence.** HiRE (Google, 2024) uses exactly this two-stage pattern (compressed approximation for high-recall top-k,
exact computation on the subset) for the softmax and FFN layers: 1.47× latency on a 1B model with nearly matching
accuracy.

**Expected (est.).** Flint: −0.13 GB per token with a Q4_1 screen (+3 %), −0.28 GB with a rank-256 screen (+7–8 %).
Qwen3.6's head is 248,320 × 2048 (540 MB at Q8_0, about a quarter of its bytes): +8 % (Q4_1 screen) to +15–18 %
(rank-256). The MTP draft head benefits the same way. Exactness here makes it compatible with the bar without a KLD
trade.

**First experiment (quant machine, hours).** Dump final hidden states for wikitext and Q51's transcripts; compute
exact head logits, Q4_1-screen and rank-r-screen logits; measure recall of the top-20 and of the top-p 0.95 nucleus at
K = 256, 512, 1,024. Gate: ≥ 99.9 % at some K ≤ 1,024.

## 6. KL-aware rounding for the always-read tensors

**Mechanism.** The tensors read on every token (attention q/o/k/v, shared expert, dense layers, head) hold most of the
bytes and, per Q18, most of the 4-bit loss. Error-feedback rounding into llama.cpp's own blocks (Q5_0, Q6_0, Q4_1)
instead of imatrix round-to-nearest lowers that loss at the same bytes. YAQA (ICML 2026) rounds against a
Kronecker-factored Hessian of the full model's KL divergence: on Llama 3.1 8B Instruct with INT4 absmax groups of 32
(Q4_0's layout) it gives KL 0.021–0.025 against 0.033 for LDLQ/GPTQ (−24 to −36 %); its sketches cost ~30–50 GPU-hours
for a ~10B model and need FP32 Hessians.

**Expected (est.).** If rounding lets attention q (0–24), attention output (0–32) and the shared expert go from Q8_0 to
Q5_0 at G3's KLD: 0.78 B parameters × 0.375 B = −0.29 GB per token, +8–9 %; to Q6_0, −0.2 GB, +5–6 %. MoE experts are
left to imatrix rounding (YAQA was not run on MoE, and per-expert Hessians for 256 experts × 47 layers are too large to
hold at once).

**First experiment.** Extend Q56's GPTQ pipeline (captured layer inputs, blocks written by script) from the head to
attention output, layers 0–32, in Q5_0; KLD against BF16. Gate: GPTQ recovers at least half of the Q8_0 → Q5_0 loss for
that class; only then build YAQA's sketch for the dense part of the model.

## 7. Query-aware KV page selection at long context

**Mechanism (Quest).** For each page of 16–64 cached tokens and each KV head, keep the channel-wise minimum and maximum
of the keys. At decode, an upper bound of q·k per page picks the top pages; attention reads only those (union over the
8 query heads of a KV head), plus the sliding-window layers as today. Flint's full-attention layers use no RoPE, so key
statistics don't drift with position; the learned sinks stay a separate term.

**Evidence.** Quest (ICML 2024): 7.03× self-attention speed-up and 2.23× end-to-end latency at 30k context with a
2,048-token budget and 4-bit weights, with negligible loss on long-dependency tasks.

**Expected (est.).** Flint reads 28 KiB of KV per context token: 0.9 GB at 32k, 3.7 GB at 131k, 7.3 GB at 262k, against
2.24 GB of weights. With a 4k-token budget and min/max metadata (~3 % of the KV bytes), attention reads ~6 % of the KV at
131k. Over Q60's predicted 19–21 tok/s at 131k that is ~2×, at 262k 2.5–3×; at 32k +10–15 %.

**Quality and first experiment.** The quality bar's 512-token chunks cannot see this. On the quant machine, emulate page
selection as an extra mask in llama.cpp's attention and run a long-context retrieval and code task set at 32k–128k
against the dense kernel (no rig time). Build the kernel only after Q59/Q60 and only if the loss is within 2 points.

## 8. Intra-expert activation sparsity

**Mechanism.** Inside each selected expert, compute the gate product densely, keep the neurons whose activation clears a
calibrated threshold, and read only their rows of up and their columns of down. In GGUF, up rows are rows already;
down columns are not (blocks run along the 512-wide input), so down is stored transposed and computed as a weighted
sum of gathered rows (FP32 accumulation, ~2–3 instructions per Q8_0 weight, within the card's issue budget, est.).
[TEAL](https://arxiv.org/abs/2408.14690) generalizes the idea to every projection with column-major weights.

**Evidence.** [Park et al. 2026](https://arxiv.org/abs/2605.08575) find large intra-expert sparsity in eight
pretrained MoE models (up to 90 % at their cut-off of 95 % of baseline score) and gain 1.5–2.5× per MoE layer, 1.04–1.20×
end to end at batch 1. On Qwen3.5-35B-A3B, average accuracy falls from 70.9 to 68.2 at 60 % sparsity and 67.5 at 85 %;
they leave shared experts dense. TEAL on Llama-3-8B: perplexity 5.87 → 5.94 at 25 %, 6.21 at 40 % (no MoE tested).

**Expected (est.).** G3's routed experts read 0.65 GB; with the gate dense, 50 % sparsity saves 0.22 GB (+5–6 %), 80 %
saves 0.35 GB (+9 %). At this project's bar only 20–40 % is likely to fit, so +2–4 % is the realistic case.

**First experiment (quant machine, 1–2 days).** A graph-level threshold on the expert intermediate (and, separately, on
the attention-output input) of G3, thresholds calibrated per layer for 20/30/40/60 % sparsity, KLD against BF16. Build
nothing unless 30 % costs ≤ 0.01 KLD.

## 9. Expert-contiguous weight layout

**Observation.** In a real S4-L0 token the routed gate/up call takes ~33 µs and the routed down call ~25 µs longer than
the same shapes in the microbenchmark, against ~13 µs for attention q; the routed weights of a layer are spread over
285–370 MB per card ([kernel note](pascal_mmvq_integration.md) section 10).

**Hypothesis.** Address translation: with 2 MiB pages, six random experts touch ~18–36 pages per layer in three
separate tensors. Volta's L1 TLB covers 32 MiB and its L2 TLB ~8 GiB, "larger than previous architectures"
([Jia et al.](https://arxiv.org/abs/1804.06826)), so a P106's L2 TLB likely covers less than the ~5.5 GB of weights each
card sweeps per token.

**Mechanism and gain (est.).** Allocate each layer's experts as one buffer with expert e's gate, up and down rows
together (~2.1 MB in Q5_0) and give the three tensors an expert stride (`nb[2]`) over it; the one-token `MUL_MAT_ID`
path and MMQ already address experts through that stride. If the 12–20 µs per call of excess over attention q is
translation, that is 1.1–1.9 ms per token (+4–7 %); if the whole gap, up to ~2.7 ms.

**First experiment (one rig card, 1 day).** In `bench/mmvq_pascal`, run the routed shapes with 6 random experts out of
256 in separate 370 MB tensors, then with the interleaved layout. Gate: the spread case reproduces ≥ 10 µs of the gap and
the interleaved case removes most of it.

## 10. Learned expert skipping

**Mechanism.** ZEDA adds parameter-free zero-output experts to each MoE layer and trains the router by two-stage
self-distillation against the frozen original, so easy tokens spend some of their six slots on experts that read no
weights. Unlike a fixed lower expert count (Q6), the model learns where skipping is cheap.

**Evidence.** On Qwen3-30B-A3B and GLM-4.7-Flash: over 50 % of expert FLOPs removed "at marginal accuracy loss" across
11 benchmarks, 1.20× end to end.

**Expected (est.).** Routed experts are ~28 % of G3's bytes; skipping 30–50 % of slots saves 0.2–0.33 GB, +8–14 %.
llama.cpp needs `MUL_MAT_ID` to skip empty slots (open PR #26631 lets it skip index −1). Training needs Flint's full
forward and backward with frozen 4-bit experts on the quant machine's RTX 3090s.

**First experiment.** Router-only adaptation of a few layers on cached hidden states, KLD of the adapted model against
BF16 before any quantization. Gate: ≤ 0.03.

## Checked, not worth it here

| Idea | Why not (est. where marked) |
|---|---|
| Trellis codes (QTIP, EXL3) | QTIP's computed codes take up to ~4 instructions per weight on modern GPUs; on sm_61 a 32-bit multiply-add is three XMADs and 3INST's FP16 arithmetic runs at 1/64 rate, so ~5–7 per weight: 13–28 per byte at 2–3 bpw against the ~6.4 the card can issue per byte. At 3 bits that lands at Q4_1's time per weight with no better quality (est.) |
| Vector quantization (AQLM, QuIP# lattices) | Shared-memory codebook lookups with random bank conflicts: ~5 instructions per byte at 4 bpw before conflicts; no gain over Q4_1 per unit time (est.) |
| Learned per-row 4-bit lookup values (any4-style) | A lookup format decodes like IQ4_XS on the fork (94–104 GB/s against 145 for Q4_1): 4.25–4.5 bpw at that rate costs more time per weight than Q4_1 (est.) |
| Expert merging, shared base + low-rank expert deltas | Merging keeps 6 experts read per token; low-rank deltas would cut routed bytes but fine-grained 256-expert models lose most under merging (REAP's study, [large-MoE note](models_large_moe_and_compression.md)); heavy retraining |
| Lookahead / Jacobi decoding | Needs wide multi-token verify steps, the expensive case on this card |

## What this changes in the current plan

- **Speculation is not blocked by verify cost.** Q19/Q20/Q43/Q64 measured batched verification. Pipelined speculation
  never verifies a batch, so it does not need Q78's multi-token kernels to pay, and a Flint drafter (Q20, dropped) is
  worth training again, as a single-token pipeline drafter with partial-depth features.
- **Q64's "only 2-token drafts pay" is a property of batched verification.** Pipelined, long copies pay in full (est.
  3–4.5× on file rewrites against 1.26×), so the copy gating and verify-width plumbing of Q52/Q53 matter less than an
  executor.
- **Q78's multi-token kernels keep their value** for prompt tails, small per-expert batches in prefill and
  two-candidate pipeline steps, not as the gate for speculation.
- **The format ladder of Q73/Q77 lacks a 6-bit rung**; Q77's Q8_0 → Q5_0 sensitivity scan should include Q6_K (proxy)
  or Q6_0.
- **`-ub 128` at long context is set by the mask's four pipeline copies**, so long-context prefill is a software limit;
  Q80's sweep should vary `GGML_SCHED_MAX_COPIES` together with `-ub`.
- **No community result on P106/P104/P102 cards beats the fork.** The one mining-card project with speculation under
  pipeline parallelism runs GA100-based CMP 170HX cards.

## Already covered here

| Idea found in this research | Where it is |
|---|---|
| Quantization-aware recovery into Q4_x (Gemma 3's Q4_0 QAT cut the perplexity loss by 54 % in ~5,000 KD steps) | Q40 (planned); highest-ceiling byte lever (all-Q4_1 at the bar would read ~1.6 GB, est.) |
| GPTQ-rounded output head | Q56 (running); candidate 6 extends it |
| Fused Q/K/V, shared expert as a seventh expert, one activation quantization per shared input | Q14, Q37, Q38 step 9 |
| FP32 activation sums, rescale removal | Q76, Q22 |
| EAGLE-3 / DFlash with batched verify, n-gram drafting with batched verify, CUDA graphs per verify width | Q19, Q20, Q53, Q64, Q52 |
| Pascal decode attention reading K/V once per KV head; q8_0 KV in TILE | Q59, Q60, Q61 |
| Host mask fill and copies for speculative steps | Q65; candidate 3 extends it to prefill and VRAM |
| Prefill `-ub`/`-b` sweep, Pascal MMQ tiles | Q80, PR #30235 (E2.3) |
| Hop cost between cards, tensor parallelism | Q81, E2.9 (ruled out on x1) |
| Multi-token kernels | Q78 |
| Expert tiering, fewer experts per token, expert pruning | Q57, Q6, survey A/C (no bytes per token saved) |
| Rotations, group-64 Q4_1, router precision | Q13/Q15, Q16/Q28, Q23/Q35 |
| Vocabulary trimming of the head (lossy) | Surveys A 3.5 and C 3.13; candidate 5 is the exact variant |
| Vulkan, `GGML_CUDA_GRAPH_OPT`, fusion switch | Q34, E2.13 |
| Headless mode, memory overclock, second SO-DIMM | Q83, Q86 |
| GPU sampling for tool requests, reasoning budget | Q67, Q85 |
| ik_llama.cpp types and engine, MXFP4/NVFP4 | E2.11, survey B L8/L9 |
| Persistent kernel mentioned as a long-term option | Survey B L7, [rec note](rec_engine.md); no registry row until now |

## Not verified

- No pipelined decoding has run on these cards: cycle time with all stages busy, the clock under sustained load at
  75 W, host dispatch cost per in-flight token on the Celeron, and the h/w shares outside Q64's averages are estimates.
- Flint's acceptance with a pipeline drafter: SPD's 0.72–0.75 (derived from its step acceptance) is for a dense 4B model
  with reasoning off.
- A megakernel on sm_61: register pressure and counter-based dependencies are untested on Pascal; the +20–35 % rests on
  H100 results and this note's time split.
- Q6_0's quality on Flint: ik's statement and Llama-3-8B's Q6_K numbers only.
- Screen recall on Flint's and Qwen3.6's heads; HiRE's numbers are for a 1B model on TPU.
- YAQA on an MoE, and its cost on the quant machine's GPUs.
- Quest on GQA layers with learned sinks and without RoPE, and the long-context quality it costs.
- Intra-expert sparsity and ZEDA on Flint; both papers' accuracy criteria are looser than this project's bar.
- That the routed-expert gap is address translation; the P106's TLB sizes are not measured (only Volta's are
  published in the cited study).
- The ~2 s fixed part of each served prompt is a fit to five turns; its cause is unknown.
- Whether CUDA graph conditional nodes (an alternative way to skip invalidated in-flight work) are available on sm_61.
- The CMP 170HX project's numbers are one repository's measurements.

## Sources

Pipelined speculation and coupling
- Y. Yu et al., "Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline
  Parallelism", arXiv 2605.30852 (v3, 2026-07-27), [abs](https://arxiv.org/abs/2605.30852),
  [html](https://arxiv.org/html/2605.30852), [code](https://github.com/yuyijiong/speculative_pipeline_decoding) (P)
- "SpecPipe: Accelerating Pipeline Parallelism-based LLM Inference with Speculative Decoding", arXiv 2504.04104,
  [html](https://arxiv.org/html/2504.04104v2) (P)
- B. Butler et al., "PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation", SC 2024,
  [arXiv 2407.11798](https://arxiv.org/abs/2407.11798), [code](https://github.com/AutonomicPerfectionist/PipeInfer) (P)
- X. Liu et al., "FlowSpec: Continuous Pipelined Speculative Decoding for Efficient Distributed LLM Inference",
  [arXiv 2507.02620](https://arxiv.org/abs/2507.02620), [code](https://github.com/Leosang-lx/FlowSpec) (P)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding",
  [arXiv 2408.07978](https://arxiv.org/abs/2408.07978) (P)
- DeepSeek-V4-Flash on 4 × CMP 170HX, [repository README](https://github.com/allover326/deepseek-v4-cmp170hx) (S)

Persistent kernels
- Hazy Research, "Look Ma, No Bubbles! Designing a Low-Latency Megakernel for Llama-1B", 2025-05-27,
  [blog](https://hazyresearch.stanford.edu/blog/2025-05-27-no-bubbles) (P)
- Cohere, "Megakernels", 2026-09-08, [blog](https://cohere.com/blog/megakernels) (P)
- "MPK: A Compiler and Runtime for Mega-Kernelizing Tensor Programs", OSDI 2026,
  [arXiv 2512.22219](https://arxiv.org/abs/2512.22219), [code](https://github.com/mirage-project/mirage/tree/mpk) (P)

Formats, rounding and training
- ik_llama.cpp, "Adding Q6_0", [PR #77](https://github.com/ikawrakow/ik_llama.cpp/pull/77) (2024-10-02);
  `block_q6_0` in `ggml/src/ggml-common.h` and `vec_dot_q6_0_q8_1` in `ggml/src/ggml-cuda/vecdotq.cuh` (P)
- llama.cpp, [perplexity README](https://github.com/ggml-org/llama.cpp/blob/master/tools/perplexity/README.md),
  LLaMA 3 8B KLD table (P)
- A. Tseng, Z. Sun, C. De Sa, "Model-Preserving Adaptive Rounding" (YAQA),
  [arXiv 2505.22988](https://arxiv.org/abs/2505.22988), [code](https://github.com/Cornell-RelaxML/yaqa) (P)
- Google, "Gemma 3 QAT Models", [blog](https://developers.googleblog.com/gemma-3-quantized-aware-trained-state-of-the-art-ai-to-consumer-gpus/) (P)
- "QTIP: Quantization with Trellises and Incoherence Processing", [arXiv 2406.11235](https://arxiv.org/abs/2406.11235) (P)

Head, attention, sparsity, experts
- Samaga B L et al., "HiRE: High Recall Approximate Top-k Estimation for Efficient LLM Inference",
  [arXiv 2402.09360](https://arxiv.org/abs/2402.09360) (P)
- J. Tang et al., "Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference", ICML 2024,
  [arXiv 2406.10774](https://arxiv.org/abs/2406.10774) (P)
- J. Liu et al., "Training-Free Activation Sparsity in Large Language Models" (TEAL), ICLR 2025,
  [arXiv 2408.14690](https://arxiv.org/abs/2408.14690), [code](https://github.com/FasterDecoding/TEAL) (P)
- J. Park et al., "Uncovering Intra-expert Activation Sparsity for Efficient Mixture-of-Expert Model Execution",
  [arXiv 2605.08575](https://arxiv.org/abs/2605.08575) (P)
- X. Lv et al., "Post-Trained MoE Can Skip Half Experts via Self-Distillation" (ZEDA),
  [arXiv 2605.18643](https://arxiv.org/abs/2605.18643) (P)
- Z. Jia et al., "Dissecting the NVIDIA Volta GPU Architecture via Microbenchmarking",
  [arXiv 1804.06826](https://arxiv.org/abs/1804.06826), section 3.8 (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q19, Q38, Q43, Q49, Q51, Q58, Q64, Q73, Q76, Q78), the
  [kernel integration note](pascal_mmvq_integration.md), [speculation plumbing](speculation_plumbing.md),
  [long-context cliff](long_context_cliff.md), [attention note](pascal_attention_long_context.md),
  [n-gram replay](../../../bench/ngram_replay/README.md), [EAGLE-3 training plan](../../../bench/eagle3_flint/TRAINING_PLAN.md),
  [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md)
- Fork source, branch `pascal-p106-v2` (`032b130f4`): `src/llama-context.cpp` (graph reuse synchronizes the scheduler
  with pipeline parallelism on), `ggml/src/ggml-backend.cpp` (graph-input copies per pipeline copy), `src/llama-graph.cpp`
  (`build_attn_inp_kq_mask`: f16 n_kv × n_tokens mask with flash attention)
