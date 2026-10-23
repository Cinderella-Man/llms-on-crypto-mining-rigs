# HyperQwen's speed-ups for Qwen3.8-27B, and which of them carry over to Flint on the P106 rig (2026-10-23)

Scope: [HyperQwen](https://github.com/syv-ai/HyperQwen) (read at commit `ce9615c`, 2026-10-22) serves the dense hybrid
Qwen3.8-27B on one RTX 3090 with a patched vLLM. Single-stream decode goes from 46 tok/s without speculation to 127–133
tok/s, and to 381 tok/s while the answer quotes its own prompt. This note lists what it does, why each piece works there,
and which pieces could speed up Flint (40B-A2.5B MoE) on nine P106-090 cards in llama.cpp. Conventions are those of the
other notes:
- "Claim to test" marks anything not measured on this rig.
- "(my calc)" marks this note's own arithmetic.
- "P" and "S" in source lists mean primary (code, docs, measurements in the repo) and secondary.

Starting point on the rig, measured: S4-L0 (the quality-bar mix) decodes 37.4 tok/s in llama-bench tg128 on six cards with
the fork's Pascal kernels (26.8 ms per token). It reads 2.24 GB of weights per token:
- dense part 1.52 GB: attention 0.74, output head 0.33, shared expert 0.31, the two dense layers 0.08, routers 0.05;
- routed experts 0.72 GB.

In a five-turn chat at temperature 1.0 on nine cards it generates 31.6–32.2 tok/s.

## Headline findings

1. **HyperQwen's gain is speculative decoding on a model whose verify step is almost free.** On a 3090 a dense 27B model
   reads the same ~13 GB whether it checks 1 token or 16, and the tensor cores have compute to spare. So every accepted
   draft token is nearly pure profit. Their per-step costs:
   - the drafter: 0.5–1 ms per MTP draft; ~5 ms for a 7-token DFlash2 block in bf16, 3.4 ms less after int4;
   - each extra verify position: ~1 ms of attention at 25k context, i.e. a 16-token block costs +6 % per step over an
     8-token block at 1.5k context and +27 % at 25k.

   Byte cuts (int8/int4 heads and embeddings) won VRAM for the KV cache and made the drafter cheap. They added little speed
   to the target itself.
2. **Flint's verify step is not free, so the same recipe yields far less.** Each verified token picks its own 6 of 256
   experts, so an n-token verify reads roughly n times the expert bytes. The fork's fast kernels also run only for a single
   token. An extra verify position costs ~5–6.5 ms on a 26.8 ms token (~20–25 % each, against HyperQwen's ~1–3 %):
   ~0.7 GB of extra experts at 120–150 GB/s, plus the extra column of the dense products (my calc). A learned drafter
   for Flint therefore models at 1.0–1.3×, not 2.8×.
   - Other old-GPU owners see the same pattern on MoE targets. A P40 running Qwen3.6-35B-A3B with MTP peaks at 2 drafts
     (1.36×) and is slower than plain decoding at 6. Two V100s with DFlash on the same model peak at 1.21× (2 tokens) and
     reach 0.83× at 8.
   - The open variable is routing locality. On Qwen3-Coder-30B-A3B, adjacent tokens share about half their experts, so
     an 8-token verify touches 51 % of the experts random routing would. Nobody has measured this for Flint, and it is a
     1–2 h statistic on the quant machine.
3. **What carries over is mostly the speculation control logic and the drafter economics, not the kernels.** Three pieces
   change what we should build:
   - **Lookup drafting with a copy-gated long block.** Draft from the request's own history. Ask for a long verify block
     only after two consecutive steps accepted everything, because match length alone is a poor trigger. llama.cpp
     already does the lookup and the exact acceptance; the gate is missing for n-gram drafts. An adaptive MTP-depth
     controller built on the same signal is open upstream (PR #27210).
   - **CUDA graphs for every verify length.** HyperQwen lost 7 % when short steps fell back to uncaptured execution. Our
     llama.cpp keeps one graph per split and re-captures on every shape change, so drafts of varying length run
     uncaptured. Graphs are worth +14 % for Flint on this rig.
   - **A draft vocabulary counted over the model's own outputs, and a quantized draft head.** HyperQwen measured +10 %
     from the vocabulary list alone. Both apply directly to the planned EAGLE-3 head for Flint (Q20) and to the 32k trim
     of Qwen3.6's MTP head (Q43).
4. **Two serving issues weigh more for agent traffic than any kernel.** HyperQwen's sampler and prefix-cache work points
   at both.
   - Tool requests switch backend sampling off in llama.cpp. Each sampled position then copies 600 KB of logits over a
     PCIe 1.0 x1 link and runs a top-k pass on the 2-core host: ~3 ms per position (my calc), ~11 % of a token (est.),
     and (k + 1)× that on every verify step.
   - On a sliding-window model, a follow-up turn whose history differs more than ~770 tokens back re-processes the whole
     prompt unless context checkpoints are on. A Flint checkpoint is only 35 MiB.
5. **At long context, attention becomes the next kernel target, which is HyperQwen's long-context lesson too.** Their
   int8-KV attention kernels gave +12–14 % at 66–100k on a 3090 and ~2× at 128k on two 3080s. On the rig, Flint's KV
   cache reads ran at ~57 GB/s on the pre-fork build (my calc from the depth benchmark), against 120–150 GB/s for the
   fork's matrix-vector kernels. At 32k of context, reading the KV cache may take ~60 % as long as reading the weights
   (claim to test).
6. **Many of HyperQwen's pieces do not apply to this rig:**
   - int8 tensor-core GEMMs, int8 prefill attention, fp8 KV and Marlin tuning (sm_61 has no tensor cores and no fp8);
   - fp16 recurrent state (Flint has no recurrent layers);
   - tensor parallelism (x1 links);
   - KV compression for capacity (262k already fits).

## What HyperQwen does

All numbers are HyperQwen's, measured on an RTX 3090 at 250 W ([docs/optimizations.md](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/optimizations.md),
[single-user/README.md](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/single-user/README.md); P).

**Single stream, step by step** (T = model default / greedy):

| Step | tok/s | Tokens per step |
|---|---|---|
| No speculation | 46 / 46 | 1.0 |
| MTP-2 as shipped (bf16 drafter, full 248k-row head) | 66 / 79 | 2.1 / 2.4 |
| MTP-4, int8 drafter, 40k-row draft head, fp16 recurrent state | 78 / 99 | 2.2 / 2.7 |
| + probabilistic draft sampling | 90 / 98 | 2.6 / 2.7 |
| + sort-free sampler, split-KV verify attention | 93 / 99 | 2.6 / 2.6 |
| + draft vocabulary counted over the model's own outputs (97.5 % coverage instead of 92 %) | 107 / 109 | 2.9 / 2.9 |
| + GPTQ int4 output head, GPTQ int4 MTP module | ~114 / 118–124 | 2.8 / 2.9–3.0 |
| DFlash2 block drafter instead of MTP (int4, 1.19 GB) | 118 / 126 | 3.14 / 3.34 |
| + lookup drafting from the context | 130; up to 259 when the answer reproduces the context | 3.3–7.8 |
| + a 16-token verify block that the lookup fills | 133; up to 381 when reproducing | 3.4–15.0 |

**The levers, grouped:**

| Lever | Mechanism | Measured effect | Precondition |
|---|---|---|---|
| Quantized output head and embedding (int8, then GPTQ int4 on 300k captured hidden states) | Fewer bytes per draft and per token; 2.6 GB of VRAM freed for KV | 370 → 516 tok/s aggregate at 64 streams via more cache pages; single stream +2 % (head), +4 % (MTP module) | Untied 248k vocabulary in bf16; GPTQ keeps quality (+0.6 % perplexity vs +1.5 % for round-to-nearest) |
| Draft vocabulary of 40,960 ids counted over 5.4M tokens of the model's own output | The drafter scores 41k rows instead of 248k; ids outside the list are certain rejections that also end the chain | +10 % single stream over a web-text list (98.0 → 108.6 greedy, 90.0 → 107.4 default sampling); Chinese text drops to 1.06 tokens per step | Large vocabulary; traffic like the counting corpus |
| Quantized drafters (int4 GPTQ MTP; DFlash2 W4A16, 3.85 → 1.19 GB) | Drafter reads fewer bytes | DFlash2 step 31.4 → 28 ms; acceptance kept at greedy, −5 % at default sampling | Drafter cost a large share of the step |
| Probabilistic drafts, truncated to the target's top-k/top-p | Drafts sampled from the draft distribution; exact rejection sampling | +15 % at T > 0, nothing at T = 0 | Sampling at T > 0 |
| Sort-free top-k, multi-block softmax | One `topk` instead of a 248k sort per row; 140 µs → 10 µs softmax | ~+4 % together with split-KV attention | Large vocabulary, several rows per step |
| Split-KV attention for the multi-token verify | FA2 splits the KV only for one query token; a Triton kernel splits it for 2–32 | 2,053 → 583 µs per layer at 25k with 16 queries | Long context, few heads |
| Lookup drafting (LABD) | Longest, most recent match of the generated suffix (6–12 tokens) anywhere in the request's history; point-mass draft distribution keeps it exact; a match may overlap its own suffix (periodic text) | 159 → 381 tok/s reproducing a 25k document; +2–3 % on prose | The answer repeats the context |
| Copy-gated long verify block | 16 positions only after two consecutive steps that accepted the whole short block, with the match long enough to fill the tail; held 3 steps after the flag drops; decision read one step stale from pinned memory | +47 % where the model reproduces its context, a few % elsewhere; match length alone needed a threshold of 22 to stop losing on prose | Extra verify positions cost something |
| CUDA graphs for both block lengths | Short steps had fallen back to piecewise execution | 27.9 → 25.9 ms per step | Verify length varies |
| Prefix caching for a hybrid model | Attention KV plus recurrent-state snapshots | Second chat turn on a 24k document 22.9 → 1.15 s | Multi-turn traffic |
| fp16 recurrent state | Halves the DeltaNet state | All 64 requests resident; 516 → 707 tok/s aggregate | Recurrent layers |
| int8 activations on tensor cores (W4A8 Marlin), int8 QK prefill attention | 2–4× the fp16 MMA rate | Prefill +27–30 %; batch decode 707 → 1,042 tok/s | Tensor cores |
| int8 / fp8 / 4-bit KV kernels (pth-flash-attn, FA2 fp8, KVarN) | Read quantized KV directly, split across blocks | Decode +12–14 % at 66–100k (3090); 44 → 86 tok/s at 128k (two 3080s) | Long context; sm80+ |

Things HyperQwen measured that did not pay:
- **Deeper drafts.** MTP k = 5 and 6 were slower than k = 4.
- **Fine-tuning the MTP head** halved the KL divergence but left top-1 agreement unchanged.
- **Drafter-free chains.** Skipping the drafter while the lookup filled the block lost 6 %. A chain variant gained 7 % on
  copies but lost 8 % at T = 1.
- **Marlin tile tuning:** +2–20 % per GEMM in isolation, +0.4 % end to end.
- **DSpark** instead of the MTP head: 3.19 against 3.77 tokens per step.
- **DFlash2 at 112k:** half of MTP's speed outside copy work, because its drafter sees only a 2,048-token window.
- **The 16-token block on chat:** positions 7–14 supplied 0.65 % of accepted tokens.

## Why the verify step decides everything

A verify step of n tokens returns between 1 and n tokens. It pays when the accepted tokens outweigh the step's extra cost
over one plain token.

**On the 3090 with a dense model** the extra cost is attention and drafting. A 16-token block costs +6 % per step at 1.5k
context and +27 % at 25k, against an 8-token block. A drafter that gets 3 tokens per step therefore nearly triples the speed.

**For Flint on the rig** the extra cost is mostly expert bytes. Each token routes to 6 of 256 experts, and with random
routing the tokens in a step share almost none. Expected distinct experts per layer are 256·(1 − (250/256)ⁿ), and the
bytes per step follow from S4-L0's split of 1.52 GB dense and 0.72 GB routed (my calc):

| n verified | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Distinct experts per layer, random routing | 6.0 | 11.9 | 23.2 | 44.2 | 80.7 |
| GB read per step (× one token) | 2.24 (1.0) | 2.94 (1.3) | 4.30 (1.9) | 6.82 (3.0) | 11.2 (5.0) |
| Best case if all n are accepted, at equal efficiency | 1.0× | 1.5× | 2.1× | 2.6× | 3.2× |
| Distinct experts with Qwen3-Coder-30B-A3B's locality | 6.0 | 9.3 | 14.6 | 22.5 | ~32 |
| GB read per step (× one token) | 2.24 (1.0) | 2.63 (1.2) | 3.27 (1.5) | 4.22 (1.9) | ~5.4 (2.4) |
| Best case if all n are accepted, at equal efficiency | 1.0× | 1.7× | 2.7× | 4.3× | ~6.7× |

The locality rows scale the random-routing counts by what [arXiv 2609.22156](https://arxiv.org/html/2609.22156) measured
for Qwen3-Coder-30B-A3B (8 of 128 experts, A100, batch 1, T = 0.7). Its layer-averaged distinct experts were 12.1 / 18.2 /
26.1 / 30.2 for 2 / 4 / 8 / 11 verified tokens. That is 78 / 63 / 51 / 46 % of the independent-routing count, and the
16-token column extrapolates to ~40 % (my calc). Flint routes 6 of 256, a finer split, so its locality may be lower.

Exploiting locality also needs a kernel that reads each distinct expert once for all tokens routed to it. llama.cpp's
multi-token MoE matrix-vector path runs one warp per token and does not deduplicate (`mmvq.cu`).

The kernel side makes it worse today:
- **One-token kernels only.** The fork's Pascal matrix-vector kernels run only for one token (`mmvq-pascal.cu`), so any
  verify step falls back to llama.cpp's stock multi-token kernels.
- **MMQ above a few tokens.** On compute 6.1 the expert products leave the matrix-vector path for MMQ above 4 tokens
  (Q8_0) or 6 (Q4_x).

So the last row of the table is a ceiling for long verifies. It is approached only on text that is accepted wholesale: file
rewrites, edits that keep context, and quoted tool output. On prose it does not apply.

## What carries over, lever by lever

### 1. Lookup drafting with a copy-gated long block (the main transferable idea)

**What llama.cpp already has** (fork at upstream `d77256999`; P):
- **Lookup over the full history.** The n-gram drafters search the request's prompt plus accepted output: the server passes
  `slot.prompt.tokens` (`tools/server/server-context.cpp:3757`). `ngram-simple` takes the most recent earlier occurrence
  of the last `size_n` tokens and copies up to `size_m` following tokens (`common/ngram-map.cpp:69-111`).
- **Exactness at any temperature.** Every drafter without a draft distribution (n-gram, EAGLE-3, DFlash, greedy drafts)
  is verified by sample-and-match (`common/sampling.cpp:690-722`). That is the same as HyperQwen's point-mass proposal,
  so lookup drafts are exact at Flint's temperature 1.0 too.
- **Draft length.** It is fixed per type: `ngram-simple` copies min(`size_m`, tokens available); `ngram-mod` needs a
  24-token key and drafts 48–64; `ngram-map-k` can only shrink its length after misses (`ngram-map.cpp:386, 503, 535`).
- **Combining types.** Several types run in a fixed priority order, and the first non-empty draft wins
  (`common/speculative.cpp:2749-2761, 2959-2962`). There is no fusion and no rule that grows the block while a copy is
  running.

**What HyperQwen adds, and why it matters more for Flint than for them:**
- **Gate the long block on acceptance history, not on match length.** HyperQwen's lookup needed a match-length threshold of
  22 before it stopped losing on prose. That is close to `ngram-mod`'s default 24-token key, which explains why
  `ngram-mod` drafts rarely. The reliable signal was "the last two steps each accepted every drafted token". On Flint a
  wasted long block costs several token-times (table above), against ~27 % of one step for HyperQwen. So the gate is
  worth more here.
- **Hold the long block for a few steps** after the flag drops (3 steps, single request). Copies have short unmatched
  lines; without the hold, tokens per step on a copy fell from 15.2 to 13.9.
- **Let a match overlap its own suffix,** so indentation, list markers and fences are proposed from their period.
  `ngram-simple` never copies into the draft itself, so it rejects short-period repetition.
- **Read the decision without a device synchronise.** HyperQwen measured 5 % for a synchronous read per step. In
  llama-server the acceptance count is already on the host after sampling, so the gate is free there.

**Proposed fork design for Flint (claim to test).** A drafter type in `common/speculative.cpp` that wraps `ngram-simple`'s
lookup with three states:
1. **No draft** unless the match is at least `N_STRONG` tokens long (6–8; start at HyperQwen's 6). A plain token keeps
   the fast one-token kernels and CUDA graphs.
2. **Short draft** of k_s = 2–3 tokens on a strong match. This stays inside the matrix-vector path for every format, and
   a wrong guess costs ~0.25–0.5 of a token (est.: a 3-token step on the stock multi-token kernels at 1.24–1.5× a
   one-token step on the Pascal kernels).
3. **Long draft** of k_l = 7–15 tokens after two consecutive short steps accepted everything, while the match still has k_l
   tokens left. Hold it for 3 steps, and drop to state 1 on the first rejection after the hold.

On text that is copied wholesale, the long state could reach the table's 2.6–3.2× with random routing, or more with real
locality (est.). How much of real agent output is copy-like decides the session-level gain.

**Upstream and published work with the same idea** (P):
- **Adaptive MTP depth, llama.cpp [PR #27210](https://github.com/ggml-org/llama.cpp/pull/27210)** (open).
  - Mechanism: `--spec-type draft-mtp-adaptive` climbs one draft step after a run of fully accepted verifies, and drops
    one when misses (weighted by unaccepted tokens) pass a threshold.
  - Qwen3.8-27B Q8_0 on two Radeon R9700 GPUs, against fixed depth 3: reasoning 51.4 vs 51.7 tok/s, prose 56.6 vs 55.9,
    code 78.0 vs 69.3, verbatim recall 147.5 vs 81.9.
  - Its controller is the nearest code to HyperQwen's gate inside llama.cpp, and a cherry-pick candidate. It is written
    for MTP; the n-gram drafter would need the same logic.
- **`ngram-mod` on copying** (same PR's measurements, Qwen3.8-27B): 292.9 against 29.9 tok/s on recall, no change on
  reasoning, prose and code.
- **SuffixDecoding** ([arXiv 2411.04975](https://arxiv.org/html/2411.04975)): lookup over a suffix tree of the prompt
  and of earlier outputs.
  - Mechanism: it caps each draft at α × match length, with α = 1–4 for agentic work.
  - Results: 5.3× on AgenticSQL, 2.5× on SWE-Bench offline, 1.8–4.5× end to end in vLLM.
  - A llama.cpp port is open as PR #26283.

### 2. CUDA graphs for every verify length (prerequisite for any speculation here)

HyperQwen's short steps ran uncaptured when only the long length was captured (27.9 against 25.9 ms per step).

In our llama.cpp (P):
- The CUDA graph cache keeps one entry per backend split, keyed by the address of its first node
  (`ggml-cuda.cu:2759-2761`).
- Any change of node properties resets the warm-up, so the next step runs eagerly, and the one after re-captures
  (`ggml-cuda.cu:5036-5078`).
- llama-context reuses a built graph only for an identical batch shape (`llama-context.cpp:1530-1573`), so the 2-core
  host also rebuilds a ~3,500-node graph on each change.

A drafter that alternates between "no draft" (1 token) and "draft" (k + 1 tokens) therefore runs most steps uncaptured.
CUDA graphs are worth +14 % for Flint (measured, [Flint benchmark](../../benchmarks/2026-10-22-flint.md)).

**Fix in the fork (claim to test):** add the batch's token count to the graph-cache key, and keep 2–3 shapes per split
(1, k_s + 1, k_l + 1). Use the same fixed lengths in the drafter. This also helps `-np 2` serving, where the batch
alternates between 1 and 2 tokens as requests start and stop.

### 3. Draft-head economics for a learned drafter (Q20 for Flint, Q43 for Qwen3.6)

**Count the draft vocabulary over the model's own outputs.**
- HyperQwen's list of 40,960 ids covers 97.5 % of generated tokens (96 % on code). The web-text list covered 92 % (83 % on
  code) and cost 10 % of single-stream speed.
- 49k ids measured no better than 41k.
- The failure mode is traffic unlike the corpus: Chinese text fell to 1.06 tokens per step.

For Flint the 32k list of [TRAINING_PLAN.md](../../../bench/eagle3_flint/TRAINING_PLAN.md) is already planned from
regenerated Flint responses.
- Record coverage on code separately, and prefer ~40k ids if 32k covers less than ~96 % of code tokens.
- Flint's vocabulary has 150,144 ids and hidden size 2048, so each 10k ids cost ~22 MB of Q8_0 per draft pass (my calc).

For Qwen3.6's 32k MTP trim (Q43), count the kept rows over Qwen3.6's own generated text. A web-text frequency list would
cost acceptance.

**Quantize the drafter, but calibrate it.**
- HyperQwen's round-to-nearest int4 head lost ~2 % acceptance, and GPTQ on captured hidden states lost none.
- For our heads the format choice is Q4_0/Q4_1 against Q8_0 on the fork's kernels: about half the draft-pass bytes (my
  calc).
- llama.cpp's quantizer has no GPTQ, so a calibrated 4-bit head would be written by a script and checked by acceptance.

**Sample drafts probabilistically at T > 0.**
- HyperQwen: greedy drafts were ~15 % slower at T > 0 and equal at T = 0.
- In llama.cpp only `draft-simple` and `draft-mtp` produce a draft distribution (`--spec-draft-sampling probabilistic`,
  `common/speculative.cpp:370-382, 1713-1719`). EAGLE-3 and DFlash drafts are greedy and verified by sample-and-match.
- Flint and Qwen3.6 run at temperature 1.0, so test `probabilistic` in Q43 at T = 1.0.
- For an EAGLE-3 head for Flint, emitting q would be fork work. Upstream merged probabilistic sampling for DFlash2 on
  2026-10-23 (PR #30663), after the fork's base.

**DFlash against EAGLE-3 for Flint.**
- llama.cpp in our tree runs DFlash, DFlash2 and DSpark drafters generically (`src/models/dflash.cpp`). Flint's graph
  already exposes the layer inputs they read (`src/models/flint.cpp:97`), so a trained checkpoint is the only missing piece.
- One DFlash pass drafts the whole block, where EAGLE-3 needs k sequential passes. Each pass costs a graph evaluation
  and launches on the 2-core host.
- DFlash's strength is long blocks, which Flint's expert fan-out punishes. At k = 2–3 it would win on drafting cost only.
- HyperQwen's DFlash2 beat a tuned MTP head by only a few per cent at C1 (118 / 126 against ~114 / 118–124 tok/s). It
  lost 2:1 at 112k context, and DSpark lost to MTP.
- **Gains on MoE targets are smaller everywhere they have been measured:**
  - The DFlash paper ([arXiv 2602.06036](https://arxiv.org/abs/2602.06036)) gets 3.5× on Qwen3-Coder-30B-A3B at
    HumanEval on a B200. Its concurrency-8 table gives 1.3–2.4× for 30–120B MoE targets.
  - Inco's DFlash2 drafter for Qwen3.6-35B-A3B
    ([model card](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2)) reaches 1.53–2.09× at T = 1 on a GB300,
    against 2.7–3.4× for the dense Qwen3.8-27B.
  - On an Intel iGPU the same MoE gains 1.3–2.2×, against 3.1–5.5× for a dense 27B at similar acceptance; the authors
    blame verify tokens routing to different experts ([blog](https://huggingface.co/blog/ofirzaf/intel-dflash-ptl)).
  - On two V100s, MTP beat DFlash at every block length
    ([sweep](https://github.com/thc1006/qwen3.6-speculative-decoding-v100)).
  - An early llama.cpp DFlash build ran Qwen3.6-35B-A3B on an RTX 3090 at 77.0 against 138.9 tok/s without it
    ([PR #22105 comment](https://github.com/ggml-org/llama.cpp/pull/22105#issuecomment-4412880745)).
- **Training.**
  - Code: SpecForge and vLLM's speculators both train DFlash and DFlash2. vLLM's Flint class has no aux-hidden-state
    interface yet, so hidden states would come from transformers, as in the EAGLE-3 plan.
  - Recipe: the DFlash paper trains on ~800k target-regenerated samples for 6 epochs and gives no GPU-hours.
  - Cost: a rough estimate is several times the EAGLE-3 pilot's 35–56 GPU-hours (inference).
- **Verdict for this rig:** a DFlash head for Flint would cost more to train than EAGLE-3. Its advantage, long blocks,
  is what the MoE verify cost and the 5-SM cards punish, so EAGLE-3 at k = 2–3 stays the better learned-drafter bet.
  Neither is worth training before the measurements in the last section.

### 4. Backend sampling for requests with tools

HyperQwen spent effort on the sampler (sort-free top-k, multi-block softmax, +4 % with the attention patch) because the
248k-wide rows were sampled several times per step. Our equivalent is moving logits off the GPU.

**What happens to tool requests** (P):
- llama-server's backend sampling (`-bs`) keeps the 150,144 logits on the GPU. It is switched off for any request with a
  grammar or a reasoning-budget sampler (`common/sampling.cpp:422-432`). Tool calls install a lazy grammar
  (`common/chat.cpp:1744-1747`), so every tool request runs without it.
- Each sampled position then copies 600 KB over the x1 link (`llama-context.cpp:2065-2077`), and the host runs a top-k 40
  partial sort over 150k entries.

**Cost** (pre-fork measurement, [speed-ups](../../benchmarks/2026-10-22-speedups.md)):
- `-bs` gained +8 % (3.3 ms per token at 41 ms). At the fork's 26.8 ms the same 3.3 ms would be ~11 % of the speed
  (est.).
- In speculation it is paid once per verified position.

For an agentic coding model nearly every request has tools, so the chat benchmarks overstate agent speed.

**Fix in the fork (claim to test):**
- Run the chain's top-k on the GPU and copy only the 40 candidates (320 B) per position.
- Apply the grammar and the reasoning budget to those on the host.
- Fetch the full row only when the grammar rejects every candidate. That is the same resample path llama.cpp's sampler
  already takes after a grammar rejection, so outputs keep llama.cpp's semantics.

First measure the real loss: the same prompt with and without `tools` in the request, with `-bs` on.

### 5. Prompt reuse across turns (already in use; keep it on)

HyperQwen's prefix caching took a follow-up turn from 23 s to 1 s. For Flint in llama-server (P):
- **When the whole prompt is re-processed.** The 35 sliding-window layers keep only pad(513 + n_ubatch, 256) cells. A new
  prompt that diverges from the cache more than ~770 tokens before its end is re-processed from token 0 unless a context
  checkpoint covers the divergence point (`server-context.cpp:4067-4164`, `llama-kv-cache-iswa.cpp:73`).
- **Why chat clients hit this.** They resend answers without their reasoning, so the history diverges at the start of the
  previous reply.
- **What a checkpoint costs.** Checkpoints are taken near the end of each prompt and at user-message starts, and they hold
  the SWA layers' window: 513 × 35 × 4 KV heads × 128 × 2 (K and V) × 2 bytes = 35 MiB at f16 (my calc).

[server_bench.sh](../../../scripts/server_bench.sh) documents this, and the five-turn chat run on the rig used
`--ctx-checkpoints 4`. Turns 3–5 there processed only the new 380–450 tokens of 1,600–2,400, at 72–84 tok/s.

Without checkpoints a 30k-token agent context would take minutes to re-read every turn at that speed (my calc). Keep 2–4
checkpoints for Flint wherever 4 GB of host RAM allows (35 MiB each). The setup guide's row on `--ctx-checkpoints`
should name sliding-window models, not only hybrids.

### 6. Attention at long context (HyperQwen's long-context lesson)

HyperQwen's long-context gains came from attention kernels that read a quantized KV cache directly (int8 with inline
scales, no dequantize pass) and split the KV across thread blocks. Results: +12–14 % at 66–100k on the 3090, and 44 → 86
tok/s at 128k on two 3080s.

**What the Flint numbers suggest** (pre-fork build, one repetition each; my calc):
- **Measured speed.** Flint went from 25.7 tok/s at 8k of context to 19.5 at 32k. That is +12.4 ms per token for 0.70 GB
  more KV (28 KiB per token over 24,576 tokens), i.e. ~57 GB/s.
- **Comparison.** The fork's matrix-vector kernels stream weights at 120–150 GB/s.
- **Projection at 32k.** The KV read is ~0.94 GB per token, about 16 ms at that rate, on top of 26.8 ms of weights.
- **Possible gain.** An attention kernel at 150 GB/s on f16 KV would take ~6 ms, and ~3 ms on a q8_0 KV it reads
  natively, i.e. +30–45 % at 32k (claim to test).
- **Existing work.** [flint_speedups.md](flint_speedups.md) already notes that a q8_0 KV cache moves sm_61 to the VEC
  kernel, which re-reads K/V for each of the 8 query heads per KV head. The GQA-aware kernel (upstream PR #30792) is the
  piece to watch or port.

Measure first: `llama-bench -d 0,8192,32768,65536` on the fork build, plus one `GGML_CUDA_PROFILE` run at 32k for
attention's share.

### 7. Output head at 4 bits: does not transfer as is

HyperQwen's GPTQ int4 head cost +0.6 % perplexity on Qwen3.8-27B.

On Flint the head is sensitive:
- Round-to-nearest imatrix Q4_1 in place of Q8_0 raised KLD 0.280 → 0.338 and perplexity +16.8 % (Q18).
- The makers' tuned 4-bit head was also worse (Q44).

A GPTQ-rounded Q4_1 head (written into the GGUF by script) would save 0.13 GB per token, ~3 % (my calc), if it passes the
quality bar. That makes it a small, optional quant-machine experiment, not a priority.

### 8. Measurement practice worth copying

From HyperQwen's [gotchas](https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/docs/gotchas.md) (P):
- **Report tokens per step, not acceptance per drafted token.** Keep request order fixed.
- **Use real prompts.** Random-token benchmarks gave 35, 83 or 151 tok/s on the same server.
- **Treat high acceptance as a warning.** A repetition loop scored 79.7 % against a normal 35–45 %. Check output, for
  example the distinct-word ratio, before believing a speculation gain.
- **Expect greedy text to change** when the verify width changes (rounding at near-ties). Flint is more sensitive than
  most (Q21), so compare speculation runs by speed and quality metrics, not by identical text.

## What does not carry over

| HyperQwen lever | Why not here |
|---|---|
| int8 tensor-core GEMMs (W4A8 Marlin), int8 QK prefill attention | sm_61 has no tensor cores; llama.cpp's prefill already multiplies int8 activations with dp4a (MMQ) |
| fp16 recurrent state, hybrid KV-group sizing, recurrent-state prefix snapshots | Flint has no recurrent layers; llama.cpp's sliding-window cache is already separate from full attention |
| fp8 KV, KVarN 4/2-bit KV, pth-flash-attn | No fp8 on sm_61; the kernels need sm80+; 262k already fits in nine cards (0.5–1.5 GB KV per card) |
| Embedding in pinned host memory | llama.cpp already keeps the embedding table on the host and reads one row per token |
| Tensor parallelism | PCIe 1.0 x1 without P2P; HyperQwen's own TP=2 without P2P was slower than one card |
| Marlin tile tuning | No Marlin; HyperQwen measured +0.4 % end to end anyway |
| Tree drafts | Flint has no recurrent layers, so trees are possible in principle, but every extra verified token adds experts |
| HyperQwen's exact model | Qwen3.8-27B is dense; on the rig it decodes ~4 tok/s, and a 3-token verify already costs 1.71× ([speed-ups](../../benchmarks/2026-10-22-speedups.md)) |

## Experiments, ranked by value for effort

1. **Flint's routing locality and verify cost** (quant machine 1–2 h, rig ~2 h; decides every speculation item).
   - Locality: dump the router's top-6 for each layer over a few thousand generated tokens of chat and code. Report
     distinct experts for n = 2, 4, 8 and 16 consecutive tokens against random routing (the table above).
   - Verify cost: Q19's [verify_cost.sh](../../../bench/eagle3_flint/verify_cost.sh) for n = 1–8 on the fork build,
     kernels on and off.
   - Together they fix k_s and k_l and say whether a deduplicating multi-token `MUL_MAT_ID` kernel is worth writing.
2. **Graph-shape cache plus the copy-gated n-gram drafter** (fork, 2–4 days; levers 1–2).
   - Measure: tokens per step and tok/s on four prompt sets with `-bs`:
     - the four chat prompts;
     - "return this 2–4k-token file with one function renamed";
     - an edit that keeps context;
     - a replayed agent session with tool output.
   - Baselines: no drafting, `ngram-simple` 3/4, `ngram-mod` defaults, and PR #27210's controller if it ports cleanly.
   - Expected: ≥ 2× on rewrites, ±2 % on chat (claim to test).
3. **Attention at depth** (rig, 1 h, then a decision; lever 6).
   - Run: `llama-bench -d 0,8192,32768,65536` on S4-L0 with the fork build, plus one profile at 32k.
   - If attention takes ≥ 30 % of the token at 32k, a GQA-aware Pascal attention kernel (f16 and q8_0 KV) is the next
     kernel project after Q50.
4. **Tool requests without backend sampling** (rig, 1 h; lever 4).
   - Run: the same prompt with and without `tools`, `-bs` on, real chat script.
   - If the loss is ≥ 8 %, build the GPU top-k candidate path (fork, ~2 days).
5. **Draft-head economics folded into Q43 and Q20** (no new experiment; lever 3).
   - Q43: rows counted over Qwen3.6's own outputs; `--spec-draft-sampling probabilistic` at T = 1.0.
   - Q20: code coverage of the 32k list; a 4-bit calibrated draft head.
6. **Checkpoints as the documented default for Flint serving** (docs only; lever 5).
7. **GPTQ-rounded Q4_1 output head** (quant machine, ~1 day; lever 7). Optional, ~3 % if it holds quality.

## Sources

HyperQwen at commit `ce9615c` (P), all under `https://github.com/syv-ai/HyperQwen/blob/ce9615c4f2309be727407918a6c979bd67968069/`:
- `README.md`: setups A–E, numbers per mode.
- `docs/optimizations.md`: every optimization, the cumulative tables, the lookup drafter, DFlash2.
- `single-user/README.md`: per-knob measurements; per-position acceptance (74/50/34/24 % at T 1.0); `DRAFT_SAMPLE`
  greedy ~15 % slower at T > 0; lookup knob defaults.
- `docs/gotchas.md`: verify-cost stairs (16 → 17 and 21 → 22 query tokens), CUDA graphs for both block lengths
  (27.9 vs 25.9 ms), lookup controller variants, measurement pitfalls, draft-vocabulary failure on Chinese.
- `docs/long-context.md`: int8/fp8/4-bit KV kernels and their decode speeds.
- `docs/multi-gpu.md`: TP=2 without P2P slower than one card; pipeline parallel with MTP.
- `drafter/README.md`: draft vocabulary (92 % → 97.5 % coverage), GPTQ heads, DFlash2 requantization, negative results.
- `patches/dflash2-lookup-drafting.patch`: the suffix-lookup kernel, the fuse rules and the adaptive long block
  (`next_num_draft_tokens`); `patches/speed-knobs-envs.patch`: shipped defaults (NMIN 6, NMAX 12, NSTRONG 6, AGREE 0,
  NMIN_TAIL 4, LONGMIN 6, STICKY 3).

llama.cpp, the fork's tree at upstream `d77256999` (P):
- `common/speculative.cpp`, `common/ngram-map.cpp`, `common/sampling.cpp`, `common/common.h`, `common/arg.cpp`
- `tools/server/server-context.cpp`
- `src/models/dflash.cpp`, `src/models/flint.cpp`, `src/llama-context.cpp`, `src/llama-kv-cache-iswa.cpp`
- `ggml/src/ggml-cuda/ggml-cuda.cu`, `mmvq.cu`, `mmvq-pascal.cu`

External (P unless marked S):
- [arXiv 2609.22156](https://arxiv.org/html/2609.22156) (Amankulov, Mamatin): distinct experts per verified block on
  Qwen3-Coder-30B-A3B with EAGLE-3, A100, batch 1 (Table A1).
- [llama.cpp PR #27210](https://github.com/ggml-org/llama.cpp/pull/27210): adaptive MTP depth (open), with the
  reasoning / prose / code / recall measurements on Qwen3.8-27B.
- [arXiv 2411.04975](https://arxiv.org/html/2411.04975): SuffixDecoding, adaptive draft length α × match length,
  agentic results.
- [arXiv 2602.06036](https://arxiv.org/abs/2602.06036): DFlash (block drafter, target-layer KV injection, MoE results).
- [incoai/Qwen3.6-35B-A3B-DFlash2](https://huggingface.co/incoai/Qwen3.6-35B-A3B-DFlash2) and
  [inco.ai/blog/dflash2](https://inco.ai/blog/dflash2): DFlash2 architecture and speed-ups, MoE against dense.
- [Intel DFlash on an iGPU](https://huggingface.co/blog/ofirzaf/intel-dflash-ptl) (S): MoE against dense gains at batch 1.
- [Qwen3.6 speculative decoding on 2 × V100](https://github.com/thc1006/qwen3.6-speculative-decoding-v100) (S): MTP and
  DFlash by block length, MoE and dense.
- [llama.cpp discussion #25978](https://github.com/ggml-org/llama.cpp/discussions/25978) (S): P40, Qwen3.6-35B-A3B MTP
  by draft depth.
- [llama.cpp PR #22105](https://github.com/ggml-org/llama.cpp/pull/22105) (DFlash in llama.cpp, with an RTX 3090 MoE
  report in the comments), [SpecForge](https://github.com/sgl-project/SpecForge),
  [vLLM speculators](https://docs.vllm.ai/projects/speculators/en/latest/user_guide/tutorials/train/).

This repository:
- [Flint benchmark](../../benchmarks/2026-10-22-flint.md): depth speeds, CUDA-graphs gain.
- [speed-ups](../../benchmarks/2026-10-22-speedups.md): backend sampling +8 %, MTP verify costs.
- [quants/experiments.md](../../../quants/experiments.md): Q18 and Q44 (head), Q21 (rounding chaos), Q38 (kernels).
- [eagle3_flint](../../../bench/eagle3_flint/README.md) and its training plan.
- [flint_speedups.md](flint_speedups.md): n-gram types, KV kernels on sm_61.
