# Unconventional ideas for Flint and Qwen3.6-35B-A3B on the P106 rig (2026-10-24)

Scope: ideas for running Flint (40B-A2.5B MoE) and Qwen3.6-35B-A3B much faster on nine P106-090 cards that a normal
llama.cpp user would not try: skipping work, changing the model, changing the file format, exploiting the nine-card
topology, and changing what the engine does per answer rather than per token. It assumes this project's stack: a
llama.cpp fork under our control, a 7-GPU quant machine for retraining and quality work (never for speed), and full
freedom to rewrite model files. It builds on the [registry](../../../quants/experiments.md) (Q1–Q91) and on
[breakthrough candidates](breakthrough_candidates.md), whose ideas are not proposed again; the mapping is in "Already
covered elsewhere". Each idea was checked against its primary source and priced with the rig's own numbers.
Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs: code and correctness on a laptop GPU, quality,
  generation and training on the quant machine, speed on the rig only.

Starting point, measured ([registry](../../../quants/experiments.md)):
- S4-L0 on six cards, fork build: 37.4 tok/s in llama-bench tg128, 26.8 ms per token (T below), 2.24 GB of weights per
  token. G3, the best file so far (KLD 0.101 against BF16), reads 2.33 GB; 2.21 GB with the GPTQ-rounded Q5_0 head
  (Q56). Real chat on nine cards at 262,144 tokens of context: 31.9 tok/s (Q58), ~31.3 ms per token.
- The quality bar: within noise of Unsloth's UD-Q4_K_XL against BF16, KLD 0.120 and 87.7 % same top token (in practice
  KLD ≲ 0.125). Task checks (Q41) use non-inferiority margins of 1.5 points (BFCL single-turn) and 3 points (multi-turn,
  LiveCodeBench) against UD-Q4_K_XL.
- With the fork's Pascal kernels decode time follows bytes: Q8_0, Q5_0 and Q4_1 stream at 150, 138 and 145 GB/s on
  large shapes. About 9 ms of a token is kernel boundaries, small kernels and non-matrix work (the persistent kernel of
  Q88 attacks it), ~3 ms hops and host.
- On a layer split each card computes 4–8 ms of a 26.8 ms token and idles the rest (Q49). The pipeline simulator puts
  the six-card ceiling with every card busy at one token per 6.28 ms (slowest stage, the one with the output head,
  plus a hop): 4.3× one stream (Q87).
- About two thirds of Flint's output is reasoning (Q64); replies think 600–7,900 tokens (Q58); single agent turns are
  89 % reasoning ([n-gram replay](../../../bench/ngram_replay/README.md)).
- Flint: 49 layers (2 dense, 47 MoE with 256 routed experts of intermediate 512, top-6 sigmoid renormalized, a shared
  expert of intermediate 1,024), hidden 2,048, 32 query and 4 KV heads of 128, 14 full-attention layers without RoPE
  and 35 sliding-window layers (513), one learned sink logit per head (`attn_sinks`, shape n_head), vocabulary 150,144.
  Consecutive tokens share about one of six experts (Q51); route weights by rank 0.31 / 0.21 / 0.16 / 0.13 / 0.11 / 0.10.

## Verdict

- **The per-token quality bar leaves about one millisecond to spend.** G3 sits ~0.02 KLD under the bar, and near the
  bar the fork's file-side frontier trades 0.015–0.025 KLD per millisecond saved (G3 → G4: +0.009 KLD for 0.39 ms;
  G3 → H3: +0.017 for 0.74 ms; Q77's cheapest group, the shared expert in layers 33–40, ~0.015 per ms). Any lossy
  structural change (dropping layers, heads or experts, merging, low-rank or shared-basis experts, sparsity, shared
  codebooks, a distilled student) must beat that rate, and most cannot: they were rejected below on that arithmetic or
  because they save no bytes per token at all. The large levers left are exact, or they act per answer and are judged
  by tasks instead of KLD.
- **The largest untried lever is a first-finish race through the idle cards.** Two or three samples of the same request
  run staggered through the pipeline, each at nearly single-stream speed because the cards idle 70–85 % of every
  token; the first sample that closes its reasoning wins and the others are dropped. Est. 1.10–1.35× per answer,
  depending on how much Flint's reasoning length varies between samples of one prompt. Published results go the right
  way for quality too: shorter chains are more often correct (First Finish Search: +15 points on AIME for DeepSeek-R1;
  short-m@k: shortest chains up to 34.5 % more accurate than the longest). The file and the model stay the same; the
  selection changes which sample the user gets, so Q41's task lanes decide, and Q41's existing multi-attempt logs
  answer both questions (time saved, pass rate of the shortest attempt) before any code is written.
- **Second: shorten the reasoning itself.** Suppressing reflection tokens ("Wait", "Hmm", "Alternatively") inside
  `<think>` cut reasoning by 13–31 % on RL-trained 32B models with accuracy from −2.7 to +6 points (NoWait), though
  GPQA lost 6 points on Qwen3-32B and distilled models lost much more. It is a logit bias that the GPU sampler already
  supports. Self-training on Flint's own shortest correct samples (−30 % output tokens at kept accuracy on math) is the
  trained follow-up, and the one top idea that needs the quant machine for retraining (LoRA SFT, est. 3–15 GPU-hours).
  Est. 1.10–1.25× per answer for suppression, 1.25–1.4× for self-training. Both change the text by design; only tasks
  can judge them.
- **Exact and large, but only at long context: KV-sharded decode attention.** The 14 full-attention layers' KV cache is
  spread over all nine cards; each card computes attention over its slice, and the partial results are merged with
  their log-sum-exp (Flash-Decoding's split-KV merge, Helix's KV parallelism, both exact). Flint's full-attention layers
  have no RoPE, so a shard needs no positions, only the query. Est. nothing at ≤ 32k, +5–20 % at 64k, +40–60 % at 131k
  and ~2× at 262k over the planned Pascal attention kernel (Q60), plus an even spread of KV memory over the cards.
- **Exact and cheap to test: guesses from Flint's own output.** A table of n-gram continuations built from tens of
  millions of tokens Flint generates on the quant machine, behind the copy lookup of the pipelined executor (Q87), plus
  Gumbel-coupled guessing: guesser and sampler share position-keyed random noise, so a guess drawn from a distribution
  close to Flint's matches Flint's sample far more often than a fixed top guess at temperature 1.0, while the output
  stays exactly Flint's (Daliri et al.'s coupling). Est. 1.4–1.55× overall on six cards against 1.34× for today's
  guessers. The coupling helps the planned trained drafter as well.
- **Small exact levers found while checking:** Q8_0 with one scale per 128 weights and Q5_0 per 64 (testable with stock
  tools by tying the scales of neighbouring blocks; −0.10 GB per token, +2.5 %); a Pascal kernel for the F16 router,
  which streams at ~50 GB/s today (~0.9 ms per token, +2 %); the output head split over several cards (+1–3 % for
  Flint, +5–7 % for Qwen3.6's 248k head, and the six-card pipeline ceiling from 4.3× to ~5.6×).
- **Lossy, for the record:** reasoning tokens through cheaper copies of the always-read tensors (est. 1.07–1.11× per
  answer) and typical or judged acceptance of in-flight guesses (+15–20 % over a trained drafter). Both cost more than
  the bar's headroom per token; only tasks could clear them.
- **A rule from the topology:** a hop between cards goes through host memory over PCIe 1.0 x1 (assumed 0.1–0.3 ms;
  Q81 measures it), so splitting any unit of work shorter than ~1 ms across cards loses. Routed experts of one layer
  (~0.09 ms), a whole MoE layer (~0.45 ms) and attention at ≤ 32k (~0.5 ms per full layer) don't qualify: no expert,
  tensor or layer parallelism. The output head (1.4–2.2 ms) barely does; attention at ≥ 64k (1–4 ms per full layer)
  does.

## The frame: what the bar and the rig leave

**Quality headroom (est.).** G3 scores 0.101 against the bar's 0.120 (0.104 on the GPU set of Q77). Near the bar the
fork's frontier exchanges KLD for time at:

| Step | KLD change | GB per token saved | ms saved at 150 GB/s | KLD per ms |
|---|---|---|---|---|
| G3 → G4 (attention q Q8_0 only in layers 0–16) | +0.009 | 0.058 | 0.39 | 0.023 |
| G3 → H3 (F3 recipe, shared expert Q5_0 in 13–45) | +0.017 | 0.111 | 0.74 | 0.023 |
| Q77's cheapest group: shared expert 33–40 to Q5_0 | +0.0019 | 0.019 | 0.13 | 0.015 |
| G3 + GPTQ Q5_0 head (Q56), for contrast | +0.0023 | 0.116 | 0.77 | 0.003 |

The remaining ~0.02 KLD buys ~1 ms (~4 %) of lossy savings. Better rounding (the last row) beats the frontier by a wide
margin, which is why the exact and near-exact levers matter more than structural cuts here.

**Idle cards.** A single stream keeps one card busy at a time. Q87's simulator puts the cost of extra work at zero as
long as the pipeline holds fewer tokens in flight than ~4.3 (six cards) to ~5.9–8.5 (nine cards, depending on how light
the last card is), minus ~10 % if the 75 W limit pulls clocks down at full duty. Pipelined speculation (breakthrough
candidate 1) spends that capacity on guesses; ideas 1, 3 and 4 below spend it on other things or better guesses.

**Splitting work across cards (est.).** Splitting a unit of U ms over N cards saves U·(1 − 1/N) and costs a scatter
and a gather on the critical path, each at least one hop (0.1–0.3 ms assumed) plus transfer at ~190 MB/s per link and
500 MB/s per switch uplink:

| Unit of work, one token | Time on one card | Split pays? |
|---|---|---|
| Routed experts of one MoE layer (G3: 13.8 MB) | ~0.09 ms | No |
| One MoE layer (weights ~41 MB, ~13 kernels) | ~0.45 ms | No |
| Attention of one full-attention layer at 32k (67 MB of f16 KV at 140 GB/s) | ~0.48 ms | Break-even at best |
| Output head, Q8_0 / GPTQ Q5_0 | 2.2 / ~1.4 ms | A little |
| Attention of one full-attention layer at 131k / 262k | ~1.9 / ~3.8 ms | Yes |

## Ranked ideas

Gains are against today's fork build unless the row says otherwise. "Per answer" counts the reasoning, which the user
waits for; "overall" is tokens per second over Q64's transcript mix.

| # | Idea | Mechanism | Evidence | Expected gain (est.) | Quality against the bar | Effort | Quant machine | First experiment | What would kill it |
|---|---|---|---|---|---|---|---|---|---|
| 1 | First-finish reasoning race | k = 2–3 samples of one request staggered through the idle cards, prompt KV shared; the first to end its reasoning wins, the rest are dropped | [FFS](https://arxiv.org/abs/2505.18149): DeepSeek-R1 82.2 % on AIME, +15 points; [short-m@k](https://arxiv.org/abs/2505.17813): up to 34.5 % more accurate than the longest chain, up to 33 % less wall time; Q87: ~4 tokens in flight cost no latency on six cards | 1.10–1.35× per answer; agent turns (89 % reasoning) up to ~1.35× | Same file and model; selects the shortest of k samples; Q41 lanes decide | ~1 week after Q87's multi-sequence loop | None | Shortest-of-k length and pass rate from Q41's per-attempt CSVs (any PC, minutes) | Shortest of 3 ≥ 0.85 × the mean length, or its pass rate below a lane margin |
| 2 | Reflection suppression, then concise self-training | Logit bias −∞ on reflection tokens inside `<think>` only; then LoRA SFT on Flint's shortest correct self-samples | [NoWait](https://arxiv.org/abs/2506.08343): −13 to −31 % reasoning on RL-trained 32B models; [self-training](https://arxiv.org/abs/2502.20122): −30 % tokens at kept accuracy | 1.10–1.25× per answer (suppression); 1.25–1.4× (self-training) | Changes the text; Q41 lanes decide; a fine-tune is quantized and KLD-checked against its own BF16 | 1 day + one lane; self-training 1–2 weeks | Retraining (LoRA SFT) and sample generation | Count reflection words in existing transcripts; E1 half pool with and without the bias | < 10 % fewer reasoning tokens, or a lane fails |
| 3 | KV-sharded decode attention | Full-attention KV spread over all cards; partial attention per slice; log-sum-exp merge on the host; sinks added once | [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (exact split-KV); [Helix](https://arxiv.org/abs/2507.07120) KV parallelism (exact; up to 1.5× lower token latency); [Medha](https://arxiv.org/abs/2409.17264) KVP | 0 at ≤ 32k; +5–20 % at 64k; +40–60 % at 131k; ~2× at 262k (over Q60) | Exact (summation order) | 2–4 weeks | KLD and needle checks | Scatter/gather round trip for N = 2–8 cards (Q81 extended); two-card shard at 131k | Round trip ≥ 1 ms, or no use beyond 32k |
| 4 | Flint-output datastore and Gumbel-coupled guesses | n-gram continuation table from 20–50 M generated tokens behind the copy lookup; position-keyed Gumbel noise shared by guesser and sampler | [SuffixDecoding](https://arxiv.org/abs/2411.04975): open-ended 1.66× against 1.45× for prompt lookup, agentic 2.4–5.3×; [REST](https://arxiv.org/abs/2311.08252) 1.62–2.36×; [coupling bound](https://arxiv.org/abs/2408.07978) (1 − TV)/(1 + TV) | 1.34× → 1.4–1.55× overall on six cards; chat 1.11× → 1.15–1.3× | Exact: sample-and-match, coupling keeps Flint's distribution | 1–2 weeks with the coupled sampler | Generation only | `pipeline_sim.py` with a datastore guesser on held-out transcripts | Right guesses on fresh text < 0.15 and coupling adds < 0.05 |
| 5 | Coarser scale groups; router kernel | Q8_0 with one FP16 scale per 128 weights, Q5_0 per 64; a Pascal F16 product for the router | Q11/Q16 (tied scales in the makers' file); profile: router 47 × ~20 µs at ~50 GB/s | −0.10 GB per token (+2.5 %); −0.55 ms (+2 %) | Q8 per 128 est. ≤ +0.003 KLD; Q5 per 64 to test; router exact | Emulation 1 day; types 3–5 days; router 1 day | KLD checks | G3 with tied scales (stock Q8_0/Q5_0 blocks), KLD against BF16 | ΔKLD > 0.003 (Q8) or > 0.005 (Q5) |
| 6 | Split output head | Head rows over 2–4 cards; each returns its top-20 and log-sum-exp; exact merge for greedy and top-k ≤ 20 | [Vocabulary parallelism](https://arxiv.org/abs/2411.05288): +5–51 % throughput in training pipelines | Flint +1–3 % single stream; Qwen3.6 +5–7 %; six-card pipeline ceiling 4.3× → ~5.6× | Exact | ~1 week | — | Q81 hop latency; split-head patch with the pipespec identity check | Hop ≥ 0.4 ms (then only Qwen3.6 and the pipeline gain) |
| 7 | Precision by token role (lossy) | Reasoning tokens use Q4_1 copies of attention q/o, shared expert and dense layers (+0.75 GB of VRAM), optionally 4 experts; answers and the KV cache stay G3 | Q30: the makers' Q4_1 matches BF16 on BFCL; [quantized reasoning models](https://arxiv.org/abs/2504.04823): W4A16 near lossless, no longer outputs | +11–17 % per reasoning token; 1.07–1.11× per answer | Reasoning near S1 or the makers' level; tasks decide | 3–5 days | Task lanes | Two-server emulation on Q41's E1 | A lane fails, or reasoning ≥ 10 % longer |
| 8 | Typical or judged acceptance (lossy) | Keep an in-flight guess when Flint gives it probability ≥ min(ε, δ·e^−H) instead of requiring the sample to match | [Medusa](https://arxiv.org/abs/2401.10774) typical acceptance; [Judge decoding](https://arxiv.org/abs/2501.19309): 9× on Llama-405B | +15–20 % over a trained drafter (p 0.5 → ~0.6–0.65) | Sampling no longer Flint's distribution; tasks decide | 2–3 days after Q20's drafter | Optional judge head | Offline on dumped probabilities | Any lane fails |
| 9 | Redundancy scan at the frontier's rate | Drop dormant heads (by sink mass), attention blocks or layers only where ΔKLD per ms saved beats 0.015–0.02 | [He et al.](https://arxiv.org/abs/2406.15786): half of Llama-2-70B's attention −2.4 %; [ShortGPT](https://arxiv.org/abs/2403.03853): generation near zero after 25 % of layers on 7B models; [FlexiDepth](https://arxiv.org/abs/2503.23798): ~1.7 of 32 layers skipped on free text | 0–3 % (likely 0) | Lossy; must beat the frontier | Scan 2–4 GPU-hours; patch 1–2 days | Scan | Per-head sink mass and per-block ΔKLD on 20 × 512 tokens | No block or head group under the gate |

## 1. First-finish reasoning race on the idle cards

### Mechanism

A request with sampling on (Flint's recommended temperature 1.0, top-p 0.95, top-k 20) starts k sequences that share
the prompt's KV cells (`llama_memory_seq_cp` in a unified cache, `-kvu`) and differ only in their random seeds. The
pipelined executor of Q87 runs them staggered: card 1 takes sequence A's token, then B's, then C's, each as an ordinary
single-token step. When one sequence emits the end-of-reasoning token, the others are removed (`llama_memory_seq_rm`)
and the winner continues alone into its answer or tool call, where pipelined guessing (breakthrough candidate 1) can
take over the idle capacity again.

### Why it fits this rig

- On a GPU server, k parallel samples are a batch, nearly free because decode is memory-bound. On these cards a
  multi-token step is expensive (Q19: a 4-token verify step 3.6 T on stock kernels; 1.3–1.8 T of GPU time for 2–3
  tokens with Q78's kernels, est.), so llama-server's own parallel samples would run each sequence at 0.55–0.77× speed,
  which cancels the race's gain (est.). Staggered single-token steps avoid that: up to ~4 sequences fit in the six-card
  pipeline's slack.
- Reasoning is two thirds of Flint's output and the part where pipelined n-gram guessing helps least (chat 1.11×,
  Q87), so the idle cards are worth more to the reasoning phase as extra samples than as guesses.

### Evidence

| Work | Setup | Result |
|---|---|---|
| [First Finish Search](https://arxiv.org/abs/2505.18149), 2025 | n parallel samples, stop at the first to finish; DeepSeek-R1, R1-Distill-Qwen-32B, QwQ-32B, Phi-4-Reasoning-Plus on AIME and GPQA Diamond | DeepSeek-R1 82.23 % on AIME, ~15 points above its standalone accuracy; motivated by shorter traces being "much more likely to be correct" |
| [Don't Overthink It (short-m@k)](https://arxiv.org/abs/2505.17813), 2025 | k parallel chains, stop when the first m finish, majority vote among them | Shortest chains up to 34.5 % more accurate than the longest for the same question; short-1@k matches or beats majority voting with up to 40 % fewer thinking tokens; short-3@k up to 33 % less wall time |
| Q87 simulator (this project) | S4-L0 on six cards, per-card stage times from the 26.8 ms token | One token per 6.28 ms with every card busy: ~4.3 tokens in flight before anything waits |

### Expected gain (est.)

The reasoning phase ends at the shortest of k sample lengths. With per-prompt reasoning lengths log-normal around their
mean (Monte Carlo, coefficient of variation CV between samples of one prompt), the shortest of k is:

| CV | Shortest of 2 | Shortest of 3 | Shortest of 4 |
|---|---|---|---|
| 0.3 | 0.84 | 0.77 | 0.72 |
| 0.45 | 0.76 | 0.67 | 0.61 |
| 0.6 | 0.70 | 0.58 | 0.52 |

Time per answer, with reasoning two thirds of the tokens and each stream 3 % (k = 2) or 5 % (k = 3) slower from host
work and the power limit: k = 2 gives 1.10–1.23×, k = 3 gives 1.15–1.35×. Agent turns (89 % reasoning) at CV 0.45 and
k = 3: ~1.36×. Flint's CV is unknown; Q58's 600–7,900 tokens per reply spans prompts, not samples of one prompt.

Costs: k times the card work in the reasoning phase (power, not time), per extra sequence ~28 KiB of full-attention KV
per reasoning token (~0.22 GB for 8k tokens) plus its sliding-window cells, and a sampler per sequence on the host.

### Quality, and how to measure it

The file is unchanged, so KLD is unchanged. What changes is which of Flint's samples the user gets: the one with the
shortest reasoning. The papers above find that this raises accuracy on math and science questions; for tool calls and
code nobody has measured it. Q41's harness already logs each attempt's output tokens and reasoning length
([bfcl_flint/score.py](../../../bench/bfcl_flint/score.py), [livecodebench_flint](../../../bench/livecodebench_flint/)),
and its pilot ran 376 BFCL tasks × 5 sampled attempts, so both effects can be computed from existing data.

### Effort and first experiment

- **Offline, no GPU (minutes, any PC).** From Q41's per-attempt CSVs: for every task and every k-subset of its attempts
  (k = 2, 3), pick the attempt with the shortest reasoning (reasoning characters in BFCL's scores, output tokens in
  LiveCodeBench's); report the mean shortest length over the mean length, and
  the pass rate of the picked attempt against mean pass@1, with a paired bootstrap interval. Repeat on E1, E2 and E3
  as their multi-attempt runs land. Gate: shortest of 3 ≤ 0.8 of the mean and the pass rate inside the lane margin.
- **Then the executor.** Q87's loop for several sequences (the slots exist, the loop does not), prompt sharing by
  `seq_cp`, per-sequence samplers, and a server option for k; about a week after the executor. Rig check: per-stream
  token time with k = 1, 2, 3 streams, clocks and power.

**What would kill it:** reasoning lengths that barely vary between samples (shortest of 3 ≥ 0.85 of the mean), or a
pass rate that falls by more than a lane's margin.

**Qwen3.6:** the same, in thinking mode; each sequence keeps its own Gated DeltaNet state, which needs no rollback here
because nothing is speculated.

## 2. Shorter reasoning: reflection suppression, then concise self-training

### Mechanism

- **Suppression.** During reasoning, the logits of the tokens that open self-reflection ("Wait", "Hmm",
  "Alternatively", "But wait", their case and spacing variants) are set to −∞. NoWait collected 15–17 such keywords from
  QwQ-32B's AIME runs, matched them as substrings in each vocabulary, and masked the tokens with a logit processor.
- **Self-training.** Flint generates N samples per prompt on tasks with checkable answers (tool calls, code with tests,
  math); the shortest correct sample per prompt becomes fine-tuning data; a LoRA on attention and the shared expert
  (routed experts frozen) is trained in BF16, merged and quantized with G3's recipe.

### Evidence

- [NoWait](https://arxiv.org/abs/2506.08343) (2025), reasoning tokens and accuracy (points):

| Model | Benchmark | Length | Accuracy |
|---|---|---|---|
| QwQ-32B | AMC 2023 / AIME 2024 / AIME 2025 | −30 / −16 / −31 % | +4.25 / −2.00 / +1.33 |
| Phi-4-Reasoning-Plus | same | −28 / −26 / −23 % | +6.00 / −0.67 / +3.34 |
| Qwen3-32B | same | −13 / −16 / −14 % | −0.83 / +2.00 / −2.67 |
| Qwen3-32B | GPQA Diamond | −15 % | −6.06 |
| QwQ-32B-Preview | AIME 2024 | −36 % | ±0 (O1-Pruner −8.7 at −52 %) |

  The paper calls Qwen3-32B RL-trained and Qwen3-4B/8B/14B distilled: the distilled ones lost 13–17 points on AIME 2025,
  so models trained by distillation tolerate it far worse. Flint was trained with reinforcement learning after SFT
  (model card), which places it with the robust group (est.).
- [Self-training elicits concise reasoning](https://arxiv.org/abs/2502.20122) (ACL Findings 2025): best-of-N and
  few-shot-conditioned self-generated concise paths, fine-tuned, give 30 % fewer output tokens on average with average
  accuracy kept, across five model families on GSM8K and MATH.

### Expected gain (est.)

Reasoning −13 to −30 % with suppression gives 1.10–1.25× per answer (reasoning two thirds of the tokens);
self-training's −30 % of output tokens, if it carries over to tool calls and code, 1.25–1.4×. Ideas 1 and 2 shorten the
same phase, so together they give less than the product: est. 1.3–1.5×.

### Quality and the bar

Both change what Flint writes, by design. KLD against BF16 cannot judge them; Q41's lanes (BFCL single- and multi-turn,
LiveCodeBench) with their non-inferiority margins can, together with the malformed-output gate. A self-trained Flint is
a new model: its quantized file is checked against its own BF16 for quantization quality, and against the original
Flint's files on tasks.

### Design details that matter

- **Inside `<think>` only.** Substring matching catches code tokens (`await`, `waitFor`), and the answer may need the
  words. The bias set must switch off at the end of reasoning: either two sampler chains switched at that token, or a
  phase-aware logit-bias sampler.
- **Backend sampling stays on.** llama.cpp's GPU sampler implements `logit_bias`, so `-bs` survives a plain bias. The
  reasoning-budget sampler, which tracks the phase today, switches `-bs` off (Q55); the phase switch must not reuse it.
- **Word-initial tokens only** (" Wait", "Wait", " Hmm", " Alternatively"), so common words inside identifiers are not
  hit; the list comes from Flint's own transcripts, not from QwQ's.

### First experiment

1. Count the reflection words per 1,000 reasoning tokens in Q51/Q64's transcripts and Q41's logged reasoning (minutes).
   Below ~5 per 1,000 there is little to remove.
2. Q41's E1 half pool, two sampled attempts, with and without the bias; tokens per answer and pass@1, paired. Gate:
   reasoning tokens −15 % or more and every lane non-inferior.
3. Only then self-training: generate 8 samples per prompt on the quant machine with llama-server (`-np 32`), select,
   LoRA SFT with transformers' native Flint across the 3090s (10–30 M tokens at 1–2k tokens/s: 3–15 GPU-hours, est.),
   merge, quantize, score.

**What would kill it:** fewer than 10 % fewer reasoning tokens, or any lane failing its margin.

**Qwen3.6:** NoWait tested Qwen3 directly; its thinking mode takes the same bias. Answer-stability probing (Certaindex,
[arXiv 2412.20993](https://arxiv.org/abs/2412.20993): up to 50 % compute saved in a serving system) is a third way to cut
reasoning, but each probe costs decode steps on these cards and tool-call answers have no single value to stabilize, so
it is not pursued here.

## 3. KV-sharded decode attention across the cards

### Mechanism

- **Placement.** The KV cache of each of the 14 full-attention layers is split along the sequence into N blocks on N
  cards; the 35 sliding-window layers keep their 513 cells local, as today.
- **Decode step at a full-attention layer on card j.** Card j computes the token's q (32 heads × 128) and k, v; q goes
  through the host to the other N − 1 cards (8 KB in FP16); every card runs attention over its slice and returns the
  partial output per head with its running maximum and sum (8 KB); the host merges the partials, rescaling each by its
  log-sum-exp and adding the head's learned sink once to the denominator, and sends the merged 4,096 values back to
  card j for the output projection.
- **Why Flint makes it simple.** The full-attention layers use no RoPE, and one sequence attends to every cached token,
  so a shard needs no positions and no mask, only q and its slice length. New K/V go to the newest block, which stays on
  card j and moves to a shard when full (Helix rotates the write target every 16 tokens).
- **Exactness.** Flash-Decoding's split-KV merge, which llama.cpp's TILE kernel already applies across thread blocks
  within one card, applied across cards: the result differs only in summation order.

### Evidence

- [Flash-Decoding](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (2023): splits keys and values into chunks
  computed in parallel and reduces "over all the splits, using the log-sum-exp to scale the contribution of each split".
- [Helix Parallelism](https://arxiv.org/abs/2507.07120) (NVIDIA, 2025): shards the KV cache along the sequence for
  attention, each GPU emits "a partial attention output and a log-sum-exp scalar per token", one all-to-all
  reconstructs the exact result; communication is independent of context length; up to 1.5× lower token latency at
  fixed batch for multi-million-token decoding.
- [Medha](https://arxiv.org/abs/2409.17264) (2024): KV-cache parallelism to keep decode interactive at very long
  context.

### Expected gain (est.)

KV per full-attention layer and token: 134 MB at 64k, 268 MB at 131k, 537 MB at 262k (f16; 28 KiB per context token
over 14 layers). Q60's planned kernel streams it at ~140 GB/s on a 26.2 ms base token. Sharding costs per layer: the
slice at 140 GB/s plus 0.25–0.6 ms of scatter, gather, host merge and return (two hops at 0.1–0.3 ms, transfers through
the switch uplinks, ~30 µs of merge on the Celeron):

| Context | Q60 attention per token | Q60 tok/s | Sharded (N) | Attention per token | tok/s |
|---|---|---|---|---|---|
| 32,768 | 6.7 ms | 30.4 | off | 6.7 ms | 30.4 |
| 65,536 | 13.4 ms | 25.2 | 4 | 6.9–11.8 ms | 26–30 (+4–20 %) |
| 131,072 | 26.8 ms | 18.9 | 9 | 6.4–11.3 ms | 27–31 (+41–62 %) |
| 262,144 | 53.7 ms | 12.5 | 9 | 9.5–14.4 ms | 22–28 (~1.8–2.2×) |

Contexts above ~90k need nine cards in f16, where the base token is ~31 ms instead of 26.2: both columns drop a little
and the ratio stays near 1.8–2.1× at 262k. Against today's stock TILE kernel (57–87 GB/s) the gains are larger.
Memory: the full-attention KV at 262k (7.5 GB) spreads evenly at ~0.83 GB per card, instead of ~1 GB on each card
holding two full-attention layers, which eases Q8_0-class files at full context. Long prefills shard their attention the same way. Sharding composes with q8_0 KV
(Q61) and with Quest-style page selection (breakthrough candidate 7), whose fewer bytes shrink the slice time until the
hops dominate.

### Quality and effort

Exact up to summation order. Tests: Q60's attention tests with sharding forced at small N, KLD at 4k context within the
~0.06 summation-order noise (Q38), a needle at 128k. Effort 2–4 weeks: the shard computation reuses Q59/Q60's split
kernel (partials out instead of a combine), but llama.cpp has no sequence-split KV cache, so the fork needs a
device-spanning attention op with 14 extra split points per token, shard-aware KV allocation and writes, and prefill
placement. In pipelined speculation the shard cards are busy with guesses; above ~64k sharding should take the idle
capacity first (est.).

### First experiment

Extend Q81's hop benchmark (laptop for the code, rig ~1 h): card j sends 8 KB to N cards, N cards return 8 KB each to
the host, the host merges, 8 KB goes back to card j; N = 2, 4, 8, cards behind both switches. Then a two-card version of
Q59's benchmark at 131k with half the KV on each card against one card. Gate: round trip ≤ 0.6 ms at N = 8 and a sharded
layer ≤ 0.6× the unsharded one at 131k.

**What would kill it:** a round trip of 1 ms or more, or agent sessions that stay under 32k.

**Qwen3.6:** 10 full-attention layers, and its Gated DeltaNet layers keep a fixed-size state, so the same mechanism
covers all of its context-dependent attention.

## 4. A Flint-output datastore and Gumbel-coupled guesses for the pipeline

### Mechanism

- **Datastore.** Flint generates 20–50 M tokens at its own sampling settings on the quant machine (chat, code, edits,
  agent turns with tools; the same generation Q20's drafter needs). A hashed table maps each 1- to 8-token context to
  its most frequent next tokens with counts (~0.3–0.5 GB, est.), held in the rig's host RAM. In Q87's executor it sits
  behind the copy lookup: when the context has no copy match, the longest matching table context proposes the guess.
- **Gumbel coupling.** The sampler at the last card draws each token as the argmax of log p + G over the tokens top-k and
  top-p keep, where G is Gumbel noise computed from (seed, position, token id) by a hash; that is an exact sample from
  Flint's distribution. The guesser takes the argmax of log q + G over its own candidates, with q its count
  distribution and the same G. Where q is close to p, the two pick the same token with probability at least
  (1 − TV)/(1 + TV) (Daliri et al.), far above p(top guess) when Flint spreads its probability: with p = 0.5 / 0.3 / 0.2
  and q = 0.45 / 0.35 / 0.2 a fixed top guess is right 50 % of the time, a coupled one at least 90 %. The output depends
  only on the seed, whatever the guesser does, and plain decoding with the same sampler gives the same tokens.
- **Hedged guesses (variant).** While the pipeline has free slots, the second-best candidate can run as a separate
  sequence at the same position; the match keeps one and drops the other. It costs card time only, and attention stays
  exact up to the order of cells.

### Evidence

- [SuffixDecoding](https://arxiv.org/abs/2411.04975) (NeurIPS 2025): a global suffix tree of earlier outputs. On
  open-ended Spec-Bench 1.66× and 1.75 tokens per step against 1.45× and 1.61 for prompt lookup; on agentic work 5.18×
  (AgenticSQL) and 2.43× (SWE-Bench); acceptance holds and speed grows as the tree grows; 27 M tokens took 1.7 GB in
  their structure, updates ~4 µs per token, lookups 10–12 µs.
- [REST](https://arxiv.org/abs/2311.08252) (NAACL 2024): retrieval from a datastore, 1.62–2.36× on code and text with 7B
  and 13B models.
- [Daliri, Musco, Suresh](https://arxiv.org/abs/2408.07978): communication-free coupling, Pr[match] ≥ (1 − TV)/(1 + TV),
  optimal in the worst case, Gumbel sampling at least as good as weighted MinHash for every pair of distributions.
- Q87's simulator: a guesser right with probability p on every token gives 1.30× (p 0.3) and 1.44× (p 0.4) alone, 1.85×
  behind the copy lookup at p 0.5; today's best guessers (copy, else per-request n-gram counts) give 1.34×, chat 1.11×.

### Expected gain (est.)

If the table and coupling bring fresh-text guesses to 0.2–0.35 right: 1.4–1.55× overall on six cards (against 1.34×),
chat 1.15–1.3×, more on tool calls, which repeat paths, commands and argument patterns across sessions. SuffixDecoding's
open-ended gain over prompt lookup (1.66× against 1.45×) is the cautious reference; its agentic gains the hopeful one.
The coupling also raises the acceptance of the trained pipeline drafter (Q20) at temperature 1.0, since that drafter
outputs a distribution too.

### Effort and first experiment

The executor already needs a random draw tied to the position for seeded sampling ([pipelined
speculation](pipelined_speculation.md), production list), so the coupled sampler is that draw done as hashed Gumbel
noise in the GPU sampler (2–3 days). The table and its guesser in `llama-pipespec` take 2–3 days; generation half a day
on the quant machine.

First experiment, offline: build the table from the generated tokens, keep Q51/Q64's 65,520 transcript tokens out of
it, and replay them in [pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py) with a table guesser behind the copy
lookup; for the coupling, dump Flint's top-20 probabilities along the same transcripts on the quant machine and compute
the coupled match probability per position. Gate: ≥ 1.45× overall, or ≥ 0.2 right guesses on fresh text.

**What would kill it:** fewer than 0.15 right guesses on fresh text and less than 0.05 added by the coupling.

**Qwen3.6:** MTP supplies its guesses; the coupled sampler raises MTP's acceptance at temperature 1.0 as well.

## 5. Small exact levers: coarser scale groups and a Pascal router kernel

### Coarser scale groups

- **Mechanism.** Q8_0 stores an FP16 scale for every 32 weights (8.5 bits per weight). One scale per 128 weights
  (130-byte blocks, 8.125 bits) loses little, because Q8_0's own error sits at Flint's KLD floor (Unsloth's Q8_0 scores
  0.052 against BF16, while the same file on two GPU types differs by 0.065, Q21). Q5_0 with one scale per 64 weights
  takes 5.25 instead of 5.5 bits. Both are fork-only types, like Q4_1_64 (Q16), and the Pascal kernels' staged loads read any block size
  once rows are contiguous (a 2,048-wide row of 130-byte blocks is 2,080 bytes, 16-byte aligned).
- **Testing without new code.** Rewrite G3's Q8_0 tensors so each run of four blocks carries the same scale (the
  largest of the four), and its Q5_0 tensors so each pair shares one: the file stays stock Q8_0/Q5_0, runs on stock
  llama.cpp, and scores exactly what the coarse types would. Q11 found the makers' Q4_1 built this way from their
  group-64 export.
- **Expected (est.).** G3 holds 1,461 MB of Q8_0 and 766 MB of Q5_0 per token: −64 MB and −35 MB, −0.10 GB per token,
  0.66 ms at 150 GB/s, +2.5 %. The Q8 step grows ~20 % (largest of 128 Gaussian values ~2.9σ against ~2.4σ of 32), so
  est. ≤ +0.003 KLD; Q5_0's error matters more and must be measured. If Q6_0 (Q90) replaces part of the Q8_0, the same
  trick applies to it (6.25 instead of 6.5 bits per weight at 64).

### The router

- **Observation.** In the S4-L0 profile the F16 router runs llama.cpp's stock kernel: 47 launches at ~25 µs, ~20 µs
  without the profiler's overhead, for 1.05 MB each, ~50 GB/s, ~0.9 ms per token (G3's recipe still has an F32 router,
  99 MB; Q35 showed F16 costs nothing in quality).
- **Lever.** A Pascal F16 product with the staged loads of the quantized kernels (~140 GB/s, ~8 µs per layer): −0.55 ms
  per token, +2 % (est.), exact up to summation order. The persistent kernel of Q88 removes it as a separate launch
  anyway; until then it is a one-day kernel.

**First experiment:** G3 with tied Q8_0 scales, G3 with tied Q5_0 pairs, both against BF16 on G3's GPU set (quant
machine, 2 h). Gate: ΔKLD ≤ 0.003 for Q8 and ≤ 0.005 for Q5. Then the two types (CPU reference, quantizer, Pascal
one-token kernel from Q8_0's and Q5_0's, MMQ dequantize), 3–5 days.

**What would kill it:** ΔKLD above the gates; the router kernel has no quality risk.

**Qwen3.6:** its best files are Q4_0 with Q8_0 attention and head; the Q8 change applies to those tensors.

## 6. A split output head ("parallel tail")

### Mechanism

The head's rows (150,144 for Flint, 248,320 for Qwen3.6) are split over 2–4 cards. After the last layer the final
hidden state (8 KB) goes to the head cards; each computes its rows' logits, keeps its top-20 with their values and the
log-sum-exp of all its rows, and returns ~200 bytes. The sampler merges: greedy and top-k ≤ 20 sampling (the makers'
top-k 20, top-p 0.95, min-p 0) need only the global top 20, which is exact; perplexity and `n_probs` use the merged
log-sum-exp. Tool requests that sample on the host (Q55) copy a fraction of the logits per card instead of 600 KB over
one link.

### Evidence

[Vocabulary parallelism](https://arxiv.org/abs/2411.05288) (2024) splits the vocabulary layers evenly over pipeline
devices, because they unbalance the pipeline's stages: 5–51 % more throughput in training pipelines. Q87's simulator
shows the same imbalance at decode: the last card's stage (5.98 ms with the head and sampling) against 3.48 ms for the
others sets the six-card ceiling at 4.3×.

### Expected gain (est.)

- **Single stream.** Flint's Q8_0 head (2.2 ms) over 3 cards: 0.73 ms plus two hops (0.2–0.6 ms) saves 0.9–1.3 ms
  (+3–4.5 %); with the GPTQ Q5_0 head (~1.4 ms) 0.3–0.7 ms (+1–2.5 %). Qwen3.6's 540 MB Q8_0 head (3.6 ms) over 4 cards
  saves 2.1–2.4 ms of 34.4 (+6–7 %).
- **Pipelined or staggered.** Moving two thirds of the head off the last card brings the six-card stages to ~3.9–4.5 ms:
  ceiling 4.3× → ~5.6×, which raises file rewrites, races (idea 1) and multi-conversation throughput. On nine cards a
  lighter last card does the same with no code (Q87's 6/…/5/2 split).

**Effort:** ~1 week in the fork (row-split head on chosen devices, per-shard top-k and log-sum-exp, merge in the backend
sampler); exactness checked with `bench/pipespec`'s identity check. **First experiment:** Q81's hop latency. **What would
kill it:** hops of 0.4 ms or more, which leave only Qwen3.6 and the pipeline balance.

## 7. Precision by token role (lossy)

### Mechanism

Reasoning tokens, two thirds of the output, run through cheaper copies of the always-read tensors; answer and tool-call
tokens run G3. The cheap set is attention q and output, the shared expert and the dense layers in Q4_1 (1.19 B
parameters, +0.75 GB of VRAM, ~83 MB per card on nine cards); optionally reasoning tokens use 4 of their 6 experts. The
attention k/v projections stay G3's for every token, so the cache keeps one precision. The fork switches tensor sets per
step, keyed on the reasoning phase.

### Expected gain (est.)

G3 reads 1,141 MB per token in those tensors, 746 MB in Q4_1: −0.40 GB, −2.6 ms, +11 % per reasoning token; with 4
experts another −0.22 GB, +17 % in all. Per answer 1.07–1.11×.

### Quality

Reasoning tokens then come from a file near S1 (KLD 0.219, 83.6 % same top token) or, with 4 experts, near the makers'
Q4_1, the tier the bar excludes. Two results suggest tasks may hold: the makers' Q4_1 matched BF16 on 300 BFCL tasks
(Q30), and weight-only 4-bit quantization of reasoning models is near lossless without longer outputs
([Liu et al.](https://arxiv.org/abs/2504.04823), COLM 2025). The gain is a third of idea 1's for a real quality risk, so
it ranks low.

**First experiment:** emulate on the quant machine with two llama-servers, a cheap file writing up to the end of
reasoning and G3 continuing from the full prefix, on Q41's E1 half pool. **What would kill it:** a lane failing its
margin or reasoning 10 % longer.

## 8. Typical or judged acceptance in the pipeline (lossy)

### Mechanism

With sample-and-match, an in-flight guess survives only when Flint's own sample equals it. Medusa's typical acceptance
keeps a guess when Flint gives it probability above min(ε, δ·exp(−H)), H being the entropy of Flint's distribution at
that position, and samples normally otherwise; Judge decoding trains a small head on the target's embeddings to accept
guesses that are wrong token for token but fine for the continuation.

### Evidence and gain

Medusa reports better speed at similar quality, with ε trading one for the other, and says plainly that it does not
keep the model's distribution. Judge decoding reports 9× over Llama-3.1-405B "while maintaining its quality on a large
range of benchmarks". On the rig (est.): a pipeline drafter at exact acceptance 0.5 gives 1.85× behind the copy lookup
(Q87); typical acceptance lifting it to 0.6–0.65 gives ~2.1–2.2×, +15–20 %.

### Quality and order

The sampled text is no longer Flint's distribution; at temperature 1.0 it leans towards the drafter's choices. Only
task lanes can judge it, and it needs Q20's drafter first. **First experiment:** on dumped probabilities along held-out
transcripts, the acceptance gain per (ε, δ); then one Q41 lane. **What would kill it:** any lane failing its margin.

## 9. Redundancy scan at the frontier's exchange rate

### Mechanism

Remove parts of Flint that contribute little, but only where the KLD cost per millisecond saved beats the file-side
frontier (0.015–0.02 KLD per ms):
- **Dormant heads.** Flint learns a sink logit per head; a head that puts nearly all its attention mass on the sink for
  almost every token outputs ~0 and can go with its 128 rows of q and 128 columns of o (~0.5 MB per head in G3).
- **Attention blocks** (q, k, v, o and the attention itself, 14–20 MB in G3, ~0.12–0.18 ms) and **whole MoE layers**
  (~41 MB, ~0.45 ms).

### Gates (est.)

One attention block must cost ≤ ~0.002–0.0035 KLD, one MoE layer ≤ ~0.007–0.009, one head ≤ ~0.00006 (so only large
groups of heads are measurable). A pruned head changes the query-to-KV-head grouping, so the fork would need per-layer
head counts and a head map in the attention kernel.

### Evidence

- [He et al.](https://arxiv.org/abs/2406.15786) (2024): half of Llama-2-70B's attention layers dropped for a 48.4 %
  speed-up and a 2.4 % performance drop.
- [ShortGPT](https://arxiv.org/abs/2403.03853) (2024): 24–27 % of layers removed keep 85–92 % of the benchmark average,
  but XSum generation falls to near zero on the 7B models (19.40 → 0.67 for Llama-2-7B).
- [FlexiDepth](https://arxiv.org/abs/2503.23798) (COLM 2025): a trained router skips 4–8 of Llama-3-8B's 32 layers at
  full benchmark scores, but only ~1.7 on free text continuation, and gives no wall-clock speed-up.
- [Active-dormant heads](https://arxiv.org/abs/2410.13835) (2024): many heads act as sinks on some input domains and not
  on others, which makes "always dormant" rare.

Flint is a 2.5B-active, RL-trained reasoning model of 49 layers, and generation is where removal hurts most, so the
prior is low: est. 0–3 %, most likely 0.

**First experiment (quant machine, 2–4 GPU-hours):** per-head sink mass over Q51's transcripts (eager attention in
transformers' Flint, or Q/K dumps), then ΔKLD on 20 × 512 tokens for each attention block and each layer removed alone.
**What would kill it:** no block, layer or head group under its gate.

## Rejected after checking

| Idea | Why not (est. where marked) |
|---|---|
| "Lazy" layers: reuse a layer's output across adjacent tokens | Adjacent tokens share about one of six experts (Q51), so each layer's FFN input changes from token to token; every token must also write every layer's K/V. No mechanism to reuse whole-layer outputs without a large error |
| Dynamic expert count by route confidence | Route weights are flat (0.31 … 0.10 by rank); the sixth expert is under 5 % on only 6.4 % of token-layers (Q51), so a cumulative-weight rule either keeps 6 or behaves like a fixed 5 (Q6: +4 % for −2 points same top token, worse than the frontier's rate). Learned skipping is breakthrough candidate 10 |
| Layer skipping for reasoning tokens only (FlexiDepth-style), or skipping attention on sliding layers per token | Needs a trained router and per-token predication; FlexiDepth skips ~1.7 of 32 layers on free continuation and keeps computing K/V for skipped tokens; a skipped Flint attention block saves ~0.12–0.18 ms. Est. < 3 % on reasoning tokens for weeks of training and a reasoning-quality risk |
| Self-speculation through skipped layers inside the pipeline (LayerSkip, SWIFT, Draft&Verify) | A guess is needed every cycle (~5 ms on six cards); a draft through a third of Flint's layers takes ≥ 9 ms. The cheap form, an adapter and exit head on early features (Kangaroo), is the partial-depth pipeline drafter of breakthrough candidate 1 (SPD); the batched-verify forms are in survey C 3.6 |
| A low-bit copy of Flint as its own drafter ([QSpec](https://arxiv.org/abs/2410.11305): up to 1.64× on GPUs) | Full depth per draft token, ≥ ~18 ms at 2–3 bits on these cards (est.), longer than a ~5 ms cycle; it also needs ~12 GB of VRAM (est.) that the 262k configuration doesn't have. A 1–2-layer drafter dominates |
| Small model drafts reasoning steps, Flint checks them ([SpecReason](https://arxiv.org/abs/2504.07891): 1.4–3.0× on GPUs) | On the rig a checking prefill costs ~5 ms per token against 27 ms to decode one (5×, not the ≥ 50× of a GPU server), and a small model on a spare card writes ~50–60 tokens/s: ≤ 1.1× (est.), plus a second tokenizer |
| Expert merging or clustering with the router remapped | Still six experts read per token: no bytes saved (breakthrough candidates, rejected list; survey C 3.12) |
| Shared-basis or low-rank experts, or each expert stored as a delta to the shared expert | [MoBE](https://arxiv.org/abs/2508.05257): 24–30 % fewer parameters at 1–2 % accuracy loss on 235B–1T models, far beyond 0.02 KLD; routed experts are only ~28 % of G3's bytes, and a shared base adds its own read. Routed and shared experts differ in width (512 and 1,024) and the shared expert exists so routed experts specialize |
| Bits per expert by usage frequency (rare experts fewer bits) | Bytes per token and KLD both scale with each expert's usage, so the KLD per byte of a change is the same for hot and cold experts: no gain in speed at equal quality. It shrinks the file, not the bytes per token |
| Codebooks shared across experts (vector quantization) | Shared-memory lookups cost more instructions per byte than the card can issue at full bandwidth (breakthrough candidates, rejected list) |
| 2:4 sparse Q8 (two of four weights kept, selected with byte-permute instructions) | Decodes cheaply (est. ~2.5 instructions per byte) at 5 bits per weight, but quality is far off the bar: 2:4 pruning of a LLaMA-2 model raises wikitext perplexity from 5.12 to 6.72 even with learned masks ([MaskLLM](https://arxiv.org/abs/2409.17481)), ≥ 10 with SparseGPT or Wanda |
| Entropy-coded weights decoded on the fly | The codes have little entropy to remove: simulated Gaussian, Laplace and Student-t blocks carry 7.3–7.7 of 8 bits (Q8_0), 4.4–4.75 of 5 (Q5_0), 3.4–3.8 of 4 (Q4_0) (est.), so 4–14 % fewer code bytes; a variable-length decoder costs ~6–10 instructions per symbol against ~4 instructions per byte of issue slack left by the Pascal Q8_0 kernel (1.2 used of ~5.3 usable at 173 GB/s). [DFloat11](https://arxiv.org/abs/2504.11651) removes ~30 % losslessly because BF16 weights have low entropy; quantized codes have little left |
| Weight sharing across layers ([Relaxed Recursive Transformers](https://arxiv.org/abs/2410.20672)) | A shared block is re-read from DRAM at every use (1.5 MB of L2 against ~20–40 MB per layer), so it saves memory, not bytes per token; it also needs uptraining |
| Dropping or merging layers with brief healing | ShortGPT's generation collapse at 25 % on 7B models; Qwen-family models collapse near 20 % even with healing (Gromov et al., survey C 3.10); one Flint layer saves ~0.45 ms and would have to cost ≤ 0.009 KLD. Idea 9 keeps the cheap scan |
| Distilling Flint into a smaller student | Width/depth pruning with distillation needed ~94 B tokens (Minitron, survey C 3.11); the quant machine trains at ~1–2k tokens/s. Nowhere near the bar's KLD |
| A superword tokenizer or vocabulary extension | [SuperBPE](https://arxiv.org/abs/2503.13423)'s 33 % fewer tokens comes from pretraining from scratch; a retrofit needs heavy healing, and a larger head adds bytes |
| Replicating hot layers or experts; expert, tensor or layer-pair parallelism across cards | A single stream is serial through the layers, so copies don't shorten it; intra-layer splits fail the splitting rule (routed experts ~0.09 ms, a layer ~0.45 ms, against ≥ 0.2 ms per scatter/gather). Running adjacent layers in parallel ([true depth](https://arxiv.org/abs/2502.02790): 1.19× and −1.5 % accuracy on Llama 2 7B on GPUs) gains nothing on one card, which is bandwidth-bound, and costs two hops per pair across cards |
| Two independent model copies for two users | G3 (~33 GB, est.) does not fit twice in 54 GB; a ≤ 24 GB file is below the bar; staggered sequences through one copy give 4.3–8.2× aggregate (Q87) against 2× |
| The 4 GB host as a weight cache | Everything already sits in VRAM; host RAM is single-channel DDR4 (~17 GB/s) with ~2.6 GB free. Its use is idea 4's table |
| Precomputed routing for repeated prompts | Prompts run as batches through MMQ, where routing is not the cost; repeated prompts are served from the prompt cache |
| Prefill on another PC with the KV cache shipped back (llama-server slot save/restore) | Outside the rig: a 30k-token Flint prompt's state is ~0.9 GB, ~9 s over gigabit, against 150–375 s of prefill on the rig (est.), but it needs a second machine with the same file; time to first token only (claim to test) |

## Already covered elsewhere

| Direction asked about | Where it is |
|---|---|
| Pipelined speculation, staggered conversations, a pipeline drafter fed by partial depths (SPD, Kangaroo-like) | Breakthrough candidate 1, Q87, [pipelined speculation](pipelined_speculation.md), Q20 |
| Compiling the model into one static program; host-free token loop | Breakthrough candidate 2, Q88, [bench/persistent_pascal](../../../bench/persistent_pascal/) |
| Attention mask on the card, fewer scheduler copies | Breakthrough candidate 3, Q89 |
| A 6-bit rung | Breakthrough candidate 4, Q90 |
| Shrinking the head: exact screen, lossy trim | Breakthrough candidate 5, Q91; survey A (trim misses 1.9–5.5 % of held-out tokens), survey C 3.13 |
| Better rounding of the always-read tensors (GPTQ, YAQA), low-rank-plus-residual variants | Breakthrough candidate 6, Q56 |
| Query-aware KV page selection | Breakthrough candidate 7 |
| Activation sparsity inside experts | Breakthrough candidate 8, survey C 3.14 |
| Re-ordering experts or rows for locality | Breakthrough candidate 9 |
| Learned expert skipping, fewer experts per token | Breakthrough candidate 10, Q6, Q51 |
| Layer-skip self-speculation with batched verify, lookahead | Survey C 3.6 |
| Head pruning, low-rank q/o, attention-block drop | Survey C 3.11, survey A (attention-block drop; scan kept as idea 9) |
| Full → sliding-window conversion, cross-layer KV sharing | Survey C 3.15 |
| Quantization-aware recovery, partial or full | Q40, survey C 3.7–3.8 |
| Expert pruning | Survey C 3.12, Q57 |
| Reasoning budget | Q85 (idea 2 adds suppression and self-training as arms) |
| Two slots, batching conversations | E2.10, Q78 (two independent tokens touch 11.9 experts per layer, so only the dense ~72 % of bytes is shared) |
| KV and prompt caching across turns | [Flint speed-ups](flint_speedups.md) section 6 (checkpoints of 45–55 MB) |
| Expert tiering, host RAM as cache | Q57, [Maya note](maya_expert_tiering.md) |
| Tensor parallelism over x1 links | E2.9, [Flint speed-ups](flint_speedups.md) section 4 |
| Group-64 Q4_1, F16 router, fused Q/K/V, shared expert as routed experts | Q16/Q28, Q23/Q35, Q14, Q37/Q50 |
| EAGLE-3, DFlash, draft vocabulary trim | Survey C 3.1–3.3, Q20 |
| Multi-token kernels | Q78 |

## What this changes in the plan

- **Q41's multi-attempt logs answer idea 1 now**, at no GPU cost: shortest-of-k length and pass rate per lane.
- **Q85 gains two arms:** reflection suppression inside `<think>`, measured as reasoning tokens per answer next to the
  hard budget; self-training only if suppression holds the lanes.
- **Q87's executor should get its multi-sequence loop early** (races, throughput), a position-keyed Gumbel sampler (exact
  coupling, seeded output) and a slot for a table guesser; Q20's drafter should output a distribution for coupling.
- **Q81 should time a scatter/gather round trip to N cards**, not only one hop: it gates ideas 3 and 6.
- **Q59/Q60's kernel should expose its partials** (output, maximum, sum per head) so that sharding can reuse it.
- **Tied-scale emulation** tests any coarser scale group (Q8_0, Q5_0, Q6_0) with stock tools before a type is written.

## Not verified

- Flint's reasoning-length spread between samples of one prompt (idea 1) and its reflection-word rate (idea 2); both
  ideas' quality on tool calls and code, where the cited papers have no data.
- NoWait on an RL-trained MoE; the paper's RL-trained models are dense 32B models, and Qwen3-32B lost 6 points on GPQA.
- Self-training's −30 % on agent and tool-call tasks (the paper used GSM8K and MATH); LoRA training of Flint in
  transformers across the quant machine's GPUs without P2P.
- The host-staged scatter/gather round trip (0.25–0.6 ms per layer assumed); a device-spanning attention op in the
  fork's scheduler; the Celeron's merge time.
- The fresh-text accuracy of a Flint-output table and the coupled acceptance at temperature 1.0.
- Per-stream speed with 2–3 staggered sequences under the 75 W limit and on the 2-core host.
- The coarse-group quality estimates and Q5_0's sensitivity to group 64.
- The router's real time per launch without the profiler (~20 µs assumed from ~25 µs with it).
- Flint's head sink masses; whether any head is dormant on nearly all tokens.
- The KV state size and transfer time for prefill on another PC.

## Sources

Reasoning length and selection
- C. Wang et al., "Wait, We Don't Need to 'Wait'! Removing Thinking Tokens Improves Reasoning Efficiency" (NoWait),
  [arXiv 2506.08343](https://arxiv.org/abs/2506.08343), [html](https://arxiv.org/html/2506.08343v2) (P)
- T. Munkhbat et al., "Self-Training Elicits Concise Reasoning in Large Language Models", ACL Findings 2025,
  [arXiv 2502.20122](https://arxiv.org/abs/2502.20122) (P)
- A. Agarwal, A. Sengupta, T. Chakraborty, "First Finish Search: Efficient Test-Time Scaling in Large Language Models",
  [arXiv 2505.18149](https://arxiv.org/abs/2505.18149) (P)
- M. Hassid, G. Synnaeve, Y. Adi, R. Schwartz, "Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM
  Reasoning", [arXiv 2505.17813](https://arxiv.org/abs/2505.17813) (P)
- Y. Fu et al., "Efficiently Scaling LLM Reasoning with Certaindex", [arXiv 2412.20993](https://arxiv.org/abs/2412.20993)
  (P)
- R. Liu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", COLM 2025,
  [arXiv 2504.04823](https://arxiv.org/abs/2504.04823) (P)

Long-context attention
- T. Dao, D. Haziza, F. Massa, G. Sizov, "Flash-Decoding for long-context inference", Stanford CRFM, 2023-10-12,
  [blog](https://crfm.stanford.edu/2023/10/12/flashdecoding.html) (P)
- N. Bhatia et al., "Helix Parallelism: Rethinking Sharding Strategies for Interactive Multi-Million-Token LLM
  Decoding", [arXiv 2507.07120](https://arxiv.org/abs/2507.07120), [html](https://arxiv.org/html/2507.07120) (P)
- A. Agrawal et al., "No Request Left Behind: Tackling Heterogeneity in Long-Context LLM Inference with Medha",
  [arXiv 2409.17264](https://arxiv.org/abs/2409.17264) (P)

Guessing and acceptance
- G. Oliaro, Z. Jia, D. Campos, A. Qiao, "SuffixDecoding: Extreme Speculative Decoding for Emerging AI Applications",
  NeurIPS 2025, [arXiv 2411.04975](https://arxiv.org/abs/2411.04975), [html](https://arxiv.org/html/2411.04975v3) (P)
- Z. He et al., "REST: Retrieval-Based Speculative Decoding", NAACL 2024,
  [arXiv 2311.08252](https://arxiv.org/abs/2311.08252) (P)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding",
  [arXiv 2408.07978](https://arxiv.org/abs/2408.07978) (P)
- T. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
  [arXiv 2401.10774](https://arxiv.org/abs/2401.10774), [html](https://arxiv.org/html/2401.10774v3) (P)
- G. Bachmann et al., "Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment",
  [arXiv 2501.19309](https://arxiv.org/abs/2501.19309) (P)
- J. Zhao et al., "QSpec: Speculative Decoding with Complementary Quantization Schemes", EMNLP 2025,
  [arXiv 2410.11305](https://arxiv.org/abs/2410.11305) (P)
- R. Pan et al., "SpecReason: Fast and Accurate Inference-Time Compute via Speculative Reasoning",
  [arXiv 2504.07891](https://arxiv.org/abs/2504.07891) (P)

Pipeline balance
- M. T. Yeung, P. Qi, M. Lin, X. Wan, "Balancing Pipeline Parallelism with Vocabulary Parallelism",
  [arXiv 2411.05288](https://arxiv.org/abs/2411.05288) (P)

Removing parts of the model
- S. He, G. Sun, Z. Shen, A. Li, "What Matters in Transformers? Not All Attention is Needed",
  [arXiv 2406.15786](https://arxiv.org/abs/2406.15786) (P)
- X. Men et al., "ShortGPT: Layers in Large Language Models are More Redundant Than You Expect",
  [arXiv 2403.03853](https://arxiv.org/abs/2403.03853), [html](https://arxiv.org/html/2403.03853v3) (P)
- X. Luo, W. Wang, X. Yan, "Adaptive Layer-skipping in Pre-trained LLMs" (FlexiDepth), COLM 2025,
  [arXiv 2503.23798](https://arxiv.org/abs/2503.23798), [html](https://arxiv.org/html/2503.23798v3) (P)
- T. Guo et al., "Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs",
  [arXiv 2410.13835](https://arxiv.org/abs/2410.13835) (P)
- R. Calvo González et al., "Leveraging the true depth of LLMs", [arXiv 2502.02790](https://arxiv.org/abs/2502.02790) (P)

Formats and compression
- X. Chen et al., "MoBE: Mixture-of-Basis-Experts for Compressing MoE-based LLMs",
  [arXiv 2508.05257](https://arxiv.org/abs/2508.05257) (P)
- G. Fang et al., "MaskLLM: Learnable Semi-Structured Sparsity for Large Language Models", NeurIPS 2024,
  [arXiv 2409.17481](https://arxiv.org/abs/2409.17481) (P)
- T. Zhang et al., "70% Size, 100% Accuracy: Lossless LLM Compression for Efficient GPU Inference via Dynamic-Length
  Float" (DFloat11), NeurIPS 2025, [arXiv 2504.11651](https://arxiv.org/abs/2504.11651) (P)
- S. Bae et al., "Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA", ICLR 2025,
  [arXiv 2410.20672](https://arxiv.org/abs/2410.20672) (P)
- A. Liu et al., "SuperBPE: Space Travel for Language Models", [arXiv 2503.13423](https://arxiv.org/abs/2503.13423) (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q6, Q11, Q16, Q19, Q21, Q30, Q35, Q38, Q41, Q45, Q49, Q51, Q55,
  Q56, Q58, Q60, Q64, Q73, Q77, Q78, Q81, Q85, Q87, Q88, Q90), [breakthrough candidates](breakthrough_candidates.md),
  [pipelined speculation](pipelined_speculation.md), [n-gram replay and pipeline simulator](../../../bench/ngram_replay/README.md),
  [expert overlap](../../../bench/expert_overlap/README.md), [kernel integration](pascal_mmvq_integration.md) section 10
  (per-tensor profile), [attention at long context](pascal_attention_long_context.md) (KV bytes, Q60 predictions),
  [survey A](quant_speed_a_file_only.md), [survey C](quant_speed_c_retraining.md), [Flint speed-ups](flint_speedups.md),
  [persistent kernel bench](../../../bench/persistent_pascal/README.md), Q41's harnesses
  ([bfcl_flint](../../../bench/bfcl_flint/), [livecodebench_flint](../../../bench/livecodebench_flint/))
- Fork source, branch `pascal-p106-v2`: `src/models/flint.cpp` (`attn_sinks` of shape n_head per layer; RoPE only on
  layers where `has_rope`)
