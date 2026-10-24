# A one-token guesser for pipelined speculation and the first-finish race (2026-10-24)

Scope: which guesser (draft model) should feed the pipelined executor of Q87 and share the idle cards with the
first-finish race of Q93 for Flint (40B-A2.5B MoE) on nine P106-090 cards, and how to build it on the 7-GPU quant
machine. The aim is a guess that matches Flint's temperature-1.0 sample at least 70 % of the time and costs a fraction of
one card's stage time. The note compares the candidate drafters, works out which Flint features each one needs and when
those features exist in a pipeline, gives a training plan, and turns the expected acceptance into speed per workload
through Q87's simulator. It builds on [breakthrough candidates](breakthrough_candidates.md) (candidate 1),
[pipelined speculation](pipelined_speculation.md), [unconventional ideas](unconventional_ideas.md) (ideas 1, 2 and 4),
[the n-gram replay and pipeline simulator](../../../bench/ngram_replay/README.md), the dropped
[EAGLE-3 plan](../../../bench/eagle3_flint/TRAINING_PLAN.md) and [survey C](quant_speed_c_retraining.md) (3.1–3.6).
Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs: code and correctness on a laptop GPU, generation,
  feature dumps and training on the quant machine (never speed), speed on the rig only.
- "p" is the per-token acceptance: the probability that the guess for a position equals Flint's sample there, given that
  every earlier guess in flight was right. Published "α" values are converted to p where the source reports something
  else; the conversion is given each time.

Starting point, measured ([registry](../../../quants/experiments.md)):
- Flint on six cards, fork build: 37.3 tok/s, 26.8 ms per token. Q87's simulator splits it into stages of 3.91 / 3.48 /
  3.48 / 3.48 / 3.48 / 5.98 ms (9/8/8/8/8/8 layers, the output head and sampling on the last card), 0.3 ms per hop and
  1.5 ms from a sample to the next launch on card 1. With every card busy one token leaves every 6.28 ms (4.3× one
  stream); about S_eff = 4.2 tokens can be in flight before a wrong guess costs time. Nine cards: 31.3 ms per token,
  stages 3.05 / 2.54 / 5.04 ms (6/6/6/6/5/5/5/5/5) or 3.05 / 2.54 / 3.52 ms with only 2 layers next to the head
  (6/6/6/6/6/6/6/5/2), S_eff 5.9 and 8.5.
- Simulated on Flint's own transcripts (Q87, Q96), six cards: copy lookup plus `ngram-cache` 1.34× overall; copy plus a
  continuation table and the request's own counts 1.44×; a guesser right with probability p on every token behind the
  copy lookup 1.85× at p = 0.5 and 2.32× at p = 0.7 (2.12× and 2.88× on the nine-card 6/…/5/2 split).
- Flint at its own sampling (temperature 1.0, top-k 20, top-p 0.95) is nearly deterministic: the most likely token holds
  0.90 of the probability on average (chat 0.87, code 0.90, edits 0.93, agent turns 0.83) and is the sample 89.7 % of
  the time; the kept set holds 1.69 tokens on average (Q96, 65,558 transcript tokens, Flint Q8_0).
- First-finish race (Q93): the shortest reasoning among k sampled attempts is 0.72 / 0.64 / 0.56 of the mean for
  k = 2 / 3 / 5, with BFCL pass rates +0.5 / +1.0 / +3.2 points over a random attempt.
- Flint: 49 layers (2 dense, 47 MoE with 256 experts top-6 and a shared expert), hidden 2,048, 32 query and 4 KV heads
  of 128, vocabulary 150,144 from its own byte-level BPE, no MTP head, about two thirds of its output is reasoning.

## Verdict

- **Recommended guesser: a partial-depth pipeline head, the design of Speculative Pipeline Decoding (SPD), adapted to
  Flint.** Two dense transformer layers at Flint's width with a 32k-row output head, trained on Flint's own
  temperature-1.0 outputs. For each token still in the pipeline it reads the deepest Flint hidden state that already
  exists (the output of whichever card that token last left), for older tokens the final hidden state, and for the
  newest token its embedding, or card 1's output when the head runs on card 1. It outputs a calibrated distribution; the
  guess is drawn with the same position-keyed Gumbel noise as Flint's sampler, and the copy lookup goes first. Est. p
  0.60–0.72 at temperature 1.0 on fresh text, ~0.9 inside copies. Cost per guess on a P106 est. 1.6–2.6 ms (~0.19 GB of
  Q8_0 and ~30 launches), inside one 6.28 ms cycle on a spare card; on card 1 a slimmer variant (one layer or a 4-bit
  head) fits card 1's ~2 ms of slack.
- **Expected speed (Q87's simulator):** at p 0.6–0.7 behind the copy lookup, 2.05–2.32× overall on six cards (chat
  ~1.85–2.09×, code ~2.03–2.27×, edits ~2.4–2.6×, agent turns ~1.9–2.16×, file rewrites ~4.1×) and 2.5–2.9× on nine
  cards with a light last card; ~1.97× at p 0.7 if the 75 W limit slows every stage by 10 %. Against today's 1.34–1.44×.
- **The dependency is real, and it decides the design.** In a pipeline the newest final hidden state belongs to a token
  about S_eff ≈ 4 (six cards) to 6–8 (nine cards) positions behind the one entering card 1. Every drafter that reads the
  previous token's final hidden state (EAGLE-3 as built, Medusa, an MTP head) must either predict 4–8 tokens ahead or
  chain that many of its own steps per guess on the slowest card, so it supports only one or two guesses in flight
  (est. 1.5–1.8× at best). Two kinds of guesser avoid it:
  - **Token-only guessers** (copy lookup, tables, a small draft LM) need nothing from Flint, and their acceptance does not
    fall along the chain of guesses; the small LM pays for that with lower acceptance (est. p 0.45–0.6) and its own full
    forward per guess.
  - **Partial-depth guessers** use the hidden states Flint computes anyway for the tokens in flight. A guess for t+1
    never waits for token t to finish: SPD starts its draft at the beginning of a cycle from what exists then, and a head
    on card 1 waits only for card 1's own output for token t, which costs card 1's slack (~2 ms per cycle on six cards,
    ~0.5 ms on the 6/…/5/2 split unless card 1 gets fewer layers).
- **70 % at temperature 1.0 is the upper end of what the evidence supports.** The only published pipeline drafter (SPD,
  dense Qwen3.5-4B, temperature 1, top-k 50) reaches p ≈ 0.68–0.80 depending on stage count and how its step acceptance
  is converted. Flint's peaked sampler helps; an MoE target, a smaller training set and coupled matching instead of
  rejection sampling cost (est. −0.05 to −0.15 together). Plan on 0.6; 0.7 is the stretch.
- **The race and the guesser share one budget of ~S_eff tokens in flight.** With a guesser at p ≥ 0.5, two race samples
  with one guess each match one sample with three guesses in reasoning time on six cards (est. 0.46–0.51 of a plain
  sample's) and keep the race's selection; four samples without guesses (0.60) only pay for their quality effect. On nine
  cards both fit: two samples with 2–3 guesses each. After `</think>` the winner takes every slot as guesses.
- **Rejected:** EAGLE-3 heads at the last card (one to two guesses in flight, on the bottleneck card); Medusa heads (a
  head 4–8 tokens ahead guesses far below 0.5); layer-skip self-drafting (layer-sparse drafts of MoE models accept 1.1–1.7
  tokens of 8, and every draft uses Flint's own cards); expert-sparse self-drafting (≥ 0.7 of a token's card time per
  guess, since 1.52 of Flint's 2.24 GB per token are dense); a drafter on the host (2 Celeron cores without AVX, single-
  channel memory). No Flint drafter or same-tokenizer small model exists to reuse (Hugging Face, checked 2026-10-24).
- **First experiment, offline:** generate ~20k Flint answers at temperature 1.0 on the quant machine, dump the hidden
  states at the six-card boundaries (layer inputs 9, 17, 25, 33, 41 and the final state) and the sampler's top-20
  probabilities, train five small heads on the same data (staircase as on card 1, staircase from a spare card, card-1
  output only, full depth as the after-miss bound, tokens only as the control), score coupled acceptance on Q51's
  held-out transcripts and feed it to `pipeline_sim.py`. About 40–70 GPU-hours, no rig time. Gate to the rig: coupled p
  ≥ 0.55 on held-out fresh text. Kill: the best head below 0.5.

## Ranked candidates

"Pipelining" says how many guesses can be in flight without the guesser slowing the cycle on six cards. Costs are for
one guess on one P106-090 with the fork's kernels at Q8_0 (est. from bytes at 100–140 GB/s for small matrices plus
5–8 µs per launch). Training costs are for a pilot on the quant machine.

| # | Candidate | p at T 1.0 (est.) | Cost per guess on a P106 (est.) | Pipelining compatible? | Training cost (est.) | First experiment | Kill criterion |
|---|---|---|---|---|---|---|---|
| 1 | **Partial-depth "staircase" head** (SPD-style): 2 layers, fusion of the deepest available Flint state per in-flight token, 32k head, Gumbel-coupled, behind the copy lookup | 0.60–0.72 fresh text; ~0.9 in copies | ~0.19 GB, ~30 launches: 1.6–2.6 ms | Yes, up to S_eff − 1 guesses. On card 1 after card 1's stage (freshest inputs, costs card-1 slack), or on a spare card from the start of each cycle | Generation 15–20 GPU-h, feature dump 8–15, training 5–15 per head; 0.3–0.6 TB of features or streamed | D1: five heads on the same dump, coupled p on held-out Q51 transcripts, then `pipeline_sim.py` | Coupled p < 0.55 on held-out fresh text at the six-card staircase |
| 2 | **Card-1 exit head** (PPSD/Kangaroo-style): 1 layer on card 1's output for the newest token only, 32k head | 0.40–0.60 | ~0.12 GB, ~15 launches: 0.9–1.3 ms | Yes; serial inside card 1 (needs ~1–2 ms of card-1 slack); no cross-card features | Same generation; one 4 KB feature per token (~0.1 TB); training 3–8 GPU-h | Trained in D1 | p < 0.5, or < 0.05 above the token-only control |
| 3 | Copy lookup, else table + request counts (Q96), top guess | 0.40 measured (0.20 fresh text) | Host lookup, ~0.01–0.05 ms | Yes, fully | None beyond generation | Done (Q96): 1.44× | – (stays as the front end and fallback) |
| 4 | Token-only draft LM, 0.1–0.15 B, vocabulary transplanted to Flint's tokenizer, healed, distilled on Flint's outputs | 0.45–0.60 | ~0.13 GB but ~300 launches: 3–4.5 ms (0.5 B class: 4.5–6 ms) | Yes, no dependency; must keep up with one guess per cycle and restart within a card-1 stage after a miss | Transplant + 1–3 B tokens of healing + distillation: 50–100 GPU-h | D4, only if feature transport fails on the rig | p < 0.5 after distillation, or > 4 ms per guess |
| 5 | EAGLE-3 head at the last card, single step on the final hidden state | 0.64–0.73 per step | 145 MB per step, 2.5–4 ms per step; D+1 sequential steps per guess | No: one to two guesses in flight; extra steps land on the slowest card. Est. 1.5–1.8× | Pilot 35–56 GPU-h ([plan](../../../bench/eagle3_flint/TRAINING_PLAN.md)) | None; D1's full-depth head is its upper bound | – (rejected by the dependency) |
| 6 | Medusa heads at the last card | ≤ 0.3–0.5 for heads 1–2 tokens beyond the next; lower deeper | ~75 MB per head, ~0.7 ms | No: needs a head 4–8 tokens ahead | Like an EAGLE head | None | – (rejected) |
| 7 | Layer-skip or expert-sparse self-drafting through Flint | ≤ 0.4 (layer-sparse); higher expert-sparse | ≥ 9 ms of Flint card time (a third of the layers) or ≥ 0.7 of a token's card time (1 of 6 experts) | No: each guess occupies Flint's own cards | None (training-free) | None | – (rejected) |
| 8 | Hybrid per token by confidence | – | – | – | – | Part of D1/D2: copy first, then the head; the head's confidence allocates slots between guesses and race samples and gates guesses under the power limit | – |

## 1. What a pipeline guesser has to do

### Rate and deadline

The executor ([pipelined speculation](pipelined_speculation.md)) keeps up to D guesses in flight behind the newest
committed token and submits a new guess whenever a slot opens. In steady state with every guess right that is one guess
per cycle, the slowest stage plus a hop:

| Layout | Cycle (every card busy) | Card 1 stage | Card-1 slack per cycle (cycle − stage − hop) | S_eff |
|---|---|---|---|---|
| 6 cards, 9/8/8/8/8/8 | 6.28 ms | 3.91 ms | ~2.1 ms | 4.2 |
| 9 cards, 6/6/6/6/5/5/5/5/5 | 5.34 ms | 3.05 ms | ~2.0 ms | 5.9 |
| 9 cards, 6/6/6/6/6/6/6/5/2 | 3.82 ms | 3.05 ms | ~0.5 ms | 8.5 |

After a miss the sampled token enters card 1 (1.5 ms after its sample), and the next guess has to be ready by the time
card 1 is free again, ~4 ms later on six cards. A guess for position j conditions on the committed tokens and on the
guesses in flight before j: those are exactly the true prefix whenever the guess matters, so the relevant acceptance is
conditional on a correct prefix (the n-α of the EAGLE papers).

### Acceptance under sample-and-match

The last card samples once per position and keeps the in-flight guess when it equals the sample, so the output is plain
decoding's output.
- **A point guess g** is kept with probability p(g), Flint's sampler probability for g. A guesser that always names
  Flint's most likely token is capped at the average top-token probability: 0.90 on Flint's transcripts, 0.83 on agent
  turns.
- **A coupled guess.** The sampler draws argmax(log p + G) over Flint's kept tokens with Gumbel noise G keyed by (seed,
  position, token); the guesser draws argmax(log q + G) over its own distribution q with the same noise. The match
  probability is Σ_i 1 / Σ_j max(p_j/p_i, q_j/q_i), at least (1 − TV)/(1 + TV) ([Daliri et al.](https://arxiv.org/abs/2408.07978)),
  and the output still depends only on the seed. On Flint's dumped probabilities with a modelled drafter (Flint's own
  log-probabilities plus Gaussian noise of 0.5–2 nats) coupling raised the match rate by 2.4–6 points (Q96). It does
  not help count-based guessers, whose counts spread where Flint is sure.
- **Rejection sampling** at the last card would keep each guess with probability 1 − TV, a little more than coupling,
  and stays exact in distribution, but the output then depends on the guesser and is no longer bit-identical to plain
  decoding with the same seed, which the executor's identity checks rely on. It is the fallback if coupling falls short
  by more than ~3 points (est.).

What the target means for the guesser (est.): with Flint's peaked distribution an uncoupled top guess reaches p ≈ 0.90·a
+ 0.05·(1 − a) for top-1 agreement a with Flint's most likely token, so p 0.70 needs a ≈ 0.77; with coupling, a ≈ 0.70–0.74
is enough. That is the top-1 agreement of EAGLE-class heads on dense chat models at temperature 0 (0-α 0.74–0.79,
section 2.1).

## 2. Candidates (question 1)

### 2.1 Published acceptance at temperature ≥ 0.7

| Drafter | Target | Setting | Per-token acceptance | Source |
|---|---|---|---|---|
| EAGLE: 1 layer on the target's final hidden state, first step (0-α) | Vicuna 7–33B, LLaMA2-Chat 7–70B | MT-bench, T 1, rejection sampling | 0.71–0.73 | [EAGLE](https://arxiv.org/abs/2401.15077), Table 2 |
| Same, after 1–4 steps on its own predicted features (1-α … 4-α) | same | same | 0.64–0.69 | same |
| Same on an MoE target | Mixtral 8x7B Instruct | T 0 only | 0-α 0.67, 1-α…4-α 0.61–0.64 (dense targets at T 0: 0.74–0.79) | same, Table 3 |
| EAGLE ablation: tokens only, features only, features + shifted token | Vicuna 7B | T 0 | speed-up 1.5×, 1.9×, 2.8×: the target's features carry most of a 1-layer head's accuracy | same, Figure 4 |
| SPD pipeline draft module, 3–4 layers, staircase of partial-depth features | Qwen3.5-4B, 8 stages | T 1, top-k 50, rejection sampling | step acceptance α 0.34 → p ≈ 0.72 (0.68–0.77) | [SPD](https://arxiv.org/abs/2605.30852), Tables 2–3 |
| Same | Qwen3.5-4B, 4 stages | same | α 0.575 → p ≈ 0.75 (0.75–0.80) | same |
| Same, 1 layer | Qwen3.5-4B, 8 / 4 stages | same | α 0.27 / 0.50 → p ≈ 0.62 / 0.67 | same, Table 3 |
| First-stage features only (PPSD as re-run in SPD) | Qwen3.5-4B, 8 / 4 stages | T 1 | p ≈ 0.35 / 0.37 (from its theoretical speed-ups 1.44 / 1.39) | same, Table 2 |
| PPSD: early exit after 20–25 % of the layers + 1 transformer layer | Vicuna-7B/13B, LLaMA-2-13B | temperature not stated | 0.67–0.85 (XSum < GSM8K < HumanEval); a norm-only exit head 0.32–0.52 | [PPSD](https://arxiv.org/abs/2509.19368) |
| Kangaroo: exit after layer 2 of 32 + 1 attention layer, target's head | Vicuna-7B | Spec-Bench | compression rate 1.41–2.22 with up to 6 drafts and an early stop (p ≈ 0.5, est.) | [Kangaroo](https://arxiv.org/abs/2404.18911), Table 1 |
| Medusa head for the token after next | Vicuna | – | top-1 ≈ 0.6, top-5 > 0.8 (authors' post); later heads lower | [Medusa](https://arxiv.org/abs/2401.10774), [authors' post](https://together.ai/blog/medusa) |
| Independent small LM, 100M / 2B / 8B | LaMDA 137B, dialog | T 1 (T 0) | 0.57 / 0.71 / 0.74 (0.61 / 0.71 / 0.75) | [Leviathan et al.](https://arxiv.org/abs/2211.17192), Table 3 |
| T5-small / base / large | T5-XXL, EnDe / CNN-DM | T 1 | 0.62 / 0.68 / 0.71; 0.53 / 0.55 / 0.56 | same |
| 115M draft (4 layers × 1,024), pretrained on 600B tokens, distilled on target outputs | Llama-2-Chat 7B | greedy, 3 drafts | block efficiency 2.4 → p ≈ 0.66; +21 % block efficiency from distillation at T 0.6 | [Goel et al.](https://arxiv.org/abs/2403.00858) |
| EAGLE-3 heads on MoE targets, chain drafts in vLLM | Qwen3-30B-A3B, gpt-oss-20b | as on the model cards | first-token acceptance 0.74 / 0.64 (from τ at k = 1) | [EAGLE-3 plan](../../../bench/eagle3_flint/README.md) table |
| Layer-sparse vs expert-sparse self-drafts, 8-token drafts | DeepSeek-V2-Lite, OLMoE | – | accepted per step 1.70 / 1.06 (layer-sparse) against 4.73 / 3.08 (expert-sparse) | [S2-MoE](https://arxiv.org/abs/2608.15018), appendix Table 5 |
| Copy lookup, else table + request counts | Flint Q8_0 | T 1, top-k 20, top-p 0.95 | 0.40 overall; 0.20 on fresh text; 0.90 in copies | Q96 |

How SPD's numbers were converted: SPD reports α = N/K, tokens per pipeline step, and a zero-draft-cost speed-up of α·n.
A right guess costs one step and a miss refills the pipeline in about n steps, so 1/α = p + (1 − p)·n. Table 3's
averages (MT-Bench, GSM8K, HumanEval) and Table 2's overall speed-ups give slightly different α, and a refill of n − 1 or
n + 1 steps moves p by ±0.04; the ranges above cover both.

### 2.2 How the published figures transfer to Flint

- **Flint's sampler is close to greedy.** At top-k 20 and top-p 0.95 the sample is the most likely token 90 % of the
  time, so the usual drop from temperature 0 to 1 (EAGLE −0.02 to −0.08 per token; T5-small −0.12 to −0.13) should be smaller for
  Flint than for SPD's top-k 50 sampling (est. +0.03 to +0.08).
- **MoE targets are harder to draft.** EAGLE on Mixtral sits ~0.07–0.12 below its dense targets at temperature 0, and
  the gpt-oss head below the Qwen3-30B-A3B head. Flint routes 6 of 256 experts with little locality (Q51), est. −0.05 to
  −0.10.
- **Reasoning and code text is more predictable than chat.** EAGLE-3 on DeepSeek-R1-Distill-LLaMA-8B at temperature 1
  accepts the longest drafts on GSM8K (τ 6.10 against 4.49 on MT-bench); PPSD accepts most on HumanEval; Flint's own top
  token is surest on edits (0.93) and code (0.90). Flint's reasoning is mostly about code, so est. +0.03 to +0.08 over
  chat-only figures, but agent turns, where Flint is least sure (0.83), sit lower.
- **Less data than the papers.** SPD trained on ~1.2 M samples for one epoch, EAGLE on ~68k dialogues; a pilot of ~20k
  Flint answers (~24 M response tokens) is small, est. 0 to −0.05.
- **Coupled matching instead of rejection sampling,** est. −0.02 to −0.04 against the 1 − TV the papers measure.
- **Partial depth on Flint's splits.** SPD's staircase comes from 4–8 equal stages of a 32-layer model; on six cards
  card 1 holds 9 of 49 layers (18 %), and the newest token reaches only card 1. The four-stage SPD results (shallower
  staircase relative to depth) are closer to Flint on six cards than the eight-stage ones.

### 2.3 The candidates one by one

**EAGLE-3 head, single step (rejected for pipelining).** It reads three of Flint's layers (inputs of layers 2, 24 and 46,
on cards 1, 3 and 6) for the previous token and drafts the next one. Without guesses in flight that works as in plain
speculative decoding, but in a pipeline the token entering card 1 is 4–8 positions newer than the newest token with
those features. Two ways out, both bad here: chain D + 1 of its own steps per guess (EAGLE's own n-α at temperature 1
falls to 0.64–0.69 after 1–4 self-predicted steps, and each step costs 2.5–4 ms on the last card, the slowest stage), or
guess only one or two tokens ahead of each fresh feature (est. D ≤ 2: the event model gives 1.62× at D = 1 and 1.95× at
D = 2 for p 0.7, lower here because the second guess uses a self-predicted feature and the extra steps load the last
card; est. 1.5–1.8×). The staircase head is EAGLE-3 with the self-predicted features replaced by the real partial-depth
features the pipeline computes for the guesses in flight anyway.

**Medusa heads (rejected).** Head k predicts token t + k + 1 from the final hidden state of token t, in parallel and
cheaply. The pipeline needs heads 4–8 tokens ahead; the head for the token after next is right ~60 % of the time on
Vicuna (the authors' figure), and accuracy falls with each further head. Medusa-style multi-head training also needs the same
feature dump as EAGLE.

**Partial-depth staircase head (recommended).** SPD's pipeline draft module is an L_s-layer decoder (L_s ≈ L/n − 1, so
it finishes within one stage) on its own GPU. Each token contributes one fused feature built from its hidden states at
anchor layers (for a 32-layer model, layers 0, 8, 16, 24, 31), restricted to the depth that token has reached; verified
positions use full depth, the newest token only its embedding. The module drafts before each pipeline step from the
states available then ("pre-step"), so drafting overlaps the stage compute; SPD found that waiting for the step's own
outputs ("post-step") raises the zero-cost speed-up but loses once the draft latency is exposed. Its head starts from the
target's output head with a smaller vocabulary and is trained by KL divergence to the frozen target's logits. On Flint:
- anchors at the card boundaries (layer inputs 9, 17, 25, 33, 41 and the final hidden state on six cards);
- 2 layers at hidden 2,048 (32 query and 4 KV heads, FFN 4,096: ~44 M parameters each), a fusion projection (~29 M) and
  a 32k head initialized from Flint's output-head rows (65.5 M): ~180 M parameters, ~0.19 GB at Q8_0;
- input embeddings from Flint's own table (kept on the host by llama.cpp, one row per token);
- trained against Flint's sampler distribution (top-k 20, top-p 0.95), so that its q is close to p where coupling needs
  it;
- per cycle it recomputes the few in-flight rows (their features deepen as they move through the cards) and caches each
  committed row once at full depth.

**Card-1 exit head (the cheap variant).** PPSD runs the first E layers of the target plus a one-layer exit head on the
first pipeline worker, drafts token i + 1 from token i's early-exit state, and feeds it straight back into the same
worker: a serial loop inside stage 1. At 20–25 % depth with a transformer head it reports 0.67–0.85; with only a norm
and a linear head 0.32–0.52. SPD's re-run of PPSD at temperature 1 implies ~0.35–0.37, and SPD calls first-stage
features "information-poor". On Flint the head would read card 1's output for the newest token (18 % depth) and nothing
from other cards: no cross-card feature traffic and one row per guess. It is the staircase head with every input but one
removed, so D1 measures what the other cards' features add.

**Token-only draft LM (fallback).** No small model shares Flint's tokenizer: Recursive publishes Flint, flint-int4, its
GGUF and an NVFP4 GGUF, and Hugging Face searches for Flint drafters (EAGLE, DFlash, MTP, speculator, small or mini
variants) return nothing (2026-10-24). Flint's 150,000-token BPE shares 68.5 % of its token strings but only the byte
tokens' ids with Qwen3.6 ([Flint speed-ups](flint_speedups.md)). Routes:
- **Vocabulary transplant** ([transplant-vocab](https://github.com/jukofyork/transplant-vocab): a donor's weights with
  the target's tokenizer, embeddings and head rows built from the donor tokens each target token decomposes into; it
  reports no acceptance), then healing on 1–3 B tokens of Flint-tokenized public text, then distillation on Flint's
  outputs with its top-20 probabilities. A 135M-class donor (30 layers × 576) costs ~3–4.5 ms per guess on a P106,
  mostly launches; a 0.5B-class donor 4.5–6 ms, too slow for the nine-card cycles.
- **Cross-vocabulary verification** ([Timor et al.](https://arxiv.org/abs/2502.05202), lossless, up to 2.8× on GPUs)
  works with off-the-shelf drafters but re-tokenizes text on every step, work the 2-core host cannot spare, and llama.cpp
  requires matching vocabularies.

Published independent drafts reach p 0.53–0.71 at temperature 1 (100M–2B), and 0.66 for a distilled 115M draft that
had 600B tokens of pretraining; a transplanted and briefly healed draft without Flint's features lands est. 0.45–0.6. Its
advantages: no dependency on Flint at all, no feature traffic, acceptance that does not change with pipeline position,
and an existing llama.cpp code path for draft models (the executor needs a streaming variant that extends one chain by
one token per cycle and rolls back on a miss).

**Layer-skip and expert-sparse self-drafting (rejected).** Both draft with Flint's own weights, which live on Flint's
cards, so each guess takes card time from the pipeline it is meant to fill. S2-MoE measured layer-sparse drafts of MoE
models at 1.06–1.70 accepted tokens per 8-token draft; expert-sparse drafts (1 of 6 experts) accept far more (3.08–4.73)
but still read every dense tensor, 1.52 of Flint's 2.24 GB per token. [Unconventional ideas](unconventional_ideas.md)
rejected the pipeline variant on the same arithmetic (a third of the layers ≥ 9 ms against a ~5 ms cycle).

**n-gram and table guessers with coupling (front end).** Measured in Q96: copy lookup, else a continuation table plus
the request's own counts, 0.40 right overall, 0.20 on fresh text, 1.44× on six cards; coupling lowers count-based
guessers slightly (39.7 → 38.6 %), so they guess their top token. They stay in front of the neural head: the copy lookup
with a ≥ 4-token match is right 83–98 % of the time inside copies, better than any head there.

**Hybrids by confidence.** The head's top probability predicts whether its guess will match. Three uses, none of which
changes the output: the copy lookup overrides the head when it has a ≥ 4-token match; slots go to race samples instead
of low-confidence guesses during reasoning (section 3.4); and under the 75 W limit low-confidence guesses can be skipped,
since a wrong guess costs card work (Q96's best table guesser cancelled 37 ms of stages per token) but no time below
S_eff.

### 2.4 Cost per guess on a P106-090

| Guesser | Weights read per guess (Q8_0) | Launches | Time on one P106 (est.) |
|---|---|---|---|
| Staircase head: fusion + 2 layers for ≤ 6 rows, 32k head for 1 row | ~195 MB | ~30 | 1.6–2.6 ms (upper end if the ≤ 6-row products run on llama.cpp's stock multi-column kernels) |
| Card-1 exit head: 1 layer + 32k head | ~117 MB | ~15 | 0.9–1.3 ms |
| Token-only LM, 135M class (30 × 576), 32k head | ~133 MB | ~330 | 3–4.5 ms |
| Token-only LM, 0.5B class (24 × 896), 32k head | ~0.41 GB | ~260 | 4.5–6 ms |
| EAGLE-3 head (1 layer, FFN 6,144, 32k head) | 145 MB per step | ~25 per step | 2.5–4 ms per step ([survey C](quant_speed_c_retraining.md) 3.1); D + 1 steps per guess |
| Medusa head (residual block + 32k head) | ~75 MB | ~5 | ~0.7 ms |
| Layer-skip / expert-sparse self-draft | ≥ 0.75 GB / ≥ 1.6 GB of Flint | hundreds | ≥ 9 ms / ≥ 0.7 of a token |
| Copy lookup + table | host RAM (the table ~0.3–0.5 GB) | 0 | ~0.01–0.05 ms on a Celeron core |

A 32k head in GPTQ-rounded 4-bit (HyperQwen's draft head lost no acceptance that way; [HyperQwen lessons](hyperqwen_lessons.md))
saves ~35 MB, ~0.3 ms. Every neural guesser also needs its token id on the host before the executor submits the guess
(one small device-to-host read and a submit, est. 0.4–1.5 ms; section 3.3).

## 3. Where the guesser runs, and what it needs from Flint (question 2)

### 3.1 What exists when

Feature availability on six cards in steady state with every guess right, from Q87's stage times; i is the guess
entering card 1 at t = 0 (cycle 6.28 ms), "L9" the input of layer 9 (card 1's output), "full" the final hidden state:

| Token | Pre-step, t = 0 | Late pre-step, t ≈ 3.5 ms | On card 1 after its stage, t ≈ 3.9 ms |
|---|---|---|---|
| i (newest guess) | embedding | embedding | L9 |
| i − 1 | L9 | L17 | L17 |
| i − 2 | L25 | L33 | L33 |
| i − 3 | L33 | L41 | L41 |
| i − 4 | L41 | full | full |
| ≤ i − 5 | full | full | full |

The guess for i + 1 is due at t = 6.28 ms. A drafter on a spare card can therefore start late (about the deadline minus
its own time and the host round trip) and see almost what a drafter on card 1 sees, everything except card 1's output
for the newest token. After a miss every older token is complete, which is the best case for any feature-based head.
Final hidden states are not needed for the tokens in flight; the guesses' own partial states are Flint's real
computation on the true prefix whenever the guess matters, and wrong guesses' states are dropped with their positions.

### 3.2 Dependency classes

| Class | Needs from Flint | Guess for t + 1 waits for | In a pipeline |
|---|---|---|---|
| Token-only (copy, table, small LM) | committed tokens and guesses | nothing from Flint | Compatible; acceptance independent of position in the chain |
| Partial depth, pre-step (SPD) | deepest state of each in-flight token at the drafter's start | nothing of the current cycle | Compatible; a full cycle of compute budget |
| Partial depth on card 1 (PPSD/Kangaroo, or the staircase with a fresh newest token) | card 1's output for token t (plus older states) | card 1 finishing token t | Compatible while drafter time + host round trip ≤ card-1 slack |
| Final hidden state (EAGLE-3 as built, Medusa, MTP) | token t's final state | token t leaving the last card | Limited to 1–2 guesses in flight, or self-predicted chains on the slowest card |

### 3.3 Placement

- **Card 1, after its stage (freshest inputs).** Six cards: head time plus the token's round trip must fit card 1's
  ~2.1 ms of slack. A card-1 exit head (0.9–1.3 ms) with a ~0.4 ms round trip fits; the staircase head fits only at the
  low end of its 1.6–2.6 ms, so on card 1 it runs with one layer, a 4-bit head, or the input embedding gathered on the
  card instead of a host round trip (est.). Nine cards: the 6/…/5/2 split leaves ~0.5 ms, so card 1 must hold one or two
  layers fewer (they move to the light last card); the 6/…/5/5 split leaves ~2 ms. VRAM
  for ~0.2 GB of head plus its KV cache must fit next to card 1's layers (est. fits at 8k on six cards and at 262k on
  nine cards for G3-class files; to be read from the rig's memory log).
- **A spare card (pre-step).** On six cards three cards are idle, so card 7 can host the head with a full cycle of
  budget and no change to Flint's split; it needs every card's boundary state each cycle (one 4 KB copy per card through
  host memory, est. ≤ 0.1 ms each, overlapped). On nine cards there is no spare card; the head then shares a card with
  Flint layers and that card needs fewer layers.
- **The last card (worst).** It holds the output head and is the slowest stage; anything added there lengthens the cycle.
- **The host.** Only for lookups: two 1.8 GHz cores without AVX and ~17 GB/s single-channel memory would need tens of
  milliseconds for a 0.1 B-parameter forward, and the host already launches every stage.

The token id: llama.cpp looks up the input embedding on the host, so a guess made on a GPU travels to the host before it
is submitted. Reading it through mapped pinned memory and submitting a pre-built graph keeps the round trip near the
~0.3 ms per card the host spends launching today (Q49); the simulator's 1.5 ms sample-to-launch gap is the pessimistic
case and would lengthen the six-card cycle by ~8 % with a 1.1 ms head on card 1 (3.91 + 1.1 + 1.5 + 0.3 = 6.8 ms
against 6.28; est.). The coupled noise is a hash of (seed,
position, token), computed on the drafter's card and in the sampler without communication.

### 3.4 Sharing the idle cards with the first-finish race

Guesses and race samples draw on one budget of ~S_eff tokens in flight (4.2 on six cards). With k samples and D guesses
each, k·(1 + D) ≤ S_eff keeps wrong guesses free, and the cards' capacity (one token per cycle) caps the sum of the
samples' speeds. Reasoning time of the winner against one plain sample (est.; Q93's shortest-of-k fractions, the event
model's speed per sample, capped by capacity):

| Six cards, reasoning phase | Guesser p 0.5 | p 0.6 | p 0.7 | Today's guessers (Q96) |
|---|---|---|---|---|
| k = 1, D = 3–5 | 0.62 | 0.54 | 0.47 | 0.85 (chat-like reasoning) |
| k = 2, D = 1 each | 0.51 | 0.50 | 0.46 | 0.65 |
| k = 4, no guesses | 0.60 | 0.60 | 0.60 | 0.60 |

- With a trained guesser, two samples with one guess each are as fast as one sample with three guesses and keep the
  race's selection effect (+0.5 points at k = 2 on BFCL, not significant); four samples only buy their quality effect
  (+1.0 at k = 3, +3.2 at k = 5).
- Without one, the race is the better use of the cards in reasoning, as [pipelined speculation](pipelined_speculation.md)
  concluded.
- On nine cards (S_eff 5.9–8.5) two samples with 2–3 guesses each fit (est. ~0.40).
- A neural guesser then serves each sample: one head step per token of each sample, which the card-1 placement does
  anyway and a spare card does twice per cycle (2 × 1.6–2.6 ms ≤ 6.28 ms).

## 4. Training plan (question 3)

### Data

- **Prompts:** ~20k first turns for the pilot (60k for a full run) in the mix of the [EAGLE-3 plan](../../../bench/eagle3_flint/TRAINING_PLAN.md)
  (chat, code, tool calls in Flint's format, reasoning), plus whole-file edits and agent turns like Q51's categories,
  since those are where Flint is least predictable (agent turns) or copies most (edits).
- **Answers:** generated by Flint at the makers' sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0) with exact
  token ids, reasoning kept, on llama-server copies across all seven GPUs: ~24 M response tokens, 3–4 h, 15–20 GPU-h
  (that plan's estimate). Q51's 38 transcripts (65,558 generated tokens) stay out of training as the test set.
- **Which file:** features from Flint Q8_0 match Q51's test transcripts and Q96's dumped probabilities; a check with
  features from the release file (G3h, K2h or K1h) follows before the rig run, since the head will read the release
  file's hidden states.

### Features and targets

- A dump tool on llama.cpp's hidden-state API (`llama_set_embeddings_layer_inp`, as in the EAGLE-3 plan) replays each
  sequence as one batch and saves the layer inputs 9, 17, 25, 33 and 41 and the final hidden state in BF16 (24 KB per
  token: ~0.6 TB for 24 M tokens; 0.3 TB in FP8 with a scale per row, or streamed to the trainer and deleted after use),
  and the sampler's top-20 log-probabilities (llama-logit-dump, already used for Q96: 160 B per token). Nine-card splits
  map their boundaries to the nearest lower dumped layer (12 → 9, 18 → 17, …), so one dump serves every split at a small
  loss of freshness (est.).
- Flint in transformers is not needed: its attention runs eagerly there, the BF16 model (80 GB) exceeds the three
  RTX 3090s, and the V100s lack BF16. SPD's trainer runs the target in-process and its repository states no license, so
  the module is reimplemented from the paper over offline features.

### Vocabulary

The 32,000 ids most frequent in the generated answers, with Flint's control tokens (`<think>`, `</think>`,
`<tool_call>`, `<arg_key>`, `<arg_value>` and closing tags) forced in, as in the EAGLE-3 plan; coverage reported per
class, ~40k if code coverage is below ~96 % (HyperQwen: a list counted over the model's own outputs gave +10 % over a
web-text list). Q51's transcripts are far too small to set it: 65,558 tokens use 7,331 distinct ids, and 17.5 % of each
transcript's tokens are ids that never occur in the other 37 transcripts. Ids outside the list are certain misses
for the head; the copy lookup still covers them.

### Model and loss

Five heads on the same data, so D1 isolates each design choice:

| Head | Inputs | Purpose |
|---|---|---|
| A: staircase, mid-step | the "on card 1" column of section 3.1 | the recommended design where card 1 has slack |
| B: staircase, pre-step | the "pre-step" column (and late pre-step) | a spare card, or nine cards without slack |
| C: card-1 exit head | L9 of every token only | no cross-card features |
| U: full depth | every token at full depth | upper bound; the after-miss case |
| T: tokens only | embeddings only, same size | what Flint's features add |

Depth per token is sampled from the staircases of all three layouts during training, so one head can serve any split
(est.; SPD trains one fixed staircase). Loss: total variation or forward KL to Flint's sampler distribution over the 32k
ids plus the mass outside them, with cross-entropy on the sampled token (DistillSpec reports that the divergence and
on-policy data choice matter for acceptance; acceptance is 1 − TV under rejection sampling and coupling also rewards a
calibrated q). Compute: ~180 M parameters × ~3 rows per position (staircase depths) × 24 M tokens × 2 epochs ≈ 1.6 × 10^17
FLOP, ~5 GPU-h at ~10 TFLOPS sustained on an RTX 3090 (est.); 3–10 GPU-h per head with data loading.

### GPU-hours

| Step | Pilot (est.) |
|---|---|
| Generation (all seven GPUs) | 15–20 GPU-h, 3–4 h |
| Feature and probability dump | 8–15 GPU-h, 1.5–3 h |
| Training heads A, B, C, U, T (RTX 3090s) | 15–40 GPU-h |
| Offline scoring and simulation | < 1 GPU-h plus CPU |
| **Total** | **~40–70 GPU-h, 2–3 days** |

The token-only LM (D4) adds 50–100 GPU-h: healing a 135M model on 1–3 B tokens takes 16–50 GPU-h at ~17k tokens/s
per RTX 3090 (6 × 0.135 × 10^9 FLOP per token at ~14 TFLOPS), plus tokenizing the corpora with Flint's tokenizer.

### Measuring acceptance offline, before any rig run

1. Dump the same features for Q51's held-out transcripts with the file that made them (Flint Q8_0); Flint's sampler
   probabilities along them already exist (`top20.bin`, Q96).
2. Run each head over the transcripts in pipeline mode: for each position the inputs that a given layout's staircase
   would provide, assuming the earlier guesses were right (the conditional acceptance that matters), and separately the
   after-miss inputs (all older tokens at full depth).
3. Per position: the coupled match probability with `coupled_match` of [pipeline_sim.py](../../../bench/ngram_replay/pipeline_sim.py)
   and the top-guess probability, split into fresh and copied text, reasoning and answer, and the four classes.
4. Feed the per-position probabilities to `pipeline_sim.py` as a probability guesser behind the copy lookup (the
   `ProbGuesser` path Q96 used), with two extensions (D2): a guess exists only once its inputs exist plus the drafter's
   time and the round trip, and the after-miss probability applies to the first guess after a miss.
5. Report speed per workload for six and nine cards, and the race with guesses (k samples × D guesses, which the
   simulator cannot run yet).

## 5. Expected speed-up per workload (question 4)

From Q87's simulator (six cards, depth 5; nine cards, depth 8), the copy lookup first, then a guesser right with
probability p on every remaining token; p 0.6 interpolated (est.):

| Setup and guesser | Chat | Code | Edits | Agent | All | File rewrites |
|---|---|---|---|---|---|---|
| Six cards, copy + `ngram-cache` (Q87) | 1.11 | 1.26 | 1.74 | 1.21 | 1.34 | 4.04 |
| Six cards, copy + table + request counts (Q96) | 1.18 | 1.35 | 1.85 | 1.37 | 1.44 | 4.08 |
| Six cards, copy + head, p 0.5 | 1.61 | 1.79 | 2.20 | 1.68 | 1.85 | 4.04 |
| Six cards, copy + head, p 0.6 (est.) | ~1.85 | ~2.03 | ~2.40 | ~1.92 | ~2.08 | ~4.06 |
| Six cards, copy + head, p 0.7 | 2.09 | 2.27 | 2.61 | 2.16 | 2.32 | 4.09 |
| Nine cards (6/…/5/2), copy + head, p 0.7 | 2.50 | 2.78 | 3.41 | 2.59 | 2.88 | 7.47 |

- In tokens per second on six cards (37.3 plain): p 0.7 gives chat 78, code 85, edits 97, agent turns 81; nine cards
  (32.0 plain) 80, 89, 109 and 83.
- The simulator charges no drafter time; the card-1 and spare-card placements in section 3.3 keep it off the cycle on
  six cards (est.). With every stage 10 % slower from the 75 W limit, a p 0.7 drafter falls to 1.97× (Q87 sensitivity).
- Per answer, counting reasoning (two thirds of the tokens; 89 % on agent turns) and using the race allocation of
  section 3.4 (est.): today's guessers with a four-sample race give ~1.5× on chat and ~1.6× on agent turns; a p 0.6 head
  with two samples × one guess in reasoning gives ~2.0× on both; p 0.7 ~2.1×; nine cards est. 2.4–2.5×.

## 6. First experiments, in order

1. **D1, head pilot (quant machine, 2–3 days, ~40–70 GPU-h).** Generation, dump, vocabulary, heads A/B/C/U/T, offline
   scoring as in section 4. **Gates:** head A or B with coupled p ≥ 0.55 on held-out fresh text (≥ 0.6 overall) goes to
   the rig; C is kept if within 0.05 of A (no cross-card features needed); U − A shows what staleness costs; A − T shows
   what Flint's features buy. **Kill:** the best head below 0.5 on fresh text: no learned guesser; the table guessers
   and the race keep the idle cards.
2. **D2, simulator extensions (CPU, a day).** Drafter time and input availability per placement, after-miss acceptance,
   and races with guesses; six cards and both nine-card splits, plus a nine-card split with one or two layers off card 1.
3. **D3, guess cost on the rig (one card, an hour, then the executor).** A random-weight head of the chosen shape on one
   P106 (like Q19's random EAGLE-3 head): time per guess and the token round trip through the 2-core host; then the head
   as an executor guesser on six cards, on card 1 and on card 7, with the identity check, acceptance against D1's
   offline value, and tokens per second per prompt set against the copy + table guesser. **Kill:** guess plus round
   trip above card 1's slack with no spare card, or a cycle with all cards busy above 0.3 of a plain token.
4. **D4, token-only fallback (only if D3 shows the feature path cannot be fed in time).** Transplant a 135M-class model,
   heal on 1–3 B Flint-tokenized tokens, distill on D1's answers; same offline scoring. **Kill:** p < 0.5 or > 4 ms per
   guess.

## What this changes in the plan

- **Q20 becomes the staircase head of D1,** not an EAGLE-3 head: no batched verify, no chains of self-predicted
  features, one guess per cycle from partial-depth states.
- **Q87's executor needs three hooks:** a GPU guesser slot whose token id reaches the host cheaply (mapped memory, or
  the input embedding gathered on the card), a 4 KB copy of each card's boundary state per token for the head, and the
  position-keyed Gumbel sampler already on its production list.
- **Q96's copy lookup stays in front,** and its table is the fallback when no head is loaded.
- **Q93's race allocation depends on the guesser:** four samples without guesses until a head exists; with a head at
  p ≥ 0.5, two samples with one guess each on six cards and two samples with 2–3 guesses each on nine.
- **The release file fixes the head's inputs:** feature dumps for the final head should come from the file the rig
  serves.

## Already covered

| Direction | Where it is |
|---|---|
| Pipelined executor, bit-identical checks, dropping instead of withdrawing, races in the executor | Q87, [pipelined speculation](pipelined_speculation.md) |
| Copy lookup, continuation table, request counts, Gumbel coupling for count-based guessers | Q96, [n-gram replay](../../../bench/ngram_replay/README.md) |
| First-finish race and its offline gate | Q93, [unconventional ideas](unconventional_ideas.md) idea 1 |
| EAGLE-3 with batched verify, its training pipeline and feature API | Q19, Q20, [EAGLE-3 plan](../../../bench/eagle3_flint/TRAINING_PLAN.md), [survey C](quant_speed_c_retraining.md) 3.1 |
| Draft vocabulary trimming, quantized draft heads | Survey C 3.2, [HyperQwen lessons](hyperqwen_lessons.md) §3 |
| Block drafters (DFlash, DSpark), MTP heads, Medusa with batched verify | Survey C 3.3–3.4 |
| Small same-tokenizer draft model with batched verify, training-free self-speculation | Survey C 3.5–3.6 |
| Self-speculation through skipped layers in the pipeline, low-bit copy of Flint as drafter, small-model reasoning drafts | [Unconventional ideas](unconventional_ideas.md), rejected list |
| Typical or judged acceptance of in-flight guesses (lossy) | Unconventional ideas idea 8 |
| Split output head, which lightens the last card and raises the pipeline ceiling | Unconventional ideas idea 6 |

## Not verified

- Every acceptance figure for Flint is an estimate; no head has been trained on Flint's features.
- SPD's per-token acceptance is derived from its step acceptance under an assumed refill of n steps; its Tables 2 and 3
  give slightly different values.
- PPSD's temperature is not stated, and SPD's re-run of a first-stage-only drafter implies half of PPSD's own acceptance;
  which applies to a card-1 head on Flint is open.
- How much an MoE target and reasoning text move a staircase head's acceptance; the transfer adjustments in section 2.2
  are judgement, not measurement.
- Coupled matching against 1 − TV for a real head (Q96's comparison used a noise model of Flint's own probabilities).
- Q51's transcripts were sampled with min-p 0.05; the makers' min-p 0 keeps slightly more tokens and may lower acceptance
  by a point or two.
- Whether ~20k answers suffice; SPD used ~1.2 M samples, EAGLE ~68k dialogues.
- Head accuracy with features from the release file instead of Q8_0.
- Guess time and the token round trip on a P106 with the 2-core host; the cost of per-card feature copies; VRAM on card 1.
- The race-with-guesses timings, which come from capacity arithmetic, not a simulation.
- Long context: the head's own attention over tens of thousands of positions (a sliding window keeps it cheap, but a
  2,048-token drafter window halved DFlash2's speed at 112k in HyperQwen; the copy lookup covers long-range copies).

## Sources

Pipeline drafters
- Y. Yu et al., "Speculative Pipeline Decoding: Higher-Accuracy Drafting with Hidden Latency via Pipeline Parallelism"
  (SPD), [arXiv 2605.30852](https://arxiv.org/abs/2605.30852), [html](https://arxiv.org/html/2605.30852): pipeline draft
  module, anchors and staircase, pre-step vs post-step, Tables 2–4; [code](https://github.com/yuyijiong/speculative_pipeline_decoding)
  (no license stated) (P)
- Li et al., "Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding" (PPSD),
  [arXiv 2509.19368](https://arxiv.org/abs/2509.19368), [html](https://arxiv.org/html/2509.19368): exit heads, acceptance
  by exit depth, Algorithm 1 (P)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding",
  [arXiv 2408.07978](https://arxiv.org/abs/2408.07978) (P)

Feature-based heads
- Y. Li, F. Wei, C. Zhang, H. Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
  [arXiv 2401.15077](https://arxiv.org/abs/2401.15077), [html](https://arxiv.org/html/2401.15077): Tables 2–3 (n-α at
  temperature 0 and 1, Mixtral), Figure 4 (input ablation) (P)
- Y. Li, F. Wei, C. Zhang, H. Zhang, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via
  Training-Time Test", [arXiv 2503.01840](https://arxiv.org/abs/2503.01840), [html](https://arxiv.org/html/2503.01840):
  Table 1 (temperature 1, DeepSeek-R1-Distill-LLaMA-8B) (P)
- F. Liu et al., "Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting",
  [arXiv 2404.18911](https://arxiv.org/abs/2404.18911) (P)
- T. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
  [arXiv 2401.10774](https://arxiv.org/abs/2401.10774); the authors' post, [together.ai/blog/medusa](https://together.ai/blog/medusa)
  (head accuracy) (P)
- RedHatAI EAGLE-3 speculators for [Qwen3-30B-A3B](https://huggingface.co/RedHatAI/Qwen3-30B-A3B-speculator.eagle3) and
  [gpt-oss-20b](https://huggingface.co/RedHatAI/gpt-oss-20b-speculator.eagle3), acceptance tables on the model cards (P)

Independent and distilled drafts
- Y. Leviathan, M. Kalman, Y. Matias, "Fast Inference from Transformers via Speculative Decoding", ICML 2023,
  [arXiv 2211.17192](https://arxiv.org/abs/2211.17192): Table 3 (α at temperature 0 and 1) (P)
- Y. Zhou et al., "DistillSpec: Improving Speculative Decoding via Knowledge Distillation",
  [arXiv 2310.08461](https://arxiv.org/abs/2310.08461) (abstract: 10–45 % over standard speculative decoding; on-policy
  data and divergence choice) (P)
- R. Goel et al., "Direct Alignment of Draft Model for Speculative Decoding with Chat-Fine-Tuned LLMs",
  [arXiv 2403.00858](https://arxiv.org/abs/2403.00858), [html](https://arxiv.org/html/2403.00858) (P)
- N. Timor et al., "Accelerating LLM Inference with Lossless Speculative Decoding Algorithms for Heterogeneous
  Vocabularies", ICML 2025, [arXiv 2502.05202](https://arxiv.org/abs/2502.05202) (P)
- transplant-vocab, [github.com/jukofyork/transplant-vocab](https://github.com/jukofyork/transplant-vocab) (P)
- Hugging Face model search, 2026-10-24: models by RecursiveSI and searches for Flint drafters (P)

Self-speculation on MoE
- "S2-MoE: Enabling Efficient Self-Speculative Decoding for Mixture-of-Experts on Edge Devices",
  [arXiv 2608.15018](https://arxiv.org/abs/2608.15018), [html](https://arxiv.org/html/2608.15018): Table 1, appendix
  Table 5 (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q19, Q20, Q43, Q49, Q51, Q53, Q64, Q87, Q93, Q94, Q96),
  [breakthrough candidates](breakthrough_candidates.md), [pipelined speculation](pipelined_speculation.md) (event model,
  races), [unconventional ideas](unconventional_ideas.md), [speculation plumbing](speculation_plumbing.md),
  [HyperQwen lessons](hyperqwen_lessons.md), [Pascal forks and quants](pascal_forks_and_quants.md),
  [Flint speed-ups](flint_speedups.md) (tokenizer), [survey C](quant_speed_c_retraining.md),
  [EAGLE-3 check](../../../bench/eagle3_flint/README.md) and [training plan](../../../bench/eagle3_flint/TRAINING_PLAN.md),
  [n-gram replay and pipeline simulator](../../../bench/ngram_replay/README.md) (Q87 and Q96 results,
  `pipeline_sim.py`), Q51's transcripts ([expert_overlap](../../../bench/expert_overlap/README.md)) for the vocabulary
  count in section 4
