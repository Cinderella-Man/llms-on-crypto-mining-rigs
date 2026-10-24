# Raising the staircase head's acceptance beyond more data (2026-10-24)

Scope: what limits the acceptance of the staircase guesser for pipelined speculation (Q20/D1,
[pipeline drafter](pipeline_drafter.md), code in [bench/staircase_drafter](../../../bench/staircase_drafter/README.md)),
what comparable published drafters reach and what limits them, which training-recipe changes have evidence of gains, and
a ranked test plan for five RTX GPUs (3 × RTX 3090, 2 × RTX 3080) on the 7-GPU quant machine on the morning of
2026-10-25, next to the data scaling that runs anyway. Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Acceptance" is the Gumbel-coupled match probability between Flint's sampler (temperature 1, top-k 20, top-p 0.95) and
  the head's distribution sharpened to q^α, with α chosen on validation answers (α = 3 for every head so far), unless a
  row says α = 1 or "top" (the probability that the head's most likely token is Flint's sample).
- "Q51" is the held-out test set: 38 transcripts, 65,520 positions, 72.3 % of them fresh text (outside ≥ 8-token
  copies). "Val" is the 3 % of training answers held out by `train.py` (60 sequences, 75,485 positions).
- "Staircase" is head A's window on card 1 after its stage (newest token at L9, then L17, L33, L41, older tokens at full
  depth); "after-miss" is the first guess after a miss (newest at L9, every older token at full depth). Ln is the input
  of Flint layer n.

Starting point, measured on the quant machine (training logs and Q51 scores of 2026-10-24):

| Run | Training tokens | Steps of 8,192 targets (minutes) | Epochs | Val staircase, α 1 / top / α 3 | Val after-miss, α 1 / top / α 3 | Q51 fresh / overall, α 3 |
|---|---|---|---|---|---|---|
| A, 3.08 M | 3.0 M | 2,275 (121) | 7 | 0.428 / 0.463 / – | 0.473 / 0.507 / – | 0.422 / 0.440 |
| A + feature regression, 3.08 M | 3.0 M | 2,058 (121) | 6 | 0.420 / 0.466 / 0.472 | 0.460 / 0.505 / – | – |
| A, 8.6 M | 8.3 M | 3,028 (120) | 3.3 | 0.469 / 0.520 / 0.527 | 0.509 / 0.557 / 0.565 | 0.463 (0.439–0.486) / 0.505; after-miss 0.504 / 0.553 |
| B (spare card), 8.6 M | 8.3 M | 1,742 (120) | 1.9 | pre-step 0.353 / 0.433 / 0.436 | 0.400 / 0.478 / 0.482 | pre-step 0.380 / 0.405 |
| T (tokens only), 3.08 M | 3.0 M | 10,828 (121) | 33 | 0.296 / 0.302 / – | – | – |

Tonight's heads on 13.9 M tokens (300-minute budgets, C 150) at their 40-minute evaluation, val α 3: A 0.404 (step 908),
4-layer A 0.357 (step 488), U 0.546 (step 2,106), C 0.505 (step 2,839), B pre-step 0.283 (step 552).

## Summary

- **The head is short of training, not of inputs.** Head B's after-miss window (newest token as its embedding only,
  every older token at full depth) is exactly the input set of EAGLE's first draft step, whose top-1 agreement is 0.67 on
  Mixtral and 0.74–0.79 on dense models at temperature 0 with one layer. Converted to sampling Flint at temperature 1
  (its top token is the sample 86–90 % of the time), an EAGLE-grade head would score ~0.60 overall there (est.); B's
  after-miss window scores 0.466 overall on Q51 at 8.6 M tokens. EAGLE's official script runs 68k dialogues for 20
  epochs and SPD trains on ~1.2 M samples: est. ~10⁹ token-passes each, against 2.5 × 10⁷ for head A at 8.6 M and
  ~5.7 × 10⁷ tonight.
- **Most of the shortfall sits on tokens Flint finds obvious.** On Q51's fresh text, 55.5 % of positions have Flint's top
  token at ≥ 0.99 probability, and the staircase gets only 0.587 of them. Of its 0.537 shortfall, 0.138 is Flint's own
  uncertainty (the ceiling for any guess near Flint's top token), 0.229 sits on those near-certain positions and 0.17 on
  the uncertain rest. The near-certain part is knowledge that only capacity and data supply; loss and calibration
  changes act on the 0.17.
- **The logs say data-limited, with the step budget mattering for the slow heads.** A at 3.08 M overfit (val flat from
  90 minutes, val KL rising from 2.58 to 2.67 while training KL fell to 0.91); A at 8.6 M was still rising at the cap but
  flattening (val KL 2.01 → 1.95 → 1.94 over the last 30 minutes, training KL 1.20 against 1.94), so a longer run would
  have added ~0.01 (est.). B was step-limited (1.9 epochs, still rising by ~0.008 per 15 minutes). The per-step price
  differs a lot: A trains 23–25 steps per minute on an RTX 3090, U 52 and C 70 on an RTX 3080, because A computes five
  window rows and two output heads per target.
- **Feature depth: the newest token's depth matters little at the shallow end, the older tokens' depth matters more,
  and neither replaces the 40 layers the newest token has not passed.** Embedding-only versus L9 for the newest token is
  worth ~0.04 at equal steps; older tokens at full depth instead of L9 ~0.04–0.06 at 40 minutes (U against C); the
  staircase (the three tokens before the newest at L17/L33/L41) costs 0.04 against full depth; an untrained head fed the
  newest token's final state reproduces Flint at 0.87. Published early-exit drafters at 6–25 % depth sit near 0.5 per
  token unless the target itself is retrained for early exit (LayerSkip, 0.55–0.78 at 18–38 % depth, greedy).
- **0.55 on fresh text is plausible but sits at the top of what these inputs give.** Two estimates agree: an EAGLE-grade
  head puts the staircase at ~0.55 fresh / ~0.60 overall (est.), and the measured slope (+0.03 per doubling of data)
  reaches 0.55 at ~60–70 M tokens with today's recipe. Recipe changes worth +0.03–0.05 together would bring that to
  ~30–40 M tokens. Tomorrow midday's central estimate is 0.52–0.53 fresh; 0.55 needs one to two more days of generation
  and training. Read copy-first, as the executor runs (copy lookup inside copies, 0.90 there in Q96), overall acceptance
  is already ~0.58 at 0.463 fresh and reaches 0.60 at ~0.49 fresh (est.), so fresh text is the binding gate.
- **Ranked changes (section 4):** (1) training throughput and warm starts, 1.5–2× more steps per GPU-hour plus robustness
  to pipeline timing; (2) fusing every anchor a token has reached (SPD's design, EAGLE-3's +14 % acceptance length),
  est. +0.01–0.03; (3) an acceptance-matched hybrid loss (LK losses: +3.8–8.2 % acceptance length for EAGLE-3 at
  temperature 1, most on MoE targets), est. 0–0.02 on top of the α sharpening; (4) feature noise and dropout against the
  overfitting, est. 0–0.015; (5) a 4-layer head, est. +0.01–0.02 at 1.85× training and ~1.6× guess time, decided by
  tonight's run; then data mix. Not worth it now: a larger vocabulary (40k already holds 0.971 of Flint's kept probability
  on fresh text), training-time test and on-policy drafter data (no self-predicted features or drafted prefixes in a
  one-guess pipeline), softer distillation temperatures.
- **Test plan (section 5):** before 08:30, seven small changes to `train.py`/`drafter.py`; 08:30–11:00, five 150-minute
  fine-tunes from tonight's checkpoints on one fixed file list (A control, A + hybrid loss, A + fused anchors on the 3090s;
  U control and U + noise/dropout on the 3080s), scored on Q51 with paired intervals; 11:30–16:30, the combined recipe on
  all data plus the 4-layer or depth-curve run.

## 1. What the logs and scores show

### 1.1 Steps or data?

| Run | What the curve does at the end | Reading |
|---|---|---|
| A, 3.08 M, 7 epochs | val coupled 0.424 → 0.428 over the last 30 minutes; val KL 2.58 (70 min) → 2.67; training KL 0.91 | overfit: data-limited |
| A, 8.6 M, 3.3 epochs | val α 3: 0.499, 0.513, 0.524, 0.527 at 75/90/105/120 minutes, partly the cosine decay; training KL 1.20 against val 1.94 (staircase) and 1.69 (after-miss) | close to converged for its data; ~+0.01 left (est.) |
| B, 8.6 M, 1.9 epochs | val α 3 +0.008 per 15 minutes at the cap; training KL ≈ val KL | step-limited: 0.380 understates B |
| T, 3.08 M, 34 epochs | val KL 3.98 at 20 minutes, 6.50 at the end, while coupled crept up | heavily overfit |
| Tonight's U and C | 9 and 6 epochs of 13.9 M in their budgets | will saturate on their data; the best-checkpoint rule protects them |

Tonight's A (300 minutes, ~6,900 steps, ~4 epochs of 13.9 M) is reasonably matched to its data. Repeating data beyond
~4 epochs adds little in language-model training ([Muennighoff et al.](https://arxiv.org/abs/2305.16264): up to 4
epochs of repeated data change the loss negligibly against unique data, then the value of repeats falls off), which is
what A at 3.08 M shows. The consequence for the morning: each new data size needs proportionally more steps, and A's
price per step makes that the binding cost.

| Head (GPU) | Steps per minute | GPU-hours per epoch at 14 M / 30 M / 60 M tokens (est.) |
|---|---|---|
| A (RTX 3090) | 23–25 | 1.2 / 2.6 / 5.2 |
| 4-layer A (RTX 3090) | 12 | 2.3 / 5.0 / 10 |
| B (RTX 3090) | 14.5 | 2.0 / 4.2 / 8.4 |
| U (RTX 3080) | 52 | 0.55 / 1.2 / 2.3 |
| C (RTX 3080) | 70 | 0.41 / 0.87 / 1.7 |

Per target, A computes four staircase rows and one after-miss row through both blocks plus two 40k output heads (~640 M
multiply-adds forward, est.), U one window row and one head (~240 M), C only the committed row and one head (~160 M).
The window attention in `drafter._window_attn` also scores every chunk of 256 targets against all committed positions
and masks the ones after each target's limit, so part of that work is thrown away.

### 1.2 Where the misses are

Q51 fresh text, head A at 8.6 M, α 3, by how sure Flint is (Flint's kept distribution from the dump's top-20):

| Flint's top-token probability | Share of fresh positions | Staircase | After-miss | Ceiling for a guess of Flint's top token |
|---|---|---|---|---|
| ≥ 0.99 | 55.5 % | 0.587 | 0.638 | 1.00 |
| 0.95–0.99 | 2.0 % | 0.381 | 0.434 | 0.96 |
| 0.8–0.95 | 13.9 % | 0.365 | 0.397 | 0.88 |
| 0.5–0.8 | 20.5 % | 0.299 | 0.325 | 0.65 |
| < 0.5 | 8.1 % | 0.217 | 0.235 | 0.40 |
| All fresh text | 100 % | 0.463 | 0.504 | 0.862 |

- Where Flint has effectively one candidate, coupled matching reduces to the head's probability for that token, and
  forward KL, cross-entropy and any acceptance loss coincide. Those 0.229 of the shortfall respond only to what the head
  knows: capacity, data and inputs.
- By prompt class (fresh, staircase): chat 0.414, code 0.476, edits 0.508, agent turns 0.502. Reasoning 0.472, answer
  0.441, tool calls 0.725. The first 64 tokens of an answer 0.619, later ~0.46.
- Agent turns are 1,100 of the 7,150 first prompts but only 2.5 % of the generated tokens (295k of 11.6 M), because
  their answers are short.

### 1.3 Calibration and the α sharpening

At α = 1 the head is overconfident on Q51's fresh text (mean top probability 0.542 against a top-token hit rate of
0.454), yet sharpening raises coupled acceptance from 0.404 to 0.463, and on val from 0.469 to 0.527, close to the top
hit rate (0.520). The reason: where Flint is nearly one-hot, the coupled match equals the head's probability on Flint's
token, which is below the hit rate of its top guess for any head unsure between candidates. So acceptance is set almost
entirely by top-1 agreement, and the gain from α 2 to 3 (+0.004 on val) shows the sharpening is used up. A loss that
learns the sharpness per position can add only what a single global α leaves (section 3.2).

### 1.4 Vocabulary

From Flint's kept distributions on Q51 and the vocabulary lists (a guess can only match a token inside the list):

| Vocabulary | Flint's kept probability inside it, all / fresh | Sampled tokens outside it, all / fresh |
|---|---|---|
| 32,000 | 0.955 / 0.960 | 4.5 % / 3.9 % |
| 40,000 (in use) | 0.967 / 0.971 | 3.3 % / 2.9 % |
| 48,000 | +0.006 / +0.006 over 40,000 (same count list) | – |

The 40k list caps acceptance at 0.971 on fresh text; a full vocabulary would lift the cap by 0.029, of which a head at
today's quality would turn est. 0.01–0.015 into matches, for 3.75× the head's bytes (82 M → 307 M parameters). 48k
would add est. 0.002–0.003 for +20 % head bytes. Not worth it before the head improves.

## 2. Published drafters (question 1)

### 2.1 Acceptance and what limits it

| Drafter | Inputs and size | Setting | Acceptance | What limits it | Source |
|---|---|---|---|---|---|
| EAGLE | 1 layer on the final hidden states of all older tokens + the newest token's embedding (the "0-α" step), target's head frozen | Vicuna 7B / Mixtral 8x7B, T 0 | 0-α 0.79 / 0.67 | MoE targets −0.07 to −0.12; drafting further on its own predicted features (n-α falls) | [EAGLE](https://arxiv.org/abs/2401.15077) Tables 2–3 |
| EAGLE training | 68k ShareGPT dialogues, fixed data; lr 3e-5; uniform feature noise; 20 epochs in the official script | – | target-generated instead of fixed data: τ 3.62 → 3.75 (+3.6 %) | "low sensitivity to training data"; plateaus with more data | same, Table 6; [training script](https://github.com/SafeAILab/EAGLE/blob/main/eagle/train/main.py) |
| EAGLE-3 | fuses low, middle and high layers (concatenation + FC), no feature loss, training-time test | LLaMA 3.1 8B | τ 4.05 (EAGLE-2) → 5.37 (no feature loss) → 6.13 (fused features), +14 % from fusion | data: acceptance grows with data once the feature loss is gone; trained on ShareGPT + UltraChat (~530k), target-generated responses | [EAGLE-3](https://arxiv.org/abs/2503.01840) Table 2, Figure 1 |
| SPD (pipeline draft module) | 1–4 layers on a staircase; each token gets every anchor it has reached, concatenated + FC per anchor count | Qwen3.5-4B (32 layers), 8 stages, T 1 | step α 0.271 / 0.328 / 0.340 / 0.344 for 1 / 2 / 3 / 4 layers, p ≈ 0.62 / 0.71 / 0.72 / 0.73 (est., conversion of [pipeline drafter](pipeline_drafter.md) 2.1) | head depth up to 2–3 layers; pre-step drafting −5 % relative; anchor choice ±2 %; no embedding anchor → unstable | [SPD](https://arxiv.org/abs/2605.30852) Table 3, §3.3 |
| SPD training | ~1.2 M samples ≤ 2,048 tokens, one epoch, lr 1e-4 linear decay, KL to target logits, random pipeline fill (full with probability 0.5, else uniform) | – | – | – | same |
| Kangaroo | target's first 2 of 32 layers (6 %) + 1 attention layer adapter (67 M), target's head; 10 epochs ShareGPT | Vicuna 7B, greedy, ≤ 6 drafts | compression 1.41–2.22 (p ≈ 0.5, est.) | shallow features; draft length stopped at confidence 0.6 | [Kangaroo](https://arxiv.org/abs/2404.18911) Table 1 |
| EESD | first 3 of 40 (13B) or 5 of 80 (70B) layers + 1 trained layer initialized from the target's last layer | LLaMA-2 13B, GSM8K | draft acceptance v_d 0.50–0.52; self-distilled data 1.82× → 1.91× | shallow features | [EESD](https://arxiv.org/abs/2406.03853) Tables 4, 10 |
| LayerSkip | early exit at layer E with the target's head, after training the target with layer dropout and early-exit loss | Llama2 7B / 13B, exit at 18–38 % depth (E 7–15), greedy | 0.55–0.78 (CNN-DM 0.69/0.75, XSUM 0.55/0.68, HumanEval 0.67/0.57) | without that training the middle layer is useless (Llama2 7B perplexity 1,900 at layer 16 against 4.3) | [LayerSkip](https://arxiv.org/abs/2404.16710) Tables 1, 3–6 |
| Draft & Verify | the target with layers skipped | LLaMA-2 13B, greedy | best skip set 0.919; skipping the first layers 0.091, middle 0.393, last 0.508 | early layers are essential, late layers redundant | [Draft & Verify](https://arxiv.org/abs/2309.08168) Table 7 |
| PPSD | exit after 20–25 % of layers + 1 transformer layer | Vicuna, LLaMA-2, temperature not stated | 0.67–0.85 (SPD's re-run of first-stage-only drafting at T 1 implies 0.35–0.37) | – | [PPSD](https://arxiv.org/abs/2509.19368) |
| Hydra++ | sequential draft heads with 4-layer MLPs, an extra prefix-attention layer, teacher distillation, 10 epochs | Vicuna 7–33B | +27–31 % throughput over Medusa; prefix attention 1.12× acceptance length; distillation 1.04× throughput; input noise hurt | – | [Hydra](https://arxiv.org/abs/2402.05109) §5–6, A.1.1 |
| Medusa / Clover | heads on the final state; Clover adds an attention "augmenting block" | Vicuna; Baichuan | Clover's block +4.9 / +9.0 / +9.4 % top-5 accuracy per head, mostly from its attention | heads beyond the next token have no feature of the token before | [Medusa](https://arxiv.org/abs/2401.10774), [Clover](https://arxiv.org/abs/2405.00263) |
| HASS / GRIFFIN | EAGLE-2 + top-K distillation loss and multi-step context alignment; GRIFFIN adds token-feature fusion and masks misaligned positions | LLaMA2-Chat 7B, T 1 | top-K loss τ 4.84 → 4.92 (+1.7 %); alignment 4.57 → 4.92; GRIFFIN 4.65 → 5.23 over EAGLE-2 | gains mostly in later draft steps | [HASS](https://arxiv.org/abs/2408.15766), [GRIFFIN](https://arxiv.org/abs/2502.11018) |
| Scylla | EAGLE-like with 1–5 decoder layers, 1–100 B tokens of pretraining | Vicuna, T 0 | τ = 0.74 · log10(layers) + 4.61; pretraining 5.13 → 5.43, saturating past 50 B tokens; 8× SFT data +0.3–0.5 τ | log-linear returns on depth and data | [Scaling laws for speculative decoding](https://arxiv.org/abs/2505.07858) |

### 2.2 What feature depth costs

- **The newest token.** In this project's runs the newest token at L9 instead of its embedding is worth ~0.04 at equal
  steps (A's after-miss window against B's at ~1,750 steps: 0.53 against 0.48 val α 3). Flint's logit lens at layer 41
  scores 0.41 before training, the final state 0.87. Next-token information appears late in a frozen model: the logit
  lens gives no interpretable prediction before layer 21 of 32 in GPT-Neo-2.7B
  ([tuned lens](https://arxiv.org/abs/2303.08112)), the middle layer of Llama2 7B has perplexity 1,900, and Draft &
  Verify's acceptance collapses when early layers are skipped but not late ones. The first 9 of 49 layers are a useful frozen encoder for the head, not a partial answer.
- **The older tokens.** Full depth instead of L9 for every older token is worth ~0.04–0.06 at 40 minutes (U 0.546 at
  2,106 steps against C 0.505 at 2,839), the staircase loses 0.04 of that (Q51: 0.463 against 0.504 fresh, uniform
  across confidence bins), and the token just before the newest (at L17) carries most of the staleness. EAGLE's 0.67–0.79
  comes from full-depth older tokens alone, which shows where the information is.
- **Nine cards.** Card 1 then holds 6 layers. The nine-card windows in `drafter.py` map card 1's output to the embedding
  because layer 6 has no anchor slot (it is dumped for every position); est. −0.01 to −0.03 against six cards, from the
  embedding-versus-L9 gap.
- **The 40 layers the newest token has not passed cannot be supplied on card 1.** Only the head's capacity and training
  stand in for them, which is why the near-certain positions of section 1.2 dominate the shortfall.

### 2.3 Data, epochs and head size

- **Data.** Published heads train on 10⁸–10⁹ tokens or token-passes; this project's slope is +0.03 per doubling
  (0.422 → 0.463 fresh from 3.08 to 8.6 M). EAGLE-3 reports acceptance rising with data once the feature loss is gone;
  Scylla finds log-linear, saturating returns. Target-generated data is worth +3.6 % (EAGLE) to +5 % (EESD) over fixed
  data; this project's data is already Flint's own output.
- **Head size.** SPD: 1 → 2 layers +0.09 per token, 2 → 4 layers +0.02 (est. conversion). Scylla: +0.22 τ per doubling
  of layers (~+5 % at τ 4.6). Hydra: MLP heads saturate at 4 layers. Tonight's 4-layer A leads the 2-layer A by ~0.01 at
  equal steps at 40 minutes but is 1.85× slower per step. The staircase head does more of the target's work than EAGLE
  (the newest token's 40 layers), so depth may matter more here than in EAGLE (est.).

### 2.4 Is 0.55 on fresh text plausible?

- EAGLE-grade (est.): EAGLE's first-step agreement on an MoE target (0.67) times the probability that Flint samples its
  top token (0.86–0.90), plus a little for matches on other tokens, gives ~0.60 overall for B's after-miss inputs; L9 for
  the newest token adds ~0.04; the staircase subtracts ~0.04; fresh text runs ~0.04–0.05 below overall. Staircase
  ≈ 0.55–0.56 fresh, ≈ 0.60 overall.
- Data slope (est.): 0.463 at 8.6 M + 0.03 per doubling reaches 0.55 at ~65 M tokens.
- Both say the gate sits where published heads land with these inputs, reachable with EAGLE/SPD-scale training plus
  recipe gains, not with today's 14 M and today's recipe.

| Step | Q51 fresh (est.) | Basis |
|---|---|---|
| A at 8.6 M (measured) | 0.463 | – |
| A at 13.9 M (tonight) | 0.47–0.49 | 0.7 doublings × 0.03 |
| warm-started on ~30 M | 0.49–0.51 | +1.1 doublings |
| + fused anchors, hybrid loss, noise/dropout | 0.51–0.55 | sections 3.2–3.4 |
| + 4-layer head or ~60 M tokens | 0.53–0.57 | sections 2.3, 3.5 |

## 3. Training-recipe changes (question 2)

### 3.1 Training scale: steps, warm starts, throughput

- **Warm-start every new run from tonight's checkpoints.** Each from-scratch run re-spends 5 GPU-hours to relearn what
  A-14m knows. The same file order with new files appended keeps tonight's validation split held out (the split is
  every 33rd record of the concatenated list).
- **`--resume` cannot load an optimizer into a changed model.** Fused anchors add parameters, so the morning needs an
  `--init-from` path: weights only (`strict=False`), new parameters at zero so the warm-started function is unchanged,
  a fresh AdamW with 100 warm-up steps and its own cosine.
- **One window per sequence instead of all of them.** Training each sequence on one window drawn at random (the nominal
  staircase, the after-miss window, or a random staircase whose depths do not decrease with age) cuts A's rows per target
  from 5 to ~3 and its output heads from 2 to 1: est. 1.5× more steps per GPU-hour. It is also SPD's random pipeline
  fill, which makes the head serve any staircase: today's A scores 0.06 on windows it never saw (pre-step, nine-card),
  so a hop that arrives late on the rig would hand it an input it cannot read.
- **Skip the fully masked keys.** Targets are processed in order, so each chunk of 256 targets needs committed keys only
  up to its last target's limit: est. 30–45 % less window-attention work, an equivalence check of one batch.
- **Epochs.** Budget ≤ 4 epochs per data size and size the run by section 1.1's table. Peak learning rate 2e-4 from
  scratch; 1e-4 for warm-started fine-tunes (SPD 1e-4, EAGLE-3 5e-5, EAGLE 3e-5).
- **Data-parallel training** over two or three RTX 3090s (no peer-to-peer; ~0.37 GB of BF16 gradients per step through
  host memory, est. 0.2–0.7 s against ~2.5 s steps) is a wall-clock tool for the production run, not a GPU-hour saving.

### 3.2 Loss

- **Hybrid acceptance loss.** [LK losses](https://arxiv.org/abs/2602.23881) train drafters with λ·KL(p‖q) + (1 − λ)·TV(p,
  q), λ = exp(−η·sg[α]) per position, η = 3: KL leads while acceptance is low, the acceptance term takes over as it rises.
  Against KL at temperature 1: EAGLE-3 +3.8 % (gpt-oss-20b), +7.7 % (gpt-oss-120b), +8.2 % (Qwen3-235B-A22B) acceptance
  length; MLP speculator +8.3 %, Medusa +7.8 %; pure TV is far worse; truncated vocabularies need no renormalization.
  Here the acceptance term is the coupled match itself (`drafter.coupled_and_top` at α = 1 is differentiable), so the head
  learns its own sharpness per position instead of one global α.
- **Expected gain, est. 0–0.02 (central +0.01).** Section 1.2 bounds it: on the 55.5 % of near-one-hot positions the
  coupled term equals cross-entropy, so the loss acts only on the uncertain positions' 0.17, and the α sharpening already
  takes the global part (+0.06 on val).
- **Top-K and divergence choice.** HASS's top-K distillation term added +1.7 % acceptance length at temperature 1;
  DistillSpec found the best divergence task-dependent, reverse KL with target data best at temperature 1. The hybrid
  covers the same ground with one switch.
- **Temperature.** Softening the targets (distillation temperature > 1) goes against the acceptance rule, which rewards
  a q sharper than the head's belief (section 1.3). One related option: KL against Flint's top-20 before top-p (more
  information about Flint's second choices), with the acceptance term on the kept distribution.

### 3.3 Inputs: using the staircase better

- **Fuse every anchor a token has reached.** Each row today gets one projected state, the deepest. SPD gives each token
  all anchors it has reached; EAGLE-3 fuses low, middle and high layers (+14 % acceptance length over its top-layer-only
  variant). Layer inputs are residual-stream sums, so L17 alone hides how much of it layers 9–16 added; separate
  projections of L9 and L17 recover that. Window rows get every six-card anchor at or below their depth (the row at L33:
  L9, L17, L25, L33); committed rows get L9 next to the final state (both stored for every position). Mid-layer anchors
  for committed rows exist only for answer positions in the dump, so adding them would need the same rule at inference.
  Cost +10–20 % per step (est.), and the head on card 1 already receives each card's boundary states. Est. +0.01–0.03,
  bounded by the 0.04 staircase gap and the 0.04–0.06 older-depth gap.
- **Shifted pairing (round 2).** EAGLE's largest input gain came from pairing each feature with the next token
  (resolving sampling randomness), and GRIFFIN's token-feature fusion collapses without the feature. Adding the previous
  token's deepest state to each row puts "Flint's view of the context up to t − 1" next to token t's embedding without
  needing an attention head to fetch it. Est. 0–0.02.
- **Randomized staircases** (section 3.1) are needed before the rig, whatever else wins. With an L6 anchor slot they
  also let one head serve the nine-card splits.

### 3.4 Regularization against the overfitting

- The 183 M-parameter head memorizes: training KL 1.20 against val 1.94 at 3.3 epochs, overconfidence on Q51, A at
  3.08 M overfit from epoch 5.
- **Feature noise.** EAGLE adds uniform noise U(−0.1, 0.1) to the features; Hydra++ found noise hurt its heads. For the
  staircase head, noise after each anchor's RMS norm (unit scale) at U(−0.2, 0.2) also covers the step from Q8_0 features
  in training to the release file's features on the rig (FP8 E4M3 storage adds ~2–4 % relative error per value; G3's
  drift is unmeasured).
- **Dropout 0.1** after attention and FFN.
- **A frozen output head** (Flint's rows, as EAGLE and Kangaroo use the target's head) removes 82 M of the trainable
  parameters and ~15 % of the step (est.); sign unknown.
- Est. 0–0.015, mixed evidence; tested cheaply on U (section 5).

### 3.5 Head size

A 4-layer head: est. +0.01–0.02 (SPD, Scylla, tonight's per-step lead), 1.85× training per step, ~1.6× guess time on a
P106 (est. 2.6–4 ms instead of 1.6–2.6 ms), which fits a spare card's 6.28 ms cycle on six cards but not card 1's slack.
In a data-limited regime a larger head also overfits sooner. Decide on tonight's 300-minute A against A4 at equal
wall-clock; if C comes within 0.01 of A's staircase, a deeper C (no cross-card features, 4× cheaper steps) is the
better place to spend depth.

### 3.6 Data

- **More of the same** stays the steady lever (+0.03 per doubling, ~100 GPU-hours of generation and dump per doubling at
  30 → 60 M, est.).
- **Mix.** Agent turns are 2.5 % of the generated tokens; chat is the weakest class (0.414 fresh) although it is the
  largest (3.9 M tokens), so it is hard rather than scarce. More agent-turn prompts in the generation queue cost nothing
  in training, est. 0–0.01.
- **On-policy data** (DistillSpec, 10–45 % speed-up) helps drafters whose own guesses become the prefix of the next
  guess. In this executor every guess is conditioned on Flint's own committed tokens or on guesses that are kept only
  when they equal Flint's samples, so Flint's own samples are already the on-policy data. Training-time test (EAGLE-3,
  HASS's alignment, GRIFFIN's masking) fixes self-predicted features, which the staircase does not use: its in-flight
  rows are Flint's real partial states.

## 4. Ranked changes (question 3)

Expected gains are absolute acceptance on Q51 fresh text (α-sharpened) over a control trained on the same data; "free"
means no recurring cost once tested.

| Rank | Change | Expected gain (est.) | Recurring training cost | Guess cost on a P106 | Test cost | Evidence |
|---|---|---|---|---|---|---|
| 1 | Throughput: one window per sequence (with random staircases), masked keys skipped, warm starts with `--init-from` | 0 directly; 1.5–2× steps per GPU-hour, worth +0.01–0.02 at a fixed morning budget; robustness to timing | −35 % | none | 0.2 GPU-h equivalence check | section 1.1; SPD's random fill |
| 2 | Fuse every reached anchor per row | +0.02 (0.01–0.03) | +10–20 % | +~0.1 ms (est.) | 2.5 GPU-h | EAGLE-3 Table 2; SPD |
| 3 | Hybrid acceptance loss (KL → coupled match, η 3) | +0.01 (0–0.02) | ~0 | none | 2.5 GPU-h | LK losses; section 1.3 |
| 4 | Feature noise + dropout (+ frozen head as a variant) | +0.005 (0–0.015) | ~0 | none | 2.5 GPU-h on an RTX 3080 (U proxy) | EAGLE; train/val gap |
| 5 | 4-layer head | +0.015 (0.01–0.02) | +85 % | ~+60 % | tonight's run | SPD Table 3; Scylla |
| 6 | Shifted pairing (previous token's deepest state in each row) | +0.01 (0–0.02) | +5 % | small | 2.5 GPU-h | EAGLE input ablation; GRIFFIN |
| 7 | More agent-turn prompts in generation | 0–0.01 | 0 | none | none | section 1.2 |
| 8 | Data doubling (30 → 60 M) | +0.03 | ~100 GPU-h generation and dump, training ×2 | none | runs anyway | section 2.3 |
| – | Vocabulary 48k / full | +0.002–0.003 / +0.01–0.015 | +5 % / +70 % | +20 % / +275 % head bytes | – | section 1.4 |
| – | Training-time test, on-policy drafter data, distillation temperature > 1 | ~0 here | – | – | – | section 3.6 |

## 5. Test plan for the morning of 2026-10-25

### 5.1 Inputs from tonight's scoring (~02:45)

| Comparison on Q51 fresh | Decision |
|---|---|
| 4-layer A ≥ A + 0.015 at the same 300 minutes | round 2's production line is 4 layers |
| C ≥ A staircase − 0.01 | open a C line (no cross-card features, 3–4× cheaper steps); a deeper C in round 2 |
| U ≈ A after-miss | confirms A's after-miss row needs no separate head |
| A at 13.9 M against 8.6 M | updates the data slope and section 2.4's table |

### 5.2 Code changes before 08:30 (laptop edits, checks on the quant machine)

1. `--init-from PATH` (weights only, new parameters zero-initialized, fresh AdamW, own warm-up and cosine).
2. `--window-sample P_STAIR,P_MISS,P_RANDOM`: one window per sequence. A random staircase puts the newest token at L9
   and each older token at a depth no lower than the newer one's, drawn from L17, L25, L33, L41 and the final state.
   `evaluate` keeps scoring every window.
3. `_window_attn`: slice the committed keys and values to the chunk's last limit + 1. Check: identical loss on one batch
   with and without the slice.
4. `--loss lk --eta 3`: per window c = `coupled_and_top(logits, top_draft, p, alpha=1)[0]`, λ = exp(−η·c.detach()),
   loss = mean(λ·KL + (1 − λ)·(1 − c)) + 0.2·CE.
5. `--fuse`: extra zero-initialized projections (one per anchor type, own RMS norm) for every reached anchor below a
   row's deepest; committed rows L9 + final state.
6. `--noise A` (uniform U(−A, A) after each anchor's RMS norm, training only) and `--dropout P`.
7. `--val-seqs 300` for the final evaluation (60 today) and the mean loader wait per step in the log.

### 5.3 Round 1, 08:30–11:00: one factor each, paired

All runs: the same `--feats` list (tonight's four packs plus the overnight pack as it stands at 08:00, appended in that
order; a run reads the complete records present when it opens the file),
`--minutes 150 --lr 1e-4 --warmup 100 --eval-every 20`, from tonight's `head.pt`. A runs add
`--window-sample 0.6,0.25,0.15` and the key slicing.

| GPU | Run | From | Change | Question |
|---|---|---|---|---|
| RTX 3090 #1 | A-ctrl | A-14m | – | control; what the new data adds; does window sampling keep the staircase at A-14m's level |
| RTX 3090 #2 | A-lk | A-14m | `--loss lk --eta 3` | rank 3 |
| RTX 3090 #3 | A-fuse | A-14m | `--fuse` | rank 2 |
| RTX 3080 #1 | U-ctrl | U-14m | – | proxy control |
| RTX 3080 #2 | U-reg | U-14m | `--noise 0.2 --dropout 0.1` | rank 4 |

Scoring, 11:00–11:30: `score.py` on Q51 at α 1 and each head's best validation α; then the per-position difference
against the control from the `scores-*.npz` files with 95 % intervals from resampling whole transcripts (paired, so far
tighter than the ±0.024 of a single head). Adopt a change at ≥ +0.005 on Q51 fresh with the paired interval above zero,
or ≥ +0.005 on both the 300-sequence val and Q51. If A-ctrl's staircase falls below A-14m's, train production on all
windows again.

If the overnight dump needs one of these GPUs, drop the U pair first. The dump writes its superset and its six-card pack
to the same HDD the morning runs read; three A runs read ~45 MB/s each in random sequence order (est.), so check the
loader wait from item 7 and move the overnight pack to the other bulk disk if the GPUs starve.

### 5.4 Round 2, 11:30–16:30

| GPU | Run |
|---|---|
| RTX 3090 #1 | production: every adopted change, from the best round-1 head, all data dumped by 11:30 (~30 M), 300 minutes (data-parallel over two 3090s if implemented) |
| RTX 3090 #2 | if A4 won tonight: A4 with the adopted changes from A4-14m; else shifted pairing from A-ctrl (rank 6) |
| RTX 3090 #3 | depth curve: a U-type head with after-miss windows whose newest token sits at L9, L17, L25 or L33 (older tokens full), 150 minutes; it says whether a deeper card 1, or waiting for card 2, could ever pay for its longer cycle; else the randomized-staircase head with an L6 anchor for nine cards |
| RTX 3080 #1–2 | the C line with the adopted changes if C qualified tonight; else U at peak lr 3e-4 against 1e-4 |

Score production at ~16:45 on Q51 and through `sim.py` (copy-first speed per workload). The evening's retrain on the
night's data starts from the production head.

## What this changes in the plan

- Q20's next runs are warm-started fine-tunes on a growing file list, not fresh 300-minute runs per data size.
- The scaling target moves from "more tokens" to "more tokens and steps in proportion": ~2.6 GPU-hours per epoch of
  30 M tokens for head A today, ~1.7 after the throughput changes (est.).
- The head must train on random staircases before any rig run, because untrained windows score ~0.06.
- The gate's overall figure is best read copy-first from `sim.py`; fresh text is the binding gate.
- Agent-turn prompts deserve a larger share of the generation queue.

## Not verified

- Tonight's 40-minute comparisons (U against C, 4-layer A against A) are early and at different learning-rate phases;
  the 02:45 scores replace them.
- The EAGLE-grade conversion (section 2.4) assumes EAGLE's temperature-0 agreement transfers to Flint's near-greedy
  sampler and to coupled matching; MoE and reasoning-text effects are judgement.
- The token-pass counts for EAGLE and SPD assume ~1,000 tokens per dialogue or sample; neither paper gives token counts.
- SPD's per-token values come from its step acceptance under an assumed refill of n steps.
- Every expected gain in section 4 is an estimate; fusion's evidence is from multi-step drafters, LK's from rejection
  sampling rather than coupled matching.
- The copy-first overall figure uses Q96's 0.90 inside copies with the same ≥ 8-token copy definition.
- Throughput gains of section 3.1 are arithmetic on the logged step rates, not measurements.
- Whether the head keeps its acceptance on the release file's features instead of Q8_0's.

## Sources

Pipeline and early-exit drafters
- Y. Yu et al., "Speculative Pipeline Decoding" (SPD), [arXiv 2605.30852](https://arxiv.org/abs/2605.30852): anchors and
  fusion (Eqs. 3–4), Table 3 (draft layers), training recipe and random pipeline fill (P)
- Li et al., "Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding" (PPSD),
  [arXiv 2509.19368](https://arxiv.org/abs/2509.19368) (P)
- F. Liu et al., "Kangaroo: Lossless Self-Speculative Decoding via Double Early Exiting",
  [arXiv 2404.18911](https://arxiv.org/abs/2404.18911): adapter, exit layer, Table 1 (P)
- M. Elhoushi et al., "LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding",
  [arXiv 2404.16710](https://arxiv.org/abs/2404.16710): token acceptance by exit layer, middle-layer perplexity (P)
- J. Zhang et al., "Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding",
  [arXiv 2309.08168](https://arxiv.org/abs/2309.08168): Table 7 (which layers to skip) (P)
- J. Liu, Q. Wang, J. Wang, X. Cai, "Speculative Decoding via Early-exiting for Faster LLM Inference with Thompson Sampling Control
  Mechanism" (EESD), [arXiv 2406.03853](https://arxiv.org/abs/2406.03853): Tables 4, 9, 10 (P)

Feature-level heads
- Y. Li et al., "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty",
  [arXiv 2401.15077](https://arxiv.org/abs/2401.15077): Tables 2, 3, 6, feature noise, training data; official
  [training script](https://github.com/SafeAILab/EAGLE/blob/main/eagle/train/main.py) (epochs, noise, learning rate) (P)
- Y. Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test",
  [arXiv 2503.01840](https://arxiv.org/abs/2503.01840): Table 2 (fusion ablation), Figure 1 (data scaling), training data (P)
- Z. Ankner et al., "Hydra: Sequentially-Dependent Draft Heads for Medusa Decoding",
  [arXiv 2402.05109](https://arxiv.org/abs/2402.05109): Hydra++ recipe, appendix A.1.1 (noise, distillation) (P)
- T. Cai et al., "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads",
  [arXiv 2401.10774](https://arxiv.org/abs/2401.10774) (P)
- B. Xiao et al., "Clover: Regressive Lightweight Speculative Decoding with Sequential Knowledge",
  [arXiv 2405.00263](https://arxiv.org/abs/2405.00263): augmenting-block ablation (P)
- L. Zhang et al., "Learning Harmonized Representations for Speculative Sampling" (HASS),
  [arXiv 2408.15766](https://arxiv.org/abs/2408.15766): top-K loss, context alignment, Tables 1, 2, 4, 7 (P)
- S. Hu et al., "GRIFFIN: Effective Token Alignment for Faster Speculative Decoding",
  [arXiv 2502.11018](https://arxiv.org/abs/2502.11018): token-alignable training and fusion ablations (P)
- S. Yan et al., "Scaling Laws for Speculative Decoding" (Scylla), [arXiv 2505.07858](https://arxiv.org/abs/2505.07858): draft depth
  and data laws (P)

Losses, distillation and data
- A. Samarin et al., "LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding",
  [arXiv 2602.23881](https://arxiv.org/abs/2602.23881): hybrid loss, η, results by drafter and target; implementation in
  [SpecForge PR #492](https://github.com/sgl-project/SpecForge/pull/492) (P)
- Y. Zhou et al., "DistillSpec: Improving Speculative Decoding via Knowledge Distillation",
  [arXiv 2310.08461](https://arxiv.org/abs/2310.08461): divergences and data sources (P)
- N. Muennighoff et al., "Scaling Data-Constrained Language Models", [arXiv 2305.16264](https://arxiv.org/abs/2305.16264):
  repeated epochs (P)
- N. Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens",
  [arXiv 2303.08112](https://arxiv.org/abs/2303.08112): logit lens by depth (P)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding",
  [arXiv 2408.07978](https://arxiv.org/abs/2408.07978) (P)

This project
- [Pipeline drafter](pipeline_drafter.md) (design, SPD conversion, the gate), [pipelined speculation](pipelined_speculation.md),
  [bench/staircase_drafter](../../../bench/staircase_drafter/README.md) (`drafter.py`, `train.py`, `score.py`); training
  logs and Q51 scores of heads A, B, T and tonight's A, 4-layer A, B, U and C on the quant machine (2026-10-24); the
  per-confidence, calibration and vocabulary breakdowns of sections 1.2–1.4 computed from head A's 8.6 M Q51 score
  files, Q51's transcripts, the dumped top-20 log-probabilities and the vocabulary lists (P)
