# A head that guesses far better: what limits Flint's MTP head, and what to try (2026-10-28)

Scope: Flint's MTP head (the one-token staircase head of [docs/drafter.md](../../drafter.md), method M4, registry Q20 and
Q169) is right ~25 % of the time at the positions it serves and makes the release line 3.2 % slower (Q193). This note
asks what limits its acceptance, quantified from the project's own data where the data allows it, what the 2023–2026
literature says about each limit, and which experiments on the existing dumps and sessions should come next, ranked by
expected gain × feasibility on the quant machine and the P106 rig. Nothing was run for it: the numbers come from the
data folders, the registry and the papers. Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- **Acceptance** is the coupled match probability between Flint's sample (temperature 1, top-k 20, top-p 0.95) and the
  head's guess, given a correct prefix ([drafter.md section 5](../../drafter.md#5-coupling-with-the-sampler)).
- **Held-out** is the 40 sessions of the 7 held-out modules of agentic run 1 (680,600 scored positions), at nine cards
  (window g9: the newest token at L6, then L12, L22, L33, L37, L44, older tokens complete), α 1, unless a row says
  otherwise. **Validation** is sessions of the training modules. **After-miss** (g9miss) is the first guess after a
  wrong one: the newest token at L6, every older token complete.
- **At-position** is the head's acceptance at the positions it serves: where the server's copy lookup and the request's
  continuations of 2-token or longer contexts do not guess (a 1-token context or nothing: 23.2 % of positions,
  [source_split.md](../../data/2026-10-27_cheap_guess/source_split.md)). **Served** is what llama-server's counters
  report on the agent check's paired replay.
- Ln is the input of Flint layer n (49 layers, hidden 2,048).

## Verdict

- **The head is limited by what it has learned, not by what its inputs hold.** The after-miss window carries the same
  information as EAGLE's first draft step and a DeepSeek-style MTP module: the final states of every older token plus
  the newest token (here also its L6 state). Those reach top-1 agreement 0.67 (EAGLE on Mixtral, T 0) to 0.74–0.79 (dense
  targets), and built-in MTP layers 0.85–0.91 (DeepSeek-V3's second token; Ornith-1.5's own layer on agent text, Q341).
  The project's own M2 heads reached 0.61–0.62 with these inputs on Q51 (old data, 27.6 M tokens, warm-started). M4 gets
  0.247 there at 240 minutes ([scores_heldout.md](../../data/2026-10-27_mtp_head/offline/scores_heldout.md) line 58). An
  EAGLE-grade head on Flint's agent text would sit near 0.55–0.65 (est.: 0.67 × Flint's top-token probability of
  0.83–0.90, plus coupled matches off the top token).
- **The binding gap is generalization to unseen code.** At 1,000 minutes validation (sessions of the training modules)
  is 0.524 and held-out modules 0.366; at 240 minutes 0.365 against 0.277. The gap grew from 0.088 to 0.158: about 55 % of
  what the extra epochs added on seen modules transferred to unseen ones (est.). The head memorizes module content and
  retrieves poorly from context: at copy positions it is right 0.310 of the time where the copy lookup is right 0.815.
- **It is under-trained, and steps still pay:** held-out rose 0.277 → 0.366 from ~1 to ~3.4 epochs, +0.049 per doubling
  of steps (est., two points), more than the +0.03 per doubling of data the old heads showed; training KL was ≈ validation
  KL at 240 minutes (2.1–2.2 against 2.44 nats). The 1,480-minute round (~5 epochs) scores 0.380, +0.027 per doubling
  since 1,000 minutes.
- **The head is already a reasoning drafter.** Fresh text is ~85 % reasoning (est. from the answer store's split), and the
  head serves mostly fresh-text positions; yet ~77 % of its training loss sits on positions it never serves.
- **After a miss it is weaker than in a full pipeline,** the reverse of every earlier head: held-out g9miss 0.247 against
  g9 0.277 (M0–M2: +0.04 to +0.07 the other way). With served acceptance near 25 %, most head guesses follow a miss
  (est.), so this window sets much of the served number.
- **What it would take (est.):** each +0.01 of served at-position acceptance is worth ~+0.5 % generation at ≤ 3 ms per
  guess (cost simulator, 0.20 → 0.40). Served ≈ 0.68 × held-out today (0.248 against 0.366). A served 0.30 (+8 % at
  ≤ 3 ms) needs held-out ~0.44; 0.40 (+14 %) needs ~0.59 or a smaller served/offline gap. The cheap training changes,
  ~5 epochs and data run 2 together are est. 0.45–0.52 held-out, so the path to ≥ +5 % needs both them and Q194's
  serving cuts (the wasted launches, the prompt-side commits, the taps' copies).
- **First experiments (offline, days, all on the existing dumps):** after-miss oversampling, served-position weighting,
  tool-result positions as extra targets, and three probes that settle capacity against information against data
  (ranked table below). The dumps (1.43 TB of features and sequences) are needed for every one of them.

## 1. Where the head stands

| | Value | Source |
|---|---|---|
| Head | 2 blocks at hidden 2,048 (16 query and 4 KV heads, FFN 4,096), 29,288 + 1 output ids initialized from Flint's head, 160.8 M parameters | [config_nine.json](../../data/2026-10-27_mtp_head/offline/config_nine.json); `drafter.py` lines 252–300 |
| Training | forward KL to Flint's kept distribution + 0.2 × cross-entropy, lr 2e-4, cosine to 10 % over the run, 8,192 targets a step, no dropout or noise, all generated positions weighted 1, window kinds 0.6 / 0.15 / 0.25 (staircase / spare card / after-miss) | config; `train.py` lines 5, 190; `drafter.py` line 393 |
| Data | 1,108 sessions, 13.7 M generated tokens trained, 73 Python and 19 Elixir modules, 7 held out | [mtp_head README](../../data/2026-10-27_mtp_head/README.md) |
| 240 minutes (~1 epoch) | held-out 0.277 (fresh 0.235); validation 0.365; α 3 0.455 on validation | [final_alpha.tsv](../../data/2026-10-27_mtp_head/offline/final_alpha.tsv) |
| 1,000 minutes (~3.4 epochs, est. from 6,471 items per 1,392 steps) | held-out 0.366 (fresh 0.323); 3–12-card head 0.355 (0.315); Q51 0.218; validation 0.524 at 946 minutes | [drafter.md](../../drafter.md#a-cheaper-guess-2026-10-27) ("On the agent check"); [the head's data folder](../../data/2026-10-27_mtp_head/README.md) |
| At-position, offline | 0.244 at 240 min, 0.314 at 1,000 min (1-token continuation positions) | [source_split.md](../../data/2026-10-27_cheap_guess/source_split.md); drafter.md |
| Served (paired replay) | 24.8 % right (1,761 of 7,104), release line 3.2 % slower | Q193 |
| Cost per guess | 2.93 ms on an idle P106; in the server ~4 ms + ~0.75 ms per token, half following, half guessing, 2 of 3 launches wasted | Q192, Q193, Q194 |

## 2. What limits acceptance

### 2.1 Information at the tap depth

How much of Flint's next-token decision exists by layer 6 of 49? The literature says little in a frozen model: the logit
lens gives no usable prediction before ~2/3 depth in some models (tuned lens paper), early layers rank mostly
high-frequency "statistical guesses" that later layers refine ("How do LLMs use their depth?"), and
skipping early layers collapses a self-draft while skipping late ones does not (Draft & Verify). But the head does not
need L6 to hold the answer. It needs the older tokens' complete states, which already encode Flint's view of the
context and part of what comes next (Future Lens: a single mid-layer state predicts the token after next with > 48 %
accuracy at the best layers of GPT-J), and the newest token's identity.

From the project's own scores:

| Inputs | Held-out (nine-card head, 240 min) | Basis |
|---|---|---|
| Newest token as its embedding only, the staircase behind it (spare card, `g9pre`) | 0.197 | scores_heldout.md line 57 |
| Newest token at L6 (`g9`) | 0.277 | line 51 |
| Card-1 depth across card counts, 3–12-card head: newest at L4 / L6 / L9 / L12 / L17 (12 / 9 / 6 / 4 / 3 cards) | 0.213 / 0.256 / 0.286 / 0.279 / 0.304 | lines 16–24 |
| Newest at L6, every older token complete (`g9miss`) | 0.247 | line 58 |
| Tokens only (M0's head T, Q51) | ~0.30 | drafter.md section 6 |
| An untrained head on the newest token's final state (Flint's own head, Q51) | 0.87 | [drafter acceptance](drafter_acceptance.md) 2.2 |

- Card 1's six layers on the newest token, with one cycle's fresher states of the older tokens in flight, are worth
  +0.08 over the spare card's window; L6 → L9 another ~+0.03 and little more up to L17 (est.; the card counts also differ
  in training weight and tokens in flight).
- The 40 layers the newest token has not passed are the largest single loss of information (0.87 against 0.28), and no
  head on card 1 can get them. Only EAGLE-class inputs are available, and those support far more than 0.28.

**Experiment (Q376, the information probe):** on the dumps' generated positions, (a) a tuned-lens probe per recorded
layer (4, 6, 9, 12, 18, 24, 30, 36, 42, 47, final: an affine map initialized at the identity, then Flint's output norm and
head rows, KL to Flint's top-20, 1 M positions; minutes each), reporting top-1 agreement with Flint's top token and
coupled acceptance on the held-out sessions; (b) an after-miss head per depth (newest at L_k, older complete, warm-started
from the 1,000-minute head, 2 h each). (a) gives Flint's decision by layer, publishable on its own; (b) gives the head's
ceiling per tap depth with context. A flat curve to L24 says depth is not the limit at nine cards; a steep one prices a
deeper card 1 (Q393).

### 2.2 The knowledge gap

- **Flint is sure, the head is wrong.** On Q51 (M0's head A) 55.5 % of fresh positions had Flint's top token at ≥ 0.99,
  and the head got 0.587 of them; 0.229 of its 0.537 shortfall sat there, 0.138 was Flint's own uncertainty, 0.17 the
  rest ([drafter acceptance](drafter_acceptance.md) 1.2). Where Flint is one-hot, every loss reduces to the head's
  probability on Flint's token: knowledge, not calibration. The M4 breakdown is not computed yet (Q385).
- **Seen code against unseen code.** Validation shares modules with training; held-out does not. The gap of 0.158 at
  1,000 minutes is module-specific knowledge (file contents, identifiers, the module's API) that the head memorized and
  that does not exist for new code. On the agent check (Exercism tasks, three of the six replayed traces in JavaScript,
  a language the head never saw) the head is in held-out territory or beyond.
- **Retrieval from context is weak.** At copy positions (52 % of positions) the head is right 0.310 where the copy lookup
  is right 0.815 (source_split.md line 12): the head sees the context through Flint's final states and its own
  attention, yet does not copy. In reasoning about code Flint repeats identifiers and phrases from the files it read;
  a mechanism that copies from context (Q381) attacks exactly the unseen-code gap.

### 2.3 Temperature 1, coupling and the training objective

- **The ceiling for a point guess** is E[p(g)] ≤ E[p_max]: Flint's top token holds 0.90 of the kept probability on
  average and 0.83 on agent turns (Q96, [pipeline drafter](pipeline_drafter.md)). A coupled guess can exceed it: with the
  same Gumbel noise and q = p the match is 1. Coupled acceptance is at least (1 − TV)/(1 + TV) (Daliri et al.), and
  maximal coupling (rejection sampling, which gives up bit-identical output) reaches 1 − TV = Σ min(p, q).
- **Today the head is a point guesser.** At α 3 validation is 0.455 and the top-guess rate 0.454 (final_alpha.tsv line
  7); α 1 gives 0.365. Acceptance equals P(head's top token = Flint's sample). The objective that matches this is the
  coupled match at the served α (a smooth version of "put the argmax on Flint's sample"), not forward KL at α 1, which
  spends capacity on the kept set's tail. With 1.69 kept tokens on average the two differ little: LK losses (a KL-to-match
  hybrid) gave +3.8 / +7.7 / +8.2 % acceptance length at T 1 on gpt-oss-20b / gpt-oss-120b / Qwen3-235B-A22B, and pure TV
  trains badly; acceptance-aware losses (EAL, window TV, a GRPO stage) report consistent gains over KL. Expected here
  +0.005–0.02 (est.; the near-one-hot positions bound it), Q378.
- **The served coupling costs little.** Inverse-CDF (one uniform per position) scored 0.007–0.010 below Gumbel on trained
  heads; on M4 the shared-uniform score is even 0.009 above it at α 1 (0.286 against 0.277, scores_heldout.md line 51).
  Σ min(p, q) is not scored yet (Q385); it bounds what dropping bit-identity could add.
- **Per-position sharpening.** One global α leaves whatever a position-dependent α would add; on M0 α 2 → 3 added +0.004,
  so the global part is used up, and a per-confidence-bin α fitted on validation is worth est. +0.005–0.015 (Q378).

### 2.4 Vocabulary truncation

The 29,288 draft ids cover 97.7 % of the held-out sessions' generated tokens (the first 4,096 / 8,192 / 16,384 ids 84.2 /
90.3 / 95.6 %; drafter.md, "What a guess may cost"). The cap is ~0.977 overall and est. ~0.97 at the head's fresh-text
positions. FR-Spec and FastMTP trim the drafting vocabulary for speed at a small acceptance cost; here the output layer
is 0.47 ms of 2.93 and coverage would fall, so the vocabulary stays.

### 2.5 One token ahead, and staleness

- Every guess is one token ahead of its newest input; the "further ahead" is the staleness of the in-flight rows. On
  M0–M2 complete older tokens beat the staircase by 0.04–0.07; on M4 the after-miss window is 0.03 below it at nine cards
  (0.247 against 0.277; 3–12-card head 0.220 against 0.256), and on validation even the after-miss window with the newest
  token at L9 scores 0.308 against the staircase's 0.365 (final_alpha.tsv lines 7, 9).
  More information scoring lower points at training (M4 started from scratch; M0–M2 descended from the after-miss
  specialist U) and makes Q371 the cheapest fix with a large share of served guesses behind it.
- In the served pipeline the window is the split's staircase whatever the fill: a token that has completed still enters
  at its staircase depth. When the pipeline is not full (most of the time at ~25 % acceptance) that throws information
  away (Q386).
- Drafting further ahead in one pass (block drafting, Q271) pays for amortized cost, not acceptance: P-EAGLE reports an
  NTP–MTP gap of ~25 % and needs 4 layers to make up for the missing intermediate tokens; DFlash and PARD likewise. At a
  first-token acceptance near 0.3 a second position would be right est. 0.15–0.2 of the time.

### 2.6 Training length, data and size

Section 6. In short: under-trained (steps still pay at ~3.4 epochs), data-limited for unseen code (the
validation/held-out gap), size untested (the 4-layer test of M0 was inconclusive).

## 3. Reasoning against tool calls

From the answer store's split of the 40 held-out sessions with the release line's lookups (`cont`), and the head's
split by guess source ([agentic.md](../../data/2026-10-27_answer_store/offline/agentic.md) lines 4–8,
[source_split.md](../../data/2026-10-27_cheap_guess/source_split.md) lines 10–16):

| Segment | Positions | Share | Fresh text share | Lookups right, all / fresh | Pipeline model, depth 3 |
|---|---|---|---|---|---|
| Reasoning | 440,543 | 64.7 % | 68.8 % | 44.4 % / 26.9 % | 1.55× |
| Tool calls | 216,183 | 31.8 % | 16.0 % | 84.3 % / 34.0 % | 2.82× |
| Answer text | 23,874 | 3.5 % | 73.2 % | 39.8 % / 24.4 % | 1.47× |

| What the lookups would guess | Share of positions | Head right (240 min) | Lookup right |
|---|---|---|---|
| copy lookup | 52.1 % | 0.310 | 0.815 |
| continuation, 2-token or longer context | 24.6 % | 0.266 | 0.434 |
| continuation, 1-token context (the head's place now) | 18.8 % | 0.244 | 0.191 |
| nothing (the head's place) | 4.4 % | 0.273 | – |

- **The head is a reasoning drafter in all but training.** Fresh text is 303k reasoning, 35k tool-call and 17k answer
  positions (est., shares × fresh fractions), so ~85 % of the head's served positions are reasoning (est.). Its loss
  weights all generated positions equally: 77 % of it lands on copy and 2-token-continuation positions it never serves.
- **Separate heads per segment** buy little: tool calls hold est. < 10 % of the head's positions, and segment
  specialization (TAPS: specialized drafters with confidence routing beat one generalist across domains) is already
  achieved by weighting where the head serves (Q372). Kept as a parked row (Q392).
- **Conditioning on the segment** (an embedding for `<think>`, tool call, answer) is mostly redundant: Flint's final
  states carry it. One reason to add it anyway: a 9–15k-token reasoning reply pushes `<think>` out of the head's
  4,096-position window. It rides along in Q372 as an arm.
- **Training on the positions where the lookups are wrong** is the direct form of specialization (Q372). The evidence
  both ways: selective distillation (AdaSPEC: drop tokens the draft cannot fit, up to +15 % acceptance over DistillSpec;
  Rho-1's selective token loss for pretraining) and the "flatter tokens" result (half the data, the flattest-target
  samples, keeps EAGLE within 4 % of full-data speed-up) say the loss should go where it changes outcomes. Against: copy
  positions teach in-context retrieval, which the head lacks. So reweight (served positions × 3–4, the rest × 0.3–1),
  don't drop.
- **Serving:** no head launches inside tool-call segments (the lookups are right 84 % there, and two of three launches go
  to waste anyway, Q194), and deeper speculation where the lookups are sure (Q342).

**Experiment (Q385, from existing files):** the held-out score files joined with the store's segment labels, the lookups'
sources and the dumps' top-20: the head's acceptance per segment × source × Flint-confidence bin × window × position in
the session. It replaces every "(est.)" in this section with a measurement and shows whether acceptance decays over long
sessions, as test-time speculation found for EAGLE-3, DFlash and PARD (acceptance length near 1 after ~20k generated
tokens).

## 4. The literature

"Fit" judges a one-token head on card 1 reading partial-depth states, ≤ 3 ms per guess on a P106, coupled sample-and-match,
temperature 1. T 0 and T 1 are the papers' own settings.

### 4.1 Feature-level heads and their training

| Work | Mechanism | Reported gain and conditions | Fit |
|---|---|---|---|
| EAGLE ([2401.15077](https://arxiv.org/abs/2401.15077)) | 1 layer on the target's final states of older tokens + the newest token's embedding; feature regression; target head frozen | 0-α 0.74–0.79 dense, 0.67 Mixtral (T 0); 0.71–0.73 at T 1 (rejection sampling); 68k dialogues, ~20 epochs | Its first step has our after-miss inputs: the reference for what they support. Feature regression gave nothing here (M0) |
| EAGLE-2 ([2406.16858](https://arxiv.org/abs/2406.16858)) | dynamic draft trees from the draft's confidence, which tracks acceptance | 3.05–4.26×, 20–40 % over EAGLE-1 | Confidence ≈ acceptance: the launch gate and adaptive depth (Q342), not trees |
| EAGLE-3 ([2503.01840](https://arxiv.org/abs/2503.01840)) | low, mid and high target layers fused; no feature loss; training-time test (TTT) | τ 4.05 → 5.37 (no feature loss) → 6.13 (fusion, +14 %); acceptance grows with data; DeepSeek-R1-Distill-LLaMA-8B at T 1 τ 6.10 on GSM8K against 4.49 on MT-bench | Fusion: yes, of the card-1 layers (Q380). TTT fixes self-predicted features, which the staircase does not use |
| HASS ([2408.15766](https://arxiv.org/abs/2408.15766)), GRIFFIN ([2502.11018](https://arxiv.org/abs/2502.11018)) | top-K distillation, context alignment; token-feature fusion, masking misaligned positions | +1.7 % τ from the top-K loss (T 1); GRIFFIN 4.65 → 5.23 over EAGLE-2 | Gains are in later draft steps; little for one token |
| Mixture of Attentions ([2410.03804](https://arxiv.org/abs/2410.03804)) | attention across the target's layers per token, cross-attention to target activations, "target layer inference": rerun the target's last N frozen layers | τ 2.14 / 2.25 / 2.54 at N 0 / 1 / 3; +25 % τ over EAGLE-2 | Reusing Flint's last layers as such sits on the last card; as an initialization (Q383) |
| SpecForge / SpecBundle ([2603.18567](https://arxiv.org/abs/2603.18567)) | EAGLE-3 training at scale | target-regenerated data raises τ on nearly all benchmarks; 320k → 1.4M conversations: τ 2.82 → 3.48 on Math500; dense drafts beat every MoE-drafter variant at equal FLOPs; TTT length 3–13 | Data volume and self-generation: yes (already self-generated). An MoE FFN in the head: parked (Q388) |
| P-EAGLE ([2602.01469](https://arxiv.org/abs/2602.01469)) | one pass drafts K positions from a shared hidden state and mask tokens | τ +2–10 % over EAGLE-3 (gpt-oss, Qwen3-Coder); 1 / 2 / 4 layers τ 2.69 / 3.58 / 3.92 (HumanEval); 20 → 40 → 60 epochs +1.5 / +2 % (HumanEval), +3.6 / +4.6 % (MT-Bench) | Layers: 1 → 2 is the large step, 2 → 4 ~+9 %. Epochs keep paying on fixed data long past ours. Block drafting: Q271 |
| DFlash ([2602.06036](https://arxiv.org/abs/2602.06036)) | block drafter: [token, MASK × (d−1)] in one pass, conditioned on 5 target layers injected as K/V | 5 layers; τ ≈ 6.5 (T 0), ≈ 5.5 (T 1) on Qwen3-4B/8B, 4–5× | Block amortization for Q271; the multi-layer K/V injection is a fusion variant |
| PARD ([2504.18583](https://arxiv.org/abs/2504.18583)), Falcon ([2412.12639](https://arxiv.org/abs/2412.12639)), Gumiho ([2503.10135](https://arxiv.org/abs/2503.10135)) | parallel drafts from mask tokens; semi-autoregressive drafting with glancing distillation; serial heads for early positions, parallel MLPs later | PARD 1.78× over AR drafts on average; Gumiho: early tokens matter most, so capacity goes there | Gumiho's argument is ours: all capacity on the one position that matters |
| Medusa ([2401.10774](https://arxiv.org/abs/2401.10774)), Hydra ([2402.05109](https://arxiv.org/abs/2402.05109)), Clover ([2405.00263](https://arxiv.org/abs/2405.00263)), ReDrafter ([2403.09919](https://arxiv.org/abs/2403.09919)) | heads on the final state; sequential heads; an attention augmenting block; an RNN head with beam search | head for the token after next ~0.6 top-1; Hydra++ +27–31 % throughput, MLP heads saturate at 4 layers, input noise hurt; Clover's attention block +4.9–9.4 % top-5 | Final-state heads cannot fill the pipeline. Hydra's MLP depth: an MLP-only block on the output row (Q382) |
| FastDraft ([2411.11055](https://arxiv.org/abs/2411.11055)) | pretrain a small draft, then align on target-generated data | 50M draft for Phi-3-mini: acceptance up to 67 %, 3× memory-bound | Pretraining a token-only draft LM: rejected earlier (no same-tokenizer model, launches) |
| DistillSpec ([2310.08461](https://arxiv.org/abs/2310.08461)), temperature-centric KD ([2410.10141](https://arxiv.org/abs/2410.10141)) | white-box distillation; divergence and data choice; distillation at the decoding temperature | 10–45 % speed-up; best divergence task-dependent; matching the temperature closes the high-temperature gap | Already distilled on Flint's T 1 kept distribution, on Flint's own samples |
| AdaSPEC ([2510.19779](https://arxiv.org/abs/2510.19779)), SFDD ([2601.18902](https://arxiv.org/abs/2601.18902)), Rho-1 ([2404.07965](https://arxiv.org/abs/2404.07965)) | filter tokens the draft cannot fit (reference-model loss); keep the samples with the flattest target distributions; selective token loss | up to +15 % acceptance over DistillSpec; 50 % of the data, > 2× faster training, speed-up within 4 % | Weight the loss where it changes served outcomes (Q372) |
| LK losses ([2602.23881](https://arxiv.org/abs/2602.23881)), acceptance-aware training ([2609.24150](https://arxiv.org/abs/2609.24150)) | λ·KL + (1 − λ)·(1 − match), λ from the current acceptance; expected-accepted-length and window-TV losses, a GRPO stage | +3.8–8.2 % acceptance length at T 1 on MoE targets; pure TV far worse; consistent gains over KL | `--loss lk` exists and was never run on M4 (Q378) |
| Scylla ([2505.07858](https://arxiv.org/abs/2505.07858)), Muennighoff et al. ([2305.16264](https://arxiv.org/abs/2305.16264)) | scaling laws for drafters; repeated data | τ = 0.74 · log10(layers) + 4.61; pretraining 5.13 → 5.43 saturating past 50 B tokens; up to ~4 epochs of repeats ≈ unique data | Log-linear returns: plan in doublings (section 6) |

### 4.2 Built-in MTP and self-speculation with the target's own layers

| Work | Mechanism | Reported gain and conditions | Fit |
|---|---|---|---|
| DeepSeek-V3 ([2412.19437](https://arxiv.org/abs/2412.19437)) | MTP module trained with the model: [final state of token t; embedding of t+1] → t+2 | second-token acceptance 85–90 %, 1.8× tokens per second | Same information set as our after-miss window; trained jointly on trillions of tokens. The upper reference |
| Ornith-1.5's own MTP layer (Q341) | built-in `nextn` layer, measured offline through llama.cpp's MTP graph | first draft 0.876 / 0.908 (35B Q8_0, agent / code, makers' sampling); 0.845 / 0.865 (122B UD-Q2_K_XL) | The same structure on agent text of this project's harness |
| FastMTP ([2509.18362](https://arxiv.org/abs/2509.18362)), self-distillation for MTP ([2603.23911](https://arxiv.org/abs/2603.23911)) | one shared MTP head fine-tuned on self-distilled data | first-draft acceptance 70 → 81 %, 2.03× | Self-distilled data again; Flint ships no MTP layer to start from |
| LayerSkip ([2404.16710](https://arxiv.org/abs/2404.16710)), Kangaroo ([2404.18911](https://arxiv.org/abs/2404.18911)), Draft & Verify ([2309.08168](https://arxiv.org/abs/2309.08168)), EESD ([2406.03853](https://arxiv.org/abs/2406.03853)) | early exit with the target's head (after retraining the target); a shallow exit + adapter; layer skipping; an exit layer initialized from the target's last layer | 0.55–0.78 at 18–38 % depth (greedy, target retrained); Kangaroo ~0.5 (est.); early layers essential, late redundant | The target cannot be retrained; init from Flint's late layers is testable (Q383); layer skipping takes Flint's own cards (rejected) |

### 4.3 Reasoning models and long outputs

| Work | Mechanism | Reported gain | Fit |
|---|---|---|---|
| Test-time speculation ([2605.09329](https://arxiv.org/abs/2605.09329)) | acceptance of EAGLE-3, DFlash and PARD decays with output length (draft cross-entropy on the target's top token 2.8 → 6.4 nats); one online distillation step per round from the target's verify distributions | τ near 1.1 after ~20k tokens; +34–41 % τ on average, up to +72 %, more on long outputs | Our head trains on whole sessions with the served window, so the decay may be smaller; measure (Q385). Online steps on card 1 cost card time: parked (Q389) |
| SpecExtend ([2505.20776](https://arxiv.org/abs/2505.20776)), LongSpec ([2502.17421](https://arxiv.org/abs/2502.17421)) | drafters for long sequences | EAGLE-3 draft accuracy falls past ~2k tokens; drop-in fixes recover it | The head already trains at the served 4,096 window |
| SpecReason ([2504.07891](https://arxiv.org/abs/2504.07891)), Lookahead Reasoning ([2506.19830](https://arxiv.org/abs/2506.19830)) | a small model drafts whole reasoning steps, the target judges or regenerates them | 1.5–2.5×; SD 1.4× → 2.1× | Lossy (semantic acceptance): the output changes. Parked (Q391) |
| TAPS ([2603.27027](https://arxiv.org/abs/2603.27027)) | drafters trained per domain, combined by confidence routing | specialized drafters win on their own workloads; routing beats a merged generalist | Supports weighting toward the served workload (Q372) |
| SuffixDecoding ([2411.04975](https://arxiv.org/abs/2411.04975)) | suffix trees over the model's own past outputs, built for agents | – | The answer store (Q161, +1.4 % measured) |

### 4.4 Pipeline-parallel speculation

| Work | Mechanism | Reported gain | Fit |
|---|---|---|---|
| SPD ([2605.30852](https://arxiv.org/abs/2605.30852)) | a draft module on its own GPU reads a staircase of partial-depth features; pre-step drafting | per-token p ≈ 0.62 / 0.71 / 0.72 / 0.73 for 1–4 layers (est. conversion, Qwen3.5-4B, T 1); first-stage features alone ~0.35 | The design this head follows; 2 layers sit near its knee |
| PPSD ([2509.19368](https://arxiv.org/abs/2509.19368)), PipeInfer ([2407.11798](https://arxiv.org/abs/2407.11798)), FlowSpec ([2507.02620](https://arxiv.org/abs/2507.02620)) | early-exit drafting in stage 1; asynchronous pipelined speculation; pipelined tree speculation | PPSD 2.01–3.81×; FlowSpec 1.37–1.73× over naive pipelines | The executor's prior art. PPSD's closed form prices acceptance on nine cards |
| SpecTr ([2310.15141](https://arxiv.org/abs/2310.15141)), SpecBranch, Saguaro (idle-cards note) | several drafts coupled by optimal transport; hedged branches | – | Top-2 hedging in spare slots (Q276): measure the top-2 hit rate first (Q387) |

## 5. Ideas and experiments, ranked

Gains are absolute held-out acceptance (nine cards, g9, α 1) against a control trained the same number of steps from the
same checkpoint, paired over the 40 sessions, unless a row says otherwise. Every training run is warm-started from the
newest reference head (the 1,000- or 1,480-minute nine-card head) unless it says "from scratch". GPU times are one RTX
3090 at the overnight rate (~4.6 steps of 8,192 targets a minute). "Served" effects come from the cost simulator and a
paired replay on the rig. Rank = expected gain × feasibility.

| # | Registry | Hypothesis | Offline experiment | Metric | Expected effect (est.) | Cost | Kill criterion |
|---|---|---|---|---|---|---|---|
| 1 | Q169 (running), Q370 | Under-trained: steps still pay at ~3.4 epochs | the 1,480-minute round (~5 epochs); future rounds on a warm-up–stable–decay schedule, so an extension does not re-warm (the restart cost ~1 h: 0.365 → 0.325 → 0.360) | held-out at the round's end | +0.02–0.03 (0.38–0.39) | running | < +0.01 over 0.366: stop adding epochs on this data |
| 2 | Q371 | The after-miss window is under-trained (−0.03 against the staircase, the reverse of M0–M2), and it carries most served guesses | window kinds 0.4 / 0.1 / 0.5 instead of 0.6 / 0.15 / 0.25, 4 h, paired with the default kinds | g9miss and g9 held-out; served-window mix from Q385 | g9miss +0.03–0.06; served +0.01–0.03 | 2 × 4 h | g9miss still below g9, or g9 down > 0.01 |
| 3 | Q372 | 77 % of the loss sits where the head never guesses; weighting toward its positions buys acceptance there | per-position source labels on the training sessions (the server's lookups replayed, CPU), loss weight 3–4 on 1-token-continuation and no-guess positions, 0.3–1 elsewhere; an arm with a segment embedding | at-position held-out (served coupling) | +0.01–0.03 at-position | CPU labels (hours) + 2 × 4 h | < +0.005 at-position, paired |
| 4 | Q373 | Free data: ~10–12 M tool-result positions in the dump (est.: 26.95 M replayed against 14.3 M generated) carry Flint's predictions of code it reads | targets at tool-result positions (Flint's top-20, no cross-entropy, weight 0.3–0.5), after-miss and staircase windows; check first that the dump holds top-20 and the grid there | held-out g9 | +0.01–0.03 | 8 h | < +0.01 |
| 5 | Q374 | Unseen code is the gap (validation 0.524 against held-out 0.366): diversity beats volume | data run 2: 2× generated tokens from new code bases already scanned (sympy, numpy, networkx, mpmath, then OTP and llama.cpp once the sandbox has their toolchains); same recipe and epochs; held-out modules unchanged | held-out g9; validation − held-out gap | +0.04–0.07 (0.41–0.44 at matched epochs) | ~9 h generation (all 7 GPUs), ~4 h dump, ~33 GPU-h of training (~17 h over two RTX 3090s once data-parallel training exists), +1.4 TB | < +0.02 at matched epochs against the same-epoch control |
| 6 | Q375 | Capacity limits the head (an open question: M0's 4-layer test was cut short) | an offline 6-layer, FFN 8,192 head (~0.6 B parameters, never served) against the 2-layer control, same data and steps, 8 h each | held-out g9 | decides: ≥ +0.03 → capacity matters (do Q382); < +0.01 → data and recipe | 2 × 8 h | – (diagnostic) |
| 7 | Q376 | The inputs at L6 support ≥ 0.5 with context (learning-limited) | tuned-lens probes per layer; after-miss heads with the newest at L4 … L47 and full depth (section 2.1) | top-1 agreement and acceptance per depth | decides; publishes Flint's next-token decision by layer | probes minutes each; 10 × 2 h | – (diagnostic) |
| 8 | Q377 | The M4 data slope and the diversity slope | from scratch, 2 epochs each: 25 / 50 / 100 % of the sessions (all modules), and 25 / 50 % of the modules (all their sessions) | held-out g9 per run | gives the per-doubling slopes that size data run 2 | ~25 GPU-h (two 3090s, ~12 h) | – (diagnostic) |
| 9 | Q384 | The served/offline gap (0.248 against 0.314) is domain and window mix, not the engine | dump the six replayed agent-check traces (36 requests, 20,544 generated tokens) like the training data, for scoring only (the check stays held out); score the served head offline by source and window | offline at-position acceptance on the check's own text | locates the 0.07: if offline ≈ 0.25, the gap is domain (JavaScript, Exercism) and data run 2 must cover it | minutes of dump, an hour | – (diagnostic) |
| 10 | Q378 | The served objective is the coupled match at α 3, not KL at α 1 | `--loss lk --eta 3` with the match at α 3 (code: α in `lk_loss`); plus an α per head-confidence bin fitted on validation | held-out at the served α | +0.005–0.02, +0.005–0.015 for the α table | 2 × 4 h; the α table on the CPU | < +0.005 paired |
| 11 | Q379 | The head memorizes modules: regularize and select by generalization | dropout 0.1, feature noise 0.2, weight EMA; validation and checkpoint choice on 3–4 extra held-out modules instead of training-module sessions | held-out g9 | +0.005–0.02 | 2 × 8 h | < +0.005, or the validation − held-out gap unchanged |
| 12 | Q380 | Fusing card 1's layers helps as EAGLE-3's fusion did (+14 % τ) | the newest token's L4 and L6 (both on card 1: no copy), and every reached grid layer for in-flight rows, zero-initialized projections | held-out g9; guess cost on one P106 | +0.01–0.03; +0.1–0.3 ms | 2 × 4 h; engine 1–2 days | < +0.01, or > 0.3 ms per guess |
| 13 | Q381 | The head cannot copy from context; a pointer fixes unseen-code identifiers | first: the share of head-position misses whose token occurs in the last 4,096 tokens (and follows the same token there); then a gated mixture of the vocabulary softmax and a successor distribution from the last block's attention (pointer-generator) | in-context share; held-out g9 | +0.01–0.04 if the share is ≥ 40 %; +~0.05–0.1 ms | CPU hours, then 8 h + engine 2–3 days | in-context share < 25 %, or < +0.01 |
| 14 | Q382 | Capacity is cheap on the output row: the pruned last block runs one row | a 2× FFN in the last block, or an MLP-only third block on the output row (~25 M parameters, ~0.25 ms each at Q8_0) | held-out g9; guess cost | +0.01–0.02; +0.25–0.4 ms | 8 h; engine 1 day | < +0.01 (only if Q375 says capacity matters) |
| 15 | Q383 | Starting from Flint's late layers gives the head Flint's own late-layer computation | the two blocks initialized from Flint's layers 47–48 (attention, 32 query heads, and the shared expert as the FFN), against from scratch at equal steps | held-out g9 at 2 h and 8 h | 0 to +0.03 | 2 × 8 h | not ahead at equal steps after 2 h |
| 16 | Q385 | The "(est.)" breakdowns of this note, measured | from existing score files: per segment × source × Flint-confidence bin × window × position in session; Σ min(p, q) against both couplings | – | – | CPU hours | – (diagnostic) |
| 17 | Q386 | When the pipeline is not full, complete tokens enter at their staircase depth and information is lost | the engine feeds the deepest existing state per row; train on mixed windows | served at-position (replay) | +0.01–0.02 after Q371 | engine 2 days, 8 h training | < +0.01 offline on mixed windows |
| 18 | Q387, Q276 | A second guess in a spare slot catches much of the first one's misses | the top-2 hit rate per position in `score.py`, then `cost_sim.py` with hedged chains (each branch costs its own head launches on card 1) | simulated tok/s | +2–6 % served if the top-2 rate is ≥ top-1 + 0.10 and launches are cut (Q194) | an hour offline, then 3–5 days engine | simulated gain < 3 % |
| 19 | Q393 | The newest token at L9 instead of L6 is worth ~+0.03 (the 3–12-card head's card counts) | the simulator with a deeper first stage or the head after it, against card-1 slack | simulated tok/s | unknown; likely negative while the head shares card 1 | an hour | simulated gain < 2 % |
| 20 | Q394, Q395 | More supervision per step | every window row also predicts its own next token; KL against the top-20 before top-p | held-out g9 | +0–0.02 each | 2 × 4 h each | < +0.005 |
| – | Q271 | Block drafting amortizes the guess | as planned | – | after first-position acceptance ≥ 0.45 | – | – |
| – | Q388–Q392 | parked: MoE FFN in the head; online test-time training on P106; an EAGLE-3/MTP head for fast GPUs; step-level reasoning speculation (lossy); separate heads per segment | – | – | – | – | reopen conditions in the registry rows |

Order of work on the quant machine (est.): items 16, 9 and the in-context share of 13 first (CPU and an hour of dump);
then 2, 3, 4 and 10 as paired 4–8 h runs on two RTX 3090s from the 1,480-minute head; 6–8 in parallel on the remaining
cards; then data run 2 (item 5) with every change adopted. Adopt a change at ≥ +0.005 with the paired interval above
zero. The combined effect, if the adopted changes add at half their sum (est.): 1,480-minute head 0.38–0.39, cheap
changes +0.03–0.06, data run 2 +0.04–0.07: 0.45–0.52 held-out, served est. 0.31–0.40 at today's served/offline ratio.

## 6. What the training curves say

From [val_curve.tsv](../../data/2026-10-27_mtp_head/offline/val_curve.tsv), the training logs and the round's
held-out scores (nine-card head at nine cards, α 1):

| Point | Steps (8,192 targets) | Epochs (est.) | Validation g9 | Held-out g9 |
|---|---|---|---|---|
| 240 minutes | 1,392 | ~1.0 | 0.365 | 0.277 |
| restart (+20 min, lr re-warmed) | 1,485 | ~1.04 | 0.337 | – |
| 316 minutes | ~1,790 | ~1.3 | 0.363 | – |
| 946 / 1,000 minutes | ~4,700 / ~4,950 | ~3.4 | 0.524 | 0.366 |
| 1,480 minutes (best checkpoint, ~1 h before the end) | ~7,000 | ~5 | 0.557 (final) | 0.380 |

- **Under-trained, yes.** At 240 minutes training KL was 2.1–2.2 nats against 2.44 on validation
  ([train_log_nine.jsonl](../../data/2026-10-27_mtp_head/offline/train_log_nine.jsonl) lines 82–85): no overfitting
  within one epoch. Held-out kept rising to ~3.4 epochs. Published drafters see far more token-passes: EAGLE ~10⁹ over
  ~20 epochs, SPD ~1.2 × 10⁹ in one epoch, Mixture of Attentions 2.4 × 10⁹, P-EAGLE 20–60 epochs; this head
  4.7 × 10⁷ (13.7 M × 3.4), 0.3 per parameter (est.).
- **Data-limited for unseen code, yes.** Validation gained +0.16 for +0.09 held-out; the gap grew from 0.088 to 0.158.
  More epochs on the same 92 modules increasingly buy memory of those modules.
- **Under-sized: unknown.** SPD's 2 → 4 layers +0.02 per token (est.), P-EAGLE's +9 % τ, Scylla's +0.22 τ per doubling of
  layers say modest; the val/held-out gap says a bigger head on the same data would mostly memorize more. Q375 decides.
- **Fit.** Through the two held-out points: A(S) = 0.277 + 0.049 · log2(S / 1,392), S the steps (est.; fresh text the
  same slope, 0.235 → 0.323). At 7,150 steps it gives 0.392; repeats lose value past ~4 epochs (Muennighoff's effective
  data at 5 epochs ≈ 90 % of unique, at 6.6 ≈ 86 %), so est. 0.38–0.39. The 1,480-minute score tests the fit: below
  0.375 means the curve is bending and the data is used up. It scored 0.380 (0.340 on fresh text), 0.011 under the fit:
  the slope since 1,000 minutes is about +0.027 per doubling. Validation through 1,792 and 4,700 steps rises +0.116 per
  doubling, 2.4× the held-out slope.

| Option | What it buys (est.) | Basis |
|---|---|---|
| Another 8 h on the same data (~2,200 steps) | +0.02–0.03 to ~5 epochs, then +0.015–0.02 for the next 8 h; ~0.40 ceiling on this data | the fit with repeat decay |
| 2× data from new code bases, same epochs | +0.04–0.07 | the steps slope (fresh data ≥ repeats) plus the diversity effect on unseen modules; SpecForge's 4.4× data +23 % τ |
| 2× head size (4 blocks) | +0.01–0.02 offline; +~1–1.2 ms per guess (two more 8-row blocks), past the ~3 ms cliff, so a net loss on nine P106 cards | SPD, P-EAGLE, Scylla; the guess profile (Q192) |
| 2× capacity on the output row only | +0.01–0.02 at +0.25–0.4 ms | Q382 |

## 7. What the existing dumps are for

The dumps of run 1 (26-layer grid, ~55 KB per replayed token, 1.43 TB with the sequences on the bulk disks) feed every
offline item above: the probes (Q375–Q377), the window and weighting runs (Q371–Q373), LK and α (Q378), regularization
(Q379), card-1 fusion (Q380: L4 and L6 exist for every position), the pointer measurement and head (Q381), output-row
capacity (Q382), the Flint-layer initialization (Q383), the breakdowns (Q385, with the dumps' top-20) and the block
guesser (Q271). They can be rebuilt from the kept session sequences in ~4 h on three dump instances (26.95 M tokens at
~1,860 tokens/s), so the sessions are the irreplaceable part; the dumps are a cache worth keeping while these
experiments run. Data run 2 adds its own (~1.4 TB at the same size).

## 8. Not verified

- The literature numbers are the papers' own; few are at temperature 1 with sample-and-match coupling, none on a layer
  split of an MoE target with a one-token drafter.
- The epoch count of the 1,000-minute round, the share of head guesses that follow a miss, the head's reasoning share
  and the tool-result position count are this note's estimates (Q385 and Q373 measure them).
- The EAGLE-grade estimate (0.55–0.65 on Flint's agent text) transfers top-1 agreement from other targets.
- The fit was made on two points; the third, the 1,480-minute score (0.380), sits 0.011 below it.
- The validation curve of the overnight round (0.363 at 316 minutes, 0.524 at 946 minutes) and the 1,000-minute scores
  are in [the head's data folder](../../data/2026-10-27_mtp_head/README.md) (`offline/train_log_*_overnight.jsonl`,
  `offline/scores_heldout_1000min.md`).

## Sources

Feature-level heads and training
- Y. Li, F. Wei, C. Zhang, H. Zhang, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty", [arXiv 2401.15077](https://arxiv.org/abs/2401.15077)
- Y. Li et al., "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees", [arXiv 2406.16858](https://arxiv.org/abs/2406.16858)
- Y. Li et al., "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test", [arXiv 2503.01840](https://arxiv.org/abs/2503.01840)
- L. Zhang et al., "Learning Harmonized Representations for Speculative Sampling" (HASS), [arXiv 2408.15766](https://arxiv.org/abs/2408.15766)
- S. Hu et al., "GRIFFIN: Effective Token Alignment for Faster Speculative Decoding", [arXiv 2502.11018](https://arxiv.org/abs/2502.11018)
- M. Zimmer et al., "Mixture of Attentions for Speculative Decoding", [arXiv 2410.03804](https://arxiv.org/abs/2410.03804) ([html](https://arxiv.org/html/2410.03804v2), Table 5)
- "SpecForge: A Flexible and Efficient Open-Source Training Framework for Speculative Decoding", [arXiv 2603.18567](https://arxiv.org/abs/2603.18567) ([html](https://arxiv.org/html/2603.18567): regenerated data, MoE drafters, TTT length)
- M. Hui et al., "P-EAGLE: Parallel-Drafting EAGLE with Scalable Training", [arXiv 2602.01469](https://arxiv.org/abs/2602.01469) ([html](https://arxiv.org/html/2602.01469v1): layer and epoch ablations)
- "DFlash", [arXiv 2602.06036](https://arxiv.org/abs/2602.06036) (via [uzu lessons](uzu_lessons.md))
- Z. An et al., "PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation", [arXiv 2504.18583](https://arxiv.org/abs/2504.18583)
- "Falcon: Faster and Parallel Inference of Large Language Models through Enhanced Semi-Autoregressive Drafting and Custom-Designed Decoding Tree", [arXiv 2412.12639](https://arxiv.org/abs/2412.12639)
- J. Li et al., "Gumiho: A Hybrid Architecture to Prioritize Early Tokens in Speculative Decoding", [arXiv 2503.10135](https://arxiv.org/abs/2503.10135)
- T. Cai et al., "Medusa", [arXiv 2401.10774](https://arxiv.org/abs/2401.10774); Z. Ankner et al., "Hydra", [arXiv 2402.05109](https://arxiv.org/abs/2402.05109); B. Xiao et al., "Clover", [arXiv 2405.00263](https://arxiv.org/abs/2405.00263); Y. Cheng et al., "Recurrent Drafter for Fast Speculative Decoding" (ReDrafter), [arXiv 2403.09919](https://arxiv.org/abs/2403.09919)
- "FastDraft: How to Train Your Draft", [arXiv 2411.11055](https://arxiv.org/abs/2411.11055)
- Y. Zhou et al., "DistillSpec", [arXiv 2310.08461](https://arxiv.org/abs/2310.08461); S. Ouyang et al., "Temperature-Centric Investigation of Speculative Decoding with Knowledge Distillation", [arXiv 2410.10141](https://arxiv.org/abs/2410.10141)
- Y. Hu et al., "AdaSPEC: Selective Knowledge Distillation for Efficient Speculative Decoders", [arXiv 2510.19779](https://arxiv.org/abs/2510.19779)
- "Flatter Tokens are More Valuable for Speculative Draft Model Training", [arXiv 2601.18902](https://arxiv.org/abs/2601.18902)
- Z. Lin et al., "Rho-1: Not All Tokens Are What You Need", [arXiv 2404.07965](https://arxiv.org/abs/2404.07965)
- A. Samarin et al., "LK Losses", [arXiv 2602.23881](https://arxiv.org/abs/2602.23881); [Nebius post](https://nebius.com/blog/posts/lk-losses)
- "Acceptance-Aware Draft Model Training for Speculative Decoding", [arXiv 2609.24150](https://arxiv.org/abs/2609.24150)
- S. Yan et al., "Scaling Laws for Speculative Decoding" (Scylla), [arXiv 2505.07858](https://arxiv.org/abs/2505.07858)
- N. Muennighoff et al., "Scaling Data-Constrained Language Models", [arXiv 2305.16264](https://arxiv.org/abs/2305.16264)
- A. See, P. Liu, C. Manning, "Get To The Point: Summarization with Pointer-Generator Networks", [arXiv 1704.04368](https://arxiv.org/abs/1704.04368)
- S. Hu et al., "MiniCPM" (warm-up–stable–decay schedule), [arXiv 2404.06395](https://arxiv.org/abs/2404.06395)

Built-in MTP and self-speculation
- DeepSeek-AI, "DeepSeek-V3 Technical Report", [arXiv 2412.19437](https://arxiv.org/abs/2412.19437) (second-token acceptance 85–90 %)
- "FastMTP: Accelerating LLM Inference with Enhanced Multi-Token Prediction", [arXiv 2509.18362](https://arxiv.org/abs/2509.18362); "Self-Distillation for Multi-Token Prediction", [arXiv 2603.23911](https://arxiv.org/abs/2603.23911)
- M. Elhoushi et al., "LayerSkip", [arXiv 2404.16710](https://arxiv.org/abs/2404.16710); F. Liu et al., "Kangaroo", [arXiv 2404.18911](https://arxiv.org/abs/2404.18911); J. Zhang et al., "Draft & Verify", [arXiv 2309.08168](https://arxiv.org/abs/2309.08168); J. Liu et al., EESD, [arXiv 2406.03853](https://arxiv.org/abs/2406.03853)
- FR-Spec, [arXiv 2502.14856](https://arxiv.org/abs/2502.14856)

Depth and lenses
- N. Belrose et al., "Eliciting Latent Predictions from Transformers with the Tuned Lens", [arXiv 2303.08112](https://arxiv.org/abs/2303.08112)
- "How Do LLMs Use Their Depth?", [arXiv 2510.18871](https://arxiv.org/abs/2510.18871)
- K. Pal et al., "Future Lens: Anticipating Subsequent Tokens from a Single Hidden State", [arXiv 2311.04897](https://arxiv.org/abs/2311.04897)

Reasoning, long outputs, routing
- "Test-Time Speculation", [arXiv 2605.09329](https://arxiv.org/abs/2605.09329) ([html](https://arxiv.org/html/2605.09329v1))
- "SpecExtend", [arXiv 2505.20776](https://arxiv.org/abs/2505.20776); "LongSpec", [arXiv 2502.17421](https://arxiv.org/abs/2502.17421)
- "SpecReason", [arXiv 2504.07891](https://arxiv.org/abs/2504.07891); "Scaling Speculative Decoding with Lookahead Reasoning", [arXiv 2506.19830](https://arxiv.org/abs/2506.19830)
- "TAPS: Task Aware Proposal Distributions for Speculative Sampling", [arXiv 2603.27027](https://arxiv.org/abs/2603.27027)
- G. Oliaro et al., "SuffixDecoding", [arXiv 2411.04975](https://arxiv.org/abs/2411.04975)

Pipeline and coupling
- Y. Yu et al., "Speculative Pipeline Decoding" (SPD), [arXiv 2605.30852](https://arxiv.org/abs/2605.30852)
- Li et al., PPSD, [arXiv 2509.19368](https://arxiv.org/abs/2509.19368); B. Butler et al., "PipeInfer", [arXiv 2407.11798](https://arxiv.org/abs/2407.11798); X. Liu et al., "FlowSpec", [arXiv 2507.02620](https://arxiv.org/abs/2507.02620)
- Z. Sun et al., "SpecTr: Fast Speculative Decoding via Optimal Transport", [arXiv 2310.15141](https://arxiv.org/abs/2310.15141)
- M. Daliri, C. Musco, A. T. Suresh, "Coupling without Communication and Drafter-Invariant Speculative Decoding", [arXiv 2408.07978](https://arxiv.org/abs/2408.07978)

This project
- [docs/drafter.md](../../drafter.md) (M0–M4, "A cheaper guess", section 7), [drafter acceptance](drafter_acceptance.md),
  [pipeline drafter](pipeline_drafter.md), [idle cards](idle_cards.md) (hedged guesses), [uzu lessons](uzu_lessons.md)
  (DFlash), [Flint v3 plan](flint_v3_plan.md)
- Data: [2026-10-27_mtp_head](../../data/2026-10-27_mtp_head/README.md) (`val_curve.tsv`, `final_alpha.tsv`,
  `scores_heldout.md`, configs, training logs), [2026-10-27_cheap_guess](../../data/2026-10-27_cheap_guess/README.md)
  (`source_split.md`, the replays, Q194's profile), [2026-10-27_answer_store](../../data/2026-10-27_answer_store/README.md)
  (`offline/agentic.md`), [2026-10-26_agentic_data](../../data/2026-10-26_agentic_data/README.md) (modules, licence scans)
- Code: [bench/staircase_drafter](../../../bench/staircase_drafter/README.md) (`drafter.py`, `train.py`, `score.py`,
  `cost_sim.py`, `source_split.py`)
- Registry ([quants/experiments.md](../../../quants/experiments.md)): Q20, Q96, Q161, Q169, Q190–Q194, Q271, Q276, Q310,
  Q341, Q342, and the new rows Q370–Q395 named above
