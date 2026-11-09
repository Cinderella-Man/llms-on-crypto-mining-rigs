# Weight-space LP: what it is, and what I can take (2026-11-09)

What I read:
- the post, [Linear programming for weight-space programming](https://nathan.rs/posts/weight-space-lp/);
- its repository, [nathanrs/weight-space-lp](https://github.com/nathanrs/weight-space-lp), with
  `docs/WRITEUP.md`. I read it as text only and ran nothing. Its page shows no licence;
- the paper it builds on: Takehi, Teoh, Zhang and Khattab,
  [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/) (MIT CSAIL, 2026);
- the related work in the sources at the end.

## The short answer

- **The post does not make a model smaller.** Its linear program (LP) minimizes how many weights an *edit* changes
  (the L1 norm of the edit), not the weights themselves. The edited model has the same size, so the cards read the same
  bytes per token.
- **What it does is stop reasoning loops,** and that touches the metric in a different way. A loop burns tokens until
  a cap, so it costs time per task. Loops grow at 2 bits (arXiv 2606.02011), and Swift Qwen3.8-Flash-Next Pascal Q2
  GGUF is a 2-bit file.
- **The cheap form of the fix is a steering vector,** and llama.cpp already applies one: `--control-vector-scaled` and
  `--control-vector-layer-range`. The graphs of all three of my models call it: `flint`, `qwen4exp` (Flash-Next) and
  `qwen35moe` (Ornith). The cost is one vector add per steered layer and token.
- **Whether my models loop enough to matter is not known.** The agent check has runaways (section 3), but nobody has
  counted how many of them are repeated text. That count comes first (Q1138).
- **Reading fewer bytes by removing weights is pruning.** Pruning and the lottery ticket hypothesis are covered in
  the Elixir models repo, which holds all the work on making models smaller (section 4).

## 1. What the post does

**The problem.** Qwen3.5-0.8B gets stuck repeating lines while it thinks. In the paper, 93.0 % of its base answers on
MATH-500 hit the 32,768-token limit.

**The paper's edit (Astra, an LLM agent working on the weights).** Astra wrote four pairs of looping and productive
texts and took the difference of their mean residual vectors as a "loop direction" u (1,024 wide). It traced the
direction to the MLPs of layers 15 and 19. It then took u out of 64 columns of each layer's `down_proj`, six times
over (w_j → w_j − 6 (uᵀw_j) u). That changed 130,188 weights, 0.016 % of the model. The paper's numbers, base → edit:

| | MATH-500 | GSM8K | IFEval | hit the token limit |
|---|---:|---:|---:|---:|
| Base | 7.9 % | 13.8 % | 21.0 % | 93.0 % |
| Astra's edit | 48.1 % | 67.1 % | 62.3 % | 33.1 % |

Finding the edit took Astra about 8 hours of its 50-forward-pass budget. Gemma 3 270M, which seldom loops, gained
nothing: the edit removes an obstacle, but it cannot add a skill the model lacks.

**The post's LP.** Removing a direction from columns of W is a rank-1 edit ΔW = u vᵀ, so only the vector v (3,584
wide, one value per MLP neuron) has to be chosen. The post chooses it with one LP per layer:

- minimize Σ|v_j| + C · Σ slack;
- looping tokens t: v · h_t ≤ −δ, so the push along u is at least δ;
- productive tokens t: |v · h_t| ≤ ε, so they barely move. ε is 1.0 ("broad") or 0.25 ("selective");
- keep the 8 neurons with the largest |v_j| · std(h_j), then solve again on those 8.

HiGHS took over 20 minutes per solve. The author's GPU PDHG solver (the PDLP algorithm) gets within 0.2 % of
HiGHS's optimum in about a minute.

**Token labels, with no LLM involved.** A token repeats when the 16 tokens ending at it appeared earlier in the reply.
A loop starts at the first point after which at least 80 % of the remaining tokens repeat. Looping tokens are repeats
at least 64 tokens into a loop. Productive tokens are at least 256 tokens before a loop, or in a reply that finished.
The "heuristic direction" is the mean residual on looping tokens minus the mean on productive tokens, taken from the
model's own traces.

**Results on the test sets:**

| Edit | weights changed | GSM8K | MATH-500 | IFEval |
|---|---:|---:|---:|---:|
| Base | 0 | 14.1 | 7.0 | 22.9 |
| Astra's edit | 130,188 | 68.2 | 50.9 | 61.9 |
| LP, Astra's direction | 16,291 | 69.6 | 51.7 | 59.7 |
| LP, heuristic direction | 16,271 | 65.4 | 47.8 | 60.4 |

With the heuristic direction and no agent, the LP gets about 93 % of Astra's result.

**A steering vector does the same job.** The LP's 8 neurons fire on 89 % of tokens (a typical neuron fires on 16 %),
so the edit is close to a constant push along u. Adding that push as a vector at layers 15 and 19 of the unedited
model scored 57.8 on the dev set and 50.8 on MATH-500 at 1×, against 62.5 and 53.2 for the LP edit and 10.9 and 7.0
for the base model. The dose window is narrow: 51.6 on the dev set at 0.5× and 31.2 at 2×. A linear probe spots a loop
that is already under way at AUC 0.93, but its start only at 0.73. So the fix works as a direction and a dose, not as
a switch that fires per token.

**Limits (the author's).** One model and one behaviour. The layers are still Astra's choice. Sampling penalties and L2
baselines were not tried. The whole study cost about 18 H100-hours and 1.5 A100-hours.

## 2. Does it shrink what the cards read? No

- The LP minimizes the edit, not the model. Every tensor keeps its shape and its type. Baking the edit into a
  quantized file means quantizing the touched blocks again: the size stays the same.
- The steering form adds one vector of the model's width (n_embd) per steered layer: kilobytes read per token,
  against the gigabytes the weights take.
- The bytes-per-token question belongs to pruning (section 4).

## 3. Where it can help: loops cost time per task

**Low bits make loops more likely.** [arXiv 2606.02011](https://arxiv.org/abs/2606.02011) (Rudas et al., June 2026)
tested Qwen3-8B and Qwen3-32B with GPTQ-style 2-bit weights (W2A16):
- 2-bit traces grow longer through repeated loops, exhausted budgets and reasoning that is never closed;
- at temperature 0.6, Qwen3-8B looped in 90.2 % of answers and Qwen3-32B in 18.0 %;
- their "loop rescue" flags a 20-gram that occurs at least 4 times in the last 1,024 tokens, then either commits to an
  earlier answer or reruns in FP16. On MATH-500 it lifts Qwen3-8B from 17.2 to 74.2 %, and in their worst setting
  mean length falls from 12,222 to 949 tokens;
- they tested no MoE models and nothing agentic.

The older [arXiv 2504.04823](https://arxiv.org/abs/2504.04823) found output lengths unchanged at 4 bits and longer
outputs from W3G128 down.

**My own runs.** An attempt is a runaway when the agent check stops it at the 12-minute or 40-turn cap. Runaways
appear on Flash-Next and Flint:
- `stockpmix` (stock + Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, 2026-11-06): 10/11, and the failed task was a runaway;
- S36, the adaptive draft on Flash-Next: one runaway in 12 tasks;
- the expert-skip arm on Flash-Next (`LLAMA_MOE_TAIL`, release v3): 11/12, one runaway;
- Flint `q5stock` (stock + Unsloth UD-Q5_K_XL, 2026-11-06): 11/12, one runaway at the 12-minute cap in a 62.7-minute
  arm;
- Flint on v2 (Q266): `bugfix-python-scale-generator` took 33–36 % of every agent-check run, through single replies of
  9–15k reasoning tokens, and hit the cap in 9 of 15 runs;
- Ornith's release arms had none (12/12).

No one has checked whether these replies repeat themselves or reason at length. The proxy keeps every reply
(`proxy.jsonl.bodies.gz`), so the post's labels and the paper's 20-gram rule can be run over all of them offline.
That count is Q1138, and it decides the rest.

**What the post adds to what is queued.**
- Q266 (a runaway guard: a repetition detector or a reasoning budget) cuts a loop once it has started.
- Q267 (a concise-reasoning steering vector) uses the same mechanism for a different direction.
- The post's fix stops loops from starting. It costs almost nothing at run time, and its direction can come from my
  own replies with no LLM in the loop. The tooling would be shared with Q267: activations on labelled spans, then a
  scale sweep. llama.cpp's `llama-cvector-generator` builds a vector from positive and negative text files
  (`--method mean` or `pca`). How it pairs and pads the two files has to be checked against unpaired loop spans; a
  small residual dump is the fallback.

**Caveats.**
- My three models are MoE, and the post edits a dense MLP. The vector is the same either way, since it adds to the
  residual whatever wrote it. A baked-in edit would go on the shared expert's down projection, where a model has one.
- The dose window is narrow (0.5× too weak, 2× breaks reasoning), so each model needs its own scale sweep with a
  quality gate on text that does not loop. That gate is the LP's ε: KLD on FN1's agent, wiki and code bases, within
  noise.
- My models loop far less than the 0.8B model did (93.0 % of answers at the limit). The Gemma control in the paper
  shows no gain where loops are rare, so the gain is capped at the share of time the runaways take.
- The agent check samples at the makers' settings (temperature 1.0, top-p 0.95, top-k 20, min-p 0). The paper's loop
  rates are at 0.6, and lower temperatures loop more.
- A constant push acts on tool-call JSON too. Q267 planned to limit its vector to the `<think>` span, and the same
  applies here (code).
- It changes the text, so the task-level gate (the pass rate) decides, as for Q266 and Q267.

**Rows:**
- Q1138: the loop census;
- Q1139: the loop vector;
- Q1140: the fix baked into a published file, or shipped as a vector file beside it.

## 4. Pruning and the lottery ticket hypothesis

Reading fewer bytes by removing weights (pruning), and the lottery ticket hypothesis, are in
[pruning_and_lottery_tickets.md](https://github.com/Cinderella-Man/elixir-agents-on-mining-rigs/blob/main/docs/research/notes/pruning_and_lottery_tickets.md), with the other work on making models smaller.

## Sources

- The post: https://nathan.rs/posts/weight-space-lp/
- Its repository: https://github.com/nathanrs/weight-space-lp
- Takehi, Teoh, Zhang, Khattab, Sparks of Weight-Space Programming (MIT CSAIL, 2026):
  https://rikiyat.github.io/blog/weight-space-programming/
- Rudas et al., Extreme Low-Bit Inference in Reasoning Models: Failure Modes and Targeted Recovery (2026):
  https://arxiv.org/abs/2606.02011
- Liu et al., Quantization Hurts Reasoning? (2025): https://arxiv.org/abs/2504.04823
