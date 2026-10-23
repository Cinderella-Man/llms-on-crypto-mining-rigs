# Task-level quality checks for Flint quants: how many tasks, which benchmarks, and a protocol for Q41 (2026-10-23)

Scope: the Flint files at the quality bar (Unsloth's UD-Q4_K_XL against BF16: KLD 0.120, 87.7 % same top token,
perplexity +3.9 % on 80 × 512 tokens of wikitext-2) sit within a few hundredths of KLD of each other, and the 300-task
BFCL check ([Q30](../../../quants/experiments.md)) cannot tell them apart. This note works out how many paired tasks
separate two quants of the same model, which tool-calling and coding benchmarks run against llama-server without a judge
model, and what the literature says about KLD as a predictor of task scores. It ends with a protocol for Q41 (larger task
checks) and the task section of the Q42 model cards. Conventions are those of the other notes:
- "Claim to test" marks anything not measured here.
- "(my calc)" marks this note's own arithmetic.
- "P" and "S" in source lists mean primary (code, data, papers, model cards) and secondary.

Starting point: the Q30 run, re-analysed here task by task. Ten complete runs of 300 single-turn BFCL v3 tasks: five
files plus five repeats on other GPUs or with another slot count, all greedy, Flint's own chat template, BFCL's AST
checker ([bench/bfcl_flint](../../../bench/bfcl_flint/)). Every file scored 78.3–83.7 % and BF16 80.7 / 81.0 %.

## Headline findings

1. **Flint's greedy tool-call verdicts behave like random draws, so one run of a file is one sample.** Across the ten Q30
   runs, 91 of 300 tasks (30 %) are solved in some runs and failed in others (my calc).
   - Two runs of any two files disagree on 5.0–15.3 % of tasks, mean 10.8 %. That equals what independent draws from the
     same per-task success rates predict (2 × E[p(1−p)] = 2 × 0.054).
   - A quant against BF16 disagrees on 11.2 % on average, no more than reruns of one file. At 300 tasks no file differs
     measurably from any other.
   - Reruns on the same GPUs are correlated (within-task ρ 0.33–0.54); a rerun on another GPU type is nearly independent
     (ρ 0.10). Repeats should therefore change the GPU type, not just restart the server.
2. **Paired tasks needed, two-sided α 0.05, power 0.8, at Flint's 11 % discordance (my calc):** 1 point ≈ 8,600 tasks,
   2 points ≈ 2,160, 3 points ≈ 960, 5 points ≈ 340. Unpaired comparisons need 3–6 times more.
   - Because the files differ only by noise, tasks and repeats are interchangeable: what counts is tasks × runs.
   - BFCL v4's full judge-free single-turn Python pool (3,491 tasks) run twice per file resolves ~1.0 point. It supports
     a non-inferiority margin of 1.5 points against UD-Q4_K_XL with ~99 % power if the files are equal.
3. **KLD filters damaged files but does not rank near-lossless ones.**
   - Across 28 GGUF quants of Qwen3.6-35B-A3B, KLD correlates with a benchmark composite (ρ −0.72). Among the 17 quants with
     KLD ≤ 0.064 the correlation is 0.00 (arXiv 2606.19558).
   - Flint's own exception: the makers' distillation-tuned Q4_1 (KLD 0.322) is as good as BF16 on tool calls (Q30,
     terelabs).
   - Keep KLD as the screening gate and tasks as the release check. Within the KLD bar, only thousands of samples per
     file can show a task difference.
4. **Quantization damage shows first in multi-step and long-reasoning tasks, and in process errors that a final score
   absorbs.**
   - Red Hat's NVFP4 Gemma-4-26B-A4B: BFCL v4 single-turn −1.9 points, multi-turn 0.0, agentic −11.8.
   - Qwen3-4B GGUF on BFCL v4 multi-turn: 25.5 (Q8_0), 24.5 (Q4_K_M), 15.0 (IQ3_XXS).
   - On τ²-bench, 4-bit agents kept their scores while tool-name hallucinations rose up to 2.5×, hidden by the
     benchmark's error budget.
   - The protocol therefore adds a multi-turn lane and gates on malformed or runaway outputs as well as accuracy.
5. **BFCL v4 is the only tool-calling pool above 2,000 tasks that is judge-free, local and deterministic to score.**
   - It has 3,491 Python single-turn tasks, plus 800 multi-turn tasks on local Python backends.
   - τ²/τ³-bench needs a simulator LLM and judge models, and resolves only ±7.5 points at 114 tasks × 4 trials.
   - When2Call's multiple-choice mode needs prompt log-probabilities, which llama-server does not return.
   - The MCP suites have 76–250 executable tasks and need live services or judges.
6. **For coding, LiveCodeBench code generation separates quants best per GPU-hour.** It is also the coding benchmark
   Flint's makers report (BF16 80.3, NVFP4 80.8).
   - 454 problems dated from 2024-08-01, 4 samples each at the makers' sampling, give a paired standard error of
     ~0.9 points (my calc).
   - HumanEval+/MBPP+ are saturated and too small: SE 3.7 / 2.5 points per run.
   - SWE-bench and Terminal-Bench cost tens of millions of tokens per run, for a standard error of 1.9–5.8 points.
   - Flint's knowledge cutoff (June 2026) is later than every LiveCodeBench problem. Absolute scores may be inflated;
     comparisons between quants of the same weights are not affected.
7. **Decoding:**
   - **BFCL: greedy**, the benchmark's own setting and Q30's. Flint's greedy output is not reproducible across GPUs anyway,
     so repeats on other GPU types act as samples.
   - **LiveCodeBench: the makers' sampling** (temperature 1.0, top-p 0.95, top-k 20, min-p 0) with 4 seeds, averaged per
     problem, as the makers average 5 attempts. Long reasoning at greedy risks repetition loops.
   - Temperature is not lowered to cut variance; Miller advises against it.
8. **Cost of the protocol: ~78 server-hours, ~180 GPU-hours, about 1.1 days of the whole quant machine** (my calc,
   claim to test).
   - Files: UD-Q4_K_XL, S4-L0 and S5 in full, Unsloth's Q8_0 as the anchor, BF16 on single-turn BFCL.
   - LiveCodeBench is two thirds of it, and its cost scales with Flint's unmeasured reasoning length.
9. **llama.cpp has no Flint-specific tool-call parser.**
   - Flint's template falls through to the generic autoparser. Its lazy grammar restricts argument names to the schema,
     so a `--jinja` client cannot see invented argument names.
   - The raw-prompt lane therefore measures the model, and a separate `--jinja` lane measures what users get.
   - Open llama.cpp items: tool-call XML left inside the reasoning (#31000); unbounded optional-argument repetition
     (draft PR #31040).

## 1. What the 300-task check shows, task by task

All numbers are my calc from Q30's per-task verdicts: ten complete runs of the same 300 tasks.
- Files: BF16, the makers' Q4_1, UD-mirror v1, S4, S1 and makers-fast.
- Repeats: BF16 and UD-mirror v1 on the same GPUs, the makers' Q4_1 on V100s with two requests in flight and with one.

| Category | Tasks | Mean accuracy, 10 runs | Tasks with mixed verdicts | E[p(1−p)] |
|---|---|---|---|---|
| simple | 50 | 93.8 % | 8 | 0.023 |
| multiple | 50 | 94.4 % | 9 | 0.032 |
| parallel | 40 | 78.7 % | 21 | 0.094 |
| parallel_multiple | 40 | 78.5 % | 19 | 0.083 |
| live_simple | 40 | 74.8 % | 18 | 0.077 |
| live_multiple | 40 | 70.3 % | 11 | 0.056 |
| irrelevance | 20 | 79.0 % | 3 | 0.039 |
| live_irrelevance | 20 | 76.0 % | 2 | 0.017 |
| **All** | **300** | **82.0 %** | **91 (30 %)** | **0.054** |

- **Per-task outcomes:** 184 tasks are right in all ten runs and 25 wrong in all. The other 91 are spread from 1 to 9
  correct runs.
- **What E[p(1−p)] is:**
  - It is the mean over tasks of k(10−k)/90, where k is the number of correct runs.
  - It estimates the per-sample variance that remains once task difficulty is paired out.
  - Twice it is the expected disagreement between two independent runs.
- **Disagreement between two runs** (45 pairs):

  | Pair | Tasks whose verdict differs |
  |---|---|
  | Same file, same GPUs | 5.0–7.3 % |
  | Same file, RTX 3090 pair against V100 pair | 9.7 % |
  | A quant against BF16 | 7.3–14.3 %, mean 11.2 % |
  | All 45 pairs | 5.0–15.3 %, median 11.0 %, mean 10.8 % |

- **Within-task correlation between two runs** is ρ = 1 − disagreement / (2 E[p(1−p)]):
  - 0.33–0.54 for same-GPU reruns;
  - 0.10 across GPU types;
  - −0.04 for quant against BF16.

  A rerun on the same GPUs repeats part of the previous run's rounding path. A rerun on another GPU type, or any change
  of weights, is close to a fresh draw.
- **What the 300 tasks can show:** the true differences between files cannot be told from zero. The paired standard
  error at 300 tasks is 1.9 points, so a difference must be ~5.3 points to be detected with 80 % power.
  - This matches Q30's McNemar p ≥ 0.20 and terelabs' p 0.29–1.0.
  - terelabs counted 73 of their 300 tasks as unstable under greedy decoding even for BF16, in line with the 91 here.
- **Projected to the full BFCL v4 pool:** weighting the per-category values by its composition gives E[p(1−p)] ≈ 0.045
  (discordance ≈ 9 %; claim to test). The pool holds proportionally more live_irrelevance and live_multiple tasks and
  fewer parallel ones.
- **The output is short:** 163 tokens on average, median 129, p99 578, almost all of it reasoning. Prompts average 493
  tokens (p90 909, max 6,100). Single-turn tool tasks therefore cost little per sample, so many samples are affordable.

## 2. How many tasks: the statistics

### 2.1 Paired beats unpaired

For two files run on the same tasks, only the discordant tasks (right for one file, wrong for the other) carry
information. For a share d of discordant tasks and a true difference δ, the number of tasks for a two-sided test at
α = 0.05 with 80 % power is n = [z₀.₉₇₅ √d + z₀.₈ √(d − δ²)]² / δ² (Connor's McNemar formula). The exact McNemar test
or a paired bootstrap then analyses the result.

**Table A. Tasks per file needed to detect a difference δ, two-sided α 0.05, power 0.8 (my calc)**

| δ (points) | Unpaired, base 80 % | Unpaired, base 50 % | Paired, d = 0.06 | d = 0.09 | d = 0.11 | d = 0.15 | d = 0.20 | d = 0.30 |
|---|---|---|---|---|---|---|---|---|
| 1 | 25,583 | 39,240 | 4,707 | 7,062 | 8,632 | 11,771 | 15,696 | 23,545 |
| 2 | 6,510 | 9,806 | 1,175 | 1,764 | 2,157 | 2,941 | 3,923 | 5,885 |
| 3 | 2,943 | 4,356 | 521 | 783 | 957 | 1,306 | 1,742 | 2,614 |
| 5 | 1,094 | 1,565 | 186 | 281 | 343 | 469 | 626 | 940 |

What the d values correspond to:
- **0.06:** a same-GPU rerun.
- **0.09:** the full BFCL v4 pool (projected).
- **0.11:** a Flint quant against BF16 in Q30.
- **0.20–0.30:** typical of long, sampled tasks such as LiveCodeBench at temperature 1 (claim to test).

Base accuracy enters a paired test only through d, which is largest when many tasks have success probabilities near 0.5.

### 2.2 Repeats and sampling noise

Miller ([arXiv 2411.00640](https://arxiv.org/abs/2411.00640), Eq. 9) gives the paired sample size with K samples per
task:

n = (z_{α/2} + z_β)² (ω² + σ²_A/K_A + σ²_B/K_B) / δ²

- **ω²** is the variance across tasks of the true per-task difference between the two models.
- **σ²** is the mean within-task variance, E[p(1−p)] for pass/fail.

For two quants of one model ω² is close to 0: Q30's quant-against-BF16 disagreement equals the rerun disagreement. So
**N × K is what counts**, and extra runs are as good as extra tasks until the task pool runs out. Miller's other rules
apply:
- average the K samples per task first, then take the standard error across tasks (a pooled K·N error is inconsistent);
- use clustered standard errors when tasks share a source;
- do not lower the temperature just to reduce variance.

Wang ([arXiv 2512.21326](https://arxiv.org/abs/2512.21326)) finds that at natural temperatures (0.7–1) the paired
prediction noise exceeds the data noise. Averaging samples then buys most of the power: on HumanEval the detectable
difference falls from 11 % (unpaired, one sample) to 2–4 % (paired, averaged).

**Table B. Minimum detectable difference in points (80 % power, two-sided α 0.05) for N tasks × K independent runs (my
calc)**

| E[p(1−p)] | Benchmark it matches | N = 300 | N = 800 | N = 2,000 | N = 3,491 |
|---|---|---|---|---|---|
| 0.045 | BFCL v4 pool, greedy (projected) | 4.9 / 3.4 / 2.4 | 3.0 / 2.1 / 1.5 | 1.9 / 1.3 / 0.9 | 1.4 / 1.0 / 0.7 |
| 0.054 | Q30's 300 tasks, greedy | 5.3 / 3.8 / 2.7 | 3.3 / 2.3 / 1.6 | 2.1 / 1.5 / 1.0 | 1.6 / 1.1 / 0.8 |
| 0.08 | e.g. multi-turn or LiveCodeBench (assumed) | 6.5 / 4.6 / 3.2 | 4.0 / 2.8 / 2.0 | 2.5 / 1.8 / 1.3 | 1.9 / 1.3 / 1.0 |
| 0.12 | noisier sampled tasks (assumed) | 7.9 / 5.6 / 4.0 | 4.9 / 3.4 / 2.4 | 3.1 / 2.2 / 1.5 | 2.3 / 1.6 / 1.2 |

Each cell gives K = 1 / 2 / 4. For LiveCodeBench's 454 problems at E[p(1−p)] = 0.08: K = 1, 2, 4 and 8 give 5.3, 3.7,
2.6 and 1.9 points.

**Correlated repeats are worth less.** With within-task correlation ρ between runs, the variance of a K-run mean is
σ²(1 + (K−1)ρ)/K. Two same-GPU greedy runs (ρ ≈ 0.42) cut the variance to 0.71 σ², not 0.5 σ², so they are worth about
1.4 independent runs. Two runs on different GPU types (ρ ≈ 0.10) are worth 1.8. Seeds at temperature > 0 are
independent.

**Greedy or sampled.**
- **Greedy** is BFCL's protocol (its client sends temperature 0.001 for every model) and Q30's. It keeps 70 % of tasks
  fully stable, so its per-sample variance is the lowest available.
- **The makers' sampling** (temperature 1.0, top-p 0.95, top-k 20) is what Flint's model card asks users to run and what
  its own evaluations use. It measures the deployed condition, gives clean independent repeats, and avoids greedy
  repetition loops in long reasoning. Its per-sample variance is higher (claim to test).
- **The choice here:** greedy for BFCL, K = 2 on different GPU types; the makers' sampling with K = 4 for LiveCodeBench.
  An optional single sampled BFCL run per finalist reports the deployed condition in the model card.

### 2.3 Non-inferiority, not "no significant difference"

"Within noise of UD-Q4_K_XL" becomes a testable rule with a margin m. A candidate passes if the lower end of the
two-sided 90 % interval of (candidate − UD-Q4_K_XL) lies above −m. That is a one-sided test at α 0.05. A non-significant
difference at 300 tasks proves nothing, because such a test passes any file when the sample is small.

**Table C. Tasks × runs (N·K) needed to pass margin m with 80 % power when the files are truly equal (my calc)**

| E[p(1−p)] | m = 1 | 1.5 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| 0.045 | 5,565 | 2,474 | 1,392 | 619 | 348 | 223 |
| 0.054 | 6,678 | 2,968 | 1,670 | 742 | 418 | 268 |
| 0.08 | 9,893 | 4,397 | 2,474 | 1,100 | 619 | 396 |
| 0.12 | 14,839 | 6,595 | 3,710 | 1,649 | 928 | 594 |

**BFCL v4 single-turn, 3,491 tasks × 2 runs (my calc, E[p(1−p)] 0.045):**
- Paired standard error 0.36 points; 90 % interval ±0.59, 95 % ±0.71.
- Power to pass a 1.5-point margin: 0.99 if the files are equal, 0.87 if the candidate is truly 0.5 points worse.
- A 1-point margin would have 0.87 and 0.40.

### 2.4 Typical quantization deltas the margins must resolve

| Study | What was compared | Delta |
|---|---|---|
| Kurtic et al., "Give Me BF16 or Give Me Death" (500k evaluations, Llama 3.1) | HumanEval / HumanEval+ recovery | 99.9 % (8-bit), 98.9 % (4-bit); Arena-Hard intervals overlap for all formats |
| Giagnorio et al. (MultiPL-E, 20 samples, McNemar with Benjamini–Hochberg) | 4-bit drops | Not significant except one model–language pair; 3-bit and 2-bit drops significant |
| Liu et al., "Quantization Hurts Reasoning?" | LiveCodeBench, R1-Distill-Qwen-14B, BF16 50.9 | 49.6 (4-bit AWQ), 48.3 (4-bit GPTQ), 43.3 (3-bit AWQ). Losses grow with task difficulty and output length; low-bit models write longer outputs |
| Dutta et al., "Accuracy is Not All You Need" | Answer flips | Accuracy within 0–2 % while up to 13.6 % of answers flip (Llama2-7B GPTQ W4A16 on MMLU); flips concentrate on low-margin questions |

4-bit files of a good model typically lose 0.5–2.5 points on coding and single-turn tool tasks. A margin of 1.5 points
(single-turn tools) to 3 points (multi-turn and coding) sits at that scale.

## 3. Does KLD predict task deltas?

**For KLD as a coarse gate:**
- Dutta et al. (Llama2, Yi; GPTQ, AWQ, SmoothQuant, BnB): flips and KLD are "well correlated", Spearman 0.981 on MMLU.
  Both are recommended alongside accuracy.
- Across whole quant ladders KLD tracks scores: ρ −0.72 (28 Qwen3.6-35B-A3B quants) and −0.86 (41 Devstral-Small-2-24B
  quants) against a composite of HumanEval, LiveCodeBench v6, IFEval, BFCL v3, GSM8K and MMLU (arXiv 2606.19558).
- On Flint, the files KLD calls broken are broken: imatrix Q4_0 (KLD 0.565, perplexity +45 %, Q2) is unusable.
- On BFCL, decision margins before quantization predict flip rates after it, with a median error of 1.0 point (arXiv
  2608.06564).
- Unsloth builds its quants on KLD and top-1 agreement. For Dynamic 3.0 it added Divergence-300@32: 300 held-out
  prompts from Terminal-Bench 2.1, DeepSWE, Harbor and MathArena, with 32 greedy tokens compared against BF16. That is a
  trajectory-level KLD on agentic text.

**Against KLD as a ranking:**
- **Within the near-lossless zone the correlation is gone:** ρ = 0.00 among the 17 Qwen3.6 quants with KLD ≤ 0.064, and
  −0.24 (p 0.36) for Devstral. The cause: "KLD mainly tracks the volume of disagreement … not the direction". The authors'
  advice is that fidelity metrics "should filter damaged candidates, not rank near-baseline ones" (arXiv 2606.19558).
  Flint's finalists sit at KLD 0.105–0.127, about twice Flint's noise floor (0.052–0.065); whether that is inside
  Flint's silent zone is what the task lanes test (claim to test).
- **High KLD with intact task quality:**
  - The makers' Q4_1: KLD 0.322 against BF16 (0.675 on code text in terelabs' test), yet 83.0–83.7 % against BF16's
    80.7–81.0 % in Q30, and 81.0 % against 79.7 % for terelabs.
  - It was trained after quantization towards task quality, not towards BF16's distribution.
  - The same holds for any quantization-aware-trained file. KLD against BF16 cannot rank such files against files
    quantized from BF16.
- **Low KLD, damaged behaviour:**
  - Process failures can grow while the score stays flat. On τ²-bench, 4-bit agents' tool-name hallucinations rose up
    to 2.5× (+17.6 points per task) with no significant score change (arXiv 2607.27275).
  - Agentic categories move where single-turn ones do not (Red Hat's Gemma-4 NVFP4 card, section 6).
  - Serving stacks break tool calls independently of the weights. Moonshot's K2 Vendor Verifier (4,000 requests,
    2025-11-15) found 72–100 % schema-valid tool calls across providers serving the same model, and the official API
    agreeing with itself on only 76–84 % of tool-call decisions at temperature 0.6–1.0.

**Consequence for this project:** KLD ≲ 0.125 and same top token ≳ 87.5 % stays the screening gate (cheap, 1 GPU-hour
per file). The task protocol is the release check. It needs enough samples to resolve ~1.5 points on single-turn tool
calls and ~3 points on multi-turn and coding, and it must log process failures separately from accuracy.

## 4. Tool-calling benchmarks

All read 2026-10-23.

| Benchmark | Tasks | Metric | Second LLM | External services | Against llama-server | Licence | Use here |
|---|---|---|---|---|---|---|---|
| **BFCL v4 single-turn, Python** (`bfcl-eval` 2026.3.23) | 3,491: simple_python 400, multiple 200, parallel 200, parallel_multiple 200, live_simple 258, live_multiple 1,053, live_parallel 16, live_parallel_multiple 24, irrelevance 240, live_irrelevance 884, live_relevance 16 | AST match of name, arguments, types and allowed values; no call / a call for (ir)relevance | none | none | yes: the existing harness, raw prompts to `/completion` | Apache-2.0 | **lane E1** |
| **BFCL v4 multi-turn** | 800: base, miss_func, miss_param, long_context × the same 200 scenarios; 1–8 user turns (mean 3.7–4.7), ≤ 20 model steps per turn, 5.7 gold calls per task | backend state after every turn must match the ground truth, and the call results must contain the ground truth's (in any order) | none | none: GorillaFileSystem, TradingBot, Travel, Vehicle and four other Python backends | yes, with a Flint handler on BFCL's local-model handler class | Apache-2.0 | **lane E2** |
| BFCL v4 agentic: web_search (200), memory (465) | 665 | answer substring | none | SerpAPI key and live web; sentence-transformers | partly | Apache-2.0 | no: web results drift |
| BFCL Java / JavaScript single-turn | 150 | AST | none | none | yes | Apache-2.0 | no: not Flint's use |
| τ²/τ³-bench (`tau2-bench` 2.0.0, 2026-10-19) | airline 50, retail 114, telecom 114, banking 97, intake 200 | pass^k on database state × natural-language assertions | user simulator (GPT-4.1, GPT-5.5 in 2.0) plus judge models | none | only with a local simulator, which then sets the score | MIT | no |
| When2Call (NVIDIA) | 3,652 multiple choice, 300 judged | log-likelihood ranking, or a GPT-4o-class judge | judge for the generative mode | none | multiple choice needs prompt log-probabilities (llama-server refuses `echo`) | Apache-2.0 / CC-BY-4.0 | no; built from BFCL live anyway |
| ACEBench (English) | 1,023 (973 judge-free) | rule/AST checker | user simulator for 30 agent tasks | none | needs a converter from Flint's XML to `[Api(k='v')]` | MIT | optional second single-turn source |
| ToolHop | 995 multi-hop queries over 3,912 local tools | answer substring | none | none (tool code runs via `exec`, so sandbox it) | OpenAI `base_url` script; `max_tokens` 256 hard-coded | Apache-2.0 / CC-BY-4.0 | alternative to E2 |
| NESTFUL | 1,861 nested call sequences | F1, full-sequence accuracy | none | none | adapter needed | Apache-2.0 | no |
| Hammer suite (API-Bank L1/L2, NexusRaven, Seal-Tools, ToolAlpaca) | 1,252 | F1 with exact-string arguments | none | none | reformatted JSON schemas | mixed; NexusRaven non-commercial | no: label noise, string-exact |
| StableToolBench | 765 | pass/win rate | GPT-4 judge and API simulator | — | — | Apache-2.0 | no |
| ComplexFuncBench | 1,000 | — | GPT-4o judge | Booking.com RapidAPI | — | Apache-2.0 | no |
| MCP suites (MCP-Bench 104, MCP-Universe 231, MCPMark 127 of which ~76 local, LiveMCPBench 95, MCP-Atlas 500, Toolathlon 108, LiveMCP-101 101, MCPVerse 250) | 76–500 | judges, or programs against live services | mostly | API keys, live servers | partly | mostly Apache-2.0/MIT | no: ±3–5 points per run at best |

**BFCL details that matter for Flint:**
- **Data version.** v4 (2025-07-17) kept the v3 single-turn data, changed 14 entries (ground-truth and question fixes
  such as `live_simple_205-116-13`) and added 2 live_irrelevance tasks. It also changed about a third of the multi-turn ground truths, for example
  `multi_turn_base_154` (2025-10-01).
  - Porting the Q30 harness is mechanical: the package is renamed `bfcl` → `bfcl_eval`, the data files are now
    `BFCL_v4_*.json`, and simple becomes simple_python.
  - The v4 wheel (2026-03-23) is the current release; the leaderboard was last updated 2026-04-12.
- **Local endpoints.**
  - BFCL's OpenAI-compatible function-calling handler reads `OPENAI_BASE_URL` and sends no `max_tokens`. It hands
    `--jinja` tool parsing to llama-server.
  - The local-model handler renders the prompt on the client and posts it to `/v1/completions`, which is Q30's method.
    It caps generation at min(4,096, context − prompt) and needs the Hugging Face tokenizer files locally.
  - A known client bug: keep-alive connections give spurious HTTP 400 errors against llama-server (gorilla #1339,
    open). Disable keep-alive.
- **Scoring is deterministic.** Generation is not: llama-server's documentation says `cache_prompt` (on by default)
  can change logits because prompt and generation batches differ.
  - The variation is already part of the per-task variance measured in Q30.
  - What must stay fixed is that every file runs with the same server flags, the same batch sizes and a balanced mix of
    GPU types.

**llama.cpp and Flint's tool-call format** (master `36e36e6`, 2026-10-23):
- **How calls are parsed.** Flint writes `<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>` after
  a `<think>` block.
  - No chat handler in `common/` names Flint, so `--jinja` uses the differential autoparser.
  - With `tool_choice: auto` the server builds a lazy grammar, triggered at the call tag.
  - Once triggered, the grammar allows only declared function names and the schema's argument names.
- **Consequences for measurement:**
  - The `--jinja` path hides the model's habit of inventing argument names, which a community report saw without a
    grammar: 41 of 85 calls failed in OpenCode.
  - The raw-prompt lane (Q30's method, no grammar) measures the model itself and is the lane that compares quants.
  - A separate `--jinja` lane checks that each released file works through llama-server's OpenAI endpoint.
- **Open items:**
  - #31000: Flint NVFP4 in OpenCode writes tool-call XML inside its reasoning, so the turn ends with no call.
  - #31040 (draft): the optional-argument rule repeats without bound, so a model can loop on an `<arg_key>` until
    `max_tokens`.

  The raw decoder needs a fixed rule for calls inside `<think>`: count them as no call, and log them separately.

## 5. Coding benchmarks

| Benchmark | Tasks | Metric, defaults | Sandbox | Against llama-server | Noise per run (between models) | Use here |
|---|---|---|---|---|---|---|
| **LiveCodeBench code generation** | `release_v6`: 1,055 problems (2023-05 to 2025-04); 454 dated from 2024-08-01; median 40 hidden tests per problem (v6 increment) | pass@1; runner defaults n = 10, temperature 0.2, `max_tokens` 2,000, 6 s per test | CPU subprocess; run in a no-network container | OpenAI client with a base URL; register the model in `lm_styles.py`; send K separate requests instead of `n` | paired SE 1.1 (v6), 1.9 (from 2024-08-01) | **lane E3** |
| HumanEval+ / MBPP+ (EvalPlus) | 164 / 378 | pass@1, greedy | Docker image | `--backend openai --base-url`; the OpenAI backend caps output at 768 tokens unless patched | 3.9 / 2.4 | broken-file smoke test only |
| BigCodeBench | 1,140 full, 148 hard | pass@1, greedy, `max_new_tokens` 1,280 | 9.3 GB evaluation image | `--base_url` | not published | runner-up single-turn set |
| Aider polyglot | 225 exercises, 6 languages | pass@2 (second try sees the test output), share of well-formed edits | Docker | `OPENAI_API_BASE` | ~3.3 per run (my calc) | optional; Unsloth's quant ladders use it |
| SWE-bench Verified / Lite | 500 / 300 | resolved rate, agent harness (mini-SWE-agent: 27–87 model calls per instance) | Docker, ~100 GB of images | litellm | 1.9 / 2.6 | no: tens of millions of tokens per run |
| Terminal-Bench 2.0 / 2.1 | 89 | resolved rate, Terminus 2 in Harbor | Docker per task | litellm | 5.8 | no |
| CRUXEval, DS-1000 | 800, 1,000 | output prediction, data-science completions | CPU | yes | 1.6–1.8, 1.5 | no: not Flint's use |

**Why LiveCodeBench:**
- **It is the makers' own coding number.** The model card's chart gives BF16 80.3 and NVFP4 80.8, read at temperature
  1.0 / top-p 0.95 / top-k 20 as the average pass@1 of 5 attempts. That anchors the harness.
- **Losses concentrate on hard, long problems** in Liu et al., and LiveCodeBench has many of them.
- **It is judge-free and runs on CPU.**
- **The cost is reasoning length.** The runner's 2,000-token default must go: Flint's card asks for at least 32,768
  output tokens on hard problems.

The window choice:
- **2024-08-01 to 2025-04-30 (454 problems)** keeps cost and noise balanced.
- **All 1,055 problems** halve the noise per sample at 2.3 times the cost.
- **The 2025-02-01 to 2025-05-01 subset** (Qwen's "v6" convention, 131 public problems) is reported separately next to
  the makers' 80.3. Which window the makers used is not stated (claim to test).

No LiveCodeBench release postdates v6 (dataset last modified 2025-06-05), so nothing is newer than Flint's June 2026
knowledge cutoff.

## 6. What Flint's makers and the community report

- **Recursive** (model card and blog, 2026-10-21):
  - BF16 against NVFP4:

    | Benchmark | BF16 | NVFP4 |
    |---|---|---|
    | Terminal-Bench 2.1 | 54.9 | 53.7 |
    | SWE-bench Pro v2 | 89.7 | 88.5 |
    | SWE-bench Verified | 74.2 | 74.1 |
    | LiveCodeBench v6 | 80.3 | 80.8 |
    | IFBench | 89.2 | 88.1 |
    | IFEval | 95.1 | 94.6 |

    The blog's summary: "within 0.7 points of the 16-bit model" on average.
  - Settings: temperature 1.0, top-p 0.95, top-k 20 for every evaluation; LiveCodeBench and the IF benchmarks single-turn,
    average pass@1 over 5 attempts; agentic benchmarks at least 5 attempts with timeouts up to 5 hours.
  - Advice to users: at least 32,768 output tokens on hard problems, keep `reasoning_content` in the history.
  - Knowledge cutoff June 2026; licence Apache-2.0.
  - Not published: any tool-calling benchmark, and any score for the int4 Q4_1 GGUF. The card advises evaluating the
    exact checkpoint and runtime one deploys.
- **terelabs** (2026-10-21):
  - BFCL v3, 300 tasks, a simplified AST checker, greedy, 4,096 tokens, one slot, BOS off, llama.cpp PR #30928 at
    `1b44e30`.
  - Results: BF16 79.7 %, makers' Q4_1 81.0 %, UD-Q4_K_XL 80.0 %, other files 78.3–81.0 %, all ±4.5 points.
    McNemar p 0.29–1.0; 73 tasks unstable under greedy; a one-slot BF16 repeat matched 100 %.
  - KLD on code text (httpx, fd, cobra, axios; context 2,048, 40 chunks).
- **Unsloth:**
  - The Flint guide recommends UD-Q4_K_XL at the makers' sampling.
  - Quality practice: KLD and top-1 on held-out text, Divergence-300@32 for Dynamic 3.0, and Aider polyglot ladders for
    large models. DeepSeek-V3.1 pass@2 runs from 55.7 (IQ1_M) through 69.7 (Q4_K_XL) to 70.7 (Q5_K_XL), each run about 3
    times.
- **Community:**
  - A vLLM NVFP4 owner re-ran 45 LiveCodeBench and 50 SWE-bench Verified tasks (sample too small to compare).
  - A schema grammar fixed invented argument names in OpenCode.
  - No independent quant comparison of Flint on coding exists.

## 7. Protocol for Q41

### 7.1 Files

| Role | File | Lanes |
|---|---|---|
| Bar | Unsloth UD-Q4_K_XL (25.1 GB) | E1, E2, E3 in full |
| Candidates | S4-L0 (32.5 GB), S5 (32.6 GB), both rescale-free (`LLAMA_NO_MOE_DOWN_RESCALE=1`) | E1, E2, E3 in full |
| Anchor | Unsloth Q8_0 (42.3 GB; KLD 0.052, the noise floor) | E1, E2, E3 at K = 2 |
| Reference | BF16 (79.6 GB) | E1 only |
| Optional | makers' Q4_1 | E1 only, to settle the distillation question at 3,491 tasks |

### 7.2 Lanes

| Lane | Tasks | Decoding | Runs per file | Gate margin m |
|---|---|---|---|---|
| **E1** BFCL v4 single-turn, raw prompts, no grammar | all 3,491 Python tasks, minus any that fail the ground-truth self-test (list fixed before the runs) | greedy (temperature 0), `n_predict` 4,096, `cache_prompt` false | 2, on different GPU types | 1.5 points |
| **E2** BFCL v4 multi-turn, raw prompts, BFCL's executor and state checker | 800 (200 scenarios × 4) | greedy, ≤ 20 steps per turn, prompt cache on (same for every file) | 2, on different GPU types | 3 points |
| **E3** LiveCodeBench code generation, `release_v6` | 454 problems dated 2024-08-01 to 2025-04-30 | temperature 1.0, top-p 0.95, top-k 20, min-p 0, `max_tokens` 32,768, seeds 1–4 | K = 4 (anchor K = 2), half on each GPU type | 3 points |
| E1-T (report only) | the E1 tasks | makers' sampling, seed 1 | 1 | — |
| J (report only) | 1,000 stratified E1 tasks through `/v1/chat/completions` with `--jinja` and `tools` | greedy | 1 | — |

### 7.3 Analysis and pass/fail rule

**Per lane:**
1. **Per-task score:** each file's mean over its K runs. For E3 this is the mean pass over its samples.
2. **Paired difference** for each task: candidate minus UD-Q4_K_XL. Report the mean, its standard error (Miller Eq. 7),
   and the 90 % and 95 % intervals.
3. **Clustering:** E2 uses clustered standard errors by scenario (Miller Eq. 8). E1 checks clustering by the shared
   source key in live task IDs and uses the clustered error if it is more than 10 % larger.
4. **Cross-checks:** a paired bootstrap (10,000 resamples of tasks, each task with all its runs) and the exact McNemar
   test on the first run pair, for continuity with Q30.
5. **Noise floor:** E1 and E2 also report the discordance between each file's own two runs.

**Gate.** A candidate passes Q41 only if all of these hold:
- **Accuracy on every gated lane:** the lower 90 % bound of (candidate − UD-Q4_K_XL) is above −m.
  - All three lanes must pass, which is an intersection–union test, so no multiplicity correction is needed.
  - Each candidate is decided on its own.
- **Format and runaway safety**, pooled over E1 and E2:
  - The failures counted: undecodable output, unknown function, an argument name not in the schema, a call inside
    `<think>`, or output stopped by the token limit.
  - The upper 90 % bound of (candidate − UD-Q4_K_XL) in this failure rate must be ≤ +0.5 points.
  - On E3, the share of samples cut off at 32,768 tokens must be ≤ UD-Q4_K_XL's + 2 points.
- **Numerical safety:** no NaN or empty output in any run, with the exact build and environment of the release.

**BF16 and Q8_0 give context, not the gate.** Report every file's difference from BF16 (E1) and from Q8_0 (E1–E3) with
95 % intervals. If UD-Q4_K_XL itself falls below the anchor by more than a margin, record that as a finding. The decided
bar stays UD-Q4_K_XL.

**Claims in model cards:**
- A file is "better" than another on a lane only if the 95 % interval excludes 0.
- Otherwise the card says "within m points of UD-Q4_K_XL".

### 7.4 Running it on the 7-GPU quant machine

**GPU sets, one llama-server each:**

| Set | Cards | Memory | Holds |
|---|---|---|---|
| A | 2 × RTX 3090 | 48 GB | any 4-bit file, E3 at 8 slots |
| B | 2 × V100 32 GB | 64 GB | any 4-bit file or Q8_0 |
| C | RTX 3090 + 2 × RTX 3080 | 46 GB | a 4-bit file, E3 at 6 slots |
| BF16 | A + C | 94 GB | BF16, as in Q30, while set B keeps running |

- Keep BF16 off the V100s, which lack BF16 arithmetic; Q30 ran BF16 on the Ampere cards.
- The 3080s' x1 links slow model loading, not decoding.
- Q8_0 fits only set B.

**Containers:**
- One container per server, labelled `pascal-quants=1`, with `CUDA_DEVICE_ORDER=PCI_BUS_ID`.
- One llama.cpp build for every file: the build used for KLD, plus the rescale switch. Check that the switch is present
  in the binary (`strings`), because a build without the patch ignores the variable silently.
- Clients run in separate containers. LiveCodeBench's code execution runs in its own container with `--network none`,
  CPU and memory limits, after generation has finished.

**Server flags** (same for every file within a lane; claim to test for memory on set C):

```bash
# E1, E2: short prompts (E1 prompts ≤ ~10k tokens + 4,096 out); unified KV pool shared by 8 slots
llama-server -m <file>.gguf -ngl 99 -fa on -np 8 -kvu -c 131072 -b 2048 -ub 512 --cache-ram 0 -ts <split>
# E3: up to 32,768 output tokens per sample; makers' sampling as server defaults
llama-server -m <file>.gguf -ngl 99 -fa on -np 8 -kvu -c 278528 -b 2048 -ub 512 --cache-ram 0 -ts <split> \
  --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0
```

- Flint's KV cache is small: 14 full-attention layers at ~28.7 KB per token in 16 bits. So 8 × 34k tokens take ~7.8 GB,
  and the sliding-window layers ~0.3 GB (my calc).
- llama-server's default min-p is 0.05, so it is set to 0 explicitly. Setting the sampling on the server also covers
  clients that cannot send top-k and min-p.

**Harness work, done before any GPU time:**
1. **E1:** port [bench/bfcl_flint](../../../bench/bfcl_flint/) to the v4 wheel (module and file names, all 11 Python
   categories, a seed-free full task list). Rerun `selftest.py` on all 3,491 tasks and freeze the exclusion list. Give
   `run.py` per-run tags (file, GPU set, run number).
2. **E2:** a Flint handler on BFCL's local-model handler class.
   - It renders prompts with the GGUF's chat template and posts them to `/v1/completions`.
   - It decodes calls with `decode_flint` from E1, keeps `reasoning_content` as the template expects, and disables
     keep-alive.
   - Validate it by replaying the ground-truth calls of all 800 tasks through BFCL's executor and checker.
3. **E3:** LiveCodeBench's runner with a model entry in OpenAI-chat style and a base URL. It sends K separate requests
   with seeds 1–4 instead of `n`, at `max_tokens` 32,768.
   - Evaluate with the runner's `--evaluate` and score with `compute_scores --start_date 2024-08-01`, plus the
     2025-02-01 to 2025-05-01 subset.

**Order:**
1. **Pilot, ~3 server-hours:**
   - UD-Q4_K_XL through E1 twice (sets A and B);
   - 100 E2 tasks (set A);
   - 100 E3 problems × 2 seeds (set B).

   It gives the full-pool E[p(1−p)], E2's and E3's per-task variance, tokens per E3 sample, truncation and failure
   rates, and job times.
2. **Fix and record the design in the Q41 row before the main runs:**
   - Keep K and the margins if E1's E[p(1−p)] ≤ 0.06, E2's (× design effect) ≤ 0.12 and E3's ≤ 0.12.
   - Otherwise raise K for that lane (E2 to 3, E3 to 6), or widen the margin and say so.
   - Freeze the task lists by ID and sha256.
3. **Main runs:**
   - Each file's runs are split evenly between a 3090-based set and set B (crossover).
   - BF16 runs on its 5-card set while set B carries Q8_0.
   - Job times are kept only for scheduling; quality-machine timings are not speed results.

### 7.5 Cost (my calc, claim to test; the pilot's job times replace these)

**Assumptions** per llama-server with 8 slots:
- decode ~400 tok/s aggregate at E1/E2 lengths and ~300 tok/s at E3 lengths;
- prompt processing ~3,000 tok/s;
- E2 costs ~9 generations of ~300 tokens per task;
- E3 costs ~8,000 tokens per sample. DeepSeek-R1 averages ~8.2k completion tokens per Aider polyglot case; Flint's
  length is unmeasured.

| Lane | Work per run | Server-hours per run | Runs per file | Server-hours per file |
|---|---|---|---|---|
| E1 | 3,491 tasks: 0.57 M output, 1.7 M prompt tokens | 0.6 | 2 | 1.2 |
| E2 | 800 tasks: ~2.2 M output, ~8 M new prompt tokens | ~2.5 | 2 | ~5 |
| E3 | 454 samples: ~3.6 M output tokens | ~3.4 | 4 | ~13.6 |

| Files | Server-hours |
|---|---|
| UD-Q4_K_XL, S4-L0, S5: E1 + E2 + E3 (~20 each) | ~60 |
| Q8_0: E1 + E2 + E3 at K = 2 | ~13 |
| BF16: E1 on 5 cards | ~2.4 (~12 GPU-hours) |
| Pilot | ~3 |
| **Total** | **~78 server-hours ≈ 180 GPU-hours ≈ 26 h of wall time on three servers** |

- Per 4-bit file on a two-card set that is ~40 GPU-hours, of which LiveCodeBench is ~27.
- If Flint averages 15k tokens per E3 sample, E3 doubles. The fallback is K = 2 with a 4-point margin, plus a
  full K = 4 only for the candidate closest to failing.
- E1 alone, the cheapest decisive check, costs ~2.4 GPU-hours per file.

### 7.6 What to log

- **Per run, a manifest:**
  - the file, with its sha256;
  - the llama.cpp commit and binary checksum, and the environment (rescale switch);
  - the container image digest, GPU set and card types, every server flag;
  - the client harness commit, `bfcl-eval` version, LiveCodeBench commit and release;
  - the task-list sha256, the decoding parameters and seeds, and the date.
- **Per request:**
  - task ID, run tag and seed;
  - prompt and output token counts, stop type and truncation flag;
  - the full raw text including `<think>`, with the reasoning length;
  - the decoded calls or extracted code, the verdict and the checker's error type.
- **Per task, for the process metrics:**
  - undecodable output, unknown function, invented argument name, missing required argument, wrong type;
  - a call inside `<think>`, a call on an irrelevance task (and its reverse), runaway output;
  - for E3: no code extracted, compile or runtime error, timeout.
- **Per run, from the server log:** warnings, NaN or overflow messages, load-time tensor types.
- **Outputs:**
  - per-task CSVs with one row per task and run;
  - the analysis output (paired differences, intervals, bootstrap, McNemar, discordance, E[p(1−p)] per lane);
  - the gate verdict table.

  Raw outputs and server logs stay out of the git repository; the per-task CSVs go with the model cards.

### 7.7 What goes into the model cards (Q42)

- **Quality table:**
  - Rows: the file, UD-Q4_K_XL, Q8_0 and BF16.
  - KLD, same top token and perplexity against BF16 (wikitext-2, 80 × 512).
  - E1, E2 and E3 scores with 95 % intervals.
  - The paired difference from UD-Q4_K_XL with its 90 % interval and the margin verdict, e.g. "BFCL v4 single-turn:
    −0.3 points (90 % interval −0.9 to +0.3), within the 1.5-point margin".
- **How it was measured:**
  - task counts and runs per file, decoding settings and token limits;
  - `bfcl-eval` 2026.3.23 data, LiveCodeBench `release_v6` and its date window, the chat template's sha256;
  - the llama.cpp commit and the GPUs used. Quality ran on RTX 3090 and V100 cards; the rig's Pascal kernels change
    output only at summation-order level (Q38, Q46).
- **Noise statement:**
  - Two runs of the same file disagree on 5–10 % of tool-call verdicts.
  - Differences below ~1 point on BFCL single-turn, ~3 points on multi-turn, and ~2.5 points on LiveCodeBench are not
    resolvable at these sample sizes.
- **Process metrics:** the malformed-call rate, runaway rate and mean reasoning tokens against UD-Q4_K_XL.
- **Serving notes:**
  - `--jinja` lane parse-success rate.
  - Recommended flags: the makers' sampling with min-p 0, keep `reasoning_content`, and the rescale switch with the
    required build.
  - Open llama.cpp issues that affect tool use (#31000, #31040).
- **The per-task CSVs**, so anyone can redo the paired analysis against their own runs.

## Sources

Read 2026-10-23 unless a date is given.

Statistics and quantization evaluation (P):
- Miller, [Adding Error Bars to Evals](https://arxiv.org/abs/2411.00640) (arXiv 2411.00640, 2024-11-01): Eqs. 1–10
  (CLT, clustered and paired standard errors, resampling, sample size and minimum detectable effect), the advice against
  changing temperature.
- Wang, [Measuring all the noises of LLM Evals](https://arxiv.org/abs/2512.21326) (arXiv 2512.21326 v3, 2026-10-13) and
  its [table of standard errors](https://all-the-noises.github.io/main/index.html) (HumanEval, MBPP, LiveCodeBench,
  SWE-bench, Terminal-Bench, CRUXEval, DS-1000).
- Dutta et al., [Accuracy is Not All You Need](https://arxiv.org/abs/2407.09141) (arXiv 2407.09141, NeurIPS 2024): flips,
  KLD–flip correlation, MT-Bench drops.
- Nikolić et al., [Displacement Is Not Direction](https://arxiv.org/abs/2606.19558) (arXiv 2606.19558, 2026-06-17): KLD
  against downstream scores for 28 Qwen3.6-35B-A3B and 41 Devstral-Small-2-24B quants, the silent zone.
- [arXiv 2608.06564](https://arxiv.org/abs/2608.06564): decision margins and flips under quantization in 16 models;
  BFCL v4 by GGUF level for Qwen3-4B (Table 3), whether-to-call against which-tool flips.
- Jang et al., [Flat Score, Amplified Failures](https://arxiv.org/abs/2607.27275) (arXiv 2607.27275, 2026-07-29):
  τ²-bench under 4-bit quantization.
- Kurtic et al., [Give Me BF16 or Give Me Death](https://arxiv.org/abs/2411.02355) (arXiv 2411.02355) and Red Hat's
  [half a million evaluations](https://developers.redhat.com/articles/2024/10/17/we-ran-over-half-million-evaluations-quantized-llms)
  (2024-10-17).
- Giagnorio et al. (arXiv 2503.07103 v2, 2026-01): quantized code models on MultiPL-E with McNemar tests.
- Liu et al., [Quantization Hurts Reasoning?](https://arxiv.org/abs/2504.04823) (arXiv 2504.04823, COLM 2025).
- Red Hat, [gemma-4-26B-A4B-it-NVFP4 model card](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4): BFCL v4
  overall, single-turn, multi-turn and agentic against the base model.
- Moonshot AI, [K2 Vendor Verifier](https://github.com/MoonshotAI/K2-Vendor-Verifier) (tables dated 2025-11-15).
- Unsloth, [Dynamic 2.0 GGUFs](https://unsloth.ai/docs/basics/unsloth-dynamic-2.0-ggufs) (KLD, flips, Divergence-300@32,
  MMLU replication) and its Aider polyglot page for DeepSeek-V3.1; [Flint guide](https://unsloth.ai/docs/models/flint) (S).
- Intel [AutoRound](https://github.com/intel/auto-round) evaluates with lm-eval averages over ~10 log-likelihood tasks,
  which do not cover tool calling (S).

Benchmarks (P):
- BFCL: [repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) at `6ea5797`
  (CHANGELOG, `_llm_response_generation.py` temperature 0.001, `MAXIMUM_STEP_LIMIT = 20`),
  [`bfcl-eval` 2026.3.23 and 3.0.0 wheels](https://pypi.org/project/bfcl-eval/) (task counts above counted from the data
  files), [leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), gorilla issue #1339.
- [tau2-bench](https://github.com/sierra-research/tau2-bench) at `cc1df47` (CHANGELOG, task splits, simulator and judge
  defaults); [When2Call](https://github.com/NVIDIA/When2Call); [ACEBench](https://github.com/ACEBench/ACEBench);
  [ToolHop](https://huggingface.co/datasets/bytedance-research/ToolHop); [NESTFUL](https://github.com/IBM/NESTFUL);
  [Hammer](https://github.com/MadeAgents/Hammer); [StableToolBench](https://github.com/THUNLP-MT/StableToolBench);
  MCP-Bench (arXiv 2508.20453), MCP-Universe, MCPMark, MCP-Atlas, LiveMCPBench, Toolathlon, LiveMCP-101, MCPVerse
  repositories.
- [LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) (README release table, `lcb_runner/runner/parser.py`
  defaults) and the [code_generation_lite](https://huggingface.co/datasets/livecodebench/code_generation_lite) dataset;
  [EvalPlus](https://github.com/evalplus/evalplus); [BigCodeBench](https://github.com/bigcode-project/bigcodebench);
  [Aider polyglot](https://aider.chat/2024/12/21/polyglot.html) and leaderboard data;
  [SWE-bench](https://github.com/SWE-bench/SWE-bench), mini-SWE-agent; Terminal-Bench 2.0 (arXiv 2601.11868).

Flint and llama.cpp:
- [RecursiveSI/flint](https://huggingface.co/RecursiveSI/flint) model card and chart, and
  [Recursive's announcement](https://www.recursive.com/updates/introducing-recursive-flint) (2026-10-21) (P).
- [terelabs/flint-GGUF](https://huggingface.co/terelabs/flint-GGUF) README and `comparison.csv` (S).
- llama.cpp master `36e36e6` (P): `common/chat-auto-parser-generator.cpp` (lazy grammar), `tools/server/README.md`
  (`cache_prompt`, `min_p` default); [#31000](https://github.com/ggml-org/llama.cpp/issues/31000),
  [#31040](https://github.com/ggml-org/llama.cpp/pull/31040), #30928 (Flint support).
- Community reports in [quant_speed_d_community.md](quant_speed_d_community.md) (L3, L7) (S).

This repository:
- [quants/experiments.md](../../../quants/experiments.md): Q2, Q21, Q30, Q38, Q46; the quality bar.
- [bench/bfcl_flint](../../../bench/bfcl_flint/): the Q30 harness.
- [flint_speedups.md](flint_speedups.md): terelabs' comparison, Unsloth's guide.
