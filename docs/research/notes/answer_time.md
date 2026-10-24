# Answer time, not tokens per second: does quantization change how long Flint reasons, and which release candidate gives the shortest time to a correct answer (2026-10-24)

Scope: Flint always reasons, and about two thirds of what it writes is reasoning. A file that decodes 3 % faster but
reasons 3 % longer gains nothing, yet the release decision has so far weighed KLD and bytes per token separately.
This note:
- measures, paired by task, the reasoning, answer and total tokens per sample of G3 and K2h against Unsloth's
  UD-Q4_K_XL in Q41's task runs, with attempt-to-attempt noise as the floor;
- collects what is published on quantization changing reasoning length;
- turns the measured lengths and each file's bytes per token into a time per correct answer for single-turn tool
  calls, multi-turn agent episodes and coding, and says how large a length difference would flip the ranking of K2h,
  G3h, G3 and UD-Q4_K_XL;
- checks how the first-finish race (Q93), reflection suppression (Q94) and quant choices interact with length;
- proposes a release metric, the measurements Q41's remaining runs should add, and two cheap first experiments.

Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (papers, code, measurements in this repository) and secondary.
- Nothing ran on the rig or the quant machine. The tables come from Q41's raw outputs (kept outside the repository),
  read on a laptop with [bench/answer_time](../../../bench/answer_time/), standard-library Python. Reasoning tokens
  are counted with Flint's own tokenizer: the count reproduces llama-server's `tokens_predicted` exactly (up to the
  stop token) on 98.9 % of 19,419 samples, and within 1–3 tokens on all but 16.
- "UD" is Unsloth's UD-Q4_K_XL, the quality bar. Ratios are candidate over UD.

Starting point, measured ([registry](../../../quants/experiments.md)):
- **Q41's data at 08:07 on 2026-10-24**, the makers' sampling (T 1.0, top-p 0.95, top-k 20, min-p 0, reasoning kept):
  - E1, BFCL v4 single-turn: all 3,477 tasks, UD and G3 at attempts 1 and 3, K2h at attempt 1 (4,096-token cap);
  - E1 pilot: UD, 376 tasks × 5 attempts;
  - E2, BFCL v4 multi-turn: 199 tasks (the pilot's 100 plus 99 new scenarios), UD at attempt 1 on all and attempt 2
    on the pilot's 100, G3 at attempt 1 (4,096 tokens per step, ≤ 20 steps per turn);
  - E3, LiveCodeBench v6: eighth 1, 56 problems, UD and G3 at attempt 1, 32,768-token budget.
- **Pass@1 on those samples:** E1 UD 76.3 %, G3 76.6 %, K2h 76.7 %; E2 UD 36.2 %, G3 31.7 % (paired −5.0 points,
  90 % interval −11.2 to +1.2); E3 UD 85.7 %, G3 80.4 %, decided by three hard problems at the 32k budget.
- **Bytes per token:** UD 2.213 GB, G3 2.327, G3h 2.211, K2h 2.139, K1h 2.095 (Q73, Q77, the
  [long-context note](long_context_quality.md)). On the fork's v2 build UD decodes 37.26 tok/s on six cards
  (26.84 ms, Q76) and S4-L0 37.4 tok/s at 2.236 GB (26.8 ms).
- **Rig costs used below:** served nine-card chat 31.3 ms per token for S4-L0 against 26.8 ms in llama-bench (Q58);
  ~3.3 ms more per token for requests that carry tools (logits copy and host sampling, Q55, est.); prompt processing
  ~7.1 ms per new token plus ~2.2 s per request in llama-server today, est. ~2.0 ms and ~1.0 s after Q108–Q110
  ([prefill note](prefill_pascal.md)); +0.22–0.5 ms per token per 1,000 tokens of context (Q54).

## Verdict

- **Quantization at this level does not make Flint reason longer.** Paired by task, against UD-Q4_K_XL:
  - E1 single-turn, 3,477 tasks: G3 total tokens 1.000 (95 % interval 0.977–1.023), median per-task ratio 0.995
    (0.987–1.003); K2h 0.990 (0.955–1.032), median 0.991 (0.981–1.000). Reasoning alone: G3 1.001, K2h 1.004.
  - E2 multi-turn, 199 conversations: G3 0.957 (0.878–1.053), reasoning 0.949 (0.854–1.067), steps 0.958.
  - E3 coding, 56 problems: G3 1.033 (0.948–1.124), median 1.003 (0.978–1.111); hard problems 1.051
    (0.983–1.134).
  - **The noise floor is as large as any of these.** UD's attempt 3 against its attempt 1 gives 1.027 (0.991–1.064)
    on E1, and G3's 0.966. The "189 against 183 tokens" of Q41's first E1 pair was attempt-to-attempt noise: with
    attempt 3 added the means are 186 and 186.
  - Hesitation words ("wait", "but", "maybe", ...), the tokens that quantization noise pushes up in 3-bit reasoning
    models, are level too: 0.99 (G3), 1.03 (K2h) on E1, 0.98 on E2, 0.97 on E3.
- **The literature agrees once the quant is gentle.** Length inflation tracks quantization damage, not bit width.
  - Qwen3-30B-A3B (MoE, 3B active) at INT4: GPTQ +3.7 % reasoning tokens, HQQ +3.9 %, AWQ +14.4 %, round-to-nearest
    +43 %; INT3 +19–60 %; BFCL v4 multi-turn with GPTQ INT4 +2.6 %.
  - DeepSeek-R1 distills at 4 bits: −8 % to 0 % at 14B and QwQ-32B, 0 to +3 % for Llama-8B, up to +23 % at 1.5B
    and 7B; 3-bit AWQ +355 % on the 1.5B model.
  - Our candidates are 5–8-bit mixes at the same KLD as UD (0.10–0.12), and on the tokens Flint writes their KLD is at
    or below UD's (Q106). Neither the published mechanism nor our data predicts a length difference between them.
- **Time per correct answer separates the candidates by only 2–3 %, and pass rate, not length, dominates it.**
  - Per served token the candidates differ from UD by G3 +1.6–1.8 %, G3h −0.6 %, K2h −1.9 to −2.2 %, K1h −2.8 to
    −3.1 % (est., from bytes).
  - Prefill and per-request costs dilute that in agent episodes: decode is ~49 % of an E2 episode's time today, ~74 %
    after Q108–Q110 (est.).
  - One point of pass@1 is worth 1.3 % of time per correct answer on E1, 2.7 % on E2 and 1.2 % on E3. Q41's
    non-inferiority margins (1.5 / 3 / 3 points) therefore allow 2.0 / 8.2 / 3.5 % more time per correct answer than
    UD, more than any candidate's speed edge.
- **Ranking at equal pass rates, measured lengths (est.):** K1h < K2h < G3h < UD < G3 on single-turn and
  multi-turn work. On coding, with G3's +3.3 % length borrowed for all our files, UD comes first (K1h 1.008,
  K2h 1.016). On E1 K2h's time per correct answer is 0.981 (0.957–1.007) of UD's with its own measured length and
  pass rate. With measured pass rates G3 loses on E2 (1.12) and E3 (1.13), both inside its quality intervals.
- **Break-even lengths.** UD's reasoning would have to be ≥ 1.9–2.2 % shorter than K2h's for UD to win (K1h: 2.8–3.1 %;
  G3h: 0.6 %); G3 must reason ≥ 1.6–1.8 % less than UD to tie it. Between candidates, 0.9 % flips K1h and K2h and
  1.3–1.6 % flips K2h and G3h. Today's lanes resolve ±2.3 % (E1 at K = 2) to ±9 % (E2, E3) on mean length, so none of
  these flips is settled by data yet; E1's medians (±1 %) put K2h at or below UD.
- **The lead does not change, with one condition.** K2h stays the lead: the only candidate with a measured time per
  correct answer below UD's, and no sign of longer reasoning. The condition is the head:
  - The screened head (Q92) is exact and is the planned release engine's head, but it works for UD's Q8_0 head too.
  - Timed on the same build, UD then reads 2.060 GB per token, K2 with a screen 2.102 and K1 2.058 (est.). K2h's
    per-token edge over UD falls from ~2 % to nothing (K2 + screen takes 1.003× UD's time), K1h's from ~3 % to
    0.6 %.
  - So the release claim must compare files with the same head treatment. On a screened-head build the K family's
    case rests on quality and length, not bytes.
- **Levers on length are per answer, not per file.** The first-finish race (Q93) cuts reasoning to 0.74–0.75 of the
  mean at k = 2 for UD and G3 alike, because their spread is the same (within-task coefficient of variation 0.26).
  - Runaways repeat on the same tasks: 5 of UD's 6–9 E1 runaways recur across attempts, and 6 of the 7–8 problems cut
    at 32k are the same for UD and G3. A race removes only a fifth to a half of runaway cost (est.).
  - Failed samples are long: 1.2× passed ones on E1, 1.45× on E2, 3.2–3.6× on E3. 35–47 % of E3's tokens go into
    failures, and short-first selection attacks exactly those.
- **First experiments:**
  1. Read the probability of `</think>` where reasoning ends from Q106's per-token dumps: no GPU, minutes. It tests
     whether any candidate shifts Flint's stop decision.
  2. Run the ten E3 problems decided near the 32k budget at 65,536 tokens, K = 4, UD against K2h: ~3–4 GPU-hours.
     It tells whether coding length differs between files once the budget no longer cuts.

## Measured tables

All tables: paired by task, each file's mean over its attempts first; 95 % intervals from a bootstrap that resamples
clusters of tasks (BFCL live sources, multi-turn scenarios, LiveCodeBench contests). "Ratio of means" divides the sums
of the per-task means. "Median ratio" is the median per-task ratio over equal numbers of attempts: a single draw
against a two-attempt mean biases a median of ratios low, because single draws are more skewed than means.
"Runaway" samples stopped at the token cap or never closed their reasoning.

### Per file

| Lane | File | Samples | Mean total tokens | Reasoning | Answer | Reasoning share | Median | p90 | p99 | Runaway |
|---|---|---|---|---|---|---|---|---|---|---|
| E1 | UD | 6,954 | 186 | 122 | 64 | 0.657 | 129 | 304 | 1,040 | 0.22 % |
| E1 | G3 | 6,954 | 186 | 122 | 64 | 0.658 | 128 | 296 | 1,084 | 0.19 % |
| E1 | K2h | 3,477 | 184 | 123 | 61 | 0.666 | 128 | 291 | 982 | 0.23 % |
| E1 pilot | UD (5 attempts) | 1,880 | 195 | 125 | 71 | 0.638 | 139 | 313 | 1,050 | 0.11 % |
| E2 (per conversation) | UD | 299 | 2,386 | 1,839 | 548 | 0.771 | 1,819 | 4,169 | 12,732 | 5.0 % |
| E2 (per conversation) | G3 | 199 | 2,311 | 1,770 | 541 | 0.766 | 1,701 | 4,208 | 10,402 | 4.0 % |
| E3 | UD | 56 | 12,876 | 12,650 | 226 | 0.982 | 11,131 | 32,768 | 32,768 | 12.5 % |
| E3 | G3 | 56 | 13,304 | 13,074 | 230 | 0.983 | 10,408 | 32,768 | 32,768 | 14.3 % |

- E2 conversations take 12.0 (UD) and 11.5 (G3) model steps of ~200 tokens each, 153 of them reasoning. The first
  prompt is ~5,300 tokens, and the context reaches a median ~8,200.
- E3's "answer" is the code after `</think>`. Above p90 every E3 quantile is the 32,768-token budget.

### Paired length ratios

| Lane | Pair | Tasks | Total: ratio of means | Total: median ratio | Reasoning: ratio of means | Reasoning: median ratio | Answer: ratio of means | Total without runaways |
|---|---|---|---|---|---|---|---|---|
| E1 | G3 / UD | 3,477 | 1.000 (0.977–1.023) | 0.995 (0.987–1.003) | 1.001 (0.969–1.032) | 0.996 (0.986–1.010) | 0.997 (0.972–1.031) | 1.007 (0.986–1.031) |
| E1 | K2h / UD | 3,477 | 0.990 (0.955–1.032) | 0.991 (0.981–1.000) | 1.004 (0.953–1.067) | 1.000 (0.981–1.010) | 0.962 (0.941–0.985) | 0.984 (0.954–1.018) |
| E1 | noise: UD attempt 3 / 1 | 3,477 | 1.027 (0.991–1.064) | 1.000 (0.985–1.007) | 1.021 (0.970–1.076) | 0.989 (0.973–1.000) | 1.038 (1.008–1.071) | 1.013 (0.982–1.045) |
| E1 | noise: G3 attempt 3 / 1 | 3,477 | 0.966 (0.924–1.007) | 0.992 (0.982–1.000) | 0.952 (0.895–1.008) | 0.985 (0.971–1.000) | 0.991 (0.939–1.037) | 0.971 (0.934–1.010) |
| E1 | G3 / UD, attempt 1 only | 3,477 | 1.031 (0.993–1.077) | 1.000 (0.989–1.007) | 1.036 (0.984–1.095) | 1.000 (0.989–1.013) | 1.021 (0.977–1.079) | 1.018 (0.984–1.052) |
| E1 pilot | noise: UD attempt 2 / 1 | 376 | 1.062 (0.964–1.175) | 1.016 (0.990–1.050) | 1.107 (0.959–1.283) | 1.044 (1.000–1.121) | 0.988 (0.931–1.050) | 1.062 (0.964–1.174) |
| E2 | G3 / UD | 199 | 0.957 (0.878–1.053) | 0.941 (0.854–1.006) | 0.949 (0.854–1.067) | 0.900 (0.841–0.970) | 0.984 (0.930–1.040) | 0.940 (0.861–1.022) |
| E2 | noise: UD attempt 2 / 1 | 100 | 0.985 (0.902–1.082) | 0.920 (0.861–0.997) | 0.990 (0.893–1.114) | 0.947 (0.883–1.034) | 0.970 (0.867–1.081) | 1.017 (0.923–1.128) |
| E3 | G3 / UD | 56 | 1.033 (0.948–1.124) | 1.003 (0.978–1.111) | 1.034 (0.947–1.125) | 1.003 (0.966–1.142) | 1.018 (0.861–1.216) | 1.032 (0.897–1.173) |

- **E1 pins G3 to UD within ±2.3 %** on means and ±0.8 % on medians. Its two attempts differ from each other by as
  much as it differs from UD.
- **K2h writes 4 % shorter answers** (0.962, 0.941–0.985; 0.981 at attempt 1 only), with as much reasoning as UD.
  It also had fewer malformed answers (15 against 25) and fewer second `</think>` tags (18 against 32), which add text
  after the first close.
- **E2's medians are noisy.** G3's reasoning median of 0.900 has an interval that excludes 1. So does UD's own
  attempt 2 against attempt 1 (total median 0.920), with ~25–50 scenarios per cluster resample. G3 is not longer on
  multi-turn tasks and may be shorter. Per user turn (841 paired turns) the ratios are the same: total 0.954,
  reasoning 0.946, steps 0.954 (0.907–1.000).
- **E3 has no noise floor yet (K = 1),** but the task bootstrap includes the sampling noise of each problem's single
  draw. Without runaways, G3/UD is 1.032 (0.897–1.173). On the 47 problems neither file cut, the median is 1.024.

### By category (total tokens, ratio of means)

| E1 category | Tasks | G3 / UD | K2h / UD |
|---|---|---|---|
| simple_python | 400 | 0.985 (0.953–1.019) | 1.008 (0.979–1.042) |
| multiple | 200 | 0.983 (0.949–1.016) | 0.967 (0.923–1.012) |
| parallel | 200 | 0.900 (0.838–0.955) | 0.912 (0.846–0.979) |
| parallel_multiple | 200 | 1.002 (0.928–1.081) | 0.916 (0.839–0.987) |
| live_simple | 255 | 0.996 (0.931–1.067) | 1.008 (0.915–1.087) |
| live_multiple | 1,042 | 0.998 (0.952–1.051) | 1.018 (0.968–1.086) |
| live_parallel / live_parallel_multiple | 16 / 24 | 1.280 / 0.944 | 1.260 / 1.019 |
| irrelevance | 240 | 1.057 (0.997–1.130) | 1.146 (0.989–1.370) |
| live_irrelevance | 884 | 1.007 (0.967–1.060) | 0.961 (0.895–1.058) |
| live_relevance | 16 | 1.101 | 0.908 |

| E2 category | Tasks | G3 / UD | | E3 difficulty | Problems | G3 / UD | Medians UD → G3 | Cut at 32k (UD / G3) |
|---|---|---|---|---|---|---|---|---|
| base | 51 | 0.885 (0.729–1.054) | | easy | 13 | 1.067 (0.491–1.423) | 1,075 → 707 | 0 / 0 |
| long_context | 55 | 0.902 (0.754–1.099) | | medium | 15 | 0.945 (0.613–1.224) | 6,082 → 3,764 | 0 / 0 |
| miss_func | 46 | 1.102 (0.911–1.347) | | hard | 28 | 1.051 (0.983–1.134) | 17,814 → 21,428 | 7 / 8 |
| miss_param | 47 | 0.889 (0.778–1.022) | | | | | | |

- **No category stands out beyond what ~20 comparisons produce by chance.** Both candidates are 9–10 % shorter on
  parallel. K2h's irrelevance interval nearly excludes 1 (reasoning 1.29, 1.01–1.71, on 240 tasks).
- **E2's miss_param, where G3 lost pass points, is not longer** (0.889). G3's whether-to-call misses are not
  overthinking.

### Where the difference sits, and failures against passes

| Lane | Pair | Mean difference (tokens per task) | from tasks both always pass | both never pass | split verdicts | tasks with a runaway | Ratio on tasks both always pass |
|---|---|---|---|---|---|---|---|
| E1 | G3 − UD | −0.1 | +1.5 | −1.4 | −0.2 | +0.8 | 1.015 (2,140 tasks) |
| E1 | K2h − UD | −1.9 | −1.3 | −0.7 | +0.1 | +0.7 | 0.988 (2,261) |
| E2 | G3 − UD | −103 | −21 | −38 | −44 | +73 | 0.927 (37) |
| E3 | G3 − UD | +428 | +237 | +2 | +189 | +178 | 1.035 (45) |

The "runaway" column overlaps the pass classes.

| Lane | File | Mean tokens, passed | failed | failed / passed | Share of all tokens in failed samples | in runaway samples |
|---|---|---|---|---|---|---|
| E1 | UD / G3 / K2h | 176 / 178 / 174 | 218 / 213 / 216 | 1.24 / 1.20 / 1.24 | 0.28 / 0.27 / 0.27 | 0.05 / 0.04 / 0.05 |
| E2 | UD / G3 | 1,845 / 1,764 | 2,701 / 2,565 | 1.46 / 1.45 | 0.70 / 0.76 | 0.19 / 0.13 |
| E3 | UD / G3 | 9,801 / 8,800 | 31,325 / 31,729 | 3.20 / 3.61 | 0.35 / 0.47 | 0.32 / 0.35 |

- **Where E3's +3 % sits:** in problems both files solve (+237 tokens per problem) and in split ones (+189), not in
  failures shared by both.
- **Shared failures are budget-bound.** All of UD's E3 failures but one, and all of G3's but three, are 32k cuts,
  and 6 of the 7–8 cut problems are the same for both files. In E1, 5 of UD's runaway tasks recur at attempts 1 and 3.
- **Runaways are properties of tasks** (loops or hard problems), not of files.

### Hesitation markers per 1,000 reasoning tokens

| Lane | UD | G3 | K2h | G3 / UD | K2h / UD | Noise (UD attempt b / a) |
|---|---|---|---|---|---|---|
| E1 | 11.66 | 11.51 | 12.03 | 0.987 (0.924–1.055) | 1.032 (0.946–1.122) | 0.998 (0.928–1.080) |
| E2 | 18.63 | 18.27 | — | 0.981 (0.854–1.158) | — | 0.917 (0.856–1.050) |
| E3 | 22.16 | 21.54 | — | 0.972 (0.898–1.050) | — | — |

- **Markers counted** (word-initial): wait, but, alternatively, perhaps, maybe, however, hmm, actually, reconsider,
  backtrack, wrong, double-check, recheck, verify, mistake, oops, hold on. This is the class that arXiv 2606.00206
  finds 2–4× more likely among the top-20 candidates at high-entropy positions in quantized models.
- **The reflection words alone** (wait, hmm, actually, alternatively, ...) run 1.8–1.9 per 1,000 reasoning tokens on
  E1, 2.9–3.4 on E2 and 4.0–4.3 on E3. They are level between files except E3, where G3 is 1.05 (1.006–1.096). That is
  one interval of ~10 and a 5 % shift of a 4-per-1,000 rate.

## Ranked levers and decisions

Effects are on the time to a correct answer (est.), against UD-Q4_K_XL on the same build unless stated.

| # | Lever or decision | Effect on time to answer (est.) | Evidence | First experiment | Kill criterion |
|---|---|---|---|---|---|
| 1 | **Decide the release on time per correct answer (TPCA), with a length gate**, not on KLD and bytes separately | Prevents trading 2–3 % of per-token speed for up to 8 % of TPCA inside the E2 quality margin (1 point of pass@1 = 2.7 % of E2's TPCA) | Measured E2: G3's −5 points cost +12 % TPCA (0.92–1.40) while its length is 0.96 | `lengths.py --time` on every Q41 pair as it lands | None: it is a reporting rule |
| 2 | **Compare against UD with the same head treatment** (screened head on both, or neither) | The screened head takes UD from 26.84 to ~25.8 ms per token (−3.8 %), exact for top-k 20; K2h's edge falls from −2.0 % to +0.3 % per token, K1h's from −2.8 % to −0.6 % | Q91 (exact on all 40,527 held-out rows), Q92 (identity checks on G3 and UD) | Rig: UD with and without the screen, K2h, K2 + screen, one session | — (arithmetic); if UD + screen ≥ K2 + screen in tok/s, the K family ships on quality only |
| 3 | **First-finish race at k = 2 (Q93)** | Reasoning 0.74–0.75 of the mean for every file; per answer 1.10–1.23× faster ([unconventional ideas](unconventional_ideas.md)); TPCA a little better than that, since short samples pass more often (+0.24 to +0.40 points on E1) | E1 3,477 tasks × 2: shortest/mean reasoning 0.740 (G3), 0.748 (UD); within-task CV 0.26 for both; pilot 0.64 at k = 3 | E3 at 64k with K = 4 (experiment 2 below) gives the coding value | Shortest of 2 ≥ 0.85 of the mean on E3 at 64k |
| 4 | **K2h as the release file** (G3h quality fallback, K1h speed option) | TPCA 0.981 (0.957–1.007) on E1 measured; est. 0.97 on E2 and 1.0–1.02 on E3 at equal pass rates (G3's length ratios borrowed) | Bytes 2.139 against 2.213 GB; E1 length 0.990 and median 0.991; marker density level | Rig speed of K2h and UD on `pascal-p106-v2` (planned); K2h on E2 at K ≥ 1 | Rig per-token ratio ≥ 0.99, or K2h's E2 length ratio upper 90 % bound > 1.03 |
| 5 | **Prefill fixes (Q108–Q110) and exact reuse of the generated turn** | E2 episode est. 183 → 120 s (−35 %). Exact reuse alone −9 % today (184 → 168 s), since every generated token is otherwise re-read once at 7.1 ms (~20 % on top of its decode time) | This note's E2 replay with both cache policies; [prefill note](prefill_pascal.md) | Already on the rig runbook (step 5) | — (covered there) |
| 6 | **E3 at 65,536 tokens; no budget below 32k** | At 32k, 12.5–14.3 % of samples are cut, all of G3's E3 deficit sits in three problems at the budget, and UD passes three problems with 23–32.6k tokens. A tighter budget would cut passes | E3 eighth 1 | Experiment 2 below | G3/K2h runaway count at 64k within +2 of UD's on 40 samples |
| 7 | **Quant choices aimed at length** (head precision, global-layer attention, GPTQ body) | None expected: every candidate's KLD on written tokens is at or below UD's (G3 0.887, G3h 0.957, K2h 1.013, K1h 1.038), and Q107's global-layer Q8_0 moves neither read nor written ratios | Q106, Q107; literature mechanism (KL at high-entropy written positions) | Experiment 1 below (p(`</think>`) from Q106's dumps) | Δ log p at `</think>` within ±0.05 nats for every candidate: close the question |
| 8 | Hesitation-marker penalty (the overthinking variant of Q94) | ≤ 5 % shorter reasoning at most (BF16 distills −5.6 to −10 %), and it changes the text. Flint has no quant-induced excess to remove | Marker density level across files (above); Q94's 2.4 reflection words per 1,000 tokens | Not pursued; the marker ratio stays as a screen for new files | Marker ratio > 1.10 with interval above 1 flags a new file before task runs |
| 9 | Precision by token role (unconventional ideas, idea 7: reasoning through Q4_1 copies) | Risk raised: INT4 round-to-nearest inflates reasoning +43 % on Qwen3-30B-A3B; reasoning at S1's KLD (0.219) is that regime | arXiv 2606.25519 Table 2 | Not before a task check | Its own kill criterion (reasoning ≥ 10 % longer) is likely to fire (est.) |

## 1. Length, paired by task

### 1.1 How it was measured

- **Data and counting.** [lengths.py](../../../bench/answer_time/lengths.py) reads each data folder's raw llama-server
  outputs and per-attempt verdict tables (the formats of [bfcl_flint](../../../bench/bfcl_flint/) and
  [livecodebench_flint](../../../bench/livecodebench_flint/)).
  - It counts reasoning as the tokens from `<think>` to the last `</think>` (an unclosed block counts as reasoning).
  - Answer tokens are the rest of llama-server's `tokens_predicted`, the stop token included.
  - Multi-turn conversations are counted per step, per user turn and per conversation, with each step's new prompt
    tokens.
- **The tokenizer.** [flint_tokens.py](../../../bench/answer_time/flint_tokens.py) is Flint's byte-level BPE in
  standard-library Python, with Unicode classes built from `unicodedata`. On E1 and E3, `tokens_predicted` minus the
  count is exactly 1, the stop token, on 98 % of samples. It is 0 on samples stopped by the cap and 2–3 on the few
  with a non-canonical token split.
- **E3's reasoning needed recounting.** The LiveCodeBench harness's `reasoning_chars` is 0 for samples cut before
  `</think>`, so the CSV column undercounts exactly the runaways.
- **Pairing and intervals.** Each file's per-task mean comes first, as [paired_tasks.py](../../../bench/paired_tasks.py)
  does for pass@1. Bootstraps resample clusters of tasks (2,000 resamples).
- **Noise floors** are one file's attempt b against its attempt a, analysed as two files: E1 UD 1 and 3, G3 1 and 3,
  the pilot's UD 1 and 2; E2 UD 1 and 2.

### 1.2 What it shows

- **Single-turn tool calls (E1).**
  - **Length.** Equal for G3 and UD in mean, median, reasoning and answer, by category and in every class of task;
    K2h equal or slightly shorter.
  - **Tails.** p90/p95 ratios are 0.96–0.98 for both candidates, p99 within noise (G3 1.04, 0.89–1.19).
  - **Runaways.** 0.16–0.22 % at the 4,096-token cap for all three files.
  - **Why means are noisy.** A 4,096-token runaway is 22 average samples, so 3 runaways more or fewer in 3,477
    samples move a mean by 2 %. This is why E1's ratio of means is ±2.3 % wide at K = 2 while its median is ±0.8 %.
- **Multi-turn agent episodes (E2).**
  - **Length.** G3 writes 4 % fewer tokens over a conversation (0.957, 0.878–1.053) in 4 % fewer steps. Reasoning
    and answers per step are unchanged (153 and ~47 tokens).
  - **Prompts.** New prompt tokens per conversation are equal (exact-reuse count 1.011, 0.983–1.047).
  - **Runaways.** Steps at the 4,096-token cap: 3.7 % of UD's conversations, 3.0 % of G3's.
  - **The pass gap** (−5.0 points) has no length signature: miss_param, where G3 loses, is 11 % shorter.
- **Coding (E3).**
  - **Length.** +3.3 % for G3 (0.948–1.124), concentrated in hard problems (1.051) and in problems both solve.
  - **Why the pass rate moved.** Under a 32k budget the three discordant hard problems flip on a few thousand tokens
    (G3 cut at 32,768 where UD finished at 32,144 and 23,005; G3 wrong at 32,763). Length differences that are not
    significant still decide passes there. Q71's decision to compare files only at 65,536 tokens is the right one.
- **Noise.**
  - UD against itself moves the E1 ratio of means by 2.7 %, G3 against itself by 3.4 %, and the pilot's UD by 6.2 %.
  - Nothing measured for G3 or K2h exceeds that, and no interval of a candidate's ratio of means excludes 1.
  - The one candidate interval that excludes 1 on any measure (E2 reasoning median) has a same-file counterpart that
    does too.

### 1.3 How much data would resolve the break-even differences (est.)

- **Scaling.** The 95 % half-width of the ratio of means scales as 1/√(tasks × attempts) (Miller's rule, as in the
  [task-level note](task_level_quality_checks.md)).
- **Today:** E1 ±2.3 % at 3,477 × 2, E2 ±9 % at 199 × 1, E3 ±9 % at 56 × 1.
- **Q41's decision run would reach:** E2 at 799 × 2 ±3.2 %; E3 at 454 × 4 (64k, higher variance per sample) est.
  ±2–3 %. E1 at K = 4 would reach ±1.6 %.
- **Medians are 3× tighter on E1** (±0.8 %), because runaways don't move them. Medians alone hide tail cost, which is
  real time, so both are reported.
- **Consequence.** A 2 % length difference between K2h and UD is resolvable on E1 medians now and on E1 means at
  K ≥ 4, but not on E2 or E3 at any size Q41 can afford. Hence the length gate in section 5 is a non-inferiority
  margin, not a detection test.

## 2. Literature: quantization and reasoning length

| Study | Models, method | Length change against BF16 | Accuracy |
|---|---|---|---|
| Liu et al., "Quantization Hurts Reasoning?", COLM 2025 (§4.1, Fig. 3, App. B.3) | DeepSeek-R1-Distill-Qwen/Llama 1.5–70B, QwQ-32B, Qwen3-8B; AWQ, GPTQ, FlatQuant, KV and activation quantization | W4 g128, W8A8KV8 and KV4 "closely align" with BF16; W3 g128 and W4A4KV4 produce longer outputs, most in the 1.5B model; harder problems are longer for every precision but quantization does not lengthen them more | W8A8 and W4A16 near lossless; LiveCodeBench, R1-Distill-Qwen-14B: BF16 50.9, AWQ W4 49.6, GPTQ W4 48.3, AWQ W3 43.3 |
| Lotfi, Kirichenko, Li, Liu, "Quantized Reasoning Models Think They Need to Think Longer, but They Do Not", 2026 (Table 2) | R1 distills 1.5B, 7B, 14B, Llama-8B, QwQ-32B; GPTQ, AWQ W4/W3, FlatQuant W4A4KV4; MATH-500 CoT tokens | 1.5B: 5,156 → +20 % (GPTQ W4), +23 % (AWQ W4), +32 % (GPTQ W3), +355 % (AWQ W3). 7B: +16 / +1 / +23 / +23 %. 14B: −8 / −3 / −3 / +15 %. Llama-8B: 0 / +3 / +34 / +16 %. QwQ-32B: −1 / 0 / +9 / +3 %. 1.5B AWQ W3 on LiveCodeBench 19,066 → 38,793 | MATH-500 at W4 within 0.6 points of BF16 except the 1.5B model (−2.6 to −2.8); 1.5B AWQ W3 85.6 → 47.0 |
| Same paper, mechanism | position-level KL against the BF16 model | KL tracks the BF16 model's next-token entropy (ρ 0.92). Hesitation markers are 2–4× more likely among the top 20 at high-entropy positions. Up to 52 % of quantized failures reach the right answer and keep thinking. A 50-marker logit penalty shortens CoT by 12–23 % (quantized) and 5.6–10 % (BF16) at kept or better accuracy | Length increase tracks accuracy loss across 28 model–quantization pairs (Spearman magnitude 0.73) |
| "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models", 2026 (Tables 1–3) | Qwen3-4B-Thinking-2507 (dense), **Qwen3-30B-A3B-Thinking-2507 (MoE, 3B active)**; weight-only g128 INT4/INT3 with RTN, AWQ, GPTQ, HQQ, ParoQuant; greedy, 100 questions per benchmark | Qwen3-30B-A3B reasoning-token inflation, mean over six benchmarks: INT4 GPTQ +3.7 %, HQQ +3.9 %, AWQ +14.4 %, RTN +43.0 %; INT3 GPTQ +19.0 %, AWQ +32.8 %, HQQ +60.1 %; LiveCodeBench (15,374 tokens in BF16): GPTQ −0.1 %, AWQ +11.0 %, RTN +82.7 %. BFCL v4 multi-turn, GPTQ INT4: 5,351 → 5,488 (+2.6 %), INT3 6,611 (+23.5 %). Qwen3-4B: INT4 +4.7 % (ParoQuant) to +42.5 % (RTN); BFCL multi-turn GPTQ INT4 +13.3 %. Inflation is stronger on problems BF16 solves | Qwen3-30B-A3B reasoning average 74.8 (BF16), 74.6 (GPTQ INT4), 75.6 (RTN INT4); BFCL multi-turn 55.9 → 54.1 (GPTQ INT4). A W4A16 kernel's 1.2–1.4× tok/s gave a 2.5 % slower end-to-end BBH run (generation capped at 128 tokens there) |
| Zhang et al., "When Reasoning Meets Compression", ICLR 2026 (App. O) | DeepSeek-R1 at 2.51, 1.73, 1.58 bits (dynamic quantization) | Longest-30 % / shortest-30 % length ratio on AIME 2024: 5.3 (unquantized), 4.9, 4.4, 3.8: the spread narrows; shorter responses score higher | 2.51-bit close to R1 |
| Chen et al., "Train Where the Quantized Model Goes", 2026 | 2.79 and 1.88 effective bits | Long generations "degenerate into repetitive loops, exhausting the decoding budget" | MATH-500 retention 35 % → 70 % with on-policy distillation |

What this means for Flint:
- **Where inflation starts.** 8-bit weights or KV: none. 4-bit weight-only with GPTQ or HQQ: −8 % to +4 % on models
  of ≥ 14B dense or the 30B-A3B MoE, up to +23 % on 1.5–7B models; AWQ +11–14 % and round-to-nearest +43 % on the
  30B-A3B. 3-bit and W4A4: −3 % to +355 %, the large values in small models. Round-to-nearest at INT4 g128 inflates
  more than 3-bit GPTQ.
- **Matched quality means matched length, as far as anyone has measured.** No study compares quants matched by KLD.
  Within one bit width, however, inflation varies tenfold with the method (Qwen3-30B-A3B INT4: +3.7 % GPTQ against
  +43 % RTN) and follows accuracy loss. The mechanism is KL at high-entropy positions turning into hesitation tokens.
  Both say length follows the quantization error, which KLD measures.
- **Our files sit in the gentle regime.** UD-Q4_K_XL and the candidates are 4.5–8-bit mixes with importance-matrix
  rounding at KLD 0.10–0.12 against BF16. On the tokens Flint writes, the candidates' KLD is 0.89–1.04× UD's (Q106).
  Expected length difference against UD: ~0, which the measurements confirm.
- **What is not known:** whether UD itself reasons longer than BF16. Q41 compares quants with UD, not with BF16 or
  Q8_0, on E2 and E3. If all 4-bit files inflate alike, the ranking is unaffected but the absolute answer time is.
- **Flint's own scale of length variation is far larger.** On GPQA-Diamond the makers' reasoning has a median of
  4.3k, p95 33k and p99 46k tokens ([rec note](rec_engine.md)). Here, within one task, the coefficient of variation
  of reasoning between two samples is 0.26 (E1, E2) and 0.36 with five samples (pilot).

## 3. Time to a correct answer

### 3.1 The model

For file f, a generated token at context depth d costs, in served conditions on nine cards (est.):

  t(f, d) = t₀ + B_f / BW + Δ_serve + Δ_tool + α·d

- **t₀ + B_f / BW: the decode time on six cards in llama-bench.** t₀ = 12.0 ms and BW = 151 GB/s.
  - **Basis:** the line through S4-L0's measured 26.8 ms at 2.236 GB, with the fork's streaming rates (Q8_0 150,
    Q5_0 138, Q4_1 145 GB/s). Of the fixed part, ~9 ms is kernel boundaries and small kernels and ~3 ms hops and host
    ([unconventional ideas](unconventional_ideas.md)).
  - **"Decode time follows bytes on the fork"** is this line (Q73 re-priced every mix by it).
  - **UD uses its measured 26.84 ms** (Q76); its bytes alone would predict 26.65, so its K-quant tensors cost ~0.7 %
    extra.
  - **Per file:** G3 27.41, G3h 26.64, K2h 26.17, K1h 25.87 ms.
- **Δ_serve = 4.5 ms:** S4-L0 served on nine cards at 262k of context (31.3 ms, Q58) against llama-bench on six.
- **Δ_tool = 3.3 ms** for requests that carry tools: the 600 KB logits copy and host sampling while backend
  sampling is off (Q55, [tool-request sampling](tool_request_sampling.md), est.). It is 0 for coding.
- **α = 0.3 ms per 1,000 tokens of context** (Q54: 0.22 on nine cards from 0 to 32k, 0.5 on six cards at 8k).

A request costs r + n_new·p before its first token:
- **Today:** r = 2.2 s and p = 7.1 ms (Q58's fit).
- **After Q108–Q110 (est.):** r ≈ 1.0 s and p ≈ 2.0 ms.
- **Multi-turn n_new.** Today llama-server re-reads the previous step's output from 4 tokens before the previous
  prompt's end, because the re-rendered tool call differs from the generated one. "Exact reuse" reads only tool output
  and new user text.

The time per answer is the sum over its requests. The time per correct answer is TPCA = E[time per attempt] / pass@1,
computed per task and summed over the tasks both files have. For two files with equal pass rates and prompts, file a
beats file b exactly when L_a / L_b < t_b / t_a, with t the per-token cost above: prefill cancels. Today's re-read
adds p to every generated token of a step that is not the last, which dilutes the per-token ratio by ~0.3 % (est.).

UD's time per attempt, split (est.):

| Workload | Per attempt | Requests + prefill | Decode (+ depth) | Decode share | Per correct answer |
|---|---|---|---|---|---|
| Single-turn tool call (E1, 186 tokens) | 13.2 s | 6.7 s | 6.5 s | 0.49 | 17.3 s |
| same, after Q108–Q110 | 8.8 s | 2.3 s | 6.5 s | 0.74 | 11.5 s |
| Multi-turn episode (E2, 12 steps, 2,386 tokens) | 184 s | 94 s | 89 s | 0.49 | 508 s |
| same, exact reuse of the generated turn | 168 s | 78 s | 89 s | 0.53 | 463 s |
| same, after Q108–Q110 | 120 s | 31 s | 89 s | 0.74 | 334 s |
| Coding (E3, 12,876 tokens, 32k budget) | 452 s | 6 s | 449 s | 0.99 | 527 s |

### 3.2 Ranking

Ratio to UD (est.; per-token costs from bytes; lengths measured where available).
- **Lengths:** G3's ratios stand in for G3h (a head-only change). On E2 and E3, G3's ratios also stand in for K2h
  and K1h, which share G3's body.
- **The "measured pass" column** applies each lane's measured pass@1 ratio where it exists.

| File | Bytes per token | Per served token: tool / coding | E1 TPCA, equal pass | E1, measured pass | E2 TPCA, equal pass | E2, measured pass | E3 TPCA, equal pass | E3, measured pass |
|---|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 2.213 | 1 / 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| G3 | 2.327 | 1.016 / 1.018 | 1.008 | 1.003 (0.987–1.019) | 0.985 | 1.12 (0.92–1.40) | 1.053 | 1.13 (1.00–1.29) |
| G3h | 2.211 | 0.994 / 0.994 | 0.997 | — | 0.975 | — | 1.030 | — |
| **K2h** | 2.139 | 0.981 / 0.978 | 0.986 | **0.981 (0.957–1.007)** | 0.969 | — | 1.016 | — |
| K1h | 2.095 | 0.972 / 0.969 | 0.982 | — | 0.966 | — | 1.008 | — |

- **After Q108–Q110 decode is a larger share,** so every gap grows by ~1.5×: K2h on E1 0.978, on E2 0.953 (est.).
- **With the screened head on every file** (UD 2.060, K2 2.102, K1 2.058, G3 2.174 GB per token, est.), the
  per-token ratios become K2 1.003, K1 0.994, G3/G3h 1.017.
  - Ranking at equal length: K1 ≈ UD < K2 < G3.
  - E1 TPCA with measured lengths: K2 0.996, K1 0.992.

### 3.3 What flips the ranking

| Pair | Per-token ratio (tool / coding) | Length ratio that ties them (later file's length over earlier file's) | Resolved by today's data? |
|---|---|---|---|
| K2h vs UD | 0.981 / 0.978 | UD reasoning 1.9–2.2 % shorter than K2h's | E1 median: K2h/UD 0.991 (0.981–1.000), so no; E2/E3: not measured for K2h |
| K1h vs UD | 0.972 / 0.969 | 2.8–3.1 % | as K2h |
| G3h vs UD | 0.994 / 0.994 | 0.6 % | no (E1 ±2.3 % on means) |
| G3 vs UD | 1.016 / 1.018 | G3 must be 1.6–1.8 % shorter than UD | E2 point estimate 0.957 would do it; E1 1.000 and E3 1.033 would not |
| K1h vs K2h | 0.991 | 0.9 % | no |
| K2h vs G3h | 0.986 | 1.3–1.6 % | no |
| G3h vs G3 | 0.978 | 2.2 % | — (same body) |
| With screened heads: K2 vs UD | 1.003 | UD wins unless K2 is ≥ 0.3 % shorter | no |

- **Pass rate is the larger term.** One point of pass@1 moves TPCA by 1.3 % (E1, p ≈ 0.76), 2.7 % (E2,
  p ≈ 0.37) and 1.2 % (E3, p ≈ 0.86). Q41's margins allow 2.0, 8.2 and 3.5 %, so a candidate can pass every quality
  gate and still lose on TPCA by more than its speed edge.
- **How the release metric handles that** is in section 5.

## 4. Levers that act on length

### 4.1 The first-finish race (Q93)

- **The race's gain is a property of Flint, not of the quant.** The spread of reasoning between samples of one task
  is the same for UD and G3: within-task CV 0.261 and 0.266 on E1; shortest of 2 = 0.748 and 0.740 of the mean;
  E2 UD 0.739.
- **The race does not care which file runs.** With equal shapes it keeps any length ratio between files unchanged:
  if file a reasons c times longer than b sample by sample, so does its shortest of k.
- **A heavier tail makes the race more valuable only when the tail is independent between samples.** It isn't:
  runaways are task properties.
  - E1: 5 of the tasks where UD ran away at attempt 1 ran away again at attempt 3 (6 and 9 runaway tasks).
  - E3: 6 problems were cut at 32k for both files.
  - With per-task runaway probability q, k = 2 removes the cost only on tasks where one sample closes, a fraction
    1 − q of the affected tasks. The observed recurrence (0.4–0.8 between attempts) leaves half to four fifths of
    runaway cost (est.).
- **What it does cut is the long tail of finished samples, and with it time per correct answer.** Failures are long
  (E1 1.2×, E2 1.45×, E3 3.2–3.6× passed samples), so short-first selection raises pass@1 as it saves time. That is
  +0.24 to +0.40 points on E1 at k = 2, +3.2 points at k = 5 on the pilot.
- **Where a quant does lengthen a tail,** as G3's +3 % on E3 hard problems may, the race shrinks that excess
  whenever the extra length is a per-sample excursion rather than a per-task shift (est.). The E3 64k run with K = 4
  measures both.
- **Under a budget the race also converts cuts into finishes.** A problem where one of two samples finishes under the
  budget is no longer a failure. That is the coding lever if 64k samples still hit the budget.

### 4.2 Reflection and hesitation suppression (Q94 and its overthinking variant)

- **Q94 was dropped because Flint reflects little:** 2.4 reflection words per 1,000 reasoning tokens.
- **The 2026 overthinking work names the mechanism by which quantization inflates length:** KL concentrated at
  high-entropy positions, where hesitation markers gain probability. Its 50-marker penalty includes "but", "maybe",
  "perhaps" and "however", a broader class than NoWait's.
- **Flint's density of that class** is 11.7 (E1), 18.6 (E2) and 22.2 (E3) per 1,000 reasoning tokens, and identical
  between files (above). There is no quant-induced overthinking to remove. A penalty would act on Flint's ordinary
  planning clauses ("But the user wants…"); the BF16 distills' 5.6–10 % is the most it could give (est.), and it
  changes the text, so only task lanes could clear it.
- **Kept as a screen:** the marker ratio against UD costs nothing to compute from any task run. A new file (GPTQ body
  Q112–Q114, a Q6_K rung Q115, any lower-bit expert format) with a ratio above 1.10 and an interval above 1 is
  inflating before its task scores show it.

### 4.3 Which quant choices could drive length

- **Q106's per-token split is the relevant yardstick,** because the published mechanism acts on sampled (written)
  tokens at uncertain positions.
  - Written-token KLD against UD: G3 0.887 (reasoning 0.85, answers 0.86, calls 0.95), G3h 0.957, K2h 1.013,
    K1h 1.038.
  - On the first token after `</think>` (call or answer) G3 is 0.69 (230 tokens).
  - On read tokens (tool schemas and outputs) every candidate is 3–5 % worse than UD, against BF16 as well.
- **Read-token error changes what Flint concludes from a tool result, not how long it deliberates.** E2's step counts
  (0.958) and new prompt tokens (1.011) show no extra tool round trips for G3.
- **Head precision.**
  - The GPTQ Q5_0 head acts on every sampled token, `</think>` included, and moves written-token KLD from 0.887 to
    0.957 (G3 → G3h). A length effect below 1 % is expected (est.) but unmeasured: G3h has no task runs.
  - The screened head (Q92) samples exactly as the full Q8_0 head for top-k 20, so it cannot change length. It is the
    length-neutral way to save head bytes.
- **Global-layer attention (Q107).** Q8_0 attention in the 14 global layers leaves read, written and long-range-copy
  ratios where they were. It is not a length lever.
- **The stop decision itself** is the probability of `</think>` at the positions where Flint ends its reasoning.
  Q106's dumps hold both models' log-probability of every actual token, so the shift at those positions can be read
  without a GPU (experiment 1).

## 5. Plan

### 5.1 Release metric and decision rule

**Metric.** For each lane and each candidate X against UD on the same build, with the same head treatment:

  TPCA ratio = [Σ_tasks T_X / Σ_tasks T_UD] × [Σ_tasks p_UD / Σ_tasks p_X]

- T is the time per attempt from the measured token counts and the rig's cost model (section 3.1, with the rig's
  measured per-token times once they exist); p is pass@1.
- It is reported with a cluster-bootstrap 90 % interval, next to the length ratio and the per-token ratio
  (`lengths.py --time --token-ms X=...`).
- **Workload weights** for one summary number (the owner's choice; proposed): multi-turn agent 0.5, single-turn tool
  calls 0.25, coding 0.25. The rig's users run agents.

**Rule.**
1. **Quality gates unchanged:** E1, E2 and E3 (at 64k) non-inferior; malformed plus runaway ≤ UD + 0.5 points; E3
   runaways ≤ UD + 2 points.
2. **Length gate:** total output tokens per task against UD, ratio of means with runaways kept (they cost time).
   - Upper 90 % bound ≤ 1.03 on E1 (K ≥ 2, resolvable: ±1.9 % at 90 %) and ≤ 1.05 on E2 (799 × 2: ±2.7 %, est.).
   - E3 at 64k is reported, not gated (±2–3 % at best).
   - A file that fails the gate cannot claim to be faster than UD.
3. **Choice among files that pass 1–2:** the lowest weighted TPCA point estimate.
   - If two files are within 1 %, which no lane can resolve, take the one with the lower written-token KLD (Q106).
4. **Claims:** "faster than UD-Q4_K_XL per correct answer" only on lanes where the TPCA ratio's upper 90 % bound is
   below 1. Otherwise the model card states the per-token speed on the rig, the length ratio with its interval, and
   "time per correct answer within x % of UD-Q4_K_XL".

On today's data this rule keeps K2h as the lead: it passes E1, its length ratio is ≤ 1, and its TPCA point estimate
is 0.98. It has not yet been run on E2 or E3.

### 5.2 What Q41's remaining runs should add

- **K2h, the lead, on E2 and on E3 at 65,536 tokens,** not only G3. Its length outside E1 is assumed from G3's
  today.
- **K ≥ 2 for every file on every lane, UD included.** This is the noise floor for length as well as pass@1, and
  E3's within-problem spread is what the race's coding value depends on.
- **E3 at 65,536 tokens** (decided in Q71). Reasoning tokens per sample come from the raw text with `lengths.py`; no
  harness change is needed. The position of the first `</think>`, where a race stops, comes free from the same text.
- **llama-server's `tokens_cached` per multi-turn step,** a count, not a timing. It gives the prefill each step
  really needs under the harness's cache, instead of this note's two policies.
- **A Q8_0 anchor on one lane** (E1 or E2 at K = 2) tells whether 4-bit files as a group reason longer than Flint
  nearly unquantized. That changes absolute answer time, not the ranking.
- **The same token caps for every file** (4,096 per step, 65,536 for E3), with runaways reported per file.

### 5.3 First experiments

1. **The stop decision from Q106's dumps.** No GPU, minutes, if the dumps are kept; the KLD dump patch run again
   otherwise, ~1 GPU-hour.
   - **Positions:** in the 24 agent conversations (98,280 scored tokens) take every position whose actual token is
     `</think>` (~230 positions, by the count of first tokens after it) and every hesitation-marker position.
   - **Statistic:** the mean difference of log-probability, candidate minus UD (and against BF16 on the 12 chunks
     scored there), bootstrapped over chunks.
   - **Reading:** a candidate that gives the actual `</think>` less probability than UD is less willing to stop and
     predicts longer reasoning; more probability on markers predicts hesitation.
   - **Kill criterion:** |Δ log p| < 0.05 nats with an interval covering 0 at `</think>` and at markers for every
     candidate. Then quant choice is not a length lever, and length enters the release only through section 5.1's
     gate.
   - **If a candidate is below −0.1 nats,** it gets experiment 2 before the decision run.
2. **E3's budget-decided problems at 64k.**
   - **What:** the 10 hard problems of eighth 1 that hit or nearly hit 32k (arc181_b, arc185_b, abc382_f, the 6 cut
     for both files, abc376_f). K = 4 per file, UD and K2h, the makers' sampling, budget 65,536, on one GPU set.
   - **Cost:** ~2.8 M generated tokens, est. 3–4 server-hours.
   - **It measures:**
     - each file's length and runaway rate on the problems where length matters;
     - pass@1 once the 32k cut is gone;
     - the shortest-of-2 and shortest-of-4 length and pass (the race's coding value, Q93's open lane).
   - **Kill criterion for "coding length differs between files":** K2h's mean length on these problems ≤ 1.03× UD's,
     and its runaways at 64k within +2 of UD's out of 40. Then E3 length leaves the release question and the decision
     run's E3 compares pass@1 only.
   - **Otherwise** length becomes the coding gate, and the race (rank 3) and a budget-aware stop move up.

## Already covered

| Topic | Where |
|---|---|
| First-finish race, offline gate and executor | Q93, [unconventional ideas](unconventional_ideas.md) idea 1, `bench/race_first_finish.py`, Q87 |
| Reflection-word counts and the dropped logit bias | Q94, [bench/reasoning_length](../../../bench/reasoning_length/) |
| Reasoning budget | Q85 (planned); this note adds that a 32k cut turns length into pass differences on hard problems |
| Flint's tokens per LiveCodeBench sample, the 64k decision | Q71, Q41's E3 pair |
| Written/read KLD split, the token after `</think>`, global-layer attention | Q106, Q107, [long-context quality](long_context_quality.md), [bench/long_context_kld](../../../bench/long_context_kld/) |
| Task protocol, noise, non-inferiority margins | [task-level quality checks](task_level_quality_checks.md), Q41 |
| Prefill costs, cache policies, end-of-turn checkpoint | [prefill on Pascal](prefill_pascal.md) |
| Tool requests on the host sampling path | [tool-request sampling](tool_request_sampling.md), Q55 |
| Bytes-to-time pricing on the fork | Q73, Q77, [GPTQ for the body](gptq_body.md), `scripts/token_cost.py` |
| Screened head | Q91, Q92 |

## Not verified

- **Per-token times of G3, G3h, K2h and K1h on the rig.** All four are from the bytes line; the rig run is planned.
  So is UD with the screened head (est. −1.0 ms).
- **Lengths not measured:** K2h's on E2 and E3, G3h's and K1h's anywhere. G3's ratios stand in for them.
- **E3 at 65,536 tokens and with K ≥ 2.** E3's length interval comes from 56 single samples at a budget that cuts
  12–14 % of them.
- **The served overheads on the v2 build:** Δ_serve 4.5 ms, the 3.3 ms of host sampling for tool requests, the
  depth slope α, and the prefill costs after Q108–Q110.
- **Whether UD-Q4_K_XL reasons longer than BF16 or Q8_0** on any lane.
- **The `</think>` log-probability shift** (experiment 1); whether Q106's dumps were kept.
- **Liu et al.'s per-bit length values** are in figures only. The token-inflation paper's runs are greedy with 100
  questions per benchmark and one run, so its ±4 % differences are within its own noise.

## Sources

Quantization and reasoning length (P)
- R. Liu et al., "Quantization Hurts Reasoning? An Empirical Study on Quantized Reasoning Models", COLM 2025,
  [arXiv 2504.04823](https://arxiv.org/abs/2504.04823), [html v2](https://arxiv.org/html/2504.04823v2) (§4.1 "Do
  Quantized Reasoning Models Think Longer?", Fig. 3, App. B.3)
- S. Lotfi, P. Kirichenko, S. Li, Z. Liu, "Quantized Reasoning Models Think They Need to Think Longer, but They Do
  Not", [arXiv 2606.00206](https://arxiv.org/abs/2606.00206), [html](https://arxiv.org/html/2606.00206v1) (Tables 1–2,
  Figs. 2, 3, 5; App. C marker list)
- "Quantization Inflates Reasoning: Token Inflation as a Hidden Cost of Low-Bit Reasoning Models",
  [arXiv 2606.25519](https://arxiv.org/abs/2606.25519), [html v2](https://arxiv.org/html/2606.25519v2) (Tables 1–3,
  App. E)
- N. Zhang et al., "When Reasoning Meets Compression: Understanding the Effects of LLMs Compression on Large Reasoning
  Models", ICLR 2026, [arXiv 2504.02010](https://arxiv.org/abs/2504.02010) (App. O, P)
- Y. Chen et al., "Train Where the Quantized Model Goes: On-Policy Distillation for Low-Bit Reasoning",
  [arXiv 2609.26708](https://arxiv.org/abs/2609.26708)

Selection by length (P)
- A. Agarwal, A. Sengupta, T. Chakraborty, "First Finish Search: Efficient Test-Time Scaling in Large Language Models",
  [arXiv 2505.18149](https://arxiv.org/abs/2505.18149)
- M. Hassid, G. Synnaeve, Y. Adi, R. Schwartz, "Don't Overthink it. Preferring Shorter Thinking Chains for Improved
  LLM Reasoning", [arXiv 2505.17813](https://arxiv.org/abs/2505.17813)
- E. Miller, "Adding Error Bars to Evals", [arXiv 2411.00640](https://arxiv.org/abs/2411.00640)

Flint
- Recursive's model card and SGLang README figures (GPQA-Diamond reasoning length), as cited in the
  [rec note](rec_engine.md) (S here)
- Flint's tokenizer (`tokenizer.json` of the published checkpoint): byte-level BPE, 150,000 tokens, 149,744 merges,
  added tokens 150000–150143 (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q41, Q54, Q55, Q56, Q58, Q71, Q73, Q76, Q77, Q85, Q91, Q92, Q93, Q94,
  Q106, Q107, Q108–Q110), Q41's raw outputs and per-attempt tables at 08:07 on 2026-10-24 (main runs and pilot; kept
  outside the repository), [bench/answer_time](../../../bench/answer_time/) (`lengths.py`, `flint_tokens.py`),
  [paired_tasks.py](../../../bench/paired_tasks.py), [race_first_finish.py](../../../bench/race_first_finish.py),
  [bench/long_context_kld](../../../bench/long_context_kld/README.md), and the notes linked above
