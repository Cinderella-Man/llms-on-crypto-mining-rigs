# v2 synthesis: what the engine and file work deliver per workload on nine P106-090 cards, and what ships when (2026-10-24)

Scope: the last research round of the 2026-10-23/24 period. It combines the 14 earlier rounds' notes, the registry
(Q1–Q140) and today's first v2 rig measurements into one picture for the release decision:
- a speed budget per workload (chat at short context, agent work, long context, several conversations, prompt
  processing for agent turns), each for six and nine cards: the time per token or per turn split into measured and
  estimated parts, for stock llama.cpp, v1, the v2 base engine, the combined v2 build, and what each pending piece adds
  if it works;
- a ranked ship list (v2.0 now, v2.1, later) with expected gain, evidence and the remaining test, and the default
  command lines per workload;
- the model-file question: our files against UD-Q4_K_XL with the screened head;
- the gaps a careful reader would ask about, each with its cheapest measurement;
- a draft of the public v2 summary.

Conventions are those of the other notes, with one addition for the many numbers here:
- **A number followed by [Qn] is measured** in registry experiment Qn: speed on the rig, quality on the 7-GPU quant
  machine. [tgcheck] is today's generation-only follow-up of Q117 (three builds alternating, three rounds, six
  cards). "Flint benchmark" and "chat benchmark" are the measured pages in [docs/benchmarks/](../../benchmarks/).
- **"(est.)" marks an estimate**: this note's arithmetic or another note's estimate, with its basis given.
  "(est., sim.)" is a result of the pipeline simulator on Flint's own transcripts, which is measured offline but an
  estimate for the rig.
- "UD" is Unsloth's UD-Q4_K_XL, the quality bar. "Served" means llama-server with backend sampling (`-bs`), one
  conversation, the makers' sampling, unless stated. "Combined" is the test build with every v2 engine piece and an off
  switch for each (Q117). "The screen" is the screened output head (Q91/Q92). T is the time of one plain token.
- Nothing ran on the rig or the quant machine for this note. Today's raw rig logs (Q117, Q118, Q55, the tg-only check)
  were read and are summarized here; they stay outside the repository.

Starting point, measured today and before ([registry](../../../quants/experiments.md)):
- **Engines, llama-bench tg128 after the prompt tests of the same run, UD, tok/s [Q117]:** six cards stock 15.2, v1
  31.2, v2 base 34.7, combined 33.5; nine cards 16.5, 32.6, 36.5, 32.9. Prompts (pp512 / 2,048 / 8,192): nine cards
  stock 185 / 493 / 593, v2 base 186 / 492 / 593, combined 228 / 613 / 723; six cards 193 / 410 / 507 against 250 / 511
  / 617.
- **Generation alone (no prompt tests before it in the process):** v2 base on nine cards 30.95 tok/s at `-t 2`, 28.85
  at `-t 1` [Q118]; six cards, three rounds: combined 30.61–30.67, combined with PR #31047 off 30.60–30.65, v2 base
  29.86 in every round [tgcheck].
- **Served, short context:** v2 base, nine cards, `-c 16384`: 31.4–33.6 tok/s over 1,000 tokens, streaming free [Q118].
  The combined build with PR #31047 off, real-chat flags (`-c 262144 -b 512 -ub 128 --ctx-checkpoints 4`), 3,000-token
  requests: chat 32.9–34.0 (nine cards) and 33.5–34.6 (six); the same request with tools 29.3–29.6 and 29.6–30.0
  [Q55]. Yesterday's five-turn chat at the same flags: v1 27.97, stock 14.20 [Q58].
- **Quality:** KLD against BF16 at 512 tokens G3 0.104, K2 0.107, K2h 0.109, K2h-r16 0.105, UD 0.120 [Q77, Q113]; at
  8k on agent text every candidate 1.03–1.04× UD overall, G3 0.89× on written tokens, 1.04–1.06× on read tokens, the
  gap sitting on tool schemas [Q106, Q107]; task checks: E1 non-inferior for G3 (K = 2) and K2h (K = 1), E2 at 458
  tasks G3 +0.87 points, 0.23 short of its margin at K = 1, miss_param confirmatory test in G3's favour [Q41].

## Verdict

- **What v2 delivers today, measured on nine cards with UD:**
  - **Chat:** 2.2× stock and 1.12× v1 in one llama-bench session (36.5 against 16.5 and 32.6 tok/s [Q117]). Served:
    31.4–34.0 tok/s [Q118, Q55], against 28.0 for v1 and 14.2 for stock in yesterday's chat benchmark [Q58] (different
    sessions and prompts, so 1.18–1.22× and ~2.3× are indicative).
  - **Prompts:** 1.22–1.29× faster with upstream PR #31047, pp8192 723 against 593 tok/s [Q117].
  - **Agent traffic:** every request carries tools, so llama-server samples it on the host. It decodes at
    29.3–29.6 tok/s, 3.6–4.8 ms (12–16 %) per token slower than chat [Q55].
  - **Not measured on v2:** generation past 32k of context, several conversations, agent tasks end to end.
- **The engine carries the speed, the file does not.** The same file goes 2.2× from stock to v2 [Q117]. On the same
  engine with the same head treatment, UD and our K2-class files read 2.05–2.06 GB per token, est. 0–1 % apart on
  chat. On tool requests the screen is off, so K2h-r16's GPTQ head makes it est. ~3 % cheaper per token there. That
  puts ≥ 95 % of the logarithmic speed-up in the engine (est.), as the owner concluded.
- **The combined build's decode loss is not real.** Generation alone, six cards, three alternating rounds: combined
  30.61–30.67, the same with PR #31047 off 30.60–30.65, v2 base 29.86 tok/s in every round [tgcheck]. The combined
  build is level with or ~2.5 % above v2 base (within each run's ±1.6 spread, but in all three rounds). No code of PR
  #31047 runs in a one-token step: `MUL_MAT_ID` with ≤ 4–8 tokens goes to the vector kernels before MMQ is considered
  (code reading, 2026-10-24). Q117's −9 % most likely came from llama-bench's state: every tg128 there ran right after
  three pp8192 tests, and such runs read 9–20 % above generation alone. The combined build can ship as it is; the
  split build stays parked.
- **The largest open term in a token is ~5 ms that the same build sometimes does not pay.**
  - The three readings of the v2 base build:
    - llama-bench after prompt tests: 24.6 ms per token of compute (27.4 ms minus its 2.8 ms logits copy, nine cards);
    - llama-bench alone: 29.5 ms;
    - served: 29.4–31.8 ms [Q117, Q118].
  - First-pass page faults on the memory-mapped embedding table explain 1–2 ms in the server: 0.23 major faults per
    token on new text against 0.015 on a repeat, and the repeats ran 0.9–1.8 ms per token faster [Q118].
  - The rest is unexplained. Recovering it would be worth up to +15–20 % served (est.), more than any pending
    single-stream kernel piece.
  - `--mlock`/`--no-mmap` (Q119), the embedding lookup on card 1 (Q121's switch) and a clock-logged pair of runs settle
    it in ≤ 30 minutes.
- **Six cards are as fast as nine at short context.** Served, six cards ran 1–2 % faster than nine [Q55]; llama-bench
  after prompts said the opposite (34.7 against 36.5 [Q117]). Nine cards are for capacity: context above ~32k with a
  K2-class file (~128k with UD, est.), and several conversations.
- **Ship list.**
  - **v2.0:** the combined build with Unsloth's UD-Q4_K_XL, so no weights are redistributed. The build holds v1's
    Pascal kernels and CUDA graphs, FP32 activation sums, shared-expert fusion, PR #31047 for prompt batches, the
    graph-reuse fix and the card-built prompt mask.
  - **v2.1, built and exact but untimed:**
    - the screened-head side file, est. +3–4 % on chat (Q125's hour);
    - the decode mask on the card, est. −3 ms per token at 32k and −12 ms at 131k;
    - the multi-token kernels;
    - the executor's pieces, once in llama-server: pipelined guesses (est., sim. 1.34–1.54×), the first-finish race
      (est. 1.10–1.23× per answer), streams (est. 118–256 tok/s total at 4–8 conversations), the host loop.
  - **Later:** GPU sampling for tool requests (Q67), whose +12–16 % on agent tokens is measured as the gap in Q55 but
    whose code is not built (first to pull forward). Then the staircase drafter (est. 2.0–2.9×), the attention kernel
    and KV sharding (≥ 64k), the persistent kernel, GPTQ for the body.
- **File recommendation: UD-Q4_K_XL**, with the side file once v2.1 lands.
  - K2sr passes the 512-token test (0.89× UD), and K2h passed E1.
  - Its written-token edge at 8k is an estimate (0.94×, Q126 unmeasured), it trails UD ~4 % on tool-schema tokens,
    and the multi-turn decision run has not run.
  - **What would change it:** Q126's written-token upper bound below 1.00 against BF16 plus non-inferiority in the
    decision run (K2sr then ships as a same-speed, higher-fidelity option); Q101's 6-bit kernel passing (G3-6sr); GPTQ
    for the body (Q112 E3).
- **Top gaps:**
  1. A same-session served comparison of stock, v1 and v2, together with the ~5 ms state effect above.
  2. Agent turns end to end: time to first token with tools and checkpoints, cache hits, `-ub 512` at 262k. Today
     they are measured only in pieces.
  3. Everything past 32k: decode, prefill, host memory and quality at 131–262k.
  - Power per token (est. 9–11 J on the cards) and stability over hours (no run longer than ~1 h) follow.

## 1. Speed budget per workload

### 1.1 What a token costs today

UD on the v2 base engine (the combined build decodes at least as fast [tgcheck]), short context, served with `-bs`:

| Part of a token | Nine cards | Six cards | Basis |
|---|---|---|---|
| Weight streaming: 2.21 GB at 138–150 GB/s (Q8_0, Q5_0), UD's Q5_K and IQ4_XS parts at 94–125 GB/s | 15–16.5 ms (est.) | same (est.) | kernel rates [Q38, Q46] |
| Kernel boundaries, small kernels, ramp-up and tails | 8–9 ms (est.) | 8–9 ms (est.) | S4-L0 profile, [breakthrough candidates](breakthrough_candidates.md) |
| Page walks while sweeping ~5 GB per card | 0–4 ms (est.) | same | Q99's hypothesis, untested |
| Embedding-table page faults on new text | 1–2 ms on a first pass, ~0 on a repeat [Q118] | not measured | 0.23 against 0.015 major faults per token |
| Hops between cards (staged through the host) | 0.4–0.8 ms (est.) | 0.25–0.5 ms (est.) | 8 or 5 hops, Q81 unmeasured |
| Host's serial gap from a sample to the next launch | 0.3–1.2 ms (est.) | same | [host decode loop](host_decode_loop.md) §1.1 |
| GPU sampling chain on the last card | 0.1–0.5 ms (est.) | same | radix top-k over 150,144 logits |
| **Served token, total** | **29.4–31.8 ms [Q55, Q118]** | **28.9–29.9 ms [Q55]** | |
| Same build in llama-bench after prompt tests, minus the 2.8 ms logits copy | 24.6 ms [Q117] | 26.0 ms [Q117] | the open ~5 ms of the verdict |
| Tool requests: 600 KB logits copy plus host sampling | +3.6–4.8 ms [Q55] | +3.8–4.8 ms [Q55] | backend sampling off for any grammar |
| Context depth | +0.21 ms per 1,000 tokens up to 32k [Q54, S4-L0] | +0.5 ms per 1,000 at 8k [Q54] | attention plus the host mask copies |

The estimated parts sum to 25–34 ms against the measured 29–32 ms; the split is a guide to where the remaining work
acts, not a measurement.

### 1.2 Chat at short context

tok/s, UD, one conversation:

| Engine or piece | Six cards | Nine cards | Status |
|---|---|---|---|
| Stock llama.cpp, llama-bench / served | 15.2 [Q117] / — | 16.5 [Q117] / 14.2 [Q58] | |
| v1, llama-bench / served | 31.2 [Q117] / — | 32.6 [Q117] / 28.0 [Q58] | released |
| v2 base and combined, llama-bench after prompts | 34.7 / 33.5 [Q117] | 36.5 / 32.9 [Q117] | |
| v2 base and combined, llama-bench alone | 29.9 / 30.6 [tgcheck] | 31.0 (v2 base) [Q118] | |
| **v2, served** | **33.5–34.6 [Q55]** | **31.4–34.0 [Q55, Q118]** | v2.0 |
| + screened head (Q92), −0.153 GB per token | est. +3–4 % (~1.0 ms) | same | v2.1 |
| + the ~5 ms of section 1.1 found (Q119, Q121) | est. up to +15–20 % | same | open |
| + memory clock +10 % (Q133) | est. +3–5 % | same | kit ready, owner |
| + persistent MoE-layer kernel (Q88) | est. +4–20 % | same | prototype exact |
| Single stream with screen, memory clock and persistent kernel | est. 38–45 | est. 36–42 | |
| + pipelined guesses, copy lookup and table (Q87, Q96) | chat est. 1.11–1.18× (sim.); all text 1.34–1.44× | all text est. 1.38–1.54× (sim.) | v2.1 (executor) |
| + staircase drafter at p 0.6–0.7 (Q20) | chat est. 1.85–2.09× (sim.) | chat est. 2.5× at p 0.7 (sim.) | training pilot |
| + first-finish race, k = 2 (Q93) | per answer est. 1.10–1.23× (reasoning 0.75 of the mean, measured offline [Q93]) | same | v2.1 (executor) |

- **Answer time, not tokens per second, is what a chat user waits for.** Two thirds of Flint's output is reasoning
  [Q64], and the race shortens it per answer; quant choice does not change reasoning length (E1 1.000, E2 0.957, E3
  1.033, within UD's own attempt noise 1.027 [answer time](answer_time.md), Q123).
- **Gains do not simply multiply.** The persistent kernel and the memory clock shorten every stage, which also moves
  the pipeline's ceiling; the drafter's gain was simulated on today's stage times (est.).

### 1.3 Agent work

**Per generated token** (tool requests, UD, tok/s):

| Engine or piece | Six cards | Nine cards | Status |
|---|---|---|---|
| v1 | — | est. 24.5–25.5 (Q58's 35.8 ms + 3.6–4.8 ms) | |
| **v2, tool requests** | **29.6–30.0 [Q55]** | **29.3–29.6 [Q55]** | v2.0 |
| + GPU sampling with a host grammar check (Q67) | est. 33.5–34.6 (Q55's chat arm) | est. 32.9–34.0 | later (2–3 days to build) |
| + the screen kept on tool requests (Q67 allows it) | est. +3–4 % | same | later |
| + pipelined guesses on agent turns (Q87, Q96) | est. 1.21–1.37× (sim.) | similar (est.) | v2.1 (executor) |
| K2h-r16 instead of UD (GPTQ Q5_0 head, no screen needed) | est. ~3 % faster per token | same | file option |

**Per agent turn** (nine cards; fixed cost + new prompt tokens + generated tokens; est. unless marked):

| Turn | v1 (released flags) | v2.0 (combined, `-ub 512`) | v2.0 + Q67 | + table guesses (Q87), Q111's per-request fixes and saved prefix |
|---|---|---|---|---|
| Tool-calling step: 400 new tokens, 200 generated | 2.2 s + 2.8 s + 8.0 s ≈ 13 s | 0.9–1.3 s + ~1.8 s + 6.8 s ≈ 9.4–9.9 s | ≈ 8.5–9.1 s | ≈ 6.5–7 s |
| Coding-agent turn: 1,800 new, 800 generated | 2.2 + 12.8 + 32 ≈ 47 s | 1.2 + 2.9–3.6 + 27.2 ≈ 31–32 s | ≈ 28–29 s | ≈ 21–23 s |
| Session start: 5,000-token system and tool prompt | 2.2 + 35.5 ≈ 38 s to the first token | 1.2 + 6.9–8.2 ≈ 8–9.5 s | same | 1–3 s with a saved state |

- **Bases.** v1: ~7.1 ms per new token plus ~2.2 s per request (a fit to Q58's turns, [prefill note](prefill_pascal.md)
  §1.5); 39–41 ms per tool-request token. v2.0: a 28-token prompt took 0.86–0.90 s to its first token without
  checkpoints [Q118], and checkpoints add est. 0.2–0.5 s. Prompts up to 512 tokens are one micro-batch at 228 tok/s
  (pp512 [Q117]), longer ones run near pp2048's 613 tok/s [Q117]; decode is 33.7–34.2 ms [Q55].
- **Where v2 gains on agents.** Mostly in prefill and the fixed cost per request (est. 1.6–4.6× there); decode, which
  is most of a v2 turn, gains only 1.15–1.20× until Q67 lands. [Agent workload check](agent_workload_check.md) estimates
  a whole coding task at 1.47× (8.1 h against 5.5 h for its 48 tasks).
- **Caching** already serves 89.6 % of the prompt tokens an agent re-sends; a saved system-prompt state and exact reuse
  of the generated turn would serve ~97.6 % (est., replay of 200 BFCL sessions, Q111). The fresh tool output stays the
  prefill load.

### 1.4 Long context

Nine cards (six cards hold ~32k with a K2-class file and ~128k with UD, est.). UD or S4-L0, tok/s at the given depth
of context:

| Depth | Measured | v2 base (est.) | + decode mask on the card, Q116 (est.) | + Pascal decode attention, Q60 (est.) | + KV-sharded attention, Q95 (est.) | Prefill to that depth: v1 / after the prefill fixes |
|---|---|---|---|---|---|---|
| ≤ 3k | v1 28.0 served [Q58]; v2 31.4–34.0 served [Q55, Q118] | — | — | — | — | — |
| 32k | 23.3 (S4-L0, v1-era fork, llama-bench) [Q54] | 26–27 | 28–30 | ~30 | — | 234 s [Q79] / est. 75–110 s |
| 131k | not measured | 9–17 | 10–22 | 17–21 | +40–60 % over Q60 | est. ~35 min / est. 10–15 min |
| 262k | not measured | 5–12 | 6–16 | 12.5–14 | ~2× over Q60 | est. ~2 h / est. 45–60 min |

- **What "full 262,144-token context" means today.** The context fits: UD and S4-L0 serve one conversation with the
  full context on nine cards [Q58]. The published chats stayed below ~11k tokens of depth, so no speed at depth beyond
  32k has been measured. The 64k point of Q54 (3.6 tok/s) was llama-bench's saved state swapping the 4 GB host (Q79).
- **The v2 base range** spans two models: the measured slope to 32k extended linearly (+0.21 ms per 1,000 tokens), and
  the attention note's bandwidth model (KV read at 57–87 GB/s plus the host mask copies), which over-predicted 32k by
  2–3×. Q116's gain assumes the mask copies are ~40 % of the slope (est.: 3 of 6.9 ms at 32k).
- **Attention is 16 % of a token at 32k** [Q54], below the 30 % gate for the attention kernel (Q59/Q60). The kernel
  and KV sharding matter from ~64k up.
- **Six cards** pay a similar or steeper cost per token of depth (S4-L0 32.6 tok/s at 8k [Q54], +0.5 ms per 1,000
  tokens), and their capacity ends at ~32k with a K2-class file (S4-L0 did not fit 32k [Q54]) or ~128k with UD (est.).
- **Prefill at depth** is the larger wait: a 131k document takes est. half an hour on v1 (llama-bench fill slope,
  [prefill note](prefill_pascal.md) §1.4).

### 1.5 Several conversations

Total generated tok/s (each conversation's rate in brackets), short context, est. unless marked
([multi-conversation serving](multi_conversation_serving.md)):

| N | llama-server `-np N` today | `-np N` with the multi-token kernels (Q78) | Staggered streams (Q104) | Measured |
|---|---|---|---|---|
| Nine cards, 1 | 31.4–34.0 [Q55, Q118] | — | — | |
| 2 | 39–47 (19–24) | 46–50 | 64 (32) | stock build, makers' Q4_1: 35–42 (21–24) [Flint benchmark] |
| 4 | ~37 (~9) | 64–75 | 118–128 (29–32) | — |
| 8 | ~32 (~4) | 82–95 (needs an 8-column kernel) | 235–256 (29–32), host-bound until Q116 and Q120 | — |
| Six cards, 2 | 45–55 | 53–59 | 75 (37) | — |
| 4 | ~43 | 74–87 | 137–149 (34–37) | — |
| 8 | ~37 | 94–109 | 145–159 (18–20); groups of two 196–235 | — |

- **Memory does not bind below ~64k per conversation:** f16 KV is 0.27 / 0.93 / 3.55 GiB per conversation at 8k / 32k
  / 131k (Q105, est.); UD on nine cards holds ~85 / ~24 / ~6 conversations (est.).
- **Tool requests cap the staggered ceiling** at est. ~150 tok/s on nine cards (~110 on six), because the 600 KB logits
  copy sits on the last card's stream, until Q67.
- **Prompt-heavy agents:** at two new prompt tokens per generated token, est. 56–72 generated tok/s total today, 130–190
  with the prefill fixes and prompt chunks as pipeline work ([prefill note](prefill_pascal.md) §3.4).
- **v2.0 setting:** explicit `-np 2` at most (one KV stream per slot; `-np auto` unifies the cache and makes every
  step read every conversation's KV).

### 1.6 Prompt processing for agent turns

| Engine | Six cards, pp512 / 2,048 / 8,192 (tok/s) | Nine cards | Served |
|---|---|---|---|
| Stock = v1 = v2 base (the same batched kernels) | 193 / 410 / 507 [Q117] | 185–187 / 492–494 / 593 [Q117] | v1 flags (`-b 512 -ub 128`): ~141 tok/s + ~2.2 s per request (fit) [Q58] |
| **Combined: PR #31047, reuse fix, card mask** | **250 / 511 / 617 [Q117]** | **228 / 613 / 723 [Q117]** | 28-token prompt 0.86–0.90 s to the first token (v2 base, no checkpoints) [Q118] |
| + `-ub 128` with the reuse fix (Q109) | est. 1.4–1.8× at ≤ 8k over today's `-ub 128` | same | not measured (Q109 does nothing at `-ub 512` [Q117]) |
| + card mask at depth (Q110) | — | est. 1.3–1.5× at 32k, 1.5–2× at 131k | +1.3 % at pp8192 [Q117] |
| + per-request fixes and saved prefixes (Q111) | — | est. −0.5–1.5 s per request; 97.6 % of re-sent tokens served | not built |

- **PR #31047 carries all of the measured prompt gain** (switching it off returns 617 to 508 tok/s [Q117]). It sizes
  MMQ's expert tiles by tokens per expert instead of 64-token tiles that are ~95 % padding at small batches.
- **Small prompts matter most to agents**: a tool turn adds a median 153 tokens. They run as one micro-batch, where
  pp512 moved from 186 to 228 tok/s on nine cards [Q117], and the per-request fixed cost is as large as the tokens.

## 2. Ship list

### 2.1 Ranked

Ranked by expected value for a rig owner, within each release.

| Feature | Status | Gain | Evidence | Remaining test |
|---|---|---|---|---|
| **v2.0: measured on the rig, exact** | | | | |
| v1's Pascal one-token kernels, CUDA graphs on Pascal, 2-warp launch shape | shipped in v1 | 2.0× stock [Q117: 32.6 against 16.5]; 1.97× served [Q58] | KLD within summation-order noise, operator tests on a P106 [Q38, Q46] | none |
| FP32 activation sums, `LLAMA_NO_MOE_DOWN_RESCALE=tg` for every file | measured | +9.6 % tg128 for UD [Q76]; with shared-expert fusion v2 base 1.11–1.12× v1 [Q117] | one-token perplexity identical | Q76's 80-chunk KLD check to record |
| Shared-expert gate/up fused with SwiGLU | measured | +0.7 % [Q50] | exact | none |
| PR #31047, MoE tiles sized by tokens per expert | measured | prompts +22–29 % at `-ub 512` [Q117]; no decode change [tgcheck] | logits byte-identical on the quant machine | a one-token and prompt KLD on the rig (~10 min) |
| Card-built prompt mask (Q110) | measured (small) | +1.3 % at pp8192 [Q117]; est. 1.3–1.5× at 32k and `-ub 512` at 262k without 1 GiB of pinned host memory | 1,606 masks identical to the host's | Q140: `-ub 512` at 262k fits; llama-batched-bench at 32k/131k |
| Graph-reuse fix for prompt micro-batches (Q109) | measured (no effect at `-ub 512`) | est. 1.4–1.8× at `-ub 128`, ≤ 8k | logits identical | the `-ub 128` grid (Q89) |
| Norm writes the Q8_1 activations (Q100) | measured (within noise) | ~0 [Q117] | exact, 287 fewer launches | none |
| Served settings: `-bs`, streaming, `--cache-ram 0`, `--ctx-checkpoints 4` | measured | no server gap, streaming free at 1,000 tokens [Q118] | | Q119 (`--mlock`/`--no-mmap`), 20 min |
| **v2.1: built and exact, untimed on the rig** | | | | |
| Screened-head side file (Q92), chat only | in the combined build, off by default | est. +3–4 % chat; +15–20 % on the multi-conversation ceiling (lighter last card) | bit-identical on 262,144 candidate logits; exact for top-k 20 on 40,527 rows [Q91] | Q125 (1 h) |
| Decode mask on the card (Q116) | branch off Q110's | est. −3 ms per token at 32k, −12 ms at 131k (10–25 % of a token at depth) | 15,524 masks identical, CUDA graphs unchanged | llama-batched-bench at 32k/131k, host against card mask (1 h) |
| Multi-token Pascal kernels (Q78) | in the combined build, on | est. 2-token step 1.30–1.44 T (today ~1.65 T): `-np 2`, verify steps, groups of two | bit-identical to one-token products | llama-batched-bench `-npl 1,2,4` (30 min) |
| Pipelined executor with copy lookup and continuation table (Q87, Q96) | `llama-pipespec` | est. 1.34–1.44× (six cards), 1.38–1.54× (nine) overall; chat 1.11–1.18×, edits 1.74–1.85×, rewrites ~4× (sim.) | 89 runs bit-identical to plain decoding | `rig_run.sh` (2–3 h); llama-server integration (1–2 weeks) |
| First-finish race, k = 2 (Q93) | `llama-pipespec` | per answer est. 1.10–1.23× | offline: reasoning 0.75 of the mean, pass +0.24 to +0.40 (n.s.) [Q93]; 31 races exact | `rig_race.sh` (45 min); server integration |
| Streams: several conversations staggered (Q104) | `llama-pipespec` | est. 118–128 tok/s at N = 4, 235–256 at N = 8 (nine cards) against ~37 for `-np` | 48 conversation runs exact | rig experiments A–D (3 h); server integration |
| Packed asynchronous uploads, device-fed next token (Q120, Q121) | `llama-pipespec` | est. +0.5–1 % and +2–5 % single stream; N = 8 at the cards' ceiling | bit-identical in all modes | `rig_hostloop.sh` (25 min) |
| Pascal Q6_K kernel (Q101) | branch | enables G3-6sr and GPTQ-Q6_K (est. +4 % at equal quality) | correctness checks pass | rig gate ≥ 130 GB/s (part of a 1 h block) |
| Persistent MoE-layer kernel (Q88) | prototype | est. +4–20 % | bit-exact on an RTX 3090 | one-card rig A/B; integration weeks |
| Memory clock offset through NVML (Q133) | kit ready | est. +3–5 % decode | exact protocol (statistics identical to a stock rerun) | GPU0 run (1.5 h, owner at the console) |
| **Later: research or not built** | | | | |
| GPU sampling for tool requests with a host grammar check (Q67) | not built | +12–16 % per agent token (the measured gap [Q55]) plus the screen on tool requests | design exact by construction | build (2–3 days), Q66's rejection rate, rig A/B (30 min) |
| Staircase drafter for the executor (Q20) | training pilot running | est. 2.05–2.32× (six cards), 2.5–2.9× (nine) | design; SPD 2.05–2.21× at T 1 on other models | offline gate p ≥ 0.55 on fresh text |
| llama-server integration of the executor (streams, guesses, race) | not built | unlocks every executor row above | — | 1–2 weeks of code |
| Per-request fixed cost and saved prefixes (Q111) | not built | est. −0.5–1.5 s per request; session start 1–3 min → 1–3 s | replay of 200 BFCL sessions | server changes (1–2 days) |
| Pascal decode attention (Q59/Q60), then KV sharding (Q95) | not built | est. 131k 9–17 → 17–21 tok/s; KV sharding +40–60 % at 131k, ~2× at 262k | attention 16 % at 32k [Q54] | Q59 microbenchmark after a 64k/131k depth run |
| GPTQ for the body (Q112–Q115) | E1 passed | est. +1.6–4.5 % at equal KLD | GPTQ recovers 0.45–0.79 of Q5_0's cost in three groups [Q112] | E2/E3 (5–12 GPU-hours) |
| TLB-aware weight layout (Q99) | bench mode ready | est. +5–12 % if page walks explain the gap | hypothesis | 2–3 h on one card |

### 2.2 Default command lines (nine cards, release engine, loopback only)

Common to all: the makers' sampling, f16 KV cache, `--cache-ram 0`, explicit `-np`, and
`LLAMA_NO_MOE_DOWN_RESCALE=tg` (never on other GPUs). `<file>` is UD-Q4_K_XL unless the owner picks another.

**Chat and long documents (one conversation, full context):**

```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg \
bin/llama-server -m <file>.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 \
  -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

- `-b 2048 -ub 512` once Q140 shows it fits at 262k with the card-built mask (host MemAvailable logged); until then
  `-ub 128`, v1's setting.
- v2.1 adds `LLAMA_SCREEN_HEAD=1 LLAMA_SCREEN_HEAD_FILE=flint-output-screen-gptq-q4_0.gguf` for chat servers.

**Agents (pi, OpenCode), one agent:**

```bash
LLAMA_NO_MOE_DOWN_RESCALE=tg \
bin/llama-server -m <file>.gguf --alias flint -ngl 99 -fa on -c 65536 -np 1 \
  -b 2048 -ub 512 -bs --ctx-checkpoints 4 --cache-ram 0 \
  --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --host 127.0.0.1 --port 8080
```

- At 65,536 tokens the micro-batch of 512 needs at most 256 MiB of pinned mask memory with four pipeline copies, within
  the host's budget even without the card-built mask ([prefill note](prefill_pascal.md) §3.2, est.). Sessions that need
  more context use the chat line.
- No screen variables: every tool request turns the screen and backend sampling off.
- Four checkpoints take up to ~350 MiB of host RAM at `-ub 512` (87.5 MiB each); if the host starts swapping, use
  `--ctx-checkpoints 2`.
- The client's context window equal to `-c` (pi compacts at window minus 16,384 tokens, which on the rig costs a full
  re-read), an output cap of 32,768 tokens.

**Two conversations:** the agent line with `-c 131072 -np 2 -b 512 -ub 256 --ctx-checkpoints 2` (65,536 tokens each;
`-ub 256` keeps the pinned mask copies within the host's budget at 131,072 cells, est.). More conversations wait for
the streams mode in llama-server.

**Six cards** (three cards free for another model): `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5` with `-c 32768`; UD fits est.
~128k there. Same speed as nine cards at short context [Q55].

## 3. The model file

| | UD-Q4_K_XL + screen | K2sr (K2 + screen + F16 router) | K2h-r16 (GPTQ head, for stock llama.cpp) | G3-6sr (needs Q101) |
|---|---|---|---|---|
| GB per token | 2.060 (2.213 without the screen) | 2.053 (2.206 without) | 2.091 | 2.066 (est.) |
| Speed against UD, chat, same head treatment | 1 | est. 1.00 | — | est. 1.00 |
| Speed against UD, tool requests (screen off) | 1 | est. 1.00 | est. ~0.97 of UD's time | — |
| KLD at 512 tokens against BF16 | 0.120 | 0.107 (K2, measured with a Q8_0 head) [Q77] | 0.105 [Q113] | est. 0.105–0.106 |
| 8k agent text, written tokens, ratio to UD (BF16 reference) | 1 | est. ~0.94 (Q126 unmeasured) | ~1.00 (K2h 0.999) [Q106] | est. 0.90–0.91 |
| 8k agent text, read tokens | 1 | ~1.04 (K2h) [Q106] | ~1.04 | est. 1.04–1.06 |
| E1 single-turn | bar | non-inferior (K2h, K = 1) [Q41] | same body | not run |
| E2 multi-turn, E3 coding at 64k | bar | not run (G3: E2 +0.87 at 458 tasks, K = 1) | not run | not run |
| Download and licence | Unsloth's file at a pinned revision, plus a 173 MB side file | our 31–33 GB upload, headers to rewrite | same | same |

- **Recommendation: UD-Q4_K_XL** (the side file from v2.1).
  - Speed per token is equal with the same head treatment.
  - The release rule ([long-context mix](long_context_mix.md) §5.2) asks a file of ours for superiority on written
    tokens at 8k and non-inferiority in the 799-task decision run. Neither is measured yet.
  - UD needs no weights from us: an owner keeps the file they already have.
  - Pass rate dominates time per correct answer (one point of pass@1 is 1.3–2.7 % of it, [answer time](answer_time.md)),
    so a 0–3 % speed edge cannot decide the file.
- **Quality evidence for our files today:** better at 512 tokens (0.87–0.91× UD) and on what Flint writes at 8k (G3
  0.89×); ~4 % worse on tool-schema tokens it reads; level or better on tasks where measured (E1 for G3 and K2h, E2 at
  K = 1, miss_param in G3's favour); reasoning length level.
- **What would change it:**
  - Q126 (0.3 GPU-hours) puts K2sr's written-token upper bound below 1.00 against BF16, and the decision run
    (799 × K ≥ 2, E3 at 64k; owner's booking of the quant machine) is non-inferior. K2sr then ships as an optional
    file: same speed, lower error on what Flint writes.
  - Q101's Q6_K kernel passes its ≥ 130 GB/s gate: G3-6sr is est. better than UD on both yardsticks at its bytes.
  - Q112 E3 makes a K2-class file 50–150 MB lighter at equal KLD (est. +1.1–3.2 %): a speed edge that would need the
    same quality evidence.
  - Agent-only servers before Q67: K2h-r16's ~3 % on tool requests is real but small against a 30 GB download and the
    open E2 evidence.

## 4. Gaps

| Gap | What exists | Cheapest measurement |
|---|---|---|
| **The ~5 ms state effect and a same-session served comparison** | llama-bench after prompts 24.6 ms of compute, alone and served ~29.5–31.8 ms [Q117, Q118, tgcheck]; stock/v1 served numbers from yesterday's chat benchmark | llama-bench `-p 0 -n 128` alone with `-mmp 0`, then after `-p 8192` in a separate process, clocks logged every 200 ms (20 min); `server_bench.sh` chat with stock, v1 and v2 in one session (~1.5 h with loads) |
| **Agent turns end to end** | prefill, fixed cost and tool-token decode measured in pieces [Q55, Q58, Q117, Q118] | a two-turn tool conversation with Q55's client, logging `prompt_n`, `prompt_ms`, `cache_n` and the first-chunk time at `--ctx-checkpoints 4` (30 min); then Q131's pilot (1 h 50 min) |
| **First-token latency for agent turns** | v1 ~7.1 ms per new token + ~2.2 s per request (fit) [Q58]; v2 base 0.86–0.90 s for 28 tokens without checkpoints [Q118] | the same two-turn run, plus the prefill note's E6 (± `-bs`, ± checkpoints, `-lv 4` timestamps; 30 min) |
| **Long context past 32k** | 32k decode on a v1-era build [Q54]; capacity at 262k [Q58]; 64k point invalid (Q79) | llama-batched-bench at `-npp 65536` and `131072`, host against card mask (rig task, 1 h); a 262k fill only after the prefill fixes |
| **Quality at 131k–262k** | 8k agent text [Q106]; RULER literature: 4-bit weights level below 64k, INT4 ~85–88 % recovery at 128k ([long-context mix](long_context_mix.md) §1.4) | quant machine: 32k–64k chunks of long transcripts against a sparse top-256 BF16 reference (L7), UD against K2sr (1–3 GPU-hours); a needle test at 131k–262k. The engine is exact, so only the file matters |
| **4 GB host RAM** | serving at 16k: ~0.59 GiB in swap, ~0.17 GiB free, ~1.6 GiB page cache [Q118 raw vmstat]; 0.23 major faults per token on new text [Q118]; llama-bench's saved state swaps at 64k (Q79) | Q140's 262k check with `/proc/meminfo` and `vmstat 1` logged at `-ub 512` and four checkpoints (87.5 MiB each), plus `--mlock` (30 min); the owner's second SO-DIMM (Q86) if it fails |
| **Power per token** | busy cards average 33–39 W in the layer split (hardware note; Q117 follow-up phases), 75 W cap reached rarely or often depending on the reading; est. ~300–350 W on nine cards, ~9–11 J per token on the cards at 33 tok/s | add `power.draw` and `clocks_event_reasons.active` to the nvidia-smi logging of an already scheduled run (no extra rig time); one reading of a wall-plug meter by the owner for host, fans and supply |
| **Stability over hours** | no run longer than ~1 h; no Xid in today's runs | a 4-hour unattended server soak with a looping chat client: tok/s per request, Xid, temperatures, MemAvailable and swap, plus a greedy canary every 30 minutes that must repeat its tokens exactly; or as part of Q131's ~20 h day |
| **The combined build on the P106 itself** | operator tests and logit identity on the quant machine (compute_61 PTX) | a 4 × 512-token one-token KLD against v1 on six cards (10 min) |
| **P106-100 rigs and other boards** | none | an owner report; same compute capability, untested |

## 5. Draft public summary

Before publishing: Q125 must time the screen if v2 ships with it, a served run of the combined build should confirm
its generation speed (the served rows below ran on the v2 base build and on the combined build with PR #31047 off),
and a same-session served run of stock, v1 and v2 would replace the cross-session row. The draft below is
written for the README or `docs/benchmarks/2026-10-xx-flint-v2.md`.

---

### v2: Flint on nine P106-090 cards

v2 is this project's second llama.cpp release for the Manli 9 × P106-090 rig. It runs the model file most people
already have, Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) of
[Flint](https://huggingface.co/RecursiveSI/flint), about 2.2 times as fast as stock llama.cpp, and reads prompts about a
quarter faster than v1.

| Nine cards, Unsloth UD-Q4_K_XL | Stock llama.cpp | v1 | **v2** |
|---|---|---|---|
| Generation, `llama-bench` tg128 (one session) | 16.5 tok/s | 32.6 | **36.5** |
| Generation, real chat (llama-server, backend sampling, one conversation) | 14.2 | 28.0 | **32.9–34.0** |
| Generation, requests that carry tools (agents) | — | — | **29.3–29.6** |
| Prompt processing, 512 / 2,048 / 8,192 tokens | 185 / 493 / 593 | 187 / 494 / 593 | **228 / 613 / 723** |
| Six cards: generation, `llama-bench` tg128 | 15.2 | 31.2 | **34.7** |
| Six cards: prompt processing, 512 / 2,048 / 8,192 | 193 / 410 / 507 | 193 / 410 / 507 | **250 / 511 / 617** |

What v2 adds to v1:
- **FP32 activation sums** in the Pascal one-token kernels, so llama.cpp's MoE rescale is skipped for every file
  (`LLAMA_NO_MOE_DOWN_RESCALE=tg`): +9.6 % generation on UD-Q4_K_XL.
- **Flint's shared expert fused** with its activation: +0.7 %.
- **MoE prompt tiles sized by tokens per expert** (upstream PR #31047): +22–29 % prompt processing.
- **Two changes for long prompts:** the attention mask built on each card for prompt batches, and micro-batches of one
  prompt that overlap across the cards at small batch sizes.

How to read the numbers:
- **Generation speed depends on how it is measured.** `llama-bench` ran each tg128 after its prompt tests; run alone,
  the same v2 build reads 31.0 tok/s on nine cards. Served chat, 31–34 tok/s, is what a user sees.
- **The real-chat row comes from two sessions:** stock and v1 from the
  [five-turn chat benchmark](2026-10-23-flint-chat.md), v2 from 3,000-token requests with the same server settings.
- **Six cards are as fast as nine for one short conversation.** Nine cards are needed for long contexts and several
  conversations.
- **Agents run slower than chat.** Every request with tools makes llama-server sample on the CPU, which copies 600 KB
  of scores over the slow link per token: 3.6–4.8 ms more per token.
- **The engine does not change the answers.** Its kernels change only the order of additions; outputs match stock
  llama.cpp within rounding, and the v2 additions leave the logits unchanged (byte-identical in every check; the FP32
  sums give the same perplexity to every printed digit).
- **The file:** v2 recommends Unsloth's UD-Q4_K_XL. This project's own mixes are more faithful on 512-token text (KLD
  0.104–0.109 against 0.120), level or better on what Flint writes in 8k-token agent conversations and ~4 % less
  faithful on the tool descriptions it reads; they are no faster on v2, and their multi-turn task checks are not
  complete.
- **Not measured yet:** v2 generation deep into a long context (the full 262,144 tokens fit on nine cards; v1's engine
  generated 23.3 tok/s at 32,768 tokens of context with Miner S4-L0), several conversations at once, coding-agent tasks
  end to end, power draw, multi-hour runs.

What comes next, built and exact but not yet timed on the rig:
- a 173 MB screened output head that serves the same tokens with less reading per token (est. +3–4 % on chat);
- the attention mask built on the cards during generation (long contexts);
- a pipelined executor that sends guessed tokens, extra samples of the same answer, or other conversations through the
  idle cards (est. 1.1–1.2× on chat and 1.3–1.5× averaged over chat, code and edits for one conversation, and 4–8×
  total throughput for several conversations).

Not built yet: GPU sampling for requests with tools, which would bring agents to chat speed.

---

## What this changes in the plan

- **Published decode speeds:** served runs, and llama-bench tg128 only from a run without prompt tests before it. Q117's
  after-prompt numbers stay valid as same-session engine ratios.
- **PR #31047 ships unsplit** in the combined build: all three tg-only rounds agree; the split build stays parked and
  the queued Q117 fix (a split-build A/B) is not needed.
- **Q119 and Q121's embedding-on-card switch move up:** the ~5 ms per token between llama-bench's best state and served
  speed is the largest single-stream term left. A 20-minute run decides whether it is reachable.
- **Q67 is the first item to pull into v2.1:** its gain is already measured (Q55), and agent traffic is what the rig's
  users run.
- **The release note's card advice changes:** six cards are as fast as nine at short context; nine are for capacity.

## Already covered

| Topic | Where |
|---|---|
| Release mechanics: side file, licences, tarballs, headers, results format, checklist | [release v2](release_v2.md) |
| Per-piece designs and estimates | [breakthrough candidates](breakthrough_candidates.md), [pipelined speculation](pipelined_speculation.md), [pipeline drafter](pipeline_drafter.md), [multi-conversation serving](multi_conversation_serving.md), [prompt processing](prefill_pascal.md), [host decode loop](host_decode_loop.md), [tool-request sampling](tool_request_sampling.md), [hardware tuning](hardware_tuning.md), [GPTQ for the body](gptq_body.md) |
| File quality: 8k agent text, written and read tokens, the release rule | [long-context quality](long_context_quality.md), [long-context mix](long_context_mix.md) |
| Answer time, reasoning length, time per correct answer | [answer time](answer_time.md) |
| The end-to-end agent check | [agent workload check](agent_workload_check.md) |
| Attention at depth and the 64k artefact | [attention at long context](pascal_attention_long_context.md), [long-context cliff](long_context_cliff.md) |

## Not verified

- **Served speed of v1 and stock at short context in the same session as v2:** the cross-session ratios (1.18–1.22×,
  ~2.3×) compare different prompts and days.
- **The tg-only check:** six cards only, generation alone; the combined build's ~2.5 % lead over v2 base is inside
  each run's spread.
- **The ~5 ms between llama-bench after prompt tests and served speed:** page faults explain 1–2 ms in the server;
  the rest has no measured cause. Clocks were not logged during Q117's runs.
- **The token decomposition of section 1.1:** every part except the totals, the tool-request extra, the page-fault
  difference and the depth slope is an estimate.
- **Every v2.1 and later gain:** estimates or simulator results; none of those pieces has run on the rig.
- **Long context:** v2 at 131k and 262k is extrapolated from a 32k point on a v1-era build with a different file
  (S4-L0); the two extrapolation models disagree by up to 2×.
- **Agent turn times:** a cost model from measured parts; no agent turn has been timed on v2.
- **Several conversations on v2:** no `-np` or streams run on the v2 build; the stock-build two-slot run used another
  file.
- **The file comparison:** K2sr's written-token ratio, its rig speed and the screen's speed on the rig (Q125, Q126);
  E2 and E3 for K-class files.
- **Power:** per-card means from earlier runs, two readings of how often the 75 W cap is active; no wall-plug
  measurement.
- **Six-card context limits and the agent line's pinned-memory headroom:** arithmetic, not run.

## Sources

This repository (P):
- [quants/experiments.md](../../../quants/experiments.md): the decision log and Q19, Q20, Q38, Q41, Q46, Q50, Q54, Q55,
  Q58, Q64, Q67, Q76–Q79, Q87–Q93, Q96, Q99–Q101, Q104–Q113, Q116–Q121, Q123–Q126, Q131–Q133, Q136–Q140.
- Benchmarks: [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md) (Q58),
  [Flint benchmark](../../benchmarks/2026-10-22-flint.md) (two conversations on the stock build).
- Notes: [breakthrough candidates](breakthrough_candidates.md), [pipelined speculation](pipelined_speculation.md),
  [pipeline drafter](pipeline_drafter.md), [multi-conversation serving](multi_conversation_serving.md),
  [prompt processing](prefill_pascal.md), [host decode loop](host_decode_loop.md),
  [tool-request sampling](tool_request_sampling.md), [answer time](answer_time.md),
  [long-context mix](long_context_mix.md), [long-context quality](long_context_quality.md),
  [agent workload check](agent_workload_check.md), [release v2](release_v2.md), [hardware tuning](hardware_tuning.md),
  [GPTQ for the body](gptq_body.md), [attention at long context](pascal_attention_long_context.md),
  [long-context cliff](long_context_cliff.md).
- [README](../../../README.md), [hardware spec](../../mining_rig.md), [overview](../../overview.md),
  [quants/README.md](../../../quants/README.md).

Raw rig logs of 2026-10-24, kept outside the repository (P), summarized here:
- Q117: llama-bench grids of stock, v1, v2 base and the combined build on six and nine cards, and the combined build's
  ablations, with card temperatures before each run.
- The Q117 generation-only follow-up: tg128 alone, combined with PR #31047 on and off and v2 base alternating, six
  cards, three rounds, clocks at each run's start.
- Q118: llama-bench tg128 at `-t 1` and `-t 2`; llama-server runs streamed and not, with per-request timings, major page
  faults before and after, per-thread CPU, card utilization and `vmstat`.
- Q55: llama-server with tools, `tool_choice` none and auto, backend sampling on and off, nine and six cards.
- The Q108 follow-up's code reading of the one-token `MUL_MAT_ID` dispatch on the combined branch and its nvidia-smi
  phase analysis of earlier runs.
