# Flint after v2.0: where an agent task's time goes, and what could still make it faster (2026-10-28)

Flint Q8_0 on nine P106-090 cards with the v2.0 release line (`-c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6
-bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4`) generates at 44.6–46.5 tok/s on the agent check (12 of 12
tasks), against 14.4 for stock llama.cpp. This note breaks an agent task's time into buckets from the measured runs,
lists every lever the [registry](../../../quants/experiments.md), the other notes and recent outside work offer, and
ranks them by expected agentic gain per unit of effort. "(est.)" marks arithmetic on measured numbers; a gain is a
measurement only where a registry row says so.

## Where the time goes

From the three release-line runs with per-request rows (2026-10-26 `d4` and `q8-d4b`,
[release setup](../../data/2026-10-26_release_setup/README.md); 2026-10-27 `base`,
[MTP head](../../data/2026-10-27_mtp_head/README.md)):

| Bucket | Share of a task | Evidence | What could be saved |
|---|---|---|---|
| Generation | 76–79 % | per-request rows | below |
| … full passes after a missed or absent guess (one pass ≈ 29.7 ms, est.) | ~70 % (91 % of generation) | 15,658 of 46,460 generated tokens came from right guesses (h = 0.34) | each +10 points of right guesses saves ~2.4 ms per token (est.) |
| … … weight products (2.70 GB per token) | ~55 % | rc3 profile scaled to Q8_0 (est.): 22–24 ms of a pass | ≤ 5–7 ms per pass to the microbenchmark floor of 17.1 ms; fewer bytes |
| … … attention at the agent's 5–24k context | 6–9 % | 2.5–4 ms per pass (est.) | 1–2 ms per pass |
| … … card hand-offs and host gap | 2–9 % | 1–4 ms per pass, never measured directly (Q81) | unknown until Q81 |
| … right-guess cycles (the last card's ~5.3 ms) | ~7 % | the cost simulator | a lighter last card |
| Prompt processing | 19–21 % | per-request rows | below |
| … reply re-reads | 5.9–8.3 % | after some long replies the next request re-reads the whole reply: 15,284 tokens in 80 s once; the worst first-token waits, 60–83 s. Flint samples non-canonical token splits (the same text in other ids), and the conversation, sent back as text and tokenized again, stops matching the cache there ([reply re-reads](../../data/2026-10-28_reply_rereads/README.md)) | nearly all of it |
| … short prompts: micro-batches barely overlap, ~0.6 s floor per request | 11–14 % | 0.55–0.8 s per 128-token micro-batch in a 15–800-token turn, 0.26 s in a fresh 13.4k prompt | ~5 % of a task (est.) |
| Tools, agent, HTTP | 2–3 % | | ~0 |

- **The median task takes 1.2 min, the mean 1.9–2.5 min.** One task (`scale-generator`) takes 33–36 % of every run
  through single replies of 9–15k reasoning tokens; the five longest replies of a run hold 53–57 % of its generation.
- **No token was generated above 32k context** (the largest peak was 24,142), so long-context work does not move this
  benchmark, though it matters for longer real sessions.
- **Reasoning is ~65 % of generated tokens** in agent sessions (tool calls ~32 %, answer text ~3.5 %); copy and
  continuation guesses are right on 44 % of reasoning positions and 84 % of tool-call positions.
- **Pipelining's +52–59 %** (34.1 → 21.5–22.4 ms per token) is about half the clock rising to boost and half right
  guesses; the release line keeps the cards at boost 81 % of the time
  ([clock share](../../data/2026-10-27_clock_share/README.md)).

## The levers, ranked

### 1. Measured or exact, an hour of rig time each

| Registry | Lever | Expected gain | Needs |
|---|---|---|---|
| Q168 | p065 or p061: Q8_0's quality at −11.0 / −7.8 % bytes per token | est. +3–9 % generation (bytes-to-speed elasticity 0.3–0.8 on this rig) | the agent check against Q8_0 |
| Q193, Q169 | the cheaper MTP guess (2.93 ms) with the overnight heads (held-out acceptance 0.366, was 0.277) | simulated +5 %; +8 % at 0.30 right as served, +14 % at 0.40 | the paired replay (its head arm failed to load on 2026-10-28 for host RAM; Q265's check too) |
| Q202 | the answer store behind the copy lookup | +0.9–4.2 % live, +4.3 % offline | the paired replay |
| Q262 | the screened-head side file on the Q8_0 release line | measured +1.5–2.3 % on UD; est. +1.0–1.6 % on agent sessions | the heads are byte-identical; the screen is off whenever a grammar is set, so every agent request needs Q360 (the screen until the tool-call grammar triggers) first |
| Q263 | the end-of-prompt checkpoint and its 4-token tail micro-batch (v2.0 already takes one ring checkpoint and skips the re-reserve) | ~0.3–0.4 s per request, 1.5–2 % of a task (est.) | built: `LLAMA_SERVER_TAIL_CKPT_MIN`, one live agent-check arm |
| Q133 | memory clock offset | +3–5 % decode, exact when stable | the owner at the console |
| Q265 | the persistent MoE kernel's switch in v3 builds | up to 8 % if a v3 build runs the per-op path | a log and source check |

### 2. Small engine changes with a measured basis

| Registry | Lever | Expected gain | Needs |
|---|---|---|---|
| Q260 | stop the reply re-reads: keep the cached ids where the prompt spells the same text (`LLAMA_SERVER_TEXT_REUSE`) | 5.9–8.3 % of task time; the worst first token 60–83 s → 0.5–3.5 s | built and checked on the CPU; one live agent-check arm |
| Q261 | the decode-only card mask (`LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_MAX=16`) | **measured on the rig: 0.4 % slower per generated token** at the agent check's ~4.4k tokens (paired replay, identical tokens; Q116's gains were at 32k+) | off; Q430 (a context threshold) for long sessions |
| Q264 | short agent prompts: micro-batches overlapped as in a fresh prompt (Q363: smaller micro-batches for short prompts) | ~5 % of a task (est.) | Q157's busy profile on agent-sized turns first, after Q260 and Q263 |

### 3. Diagnostics that decide large engine work

| Registry | Lever | Expected gain | Needs |
|---|---|---|---|
| Q81 → Q270 | card-to-card hand-offs through mapped host memory and stream memory operations (or a one-warp spin kernel) instead of staged copies and cross-device events | est. +8–15 %, exact, if the 0.55–0.65 ms per boundary is software | Q81's hop measurement and the stream-memop device attribute (1–2 h), then 1–2 weeks |
| Q99 | page-walk (TLB) cost of sweeping a card's weights | est. +5–12 %; recount for Q8_0's ~4.7 GB per card | the 2–3 h probe, then 3–5 days |
| Q278 | the verify cost of 2–8-token steps on v2.0 now that Q78's multi-token kernels ship, which reopens Q52/Q53's batched copy blocks | copies cover 53 % of agent positions at 82 % right | 1 h of rig time |

### 4. Shorter reasoning: the largest lever on time per task (changes the text, so the agent check and task gates decide)

| Registry | Lever | Expected gain | Needs |
|---|---|---|---|
| Q267 | a concise-reasoning steering vector (SEAL, arXiv 2504.07986; ASC, arXiv 2507.04742) added inside `<think>` through llama.cpp's control vectors, plus a zero-cost "reason tersely" system-prompt arm | est. 1.1–1.3× per task if 15–30 % shorter reasoning keeps the pass rate | 2–3 days on the quant machine from agentic run 1, then gating the vector to the reasoning span |
| Q268 | early exit when reasoning converges (PUMA, arXiv 2605.17672; CoDE-Stop, arXiv 2604.04930) | est. +5–15 % on answer turns | an offline replay on existing logs |
| Q266 | a runaway guard for 9–15k-token replies | sets the p90 task time | a task-level gate |
| Q93 | the first-finish race (k = 2) | 0.72–0.75× reasoning offline; est. 1.10–1.23× per answer | 1–2 weeks of server work; competes with guesses for the idle cards |
| Q85 | a reasoning budget | unmeasured | task gate |
| Q269 | a short RL phase for concise reasoning (arXiv 2504.05185) | est. 1.2–1.5× | weeks; parked |

### 5. Engine and guessing work of a week or more

| Registry | Lever | Expected gain |
|---|---|---|
| Q271 | block drafting for the staircase head (P-EAGLE, arXiv 2602.01469; DFlash): one pass fills 2–3 pipe slots | est. +3–8 % |
| Q276 | top-κ hedged guesses in spare slots | est. up to +10–30 % over a top-1 drafter (idle cards note) |
| Q272 | an answer store that grows online from the user's own sessions (SuffixDecoding) | est. +1–4 % |
| Q275 | a successor table from the screened head's top-K (Token Recycling, arXiv 2408.08696) | est. +1–3 % |
| Q277 | a lighter last card for Q8_0 | shorter right-guess cycles; needs a fit probe |
| Q273, Q248 | past p065: low-rank error compensation for the early layers (EoRA, arXiv 2410.21271); Q6_K and GPTQ choices per group after Q101's gate | est. +3–6 % |
| Q97 | Q8_0 with one scale per 128 weights; the F16 router order | est. +2–4 % |
| Q98 | split output head | +1–3 %; overlaps Q262 |

Parked, with the reason in their rows: expert skipping (lossy on Flint's flat routing, Q288), Hogwild! inference
(Q279), position-independent KV reuse (Q280), top-5 experts (Q281), sliding-window conversion of the global layers
(Q282), per-stage cancel (Q283).

## What it could add up to

- **Within a few days, from groups 1 and 2:** generation est. +12–20 % (p065, the head, the store, the screened head,
  the memory clock; these overlap partly), and time per task a further −8–10 % from the re-reads and
  the end-of-prompt checkpoint. About 45 → 52–55 tok/s, with the worst first-token waits gone.
- **With group 3's engine work, if the diagnostics confirm:** hand-offs and page walks est. +10–20 % more.
- **Shorter reasoning** multiplies on top of everything: 15–30 % fewer reasoning tokens at the same pass rate would cut
  time per task by 10–20 % with no change to tok/s.
