# Idle cards: where each card's time goes, and how to keep all nine working (2026-10-24)

Scope: when the rig's cards sit idle while Flint (49 layers, 40B-A2.5B MoE) runs as a layer split over six or nine
P106-090 cards, how much of that idle time the layer split forces and how much host work, waits, clocks, syncs and
transfers add, and every published or upstream way to fill idle pipeline stages: speculation variants, asynchronous
pipelines, prompt chunks of other requests, drafters on idle stages, early-exit and Jacobi decoding, races, staggered
micro-batches, expert parallelism, attention offload and side work. It reads the Q87/Q104/Q120/Q121 rig runs token by
token, then surveys papers (2023–2026), llama.cpp pull requests, issues and discussions, ik_llama.cpp and other forks,
and other engines. It builds on and does not repeat [pipelined speculation](pipelined_speculation.md),
[pipeline drafter](pipeline_drafter.md), [multi-conversation serving](multi_conversation_serving.md),
[host decode loop](host_decode_loop.md), [prefill on Pascal](prefill_pascal.md), [power and tuning](power_and_tuning.md),
[slim Pascal fork](pascal_slim_fork.md), [breakthrough candidates](breakthrough_candidates.md),
[unconventional ideas](unconventional_ideas.md), [speculation plumbing](speculation_plumbing.md),
[multi-GPU physics](multigpu_pascal_performance.md), [distributed inference](distributed_inference.md),
[v2 synthesis](v2_synthesis.md) and [fork scope](fork_scope.md). Conventions are those of the other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Boost" means the cards' top clock bins (~1,830–1,911 MHz), "base" the 1,354 MHz base clock.
- L is the time one token needs to pass through every card; c is the pipeline cycle, the time between two tokens leaving
  the last card when the cards are full; S_eff = L / c is how many tokens the pipeline holds before anything waits.
- "Busy" is the share of wall time a card spends computing.

Data (measured; K2h-r16 with the MoE down rescale off, fork build `pascal-p106-v2-pipespec2`, 512 generated tokens per
job, nine cards on the default split (est. 6/6/6/6/5/5/5/5/5 layers) and six cards (9/8/8/8/8/8); per-token traces, a
per-second clock and utilization log of every card, the executor's host timers; raw logs are kept outside the
repository):
- Plain decoding 31.0 tok/s (nine cards) and 32.6 (six) in the Q87 runs, 33.7–34.2 in the Q120 runs; every
  pipelined row of Q87 and Q120 in [the registry](../../../quants/experiments.md).
- The all-right oracle at depth 8 on nine cards: 186 tok/s; every guess wrong at depth 3: 38.5 tok/s (nine cards) and
  40.8 (six); 70 % right at depth 3: 81.9 and 83.9.
- Eight staggered conversations (Q120, packed uploads): 66.3 tok/s in total; device-fed tokens (Q121): 22.3.
- Prompts (Q117, combined build, `-ub 512`, nine cards): 228 / 613 / 723 tok/s at 512 / 2,048 / 8,192 tokens.

## Verdict

- **The layer split forces most of the idle time; the measured shortfalls are fixable.** In single-stream decode each
  card computes 1/9 of a token's pass (8.7 % of wall time on nine cards, 14 % on six, est.); nothing but more work in
  flight changes that. The gaps between what was measured and what the cards can do (31 against 38.5 tok/s single
  stream, 186 against ~290 for the all-right oracle, 66 against ~265 for eight conversations, 723 against ~1,360 for an
  8k prompt) come from the clock, the hand-offs between cards, the executor's rebuild stalls, prefill blocking decode and
  the host's waits in prompt processing, all fixable.
- **Idle cards are slow cards: the clock costs single-stream decode ~20 %.** With each card ~11 % busy, NVIDIA's
  utilization-based clock policy holds the cards at the 1,354 MHz base clock. The same chat prompt decoded by the same
  plain loop in one process ran at 34.0 tok/s at base clock and 38.5 tok/s at boost (29.3 against 26.0 ms per token).
  Guesses raise every card's duty and with it the clock: on nine cards every-guess-wrong at depth 3 decodes at 38.5
  tok/s at ~1,805 MHz against plain's 31.0 at 1,354 MHz, 1.24× with no right guess at all (six cards 1.25×).
- **Q87's 1.3–1.55× is two effects.** At matched clocks (against the every-guess-wrong rows) today's copy lookup and
  continuation table give 1.10–1.15× on chat and code and ~1.2–1.3× on rewrites; the rest is the clock rising from base
  to boost. The 70 %-right oracle is 2.06–2.13× at matched clocks (2.57–2.64× against plain). Speed-ups of any filling
  scheme must be read against a baseline at the same clock; the every-guess-wrong oracle is that baseline.
- **The pipeline is faster than its oracle ceiling suggests.** The median time between tokens in the all-right oracle at
  depth 4–8 on nine cards is 3.37–3.46 ms (292 tok/s, S_eff 7.2), and a token's pass at boost is 24.6–24.8 ms; on six
  cards 4.5–4.6 ms (219 tok/s, S_eff 5.1) and 23.4–23.6 ms. The measured 186 tok/s mean is dragged down by the
  executor's graph rebuilds at every 256-cell KV step (Q153): at depth 8, 90 of 510 intervals take 8–12.5 ms, 24 % of
  the run.
- **Eight staggered conversations do not stall in steady state; prefill does.** In Q120's eight-conversation job the
  first ~46 s of 62 s were spent prefilling 5,740 prompt tokens (two 2,076-token and two 750-token prompts) as 64-token
  chunks with at most two chunks in flight. While a chunk occupies a card (~0.1 s, est.), every decode token queued
  behind it on that card's single in-order stream waits, so decode crawled at ~5 tok/s in total. Once every prompt was
  in, all nine cards ran at 61–98 % utilization (last card 98 %) and the conversations finished within 1.3 s of each
  other: the steady state is est. ~265 tok/s (8 × ~33), 90 % of the 292 tok/s ceiling. Four conversations: est. ~152
  tok/s (4 × 38), as the closed-pipeline model min(N / (L + h), 1 / c) predicts. The 66 tok/s and the "13 ms of waits
  per token" are averages over a run dominated by prefill. Packed uploads (Q120) are worth ~1.9× in that steady state
  (est. ~140 tok/s with synchronous copies), not the +21 % the diluted totals showed.
- **Device-fed tokens collapse streams because of where the token copy is queued.** The 4-byte copy of a sampled id
  goes onto the last card's stream at submit time, behind every item already queued there, so each fed token waits for
  the whole pipeline to drain through the last card: the fed totals are 22.1 tok/s at N = 4 and 22.3 at N = 8, the same
  rate whatever N (est. mechanism). Queuing the copy on card 1's stream behind a wait on the source slot's own event
  removes it.
- **Prompt processing leaves 65–89 % of card time idle (est.).** A 512-token prompt at `-ub 512` is one micro-batch, so
  one card works at a time. Longer prompts overlap, but llama.cpp's scheduler waits on the host for each card's input
  copy, which caps the micro-batches in flight at about the four scheduler copies (est. 35 % busy at 8k on nine cards).
  Upstream measured +25 % at 16k from eight copies on a ROCm multi-GPU box (#20395); the device-side wait of Q120's
  packed uploads, ported to prompt micro-batches, removes the cap.
- **Nobody upstream overlaps decode steps of different sequences across a layer split.** llama.cpp's pipeline
  parallelism (#6017) overlaps one call's prompt micro-batches only; graph reuse synchronizes the scheduler before
  every step (#20927); the server merges all slots into one step and samples after it; prefill chunks block every slot
  (#29175, open). The one public implementation of staggered multi-slot decode on llama.cpp is a downstream fork
  (llama.cpp-multigpu, "decode groups": five sessions at 250k from 2.3 to 27.3 tok/s each on six RTX 3090s). vLLM (MRV2)
  and SGLang added speculative decoding under pipeline parallelism in September 2026, with the drafter on the last stage;
  none of them runs on compute 6.1.
- **Expert parallelism, attention offload and disaggregation stay rejected:** expert parallelism moves 1.6–3.2 MB per
  token (25–50× the layer split's 64 KB) through 98 host-mediated exchange phases per token; attention offload needs
  ~450–500 KB and 98 round trips per token; a second copy of the weights does not fit.
- **First experiment (rig, ~25 min, no code):** `rig_hostloop.sh` with `CONFIGS=u2` and short prompts only
  (`chat.txt,code.txt`), N = 1/4/8, then the same with the light last card. It shows the steady-state rate without
  prefill (expected 250–290 tok/s at N = 8 on the default split, est.) and decides whether any steady-state stall is
  left to find. Kill for the reading above: N = 8 below 180 tok/s.

## Idle-time accounting

Per card and per token, nine cards unless noted. Card compute per token is est. 20.2 ms at boost (49 layers × ~0.38 ms
plus ~1.5 ms for the output head, sampler and output copy); the rest of a 24.7 ms pass is ~0.55–0.65 ms per hand-off
between cards (est., section 1.1). Wasted work is dropped guesses, which run through every card.

| Mode | tok/s (useful) | Busy per card: mean / bottleneck card | Rest of each card's time | Structural | Fixable |
|---|---|---|---|---|---|
| Single stream, plain, at base clock (as it runs) | 31.0–34.0 (29.3–32.3 ms) | ~9 % / ~13 % (est.) | other cards' turn ~73 %; hand-offs ~14 %; host gap ~3 %; slices ~18 % slower than at boost | 8/9 of each pass | clock (+20–24 %), hand-offs, host gap |
| Single stream at boost (Q120 references after a prompt, streams N = 1) | 38.2–39.4 (25.4–26.2 ms) | 8.7 % / 13 % (est.) | other cards' turn ~70 %; hand-offs ~17 %; host gap ~4 % | 8/9 | hand-offs, host gap |
| Six cards, single stream at boost (every-guess-wrong oracle) | 40.8 (24.5 ms) | 14 % / 19 % (est.) | other cards' turn ~69 %; hand-offs ~13 %; host ~4 % | 5/6 | as above |
| Pipelined speculation, `cont`, chat, D = 1 / 3 / 5 | 35.8 / 38.3 / 42.7 | 11 / 20 / 30 % (est.), of it useful 8 / 9 / 10 % | dropped guesses 3 / 11 / 20 %; idle 89 / 80 / 70 % | – | guess quality; KV-step stalls |
| Same, six cards, D = 3 | 44.8 | 34 % (est.), useful 15 % | dropped guesses 19 %; idle 66 % | – | as above |
| Oracle, 70 % right, D = 3 | 81.9 (six: 83.9) | 35 % (est.), useful 18 % (six: 53 % / 28 %) | dropped 16 %; idle 65 % | – | stalls |
| Oracle, all right, D = 8 (six: D = 5) | 186 (six: 168) | 42 % (est.); last card 63 % computing + 24 % stalled (six: 57 %; last 77 % + 13 %) | idle 58 % | last card sets the cycle (3.4 ms; six 4.6 ms) | KV-step stalls (→ ~290 / ~219 tok/s) |
| 8 staggered conversations, whole job (Q120) | 66.3 | – | 75 % of the job was prefill | – | prefill scheduling |
| … prefill phase (46 s) | ~5 decode + ~125 prompt | 1–3 cards busy (utilization sum 100–300 %) | a 64-token chunk holds one card ~0.1 s while decode waits behind it | – | chunk size, chunks in flight, priority |
| … steady phase (15 s) | ~265 (est.) | 74 % (utilization sum 662 %) / last card 98 % | non-bottleneck cards 20–40 % idle | last card is the slowest stage | KV-step stalls; light last card |
| 4 staggered conversations, steady phase | ~152 (est.) | 41 % (utilization sum 373 %) | – | 4 / S_eff of capacity | more conversations |
| Prompt processing, pp512, `-ub 512` | 228 | ~11 % (one micro-batch) | 8/9 idle | yes, at one micro-batch | smaller `-ub` with PR #31047 |
| pp2048 / pp8192, `-ub 512` | 613 / 723 | ~30 % / ~35 % (est.) | fill and drain; host waits per input copy cap ~4–5 micro-batches in flight | fill/drain only | device-side input waits; more micro-batches; copies |

Structural against fixable, single stream at boost on nine cards (25.7 ms per token, est.):

| Where a card's time goes | ms per token | Share | Kind |
|---|---|---|---|
| Its own layers (mean card) | 2.2 | 8.7 % | busy |
| The other eight cards' layers | 18.0 | 70 % | structural (the layer split) |
| Hand-offs between cards (8 × ~0.55–0.65) | ~4.5 | ~17.5 % | fixable in part (hop mechanism); overlapped once ≥ 2 tokens are in flight |
| Host gap between sample and next launch | ~1.0 | ~4 % | fixable (input layer on card 1, fed tokens) |
| At base clock instead of boost: every slice stretched | +3.6–5.7 | +14–22 % | fixable (duty, core offset) |

## Ranked ideas

Gains are against today's fork at the same clock unless stated; "busy" is the mean card's busy share. Nine cards.

| # | Idea | Busy | Single stream | N conversations | Effort | Evidence | First experiment (rig ≤ 30 min unless noted) | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| 1 | **Prompts as pipelined chunks with decode priority in the streams mode** (the combined build with PR #31047; several chunks of one prompt in flight, since each card runs them in order; 32–128-token chunks; at most one chunk per card per decode round, or a prefill wave with decode paused when the backlog is large) | prefill phase 1–3 → 5–7 cards | time to first token of a 2k prompt next to 8 conversations ~32 s → 3–6 s (est.) | the Q120 job mix: 66 → est. 150–200 tok/s; decode during prefill ~5 → 60–100 tok/s (est.) | 2–4 days in pipespec | Q120 logs (section 4); Medha stream pipeline, Jupiter, SGLang chunked pipeline 3.31×, TD-Pipe, PipeSwift, THETA, llama.cpp #29175, llama.cpp-multigpu patches 28–36 | `rig_hostloop.sh CONFIGS=u2 PROMPTS=chat.txt,code.txt` (steady state); then `CHUNK=128` with all four prompts | N = 8 short-prompt total < 180 tok/s: a steady-state stall exists, profile it first |
| 2 | **Count the clock and use it: keep ≥ 3 tokens in flight on nine cards (≥ 1 on six)** whenever a conversation generates, guesses or not (the `cont` fallback always has one); the server's `--pipe-depth` path does this for a lone slot | 9 % → 25–35 % | +20–24 % over plain as it runs (31.0 → ~38.5, measured as every-guess-wrong); +60–120 W (est.) | none (cards busy anyway) | none in pipespec; llama-server via the pipeserve branch | gpu logs of Q87/Q120 (section 1.6); [power and tuning](power_and_tuning.md) | plain, every-guess-wrong oracle at D = 1/2/3 and `cont` D = 3 on chat, with the clock log and wall power | every-guess-wrong D = 3 < 1.15× plain in the same session |
| 3 | **Remove the executor's 256-cell KV-step stalls** (Q153: capture without an eager token, prebuild the next bucket per slot, or one fixed decode graph) | all-busy modes +5–30 % | +1–3 % | +25–31 % for deep guessing and races (oracle D = 8: 186 → 235–290 tok/s); est. +5–10 % for N = 8 streams | 1–3 days | Q87 traces: mean 4.54 against median 3.46 ms at D = 8; 18 slots × +9.7 ms per step ([slim Pascal fork](pascal_slim_fork.md)) | offline: stall counts per job from the existing traces; after the fix, oracle D = 8 | mean within 5 % of median after the fix, or no tok/s change |
| 4 | **Device-side waits for prompt micro-batch inputs** (Q120's packed uploads in `llama_decode`, or upstream #25319), plus more micro-batches (`-ub` 128–256 with PR #31047 and Q109) or 8 scheduler copies | pp8192 35 → 60–80 %; pp512 11 → 30–40 % (est.) | prompts: pp8192 723 → 1,000–1,400; pp512 228 → 350–450 tok/s (est.) | prompt intake for agents | copies: a build flag; staged inputs 2–4 days | #20395 (+25 % at 16k, 4 → 8 copies, ROCm multi-GPU); #19922; TurboPrefill 3–4.5× on x1 links; llama.cpp-multigpu patch 12 | `llama-bench -p 512,2048,8192 -ub 128,256,512` on the combined build with 4 and 8 copies (an 8-copy build from the quant machine) | pp8192 gains < 10 % with 8 copies at `-ub 256` |
| 5 | **Fed tokens without the drain:** queue the 4-byte token copy on card 1's stream behind a wait on the source slot's output event | – | +4–5 % (d0-fed 33.9 vs d0 32.4), less after the input layer moves to card 1 | removes the 3× regression (n8-fed 22 tok/s → n8's rate) | ½–1 day | Q121: fed totals equal at N = 4 and 8 (22.1, 22.3) | n4-fed / n8-fed with short prompts after the change | n8-fed below n8 |
| 6 | **Trained staircase drafter (Q20)**, behind the copy lookup | up to ~60 % | at matched clock (PPSD's formula, S_eff 7.2): α 0.44 → 1.6×, 0.55 → 1.9×, 0.7 → 2.5×; six cards (S_eff 5.1) 1.55 / 1.8 / 2.3× (est.) | guesses only in the slack below S_eff | running (training) | Q20 pilot 0.38–0.44 and rising; PPSD, SPD | covered by Q20's gate | coupled α < 0.5 on held-out text |
| 7 | **Top-κ hedged guesses on spare slots:** when the drafter's top guess is uncertain, the second candidate's chain runs as its own sequence | +5–15 % | +10–30 % over a top-1 drafter at α 0.4–0.6 (est.) | only below S_eff | 3–5 days (a sequence per branch; the race path exists) | Mirror-SD, Speculative Speculative Decoding (Saguaro), SpecBranch | offline: `pipeline_sim.py` with Q20's top-2 probabilities | simulated gain < 5 % |
| 8 | **Light last card** (`-ts 6,6,6,6,6,6,6,5,2`) | bottleneck shifts to a 6-layer card | none | ceiling 292 → est. 330–345 tok/s (+13–18 %), only at ≥ 7 in flight | a flag | the 3.4 ms cycle is the last card (5 layers + head) | oracle D = 8 and N = 8 short prompts with the light split (15 min) | median cycle ≥ 3.3 ms |
| 9 | **Cheaper hand-offs between cards** (Q81 first; zero-copy mailboxes in mapped host memory if a hop costs ≥ 0.2 ms; six cards for one conversation) | – | +5–12 % if the hand-off halves (est.) | none (overlapped) | Q81 1–2 h; mailbox 1–2 weeks | est. 0.55–0.65 ms per boundary from L and c on six and nine cards | Q81's hop benchmark | hop ≤ 0.2 ms |
| 10 | **Race control and consensus guesses** (Q93's race plus early-stop rules and samples guessing for each other) | raises useful share in races | answer time −20–50 % on suitable tasks (quality-gated, est.) | – | days + task checks | ESC, ST-BoN, DeepConf, DEER, Dynasor; multi-sample speculative decoding (EMNLP 2025 Findings) | offline on Q41's five-attempt logs: cross-sample n-gram acceptance | acceptance gain < 5 % |
| 11 | **llama-server multi-slot staggered decode** (the streams mode in the server; covered by Q104) | as streams | – | `-np N` today ~37 → est. 150–265 tok/s at N = 4–8 | 1–2 weeks | llama.cpp-multigpu decode groups; vLLM/SGLang pipeline loops | design read of the fork's patches 19–28 (no rig) | – |
| 12 | Side work on idle cards (hidden-state probes for streaming moderation or confidence, a 0.6B guard model) | small | none | none | days | probes < 1 ms per token; Stream Qwen3Guard | – | – |
| – | Rejected: expert parallelism and hybrid layer + expert splits; attention or KV offload below ~64k; prefill/decode disaggregation; CPU drafters; Jacobi, lookahead, CLLMs; batched trees; layer-skip self-drafting; lossy early exit and lenient acceptance; `-sm tensor` and ik_llama.cpp's graph mode; one rpc-server per card; one host thread per card | | | | | section 2 | | |

## llama.cpp pull requests, issues, discussions and forks

States checked with the GitHub API on 2026-10-24. "Helps us" judges the rig (layer split, x1 links, no P2P, 2-core host).

| Item | State (dates) | What it does or reports | Helps us |
|---|---|---|---|
| PR #6017 pipeline parallelism | merged 2024-03-13 | Prompt micro-batches overlap across GPUs with `GGML_SCHED_MAX_COPIES` (4) input copies; pp8192 4,373 → 16,992 t/s on 8 × A100 PCIe, generation 146 → 135 | the prompt overlap we use; nothing for decode |
| PR #4918 pipeline demo | closed 2024-03-01 | First micro-batch demo | history |
| Issue #13751 / PR #13814 | closed / merged 2025-05-27 | Layer split with 4 copies slower than one GPU because CUDA graphs were off under pipeline parallelism; #13814 allows graphs; decode always reuses copy 0 | confirms decode steps never overlap |
| PR #20463 / PR #20927 | merged 2026-03-12 / 2026-03-24 | Graph reuse off under pipeline parallelism (a perplexity bug), then back on with a full scheduler synchronize before `set_inputs` | the wait behind Q109 and behind "one decode step at a time" |
| PR #17795 → #20793 → revert #25138 (issue #25062) | merged 2026-03-05, 2026-06-26, reverted 2026-06-30 | Fewer synchronizations between tokens (+1–2 %); the reintroduction cost ~50 % prefill on 4 × MI50 | sync removal must be per backend; keep ours in the executor |
| PR #25319 asynchronous scheduler memory copies | open draft (2026-07-05) | Consensus design for host-free input copies; +5 % TG on a UMA device, prefill gains on two GPUs | yes, for prompt micro-batches (idea 4) |
| PR #19922 event-driven pipeline scheduler + "xdev host staging" | closed 2026-03-03 | Built on 10 × MI50: the CPU blocked at split sync points and left GPUs idle; GPU events, a queue throttle, host-staged cross-GPU copies; mixed tester results | read for idea 4 |
| PR #20395 `--sched-n-copies` | closed 2026-05-08 | gpt-oss-120B pp16384 2,166 → 2,706 t/s with 8 copies (16 no better) | yes, idea 4 |
| PR #19042 `CUDA_SCALE_LAUNCH_QUEUES=4x` | merged 2026-01-27 | The CPU-side launch queue filled under pipeline parallelism and left GPU bubbles; +25 % prompts | check it is active on the rig's CUDA 12.9 |
| Issue #22926 / PR #28387 | closed 2026-06-25 / merged 2026-09-07 | The CPU embedding adds a graph split; `token_embd` on GPU 0 gave 1.5–3× prefill on two GPUs; inputs no longer force a split | our `LLAMA_INPUT_ON_GPU` (slim note B2) |
| PR #26040 split race fix | merged 2026-08-20 | A split without inputs waits on the previous split's event | the scheduler stays sequential |
| PR #27311 / #29450 input ring buffers | open drafts | Ring buffers for inputs to allow overlap | prompt overlap |
| PR #27423 adaptive ubatch shrink under pipeline parallelism | open draft | 2 × MI210 utilization 62 → 85.6 % | prompt fill (idea 4) |
| PR #28872 keep the scheduler on re-reserve | open draft | 115 + 424 ms of pinned frees and allocations per request at 262k | per-request cost (Q111) |
| Issue #29829 | open (2026-10-01) | Non-blocking dispatch of successive prompts on one GPU | the same API gap as overlapped decode |
| Discussion #20252 "Does llama.cpp actually support pipeline parallelism?" | 2026-03-08 | Yes, for prompts; P40 and RX 6800 owners saw one GPU active at a time and no gain without P2P | P2P-less boards may get little prompt overlap: measure |
| Discussions #16621, #9364, #17220 | no replies / idea | Using idle GPU periods; overlapping attention and FFN of two batches | – |
| PRs #10497 / #10455 `--device-draft` | merged 2024-11-25 | Draft model on its own GPU, run in lockstep with the target | no (no same-tokenizer drafter) |
| PR #22838 parallel drafting | merged 2026-05-11 | One draft context drafts for all slots at once | batched, not asynchronous |
| Issue #27428 / PR #29523 | open / closed 2026-09-27 | Draft-MTP halves layer-split prompt speed (973 → 532 t/s); #29523 lagged the MTP catch-up through pinned host views so the next device can start | only for MTP models on a layer split (Qwen3.6 track) |
| PR #25818 remote speculative decoding | open | Drafts from another llama-server over Ethernet; synchronous per step | no |
| PR #3334 (MPI, PipeInfer's author) / PR #7395 | open draft since 2023 / merged 2024-05-19 | MPI per-node options; the MPI backend was removed for RPC; Discussion #10344 is the only upstream mention of PipeInfer | no upstream pipelined speculation exists |
| Discussion #6853 asynchronous speculative execution | 2024-04-23 | Draft on CPU cores while the target runs on the GPU | no (2-core host) |
| PR #23287 / #28549 | merged 2026-05-20 / 2026-09-16 | MTP draft with backend sampling (+7 % on 2 × 5090); CUDA graphs for the MTP draft | MTP only |
| PR #10718 server chunked prefill | open draft since 2024-12-08 | Proof of concept for not blocking generating slots; the maintainer doubts its value | the problem of section 4.2 |
| PR #14343 per-slot token cap | closed 2025-06-23 | n_batch/4 per slot per step; rejected for lower throughput | – |
| Issue #29175 | open (2026-09-20) | One prefill chunk blocks decode for every sequence; stalls = ceil(prompt / n_batch); `-b` 4,096 → 16,384 turned a co-tenant's worst stall from ~8 s into ~31 s; the reporter caps prompt tokens per step at 1,024 while slots decode (3.5–3.8× smaller stall) | yes, for llama-server `-np` today |
| Discussion #26022 decode-priority policy | 2026-07-23 | 4 × A100 layer split: strict decode priority cut short-request latency 17.5 → 8.8 s and TPOT 106.5 → 37.8 ms at equal throughput, starving prefill | policy evidence for idea 1 |
| Issue #6607 | closed 2026-03-17 | FIFO prompt batching blocks slots | – |
| Issue #28495 / PR #29510 | open / open | `--kv-unified` costs 42–54 % prompt speed (FA kernels skip only the mask tail); a rows variant closes most of it | keep one stream per conversation |
| Issue #27050 | closed 2026-09-28 | Backend sampling +48 % at 32 slots (706 → 1,046 tok/s on an RTX 5090): the logits copy and CPU sampling dominated | GPU sampling per conversation (done in pipespec) |
| Issue #25840 | closed 2026-09-03 | Non-contiguous busy slots decode 4.4× slower (hybrid model) | slot binding |
| Issue #21266 / PRs #25675, #27058 | open / closed / open | Disaggregated prefill on other nodes | no (needs a second copy) |
| Issue #24355, PRs #24095/#24204/#24205, #25997 | closed (2026-06 / 07) | Runtime switches for copies and pipeline parallelism; 0.5–1.5 GB of VRAM for idle copies | copies trade VRAM |
| PR #22789 | merged 2026-08-03 | Dynamic split-input limits (fixes an assertion on > 1 GPU) | – |
| PR #18626 RPC events and async backend | merged 2026-08-26 | Pipeline parallelism over RPC; #24675's numbers: 2 × MI25 workers 528 → 905 t/s prompts | not on one host with 2 cores |
| PRs #7915, #8032, #29287 | open / open draft / open draft | Async RPC, server-to-server copies, RPC graph cache (TG 2.44–2.51× for `-sm tensor` over RPC) | no |
| PR #1898 | merged 2023-06-17 | One CUDA stream per device: the origin of in-order work per card (sections 4.2, 4.3) | explains head-of-line blocking |
| PR #4719 multi-stream graph evaluation | open draft since 2023-12-31 | Slower than one stream on the author's systems | – |
| PR #16991 stream concurrency in one GPU | merged 2025-11-30 | Q/K/V branches on extra streams (`GGML_CUDA_GRAPH_OPT=1`), +1–9 % TG | small, 5 SMs |
| PR #21846 | closed 2026-04-13 | "Avoid cudaMemcpyPeerAsync without P2P" rejected: the staged fallback works | hops stay staged copies |
| PR #25228 virtual CUDA devices | merged 2026-07-16 | Several devices on one GPU for multi-GPU tests | used for the executor's checks |
| PR #30804 chunked CUDA graphs | merged 2026-10-23 | 1–2 % TG from amortized graph launch | small |
| PR #16548 CUDA graph plans | open | CUDA graphs for split graphs | – |
| PR #19378 `-sm tensor` ("meta" backend) | merged 2026-04-09 | TP with AllReduce per layer; a 35B-A3B MoE decodes at 0.69× on 2 × 4090; experts are sliced across GPUs, not placed (no expert parallelism) | no (x1 links) |
| PRs #22299, #22729 | merged | Internal AllReduce; `docs/multi-gpu.md`: layer mode tolerates slow links, tensor mode does not | – |
| Issues #29466, #30075 | open | `-sm tensor` on P100s crashes or degrades | no |
| PR #11397 `-ot` | merged 2025-04-02 | A layer's experts are one 3-D tensor: whole blocks can move, one layer's experts cannot be spread | no expert placement without new code |
| Issue #23855 | closed 2026-08-04 | Request for "KV-cache GPUs" and "MoE GPUs" | – |
| ik_llama.cpp PR #1022, #1067, #1080 | merged 2025-12 | Graph-parallel (TP) mode for MoE; TG 60.5 vs 91.8 t/s for layer mode at zero context on 2 × 3090, ahead only past ~20k; 2.4× at 100k on 4 × 3090 | no (AllReduce per layer) |
| ik_llama.cpp PR #1069 | merged 2025-12-17 | Don't wait for the previous split before copying the next inputs (+25 % prompts) | the idea behind idea 4 |
| ik_llama.cpp PR #1089 | merged 2025-12-27 | One submitting thread per GPU (`-sas`), graph mode only | no (2 cores) |
| ik_llama.cpp PR #1092 | merged 2025-12-26 | Ring reduction, ~+40 % without P2P | graph mode only |
| ik_llama.cpp PR #751 | merged 2025-09-02 | Default copies = 1 (no prompt overlap on `-sm layer` unless rebuilt) | stay on mainline for prompts |
| ik_llama.cpp PR #2744 | open (2026-10-24) | Fixes `-sm attn` for Flint (CUDA_Split buffer 22,873 → 891 MiB) | attention-only split, untested here |
| Fork llama.cpp-multigpu (lukolszewski) | active (2026-09-11 → 10-22), MIT | 47 patches: per-sequence decode ubatches on a stream-agnostic graph, several decode batches in flight with per-batch output slots ("decode groups"), samplers that wait only for their own batch, rotating pinned input staging, the KQ mask on the device, a hop copy that waits for the destination's previous split, no pipeline drain on re-plan, 256-token prefill chunks while slots decode. Six RTX 3090s, constrained PCIe: five sessions at 250k from 2.3 to 27.3 tok/s each; two users at 58k see ~280 ms between tokens instead of 1.3 s while a 200k prompt prefills | the closest public design to Q104 in llama-server |
| TurboPrefill (RFC #24219, closed 2026-08-17) | file overlay | Intra-prompt pipeline scheduling; 5 × Radeon VII on Gen3 x1: 16k prefill 115 → 347 t/s; 4.5× on 10 × P104-100 at Gen1 x1 | prompt fill on x1 rigs (covered in the prefill note) |
| PipeInfer fork | 2024 | MPI asynchronous pipelined speculation on a 2023 llama.cpp | covered |
| prima.cpp | ICLR 2026 | Piped-ring parallelism over home Wi-Fi clusters; disk-bound 70B at 674 ms per token | no (weights fit in VRAM) |
| P100 patch set (shinbunbun) | 45 patches | sm_60 AllReduce for `-sm tensor`, scheduler tweaks (+0.97 % with 4 decode slots) | no |

## 1. Idle-time accounting (question 1)

### 1.1 What the traces measure

Each pipespec trace records the time of every sampled token and what was in flight behind it; the clock log gives each
card's clock and utilization every second.

| Quantity | Nine cards | Six cards | Read from |
|---|---|---|---|
| L, one token's pass at boost | 24.6–24.8 ms | 23.4–23.6 ms | median interval with every guess wrong (oracle D = 3), and after misses of `cont` |
| c, the cycle with the cards full | 3.37–3.46 ms (292 tok/s) | 4.51–4.64 ms (219 tok/s) | median interval of the all-right oracle at D = 4–8 (D = 3–5 on six) |
| S_eff = L / c | 7.2 | 5.1 | |
| Per layer | ~0.38 ms (est.) | | the last card holds 5 layers + head (c = 3.4 ms) on nine, 8 + head (4.55 ms) on six |
| Output head, sampler, output copy | ~1.5 ms (est.) | | same two equations |
| Hand-off between cards | ~0.55–0.65 ms each (est.) | | L − 49 × 0.38 − 1.5, over 8 and 5 boundaries |
| Host gap, plain decoding | ~1.0 ms (est.) | | plain at boost (25.7 ms) − L |
| Plain at base clock | 29.3–31.4 ms | 29.8 ms | Q120 references at 1,354 MHz; Q87 plain at 1,354–1,460 MHz |

The hand-off term holds everything per card boundary that is not layer compute: the staged copy through host memory,
the cross-device event, the gap before the next card's graph starts. Q81 has not measured a hop; the simulator assumed
0.3 ms. If the 0.55–0.65 ms holds, hand-offs are ~18 % of a nine-card pass and the reason six cards decode a single
stream about as fast as nine.

### 1.2 Single-stream decode

One token in flight: card j computes its slice (2–3.4 ms) and idles for the rest of the token. At boost the mean card is
busy 8.7 % of the time on nine cards (14 % on six); the summed NVML utilization of the nine cards reads 93–100 %, one
card at a time. Of the idle 91 %, 70 points are the other cards' slices (structural), ~17.5 the hand-offs and ~4 the
host gap (table above). As single-stream decoding actually runs, the cards sit at base clock (section 1.6) and every
slice takes ~15 % longer: 29.3–31.4 ms per token instead of 25.7.

### 1.3 Pipelined speculation at depth 1–5

- **Right guesses** leave the pipeline one cycle apart (3.4 ms on nine cards); **wrong ones** cost a full pass from the
  corrected token (L ≈ 24.7 ms), and their work is dropped, not withdrawn, so every dropped guess still runs through
  every card. With `cont` on chat (56 right and 218 wrong of 512), each depth adds 0.43 dropped tokens per useful one:
  at D = 1 / 3 / 5 the cards push 1.43 / 2.28 / 3.13 tokens through per token kept.
- **Busy (est., compute 20.2 ms per token):** 11 / 20 / 30 % on nine cards, of it 8 / 9 / 10 points useful; six cards
  D = 3 34 % (15 useful). The 70 %-right oracle keeps cards 35 % busy (18 useful) on nine and 53 % (28 useful) on six.
- **Wall time** is set by the miss rate: a run of misses is a run of full passes, which is why depth beyond 3 adds
  little on chat. Dropped guesses cost no wall time below S_eff but raise the cards' duty, and with it the clock: that
  is the 1.24× of the every-guess-wrong rows.
- **At matched clocks** (every-guess-wrong rows as baseline, 38.5 tok/s nine cards, 40.8 six): chat 1.11× (nine, D = 5)
  and 1.10× (six, D = 3), code 1.15× and 1.13×, rewrite ≥ 1.2–1.3× (its plain run was already near boost: 35.9 tok/s at
  1,807 MHz against 46.3 at D = 3), the 70 %-right oracle 2.13× and 2.06×.
- **PPSD's closed form** (speed-up = S / (α + (1 − α) S), S = S_eff) with S 7.2 predicts 2.5× at α 0.7 on nine cards;
  the oracle's 2.13× is 15 % lower, est. the KV-step stalls. At α 0.3–0.4 the formula gives 1.35–1.53×.
- **All right, deep:** at D = 8 the median interval is 3.46 ms, but 90 of 510 intervals take 8–12.5 ms in clusters of
  ~26, one per executor slot after each 256-cell KV step: 24 % of the run (13 % on six cards at D = 5). Without them the
  all-right ceiling is ~290 tok/s on nine cards.

### 1.4 Eight staggered conversations

From the Q120 run's per-conversation timings and the clock log (u2: packed uploads, temperature sampling on the last
card, prompts chat/code/edit/rewrite cycled, 512 tokens each, 64-token prompt chunks, two chunk slots):

| Phase | Time | What runs | Cards | Rate |
|---|---|---|---|---|
| Prefill | 0–46 s | 5,740 prompt tokens as 64-token chunks, one in flight per conversation, at most two at once; decode of the conversations already started | a wave of 1–3 busy cards moving down the pipeline (utilization sum 100–300 %) | ~125 prompt tok/s; ~5 generated tok/s in total (est.) |
| Steady | ~46–61 s | all eight generating, one token each in flight | every card 61–80 %, last card 98 % (sum 662 %), all at ~1,870 MHz | est. ~265 tok/s: the last 2,076-token conversation to start generated its 512 tokens in 15.3 s while the seven others finished in the same 1.3 s |

- Round-robin: every conversation has exactly one token in flight and resubmits as soon as it is sampled, so all eight
  progress at the same rate in steady state; their end times (60.0–61.3 s) fit ~33 tok/s each.
- The closed-pipeline model min(N / (L + h), 1 / c) with L 24.7 ms, h 1.6–1.8 ms (the streams loop's submit time) and
  c 3.4 ms gives 152 tok/s at N = 4 (est. measured ~152: the long-prompt conversation ran 38 tok/s with three others) and
  292 at N = 8 (est. measured ~265, 90 %).
- With synchronous input copies (u0) the same steady state ran ~18 tok/s per conversation (~140 in total, est.): the
  host's ~45–60 synchronous copies per token were the limit, as the host note predicted. Packed uploads (Q120) are
  worth ~1.9× there.

### 1.5 Prompt processing

- **pp512 at `-ub 512`** is one micro-batch: it passes the cards one after another, ~0.25 s per card (est.: 2.25 s over
  nine stages), and each card is busy ~11 %.
- **pp2048** is four micro-batches: fill and drain over nine stages allow at most 4 / (4 + 8) of the cards' time, est.
  ~30 % busy (613 tok/s against ~680 for perfect overlap of four).
- **pp8192** is sixteen: perfect overlap would take ~6 s (est. ~1,360 tok/s); it takes 11.3 s. Before each card's split
  the scheduler waits on the host for that card's previous use of the same input copy (`compute_splits`: an event
  synchronize per user input), and the host submits micro-batches card by card in order, so card 1 idles while the host
  waits for card 9; in-flight micro-batches stay near the number of copies (four), est. ~35 % busy. #20395's eight
  copies gave +25 % at 16k on the author's ROCm multi-GPU box; a device-side wait (an asynchronous copy behind an event
  wait on the card's stream, as Q120's packed uploads do for decode) removes the host from the chain.
- **In the streams mode** the 64-token chunks run ~1 s per chunk through nine cards without PR #31047 (pipespec2 lacks
  it), one per conversation in flight: ~125 prompt tok/s with two prompts in flight, 10–30 % busy.

### 1.6 The clock: idle cards are slow cards

| Job (nine cards) | tok/s | Mean clock of the used cards | Utilization sum |
|---|---|---|---|
| plain, chat (Q87) | 31.0 | 1,354 MHz | 100 % |
| pipe API, no guesses (`d0`) | 32.5 | 1,354 | 96 % |
| every guess wrong, D = 1 | 35.6 | 1,594 | 188 % |
| every guess wrong, D = 3 | 38.5 | 1,805 | 357 % |
| every guess wrong, D = 5 | 38.4 | 1,873 | 486 % |
| plain, rewrite (after a 750-token prompt) | 35.9 | 1,807 | 115 % |
| plain, chat, Q120 reference c0 | 34.0 | 1,354 | 94 % |
| plain, chat, Q120 reference c4 (after a 750-token prompt job) | 38.5 | 1,867 | 96 % |
| eight conversations, steady | ~265 total | 1,873 | 662 % |
- Every-guess-wrong runs can only be as fast as plain decoding in tokens; their 1.15–1.24× is the clock. The
  [power note](power_and_tuning.md) found the cause (the utilization-based policy holds cards ~11–17 % busy at base clock
  with no event bit) and Q141 found PowerMizer's "Prefer Maximum Performance" does not move it; these runs show the
  policy does respond to duty: ~2 tokens in flight per pass (D = 1) lifts the clock part-way, ~4 (D = 3) to boost.
- Locked clocks are Volta and newer and application clocks are unsupported on these cards, so the levers are duty
  (filling) and the core offset, which on Pascal also lifts the base level (+7 % at base per +100 MHz, power note).
- Consequence for every comparison: a filling scheme measured against plain decoding at base clock is credited with the
  clock. Compare against the every-guess-wrong oracle at the same depth, or log the clock.

## 2. Every known way to fill idle stages (question 2)

### 2.1 Pipeline-parallel and asynchronous speculation after PipeInfer and SPD

| Work | Mechanism | Published numbers | Multi-token verify? | Fit and gain here (est.) |
|---|---|---|---|---|
| SPEED (NeurIPS 2023 ENLSP workshop) | Each decoder stage processes a different token, started from an early-layer prediction; invalidation flushes wrong tokens and their KV | T5-Base on one A5000 (latency plots) | no | the executor's mechanism; prior art, nothing new |
| PPSD (arXiv 2509.19368, 2025) | Stage 1 = first E layers + exit head drafts continuously; later stages check one token each; closed-form speed-up | 2.01–3.81× (Vicuna-7B/13B, LLaMA-2-13B/70B, 8 × A100/H100) | no | the yardstick for Q20 (section 1.3) |
| PipeSpec (ACL Findings 2025) | k models on their own GPUs, asynchronous, each verifying the one below | up to 2.54× (Llama-3.1-70B, 4 × A100) | yes | needs a model hierarchy that does not fit 6 GB cards; 0 |
| AMUSD (arXiv 2410.17375) | Draft runs continuously on GPU 0, verifier on GPU 1 | 1.31–1.96× (2 × A100, Llama-3.1-8B + 1B) | yes | a small same-tokenizer drafter on a spare card would feed one-token guesses; no such model exists for Flint |
| PEARL (ICLR 2025) | Pre-verify the first draft token as a one-token step while drafting continues; adaptive draft length | up to 4.43× (CodeLlama 7B & 70B) | partly | pre-verify is what the executor does; adaptive depth by confidence +0–10 % |
| DSI (ICLR 2025) | Pool of target instances verifies staggered prefixes; provably never slower | 1.29–1.92× (simulated) | – | the "never slower" rule: guesses pre-emptible, the real token first; 0–5 % |
| Speculative Speculative Decoding / Saguaro (ICLR 2026) | While the target verifies, the drafter prepares the next speculation for each likely outcome | ~30 % over the strongest SD baselines, up to ~5× over AR (Llama-3.1-70B, 4 × H100) | yes | one-token translation: pre-launch a chain for the likely correction; +5–15 % (idea 7) |
| SpecBranch (ICLR 2026) | Parallel speculative branches hedge likely rejections; confidence-adaptive length | 1.8–4.5× over AR; ~50 % fewer rollback tokens | yes | spawn a second chain only when the top guess is uncertain; +5–15 % (idea 7) |
| MineDraft (ICML 2026), DPara (arXiv 2609.27396) | Draft one batch while verifying another; diffusion drafter precomputes drafts per acceptance boundary | +75 % throughput; 3.21–3.52× | yes | no |
| Minions, Ouroboros (2024) | Majority vote of several drafters; phrase-level drafts reusing rejected tails | 2.8–3.9× (Ouroboros) | yes | voting among copy, table and staircase guessers; +0–10 % |
| Faster Cascades (ICLR 2025) | Lenient deferral rule instead of strict matching | cost/quality curves | – | lossy; rejected by the bar ([unconventional ideas](unconventional_ideas.md)) |
| vLLM PR #50514, SGLang PR #30775 (2026-09) | Speculation under pipeline parallelism: the drafter on the last stage; SGLang drafts round r+1 at the end of round r ("tail-drafting") and relays accept lengths, bonus token and tree to stage 0 | vLLM: 2.34× at concurrency 1 over TCP-linked TP2 × PP2; SGLang on PCIe without NVLink: TP4 × PP2 + MTP 154.5 / 218 tok/s at batch 8 / 32 against 108.7 / 120.1 for TP8 + MTP | yes | batched verify; the last-stage drafter placement matches the staircase head's spare-card option |

### 2.2 Drafters on idle stages or devices

- **Mirror-SD** (arXiv 2510.13161): at a mid-depth exit the target's head gives top-κ (κ = 8) candidates, a drafter on
  other hardware rolls each out while the target's later layers compute, and the matching branch is reused; 2.8–5.8× on
  SpecBench (Qwen3-14B/32B). Closest published relative of the staircase head plus top-κ hedging.
- **Edge-cloud early exits** (arXiv 2505.21594): early exits inside the target emit provisional tokens so the drafter
  starts early (+11 % from that alone): the staircase head's principle, confirmed.
- **SpecOffload** (arXiv 2505.10259): the drafter runs in the GPU's idle gaps of an offloading engine;
  throughput-oriented.
- **CPU drafters** (DuoDecoding, Dovetail, Ghidorah): rejected on a 2-core host.
- Covered and still the plan: the staircase head on card 1 or a spare card ([pipeline drafter](pipeline_drafter.md)).

### 2.3 Early exit and self-speculation from the first layers

- **PPSD, SPEED and EE-LLM** (ICML 2024, "pipeline-based early-exit inference": an early-exiting token returns to stage 1
  while later stages fill its KV; 2× or more on HELM tasks, 4 × A100) draft from the card holding the first layers while
  later layers still compute. Only EE-LLM is lossy.
- **LayerSkip, Draft & Verify, SWIFT, CLaSp, KnapSpec, DEL, S3D** skip layers spread over every card, so each draft still
  crosses all hand-offs; rejected (pipeline drafter note). **Kangaroo**'s confidence stop and **HiSpec**'s intermediate
  verifiers (an exit head on a middle card cancels a doomed chain before it occupies later cards) carry over as rules.
- **KnapSpec**'s cost model (maximize tokens per unit time with measured per-layer costs) is the way to choose which
  card's hidden state the staircase head reads.

### 2.4 Jacobi, parallel and lookahead decoding

CLLMs (ICML 2024, 2.4–3.4×), Jacobi Forcing (ICML 2026, ~3.8× wall clock) and lookahead decoding are multi-token forward
passes by construction; no variant was found that works with one-token steps or fills idle stages. Their n-gram pools
are already in the guessers. Rejected.

### 2.5 Tree speculation for one-token steps

SpecPipe and FlowSpec (covered) remain the only pipeline-adapted trees, and both need cheap multi-token steps. Sequoia's
dynamic program over the measured verify cost c(k) confirms that batched trees cannot pay at c(3) = 2.2 and c(4) = 3.6.
On a one-token pipeline a "tree" is top-κ chains as separate sequences: idea 7. Not evaluated anywhere on a layer split.

### 2.6 Several samples, races and best-of-N

- Q93's first-finish race is covered. New, usable on idle slots: **ESC** (stop sampling once a window of samples agrees;
  −34 to −84 % samples), **ST-BoN** (pick the best sample early by internal consistency; full best-of-N quality at
  −70–80 % cost), **DeepConf** (token-confidence filtering; up to −84.7 % tokens), **DEER** (trial answers at "Wait"
  points; reasoning −19 to −80 % at +0.3–5 points), **Dynasor/Certaindex** (probe answers; up to 50 % compute saved).
  Trial answers and probes are forked sequences that can run on idle slots without delaying the main one. All change the
  output and need task checks.
- **Speculative decoding for multi-sample inference** (EMNLP 2025 Findings): drafts from consensus n-grams across
  parallel samples; in a race each sample's text guesses for the others (+5–15 % acceptance during races, est.).
- **Hogwild! Inference** (NeurIPS 2025: QwQ-32B 20.1 → 68.9 tok/s with 4 workers sharing one KV cache on an L40S),
  **Group Think**, **APR**, **Multiverse**, **ParaThinker**: parallel workers or threads; most need fine-tuning and the
  rest depend on whether Flint can collaborate. Not now.

### 2.7 Micro-batching several conversations with staggered phases

Covered in principle ([multi-conversation serving](multi_conversation_serving.md)). New since that note:
- **vLLM Model Runner V2** is now the default runner and supports pipeline parallelism with asynchronous scheduling: the
  last rank broadcasts sampled tokens on a side stream with its own communicator; a request decodes at most once every
  pp_size steps, so pp_size concurrent requests fill the pipeline.
- **gLLM** (arXiv 2504.14775): per-micro-batch decode and prefill token budgets (#D = running decode tokens / depth;
  prefill throttled by free KV); +11 % to +398 % over vLLM and SGLang on PCIe-only boxes with P2P disabled; vLLM's
  pipeline spent ~17 % of its time preparing inputs on the CPU.
- **SiPipe** (arXiv 2506.22033): three pipeline-bubble classes (load imbalance, intra-stage, inter-stage); 2.1×.
- **llama.cpp-multigpu's decode groups** (table above): the design to borrow for llama-server.
- At N ≥ S_eff (7 on nine cards) groups of two on Q78's kernels remain the way past the one-token ceiling (covered).

### 2.8 Prompt chunks of other requests in idle stages

| Work | What it does | Numbers | Fit here |
|---|---|---|---|
| Medha (arXiv 2409.17264) | Stream pipeline parallelism (chunks of one prompt in flight across stages) and adaptive chunk size (largest chunk that fits a target iteration time) | 5.7× throughput; 30× / 174× lower median / p99 latency; 32-token chunks only 11 % costlier than 2,048 on dense Llama-3 | yes for chunks in flight; small chunks cost far more on a 256-expert MoE (a 64-token chunk touches ~78 % of experts per layer) |
| Jupiter (INFOCOM 2025) | Intra-sequence pipelining of prefill across edge devices | prefill 1.4–7.4× faster (Jetsons, 100 Mbps–1 Gbps) | yes (idea 1) |
| SGLang chunked pipeline parallelism (blog 2026-01-15) | Chunks of one request on different stages at once; dynamic chunk size from a quadratic latency model | PP4 × TP8 prefill 3.31× TP8 (H20) | the dynamic sizing matters from ~8k |
| TD-Pipe (ICPP 2025) | Prefill and decode in separate time windows for the whole pipeline; switch rule from spatial and temporal intensity | up to 1.91× over TP, 2.73× over PP (PCIe-only L20/A100) | a wave policy for large prompt backlogs |
| PipeSwift (arXiv 2609.16491) | Prefill waves of K micro-batches, K* ≈ √((P − 1) W / γ); an interval of decode steps between waves | 1.45–2.33× job completion on 64 × H800 | sizing rule: est. `-ub` ~170 for 2k-token prompts on nine cards |
| THETA (ICML 2026) | Mixed prefill + decode batches pay an interference penalty that grows as memory bandwidth falls | exclusive batching up to +41.9 % on bandwidth-constrained GPUs | a 179 GB/s card argues for waves or pipelined chunks, not piggybacking |
| DéjàVu (ICML 2024) | Bubble model of colocated pipelines; disaggregates prompt and token pipelines | up to 2× over FasterTransformer | its model says prompt bubbles dominate here (prompt/token time ratio ~100 ≫ ~2.3); disaggregation itself does not fit |
| VPP (arXiv 2608.26523) | V-shaped virtual stages overlap a chunk's heavy middle with light neighbours | +6.7–13.1 %, bubble 6.4 → 0.1 % at 512k | long prompts only |
| ExLlamaV3 | `prefill_cooldown = 4`: one prefill chunk every fifth iteration while jobs generate; shared chunk half size | – | a simple decode-priority rule |
| vLLM | `long_prefill_token_threshold` (adaptive fair share); decodes scheduled first | – | same family |

### 2.9 Expert parallelism instead of a layer split

Covered by [multi-GPU physics](multigpu_pascal_performance.md) Q4 and [distributed inference](distributed_inference.md)
§3; updated arithmetic (est.): with 256 experts over nine cards and six per token, a token reaches ~4.6 cards per layer
(4.05 remote); dispatch and combine move 1.6 MB (FP16) to 3.2 MB (F32) per token over 49 layers, against the layer
split's 8 hops × 8 KB = 64 KB; staged through the host the best case is ~2 MB per token, ~360 tok/s of bandwidth ceiling
on ~715 MB/s, before latency; and it needs 98 host-mediated exchange phases per token (2 per layer), est. 5–15 ms of
synchronization on the Celeron. ThunderEP (arXiv 2609.40093) is the only expert-parallel system built for CPU-staged
consumer PCIe (dispatch 2.00×, combine 1.53× over NCCL, 1.66× end to end on RTX 4090/5090 boxes) and its links are two
to three orders of magnitude faster. llama.cpp's `-sm tensor` slices every expert across GPUs (two AllReduces per layer),
which is not expert parallelism. Rejected.

### 2.10 Hybrid layer + expert splits

Attention on some cards and experts on others (MegaScale-Infer's ping-pong pipeline needs m ≥ 2 (1 + T_c / T_f), three
to four micro-batches per layer; Step-3's attention-FFN disaggregation) moves the same per-token volume as expert
parallelism and puts the host in every layer. Affinity placement (ExFlow, MoETuner, Occult) cuts expert traffic by up to
~20–67 %, still 8–40× the layer split. Rejected.

### 2.11 KV cache or attention on idle cards

Lamina's attention offload needs ~450–500 KB and 98 round trips per token for this model (est.); Infinite-LLM, Adrenaline
and Helix parallelism assume fast fabrics; FastDecode, NEO and APEX put attention on a strong CPU. The one exact form that
fits is KV-sharded decode attention at long context (Q95, covered: nothing at ≤ 32k, ~2× at 262k). Tree attention and
Medha's KV parallelism are the same idea.

### 2.12 Decode-time compute for free

- **Hidden-state probes** (arXiv 2606.10487): a linear probe on one mid-layer hidden state recovers most of a strong
  guard's streaming moderation decisions at < 1 ms per token; the same trick fits any per-token classifier (confidence,
  topic, language) and the hidden state is already on the card holding that layer.
- **Stream Qwen3Guard** (0.6B, token-level monitoring head) fits a card's free ~0.5 GB and runs one step per emitted
  token off the critical path. **Streaming content monitoring** (NeurIPS 2025) decides after 18 % of a reply on average.
- Reward-model pruning of samples (speculative rejection, RSD) needs a model larger than a card. Embeddings and reranking
  need a second model and a use case the rig's owners have not asked for.
- None of these speeds anything up; they are features that idle cards make nearly free.

### 2.13 Engines (other than llama.cpp)

| Engine | Split and decode overlap | Speculation with pipeline parallelism | Prompt chunks | Compute 6.1 |
|---|---|---|---|---|
| vLLM | TP, PP, DP, EP, context parallel; MRV2 default with PP + async scheduling, side-stream token broadcast | MRV2: MTP (#46994), EAGLE-3/DFlash/DSpark (#50514), drafter on the last rank | chunked prefill default, `max_num_batched_tokens` 2,048 on < 70 GB GPUs | no (≥ 7.5) |
| SGLang | TP, PP (`pp_size + async depth` micro-batches), EP backends | `SGLANG_ENABLE_PP_SPEC=1` (#30775, experimental): tail-drafting on the last stage | `chunked_prefill_size` by memory; PP-only dynamic chunking | no (sm80 kernels) |
| TensorRT-LLM | TP, PP ring, EP, Helix | speculation × PP marked "No" | chunked prefill × PP "Untested" | no |
| ExLlamaV3 / TabbyAPI | sequential layer split; TP with EP for MoE through a host shared-memory AllReduce (no P2P) | draft models, n-gram suffix drafting, confidence-calibrated draft length; lockstep | `prefill_cooldown` | no (sm75+) |
| MLC-LLM | disco TP and PP stages, no micro-batching | draft/EAGLE/Medusa; PP + spec unverified | split-fuse | Vulkan only, unverified |
| distributed-llama | TP only, all-gather over TCP per layer; nodes ≤ KV heads (4 for Flint) | none | 32-token prompt batches | Vulkan FP16, unverified |
| exo | MLX; pipeline mode all-gathers the last hidden state and every rank samples in lockstep (#2229 open: relay one int) | none | staggered pipelined prefill | no (MLX CUDA ≥ 7.5) |
| Petals | client-driven chains | primitives (2024) | – | – |
| PowerInfer | one GPU + CPU, ReLU-sparse models | – | – | unverified |
| ktransformers | CPU+GPU MoE; "expert deferral" (1–4 experts applied a layer late to overlap CPU and GPU) | – | – | no |

Ideas that transfer: the drafter on the last stage with relayed accept lengths; a side-stream token relay that never
blocks the host; decode-first prefill with a cooldown; dynamic chunk sizing; confidence-calibrated depth.

## 3. llama.cpp specifically (question 3)

- **Decode overlap across sequences:** none upstream. #6017 overlaps one call's prompt micro-batches; decode reuses copy
  0 (#13814); graph reuse under pipeline parallelism synchronizes the whole scheduler before `set_inputs` (#20927); the
  server merges every slot's token into one step, samples (a full synchronize per logits read) and only then submits
  the next step. An untested reading of the code: with graph reuse disabled and `n_ubatch` below the
  number of decode tokens, one call's per-sequence micro-batches could rotate copies, but `n_ubatch` also caps prompt
  micro-batches. The fork's executor (Q87/Q104) and llama.cpp-multigpu's decode groups are the only implementations.
- **Speculation on a layer split:** draft placement (`-devd`) and parallel drafting across slots exist, always in
  lockstep; draft-MTP hurts layer-split prompts (#27428), and the fix that overlapped the MTP catch-up (#29523) was
  closed by a bot. PipeInfer was never upstreamed (MPI removed in #7395).
- **Chunked prefill and slot fairness:** the problem is reported (#6607, #29175, Discussion #26022) and the proof of
  concept (#10718) has stalled since 2024. #29175's 1,024-token cap per step while slots decode is the cheapest fix for
  llama-server `-np` today.
- **`GGML_SCHED_MAX_COPIES`:** default 4 upstream (1 in ik_llama.cpp); 8 copies +25 % at 16k on a ROCm multi-GPU box
  (#20395, closed); copies cost 0.5–1.5 GB of VRAM (#24355); the host-side event synchronize per input copy is what
  makes the count matter (section 1.5), and the asynchronous copy work (#25319, #30685, #29963, #27311) is what would
  make it matter less.
- **RPC:** events and async APIs merged (#18626) enable pipeline parallelism over RPC between hosts; one rpc-server
  serving several GPUs is not pipelined; on two cores the TCP path costs more than it gains.
- **CUDA streams:** one stream per device since #1898; multi-stream graph evaluation was slower (#4719); stream
  concurrency inside one GPU exists for Q/K/V (#16991). Nothing upstream gives each in-flight sequence its own stream on
  a card, which is what would stop one item's cross-device wait from blocking the others (section 4.3).
- **ik_llama.cpp:** its multi-GPU work is graph-parallel tensor parallelism (#1022, #1067, #1080, #1092) with AllReduce
  per layer, worse than layer mode at short context for small-active MoEs; #1069 (no wait before copying the next
  split's inputs, +25 % prompts) is the one idea that transfers. No decode overlap across sequences.

## 4. Why eight staggered conversations reached 66 tok/s (question 4)

### 4.1 They did not stall in steady state

Section 1.4: once the four long prompts were in, the eight conversations ran at est. ~265 tok/s with the last card at
98 % utilization, 90 % of the 292 tok/s ceiling, and four ran at est. ~152, on the model's line. The causes listed in the
question each check out as not binding there:

| Suspected cause | Status in the streams mode | Evidence |
|---|---|---|
| Per-backend synchronizes before inputs (#20927) | absent: slots use `caller_sync`, so the scheduler skips host waits | steady state at the model's rate |
| Event waits between cards | asynchronous, device-side (`cudaStreamWaitEvent`) | same |
| One stream per device | in-order work per card; harmless while every item is a one-token step, harmful behind a chunk (4.2) and a fed copy (4.3) | prefill phase; fed runs |
| Pinned-memory copies on x1 | the packed upload is one asynchronous copy per card per token; the synchronous path (u0) was host-bound at ~140 tok/s | u0 against u2 |
| Unified KV layout | one stream per conversation (`kv_unified = false`), so attention reads only its own cells | streams mode design |
| `n_seq` limits | 8 sequences, one stream each; no limit reached | – |
| Rebuilds at each 256-cell KV step | present, est. 5–10 % of steady state (Q153) | traces of the single-sequence jobs |
| The last card | the bottleneck at 3.4 ms per token | utilization 98 % |

### 4.2 Prefill chunks block every card they pass

- A 64-token chunk takes ~1 s through nine cards (the 2,076-token prompt needed 33 chunks in 32–35 s with one chunk in
  flight), ~0.1 s per card (est.), about 30 decode cycles. Each card runs its stream in order, so every conversation's
  token queued behind a chunk on that card waits for it: decode fell to ~5 tok/s in total for 46 s.
- One chunk in flight per conversation (and two chunk slots) serializes each prompt: prompt intake ~125 tok/s, against
  613 tok/s for llama-bench's pp2048.
- pipespec2 lacks PR #31047, so each 64-token chunk tiles every expert for 64 tokens while each sees ~1.5 (est. 1.4–2×
  slower than with the PR, [prefill note](prefill_pascal.md)).
- Fix (idea 1): the combined build; several chunks of one prompt in flight, which is exact because each card applies
  them in order and a chunk's KV is written before the next chunk reads it on that card; 32–128-token chunks; and a rule
  that admits at most one chunk per card per round of decode tokens (ExLlamaV3's cooldown, llama.cpp-multigpu's
  "one ubatch and one slot while other slots generate") or, for large backlogs, a wave with decode paused (TD-Pipe,
  PipeSwift). Est. with 64-token chunks pipelined one per round next to eight decoding conversations: a round of ~22 ms
  of decode cycles plus one ~55–110 ms chunk stage, so prompts enter at ~480–830 tok/s and each conversation still
  decodes at 8–13 tok/s; the Q120 job mix's 5,740 prompt tokens would take ~7–12 s instead of 46 (est.).

### 4.3 Device-fed tokens wait for the whole pipeline

- The fed path (Q121) queues the 4-byte copy of slot s's sampled id on the last card's stream at the moment slot s + 1 is
  submitted ([host decode loop](host_decode_loop.md) §6.2). By then the last card's stream already holds the work of every
  item submitted after slot s, so the copy, and card 1's work behind its event, waits until all of them have left the
  last card: one pipeline drain per fed token.
- That predicts a total rate independent of N, set by the drain: measured 22.1 tok/s at N = 4 and 22.3 at N = 8 (waits
  43 ms per token), while a lone conversation, with nothing behind it, gains (d0-fed 33.9 against d0 32.4).
- Fix (idea 5): record the event right after slot s's sampler (the slot's output event already exists) and issue the
  copy on card 1's stream after `cudaStreamWaitEvent` on it, or on a separate copy stream; card 1 then waits only for
  slot s's own sample.

### 4.4 Known causes and fixes elsewhere

- llama.cpp: graph reuse synchronize (#20927), copy 0 for decode (#13814), full synchronize per logits read (the
  multigpu fork's patches 25–26 make samplers wait only for their own batch), prefill chunks blocking slots (#29175,
  #10718), unified KV costing 42–54 % (#28495), the CPU embedding adding a split (#22926), launch queue overflow under
  pipeline parallelism (#19042), CPU blocking at split sync points (#19922).
- vLLM's pipeline spent ~17 % of its time preparing inputs on the CPU (gLLM); SiPipe moves sampling and bookkeeping to
  spare cores (the opposite of what a 2-core host can do; keep sampling on the cards).
- The fork's executor already avoids the first three; the remaining fixes are ideas 1, 3 and 5.

## 5. The plan (question 5)

### 5.1 Ranked

The ranked table above, in order of value per day: idea 1 (prefill as pipelined, bounded chunks; the largest measured
loss), 2 (count and use the clock; free), 3 (KV-step stalls; +25–31 % in every all-busy mode), 4 (prompt micro-batch
waits; agents' prefill), 5 (fed-token copy; a day), then the drafter work (6, 7) that is already running, the light last
card (8, a flag), hand-off cost (9, after Q81), race control (10, quality-gated) and the server integration (11).

### 5.2 First experiments

1. **Steady state without prefill (~25 min, no code).** Nine cards, the `pascal-p106-v2-pipespec2` build with the window
   fix, K2h-r16:

   ```
   ssh rig "setsid nohup env BIN=$B MODEL=<K2h-r16> TAG=idle-steady GPUS=0,1,2,3,4,5,6,7,8 \
       ENVS=LLAMA_NO_MOE_DOWN_RESCALE=1 CONFIGS=u2 NS='1 4 8' \
       PROMPTS=/tmp/pipespec/prompts/chat.txt,/tmp/pipespec/prompts/code.txt \
       bash /tmp/pipespec/rig_hostloop.sh > /tmp/pipespec/idle-steady.out 2>&1 < /dev/null &"
   # then TAG=idle-steady-light with TS=6,6,6,6,6,6,6,5,2
   ```

   Read `summary.tsv` (n4, n8 totals) and `gpu.csv` (clock, utilization sum). Expected (est.): n8 250–290 tok/s on the
   default split, n4 145–160, the light split +10–18 % at n8. Kill for section 4.1's reading: n8 < 180 tok/s.
2. **The clock baseline (~15 min).** `rig_run.sh` with `PROMPTS=chat` and the oracle at 0 % and 100 % wrong for D = 1–3,
   plus `cont` at D = 3, with the clock log and wall power: the every-guess-wrong rows are the matched baseline for every
   later filling result.
3. **Prompt micro-batches (~30 min, after an 8-copy build on the quant machine).** `llama-bench -p 512,2048,8192 -n 0
   -ub 128,256,512` on the combined build with 4 and 8 copies.
4. **Offline, no rig:** stall counts per job from the existing traces (Q153's size in every mode), and `pipeline_sim.py`
   with Q20's top-2 probabilities for idea 7.

### 5.3 What to take from unmerged work now

- **llama.cpp-multigpu (MIT):** patch 15 (a hop's source stream waits for the destination's previous split), 12/16/17
  (rotating pinned input staging only for pipelined decode micro-batches), 19–28 and 36 (decode groups, per-batch output
  slots, samplers that wait for their own batch, prefill chunk policy while slots decode), 30 (no pipeline drain on
  re-plan). Read before the streams mode goes into llama-server.
- **#25319** (asynchronous scheduler copies) and **#20395** (runtime copies) for idea 4; **ik_llama.cpp #1069**'s
  no-wait input copy as the minimal version.
- **#29175**'s cap on prompt tokens per step while slots decode, for llama-server `-np` until the streams mode lands.
- **#29523** (lagged MTP catch-up) if the Qwen3.6 track uses MTP on a layer split.
- Already planned elsewhere: one synchronize per served token (#30196, #29796; slim note B3), the input layer on card 1
  (#22926; slim note B2).

## What this changes in the plan

- **Q104's measured 66 tok/s is a prefill number, not a serving ceiling.** The steady state is est. ~265 tok/s on nine
  cards; the registry row and the v2 synthesis's several-conversations table should carry both numbers once experiment 1
  confirms it, and the streams benchmark should report steady-state and prefill phases separately.
- **Every pipelined speed-up needs a clock-matched baseline.** Q87's 1.3–1.55× splits into ~1.24× clock and 1.10–1.3×
  guesses; the Q20 drafter's payoff is est. 1.6–2.5× over the boost baseline, ~2–3× over plain as it runs today.
- **The executor's next items are scheduling, not kernels:** prompt chunks in flight with decode priority (idea 1), the
  KV-step stalls (Q153, now worth +25–31 % in all-busy modes), and the fed-token copy (idea 5).
- **The light last card is worth less than the serving note estimated** (est. +13–18 % at ≥ 7 in flight, not +40 %),
  because the measured cycle at boost is 3.4 ms, not 5 ms.
- Suggested registry rows (not added here): the steady-state streams run (experiment 1); the clock-matched baseline in
  every pipespec run; the fed-copy fix; prompt chunks in flight with a decode-priority rule; 8 copies or device-side
  input waits for prompt micro-batches.

## Already covered

| Direction | Where it is |
|---|---|
| Pipelined speculation, executor, dropping vs cancelling, races, streams mode | [Pipelined speculation](pipelined_speculation.md); Q87, Q93, Q104 |
| PipeInfer, SPD, SpecPipe, FlowSpec, SpecPipe-DB | [Pipelined speculation](pipelined_speculation.md), [multi-conversation serving](multi_conversation_serving.md) |
| EAGLE-3, Medusa, MTP, layer-skip and expert-sparse self-drafting, small same-tokenizer drafters (none exist) | [Pipeline drafter](pipeline_drafter.md) |
| Orca, DeepSpeed Inference, vLLM/SGLang pipeline batch queues, Megatron, Sarathi-Serve, Petals, exo #2312 | [Multi-conversation serving](multi_conversation_serving.md) §3.1 |
| Staggered conversations' throughput model, KV per conversation, groups of two | [Multi-conversation serving](multi_conversation_serving.md) |
| Host per-token budget, packed uploads, fed tokens, conditional-node loops, mailboxes | [Host decode loop](host_decode_loop.md); Q118–Q122 |
| Host slimming, input layer on card 1, per-slot rebuilds at KV steps | [Slim Pascal fork](pascal_slim_fork.md); Q150–Q153 |
| MMQ tiles (PR #31047), graph-reuse wait (Q109), on-card masks (Q110, Q116), TurboPrefill, prompt caching | [Prefill on Pascal](prefill_pascal.md); Q108–Q111 |
| Base clock in low-duty decode, PowerMizer, core offset, power in all-busy modes | [Power and tuning](power_and_tuning.md); Q141, Q143 |
| First-finish race, judged/typical acceptance, KV-sharded attention, split output head, hop rule | [Unconventional ideas](unconventional_ideas.md); Q93, Q95 |
| Tensor and expert parallelism over x1, RPC between machines | [Multi-GPU physics](multigpu_pascal_performance.md) Q3–Q4; [distributed inference](distributed_inference.md) |
| Batched verify costs, per-shape graphs | [Speculation plumbing](speculation_plumbing.md); Q19, Q52 |
| Persistent MoE kernel | Q88 |
| Pipelined decoding in llama-server for a lone slot | the fork's `pascal-p106-v2-pipeserve` branch (in progress) |

## Not verified

- The steady-state rates of section 1.4 (~265 tok/s at N = 8, ~152 at N = 4) are inferred from per-conversation end
  times, round-robin fairness and the clock log, not timed per token; experiment 1 measures them.
- The decomposition of a pass into layers (~0.38 ms), head (~1.5 ms) and hand-offs (~0.55–0.65 ms) rests on two
  equations from the six- and nine-card runs; the hand-off has not been measured (Q81).
- The fed-token mechanism (section 4.3) is inferred from the executor's described copy placement and the N-independent
  rate, not from a trace.
- The host-wait cap on prompt micro-batches (section 1.5) is read from the scheduler's code path and #20395's numbers on
  other hardware; the rig's pp8192 busy share is an estimate.
- The duty at which the clock policy lifts the cards to boost (between ~1 and ~3 extra tokens in flight on nine cards) is
  bracketed by three runs.
- Chunk policies' effects (section 4.2) use an est. chunk cost per card of 55–110 ms for 64 tokens.
- Clock-matched gains for edit and rewrite prompts lack an every-guess-wrong run on the same prompt.
- Engine and paper numbers are as published; none was reproduced; vLLM, SGLang and TensorRT-LLM do not run on compute 6.1.

## Sources

Pipelined and asynchronous speculation (P)
- C. Hooper et al., "SPEED: Speculative Pipelined Execution for Efficient Decoding", [arXiv 2310.12072](https://arxiv.org/abs/2310.12072)
- R. Li et al., "Pipeline Parallelism is All You Need for Optimized Early-Exit Based Self-Speculative Decoding" (PPSD), [arXiv 2509.19368](https://arxiv.org/abs/2509.19368)
- B. McDanel et al., "PipeSpec: Breaking Stage Dependencies in Hierarchical LLM Decoding", ACL Findings 2025, [arXiv 2505.01572](https://arxiv.org/abs/2505.01572)
- B. McDanel, "AMUSD: Asynchronous Multi-Device Speculative Decoding", [arXiv 2410.17375](https://arxiv.org/abs/2410.17375)
- T. Liu et al., "PEARL: Parallel Speculative Decoding with Adaptive Draft Length", ICLR 2025, [arXiv 2408.11850](https://arxiv.org/abs/2408.11850)
- N. Timor et al., "Distributed Speculative Inference (DSI)", ICLR 2025, [arXiv 2405.14105](https://arxiv.org/abs/2405.14105)
- T. Kumar, T. Dao, A. May, "Speculative Speculative Decoding", ICLR 2026, [arXiv 2603.03251](https://arxiv.org/abs/2603.03251)
- Y. Shen et al., "SpecBranch", ICLR 2026, [arXiv 2506.01979](https://arxiv.org/abs/2506.01979)
- Z. Tang et al., "MineDraft", ICML 2026, [arXiv 2603.18016](https://arxiv.org/abs/2603.18016); F. Liu et al., "When Parallel Drafter Meets Parallel Speculative Decoding" (DPara), [arXiv 2609.27396](https://arxiv.org/abs/2609.27396)
- H. Narasimhan et al., "Faster Cascades via Speculative Decoding", ICLR 2025, [arXiv 2405.19261](https://arxiv.org/abs/2405.19261)
- N. Bhendawade et al., "Mirror Speculative Decoding", [arXiv 2510.13161](https://arxiv.org/abs/2510.13161)
- Y. Venkatesha et al., "Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits", [arXiv 2505.21594](https://arxiv.org/abs/2505.21594)
- Y. Chen et al., "EE-LLM", ICML 2024, [arXiv 2312.04916](https://arxiv.org/abs/2312.04916); M. Elhoushi et al., "LayerSkip", ACL 2024, [arXiv 2404.16710](https://arxiv.org/abs/2404.16710); F. Liu et al., "Kangaroo", NeurIPS 2024, [arXiv 2404.18911](https://arxiv.org/abs/2404.18911); S. Cha et al., "KnapSpec", ICML 2026, [arXiv 2602.20217](https://arxiv.org/abs/2602.20217); A. Kumar et al., "HiSpec", [arXiv 2510.01336](https://arxiv.org/abs/2510.01336)
- Z. Chen et al., "Sequoia", [arXiv 2402.12374](https://arxiv.org/abs/2402.12374); S. Kou et al., "CLLMs", ICML 2024, [arXiv 2403.00835](https://arxiv.org/abs/2403.00835); L. Hu et al., "Jacobi Forcing", [arXiv 2512.14681](https://arxiv.org/abs/2512.14681)
- A. Saxena et al., "Utility-Driven Speculative Decoding for Mixture-of-Experts", [arXiv 2506.20675](https://arxiv.org/abs/2506.20675)
- vLLM [PR #50514](https://github.com/vllm-project/vllm/pull/50514), [PR #46994](https://github.com/vllm-project/vllm/pull/46994); SGLang [PR #30775](https://github.com/sgl-project/sglang/pull/30775)

Races, samples and side work (P)
- Y. Li et al., "Escape Sky-high Cost: Early-stopping Self-Consistency", ICLR 2024, [arXiv 2401.10480](https://arxiv.org/abs/2401.10480)
- Y. Wang et al., "Self-Estimating the Best-of-N Sampling in Early Decoding" (ST-BoN), NeurIPS 2025, [arXiv 2503.01422](https://arxiv.org/abs/2503.01422)
- Y. Fu et al., "Deep Think with Confidence", [arXiv 2508.15260](https://arxiv.org/abs/2508.15260); C. Yang et al., "Dynamic Early Exit in Reasoning Models" (DEER), [arXiv 2504.15895](https://arxiv.org/abs/2504.15895); Y. Fu et al., "Efficiently Scaling LLM Reasoning with Certaindex", [arXiv 2412.20993](https://arxiv.org/abs/2412.20993)
- Y. Li et al., "Speculative Decoding for Multi-Sample Inference", EMNLP 2025 Findings, [arXiv 2503.05330](https://arxiv.org/abs/2503.05330)
- G. Rodionov et al., "Hogwild! Inference", NeurIPS 2025, [arXiv 2504.06261](https://arxiv.org/abs/2504.06261)
- H. Shu et al., "Stop Early, Spend Less: Hidden-State Probes …", [arXiv 2606.10487](https://arxiv.org/abs/2606.10487); Qwen, "Qwen3Guard Technical Report", [arXiv 2510.14276](https://arxiv.org/abs/2510.14276)

Pipelines, prefill and serving (P)
- A. Agrawal et al., "SARATHI", [arXiv 2308.16369](https://arxiv.org/abs/2308.16369); "No Request Left Behind … with Medha", [arXiv 2409.17264](https://arxiv.org/abs/2409.17264)
- S. Ye et al., "Jupiter", INFOCOM 2025, [arXiv 2504.08242](https://arxiv.org/abs/2504.08242)
- H. Zhang et al., "TD-Pipe", ICPP 2025, [arXiv 2506.10470](https://arxiv.org/abs/2506.10470); "EcoServe", [arXiv 2504.18154](https://arxiv.org/abs/2504.18154)
- S. Wang et al., "PipeSwift", [arXiv 2609.16491](https://arxiv.org/abs/2609.16491); W. Zhang et al., "THETA" (ICML 2026 poster), <https://icml.cc/virtual/2026/poster/62774>
- F. Strati et al., "DéjàVu", ICML 2024, [arXiv 2403.01876](https://arxiv.org/abs/2403.01876)
- Y. Guo et al., "gLLM", [arXiv 2504.14775](https://arxiv.org/abs/2504.14775); Y. He et al., "SiPipe", [arXiv 2506.22033](https://arxiv.org/abs/2506.22033)
- "VPP: Virtual Pipeline Parallelism for Efficient Chunked Prefill", [arXiv 2608.26523](https://arxiv.org/abs/2608.26523)
- D. Arfeen et al., "PipeFill", MLSys 2025, [arXiv 2410.07192](https://arxiv.org/abs/2410.07192)
- Z. Li et al., "Prima.cpp", [arXiv 2504.08791](https://arxiv.org/abs/2504.08791)
- LMSYS, chunked pipeline parallelism (2026-01-15), <https://lmsys.org/blog/2026-01-15-chunked-pipeline>; SGLang
  [pipeline parallelism docs](https://docs.sglang.io/advanced_features/pipeline_parallelism.html)

Expert parallelism and attention offload (P)
- J. Lee et al., "Efficient Expert-Parallel Communication on PCIe-Connected Consumer GPUs" (ThunderEP), [arXiv 2609.40093](https://arxiv.org/abs/2609.40093)
- R. Zhu et al., "MegaScale-Infer", [arXiv 2504.02263](https://arxiv.org/abs/2504.02263); J. Yao et al., "Exploiting Inter-Layer Expert Affinity …" (ExFlow), [arXiv 2401.08383](https://arxiv.org/abs/2401.08383)
- S. Chen et al., "Efficient Heterogeneous LLM Decoding with Model-Attention Disaggregation" (Lamina), [arXiv 2405.01814](https://arxiv.org/abs/2405.01814)

Engines (P)
- vLLM main `5f63ea46` (2026-10-24): `vllm/v1/worker/gpu/pp_utils.py`, `vllm/v1/core/sched/scheduler.py`, `docs/features/speculative_decoding/README.md`
- SGLang main `bf1a2529`: `scheduler_pp_mixin.py`, `dynamic_chunk_sizer.py`, `validation_hook.py`
- TensorRT-LLM `032eb456`: `docs/source/features/feature-combination-matrix.md`; ExLlamaV3 `da68d57d`: `model_tp_backend.py`, `generator.py`
- distributed-llama `4074c0f3`, Discussions #147, #202, #255; exo `bc412e22`, PR #2229; MLC-LLM PR #2825; ktransformers `kt-kernel/README.md`

llama.cpp, ik_llama.cpp and forks (P): every item in the table above, at `https://github.com/ggml-org/llama.cpp/pull/N`,
`/issues/N` and `/discussions/N` and `https://github.com/ikawrakow/ik_llama.cpp/pull/N`; the fork
[lukolszewski/llama.cpp-multigpu](https://github.com/lukolszewski/llama.cpp-multigpu) (`docs/multigpu/patches.md`,
`docs/multigpu/benchmarks.md`)

This project
- [Quant registry](../../../quants/experiments.md) (Q20, Q81, Q87, Q88, Q93, Q104, Q108–Q110, Q117, Q120, Q121, Q141,
  Q153); the Q87 and Q120 rig runs (per-token traces, per-second clock and utilization logs, host timers; raw logs outside
  the repository); [bench/pipespec](../../../bench/pipespec/README.md); the notes linked in the scope
