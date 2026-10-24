# Serving several conversations at once on the layer split (2026-10-24)

Scope: how one copy of Flint (40B-A2.5B MoE), split by layers over six or nine P106-090 cards, should serve N independent
conversations (an agent with parallel sub-agents, tool calls, a small team on one rig) so that the total rate grows
several-fold while each conversation stays close to single-stream speed. The note compares llama-server's batching
(`-np N`), staggered single-token pipelining through the fork's pipelined executor (Q87) and hybrids that stagger small
batches on the multi-token kernels (Q78); sizes the KV cache per conversation; reads the prior art and the fork's
scheduler, llama-context and server code for what has to change; and sets one policy for sharing the idle cards between
conversations, pipelined guesses (Q87) and race samples (Q93). It builds on [breakthrough candidates](breakthrough_candidates.md)
(candidate 1), [pipelined speculation](pipelined_speculation.md), [pipeline drafter](pipeline_drafter.md),
[unconventional ideas](unconventional_ideas.md), [speculation plumbing](speculation_plumbing.md), section 14 of the
[kernel integration note](pascal_mmvq_integration.md), the [long-context cliff](long_context_cliff.md) and the
[pipeline simulator](../../../bench/ngram_replay/README.md#pipelined-speculation-q87). Conventions are those of the
other notes:
- "(est.)" marks this note's own arithmetic or inference; the basis is given next to it.
- "P" and "S" in the source list mean primary (paper, code, measurements in a repository) and secondary.
- "Laptop", "quant machine" and "rig" are the three places work runs: code and correctness on a laptop GPU, quality and
  training on the quant machine (never speed), speed on the rig only.
- `file:line` refers to the fork's branch `pascal-p106-v2` at `032b130f4`; "pipespec" marks the branch
  `pascal-p106-v2-pipespec` at `a7204884d`, which adds the executor.
- N is the number of conversations generating at the same time; "total" is the sum of their generated tokens per second.

Starting point, measured ([registry](../../../quants/experiments.md)):
- Single stream, S4-L0 on six cards, fork build: 37.4 tok/s in llama-bench tg128, 26.8 ms per token (T); UD-Q4_K_XL
  37.3 (Q76). Real chat on nine cards at 262,144 tokens of context: 31.9 (S4-L0) and 28.0 (UD-Q4_K_XL) tok/s, ~31.3 ms
  per token (Q58). G3 reads 2.33 GB per token, 0.65 GB of it routed experts; 2.21 GB with the GPTQ Q5_0 head (G3h).
- Q87's simulator splits the six-card token into stages of 3.91 / 3.48 / 3.48 / 3.48 / 3.48 / 5.98 ms (9/8/8/8/8/8
  layers; output head and GPU sampling on the last card), 0.3 ms per hop and 1.5 ms from a sample to the next launch;
  nine cards: 3.05 (×4) / 2.54 (×4) / 5.04 ms (6/6/6/6/5/5/5/5/5 layers) or 3.05 (×7) / 2.54 / 3.52 ms
  (6/6/6/6/6/6/6/5/2). With every card busy, one token leaves per slowest stage plus a hop: 159, 187 and 262 tok/s
  (4.3×, 5.9× and 8.2× one stream). S_eff, the tokens the pipeline holds before anything waits, is 4.3, 5.9 and 8.2.
- Two conversations on the stock build (makers' Q4_1, nine cards, `-np 2 -bs`): 21.2–23.6 tok/s each against 24.3
  alone, 35–42 tok/s together ([Flint benchmark](../../benchmarks/2026-10-22-flint.md)). On the fork the single stream
  runs on the Pascal one-token kernels, while a 2-token step still runs llama.cpp's stock multi-token kernels.
- Multi-token steps on the fork (Q19, server, synthetic acceptance): 2.90 tokens in 57.7 ms and 4.05 tokens in 92.1 ms
  against 25.9 ms for one (2.23× and 3.56×); llama-bench pp8 212 ms. Q78's 2–4-column Pascal kernels pass every
  correctness test; their rig speed is unknown, and section 14 of the kernel note models a step's GPU time at
  1.30–1.44 (2 tokens), 1.55–1.80 (3) and 1.79–2.14 (4) times one token's.
- Independent tokens share few experts: 2, 4 and 8 random tokens touch 11.9, 23.2 and 44.2 of 256 experts per layer
  (Q51), so a batch of N conversations reads nearly N times the routed bytes; only the dense ~72 % is shared.
- The host launches one token's six cards' graphs within ~1.9 ms of the token's start (Q49): ~0.3 ms per card.

## Verdict

- **Stagger the conversations through the cards instead of batching them.** Each conversation keeps one token in flight
  and submits the next one the moment it is sampled; the cards then work on different conversations' tokens at once,
  every step stays a one-token step on the fork's kernels and CUDA graphs, and each conversation runs at nearly its
  single-stream speed until the pipeline is full. Est. on nine cards with a light last card (6/…/5/2): **64 tok/s total
  at N = 2, 118–128 at N = 4, 235–256 at N = 8, each conversation at 29–32 tok/s** (one stream: 31.9). Six cards:
  75, 137–149 and 145–159 (each 37, 34–37 and 18–20). Q87's executor already runs exactly this schedule for the k
  samples of a race; it needs a "streams" mode, not a new design.
- **llama-server's `-np N` scales badly on this rig.** It merges the slots' tokens into one ubatch that passes the cards
  one after another (five of six cards idle at any moment), reads each conversation's experts separately, and every
  step of two or more tokens leaves the Pascal one-token kernels. Est. on six cards today: 45–55 tok/s total at N = 2,
  ~43 at N = 4 and ~37 at N = 8 (each 23–28, ~11 and ~4.6): no gain beyond two conversations. With Q78's kernels:
  53–59, 74–87 and 94–109 (the last needs an 8-column kernel; Q78 stops at 4).
- **Above S_eff, stagger groups of two.** Two conversations batched on Q78's 2-column kernel read 1.27× one token's
  bytes for two tokens. Six cards, N = 8: est. 196–235 tok/s total at 25–29 each, against 145–159 at 18–20 for single
  tokens; nine cards, N = 16: 337–403 at 21–25. Groups of four cut each conversation to ~0.55× of single speed and win
  only on total.
- **Memory does not limit N at ≤ 32k on nine cards.** f16 KV is 28 KiB per token over the 14 full-attention layers plus
  ~53 MiB of sliding-window cells per conversation: 0.27, 0.93 and 3.55 GiB per conversation at 8k, 32k and 131k.
  Beside a ~33 GB file (G3h class) nine cards hold ~16–18 GiB of KV: ~60, ~17 and 4–5 conversations (about 3 at 131k
  on the light-last-card split); beside UD-Q4_K_XL (25.1 GB) ~22–24 GiB: ~85, ~24 and 6. Six cards with a G3h-class
  file hold 1–3 conversations at 8k (S4-L0 at 32k already fails there, Q54), so multi-conversation serving runs one
  copy on all nine cards. Two copies do not fit at the quality bar.
- **The limits move from the cards to the plumbing.** Five things decide whether the estimate holds:
  - **sampling on the GPU, per sequence and seeded:** host sampling copies 0.6 MB of logits per token over the last
    card's x1 link on its compute stream (~2.8 ms), which lowers the six-card ceiling from 159 to ~110 tok/s;
  - **one KV stream per conversation:** with llama.cpp's unified cache (the race's mode, and llama-server's default
    for `-np auto`) every one-token step reads the KV of every conversation;
  - **the attention mask:** it is copied synchronously to every card for every token, ~2 ms per token at 32k on six
    cards and ~12 ms at 131k on nine, which makes long-context serving host-bound unless the mask is built on the card;
  - **the host's submit rate:** at ~0.3 ms per card per token one Celeron core feeds ~240–400 tokens/s on nine cards
    at short context, just around the 262 tok/s ceiling;
  - **prompts:** a new conversation's prompt goes through `llama_decode`, which drains the pipeline and stalls every
    other conversation for the whole prefill; prompt chunks must become pipeline work items.
- **Prompt tokens cost about as much card time as generated ones** (~5–7 ms of pipeline time per prompt token against
  3.8–6.3 ms per generated token at saturation). For tool-heavy agents, with two or more new prompt tokens per generated
  token, prefill sets the total (est. 56–72 generated tok/s on nine cards at 2:1), so the prefill work (Q80, Q89, the
  on-card mask of breakthrough candidate 3) belongs to this goal as much as staggering does.
- **Speculation and races fill what the conversations leave.** Per free slot in the pipeline the expected useful work
  ranks: a waiting conversation's token (1.0) > a guess inside a copy run (0.83–0.98) > a trained head's first guess
  (0.6–0.72) > a second race sample during reasoning (~0.39) > a fresh-text table guess (~0.2) > a third race sample
  (~0.17). Tokens in flight stay at or below S_eff, so a wrong guess never delays another conversation (PipeInfer's
  rule: speculate only when the stage would otherwise idle). With N ≥ S_eff there is no speculation.
- **First experiments, rig, no code, ~3 h:** `llama-batched-bench -npl 1,2,3,4,6,8` on the v2 and the multi-token
  builds (the step cost of N tokens); `bench/pipespec/rig_run.sh`'s all-right oracle at depths 0–5 (six cards) and
  0–8 (nine cards) with GPU greedy sampling, whose rate at depth D is the staggered rate of D + 1 conversations, with
  clocks, power and host CPU logged; and a race of N samples with a stop text that never appears, which runs N real
  sequences with host sampling. Then a streams mode in `llama-pipespec` (3–5 days) and a multi-stream extension of
  `pipeline_sim.py`.

## Ranked approaches

Rates are generated tokens per second, per conversation / total (est.), at short context (≤ 8k) with GPU sampling; the
lower ends include a 10 % stage slowdown from the 75 W limit once the cards run at full duty. "Memory limit" is for
nine cards with f16 KV beside a G3h-class file (~33 GB).

| # | Approach | N = 2 | N = 4 | N = 8 | Memory limit | Effort | First experiment | Kill criterion |
|---|---|---|---|---|---|---|---|---|
| 1 | **Staggered single-token conversations** (Q87's executor in a streams mode: one KV stream and one pipe slot per conversation, seeded GPU sampling per sequence), nine cards, light last card (6/…/5/2) | 32 / 64 | 30–32 / 118–128 | 29–32 / 235–256 | ~17 conversations at 32k; ~3 at 131k on this split (4–5 on a balanced one); 2 at 262k | Streams mode 3–5 days; GPU sampling per sequence 2–4 days; on-card mask 1–5 days; prompt chunks as pipeline work 3–5 days; llama-server integration 1–2 weeks | All-right oracle at depths 0–8 with GPU greedy sampling (staggered rate of D + 1 conversations); race of N samples with an impossible stop | Cycle with every card busy above 0.3 T (below 3.3× one stream on six cards), or a submit thread that needs more than one cycle per token |
| 1b | Same on six cards (9/8/8/8/8/8) | 37 / 75 | 34–37 / 137–149 | 18–20 / 145–159 | UD-Q4_K_XL: ~9 at 32k; G3h: 1–3 at 8k | as row 1 | as row 1 | as row 1 |
| 2 | **Hybrid: groups of two conversations on Q78's 2-column kernel, staggered** (only above S_eff) | worse than row 1 | worse than row 1 | six cards 25–29 / 196–235; nine cards (N = 16) 21–25 / 337–403 | as row 1 | Multi-token pipe slots 2–3 days, after row 1 and Q78's rig A/B | The 2-token step of `llama-batched-bench` on the multi-token build | Rig 2-token step ≥ 1.6 T on the multi-token build (then groups never beat single tokens) |
| 3 | llama-server `-np N` with Q78's kernels | six cards 27–29 / 53–59; nine 23–25 / 46–50 | 18–22 / 74–87; nine 16–19 / 64–75 | 12–14 / 94–109 with an 8-column kernel (else ~4.6 / ~37); nine 10–12 / 82–95 | same KV per conversation | None beyond Q78's A/B; 8 columns 2–4 days | `llama-batched-bench -npl 1,2,3,4,6,8`, multi-token build | Not killed: it is the fallback without the executor |
| 4 | llama-server `-np N` today (stock multi-token kernels) | six cards 23–28 / 45–55; nine 19–24 / 39–47 (stock build measured 21–24 / 35–42) | ~11 / ~43; nine ~9 / ~37 | ~4.6 / ~37; nine ~4 / ~32 | same | None | `llama-batched-bench`, v2 build | Use only up to two conversations |
| 5 | Two model copies | – | – | – | A bar-quality file (25–33 GB) needs ≥ 5 cards per copy; nine cards hold one | – | – | Rejected |
| 6 | Overlapping `llama_decode` calls of different conversations without the executor | – | – | – | – | – | – | Rejected: graph reuse synchronizes the scheduler, every logits read synchronizes the whole context, and the rebuild path costs ~8–10 ms of host time per call (section 3.2) |

## 1. Throughput for N conversations (question 1)

### 1.1 What each approach does on the cards

- **(a) Batching, `-np N`.** `update_slots` collects one token per generating slot into one batch
  (`server-context.cpp:3700-3725`, `:3836`, `:587-615`), decodes it with one `llama_decode` per `n_batch` chunk
  (`:3514-3522`) and samples every slot afterwards (`:4664-4760`). The N tokens form one ubatch, which passes card 1,
  then card 2, and so on: one card works at a time, exactly as for one token. A step costs v(N) plain tokens and
  serves N tokens, so the total is N / (v(N) · T).
- **(b) Staggered single tokens.** Each conversation has one token in flight. Card j runs whichever token reaches it
  next, so with N conversations up to N cards work at once. For a closed pipeline of deterministic stages the rate is
  min(N / T, 1 / c): every conversation runs at its single-stream speed 1/T until N reaches S_eff = T / c, after which
  the cards are full and the total stays at 1/c, where c is the slowest stage plus a hop.
- **(c) Hybrids.** G groups of B conversations; each group is one B-token ubatch on Q78's kernels and the groups are
  staggered. Every stage takes r(B) times its one-token time, where r(B) is the multi-token step ratio, so the total is
  B · min(G / T_B, 1 / c_B) with T_B ≈ r(B) · T and c_B ≈ r(B) · c (plus ~0.15 ms of host time per extra token).
  (a) is the case G = 1, (b) the case B = 1.

Bytes read per step (G3's table: 1.68 GB dense, 0.65 GB routed for 6 experts per layer; independent tokens touch
256 · (1 − (250/256)^N) experts per layer):

| Tokens per step | Distinct experts per layer | GB per step | Per token | Ratio to one token |
|---|---|---|---|---|
| 1 | 6 | 2.33 | 2.33 | 1.00 |
| 2 | 11.9 | 2.96 | 1.48 | 1.27 |
| 4 | 23.2 | 4.19 | 1.05 | 1.80 |
| 8 | 44.2 | 6.46 | 0.81 | 2.78 |

Staggering (b) reads 2.33 GB per token whatever N is, but each card reads only its own share and the cards stream in
parallel: at 256 tok/s on nine cards that is ~66 GB/s per card on average, well below the ~150 GB/s the kernels reach,
so (b)'s ceiling is set by the per-token fixed costs of the slowest stage, not by bandwidth. Batching saves bytes per
token, which is why (c) can pass (b)'s ceiling once the cards are full.

### 1.2 Inputs

- Stage times, hop and gap: Q87's simulator setups above. The hop (0.1–0.3 ms, Q81 unmeasured) moves the result by
  ≤ 3 %: T is measured, so a shorter hop leaves more time in the stages.
- v(N) today: v(2) = 1.35–1.65 (1.65 interpolated from Q19; 1.35 from the stock build's two slots at 1.03–1.14 of its
  own single step, applied to the fork's single step, est.), v(4) = 3.50 (Q19), v(8) = 7.9–8.1 (llama-bench pp8 and
  Q19's last slope).
- r(B) with Q78's kernels: 1.30–1.44 (B = 2), 1.55–1.80 (3), 1.79–2.14 (4), from section 14 of the kernel note; an
  8-column kernel est. 2.9–3.4 (the byte ratio 2.78 plus fixed costs).
- The output head and sampling stay on the last card, which is the slowest stage in every setup.

### 1.3 Results (est.)

Six cards (T 26.8 ms, ceiling 159 tok/s, S_eff 4.3), per conversation / total:

| N | (a) today | (a) Q78 | (b) staggered | (c) groups of 2 | (c) groups of 4 |
|---|---|---|---|---|---|
| 1 | 37.3 | 37.3 | 37.3 | – | – |
| 2 | 22.6–27.6 / 45–55 | 26.7–29.3 / 53–59 | 37.3 / 75 | = (a) Q78 | – |
| 4 | 10.7 / 43 | 18.4–21.7 / 74–87 | 34.3–37.3 / 137–149 | 26.7–29.3 / 107–117 | = (a) Q78 |
| 8 | 4.6 / 37 | 11.8–13.7 / 94–109 (8 columns) | 18.2–19.9 / 145–159 | 24.5–29.3 / 196–235 | 18.4–21.7 / 147–174 |
| 16 | – | – | 9–10 / 145–159 | 12.8–15.5 / 205–248 | 16.6–21.7 / 265–347 |

Nine cards, 6/6/6/6/6/6/6/5/2 (T 31.3 ms, ceiling 262 tok/s, S_eff 8.2):

| N | (a) today | (a) Q78 | (b) staggered | (c) groups of 2 |
|---|---|---|---|---|
| 1 | 31.9 | 31.9 | 31.9 | – |
| 2 | 19.4–23.7 / 39–47 | 23.0–25.2 / 46–50 | 31.9 / 64 | = (a) Q78 |
| 4 | 9.1 / 37 | 15.9–18.7 / 64–75 | 29.4–31.9 / 118–128 | 23.0–25.2 / 92–101 |
| 8 | 4.0 / 32 | 10.2–11.8 / 82–95 (8 columns) | 29.4–31.9 / 235–256 | 21.1–25.2 / 169–202 |
| 16 | – | – | 14.7–16.4 / 235–262 | 21.1–25.2 / 337–403 |

On the memory-balanced nine-card split (6/6/6/6/5/5/5/5/5, ceiling 187 tok/s, S_eff 5.9) staggering gives 128 at N = 4
and 171–187 at N = 8 (21–23 each), and groups of two give 241–292 at N = 16.

- Up to S_eff conversations, staggering keeps everyone at full speed: no batching scheme comes close, because every
  multi-token step costs each conversation speed.
- Beyond S_eff, groups of two give each conversation 25–29 tok/s on six cards where single tokens give 18–20, and lift
  the total above the one-token ceiling. Groups of four only pay where the total matters more than each conversation.
- The ceiling is the last card. Anything that lightens it raises the multi-conversation total more than it raises
  single-stream speed: the GPTQ Q5_0 head of G3h (~0.8 ms less on the last card) lifts the six-card ceiling from 159 to
  ~183 tok/s and the nine-card one from 262 to ~299 (est.), against ~3 % for a single stream; the screened head (Q92)
  and a split head (unconventional idea 6) do the same.

### 1.4 The host

Per token the host builds the inputs, copies the user inputs (positions, KV indices, masks) to every card and launches
each card's graph: ~0.25–0.35 ms per card (Q49). The full-attention mask (n_kv × 2 B per card, f16) is copied
synchronously to every card that holds a full-attention layer (all of them on six and nine cards; Flint's full-attention
layers are 0, 1, 5, 9, …, 45, 48), at ~190 MB/s. With one submit thread, ~0.25 ms of server work per token and GPU
sampling (est.):

| Cards | Context per conversation | Host time per token | Rate one core can submit | Staggered ceiling |
|---|---|---|---|---|
| 6 | ≤ 1k | 1.8–2.3 ms | 430–570 tok/s | 159 |
| 6 | 32k | 3.8–4.4 ms | 225–260 | ~130 (est.) |
| 6 | 64k | 5.9–6.5 ms | 155–170 | ~110 (est.) |
| 9 | ≤ 1k | 2.5–3.4 ms | 295–400 | 262 |
| 9 | 8k | 3.3–4.2 ms | 240–305 | 262 |
| 9 | 32k | 5.6–6.5 ms | 155–180 | ~210 (est.) |
| 9 | 131k | 14.9–15.8 ms | 63–67 | ~110 (est.) |

- On nine cards the host is the next limit at N ≈ 8 even at 8k; at 32k it binds first. Masks built on the card (or at
  least uploaded once per device and asynchronously, as upstream PR #29963 does) remove the context column; groups of
  two halve the launches per token.
- Host sampling adds 0.4–0.7 ms of host time and ~2.8 ms on the last card's link per token (Q55), queued on its
  compute stream behind every token: six-card ceiling ~110 tok/s, nine-card ~150 (est.). Tool requests sample on the
  host today (Q55), so a share f of tool-request tokens adds ~2.8·f ms to the last stage; GPU sampling with a host
  grammar check (Q67) removes it.
- The Celeron also runs the HTTP server, detokenization and the desktop (Q83's headless mode frees some of it).

### 1.5 Context, power and prompts

- **Attention at depth** adds ~6.8 ms per token at 32k on nine cards (Q54), shared by the cards in proportion to their
  full-attention layers. The cards with three of them (card 1 on nine cards; cards 1 and 6 on six) gain ~1.5 ms at
  32k and ~6 ms at 131k, so the ceilings in section 1.4's table fall to ~130 (six cards) and ~210 tok/s (nine) at 32k
  and to ~110 at 64k–131k (est., linear in context). Batching does not share attention either: each conversation reads
  its own KV in (a) as in (b).
- **The 75 W limit** applies once the cards run at full duty: ≤ 10 % per stage (est., bandwidth-bound kernels), in the
  lower ends above.
- **Prompts use the same cards.** Prefill runs at ~200 tok/s on six cards in llama-bench pp512 and ~7 ms per token
  plus ~2 s per request when served (est. in breakthrough candidate 3; 80–85 tok/s at 262k on nine cards, Q58): ~5–7 ms
  of pipeline time per prompt token, about what a generated token costs at saturation (6.28 ms on six cards, 3.82 ms on
  nine). With ρ new prompt tokens per generated token the total generated rate is 1 / (ρ · p + c):

| ρ (new prompt tokens per generated token) | Nine cards, staggered | Nine cards, one stream | Six cards, staggered |
|---|---|---|---|
| 0 (pure generation) | 262 | 32 | 159 |
| 0.5 (chat with short history growth) | 137–158 | 28–29 | 102–114 |
| 2 (agent turns with tool output) | 56–72 | 22–24 | 49–61 |
| 8 (long tool outputs, short replies) | 17–23 | 11–14 | 16–22 |

  Prefix caching keeps ρ at the new tokens only (llama-server reuses a slot's cached prefix; Flint's sliding-window
  layers need `--ctx-checkpoints`). For agents with long tool outputs, faster prefill (Q80, Q89's scheduler copies,
  the on-card mask of breakthrough candidate 3) is worth as much as staggering.

## 2. Memory (question 2)

### 2.1 KV per conversation

Flint's 14 full-attention layers keep every token: 4 KV heads × 128 × 2 (K and V) × 2 B = 2 KiB per layer and token,
28 KiB per token in f16. The 35 sliding-window layers keep n_swa + n_ubatch cells per stream, padded to 256
(`llama-kv-cache-iswa.cpp:73`): 768 cells at `-ub 128`, 52.5 MiB per conversation (87.5 MiB at `-ub 512`).

| Context | Full-attention KV | Sliding-window KV | Per conversation (f16) | q8_0 KV (est.) |
|---|---|---|---|---|
| 8,192 | 224 MiB | 52.5 MiB | 0.27 GiB | 0.15 GiB |
| 32,768 | 896 MiB | 52.5 MiB | 0.93 GiB | 0.49 GiB |
| 131,072 | 3.5 GiB | 52.5 MiB | 3.55 GiB | 1.89 GiB |
| 262,144 | 7.0 GiB | 52.5 MiB | 7.05 GiB | 3.74 GiB |

q8_0 KV runs ~30 % slower on these cards (the VEC kernel, Q54) and still keeps an f16 scratch at long context (Q61),
so it is the capacity fallback, not the default.

### 2.2 What fits

Usable VRAM is 6,019 MiB per card with the desktop running ([hardware spec](../../mining_rig.md)): 52.9 GiB on nine
cards, 35.3 GiB on six. Per card ~0.25–0.4 GiB goes to the CUDA context, compute buffers and the executor's slots
(est.), and layer granularity plus the uneven spread of full-attention layers leaves ~85–90 % of the rest usable for KV
(est.).

| Cards | File | KV budget (est.) | 8k | 32k | 131k | 262k |
|---|---|---|---|---|---|---|
| 9 | UD-Q4_K_XL, 25.1 GB | 22–24.5 GiB | 81–90 | 23–26 | 6 | 3 |
| 9 | G3h class, ~33 GB | 16–18 GiB | 58–66 | 17–19 | 4–5 | 2 |
| 6 | UD-Q4_K_XL | 8–9.4 GiB | 29–34 | 8–10 | 2 | – |
| 6 | G3h class | 1.9–2.8 GiB nominal | 1–3 in practice | 0 (S4-L0 fails at 32k, Q54) | – | – |

Checks: the makers' Q4_1 (26.0 GB) with two 262k conversations on nine cards used 3.9–5.5 GB per card, and S4-L0
(32.5 GB) serves one 262k conversation on nine cards (Q58, [Flint benchmark](../../benchmarks/2026-10-22-flint.md)).
For the throughput mode N stays at ≤ ~8 on nine cards (S_eff) or ~16 with groups of two, so memory binds only from
~64k per conversation up.

### 2.3 How to split the cards

- **One copy on all nine cards.** A bar-quality file is 25–33 GB and needs at least five cards; two copies would need
  ten. Six cards are enough only for UD-Q4_K_XL-class files and ≤ 4 conversations at ≤ 32k.
- **Memory first.** KV lives on the card of its layer. Per card: its layers' weights + 2 KiB × N × L for each of its
  full-attention layers + 1.5 MiB × N for each sliding-window layer + ~0.3 GiB. On 6/6/6/6/6/6/6/5/2, card 1 holds
  layers 0, 1 and 5 (three full-attention layers, 6 KiB per token and conversation) and cards 3, 5 and 7 two each, so
  at 131k card 1 and card 3 allow ~3 conversations. `-ts` must keep each card under ~5.6 GiB at the target N × L, as
  for Q8_0 at 262k (Q34).
- **Then balance the stages.** The output head and sampling cost ~2.5 ms on the last card, about five layers' worth,
  so the last card should carry 2–3 layers when memory allows: 6/…/5/2 gives a 262 tok/s ceiling against 187 for
  6/6/6/6/5/5/5/5/5. KV-heavy setups (N × L above ~0.5 M tokens) fall back to the balanced split or to q8_0 KV.
- **Keep room on card 1** if a learned guesser (Q20's head) shares it ([pipeline drafter](pipeline_drafter.md)
  section 3.3).

### 2.4 Host RAM

- Per pipe slot: a pinned 0.6 MB output buffer, its own scheduler and graph metadata (a few MB touched, est.).
- llama-server keeps `--ctx-checkpoints` per slot in host RAM: 52.5 MiB each (the sliding-window cells). Eight
  conversations with 4 checkpoints each would take 1.6 GiB of the ~2.6 GiB free; with N ≥ 4 the server needs 1–2
  checkpoints per slot, or the second SO-DIMM (Q86).
- `--cache-ram 0` stays on: the host prompt cache copies whole sequence states (28 KiB per token).

## 3. Prior art and what has to change (question 3)

### 3.1 Prior art

| System | How several requests share a pipeline | What carries over |
|---|---|---|
| [Orca](https://www.usenix.org/system/files/osdi22-yu.pdf) (OSDI 2022), §4.2, Fig. 8 | Iteration-level scheduling: the scheduler does not wait for a batch until the number of scheduled batches reaches the number of workers (stages), so every stage processes one of the batches; contrasted with FasterTransformer, which splits one batch into micro-batches and trades batching efficiency against bubbles | Exactly approach (c): G = stages, any batch size per group; with B = 1 it is (b) |
| [DeepSpeed Inference](https://arxiv.org/abs/2207.00032) (SC 2022), §IV-C.1, Fig. 2–3 | Pipeline-parallel generation with the number of micro-batches set to the pipeline depth, generated-token micro-batches queued dynamically until sequences end; more micro-batches for prompts, fewer for generation ("hybrid scheduling") | The schedule of (b)/(c). Its premise, that a micro-batch's time hardly depends on its size because weights dominate, fails for a 256-expert MoE (table in 1.1), which is why B stays at 1–2 here |
| vLLM (main `7dc44a0`) | `max_concurrent_batches` = pipeline size ("PP requires PP-size concurrent batches to fill the pipeline", `vllm/config/vllm.py:588-598`); `EngineCore.step_with_batch_queue` keeps that many batches in flight (`vllm/v1/engine/core.py:205-211, 698-760`); a request decodes at most once per `pp_size` scheduler steps (`vllm/v1/core/sched/scheduler.py:794-795`) | (c) with groups formed by the scheduler; each request belongs to one in-flight batch |
| SGLang (main `49a2fbf`) | `event_loop_pp` cycles over `pp_loop_size` = `pp_size` + async depth micro-batches, each with its own running batch (`python/sglang/srt/managers/scheduler_pp_mixin.py:104-136, 562-571`); default micro-batch size `max_running_requests // pp_size` (`scheduler.py:1220-1226`); asynchronous P2P sends ([docs](https://docs.sglang.io/advanced_features/pipeline_parallelism.html)) | The same; per-micro-batch request sets |
| Megatron-LM core inference (main `9ef1a71`) | `forward_pass_with_pipeline_parallel` receives, computes and sends one batch per stage (`megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py:292-330`); no interleaving of decode micro-batches | What llama.cpp does today |
| [SpecPipe](https://arxiv.org/abs/2504.04104), §4, §4.4 | SpecPipe-DB adds dynamic batching (ragged tensors, paged KV); new tasks start whenever fewer than B run; 1.64–2.08× throughput and 1.61–2.06× lower time between tokens than pipeline-parallel vLLM at B ≥ 2 (RTX 3090/4090 pipelines) | Speculation and several requests in one pipeline; trees need cheap multi-token steps |
| [PipeInfer](https://arxiv.org/abs/2407.11798) (SC 2024), abstract, §IV-B | Single-request scope; notes that pipeline-parallel designs need many user requests to stay busy; "continuous speculation" generates speculative work whenever the head node would otherwise be idle | The policy of section 4: speculate only into idle capacity |
| [Petals](https://arxiv.org/abs/2312.08361) (NeurIPS 2023), §3.2, §4.2 | Each client drives its own chain of servers over the internet with its own attention cache per server; 3 concurrent clients on 3 × A100 keep 1.65 of 1.71 steps/s each; large batches split into 1,024-token micro-batches | Independent sequences through shared stages lose little per sequence while stages have slack |
| exo ([PR #2312](https://github.com/exo-explore/exo/pull/2312), [issue #2108](https://github.com/exo-explore/exo/issues/2108)) | Prompt chunks of 256 tokens interleaved with active decode, one prefill at a time (the pipeline-parallel path keeps synchronous prefill); pipeline-parallel serving deadlocks at 32+ concurrent requests | Bounded prompt chunks with decode first |
| [Sarathi-Serve](https://arxiv.org/abs/2403.02310) | Chunked prefills and stall-free schedules that add new requests without pausing ongoing decodes; uniform batches keep pipeline bubbles small | Prompt chunks as pipeline work items (3.3) |
| llama.cpp [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017) (merged 2024-03) | Pipeline parallelism for prompt ubatches (`n_copies` input copies, events); "applications can also take advantage of pipeline parallelism by making multiple calls to `llama_decode` without synchronizing" | The overlap mechanism; generation lost it once graph reuse arrived (3.2) |
| llama.cpp [PR #14363](https://github.com/ggml-org/llama.cpp/pull/14363) (merged 2025-07-16) | One KV stream per sequence by default, to avoid "cross-sequence attention" on a unified buffer; `--kv-unified` restores the old mode | Per-conversation streams for (b) |
| llama.cpp [issue #28056](https://github.com/ggml-org/llama.cpp/issues/28056), [PR #29963](https://github.com/ggml-org/llama.cpp/pull/29963) (open), [PR #24219](https://github.com/ggml-org/llama.cpp/pull/24219) (closed RFC) | #28056: why the reuse path synchronizes before writing inputs; #29963: user inputs uploaded once per device and asynchronously (compute buffers 9,133 → 2,988 MiB per GPU at 262k); #24219: pipelining one prompt's ubatches, prefill only | The mask upload fix; nothing upstream overlaps decode steps of different sequences |

No llama.cpp or ik_llama.cpp pull request that overlaps decode steps of different sequences across a layer split was
found (GitHub search, 2026-10-24). The fork's pipelined executor (Q87) is the only one known for this engine family.

### 3.2 Why stock llama.cpp cannot overlap conversations

- **Graph reuse waits.** When `process_ubatch` reuses the previous graph with pipeline parallelism on, it synchronizes
  the scheduler before `set_inputs` (`src/llama-context.cpp:1540-1545`), because the inputs of a reused graph are the
  same buffers the previous step still reads (issue #28056).
- **One input copy during generation.** `ggml_backend_sched_alloc_graph` rotates the input copy only when a graph is
  allocated (`ggml/src/ggml-backend.cpp:2094-2100`), and `ggml_backend_sched_synchronize` resets the next copy to 0 so
  CUDA graphs keep their addresses (`:2133-2145`).
- **Every logits read waits for everything.** `llama_get_logits_ith` calls `synchronize()` on the whole context
  (`src/llama-context.cpp:4240-4241`), so sampling conversation A waits for B, C and D. PR #6017's "several
  `llama_decode` calls without synchronizing" works for prompt ubatches without outputs, not for decode with sampling.
- **A different sequence means a rebuild.** With one KV stream per sequence the ubatch split is `split_equal`
  (`src/llama-kv-cache-iswa.cpp:159-215`), and a graph is reused only for the same sequence ids
  (`src/llama-graph.h:839-866`), so alternating single-sequence calls rebuild every time: ~3–5 ms of graph build plus an
  uncaptured step (~5 ms) on the Celeron ([speculation plumbing](speculation_plumbing.md) 1.6), more than a six-card
  cycle.
- **User inputs are copied synchronously.** The scheduler copies host inputs (positions, KV indices, masks) to each
  card before that card's split and blocks on each copy (`ggml/src/ggml-backend.cpp:1829-1840`).
- **The server merges everything into one step.** `update_slots` builds one batch from all slots, prompt chunks
  included (`tools/server/server-context.cpp:3629-3860`), decodes it and then samples every slot
  (`:3431-3549`, `:4664-4760`). A prompt chunk in the batch makes every conversation wait for the chunk.
- **`-np auto` uses a unified cache.** llama-server sets 4 slots and `kv_unified` when `-np` is auto
  (`tools/server/server.cpp:169-174`). On a unified cache `n_kv` is the highest used cell over all sequences
  (`src/llama-kv-cache.cpp:1285-1299`), and the flash-attention kernels skip the masked tail only when the mask is
  scanned, which happens for ≥ 1,024 queries or several streams (`ggml/src/ggml-cuda/fattn-common.cuh:1114-1133`;
  TILE kernel bound at `fattn-tile.cuh:954`). A one-token step on a unified cache therefore reads every conversation's
  KV. Explicit `-np N` keeps one stream per slot.

### 3.3 Changes to Q87's executor: a streams mode

What exists (pipespec): `llama_pipe_init` gives each slot its own scheduler with one input copy, a one-token graph
reserved over the whole cache and `caller_sync` (`src/llama-context.cpp:1615-1720`, scheduler at `:1674-1677`);
`llama_pipe_submit` takes a sequence id per token (`:1723-1859`); the race loop keeps one token of each of k sequences
in flight and waits on them in submission order (`examples/pipespec/pipespec.cpp:447-497`), which is optimal on a
pipeline that completes tokens in order. Changes, in order:

| Change | Where | Why |
|---|---|---|
| N independent prompts, each decoded into its own sequence with `llama_decode` before the loop | new job key in `pipespec.cpp` (`run_job`, `:276`) | The race shares one prompt via `seq_cp` |
| One KV stream per conversation: `kv_unified = false`, `n_seq_max = N`, `-c` = N × context | `pipespec.cpp:892-895` forces `kv_unified` for races; `src/llama-context.cpp:390-401` (same on both branches) divides `n_ctx` per stream | Each token's attention reads only its own conversation (3.2) |
| Pipe slots bound to sequences (slot s for conversation s, plus guess slots per conversation) | the race's round-robin slot choice, `pipespec.cpp:457-462` | With streams, a slot's graph is reusable only for the same sequence (`src/llama-graph.h:839-866`, same on both branches); rotating slots would rebuild every token |
| Seeded GPU sampling per sequence, the random draw keyed by (seed, position) | `pipe_init` refuses more than one backend sampler (`src/llama-context.cpp:1620-1623`); the token or logits copy is queued on the output device's stream (`:1838`) | Host sampling puts 0.6 MB per token on the last card's link: ceiling 159 → ~110 tok/s on six cards |
| Masks built on the card, or uploaded once per device and asynchronously | `ggml/src/ggml-backend.cpp:1829-1840` on v2 (+4 lines on pipespec): synchronous user-input copies, where `caller_sync` skips the wait but not the copy; PR #29963's first commit; breakthrough candidate 3 | Host time per token grows with n_kv × cards (1.4) |
| Prompt chunks (32–128 tokens) as pipe work items on their own n_ubatch-sized slot | `decode()` drains every pipe slot first (`src/llama-context.cpp:2148-2149`) | Today a new conversation's prefill stops all the others |
| A submit thread separate from the server's work | the executor loop | One Celeron core per ~240–400 tok/s (1.4) |

The identity check carries over directly: each conversation's tokens and logits must equal its own plain run with the
same seed, since streams share nothing but the weights.

### 3.4 Changes to llama-server

- Replace the merged step with a per-slot loop: when a slot's token is sampled (its pipe event fired), run
  `process_token` (stop strings, grammar, streaming) and submit the slot's next token at once; prompt chunks enter as
  separate, lower-priority work items with a bounded chunk size (`update_slots`, `server-context.cpp:3431-3549`;
  `pre_decode`, `:3629`; `post_decode`, `:4664`).
- Speculative drafting per slot becomes guesses in the same submit queue under the policy of section 4; the server's
  draft and verify machinery (`spec_i_batch`, `:617-630`) is not used in this mode.
- Context checkpoints at committed positions only, 1–2 per slot (2.4); grammar and stop checks on sampled tokens only
  (sample-and-match keeps them exact); tool requests sample on the host until Q67.
- Explicit `-np N` (one stream per slot); races within a conversation need the conversation's samples in one stream,
  since `seq_cp` across streams copies the whole buffer (`src/llama-kv-cache.cpp:516-528`): a sequence-to-stream map
  that allows several sequences per stream.

### 3.5 Groups of two (hybrid)

`llama_pipe_submit` takes one token; groups need a B-token variant: `ubatch_reserve(B, B)` per slot
(`src/llama-context.cpp:1682` in pipespec reserves 1), `n_outputs = B`, B sampler chains, and a slot bound to a fixed
pair of conversations (graph reuse needs the same sequence ids). Pairs are re-formed only when a conversation starts or
ends (one rebuild and one CUDA-graph capture per change, ~10 ms). Q78's kernels must have landed on `pascal-p106-v2`.

## 4. Speculation, races and conversations on one budget (question 4)

The pipeline holds ~S_eff tokens in flight (4.3 on six cards, 5.9–8.2 on nine) before any token waits. Conversations,
guesses (Q87) and race samples (Q93) all draw on it. Expected useful tokens per slot and pass:

| Work for a free slot | Useful tokens (est.) | Basis |
|---|---|---|
| The next token of a waiting conversation | 1.0 | always kept |
| A guess inside a copy run (copy lookup ≥ 4) | 0.83–0.98 | Q96 |
| A trained head's first guess | 0.60–0.72 | [pipeline drafter](pipeline_drafter.md), est. |
| A second race sample during reasoning | ~0.39 | Q93: shortest of 2 is 0.72 of the mean, 1/0.72 − 1 |
| A trained head's second guess | 0.36–0.52 | p² |
| A table or count guess on fresh text | ~0.2 | Q96 |
| A third race sample | ~0.17 | 1/0.64 − 1/0.72 |
| A fourth or fifth race sample | ~0.06 each | (1/0.56 − 1/0.64)/2 |

The race's quality effect (+0.5, +1.0 and +3.2 points at k = 2, 3, 5) comes on top of its time value.

Policy:
1. A conversation's next token is submitted the moment it is sampled, ahead of all speculative work.
2. Free slots F = floor(S_eff) − conversation tokens in flight. Each time a token leaves the pipeline, the free slot goes
   to the highest-valued candidate in the table, across all conversations.
3. No speculative token is submitted when F = 0. The executor queues all of a token's stages at submit time, so a
   speculative token submitted into a full pipeline delays the next conversation token by one stage on every card.
   Per-stage launch would allow preemption but costs host time per stage; below saturation it gains nothing.
4. Near full duty, wrong guesses cost clock speed for everyone (Q96's table guesser cancelled 37 ms of stage work per
   token): with ≥ 2 active conversations, skip guesses with p below ~0.3 (est.).
5. Interactive conversations get guesses and races first; batch agents get throughput only.

What it gives (est.): N = 1 is today's Q87/Q93 behaviour (all slack to guesses or the race); at N = 2 on six cards each
conversation can take one guess (a p 0.7 guesser gives 1.62× at depth 1 in Q87's event model, ~55–60 tok/s each);
at N ≥ S_eff speculation stops and the cards serve conversations only.

## 5. First experiments (question 5)

### 5.1 On the rig

1. **Step cost of N tokens, both builds (~1 h, no code).** On `pascal-p106-v2` and `pascal-p106-v2-multicol`, six cards,
   S4-L0 or G3h with `LLAMA_NO_MOE_DOWN_RESCALE=1` (UD-Q4_K_XL with `=tg`):
   `llama-batched-bench -m M -ngl 99 -fa on -c 16384 -b 512 -ub 128 -npp 512 -ntg 128 -npl 1,2,3,4,6,8`
   (one KV stream per sequence by default; tokens are random, so routing is independent). S_TG gives v(N) for rows 2–4.
   Repeat N = 1, 2, 4 on nine cards with `-ts 6,6,6,6,6,6,6,5,2`.
2. **Staggered ceiling (~1 h, no code).** `bench/pipespec/rig_run.sh` with `BG=1 GPU_LOG=1`, the all-right oracle at
   depths 0–5 on six cards and 0–8 on nine (`TS=6,6,6,6,6,6,6,5,2`, `DEPTHS='1 2 3 4 5 6 7 8'`): with every guess right,
   depth D keeps D + 1 one-token steps in flight, the same card load as D + 1 conversations. Log host CPU of the process
   (`pidstat -u -p <pid> 1`). Expected (est.): 75 / 112 / 149 / 159–167 tok/s for 2 / 3 / 4 / ≥ 5 in flight on six
   cards (the two event models of Q87 put the ceiling at 4.3× and 4.48×); 64 … 256 on nine.
3. **N real sequences (~1 h, no code).** `llama-pipespec` jobs with `mode=pipe race=N race_stop=@@never@@ n=256` on
   `prompts/chat.txt` with `chat=1` (long answers, so no sample ends early), N = 1, 2, 3, 4, 6, 8, at temperature 1
   with host sampling: N sequences of their own (sharing only the prompt's cells), one token each in flight, until the
   first reaches 256 tokens (`t_race`). Expected below experiment 2 by the logits copy: ~110 tok/s ceiling on six
   cards; the difference prices host sampling.
4. **Streams mode (code 3–5 days, then rig).** Section 3.3's first four changes in `llama-pipespec`, identity against N
   plain runs, then tokens/s at 8k and 32k per conversation with N = 2–8.

Gates: experiment 2 at ≥ 3.3× one stream with every card busy on six cards, and the submit thread below one cycle per
token; experiment 1's 2-token step on the multi-token build ≤ 1.5 T before the hybrid is built.

### 5.2 Simulator extension

`bench/ngram_replay/pipeline_sim.py` replays one transcript at a time (`simulate`, `:451`). A `simulate_streams` next to
it would take:
- N transcripts (Q51's 38 as conversations, staggered starts, optional Poisson arrivals with their prompts), each with
  its own guesser and window, sharing one `card_free` per card;
- the executor's queueing (all stages queued at submit, no extra hop after a free card) and, as an option, per-stage
  launch;
- groups of B with stage times scaled by r(B) from experiment 1;
- a host model: one submit thread at a + b · cards + cards · n_kv · 2 B / 190 MB/s per launch, sampling on the GPU or
  on the host (+2.8 ms on the last card's stream), a second thread for server work;
- the 75 W slowdown as a function of each card's duty;
- prompt chunks as work items with stage times from the measured prefill rate;
- section 4's policy (priority classes, budget S_eff);
- output: total tok/s, per-conversation tok/s (p10/p50/p90), p95 time between tokens, time to first token, card and host
  utilization, cancelled card-ms.

Validate it on experiments 2 and 3, then use it to pick the split, B and the policy thresholds.

## What this changes in the plan

- **Q87's executor becomes the serving engine,** not only a speculation trick. Seeded per-sequence GPU sampling,
  per-conversation KV streams with slots bound to them, and prompt chunks as pipeline work move to the top of its
  production list ([pipelined speculation](pipelined_speculation.md), "What a production version needs").
- **Priority 8 (agent serving) gains the multi-conversation goal:** row 1 above, with experiments 1–3 as its first rig
  work; the on-card mask (breakthrough candidate 3) and Q89's scheduler copies serve it twice, through prefill and
  through host time per token.
- **Q78 matters for groups of two above S_eff and for llama-server's fallback path,** not for N ≤ S_eff.
- **Lighter last cards pay twice:** Q92's screened head, Q56's GPTQ head and a split head each raise the
  multi-conversation ceiling by ~15–20 % on six cards (est.; they overlap, so the gains do not add up).
- **llama-server settings today:** explicit `-np 2` at most (one stream per slot), not `-np auto`, which unifies the KV
  cache; `--ctx-checkpoints` 1–2 per slot on the 4 GB host.

## Already covered

| Direction | Where it is |
|---|---|
| Staggered conversations through the idle cards, ~5× aggregate | [Breakthrough candidates](breakthrough_candidates.md), candidate 1; Q87 |
| Executor, slots with their own schedulers, dropping, KV order, races | [Pipelined speculation](pipelined_speculation.md); Q87, Q93 |
| Pipeline ceilings 4.3× / 5.9× / 8.2×, depth and hop sensitivity | [n-gram replay](../../../bench/ngram_replay/README.md#pipelined-speculation-q87) |
| Sharing slots between race samples and guesses | [Pipeline drafter](pipeline_drafter.md) section 3.4 |
| Multi-token kernels and their cost model | Q78; [kernel note](pascal_mmvq_integration.md) section 14 |
| CUDA graphs per batch shape, graph rebuild costs, two-slot drafting shapes | Q52; [speculation plumbing](speculation_plumbing.md) |
| Scheduler copies, attention mask on the card | Q89; breakthrough candidate 3 |
| Hop cost between cards | Q81 |
| Split output head as a pipeline-ceiling lever | [Unconventional ideas](unconventional_ideas.md), idea 6 |
| KV-sharded attention at long context (competes for the same idle cards) | Q95 |
| Two copies for two users | [Unconventional ideas](unconventional_ideas.md), rejected list |
| Host sampling for tool requests, GPU sampling with a host grammar check | Q55, Q67 |
| Prompt processing sweeps | Q80 |
| Headless mode, second SO-DIMM | Q83, Q86 |

## Not verified

- No staggered run of several sequences has been timed on these cards: the cycle with every card busy, the clocks at
  full duty under the 75 W limit, and the host's submit rate on the Celeron are estimates.
- v(N) on the fork: v(2) is interpolated (Q19) or carried over from the stock build's two-slot run; Q78's ratios are a
  model; an 8-column kernel does not exist.
- That every stage scales by the same r(B) in a group; the last card's head and sampling may scale differently.
- Host costs: ~0.3 ms per card per token comes from one profile (Q49); the mask-copy cost assumes ~190 MB/s per link and
  serial copies; server work per token (~0.25 ms) is a guess.
- KV budgets: compute-buffer sizes per card with N streams and the executor's slots, and the packing efficiency, are
  estimates; the light-last-card split's memory per card at N × L must be read from the rig's logs.
- The prefill cost per prompt token in a mixed pipeline (5–7 ms assumed) and the effect of prompt chunks on the other
  conversations' time between tokens.
- The value of race samples and guesses per slot comes from offline data (Q93, Q96) and a modelled drafter.
- Whether prompt chunks of 32–128 tokens keep enough prefill efficiency on 256-expert layers (a 32-token chunk touches
  ~135 of 256 experts, est.).

## Sources

Serving systems and schedules
- G.-I. Yu et al., "Orca: A Distributed Serving System for Transformer-Based Generative Models", OSDI 2022,
  [PDF](https://www.usenix.org/system/files/osdi22-yu.pdf), §4.2 and Fig. 8 (pipelined batches, FasterTransformer
  micro-batching) (P)
- R. Y. Aminabadi et al., "DeepSpeed Inference: Enabling Efficient Inference of Transformer Models at Unprecedented
  Scale", SC 2022, [arXiv 2207.00032](https://arxiv.org/abs/2207.00032), §IV-C.1, Fig. 2–3 (P)
- vLLM, main `7dc44a0` (2026-10-24): [`vllm/config/vllm.py`](https://github.com/vllm-project/vllm/blob/main/vllm/config/vllm.py)
  `max_concurrent_batches` (:588-598), [`vllm/v1/engine/core.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/engine/core.py)
  batch queue and `step_with_batch_queue` (:205-211, :698-760),
  [`vllm/v1/core/sched/scheduler.py`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)
  (:794-795) (P)
- SGLang, main `49a2fbf` (2026-10-24): [`scheduler_pp_mixin.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler_pp_mixin.py)
  `event_loop_pp` (:104-136), `init_pp_loop_state` (:562-571);
  [`scheduler.py`](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/managers/scheduler.py) (:1220-1226);
  [pipeline parallelism docs](https://docs.sglang.io/advanced_features/pipeline_parallelism.html) (P)
- Megatron-LM, main `9ef1a71`: [`abstract_model_inference_wrapper.py`](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/inference/model_inference_wrappers/abstract_model_inference_wrapper.py)
  `forward_pass_with_pipeline_parallel` (:292-330) (P)
- A. Agrawal et al., "Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve",
  [arXiv 2403.02310](https://arxiv.org/abs/2403.02310) (P)
- A. Borzunov et al., "Distributed Inference and Fine-tuning of Large Language Models Over The Internet",
  [arXiv 2312.08361](https://arxiv.org/abs/2312.08361), §3.2, §4.2 and Tables 2–3 (P)
- exo: [PR #2312](https://github.com/exo-explore/exo/pull/2312) (interleaved prefill), [issue #2108](https://github.com/exo-explore/exo/issues/2108)
  (pipeline-parallel concurrency) (P)

Pipelined speculation
- "SpecPipe: Accelerating Pipeline Parallelism-based LLM Inference with Speculative Decoding",
  [arXiv 2504.04104](https://arxiv.org/html/2504.04104v2), §4 and §4.4 (SpecPipe-DB) (P)
- B. Butler et al., "PipeInfer: Accelerating LLM Inference using Asynchronous Pipelined Speculation", SC 2024,
  [arXiv 2407.11798](https://arxiv.org/abs/2407.11798), abstract and §IV-B (P)
- Y. Yu et al., "Speculative Pipeline Decoding", [arXiv 2605.30852](https://arxiv.org/abs/2605.30852) (P)

llama.cpp
- [PR #6017](https://github.com/ggml-org/llama.cpp/pull/6017) "llama : add pipeline parallelism support" (P)
- [PR #14363](https://github.com/ggml-org/llama.cpp/pull/14363) "llama : add high-throughput mode" (per-sequence KV
  streams) (P)
- [Issue #28056](https://github.com/ggml-org/llama.cpp/issues/28056) (synchronization before `set_inputs`),
  [PR #29963](https://github.com/ggml-org/llama.cpp/pull/29963) (user inputs uploaded once per device),
  [PR #24219](https://github.com/ggml-org/llama.cpp/pull/24219) (intra-prompt pipeline scheduling) (P)
- Fork source, branch `pascal-p106-v2` (`032b130f4`): `src/llama-context.cpp` (:390-401, :526-559, :1527-1547,
  :4240-4241), `src/llama-graph.h` (:839-866), `src/llama-graph.cpp` (:49-66), `src/llama-kv-cache.cpp` (:473-528,
  :733, :1285-1317), `src/llama-kv-cache-iswa.cpp` (:73, :159-215), `ggml/src/ggml-backend.cpp` (:911-913, :1823-1946,
  :2094-2145), `ggml/src/ggml-cuda/fattn-common.cuh` (:1114-1133), `ggml/src/ggml-cuda/fattn-tile.cuh` (:954),
  `tools/server/server.cpp` (:169-174), `tools/server/server-context.cpp` (:521-531, :587-615, :3431-3549, :3629-3860,
  :4664-4760), `tools/batched-bench/batched-bench.cpp` (P)
- Fork source, branch `pascal-p106-v2-pipespec` (`a7204884d`): `src/llama-context.cpp` (:1615-1859, :2148-2149),
  `ggml/src/ggml-backend.cpp` (`ggml_backend_sched_set_caller_sync`), `examples/pipespec/pipespec.cpp` (:276, :447-497,
  :892-895, :924) (P)

This project
- [Quant registry](../../../quants/experiments.md) (Q19, Q34, Q49, Q51, Q54, Q55, Q56, Q58, Q67, Q76, Q78, Q80, Q81,
  Q83, Q86, Q87, Q89, Q92, Q93, Q95, Q96), [Flint benchmark](../../benchmarks/2026-10-22-flint.md) (two conversations),
  [Flint chat benchmark](../../benchmarks/2026-10-23-flint-chat.md), [hardware spec](../../mining_rig.md),
  [n-gram replay and pipeline simulator](../../../bench/ngram_replay/README.md), [bench/pipespec](../../../bench/pipespec/README.md),
  and the notes linked in the scope
