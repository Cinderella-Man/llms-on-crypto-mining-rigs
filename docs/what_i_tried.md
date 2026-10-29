# What I tried, and why

A guided tour of the first week of this project (2026-10-22 to 2026-10-29): every line of work, why I started it, what
the measurements said, what each result taught, and where it stands now. It is written to be read front to back by
someone who knows computers but not GPU kernels. The detail behind every claim is one link away: the
[quant registry](../quants/experiments.md#registry) has a row for every experiment (the `Qnnn` IDs used below), and
each result links its data folder under [data/](data/README.md).

Numbers are measured unless marked *est.* (my own arithmetic or a simulator). Speeds marked "agent check" come from
[bench/agent_check](../bench/agent_check/README.md), the project's release metric; `llama-bench` numbers (tg128,
pp512) are diagnostics.

**Contents**

1. [The week at a glance](#1-the-week-at-a-glance)
2. [Six ideas you need first](#2-six-ideas-you-need-first)
3. [Storyline A: the rig, and learning to measure it](#3-storyline-a-the-rig-and-learning-to-measure-it)
4. [Storyline B: the engine (Miner Llama)](#4-storyline-b-the-engine-miner-llama)
5. [Storyline C: model files and quality](#5-storyline-c-model-files-and-quality)
6. [Storyline D: guessing ahead (speculation and drafters)](#6-storyline-d-guessing-ahead-speculation-and-drafters)
7. [Storyline E: beyond one rig, and other people's claims](#7-storyline-e-beyond-one-rig-and-other-peoples-claims)
8. [Where each model stands](#8-where-each-model-stands)
9. [What's next](#9-whats-next)
10. [The lessons that cut across everything](#10-the-lessons-that-cut-across-everything)
11. [Glossary](#11-glossary)

## 1. The week at a glance

| Day | What happened |
|---|---|
| 10-22 | Inventory and calibration of the rig; the first models; Flint chosen as the flagship; focus narrowed to two MoE models |
| 10-23 | The quality bar set (Unsloth's UD-Q4_K_XL); **v1 released**: CUDA graphs, Pascal kernels and a fast Flint file, about 2× stock llama.cpp in chat; batched draft-checking measured and found too expensive |
| 10-24 | The pipelined executor (guesses run on idle cards); the host decode loop measured; prompt-processing work; hardware tuning researched; GPTQ rounding for the output head |
| 10-25 | v2 release candidates; the first trained drafter served on the rig; all training data so far excluded for provenance |
| 10-26 | **v2.0 released**: Flint Q8_0 at 44.6–45.9 tok/s on the agent check (stock 14.4); agentic speed becomes the only release metric; a publishable agentic data set generated overnight |
| 10-27 | v3 starts with Ornith-1.5-122B: its baseline, Pascal IQ kernels, a Q8_0 reference, new 2–3-bit formats; the answer store; Flint's MTP heads; SLQ research; the near-Q8_0 Flint file |
| 10-28 | Ornith-122B at 14.5 tok/s (valid1, 2.3× stock); the near-Q8_0 Flint file 6.3 % faster than Q8_0; the reply re-read fix; Ornith's built-in draft layer measured; 2–4-token kernels; the page-walk theory closed |
| 10-29 | The last MTP-head tests (fast GPUs, round 3, diagnostics, two training changes): the head is closing; cheaper drafts for Ornith designed |

## 2. Six ideas you need first

### 2.1 Generating text is mostly reading memory

A model writes one token (about ¾ of a word) at a time, and for each token the GPU reads every weight that takes part.
The arithmetic per weight is tiny, so for a single user

> tokens per second ≤ memory bandwidth ÷ bytes of weights read per token.

A P106-090 reads ~173 GB/s. A file that reads ~2 GB per token therefore has a floor of ~12 ms per token. Real passes
take roughly twice that: real kernels reach only 40–55 % of the bandwidth on these cards
([calibration](benchmarks/2026-10-22-single-card-calibration.md)), and every tiny kernel launch, every hand-off between
cards and every host step adds time. Most of this project is about closing that gap.

### 2.2 Mixture-of-experts (MoE) models read little per token

An MoE layer holds many small "expert" networks and a router that picks a few of them per token. Flint has 39.8 B
weights but uses only ~2.5 B per token, so it reads a small slice of a big model. That is why MoE models are the right
class for this rig: the dense Qwen3.8-27B ran at 7.5 tok/s where Qwen3.6-35B-A3B (an MoE) ran ~4.7× faster.

### 2.3 A layer split: nine cards working in turn

No card holds a whole model, so each card holds a slice of consecutive layers (`-ts` sets the shares). A token passes
card 1 → card 2 → … → card 9, and only one card works at a time. On nine cards each card computes ~8.7 % of a token
(*est.*) and waits the rest, idle. The hand-off between cards is small (the hidden state, ~8 KB) but slow: there is no
card-to-card link, so it goes through host memory over PCIe 1.0 x1, est. 0.55–0.65 ms per boundary. Idle cards are the
rig's biggest waste, and filling them is the idea behind [storyline D](#6-storyline-d-guessing-ahead-speculation-and-drafters).

### 2.4 Quantization, and how quality is judged

Weights are stored in fewer bits to fit and to read less: Q8_0 is ~8.5 bits per weight, Q4_0 ~4.5, the 2–3-bit IQ
formats less. Fewer bits change the model's predictions slightly. Quality is measured by running a reference file and
a candidate over the same text and comparing their next-token distributions at every position:

- **KLD** (KL divergence): how far the candidate's distribution drifts; 0 is identical.
- **Same top token**: how often both pick the same most likely token.

Two runs of the *same* file on different GPU kernels already differ by KLD ~0.065 on Flint, because rounding flips
which expert the router picks; that is the noise floor (Q21).

### 2.5 Guessing ahead (speculative decoding)

Cheap guesses for the next tokens can be checked by the full model; when a guess is right, a token comes for free.
There are two ways to check:

- **Batched verify:** run the model once over several tokens. On a P106 this is expensive: checking ~3 tokens costs
  2.2× one token (Q19).
- **Pipelined (Miner Llama's way):** feed guesses into the idle cards as ordinary one-token steps while later cards
  finish earlier tokens. A wrong guess is simply dropped; the output is bit-identical to plain decoding.

What matters is **acceptance** p, the chance a guess is right. With guesses in flight a pass yields on average about
1 + p + p² + … ≈ 1/(1 − p) tokens: 1.33 at p = 0.25 however many cards there are, 3.3 at p = 0.7. Card count pays only
when acceptance is high.

### 2.6 Measuring what a user feels

The release metric is the **agent check**: a coding agent (pi) fixing 12 real bugs in Exercism exercises against
llama-server, timed per task, with multi-turn tool calls and a context that grows every turn. Single-request numbers
(chat tok/s, `llama-bench`) only help find causes. For effects of a few percent, live runs are too noisy (the same
server measured 45.0 and 46.5 tok/s on one day), so I use **paired replays**: recorded agent requests re-sent to two
setups with the same seeds; both generate identical tokens, so the ratio of time per token is the change alone, with a
95 % interval.

## 3. Storyline A: the rig, and learning to measure it

### 3.1 The hardware (10-22)

**Why.** Mining cards were built for memory bandwidth, which is what text generation needs. Everything else about the
rig is hostile, so I first had to know exactly what it is.

**What I did.** A read-only inventory ([scripts/rig_inventory.sh](../scripts/rig_inventory.sh)) and transfer tests.

**Result.** Nine P106-090 cards, each 6 GB, 5 SMs, ~173 GB/s (a plain streaming read reaches 179 GB/s), FP16 at 1/64
of the FP32 rate, no tensor cores, 75 W limit. Every card sits on a PCIe 1.0 x1 link: ~189 MB/s host to card, ~715 MB/s
with all nine at once, no peer-to-peer. The host is a 2-core Celeron without AVX, 4 GB of RAM, a 64 GB SSD.

**Lesson.** Every weight must live in VRAM (54 GB in all); nothing may be built with AVX; builds use CUDA 12.9 for
sm_61 on driver 580, the last lines that support Pascal. PCIe 2.0 is impossible: the cards advertise only 2.5 GT/s
(Q86). Details: [mining_rig.md](mining_rig.md), [setup.md](setup.md).

### 3.2 A survey before buying anything (10-22)

**Why.** No published LLM benchmark existed for a 5-SM card.

**What I did.** A literature and source survey: 82 claims to test, 81 experiments in five phases
([survey](research/moe-inference-on-the-p106-rig.md), [tracker](experiments.md)).

**Result.** Only the llama.cpp family runs on sm_61 (ik_llama.cpp needs AVX2; other engines need newer GPUs). The
planning range was *est.* 20–50 tok/s for ~3 B-active MoEs.

**Lesson.** Calibrate first, shop for models second. Most of the survey's later phases became irrelevant once the
focus narrowed; the [registry](../quants/experiments.md#registry) replaced the tracker.

### 3.3 Single-card calibration (10-22)

**Why.** Are the nine cards equal, and how much of the bandwidth does real code use?

**What I did.** Llama-2-7B (llama.cpp's scoreboard model) on each card in turn, and split over 2 and 9 cards.

**Result.** Q4_0: ~19.5 tok/s generation, ~218 prompt; all nine cards within 5 % of each other. Effective bandwidth:
Q4_0 73 GB/s (42 %), Q4_K_M 60 GB/s (35 %), Q8_0 93 GB/s (54 %); Q4_K_M is 7 % bigger than Q4_0 but 23 % slower. Two
cards are 4–5 % faster than one; nine are ~13 % slower than two, about 1 ms per card boundary.

**Lesson.** These cards are limited by their few compute units, not by memory: simple formats (Q8_0, Q4_0) beat
K-quants of similar size. That shaped every file choice that followed
([page](benchmarks/2026-10-22-single-card-calibration.md), [data](data/2026-10-22_e0.8_calibration/README.md)).

### 3.4 Small models, tiny kernels (10-22)

**Why.** Small MoEs should reach ~100 tok/s on paper. Was the 2-core CPU the brake?

**What I did.** Five small MoEs on one card, with one or both CPU cores kept busy; then a build with upstream PR #27721,
which turns CUDA graphs back on for Pascal.

**Result.** 28.9–54.1 tok/s at ~10–11 µs per graph node; the GPU was busy 95–98 % of the time and taking away a CPU
core cost at most 6 %. CUDA graphs: +16 % on granite-3.1, +7 % on Qwen3.6, +14 % on Flint.

**Lesson.** The cost is thousands of tiny kernels on the GPU, not the CPU. "Fewer, larger kernels" became the engine's
first lever ([page](benchmarks/2026-10-22-small-moe-overhead.md)).

### 3.5 The first real models, and the 4 GB host (10-22)

**Why.** Find a real model's speed and longest context.

**What I did.** Qwen3.6-35B-A3B and the dense Qwen3.8-27B with `llama-bench` and real chat requests.

**Result.** Qwen3.6: 18.5 tok/s on 4, 5 or 9 cards alike; 14.0 at 32k context, 8.1 at 64k; the full 262k context fits
in 27 GB. With CUDA graphs, its built-in MTP draft and GPU sampling (`-bs`), real chat went 15.5 → 27.7 tok/s. The dense
27B: 3.9 → 7.5. At 262k the default `-ub 512` pinned ~1 GiB of host RAM and the out-of-memory killer took the server
(and parts of the desktop) down; `-ub 128` cuts that buffer to 258 MiB, and llama-server also needs `--cache-ram 0` and
few context checkpoints.

**Lesson.** MoE is the right class here, and the host's 4 GB decides the safe flags. Those flags are in every command
line since ([baselines](benchmarks/2026-10-22-qwen-baselines.md), [speed-ups](benchmarks/2026-10-22-speedups.md)).

### 3.6 Choosing the models (10-22 onwards)

**Why.** Flint (39.8 B MoE, 2.5 B active, mostly sliding-window attention, so a small KV cache) reads less per token
than Qwen3.6.

**What I did.** Compared files, card counts and quants.

**Result.** The makers' Q4_1 ran 27.9–28.1 tok/s (`llama-bench`, with CUDA graphs), 1.4× Qwen3.6, and 24.3 tok/s in
real chat at 262k. But format trades against quality: Q4_1 matched the reference's top token 78.6 % of the time, Q5_0
85.0 %, UD-Q4_K_XL 87.6 % (at 24.3 / 18.9 / 15.8 tok/s).

**Decisions.**
- 10-22: two MoE tracks, Flint and Qwen3.6, tuned deep; the dense model and the survey's other candidates dropped.
- 10-23: the quality bar is Unsloth's UD-Q4_K_XL; the Q4_1 tier is too lossy ([storyline C](#5-storyline-c-model-files-and-quality)).
- 10-24: the project stays Pascal-specific and becomes multi-model, with Ornith-1.5-122B next (Q158).
- Flint Q8_0 on nine cards is the headline configuration.

Links: [Flint page](benchmarks/2026-10-22-flint.md), [models/flint](../models/flint/README.md).

### 3.7 Vulkan (10-23)

**Why.** A community A/B on a GTX 1060 (the same GP106 chip) found Vulkan +11 % on a dense model and +47 % on a small
MoE.

**What I did.** A no-AVX Vulkan build, Flint's dispatch lists and a rig test plan
([vulkan_on_the_rig.md](vulkan_on_the_rig.md)).

**Result.** Not run on the rig yet. *Est.* risk: Vulkan records every dispatch on the Celeron, so above ~20 µs per
dispatch the host would set the speed.

**Status.** Open.

### 3.8 Storage and loading

**Why.** The 64 GB SSD had ~26 GB free against 4–42 GB model files.

**What I did.** Models on an ext4 USB 3 drive with a `nofail` mount ([setup.md](setup.md#storing-models)).

**Result.** Load times on nine cards: 199 s for 25.1 GB, 438 s for 32.5 GB, 568 s for 42.3 GB; Ornith-122B ~5.5–6 min.
Each card loads at ~190 MB/s.

**Lesson.** Storage limits load time, never speed: once loaded, weights never cross PCIe again. Disk space is not a
reason to rule out a model.

### 3.9 Measurement traps (10-23/24)

Several "effects" turned out to be artefacts, and each became a standing rule:

| Trap | What happened | Rule since |
|---|---|---|
| Warm-up (Q43) | Qwen3.6's MTP looked like 1.24×; the plain run was still warming up. Real: 1.04× | Warm up before timing |
| Run order (Q117) | A build seemed 3.6–9.8 % slower; the prompt runs before each test caused it | Same order in every arm |
| The bench tool's own memory (Q79) | A "cliff" from 23.3 tok/s at 32k to 3.6 at 64k: `llama-bench` held 1.78 GiB in host RAM and the 4 GB host swapped (*est.* ~80 % likely) | Check host memory |
| Cold clocks (Q141) | Generation from idle runs +18 % slower than after a prompt | Log clocks |
| Prompt speed tools | `llama-batched-bench` reports ~2.3× less prompt speed than the server | Take agent numbers from the server |
| Task noise | Two runs of one file disagree on ~11 % of tool-call tasks; a 2-point difference needs ~2,160 paired tasks | KLD and paired tests, not small task sets |

### 3.10 Hardware tuning: clocks, power, memory (10-24 to 10-28)

**Why.** Free settings could add a few percent with no code.

**What I did.** A research round ([hardware_tuning.md](research/notes/hardware_tuning.md),
[power_and_tuning.md](research/notes/power_and_tuning.md)), power logging, a PowerMizer A/B, clock-share tallies, and a
kit for a memory-clock offset.

**Result.**
- Power (Q143): 35–39 W per card while decoding, 317–329 W in all, ~8.9 J per token; the 75 W cap shows in under 4 % of
  busy samples, so power limits change nothing.
- PowerMizer "maximum performance" (Q141): no change (29.75 against 29.79 tok/s).
- **Idle cards drop to the 1,354 MHz base clock** (Q154): the same prompt runs at 38.5 tok/s at boost, 34.0 at base.
  Flint's release line keeps the cards at boost 81 % of the time, Ornith on v2.0 only 50 %. No setting holds the clock:
  application clocks are not supported on these cards and locked clocks need Volta.
- Memory offset (Q133): *est.* +3–5 % decode at +10 % memory clock (only ~55 % of a token is weight streaming). The kit
  is ready; it runs only with someone at the console, proven by a bit-exact output check since the cards have no ECC.

**Lesson.** Hardware settings give little; the clock finding mattered most, because it showed that part of
pipelining's gain is simply keeping the cards busy enough to stay at boost
([clock data](data/2026-10-27_clock_share/README.md)).

### 3.11 The metric moves to agentic speed (10-24 to 10-26)

**Why.** Agents send tools with every request, and that switched GPU sampling off: tool requests ran 12–16 % slower per
token (Q55). And on agent work, v2's per-token gain looked like only ~1.10× (*est.*) against ~3.5× on prompt reading.
Chat numbers were measuring the wrong thing.

**What I did.** [bench/agent_check](../bench/agent_check/README.md): the pi agent on Exercism tasks with hidden tests,
a logging proxy, and a replay tool that re-sends recorded requests with the same seeds.

**Result.** Release settings were chosen on it: pipeline depth 4 (Flint-Pascal-Q5 +11 % over depth 3), `-b 512 -ub 128`
(larger batches ran the host out of memory at 262k), the agent prompt cache off (11 s fixed per task, no gain; Q111).

**Lesson.** Measure what the user feels; standalone numbers are diagnostics.

### 3.12 Paired replays (10-27/28)

**Why.** Live agent runs can't resolve a few percent.

**Result.** The answer store looked like +0.9 % or +4.2 % live depending on the control run; replayed it is +1.4 %
(95 % interval 0.4–1.9 %), with the offline model's +4.3 % about 3× too high (Q201, Q202). The same method showed the
MTP head 3.2 % slower and the decode-only mask 0.4 % slower.

**Lesson.** Few-percent decisions need paired replays and a decision rule written down before the run.

## 4. Storyline B: the engine (Miner Llama)

Miner Llama is my Pascal fork of llama.cpp. Every feature ships with an off switch and an exactness check, so each one
can be kept or dropped on evidence. The full list: [engine_features.md](engine_features.md).

### 4.1 v1: making a P106 decode Flint at all (10-22/23)

**Why.** Stock llama.cpp turns CUDA graphs off below Volta and launches many small kernels, and a 5-SM card pays dearly
for each launch. K-quants also decode slowly here.

**What I did.** Five changes: CUDA graphs on Pascal (PR #27721); a 2-warp launch shape for wide rows (Q34); Pascal
matrix-vector kernels for Q4_0, Q4_1, Q5_0 and Q8_0 (Q38), then Q5_K and IQ4_XS (Q46), with rows staged through shared
memory in 128-bit loads; a switch for llama.cpp's MoE overflow rescale; and an event profiler, because NVIDIA's own
profilers refuse mining cards.

**Result.** On six cards with UD-Q4_K_XL (`llama-bench`): stock 17.6 → CUDA graphs 19.5 → 2-warp shape 24.5 → Pascal
kernels 34.1 tok/s. In a five-turn chat at 262k on nine cards, 1.97–2.00× stock (UD 14.20 → 27.97, Q8_0 14.78 → 29.55).
Output stays within summation-order noise. Repacking weights into separate planes (Q39) was dropped: staging the
unchanged blocks was just as fast.

**Lesson.** Fewer, larger launches and lean decoding matter more than bytes
([chat page](benchmarks/2026-10-23-flint-chat.md), [v1 notes](../release/notes-v1.0.md)).

### 4.2 v2: the one-token engine (10-23 to 10-26)

**Why.** About 9 ms of a ~25 ms token was not weight streaming, and the MoE rescale existed only because of FP16 block
sums.

**What I did.** FP32 activation sums, so the rescale goes (Q76); the shared expert fused with its activation (Q50); one
persistent kernel per MoE layer instead of 13 launches (Q88); Pascal kernels for 2–4-token steps (Q78) and for
Q4_K/Q6_K (Q149, Q101); the norm writing 8-bit activations with the residual add folded in (Q100); prompt-processing
fixes (4.4); and the pipelined executor (6.3).

**Result.** Q76 +9.6 % with identical perplexity; Q50 +0.7 %; Q88 +5.6–8.8 %, bit-exact. Of v2's decode gain, 87 % is
Pascal-specific (Q144). The first release candidate with everything on was *slower* than its parts (36.8 against
41.8 tok/s); turning switches off one at a time found the culprits (a decode-attention kernel −9 %, an embedding row on
card 1 and packed uploads with no gain, a card-built mask that made prompts 3.6× slower), which ship off. On the agent
check, Q8_0 on nine cards: **stock 14.4 → v1 25.7 → v2.0 44.6–45.9 tok/s** (29.3 without pipelining).

**Lesson.** Measure every switch inside the release configuration; parts that win alone can lose together
([v2 notes](../release/notes-v2.0.md), [release results](../release/flint-v2/results.md)).

### 4.3 The host decode loop (10-24)

**Why.** Could the 2-core Celeron be the limit? Before the fork, llama-server ran ~8–10 ms per token slower than
`llama-bench`.

**What I did.** Measured the host's share per token (Q118); tried packed asynchronous uploads (Q120), tokens fed on the
device (Q121, fixed in Q156), one host thread, PowerMizer mode 1.

**Result.** Within a token the host stays ahead of the cards (Q49). On the fork the server gap is only 0.3–2.3 ms, and
streaming costs nothing. Packed uploads: 8 staggered conversations 54.9 → 66.3 tok/s (+21 %); with short prompts 8
conversations reach 262.6 tok/s (Q155). CUDA events distort timing badly: 12 events per token took 37.4 to 23.4 tok/s.

**Lesson.** The host is not the single-stream limit; it matters only for many conversations
([host_decode_loop.md](research/notes/host_decode_loop.md)).

### 4.4 Prompt processing (10-24/25)

**Why.** On compute 6.1 llama.cpp's batched kernel (MMQ) computes each expert in 64-token tiles while an expert sees
~3 tokens, so *est.* 95 % of the expert arithmetic multiplied padding. Graph reuse also stalled every card, and the
host copied the attention mask to each card one after another.

**What I did.** Cherry-picked upstream PR #31047, which sizes the tiles by tokens per expert (Q108); stopped graph
reuse while a micro-batch is in flight (Q109); a register-tiled prompt-attention kernel reading a Q8_0 cache (Q163);
8 scheduler copies instead of 4 (Q157); a mask built on each card (Q110).

**Result.** Q108 carries the prompt gain, +22–29 %. Micro-batch grid on nine cards, 8k prompt: 565 / 636 / 714 / 692
tok/s at `-ub` 128 / 256 / 512 / 1024 (but `-ub 512` runs the 4 GB host out of memory at 262k, so the release stays at
128). Q163: 8k prompts 699 → 766 tok/s. Q157: 765 against 766, no gain. The card-built mask made 8k prompts 3.6× slower,
so it is off.

**Lesson.** Padding, not bandwidth, was the prompt cost. For agent work the fixed per-request costs matter more than
peak prompt speed: short turns pay 0.55–0.8 s per 128-token micro-batch against 0.26 s inside a long prompt (Q264,
planned) ([prefill_pascal.md](research/notes/prefill_pascal.md)).

### 4.5 Attention at long context (10-23 to 10-28)

**Why.** On sm_61 Flint's decode attention reads the KV cache at 57–87 GB/s, capped by shared-memory traffic (*est.*
from the machine code). A kernel aware of grouped-query attention was *est.* to reach 130–145 GB/s.

**What I did.** Built a Pascal decode-attention kernel (Q151); later profiled Ornith-122B after a 39,448-token prompt
(Q210).

**Result.** Q151 is correct but slower: −9 % at short context, 15.0 against 25.4 tok/s after 32k, so it ships off. Q210:
attention is 32 % of a token at 39k (45.95 of 142.9 ms) but ~6 % on agent-check sessions (≤ 7.5k tokens).

**Lesson.** Model-level timing overturned a sound-looking estimate; attention matters only for long sessions
([attention note](research/notes/pascal_attention_long_context.md)).

### 4.6 CUDA graphs and batched checking (10-23/24)

**Why.** Speculation changes the batch shape, and a 1-token and a k-token graph share one cache key.

**Result.** One shape switch costs *est.* 10–20 ms. More decisive: checking 2.9 tokens costs 2.22× a plain token and 4
tokens 3.55× (Q19), so even perfect drafts would give at most ~1.3×. Batched checking was parked; the idle cards were
used instead ([storyline D](#6-storyline-d-guessing-ahead-speculation-and-drafters)).
([speculation_plumbing.md](research/notes/speculation_plumbing.md))

### 4.7 Pascal kernels for Ornith's IQ formats (route 1, 10-27/28)

**Why.** Ornith-1.5-122B fits 54 GB only at 2–3 bits, in IQ2_S/IQ3_XXS. On v2.0 its routed IQ experts took 43 % of a
token at 19–29 GB/s, while the Q8_0 tensors ran at ~150 GB/s.

**What I did.** Two rounds: one super-block per lane with the lookup grid in shared memory; then 11 variants timed on a
P106 with every call reading more than the 1.5 MB L2 cache (a lesson from another engine: benchmark with cold caches).

**Result.** The winner keeps the sign masks in registers and runs 3 blocks per SM: gate/up 6.67–7.28 ps per weight
(llama.cpp's kernel 9.97–11.07), down 7.07–9.19 (17.4–18.7). A token's IQ expert products: 17.25 ms against 31.63 for
llama.cpp. Agent check on UD-Q2_K_XL: 11.3 against 9.7 tok/s.

**Lesson.** Register pressure and table lookups, not bandwidth, limited these kernels
([data](data/2026-10-27_iq_kernels/README.md)).

### 4.8 New 2–3-bit formats decoded in registers: N2P and N3P (route 2, 10-27/28)

**Why.** Below ~3.5 bits a P106 is limited by instructions per weight, not bytes, so fewer bits buy memory, not time. A
format whose levels decode in registers could reach Q4_0's ~4 ps per weight (*est.* 3.5–4) instead of the IQ codes'
lookups.

**What I did.** Built the N2P (~2.69 bits) and N3P (~3.25 bits) types with CPU reference, quantizer and Pascal kernels
(Q185); timed them on a P106 (Q188); ran the agent check with an Ornith file made of them (Q189, file in
[storyline C](#5-storyline-c-model-files-and-quality)).

**Result.** One token, ps per weight (gate, down, dense): N2P 4.18 / 4.66 / 4.16, N3P 3.80 / 4.27 / 3.81, Q4_0's level.
Down products are 1.5–1.7× faster than the tuned IQ kernels. Agent check: **14.5 against 11.3 tok/s (1.28×)**, 12/12;
prompts ~16 % slower (untuned batched tiles, Q351).

**Lesson.** Changing the format beat tuning the kernels for the old one
([pascal_lowbit_format.md](research/notes/pascal_lowbit_format.md), [data](data/2026-10-27_lowbit/README.md)).

### 4.9 Kernels that read each weight once for 2–4 tokens (Q211, 10-28)

**Why.** Ornith's built-in draft layer is right 85–91 % of the time, yet v2.0 with it fell from 8.8 to 7.9 tok/s: the
Pascal kernels paid 2–3× per expert when checking 2–4 tokens.

**What I did.** A kernel that groups the tokens that picked the same expert, decodes each weight block once for all of
them, and stays bit-identical to the one-token kernel.

**Result.** Correct on the P106. At 3 tokens, cost against one token (the floor is ~2.6×, because 3 tokens touch ~21
distinct experts): N2P gate/up 2.60×, down 3.67× (2.89× with 4 row blocks), dense 1.57× (earlier kernel 2.72×). Worse
than llama.cpp's for Q8_0/Q5_0 experts (5.4–6.3× against 3.7–4.1×).

**Status.** In progress: per-type kernel choice, then the agent check with the draft layer, *est.* ~1.25–1.29×
([data](data/2026-10-28_multitoken_kernels/README.md)).

### 4.10 Do GPU page walks slow the kernels? (Q99, 10-28)

**Why.** Kernels ran 13–33 µs slower per call inside the model than in the benchmark. A card sweeps ~5 GB per token,
beyond the ~2 GiB its address-translation cache covers; *est.* +5–12 % if fixed.

**Result.** No: a pointer chase shows no knee from 16 MiB to 5 GiB, and spreading the weights over 5 GB moves MoE calls
by −0.7 to +1.7 µs against a 10 µs kill line. Closed ([data](data/2026-10-28_q99_tlb/README.md)).

### 4.11 Idle cards and the hand-off cost (10-24 to 10-28)

**Why.** Each card computes ~8.7 % of a nine-card token. What fills the rest?

**Result.** Hand-offs cost *est.* 0.55–0.65 ms per card boundary, about 17–18 % of a pass, which is also why six cards
decode one conversation about as fast as nine. That figure is inferred from six- and nine-card runs, not timed. The
weights never cross PCIe during decode; only ~8 KB per hop does, so this is latency and software, not bandwidth.

**Next.** Time a hop (Q81, 1–2 h), then hand-offs through mapped host memory without driver copies (Q270, *est.*
+8–15 %), the biggest engine lever left for Flint ([idle_cards.md](research/notes/idle_cards.md)).

### 4.12 The prompt path: why agent turns sometimes waited a minute (10-28)

**Why.** On the agent check the worst waits for a first token were 60–83 s, and re-reading the previous reply took
5.9–8.3 % of task time.

**What I did.** Compared token ids across turns; found the cause; built two server switches; checked them for
exactness and live.

**Result.** Flint sometimes writes a word in token ids the tokenizer would never choose (`file` + `type` where the
tokenizer writes `filetype`). When the agent sends the conversation back as text, it re-tokenizes differently and the
cache match breaks mid-reply: in 6.1 % of turns, and 44.9 % of replies over 4k tokens. The fix,
`LLAMA_SERVER_TEXT_REUSE=1`, keeps the cached ids while both spell the same bytes; `LLAMA_SERVER_TAIL_CKPT_MIN=4096`
spaces the end-of-prompt checkpoints. Live: no re-reads; worst first token ~12 s; prompt time per task 23.6–29.0 →
16.2 s; a short turn 684 → 379 ms. A decode-only card mask (Q261) was 0.4 % slower at agent-check contexts and stays
off.

**Status.** Both switches are planned for the v3 release line
([data](data/2026-10-28_reply_rereads/README.md)).

### 4.13 Tried and didn't pay

| Idea | What happened | Status |
|---|---|---|
| Decode-attention kernel (Q151) | −9 % short context; 15.0 vs 25.4 tok/s at 32k | built, off |
| Embedding row on card 1, packed uploads in plain decoding (Q150) | within noise | built, off |
| Card-built mask on every micro-batch (Q110) | prompts 3.6× slower | off |
| Decode-only card mask at agent contexts (Q261) | 0.4 % slower | off |
| 8 scheduler copies (Q157) | no gain | not shipped |
| Agent prompt cache (Q111) | 11 s fixed per task, no gain | off |
| Page-walk theory (Q99) | no cost | closed |
| PowerMizer mode 1 (Q141) | no change | dropped |
| Weight repacking into planes (Q39) | as fast as staging | dropped |
| Batched checking (Q19) | 3 tokens cost 2.2× one | parked |
| Fused N2P/N3P gate/up | 1.3× slower than two separate products | fix planned (Q350) |

## 5. Storyline C: model files and quality

Model files are made and scored on a separate 7-GPU quant machine (3 × RTX 3090, 2 × V100 32 GB, 2 × RTX 3080); only
speed is measured on the rig. Method: [quants/guide.md](../quants/guide.md); every file: the
[registry](../quants/experiments.md#registry).

### 5.1 How quality is judged, and why Unsloth's file is the bar (10-22/23)

**Why.** A faster file is worthless if it answers worse, and task benchmarks are too noisy to separate files cheaply:
on 300 tool-call tasks every file lands within ±4–5 points of BF16 (Q30), and two runs of one file disagree on 5–15 % of
tasks (Q41).

**What I did.** KLD and same-top-token against a reference on fixed texts: Flint against BF16 on wikitext-2
(80 × 512 tokens); later 8,192-token chunks of Flint's own agent conversations (Q106); Ornith against the makers' Q8_0
on three texts (wiki, held-out agent sessions, the agent check's code), because its BF16 doesn't fit the quant
machine.

**Result.** The noise floor (Q21): the same Flint file on two GPU types differs by KLD 0.065 and 91.3 % same top;
Unsloth's Q8_0 sits at 0.052 / 93.3 %. Only same-GPU comparisons count. Unsloth's UD-Q4_K_XL scores KLD 0.120 and
87.7 % same top against BF16, and it is what people already download.

**Decision (10-23).** A Flint file counts only near KLD ≲ 0.125 and same top ≳ 87.5 %: nobody gives up 9 points of token
agreement for speed.

### 5.2 Flint: from the makers' files to the v1 mix (10-22/23)

**Why.** Get UD's quality in formats that decode fast on Pascal (Q4_1, Q5_0, Q8_0).

**What I did.** Scored the makers' Q4_1, UD and single-format quants; mirrored UD's per-tensor bit budget in fast
formats (Q7); ranked tensor classes by KLD gained per ms of decode (Q18); built mixes S1–S4 and lowered upgrades one
8-layer group at a time (Q27, Q36, Q45).

**Result.** Against Q8_0: makers' Q4_1 78.6 % / 0.316; UD 87.6 % / 0.124; plain Q4_0 breaks Flint (PPL +45 %) because
of 7 tensors with dominant channels (Q24). The makers' Q4_1 turned out to be a plain round-to-nearest repack, not
quantization-aware trained (Q74). Rotations made Q4_x slightly worse (Q13). The winner, **S4-L0: KLD 0.123, 88.7 %,
32.5 GB**, +21 % chat speed over UD at 262k: the v1 file.

**Lesson.** Where the bits go matters more than the format: layers 0–24 carry Flint's sensitivity, and the output head
must stay at 8 bits ([quants/README.md](../quants/README.md), [recipes](../quants/recipes/)).

### 5.3 Re-pricing for Miner Llama's kernels, GPTQ, and the v2 file (10-23 to 10-26)

**Why.** With the fork's kernels decode time follows bytes: per weight Q5_0 now costs 1.1× Q4_1 and Q8_0 1.7× (stock
kernels 1.32× and 1.50×), so Q5_0 became cheap and every mix needed re-pricing.

**What I did.** New rounds from the best recipe; then each 8-bit group dropped to Q5_0 alone and its cost per MB saved
measured (Q77); GPTQ rounding for the output head (Q56) and three body groups (Q112); F16 routers (Q113).

**Result.**
- Group costs span 0.001 to 0.11 KLD per 100 MB saved and add up near the base; the cheap groups are late layers and
  early attention queries.
- A GPTQ-rounded Q5_0 head costs +0.002 KLD against round-to-nearest's +0.023, saving 0.116 GB per token.
- F16 routers save 49 MB per token at no measurable cost.
- **K2h-r16: KLD 0.105, 88.8 %, 29.4 GB, 2.09 GB per token**: +4.9 % over UD on `llama-bench`, and on the agent check
  53.8 tok/s against UD's 50.7. It ships in v2.0 as **Flint-Pascal-Q5**, for six cards or the most speed; Q8_0 on nine
  cards stays the headline ([models/flint](../models/flint/README.md)).

### 5.4 Does short-text KLD predict agent quality? (10-23/24)

**Why.** A 512-token wiki score might miss multi-turn agent behaviour.

**What I did.** Pre-registered non-inferiority tests on tool-call benchmarks (BFCL v4 single- and multi-turn) and
LiveCodeBench (Q41); KLD on 8k-token agent conversations (Q106); probes that spent bytes on attention and on UD's
tensors (Q107, Q127).

**Result.** On 3,477 single-turn tasks K2h is non-inferior to UD (+0.14 points, 90 % interval −0.87 to +1.16). At 8k
every candidate is 1.03–1.04× UD's KLD: tokens the model writes are better (0.89×), tokens it reads (tool schemas)
worse, and no tensor class explains the gap.

**Decision.** Short-text KLD no longer decides alone; written-token KLD on agent text and task checks gate a release
([long_context_quality.md](research/notes/long_context_quality.md)).

### 5.5 Near-Q8_0 Flint at fewer bytes: p065 (10-27/28)

**Why.** The headline is Q8_0; a file within noise of Q8_0 that reads fewer bytes would be pure speed.

**What I did.** I borrowed the allocator idea from the SLQ paper (measured per-group costs and a knapsack, not its
format; [slq.md](research/notes/slq.md)): starting from Q8_0 with F16 routers, 23 switches each moved one tensor class of
one group to Flint-Pascal-Q5's Q5_0 tensors; each was scored alone and a dynamic program chose the set per byte budget.

**Result.** p061: KLD 0.0611 at −7.8 % bytes; **p065: 0.0656, 91.78 % same top, −11.0 % bytes (40.4 GB)**, against
0.0580 for the Q8_0 starting point; predictions landed within 0.0011. On the rig, paired against Q8_0: **6.3 % faster
per generated token** (0.941, 95 % interval 0.930–0.980); live 12/12 at 47.1 tok/s.

**Status.** Ready; the release decision is next ([data](data/2026-10-27_flint_near_q8/README.md)).

### 5.6 Ornith-1.5-122B: from makers' files to my own formats (10-24 to 10-28)

**Why.** At 54 GB a 122 B model fits only at 2–3 bits per weight.

**What I did.**
1. Ran two community files: AtomicChat's AD-IQ3_XXS-IQ2_S (47.6 GB) and Unsloth's UD-Q2_K_XL (44.5 GB).
2. Built a Q8_0 reference (132.6 GB, all seven GPUs) with three bases.
3. Made files in the new N2P/N3P formats (section 4.8 of [storyline B](#4-storyline-b-the-engine-miner-llama))
   from the Q8_0 with Unsloth's importance matrix: file c (experts N2P/N3P, dense Q5_0, 47.58 GB), b (dense Q8_0,
   49.32 GB), d (formats swapped, a donor).
4. Allocated per 8-layer group: 18 switches, an exhaustive search over 2^18 sets, validated by a real file.

**Result.**
- UD beats AD on all three texts (wiki 0.184 against 0.233), and on v2.0 UD finishes 12/12 at 9.7 tok/s.
- A Q5_0 output layer costs nothing measurable and reads 524 MB instead of 626 (UD).
- File c beats AD at the same size but not UD: allocation matters more than format.
- **valid1** (the allocated mix, +0.91 GB) beats UD on all three texts; with the Q5_0 head (48.20 GB, Q212) by
  1.5–2.6 standard errors. On the agent check, **14.5 against 11.3 tok/s**.
- Ornith's sensitive layers are the *late* ones, the reverse of Flint, so costs must be measured per model.

Links: [Ornith page](benchmarks/2026-10-27-ornith-122b.md), [KLD data](data/2026-10-27_ornith_kld/README.md),
[allocation data](data/2026-10-27_ornith_alloc/README.md), [models/ornith-1.5-122b](../models/ornith-1.5-122b/README.md).

### 5.7 Ornith-1.5-35B and Qwen3.6-35B-A3B

**Ornith-35B (Q159).** Nine files from the BF16 against the makers' Q8_0 (wiki / agent / code). **P-Q5** (experts
Q5_0, everything else Q8_0, 25.4 GB): 0.028 / 0.029 / 0.012, better than Q6_K at 29.2 GB. PK-Q4-i (experts Q4_K with an
importance matrix, 21.3 GB): 0.040 / 0.036 / 0.019. Keeping the dense tensors at 8 bits is the lever: they are ~75 % of
a token's bytes. The rig check (Q311) is next ([models/ornith-1.5-35b](../models/ornith-1.5-35b/README.md)).

**Qwen3.6 (Q29–Q33, Q82).** My mixes reached KLD 0.020; Unsloth's UD-Q4_K_XL scores 0.0129 / 95.4 %, far better, so it
is the bar there too. Qwen3.6 is much less sensitive than Flint. Its rig matrix (Q148) is planned
([models/qwen3.6-35b-a3b](../models/qwen3.6-35b-a3b/README.md)).

### 5.8 Qwen3.8-Flash-Next and sub-1-bit ideas (research only, 10-27/28)

**Why.** Flash-Next (125 B, ~6 B active, plus a 28.8 GB n-gram table on disk) has a favourite community file at
54.8 GB; the cards hold ~47 GB of weights. Could sub-1-bit methods close the gap?

**Result.** The rig leaves 2.70–2.84 bits per expert weight. The best 1-bit post-training results raise perplexity
1.6–4× on dense models and none has run on an MoE. So: no sub-1-bit work. The plan instead scores the 47.0 GB IQ3_XXS
build, composes a per-group allocation across the published builds, and adds 128-weight N3P blocks for its 640-wide
rows. Ideas kept as parked rows: 1-bit cold experts, expert pruning, trellis and codebook formats (*est.* 9–28 ps per
weight here), IQ1 files (Q238–Q244) ([flash_next_variants.md](research/notes/flash_next_variants.md),
[sub_1_bit_quants.md](research/notes/sub_1_bit_quants.md)).

### 5.9 Tried and didn't pay

| Idea | Why it didn't pay |
|---|---|
| Hadamard rotations (Q13, Q304) | Flint's 8-bit activation blocks absorb the outliers; on 2–3-bit experts +9 % error |
| The makers' tensors in my files (Q44) | they only work inside their own file |
| Fewer experts per token (Q6) | +4–9 % speed for +12–23 % perplexity |
| Q4_1-tier speed files | below the quality bar |
| Spending bytes on the long-context gap (Q107, Q127) | moved nothing |
| A round-to-nearest Q5_0 head | too costly; GPTQ fixed it |
| SLQ's own formats | +27–53 % bytes against GGUF types at equal fidelity |
| Uniform grids, Q2_K/Q3_K at 2–3 bits (Q305) | +11–24 % error |
| An importance matrix for Q4_0 experts | no gain (it helps K-quants by 25–29 %) |

## 6. Storyline D: guessing ahead (speculation and drafters)

The idea from [section 2.5](#25-guessing-ahead-speculative-decoding): idle cards could check guessed tokens for free,
and acceptance decides how much that is worth. The living process document is [drafter.md](drafter.md).

### 6.1 Batched checking, and the first drafter plan (10-23)

**Why.** The classic recipe: a draft guesses, the model checks several tokens in one pass. Qwen3.6 ships an MTP draft
layer, and a trained head for Flint looked possible.

**Result.** On the rig, checking 2.90 tokens took 57.7 ms and 4.05 tokens 92.1 ms against 25.9 ms for one (2.22× and
3.55×; Q19); half of that was llama.cpp's stock multi-token MoE kernels. Qwen3.6's MTP gave only 1.04× with 82 % of
drafts accepted (Q43).

**Decision.** Batched checking is not worth it on Pascal until multi-token kernels exist; the drafter plan (Q20) waited.
That question came back for Ornith as Q211 ([4.9](#4-storyline-b-the-engine-miner-llama)).

### 6.2 Free guesses, run on the idle cards (10-23/24)

**Why.** Flint copies a lot: 28 % of its output sits in copies of ≥ 8 tokens (99 % of rewrites), and looking guesses up
in the context costs nothing. And guesses could run through the idle cards as one-token steps, so no batched check is
needed.

**What I did.** Replayed lookup guessers offline (Q64); built the **pipelined executor** (`--pipe-depth`, Q87): guesses
enter card 1 while later cards finish earlier tokens, the last card keeps a guess only if it equals its own sample, so
the output is bit-identical to plain decoding; added a table of continuation counts behind the copy lookup (Q96).

**Result.** Exactness: 89 runs bit-identical. Speed (`llama-bench`, best guesser): 1.30–1.55× across chat, code, edit
and rewrite on six and nine cards. Part of it is clock: with every guess wrong, nine cards still ran 1.24× faster,
because the extra work keeps them at boost (Q154). On the v2.0 agent check pipelining is +52 % (29.3 → 44.6 tok/s),
*est.* about half clock, half right guesses.

**Status.** In v2.0. **Lesson:** on Pascal pipelined beats batched, and the gain scales with acceptance
([pipelined_speculation.md](research/notes/pipelined_speculation.md)).

### 6.3 A trained guesser: the staircase drafter (10-24/25)

**Why.** Lookups only guess copied text. Heads that read the model's final hidden state allow only 1–2 guesses in
flight; a "staircase" head instead reads, for each token in flight, the deepest state it has reached so far. *Est.* p
0.60–0.72, 2.5–2.9× on nine cards, 1.6–2.6 ms per guess.

**What I did.** A 2-layer head trained on Flint's own answers with Q8_0 targets, exported as a side file, run in the
engine and served on the rig.

**Result.** Offline acceptance 0.42 → 0.46 with 3× more data (about +0.03 per doubling), 0.50–0.54 with later methods;
the gate (≥ 0.55 on fresh text) was not reached. Served: 50.4 tok/s with the head against 48.6 without (+3.7 %),
served acceptance 53.8 % exactly as predicted offline, but **one guess cost 5.3–7.3 ms** against the 1.6–2.6 ms
estimate. Then all training prompts so far were excluded because their licences and provenance weren't publishable,
so these numbers measure the method only.

**Lesson.** The offline score predicts served acceptance; guess cost was the limit
([drafter.md](drafter.md) §3, §6).

### 6.4 A publishable data set (10-26/27)

**What I did.** An agent loop in network-less sandboxes over licence-checked code (Python 3.12's standard library,
Elixir's), with Flint writing the sessions and every stage gated.

**Result.** **1,177 sessions**, 18,885 turns, 15.4 M generated tokens in one overnight run. It feeds the answer store
and the MTP heads; its narrow code variety later proved to be the heads' binding gap
([agentic_data_pipeline.md](research/notes/agentic_data_pipeline.md), [data](data/2026-10-26_agentic_data/README.md)).

### 6.5 The answer store (10-27/28)

**Why.** A host-side index of Flint's own past output, consulted after the context lookups, costs no card time.

**Result.** Offline: right guesses 56.9 → 59.6 %, *est.* +4.3 %. Paired replay on identical tokens: **+1.4 %** (22.87 →
22.55 ms per token, 95 % interval 0.4–1.9 %).

**Status.** It passed its pre-set rule; it ships as an optional download with one flag on the existing engine
([page](benchmarks/2026-10-27-answer-store.md)).

### 6.6 Flint's MTP head, start to verdict (10-27 to 10-29)

**Why.** A head trained on the agent data (method M4) could guess the fresh text lookups miss.

| Step | Result |
|---|---|
| First heads, 240 min of training (Q169) | held-out acceptance 0.277 / 0.256; on the agent check **29.7 / 30.7 tok/s against 45.0**: a 5.3 ms guess, and the head displaced continuations that were right more often |
| Cheaper guess (Q190–Q192) | 4.73 → **2.93 ms** per guess (−39 %) by moving the last layer and a Q8_0 output layer; a simulator says the head pays from ~0.13 acceptance at ≤ 3 ms (*est.*) |
| Paired replay, 1,000-min head (Q193) | right **24.8 %** as served (as predicted), yet **3.2 % slower**: a guess costs more inside the server than on an idle card |
| Where the cost goes (Q194) | following the conversation costs as much as guessing, and 2 of 3 head launches are wasted; with every removable cost gone the ceiling is *est.* +5.2 % |
| Three RTX 3090s (Q310) | still slower: ×1.031 and ×1.118. The cost is not Pascal-specific |
| Round 3, 1,480 min (Q169) | held-out 0.380 / 0.367; gains slowed to +0.027 per doubling of training |
| Diagnostics (Q384, Q385) | 0.384 held-out → 0.309 on the agent check's text → 0.248 served: 0.04–0.06 lost in serving (Q195); right only 0.52 where Flint is > 0.99 sure; acceptance falls from 0.62 under 4k tokens to 0.36–0.39 from 16k |
| Two training changes (Q371, Q372) | +0.012 and +0.001: both below their kill lines |

**Verdict.** The rule set beforehand: continue only with a credible path to ≥ +5 % on the P106 release line within a
week of engine work. The serving cuts' ceiling is +5.2 %, no margin; more training gives ~0.01; the head slows fast
GPUs too. **The head is closing.** The one open lever is the serving gap (Q195), engine work. The research behind it:
the head is limited by learning, not inputs; it saw ~4.7×10⁷ token-passes against the ~10⁹ of published heads
([mtp_head_research.md](research/notes/mtp_head_research.md), [data](data/2026-10-27_mtp_head/README.md),
[cheap-guess data](data/2026-10-27_cheap_guess/README.md)).

### 6.7 Ornith's built-in draft layer (10-27/28)

**Why.** Ornith ships an MTP layer trained together with the model.

**Result.**

| File (layer type) | First draft right, agent text | Code |
|---|---|---|
| 35B Q8_0 | **0.876** | **0.908** |
| 122B UD-Q2_K_XL (Q8_0) | **0.845** | **0.865** |
| 122B AD (Q4_K) | 0.799 | 0.831 |

- Chained drafts on the 122B: 0.845 / 0.738 / 0.666 for drafts 1/2/3 (Q366).
- On v2.0 with batched checking it lost speed (7.9 against 8.8 tok/s), hence Q211.
- Cheaper drafts (Q396): a Q5_0 or Q4_0 head costs ≤ 0.1 points; a 64k-token shortlist costs 0.1–0.6 points; the
  122B's draft block itself is sensitive to quantization. Best design: the block as shipped plus a shortlist head,
  *est.* 7.6 → 3.7 ms per draft on the 122B.
- valid1's own draft layer is as good as UD's at half the bytes (0.843 / 0.740 / 0.656; Q397).

**Next.** Q211's agent check with the draft layer (*est.* ~1.25×), then **R8**, pipelined drafting for this hybrid model
(it needs a way to roll back its recurrent state), *est.* 2.4–2.9×, about 35–41 tok/s, from 1–2 weeks of engine work
([cheap_mtp_draft.md](research/notes/cheap_mtp_draft.md), [data](data/2026-10-28_ornith_mtp_accept/README.md)).

### 6.8 Why a draft trained with the model wins

Flint's add-on head and a joint-trained layer see the same inputs, yet Ornith's layer is right 0.85–0.91 of the time and
Flint's head 0.25–0.37. A joint layer trains on the whole pretraining corpus; the add-on saw 92 code modules and
memorizes them. A draft layer is not a small model of its own: it predicts from the full model's hidden state, so it
can never replace the model, only feed it cheap guesses. On these cards that difference in acceptance is the
difference between a slowdown and a 2–3× speed-up.

## 7. Storyline E: beyond one rig, and other people's claims

### 7.1 Several rigs, one model (10-28, research only)

**Why.** Thousands of these rigs exist; could 2–5 of them run one bigger model together over Ethernet?

**Result (all *est.*).** Gigabit carries a layer split's 8–16 KiB hidden state in 0.2–0.4 ms per crossing, about one
card hand-off, so faster network cards gain nothing; the layer split itself is the limit. Rigs buy capacity and
quality, not single-stream speed: Ornith at 4–8 bits on 2–3 rigs *est.* 8–13 tok/s, 200–400 B MoEs on 4–5 rigs 4–9
tok/s. Flint should run one copy per rig. Tensor and expert parallelism add 14–72 ms per token and are rejected. A
loopback test on one rig (Q320) is the first step ([multi_rig.md](research/notes/multi_rig.md)).

### 7.2 Checking a "2× faster" claim (10-28)

**Why.** A project seen online claimed decoding "over 2× faster than Ollama".

**What I did.** Read its source and benchmark method only; nothing downloaded or run.

**Result.** The workload recites 64k tokens of a novel with block drafters on copied text and a smaller KV cache; the
engine alone is 1.18× llama.cpp, and *est.* ~2.1× of the headline is the drafter on copied text. It needs compute 8.0+,
so it can't run here. Not a scam, but the headline overstates general speed. Three ideas from it became registry rows
(Q414–Q416) ([magnitude.md](research/notes/magnitude.md)).

## 8. Where each model stands

**Flint** (agent check, nine cards unless noted):

| Configuration | Generation | Notes |
|---|---|---|
| Q8_0, stock llama.cpp (64k context) | 14.4 tok/s | 8 of 12 tasks |
| Q8_0, v1 | 25.7 tok/s | 11 of 12 |
| Q8_0, v2.0 (262k) | 44.6–45.9 tok/s | 12 of 12, the headline |
| Flint-Pascal-Q5, v2.0 | 53.8 tok/s | 12 of 12, for six cards or the most speed |
| v3 pieces, each measured alone | p065 6.3 % faster per token than Q8_0 (live 47.1 tok/s); answer store +1.4 %; re-read fix: prompt time 23.6–29.0 → 16.2 s per task, worst first token 60–83 → ~12 s | not yet measured together |

**Ornith-1.5-122B** (agent check, UD-Q2_K_XL unless noted): stock 6.2 → v2.0 9.7 → v3 IQ kernels 11.3 → **valid1 14.5
tok/s** (2.3× stock, closer to Q8_0 than UD). Next: Q211 with the draft layer (*est.* ~18 tok/s), then R8 (*est.*
35–41).

**Ornith-1.5-35B:** files made and scored (P-Q5 best); the rig check is next (Q311).

**Qwen3.6-35B-A3B:** v1-era numbers only (Q4_0 29.0 tok/s, `llama-bench`); the rig matrix is planned (Q148).

**Qwen3.8-Flash-Next:** research and a plan; no file on the rig yet.

## 9. What's next

In rough order; each has a registry row with its kill criterion.

1. **Ornith with its built-in draft:** Q211's agent check (~1.5 h of rig time), then R8, pipelined drafting for hybrid
   models (1–2 weeks of engine work).
2. **Flint's hand-off cost:** time a hop (Q81), then mapped-memory hand-offs (Q270, *est.* +8–15 %).
3. **Shorter reasoning for Flint** (Q267, Q268): the largest lever per task in the Flint plan, *est.* 1.1–1.3×
   ([flint_v3_plan.md](research/notes/flint_v3_plan.md)).
4. **The v3 release:** combine p065, the answer store, the re-read switches, valid1 and the IQ kernels, and measure them
   together.
5. Smaller items: Ornith-35B on the rig (Q311), short-prompt fixed costs (Q264), the memory-clock offset (Q133), the
   serving gap of the MTP head (Q195), Vulkan.

## 10. The lessons that cut across everything

1. **The lever is software, not hardware.** Real kernels use only 42–54 % of a card's bandwidth and pay ~10 µs per
   launch; kernels, CUDA graphs and fewer launches carried most of v1's and v2's gains. Hardware settings give little.
2. **A 5-SM Pascal card is limited by instructions and latency, not bytes, at low bit widths.** Below ~3.5 bits, fewer
   bits buy memory, not time; formats that decode cheaply (Q5_0, N2P/N3P) beat formats that are smaller.
3. **Where the bits go matters more than the format.** Measured per-group costs plus a knapsack predict a composed file
   within 0.001–0.006 KLD; the sensitive layers differ per model (early in Flint, late in Ornith).
4. **Idle cards are the rig's biggest resource.** One card works at a time; idle cards also drop to base clock, so
   filling them pays twice. Hand-offs between cards are the next cost to measure.
5. **Acceptance is the currency of speculation.** At ~25 % no engine cleverness pays; at 85 % the pipeline becomes the
   lever. A draft trained with the model beats an add-on head by ~3×.
6. **Measure a cost where it runs.** A guess was 1.6–2.6 ms estimated, 5.3 ms measured, 2.93 ms on an idle card and
   more inside the server. Model-level timing overturned several good-looking estimates.
7. **Measure what the user feels.** Agent sessions differ from chat and `llama-bench`; the release is chosen on the
   agent check.
8. **Pair the comparison and set the rule first.** Live runs can't resolve a few percent; paired replays on identical
   tokens can, and an estimate of +4.3 % became a measured +1.4 %.
9. **Control the state.** Warm-up, clock state, run order and a tool's own memory use all produced fake effects.
10. **The 4 GB host sets the flags.** `-ub 128`, `--cache-ram 0`, few checkpoints and the card split by memory are
    what keep the server alive at 262k.
11. **Provenance can reset work.** Excluding unpublishable training data turned a week of drafter numbers into method
    measurements only; every data set since is licence-checked first.

## 11. Glossary

The terms used above; the repository's full list is [GLOSSARY.md](../GLOSSARY.md).

| Term | Meaning |
|---|---|
| Acceptance (p) | the chance a guessed token equals the model's own choice, given the earlier guesses were right |
| Agent check | the release benchmark: a coding agent fixing 12 real bugs against llama-server, timed per task |
| Allocation (knapsack) | choosing a format per tensor class and layer group for the least predicted KLD at a byte budget |
| α (in drafter.md) | the sharpening exponent applied to a head's distribution (α 3 as served); in formulas, also acceptance |
| Base / boost clock | 1,354 MHz when a card idles / up to ~1,911 MHz when busy |
| Batched verify | checking several drafted tokens in one model pass |
| BPW | bits per weight, scales included |
| CUDA graph | a recorded sequence of kernels replayed with one launch |
| Dense tensors | everything except routed experts (attention, shared expert, output head, router), read every token |
| Draft layer / MTP | a small extra layer that predicts the next token from the model's hidden state |
| ECC | error-correcting memory; these cards have none, so an overclock is proven by bit-exact output |
| GPTQ | rounding that compensates each weight's error in the remaining weights using second-order statistics |
| Hand-off | passing the hidden state from one card to the next through host memory |
| Hybrid model | most layers use a recurrent linear attention with a fixed-size state; a few use full attention |
| imatrix | per-channel activation statistics from a calibration text, used to weight rounding error |
| IQ formats (IQ2_S, IQ3_XXS, …) | codebook formats, best per byte at 2–4 bits, slow to decode on Pascal |
| K-quants (Q4_K, Q6_K, …) | 256-weight super-blocks with sub-block scales; good per byte, slower to unpack on 5 SMs |
| KLD | KL divergence: how far a file's next-token distribution drifts from the reference's |
| KV cache | stored attention keys and values for the context so far |
| Layer split (`-ts`) | each card holds a slice of consecutive layers; a token passes card to card |
| MMQ / MMVQ | llama.cpp's quantized matrix-matrix (prompt batches) / matrix-vector (one token) kernels |
| MoE | mixture of experts: a router picks a few expert networks per token |
| N2P / N3P | Miner Llama's 2.69- and 3.25-bit expert formats, decoded in registers |
| Noise floor | the KLD between two runs of the same file on different GPU kernels (~0.065 for Flint) |
| Non-canonical token split | the same text in token ids the tokenizer would not choose |
| Paired replay | recorded agent requests re-sent to two setups that then generate identical tokens |
| Pipelined speculation (`--pipe-depth`) | guesses run through idle cards as one-token steps; output identical to plain decoding |
| Prefill / decode | reading the prompt / generating one token at a time |
| ps per weight | kernel time per weight on one card; the speed unit for low-bit formats |
| Q4_0 / Q4_1 / Q5_0 / Q8_0 | simple 32-weight blocks with one scale (plus an offset for Q4_1); fast on Pascal |
| Same top token | how often a file's most likely next token matches the reference's |
| SM | streaming multiprocessor, a GPU compute unit; a P106-090 has 5 |
| Staircase head | a drafter that reads, for each token in flight, the deepest model state it has reached |
| Σ min(p, q) | the overlap of the model's and the draft's distributions; the most a guess can be accepted |
| `-ub` | the prompt micro-batch size; its host buffer grows with the context |
| UD-Q4_K_XL | Unsloth's dynamic 4-bit mix, the quality bar for Flint |

