# 9. The models: each one's journey

> **In short**
> - Four large models ran on the rig. Flint was the flagship of v1 and v2: near-lossless Q8_0 on nine cards with its
>   full 262,144-token context, 44.6–45.9 tok/s for a coding agent, three times stock llama.cpp.
> - Qwen3.6-35B-A3B showed early that a mixture-of-experts model runs fast here. Ornith-1.5-122B showed that a 122B model
>   fits at 2–3 bits, and reached 14.5 tok/s.
> - Since 2026-11-02 there is one goal: Qwen3.8 Flash-Next, a 125B-parameter model, well above 5–8 tok/s on one rig. On
>   2026-11-04 it stands at 12.2 tok/s with a draft network.
> - Across all four, the engine carried most of the speed, the file decided the quality, and the 4 GB host decided what
>   could load at all. Facts as of 2026-11-04.

The earlier chapters explain the tools; this one follows each model through them. For quantization and file mixes see
[Chapter 4](04-smaller-models.md), for the card split and the host memory [Chapter 5](05-nine-cards.md), for the
kernels [Chapter 6](06-gpu-code.md), for experts [Chapter 7](07-experts.md) and for drafting
[Chapter 8](08-guessing-ahead.md).

## 9.1 The four models at a glance

| Model | Parameters, total / per token | Kind | File on the rig | Best coding-agent speed | Where it stands |
|---|---|---|---|---|---|
| Flint | 39.8 B / 2.5 B | mixture of experts | Q8_0, 42.3 GB | 44.6–45.9 tok/s (Q8_0), 53.8 (own Q5 file) | released in v2.0 |
| Qwen3.6-35B-A3B | 36 B / 3 B | hybrid mixture of experts | Q4_0, 20.8 GB | no agent check; 29.0 tok/s on one request | waiting |
| Ornith-1.5-122B | 122 B / 10 B | hybrid mixture of experts | own 2–3-bit file, 48.2 GB | 14.5 tok/s | paused since 2026-10-28 |
| Qwen3.8 Flash-Next | 125 B / 6 B (plus a 51 B n-gram table and a 4 B MTP block: 180 B in the weights) | hybrid mixture of experts with an n-gram table | Swift Q2_0, 37.7 GB on the cards | 12.2 tok/s | the v3 goal |

"Per token" is what matters for speed: the cards read only that part of the weights for each token
([Chapter 2](02-how-an-llm-makes-text.md)). A *hybrid* model mixes attention layers, which cache every past token,
with recurrent layers that keep a fixed-size running state. Speeds are from the agent check, a coding agent fixing
twelve small bugs with real tool calls ([Chapter 3](03-measuring.md)), unless marked otherwise.

## 9.2 Flint: the flagship

**What it is.** Flint is a 39.8B mixture-of-experts model that uses 2.5B parameters per token and keeps most of its
attention local. It reads up to 262,144 tokens of context and reasons before every answer, so replies run 600–7,900
tokens. Its makers released it under Apache-2.0 on 2026-10-20. Two days later it was the fastest model I had tried on
the rig, and it became the model every engine change was built and measured on.

**What fit.** With little read per token and 54 GB of memory across the cards, almost every file fit. The question was
which one to run:

| File | Size | Quality: KLD against BF16 (lower is better) | Same top token | Agent check, v2 engine |
|---|---|---:|---:|---|
| The makers' Q4_1 | 26.0 GB | 0.322 | 78.4 % | not run |
| Plain Q4_0 | 22.6 GB | 0.565 | 71.7 % | broken, not run |
| Unsloth UD-Q4_K_XL, the quality bar | 25.1 GB | 0.120 | 87.7 % | 50.7 tok/s, 10 passed |
| Flint-Pascal-Q5, my own file | 29.4 GB | 0.105 | 88.8 % | **53.8 tok/s, 12 of 12** |
| Unsloth Q8_0 | 42.3 GB | 0.052 | 93.3 % | **44.6–45.9 tok/s, 12 of 12** |

KLD is a diff score between the file's predictions and the full-precision model's ([Chapter 3](03-measuring.md)).
Plain Q4_0 broke Flint: seven tensors carry a few dominant channels that 4 bits can't hold, and perplexity rose by
45 %. The makers' fast Q4_1 agreed with the original only 78 % of the time. On 2026-10-23 I set Unsloth's UD-Q4_K_XL as
the quality bar: people use Unsloth's files, and nobody trades nine points of agreement for speed. My own files had to
match it or beat it ([Chapter 4](04-smaller-models.md)).

**The speed journey.** The same agent check, Q8_0 on nine cards:

| Engine | Context | Generation | Tasks passed | First token per turn |
|---|---|---:|---|---:|
| Stock llama.cpp | 65,536 | 14.4 tok/s | 8 of 12 | 3.7 s |
| Miner Llama v1 (2026-10-23) | 65,536 | 25.7 tok/s | 11 of 12 | 4.0 s |
| **Miner Llama v2 (2026-10-26)** | **262,144** | **44.6–45.9 tok/s** | **12 of 12** | **2.6 s** |

v1 made a P106 decode Flint well at all. It turned on CUDA graphs, which llama.cpp switches off for cards this old,
gave one-token products a launch shape for 5-SM cards, and added Pascal matrix-vector kernels. On a single-request
benchmark those three steps took 17.6 tok/s to 19.5, 24.5 and 34.1. v2 added 32-bit sums in those kernels (+9.6 %),
one persistent kernel per expert layer (+5.6–8.8 %), faster prompt reading (+22–29 %), tool requests sampled on the
GPU (+13–15 %) and pipelined speculation, the biggest single step (+52 %, [Chapter 8](08-guessing-ahead.md)).

**The key lesson.** The engine carries the speed. In single-request tests on nine cards, going from stock to v2
(16.5 to 36.5 tok/s) owed at least 95 % to the engine and only 0–3 % to the choice of file. And a 5-SM card spends
much of a token's time unpacking compressed weights ([Chapter 1](01-the-rig.md)). The engine decodes the simple 8-bit
format at close to memory speed, while the elaborate 4-bit formats need more instructions per byte. So Q8_0, which
reads 2.70 GB per token against UD-Q4_K_XL's 2.21 GB, runs within about 15 % of the 4-bit files. The release headline
is the near-lossless file, not the smallest one.

**After v2.** Three pieces for Flint were measured one at a time, each against the release line:

- a near-Q8_0 file that reads 11 % fewer bytes per token at almost Q8_0's quality (KLD 0.066 against 0.063 on the
  same machine): 6.3 % faster per token, 47.1 tok/s and 12 of 12 on the agent check;
- the answer store, +1.4 % ([Chapter 8](08-guessing-ahead.md));
- a fix for replies the server read twice: prompt time fell from 24–29 s to 16 s per task, and the worst wait for a
  first token from 60–83 s to about 13 s.

My own drafter for Flint did not pay and was stopped on 2026-10-29 ([Chapter 8](08-guessing-ahead.md)). The v2.1
release that was to carry it is not released.

**What is open.** The three pieces have not been measured together. Shorter reasoning is the largest lever per task:
about 65 % of what Flint writes in an agent session is reasoning, and a steering vector that nudges it towards brevity
is estimated at 1.1–1.3× per task. Cheaper hand-offs between cards are estimated at +8–15 %. Both are planned; no Flint
work has run since the lead moved to Flash-Next on 2026-11-02.

## 9.3 Qwen3.6-35B-A3B: the first proof

**What it is.** Qwen's 36B hybrid mixture-of-experts model uses about 3B parameters per token. 30 of its 40 layers are
recurrent (Gated DeltaNet), 10 are full attention. It reads 262,144 tokens of context, ships a built-in MTP draft
layer, and is under Apache-2.0.

**What fit.** Easily. Its full context takes 27 GB, so it fits on four or five cards, and two copies fit side by side:
one on cards 0–4 with 262,144 tokens of context, one on cards 5–8 with 131,072. Each kept about 16 tok/s while the
other was busy, 31–35 tok/s in total.

**What happened.** These are single-request numbers from the first days; it never got an agent check.

| Setup | Generation |
|---|---:|
| Stock llama.cpp, Q4_0 | 18.5 tok/s (14.0 at 32k of context, 8.1 at 64k) |
| Stock, real chat with its MTP draft and GPU sampling | 27.7 tok/s |
| Miner Llama v1, Q4_0, four cards | 29.0 tok/s |
| v1, five cards, plain against MTP | 31.1 against 32.4 tok/s (1.04×) |

Quality was no problem: Unsloth's UD-Q4_K_XL (22.4 GB) scores a KLD of 0.013 against Q8_0, and the common Q4_0 0.036.

**The key lesson.** It was the first proof that a 2026 mixture-of-experts model runs at memory-reading speed on this
rig, with its full context: 4.7 times faster than the dense Qwen3.8-27B. It also gave the first warning about drafting.
Its MTP draft was right 82 % of the time and still gained only 1.04×, because my fast kernels were one-token kernels and
the checks ran on stock multi-token code ([Chapter 8](08-guessing-ahead.md), section 8.2).

**What is open.** The model page says "coming soon". A full matrix of engines, files and settings on the rig is
planned. Flint was faster, so Qwen3.6 waited.

## 9.4 Ornith-1.5-122B: a big model at 2–3 bits

**What it is.** A fine-tune of Qwen3.5-122B-A10B for terminal coding agents, under the MIT licence. It has 122B
parameters and uses about 10B per token, four times as many as Flint. It is a hybrid (36 recurrent and 12
attention layers) with 256 experts per layer, of which 8 are used plus one shared, and it ships an MTP draft layer.

**What fit.** Its 8-bit file is 132.6 GB, two and a half times the rig's 54 GB. Only 2–3-bit files fit: Unsloth's
UD-Q2_K_XL (44.5 GB), and later my own file in two new 2–3-bit formats built for these cards (48.2 GB). On the cards it
took 49,152 tokens of context at the default micro-batch.

**What happened.** The agent check climbed in four steps (2026-10-27 and 10-28):

| Engine and file | Generation | Tasks passed |
|---|---:|---|
| Stock llama.cpp, UD-Q2_K_XL | 6.2 tok/s | 8 of 9 done |
| Miner Llama v2.0, UD-Q2_K_XL | 9.7 tok/s | 12 of 12 |
| v2.0 plus Pascal kernels for its 2–3-bit formats | 11.3 tok/s | 12 of 12 |
| **My own 2–3-bit file on those kernels** | **14.5 tok/s** | **12 of 12** |

That is 2.3× stock. My file was also better than Unsloth's on all three quality texts (KLD 0.176 against 0.184 on
wiki text, 0.052 against 0.054 on agent sessions, 0.062 against 0.071 on code), for 3.7 GB more.

Its MTP draft layer was right 0.85 of the time on agent text for the first guess, 0.74 and 0.67 for the second and
third. But on v2.0, with checks run on stock multi-token code, the draft made it slower: 7.9 against 8.8 tok/s.

**The key lesson.** Where the bits go beats which format holds them. Ornith's sensitive layers are the late ones, the
reverse of Flint, so a mix copied from Flint would have spent bits in the wrong place ([Chapter 4](04-smaller-models.md)).
And a draft trained with the model guesses well; it only pays once checking several tokens is cheap.

**What is open.** Kernels that read each weight once for 2–4 tokens were built and checked on a P106. The agent check
with the draft layer on them is estimated at about 18.7 tok/s and has not run. Pipelined drafting for hybrid models
was estimated at 35–41 tok/s; Flash-Next's 1.6× slowdown with pipelined guesses ([Chapter 8](08-guessing-ahead.md))
now argues for caution. Nothing has run on Ornith since 2026-10-28.

**Its smaller sibling, Ornith-1.5-35B,** has Qwen3.6's shapes. My best file for it (25.4 GB) scores a KLD of 0.028 on
wiki text, and its draft layer is right 0.88–0.91 of the time. It has not run on the rig yet.

## 9.5 Qwen3.8 Flash-Next: the single goal

**What it is.** Qwen released Flash-Next on 2026-08-24. It is a 125B mixture-of-experts model that uses about 6B
parameters per token, plus a 51B-parameter n-gram table and a 4B multi-token-prediction block: 180B parameters in all. The table is a lookup of embeddings
for hashed sequences of two and three tokens. Each token reads only a few of its rows, like a huge read-only hash table
that lives on disk. The model has 48 layers (36 recurrent, 12 attention with a sparse-attention index that picks the
top 2,048 past tokens), 512 experts per layer with 10 used plus one shared, and 262,144 tokens of context.

Since 2026-11-02 it is the project's one goal: the best-quality Flash-Next on one rig, well above the 5–8 tok/s of the
first run.

**What fit.** The rig runs UkisAI's Swift-1.5, a fine-tune of Flash-Next that thinks less, quantized by its makers to
2 bits (Q2_0). It is a 66.5 GB download. I split it into three parts with the bytes unchanged:

```
           nine cards (54 GB)                      SSD (64 GB)                 host (4 GB RAM)
  ┌──────────────────────────────────┐     ┌──────────────────────┐      ┌──────────────────┐
  │ transformer: 37.7 GB over 9 cards│ ◄── │ n-gram table 28.8 GB │      │ the engine, its  │
  │ layers, experts, KV cache        │     │ memory-mapped,       │      │ buffers, the OS  │
  └──────────────────────────────────┘     │ 11–21 rows per token │      └──────────────────┘
                                           └──────────────────────┘
```

The table's reads cost about 1 % of a token's time from the internal SSD; from a USB hard disk they would cost far
more. The host was the hard part: loading needed staged direct reads instead of memory mapping, and a kernel boot
option (`kho=off`) to free memory the GPU driver couldn't use ([Chapter 5](05-nine-cards.md)).

**Quality.** I scored eight files against Swift-1.5's own 8-bit version on agent sessions (KLD, lower is better; the
noise floor is 0.03):

| File | KLD, agent text | Same top token | Fits nine cards? |
|---|---:|---:|---|
| IQ3_S (two makers) | 0.117–0.121 | 93.4–93.7 % | no, at any context |
| **Swift IQ3_XXS** | **0.176** | 92.3 % | yes, the best that fits |
| Swift IQ2_XS | 0.225 | 91.5 % | yes |
| **Swift Q2_0, the rig's file today** | **0.285** | 90.7 % | yes, with the most room |

So the file that runs today is the roughest of the ones that fit. Better quality costs speed: IQ3_XXS is estimated at
about 10 tok/s plain against Q2_0's 12.

**What happened: the speed ladder.** The agent check, step by step (40 minutes per arm, so some arms started fewer
than twelve tasks):

| Step | Date | Generation | Tasks passed |
|---|---|---:|---|
| Stock llama.cpp | 2026-11-03 | 6.4 tok/s | 7 of 8 |
| Miner Llama v3's base (the v2.0 code) | 2026-11-03 | 7.8 tok/s | 9 of 9 |
| + pipelined guesses on the idle cards | 2026-11-03 | 4.9 tok/s (not taken) | 8 of 8 |
| + hyper-connections and router retyped to faster formats | 2026-11-03 | 9.0 tok/s | 12 of 12 |
| + Gyro MTP draft, base engine | 2026-11-03 | 8.8 tok/s | 10 of 11 |
| + DFlash2 block drafter, base engine | 2026-11-03 | 9.5 tok/s | 11 of 11 |
| **MTP draft on a newer build, after the memory fix** | **2026-11-04** | **12.2 tok/s** | **11 of 12** |

The last row mixes two changes, a newer engine and the draft; a same-build run without the draft will split them. The
reasons behind the drafting rows are in [Chapter 8](08-guessing-ahead.md).

**The key lesson.** On Flash-Next the limit is no longer reading weights. A token reads about 4.2 GB, and the cards
could do that in 28.7 ms, a floor of about 35 tok/s. A one-token pass on today's best build takes 75–82 ms, about
three times the floor.
The time goes to many small dense products, the hops between cards, the 4 GB host and, for drafts, the steep cost of
checking several tokens at once.

**Where it stands on 2026-11-04.** Several lines of work are out, none of them on the rig's line yet:

- **Pascal mixes:** the makers' experts kept byte for byte, every other weight moved to the format a P106 decodes
  fastest. The speed file's provisional KLD is 0.279 against Q2_0's 0.287 on the same cards, at an estimated 82 to
  72 ms per token.
- **Fewer experts:** the IQ3_S files cut from 512 to 464, 448 or 416 experts per layer, keeping the most used, so that
  the better format fits (50.2, 48.6 and 45.5 GB). A first CPU check put their KLD against the uncut file at
  0.04–0.06; the GPU scoring will decide.
- **The stacked run:** the speed file, the newest kernels and their best setting, without and with the MTP draft,
  estimated at about 15 tok/s before any draft.
- **Two rigs:** 18 cards over the network for a larger file, planned, estimated at 5.5–6.3 tok/s plain.

## 9.6 The releases

| Release | Date | What it held | Headline |
|---|---|---|---|
| **v1** | 2026-10-23 | CUDA graphs on Pascal, a launch shape for 5-SM cards, Pascal matrix-vector kernels for six formats, a profiler that works on mining cards | Flint chat 28.0 against 14.2 tok/s on stock; agent check 25.7 against 14.4 |
| **v2.0** | 2026-10-26 | 32-bit sums, a persistent kernel per expert layer, the fused shared expert, faster prompt reading, GPU sampling for tool requests, pipelined speculation; my Flint-Pascal-Q5 file | Flint Q8_0 at 262,144 tokens: 44.6–45.9 tok/s, 12 of 12 |
| **v2.1** | not released | was to add the answer store and a trained drafter | the drafter was stopped; the store measured +1.4 % |
| **v3** | TODO(tag) | Flash-Next first; also Flint's near-Q8_0 file and re-read fix, Ornith's own file and kernels; every feature behind a switch, a preset per model ([chapter 10](10-v3.md)) | Flash-Next 20.40 tok/s for a coding agent; Flint 45.9 and Ornith 16.3 on the same build |

Every release keeps the answers the same: the kernels change only the order of additions, and the speculation keeps
exactly the tokens plain decoding would produce. Each one ships with the command line and the measurements behind its
headline.

## What to remember

- Pick models by what they read per token. Flint reads 2.5B parameters per token and runs at 45 tok/s; Ornith reads
  10B and runs at 14.5.
- The engine carries the speed, the file sets the quality, and the 4 GB host decides what loads at all.
- On these cards a near-lossless 8-bit file can be the headline: it was within about 15 % of the 4-bit files.
- A big model fits only at 2–3 bits. There, where the bits go matters more than the format, and my own file beat the
  common one on both speed and quality.
- Flash-Next went from 7.8 to 12.2 tok/s in two days, but its floor is about 35 tok/s; most of the distance is still
  ahead.

## Where the details are

- Flint: [model page](../../models/flint/README.md), [release/flint-v2](../../release/flint-v2/README.md) and its
  [results](../../release/flint-v2/results.md), [benchmark](../benchmarks/2026-10-22-flint.md),
  [chat benchmark](../benchmarks/2026-10-23-flint-chat.md), [v2 synthesis](../research/notes/v2_synthesis.md) (the
  engine's share of the speed-up, decision log of 2026-10-24), [Flint v3 plan](../research/notes/flint_v3_plan.md);
  registry rows Q2, Q36, Q74, Q117, Q168 (near-Q8_0 file), Q260 (re-read fix), Q267, Q268, Q81, Q270. Sources give
  44.6–46.5 tok/s for v2 when a later run on 2026-10-27 is counted; the release results table gives 44.6–45.9.
- Qwen3.6-35B-A3B: [model page](../../models/qwen3.6-35b-a3b/README.md),
  [baselines](../benchmarks/2026-10-22-qwen-baselines.md), [speed-ups](../benchmarks/2026-10-22-speedups.md),
  [overview](../overview.md); Q38, Q43, Q82, Q148.
- Ornith-1.5-122B: [model page](../../models/ornith-1.5-122b/README.md), [benchmark](../benchmarks/2026-10-27-ornith-122b.md),
  [low-bit rig run](../data/2026-10-28_lowbit_rig/README.md), [draft acceptance](../data/2026-10-28_ornith_mtp_accept/README.md),
  [cheap_mtp_draft.md](../research/notes/cheap_mtp_draft.md); Q164, Q166, Q167, Q170, Q189, Q211, Q212, Q341, Q366.
  Ornith-1.5-35B: [model page](../../models/ornith-1.5-35b/README.md); Q159, Q311.
- Flash-Next: [model page](../../models/qwen3.8-flash-next/README.md), [release/flash-next-v3](../../release/flash-next-v3/README.md),
  [file choice](../data/2026-11-04_file_choice/README.md) (Q811), [retype](../data/2026-11-03_retype_agent/README.md)
  (Q218, Q551), [n-gram table reads](../data/2026-11-03_ngram_replay/README.md) (Q228), [the 12.2 tok/s run](../data/2026-11-04_mtp_redo/README.md)
  (Q558, Q692, Q841), [floor and engine](../research/notes/flash_next_floor_and_engine.md),
  [two rigs](../data/2026-11-04_two_rigs/README.md) (Q505, Q816); the baseline is Q215, the Pascal mixes Q880–Q885,
  the expert cuts Q890–Q891 (fork tool `scripts/gguf_prune.py`), the stacked run Q899.
- Releases: [v1 notes](../../release/notes-v1.0.md), [v2.0 notes](../../release/notes-v2.0.md) (fork tag
  `miner-llama-v2.0`, engine `37e717ff6`; v1 was `eeaa59660`), [v2.1](../../release/flint-v2.1/README.md).
