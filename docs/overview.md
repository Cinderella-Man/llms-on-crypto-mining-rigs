# Overview: why a mining rig, first results, and what they taught

Background for the [README](../README.md): why a retired GPU mining rig suits large language models at all, the first
round of measurements (2026-10-22, stock and lightly patched llama.cpp), what those measurements taught, and the terms
used throughout this repository. Current results are in the README and on each model's page under
[models/](../models/).

## Why a mining rig could work for LLMs

**Text generation is mostly a memory-reading problem.** An LLM produces text one *token* at a time (a token is roughly
three quarters of an English word). For every token, the GPU must read each weight of the model that takes part in
that token from its memory. For a single user, the arithmetic per weight is small and the reading dominates, so

> generation speed (tokens/s) ≤ memory bandwidth (bytes/s) ÷ weight bytes read per token.

**Mining cards were built for memory bandwidth.** Ethereum mining was also memory-bound, so these cards pair a mid-range
chip with fast memory: each P106-090 reads ~173 GB/s, and the nine together hold 54 GB.

**Everything else about the rig is hostile:**

- Each card talks to the host over one PCIe 1.0 lane (~0.2 GB/s, almost 1000× slower than the card's own memory), and
  cards cannot exchange data directly; everything passes through the host.
- The host CPU has 2 cores, 4 GB of RAM and no AVX vector instructions, so it cannot help with the computation.
  Every weight has to sit in GPU memory.
- The GPUs are from NVIDIA's Pascal generation: no tensor cores, and 16-bit floating-point arithmetic runs at 1/64 of
  the 32-bit rate. Many modern inference engines don't run on them at all.

**Mixture-of-Experts models fit this profile.** A Mixture-of-Experts (MoE) model contains many "expert" sub-networks
and uses only a few of them for each token. Qwen3.6-35B-A3B, for example, has 35 billion parameters but uses about 3
billion per token. That means a model needs a lot of total memory but only a little is read per token: a good match for
nine cards with plenty of combined memory. Splitting the model by layers across cards means only a small activation
vector (a few kilobytes) crosses the slow links per token.

## Results on 2026-10-22

### Current models

| Model | Type | Parameters used per token / stored | Native context | Context tested | Cards | Prompt processing (tokens/s) | Generation (tokens/s) | Details |
|---|---|---|---|---|---|---|---|---|
| Flint, Q4_1 (released 2026-10-20) | MoE | ~2.5 B / 39.8 B | 262,144 | up to 262,144 (fits on 9 cards, 33.9 GB) | 5–9 | 161–171 | **27.9–28.1** with the CUDA-graphs build (24.6 without); 19.5 at 32k tokens of context; **24.3 on real chat requests** at 262k context with backend sampling | [Flint](benchmarks/2026-10-22-flint.md), [speed-ups](benchmarks/2026-10-22-speedups.md) |
| Qwen3.6-35B-A3B, Q4_0 | MoE | 3 B / 35.5 B | 262,144 | up to 262,144 (fits) | 4–9 | 215–247 | **18.5** with an empty context (19.6 with CUDA graphs); 14.0 at 32k and 8.1 at 64k tokens of context; **27.7 on real chat requests** with CUDA graphs, multi-token prediction and backend sampling | [Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md), [speed-ups](benchmarks/2026-10-22-speedups.md) |
| Qwen3.8-27B, Q4_0 (UD-Q4_K_M at first) | dense | 27.3 B / 27.3 B | 262,144 | 32,768 | 4 | 51–54 | 3.9 (UD-Q4_K_M), 4.3 (Q4_0 + CUDA graphs); **7.5 on real chat requests** with multi-token prediction (2 drafted tokens) | [Qwen baselines](benchmarks/2026-10-22-qwen-baselines.md), [speed-ups](benchmarks/2026-10-22-speedups.md) |

### Calibration and small models

| Benchmark | Model | Native context | Cards | Prompt processing (tokens/s) | Generation, empty context (tokens/s) | Generation with context (tokens/s) |
|---|---|---|---|---|---|---|
| [Single-card calibration](benchmarks/2026-10-22-single-card-calibration.md) | Llama-2-7B, Q4_0 (dense, 6.7 B parameters) | 4,096 | 1 | 218 | 19.5 | 18.4 at 1,024 |
| | Llama-2-7B, Q4_K_M | 4,096 | 1 | 196 | 15.0 | — |
| | Llama-2-7B, Q8_0 | 4,096 | 2 | 204 | 13.2 | — |
| | Llama-2-7B, Q4_0 | 4,096 | 2 | 211–216 | 20.2–20.4 | — |
| | Llama-2-7B, Q4_0 | 4,096 | 9 | 188 | 17.7 | — |
| [Small MoE models](benchmarks/2026-10-22-small-moe-overhead.md) | OLMoE-1B-7B, Q4_K_M (1.3 B of 6.9 B parameters per token) | 4,096 | 1 | — | 54.1 | 46.5 at 4,096 |
| | granite-3.1-3b-a800m, Q4_K_M (0.8 B of 3.3 B) | 131,072 | 1 | — | 46.4 | 35.5 at 4,096 |
| | LFM2.5-8B-A1B, Q4_K_M (1.5 B of 8.5 B) | 128,000 | 1 | — | 39.2 | 38.2 at 4,096 |
| | granite-4.0-h-tiny, Q4_K_M (1 B of 6.9 B) | 131,072 | 1 | — | 30.1 | 29.3 at 4,096 |
| | Ling-3.0-tiny, Q4_K_M (1.3 B of 7.9 B) | 131,072 | 1 | — | 28.9 | 27.6 at 4,096 |

All runs use llama.cpp with flash attention on. "Native context" is the longest input the model was trained for;
"with context" means that many tokens were already in the model's cache when generation was measured, which is what
a long conversation or a large document feels like. Llama-2-7B is old, but it is the reference model of the
[llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013), so it calibrates these cards
against hundreds of other GPUs.

### What we have learned

1. **All nine cards behave the same.** On the same benchmark they are within 5 % of each other, with no GPU errors.
2. **A P106-090 uses only ~42 % of its memory bandwidth when generating.** It reaches 73 GB/s of the 173 GB/s available.
   With only 5 SMs, the card can't issue memory requests and unpack compressed weights fast enough to keep its memory
   busy. A P106-100, the same chip with 10 SMs, reaches ~115 GB/s on the same file.
3. **The weight compression format matters as much as its size.** Simple formats make better use of memory bandwidth:
   Q8_0 reaches 54 % of peak, Q4_0 42 % and Q4_K_M only 35 %. Q4_K_M is 7 % bigger than Q4_0 but 23 % slower.
4. **The 75 W power limit slows generation.** A busy card spends most of its time at the limit with its clock pulled
   down. Splitting a model over two cards lets each one rest half the time and makes generation 4–5 % faster than on
   one card. Spreading it over all nine costs ~13 %: each card-to-card handoff adds about 1 ms per token, and idle
   cards clock down.
5. **Small MoE models are limited by the number of GPU operations, not by memory.** Each generated token runs one to
   three thousand small GPU operations, and on this card each costs about 10 µs regardless of how much data it
   touches. A model that would need ~10 ms per token to read its weights therefore takes ~33 ms. The 2-core CPU is not
   the main culprit: the GPU is busy 95–98 % of the time, and occupying one CPU core slows generation by at most 6 %.
   Fewer, larger GPU operations are the lever. CUDA graphs, which llama.cpp disables on this GPU generation, record
   the operations once and replay them with one launch; a patched build gains 16 % on a small MoE and 7 % on
   Qwen3.6-35B-A3B ([build script](../scripts/build_llamacpp_docker.sh), option `--cuda-graphs-pascal`).
6. **A 2026 MoE model runs at reading speed, with its full context.** Qwen3.6-35B-A3B generates 18.5 tokens/s,
   4.7 times faster than the dense Qwen3.8-27B, and its full 262,144-token context fits in 27 GB of the 54 GB of GPU
   memory. Multi-token prediction, where a small extra layer drafts the next tokens and the model verifies them in one
   pass, raises real chat generation from 15.5 to 22.3 tokens/s.
7. **On a 4 GB host, llama.cpp's defaults can crash the machine at long contexts.** At 262,144 tokens of context the
   default micro-batch of 512 makes llama.cpp pin ~1 GiB of host RAM, and the kernel's out-of-memory killer takes the
   server and parts of the desktop with it. `-ub 128` cuts that buffer to 258 MiB and costs nothing in generation speed.
8. **The rig can serve two conversations at full speed.** Qwen3.6 needs only four or five cards, so two copies fit:
   one on cards 0–4 with 262,144 tokens of context and one on cards 5–8 with 131,072. Each keeps ~16 tokens/s while the
   other is busy, 31–35 tokens/s in total. One copy serving two requests at once manages only 23–24 tokens/s in total,
   but gives both conversations the full 262,144 tokens.
9. **Flint is the fastest model so far.** Released two days before these tests, it uses 2.5 B of its 39.8 B parameters
   per token and keeps most of its attention local. With the CUDA-graphs build it generates 28 tokens/s (22.5 on real
   chat requests with the full 262,144-token context loaded), 1.4 times Qwen3.6-35B-A3B. One copy on all nine cards
   can serve two conversations with 262,144 tokens of context each at 21–24 tokens/s each, 35–42 tokens/s in total.
10. **Software settings nearly double the speed.** A Q4_0 file, the CUDA-graphs build, multi-token prediction and
    sampling on the GPU (`-bs`) take Qwen3.6-35B-A3B from 15.5 to 27.7 tokens/s on real chat requests and the dense
    Qwen3.8-27B from 3.9 to 7.5. Dense models want fewer drafted tokens (1–2) than llama.cpp's default of 3, because on
    these cards checking several tokens at once costs noticeably more than checking one
    ([speed-ups](benchmarks/2026-10-22-speedups.md)).
11. **For one model, the file format is a speed/fidelity dial.** For Flint at ~25 GB: the makers' Q4_1 generates 24.3
    tokens/s on real requests, a Q5_0 made with an importance matrix 18.9, and Unsloth's K-quant-based UD-Q4_K_XL 15.8;
    their predictions agree with the 8-bit reference's top token 78.6 %, 85.0 % and 87.6 % of the time. Q4_0 breaks
    Flint, and using fewer experts per token costs far more quality than the 4–9 % speed it gains
    ([Flint](benchmarks/2026-10-22-flint.md); how to score and make quants on any PC:
    [quants/guide.md](../quants/guide.md)).

## How to read the numbers

- **Token:** the unit of text a model reads and writes, roughly three quarters of a word.
- **Prompt processing (pp):** reading the user's input. All prompt tokens are processed together, so the speed is set
  by arithmetic throughput. A 512-token prompt at 200 tokens/s takes 2.5 s before the first word appears.
- **Generation (tg):** producing the answer one token at a time. This is the speed at which text streams out, and it
  is set by memory reading as described above. Comfortable reading speed is roughly 5–10 tokens/s.
- **Quantization:** storing weights with fewer bits. Q4_0 packs 32 weights into a block with one scale, ~4.5 bits per
  weight. Q4_K_M uses a two-level block structure at a similar size. Q8_0 uses ~8.5 bits per weight. Fewer bits
  means a smaller file and less to read per token, at some cost in accuracy.
- **Effective bandwidth:** weight bytes on the GPU × generated tokens per second, i.e. how much of the memory bandwidth
  the software actually uses.
- **Layer split:** the model's layers are divided into consecutive blocks, one per card. For each token the cards work
  one after another, passing a small activation vector between them, so adding cards adds memory but not speed.
- **Flash attention (FA):** a fused implementation of the attention step that avoids writing large intermediate
  matrices to memory.
- **Graph nodes and kernels:** llama.cpp expresses one model step as a graph of operations (nodes); each runs as one
  or more GPU programs (kernels), and every kernel launch carries a fixed start-up cost.
- **Mixture of Experts (MoE):** a model whose feed-forward layers consist of many experts, of which a router picks a
  few per token. "1.3 B of 6.9 B parameters per token" means 1.3 billion weights are read for each token out of
  6.9 billion stored.
