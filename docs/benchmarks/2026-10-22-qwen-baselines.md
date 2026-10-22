# First 2026 models: Qwen3.6-35B-A3B and Qwen3.8-27B

*Measured 2026-10-22.*

These are the first baselines with current models: Qwen3.6-35B-A3B, a Mixture-of-Experts (MoE) model, and
Qwen3.8-27B, a dense model of similar quality class. They answer the practical questions an owner of this rig has:
how fast a modern model generates text, how much context fits, what multi-token prediction adds, and which settings
crash a machine with only 4 GB of host RAM.

## The two models

| | Qwen3.6-35B-A3B | Qwen3.8-27B |
|---|---|---|
| Type | Mixture of Experts: 256 experts, 8 used per token, plus a shared expert | Dense: every weight is used for every token |
| Parameters | 35.5 B stored, ~3 B used per token | 27.3 B, all used per token |
| Layers | 40: 30 Gated DeltaNet (linear attention) + 10 full attention | 64: 48 Gated DeltaNet + 16 full attention |
| Native context | 262,144 tokens | 262,144 tokens |
| File tested | [bartowski Q4_0](https://huggingface.co/bartowski/Qwen_Qwen3.6-35B-A3B-GGUF), 20.8 GB, plus its separate 1.2 GB multi-token-prediction head | [Unsloth UD-Q4_K_M](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF), 16.5 GB |

Both are hybrids: most layers use a linear-attention mechanism (Gated DeltaNet) whose memory doesn't grow with
context, and only a quarter of the layers keep a classic key-value (KV) cache. That is what makes very long contexts
cheap on this rig: Qwen3.6 needs only ~20 KiB of KV cache per token, so 262,144 tokens take ~5 GiB.

Q4_0 was chosen for Qwen3.6 because simple quantization layouts make the best use of these cards' memory bandwidth
([single-card calibration](2026-10-22-single-card-calibration.md)). The Qwen3.8-27B file is the widely used Unsloth
dynamic quant, which is K-quant based.

## Setup

| Item | Value |
|---|---|
| Inference engine | [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `85d5703a3` (2026-09-04), CUDA 12.9.2, compute capability 6.1, flash attention on, CUDA graphs inactive (disabled by llama.cpp on Pascal) |
| Synthetic benchmarks | [scripts/bench.sh](../../scripts/bench.sh): `llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3`; context-depth runs with `-p 0 -n 64 -d 0,8192,32768,65536 -r 1` |
| Real requests | [scripts/server_bench.sh](../../scripts/server_bench.sh): `llama-server` on 127.0.0.1, four fixed chat prompts (code, explanation, arithmetic word problem, rewrite) at temperature 0, 256 generated tokens each |
| Models | Loaded from a USB 3 flash drive; a model is spread over the cards by layers (`--split-mode layer`, the default) |

## Results

### Generation speed and number of cards

| Model | Cards | Prompt processing, pp512 (tokens/s) | Generation, tg128 (tokens/s) |
|---|---|---|---|
| Qwen3.6-35B-A3B Q4_0 | 4 | 231.6 ± 5.5 | 18.46 ± 0.32 |
| | 5 | 229.8 ± 5.5 | 18.31 ± 0.33 |
| | 9 | 215.3 ± 13.2 | 18.91 ± 0.21 |
| Qwen3.8-27B UD-Q4_K_M | 4 | 50.6 ± 1.2 | 3.94 ± 0.06 |
| | 9 | 49.1 ± 1.2 | 3.43 ± 0.20 |

Qwen3.6 fits on four cards and runs at the same speed on four, five or nine. The MoE model generates text **4.7 times
faster** than the dense one and reads prompts 4.6 times faster, although its file is larger.

Loading Qwen3.6 takes about 2 min 45 s (measured on server start-up); the smaller Qwen3.8-27B file takes roughly
2 minutes. llama.cpp uploads the weights one card at a time, so a single card's ~190 MB/s PCIe link sets the pace.

### Real chat requests and multi-token prediction

`llama-bench` measures the raw model; a real server also has to sample every token from the model's ~248,000-word
vocabulary on the CPU, apply the chat template and stream the result. With Qwen3.6 on all nine cards and a
32,768-token context:

| Prompt | Plain decoding (tokens/s) | With multi-token prediction (tokens/s) | Drafted tokens accepted |
|---|---|---|---|
| Code | 15.09 | 21.36 | 176 / 235 (75 %) |
| Explanation | 15.25 | 19.47 | 168 / 258 (65 %) |
| Arithmetic word problem | 15.46 | 23.65 | 181 / 218 (83 %) |
| Rewrite a paragraph | 16.02 | 25.47 | 185 / 209 (89 %) |
| **All four** | **15.51** | **22.34 (+44 %)** | **710 / 920 (77 %)** |

**Multi-token prediction (MTP)** uses a small extra network layer, shipped with the model, to guess the next few tokens
cheaply; the full model then checks all the guesses in one pass and keeps the ones it agrees with. Because the full
model costs about the same whether it checks one token or four, every accepted guess is nearly free. It is enabled
with `--spec-type draft-mtp -md <mtp head file>` and costs 2.2 GB of extra GPU memory here.

Plain decoding through the server is ~16–18 % slower than `llama-bench` (15.5 against 18.5–18.9 tokens/s). The likely cause is
sampling over the large vocabulary on the 2-core CPU without vector instructions, a few milliseconds per token.

### Long context

Generation speed of Qwen3.6 with tokens already in the context, on nine cards (one repetition each, so the numbers are
approximate):

| Context already in the cache | 0 | 8,192 | 32,768 | 65,536 |
|---|---|---|---|---|
| Generation (tokens/s) | 14.7 | 16.6 | 14.0 | 8.1 |

Speed holds up to ~32k tokens and halves by 64k. Every generated token has to read the whole KV cache of the ten
full-attention layers, and on Pascal the flash-attention kernels run in 32-bit floating point.

### Maximum context, and how to avoid crashing a 4 GB host

**The full 262,144-token context fits.** With `-c 262144 -b 512 -ub 128`, Qwen3.6 uses 27.0 GB of the 54 GB of GPU
memory (at most 3.65 GB on any card) and 530 MB of host RAM, and loads in 2 min 43 s.

**With llama.cpp's default micro-batch of 512 tokens, the same context brings the whole machine down.** llama.cpp keeps
the attention mask, a matrix of context length × micro-batch entries, in pinned host memory, and with several cards it
keeps four copies so that cards can work on different micro-batches at once. Pinned memory can't be swapped out. At
262,144 tokens that is ~1 GiB on top of the desktop and the model loading. The kernel's out-of-memory killer then ends
the server, and in our runs it also killed the login screen's processes, so the display went dark until the display
manager restarted. Measured sizes of that host buffer:

| Context | Micro-batch (`-ub`) | Pinned host buffer | GPU compute buffer per card | KV cache per card (f16) |
|---|---|---|---|---|
| 65,536 | 512 | 264 MiB | 372 MiB | 128 MiB |
| 131,072 | 512 | 520 MiB | 628 MiB | 256 MiB |
| 262,144 | 512 | ~1 GiB (process killed) | — | — |
| 262,144 | 128 | 258 MiB | 285 MiB | 512 MiB |

The card holding the most full-attention layers needs twice the KV cache of the others. A smaller micro-batch makes
reading long prompts somewhat less efficient but has no effect on generation speed. On a 4 GB host, set `-ub 128` for
any context above ~100k tokens. [scripts/server_bench.sh](../../scripts/server_bench.sh) also runs a watchdog that kills
the server if free host memory drops below 400 MB, before the kernel starts killing other processes.

### Newer llama.cpp and CUDA graphs

The same model on four cards with three builds, all `llama-bench -ngl 99 -fa on -p 512 -n 128 -r 3`:

| Build | pp512 (tokens/s) | tg128 (tokens/s) |
|---|---|---|
| Commit `85d5703a3` (2026-09-04), built on the rig | 231.4 ± 4.8 | 18.31 ± 0.31 |
| Commit `d77256999` (2026-10-22), built in Docker ([scripts/build_llamacpp_docker.sh](../../scripts/build_llamacpp_docker.sh)) | 246.8 ± 4.1 | 18.30 ± 0.32 |
| Same commit with CUDA graphs enabled for Pascal ([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) | 246.1 ± 6.7 | **19.61 ± 0.41** |

The newer llama.cpp reads prompts 7 % faster. CUDA graphs, which upstream llama.cpp disables on Pascal, add another
7 % to generation: they record each card's sequence of GPU operations once and replay it with a single launch. The
gain is smaller than on single-card models ([+16 % on granite-3.1](2026-10-22-small-moe-overhead.md)) because with
the model split over four cards, each card still runs its own graph segment and waits for the previous card.

### Qwen3.8-27B with multi-token prediction: first attempt

*Follow-up: with a Q4_0 file, the CUDA-graphs build and two drafted tokens per step, the dense model reaches 7.5
tokens/s; see [speed-ups](2026-10-22-speedups.md).*

The dense model on five cards through `llama-server`, 128 generated tokens per prompt, with Unsloth's separate MTP head
file and llama.cpp's default of 3 drafted tokens per step:

| Prompt | Plain (tokens/s) | With multi-token prediction (tokens/s) | Drafted tokens accepted |
|---|---|---|---|
| Code | 3.70 | 4.16 | 73 / 159 (46 %) |
| Explanation | 3.65 | 4.15 | 73 / 160 (46 %) |
| Arithmetic word problem | 3.74 | 4.89 | 82 / 131 (63 %) |

The gain is 13–31 %, far below the MoE model's 44 %. For a dense model every verification step reads all 15 GB of
weights, and on these 5-SM cards checking four tokens at once costs noticeably more than checking one, so a lower
acceptance rate eats most of the benefit. Fewer drafted tokens per step and the faster Q4_0 file are the next things to
try. (The MTP layer is also embedded in the main Qwen3.8-27B file, so `--spec-type draft-mtp` works without a separate
head.) Five cards are needed because the 4-card layout leaves no room for the head.

These runs stopped at the fourth prompt: free host memory fell below 400 MB *during* requests and the watchdog ended the
server. llama-server keeps up to 32 "context checkpoints" per conversation and an 8 GiB prompt cache in host RAM by
default; for hybrid models like Qwen3.8 each checkpoint holds the recurrent state of every linear-attention layer. On a
4 GB host, start llama-server with `--cache-ram 0` and a small `--ctx-checkpoints` value (the benchmark scripts now
pass `--cache-ram 0 --ctx-checkpoints 0`).

### Serving two conversations at once

Qwen3.6 needs only four or five of the nine cards, so the rig can hold two copies, or one copy with room for a second
conversation. Both layouts were measured with [scripts/parallel_bench.sh](../../scripts/parallel_bench.sh) and two chat
prompts of 256 generated tokens, each copy or slot getting one request at a time.

| Layout | Context per conversation | Generation, one request alone (tokens/s) | Generation, two requests at once (tokens/s) | Combined |
|---|---|---|---|---|
| Two copies: cards 0–4 (f16 KV cache) and cards 5–8 (q8_0 KV cache) | 262,144 and 131,072 | 15.5–16.2 and 15.9–16.5 | 16.7–16.8 and 16.4–16.7 | **31–32 tokens/s** |
| Two copies: cards 0–4 and cards 5–8, both f16 KV cache | 262,144 and 98,304 | 15.5–15.9 and 17.2–18.4 | 16.0–16.6 and 17.2–18.7 | **~33–35 tokens/s** (sum of the two) |
| One copy on all nine cards, two slots (`-np 2`) | 262,144 each | 15.5 (single-slot server) | 13.1 and 12.8–13.1 | **23–24 tokens/s** |

**Two separate copies run fully in parallel.** Each copy keeps its own speed when the other is busy, because they use
different cards, different PCIe switches and one CPU core each. The four-card copy is even slightly faster than the
five-card one. Together they produce 31–35 tokens/s, about twice a single conversation.

**One copy with two slots gives more context but less speed.** Its two conversations are processed as one batch of two
tokens. That batch still runs all ~3,700 operations once and reads more experts than a single token does, so each
conversation slows to ~13 tokens/s, for 23–24 tokens/s combined. In exchange, both conversations get the full 262,144
tokens, the weights exist only once, and the cards stay a third empty (2.7–4.7 GB used per card).

**Four cards leave little room for context.** The 19.4 GiB of weights fill four 6 GB cards up to ~4.9 GB each. The
largest contexts that work on four cards are 98,304 tokens with an f16 KV cache (~200 MB to spare on the fullest card)
and 131,072 tokens with a q8_0 KV cache (`-ctk q8_0 -ctv q8_0`, half the memory per token). 131,072 tokens in f16 load
but crash on the first request, when the server allocates working memory and finds the fullest card completely full.
The q8_0 cache costs the four-card copy 1–2 tokens/s against f16. Both copies loaded together leave ~1.3 GB of the 4 GB
host RAM free.

## Interpretation

- **MoE is the right model class for this rig.** Qwen3.6-35B-A3B reads ~2–2.7 GB per token against ~15 GB for the dense
  27B and generates 4.7 times faster; multi-token prediction adds another 44 % on real requests.
- **Per-operation overhead still dominates the MoE model.** Its compute graph has 3,727 operations per token. At the
  ~10 µs each measured for small MoE models ([small MoE benchmark](2026-10-22-small-moe-overhead.md)), that alone is
  ~37 ms of the ~54 ms per token. That is why Qwen3.6 is no faster than the much smaller dense Llama-2-7B, and why
  fewer, fused operations (and CUDA graphs) are the next levers.
- **The dense model is limited by memory reading.** 15.3 GiB per token at 3.94 tokens/s is ~65 GB/s, in line with the
  K-quant efficiency measured on a single card. A Q4_0 file and multi-token prediction are the obvious next steps for it.
- **Number of cards barely matters for speed** as long as the model fits; more cards buy room for context.
- **The rig is best used as two machines.** Two copies of Qwen3.6 on separate card groups double the total output
  (31–35 tokens/s) while each conversation keeps its full speed, with 262,144 and 131,072 tokens of context.
