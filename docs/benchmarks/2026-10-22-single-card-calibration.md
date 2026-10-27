# Single-card calibration: Llama-2-7B on one P106-090

*Measured 2026-10-22.*

This benchmark measures how fast one P106-090 card runs a standard model on its own, on each of the nine cards in
turn. It answers three questions that every later result depends on:

1. Do all nine cards perform the same?
2. How does a P106-090 compare with other GPUs on a benchmark many people have run?
3. How much of the card's memory bandwidth does real inference code actually use?

The model is Llama-2-7B, a dense language model with 6.7 billion parameters. It is old, but it is the reference
model of the [llama.cpp CUDA scoreboard](https://github.com/ggml-org/llama.cpp/discussions/15013), where hundreds of
GPUs have been measured with exactly the same file and command. The project's real targets are much newer
Mixture-of-Experts models; this benchmark calibrates the hardware before those.

## Background: what limits generation speed

A language model produces text one *token* at a time (a token is roughly three quarters of an English word). To
produce each token, the GPU multiplies the current state by every weight matrix in the model, so it has to read
every weight from its memory once per token. For one user at a time, the arithmetic per weight is tiny and the
reading dominates. Generation speed is therefore bounded by

> tokens per second ≤ memory bandwidth ÷ bytes of weights read per token.

A P106-090 has ~173 GB/s of measured memory bandwidth ([hardware spec](../mining_rig.md)) and the Llama-2-7B file
used here puts 3.75 GB of weights on the GPU, so the upper bound is ~46 tokens/s. How close a card gets to that
bound depends on whether its compute units can unpack the compressed weights as fast as memory delivers them.

Reading the prompt is a different workload. All prompt tokens are processed together, so each weight read from
memory is reused hundreds of times, and the speed is limited by arithmetic throughput instead of memory.

## Setup

| Item | Value |
|---|---|
| Inference engine | [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `85d5703a3` (2026-09-04), built on the rig with CUDA 12.9.2 for compute capability 6.1, flash-attention kernels enabled |
| Driver and OS | NVIDIA 580.178.04, Ubuntu 24.04, desktop session running |
| Command | `llama-bench -ngl 99 -fa 0,1 -p 512 -n 128 -r 5`, with one card made visible at a time through `CUDA_VISIBLE_DEVICES` |
| Models | [TheBloke/Llama-2-7B-GGUF](https://huggingface.co/TheBloke/Llama-2-7B-GGUF): Q4_0 (3.83 GB), Q4_K_M (4.08 GB), Q8_0 (7.16 GB); checksums match the published files |
| Monitoring | `nvidia-smi` sampled every 0.5 s: P-state, SM and memory clocks, power, temperature, utilisation, throttle reasons |
| Script | [scripts/bench.sh](../../scripts/bench.sh) with `GPU_SETS="0 1 2 3 4 5 6 7 8"` and `BENCH_ARGS="-ngl 99 -fa 0,1 -p 512 -n 128 -r 5"`; the split runs used `GPU_SETS="0,1 0,4 0,1,2,3,4,5,6,7,8"`; power and clock summary per card: [docs/data/2026-10-22_e0.8_calibration](../data/2026-10-22_e0.8_calibration/README.md) |

How to read the result columns:

- **pp512** (prompt processing): tokens per second while reading a 512-token prompt.
- **tg128** (token generation): tokens per second while generating 128 tokens. This is the speed a user sees while
  text streams out.
- **FA**: flash attention, a fused attention kernel. It is either off or on.
- **±** is the standard deviation over five repetitions.
- **Quantization** names the compression format of the weights: Q4_0 stores weights in blocks of 32 at ~4.5 bits each
  with one scale per block; Q4_K_M uses a more elaborate two-level block structure at a similar size; Q8_0 uses
  ~8.5 bits per weight with the simplest layout.

## Results

### Every card, Q4_0

| GPU | pp512 | pp512, FA | tg128 | tg128, FA |
|---|---|---|---|---|
| 0 | 209.3 | 217.9 | 17.88 | 19.37 |
| 1 | 204.3 | 209.6 | 17.40 | 18.78 |
| 2 | 213.1 | 221.9 | 18.12 | 19.67 |
| 3 | 212.0 | 219.7 | 17.98 | 19.56 |
| 4 | 211.4 | 219.9 | 18.13 | 19.61 |
| 5 | 213.0 | 219.4 | 18.07 | 19.53 |
| 6 | 211.3 | 217.5 | 17.88 | 19.41 |
| 7 | 214.1 | 222.0 | 18.16 | 19.63 |
| 8 | 210.3 | 218.1 | 17.91 | 19.47 |

All values are tokens per second; run-to-run spread is below 1 %. The nine cards are within 5 % of each other. GPU1 is
the slowest and also ran the warmest (57 °C peak, against 46–53 °C for the others). Flash attention speeds up
generation by 8 % and prompt processing by 3–4 % on every card. No GPU errors were logged.

For comparison, the scoreboard lists a P106-100, the same GP106 chip with 10 streaming multiprocessors (SMs) instead of
5, at about 30.6 tokens/s generation and 407–438 tokens/s prompt processing on the same file
([discussion #15013](https://github.com/ggml-org/llama.cpp/discussions/15013)). The P106-090 reaches about 64 % of its
generation speed and half its prompt speed.

### Effective memory bandwidth by quantization format

*Effective bandwidth* is the bytes of weights the GPU reads per token times the tokens generated per second. Weights
on the GPU are the file size minus the token-embedding table, which llama.cpp keeps in host memory.

| Model | GPUs | pp512, FA | tg128, FA | Effective bandwidth | Share of the 173 GB/s measured peak |
|---|---|---|---|---|---|
| Q4_0 | 1 (mean of all nine) | 218.4 | 19.45 | 73 GB/s | 42 % |
| Q4_K_M | 1 | 195.8 | 14.99 | 60 GB/s | 35 % |
| Q8_0 | 2 (too large for one 6 GB card) | 204.3 | 13.23 | 93 GB/s | 54 % |

Q4_K_M is 7 % larger than Q4_0 yet generates 23 % slower. Q8_0 is 87 % larger than Q4_0 but only 32 % slower. The
format that is cheapest to unpack makes the best use of memory bandwidth.

### Splitting one model across several cards

llama.cpp can split a model by layers across GPUs: each card holds a contiguous block of layers, and the cards work
one after another for every token, passing a small activation vector between them. Q4_0 fits on one card, so the same
file was also run split across several:

| GPUs | Connection | pp512, FA | tg128, FA |
|---|---|---|---|
| 1 | — | 218.4 | 19.45 |
| 2 (GPU0 + GPU1) | behind the same PCIe switch | 211.3 | 20.19 |
| 2 (GPU0 + GPU4) | through the CPU's root ports | 215.7 | 20.43 |
| 9 | all cards | 188.1 | 17.67 ± 1.31 |

## What the clock and power logs show

- **Every busy card ran in the P0 performance state with the full 4006 MHz memory clock.** Compute workloads do not
  drop these cards to a lower memory clock.
- **Generation runs into the 75 W power limit.** A single card generating text drew ~69 W on average with spikes to
  the limit; the driver reported its power cap as the active throttle reason in 50–90 % of samples, and the core
  clock averaged 1710–1760 MHz instead of the 1911 MHz maximum. The power limit cannot be set above 75 W.
- **Splitting across two cards relieves the power limit.** Each card is busy about 45 % of the time, the cap is
  reported in only 14–22 % of samples, the core clock averages ~1830 MHz, and generation is 4–5 % faster than on one
  card.
- **Splitting across all nine cards costs ~13 % against two.** Each card is busy only ~10 % of the time, its average
  core clock drops to ~1595 MHz, and the run-to-run spread grows. The seven extra card-to-card boundaries add ~7 ms per
  token, about 1 ms each.

## Interpretation

- **Generation on this card is limited by its compute units, not by memory.** Only 42 % of the measured bandwidth is
  used with Q4_0. With 5 SMs, the card cannot issue enough memory requests and unpack enough weights to keep its
  memory busy. Against the same 192 GB/s rated figure, the 10-SM P106-100 reaches ~115 GB/s (60 %) on the same file
  and the P106-090 73 GB/s (38 %).
- **Unpacking cost matters as much as file size.** The simpler the quantization layout, the higher the effective
  bandwidth (Q8_0 > Q4_0 > Q4_K_M). Many popular 2026 model files use K-quant-based mixed formats, which this result
  suggests are slow on these cards; formats like Q4_0 and Q8_0 are worth preferring and testing.
- **Clock speed matters because the card is compute-limited.** That is why the power limit slows generation and why
  two cards beat one.
- **Using more cards than needed has a cost.** Each extra boundary adds about a millisecond per token on this rig's
  single-lane PCIe links without direct GPU-to-GPU transfers, and lightly loaded cards clock down between bursts.

## What this means for Mixture-of-Experts models

A Mixture-of-Experts (MoE) model reads only a small part of its weights for each token. For example, Qwen3.6-35B-A3B
holds 35 billion parameters but uses about 3 billion per token. Its 21 GB file is spread over four or more cards, yet
only ~2–2.7 GB is read per generated token. At the 60–93 GB/s effective bandwidth measured here, that sets a ceiling
of roughly 22–46 tokens/s before any other overhead, which the MoE benchmarks will test directly.
