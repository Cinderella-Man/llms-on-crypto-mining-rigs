# Small MoE models: where the time per token goes

*Measured 2026-10-22.*

Mixture-of-Experts (MoE) models read only a small fraction of their weights for each token, so on paper small MoE
models should generate text very fast on this rig. A model that reads ~0.5–1 GB per token could approach 100 tokens/s
on a card that streams 60–90 GB/s ([single-card calibration](2026-10-22-single-card-calibration.md)). This benchmark
runs five small MoE models on one card to find out what actually limits them, and in particular whether the rig's slow
2-core CPU is the bottleneck.

## Background: compute graphs and kernel launches

llama.cpp turns one step of the model into a *compute graph*: a list of operations (matrix multiplications,
normalisations, expert routing, attention and so on) that it runs one after another on the GPU. Each operation, called
a *node*, becomes one or more GPU *kernel* launches, and every launch carries a fixed cost on the CPU (preparing and
submitting it) and on the GPU (starting it up). A dense model needs about a thousand nodes per token; MoE models with
expert routing, or hybrid models that mix attention with recurrent layers, need two to three thousand. When the work
inside each kernel is small, these fixed costs can dominate.

CUDA has a feature called *CUDA graphs* that records the whole sequence of kernels once and replays it with a single
launch, removing most of the per-kernel CPU cost. llama.cpp currently disables CUDA graphs on Pascal GPUs such as the
P106-090; the log line "disabling CUDA graphs due to GPU architecture" confirms this on the rig. All results below are
therefore without CUDA graphs. An unmerged llama.cpp change
([PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) re-enables them for this GPU generation and reports
+8 % to +78 % generation speed for these same models on a GTX 1080.

## Setup

| Item | Value |
|---|---|
| Inference engine | [llama.cpp](https://github.com/ggml-org/llama.cpp) commit `85d5703a3` (2026-09-04), CUDA 12.9.2, built for compute capability 6.1 |
| GPU | One P106-090 (GPU0), selected with `CUDA_VISIBLE_DEVICES` |
| Command | `llama-bench -ngl 99 -p 0 -n 64 -d 0,4096 -t 1,2 -fa off,on -r 3` |
| CPU contention | The generation test was repeated while a busy loop occupied one CPU core, then both cores |
| Graph size | Number of graph nodes read from llama.cpp's verbose log for each model |
| Monitoring | `nvidia-smi` every 0.25 s: GPU utilisation, clocks, power, throttle reasons |
| Script | [scripts/launch_overhead_probe.sh](../../scripts/launch_overhead_probe.sh) (`DEPTHS=0,1024` for Llama-2-7B); every llama-bench row: [docs/data/2026-10-22_e0.9_launch_overhead](../data/2026-10-22_e0.9_launch_overhead/README.md) |

The models are the ones used in PR #27721, at Q4_K_M quantization so that each fits on one 6 GB card, plus the dense
Llama-2-7B from the calibration as a control:

| Model | Architecture | Parameters, total / per token | File |
|---|---|---|---|
| [granite-3.1-3b-a800m](https://huggingface.co/bartowski/granite-3.1-3b-a800m-instruct-GGUF) | 32 layers, 40 experts, 8 per token | 3.3 B / 0.8 B | 2.0 GB |
| [OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125-Instruct-GGUF) | 16 layers, 64 experts, 8 per token | 6.9 B / 1.3 B | 4.2 GB |
| [granite-4.0-h-tiny](https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF) | 36 Mamba-2 + 4 attention layers, 64 experts, 6 per token | 6.9 B / 1 B | 4.2 GB |
| [Ling-3.0-tiny](https://huggingface.co/bartowski/Ling-3.0-tiny-GGUF) | 24 layers, 128 experts, 8 per token | 7.9 B / 1.3 B | 4.9 GB |
| [LFM2.5-8B-A1B](https://huggingface.co/LiquidAI/LFM2.5-8B-A1B-GGUF) | 18 convolution + 6 attention layers, 32 experts, 4 per token | 8.5 B / 1.5 B | 5.2 GB |
| [Llama-2-7B](https://huggingface.co/TheBloke/Llama-2-7B-GGUF) (Q4_0, dense control) | 32 layers, no experts | 6.7 B / 6.7 B | 3.8 GB |

Architecture details are from the model cards and PR #27721.

## Results

### Generation speed and graph size

Generation of 64 tokens with flash attention on, 2 CPU threads and an idle CPU. "d4096" means the same test after
4096 tokens of context are already in the cache (Llama-2-7B: 1024, because a 4096-token cache doesn't fit next to its
weights on one card).

| Model | Graph nodes | tokens/s | tokens/s at d4096 | ms per token | ms per token ÷ nodes | GTX 1080, no graphs / with graphs (PR #27721) |
|---|---|---|---|---|---|---|
| granite-3.1-3b-a800m | 1960 | 46.4 | 35.5 | 21.6 | 11.0 µs | 81.2 / 124.5 |
| OLMoE-1B-7B | 934 | 54.1 | 46.5 | 18.5 | 19.8 µs | 150.5 / 169.5 |
| granite-4.0-h-tiny | 3418 | 30.1 | 29.3 | 33.3 | 9.7 µs | 57.4 / 79.5 |
| Ling-3.0-tiny | 3090 | 28.9 | 27.6 | 34.6 | 11.2 µs | 52.7 / 79.8 (Q6_K) |
| LFM2.5-8B-A1B | 1274 | 39.2 | 38.2 | 25.5 | 20.0 µs | 116.2 / 128.5 |
| Llama-2-7B (dense) | 998 | 19.4 | 18.4 (d1024) | 51.5 | 51.6 µs | — |

Spread between repetitions is below 0.5 % for all idle runs.

### Sensitivity to CPU load

Generation at d0, flash attention on. "One core busy" and "both busy" ran a spinning loop pinned to one or both of the
two CPU cores during the test. GPU utilisation is the share of time the GPU had a kernel running, from `nvidia-smi`.

| Model | Idle CPU | One core busy | Both cores busy | GPU utilisation, idle → both busy |
|---|---|---|---|---|
| granite-3.1-3b-a800m | 46.4 | 46.3 (−0.2 %) | 44.3 (−4.6 %) | 96 % → 92 % |
| OLMoE-1B-7B | 54.1 | 53.7 (−0.6 %) | 48.2 (−10.8 %) | 97 % → 66 % |
| granite-4.0-h-tiny | 30.1 | 28.2 (−6.3 %) | 27.5 (−8.6 %) | 95 % → 86 % |
| Ling-3.0-tiny | 28.9 | 27.2 (−5.9 %) | 26.1 (−9.7 %) | 96 % → 80 % |
| LFM2.5-8B-A1B | 39.2 | 37.7 (−3.8 %) | 37.7 (−3.8 %) | 98 % → 89 % |
| Llama-2-7B (dense) | 19.4 | 19.0 (−2.4 %) | 18.8 (−3.5 %) | — |

### Other settings

- **CPU threads:** 1 and 2 threads give the same speed within measurement noise for every model. With all weights on
  the GPU, llama.cpp's CPU threads have almost nothing to do.
- **Flash attention at long context:** at d4096, flash attention raises generation by 44 % (granite-3.1), 55 %
  (OLMoE) and 53 % (Ling-3.0-tiny), and by 5–16 % for the hybrid models whose few attention layers keep the cache
  small. It is essential for long contexts on these cards.
- **Context length:** with flash attention, the hybrid models (granite-4.0-h, Ling-3.0, LFM2.5) lose only 3–5 % at
  4096 tokens of context, while the plain-attention models lose 14–23 %.
- **Clocks and power:** the card held 1833–1847 MHz with no power throttling for four of the five models; LFM2.5 drew
  ~70 W and touched the 75 W limit.

## Interpretation

- **Small MoE models are limited by the number of operations, not by memory bandwidth.** The three models with the
  largest graphs (1960–3418 nodes) take almost exactly 10–11 µs per node per token, whatever their size, and all five
  run far below what memory bandwidth alone would allow. granite-4.0-h-tiny, for example, reads roughly 0.6 GB per
  token, which would take ~10 ms at the measured bandwidth, yet each token takes 33 ms.
- **The CPU is not the main bottleneck.** The GPU reports a kernel running 95–98 % of the time, and taking away one of
  the two CPU cores slows generation by at most 6 %. Taking away both costs 4–11 %, far less than the ~33 % a purely
  CPU-bound process would lose. The CPU keeps just ahead of the GPU; the per-node time is mostly spent on the GPU,
  running thousands of very small kernels that each carry a fixed start-up cost and give its 5 SMs too little work.
- **The levers are fewer and larger kernels.** Fusing operations, fused expert-routing kernels, and CUDA graphs (which
  also shorten the gaps between kernels on the GPU side) all attack this overhead. A faster CPU would help little.
- **Compared with a GTX 1080 desktop without CUDA graphs**, the P106-090 reaches 34–57 % of its speed on the same
  models. The GTX 1080 has four times as many SMs.

## Follow-up: CUDA graphs on Pascal

A build of llama.cpp commit `d77256999` (2026-10-22) with PR #27721 applied, against the same commit without it, on
granite-3.1-3b-a800m (one card, `llama-bench -ngl 99 -fa on -p 0 -n 64 -r 3`):

| Build | Generation, tokens/s |
|---|---|
| Commit `85d5703a3` (the build used above) | 46.6 |
| Commit `d77256999` | 49.8 |
| Commit `d77256999` with CUDA graphs enabled for Pascal | **57.7 (+16 %)** |

CUDA graphs remove part of the per-operation cost, but much less than the +53 % the PR reports for the same model on a
GTX 1080 with a desktop CPU. That fits the result above: on the rig most of the per-node time is spent on the GPU, not
on the CPU launching work.

## What this means for larger MoE models

A 2026 model such as Qwen3.6-35B-A3B has a larger graph than any of these and reads ~2–2.7 GB per token. If the
per-node cost stays near 10 µs, a few thousand nodes add 25–35 ms per token on top of the weight-reading time, and
removing per-node overhead becomes as important as choosing a fast quantization format. The next benchmarks measure
that model directly, with and without CUDA graphs.
