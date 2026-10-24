# LLMs on crypto-mining rigs

<p align="center"><img src="docs/images/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="280"></p>

Modern open-weight LLMs on a retired GPU mining rig: **nine NVIDIA P106-090 cards** (6 GB each, every card on a single
PCIe 1.0 lane) driven by a 2-core Celeron with 4 GB of RAM. This repository holds a
[llama.cpp fork](https://github.com/Cinderella-Man/llama.cpp/tree/pascal-p106) tuned for these cards, model files made
for them, and every measurement behind the numbers below.

## Result: a real chat, stock llama.cpp against this project

[Flint](https://huggingface.co/RecursiveSI/flint) (a 39.8 B-parameter mixture-of-experts model, 2.5 B parameters per
token) answering one conversation of five user messages that build on each other, on all nine cards with the full
262,144-token context. Flint thinks before every answer, so a reply is 600–7,900 tokens. Both engines run with the same
settings, sampling and seeds; only the code differs.

| Model file | Size | Same next token as the original model | Stock llama.cpp | This project | Speed-up |
|---|---|---|---|---|---|
| Unsloth [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) (4-bit, the usual download) | 25.1 GB | 87.7 % | 14.20 tok/s | **27.97 tok/s** | **1.97×** |
| Unsloth [Q8_0](https://huggingface.co/unsloth/flint-GGUF) (8-bit, near-lossless) | 42.3 GB | 93.3 % | 14.78 tok/s | **29.55 tok/s** | **2.00×** |
| **Miner S4-L0 (Q4)**, ours: Q4 quality, built from formats these cards decode fast | 32.5 GB | 88.7 % | 16.06 tok/s | **31.86 tok/s** | **1.98×** |

Generation speed over the whole conversation (thinking included): a 1,000-token reply takes 31–36 s here instead of
62–70 s. Quality is measured against Flint's original 16-bit weights on wikitext-2. Method, every turn, waiting times and the commands: [the benchmark](docs/benchmarks/2026-10-23-flint-chat.md).

## What we changed

Each step measured with `llama-bench` (128 generated tokens) on Unsloth's UD-Q4_K_XL, six cards:

| Change | Generation | Where |
|---|---|---|
| Stock llama.cpp (`d77256999`) | 17.6 tok/s | |
| CUDA graphs on Pascal: llama.cpp switches them off for every GPU before Volta; replaying each card's few hundred GPU operations as one graph cuts the launch cost | 19.5 tok/s | fork commit, [upstream PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721) |
| 2 warps per block for single-token products on rows of 2,048 values or more, 4 warps for narrower ones | 24.5 tok/s | fork commit |
| **Pascal matrix-vector kernels** for Q8_0, Q4_0, Q4_1, Q5_0, Q5_K and IQ4_XS: each row gets a group of threads sized to its width, several rows share each load of the input, and weight blocks reach the threads through shared memory in wide, coalesced loads | **34.1 tok/s** | [`mmvq-pascal.cu`](https://github.com/Cinderella-Man/llama.cpp/blob/pascal-p106/ggml/src/ggml-cuda/mmvq-pascal.cu), [design](docs/research/notes/pascal_mmvq_integration.md) |

Also part of the result:
- **Miner S4-L0 (Q4)**, our own Flint file: Unsloth's Q4 quality, but in formats these cards decode quickly ([recipe](quants/recipes/flint-speed-s4-l0.txt),
  [how it was found](quants/README.md)). With all changes it generates 37.4 tok/s on six cards against UD-Q4_K_XL's 34.1.
- **`LLAMA_NO_MOE_DOWN_RESCALE=1`** switches off a safety rescale llama.cpp applies to Flint's expert layers, for files
  that don't need it (Q8_0, Miner S4-L0): +6 % on Q8_0.
- **A profiler that works on mining cards.** NVIDIA's profilers refuse the P106 ("does not support NVIDIA Crypto Mining
  Processors"); `GGML_CUDA_PROFILE=<file>` in the fork times every GPU operation with CUDA events.
- The kernels change only the order of additions: the output matches stock llama.cpp within rounding (KL divergence
  0.05–0.08, the same as between two launch shapes of llama.cpp's own kernel), and llama.cpp's operator tests pass.

## How to run it

Build the fork for compute capability 6.1 with CUDA 12.x, on the rig itself (slow on two cores; driver and CUDA setup in
[docs/setup.md](docs/setup.md)) or on any Linux PC with Docker using
[scripts/build_llamacpp_docker.sh](scripts/build_llamacpp_docker.sh) (see [patches/README.md](patches/README.md)):

```bash
git clone --branch pascal-p106 https://github.com/Cinderella-Man/llama.cpp
```

```bash
cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build llama.cpp/build -j 2 --target llama-server
```

Then serve, e.g. Unsloth's Q8_0 on all nine cards with the full context:

```bash
LLAMA_NO_MOE_DOWN_RESCALE=1 llama.cpp/build/bin/llama-server -m flint-Q8_0.gguf -ngl 99 -fa on -c 262144 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 -ts 6,6,5,5,6,5,5,6,6
```

For UD-Q4_K_XL drop `LLAMA_NO_MOE_DOWN_RESCALE=1` (its expert layers need the rescale) and `-ts`. `-ub 128`,
`--cache-ram 0` and a few checkpoints keep the 4 GB host from running out of memory at this context; `-bs` samples on
the GPU instead of sending 150,000 scores per token over the slow link. Miner S4-L0 (Q4) is not published yet.

## The machine

| | |
|---|---|
| GPUs | 9 × NVIDIA P106-090 (GP106 chip, Pascal, compute capability 6.1), 6 GB GDDR5 each |
| Per card | 5 streaming multiprocessors (SMs), ~173 GB/s measured memory bandwidth, ~2 TFLOPS FP32, 75 W power limit |
| Interconnect | Every card on a PCIe 1.0 x1 link (~190 MB/s), eight of them behind two 4-port switches; no GPU-to-GPU access |
| Host | Intel Celeron 3865U (2 cores, 1.8 GHz, no AVX), 4 GB RAM, 64 GB SSD |
| Software | Ubuntu 24.04, NVIDIA driver 580 and CUDA 12.9 (the last versions that support Pascal) |

## More

- [docs/overview.md](docs/overview.md): why a mining rig suits LLMs, the first results with other models (Qwen3.6,
  Qwen3.8, small MoE models), what they taught, and the terms used here.
- [docs/benchmarks/](docs/benchmarks/): one page per benchmark, with setup, commands and full results.
- [quants/](quants/): making and measuring model files for these cards; the plan and every experiment is in
  [quants/experiments.md](quants/experiments.md).
- [bench/mmvq_pascal](bench/mmvq_pascal/): the microbenchmark the kernels were developed in.
- [docs/mining_rig.md](docs/mining_rig.md) (hardware, measured), [docs/setup.md](docs/setup.md) (OS, driver, CUDA,
  llama.cpp), [docs/research/](docs/research/) (surveys and research notes), [scripts/](scripts/) (everything used to
  measure).

## License

The scripts, benchmarks and documentation in this repository are released under the [MIT License](LICENSE). Model
weights, llama.cpp and NVIDIA's libraries keep their own licences: Flint is Apache-2.0, llama.cpp is MIT.
