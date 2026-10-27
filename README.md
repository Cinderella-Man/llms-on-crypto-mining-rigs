# Miner Llama: LLMs on crypto-mining rigs

<p align="center"><img src="docs/images/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="280"></p>

**Retired GPU mining rigs can run today's language models.** A Manli rig with nine NVIDIA P106-090 cards has 54 GB of
GPU memory, but every card sits on a single PCIe 1.0 lane and the host is a 2-core Celeron with 4 GB of RAM, so stock
software runs models slowly on it. Miner Llama makes such rigs useful: a llama.cpp fork tuned for Pascal cards (the
"Pascal engine" in the docs), tested model files and command lines, and every measurement behind the numbers.

## What it does

[Flint](models/flint/README.md), a 39.8B mixture-of-experts model, runs on the nine cards in near-lossless 8-bit
(Unsloth's Q8_0, 42.3 GB) with its full 262,144-token context. A coding agent fixing twelve small bugs, with tool calls
and real test runs ([the agent check](bench/agent_check/README.md)):

| Engine, Unsloth Q8_0 on nine P106-090 | Generation | Tasks passed | First token per turn |
|---|---|---|---|
| Stock llama.cpp, 65,536-token context | 14.4 tok/s | 8 of 12 | 3.7 s |
| Miner Llama v1, 65,536-token context | 25.7 tok/s | 11 of 12 | 3.9 s |
| **Miner Llama v2**, 262,144-token context | **44.6–45.9 tok/s** | **12 of 12** | **2.6 s** |

More speed or fewer cards: [Flint-Pascal-Q5](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) (29.4 GB, fits
six cards) generates 53.8 tok/s in the same test, a little further from the original model than Q8_0. Every number
with its command: [results.md](release/flint-v2/results.md).

## Run it

You need the rig, a drive it reads fast for the model file (a USB 3 stick works, [storing models](docs/setup.md#storing-models)),
and a PC to connect from.

1. **Ubuntu 24.04 and NVIDIA's driver 580**, the last driver for Pascal cards. No CUDA toolkit is needed.

   ```bash
   sudo apt install nvidia-driver-580
   ```

   Reboot; `nvidia-smi` should list your cards ([setup guide](docs/setup.md)).

2. **Miner Llama v2:** from its [GitHub release](https://github.com/Cinderella-Man/llama.cpp/releases/tag/miner-llama-v2.0), download the two archives and `SHA256SUMS` into one
   folder, then:

   ```bash
   sha256sum -c SHA256SUMS
   ```

   ```bash
   tar xzf llama-p106-v2.0-bin-ubuntu-cuda12.9-sm61.tar.gz && tar xzf cudart-llama-p106-v2.0-bin-ubuntu-cuda12.9.tar.gz
   ```

3. **Flint** (42.3 GB) onto the model drive:

   ```bash
   wget -c -P /mnt/models https://huggingface.co/unsloth/flint-GGUF/resolve/main/flint-Q8_0.gguf
   ```

4. **Check the rig** (driver, cards, host memory, the model's read speed):

   ```bash
   git clone https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs
   ```

   ```bash
   BIN=$PWD/llama-p106-v2.0 MODEL=/mnt/models/flint-Q8_0.gguf llms-on-crypto-mining-rigs/scripts/p106_check.sh
   ```

5. **Start the server:**

   ```bash
   llama-p106-v2.0/llama-server -m /mnt/models/flint-Q8_0.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --host 127.0.0.1 --port 8080
   ```

6. **Connect from your PC** through an SSH tunnel, then open `http://127.0.0.1:8080`, or point any OpenAI-compatible
   client or coding agent at it:

   ```bash
   ssh -N -L 8080:127.0.0.1:8080 <your rig>
   ```

What every flag does, six cards, and coding agents: [the Flint page](models/flint/README.md#command-lines). To build
the engine yourself: [reproduce/engine.md](release/flint-v2/reproduce/engine.md). Coming in v2.1: a trained guesser
that makes it faster still ([how it works](docs/drafter.md)).

## Models

| Model | What it is | On the reference rig |
|---|---|---|
| [Flint](models/flint/) | 39.8B mixture of experts, 2.5B active per token, 262,144-token context | v2 release: Q8_0 on nine cards, 44.6–45.9 tok/s for a coding agent |
| [Qwen3.6-35B-A3B](models/qwen3.6-35b-a3b/) | 36B hybrid mixture of experts, 3B active per token | coming: four cards, 29.0 tok/s with v1 (stock 18.5), one request |
| [Ornith-1.5-35B](models/ornith-1.5-35b/) | 35B hybrid mixture of experts with Qwen3.6's shapes | coming: not run yet |
| [Ornith-1.5-122B](models/ornith-1.5-122b/) | 122B hybrid mixture of experts for coding agents | first rig run: a 2–3-bit file on nine cards, 8.8 tok/s for a coding agent with v2 (stock 5.4) |

Each model page gives the file, the command line, speed, quality and what is not measured yet. Model names only
identify the models; this project is not affiliated with their makers.

## What Miner Llama changes

A [fork of llama.cpp](https://github.com/Cinderella-Man/llama.cpp) for GPUs of compute capability 6.1: the P106 and
P104 mining cards, the GTX 10 series and the Tesla P4/P40.
- **v1:** CUDA graphs on Pascal (upstream switches them off before Volta), a launch shape and matrix-vector kernels
  made for 5-SM cards: 1.8 times stock llama.cpp's speed in the agent test.
- **v2:** 32-bit sums in those kernels, one persistent kernel per mixture-of-experts layer, the shared expert fused
  with its activation, tool requests sampled on the GPU, faster prefill, and pipelined speculation: the idle cards of a
  layer split run guessed tokens, and only those the real pass agrees with are kept (`--pipe-depth 4`).
- **The answers don't change:** the kernels change only the order of additions, and speculation gives exactly the
  tokens of plain decoding.

Each change with its measured gain: [the Flint page](models/flint/README.md#how-it-got-here) and
[the fork's scope](docs/research/notes/fork_scope.md). Other Pascal cards run the same code but were not measured. NVIDIA's
profilers refuse mining cards; `GGML_CUDA_PROFILE=<file>` times every GPU operation instead.

## Hardware tuning

The numbers above are at the cards' stock clocks. On this rig, keep `-ub 128` and `--cache-ram 0` at the full context
(the 4 GB host runs out of memory otherwise), keep the host quiet while it works, and hold the NVIDIA packages: a
background driver upgrade breaks CUDA until the next reboot. Clock offsets are measured in a later release.
Details: [hardware tuning](docs/research/notes/hardware_tuning.md).

## The reference rig

| | |
|---|---|
| GPUs | 9 × NVIDIA P106-090 (GP106 chip, Pascal, compute capability 6.1), 6 GB GDDR5 each |
| Per card | 5 streaming multiprocessors (SMs), ~173 GB/s measured memory bandwidth, ~2 TFLOPS FP32, 75 W power limit |
| Interconnect | Every card on a PCIe 1.0 x1 link (~190 MB/s), eight of them behind two 4-port switches; no GPU-to-GPU access |
| Host | Intel Celeron 3865U (2 cores, 1.8 GHz, no AVX), 4 GB RAM, 64 GB SSD |
| Software | Ubuntu 24.04, NVIDIA driver 580 and CUDA 12.9 (the last versions that support Pascal) |

It is a Manli M-P106L9-N6G; thousands of these rigs were built for mining. Measured details: [docs/mining_rig.md](docs/mining_rig.md).

## More

- [release/flint-v2](release/flint-v2/README.md): the v2 release in one folder, with how to rebuild the engine and the
  model file from scratch.
- [docs/overview.md](docs/overview.md): why a mining rig suits LLMs at all, and the terms used here.
- [docs/benchmarks/](docs/benchmarks/), [bench/](bench/) and [scripts/](scripts/): every benchmark and the tools
  behind it.
- [quants/](quants/README.md): making and measuring model files for these cards ([registry](quants/experiments.md)).
- [docs/research/notes/](docs/research/notes/README.md): the research behind each change; [GLOSSARY.md](GLOSSARY.md).

## License

The scripts, benchmarks and documentation in this repository are released under the [MIT License](LICENSE). Model
weights, llama.cpp and NVIDIA's libraries keep their own licences: Flint and Qwen3.6 are Apache-2.0, Ornith-1.5 is MIT,
llama.cpp is MIT.
