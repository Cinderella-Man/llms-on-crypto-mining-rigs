<p align="center"><img src="docs/images/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="240"></p>

<h1 align="center">Miner Llama</h1>
<p align="center"><b>Today's LLMs on retired GPU-mining rigs</b><br>
A llama.cpp fork for Pascal cards: nine P106-090s, one PCIe 1.0 lane each, a 2-core Celeron, 4 GB of RAM</p>

<p align="center">
  <a href="https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v3.0"><img alt="release" src="https://img.shields.io/badge/release-v3.0-d4a017"></a>
  <a href="https://huggingface.co/CinderellaMan"><img alt="models" src="https://img.shields.io/badge/models-Hugging%20Face-ffcc4d"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

| A coding agent, tok/s | Stock llama.cpp b12076 | **Miner Llama v3** | **Miner Llama v3, my quant** |
|---|---:|---:|---:|
| Qwen3.8-Flash-Next 125B, 2-bit | 8.20 | **20.40** (2.48×) | — |
| Flint 39.8B, 8-bit | 14.4 | **45.9** (3.18×) | **47.4**<br>[Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF) |
| Flint 39.8B, 5-bit | 13.7 | **46.4** (3.38×) | **49.4**<br>[Flint Pascal Q5 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) |
| Ornith-1.5-122B, 2-bit | 6.0 | **12.6** (2.10×) | **16.3**<br>[Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) |

The public files are UkisAI's Swift-1.5 GSQ-RCO Q2_0 and Unsloth's Q8_0, UD-Q5_K_XL and UD-Q2_K_XL. Stock llama.cpp
runs Flint at 65,536 tokens of context, because at 262,144 its attention mask does not fit in the 4 GB host; Miner Llama
runs the full 262,144. Stock reached 9 of Flash-Next's 12 tasks in its time budget; on those nine Miner Llama v3
generated 20.25 tok/s. Flint Pascal Q5 GGUF's 49.4 includes one long session (complex-numbers, 14,024 tokens); on the other
eleven tasks it generated 53.64 tok/s, as on v2.0 (54.04). Ornith's 16.3 ran on `1f7215948`, the release branch before
its last fix; on the release build `1551f5fe1` the same preset ran 16.42 on a second rig.


**Retired GPU mining rigs can run today's language models.** A Manli rig with nine NVIDIA P106-090 cards has 54 GB of
GPU memory, but every card sits on a single PCIe 1.0 lane and the host is a 2-core Celeron with 4 GB of RAM, so stock
software runs models slowly on it. Miner Llama makes such rigs useful: a llama.cpp fork tuned for Pascal cards (the
"Pascal engine" in the docs), tested model files and command lines, and every measurement behind the numbers.

## What it does

**Miner Llama v3 runs three models on one build, each with its own preset** ([release notes](release/notes-v3.0.md)):

- **[Qwen3.8-Flash-Next](release/flash-next-v3/README.md)**, a 125B-parameter mixture of experts with 6B active per
  token, in UkisAI's 2-bit Swift-1.5 GSQ-RCO Q2_0, with a 65,536-token context.
- **[Flint](release/flint-v3/README.md)**, a 39.8B mixture of experts, in near-lossless 8-bit (Unsloth's Q8_0,
  42.3 GB) with its full 262,144-token context.
  [Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF) (40.4 GB) generates 3.3 % faster,
  and [Flint Pascal Q5 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF) (29.4 GB) is the fastest file of all, 49.4 tok/s at
  the full context, and runs on six cards at a shorter context.
- **[Ornith-1.5-122B](release/ornith-v3/README.md)** in
  [Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) (48.2 GB),
  whose 2- and 3-bit types only Miner Llama loads, with a 49,152-token context.

Every speed in the table above comes from [the agent check](bench/agent_check/README.md): a coding agent fixing twelve
small bugs in Python and JavaScript, with tool calls, real test runs and a context that grows every turn, timed over
the whole session. Each number's command and data are in [docs/data](docs/data/README.md). Flint's path from stock
llama.cpp through v1 to v2 is in [release/flint-v2/results.md](release/flint-v2/results.md).

## Run it

You need the rig, a drive it reads fast for the model file (a USB 3 stick works, [storing models](docs/setup.md#storing-models)),
and a PC to connect from.

1. **Ubuntu 24.04 and NVIDIA's driver 580**, the last driver for Pascal cards. No CUDA toolkit is needed.

   ```bash
   sudo apt install nvidia-driver-580
   ```

   Reboot; `nvidia-smi` should list your cards ([setup guide](docs/setup.md)).

2. **Miner Llama v3:** from its [GitHub release](https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v3.0),
   download the two archives and `SHA256SUMS` into one folder, then:

   ```bash
   sha256sum -c --ignore-missing SHA256SUMS
   ```

   ```bash
   tar xzf llama-p106-v3.0-bin-ubuntu-cuda12.9-sm61.tar.gz && tar xzf cudart-llama-p106-v3.0-bin-ubuntu-cuda12.9.tar.gz
   ```

3. **A model** onto the model drive. Flint in Unsloth's Q8_0 (42.3 GB):

   ```bash
   wget -c -P /mnt/models https://huggingface.co/unsloth/flint-GGUF/resolve/20468edfc38ab43d6914cc88b08cbecb05d0fc62/flint-Q8_0.gguf
   ```

   The [Qwen3.8-Flash-Next](release/flash-next-v3/README.md) and [Ornith-1.5-122B](release/ornith-v3/README.md) pages
   list their own files.

4. **Check the rig** (driver, cards, host memory, the model's read speed):

   ```bash
   git clone https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs
   ```

   ```bash
   BIN=$PWD/llama-p106-v3.0 MODEL=/mnt/models/flint-Q8_0.gguf llms-on-crypto-mining-rigs/scripts/p106_check.sh
   ```

5. **Start the server with the model's preset** (Flint here; the other models have their own `settings.env`):

   ```bash
   MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/flint-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
   ```

6. **Connect from your PC** through an SSH tunnel, then open `http://127.0.0.1:8080`, or point any OpenAI-compatible
   client or coding agent at it:

   ```bash
   ssh -N -L 8080:127.0.0.1:8080 <your rig>
   ```

What each preset sets and why: the [Qwen3.8-Flash-Next](release/flash-next-v3/README.md),
[Flint](release/flint-v3/README.md) and [Ornith-1.5-122B](release/ornith-v3/README.md) pages. To build the engine
yourself: `TESTS=1 scripts/build_llamacpp_docker.sh miner-llama-v3.0`
([reproduce/engine.md](release/flint-v2/reproduce/engine.md) walks through it for v2.0).

## Models

| Model | What it is | On the reference rig |
|---|---|---|
| [Flint](models/flint/) | 39.8B mixture of experts, 2.5B active per token, 262,144-token context | v3 release: Q8_0 on nine cards, 45.9 tok/s for a coding agent (v2.0: 45.9 and 44.6 in two runs) |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next/) | 125B mixture of experts, ~6B active per token, plus a 51B-parameter n-gram table | v3 release: UkisAI's Swift-1.5 Q2_0 on nine cards, the table on the SSD, the MTP draft; 20.40 tok/s and 12 of 12 tasks for a coding agent (stock llama.cpp b12076 8.20 on the same file) |
| [Ornith-1.5-122B](models/ornith-1.5-122b/) | 122B hybrid mixture of experts for coding agents | v3 release: Ornith-1.5-122B Pascal N2P GGUF (48.2 GB) on nine cards, 16.3 tok/s and 12 of 12 tasks for a coding agent (stock llama.cpp b12076 6.0 on Unsloth's UD-Q2_K_XL) |

Each model page gives the file, the command line, speed, quality and what is not measured yet. Model names only
identify the models; this project is not affiliated with their makers.

## What Miner Llama changes

A [fork of llama.cpp](https://github.com/Cinderella-Man/miner-llama) for GPUs of compute capability 6.1: the P106 and
P104 mining cards, the GTX 10 series and the Tesla P4/P40.
- **v1:** CUDA graphs on Pascal (upstream switches them off before Volta), a launch shape and matrix-vector kernels
  made for 5-SM cards: 1.8 times stock llama.cpp's speed in the agent test.
- **v2:** 32-bit sums in those kernels, one persistent kernel per mixture-of-experts layer, the shared expert fused
  with its activation, tool requests sampled on the GPU, faster prefill, and pipelined speculation: the idle cards of a
  layer split run guessed tokens, and only those the real pass agrees with are kept (`--pipe-depth 4`).
- **v3:** Pascal kernels for BF16, 2-bit and my own N2P/N3P weights and for 1–8-token steps, the model's state cast to
  8 bits between cards, an MTP draft with a cheaper output head, a keep-alive that holds every card at its boost clock,
  and every feature behind its own switch, with one preset per model ([release notes](release/notes-v3.0.md),
  [what v3 changed and why](docs/book/10-v3.md)).
- **v4 (in progress, not released):** a newer upstream base, attention that reads only the cache cells the model
  picks, kernels for the pass that checks guessed tokens, and checkpoints kept on the cards and taken without stopping
  the pipeline. On 2026-11-11 this line wrote 45.97 % faster than v3 over a conversation of up to 49,152 tokens and
  took 9.4 % less time for a coding agent's task, on the same rig the same day
  ([the work since v3](docs/book/11-after-v3.md)).
- **The answers stay the model's own:** the kernels change only the order of additions, and a guessed token is kept
  only when it is the one the model picks itself. Flint's pipelined speculation gives plain decoding's tokens bit for
  bit. Flash-Next's draft is checked several tokens at a time, and such a pass adds the same numbers in another order
  than a one-token pass. That model amplifies the last bits, so at temperature 0 its text can differ from plain
  decoding's ([why](docs/book/13-the-check-pass.md#134-story-a-check-was-a-different-arithmetic)).

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
- [release/notes-v3.0.md](release/notes-v3.0.md): the v3 release, and its three models:
  [Qwen3.8-Flash-Next](release/flash-next-v3/README.md), [Flint](release/flint-v3/README.md) and
  [Ornith-1.5-122B](release/ornith-v3/README.md), each with the files to download, its preset and one command.
- [docs/overview.md](docs/overview.md): why a mining rig suits LLMs at all, and the terms used here.
- [docs/what_i_tried.md](docs/what_i_tried.md): a guided tour of everything tried in the first week, why, what each
  result taught, and where each model stands.
- [docs/benchmarks/](docs/benchmarks/), [bench/](bench/) and [scripts/](scripts/): every benchmark and the tools
  behind it.
- [quants/](quants/README.md): making and measuring model files for these cards ([registry](quants/experiments.md)).
- [docs/research/notes/](docs/research/notes/README.md): the research behind each change; [GLOSSARY.md](GLOSSARY.md).
- [docs/data/](docs/data/README.md): the data and method behind every experiment (tables, exact commands, run scripts),
  and [docs/engine_features.md](docs/engine_features.md): each v2.0 engine feature, how it was checked and measured.
- [docs/switches.md](docs/switches.md): every switch of the engine, its default, what it does, the branch that brought
  it and its registry row, so each change can be turned off on its own.

## License

The scripts, benchmarks and documentation in this repository are released under the [MIT License](LICENSE). Model
weights, llama.cpp and NVIDIA's libraries keep their own licences: Flint is Apache-2.0, Ornith-1.5 is MIT,
llama.cpp is MIT.
