<p align="center"><img src="docs/images/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="240"></p>

<h1 align="center">Miner Llama</h1>
<p align="center"><b>Today's LLMs on retired GPU-mining rigs</b><br>
A llama.cpp fork for Pascal cards: nine P106-090s, one PCIe 1.0 lane each, a 2-core Celeron, 4 GB of RAM</p>

<p align="center">
  <a href="https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v4.0"><img alt="release" src="https://img.shields.io/badge/release-v4.0-d4a017"></a>
  <a href="https://huggingface.co/CinderellaMan"><img alt="models" src="https://img.shields.io/badge/models-Hugging%20Face-ffcc4d"></a>
  <a href="LICENSE"><img alt="license" src="https://img.shields.io/badge/license-MIT-blue"></a>
</p>

| A coding agent on nine P106 cards, two runs each | Miner Llama v3 | **Miner Llama v4** |
|---|---:|---:|
| **Qwen3.8-Flash-Next 125B, 2-bit:** tokens written per second | 20.66, 18.92 | **27.61, 26.67** |
| minutes a task, at equal work | 3.08, 3.19 | **2.37, 2.43** |
| 49,152 tokens into a conversation: prompt tok/s | 70.3, 70.6 | **180.7, 180.8** |
| 49,152 tokens into a conversation: tokens written per second | 10.66, 10.98 | **24.26, 24.14** |
| **Ornith-1.5-122B, 2–3-bit:** minutes a task, at equal work | 2.52, 2.54 | **2.11, 2.13** |
| prompt tok/s | 61.1, 58.4 | **103.0, 105.9** |
| **Flint 39.8B, 8-bit:** seconds to the first token of a turn | 2.65, 2.61 | **1.74, 1.82** |
| tokens written per second | 48.55, 48.02 | 49.32, 46.99 |

Every value is one run on the same rigs, each engine with the model's own preset; "equal work" is the same prompt and
reply tokens at each run's own speeds. The files are UkisAI's Swift-1.5 GSQ-RCO Q2_0 and my own
[Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) and
[Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF). Flint and Ornith write one token a
step, so v4 changes how fast they read, not how fast they write ([release notes](release/notes-v4.0.md)).

Against stock llama.cpp, measured for v3 (stock was not measured again for v4):

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

**Miner Llama v4 runs three models on one build, each with its own preset** ([release notes](release/notes-v4.0.md)):

- **[Qwen3.8-Flash-Next](release/flash-next-v4/README.md)**, a 125B-parameter mixture of experts with 6B active per
  token, in UkisAI's 2-bit Swift-1.5 GSQ-RCO Q2_0, with a 65,536-token context. A coding agent gets 27.61 and 26.67
  tok/s, and the speed holds deep into a long conversation.
- **[Flint](release/flint-v4/README.md)**, a 39.8B mixture of experts, in near-lossless 8-bit with its full
  262,144-token context: [Flint Pascal Q8 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF) (40.4 GB)
  or Unsloth's Q8_0 (42.3 GB). [Flint Pascal Q5 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF)
  (29.4 GB) is the fastest file of all, 52.01 tok/s at the full context on v4, and runs on six cards at a shorter
  context.
- **[Ornith-1.5-122B](release/ornith-v4/README.md)** in
  [Ornith-1.5-122B Pascal N2P GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) (48.2 GB),
  whose 2- and 3-bit types only Miner Llama loads, with a 49,152-token context.

Every speed in the tables above comes from two tests. [The agent check](bench/agent_check/README.md) is a coding
agent fixing twelve small bugs in Python and JavaScript, with tool calls, real test runs and a context that grows
every turn, timed over the whole session. [The deep lane](bench/deep_lane/README.md) times the same requests at set
depths of a conversation, up to 49,152 tokens. Each number's command and data are in
[docs/data](docs/data/README.md). The whole story, from stock llama.cpp to v4, with what failed on the way, is told
in [the book](docs/book/00-preface.md).

## Run it

You need the rig, a drive it reads fast for the model file (a USB 3 stick works, [storing models](docs/setup.md#storing-models)),
and a PC to connect from.

1. **Ubuntu 24.04 and NVIDIA's driver 580**, the last driver for Pascal cards. No CUDA toolkit is needed.

   ```bash
   sudo apt install nvidia-driver-580
   ```

   Reboot; `nvidia-smi` should list your cards ([setup guide](docs/setup.md)).

2. **Miner Llama v4:** from its [GitHub release](https://github.com/Cinderella-Man/miner-llama/releases/tag/miner-llama-v4.0),
   download the two archives and `SHA256SUMS` into one folder, then:

   ```bash
   sha256sum -c --ignore-missing SHA256SUMS
   ```

   ```bash
   tar xzf llama-p106-v4.0-bin-ubuntu-cuda12.9-sm61.tar.gz && tar xzf cudart-llama-p106-v4.0-bin-ubuntu-cuda12.9.tar.gz
   ```

3. **A model** onto the model drive. Flint in my Flint Pascal Q8 GGUF (40.4 GB), the file Flint's v4 preset names:

   ```bash
   wget -c -P /mnt/models https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF/resolve/main/flint-pascal-Q8.gguf
   ```

   The [Qwen3.8-Flash-Next](release/flash-next-v3/README.md#run-it) and
   [Ornith-1.5-122B](release/ornith-v3/README.md#the-file) pages list their own files, and
   [the Flint page](release/flint-v3/README.md#the-files) Unsloth's Q8_0.

4. **Check the rig** (driver, cards, host memory, the model's read speed):

   ```bash
   git clone https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs
   ```

   ```bash
   BIN=$PWD/llama-p106-v4.0 MODEL=/mnt/models/flint-pascal-Q8.gguf llms-on-crypto-mining-rigs/scripts/p106_check.sh
   ```

5. **Start the server with the model's preset** (Flint here; the other models have their own `settings.env`):

   ```bash
   MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/flint-v4/settings.env && env $PRESET_ENV llama-p106-v4.0/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
   ```

6. **Connect from your PC** through an SSH tunnel, then open `http://127.0.0.1:8080`, or point any OpenAI-compatible
   client or coding agent at it:

   ```bash
   ssh -N -L 8080:127.0.0.1:8080 <your rig>
   ```

What each preset sets and why: the [Qwen3.8-Flash-Next](release/flash-next-v4/README.md),
[Flint](release/flint-v4/README.md) and [Ornith-1.5-122B](release/ornith-v4/README.md) pages. To build the engine
yourself: `TESTS=1 scripts/build_llamacpp_docker.sh miner-llama-v4.0`
([reproduce/engine.md](release/flint-v2/reproduce/engine.md) walks through it for v2.0). v3.0 and its presets
(`release/<model>-v3/`) keep working as they are.

## Models

| Model | What it is | On the reference rig |
|---|---|---|
| [Flint](models/flint/) | 39.8B mixture of experts, 2.5B active per token, 262,144-token context | v4 release: Flint Pascal Q8 GGUF on nine cards, 49.32 and 46.99 tok/s for a coding agent, the first token of a turn after 1.74 and 1.82 s (v3: 48.55 and 48.02 tok/s, 2.65 and 2.61 s; Unsloth's Q8_0 on v3: 45.9 tok/s) |
| [Qwen3.8-Flash-Next](models/qwen3.8-flash-next/) | 125B mixture of experts, ~6B active per token, plus a 51B-parameter n-gram table | v4 release: UkisAI's Swift-1.5 Q2_0 on nine cards, the table on the SSD, the MTP draft; 27.61 and 26.67 tok/s and 12 of 12 tasks twice for a coding agent (v3: 20.66 and 18.92; stock llama.cpp b12076: 8.20 on the same file) |
| [Ornith-1.5-122B](models/ornith-1.5-122b/) | 122B hybrid mixture of experts for coding agents | v4 release: Ornith-1.5-122B Pascal N2P GGUF (48.2 GB) on nine cards, 16.52 and 16.13 tok/s and 12 of 12 tasks twice for a coding agent, a task in 2.11 and 2.13 minutes at equal work (v3: 2.52 and 2.54; stock llama.cpp b12076: 6.0 tok/s on Unsloth's UD-Q2_K_XL) |

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
- **v4:** a newer upstream base (b12360), attention that reads only the cache cells the model picks, kernels that
  read each weight once for the pass that checks guessed tokens, less host work between two steps, and checkpoints
  kept on the cards and taken without stopping the pipeline. Flash-Next writes 58.4 % faster and reads prompts
  139.4 % faster than on v3 over a conversation of up to 49,152 tokens; Flint and Ornith read prompts faster
  ([release notes](release/notes-v4.0.md), [how v4 came about](docs/book/16-v4.md)).
- **The answers stay the model's own:** the kernels change only the order of additions, and a guessed token is kept
  only when it is the one the model picks itself, or, on Flash-Next's v4 preset, one it accepts by its own
  probabilities. Flint's pipelined speculation gives plain decoding's tokens bit for bit, and with the same seed v4
  writes v3's tokens on Flint. Flash-Next amplifies the last bits, so at temperature 0 its text can differ from
  plain decoding's, and v4's differs from v3's
  ([why](docs/book/13-the-check-pass.md#134-story-a-check-was-a-different-arithmetic)). One v4 prompt switch, the
  micro-batch taper, changes Ornith's numbers about as much as its file's quantization does, so Ornith's preset
  leaves it off ([the measurement](docs/book/16-v4.md#169-story-results-unchanged)).

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

- [release/notes-v4.0.md](release/notes-v4.0.md): the v4 release, and its three models:
  [Qwen3.8-Flash-Next](release/flash-next-v4/README.md), [Flint](release/flint-v4/README.md) and
  [Ornith-1.5-122B](release/ornith-v4/README.md), each with its preset, one command and its tests against v3.0.
- [release/notes-v3.0.md](release/notes-v3.0.md): the v3 release, and its three models:
  [Qwen3.8-Flash-Next](release/flash-next-v3/README.md), [Flint](release/flint-v3/README.md) and
  [Ornith-1.5-122B](release/ornith-v3/README.md), each with the files to download, its preset and one command.
- [release/flint-v2](release/flint-v2/README.md): the v2 release in one folder, with how to rebuild the engine and the
  model file from scratch.
- [docs/book/](docs/book/00-preface.md): the project as a book for backend developers, from the rig to v4: what I
  tried, how each idea was meant to work, what happened and why.
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
