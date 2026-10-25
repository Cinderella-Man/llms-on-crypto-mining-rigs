# LLMs on crypto-mining rigs

<p align="center"><img src="docs/images/miner-llama.jpg" alt="Miner Llama: a llama in a miner's helmet over two crossed pickaxes" width="280"></p>

**Thousands of retired GPU mining rigs built from NVIDIA Pascal cards are collecting dust.** They hold a lot of fast GPU
memory: a Manli rig with nine P106-090 cards has 54 GB. But every card hangs on a single PCIe 1.0 lane, has only a few
compute units, and the host is a 2-core Celeron with 4 GB of RAM. Stock software runs today's language models slowly on
such hardware, so the rigs sit idle. This project makes them useful: a llama.cpp fork tuned for Pascal cards (the
**Pascal engine**), tested files and command lines for each model, and every measurement behind the numbers.

## v2: the Flint release

[Flint](models/flint/README.md) is a 39.8 B-parameter mixture-of-experts model that reasons before it answers and reads
up to 262,144 tokens of context. On the nine cards of the [reference rig](#the-reference-rig), the v2 engine runs it
RIG-TBD times as fast as stock llama.cpp, with the file most people already have:

<!-- RIG-TBD: the stock-clock block of today's bench/release_matrix run on the release engine -->

| Nine P106-090 cards, Unsloth's UD-Q4_K_XL, tokens per second | Stock llama.cpp | v1 engine | **v2** |
|---|---|---|---|
| Generation (`llama-bench`, 128 tokens) | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Served chat | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Served chat, pipelined speculation (`--pipe-depth 3`) | — | — | **RIG-TBD** |
| Prompt processing, 512 tokens | RIG-TBD | RIG-TBD | **RIG-TBD** |

<!-- begin:drafter -->
**With the staircase drafter**, served chat reaches **RIG-TBD tok/s** on nine cards and **RIG-TBD** on six. The drafter
is a small trained head, a 0.29 GB side file shipped with the release, that guesses Flint's next token from the hidden
states of the tokens already in flight; the cards check every guess
([how it works](models/flint/README.md#the-staircase-drafter)).
<!-- end:drafter -->

The answers don't change: the engine's kernels change only the order of additions, and pipelined speculation gives
exactly the tokens of plain decoding. To run it:
1. **The engine:** the two archives of the fork's release RIG-TBD (tag and link), unpacked into one folder; the rig
   needs NVIDIA's driver 580 and no CUDA toolkit ([Run it](#run-it)).
2. **The model:** Unsloth's [UD-Q4_K_XL](https://huggingface.co/unsloth/flint-GGUF) (25.1 GB). RIG-TBD: our own
   K2h-r16 file too, if the release includes it.
3. **The command line** on the [Flint page](models/flint/README.md#command-lines), with what every flag does.
4. **The drafter** (optional): the release's side file, `--pipe-guess staircase --pipe-draft <file>`. <!-- only:drafter -->

## Models

Generation speed in tokens per second (tok/s) for one conversation, measured on the [reference rig](#the-reference-rig):

| Model | What it is | Speed | Page |
|---|---|---|---|
| [![Flint](https://img.shields.io/badge/Flint-v2%20release-2b6cb0)](models/flint/) | 39.8 B mixture of experts, 2.5 B parameters per token, 262,144-token context | Nine cards, v2: **RIG-TBD tok/s** generation with Unsloth's UD-Q4_K_XL, against RIG-TBD on stock llama.cpp; **RIG-TBD** served chat with pipelined speculation | [models/flint](models/flint/) |
| [![Qwen3.6-35B-A3B](https://img.shields.io/badge/Qwen3.6--35B--A3B-coming%20soon-718096)](models/qwen3.6-35b-a3b/) | 36 B hybrid mixture of experts, 3 B per token, 262,144-token context | Four cards: 29.0 tok/s with the v1 engine, against 18.5 on stock llama.cpp | [models/qwen3.6-35b-a3b](models/qwen3.6-35b-a3b/) |
| [![Ornith-1.5-35B](https://img.shields.io/badge/Ornith--1.5--35B-coming%20soon-718096)](models/ornith-1.5-35b/) | 35 B hybrid mixture of experts with Qwen3.6's shapes, 3 B per token, 262,144-token context | Not run yet; its 21.7 GB Q4_K_M fits four or five cards (est. 19–23 tok/s today, 30–36 with the planned K-quant kernels) | [models/ornith-1.5-35b](models/ornith-1.5-35b/) |
| [![Ornith-1.5-122B](https://img.shields.io/badge/Ornith--1.5--122B-coming%20soon-718096)](models/ornith-1.5-122b/) | 122 B hybrid mixture of experts, 10 B per token, for coding agents | Not run yet; fits nine cards only at 2–3 bits per weight (est. 8–16 tok/s) | [models/ornith-1.5-122b](models/ornith-1.5-122b/) |

Each model page gives the file to download, the command line, measured speed and quality, and what is not measured
yet. Model names only identify the models; this project is not affiliated with their makers.

## The Pascal engine

The engine is a [fork of llama.cpp](https://github.com/Cinderella-Man/llama.cpp) for GPUs of compute capability 6.1:
the P106 and P104 mining cards, the GTX 10 series and the Tesla P4/P40. Upstream llama.cpp switches CUDA graphs off for
every GPU older than Volta, and its matrix-vector kernels leave a 5-SM card waiting on instruction issue rather than on
memory. The fork changes that, mostly without regard to which model runs:

| Change | Which models gain | Measured on Flint |
|---|---|---|
| CUDA graphs on Pascal: each card's few hundred GPU operations per token replayed as one graph ([upstream PR #27721](https://github.com/ggml-org/llama.cpp/pull/27721)) | every model | 17.6 → 19.5 tok/s |
| A launch shape for one-token products: 2 warps per block on wide rows, 4 on narrow ones | every model | → 24.5 tok/s |
| Pascal matrix-vector kernels for Q4_0, Q4_1, Q5_0, Q8_0, Q5_K and IQ4_XS: lane groups sized to each row, several rows per input load, wide coalesced loads ([design](docs/research/notes/pascal_mmvq_integration.md)) | every file made of these types (Qwen3.6 Q4_0: +27 %) | → 34.1 tok/s (v1) |
| 32-bit activation sums in those kernels, so llama.cpp's MoE safety rescale can go | the models llama.cpp rescales (Flint among them) | +9.6 % |
| Shared-expert projections fused with their activation | MoE models with a shared expert | +0.7 % |
| MoE prompt tiles sized by tokens per expert (upstream PR #31047) | MoE models, prompts | prompts +22–29 % |
| One persistent kernel per MoE layer at one token, instead of 13 launches | Flint's layer shapes; other models fall back to the normal path | +5.6–8.8 % |
| A Pascal decode-attention kernel for one token | attention of Flint's shape (head size 128, 8 query heads per KV head); other shapes take the normal path | RIG-TBD |
| Pascal matrix-vector kernels for Q4_K and Q6_K too | the common Q4_K_M downloads of other models | Flint's UD-Q4_K_XL: est. ≤ 1 % |
| Pipelined speculation in llama-server (`--pipe-depth`): guessed tokens enter the idle cards of a layer split, checked against the real token | models with an attention-only cache (hybrid models next) | served chat 1.04–1.18×, edits up to 1.56×, rewrites up to 2.79× |
| A trained drafter as the guesser of pipelined speculation (`--pipe-guess staircase`): a small head that reads the hidden states of the tokens in flight | Flint (a drafter is trained per model) | RIG-TBD | <!-- only:drafter -->
| Tool requests sampled on the GPU, with the grammar checked on the host | every model, agent traffic | RIG-TBD (they decoded 12–16 % slower than chat before) |
| The attention mask built on each card | models with an attention-only cache, long contexts | decode at 64k tokens of context +12 % |
| A Pascal prompt-attention kernel that skips fully masked key tiles and reads a Q8_0 cache directly | attention with head size 128 or 256 and 8 query heads per KV head | long prompts RIG-TBD | <!-- only:q163-q111 -->
| Prompt caching for agents: the shared system and tool prompt restored from a file (`--prefix-cache`), checkpoints that hold the whole sliding-window ring | agent sessions that re-send long prompts | RIG-TBD | <!-- only:q163-q111 -->

The first three rows were measured on six cards with UD-Q4_K_XL, step by step; the rest are each change's own gain
([Flint page](models/flint/#how-it-got-here)). Of v2's decode gain on Flint, about 87 % comes from the Pascal-only changes
([fork scope](docs/research/notes/fork_scope.md)).

- **The output stays the same.** The kernels change only the order of additions: outputs match stock llama.cpp within
  rounding, and llama.cpp's operator tests pass. The v2 additions leave the output unchanged, and pipelined speculation
  gives exactly the tokens of plain decoding.
- **Other Pascal cards** (P106-100, P104-100, GTX 1060–1080 Ti, P40) run the same code. They have more compute units
  per byte of memory bandwidth than the P106-090, so the kernels likely gain less there; none was measured. The P100
  (compute capability 6.0) gets none of the Pascal pieces.
- **A profiler that works on mining cards.** NVIDIA's profilers refuse the P106 ("does not support NVIDIA Crypto Mining
  Processors"); `GGML_CUDA_PROFILE=<file>` times every GPU operation with CUDA events.

## Hardware tuning

What the cards and the host can give on top of the software. These settings apply to stock llama.cpp as much as to the
fork. Every engine number elsewhere in this repository is measured without them, and the tables below never mix the
two: each engine at stock clocks, each engine tuned, then what the tuning adds to each.

<!-- RIG-TBD: replace the three tables with tables.md from bench/release_matrix (run of 2026-10-25); drop the K2h-r16
columns if the release ships Unsloth's file only; if the release ships before the tuned run, keep the stock-clock table
and say the tuned tables follow -->


**Stock clocks**, nine cards, Flint, tokens per second:

| Engine | UD-Q4_K_XL generation | UD-Q4_K_XL served | K2h-r16 generation | K2h-r16 served |
|---|---|---|---|---|
| Stock llama.cpp | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v1 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine, pipelined speculation (`--pipe-depth 3`) | — | RIG-TBD | — | RIG-TBD |

**Tuned: memory +RIG-TBD MT/s on every card**, nine cards, Flint, tokens per second:

| Engine | UD-Q4_K_XL generation | UD-Q4_K_XL served | K2h-r16 generation | K2h-r16 served |
|---|---|---|---|---|
| Stock llama.cpp | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v1 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine, pipelined speculation (`--pipe-depth 3`) | — | RIG-TBD | — | RIG-TBD |

**What the tuning adds**, each engine against itself at stock clocks:

| Engine | UD-Q4_K_XL generation | UD-Q4_K_XL served | K2h-r16 generation | K2h-r16 served |
|---|---|---|---|---|
| Stock llama.cpp | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v1 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine | RIG-TBD | RIG-TBD | RIG-TBD | RIG-TBD |
| v2 engine, pipelined speculation (`--pipe-depth 3`) | — | RIG-TBD | — | RIG-TBD |

Generation is `llama-bench`: 128 tokens right after a 512-token prompt, so the cards are at their working clocks.
Served is llama-server with the model page's chat command line at the full 262,144-token context: the median of chat
requests of up to 1,000 tokens with the makers' sampling. With the offset, every engine's greedy output stayed identical
to its output at stock clocks (RIG-TBD). Method and script: [bench/release_matrix](bench/release_matrix/README.md).

The settings, one by one:

| Setting | What it does on this rig | Status |
|---|---|---|
| **Keep the cards busy** | In a layer split each card idles most of every token and drops to its 1,354 MHz base clock: the same chat prompt decodes at 34.0 tok/s on base-clocked cards against 38.5 at boost. Pipelined speculation keeps tokens in every card; with every guess wrong, nine cards still run 1.24× faster | measured ([idle cards](docs/research/notes/idle_cards.md)) |
| Memory clock offset through NVML | est. +3–5 % decode: only about half of a token's time streams weights. Applied at run time, cleared by offset 0 or a reboot; [the test kit](bench/mem_offset/) accepts an offset only if the model's output stays bit-identical, since the cards have no ECC | test kit ready, not measured yet |
| Core clock offset | est. +0.5–2 % decode, +3–5 % prompts; same protocol | planned |
| PowerMizer "Prefer Maximum Performance" | nothing: 29.75 tok/s from idle with it against 29.79 without, 33.25 against 33.30 served | measured, not needed |
| Power limit | Busy cards draw 35–39 W of their 75 W limit in decode (317–329 W for nine, ~8.9 J per token, hottest card 57 °C); the limit is the cards' maximum and almost never binds | measured |
| PCIe 2.0 | not possible: the cards themselves advertise only 2.5 GT/s | checked |
| Host memory (4 GB) | At 262,144 tokens of context llama.cpp's default micro-batch pins ~1 GiB of host memory and the out-of-memory killer takes the server; `-ub 128` and `--cache-ram 0` avoid it with generation speed unchanged (prompts run slower than at `-ub 512`) | measured |
| A quiet host | No apt, snap or copy jobs during timings, and the NVIDIA packages held: a background driver upgrade breaks CUDA until the next reboot | recommended |

Details: [hardware tuning](docs/research/notes/hardware_tuning.md), [power and tuning](docs/research/notes/power_and_tuning.md).

## Run it

1. **Driver and system:** Ubuntu 24.04 with NVIDIA driver 580, the last driver branch for Pascal
   ([setup](docs/setup.md)).
2. **Get the engine.** The prebuilt release needs only the driver: download the two archives of the fork's release
   RIG-TBD (tag and link once published), check them with `sha256sum -c SHA256SUMS`, and unpack both into one folder;
   then run [scripts/p106_check.sh](scripts/p106_check.sh), which checks the driver, the cards, host memory, the model's
   read speed and the shipped command lines. Or build it for compute capability 6.1 with CUDA 12.x, on the rig itself
   (slow on two cores) or on any Linux PC with Docker using
   [scripts/build_llamacpp_docker.sh](scripts/build_llamacpp_docker.sh).
   The v2 engine is commit `ab4a5cd02` of the fork (RIG-TBD: the release branch or tag). <!-- only:rc4 -->
   The v2 engine is commit `7e6ec762f` of the fork (RIG-TBD: the release branch or tag). <!-- only:rc3 -->
   The v2 engine is commit `594d20e08` of the fork (RIG-TBD: the release branch or tag). <!-- only:rc1 -->
   v1, as released on 2026-10-23, is `pascal-p106`.

   ```bash
   git clone --branch RIG-TBD https://github.com/Cinderella-Man/llama.cpp
   ```

   ```bash
   cmake -S llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61 && cmake --build llama.cpp/build -j 2 --target llama-server
   ```

3. **Pick a model page** for the file and the command line: [Flint](models/flint/#command-lines).

## The reference rig

| | |
|---|---|
| GPUs | 9 × NVIDIA P106-090 (GP106 chip, Pascal, compute capability 6.1), 6 GB GDDR5 each |
| Per card | 5 streaming multiprocessors (SMs), ~173 GB/s measured memory bandwidth, ~2 TFLOPS FP32, 75 W power limit |
| Interconnect | Every card on a PCIe 1.0 x1 link (~190 MB/s), eight of them behind two 4-port switches; no GPU-to-GPU access |
| Host | Intel Celeron 3865U (2 cores, 1.8 GHz, no AVX), 4 GB RAM, 64 GB SSD |
| Software | Ubuntu 24.04, NVIDIA driver 580 and CUDA 12.9 (the last versions that support Pascal) |

It is a Manli M-P106L9-N6G; thousands of these rigs were built for mining. Measured hardware details: [docs/mining_rig.md](docs/mining_rig.md).

## More

- [docs/overview.md](docs/overview.md): why a mining rig suits LLMs at all, the first measurements with several models,
  and the terms used here.
- [docs/benchmarks/](docs/benchmarks/): one page per benchmark, with setup, commands and full results.
- [quants/](quants/README.md): making and measuring model files for these cards; every experiment and decision is in the
  [registry](quants/experiments.md).
- [docs/research/notes/](docs/research/notes/README.md): research notes behind each change, from kernels to release
  settings.
- [bench/](bench/) and [scripts/](scripts/): the tools every measurement here was taken with;
  [patches/](patches/README.md): patches for stock llama.cpp; [GLOSSARY.md](GLOSSARY.md): terms.

## License

The scripts, benchmarks and documentation in this repository are released under the [MIT License](LICENSE). Model
weights, llama.cpp and NVIDIA's libraries keep their own licences: Flint and Qwen3.6 are Apache-2.0, Ornith-1.5 is MIT,
llama.cpp is MIT.
