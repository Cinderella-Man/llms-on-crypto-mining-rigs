---
license: apache-2.0
base_model: RecursiveSI/flint
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - flint
  - pascal
  - mining-rig
---

# Flint K2h-r16: Flint for old NVIDIA cards and crypto-mining rigs

**What this is.** [Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8-billion-parameter AI model that thinks
before it answers, writes code and calls tools, and reads up to 262,144 tokens of context. This file is Flint packed
into 29.4 GB so that it runs fast on old NVIDIA "Pascal" cards: the P106 and P104 cards of retired mining rigs, the
GTX 10 series, and the Tesla P4/P40.

**Why it is faster there.** A model file stores its weights compressed. Modern graphics cards unpack any compression
format almost for free, so the popular files use elaborate formats that save a little space. Old Pascal mining cards
have few compute units and spend real time unpacking those formats. This file uses only the two simplest formats
(Q5_0 and Q8_0), which the [Pascal engine](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs) reads at full
speed. The formats were chosen part by part, so the file stays closer to the original model than the popular 4-bit
download, Unsloth's UD-Q4_K_XL.

**How fast.** On a Manli mining rig with nine P106-090 cards (6 GB each, every card on a single PCIe 1.0 lane, a
2-core Celeron host): RIG-TBD tokens per second, against RIG-TBD on stock llama.cpp with the usual file.

**Which Flint file should you pick?**
- **Nine 6 GB cards, best quality: Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF)** (42.3 GB). It is
  near-lossless (it picks the original model's next word 93.3 % of the time), it fits the full 262,144-token context
  on nine cards, and with the Pascal engine it is not slower than the 4-bit files: 29.55 tokens per second against
  27.97 for UD-Q4_K_XL in a long chat on the v1 engine (v2: RIG-TBD). This is the headline of the release.
- **Six cards, or more room for the context: this file.** It is a third smaller than Q8_0, closer to the original than
  UD-Q4_K_XL, and faster than it on the same engine.

## Run it in three steps

1. **Driver:** Ubuntu 24.04 or newer with NVIDIA's driver 580, the last one for Pascal cards
   ([setup guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/setup.md)).
2. **Download** the Pascal engine (two archives from the
   [release page](https://github.com/Cinderella-Man/llama.cpp/releases), RIG-TBD: tag), unpack both into one folder,
   and download `flint-K2h-r16.gguf` from this page.
3. **Start it**, here on nine cards with the full context:

   ```bash
   llama-p106-v2.0/llama-server -m flint-K2h-r16.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 3 --host 127.0.0.1 --port 8080
   ```

   Then open `http://127.0.0.1:8080` through an SSH tunnel from your PC (`ssh -N -L 8080:127.0.0.1:8080 <your rig>`),
   or point any OpenAI-compatible client at it.

What each flag does, how to run on six cards, and how to use it for coding agents:
[the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md).
For even more speed, add the trained guesser from
[the drafter's page](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter).

## What you need

- **About 30 GB of GPU memory plus room for the context:** five or six 6 GB cards for short conversations, all nine
  for the full 262,144 tokens.
- **The Pascal engine** for the speed. The file also works in stock llama.cpp (release b11803 or newer) and in apps
  built on it, just without the Pascal speed-up.
- **On a modern GPU or a Mac, use Unsloth's UD-Q4_K_XL instead:** it is smaller for the same quality there.

## What we measured

| | This file | Unsloth UD-Q4_K_XL | Unsloth Q8_0 |
|---|---|---|---|
| Size | 29.4 GB | 25.1 GB | 42.3 GB |
| Speed on nine P106-090, Pascal engine | RIG-TBD tok/s | RIG-TBD tok/s | RIG-TBD tok/s |
| Picks the same next word as the original model | 88.8 % | 87.7 % | 93.3 % |
| Distance from the original (KL divergence, lower is better) | 0.105 | 0.120 | 0.052 |

- **Tool calling:** on 3,477 single-turn tasks it scores the same as UD-Q4_K_XL (+0.14 points, within the
  noise), with fewer malformed answers.
- **The engine does not change the answers:** it only changes the order in which numbers are added up.

Every number with its command, date and conditions:
[results.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/results.md).

## Rebuild it yourself

Everything needed to rebuild this file from Flint's original weights is public: the exact list of formats per
tensor, a script that runs every step, and per-tensor checksums to compare your result with this file.
[The rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/model.md),
and the whole release in one folder:
[release/flint-v2](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/release/flint-v2).

## Licence and credits

- Flint is made by RecursiveSI and released under Apache-2.0; this file is a quantized derivative under the same
  licence ([LICENSE](LICENSE)). Changes: the weights are quantized to Q5_0 and Q8_0, the output head rounded with GPTQ,
  the routers stored in 16-bit floats, all in the GGUF format.
- Unsloth's importance matrix (Apache-2.0) guided the quantization; the file's header records it.
- Quantized by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X), as the file's header says
  (`general.quantized_by`), for
  [LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). Not affiliated with
  RecursiveSI or Unsloth.

| File | Size | SHA-256 |
|---|---|---|
| `flint-K2h-r16.gguf` | 29.4 GB | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`; the same checksums are in the
[GitHub repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/CHECKSUMS).
