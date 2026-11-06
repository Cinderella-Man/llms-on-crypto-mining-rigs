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
  - miner-llama
---

# Flint Pascal Q8 GGUF: Flint at Q8_0's quality, 11 % smaller, for old NVIDIA cards and mining rigs

**What this is.** [Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8-billion-parameter AI model that thinks
before it answers, writes code and calls tools, and reads up to 262,144 tokens of context. This file (its internal
name is `p065`) is Unsloth's Q8_0 of Flint with the tensors that cost the least quality moved to 5-bit Q5_0: 40.4 GB
against Q8_0's 42.3 GB, and 11.0 % fewer bytes read per generated token. It is made for old NVIDIA "Pascal" cards
(compute capability 6.1: the P106 and P104 cards of retired mining rigs, the GTX 10 series, the Tesla P4/P40); its
speed is measured on P106-090 mining cards only.

**Why it is faster there.** A Pascal mining card spends a little over half of each generated token reading weights
from its memory. Fewer bytes per token means less of that: in a paired replay of the same agent sessions this file
generated each token in 0.941 of Q8_0's time, at the same prompt speed. Its KL divergence against the original
16-bit model is within two standard errors of Q8_0's.

**How fast.** A coding agent fixing twelve small bugs on a Manli mining rig with nine P106-090 cards (6 GB each, every
card on a single PCIe 1.0 lane, a 2-core Celeron host with 4 GB of RAM), the full 262,144-token context: **47.4
tokens per second**, all twelve tasks passed, on the Miner Llama v3 release build, against 45.9 for Unsloth's Q8_0 on
the same build.

**Which Flint file should you pick?**
- **Nine 6 GB cards, best quality: this file or Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF/tree/20468edfc38ab43d6914cc88b08cbecb05d0fc62)**
  (42.3 GB). Both fit the full context on nine cards; this one is Q8_0's quality at fewer bytes per token.
- **Six to eight cards, or the most speed on nine:
  [Flint Pascal Q5 GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF)** (29.4 GB), at a lower quality.
- **On a modern GPU or a Mac: Unsloth's Q8_0.** This file is measured on Pascal cards only.

## Run it in three steps

1. **Driver:** Ubuntu 24.04 or newer with NVIDIA's driver 580, the last one for Pascal cards
   ([setup guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/setup.md)).
2. **Download** the Pascal engine, Miner Llama v3.0, from the fork's
   [releases](https://github.com/Cinderella-Man/miner-llama/releases), unpack it, and download `flint-pascal-Q8.gguf`
   from this page.
3. **Start it**, on nine cards with the full context (v2's release line, Flint's v3 preset):

   ```bash
   llama-p106-v3.0/llama-server -m flint-pascal-Q8.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4 --host 127.0.0.1 --port 8080
   ```

   Then open `http://127.0.0.1:8080` through an SSH tunnel from your PC (`ssh -N -L 8080:127.0.0.1:8080 <your rig>`),
   or point any OpenAI-compatible client at it.

What each flag does and how to use it for coding agents:
[the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md); the
v3 release: [release/flint-v3](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v3/README.md).

## What you need

- **About 41 GB of GPU memory plus room for the context:** nine 6 GB cards for the full 262,144 tokens.
- **Miner Llama for the speed.** The file uses only llama.cpp's own types (Q8_0, Q5_0, F16), so stock llama.cpp and
  apps built on it load it too, without the Pascal kernels.

## Speed

The [agent check](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/bench/agent_check) gives the
[pi](https://github.com/earendil-works/pi) coding agent twelve small bug-fix tasks (Python and JavaScript, real test
runs, the context growing every turn) and records every turn. Nine P106-090, 262,144-token context, the command line
above, one run per row:

| Engine and file | Tasks passed | Generation, tokens/s | Prompt reading, tokens/s | First token per turn (median) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp b12076 (`20a51f24a`), Unsloth Q8_0, 65,536-token context, 2026-11-05 and 2026-11-06 | 10 of 12 | 14.4 | 148 | 4.13 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v3 (`1f7215948`), Unsloth Q8_0, 2026-11-05 | 12 of 12 | 45.9 | 168 | 2.67 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| **Miner Llama v3 (`1551f5fe1`), this file, 2026-11-06** | **12 of 12** | **47.4** | 175 | 2.66 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v2.0 (`37e717ff6`), this file, 2026-10-28 | 12 of 12 | 47.1 | — | — | [2026-10-27_flint_near_q8](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-27_flint_near_q8/README.md) |

- **`1f7215948`** gives the same greedy output as the release build `1551f5fe1`.
- **On the release build** this file generates 3.3 % faster than Unsloth's Q8_0 (47.4 against 45.9 tokens/s), with a
  median task of 1.0 against 1.2 minutes. Its sessions wrote more text, so all twelve tasks took 22.3 against 19.2
  minutes.
- **Stock** runs at 65,536 tokens of context, because at 262,144 its host-side attention mask does not fit in 4 GB, on
  the line a llama-bench pass chose for it (`-b 2048 -ub 512 -fa on`, the same split). Its two misses ran into the
  12-minute cap.
- **The paired replay** (v2.0, the same six recorded agent sessions sent to Q8_0 and then to this file, replies
  capped at their recorded lengths): 0.941 of Q8_0's time per generated token (95 % interval 0.930–0.980 over the
  sessions), 1.002 per prompt token, 0.963 of the time for the same work.

## Quality

Against Flint's original 16-bit weights, 80 × 512 tokens of wikitext-2, all three files on the same GPU set:

| File | Size | Bytes read per token | Picks the same next word as the original | Distance from the original (KL divergence, lower is better) |
|---|---|---|---|---|
| Unsloth Q8_0 | 42.3 GB | 2.696 GB | 92.39 ± 0.19 % | 0.0627 ± 0.0032 |
| **This file** | **40.4 GB** | **2.400 GB** | **91.78 ± 0.19 %** | **0.0656 ± 0.0029** |
| Flint Pascal Q5 GGUF | 29.4 GB | 2.090 GB | 88.44 ± 0.22 % | 0.1108 ± 0.0037 |

- **On coding-agent text** (12 × 8,192 tokens, against Q8_0) this file scores 0.2292 and 93.42 %; Q8_0 with only its
  routers in F16, the floor for every file with F16 routers, scores 0.2273 and 93.78 %.
- **The engine keeps the answers:** its kernels change only the order in which numbers are added up, and pipelined
  speculation gives exactly the tokens of plain decoding.

## How I made it, and what changed

Every tensor of Flint was given a cost: the KL divergence that moving it alone from Q8_0 to Flint Pascal Q5 GGUF's
Q5_0 adds, by tensor class and layer group. Dynamic programming over the bytes each move saves per token picked the
set with the least added divergence at each size. Of the three picks composed and scored, this file saves the most
while its KL divergence stays within noise of Q8_0's. The
changes against [Unsloth's Q8_0](https://huggingface.co/unsloth/flint-GGUF) (revision `20468edf`), in the GGUF format:
- **The routers** in 16-bit floats (F16), taken from Flint Pascal Q5 GGUF.
- **In Q5_0, taken from Flint Pascal Q5 GGUF:** attention q, k, v and output in layers 33–48, the shared expert in
  layers 33–48, the routed experts' gate and up in layers 41–48, and the output head (Q5_0 rounded with GPTQ).
- **Everything else** is Unsloth's Q8_0, unchanged. Early layers stay at Q8_0: they carry most of the cost.

The scores of every move, the picks and the composing script:
[2026-10-27_flint_near_q8](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-27_flint_near_q8/README.md).

## Licence and credits

- Flint is made by RecursiveSI and released under Apache-2.0; this file is a quantized derivative under the same
  licence ([LICENSE](LICENSE)), with the changes listed above.
- Built from Unsloth's Flint Q8_0 (Apache-2.0); Unsloth's importance matrix guided the Q5_0 tensors' rounding.
- Quantized by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X), for
  [LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). Not affiliated with
  RecursiveSI or Unsloth.

| File | Size | SHA-256 |
|---|---|---|
| `flint-pascal-Q8.gguf` | 40.4 GB | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`.
