---
license: mit
base_model: ornith-ai/Ornith-1.5-122B
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - ornith
  - pascal
  - mining-rig
  - miner-llama
---

# Ornith-1.5-122B Pascal N2P GGUF: a 122B coding-agent model on old NVIDIA cards and mining rigs

**What this is.** [Ornith-1.5-122B](https://huggingface.co/ornith-ai/Ornith-1.5-122B) is a 122B-parameter
mixture-of-experts model for terminal coding agents that uses about 10B parameters per token. This file (its
internal name is `valid1`) holds its routed experts in N2P and N3P, Miner Llama's 2- and 3-bit types for Pascal
cards, with more bits in the late layers, where Ornith is most sensitive. It is 48.2 GB, made for old NVIDIA "Pascal"
cards (compute capability 6.1: the P106 and P104 cards of retired mining rigs, the GTX 10 series, the Tesla P4/P40);
its speed is measured on P106-090 mining cards only.

**It runs only on Miner Llama v3.** N2P and N3P are the fork's own types: stock llama.cpp, the apps built on it and
older Miner Llama releases cannot load this file.

**Why pick it.** Against Ornith's own 8-bit file it is closer than Unsloth's UD-Q2_K_XL, the common file of about the
same size, on all three texts measured (prose, coding-agent sessions, code), for 3.66 GB more.

**How fast.** A coding agent fixing twelve small bugs on a Manli mining rig with nine P106-090 cards (6 GB each, every
card on a single PCIe 1.0 lane, a 2-core Celeron host with 4 GB of RAM), 49,152 tokens of context: **16.3 tokens per
second** with Miner Llama v3's preset, all twelve tasks passed; 14.9 without its keep-alive switch.

**Which Ornith file should you pick?**
- **Nine 6 GB Pascal cards and Miner Llama v3: this file.**
- **Any other engine or GPU: Unsloth's [UD-Q2_K_XL](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF)**
  (44.5 GB), which every llama.cpp build loads.

## Run it in three steps

1. **Driver:** Ubuntu 24.04 or newer with NVIDIA's driver 580, the last one for Pascal cards
   ([setup guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/setup.md)).
2. **Download** the Pascal engine, Miner Llama v3.0, from the fork's
   [releases](https://github.com/Cinderella-Man/llama.cpp/releases), unpack it, and download
   `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` from this page.
3. **Start it**, on nine cards with Ornith's v3 preset:

   ```bash
   GGML_CUDA_KEEPALIVE=decode llama-p106-v3.0/llama-server -m Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off --host 127.0.0.1 --port 8080
   ```

   `GGML_CUDA_KEEPALIVE=decode` is the keep-alive, which holds the cards at their boost clock. Then open
   `http://127.0.0.1:8080` through an SSH tunnel from your PC (`ssh -N -L 8080:127.0.0.1:8080 <your rig>`), or point
   any OpenAI-compatible client at it.

The preset as a file and what each setting does:
[the release page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/ornith-v3/README.md);
the model on these rigs: [the Ornith page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/ornith-1.5-122b/README.md).

## What you need

- **Nine 6 GB cards:** the file puts 46.4 GB on them, the context takes the rest. The layer split above is set by
  memory; another card count needs its own `-ts`.
- **Miner Llama v3.** The rig's 4 GB of host memory is enough: on 2026-10-28 the agent check left 563 MB free at the lowest.

## Speed

The [agent check](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/bench/agent_check) gives a
coding agent twelve small bug-fix tasks (Python and JavaScript, real test runs, the context growing every turn) and
records every turn. Nine P106-090, 49,152-token context, one run per row:

| Engine and file | Tasks passed | Generation, tokens/s | Prompt reading, tokens/s | First token per turn (median) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp b12076 (`20a51f24a`), tuned (`-b 2048 -ub 256 -fa on`), Unsloth UD-Q2_K_XL, 2026-11-06 | 11 of 12 | 6.0 | 55 | 12.57 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v3 (`1551f5fe1`), Unsloth UD-Q2_K_XL, no switch, 2026-11-06 | 12 of 12 | 12.6 | 69 | 7.90 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v3 (`1f7215948`), this file, no switch, 2026-11-05 | 12 of 12 | 14.9 | 59 | 9.41 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| **Miner Llama v3 (`1f7215948`), this file, the preset (`GGML_CUDA_KEEPALIVE=decode`), 2026-11-05** | **12 of 12** | **16.3** | 58 | 8.39 s | [2026-11-05_release_regression](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v2.0, Unsloth UD-Q2_K_XL, 2026-10-27 | 12 of 12 | 9.7 | — | — | [2026-10-27_ornith](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-27_ornith/README.md) |

- **`1f7215948`** gives the same greedy output as the release build `1551f5fe1`.
- **The keep-alive** runs a one-thread loop on each card while the model decodes, so the driver keeps every card at
  its boost clock.
- **The first token is slow** (9.41 s median): Miner Llama's prompt kernels are not tuned for N2P and N3P yet.

## Quality

KL divergence against Ornith-1.5-122B's Q8_0 (lower is better) and the share of tokens whose top choice matches
it, on wikitext-2 (80 × 512 tokens), held-out coding-agent sessions (12 × 8,192) and the agent check's Exercism code
(80 × 512):

| File | Size | Wiki | Agent sessions | Code |
|---|---|---|---|---|
| Unsloth UD-Q2_K_XL | 44.53 GB | 0.1843 ± 0.0026 / 83.55 % | 0.0541 ± 0.0011 / 94.93 % | 0.0713 ± 0.0028 / 96.32 % |
| **This file** | **48.20 GB** | **0.1762 ± 0.0028 / 84.14 %** | **0.0518 ± 0.0010 / 95.04 %** | **0.0615 ± 0.0026 / 96.69 %** |

In standard errors of a difference, this file is closer to the Q8_0 by 2.1 on wikitext, 1.5 on agent sessions and
2.6 on code. Every score and how the file was composed:
[2026-10-28_ornith_valid1](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-28_ornith_valid1/README.md).

## How I made it, and what changed

The changes against ornith-ai's
[Ornith-1.5-122B-Q8_0.gguf](https://huggingface.co/ornith-ai/Ornith-1.5-122B-GGUF), in the GGUF format:
- **The routed experts** requantized to N2P and N3P with Unsloth's importance matrix for Ornith-1.5-122B: gate and
  up at N2P in layers 0–23 and N3P in 24–47, down at N2P in layers 0–31 and N3P in 32–47: more bits where a
  per-layer search found Ornith most sensitive.
- **The output layer** in Q5_0, from the same Q8_0; it cost nothing measurable and saves 287 MB.
- **Every other weight** in Q5_0, Q8_0 or F32; the MTP block is included.

The formats: [pascal_lowbit_format.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/research/notes/pascal_lowbit_format.md);
the layer search: [2026-10-27_ornith_alloc](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-10-27_ornith_alloc/README.md).

## Licence and credits

- **Ornith-1.5-122B** is made by ornith-ai and released under the MIT licence, as declared on its model card; this
  file is a quantized derivative under the same licence ([LICENSE](LICENSE)), with the changes listed above.
- **Its base, Qwen3.5-122B-A10B,** is made by Qwen and released under Apache-2.0
  ([LICENSE-QWEN3.5](LICENSE-QWEN3.5)). The makers' card says Ornith was developed on top of Qwen3.5 and Gemma 4; its
  architecture is Qwen3.5's mixture of experts, and the makers' Q8_0 names itself Qwen3.5-122B-A10B in its header.
- **Unsloth's importance matrix** for Ornith-1.5-122B (`imatrix_unsloth.gguf_file`, from
  [unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF), MIT) decided which weights get
  more bits, so their calibration work is part of this file, with thanks.
- Quantized by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X), for
  [LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). Not affiliated with
  ornith-ai, Qwen or Unsloth.

| File | Size | SHA-256 |
|---|---|---|
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 48.2 GB | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`.
