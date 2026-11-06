---
license: other
license_name: swift-open-license-1.0
license_link: https://huggingface.co/CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF/blob/main/LICENSE
base_model: ukisai/Swift1.5-Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
tags:
  - gguf
  - llama.cpp
  - qwen3.8-flash-next
  - pascal
  - mining-rig
  - miner-llama
---

# Swift Qwen3.8-Flash-Next Pascal Q2 GGUF: a 180B-parameter model on old NVIDIA cards and mining rigs

**What this is.** Qwen3.8-Flash-Next is a 125B mixture-of-experts model with about 6B active parameters per token,
plus a 51B-parameter n-gram table. This file is UkisAI's 2-bit Swift-1.5 fine-tune of it (Swift-1.5 GSQ-RCO Q2_0)
with a Pascal type mix (its internal name is `pmix-q2-fast`), made for old NVIDIA "Pascal" cards (compute capability
6.1: the P106 and P104 cards of retired mining rigs, the GTX 10 series, the Tesla P4/P40). Its speed is measured on
P106-090 mining cards only.

**Not affiliated with or endorsed by UkisAI or Qwen.** "Swift" in the name only says where the file comes from.

**What changed against UkisAI's Q2_0.** The routed experts (most of the file) are UkisAI's bytes, unchanged. The
weights a token reads every time (attention and recurrent projections, the shared expert, the router) are in types
the P106 reads fast, so each token costs an estimated 10 ms less at the same quality. The n-gram table is not in this
repository: you cut it out of UkisAI's own download (below).

**How fast.** A coding agent fixing twelve small bugs on a Manli mining rig with nine P106-090 cards (6 GB each, every
card on a single PCIe 1.0 lane, a 2-core Celeron host with 4 GB of RAM): **20.75 tokens per second**, all twelve tasks
passed, with the Miner Llama v3 engine and its MTP draft.

**Which file should you pick?**
- **Nine 6 GB Pascal cards: this file**, with the preset below.
- **Any other GPU or a Mac: UkisAI's own Q2_0.** It has the same quality; this mix is faster only on Pascal cards.

## Run it

You need the rig (or nine 6 GB Pascal cards), about 30 GB free on an internal SSD for the n-gram table, 41 GB on the
model drive, and a PC with Python 3 and room for UkisAI's 66.5 GB download.

1. **Driver:** Ubuntu 24.04 or newer with NVIDIA's driver 580, the last one for Pascal cards
   ([setup guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/setup.md)).
2. **Engine:** download Miner Llama v3.0 from the fork's
   [releases](https://github.com/Cinderella-Man/llama.cpp/releases) and unpack it.
3. **Files:** this repository's two shards, the MTP draft, and UkisAI's two Q2_0 shards at the revision measured here:

   ```bash
   hf download CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF --include '*.gguf' --local-dir swift_q2_0_pmix_fast
   ```

   ```bash
   hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF mtp-Qwen3.8-Flash-Next-draft.gguf --revision 273edc1d2ffae5241fe3cab69568d682dad4c718 --local-dir .
   ```

   ```bash
   hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --include '*Q2_0-0000*-of-00002.gguf' --local-dir swift
   ```

4. **The n-gram table:** cut it out of UkisAI's download with
   [gguf_reshard.py](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/scripts/gguf_reshard.py)
   (copies every byte, ~8 minutes on an NVMe drive), put the third new file on the SSD, and link it beside the two
   shards under this file's name:

   ```bash
   python3 -I gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir swift_q2_0 swift/*Q2_0-0000*-of-00002.gguf
   ```

   ```bash
   ln -s <SSD>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf swift_q2_0_pmix_fast/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00003-of-00003.gguf
   ```

5. **Start it**, on nine cards with a 65,536-token context:

   ```bash
   GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode llama-p106-v3.0/llama-server -m swift_q2_0_pmix_fast/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf -md mtp-Qwen3.8-Flash-Next-draft.gguf -ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host --host 127.0.0.1 --port 8080
   ```

   Then open `http://127.0.0.1:8080` through an SSH tunnel from your PC (`ssh -N -L 8080:127.0.0.1:8080 <your rig>`),
   or point any OpenAI-compatible client at it.

The same line as a preset file, what every setting does and what it bought, the checks of the cut table, and another
card count: [the release page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v3/README.md).

## What you need

- **Nine 6 GB cards:** the model takes 35,735 MiB of their 54 GB, the draft and the context the rest. The 28.8 GB
  n-gram table is read from the SSD as tokens need it, so 4 GB of host memory is enough.
- **Miner Llama v3 for the speed.** The file uses only llama.cpp's own types, so stock llama.cpp can load it too,
  without the Pascal kernels; its stock speed is in the table below.

## Speed

The [agent check](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/bench/agent_check) gives a
coding agent twelve small bug-fix tasks (Python and JavaScript, real test runs, the context growing every turn) and
records every turn. Nine P106-090, 65,536-token context, one run per row:

| Engine and file | Tasks passed | Generation, tokens/s | Prompt reading, tokens/s | First token per turn (median) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp b12076 (`20a51f24a`), UkisAI's Swift Q2_0, 2026-11-05 | 9 of 9 (3 not reached) | 8.20 | — | 10.22 s | [2026-11-05_release_flashnext](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_flashnext/README.md) |
| Stock llama.cpp b12076 (`20a51f24a`), this file | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) |
| Miner Llama v3 (`1551f5fe1`), UkisAI's Swift Q2_0, the preset, 2026-11-06 | 12 of 12 | 20.40 | 69.4 | 6.89 s | [2026-11-05_release_flashnext](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_flashnext/README.md) |
| **Miner Llama v3 (`1f7215948`), this file, the preset, 2026-11-05** | **12 of 12** | **20.75** | 69.0 | 7.60 s | [2026-11-05_release_flashnext](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-05_release_flashnext/README.md) |

- **`1f7215948`** gives the same greedy output as the release build `1551f5fe1`.
- **The draft** guesses three tokens ahead; 0.714 of its guesses were accepted, 3.14 tokens a step. The text is the
  main model's own.
- **On the nine tasks the stock row finished,** this row did a task's work in 1.87 minutes against 3.74. Those two
  rows differ in both engine and file; the two middle rows separate the two.

## Quality

KL divergence against Swift-1.5's 8-bit file (Q8_0), on wikitext-2, held-out coding-agent sessions and code; lower
is better. Both files scored on the same five cards:

| File | Size on the cards | Wiki KLD | Agent KLD | Code KLD | Same top token: wiki / agent / code |
|---|---|---|---|---|---|
| UkisAI's Swift Q2_0 | 35,730 MiB | 0.4282 ± 0.0059 | 0.2866 ± 0.0056 | 0.1121 ± 0.0040 | 77.85 / 90.67 / 95.97 % |
| **This file** | **35,735 MiB** | **0.4264 ± 0.0059** | **0.2792 ± 0.0054** | **0.1080 ± 0.0038** | **77.71 / 90.70 / 96.12 %** |
| Swift-1.5 Q8_0 at `-ub 256`, the floor (seven cards) | — | 0.0170 | 0.0291 | 0.0036 | 95.57 / 97.10 / 99.24 % |

No tensor got fewer bits than UkisAI gave it. Every score:
[2026-11-04_pascal_mix](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_pascal_mix/README.md).

## How I made it, and what changed

This is a modified copy of UkisAI's work. The changes, against
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(revision `99bb8f7f`):
- **168 tensors retyped:** requantized by llama-quantize from Swift-1.5's Q8_0
  ([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GGUF),
  revision `0142554c`) with my importance matrix `imatrix-swift15-q8_0-agent-train.gguf`, computed on this project's
  own coding-agent sessions. Q3_K attention, recurrent and shared-expert matrices became IQ4_XS; the shared expert's
  IQ4_NL down projection became Q5_0; the shared expert's gate and up got one type per layer, the larger of the two.
- **The router** went from F32 to BF16. Its values were BF16 already, so this is exact.
- **The n-gram table** moved to a shard of its own and is not included.
- **Everything else** is UkisAI's bytes, unchanged: the routed experts, the hyper-connections, the output layer, the
  token embeddings and every norm.

The recipe and the rebuild script:
[2026-11-04_pascal_mix](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/data/2026-11-04_pascal_mix/README.md)
and [bench/pascal_mix](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/bench/pascal_mix).
How Miner Llama v3 got from 7.8 to 20.75 tokens per second on this model, one change at a time:
[the v3 chapter](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/book/10-v3.md).

## Licence and credits

Copyright 2026 UkisAI. Swift Contribution licensed under the Swift Open License v1.0 ([LICENSE](LICENSE)).
Derivative of Qwen3.8-Flash-Next, Copyright (c) 2026 Qwen, licensed under the Qwen Community License 1.0
([LICENSE-QWEN](LICENSE-QWEN)). UkisAI's attributions, including ISTA-DASLab's GSQ-RCO allocation profiles, are in
[NOTICE](NOTICE). All three files are UkisAI's, unchanged.

**What the licences let you do,** in plain words (read the files for the exact terms):
- **Free to use, also commercially, while you are small:** the Swift licence covers commercial use (direct or
  indirect advantage) only while you, counted together with any company that controls you or that you control, had
  less than US$1,000,000 of gross revenue in the last fiscal year. Above that you need a Swift Enterprise License from
  UkisAI. Qualified non-profits are exempt for non-commercial or research use.
- **Model-as-a-service and assistant businesses need Qwen's licence:** if your group sells others inference access
  to models, or sells a product mainly for AI-assisted coding or office work, the Qwen licence asks for Qwen's
  separate licence before any commercial use. Internal use that gives no third party access is exempt.
- **Very large products** (over 100 million monthly active users or US$20 million monthly revenue) must show the
  model's name prominently.
- **Breaking the Swift licence's terms ends it at once.**

Quantized by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X), non-commercially, for
[LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs).

| File | Size | SHA-256 |
|---|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf` | 11.0 GB | see [CHECKSUMS](CHECKSUMS) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf` | 26.8 GB | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`. The third shard is UkisAI's n-gram table; its checksum is in the
release's [CHECKSUMS](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flash-next-v3/CHECKSUMS).
