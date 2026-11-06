# Ornith-1.5-122B on Miner Llama v3

A 122B mixture-of-experts model for coding agents on nine 6 GB mining cards, in my own 2- and 3-bit file. This folder
says which file to download, the one command that starts it, and its preset. The engine's release notes are
[notes-v3.0.md](../notes-v3.0.md); the model's page is [models/ornith-1.5-122b](../../models/ornith-1.5-122b/README.md).

## Speed

A coding agent fixing twelve small bugs, with real tool calls and test runs ([bench/agent_check](../../bench/agent_check/README.md)),
on nine P106-090 cards, 49,152 tokens of context:

| Engine and file | Tasks passed | Generation tok/s | Prompt tok/s | First token per turn, s (median, p90) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp (an older commit), Unsloth UD-Q2_K_XL | — | 6.2 | — | — | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| Stock llama.cpp b12076 (`20a51f24a`), Unsloth UD-Q2_K_XL | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) |
| Miner Llama v2.0, UD-Q2_K_XL | 12 of 12 | 9.7 | — | — | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| Miner Llama's N2P/N3P build of 2026-10-28, Ornith-1.5-122B Pascal N2P GGUF (`valid1`) | 12 of 12 | 14.5 | — | 9.6 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| v3 (`1f7215948`), the same file, no switch | 12 of 12 | 14.9 | 59 | 9.41, 19.85 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| **v3, + the keep-alive (`GGML_CUDA_KEEPALIVE=decode`): the preset** | **12 of 12** | **16.3** | 58 | 8.39, 19.38 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| v3, + Ornith's own MTP layer as the draft (`mtp`, `mtpadapt`, `mtpka`) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) |

- The keep-alive is a gain, so the preset sets it: 16.3 against 14.9 tok/s generation (+9.4 %), the median first token
  8.39 against 9.41 s.
  All twelve tasks took the same time (32.0 against 32.2 minutes), since answers at the makers' sampling differ in
  length.
- Stock llama.cpp cannot load Ornith-1.5-122B Pascal N2P GGUF (its N2P and N3P types are this fork's), so stock runs
  Unsloth's UD-Q2_K_XL, the common file of about the same size.
- The expert skip is not in Ornith's preset: at p = 0.9 its KL divergence on the Ornith-1.5-122B Pascal N2P GGUF is above noise on all three texts (agent +0.00756 ± 0.00090, code +0.01021 ± 0.00178, wiki +0.02383 ± 0.00280; [2026-11-05_night_quant_chain](../../docs/data/2026-11-05_night_quant_chain/README.md#results)). The Q8_0 hop cast is not either: Ornith's graph does not call it, so the switch changes nothing there.
- The first token is slow (8.39 s median): the prompt-processing tiles are not tuned for N2P and N3P yet (Q351 in
  [quants/experiments.md](../../quants/experiments.md)).

Quality: on the P106 cards the release build computes Ornith's prompt passes a little differently from the low-bit
build of 2026-10-28 (`d7a75deb3`). On wikitext-2 their KL divergence is 0.021653 ± 0.000881, with the same top token
93.873 % of the time, and the release build's perplexity is the lower, 5.9938 ± 0.2410 against 6.0735 ± 0.2449
([data](../../docs/data/2026-11-05_release_regression/README.md#how-large-the-difference-is)). Which of the two is
closer to the model: TODO(t3e).

## The file

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 48.2 GB | [CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF](https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF) | `0bfa0cf2…` in its [CHECKSUMS](huggingface/Ornith-1.5-122B-Pascal-N2P-GGUF/CHECKSUMS) |

```bash
wget -c -P /mnt/models https://huggingface.co/CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF/resolve/main/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
```

Ornith-1.5-122B Pascal N2P GGUF (`valid1`) holds the routed experts in N2P and N3P, my Pascal 2- and 3-bit types, with more bits in the late layers, where
Ornith is most sensitive, and a Q5_0 output layer. It is closer to the makers' Q8_0 than Unsloth's UD-Q2_K_XL and
faster on the rig; how it was made and scored: [2026-10-28_ornith_valid1](../../docs/data/2026-10-28_ornith_valid1/README.md). Which weights get more bits was decided with [Unsloth](https://huggingface.co/unsloth)'s published importance matrix for Ornith-1.5-122B (MIT), so their calibration work is part of this file, with thanks.
Ornith-1.5 is under the MIT licence.

## Run it

From a clone of this repository, with the v3 engine unpacked beside it ([notes-v3.0.md](../notes-v3.0.md), "Install"):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/ornith-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Then connect from your PC through an SSH tunnel ([README](../../README.md#run-it)), and keep the host quiet while it
works: on 2026-10-28 its agent check left 563 MB of host memory at the lowest.

## The preset: [settings.env](settings.env)

```bash
PRESET_MODEL=Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
PRESET_ENV="GGML_CUDA_KEEPALIVE=decode"
PRESET_ARGS="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
```

| Setting | What it does |
|---|---|
| `-ts 6,6,6,6,5,5,5,5,6` | the layer split by memory, so that 48.2 GB and the context fit nine 6 GB cards |
| `-c 49152` | the context the agent check ran at |
| `-fit off` | skips llama.cpp's fit pass; this split was checked to fit |
| `--cache-ram 0`, `--ctx-checkpoints 4` | what keeps the 4 GB host from running out of memory |
| `-bs` | sampling on the GPU, also for tool requests |

One switch is set, the keep-alive (`GGML_CUDA_KEEPALIVE=decode`): 16.3 against 14.9 tok/s. The N2P/N3P kernels and
the other Pascal kernels are on by default. TODO(ornithmtp): Ornith's own MTP layer as the draft, if one of its arms
beats the preset.

## The recipe: every setting and what it bought

The rig has nine 6 GB cards, each on one PCIe 1.0 lane, a 2-core Celeron and 4 GB of host memory. Almost every
setting below works around one of those. Each speed is generation tok/s from one run of the coding-agent check at
49,152 tokens of context, unless a row says otherwise.

### The ladder: from stock llama.cpp to 16.3 tok/s

One change at a time:

| Step | What I added | Tasks passed | Generation tok/s | Data |
|---|---|---|---|---|
| 0 | stock llama.cpp (an older commit), Unsloth's UD-Q2_K_XL | 8 of 9 tried | 6.2 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| 0 | stock llama.cpp b12076 (`20a51f24a`), the upstream release v3 is built on, UD-Q2_K_XL | TODO(stock) | TODO(stock) | TODO(stock) |
| 1 | Miner Llama v2.0's Pascal kernels, the same file | 12 of 12 | 9.7 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| 2 | + kernels for the IQ2_S and IQ3_XXS weights the file is made of (`ea55c7865`) | 12 of 12 | 11.3 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| 2 | the v3 build on the same file | TODO(ornithud) | TODO(ornithud) | TODO(ornithud) |
| 3 | my file, Ornith-1.5-122B Pascal N2P GGUF, its experts in my Pascal 2- and 3-bit types, on the low-bit build of 2026-10-28 (`d7a75deb3`) | 12 of 12 | 14.5 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/agent/valid1/compare.md) |
| 4 | the v3 build (`1f7215948`), the same file | 12 of 12 | 14.9 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| 5 | + the keep-alive (`GGML_CUDA_KEEPALIVE=decode`): **the preset** | 12 of 12 | **16.3** | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| 6 | + Ornith's own MTP layer as the draft | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) |

Steps 0 to 2 ran the same file on three engines. Step 3 changes the file and the build at once; the v3
build on Unsloth's file splits the two.

### Every setting

| Setting | What it does | Why this rig needs it | What it measured | Data |
|---|---|---|---|---|
| Ornith-1.5-122B Pascal N2P GGUF (the file) | the routed experts in N2P and N3P, my Pascal 2- and 3-bit types, with more bits in the late layers and a Q5_0 output layer | 122B parameters fit 54 GB of cards only at 2 to 3 bits, and these types decode about as fast as Q4_0 on Pascal | 14.5 against 11.3 tok/s for UD-Q2_K_XL (different builds, above), and closer to the makers' Q8_0 | [ornith_valid1](../../docs/data/2026-10-28_ornith_valid1/README.md) |
| `-ts 6,6,6,6,5,5,5,5,6`, `-fit off` | how many layers each card holds; skips llama.cpp's own fit pass | 48.2 GB plus the cache must fit nine separate 6 GB pools | a fit setting, not a speed one; check another card count with [fit_probe.sh](../../scripts/fit_probe.sh) | — |
| `--cache-ram 0`, `--ctx-checkpoints 4` | no prompt cache in host memory; at most four saved context states | the 4 GB host: the agent check left 563 MB free at the lowest | not measured on its own: every run on this page used them | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| `-bs` | samples the next token on the card, also for tool requests | otherwise every token's scores cross the slow link | not measured on its own for Ornith; on Flint, tool requests ran at 37.9 against 33.6 tok/s sampled on the host | [Flint v2 results](../flint-v2/results.md) |
| `-ngl 99 -fa on -c 49152 -np 1 -b 2048 -ub 512 --jinja --reasoning-format deepseek`, the sampling | every layer on the cards, flash attention, the context, one request at a time, prompt batches of 512, the model's chat template with its reasoning kept apart; the makers' sampling settings | ordinary llama.cpp settings | — | — |
| `PRESET_ENV="GGML_CUDA_KEEPALIVE=decode"` | a small kernel on every card keeps it at its boost clock while it decodes | between the short kernels of one token the cards fall back to their base clock | 16.3 against 14.9 tok/s; Ornith's own MTP draft: TODO(ornithmtp) | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |

### On by default

These need no setting. Each has a switch that turns it off ([docs/switches.md](../../docs/switches.md)):

| Feature | Turning it off | Data |
|---|---|---|
| The fork's kernels for N2P and N3P (part of `GGML_CUDA_MMVQ_PASCAL=1`) | there is no other path: stock llama.cpp cannot load Ornith-1.5-122B Pascal N2P GGUF at all | [ornith_valid1](../../docs/data/2026-10-28_ornith_valid1/README.md) |
| The fork's kernels for IQ2_S and IQ3_XXS, for Unsloth's file | `GGML_CUDA_MMVQ_PASCAL_SKIP=iq2_s,iq3_xxs` gives llama.cpp's kernels: 9.7 against 11.3 tok/s on the agent check (v2.0 without them), 103.4 against 88.6 ms per generated token | [2026-10-27_iq_kernels](../../docs/data/2026-10-27_iq_kernels/README.md) |
