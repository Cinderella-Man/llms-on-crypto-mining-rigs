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
| Miner Llama's N2P/N3P build of 2026-10-28, valid1 | 12 of 12 | 14.5 | — | 9.6 | [2026-10-27_ornith](../../docs/data/2026-10-27_ornith/README.md) |
| **v3 (`1f7215948`), the preset, valid1** | **12 of 12** | **14.9** | 59 | 9.41, 19.85 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| v3, + the keep-alive | TODO(ornithka) | TODO(ornithka) | TODO(ornithka) | TODO(ornithka) | TODO(ornithka) |
| v3, + Ornith's own MTP layer as the draft (`mtp`, `mtpadapt`, `mtpka`) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) | TODO(ornithmtp) |

- Stock llama.cpp cannot load valid1 (its N2P and N3P types are this fork's), so stock runs Unsloth's UD-Q2_K_XL, the
  common file of about the same size.
- The hop casts and the expert skip are not in Ornith's preset: each needs Ornith's own quality check first.
- The first token is slow (9.4 s median): the prompt-processing tiles are not tuned for N2P and N3P yet (Q351 in
  [quants/experiments.md](../../quants/experiments.md)).

## The file

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 48.2 GB | TODO(upload): this project's Hugging Face repository | TODO(upload) |

valid1 holds the routed experts in N2P and N3P, my Pascal 2- and 3-bit types, with more bits in the late layers, where
Ornith is most sensitive, and a Q5_0 output layer. It is closer to the makers' Q8_0 than Unsloth's UD-Q2_K_XL and
faster on the rig; how it was made and scored: [2026-10-28_ornith_valid1](../../docs/data/2026-10-28_ornith_valid1/README.md).
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
PRESET_ENV=""
PRESET_ARGS="-ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,5,5,5,6 -fit off"
```

| Setting | What it does |
|---|---|
| `-ts 6,6,6,6,5,5,5,5,6` | the layer split by memory, so that 48.2 GB and the context fit nine 6 GB cards |
| `-c 49152` | the context the agent check ran at |
| `-fit off` | skips llama.cpp's fit pass; this split was checked to fit |
| `--cache-ram 0`, `--ctx-checkpoints 4` | what keeps the 4 GB host from running out of memory |
| `-bs` | sampling on the GPU, also for tool requests |

No switch is set: the N2P/N3P kernels and the other Pascal kernels are on by default. TODO(ornithka), TODO(ornithmtp):
the settings that win go into `PRESET_ENV` and `PRESET_ARGS`.
