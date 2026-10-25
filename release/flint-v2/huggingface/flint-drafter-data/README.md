---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - flint
  - speculative-decoding
  - synthetic
---

# Flint drafter data: what the staircase drafter was trained on

The prompts, Flint's answers with their exact token ids, and the trained heads with their training checkpoints behind
the [staircase drafter](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter), a small model that guesses
Flint's next token so that a rig of old graphics cards runs Flint faster. With this data and Unsloth's Flint Q8_0
anyone can redo the drafter's training, or continue it from our checkpoints. Every step, command and data size is in
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).

## What is here

| Path | What | Size |
|---|---|---|
| `prompts/prompts*.jsonl` | the four prompt files, without the ShareGPT rows (below): 76,431 − COUNT-TBD prompts, each with its category and source | under 0.3 GB |
| `answers/gen-*.jsonl` | Flint's answers to them with the prompt's and the answer's token ids: 71,711 − COUNT-TBD answers, 74.66 M − COUNT-TBD answer tokens | under 0.6 GB |
| `manifest.json` | one entry per answer file: the writer (model file and SHA-256, engine and commit, server flags), the prompt file, the sampling, the number of answers and answer tokens, the file's SHA-256 | small |
| `vocab.npy` | the drafter's vocabulary: the 40,000 token ids it can guess | small |
| `heads/<run>/head.pt` | each run's best weights in PyTorch format; `export_gguf.py` turns one into the GGUF the engine loads | 0.73–0.81 GB each |
| `heads/<run>/ckpt.pt` | each run's last training checkpoint: weights, optimizer and scaler state, step and elapsed time, for `train.py --resume` | 2.1–2.4 GB each |
| `heads/<run>/config.json`, `log.jsonl` | the run's settings and its training log | small |

The runs, each started from the one before ([training](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md#5-training)):

| Run | Started from | What it is |
|---|---|---|
| U | scratch | the first head, trained on six-card windows |
| R1 | U | six-card windows with fused anchors, 13.5 M answer tokens |
| R2-9 | R1 | the release's nine-card head: `flint-staircase-drafter-9cards-v2.0.gguf` |
| R2-gen | R1 | the release's depth-general head for 3–12 cards: `flint-staircase-drafter-v2.0.gguf` |

## Sources and licences

The prompts come from these public sources, with the licences their pages state (checked 2026-10-25). Every prompt
keeps its source in the `source` field.

| Source | Licence | Prompts for |
|---|---|---|
| [HuggingFaceH4/ultrachat_200k](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | MIT | chat |
| [ise-uiuc/Magicoder-Evol-Instruct-110K](https://huggingface.co/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K) | Apache-2.0 | code |
| [ise-uiuc/Magicoder-OSS-Instruct-75K](https://huggingface.co/datasets/ise-uiuc/Magicoder-OSS-Instruct-75K) | MIT | code |
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (single-turn and glaive subsets) | Apache-2.0 | tool calls |
| [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Apache-2.0 | reasoning |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Apache-2.0 | reasoning |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | tool calls (10,581 prompts) |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | tool calls (4,749 prompts) |
| Python standard library source files | Python Software Foundation License 2.0 | whole-file edits and coding-agent turns |
| llama.cpp source files | MIT | whole-file edits and coding-agent turns |

**Left out: the ShareGPT rows.** The drafter was also trained on Flint's answers to COUNT-TBD prompts from
[Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) (COUNT-TBD answer
tokens). That dataset collects conversations people shared from ChatGPT, and the terms for reusing them are unclear
whatever its own licence tag says, so neither those prompts nor Flint's answers to them are published. The published heads and checkpoints were trained with them; a head trained on
this dataset alone is equivalent, not identical.

## Where the answers come from

| Writer | Engine | Answers | Answer tokens |
|---|---|---|---|
| Unsloth's Flint Q8_0 | llama.cpp (llama-server) | 7,148 | 8.61 M |
| G3, this project's mix | llama.cpp | 14,615 | 9.58 M |
| Unsloth's UD-Q4_K_XL | llama.cpp | 7,763 | 3.73 M |
| The makers' NVFP4 GGUF | vLLM | 42,185 | 52.74 M |

The counts include the ShareGPT answers that are left out (COUNT-TBD per writer). Sampling for every writer:
temperature 1.0, top-k 20, top-p 0.95, min-p 0, reasoning kept, at most 2,048 answer tokens, seed 20000 + the prompt's
number. Answers sampled on GPUs differ between machines in the details, so only these files reproduce the drafter's
exact training text; the writers and seeds above regenerate an equivalent set.

## What is not here, and how to recompute it

- **The hidden-state dumps** (~58 GB per million answer tokens, ~1.3 TB in all): each answer replayed through Flint
  Q8_0 with `llama-staircase-dump`, which records 26 layers' inputs and the top-20 next-token probabilities per position
  ([step 3 of the guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md#3-the-dump)).
- **The packed training sets**, written by the same command next to the dumps.
- **The test set** the drafter is scored on, 38 held-out Flint conversations, is in the GitHub repository
  ([reproduce/data/q51](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/release/flint-v2/reproduce/data/q51)).

## Licence

Apache-2.0 ([LICENSE](LICENSE)) for Flint's answers (Flint is Apache-2.0, by RecursiveSI), the heads, the checkpoints,
the vocabulary and the manifest. The prompts, and the prompt token ids inside the answer files, keep their sources'
licences listed above.

Made by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X) for
[LLMs on crypto-mining rigs](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs). SHA-256 checksums of every
file: [CHECKSUMS](CHECKSUMS).
