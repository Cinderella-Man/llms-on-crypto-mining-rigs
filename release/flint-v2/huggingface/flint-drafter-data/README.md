---
license: apache-2.0
task_categories:
  - text-generation
tags:
  - flint
  - speculative-decoding
  - synthetic
---

<!-- Draft for a later release, not part of v2.0: publishing the data is parked (2026-10-25). -->

# Flint drafter data: what the staircase drafter was trained on

The prompts, Flint's answers with their exact token ids, and the trained heads with their training checkpoints behind
the [staircase drafter](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter), a small model that guesses
Flint's next token so that a rig of old graphics cards runs Flint faster. With this data and Unsloth's Flint Q8_0
anyone can redo the drafter's training, or continue it from our checkpoints. Every step, command and data size is in
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).

## What is here

| Path | What | Size |
|---|---|---|
| `prompts/prompts*.jsonl` | the four prompt files, without the ShareGPT rows (below): 66,863 prompts, each with its category and source | under 0.3 GB |
| `answers/gen-*.jsonl` | Flint's answers to them with the prompt's and the answer's token ids: 62,846 answers, 62.05 M answer tokens | under 0.6 GB |
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

**Left out: the ShareGPT rows.** The drafter was also trained on Flint's answers to 9,568 prompts from
[Aeala/ShareGPT_Vicuna_unfiltered](https://huggingface.co/datasets/Aeala/ShareGPT_Vicuna_unfiltered) (1,100, 3,017
and 5,451 in the three general prompt files, none in the tool file): 8,865 answers with 12.62 M answer tokens, 16.9 %
of the training answers. That dataset collects conversations people shared from ChatGPT, and the terms for reusing
them are unclear whatever its own licence tag says. LMSYS, who collected these conversations to train Vicuna, wrote
"We will not release the ShareGPT dataset" ([FastChat](https://github.com/lm-sys/FastChat)), and the drafter projects
this one follows (EAGLE, Medusa, Kangaroo) published their trained heads without republishing it. So do we: the heads
and checkpoints, trained with those rows, are published; the rows are not. A head trained on this dataset alone is
equivalent, not identical.

## Where the answers come from

| Writer | Engine | Answers here | Answer tokens here | Left out (ShareGPT): answers / tokens |
|---|---|---|---|---|
| Unsloth's Flint Q8_0 | llama.cpp (llama-server) | 6,048 | 7.07 M | 1,100 / 1.54 M |
| G3, this project's mix | llama.cpp | 13,722 | 8.32 M | 893 / 1.26 M |
| Unsloth's UD-Q4_K_XL | llama.cpp | 7,525 | 3.38 M | 238 / 0.35 M |
| The makers' NVFP4 GGUF | vLLM | 35,551 | 43.27 M | 6,634 / 9.47 M |
| **In all** | | **62,846** | **62.05 M** | 8,865 / 12.62 M |

The drafter was trained on both columns: 71,711 answers, 74.7 M answer tokens. Sampling for every writer:
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
