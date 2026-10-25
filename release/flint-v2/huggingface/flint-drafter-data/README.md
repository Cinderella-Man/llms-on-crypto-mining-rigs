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
Flint's next token so that a rig of old graphics cards runs Flint faster. Every prompt comes from a source whose
answers may be published. With this data and Unsloth's Flint Q8_0 anyone can redo the drafter's training, or continue
it from our checkpoints. Every step, command and data size is in
[the drafter rebuild guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md).

## What is here

| Path | What | Size |
|---|---|---|
| `prompts/prompts*.strict.jsonl` | the four prompt files with the strict sources only: 44,456 prompts, each with its id, category and source | RIG-TBD |
| `answers/gen-*.jsonl` | Flint's answers to them with the prompt's and the answer's token ids: RIG-TBD answers and answer tokens | RIG-TBD |
| `manifest.json` | one entry per answer file: the writer (model file and SHA-256, engine and commit, server flags), the prompt file, the sampling, the number of answers and answer tokens, the file's SHA-256 | small |
| `vocab.npy` | the drafter's vocabulary: the 40,000 token ids it can guess | small |
| `heads/<run>/head.pt` | each run's best weights in PyTorch format; `export_gguf.py` turns one into the GGUF the engine loads | ~0.7–0.8 GB each |
| `heads/<run>/ckpt.pt` | each run's last training checkpoint: weights, optimizer and scaler state, step and elapsed time, for `train.py --resume` | ~2.1–2.4 GB each |
| `heads/<run>/config.json`, `log.jsonl` | the run's settings and its training log | small |

The runs: RIG-TBD, the rebuilt heads' chain from scratch to the two release heads
([training](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/reproduce/drafter.md#5-training)).

## Sources and licences

Only sources whose answers may be published, with the licences their pages state (checked 2026-10-25). Every prompt
keeps its source in the `source` field.

| Source | Licence | Prompts for | Prompts |
|---|---|---|---|
| [NousResearch/hermes-function-calling-v1](https://huggingface.co/datasets/NousResearch/hermes-function-calling-v1) (single-turn and glaive subsets) | Apache-2.0 | tool calls | 4,497 |
| [AI-MO/NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | Apache-2.0 | reasoning | 3,480 |
| [open-r1/OpenR1-Math-220k](https://huggingface.co/datasets/open-r1/OpenR1-Math-220k) | Apache-2.0 | reasoning | 2,743 |
| [Team-ACE/ToolACE](https://huggingface.co/datasets/Team-ACE/ToolACE) | Apache-2.0 | tool calls | 10,581 |
| [glaiveai/glaive-function-calling-v2](https://huggingface.co/datasets/glaiveai/glaive-function-calling-v2) | Apache-2.0 | tool calls | 4,749 |
| Python standard library source files | Python Software Foundation License 2.0 | whole-file edits and coding-agent turns | 18,406 with llama.cpp's |
| llama.cpp source files | MIT | whole-file edits and coding-agent turns | (above) |

**Not used:** prompts from ShareGPT, UltraChat 200k and Magicoder (Evol-Instruct and OSS-Instruct). ShareGPT collects
conversations people shared from ChatGPT, and the terms for reusing them are unclear; LMSYS, who collected them to
train Vicuna, wrote "We will not release the ShareGPT dataset" ([FastChat](https://github.com/lm-sys/FastChat)).
UltraChat and Magicoder carry MIT and Apache-2.0 licences but were generated with OpenAI models. The project's first
drafter heads were trained partly on answers to those prompts; they are not published.

## Where the answers come from

| Writer | Engine | Answers | Answer tokens |
|---|---|---|---|
| Unsloth's Flint Q8_0 | llama.cpp (llama-server) | RIG-TBD | RIG-TBD |
| G3, this project's mix | llama.cpp | RIG-TBD | RIG-TBD |
| Unsloth's UD-Q4_K_XL | llama.cpp | RIG-TBD | RIG-TBD |
| The makers' NVFP4 GGUF | vLLM | RIG-TBD | RIG-TBD |

Sampling for every writer: temperature 1.0, top-k 20, top-p 0.95, min-p 0, reasoning kept, at most 2,048 answer
tokens, seed 20000 + the prompt's id number. Answers sampled on GPUs differ between machines in the details, so only
these files reproduce the drafter's exact training text; the writers and seeds above regenerate an equivalent set.

## What is not here, and how to recompute it

- **The hidden-state dumps** (~58 GB per million answer tokens): each answer replayed through Flint Q8_0 with
  `llama-staircase-dump`, which records 26 layers' inputs and the top-20 next-token probabilities per position
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
