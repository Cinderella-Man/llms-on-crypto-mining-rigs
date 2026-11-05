# Flint on Miner Llama v3

Flint runs on the v3 build at v2's speed with v2's command line. This folder says which file to download, the one
command that starts it, and its preset. The engine's release notes are [notes-v3.0.md](../notes-v3.0.md); the v2 release
with every flag explained is [release/flint-v2](../flint-v2/README.md), and the model's page is
[models/flint](../../models/flint/README.md).

## Speed

A coding agent fixing twelve small bugs, with real tool calls and test runs ([bench/agent_check](../../bench/agent_check/README.md)),
Unsloth's Q8_0 on nine P106-090 cards:

| Engine and line | Context | Tasks passed | Generation tok/s | Prompt tok/s | First token per turn, s (median, p90) | Data |
|---|---|---|---|---|---|---|
| Stock llama.cpp (v2's run, an older commit) | 65,536 | 8 of 12 | 14.4 | — | 3.7 | [results.md](../flint-v2/results.md) |
| Stock llama.cpp b12076 (`20a51f24a`) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) |
| Miner Llama v2.0 | 262,144 | 12 of 12 | 44.6–45.9 | — | 2.6 | [results.md](../flint-v2/results.md) |
| **v3 (`1f7215948`), the preset: v2's line** | 262,144 | **12 of 12** | **45.9** | 168 | 2.67, 6.48 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| v3, + the re-read fix, the tail checkpoint rule and the keep-alive | 262,144 | 12 of 12 | 44.7 | 168 | 2.45, 6.89 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |

- Nothing is lost against v2.0: 45.9 tok/s is the top of v2.0's range.
- The second v3 row sets `LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 GGML_CUDA_KEEPALIVE=decode`. One run
  each, 1.2 tok/s apart, about the spread of v2.0's own runs (44.6–45.9), so the preset stays v2's line. Each of v3's
  additions on its own against the preset (the keep-alive, the re-read fix, the staged hop, the answer store, the
  negligible group): TODO(flintside).
- The Q8_0 hop cast and the expert skip are not in Flint's preset: each needs Flint's own quality check first.

Quality: v3's greedy tokens leave v2.0's after 60–133 tokens on four test prompts, each time at a token whose
two best scores were close; the KL divergence against v2.0: TODO(kld).

## The file

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `flint-Q8_0.gguf` | 42.3 GB | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) | the SHA-256 Hugging Face lists for the file |

```bash
wget -c -P /mnt/models https://huggingface.co/unsloth/flint-GGUF/resolve/main/flint-Q8_0.gguf
```

## Run it

From a clone of this repository, with the v3 engine unpacked beside it ([notes-v3.0.md](../notes-v3.0.md), "Install"):

```bash
MODELS=/mnt/models; . llms-on-crypto-mining-rigs/release/flint-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" $PRESET_ARGS --host 127.0.0.1 --port 8080
```

Then connect from your PC through an SSH tunnel ([README](../../README.md#run-it)).

## The preset: [settings.env](settings.env)

```bash
PRESET_MODEL=flint-Q8_0.gguf
PRESET_ENV=""
PRESET_ARGS="--alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6 -bs --ctx-checkpoints 4 --cache-ram 0 --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --pipe-depth 4"
```

This is v2's command line, unchanged; every flag is explained on the [Flint page](../../models/flint/README.md#command-lines).
No switch is set: every fork feature Flint uses (the Pascal kernels, the persistent MoE kernel, GPU sampling for tool
requests, pipelined speculation with `--pipe-depth 4`) is on by default. TODO(flintside): the switches that win in the
side-by-side go into `PRESET_ENV`.
