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

# Flint-Pascal-Q5: Flint for old NVIDIA cards and crypto-mining rigs

**What this is.** [Flint](https://huggingface.co/RecursiveSI/flint) is a 39.8-billion-parameter AI model that thinks
before it answers, writes code and calls tools, and reads up to 262,144 tokens of context. This file is Flint in
29.4 GB, made for old NVIDIA "Pascal" cards (compute capability 6.1: the P106 and P104 cards of retired mining rigs, the
GTX 10 series, the Tesla P4/P40); its speed is measured on P106-090 mining cards only. **Q5** because most of its weights
are stored in 5-bit blocks (Q5_0) and the rest in 8-bit blocks (Q8_0): 5.9 bits per weight on average, against 6.0 for
Unsloth's UD-Q5_K_XL and 8.5 for its Q8_0.

**Why it is faster there.** A model file stores its weights compressed. Modern GPUs have compute to spare, so the
popular files use elaborate formats that get more quality out of each bit. Old Pascal mining cards have few compute
units and spend real time unpacking those formats. This file uses only the two simplest formats (Q5_0 and Q8_0), which
the [Pascal engine](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs) decodes fastest. The price is some
quality: Unsloth's UD-Q5_K_XL, the same size in elaborate formats, stays closer to the original model, but on these
cards it ran only at the speed of the much larger Q8_0. In the test below this file is 22 % faster.

**How fast.** A coding agent fixing twelve small bugs on a Manli mining rig with nine P106-090 cards (6 GB each, every
card on a single PCIe 1.0 lane, a 2-core Celeron host): it generated **53.8 tokens per second** over its sessions and
finished all twelve tasks in 17 minutes, a median of 1.2 minutes per task. Unsloth's UD-Q5_K_XL in the same test: 44.2
tokens per second, 25 minutes for the twelve tasks. This file on stock llama.cpp: 15.6 tokens per second (at 65,536
tokens of context).

**Which Flint file should you pick?**
- **Nine 6 GB cards, best quality: Unsloth's [Q8_0](https://huggingface.co/unsloth/flint-GGUF)** (42.3 GB). It is
  near-lossless (it picks the original model's next word 93.3 % of the time) and fits the full 262,144-token context on
  nine cards; with the Pascal engine it generates 44.6–45.9 tokens per second in the same agent test.
- **Six to eight cards, or the most speed on nine: this file.** It is a third smaller than Q8_0, runs on six cards with a
  shorter context, and is 17–21 % faster than Q8_0 and 22 % faster than UD-Q5_K_XL on nine, at a lower quality than
  both (it picks the original's next word 88.8 % of the time, UD-Q5_K_XL 90.7 %).

## Run it in three steps

1. **Driver:** Ubuntu 24.04 or newer with NVIDIA's driver 580, the last one for Pascal cards
   ([setup guide](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/setup.md)).
2. **Download** the Pascal engine, Miner Llama v2 (the two archives of its
   [release](https://github.com/Cinderella-Man/llama.cpp/releases)), unpack both into one folder,
   and download `flint-pascal-Q5.gguf` from this page.
3. **Start it**, here on nine cards with the full context:

   ```bash
   llama-p106-v2.0/llama-server -m flint-pascal-Q5.gguf --alias flint -ngl 99 -fa on -c 262144 -np 1 -b 512 -ub 128 -bs --ctx-checkpoints 4 --cache-ram 0 --pipe-depth 4 --host 127.0.0.1 --port 8080
   ```

   Then open `http://127.0.0.1:8080` through an SSH tunnel from your PC (`ssh -N -L 8080:127.0.0.1:8080 <your rig>`),
   or point any OpenAI-compatible client at it.

What each flag does, how to run on six cards, and how to use it for coding agents:
[the Flint page](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/models/flint/README.md).
A trained guesser for more speed is planned for a later release
([how it works](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/docs/drafter.md)).

## What you need

- **About 30 GB of GPU memory plus room for the context:** six 6 GB cards for shorter conversations, all nine for the
  full 262,144 tokens.
- **The Pascal engine** for the speed. The file also works in stock llama.cpp (tested with release b11820) and in apps
  built on it, just without the Pascal speed-up.
- **On a modern GPU or a Mac, use Unsloth's UD-Q5_K_XL instead:** the same size and closer to the original; this file
  is tuned for Pascal cards only.

## Speed

The measure is a coding agent at work, because that is what a rig like this is for: the
[agent check](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/bench/agent_check) gives the
[pi](https://github.com/earendil-works/pi) coding agent twelve small bug-fix tasks (Python and JavaScript, real test
runs, five to ten tool calls each, the context growing every turn) within 25 minutes, and records every turn. Nine
P106-090, 262,144-token context, the command line above unless noted, 2026-10-26:

| File and engine | Tasks done in the 25 minutes (passed) | Minutes per task (median) | Generation, tokens/s | Prompt reading, tokens/s | First token (median) |
|---|---|---|---|---|---|
| **This file, Pascal engine** | **12 (12), all done in 17 min** | 1.2 | **53.8** | 178 | 2.6 s |
| Unsloth Q8_0, Pascal engine (two runs) | 12 (12) and 12 (12) | 1.1, 1.4 | 45.9, 44.6 | 176, 185 | 2.6 s |
| Unsloth UD-Q5_K_XL, Pascal engine | 12 (12) in 25 min | 1.0 | 44.2 | 185 | 2.6 s |
| This file, stock llama.cpp, 65,536-token context | 11 (10) | 2.1 | 15.6 | 143 | 3.9 s |
| Unsloth Q8_0, stock llama.cpp, 65,536-token context | 9 (8) | 2.9 | 14.4 | 144 | 3.7 s |

- **One run per row** unless noted; two runs of the same setup differed by 3 % (Q8_0: 44.6 and 45.9).
- **UD-Q5_K_XL** reads about as many bytes per token as this file, but its K-quant blocks take these cards more work to
  unpack: in this test it ran at Q8_0's speed.
- **Stock llama.cpp** runs at 65,536 tokens of context, with 35 minutes for the tasks: at 262,144 its attention mask,
  built on the host, does not fit the rig's 4 GB of RAM. In this test the Pascal engine made this file 3.4 times as
  fast.
- **Prompt reading** above is the agent's turns, a few hundred to a few thousand new tokens each (85–89 % of every
  prompt comes from the cache). A whole fresh prompt reads faster: 767 tokens per second for 8,192 tokens with this
  file and larger batches (`-b 8192 -ub 512`, which fit at shorter contexts), against 593–602 on stock llama.cpp.

## Quality

Against Flint's original 16-bit weights, 80 × 512 tokens of wikitext-2:

| File | Size | Picks the same next word as the original | Distance from the original (KL divergence, lower is better) | Perplexity change |
|---|---|---|---|---|
| Unsloth Q8_0 | 42.3 GB | 93.3 % | 0.052 | +3.3 % |
| Unsloth UD-Q5_K_XL | 29.9 GB | 90.7 % | 0.085 | +3.1 % |
| **This file** | **29.4 GB** | **88.8 %** | **0.111** | **+2.4 %** |

- **UD-Q5_K_XL and this file** were measured side by side in one run, part of their expert weights computed on the
  CPU; with every weight on the GPUs this file measures 0.105 and +2.7 %, and Q8_0's row is from such a run.
- **Perplexity** changes by a few percent for every file and does not rank these three reliably; the share of words
  picked the same and the KL divergence do.
- **The engine keeps the answers:** its kernels change only the order in which numbers are added up, so results match
  stock llama.cpp within rounding, and pipelined speculation gives exactly the tokens of plain decoding, checked token
  by token.

## How we got there

**The file.** Flint-Pascal-Q5 (its recipe is called K2h-r16 in the project's notes, measurements and rebuild script)
came out of a search over mixes of Q5_0 and Q8_0, chosen tensor group by tensor group (attention,
the shared expert, the routed experts, per layer), each candidate scored by its KL divergence against the 16-bit model,
the leading ones checked on the tool-calling tasks against UD-Q4_K_XL and timed on the rig. On top of the mix:
- **Unsloth's importance matrix** weights the rounding toward the inputs that matter.
- **The output head** is Q5_0 rounded with GPTQ (rounding that corrects each column's error in the next ones).
- **The router** that picks each token's experts is stored in 16-bit floats.
- **Only Q5_0 and Q8_0**, so every quantized weight goes through the Pascal kernels' fastest path, and the persistent expert
  kernel below runs on the file's expert layers.

**The engine** ([the fork](https://github.com/Cinderella-Man/llama.cpp), `pascal-p106` branches). Each step measured
on these cards:

| Change | Effect |
|---|---|
| v1: CUDA graphs switched on for Pascal (llama.cpp switches them off before Volta), warps per block tuned for one-token products, Pascal matrix-vector kernels for Q8_0, Q5_0 and the other common formats | 1.97–2.00× stock llama.cpp in a long chat on the three Flint files measured |
| FP32 activation sums in the Pascal kernels | +9.6 % |
| The shared expert fused with its activation | +0.7 % |
| Prompt processing ([upstream PR #31047](https://github.com/ggml-org/llama.cpp/pull/31047)) | prompts +22–29 % |
| A persistent kernel for each expert layer at one token (one launch instead of 13) | +5.6–8.8 % per token |
| A Pascal attention kernel for long prompts | prompts +10 % at 8,192 tokens, +3 % at 32,768 |
| Tool requests sampled on the GPU, the grammar checked on the host | tool requests +13–15 % |
| Pipelined speculation: the idle cards check guessed tokens (from the request's own text) while the later cards finish the current one, `--pipe-depth 4` | +52 % in the agent test on Q8_0 (29.3 → 44.6 tokens/s at depth 3); this file 48.6 at depth 3, 53.8 at depth 4 |

Every number with its command, date and conditions:
[results.md](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/results.md);
the search for the file: [quants/](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/tree/main/quants).

## Rebuild it yourself

Everything needed to rebuild this file from Flint's original weights is public: the exact list of formats per
tensor, a script that runs every step, and per-tensor checksums to compare your result with this file. An output head
computed from scratch matches within noise, not bit for bit; taking it from this file gives an identical rebuild.
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
| `flint-pascal-Q5.gguf` | 29.4 GB | see [CHECKSUMS](CHECKSUMS) |

Check a download with `sha256sum -c CHECKSUMS`; the same checksums are in the
[GitHub repository](https://github.com/Cinderella-Man/llms-on-crypto-mining-rigs/blob/main/release/flint-v2/CHECKSUMS).
