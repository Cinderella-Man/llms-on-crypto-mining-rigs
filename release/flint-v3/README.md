# Flint on Miner Llama v3

Flint runs on the v3 build at v2's speed with v2's command line. This folder says which file to download, the one
command that starts it, and its preset. The engine's release notes are [notes-v3.0.md](../notes-v3.0.md); the v2 release
with every flag explained is [release/flint-v2](../flint-v2/README.md), and the model's page is
[models/flint](../../models/flint/README.md).

## Speed

A coding agent fixing twelve small bugs, with real tool calls and test runs ([bench/agent_check](../../bench/agent_check/README.md)),
on nine P106-090 cards; Flint's 8-bit lane: Unsloth's Q8_0, and my Flint Pascal Q8 GGUF (`p065`) on the last row:

| Engine and line | Context | Tasks passed | Generation tok/s | Prompt tok/s | First token per turn, s (median, p90) | Data |
|---|---|---|---|---|---|---|
| Stock llama.cpp (v2's run, an older commit) | 65,536 | 8 of 12 | 14.4 | — | 3.7 | [results.md](../flint-v2/results.md) |
| Stock llama.cpp b12076 (`20a51f24a`), tuned: `-b 2048 -ub 512 -fa on` | 65,536 | 10 of 12 | 14.4 | 148 | 4.13, 8.04 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| Miner Llama v2.0 | 262,144 | 12 of 12 | 45.9 and 44.6 (two runs) | — | 2.6 | [results.md](../flint-v2/results.md) |
| **v3 (`1f7215948`), the preset: v2's line** | 262,144 | **12 of 12** | **45.9** | 168 | 2.67, 6.48 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| v3, + the re-read fix, the tail checkpoint rule and the keep-alive | 262,144 | 12 of 12 | 44.7 | 168 | 2.45, 6.89 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| v3 (`1551f5fe1`), v2's line, Flint Pascal Q8 GGUF | 262,144 | 12 of 12 | 47.4 | 175 | 2.66, 6.75 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |

- Nothing is lost against v2.0: 45.9 tok/s equals v2.0's faster run (45.9 and 44.6 in two runs).
- Against stock llama.cpp b12076 the preset generates 3.2 times as fast (45.9 against 14.4 tok/s) and passes 12 of 12
  tasks against 10 of 12, with a median task of 1.2 against 2.9 minutes. Stock runs at 65,536 tokens, because at 262,144
  its host-side attention mask does not fit in 4 GB, on the line its llama-bench pass chose; its two misses ran into the
  12-minute cap.
- The second v3 row sets `LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096 GGML_CUDA_KEEPALIVE=decode`. One run
  each, 1.2 tok/s apart, about the spread between v2.0's own two runs (45.9 and 44.6), so the preset stays v2's line. Each of v3's
  additions on its own against the preset, on one rig: the re-read fix (with the tail checkpoint rule) is negligible,
  2.59 against 2.60 minutes a task at equal work, with the first token at 2.25 against 2.64 s (median), one run each
  ([2026-11-06_flint_side_rig2](../../docs/data/2026-11-06_flint_side_rig2/README.md)); the keep-alive, the staged
  hop, the answer store and the negligible group were not run in v3.0.
- The expert skip is not in Flint's preset: at p = 0.9 its KL divergence on Unsloth's Q8_0 is above noise on both texts (agent +0.08832 ± 0.02050, wiki +0.12198 ± 0.00569; [2026-11-05_night_quant_chain](../../docs/data/2026-11-05_night_quant_chain/README.md#results)). The Q8_0 hop cast is not either: Flint's graph does not call it, so the switch changes nothing there.

Quality: on wikitext-2 the release build's KL divergence against v2.0 is −0.000000 ± 0.000000 (0.000056 at most), with
the same top token 100.000 % of the time and the same perplexity, 24.4293: prompt processing matches v2.0 to rounding.
At temperature 0 the generated tokens leave v2.0's after 123, 60, 99 and 133 tokens on four test prompts, each time at
a near tie (v2.0's two best tokens 0.011, 0.024, 0.056 and 0.343 nats apart), since one-token decoding rounds a little
differently and a generation compounds it
([data](../../docs/data/2026-11-05_release_regression/README.md#how-large-the-difference-is)).

Flint Pascal Q8 GGUF generates 3.3 % faster than Unsloth's Q8_0 on the release build (47.4 against 45.9 tok/s), with
a median task of 1.0 against 1.2 minutes; its sessions wrote more text, so all twelve tasks took 22.3 against 19.2
minutes. Its KL divergence against the 16-bit model is within two standard errors of Q8_0's, 0.0656 ± 0.0029 against
0.0627 ± 0.0032, in 40.4 GB against 42.3 ([2026-10-27_flint_near_q8](../../docs/data/2026-10-27_flint_near_q8/README.md)).

Stock llama.cpp on Flint Pascal Q8 GGUF, and Flint's second lane, 5-bit (Unsloth's UD-Q5_K_XL and Flint Pascal Q5
GGUF, each on stock and on v3): TODO(flintq5).

## The files

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `flint-Q8_0.gguf` | 42.3 GB | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) | the SHA-256 Hugging Face lists for the file |
| `flint-pascal-Q8.gguf` (Flint Pascal Q8 GGUF) | 40.4 GB | [CinderellaMan/Flint-Pascal-Q8-GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF) | `15c50169…` in its [CHECKSUMS](huggingface/Flint-Pascal-Q8-GGUF/CHECKSUMS) |

```bash
wget -c -P /mnt/models https://huggingface.co/unsloth/flint-GGUF/resolve/main/flint-Q8_0.gguf
# or my file, Flint Pascal Q8 GGUF:
wget -c -P /mnt/models https://huggingface.co/CinderellaMan/Flint-Pascal-Q8-GGUF/resolve/main/flint-pascal-Q8.gguf
```

Flint Pascal Q8 GGUF runs on the same line: after sourcing the preset below, set `PRESET_MODEL=flint-pascal-Q8.gguf`.

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
requests, pipelined speculation with `--pipe-depth 4`) is on by default. No v3 switch beat this line in the
side-by-side (the re-read fix negligible; the keep-alive, the staged hop, the answer store and the negligible group
not run in v3.0), so `PRESET_ENV` stays empty.

## The recipe: every setting and what it bought

The rig has nine 6 GB cards, each on one PCIe 1.0 lane, a 2-core Celeron and 4 GB of host memory. Almost every
setting below works around one of those. Each speed is generation tok/s from one run of the coding-agent check, Unsloth's
Q8_0, unless a row says otherwise.

### The ladder: from stock llama.cpp to 45.9 tok/s

One change at a time:

| Step | What I added | Context | Tasks passed | Generation tok/s | Data |
|---|---|---|---|---|---|
| 0 | stock llama.cpp (v2's run, an older commit) | 65,536 | 8 of 12 | 14.4 | [results.md](../flint-v2/results.md) |
| 0 | stock llama.cpp b12076 (`20a51f24a`), the upstream release v3 is built on | 65,536 | 10 of 12 | 14.4 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| 1 | Miner Llama v2.0's engine: the Pascal kernels, the persistent expert kernel, GPU sampling for tool requests; and the full context | 262,144 | 9 of 12 | 29.3 | [results.md](../flint-v2/results.md) |
| 2 | + pipelined speculation, two tokens ahead (`--pipe-depth 2`) | 262,144 | 10 of 12 | 40.1 | [results.md](../flint-v2/results.md) |
| 3 | three tokens ahead (`--pipe-depth 3`), two runs | 262,144 | 11 and 9 of 12 | 44.6 and 44.6 | [results.md](../flint-v2/results.md) |
| 4 | four tokens ahead (`--pipe-depth 4`): v2.0's line, two runs | 262,144 | 12 and 12 of 12 | 45.9 and 44.6 | [results.md](../flint-v2/results.md) |
| 5 | the v3 build (`1f7215948`), the same line: **the preset** | 262,144 | 12 of 12 | **45.9** | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |
| 6 | + the re-read fix, the tail checkpoint rule and the keep-alive | 262,144 | 12 of 12 | 44.7 | [2026-11-05_release_regression](../../docs/data/2026-11-05_release_regression/README.md) |

- Stock runs at 65,536 tokens: at 262,144 its host-side attention mask does not fit in 4 GB. Step 1 is v2.0's whole
  engine at once; what each part of it is worth is under "On by default" below.
- Steps 1 to 4 ran on the v2.0 release engine on 2026-10-26, so they compare cleanly. Step 6 is 1.2 tok/s below step
  5, one run each, about the spread of v2.0's own two runs at depth 4. That is negligible, so step 6 is not in the
  preset; each of its parts on its own against step 5's line: the re-read fix is negligible (2.59 against 2.60 minutes
  a task at equal work, rig 2); the others were not run in v3.0.

### Every setting

| Setting | What it does | Why this rig needs it | What it measured | Data |
|---|---|---|---|---|
| `--pipe-depth 4` | while the later cards finish the current token, the idle earlier cards start on up to four guessed tokens, taken from the request itself; the text is the same as without it | on a layer split only one card works at a time | 45.9 and 44.6 tok/s (two runs) against 29.3 without it | [results.md](../flint-v2/results.md) |
| `-ts 6,6,5,5,6,5,5,6,6` | how many layers each card holds, chosen by memory instead of by count | 42.3 GB plus the 262,144-token cache must fit nine separate 6 GB pools; llama.cpp's even split puts two full-attention layers, ~6.3 GB, on one card | needed to load at this context | [models/flint](../../models/flint/README.md#command-lines) |
| `-b 512 -ub 128` | prompt batches of 128 tokens | the default batch pins ~1 GiB of host memory at 262,144 tokens | with `-b 2048 -ub 256` a card lost the room for pipeline parallelism and a 16k-token prompt read at 133 tok/s against 214; `-b 2048 -ub 512` ran the host out of memory while loading Flint Pascal Q5 GGUF | [results.md](../flint-v2/results.md) |
| `-bs` | samples the next token on the card, also for tool requests, with each token checked against the tool grammar on the host | otherwise 600 KB of scores per token cross the slow link | tool requests at 37.9 against 33.6 tok/s sampled on the host (Flint Pascal Q5 GGUF, temperature 0, 2,011 tokens) | [results.md](../flint-v2/results.md) |
| `--cache-ram 0`, `--ctx-checkpoints 4` | no prompt cache in host memory; at most four saved context states | the 4 GB host | not measured on its own: every run on this page used them | — |
| `-ngl 99 -fa on -c 262144 -np 1 --alias flint`, the sampling | every layer on the cards, flash attention, the full context, one request at a time, the name clients see; the makers' sampling settings | ordinary llama.cpp settings | — | — |
| `PRESET_ENV=""` | no switch: v3's additions stay at their defaults | — | the re-read fix on its own: negligible, 2.59 against 2.60 minutes a task at equal work ([data](../../docs/data/2026-11-06_flint_side_rig2/README.md)); the keep-alive, the staged hop, the answer store and the negligible group, each on its own: not run in v3.0 | — |

### On by default

These need no setting. Each has a switch that turns it off ([docs/switches.md](../../docs/switches.md)), and what that
costs where it was measured (llama-bench, Flint Pascal Q5 GGUF on nine cards, unless a row says otherwise):

| Feature | Turning it off | Data |
|---|---|---|
| The fork's Pascal matrix-vector kernels (`GGML_CUDA_MMVQ_PASCAL=1`) | 31.9 against 36.8 tok/s with everything on (they are worth +15.5 %) | [results.md](../flint-v2/results.md) |
| The persistent expert kernel, one kernel per expert layer | 34.0 against 36.8 tok/s (+8 %) | [results.md](../flint-v2/results.md) |
| Dropping llama.cpp's safety rescale of the expert layers, which the fork's 32-bit sums make unneeded; it is what lets each expert layer run as one kernel | `LLAMA_NO_MOE_DOWN_RESCALE=0` keeps the rescale; not measured on its own | [models/flint](../../models/flint/README.md#command-lines) |
| GPU sampling for tool requests | `LLAMA_GRAMMAR_HOST_SAMPLING=1`: 33.6 against 37.9 tok/s (llama-server, temperature 0) | [results.md](../flint-v2/results.md) |
| The prefill-attention kernel | with it, prompts of 8,192 and 32,768 tokens read 10 % and 3 % faster than without | [results.md](../flint-v2/results.md) |
