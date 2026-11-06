# Qwen3.8-Flash-Next on Miner Llama v3: a 125B-parameter model on a mining rig

Flash-Next is Miner Llama v3's headline model. This folder says which files to download, the one command that starts
it, and the preset it was measured with; the engine's release notes are [notes-v3.0.md](../notes-v3.0.md). The model's
own page is [models/qwen3.8-flash-next](../../models/qwen3.8-flash-next/README.md).

## The headline: 20.40 tok/s for a coding agent on nine 6 GB mining cards, a 2-core Celeron and 4 GB of RAM

Qwen3.8-Flash-Next is a 125B mixture-of-experts model with about 6B active parameters per token. Its original weights
add a 51B-parameter n-gram table and a 4B multi-token-prediction block, 180B in all. In UkisAI's Swift-1.5 fine-tune at 2 bits it runs on a rig
built for mining: the transformer split over nine 6 GB cards on single PCIe 1.0 lanes, the 28.8 GB n-gram table read
from the rig's SSD.

A coding agent fixing twelve small bugs, with real tool calls and test runs, 65,536 tokens of context
([bench/agent_check](../../bench/agent_check/README.md)), UkisAI's Swift-1.5 GSQ-RCO Q2_0:

| Engine and line | Tasks passed | Generation tok/s | Prompt tok/s | First token per turn, s (median, p90) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp `d77256999`, no draft (2026-11-03) | 7 of 8 tried | 6.4 | — | — | [2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| Miner Llama v2.0's code, no draft | 9 of 9 | 7.8 | 59 | 8.70 | [2026-11-04_mtp_redo](../../docs/data/2026-11-04_mtp_redo/README.md) |
| Stock llama.cpp b12076 (`20a51f24a`), with the MTP draft, `-ub 512` | 9 of 9 (3 not reached) | 8.20 | 53.7 | 10.22, 24.85 | [2026-11-05_release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |
| **Miner Llama v3 (`1551f5fe1`), the preset** | **12 of 12** | **20.40** | **69.4** | **6.89, 15.89** | [2026-11-05_release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |

- **The engine halves a task's time.** On the nine tasks stock reached, Miner Llama v3 does a task's work in 1.90
  minutes against stock's 3.74, and generates 2.49 times as fast.
- **Stock's line** is the preset's server line with the MTP draft and without the fork's switches. Stock's llama-bench
  pass read a 2,048-token prompt fastest at `-ub 256` (153.44 against 138.93 tok/s at `-ub 512`); on the agent check
  `-ub 256` was not faster (3.78 against 3.74 minutes a task), so stock's row is `-ub 512`.
- **The draft's guesses** are accepted 0.718 of the time in the preset, 3.15 tokens a step. With the keep-alive a task
  takes 8.4 % less time than without it, at equal work ([2026-11-05_keepalive](../../docs/data/2026-11-05_keepalive/README.md)).
- **My Pascal type mix of the same file** (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, `pmix-q2-fast`: the dense and
  shared-expert weights in types the cards read faster) runs the preset at 20.75 tok/s, 12 of 12 (`t1base`, on
  `1f7215948`, which gives the same greedy output as `1551f5fe1`). That is 2.45 against 2.47 minutes a task at equal
  work, inside the 3.7 % between two runs of one line, so the release uses UkisAI's file as published. On stock the mix
  runs at 8.47 tok/s. Its recipe: [2026-11-04_pascal_mix](../../docs/data/2026-11-04_pascal_mix/README.md).
- **Left out of the preset**, each tried on the mix against its 20.75: the expert skip at p = 0.9 is negligible
  (`t1skip`: 11 of 12, one task ran into the 12-minute cap, 21.29 tok/s, −2.9 % a task at equal work, inside the 3.7 %
  between two runs); every switch of the "negligible" group together is a loss (`t1neg`: 12 of 12, 19.05 tok/s, +5.3 %
  a task).

How v3 got there from 7.8 tok/s, one change at a time: [docs/book/10-v3.md](../../docs/book/10-v3.md).

## The files

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`, `…-00002-of-00003.gguf` (the transformer, on the cards) | 37.7 GB | cut from UkisAI's Swift-1.5 GSQ-RCO Q2_0 download with `gguf_reshard.py` (below) | `690252d2…`, `a893db93…` in [CHECKSUMS](CHECKSUMS) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf` (the n-gram table, on the SSD) | 28.8 GB | the same cut | `05716382…` in [CHECKSUMS](CHECKSUMS) |
| `mtp-Qwen3.8-Flash-Next-draft.gguf` (the MTP draft) | 2.7 GB | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), revision `273edc1d2ffae5241fe3cab69568d682dad4c718` | `00818378…` in [CHECKSUMS](CHECKSUMS) |
| Miner Llama v3.0 engine archives | TODO(tag) | the fork's [GitHub release](https://github.com/Cinderella-Man/llama.cpp/releases) | the release's `SHA256SUMS` |

The three model shards are UkisAI's two Q2_0 shards byte for byte, with the n-gram table moved into a third shard of
its own: the table can then live on the SSD while shards 1 and 2 load to the cards.

The model is under the Swift Open License 1.0 (free below US$1M of yearly revenue) and the Qwen Community License 1.0.

## Run it

You need the rig, about 30 GB free on its internal SSD, a model drive with 41 GB free, and a PC with Python 3 and
room for Swift's 66.5 GB download and the cut copy of it (another 66.5 GB). Another card count needs its own `-ts`,
checked with [scripts/fit_probe.sh](../../scripts/fit_probe.sh) before the first load.

1. **Download** the MTP draft and Swift's two Q2_0 shards at the revision measured here:

   ```bash
   hf download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF mtp-Qwen3.8-Flash-Next-draft.gguf --revision 273edc1d2ffae5241fe3cab69568d682dad4c718 --local-dir swift
   ```

   ```bash
   hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --include '*Q2_0-0000*-of-00002.gguf' --local-dir swift
   ```

2. **Cut the n-gram table out of Swift's download** into a shard of its own (copies every byte, ~8 minutes on an NVMe
   drive), then check the new files against the downloaded ones:

   ```bash
   python3 -I scripts/gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir swift_q2_0 swift/*Q2_0-0000*-of-00002.gguf
   ```

   ```bash
   python3 -I scripts/gguf_reshard.py verify swift/*Q2_0-0000*-of-00002.gguf --new swift_q2_0/*-of-00003.gguf
   ```

   The split is deterministic: `sha256sum -c --ignore-missing CHECKSUMS` in `swift_q2_0/` checks the three new files.

3. **Copy the new shards 1 and 2 and the draft to the model drive, and shard 3 (the table) to the SSD.** Then link the
   table beside shards 1 and 2, where llama.cpp looks for it:

   ```bash
   ln -s <SSD>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf <model drive>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf
   ```

4. **Start the server** with the preset, from a clone of this repository:

   ```bash
   . llms-on-crypto-mining-rigs/release/flash-next-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" -md "$MODELS/$PRESET_DRAFT" $PRESET_ARGS --host 127.0.0.1 --port 8080
   ```

   with `MODELS` set to the model drive's folder.

5. **Connect from your PC** through an SSH tunnel, as for Flint ([README](../../README.md#run-it)), and keep the host
   quiet while it works: the 4 GB host has a few hundred MiB to spare.

## The preset: [settings.env](settings.env)

The line of the release build's agent check (`t1swift`: 12 of 12, 20.40 tok/s):

```bash
PRESET_MODEL=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf
PRESET_DRAFT=mtp-Qwen3.8-Flash-Next-draft.gguf
PRESET_ENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode"
PRESET_ARGS="-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
```

What each setting does, why the rig needs it and what it bought: [the recipe](#the-recipe-every-setting-and-what-it-bought).

## The recipe: every setting and what it bought

The rig has nine 6 GB cards, each on one PCIe 1.0 lane, a 2-core Celeron and 4 GB of host memory. Almost every
setting below works around one of those. Each speed is generation tok/s from one run of the coding-agent check, unless
a row says otherwise.

### The ladder: from stock llama.cpp to the preset

One change at a time:

| Step | What I added | Tasks passed | Generation tok/s | Data |
|---|---|---|---|---|
| 0 | stock llama.cpp `d77256999`, Swift Q2_0 | 7 of 8 tried | 6.4 | [2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| 0 | stock llama.cpp b12076 (`20a51f24a`), the upstream release v3 is built on, with the MTP draft | 9 of 9 (3 not reached) | 8.20 | [2026-11-05_release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |
| 1 | Miner Llama's Pascal kernels (v2.0's code, v3's base `37e717ff6`) | 9 of 9 | 7.8 | [2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| 2 | the slowest BF16 weights retyped to Q8_0, which the cards read fast | 12 of 12 | 9.0 | [2026-11-03_retype_agent](../../docs/data/2026-11-03_retype_agent/README.md) |
| 3 | v3's kernels for the model's shapes on upstream b12076, and my Pascal type mix of the file; no draft | 11 of 12 | 14.3 | [2026-11-04_stacked_gains](../../docs/data/2026-11-04_stacked_gains/README.md) |
| 4 | + the MTP draft, three tokens ahead | 11 of 12 | 14.6 | [2026-11-04_stacked_gains](../../docs/data/2026-11-04_stacked_gains/README.md) |
| 5 | + the draft's shortlist head | 12 of 12 | 16.0 | [2026-11-04_stacked_gains](../../docs/data/2026-11-04_stacked_gains/README.md) |
| 6 | + newer kernel shapes and the 8-bit hop between cards | 12 of 12 | 18.8 | [2026-11-04_s35_stack](../../docs/data/2026-11-04_s35_stack/README.md) |
| 7 | the same line on the second rig, the control for step 8 | 12 of 12 | 19.19 | [2026-11-05_keepalive](../../docs/data/2026-11-05_keepalive/README.md) |
| 8 | + the keep-alive | 12 of 12 | 20.75 | [2026-11-05_keepalive](../../docs/data/2026-11-05_keepalive/README.md) |
| 9 | the release build (`1f7215948`, the same greedy output as `1551f5fe1`), the same line | 12 of 12 | 20.75 | [2026-11-05_release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |
| 10 | UkisAI's Q2_0 as published instead of my mix, on `1551f5fe1`: **the preset** | 12 of 12 | **20.40** | [2026-11-05_release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |

Steps 3, 4 and 5 are three arms of one session, and steps 7 and 8 two arms of another, so those compare cleanly. Steps
9 and 10 differ only in the file: 2.45 against 2.47 minutes a task at equal work, inside the spread of two runs. The
other steps ran on two identical rigs and on different builds; each one's own side-by-side is in its data folder.

### Every setting

| Setting | What it does | Why this rig needs it | What it measured | Data |
|---|---|---|---|---|
| UkisAI's Swift Q2_0, its n-gram table cut into a third shard (`gguf_reshard.py`) | the same weights, byte for byte; the table in a shard of its own | the 28.8 GB table fits neither the cards nor 4 GB of host memory; the SSD reads it at 6,039 random 4 KiB reads a second, the USB disk at 95 | the table's reads cost about 1 % of a token's time | [table_storage](../../docs/data/2026-11-03_table_storage/README.md), [ngram_replay](../../docs/data/2026-11-03_ngram_replay/README.md) |
| `-md mtp-…gguf`, `--spec-type draft-mtp`, `--spec-draft-n-max 3` | a small draft model guesses three tokens; the main model checks them in one pass and keeps those it agrees with, so the text is the main model's own | checking four tokens costs about twice one token (149.0 against 74.1 ms on nine cards), not four times | 14.6 against 14.3 tok/s on its own; 16.0 with the shortlist below; in the preset on the release build, 0.718 of the guesses are accepted, 3.15 tokens a step | [stacked_gains](../../docs/data/2026-11-04_stacked_gains/README.md), [release_flashnext](../../docs/data/2026-11-05_release_flashnext/README.md) |
| `LLAMA_DRAFT_SHORTLIST=32768,8192,2` | the draft scores a shortlist of likely tokens instead of all 248,320, and falls back to the full list when the shortlist might miss the best one; the guesses are the same | once the main model got fast, scoring every token for every guess ate most of what the draft saved | 16.0 against 14.6 tok/s in the same session; acceptance 0.705 against 0.697 | [stacked_gains](../../docs/data/2026-11-04_stacked_gains/README.md) |
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` | the kernels that read each weight once for all the tokens a draft step checks, with 32 KB of shared memory and without their dispatch table | a card has 5 SMs and a slow memory; reading a weight once instead of four times is most of a draft step's saving | a pass over five tokens 180.2 ms against 188.2 at the defaults; one token 74.1 against 80.6 ms | [kshape](../../docs/data/2026-11-04_kshape/README.md) |
| `LLAMA_HOP_CAST=q8_0` | the model's state crosses each of the eight card boundaries in 8-bit Q8_0 instead of 32-bit floats: a quarter of the bytes | each card has one PCIe 1.0 lane (~190 MB/s) and no direct path to the next card, so every boundary goes through the host | decode +12 %, +13 % and +15 % on a short, a ~1k and a ~8k-token request; prompts +22 % and +31 % on the ~1k and ~8k ones; no measurable change in KL divergence | [card_hops](../../docs/data/2026-11-04_card_hops/README.md) |
| `GGML_CUDA_HOP_STAGED=1` | the hop goes through a ring of pinned host memory that the next card reads, instead of the driver's copy | the same links | on Q8_0 hops negligible: decode −1.6, +0.1 and +0.1 % on those three requests, prompts faster on all five by no more than two loads differ; kept as a small gain. With the cast: plain decoding 17.06 against 14.98 tok/s | [card_hops](../../docs/data/2026-11-04_card_hops/README.md), [s35_stack](../../docs/data/2026-11-04_s35_stack/README.md) |
| `GGML_CUDA_KEEPALIVE=decode` | a one-thread loop on each card's lowest-priority stream while the model decodes, so no card looks idle | on a layer split each card works about a ninth of the time, so the driver held every card at 1,354 MHz instead of its boost clock | 20.75 against 19.19 tok/s (+8.1 %) and prompts 77.4 against 68.8 (+12.5 %) in the same session; the nine cards draw 72 W more while decoding | [keepalive](../../docs/data/2026-11-05_keepalive/README.md) |
| `-otd token_embd=CUDA4`, `--spec-draft-n-ubatch 128`, `--no-host` | the draft's 644 MiB token table on the fifth card instead of host memory; a smaller batch for the draft; no pinned host copies of the weights | 4 GB of host memory leaves a few hundred MiB free while the model runs | without them the draft did not load: usable host memory fell to 133 MiB and the watchdog stopped it. With them the agent check ran to the end, at 129 MiB at the lowest | [mtp_redo](../../docs/data/2026-11-04_mtp_redo/README.md), [draft_mem](../../docs/data/2026-11-04_draft_mem/README.md) |
| `-lm dio` | loads the files with direct reads instead of mapping them into memory | the memory map first fills 37.7 GB of mappings, which 4 GB cannot hold | the load takes 326 s from the USB disk; with the memory map it was stopped after 30 GB had been read and nothing had reached the cards | [flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| `--cache-ram 0`, `--ctx-checkpoints 4` | no prompt cache in host memory; at most four saved context states | the same 4 GB | not measured on its own: every run on this page used them | — |
| `-ts 6,6,6,6,5,6,5,6,3`, `-fit off` | how many layers each card holds; skips llama.cpp's own fit pass | the model and the draft must fit nine separate 6 GB pools; the last card also holds the output layer | a fit setting, not a speed one; check another card count with [fit_probe.sh](../../scripts/fit_probe.sh) | — |
| `-bs` | samples the next token on the card | otherwise the scores for 248,320 tokens (est. 1 MB) cross the slow link every token | not measured on its own here | — |
| `-ngl 99 -fa on -c 65536 -np 1 -b 2048 -ub 512 --jinja --reasoning-format deepseek`, the sampling | every layer on the cards, flash attention, the context, one request at a time, prompt batches of 512, the model's chat template with its reasoning kept apart; the makers' sampling settings | ordinary llama.cpp settings | — | — |

### On by default

These need no setting. Each has a switch that turns it off ([docs/switches.md](../../docs/switches.md) lists all of
them), and what that costs where it was measured:

| Feature | Turning it off | Data |
|---|---|---|
| The fork's Pascal matrix-vector kernels (`GGML_CUDA_MMVQ_PASCAL=1`) | not measured alone on Flash-Next; on Flint they are worth +15.5 % (llama-bench, 36.8 against 31.9 tok/s) | [Flint v2 results](../flint-v2/results.md) |
| The multi-token expert kernel, which reads each routed expert once for every token of a draft step (`GGML_CUDA_MMVQ_PASCAL_MTOK`) | `MTOK=0`: a pass over five tokens takes 221.0 ms instead of 188.2 at the defaults | [kshape](../../docs/data/2026-11-04_kshape/README.md) |

## The machine: how little there is

| Part | The rig | For scale |
|---|---|---|
| GPUs | 9 × NVIDIA P106-090 (Pascal, compute capability 6.1), **6 GB each**, 54 GB in nine separate pools; 5 SMs, no tensor cores, 75 W each | the model's file is 66.5 GB |
| GPU links | **one PCIe 1.0 lane per card** (~190 MB/s measured), eight of them behind two 4-port switches; no GPU-to-GPU access | a desktop card's 16 lanes of PCIe 3.0 or newer carry about 60× that |
| CPU | Intel Celeron 3865U: **2 cores at 1.8 GHz, no AVX**, 15 W | it runs no part of the model |
| Host memory | **4 GB DDR4**, one channel | the n-gram table alone is 28.8 GB |
| Storage | 64 GB SATA SSD inside; the model files on a 4.5 TB USB 3 hard disk (~127 MB/s) | — |
| Software | Ubuntu 24.04 desktop, NVIDIA driver 580, CUDA 12.9: the last versions for Pascal | — |

It is a Manli M-P106L9-N6G, one of thousands built for mining. Measured details: [docs/mining_rig.md](../../docs/mining_rig.md).

## Where it started: the baseline of 2026-11-03

On 2026-11-03 the model ran on the rig for the first time, Swift-1.5 Q2_0 with stock llama.cpp and with v2.0's code,
single requests ([data](../../docs/data/2026-11-03_flash_next_baseline/README.md)):

| Swift-1.5 GSQ-RCO Q2_0, nine P106-090 cards, 65,536-token context, one request | Stock llama.cpp `d77256999` | Miner Llama v3's base `37e717ff6` |
|---|---|---|
| Reading a 1.4k / 7.2k / 32.7k-token prompt | 59 / 116 / 128 tok/s | 72 / 139 / 149 tok/s |
| Time to the first token for those prompts | 23 / 62 / 257 s | 20 / 52 / 221 s |
| Writing the answer | 5.7–7.2 tok/s | 5.4–8.9 tok/s |
| Loading from a USB 3 hard disk | 322 s | 326 s |
| Host memory available while it works | 230–410 MiB of 4 GB | 330–460 MiB |

`GGML_CUDA_PROFILE` put a token at 136 ms of GPU time across the nine cards, 30 % of it in BF16 projections that
Pascal has no fast path for ([decode_profile_v3.txt](../../docs/data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)).
They were v3's first target.

| Question | Answer | Data |
|---|---|---|
| Does it fit nine 6 GB cards? (Q213) | Q2_0 fits with room up to 128k tokens of context, or 262k at `-ub 64`; the better IQ3_XXS file up to 48k | [2026-11-03_flash_next_fit](../../docs/data/2026-11-03_flash_next_fit/README.md) |
| Where can the n-gram table live? (Q214) | on the SSD: 6,039 random 4 KiB reads per second against 95 on the USB hard disk | [2026-11-03_table_storage](../../docs/data/2026-11-03_table_storage/README.md) |
| What do the table's reads cost? (Q228) | about 1 % of a token's time on the SSD at the rig's ~2 GB page cache | [2026-11-03_ngram_replay](../../docs/data/2026-11-03_ngram_replay/README.md) |
| Can the SSD hold it? (Q435) | yes, once what filled it moved to the hard disk (6.2 → 35.8 GB free) | [2026-11-03_ssd_room](../../docs/data/2026-11-03_ssd_room/README.md) |

## Checksums

[CHECKSUMS](CHECKSUMS) is in `sha256sum -c` format; in the folder that holds your files,
`sha256sum -c --ignore-missing CHECKSUMS` checks the ones you have. The split is deterministic: the same downloaded
shards give the same three files.

## What is not measured yet

- **One context size.** 65,536 tokens ran; 128k and 262k are memory projections only.
- **Quality on the agent's own tasks** is measured as KL divergence against Swift's 8-bit file; the full-precision
  reference is Q216, Q217 in [quants/experiments.md](../../quants/experiments.md).
- **Two rigs together** (18 cards, a better file than one rig holds): Q505.
