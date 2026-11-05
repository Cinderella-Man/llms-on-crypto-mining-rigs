# Qwen3.8-Flash-Next on Miner Llama v3: a 180B-parameter model on a mining rig

Flash-Next is Miner Llama v3's headline model. This folder says which files to download, the one command that starts
it, and the preset it was measured with; the engine's release notes are [notes-v3.0.md](../notes-v3.0.md). The model's
own page is [models/qwen3.8-flash-next](../../models/qwen3.8-flash-next/README.md).

## The headline: 20.75 tok/s for a coding agent on nine 6 GB mining cards, a 2-core Celeron and 4 GB of RAM

Qwen3.8-Flash-Next is a 125B mixture-of-experts model with about 6B active parameters per token, plus a 51B-parameter
n-gram table: 180B parameters in its original weights. In UkisAI's Swift-1.5 fine-tune at 2 bits it runs on a rig
built for mining: the transformer split over nine 6 GB cards on single PCIe 1.0 lanes, the 28.8 GB n-gram table read
from the rig's SSD.

A coding agent fixing twelve small bugs, with real tool calls and test runs, 65,536 tokens of context
([bench/agent_check](../../bench/agent_check/README.md)):

| Engine and line | Tasks passed | Generation tok/s | Prompt tok/s | First token per turn, s (median) | Data |
|---|---|---|---|---|---|
| Stock llama.cpp `d77256999`, Swift Q2_0, no draft | 7 of 8 tried | 6.4 | — | — | [2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| Stock llama.cpp b12076 (`20a51f24a`), Swift Q2_0 | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) |
| Stock llama.cpp b12076 (`20a51f24a`), `pmix-q2-fast` | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) | TODO(stock) |
| Miner Llama v2.0's code, Swift Q2_0, no draft | 9 of 9 | 7.8 | 59 | 8.70 | [2026-11-04_mtp_redo](../../docs/data/2026-11-04_mtp_redo/README.md) |
| v3, the preset without the keep-alive | 12 of 12 | 19.19 | 68.8 | 7.27 | [2026-11-05_keepalive](../../docs/data/2026-11-05_keepalive/README.md) |
| **v3, the preset** | **12 of 12** | **20.75** | **77.4** | **6.41** | [2026-11-05_keepalive](../../docs/data/2026-11-05_keepalive/README.md) |

- The two v3 rows ran on the `-keepalive` build (`f6c1c6a4a`), whose code the release build holds; the preset on the
  release build itself: TODO(preset).
- With the preset, a task takes 8.4 % less time than without the keep-alive, at equal work. The draft's guesses are
  accepted 0.69–0.71 of the time, about 3.1 tokens a step.
- The expert skip at p = 0.9 on top of the preset: TODO(t1skip). Every switch of the "negligible" group on top of it:
  TODO(t1neg).

How v3 got from 7.8 to 20.75 tok/s, one change at a time: [docs/book/10-v3.md](../../docs/book/10-v3.md).

## The files

| File | Size | Where from | Checksum |
|---|---:|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf`, `…-00002-of-00003.gguf` (`pmix-q2-fast`, on the cards) | TODO(upload) | TODO(upload): this project's Hugging Face repository | TODO(upload) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf` (the n-gram table, on the SSD) | 28.8 GB | made from UkisAI's Swift-1.5 Q2_0 download with `gguf_reshard.py` (below) | `05716382…` in [CHECKSUMS](CHECKSUMS) |
| `mtp-Qwen3.8-Flash-Next-draft.gguf` (the MTP draft) | 2.7 GB | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF), revision `273edc1d2ffae5241fe3cab69568d682dad4c718` | `00818378…` in [CHECKSUMS](CHECKSUMS) |
| Miner Llama v3.0 engine archives | TODO(tag) | the fork's [GitHub release](https://github.com/Cinderella-Man/llama.cpp/releases) | the release's `SHA256SUMS` |

`pmix-q2-fast` is Swift-1.5's Q2_0 with a Pascal type mix: the dense and shared-expert weights in types the cards read
faster, the routed experts unchanged. It keeps Swift Q2_0's quality (KL divergence against Swift's 8-bit file) at an estimated 10 ms a token less, and
fits the same splits; how it was made, with the recipe to rebuild it:
[2026-11-04_pascal_mix](../../docs/data/2026-11-04_pascal_mix/README.md) and [bench/pascal_mix](../../bench/pascal_mix/).
Its third shard is Swift's n-gram table, byte for byte, so you make it from Swift's own download.

The model is under the Swift Open License 1.0 (free below US$1M of yearly revenue) and the Qwen Community License 1.0.

## Run it

You need the rig, about 30 GB free on its internal SSD, a model drive with 41 GB free, and a PC with Python 3 and
room for Swift's 66.5 GB download. Another card count needs its own `-ts`, checked with
[scripts/fit_probe.sh](../../scripts/fit_probe.sh) before the first load.

1. **Download** `pmix-q2-fast`'s shards 1 and 2 (TODO(upload): the command), the MTP draft, and Swift's two Q2_0 shards
   at the revision measured here:

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

   Only the third new file is needed: `swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf`.

3. **Copy `pmix-q2-fast`'s shards 1 and 2 and the draft to the model drive, and the table to the SSD.** Then link the
   table beside shards 1 and 2 under `pmix-q2-fast`'s name, where llama.cpp looks for it:

   ```bash
   ln -s <SSD>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf <model drive>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00003-of-00003.gguf
   ```

4. **Start the server** with the preset, from a clone of this repository:

   ```bash
   . llms-on-crypto-mining-rigs/release/flash-next-v3/settings.env && env $PRESET_ENV llama-p106-v3.0/llama-server -m "$MODELS/$PRESET_MODEL" -md "$MODELS/$PRESET_DRAFT" $PRESET_ARGS --host 127.0.0.1 --port 8080
   ```

   with `MODELS` set to the model drive's folder.

5. **Connect from your PC** through an SSH tunnel, as for Flint ([README](../../README.md#run-it)), and keep the host
   quiet while it works: the 4 GB host has a few hundred MiB to spare.

## The preset: `settings.env`

TODO(preset): T1 writes `release/flash-next-v3/settings.env` from the release build's arms. Until then, the line
measured at 20.75 tok/s is:

```bash
# Qwen3.8-Flash-Next, Swift-1.5 pmix-q2-fast on nine P106-090, 65,536-token context, the MTP draft
PRESET_MODEL=Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf
PRESET_DRAFT=mtp-Qwen3.8-Flash-Next-draft.gguf
PRESET_ENV="GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0 GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode"
PRESET_ARGS="-ngl 99 -fa on -c 65536 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host"
```

What each part does:

| Setting | What it does |
|---|---|
| `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0` | the k-column kernels with a 32 KB shared-memory budget and without their dispatch table: the fastest setting on the rig ([2026-11-04_kshape](../../docs/data/2026-11-04_kshape/README.md)) |
| `LLAMA_HOP_CAST=q8_0`, `GGML_CUDA_HOP_STAGED=1` | the model's state crosses the eight card boundaries in Q8_0, through a pinned host ring |
| `LLAMA_DRAFT_SHORTLIST=32768,8192,2` | the draft's output head over a shortlist of tokens, with an exact fallback |
| `GGML_CUDA_KEEPALIVE=decode` | every card held at its boost clock while the split decodes |
| `-ts 6,6,6,6,5,6,5,6,3`, `-otd token_embd=CUDA4` | the layer split with the draft on the cards; the draft's token embeddings on the fifth card |
| `-md`, `--spec-type draft-mtp --spec-draft-n-max 3`, `--spec-draft-n-ubatch 128` | the MTP draft, three tokens ahead |
| `-lm dio` | direct I/O for the load: a 4 GB host cannot memory-map the 37.7 GB of shards 1 and 2 |
| `--no-host`, `--cache-ram 0`, `--ctx-checkpoints 4` | what keeps the 4 GB host from running out of memory |
| `-fit off` | skips llama.cpp's fit pass; this split was checked to fit |

Every switch: [docs/switches.md](../../docs/switches.md).

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
shards give the same three files. `pmix-q2-fast`'s shards 1 and 2: TODO(upload).

## What is not measured yet

- **One context size.** 65,536 tokens ran; 128k and 262k are memory projections only.
- **Quality on the agent's own tasks** is measured as KL divergence against Swift's 8-bit file; the full-precision
  reference is Q216, Q217 in [quants/experiments.md](../../quants/experiments.md).
- **Two rigs together** (18 cards, a better file than one rig holds): Q505.
