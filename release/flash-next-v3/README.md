# Qwen3.8-Flash-Next (v3, in preparation): a 180B-parameter model on a mining rig, the stock llama.cpp baseline

Not released. Flash-Next is the model Miner Llama v3 works towards; this folder is where its release takes shape. It
holds the state on 2026-11-03: the model running on all nine cards of a retired mining rig with stock llama.cpp, what
was measured, every file with its checksum, how to run it yourself, and what is missing before it ships. The model's
own page is [models/qwen3.8-flash-next](../../models/qwen3.8-flash-next/README.md).

## The headline: 180B parameters on nine 6 GB mining cards, a 2-core Celeron and 4 GB of RAM

Qwen3.8-Flash-Next is a 125B mixture-of-experts model with about 6B active parameters per token, plus a 51B-parameter
n-gram table: 180B parameters in its original weights. In UkisAI's Swift-1.5 fine-tune at Q2_0 it runs on a rig
built for mining: the 37.7 GB transformer split over nine 6 GB cards on single PCIe 1.0 lanes, the 28.8 GB n-gram table
read from the rig's SSD.

| Swift-1.5 GSQ-RCO Q2_0, nine P106-090 cards, 65,536-token context, one request | Stock llama.cpp `d77256999` | Miner Llama v3's base `37e717ff6` |
|---|---|---|
| Reading a 1.4k / 7.2k / 32.7k-token prompt | 59 / 116 / 128 tok/s | 72 / 139 / 149 tok/s |
| Time to the first token for those prompts | 23 / 62 / 257 s | 20 / 52 / 221 s |
| Writing the answer | 5.7–7.2 tok/s | 5.4–8.9 tok/s |
| A short question, to its first token | 7.1 s | 6.5 s |
| Loading from a USB 3 hard disk | 322 s | 326 s |
| Memory left after the load | 0.8–1.6 GB on each card | the same |
| Host memory available while it works | 230–410 MiB of 4 GB | 330–460 MiB |

Miner Llama v3's base is the v2.0 engine's code. It reads prompts 15–21 % faster. Writing speed is the same on both
within the noise of single requests. These are single requests, a diagnostic in this project's terms. A release is
measured on a coding agent at work ([bench/agent_check](../../bench/agent_check/README.md)), and that run on both
engines is under way. Every number with its command:
[docs/data/2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md).

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

## What runs

| Piece | What it is | Where it comes from |
|---|---|---|
| **The model** | Qwen3.8-Flash-Next (Qwen, 2026-08-24) in UkisAI's **Swift-1.5** fine-tune, quantized with GSQ-RCO to **Q2_0**: two shards, 66.5 GB | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), revision `99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d` |
| **On the cards** | the transformer, 37.7 GB, over all nine cards (`-ts 6,5,5,6,5,5,6,5,6`) | shards 1 and 2 of the three-shard split |
| **On the SSD** | the **n-gram table** (`per_layer_token_embd`, IQ4_NL, 28.8 GB), memory-mapped; each generated token reads 11–21 rows of it from the SSD | shard 3, written by [scripts/gguf_reshard.py](../../scripts/gguf_reshard.py) with the bytes unchanged |
| **The engine** | stock llama.cpp `d77256999`, or Miner Llama v2.0 (`37e717ff6`, v3's base); both built for compute 6.1 without AVX, both know Flash-Next's `qwen4exp` architecture and Q2_0 | [scripts/build_llamacpp_docker.sh](../../scripts/build_llamacpp_docker.sh); v2.0's [GitHub release](https://github.com/Cinderella-Man/llama.cpp/releases/tag/miner-llama-v2.0) |

The model is under the Swift Open License 1.0 (free below US$1M of yearly revenue) and the Qwen Community License 1.0.
This project ships no weights: you download them from UkisAI's repository and split them yourself.

## What was measured

| Question | Answer | Data |
|---|---|---|
| Does it fit nine 6 GB cards? (Q213) | Q2_0 fits with room up to 128k tokens of context, or 262k at `-ub 64`; the better IQ3_XXS file up to 48k | [2026-11-03_flash_next_fit](../../docs/data/2026-11-03_flash_next_fit/README.md) |
| Where can the n-gram table live? (Q214) | on the SSD: 6,039 random 4 KiB reads per second against 95 on the USB hard disk | [2026-11-03_table_storage](../../docs/data/2026-11-03_table_storage/README.md) |
| What do the table's reads cost? (Q228) | about 1 % of a token's time on the SSD at the rig's ~2 GB page cache, ~0.1 s a token on the hard disk; from 3,210 replayed agent requests | [2026-11-03_ngram_replay](../../docs/data/2026-11-03_ngram_replay/README.md) |
| Can the SSD hold it? (Q435) | yes, once what filled it moved to the hard disk (6.2 → 35.8 GB free); Q2_0 and IQ3_XXS share one byte-identical table | [2026-11-03_ssd_room](../../docs/data/2026-11-03_ssd_room/README.md) |
| How fast is it? (Q215) | the headline above; the agent check is running | [2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md) |
| Does the desktop cost much? (Q83) | it holds ~280 MB of host RAM and 5 MiB a card; it stays, for the memory-clock route | [2026-11-03_headless](../../docs/data/2026-11-03_headless/README.md) |

## Where a token's time goes

`GGML_CUDA_PROFILE` on v3's base ([decode_profile_v3.txt](../../docs/data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)):
136 ms of GPU time per generated token across the nine cards, so writing is bound by the GPUs, not by the links, the
host or the SSD.

| Part of the model | Time per token | Share |
|---|---:|---:|
| **Hyper-connection projections in BF16** (`hc_attn_up`, `hc_ffn_up`) | ~41 ms | **30 %** |
| Routed experts, Q2_0 | ~26 ms | 19 % |
| Attention projections (Q3_K, IQ4_XS) | ~15 ms | 11 % |
| The rest: flash attention, recurrent state, router, output layer and more | ~54 ms | 40 % |

The BF16 projections run about 8× slower than their bytes should allow, most likely because Pascal has no fast BF16
path. They are v3's first target.

Two more limits came out of the runs:
- **Short prompts wait 6.5–7.1 s for the first token.** Every request first restores a 112.6 MiB recurrent-state
  checkpoint (0.63 s), and each micro-batch passes all nine cards in turn over x1 links.
- **A prompt that changes far back is read again in full.** Checkpoints are saved after the chat template's prefix and
  in the last ~512 tokens of each prompt, so a 31k-token follow-up that changed its last 1,500 tokens took 209 s (v3)
  and 240 s (stock) to its first token.

## Run it yourself

You need the rig, about 30 GB free on its internal SSD, a model drive with 67 GB free, and a PC with Python 3 and room
for a second copy of the model for the split. Another card count needs its own `-ts`, checked with
[scripts/fit_probe.sh](../../scripts/fit_probe.sh) before the first load.

1. **Download** the two Q2_0 shards (66.5 GB) at the revision measured here:

   ```bash
   hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --include '*Q2_0-0000*-of-00002.gguf' --local-dir swift
   ```

   Their SHA-256 are in [CHECKSUMS](CHECKSUMS).

2. **Move the table into a shard of its own** (copies every byte, ~8 minutes on an NVMe drive), then check that the
   new files hold the same bytes as the downloaded ones:

   ```bash
   python3 -I scripts/gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir swift_q2_0 swift/*Q2_0-0000*-of-00002.gguf
   ```

   ```bash
   python3 -I scripts/gguf_reshard.py verify swift/*Q2_0-0000*-of-00002.gguf --new swift_q2_0/*-of-00003.gguf
   ```

   `verify` prints the downloaded shards' SHA-256, rebuilt from the new files; they match [CHECKSUMS](CHECKSUMS), as do
   the three new files.

3. **Copy shards 1 and 2 to the model drive and shard 3 to the SSD**, and link shard 3 beside the other two, where
   llama.cpp looks for it:

   ```bash
   ln -s <SSD>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf <model drive>/
   ```

4. **Start the server** on all nine cards:

   ```bash
   llama-server -m <model drive>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf -ngl 99 -fa on -fit off -c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -lm dio --host 127.0.0.1 --port 8080
   ```

   `-lm dio` reads the weights with direct I/O. With the default memory map, llama.cpp first maps the two 37.7 GB
   shards, which a 4 GB host cannot hold: a load was stopped after 30 GB had been read and nothing had reached the
   cards. `-fit off` skips llama.cpp's memory-fit pass; this split was checked to fit (Q213). For 128k tokens of
   context: `-c 131072 -ub 256 -ts 5,6,5,6,5,5,6,5,6`.

5. **Connect from your PC** through an SSH tunnel, as for Flint ([README](../../README.md#run-it)), and keep the host
   quiet while it works: the 4 GB host has a few hundred MiB to spare.

## Files and checksums

| File | Size | Made by | Checksum |
|---|---:|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf`, `…-00002-of-00002.gguf` (the download) | 39.8 + 26.8 GB | UkisAI | SHA-256 `79e2a387…`, `ef3bb04f…` in [CHECKSUMS](CHECKSUMS) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf` (what runs) | 11.0 + 26.8 + 28.8 GB | `gguf_reshard.py split` | in [CHECKSUMS](CHECKSUMS) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-0000{1,2}-of-00002.gguf` (the better file, not yet run) | 39.8 + 36.2 GB | UkisAI | `3bddaa66…`, `b0b15f78…` in [CHECKSUMS](CHECKSUMS); its shard 3 is the Q2_0 table, byte for byte |
| `mtp-Qwen3.8-Flash-Next-draft.gguf` (MTP draft, not yet run) | 2.7 GB | [agentionai/Qwen3.8-Flash-Next-Gyro-GGUF](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) `273edc1d2ffae5241fe3cab69568d682dad4c718` | `00818378…` in [CHECKSUMS](CHECKSUMS) |
| Miner Llama v2.0 engine archives | 253 / 554 MB | `scripts/package_release.sh` on `37e717ff6` | in [release/flint-v2/CHECKSUMS](../flint-v2/CHECKSUMS) and the release's `SHA256SUMS` |

[CHECKSUMS](CHECKSUMS) is in `sha256sum -c` format; in the folder that holds your files,
`sha256sum -c --ignore-missing CHECKSUMS` checks the ones you have. The split is deterministic: the same downloaded
shards give the same three files.

## Gaps

- **No agentic number yet.** The coding agent on both engines is the measurement that decides the release settings;
  it is running.
- **No quality number yet.** How far Q2_0 and IQ3_XXS are from the original weights is measured against a Q8_0
  reference built from the BF16 model (Q216, Q217), which is still downloading.
- **One context size measured.** 65,536 tokens ran; 128k and 262k are memory projections only.
- **Q2_0 only.** IQ3_XXS (47.2 GB on the cards, up to 48k tokens) and the MTP draft (one more layer, 2.7 GB) are
  checked to fit but not run.

## Before it ships with v3

| Step | What it brings | Registry |
|---|---|---|
| The agent check on both engines (running) | tasks passed, minutes per task and tokens per second over a session: the release's own numbers | Q215 |
| The BF16 projections to Q8_0, and Pascal kernels for Flash-Next's shapes | the 30 % above, then the experts | Q218, Q223, Q225 |
| The MTP draft | several tokens per step, as pipelined speculation does for Flint | Q226 |
| The quality of Q2_0 and IQ3_XXS against the original, then the agent check of the best two files | which file ships | Q216, Q217, Q227 |
| Checkpoints kept further back in a prompt | follow-ups that change older context, as a coding agent's do, without reading the whole prompt again | Q440, Q448 |
| Two rigs together | 18 cards and 108 GB, for a better file than one rig holds | Q505 |

Every step has its row in [quants/experiments.md](../../quants/experiments.md); the research behind them is in
[flash_next_variants.md](../../docs/research/notes/flash_next_variants.md) and
[strata.md](../../docs/research/notes/strata.md).
