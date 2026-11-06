# Qwen3.8-Flash-Next on a mining rig: the first run

Qwen3.8-Flash-Next is a 125B mixture-of-experts model with about 6B active parameters per token, plus a 51B-parameter
n-gram table: 180B parameters in its original weights. On 2026-11-03 it ran for the first time on a retired mining rig
built around 6 GB cards on single PCIe 1.0 lanes, a 2-core Celeron and 4 GB of RAM. This page is the state of that work
as it stands, with stock llama.cpp as the baseline. Where it goes next is at the end. Miner Llama v3 runs it at 20.40
tokens per second for a coding agent; every file's checksum and the steps to run it:
[release/flash-next-v3](../../release/flash-next-v3/README.md).

## The headline: a 180B-parameter model on nine 6 GB mining cards and 4 GB of RAM, with stock llama.cpp

| Stock llama.cpp `d77256999`, Swift-1.5 GSQ-RCO Q2_0, 65,536-token context | |
|---|---|
| Reading a prompt of 1.4k / 7.2k / 32.7k tokens | 59 / 116 / 128 tok/s (first token after 23 s / 62 s / 257 s) |
| Writing the answer | 5.7–7.2 tok/s |
| A short question, to its first token | 7.1 s |
| Loading the model from a USB hard disk | 5.4 min |
| Memory left | 0.8–1.6 GB on each card; 230–460 MiB of the host's 4 GB |

Miner Llama v3's base, the project's Pascal engine (v2.0's code), reads prompts 15–21 % faster: 72 / 139 / 149 tok/s.
It writes at the same 5.4–8.9 tok/s, because these single requests are too noisy to separate the two. These are
single-request numbers. The project's measure is a coding agent at work ([bench/agent_check](../../bench/agent_check/README.md)),
whose run on both engines is under way. All the numbers, the commands and the scripts are in
[docs/data/2026-11-03_flash_next_baseline](../../docs/data/2026-11-03_flash_next_baseline/README.md).

## The machine: how little there is

| Part | This rig | For scale |
|---|---|---|
| GPUs | 9 × NVIDIA P106-090, 6 GB each, 54 GB in nine separate pools; Pascal (compute 6.1), no tensor cores, 75 W each | the model's file is 66.5 GB |
| GPU links | **one PCIe 1.0 lane per card** (~190 MB/s measured), through two 4-port packet switches; no GPU peer-to-peer | a desktop card gets 16 lanes of PCIe 3.0 or newer: about 60× that |
| CPU | Intel Celeron 3865U: **2 cores, 1.8 GHz, no AVX**, 15 W | too slow to compute any part of the model |
| Host memory | **4 GB DDR4**, one channel | the n-gram table alone is 28.8 GB |
| Storage | 64 GB SATA SSD; the model files on a 4.5 TB USB 3 hard disk | — |

The full hardware notes: [docs/mining_rig.md](../../docs/mining_rig.md).

## What runs

| Piece | What it is | Where |
|---|---|---|
| **The model** | Qwen3.8-Flash-Next (Qwen, 2026-08-24) in UkisAI's **Swift-1.5** fine-tune, quantized with GSQ-RCO to **Q2_0** | [ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF) (Swift Open License 1.0 and the Qwen Community License 1.0) |
| **On the cards** | the transformer, 37.7 GB, split over all nine cards (`-ts 6,5,5,6,5,5,6,5,6`) | the file's shards 1 and 2 |
| **On the SSD** | the **28.8 GB n-gram table** (IQ4_NL), memory-mapped by llama.cpp; every token reads 11–21 rows of it from disk | the file split into three shards, the table alone in the third, by [scripts/gguf_reshard.py](../../scripts/gguf_reshard.py) |
| **The engine** | stock llama.cpp, built for compute 6.1 without AVX | [scripts/build_llamacpp_docker.sh](../../scripts/build_llamacpp_docker.sh) |

How each piece was chosen:
- **It fits** with room up to 128k tokens of context, or 262k at `-ub 64`; the better IQ3_XXS file fits up to 48k
  ([fit probe](../../docs/data/2026-11-03_flash_next_fit/README.md)).
- **The table belongs on flash.** At the rig's ~2 GB page cache it costs about 1 % of a token's time on the SSD
  ([replay](../../docs/data/2026-11-03_ngram_replay/README.md)), but ~0.1 s a token on the USB hard disk
  ([disk reads](../../docs/data/2026-11-03_table_storage/README.md)).
- **The SSD was made room for:** what filled it went to the hard disk ([SSD room](../../docs/data/2026-11-03_ssd_room/README.md)).

## Run it

1. Split the table into its own shard, put that shard on the SSD, and link it beside the other two:
   `scripts/gguf_reshard.py split --isolate '^per_layer_token_embd\.' --out-dir <model drive> --isolated-dir <SSD> <the two original shards>`.
2. Start the server on all nine cards:

   ```bash
   llama-server -m <model drive>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf -ngl 99 -fa on -fit off \
     -c 65536 -b 2048 -ub 512 -ts 6,5,5,6,5,5,6,5,6 -lm dio --host 127.0.0.1
   ```

   `-lm dio` reads the weights with direct I/O. With the default memory map, llama.cpp first maps the two 37.7 GB
   shards, which a 4 GB host cannot hold, and a load was stopped after 30 GB had been read and nothing had reached the
   cards. For 128k tokens of context use `-c 131072 -ub 256 -ts 5,6,5,6,5,5,6,5,6`.

## Where a token's time goes

Profiled on Miner Llama v3's base ([decode_profile_v3.txt](../../docs/data/2026-11-03_flash_next_baseline/decode_profile_v3.txt)):
136 ms of GPU time per generated token across the nine cards, so writing is bound by the GPUs.

| Part of the model | Time per token | Share |
|---|---:|---:|
| **Hyper-connection projections, BF16** (`hc_attn_up`, `hc_ffn_up`) | ~41 ms | **30 %** |
| Routed experts, Q2_0 | ~26 ms | 19 % |
| Attention projections (Q3_K, IQ4_XS) | ~15 ms | 11 % |
| The rest: flash attention, recurrent state, router, output layer and more | ~54 ms | 40 % |

The BF16 projections are the biggest single cost. They run about 8× slower than their bytes should allow, most likely
because Pascal has no fast BF16 path. Converting them to Q8_0 (FN3) or giving them a Pascal kernel is the first
speed-up.

## Next

| Step | What it brings |
|---|---|
| The coding agent on both engines (running) | the project's own measure: tasks passed, minutes per task, tokens per second over a session |
| The BF16 projections to Q8_0, and Pascal kernels for Flash-Next's shapes | the 30 % above, then the experts |
| An MTP draft (2.7 GB) | several tokens per step, as pipelined speculation does for Flint |
| The better file: Swift IQ3_XXS (47.2 GB on the cards), and the quality of each file against the original BF16 | which file to ship |
| Two rigs together | 18 cards and 108 GB, for a better file than one rig holds |

Every step has its row in the registry, [quants/experiments.md](../../quants/experiments.md), and its research in
[flash_next_variants.md](../../docs/research/notes/flash_next_variants.md) and [strata.md](../../docs/research/notes/strata.md).
