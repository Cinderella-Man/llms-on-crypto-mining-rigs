# Edge0 145B and 120B: what they are, how they run, and whether a P106 rig can run them (2026-11-12)

[Edge0](https://github.com/Edge0-AI/Edge0) released two models on 2026-11-12, Edge0 145B and Edge0 120B, with the claim
"145B on a laptop". This note answers four questions: what the models are, how they are run, whether the files and
the engine are safe to use, and where they fit: an RTX 3090 machine and a nine-card P106 rig. Sources were read on
2026-11-12, one day after the weights were uploaded: the Edge0 repository at `b7900cb`, three of the four model
folders on Hugging Face (both Q3_K folders and the 145B's NVFP4), and upstream
[Strata](https://github.com/Niko1221/Strata) at `c62d5e3`. The engine was built, both Q3_K folders were checked file
by file, and the 145B was loaded on one RTX 3090 and answered a request.

**Edge0 is Qwen3.8-Flash-Next with narrower experts, run by a fork of Strata, and it runs on RTX 20-class cards and
newer, not on Pascal as released.** An RTX 3090 machine runs it today: the engine builds for its cards with CUDA 13,
every file checks out, and the 145B answers from one card. A P106 rig cannot: the release needs CUDA 13, one card with far more host RAM than the
rig's 4 GB, and kernels written for tensor cores. The 120B model is still the most interesting file for the rig that
has appeared since Flash-Next itself, because its experts take 28.6 GB where the experts of Flash-Next's 2-bit file
take about 34 GB. Getting it onto nine 6 GB cards means either teaching Miner Llama the new expert layout or porting
the Strata fork to Pascal, and both are projects of weeks.

*Tags: **[maker]** stated by Edge0 in its repository, model cards or release notes. **[code]** read in source.
**[file]** read from the published files or their hashes. **[project]** done or measured by this project.
**[estimate]** arithmetic or judgement, never run.*

## What the models are

Both are derived from [Qwen3.8-Flash-Next](flash_next_variants.md) and keep its skeleton: 48 layers, 512 routed
experts per layer plus one shared expert, ten routed experts per token, the per-layer n-gram embedding table and the
multi-token-prediction layer for drafting **[maker]**. What changes is the expert pool. In Flash-Next an expert reads
the full 2,560-wide hidden state. In Edge0 a shared projection **D** first compresses the state to a latent width,
the experts work at that width, and a shared projection **U** expands the result. Each layer has eight groups of D and
U, and a label per expert picks its group **[maker]**. The maker calls this a latent mixture of experts and says the
models were trained at 4 bits from the start.

| | Edge0 145B | Edge0 120B |
|---|---|---|
| Latent width | 1,792 | 1,280 |
| Expert pool | 84.6B parameters | 60.4B parameters |
| One Q3_K expert on disk | 1,630,720 B | 1,164,800 B |
| Active per token | 10.12B | 8.64B |
| N-gram table | 51.2B parameters, 8-bit float | the same file |
| Draft layer | 1.26B parameters | the same files |

The parameter counts are the maker's; the record sizes are in the files. The 145B folder's `model.json` gives 6.6B
active parameters where the model card gives 10.12B **[file]**.

Each model comes in two expert formats, as separate folders: **Q3_K** (Q3_K gate and up, IQ4_NL down, with column
scales) and **NVFP4**. The D and U projections come as Q8 and Q4 in every folder, and the launcher picks Q8 for the
145B and Q4 for the 120B.

### The maker's numbers

Quality, as published by Edge0 for the models, not per format **[maker]**:

| Model | DeepSWE 1.1 | Terminal-Bench 2.1 | AA-LCR | IFBench | GPQA-Diamond | HLE |
|---|---:|---:|---:|---:|---:|---:|
| Edge0 145B | 64.6 | 86.5 | 79.3 | 80.6 | 90.4 | 35.4 |
| Edge0 120B | 51.3 | 75.3 | 81.0 | 79.3 | 87.4 | 31.0 |
| Qwen3.8-Flash-Next | 58.7 | 83.3 | 79.1 | 81.3 | 91.7 | 35.9 |

Speed, Q3_K, in tokens per second **[maker]**:

| Machine | Model | 4K prefill | 16K prefill | Chat decode | Agent decode |
|---|---|---:|---:|---:|---:|
| RTX 5090 D 24 GB | 145B | 2,169.9 | 3,554.9 | 99.7 | 121.2 |
| RTX 4090 24 GB | 145B | 2,042.0 | 3,179.4 | 69.9 | 91.7 |
| RTX 5090 D held to 16 GiB VRAM + 20 GiB RAM | 145B | 249 | 335 | 36.8 | 44.9 |
| RTX 5090 D 24 GB | 120B | 3,236.7 | 4,576.5 | 130.2 | 144.6 |
| RTX 4090 24 GB | 120B | 2,637.5 | 3,551.1 | 95.0 | 104.9 |
| RTX 5090 D held to 16 GiB VRAM + 20 GiB RAM | 120B | 430 | 778 | 62.4 | 73.3 |

On a 24 GB card the maker's engine holds the 145B's Q3_K weights as 23.14 GiB on the card and 31.28 GiB in host RAM,
and the 120B's as 22.93 GiB and 18.40 GiB **[maker]**. No third-party measurement exists yet. For scale, upstream
Strata ran Flash-Next itself at about 78 tok/s on an RTX 3090 with drafting off and about 119 as shipped
([strata.md](strata.md)), so the headline is not raw speed. The claim is Flash-Next's class of quality from files that
fit a 24 GB card with 31 GiB of RAM, or a 16 GiB card with 20 GiB at about a third of the speed.

## How they run

There are no safetensors files and no single GGUF because the weights ship in the engine's own layout, a Strata
"pack". The engine keeps a few thousand experts on the card and streams the rest from host memory, so it wants every
expert as a fixed-size record it can fetch by offset **[maker]**.

| File (145B Q3_K) | Bytes | What it is |
|---|---:|---|
| `experts.bin` | 40,076,574,720 | 48 × 512 expert records of 1,630,720 B, nothing else |
| `latent_dense_q8.bin`, `latent_dense_q4.bin` | 3,743,416,320 and 1,981,808,640 | the D and U projections, Q8 and Q4 |
| `model-skeleton.gguf` | 5,991,569,248 | attention, routing, shared experts, embeddings, output |
| `ple-fp8.gguf` | 51,200,246,112 | the n-gram embedding table |
| `dense.bin`, `index.txt`, `native_experts.txt`, `labels.bin`, `scales/` | 1,538,625,024 and small | the rest of the pack: an index, per-expert scales, group labels |
| `mtp/` | 824,314,919 | the draft layer |
| `tokenizer/`, `profile.bin`, `model.json`, `start.sh` | small | tokenizer, the ranking of experts to keep on the card, a launch description |

The folder is 105.37 GB. The other folders share most of it byte for byte: the skeleton, the table, `dense.bin`, the
draft layer and the tokenizer have the same SHA-256 in all folders read **[file]**. A second folder therefore costs
only its experts and projections: 32.7 GB for the 120B Q3_K and 47.6 GB for the 145B NVFP4, with hard links for the
rest ([fetch_edge0.sh](../../data/2026-11-12_edge0_setup/run/fetch_edge0.sh), `LINK_FROM`).

Two programs run it **[code]**:

- **edge0-strata**, the engine: a fork of Strata taken at upstream commit `c62d5e3` (2026-10-31, the commit
  [strata.md](strata.md) describes), with the latent experts added. It is a C++ and CUDA program plus Strata's Python
  server, MIT-licensed.
- **edge0-infer**, the launcher: `edge0-infer serve --model <folder>` starts the engine on a loopback port, waits until
  the model is loaded and forwards OpenAI- and Anthropic-style requests to it. It needs no GPU of its own.

The documented requirements are Linux or Windows, one NVIDIA card of compute capability 7.5 or newer and CUDA 13
**[maker]**. The engine's own notes say that identical requests at temperature 0 can give different answers. The
models carry the Edge0 Community Model License 1.0 on top of the Qwen Community License 1.0: personal, research and
internal use and non-commercial sharing are allowed, selling or hosting the model is not. Read both before publishing
a converted file.

## Is it safe to run

The review covered the files, the launch path and the engine's code. It found nothing that runs code from the model
folder and nothing that contacts another machine on its own.

- **The model folder is data.** It holds no pickle, PyTorch or Python file. `experts.bin` is exactly 24,576 records
  of 1,630,720 bytes, the scale files are 1,792 floats each, the indexes are text, and the two GGUF files are read by
  ggml **[file]**. The chat template is rendered in Jinja's immutable sandbox **[code]**.
- **`start.sh` and `model.json` in the folder are not needed and were not used.** `start.sh` hands the folder's
  `model.json` to the engine's server as its configuration, and that server can run a command named in a
  configuration (`before_load`). The published `model.json` names none. The launcher reads only `model_name` from it
  and takes the engine's command line from a template in the source tree
  (`edge0-strata/edge0/models/edge0-145b.json`) **[code]**. Use the launcher, not `start.sh`.
- **The engine is upstream Strata plus arithmetic.** Against Strata `c62d5e3`, the fork changes 43 files and adds 54,
  all C++, CUDA and build files apart from one line of `setup.py` and an optional log in the benchmark tool. Every
  file of the Python server is identical to upstream's. No added line starts a process, opens a socket or names a URL
  **[code]**.
- **What the server reaches by itself is nothing.** It listens on 127.0.0.1 unless told otherwise. It fetches a URL
  only when a request carries an image URL, and talks to tool servers only when a configuration names them
  **[code]**.
- **The build's inputs are pinned.** ggml comes from llama.cpp `3cf0325`, a commit on upstream's main branch (tag
  `b11062`) that upstream Strata pins too, and the Python packages are pinned by version **[code]**.
- **Use the official sources.** Copies of the repository exist under other accounts; the review covers
  `Edge0-AI/Edge0` at `b7900cb` and the Hugging Face organization `Edge0` only. Every large file was checked against
  the SHA-256 the repository lists.

Two limits remain. About a megabyte of new C++ and CUDA was searched for process, network and file calls, not read
line by line, so run the engine in a container, as here. And nothing can show from a file what a model was trained to
write: its output deserves the caution any model's does, most of all when it drives tools.

## An RTX 3090 machine can run it

An RTX 3090 is compute capability 8.6 with 24 GB, the class the maker measured on. The engine was built for it from
`b7900cb` with CUDA 13.4, in a container, with
[build_edge0.sh](../../data/2026-11-12_edge0_setup/run/build_edge0.sh) **[project]**. The method, the file list with
hashes and the scripts are in [the data folder](../../data/2026-11-12_edge0_setup/README.md).

The 145B at Q3_K loaded on one card and answered **[project]**: 23.6 GB on the card, and one request of 532 tokens
at 34.6 tok/s by the engine's own timing, with 334 of 398 drafted tokens accepted. That is a single cold request on
a busy machine, a check that it works. The maker's RTX 4090 row gives 69.9 tok/s for chat.

Three things will set its speed below the maker's RTX 4090 row, by an amount only a run can give **[estimate]**:

- A 3090 is the slower card.
- The engine sends the experts that are not on the card over PCIe or computes them on the CPU, and a card on a
  narrow or old link pays more for each one than the maker's did. Pick the card with the widest link.
- With `ple-io=direct` the engine reads the n-gram table from disk for every token, which needs an SSD. On a hard
  disk use `ple-io=ram`, which loads all 51 GB into memory once.

The release is for one card. Whether its latent path also works with Strata's split of layers across cards, which
would put all 37 GiB of the 145B's experts in VRAM on two 24 GB cards, is untested by the maker (Q1440).

## A P106 rig cannot run it as released

| Requirement of the release | The rig | Consequence |
|---|---|---|
| CUDA 13 **[maker]** | Pascal needs CUDA 12.x; CUDA 13 cannot compile for it | a build with Strata's community switch `STRATA_EXPERIMENTAL_SM60` and CUDA 12.9, which Edge0 does not document **[code]** |
| Tensor-core kernels for the latent Q3_K prompt path (`mma` instructions, compute 8.0, with a Turing variant) **[code]** | 5 SMs, no tensor cores | the unfused path (`STRATA_LATENT_FUSED=0`) or new kernels; the decode kernel uses `dp4a`, which Pascal has |
| One card, with 18–31 GiB of experts in host RAM **[maker]** | nine 6 GB cards, 4 GB of RAM, no AVX | every expert must sit in VRAM, split over nine cards, a path the maker has not run |
| Experts sent over PCIe at speed | PCIe 1.0 x1 | the same conclusion: nothing streamed |

With every weight on the cards, the sums are **[estimate]**:

| GiB | Edge0 145B Q3_K | Edge0 120B Q3_K |
|---|---:|---:|
| Experts | 37.32 | 26.66 |
| D and U projections (Q8 for the 145B, Q4 for the 120B) | 3.49 | 1.32 |
| Skeleton, pack remainder, draft layer | 7.78 | 7.78 |
| Sum | 48.59 | 35.76 |
| Nine P106 cards hold 53.3 GiB before caches and working memory | does not fit with a reserve on each card | fits, with about 17 GiB for the cache, buffers and reserves |

So the 120B is the candidate. Flash-Next's own file for the rig, Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, is 37.8 GB
and needs seven of the nine cards at a 64k context ([fit probe](../../data/2026-11-03_flash_next_fit/README.md)). An
Edge0 120B at Q3_K would hold experts at about 3.8 bits per weight in less memory than Flash-Next's at 2 bits, if the
maker's quality table holds for it.

Two routes lead there, and neither is short:

1. **Teach Miner Llama the latent experts** (Q1441). The skeleton is already a `qwen4exp` GGUF. What is new is a
   converter from the pack to GGUF tensors, and the graph for one layer: D per group, the routed experts at the
   latent width, U back to 2,560. Every kernel the engine already has for Q3_K and IQ4_NL products applies. This keeps
   one engine for every model and every card count.
2. **Port edge0-strata to compute 6.1** (Q1442). This is the Strata fork question of [strata.md](strata.md) again,
   with the latent kernels added to its list, and it inherits that note's verdict: an estimated 12–27 engineer-weeks
   for one model family on an engine that has never run with nine cards or 4 GB of RAM.

One check comes first because it is cheap: whether `ple-fp8.gguf` is Flash-Next's own n-gram table (Q1443). If it is,
a rig that holds Flash-Next already holds 51 of the 120B's 92 GB.

## Registry

Rows Q1438–Q1444 of the [registry](../../../quants/experiments.md#rigs-and-other-models): the set-up and first load
(Q1438), the agent check for quality (Q1439), experts across cards (Q1440), the latent layout in Miner Llama (Q1441),
the Pascal port of the fork (Q1442, parked with Q466), the table's identity (Q1443) and the NVFP4 folders (Q1444,
parked).
