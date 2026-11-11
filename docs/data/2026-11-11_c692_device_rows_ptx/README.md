# C692 (Q1225): upstream #31927's device rows with the fork's draft switches, on the whole model through the compute 6.1 PTX (2026-11-11)

Flash-Next's MTP draft reads the target's hidden rows: after each target pass (a prompt call or a speculative verify
step) the draft's catch-up takes one row per token, 10,240 F32 values each. In the v3.0 release build (`1551f5fe1`) the
target copies those rows to host memory and the draft uploads them again. Upstream llama.cpp #31927, merged into Miner
Llama's v4 line and on by default (`LLAMA_MTP_H_DEV=1`, [switches](../../switches.md)), keeps them in a buffer on the
target's output card, where the draft's graph reads them. A CPU build has no output card and takes the host path, so
none of the v4 line's CPU checks ran the device path. This run checks it on cards, on the whole model, with the fork's
draft switches: Swift Qwen3.8-Flash-Next Pascal Q2 GGUF's file as the v3 preset loads it (Swift Q2_0, its n-gram table as
a third shard) with its MTP draft, on three RTX 3090s, the v4 day build (`pascal-p106-v4-b12246-day` `18d0fa309`)
against v3.0. Both rig builds hold sm_61 machine code and compute_61 PTX only, so every kernel they ran on these cards
is the PTX a P106 gets, JIT-compiled by the cards' driver. Correctness only: no time was read. The registry row is Q1225
in [quants/experiments.md](../../../quants/experiments.md); on a P106 rig the buffer's load and the greedy identity were
R57's ([data](../2026-11-11_r57_v4night_p106/README.md), Q1224), and the device path a gain over the host path in R92
([data](../2026-11-11_r92_upstream_defaults/README.md)).

**Result: the device rows draft exactly what the host rows draft, with each of the fork's draft switches.** Seven loads
(the preset with its head shortlist, then with the embedding row cache, then with the hidden-row carry, each on both
builds, and the v4 build with the device rows turned off) wrote the same greedy replies and the same speculative steps,
byte for byte: in each load the same 1,101 verify steps with the same drafted tokens, the same eight candidates per
drafted position with the same probabilities, the same tokens the target kept and the same eight most probable target
tokens with the same probabilities (to the dump's 1e-5). The seven `LLAMA_SPEC_DUMP` files are one file once each
server's start stamp is removed (md5 `48b3f959b56692877d75b2c5c8b64dc9`). The v4 build put the buffer on the target's
output card, `CUDA2 h_nextn buffer size = 80.04 MiB`, and that card held 82 MiB more after loading than with v3.0
(15,080 against 14,998 MiB), as on a P106's card 8 in R57 (4,647 against 4,565 MiB).

## The loads

The order they ran in, on the same three cards, 17:34–17:52 ([loads_cards.tsv](loads_cards.tsv),
[loads.tsv](loads.tsv) has each load's whole line):

| Load | Build | Added to the preset | Where the draft read the target's rows | Used memory after loading, CUDA0 / CUDA1 / CUDA2 (MiB) | Draft graph reallocations |
|---|---|---|---|---:|---:|
| `rel-pre` | v3.0 `1551f5fe1` | nothing | host memory | 13,938 / 13,810 / 14,998 | 2,257 |
| `day-pre` | v4 day `18d0fa309` | nothing | `CUDA2 h_nextn buffer size = 80.04 MiB`; "target h rows on the device: yes" | 13,938 / 13,810 / 15,080 | 66 |
| `day-hdev0` | v4 day | `LLAMA_MTP_H_DEV=0` | host memory: "LLAMA_MTP_H_DEV=0, extracting the h_nextn rows to host memory"; "target h rows on the device: no" | 13,938 / 13,810 / 14,998 | 2,257 |
| `rel-cache` | v3.0 | `LLAMA_DRAFT_EMBD_CACHE=8192` | host memory | 13,938 / 13,810 / 15,020 | 2,252 |
| `day-cache` | v4 day | `LLAMA_DRAFT_EMBD_CACHE=8192` | the 80.04 MiB buffer on `CUDA2` | 13,938 / 13,810 / 15,102 | 60 |
| `rel-carry` | v3.0 | `LLAMA_DRAFT_H_CARRY=1` | host memory | 13,938 / 13,810 / 14,998 | 58 |
| `day-carry` | v4 day | `LLAMA_DRAFT_H_CARRY=1` | the 80.04 MiB buffer on `CUDA2` | 13,938 / 13,810 / 15,080 | 58 |

Every load is the v3 preset's line ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env):
the k-column settings, the Q8_0 hop cast, the staged hop, `LLAMA_DRAFT_SHORTLIST=32768,8192,2`, 65,536 tokens of
context, `-b 2048 -ub 512`, the MTP draft at three tokens with its own micro-batch of 128) with three changes for three
cards: `-ts 1,1,1` for the nine-card split, `-otd token_embd=CUDA1` for `token_embd=CUDA4` (the draft's token table on a
card between the first and the output card, as card 4 of nine), and no `GGML_CUDA_KEEPALIVE` (no clock to hold). Each
switch acted, with the same counts on both builds ([load_lines.tsv](load_lines.tsv)):

- the shortlist: "32801 ids (below 32768, 33 special, ...)" at the start; "949 drafts with it, 51 with the full head,
  26 target tokens outside it; 32913 ids in it" at the end of every load;
- the row cache: "8192 rows of q8_0 (21.25 MiB) in CUDA2, for batches of up to 16 tokens; the table is in CUDA1", and
  at the end "7024 token rows read from it, 490 of them copied from the table first; 56 batches read the table";
- the carry: "draft hidden rows kept on the device: 1 rows of 10240 F32 (0.04 MiB) in CUDA2" and "a step's drafting
  passes keep them on the device".

No load logged a warning or an error.

## The replies and the steps

Each load sent the same seven requests in the same order ([run/c692_client.py](run/c692_client.py)): a warm-up of 32
tokens, then three prompts twice each, 512 tokens greedy (temperature 0, top-k 1) with the end of text ignored and
without the prompt cache. The prompts, each one user message in the model's chat template, rendered once by the first
load and sent as the same text by every load ([prompts_rendered.json](prompts_rendered.json)): R57's coding request (79
tokens), R57's prose request (71 tokens), and a review of a Python script of this repository (3,298 tokens: two calls of
the 2,048-token batch, so the draft's catch-up over the prompt reads rows written by several micro-batches and two
calls). Every request in every load ([replies.tsv](replies.tsv)):

| Request | Prompt tokens | Reply ids (md5, 512 tokens; the warm-up 32) | Drafted | Accepted | Verify steps |
|---|---:|---|---:|---:|---:|
| warm-up | 8 | `57c0b902dbf643a1a8d3919d61940645` | 33 | 19 | 11 |
| code, first | 79 | `e33ea82c26d52f727568591a6adf3b77` | 497 | 345 | 166 |
| code, second | 79 | `e33ea82c26d52f727568591a6adf3b77` | 487 | 348 | 163 |
| prose, first | 71 | `3536b2785920ba68b436e3b1581d5541` | 573 | 320 | 191 |
| prose, second | 71 | `3536b2785920ba68b436e3b1581d5541` | 561 | 324 | 187 |
| review, first | 3,298 | `2bce9407dc795330ff52e8c22e698172` | 578 | 318 | 193 |
| review, second | 3,298 | `2bce9407dc795330ff52e8c22e698172` | 569 | 321 | 190 |

Every row is the same in all seven loads. [pairs_summary.tsv](pairs_summary.tsv) holds the 21 pairs of loads: 7 of 7
requests with the same ids and 7 of 7 with the same steps in each, and 0.00000 as the largest difference of the
drafter's and of the target's listed probabilities; [pairs.tsv](pairs.tsv) has each pair per request, and each load's
second request against its first.

**Within a load, a prompt's second request is not its first.** It writes the same reply but drafts otherwise (code: 487
drafted against 497); the first difference is the order of two candidates whose probability rounds to 0 (code, step 0:
ids 14235 and 1902). The draft head's shortlist takes in each request's tokens (32,801 ids at the start of a load, 32,913
at its end), so a later request drafts over a longer list. All seven loads repeat this sequence exactly, so the pairs
above compare request i of one load with request i of another.

## Where the model ran

- **The cards:** llama-server lists `CUDA0`, `CUDA1` and `CUDA2` as the three RTX 3090s (23,861, 23,826 and 23,861 MiB
  free). The target: `offloaded 49/49 layers to GPU`, model buffers of 12,484.53, 11,755.51 and 11,490.00 MiB on
  `CUDA0`, `CUDA1`, `CUDA2`; on the host a 260.50 MiB `CPU` buffer and the 27,465.95 MiB `CPU_Mapped` buffer of the
  n-gram table's shard; its KV, recurrent-state and compute buffers on the three cards. The draft: 644.14 MiB on `CUDA1`
  (its token table, `-otd`) and 1,917.90 MiB on `CUDA2` (its layer and head), its KV and compute buffers on `CUDA2`
  (compute 57.46 MiB, 1.25 MiB on `CUDA1`).
- **The PTX:** both servers hold only sm_61 code and its PTX (`cuobjdump`, [archs.txt](archs.txt)): v3.0's 187 sm_61
  cubins and 187 compute_61 PTX, the v4 day build's 192 and 192. Each build's JIT cache started empty; after its first
  load it held 49 files (181,231,828 B, v3.0) and 50 files (185,488,881 B, v4), one more after a later arm on each (50
  and 51 files). The cards ran the builds' PTX.
- **Which of the fork's kernels:** the PTX-mode settings (below) force the fork's matrix-vector kernels on (`=2`) with a
  P106-090's 5 SMs and shared memory: the one-token and multi-token Pascal kernels, the k-column kernels and the BF16
  kernel. The Pascal attention kernels (`GGML_CUDA_FATTN_TILE_PASCAL`, `GGML_CUDA_ATTN_BLOCK`) stayed at their
  defaults, which ask for compute 6.1, so attention ran on llama.cpp's own kernels (Q1178 lists the switches a newer card
  needs).

## Beside the result

- **Draft graph reallocations** (the scheduler's `graph reallocation` lines at `-lv 4`, one per reallocation; in
  `rel-pre`, `day-pre` and `rel-carry` every one is a draft graph of 206, 207, 216 or 220 nodes in 2 splits): 2,257 in a load with the rows in host memory, 66 with the device rows,
  58 with the carry on either build (60 with the row cache and the device rows), over the same 1,101 steps. This counts work and does
  not time it; which input makes the host path reallocate was not traced. On a P106 rig the device rows were a gain
  (R92), the carry negligible (Q961).
- On rig 2, v3.0 with the carry drafted one token more than the preset in a greedy reply (Q1210, 169 of 258 accepted
  against 170 of 255, the same text). Here, on these cards, the carry's drafts are the preset's on both builds.

## Limits

- Three cards, not nine: two card boundaries instead of eight, so fewer hops (each through the staged host ring with
  the Q8_0 cast, as on the rig).
- One slot, greedy, prompts of 71 to 3,298 tokens and replies of 512 tokens. Deeper contexts, sampled requests and
  probabilistic drafting were not run.
- The fork's attention kernels ran as llama.cpp's (above); the matrix-vector kernels as on a P106.

## Commands

On a build machine with Docker and three free cards, from a checkout of this repository (`REPO`), with the two rig
tarballs unpacked:

```bash
# the cards' CUDA, in the job's image (scripts on the build machine: cuda_preflight.sh)
IMAGE=nvidia/cuda:12.9.2-devel-ubuntu24.04 PREFLIGHT_DOCKER_ARGS="--label pascal-quants=1 --cpu-shares 256" \
  cuda_preflight.sh -- --gpus '"device=<card>,<card>,<card>"'
WORK=<work folder> REPO=<checkout> MODEL=<Swift Q2_0 shard 1 of 3> MODEL_DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> \
  BIN_day=<v4 day tree>/bin/llama-server BIN_rel=<v3.0 tree>/bin/llama-server GPUS=<card>,<card>,<card> \
  MOUNTS="<every folder those paths lie on>" \
  setsid -f nohup nice -n 19 bash run/run_c692.sh > <work folder>/console.txt 2>&1 < /dev/null
```

The driver ([run/run_c692.sh](run/run_c692.sh)) lists each build's GPU code with `cuobjdump`, then runs the loads
(default `LOADS="rel:pre day:pre day:hdev0 rel:cache day:cache rel:carry day:carry"`), each one llama-server in a
container (`--gpus`, `--cpu-shares 256`, label `pascal-quants=1`) with:

- the preset's line as above, plus `-lv 4` (the load lines that name each buffer's card and the h_nextn buffer);
- the PTX-mode settings `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
  GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
  GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` and `CUDA_FORCE_PTX_JIT=0`: the builds have no code for these cards, so the
  driver JIT-compiles their PTX anyway, and `=1` would force cuBLAS onto PTX that cuBLAS does not ship for them (Q876);
- the build's own JIT cache (`CUDA_CACHE_PATH`, emptied before the first load) and
  `LLAMA_SPEC_DUMP=<work>/dump_<load>.jsonl`.

[run/c692_client.py](run/c692_client.py) sends each load's requests on the loopback; at the end
[run/c692_compare.py](run/c692_compare.py) writes `replies.tsv`, `pairs.tsv`, `pairs_summary.tsv` and `load_lines.tsv`
(paths in them replaced by `$WORK`, `$BIN_day`, `$BIN_rel`, `$MODEL`, `$MODEL_DRAFT`). The dumps' identity:
`sed -E 's/"run":[0-9]+,//' dump_<load>.jsonl | md5sum` for each load.

## Files

| File | What it holds |
|---|---|
| [loads_cards.tsv](loads_cards.tsv) | per load: build, arm, used memory of each card after loading, the JIT cache's files and bytes after the load, the draft graph reallocations, the dump's steps and its md5 without the start stamp |
| [loads.tsv](loads.tsv) | per load: the variables and the llama-server line |
| [replies.tsv](replies.tsv) | per load and request: the prompt's tokens and their md5, the reply's ids (md5), the draft counters, the verify steps (none rejects by sampling: the requests are greedy) |
| [pairs_summary.tsv](pairs_summary.tsv), [pairs.tsv](pairs.tsv) | every two loads, in all and per request; each load's repeats against its first request |
| [load_lines.tsv](load_lines.tsv) | each load's lines on the devices and buffers, the h_nextn buffer, the shortlist, the row cache and the carry |
| [archs.txt](archs.txt) | each build's llama-server: md5, its machine code and PTX by architecture |
| [prompts_rendered.json](prompts_rendered.json) | the three prompts as sent |
| [run/](run/) | the driver, the client, the tables |

## Provenance and licence

The builds are Miner Llama's rig tarballs `llama.cpp-20261111-18d0fa309-pascal-p106-v4-b12246-day` (md5
`ad766d3f8198be0c77f9f41ebf9c69d5`) and the v3.0 release `llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5`
(MIT, as llama.cpp). The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as the v3
preset uses them ([release/flash-next-v3](../../../release/flash-next-v3/README.md)). The prompts are R57's two
requests and a review of [exact_client.py](../2026-11-10_draft_line_exactness/run/exact_client.py) of this repository;
the tables hold token-id hashes and counts, no model text.
