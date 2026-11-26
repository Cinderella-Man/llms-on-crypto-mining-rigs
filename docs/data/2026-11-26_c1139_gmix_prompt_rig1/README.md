# Why `gmix-q2` reads prompts a third slower (C1139, Q1872, 2026-11-26)

What this measured: why Qwen3.8-Flash-Next's better-quality file `gmix-q2` reads prompts about a third slower than
`pmix-q2-fast`, the file it was built from (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF), on nine P106-090 cards. On the
coding-agent check ([C1173](../2026-11-26_c1173_gmix_agent_rig1/README.md)) `gmix-q2` took 8.7 % longer a task, and 68 %
of that was prompt reading. [R125](../2026-11-25_r125_gmix_step/README.md) had found an 8,128-token prompt 32.4 % slower.
`gmix-q2` puts the routed experts' gate and up matrices of 24 of its 48 layers in grid formats (IQ2_XXS, IQ2_S, IQ2_XS,
IQ1_M); everything else matches `pmix-q2-fast` apart from a few dense tensors. Here each file's prompt reads were taken
apart by part of the model, by matrix product and by weight type, and the fork's code was read to name the kernel each
type's prompt product runs on. Registry row [Q1872](../../../quants/experiments.md).

## Result

**Almost all of the prompt gap comes from one weight type, IQ1_M, which sits in only two layers.** The fork's fast
batched kernel for expert products, MMQ, takes Q2_0 and the three IQ2 types but not IQ1_M. So IQ1_M's prompt products
fall back to a generic path. That path copies the expert ids to the host and waits for the card twice per product,
then runs one small product per expert. The wait also switches off CUDA graphs for the whole graph of that card. The
two IQ1_M layers sit on cards 1 and 2. On those two cards the host spends 18.3 and 13.8 s inside their evaluations of
the 8,128-token prompt, against 1.1 and 1.5 s with `pmix-q2-fast`; every other card is within 0.35 s of its
`pmix-q2-fast` figure. One host thread drives all nine cards, so while it waits there, the cards stop overlapping their
work.

- **8,128 tokens from an empty cache** (micro-batches of 512): 145.1 and 144.5 tok/s against 221.0 and 222.5 (two loads
  each), +19,246.7 and +19,717.1 ms. Only est. 3,149 and 2,936 ms of that is GPU time. The rest, est. 16,097 and 16,781
  ms, is lost overlap.
- **1,028 tokens on top of 7,612 cached** (micro-batches of 64, the size that loses most on the agent check): 94.3 tok/s
  against 154.5, +4,243.8 ms (+63.8 %). Est. 355 ms is GPU time; est. 3,889 ms is lost overlap.
- **By type** (time per launch, one layer's product for one micro-batch, against Q2_0's in the same file): at 512 tokens
  IQ1_M takes 6.53 and 6.33 × Q2_0's time (gate, up; lane 2: 6.50 and 6.31), IQ2_XS 1.20 ×, IQ2_S 1.06 ×, IQ2_XXS
  1.04 ×; at 64 tokens IQ1_M 2.47 ×, IQ2_XS 1.36 ×, IQ2_S 1.12 ×, IQ2_XXS 1.20 ×. The four types hold about the bytes of
  Q2_0 (2.25 bits a weight): IQ1_M 1.75, IQ2_XXS 2.06, IQ2_XS 2.31, IQ2_S 2.56.
- **What a fix wins back:** IQ1_M on MMQ at Q2_0's rate, with no wait, would win est. 95.6 % and 96.7 % of the
  8,128-token gap (two loads), 91.5 % of the 1,540-token one and 95.0 % of the 1,028-token one. On the agent check that
  is est. +3.1 % a task instead of +8.7 %, which leaves the decode gap. The IQ2 types' MMQ tiles are worth est. 0.35 s
  of the 8,128-token prompt (1.8 %). The dense tensors' larger types cost est. 0.33 and 0.30 s; that is their bytes,
  not a kernel.
- **The code row this names:** C1179 (Q1914), IQ1_M on MMQ. Upstream llama.cpp's tag b12756 has no IQ1_M MMQ path
  either.

Every load of each file wrote the same replies, and the same as C1146's and R125's loads of that file.

### The prompts, plain loads (no profiler)

The deep lane primes each depth with that depth's prompt minus its tail. `prime d8192` reads 8,128 tokens on an empty
cache: two calls of up to 4,096 tokens in micro-batches of 512, 17 evaluations a card (the last one small). After the
depth-8,192 requests, the server resumes from its context checkpoint at 7,612 tokens; a hybrid model's recurrent state
can only resume at a checkpoint. So `prime d9216` read 1,540 tokens (more than the preset's taper limit of 1,400,
hence micro-batches of 512, 4 evaluations a card). `prime d8704` (the second lane) read 1,028 tokens in micro-batches
of 64 (17 evaluations a card).

| prime | lane, load | tokens read | `pmix-q2-fast` ms (tok/s) | `gmix-q2` ms (tok/s) | gap |
|---|---|---:|---:|---:|---:|
| d8192 | 1, `pu1` `gu1` | 8,128 | 36,786.5 (221.0) | 56,033.2 (145.1) | +19,246.7 ms (+52.3 %) |
| d8192 | 2, `pu2` `gu2` | 8,128 | 36,533.6 (222.5) | 56,250.7 (144.5) | +19,717.1 ms (+54.0 %) |
| d9216 | 1, `pu1` `gu1` | 1,540 | 8,996.6 (171.2) | 13,421.4 (114.7) | +4,424.8 ms (+49.2 %) |
| d8704 | 2, `pu2` `gu2` | 1,028 | 6,651.8 (154.5) | 10,895.6 (94.3) | +4,243.8 ms (+63.8 %) |

R125 measured the 8,128-token prime at 218.9 and 148.0 tok/s on the `eee615cdb` build; C1146 at 224.4 and at 146.7
and 147.5 on this build.

### Where the gap goes: GPU time and lost overlap (est.)

The group profile synchronizes each card after every evaluation, so it gives each product's time with the cards run
one after another. A plain read overlaps the cards (pipeline parallelism over the micro-batches of a call), so it takes
only a fraction f of the cards' summed GPU time: `pmix-q2-fast`'s plain time over its group-profile GPU time. If
`gmix-q2` overlapped as well, it would take f × its own GPU time. What its plain time adds beyond that is the lost
overlap. Each GPU-time difference is priced at f ([summary.md](summary.md), from `prompt_parts.py`'s pricing):

| prime | lane | f | gap ms | GPU difference at f | lost overlap | IQ1_M gate/up at f | IQ2_XS at f | IQ2_XXS at f | IQ2_S at f | IQ1_M on MMQ at Q2_0's rate, no wait (est.) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| d8192 | 1 | 0.4197 | +19,246.7 | +3,149 | +16,097 | +2,311 | +168 | +99 | +88 | +18,408 (95.6 %) |
| d8192 | 2 | 0.4145 | +19,717.1 | +2,936 | +16,781 | +2,276 | +166 | +87 | +85 | +19,057 (96.7 %) |
| d9216 | 1 | 0.4760 | +4,424.8 | +892 | +3,533 | +517 | +40 | +30 | +34 | +4,050 (91.5 %) |
| d8704 | 2 | 0.2886 | +4,243.8 | +355 | +3,889 | +144 | +70 | +110 | +40 | +4,033 (95.0 %) |

The last column is IQ1_M's GPU excess at f plus the lost overlap. The host profile below puts the lost overlap on the
two cards that hold the IQ1_M layers, and the code puts the wait in IQ1_M's path, so both parts go if IQ1_M takes MMQ.
How the two parts split depends on the overlap model; their sum does not.

### The host's wait (host profile: nothing synchronized)

`GGML_CUDA_PROFILE_HOST` records, per card and evaluation, the host time from entering the evaluation to leaving it,
and the GPU time. The profiler's own timing breaks most of the overlap here (the 8,128-token prime takes 57,082.5 ms
with `pmix-q2-fast` under it, not 36,786.5), but it shows where the host waits. Prime d8192, lane 1 (ms):

| Card | `pmix-q2-fast` host | `pmix-q2-fast` GPU | `gmix-q2` host | `gmix-q2` GPU | host difference |
|---:|---:|---:|---:|---:|---:|
| 0 | 421.0 | 11,152.4 | 440.6 | 11,480.4 | +19.6 |
| **1** | 1,125.5 | 11,729.3 | **18,348.4** | 15,441.6 | **+17,222.9** |
| **2** | 1,474.2 | 10,167.8 | **13,800.1** | 14,227.6 | **+12,325.8** |
| 3 | 1,379.8 | 11,351.3 | 1,036.9 | 11,508.0 | −342.9 |
| 4 | 1,435.4 | 8,742.6 | 1,457.4 | 9,007.5 | +22.0 |
| 5 | 1,202.0 | 10,251.9 | 1,254.7 | 10,283.1 | +52.6 |
| 6 | 1,397.6 | 9,772.6 | 1,409.0 | 9,722.9 | +11.4 |
| 7 | 1,270.3 | 10,351.3 | 1,258.7 | 10,320.8 | −11.6 |
| 8 (with the draft) | 729.8 | 5,424.4 | 731.7 | 5,331.4 | +2.0 |

Prime d9216 (1,540 tokens, micro-batches of 512): cards 1 and 2 spend 4,067.2 and 2,942.6 ms of host time against 13.4
and 12.3, every other card within 38 ms. Prime d8704 (1,028 tokens, micro-batches of 64, lane 3): cards 1 and 2 spend
5,019.9 and 3,055.5 ms against 60.3 and 56.8, every other card within 20 ms. R125's loads on the older build show the
same: cards 1 and 2 at 18,248.8 and 13,738.3 ms against 1,071.5 and 1,463.1.

### By part (group profile, ms of GPU time for the request, all cards)

| Part | d8192 lane 1: `pmix-q2-fast` | `gmix-q2` | difference | d8192 lane 2: difference | d8704 (64): `pmix-q2-fast` | `gmix-q2` | difference |
|---|---:|---:|---:|---:|---:|---:|---:|
| routed experts | 39,497.5 | 45,919.3 | +6,421.8 | +6,203.0 | 14,042.8 | 15,163.4 | +1,120.6 |
| attention and recurrent projections | 11,942.1 | 12,726.2 | +784.1 | +715.3 | 1,705.0 | 1,792.1 | +87.1 |
| hyper-connections (BF16) | 11,181.4 | 11,269.9 | +88.5 | +37.1 | 3,571.1 | 3,583.5 | +12.4 |
| attention (FLASH_ATTN_EXT) | 12,678.3 | 12,752.3 | +74.0 | −2.4 | 1,814.2 | 1,818.5 | +4.3 |
| shared expert | 1,283.2 | 1,333.3 | +50.2 | +43.5 | 196.8 | 199.6 | +2.8 |
| everything else (router, Gated DeltaNet, output, other ops and weights) | 9,842.1 | 9,925.9 | +83.8 | +86.1 | 1,563.7 | 1,566.5 | +2.8 |
| the MTP draft's catch-up over the prompt | 1,219.7 | 1,220.7 | +1.1 | +0.7 | 155.3 | 155.1 | −0.2 |
| **all** | **87,644.3** | **95,147.8** | **+7,503.5** | **+7,083.3** | **23,048.9** | **24,278.6** | **+1,229.7** |

The routed experts' difference is all in the gate and up products: the down products are Q2_0 in both files and differ
by +96.8 ms at 8,128 tokens (lane 1) and +30.8 at 1,028. The attention and recurrent projections' difference is the
dense tensors `gmix-q2` took from `pmix-iq3-fast` (Q5_K and Q6_K `attn_qkv` where `pmix-q2-fast` has Q2_0 in some
layers: +407.0 ms at 8,128 tokens), as in R125's verify pass. Per card, `gmix-q2`'s GPU time grows most on cards 1
and 2 (+3,033.5 and +3,295.5 ms at 8,128 tokens), the IQ1_M cards. Every product: [prompt_parts.md](prompt_parts.md)
(lane 1) and [prompt_parts2.md](prompt_parts2.md) (lanes 2 and 3).

### The routed gate/up by weight type, and the kernel each one runs

Read in the fork at `883f21f2c` (`ggml/src/ggml-cuda/ggml-cuda.cu`, `ggml_cuda_mul_mat_id`; `mmq.cu`,
`ggml_cuda_should_use_mmq`). A prompt micro-batch of 64 or 512 tokens is more than `MMVQ_MAX_BATCH_SIZE` (8) and more
than few-row MMQ's 24 columns. On a P106 (compute 6.1, no FP16 tensor cores) `ggml_cuda_should_use_mmq` then returns
true for every type on MMQ's list, at any batch, when the product has experts:

| Type | gate/up layers | bits a weight | prompt product on a P106 |
|---|---:|---:|---|
| Q2_0 | 24 (`pmix-q2-fast`: 48) | 2.25 | MMQ: `mul_mat_q` with the expert ids on the card, DP4A tiles |
| IQ2_XXS | 11 | 2.06 | MMQ (`load_tiles_iq2_xxs`) |
| IQ2_S | 7 | 2.56 | MMQ (`load_tiles_iq2_s`) |
| IQ2_XS | 4 | 2.31 | MMQ (`load_tiles_iq2_xs`) |
| IQ1_M | 2 (cards 1 and 2) | 1.75 | **not on MMQ's list** (nor few-row MMQ's or MMF's), so the generic fallback: the ids copied to the host and the stream synchronized, the sorted ids copied back and the stream synchronized again, then one product per expert that has tokens (MMVQ up to 8 tokens, else IQ1_M dequantized to F32 and cuBLAS SGEMM, as compute 6.1 has no fast FP16). `ggml_cuda_mul_mat_id_needs_sync` is true for it, and `ggml_cuda_graph_check_compability` then turns CUDA graphs off for the card's whole graph |

ms a launch (one layer's gate or up for one micro-batch, 512 experts of 2,560 × 640 each, so a launch compares per
weight), and the ratio to Q2_0's launch in the same file:

| Type | 512 tokens, lane 1: gate, up | lane 2: gate, up | × Q2_0 | 64 tokens (lane 2): gate, up | × Q2_0 |
|---|---|---|---:|---|---:|
| Q2_0 (`gmix-q2`) | 15.843, 15.829 | 15.879, 15.868 | 1.00 | 5.296, 5.289 | 1.00 |
| Q2_0 (`pmix-q2-fast`) | 15.858, 15.850 | 15.960, 15.949 | 1.00 | 5.412, 5.405 | 1.00 |
| IQ2_XXS | 16.520, 16.489 | 16.484, 16.462 | 1.04 | 6.381, 6.373 | 1.20 |
| IQ2_S | 16.771, 16.762 | 16.787, 16.787 | 1.06 | 5.923, 5.912 | 1.12 |
| IQ2_XS | 18.955, 18.965 | 19.013, 18.993 | 1.20 | 7.203, 7.196 | 1.36 |
| IQ1_M | 103.499, 100.221 | 103.252, 100.087 | 6.53, 6.33; 6.50, 6.31 | 13.101, 13.077 | 2.47 |

Under the group profile a launch group's time runs from the event before it to the event after it. For IQ1_M that
span also holds the fallback's two waits and its launches, one per expert with tokens.

### What a faster prompt kernel per type would win back (est., plain time)

| Type or part | 8,128 tokens (lanes 1; 2) | 1,540 tokens | 1,028 tokens, micro-batches of 64 | what it needs |
|---|---:|---:|---:|---|
| IQ1_M (2 layers) | 18,408; 19,057 ms | 4,050 | 4,033 | IQ1_M on MMQ: no wait, CUDA graphs kept, Q2_0's rate (C1179) |
| IQ2_XS (4) | 168; 166 | 40 | 70 | a faster MMQ tile loader (C1179, second part) |
| IQ2_XXS (11) | 99; 87 | 30 | 110 | the same |
| IQ2_S (7) | 88; 85 | 34 | 40 | the same |
| dense projections | 329; 296 | 103 | 25 | bytes (the file's choice), no kernel |
| the gap | 19,246.7; 19,717.1 | 4,424.8 | 4,243.8 | |

On the agent check (C1173) the prompt part of a task was 0.477 and 0.457 minutes with `gmix-q2` against 0.330 and
0.331. Taking away est. 95 % of that gap leaves a task est. +3.1 % longer than with `pmix-q2-fast` (from +8.7 %), the
decode gap that remains after C1146's kernels.

### Replies

Every load of a file wrote the same greedy and sampled replies at every depth: `pg`, `ph`, `pu1` alike, `gg`, `gh`, `gu1`
alike, and so on for lanes 2 and 3 ([replies_pmix.md](replies_pmix.md), [replies_gmix.md](replies_gmix.md),
[replies_pmix2.md](replies_pmix2.md), [replies_gmix2.md](replies_gmix2.md)). At depths 0 and 8,192 they are the
replies of C1146's `gmix-q2` loads at `=2` (`g2`, `u2`) and of its `pmix-q2-fast` load (`p0`), and of R125's
`pmix-q2-fast` load.

## What ran

- **Rig and build:** nine P106-090 cards at stock clocks (no offsets this boot), the fork's `883f21f2c`
  (`pascal-p106-v5-iq-mtok-fast`, C1146's build, `llama-server` md5 `dae09aa55e6fe7e875a0d57bbaf0b796`).
- **Preset:** Flash-Next's v5 preset as it stood at commit `26e6575a` (`release/flash-next-v5/settings.env`, before
  `d296f5ce` changed the sparse attention's splits and threshold; both files ran the same line), the MTP draft on,
  by the deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) with R125's settings: R125's recorded agent
  text (`TAG='^t1swift/'`), 256 tokens a request, a sampler pass, four idle seconds before each request. `gmix-q2` with
  `GGML_CUDA_MTOK_IQ_FAST=2` (the PTX grid kernels of the four-token verify, as in C1173); `pmix-q2-fast` without it.
  The dry run checked that each `gmix-q2` line is its `pmix-q2-fast` line plus that switch and `-m <gmix-q2>`.
- **Lane 1** (13:33:54–14:37:09): depths 0, 8,192, 9,216; loads `warm` (untimed), `pg` `gg` (group profile), `ph` `gh`
  (host profile), `pu1` `gu1` (no profiler).
- **Lane 2** (14:39:14–15:18:16): depths 0, 8,192, 8,704, so that the last prime reads 1,028 tokens in micro-batches of
  64 (lane 1's 9,216 read 1,540 because of the checkpoint); loads `gg2` `pg2` (group profile), `gu2` `pu2` (none).
- **Lane 3** (15:20:17–15:38:51): lane 2's depths; loads `gh3` `ph3` (host profile), for the host's wait at 64 tokens.

## Files

| File | What it is |
|---|---|
| [summary.md](summary.md) | every prime of lanes 1 and 2: tokens, micro-batches, plain times, the gap and its pricing |
| [prompt_parts.md](prompt_parts.md), [prompt_parts.tsv](prompt_parts.tsv) | lane 1's primes: plain, group and host profile; by part, card, routed type and product; pricing; the host table |
| [prompt_parts2.md](prompt_parts2.md), [prompt_parts2.tsv](prompt_parts2.tsv) | the same for lane 2's primes (group profile and plain) with lane 3's host profile |
| `replies_*.md` | the replies of every load, side by side |
| [table.md](table.md), [table2.md](table2.md), `lane_classes*.tsv` | the deep lane's own tables of lanes 1 and 2 (decode speed by depth; not this row's question) |
| `requests1.csv`, `requests2.csv`, `requests3.csv` | every request of each lane (deep lane format: kind, depth, tokens read, prompt and generation times, reply md5) |
| `loads1.tsv`, `loads2.tsv`, `loads3.tsv` | each load's server line (rig paths as `<BIN>`, `<MODELS>`, `<OUT>`) |
| [run/c1139.sh](run/c1139.sh) | the driver (R125's lane with the three profilers per file) |
| [run/prompt_parts.py](run/prompt_parts.py) | the tables: profiler files cut at the requests' idle gaps, the primes by part (R41's parts), card, product and routed type, the host table, the pricing |
| [run/summary.py](run/summary.py), [run/replies.py](run/replies.py), [run/tables.sh](run/tables.sh) | the summary, the reply check, and the script that ran them all |

The profiler traces (group profile 121, 131, 122 and 137 MB for `pg`, `gg`, `pg2`, `gg2`; host profile about 0.4 MB
each) are not published; they summarize into the tables above.

## Commands

From the repository's root on the operator's machine (`RIG` the rig's ssh alias; `BIN`, `MODEL`, `GMIX`, `MODEL_DRAFT`
paths on the rig; `TEXT_LOCAL` the release check's `proxy.jsonl.bodies.gz`, md5 `820be205246dd1113f4dff327c4b27d1`,
with its `proxy.jsonl`):

```bash
export RIG=rig BIN=<llama-server of 883f21f2c> MODEL=<pmix-q2-fast shard 1> GMIX=<gmix-q2 shard 1> \
  MODEL_DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> TEXT_LOCAL=<proxy.jsonl.bodies.gz>
D=docs/data/2026-11-26_c1139_gmix_prompt_rig1/run
# lane 1
WORK=/var/tmp/c1139 STEP=dry bash $D/c1139.sh && WORK=/var/tmp/c1139 STEP=start bash $D/c1139.sh
WORK=/var/tmp/c1139 STEP=wait bash $D/c1139.sh && WORK=/var/tmp/c1139 OUT=<out1> STEP=collect bash $D/c1139.sh
# lane 2
WORK=/var/tmp/c1139b DEPTHS="0 8192 8704" LOADS="gg2:gg pg2:pg gu2:gu pu2:pu" STEP=start bash $D/c1139.sh  # then wait, collect to <out2>
# lane 3
WORK=/var/tmp/c1139c DEPTHS="0 8192 8704" LOADS="gh3:gh ph3:ph" STEP=start bash $D/c1139.sh  # then wait, collect to <out3>
# tables (laptop, CPU only)
S=$PWD OUT=<out1> OUT2=<out2> OUT3=<out3> RES=<tables> bash $D/tables.sh
```

## Provenance and licence

Measured on one rig on 2026-11-26 with the fork's build `883f21f2c` and the files `pmix-q2-fast` and `gmix-q2` (built
from UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next; `gmix-q2` is registry row Q1382's file). The prompts are
cut from the release check's recorded agent sessions (this repository's
`bench/agent_check` tasks). Tables and scripts: this repository's licence.
