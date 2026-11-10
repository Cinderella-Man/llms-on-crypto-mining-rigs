# A served step at depth, part by part (R51, Q1213, 2026-11-10)

What this measured: what still grows with context depth in a served speculative step once the sparse Pascal attention
is on, on the Miner Llama v3.0 release line (`pascal-p106-v3-b12076-ladder5` `1551f5fe1`, the v3 preset exactly:
UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three tokens) with
`GGML_CUDA_FATTN_SPARSE_PASCAL=1` ([Q1213](../../../quants/experiments.md)). With the switch on, the deep lane's
repeated step took 149.73 ms at 16,384 tokens, 165.86 at 32,768 and 180.77 at 49,152
([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md), R39a, `sparse1`): 31.04 ms from 16,384 to 49,152. This
run profiles that step, the verify pass and the draft's passes together, at 16,384, 32,768 and 49,152 tokens with the
fork's profiler in group mode, beside the shallow step of
[2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md) (R49, depth 79 to 335). A second run on
the same rig the same evening (R64, [Q1243](../../../quants/experiments.md)) measured the part outside the kernels
with the fork's host profile: [Outside the kernels, measured](#outside-the-kernels-measured-r64).

## Result

**Less than half of the growth is in the kernels, and most of that is one gather.** From 16,384 to 49,152 tokens the
unprofiled step grows by 30.44 ms on pass 2 and 32.68 on pass 1 (rig 1, this run's unprofiled load). The kernels
explain 13.98 and 14.11 ms of it:

- **the gather of every pooled indexer key, 9.71 ms** (31.9 % and 29.7 % of the growth). For each QSA layer and pass,
  the model gathers all pooled keys by cell (one key per block of four cells) into a new tensor with a `GET_ROWS`, and
  the indexer reads them from there. That is 12 gathers in a verify pass (11 on cards 0 to 7, one on card 8) and four
  in a step's draft passes; at 49,152 tokens card 1's two took 942.08 and 943.10 µs (raw) in one verify pass;
- **the indexer's scores (`LIGHTNING_INDEXER`), 3.41 ms** (11.2 % and 10.4 %);
- **little else:** the attention kernel with its list of kept cells +0.25 and +0.26, the indexer's top-k and mask
  +0.18 and +0.19, the draft's shortlist head +0.43 and +0.56, everything else +0.01 and −0.01.

The other est. 16.46 and 18.57 ms (54.1 % and 56.8 %) are outside the kernels: work on the host or the links that
grows with the cache. This run cannot say what it is.

**Cut next:** the gather is measured, exact to remove and 14.69 ms of a 181.86 ms step at 49,152 tokens
(new row Q1244). First, though, measure what grows outside the kernels (Q1243), since that part is larger.

### Where the growth goes

ms a step, the mean over the typical steps of one request, the profiler's 7.6 µs subtracted per launch group
([buckets.tsv](buckets.tsv); the first rule that matches a launch group wins, `run/depth_parts.py`, `BUCKETS`). Every
column is one request of the one profiled load; p1 is the first request at a depth, p2 the same request again. The
unprofiled step is the server's own figure (generation time over steps) for the same request in the load before it.

| Where | 16,384 p1 | 16,384 p2 | 32,768 p1 | 32,768 p2 | 49,152 p1 | 49,152 p2 | 49,152 − 16,384, p1 | 49,152 − 16,384, p2 | Share of the growth, p1 / p2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| the gather of every pooled key (`GET_ROWS` before the indexer) | 4.97 | 4.97 | 9.84 | 9.84 | 14.69 | 14.69 | +9.71 | +9.71 | 29.7 % / 31.9 % |
| the indexer's scores (`LIGHTNING_INDEXER`) | 1.98 | 1.98 | 3.69 | 3.69 | 5.39 | 5.39 | +3.41 | +3.41 | 10.4 % / 11.2 % |
| the indexer's top-k and mask (`TOP_K`, `TOP_K_MASK`) | 0.89 | 0.90 | 0.98 | 0.98 | 1.08 | 1.08 | +0.19 | +0.18 | 0.6 % / 0.6 % |
| the indexer's projections, norms and RoPE | 0.70 | 0.70 | 0.68 | 0.70 | 0.69 | 0.70 | −0.01 | −0.01 | |
| attention (`FLASH_ATTN_EXT`: the list of kept cells and the sparse kernel) | 14.81 | 14.81 | 14.77 | 14.78 | 15.07 | 15.06 | +0.26 | +0.25 | 0.8 % / 0.8 % |
| the output heads (the verify pass's and the draft's shortlist head) | 10.79 | 10.63 | 11.08 | 10.80 | 11.35 | 11.06 | +0.56 | +0.43 | 1.7 % / 1.4 % |
| everything else | 100.11 | 100.12 | 99.79 | 99.80 | 100.09 | 100.13 | −0.01 | +0.01 | |
| **GPU time of the step** | **134.24** | **134.12** | **140.83** | **140.59** | **148.36** | **148.10** | **+14.11** | **+13.98** | **43.2 % / 45.9 %** |
| outside the kernels, unprofiled (est.: the unprofiled step less the GPU time) | 35.83 | 17.30 | 43.24 | 25.08 | 54.40 | 33.77 | +18.57 | +16.46 | 56.8 % / 54.1 % |
| **the step, unprofiled** | **170.07** | **151.42** | **184.07** | **165.67** | **202.76** | **181.86** | **+32.68** | **+30.44** | 100 % |

- **Linear in the depth.** The gather takes 4.97, 9.84 and 14.69 ms a step at the three depths (+4.87 and +4.85 per
  16,384 tokens), the scores 1.98, 3.69 and 5.39. Est. at 65,536 tokens: 19.5 and 7.1 ms.
- **From 32,768 to 49,152** (the brief's two depths): the step grows 18.69 and 16.20 ms unprofiled, the GPU time 7.53
  and 7.51 (the gather +4.85, the scores +1.70, the attention +0.30 and +0.29).
- **Against R39a's 31.04 ms on rig 2** (`sparse1`, pass 2), the kernels' 13.98 ms are 45.0 %. This run's unprofiled
  pass-2 steps (151.42, 165.67, 181.86) lie within 1.69 ms of R39a's (149.73, 165.86, 180.77).
- **The attention does not grow.** The sparse kernel reads about 2,051 cells whatever the depth, as designed. Its list
  of kept cells is built from a mask as long as the cache (Q441), and that adds at most 0.26 ms a step over the range.
  The kernel itself costs 14.81, 14.77 and 15.07 ms a step on pass 1 at the three depths (12.11 on cards 0 to 7 at
  49,152, for 11 layers and four query tokens), against 1.27 ms for the dense kernel at shallow depth. That cost is
  the target of Q643's union gather; it does not grow with depth.
- **The draft's own passes grow by 3.55 and 3.41 ms:** the gather +2.43, the scores +0.44, the shortlist head +0.55 and
  +0.43, the attention +0.07 and +0.06. The shortlist adds up to 8,192 rows from the context's tokens to its 32,768
  (`LLAMA_DRAFT_SHORTLIST=32768,8192,2`), so a deeper context fills more of them (est.: its `MUL_MAT_ID output`
  groups take 0.76, 0.93 and 1.19 ms a step on pass 2). **The draft's attention does not read its whole
  cache:** the sparse kernel takes its one-token calls too (1.81, 1.84 and 1.87 ms a step for four draft passes). What
  grows in the draft is its indexer, as in the main model (Q1153).
- **Pass 1 differs from pass 2 only outside the kernels.** The GPU time is the same on both passes (134.24 and 134.12
  at 16,384). The unprofiled step is 18.65, 18.40 and 20.90 ms longer on pass 1. During pass 1 at 16,384 the host
  moved 113 MiB (unprofiled load) and 110 MiB (profiled load) to swap, and none during pass 2
  ([host_memory.tsv](host_memory.tsv)). That is Q1133's first-request cost.

### What grows outside the kernels (est.)

- **Under the profiler**, the time outside the kernels grows by 28.41 and 27.87 ms (53.86 → 82.27 and 37.97 → 65.84 ms a
  step). This profiler synchronizes after every evaluation, so it holds more than the unprofiled est. of 18.57 and
  16.46.
- **The candidate is the QSA k-pool inputs** (Q441, Q640). Each pass, the host fills `pool_cells`, `pool_idxs`,
  `pool_mask` and the tails, and every card that runs a QSA layer receives them: all nine cards in a verify pass, and
  card 8 in each draft pass.
  - Size: 28 bytes per pool of four cells for a four-token pass, 22 for a one-token pass. At 49,152 tokens that is
    est. 344 KB per card for a verify pass and 270 KB for a draft pass: est. 4.2 MB a step, against 1.4 MB at 16,384.
  - The extra 2.8 MB is est. 14.7 ms at the ~190 MB/s of a card's PCIe 1.0 x1 link, if no copy overlaps with the
    cards' work.
  - Q441 estimated that these copies overlap with the cards before them and put ~1 to 1.5 ms at 32k on the token's
    path. This run's 16.46 ms says that something of their size is on the path, but it does not show that it is them.
- **Measured in R64** ([below](#outside-the-kernels-measured-r64)): the uploads of these inputs are on the token's
  path, and est. 9.76 of the 15.69 ms that grow outside the kernels follow their size. Q640 reopens.

### The step at each depth

ms a step ([totals.tsv](totals.tsv), [parts.tsv](parts.tsv) by R41's parts). R49 r1 and r2 are R49's two requests at
depth 79 to 335, cut with this run's script from R49's trace; their GPU time per verify pass is R49's (88.13 and 12.86
or 12.87 ms).

| | R49 r1 | R49 r2 | 16,384 p1 | 16,384 p2 | 32,768 p1 | 32,768 p2 | 49,152 p1 | 49,152 p2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| verify pass, cards 0 to 7 | 88.13 | 88.13 | 104.71 | 104.75 | 109.03 | 109.07 | 114.30 | 114.34 |
| verify pass, card 8 | 12.86 | 12.87 | 14.50 | 14.49 | 14.96 | 14.97 | 15.47 | 15.46 |
| the draft's four passes | 13.05 | 11.05 | 15.04 | 14.88 | 16.83 | 16.54 | 18.59 | 18.29 |
| **GPU time of the step** | **114.04** | **112.05** | **134.24** | **134.12** | **140.83** | **140.59** | **148.36** | **148.10** |
| wall time a step under the profiler | 159.13 | 159.00 | 208.58 | 192.57 | 219.12 | 212.28 | 251.10 | 234.42 |
| outside the kernels under the profiler | 24.72 | 26.56 | 53.86 | 37.97 | 57.81 | 51.22 | 82.27 | 65.84 |
| the server's step, profiled | 172.79 | 157.72 | 206.35 | 190.52 | 216.91 | 210.18 | 262.99 | 230.37 |
| the server's step, unprofiled load | 144.40 | 123.93 | 170.07 | 151.42 | 184.07 | 165.67 | 202.76 | 181.86 |

- **From the shallow step to 16,384** (r2 against p2), the GPU time grows by 22.07 ms. The attention accounts for
  +13.55 of it (the sparse kernel against the dense kernel on a short cache), the gather +4.85, the scores +1.83, the
  indexer's top-k and mask +0.59, and the draft's shortlist head +0.61. Outside the kernels the est. grows by 5.42 ms
  (17.30 against R49's unprofiled 123.93 less 112.05).
- **Per card** ([per_card.tsv](per_card.tsv)): cards 1, 3 and 6 hold two QSA layers each, and each of them grows
  1.79 ms from 16,384 to 49,152 on pass 2. The other cards hold one each and grow 1.00 (card 0), 0.79, 0.85, 0.77,
  0.81 and 0.97 ms (cards 2, 4, 5, 7 and 8).
- R49's server figures are R49's ([replies.md](../2026-11-10_served_verify_parts/replies.md)).

### What the code rows could save (est.)

| Row | What it removes | est., ms a step at 16,384 / 49,152 |
|---|---|---:|
| new [Q1244](../../../quants/experiments.md) (C704): the indexer reads the pooled keys where they are, with no gather into a new tensor (a `LIGHTNING_INDEXER` that takes `pool_cells`, or the pooled keys kept in pool order so that the gather is a view); exact | the gather (and, fused, part of the scores' read) | up to 4.97 / 14.69 (3.3 % / 8.1 % of the unprofiled 151.42 / 181.86) |
| [Q1153](../../../quants/experiments.md)'s index reuse: the draft's passes reuse the selection of the step's last accepted row | the draft's gather and scores | up to 1.53 / 4.40 |
| [Q640](../../../quants/experiments.md), the k-pool tables kept on the cards (reopened by R64) | the inputs' uploads and host build | est. 2.70 / 12.43 (R64: card 0's and the draft's four uploads at 188 MB/s, 2.70 and 8.02, and at 49,152 the verify pass's rest that grows with the other cards' uploads, +4.41) |
| [Q643](../../../quants/experiments.md), the union gather for verify passes | part of the attention, which is flat in depth | not estimated here |

## Outside the kernels, measured (R64)

The same line on the same rig at 19:26 to 19:59 ([Q1243](../../../quants/experiments.md)): an unprofiled load, then
one load with the fork's host profile (`GGML_CUDA_PROFILE_HOST`) at 16,384 and 49,152 tokens, the same requests as
above. The host profile writes, per card and graph evaluation, the host's entry to and exit from `graph_compute` and
the GPU time between two CUDA events around the evaluation's work; it adds two events an evaluation and synchronizes
nothing. The scheduler copies a card's host inputs (the QSA k-pool inputs among them) right before it enters that
card's `graph_compute`, so each upload lies in the host's gap before that entry.

### Result

**Q640 reopens: the k-pool inputs' uploads are on the token's path, and est. 9.76 of the 15.69 ms that grow outside
the kernels follow their size.** From 16,384 to 49,152 tokens the step grows 30.33 ms on pass 2 (153.26 → 183.58 ms),
the GPU time of its evaluations 14.64 (135.00 → 149.64; the group profile above: 13.98) and the time outside the
kernels 15.69 (18.26 → 33.94; the estimate above: 16.46). ms a step, the mean over 66 and 69 typical steps of one
request each ([host_depth.md](host_depth.md); the parts add up to the step):

| Outside the kernels, pass 2 | 16,384 | 49,152 | growth | share of the growth |
|---|---:|---:|---:|---:|
| the verify pass's rest (its span less the nine cards' GPU time: the eight crossings, the rows back, the acceptance, the server before the draft) | 6.65 | 11.05 | +4.41 | 28.1 % |
| the draft's four crossings (card 4's exit to card 8's entry: the token's row through the host and card 8's uploads) | 5.13 | 9.26 | +4.12 | 26.3 % |
| the draft's turns before its last pass (card 8's end to the next pass's entry) | 1.53 | 3.40 | +1.86 | 11.9 % |
| the server between the last draft pass and the next verify pass (with the verify pass's input build and card 0's uploads) | 4.93 | 10.23 | +5.30 | 33.8 % |
| **outside the kernels** | **18.26** | **33.94** | **+15.69** | 100 % |
| the host's launch of the verify pass, inside its span (card 0's entry to card 8's exit) | 8.74 | 18.72 | +9.98 | |

- **Every upload interval grows by the inputs' extra bytes at the link's speed.** The k-pool inputs take 28 bytes per
  pool of four cells for a four-token pass and 22 for one token; with 4,160 and 12,352 pools (est., from the tensors'
  shapes) that is 229,376 bytes more a card for a verify pass and 180,224 for a drafting pass.
  - In the verify pass the host's gap before cards 1 to 8 grows by 1.23, 1.23, 1.22, 1.21, 1.23, 1.28, 1.23 and
    1.21 ms (from 0.92, 0.89, 0.89, 0.82, 0.88, 0.88, 0.89 and 0.94 at 16,384): 186.5 MB/s on average.
  - Each drafting pass's crossing grows by 0.96 ms (187.7 MB/s), the catch-up pass's by 1.24 (185.0 MB/s at the
    four-token size; it reads the step's accepted tokens).
  - The rig's links move ~189 MB/s from the host to one card at a time, and the cards behind one PCIe switch share
    its upstream link when several take data at once ([mining_rig.md](../../mining_rig.md), its PCIe table).
- **What lies on the token's path:**
  - *the draft's uploads:* its passes run one after another, and each waits for card 8's inputs, so the four
    crossings' +4.12 ms add to the step;
  - *card 0's upload:* it lies in the server's interval before the verify pass, est. +1.23 of that interval's +5.30,
    as on every other card;
  - *the uploads to cards 1 to 8:* they run while the cards before them compute (the host's launch takes 18.72 ms
    against the cards' 131.04 ms of GPU time). Yet the verify pass's rest grows by 4.41 ms, though its crossings move
    the same bytes at every depth: est. the uploads hold the switches' shared links while the crossings need them.
- **Together est. 9.76 ms:** 4.12 + 1.23 (est.) + 4.41 (est.). The other est. 5.93 ms is host work in the turns (the
  server's interval beyond card 0's upload +4.07, the draft's turns +1.86). The host's build of the inputs (their
  `set_input`) is part of it; how much was not measured (below).
- **What keeping the tables on the cards could save** (Q640: a few hundred bytes a token instead of the whole layout):
  est. 2.70 ms a step at 16,384 (card 0's and the draft's uploads at 188 MB/s) and 12.43 at 49,152 (8.02 of uploads and
  the +4.41 of the verify pass's rest), 1.8 % and 6.8 % of the step, plus the host's build. The draft's four uploads a
  step carry nearly the same tables (one token more each): kept on card 8 alone, they are est. 2.08 and 6.18 ms of it.
- **Not measured: the inputs' host time and their bytes per ubatch.** `LLAMA_QSA_INPUT_STATS=1` was set, but it logs at
  the library's info level, which llama-server prints only from `-lv 4` on (its default is 3): the server log holds no
  such line. The bytes above are from the tensors' shapes. Q1255 repeats the load with `-lv 4`.

**Pass 1** (the first request at a depth):

| | 16,384 p1 | 49,152 p1 |
|---|---:|---:|
| the step, mean | 196.35 | 188.27 |
| the step, median | 154.84 | 187.03 |
| outside the kernels, median | 18.37 | 37.52 |
| the server's interval before the verify pass, mean | 47.65 | 13.91 |
| the server's interval, median | 4.87 | 13.93 |

- At 16,384, 23 of the 66 typical steps spent over 100 ms each in the server's interval before the verify pass
  (2,903.4 ms together; the other 43 steps 5.62 ms on average), while the host moved 147 MiB to swap and its
  MemAvailable fell to 61 MiB ([host_memory_r64.tsv](host_memory_r64.tsv)). The cards' GPU time is the same as on pass 2
  (135.32 and 135.00 ms). That is Q1133's first-request cost: host waits, not kernels.
- At 49,152 pass 1's step is 4.69 ms longer than pass 2's, 3.68 of it in the same interval (13.91 against 10.23), with
  no step over 100 ms.

**The host profile changes the decode step little:** the server's own step on pass 2 is 151.44 and 181.48 ms in the
profiled load against 150.33 and 180.87 in the unprofiled one (R51's unprofiled load: 151.42 and 181.86). Prompts
read slower under it: 107.5 and 71.7 tok/s against 122.8 and 80.4 for the same two chunks (16,260 tokens on 60 cached,
33,284 on 15,804).

### What ran (R64)

One deep-lane run on the rig, nothing else on the rig during the loads, 19:26:24 to 19:59:42 (33.3 minutes):

| Load | What | Loaded in | Requests |
|---|---|---:|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1`: the warm-up load, and the unprofiled step of the same requests | 308 s | the client's warm-up; at 16,384 and 49,152 tokens a prime and two decodes of 256 greedy tokens |
| `host` | the same with `GGML_CUDA_PROFILE_HOST=$OUT/host.csv` and `LLAMA_QSA_INPUT_STATS=1` | 311 s | the same |

- **The prompts:** the primes read 16,260 tokens on 60 cached, then 33,284 on 15,804 (the 49,152 prime continues
  from the 16,384 one's checkpoint).
- **The replies:** each load wrote its own greedy reply at each depth and repeated it on pass 2; the profiled load's
  draft accepted 186 of 206 at 16,384 and 182 of 218 at 49,152, the unprofiled load's 184 of 210 at both.
- **The host's memory:** the lowest MemAvailable was 196 MiB over the unprofiled load (its first prime) and 61 MiB over
  the profiled one (its first decode at 16,384); the RAM watchdog (floor 60 MiB) never acted.
- **The profile** holds 6,774 evaluations: the load's own, then one piece per request at the pauses (`PAUSE=4`), with
  each depth's first decode cut once more after its tail's pass, where the server saves a context checkpoint
  ([host_pieces.tsv](host_pieces.tsv)).
- **The tables ran on a build machine's CPU** (`nice -n 19`, under a second), never on the rig.

### How the host profile is cut

`run/host_depth.py` (its header has the definitions; `--self-test`, 29 checks, all pass) uses the step and its parts
of [2026-11-10_served_step_split](../2026-11-10_served_step_split/README.md)'s `run/step_split.py` (S37):
- **The pieces:** gaps of more than 3 s cut the profile (a prompt micro-batch at depth leaves the host idle up to
  ~2.5 s; the client pauses 4 s before every request). A piece of fewer than ten main passes after the load's own is
  joined to the next one: a first decode's tail pass, cut off by the checkpoint the server saves after it. The pieces
  then match the load's requests in `requests.csv` one to one, or the script stops.
- **A main pass** is one evaluation on each of the nine cards in order; **a draft pass** the evaluations after it up to
  one on card 8 (each on card 4, the draft's token rows, then on card 8). A step is a main pass and its draft passes.
- **Typical steps** as in S37: after the request's prompt passes, four draft passes, no stray evaluation, the verify
  pass within 12 % of the median GPU time, not among the request's first three steps (66, 66, 72 and 69 steps).
- **Added to S37's parts:** per card the host's gap before its entry (from the previous card's exit) and its host
  span; per draft pass its crossing (card 4's exit to card 8's entry); outside the kernels = the step less the GPU time
  of every evaluation in it.
- **The k-pool lines** (when the server log holds them): the lines between a request's `launch_slot_` and `release`
  lines are that request's, and the j-th of them is the request's j-th pass (main or draft) in the profile; a request
  whose counts differ is left unmatched.

## What ran

One deep-lane run on the rig, nothing else on the rig during the loads, 16:35:53 to 17:19:27 (43.6 minutes):

| Load | What | Loaded in | Requests |
|---|---|---:|---|
| `warm` | the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1`, no profile: the warm-up load, and the unprofiled step of the same requests | 309 s | the client's warm-up (64 ids, 32 tokens); at 16,384, 32,768 and 49,152 tokens a prime and two decodes of 256 greedy tokens |
| `prof` | the same under `GGML_CUDA_PROFILE=$OUT/prof-depth.csv GGML_CUDA_PROFILE_MODE=group` | 311 s | the same |

- **The line:** [release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env) exactly
  (`PRESET_ENV`, `PRESET_ARGS`, the MTP draft) with the sparse switch, through
  [bench/deep_lane](../../../bench/deep_lane/README.md) with its quiet check (no llama process and 0 MiB on every card
  before each load) and the RAM watchdog (floor 60 MiB; it never acted).
- **The prompts:** the deep lane's, from the release check's recorded agent requests (arm `t1swift`) with the same
  tail as R39a. Each prime read the chunk past the previous depth's checkpoint: 16,260 tokens on 60 cached, then 16,900 on
  15,804 and 16,900 on 32,188.
- **`PAUSE=4`:** the client waited 4 s before every request, so the trace falls into one piece per request at the
  idle gaps. It changes nothing the server times.
- **16,384 as well as the brief's 32,768 and 49,152**, because the 31 ms are counted from 16,384. The extra depth
  cost only its two decodes: the primes read the same chunks either way.
- **The prompts were profiled too.** The trace holds every evaluation of the load: 577 per card for the model, 2,256
  each on cards 4 and 8 for the draft. It is 85.8 MB raw and 10.4 MB gzipped, which the rig's disk holds easily.
  Under the profiler the primes read 62.0, 46.4 and 36.0 tok/s, against 121.7, 90.0 and 69.8 unprofiled.
- **The replies:** each load wrote its own greedy reply at each depth and repeated it on pass 2 (the same text and
  ids); the two loads' replies differ, as R39a's loads did from 8,192 on. The step's parts do not depend on the reply.
  The share of the draft each reply accepts does (`requests.csv`).
- **The cards** held 5,059, 5,095, 4,963, 5,111, 4,875, 4,979, 4,381, 4,991 and 4,565 MiB after loading, in both
  loads, the same as R39a's.
- **The host's memory:** the lowest MemAvailable was 230 MiB over the unprofiled load and 73 MiB over the profiled one,
  during its first prime and first decode at 16,384 ([host_memory.tsv](host_memory.tsv)).
- **The tables ran on a build machine's CPU** (`nice -n 19`, 11 s), never on the rig.

## How the step is cut

`run/depth_parts.py` (its header has the definitions; `--self-test`, 89 checks, all pass):
- **The draft's evaluations:** those with a launch group labelled `mtp_`. The j-th model evaluation on every card is
  main pass j; under this profiler the cards run one after another, and no pass is out of order.
- **The requests:** pauses of more than 3 s cut the trace into pieces: the load, then the client's warm-up, and per
  depth the prime, decode p1 and decode p2 (1, 13, 33, 72, 71, 34, 82, 81, 34, 79 and 77 main passes). The pieces are
  matched to `requests.csv` in order, and the script stops if the counts differ.
- **A step:** a main pass and the draft's evaluations before the next one. A typical step has four draft passes, no
  stray evaluation, a main pass within 12 % of the median, and is not among a request's first three steps. That keeps
  67, 67, 77, 77, 73 and 72 steps. At 49,152, one step of each request has a single draft pass.
- **The gather:** the `GET_ROWS` among the two launch groups before each `LIGHTNING_INDEXER`. In qwen4exp's
  `build_qsa_sel` this is the gather of all `n_pool` pooled keys by `pool_cells`. Its neighbour, the gather of the
  new pools' members, took 10.24 and 12.29 µs (raw) in the same verify pass.
- **The wall time** of a step runs from the end of the previous step's last evaluation to the end of its own last one.
  Outside the kernels is that wall time less the step's raw GPU time.
- **R49's trace** is cut the same way with `--shallow`: its pieces of ten or more main passes are its two requests.
  This gives R49's 81 and 80 verify passes and the same GPU time per verify pass.

## Files

| File | What it holds |
|---|---|
| [depth_parts.md](depth_parts.md) | the full report: the cut, the step, where the growth goes, by part per card group, the launch groups that grow most, per card |
| [buckets.tsv](buckets.tsv) | per request and card group (verify pass on cards 0 to 7, on card 8, the draft): ms a step in each bucket of "where the growth goes", and the est. outside the kernels |
| [parts.tsv](parts.tsv) | per request and card group: ms a step by R41's parts (`kpass_parts.py`) |
| [groups.tsv](groups.tsv) | per request and card group: every launch group's label (node numbers dropped), launch groups a step and ms a step |
| [per_card.tsv](per_card.tsv) | per request: ms a step per card, verify pass and draft |
| [totals.tsv](totals.tsv) | per request: the step's totals, wall time and profiler cost, the server's step profiled and unprofiled |
| [requests.tsv](requests.tsv) | the cut: every piece of both traces, its request, main passes, steps and typical steps |
| [steps.tsv](steps.tsv) | every main pass of every decode: kind, typical or not, draft passes, raw GPU time, wall time |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's lines per request and per load, both loads |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables of the profiled load (it skips the warm-up load) |
| [run/depth_parts.py](run/depth_parts.py) | the cut into requests and steps, the parts and buckets, the self-test |
| [host.csv.gz](host.csv.gz) | R64: the host profile of load `host`, whole: per card and evaluation the host's entry and exit (ms) and the GPU time (µs); 6,774 lines |
| [host_depth.md](host_depth.md) | R64: the host profile's cut, the step by part per decode (means and medians) and the growth from 16,384 to 49,152 |
| [host_steps.tsv](host_steps.tsv) | R64: every generation step of the four decodes: its parts, the GPU time, host gap and host span per card, each draft pass's lead, crossing, GPU time and turn |
| [host_summary.tsv](host_summary.tsv), [host_pieces.tsv](host_pieces.tsv) | R64: each part's mean and median per decode; the cut into pieces and requests |
| [host_memory_r64.tsv](host_memory_r64.tsv) | R64: the host's memory per load and request from the RAM watchdog |
| [r64_requests.csv](r64_requests.csv), [r64_loads.tsv](r64_loads.tsv), [r64_table.md](r64_table.md), [r64_classes.tsv](r64_classes.tsv) | R64: the deep lane's lines per request and per load, both loads; its own tables of the profiled load |
| [run/host_depth.py](run/host_depth.py) | R64: the host profile cut into requests and steps (with S37's `step_split.py`), the parts, the k-pool lines matched to passes, the self-test |

## Commands

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN, MODEL, MODEL_DRAFT: paths on the rig;
# WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests, proxy.jsonl.bodies.gz)
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse prof" ARM_ENV_prof="GGML_CUDA_PROFILE=$WORK/out/prof-depth.csv GGML_CUDA_PROFILE_MODE=group" \
  LOADS="warm:sparse prof:prof" DEPTHS="16384 32768 49152" WARM_DEPTHS="16384 32768 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
ssh $RIG "gzip -k -9 $WORK/out/prof-depth.csv"   # after the run; then copy $WORK/out off the rig

# on a build machine, CPU only, from this repository's root. TRACE: prof-depth.csv (85,836,524 B, md5
# cc4d9217ac7260f3a46c788eb092ee13; the .gz 10,432,467 B, md5 c46ff57244866019ef5dfb43a543d7f6); SHALLOW: R49's
# prof-verify.csv (md5 92a891f032d1aa9c6278f0e59acf34ce); OUT: the run's out folder; RES: the tables
R=docs/data
V="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/depth_parts.py \
  --verify-parts $R/2026-11-10_served_verify_parts/run/verify_parts.py --kpass-parts $R/2026-11-04_kpass_profile/run/kpass_parts.py"
$V --self-test
$V --event-us 7.6 --shallow $SHALLOW --shallow-label R49 --shallow-depth "79 to 335" \
  --deep $TRACE --requests $OUT/requests.csv --load prof --ref-load warm --out $RES > $RES/depth_parts.md
nice -n 19 python3 -I $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
```

R64, the host profile (add `ARM_ARGS_host="-lv 4"` to get the k-pool lines into the server log as well):

```bash
# from a machine that reaches the rig over ssh, as above
RIG=$RIG WORK=$WORK BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse host" ARM_ENV_host="LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/host.csv" \
  LOADS="warm:sparse host:host" DEPTHS="16384 49152" WARM_DEPTHS="16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only, from this repository's root. OUT: the collected folder (host.csv 246,402 B, md5
# 887a4c1275da4fa8488e80e9adc1145d, this folder's host.csv.gz gunzipped; requests.csv as r64_requests.csv); RES: the
# tables
R=docs/data
H="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py"
$H --self-test
$H --host $OUT/host.csv --requests $OUT/requests.csv --load host --ref-load warm --grow 16384:49152 \
  --out $RES > $RES/host_depth.md   # with the k-pool lines in the log: --log $OUT/server-host.log
nice -n 19 python3 -I $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
```

## Limits

- **One profiled load.** Each depth's two requests are two samples of every part of that load. They agree within
  0.04 ms in every bucket but the output heads (0.16, 0.28 and 0.29 ms higher on pass 1: the draft's full-head
  fallback passes, which R49 saw on its first request only), and their GPU time differs by 0.12, 0.24 and 0.26 ms.
- **What lies outside the kernels unprofiled is an estimate:** the unprofiled step of the load before, which wrote other
  replies, less the profiled GPU time. That assumes the profiler leaves the kernels' own time unchanged.
- **What grows outside the kernels is not measured.** The k-pool inputs are the candidate by their size (est.), and
  Q1243 measures them.
- **The unprofiled load was the run's first load.** The first load after an idle rig can run slow (Q1055), but its
  pass-2 steps match R39a's second sample on the other rig to within 1.69 ms.
- **The profiler's 7.6 µs a launch group is an estimate:** C140's figure, the one R41 and R49 used, 20.48 ms a step
  here. It is the same at every depth, so the growth does not depend on it.
- **R64: one profiled load,** every value one request. Which upload an interval holds is read from where the
  scheduler copies a card's inputs; that the growth is the k-pool inputs rests on the rate (the extra bytes, from the
  tensors' shapes, over each interval's growth gives 186.5, 187.7 and 185.0 MB/s for the verify pass's cards, the
  drafting passes and the catch-up pass, the link's speed), not on a count of bytes. The share of
  the verify pass's rest that is the uploads holding the links is an estimate, and the host's own build of the inputs
  is unmeasured (the k-pool lines need `-lv 4`).

## Provenance and licence

Measured on 2026-11-10 on one rig of nine P106-090 cards, with the release build of the Miner Llama fork (llama.cpp,
MIT). The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they and the replies are not part of this folder (`requests.csv` and `r64_requests.csv` hold only
checksums). R64's host profile is here whole (`host.csv.gz`); R51's group-mode trace (85.8 MB) is not.
