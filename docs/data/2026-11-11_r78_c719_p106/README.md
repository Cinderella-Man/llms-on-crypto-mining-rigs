# The QSA pool mask built on the cards, on a P106 (R78, Q1284, 2026-11-11)

What this is: the rig run of C719's switch `LLAMA_QSA_POOL_MASK_DEV=1`
([design and CPU check](../2026-11-11_c719_pool_mask_on_cards/README.md), [Q1274](../../../quants/experiments.md)),
which builds Qwen3.8-Flash-Next's QSA pool mask on every card from 8 bytes a token instead of uploading the F16 mask
(2 bytes a pool and token), on top of C100's `LLAMA_QSA_KPOOL_DEV=1`, which keeps the k-pool tables on the cards
([R74](../2026-11-11_r74_c100_p106/README.md)). One deep-lane run on nine P106-090 cards with the v4 night merge
(`pascal-p106-v4-b12246-night` `c408dabcb`, which holds both branches; every other switch off), at 0, 16,384 and
49,152 tokens: the control, C100's switch alone, both switches, both with both exactness checks, and both under the
host profile R64 used to find the uploads ([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md#outside-the-kernels-measured-r64)).
The recorded `RANGE_MASK` cases already passed on a card in the night build's native replay
([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md): suite `range-mask` 22 of 22, its fault run
failing as it must), so this run is the design's step 2. Registry [Q1284](../../../quants/experiments.md).

## Result

**The switch is exact on the cards and a gain at depth: on the repeated request the step is 1.27 ms shorter at
16,384 tokens and 1.83 ms shorter at 49,152 than with the k-pool tables alone (145.43 against 146.70, 172.04 against
173.87).** C719 estimated 1.00 and 2.96 ms. The estimate counted the verify pass's uploads to cards 1 to 8 as well,
and those do not hold up the token: with the mask built on the cards they are gone, and the verify pass's rest still
grows 4.48 ms from 16,384 to 49,152 tokens, as it did with them (4.44 in R74; [Q1356](../../../quants/experiments.md)).

- **Exact.** The `check` load compared every card's mask with the host mask after every micro-batch and found no
  difference: 561 micro-batches of the main model (5,049 masks, nine a micro-batch) and 2,196 of the MTP draft on
  card 8 (2,196 masks). C100's check in the same load compared the k-pool tables' copies over the same micro-batches,
  with no difference. Nothing aborted. All five loads wrote the same greedy reply (token ids' md5) on both passes at
  every depth: `08569d4a…` at 0, `41cfc791…` at 16,384 and `3a311cdd…` at 49,152, the same as R74's
  ([replies.tsv](replies.tsv)).
- **Nothing the k-pool inputs send grows with depth any more.** From 16,384 to 49,152 tokens on pass 2, under the
  host profile:
  - The host's gap before each of cards 1 to 8 in the verify pass grows 0.00 ms (0.23 to 0.32 ms at both depths),
    against 0.35 to 0.42 ms with the tables alone (R74's `hostdev`) and 1.22 to 1.23 ms with neither.
  - Each drafting pass's crossing to card 8 grows 0.00 ms (0.51 ms at both depths), against 0.09 ms with the tables
    alone; the catch-up's crossing 0.00 against 0.35.
  - A card receives 3.2 KiB a verify pass and 3.1 KiB a drafting pass at every depth, against 35.7 and 99.7 KiB a
    verify pass and 11.2 and 27.2 KiB a drafting pass at 16,384 and 49,152 with the tables alone.
- **The verdict on pass 2 is a gain at 16,384 and 49,152, and negligible at depth 0** ([verdict.tsv](verdict.tsv)).
  Pass 1, the first request at a depth, is negligible when R74's and R80's loads are counted: there the host swapped
  and the spreads are 31.10 and 38.87 ms.
- **The cards and the host gain memory.** With the mask built on the cards, the graph's compute buffers are 53.05 MiB
  smaller on cards 0, 2, 4, 5, 7 and 8 and 36.67 MiB smaller on cards 1, 3 and 6, and the pinned host buffer is
  64.23 MiB smaller for the model and 4.01 MiB for the draft. After loading, cards 0, 2, 4, 5, 7 and 8 hold 54 MiB less and cards 1, 3 and 6
  36 MiB less, and the host had 979 MiB available after loading against 868 with the tables alone.

### The step, ms

ms a step = the request's generation time over its steps, which are its tokens less the accepted draft tokens. One
request per cell. Every load wrote the same reply, so every load took the same steps: 78 and 75 at depth 0, 72 at
16,384 and 74 at 49,152.

| Depth | Pass | `warm` | `kpool` | `mask` | `hostmask` | `mask` − `kpool` | Δ, this run | spread | Δ, five base loads | spread | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 125.41 | 125.98 | 125.58 | 126.15 | −0.40 | −0.11 | 0.57 | −0.03 | 0.93 | negligible |
| 0 | 2 | 121.40 | 122.18 | 121.60 | 121.96 | −0.58 | −0.40 | 0.36 | −0.12 | 0.66 | negligible |
| 16,384 | 1 | 167.35 | 165.63 | 195.55 | 164.44 | +29.91 | +14.36 | 31.10 | +9.62 | 31.10 | negligible |
| 16,384 | 2 | 149.33 | 146.70 | 145.43 | 146.11 | −1.27 | −0.93 | 0.68 | −1.05 | 0.68 | **gain** |
| 49,152 | 1 | 226.26 | 208.38 | 206.63 | 207.07 | −1.74 | −1.53 | 0.43 | −15.85 | 38.87 | negligible |
| 49,152 | 2 | 180.38 | 173.87 | 172.04 | 172.82 | −1.83 | −1.44 | 0.78 | −1.61 | 0.78 | **gain** |

- **The class.** The switch's arm has two loads, `mask` and `hostmask` (both switches, without and with the host
  profile and `-lv 4`). The base arm is C100's switch alone: `kpool` in this run, and in the wider set also R74's
  `dev` and `hostdev` and R80's `dev` and `hostdev`, the same build, rig, model and settings earlier the same day
  ([R74](../2026-11-11_r74_c100_p106/steps.tsv), [R80](../2026-11-11_r80_c720_p106/steps.tsv)). Δ is the switch arm's
  mean less the base arm's. The spread is the largest difference between two loads of one arm; the switch arm's pair
  holds the profile's own cost, so it errs against a gain. The base loads by value at 16,384, pass 2: 146.70
  (`kpool`), 146.62 and 146.97 (R74), 146.77 and 147.05 (R80); at 49,152: 173.87, 173.97 and 174.15, 174.00 and
  174.24.
- **Depth 0.** This run alone puts the switch 0.40 ms below `kpool` on pass 2, 0.04 ms beyond its spread of 0.36;
  with the five base loads the difference is 0.12 ms against a spread of 0.66. At depth 0 the mask covers 64 to 128
  pools, about 1 KiB a card, so depth 0 is classed negligible.
- **Pass 1** at 16,384 in `mask`: the host moved 230 MiB to swap during that request
  ([host_memory.tsv](host_memory.tsv)), the first-request cost of Q1133. R74's and R80's pass-1 loads at 49,152
  swapped the same way.
- **The `check` load** (146.56, 174.26 and 207.47 ms on pass 2) synchronizes and reads every card's tables and masks
  back on every micro-batch. It is the exactness check, not a speed.
- **The profile and `-lv 4` cost** 0.68 and 0.78 ms a step at 16,384 and 49,152 (`hostmask` − `mask`).
- **Generation** on pass 2: 24.45 against 24.24 tok/s at 16,384 and 20.11 against 19.90 at 49,152 (`mask` against
  `kpool`). **Prompt reading:** unchanged; the primes read 131.6 and 82.3 tok/s in `kpool`, 132.0 and 84.5 in
  `mask`.

### Where the step changes (host profile, pass 2)

ms a step, the mean over each request's typical steps (67 at 16,384, 69 at 49,152). One request per column. `hostdev`
is R74's load with the tables alone (the same build, rig and settings, [host_depth_hostdev.md](../2026-11-11_r74_c100_p106/host_depth_hostdev.md));
`hostmask` is this run's ([host_depth_hostmask.md](host_depth_hostmask.md)):

| Pass 2 | `hostdev` 16,384 | `hostdev` 49,152 | growth | `hostmask` 16,384 | `hostmask` 49,152 | growth |
|---|---:|---:|---:|---:|---:|---:|
| the step | 149.36 | 176.10 | +26.74 | 148.52 | 174.64 | +26.13 |
| GPU time of every evaluation | 135.01 | 149.11 | +14.10 | 134.70 | 148.68 | +13.98 |
| outside the kernels | 14.34 | 26.99 | +12.64 | 13.82 | 25.97 | +12.15 |
| the verify pass's rest (its span less the nine cards' GPU time) | 6.10 | 10.54 | +4.44 | 6.04 | 10.53 | +4.48 |
| the draft's crossings, four passes | 2.37 | 2.99 | +0.62 | 2.06 | 2.06 | +0.00 |
| the draft's turns before its last pass | 1.69 | 4.49 | +2.80 | 1.67 | 4.35 | +2.68 |
| the server between the draft and the next verify pass | 4.07 | 8.85 | +4.78 | 3.93 | 8.91 | +4.98 |
| the host's launch of the verify pass (inside its span) | 5.22 | 8.13 | +2.91 | 3.78 | 3.80 | +0.02 |
| the host's gap before card 1 / card 8 | 0.48 / 0.50 | 0.83 / 0.84 | +0.35 / +0.35 | 0.29 / 0.32 | 0.29 / 0.32 | +0.00 / +0.00 |
| a drafting pass's crossing (each of three) | 0.56 | 0.65 | +0.09 | 0.51 | 0.51 | +0.00 |
| the catch-up pass's crossing | 0.70 | 1.05 | +0.35 | 0.52 | 0.52 | +0.00 |
| the k-pool inputs' `set_input`, µs a step | 222.79 | 583.96 | +361.17 | 209.40 | 547.20 | +337.80 |
| k-pool KiB a card, the verify pass | 35.7 | 99.7 | +64.0 | 3.2 | 3.2 | +0.0 |

- **Most of the saving on the token's path is the draft's crossings:** 0.31 ms at 16,384 and 0.93 ms at 49,152
  against R74's `hostdev`. Of the 1.46 ms the profiled step drops at 49,152, the rest is 0.43 ms of GPU time and
  0.14 ms of the draft's turns; the server's interval before the verify pass is 0.06 ms longer.
- **The verify pass's uploads to cards 1 to 8 are gone and the step does not notice them.** The host's launch of the
  verify pass shrinks from 8.13 to 3.80 ms at 49,152 and no longer grows. It runs ahead of the cards (their GPU time
  is 130.23 ms), so it is not on the token's path. C719's estimate scaled every card's mask bytes by R64's
  bytes-to-time line; at 16,384 the measured step matches it (−1.27 ms in this run's pair, −0.93 and −1.05 as arm
  means, est. −1.00) and at 49,152 it is about half (−1.83, −1.44 and −1.61, est. −2.96).
- **What still grows with depth outside the kernels:** the verify pass's rest (+4.48 ms), the server between the
  draft and the next verify pass (+4.98) and the draft's turns before its last pass (+2.68, of it the catch-up's
  turn +2.26). None of them is a k-pool upload now; [Q1356](../../../quants/experiments.md) (R109) splits them with
  C405's step trace at depth.
- **`set_input`** counts the k-pool inputs' host time: it fills the ranges (8 bytes a token) and the other k-pool
  inputs, and the host mask only for the check.

## What ran

One deep-lane run on one rig, nothing else on the rig during the loads, 17:59:22 to 19:22:50 (83.5 minutes). The loads
came up in 224, 224, 228, 225 and 226 s:

| Load | Arm | Start | Its own variables and flags |
|---|---|---|---|
| `warm` | the control | 17:59:25 | [the v3 preset](../../../release/flash-next-v3/settings.env) with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (R64's line) |
| `kpool` | C100's switch | 18:14:13 | + `LLAMA_QSA_KPOOL_DEV=1` |
| `mask` | both switches | 18:28:58 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1` |
| `check` | both, checked | 18:43:43 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 LLAMA_QSA_POOL_MASK_DEV_CHECK=1`, `-lv 4` |
| `hostmask` | both, host profile | 19:07:09 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostmask.csv`, `-lv 4` |

- **Each load:** the client's warm-up, then at 0, 16,384 and 49,152 tokens a prime and two decodes of 256 greedy
  tokens, with a 4 s pause before every request (`PAUSE=4`), so the host profile cuts into one piece per request.
  The primes read 16,320 tokens on none cached, then 33,284 on 15,804.
- **`warm` is a timed control at all three depths** (`WARM_DEPTHS="0 16384 49152"`, as in R74 and R80), so its depth-0
  reply is there to compare. The deep lane's own tables ([table.md](table.md)) skip it and take `kpool` as their
  control.
- **The build:** the night merge's rig tarball (md5 `736363d6ca12e6c81ff5f5b684ef31b3`) in place of C719's own
  tarball. Every other branch's switch is off in every load, and with every switch off this build computes what
  `pascal-p106-v4` does ([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)).
- **The switches on the rig:** the server log of `check` and `hostmask` shows the tables kept on nine cards for the
  model (16,448 rows, 2.87 MiB on the cards) and on one card for the draft (0.32 MiB), and the pool mask built on the
  cards for both. When each memory is freed it logs its counts: 561 micro-batches for the model with 1 full update
  and 31,084 of 3,074,944 table rows sent (55.4 a micro-batch), and 2,196 for the draft with 1 full and 87,929 of
  12,059,712 rows (40.0 a micro-batch), the mask built on the cards in every one ([kpool_lines.tsv](kpool_lines.tsv)).
- **The cards** held, after loading, 5,061, 5,097, 4,965, 5,113, 4,875, 4,981, 4,383, 4,993 and 4,647 MiB with the
  tables alone and 5,007, 5,061, 4,911, 5,077, 4,821, 4,927, 4,347, 4,939 and 4,593 MiB with both switches. Of
  6,144 MiB, the least free in any load with both switches (`mask`, `hostmask`) was 1,097, 1,043, 1,193, 1,027, 1,285,
  1,177, 1,757, 1,165 and 1,491 MiB, against 1,043, 1,007, 1,139, 991, 1,231, 1,123, 1,721, 1,111 and 1,437 in
  `kpool` (sampled every 10 s, [cards.tsv](cards.tsv)). The compute buffers come from the `-lv 4` logs: R74's
  `hostdev` 344.38 MiB on card 0, 281.00 on cards 1, 3 and 6 and 280.62 on the others, 123.51 MiB pinned on the host;
  `hostmask` 291.33, 244.33 and 227.57, 59.28 on the host.
- **The host:** the lowest MemAvailable over each load (the RAM watchdog's samples, every second) was 160 MiB in
  `warm`, 91 in `kpool`, 512 in `mask`, 157 in `check` and 147 in `hostmask`. In `mask` the kernel moved pages to swap
  (swap in use up to 1,042 MiB), which kept MemAvailable higher there. The watchdog (floor 60 MiB) never acted.
- **The tables ran on a build machine's CPU** (`nice -n 19`, seconds each), never on the rig.
- **The host profile is cut as in R74:** [fold_updates.py](../2026-11-11_r74_c100_p106/run/fold_updates.py) folds
  each draft pass's update split on card 8 into the embedding card's evaluation before `host_depth.py` cuts it. It
  folded 2,190 of the profile's 11,628 evaluations (R74's `hostdev`: 2,188 of 11,628).

## Files

| File | What it holds |
|---|---|
| [verdict.tsv](verdict.tsv) | per set (this run; with R74's and R80's base loads), depth and pass: `warm`, each base and switch load by value, `mask` − `kpool`, the arms' means, Δ, the spread and the class |
| [steps.tsv](steps.tsv) | per load, depth and pass: tokens, accepted of drafted, steps, generation ms, ms a step, tok/s, the reply's md5s |
| [replies.tsv](replies.tsv) | per depth, pass and load: the reply's token ids' and text's md5 |
| [kpool_lines.tsv](kpool_lines.tsv) | per server log: each k-pool and pool-mask line kind (allocation, the switch, both checks' counts, the counts at free, `LLAMA_QSA_INPUT_STATS`), its count, first and last line |
| [cards.tsv](cards.tsv) | per load and card: MiB used and free after loading, the most used over the load |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [host_depth_hostmask.md](host_depth_hostmask.md) | `host_depth.py`'s report on `hostmask`: the cut, the k-pool lines by micro-batch size, the step by part per decode (means and medians), the growth from 16,384 to 49,152 |
| `host_summary_hostmask.tsv`, `host_steps_hostmask.tsv`, `host_pieces_hostmask.tsv`, `qsa_hostmask.tsv` | the same: each part's mean and median per decode, every generation step's parts, the cut into pieces, every k-pool line |
| [hostmask.csv.gz](hostmask.csv.gz) | the host profile, whole (11,628 evaluations, before the fold; 414,192 B, md5 `6521f88a831dc312ad3917576299b031`) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's line per request and per load (the profile's path written as `$OUT`) |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables (control `kpool`; `warm` skipped) |
| [run/r78_tables.py](run/r78_tables.py) | the step, the verdict (with other runs' base loads), the replies, the k-pool and pool-mask lines and the memory from the deep lane's folder |

## Commands

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the night merge tarball's bin/ on the rig;
# MODEL, MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse kpool mask check hostmask" \
  ARM_ENV_kpool="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_mask="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1 LLAMA_QSA_POOL_MASK_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  ARM_ENV_hostmask="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_POOL_MASK_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostmask.csv" ARM_ARGS_hostmask="-lv 4" \
  LOADS="warm:sparse kpool:kpool mask:mask check:check hostmask:hostmask" DEPTHS="0 16384 49152" \
  WARM_DEPTHS="0 16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only, from this repository's root (OUT: the collected folder; RES: the tables)
R=docs/data
nice -n 19 python3 -I $R/2026-11-11_r78_c719_p106/run/r78_tables.py $OUT --out $RES \
  --ref R74dev=$R/2026-11-11_r74_c100_p106/steps.tsv:dev --ref R74hostdev=$R/2026-11-11_r74_c100_p106/steps.tsv:hostdev \
  --ref R80dev=$R/2026-11-11_r80_c720_p106/steps.tsv:dev --ref R80hostdev=$R/2026-11-11_r80_c720_p106/steps.tsv:hostdev \
  > $RES/r78.md
nice -n 19 python3 -I $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
nice -n 19 python3 -I $R/2026-11-11_r74_c100_p106/run/fold_updates.py $OUT/hostmask.csv $RES/hostmask.csv
mkdir -p $RES/hostmask
nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py \
  --host $RES/hostmask.csv --requests $OUT/requests.csv --load hostmask --ref-load warm --grow 16384:49152 \
  --log $OUT/server-hostmask.log --out $RES/hostmask > $RES/hostmask/host_depth.md
grep -a -E 'pool_mask_dev_check|kpool_dev_check|QSA pool mask built|k-pool tables on' $OUT/server-check.log  # no "differs" line
grep -a -E 'sched_reserve: +CUDA' $OUT/server-hostmask.log                                                 # the compute buffers
```

`r78_tables.py` gives R74's verdict on R74's folder with `--base warm,host --on dev,hostdev`.

## Limits

- **One load per arm and setting in this run.** The base arm has one load here (`kpool`); the wider set adds four
  loads of the same line from two earlier runs that day. Each value is one request.
- **The loads ran in one order** (`warm`, `kpool`, `mask`, `check`, `hostmask`).
- **The profile comparison pairs two runs** (R74's `hostdev` at 13:20, this run's `hostmask` at 19:07) on the same
  build, rig and settings.
- **What still grows in the verify pass's rest, the server's interval and the catch-up's turn is not split here**
  (Q1356).

## Provenance and licence

Measured on 2026-11-11 on one rig of nine P106-090 cards, with the Miner Llama fork's v4 night merge (llama.cpp, MIT).
The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they and the replies are not part of this folder (`requests.csv` and the tables hold only checksums).
