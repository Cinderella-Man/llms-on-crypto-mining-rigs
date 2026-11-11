# The small k-pool inputs in the tables' update, on a P106 (R80, Q1293, 2026-11-11)

What this is: the rig run of C720's switch `LLAMA_QSA_KPOOL_STAGE=1`
([design and CPU check](../2026-11-11_c720_kpool_small_inputs/README.md), [Q1275](../../../quants/experiments.md)).
It works on top of C100's `LLAMA_QSA_KPOOL_DEV=1` ([R74's rig run](../2026-11-11_r74_c100_p106/README.md)). With C100
on, each card that runs a QSA layer gets seven copies a micro-batch: the tables' update, the pool mask, and five small
inputs (`k_idxs`, `tail_idxs`, `new_pool_idxs`, `new_pool_rep`, `new_pool_pos`). The switch puts the five into the
update's copy and casts them back on the card, so the card gets two copies instead of seven.

One deep-lane run on nine P106-090 cards with the v4 night merge (`pascal-p106-v4-b12246-night` `c408dabcb`, which
holds C720's branch; every other switch off), at 0, 16,384 and 49,152 tokens. It had six loads: the preset's control,
C100 alone, C100 with the stage, the stage with C100's exactness check, and C100 alone and with the stage again under
the host profile that R64 used to time each card's inputs
([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md#outside-the-kernels-measured-r64)). Registry
[Q1293](../../../quants/experiments.md).

## Result

**The switch is exact on the cards and a small gain at depth: on the repeated request the step is 0.48 ms shorter at
16,384 tokens and 0.32 ms shorter at 49,152 (the mean of two loads each), against spreads of 0.42 and 0.24 ms.** On
its own: 146.22 against 146.77 ms at 16,384 and 173.89 against 174.00 at 49,152. Under the host profile: 146.64
against 147.05 and 173.71 against 174.24. Each card's five copies cost 0.07 ms of the host's time, a card and a pass.

- **Exact.** The `check` load ran C100's check with the stage on. It compared every micro-batch's copies with the
  host's tables and every other k-pool input byte for byte. It found no difference in 561 micro-batches of the main
  model on all nine cards and 2,196 of the MTP draft on card 8, and nothing aborted. Its log has the stage switch's
  line once for each memory (model and draft).
- **The same replies.** All six loads wrote the same greedy reply (token ids' md5) on both passes at every depth:
  `08569d4a…` at 0, `41cfc791…` at 16,384 and `3a311cdd…` at 49,152 ([replies.tsv](replies.tsv)). These are R74's
  three replies too.
- **Each copy has a fixed cost, and it is gone.** On pass 2 under the host profile, with the stage against without:
  - The host's gap before each of cards 1 to 8 in the verify pass is 0.07 ms shorter at every depth (card 4's 0.06,
    cards 2 and 3 0.08 at 49,152). For example, card 1's is 0.41 against 0.48 ms at 16,384 and 0.75 against 0.83 at 49,152.
  - Each of the three drafting passes' crossings to card 8 is 0.07 ms shorter at every depth (0.49 against 0.56 ms at
    16,384, 0.58 against 0.65 at 49,152). The catch-up pass's crossing is 0.07 and 0.05 ms shorter.
  - The host's launch of the verify pass is 0.51, 0.53 and 0.54 ms shorter at 0, 16,384 and 49,152. That is about eight cards'
    gaps, and it runs ahead of the cards.
  - None of these change with depth: each shrinks by the same amount at 0, 16,384 and 49,152 tokens, and each part
    grows from 16,384 to 49,152 tokens as it did without the switch. So each of the five copies cost est. 14 µs (0.07 ms
    over five), its fixed cost, not its bytes.
- **What reaches the step is the draft's four crossings.** They take 0.29 and 0.27 ms less a step at 16,384 and
  49,152, and the profiled step is 0.38 and 0.28 ms shorter. The verify pass's rest is the same (6.13 against 6.17,
  10.59 against 10.55 ms). The host still builds the five inputs, so the k-pool inputs' `set_input` stays at 224.91
  against 219.54 µs a step at 16,384 and 582.94 against 584.74 at 49,152.
- **The verdict on pass 2 is a gain at 16,384 and 49,152 and negligible at depth 0.** It is classed on two pairs of
  loads ([verdict.tsv](verdict.tsv)). Pass 1, the first request at a depth, is negligible at every depth: there the
  host swapped in `hostdev` (spreads 7.14 and 12.49 ms).
- **Memory: +14.90 MiB of compute buffer on cards 1, 3 and 6.** With the stage, those three cards' compute buffers
  grow from 281.00 to 295.90 MiB, while the other six shrink by 0.26 MiB. The draft's buffer on card 8 grows from 57.15
  to 58.40 MiB, and the host's compute buffers grow from 123.51 to 125.80 MiB and from 10.02 to 10.59. Card 3's least
  free over the run went from 991 MiB without the stage to 977 with it, the run's lowest. New
  [Q1385](../../../quants/experiments.md) looks for what keeps that memory.
- **The host's lowest MemAvailable** was 183 MiB in `warm`, 84 in `dev`, 81 in `stage`, 178 in `check`, 245 in
  `hostdev` and 84 in `hoststage`. The loads that swapped at 49,152 (`warm`, `check`, `hostdev`, up to 810 MiB in
  swap) have the higher minima. The RAM watchdog (floor 60 MiB) never acted.

### The step, ms

ms a step = the request's generation time over its steps, which are its tokens less the accepted draft tokens. One
request per cell. Every load wrote the same reply, so every load took the same steps: 78 and 75 at depth 0, 72 at
16,384 and 74 at 49,152.

| Depth | Pass | `warm` | `dev` | `stage` | `stage` − `dev` | `hostdev` | `hoststage` | `hoststage` − `hostdev` | spread | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 125.46 | 125.89 | 125.60 | −0.29 | 126.31 | 126.72 | +0.41 | 1.12 | negligible |
| 0 | 2 | 121.42 | 121.52 | 121.54 | +0.02 | 122.15 | 122.01 | −0.14 | 0.63 | negligible |
| 16,384 | 1 | 167.38 | 165.40 | 165.18 | −0.23 | 172.55 | 165.44 | −7.11 | 7.14 | negligible |
| 16,384 | 2 | 148.57 | 146.77 | 146.22 | −0.55 | 147.05 | 146.64 | −0.41 | 0.42 | **gain** |
| 49,152 | 1 | 225.44 | 208.81 | 208.81 | +0.01 | 221.29 | 208.64 | −12.66 | 12.49 | negligible |
| 49,152 | 2 | 180.37 | 174.00 | 173.89 | −0.11 | 174.24 | 173.71 | −0.53 | 0.24 | **gain** |

- **The class:** each arm has two loads, one unprofiled and one under the host profile: C100 alone (`dev`,
  `hostdev`) and C100 with the stage (`stage`, `hoststage`). Each arm's mean is taken over its two loads. The spread
  is the larger of the two arms' differences between their loads; it holds the profile's own cost. The stage's mean is
  0.48 ms below C100's at 16,384 and 0.32 ms below at 49,152 (−0.06 at depth 0).
- **`warm`** (the preset with `GGML_CUDA_FATTN_SPARSE_PASCAL=1`, no k-pool switch) is the warm-up load and R74's
  control. The deep lane's own tables skip it, but it ran at all three depths, so its replies are there to compare. Against it, C100 alone repeats R74: 146.77 against 148.57 ms at 16,384 and 174.00 against 180.37 at 49,152.
- **The `check` load** (144.07, 168.79 and 197.98 ms on pass 2) synchronizes and reads every card's copy back on every
  micro-batch. It is the exactness check, not a speed.
- **Pass 1** at 16,384 and 49,152 in `hostdev`: the host moved 68 MiB to swap during the 16,384 request and 610 MiB
  during the 49,152 prime ([host_memory.tsv](host_memory.tsv)). This is the first-request cost of Q1133.
- **Generation** on pass 2: 24.32 against 24.22 tok/s at 16,384 and 19.89 against 19.88 at 49,152 (`stage` against
  `dev`). **Prompt reading:** the primes read 128.3 and 82.6 tok/s in `stage` and 130.4 and 82.6 in `dev`; under the
  profile 115.0 and 73.7 in `hoststage` and 114.9 and 73.6 in `hostdev`. That is one prime each, and the switch does
  not touch a 512-token micro-batch's work beyond its casts.

### Where the step changes (host profile, pass 2)

ms a step, the mean over each request's typical steps (71 at depth 0, 67 at 16,384, 69 at 49,152). One request per
column ([host_pair_p2.tsv](host_pair_p2.tsv), from [host_depth_hostdev.md](host_depth_hostdev.md) and
[host_depth_hoststage.md](host_depth_hoststage.md)):

| Pass 2 | `hostdev` 0 | `hoststage` 0 | `hostdev` 16,384 | `hoststage` 16,384 | diff | `hostdev` 49,152 | `hoststage` 49,152 | diff |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| the step | 123.64 | 123.47 | 149.59 | 149.22 | −0.38 | 176.16 | 175.89 | −0.28 |
| GPU time of every evaluation | 113.07 | 113.28 | 135.11 | 135.05 | −0.06 | 149.33 | 149.36 | +0.03 |
| outside the kernels | 10.56 | 10.19 | 14.48 | 14.16 | −0.32 | 26.83 | 26.53 | −0.30 |
| the verify pass's rest (its span less the nine cards' GPU time) | 4.26 | 4.34 | 6.17 | 6.13 | −0.05 | 10.55 | 10.59 | +0.04 |
| the host's launch of the verify pass (inside its span) | 4.70 | 4.19 | 5.21 | 4.68 | −0.53 | 7.99 | 7.45 | −0.54 |
| the host's gap before card 1 | 0.30 | 0.22 | 0.48 | 0.41 | −0.07 | 0.83 | 0.75 | −0.07 |
| the host's gap before card 4 | 0.24 | 0.18 | 0.40 | 0.34 | −0.06 | 0.73 | 0.67 | −0.06 |
| the host's gap before card 8 | 0.32 | 0.25 | 0.50 | 0.43 | −0.07 | 0.84 | 0.78 | −0.07 |
| the catch-up pass's crossing to card 8 | 0.53 | 0.46 | 0.70 | 0.63 | −0.07 | 1.05 | 0.99 | −0.05 |
| a drafting pass's crossing to card 8 (each of three) | 0.52, 0.51, 0.51 | 0.44, 0.44, 0.44 | 0.56, 0.56, 0.56 | 0.49, 0.48, 0.48 | −0.07 | 0.65, 0.65, 0.65 | 0.58, 0.57, 0.57 | −0.07 |
| the draft's crossings, four passes | 2.08 | 1.79 | 2.37 | 2.08 | −0.29 | 2.99 | 2.72 | −0.27 |
| the draft's turns before its last pass | 1.42 | 1.41 | 1.75 | 1.68 | −0.08 | 4.31 | 4.37 | +0.06 |
| the server between the draft and the next verify pass | 2.70 | 2.55 | 4.07 | 4.17 | +0.10 | 8.86 | 8.74 | −0.12 |
| the k-pool inputs' `set_input`, µs a step | 41.28 | 43.55 | 219.54 | 224.91 | +5.37 | 584.74 | 582.94 | −1.80 |
| the k-pool inputs, KiB a card a verify pass | 3.82 | 3.72 | 35.70 | 35.60 | −0.10 | 99.70 | 99.60 | −0.10 |

- **Cards 2, 3, 5, 6 and 7** lose 0.07 ms each too (cards 2 and 3 0.08 at 49,152) ([host_pair_p2.tsv](host_pair_p2.tsv) has every
  card). From 16,384 to 49,152 tokens each card's gap grows by 0.35 ms in both loads (card 4's 0.33, card 3's 0.36
  in `hostdev`): the pool mask, which
  still goes up whole (C719, [Q1274](../../../quants/experiments.md)).
- **Bytes:** the five held 160 B a card for a four-token verify pass and 136 B in the update; the profile's KiB a card
  falls by 0.10. At the link's ~187 MB/s that is under 1 µs; the 0.07 ms is the five copies' launches and waits.
- **The draft's four crossings are on the token's path**, so their −0.27 to −0.29 ms reaches the step. The verify
  pass's launch runs ahead of the cards (their GPU time is 130.82 ms at 49,152), so its −0.54 ms does not.
- **The next copy:** C733 ([Q1294](../../../quants/experiments.md)) would put the pool mask's ranges (C719) into the
  same update. At this run's ~14 µs a copy that is est. 0.06 ms a step on the draft's path, plus the verify pass's
  card gaps.

## What ran

One deep-lane run on one rig, nothing else on the rig during the loads, 16:16:51 to 17:56:18 (99.4 minutes). Each
load came up in 225, 227, 227, 227, 227 and 224 s, in order:

| Load | Arm | Start | Minutes | Its own variables and flags |
|---|---|---|---:|---|
| `warm` | the preset's control | 16:16:51 | 14.8 | [the v3 preset](../../../release/flash-next-v3/settings.env) with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (R64's line) |
| `dev` | C100 | 16:31:40 | 14.8 | + `LLAMA_QSA_KPOOL_DEV=1` |
| `stage` | C100 and C720 | 16:46:27 | 14.8 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| `check` | the stage, checked | 17:01:16 | 23.2 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |
| `hostdev` | C100, host profile | 17:24:29 | 16.0 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostdev.csv`, `-lv 4` |
| `hoststage` | C100 and C720, host profile | 17:40:27 | 15.8 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hoststage.csv`, `-lv 4` |

- **Each load:** the client's warm-up, then at 0, 16,384 and 49,152 tokens a prime and two decodes of 256 greedy
  tokens, with a 4 s pause before every request (`PAUSE=4`), so the host profile cuts into one piece per request.
  The primes read 16,320 tokens on none cached, then 33,284 on 15,804.
- **`warm` ran at all three depths** (`WARM_DEPTHS="0 16384 49152"`, as in R74, where the design said
  `"16384 49152"`), so its depth-0 reply is there to compare. It cost two short decodes. The deep lane's own tables
  ([table.md](table.md)) skip it and take `dev` as their control.
- **The build:** the night merge's rig tarball (md5 `736363d6ca12e6c81ff5f5b684ef31b3`) in place of C720's own
  tarball. Every other branch's switch is off in every load. With every switch off this build computes what
  `pascal-p106-v4` does ([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)).
- **The switch on the rig:** the server logs of `check` and `hoststage` show the stage switch's line for the model and
  for the draft, and the tables kept on nine cards for the model (16,448 rows, 2.87 MiB on the cards, 0.31 MiB on the
  host) and on one card for the draft (0.32 MiB). When each memory is freed it logs its updates. The model had 561
  micro-batches with 1 full update and 31,084 of 3,074,944 table rows sent (55.4 a micro-batch). The draft had 2,196
  micro-batches with 1 full update and 87,929 of 12,059,712 rows (40.0 a micro-batch). These are R74's counts
  ([kpool_lines.tsv](kpool_lines.tsv)).
- **The cards** held 5,061, 5,111, 4,965, 5,127, 4,875, 4,981, 4,397, 4,993 and 4,649 MiB after loading with the
  stage, and 5,061, 5,097, 4,965, 5,113, 4,875, 4,981, 4,383, 4,993 and 4,647 MiB with C100 alone. Of 6,144 MiB, the
  least free in any load was 1,043, 993, 1,139, 977, 1,229, 1,123, 1,707, 1,111 and 1,435 MiB (sampled every 10 s,
  [cards.tsv](cards.tsv)). Card 8's figure includes the night build's 80.04 MiB `h_nextn` buffer (R57).
- **The tables ran on a build machine's CPU** (`nice -n 19`, seconds each), never on the rig. `host_depth.py` read
  both profiles after R74's [fold_updates.py](../2026-11-11_r74_c100_p106/run/fold_updates.py), since both loads
  have the draft's extra update split on card 8 (R74, "How the host profile is cut with the switch").

## Files

| File | What it holds |
|---|---|
| [verdict.tsv](verdict.tsv) | per depth and pass: the step of the four speed loads, the two differences, the arms' means, the spread and the class |
| [steps.tsv](steps.tsv) | per load, depth and pass: tokens, accepted of drafted, steps, generation ms, ms a step, tok/s, the reply's md5s |
| [replies.tsv](replies.tsv) | per depth, pass and load: the reply's token ids' and text's md5 |
| [kpool_lines.tsv](kpool_lines.tsv) | per server log: each k-pool line kind (the stage switch's line, allocation, the check's counts, the counts at free, `LLAMA_QSA_INPUT_STATS`), its count, first and last line |
| [cards.tsv](cards.tsv) | per load and card: MiB used and free after loading, the most used over the load |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [host_pair_p2.tsv](host_pair_p2.tsv), [host_pair_p1.tsv](host_pair_p1.tsv) | every part of the step, `hostdev` against `hoststage` per depth, and each one's growth from 16,384 to 49,152 |
| [host_depth_hostdev.md](host_depth_hostdev.md), [host_depth_hoststage.md](host_depth_hoststage.md) | `host_depth.py`'s report per profiled load: the cut, the k-pool lines by micro-batch size, the step by part per decode (means and medians), the growth from 16,384 to 49,152 |
| `host_summary_*.tsv`, `host_steps_*.tsv`, `host_pieces_*.tsv`, `qsa_*.tsv` | the same per load: each part's mean and median per decode, every generation step's parts, the cut into pieces, every k-pool line |
| [hostdev.csv.gz](hostdev.csv.gz), [hoststage.csv.gz](hoststage.csv.gz) | the host profiles, whole (11,628 evaluations each, before the fold) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's line per request and per load (profile paths written as `$OUT`) |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables (control `dev`; `warm` skipped) |
| [run/r80_tables.py](run/r80_tables.py) | the step, the verdict, the replies, the k-pool lines and the memory from the deep lane's folder (R74's `r74_tables.py` with this run's loads) |
| [run/host_pair.py](run/host_pair.py) | two profiled loads' parts side by side from `host_depth.py`'s summaries |

## Commands

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the night merge tarball's bin/ on the rig;
# MODEL, MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse dev stage check hostdev hoststage" \
  ARM_ENV_dev="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_stage="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  ARM_ENV_hostdev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdev.csv" ARM_ARGS_hostdev="-lv 4" \
  ARM_ENV_hoststage="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hoststage.csv" ARM_ARGS_hoststage="-lv 4" \
  LOADS="warm:sparse dev:dev stage:stage check:check hostdev:hostdev hoststage:hoststage" DEPTHS="0 16384 49152" \
  WARM_DEPTHS="0 16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only, from this repository's root (OUT: the collected folder; RES: the tables)
R=docs/data
D=$R/2026-11-11_r80_c720_p106/run
nice -n 19 python3 -I $D/r80_tables.py $OUT --out $RES > $RES/r80.md
nice -n 19 python3 -I $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
H="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py"
for l in hostdev hoststage; do
  nice -n 19 python3 -I $R/2026-11-11_r74_c100_p106/run/fold_updates.py $OUT/$l.csv $RES/$l.csv
  mkdir -p $RES/$l
  $H --host $RES/$l.csv --requests $OUT/requests.csv --load $l --ref-load warm --grow 16384:49152 \
    --log $OUT/server-$l.log --out $RES/$l > $RES/$l/host_depth.md
done
for p in p1 p2; do
  nice -n 19 python3 -I $D/host_pair.py $RES/hostdev/summary.tsv $RES/hoststage/summary.tsv \
    --names hostdev,hoststage --pass $p --tsv $RES/host_pair_$p.tsv > $RES/host_pair_$p.md
done
grep -a -E 'kpool_dev_check|k-pool tables|go up in the k-pool' $OUT/server-check.log   # the stage line, the counts
grep -a -c -i 'differ' $OUT/server-check.log                                          # 0
grep -a 'compute buffer size' $OUT/server-hostdev.log $OUT/server-hoststage.log         # the cards' buffers
```

## Limits

- **One load per arm and setting.** The class pairs an unprofiled and a profiled load per arm. Their difference
  includes the profile's cost, which makes the spread conservative. Each value is one request. The gain is small: at
  16,384 tokens the stage's mean is below C100's by 0.48 ms against a spread of 0.42.
- **The loads ran in one order** (`warm`, `dev`, `stage`, `check`, `hostdev`, `hoststage`), each arm's profiled load
  after its unprofiled one.
- **Why cards 1, 3 and 6 keep 14.90 MiB more is not found here** ([Q1385](../../../quants/experiments.md)). The
  scheduler's reserve sizes each card's compute buffer from the largest graph, a 512-token micro-batch's. Which tensor
  lives longer there with the stage is not read from these logs.
- **The host's work is unchanged by design.** The host still builds the five inputs and then copies them into the
  update. Only the copies to the cards go.

## Provenance and licence

Measured on 2026-11-11 on one rig of nine P106-090 cards, with the Miner Llama fork's v4 night merge (llama.cpp, MIT).
The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they and the replies are not part of this folder (`requests.csv` and the tables hold only checksums).
