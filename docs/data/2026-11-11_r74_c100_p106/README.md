# The k-pool tables on the cards, on a P106 (R74, Q1273, 2026-11-11)

What this is: the rig run of C100's switch `LLAMA_QSA_KPOOL_DEV=1`
([design and CPU check](../2026-11-11_c100_kpool_on_cards/README.md), [Q640](../../../quants/experiments.md)), which
keeps Qwen3.8-Flash-Next's QSA k-pool tables (`pool_cells`, `pool_idxs`) on every card that runs a QSA layer and sends
each micro-batch only the rows that changed. One deep-lane run on nine P106-090 cards with the v4 night merge
(`pascal-p106-v4-b12246-night` `c408dabcb`, which holds C100's branch; every other switch off), at 0, 16,384 and
49,152 tokens: the control, the switch, the switch with its exactness check, and the two again under the host profile
that R64 used to find the tables' uploads ([2026-11-10_step_parts_depth](../2026-11-10_step_parts_depth/README.md#outside-the-kernels-measured-r64)).
Then `test-backend-ops -o SET_ROWS` on one card. Registry [Q1273](../../../quants/experiments.md); this run also
stands in for R71's load ([Q1255](../../../quants/experiments.md)).

## Result

**The switch is exact on the cards and a gain at depth: the step on the repeated request is 1.59 ms shorter at
16,384 tokens and 5.99 ms shorter at 49,152 (146.62 against 148.21, 173.97 against 179.96), and 1.83 and 6.17 ms under
the host profile.** R64 estimated 2.70 and 12.43 ms. The difference is the verify pass's rest, which grows 4.44 ms from
16,384 to 49,152 tokens with the tables on the cards and 4.41 without, so R64's reading that the uploads caused it
was wrong (new [Q1356](../../../quants/experiments.md)).

- **Exact.** The `check` load compared every micro-batch's copies with the host path's tables and found no
  difference: 561 micro-batches of the main model on all nine cards and 2,196 of the MTP draft on card 8. It also
  compared every other k-pool input byte for byte, and nothing aborted. All five loads wrote the same greedy reply
  (token ids' md5) on both passes at every depth: `08569d4a…` at 0, `41cfc791…` at 16,384 and `3a311cdd…` at 49,152
  ([replies.tsv](replies.tsv)).
- **The uploads stop growing with the tables.** From 16,384 to 49,152 tokens on pass 2:
  - The host's gap before each of cards 1 to 8 in the verify pass grows 0.35 to 0.42 ms with the switch, against 1.22
    to 1.23 ms without (R64: 1.23).
  - Each drafting pass's crossing to card 8 grows 0.09 ms against 0.96 (R64: 0.96), and the catch-up's 0.35 against
    1.22.
  - What still grows is the pool mask, which still goes up whole: 64 KiB more a card for a four-token pass and 16 KiB
    for one token, at the link's ~187 MB/s ([Q1274](../../../quants/experiments.md), C719 builds it on the cards). The
    k-pool bytes a card receives drop from 113.9 and 337.9 KiB a verify pass at 16,384 and 49,152 to 35.7 and
    99.7 KiB, and a drafting pass's from 89.4 and 265.4 KiB to 11.2 and 27.2 KiB.
- **The verdict on pass 2 is a gain at 16,384 and 49,152 and negligible at depth 0.** It is classed on two pairs
  of loads ([verdict.tsv](verdict.tsv)). Pass 1, the first request at a depth, is negligible: there the host swapped
  and the spreads are 17.10 and 19.45 ms.
- **Memory.** With the switch the cards hold 2 MiB more on cards 0 to 3 and 5 to 7 after loading, 0 on cards 4 and 8,
  and the same peak in every load. The host's lowest MemAvailable was 173 MiB in `warm`, 257 in `dev`, 204 in `host`
  and 194 in `hostdev`. The `check` load went down to 91 MiB. The RAM watchdog (floor 60 MiB) never acted.
- **CUDA's `SET_ROWS` with an I32 destination is right on a P106:** `test-backend-ops -o SET_ROWS` on card 0 passed
  162 of 162 cases, the three I32 cases among them. The other 728 cases are types CUDA does not take
  ([setrows.tsv](setrows.tsv)).
- **R71's question:** the k-pool inputs' `set_input` takes 363.43 µs a step on the host at 16,384 tokens and est.
  994 at 49,152. That is est. 0.63 ms of the est. 5.93 ms of host work that grows in the turns (R64). With the
  switch it takes 222.79 and 583.96 µs. `-lv 4` costs nothing measurable: `host` (the profile and `-lv 4`) against
  `warm` is +0.59 and +0.35 ms a step on pass 2, where R64 measured +1.11 and +0.61 for the profile alone.

### The step, ms

ms a step = the request's generation time over its steps, which are its tokens less the accepted draft tokens. One
request per cell. Every load wrote the same reply, so every load took the same steps: 78 and 75 at depth 0, 72 at
16,384 and 74 at 49,152.

| Depth | Pass | `warm` | `dev` | `dev` − `warm` | `host` | `hostdev` | `hostdev` − `host` | spread | class |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 0 | 1 | 125.34 | 125.38 | +0.04 | 125.67 | 125.94 | +0.27 | 0.57 | negligible |
| 0 | 2 | 124.43 | 121.66 | −2.77 | 122.04 | 121.97 | −0.07 | 2.40 | negligible |
| 16,384 | 1 | 166.84 | 182.70 | +15.86 | 168.12 | 165.60 | −2.53 | 17.10 | negligible |
| 16,384 | 2 | 148.21 | 146.62 | −1.59 | 148.80 | 146.97 | −1.83 | 0.59 | **gain** |
| 49,152 | 1 | 225.22 | 227.80 | +2.59 | 227.48 | 247.25 | +19.77 | 19.45 | negligible |
| 49,152 | 2 | 179.96 | 173.97 | −5.99 | 180.31 | 174.15 | −6.17 | 0.35 | **gain** |

- **The class:** each arm has two loads, one unprofiled and one under the host profile: the control `warm` and
  `host`, the switch `dev` and `hostdev`. Each arm's mean is taken over its two loads. The spread is the larger of
  the two arms' differences between their loads; it holds the profile's own cost, so it errs against a gain. The
  switch's mean is 1.71 ms below the control's at 16,384 and 6.08 ms below at 49,152.
- **The `check` load** (143.97, 169.16 and 198.21 ms on pass 2) synchronizes and reads every card's copy back on every
  micro-batch. It is the exactness check, not a speed.
- **Pass 1** at 16,384 in `dev` and at 49,152 in `hostdev`: the host moved 117 and 157 MiB to swap during those
  requests ([host_memory.tsv](host_memory.tsv)), the first-request cost of Q1133.
- **Generation** on pass 2: 24.25 against 23.99 tok/s at 16,384 and 19.89 against 19.22 at 49,152 (`dev` against
  `warm`). **Prompt reading:** unchanged. The primes read 131.9 and 82.9 tok/s in `warm`, 131.6 and 82.9 in `dev`.

### Where the step changes (host profile, pass 2)

ms a step, the mean over each request's typical steps (67 at 16,384, 69 at 49,152). One request per column
([host_depth_host.md](host_depth_host.md), [host_depth_hostdev.md](host_depth_hostdev.md)):

| Pass 2 | `host` 16,384 | `host` 49,152 | growth | `hostdev` 16,384 | `hostdev` 49,152 | growth |
|---|---:|---:|---:|---:|---:|---:|
| the step | 151.27 | 182.36 | +31.09 | 149.36 | 176.10 | +26.74 |
| GPU time of every evaluation | 134.90 | 149.60 | +14.71 | 135.01 | 149.11 | +14.10 |
| outside the kernels | 16.37 | 32.76 | +16.38 | 14.34 | 26.99 | +12.64 |
| the verify pass's rest (its span less the nine cards' GPU time) | 6.01 | 10.42 | +4.41 | 6.10 | 10.54 | +4.44 |
| the draft's crossings, four passes | 4.30 | 8.39 | +4.09 | 2.37 | 2.99 | +0.62 |
| the draft's turns before its last pass | 1.56 | 3.86 | +2.30 | 1.69 | 4.49 | +2.80 |
| the server between the draft and the next verify pass | 4.49 | 10.07 | +5.58 | 4.07 | 8.85 | +4.78 |
| the host's launch of the verify pass (inside its span) | 8.74 | 18.68 | +9.94 | 5.22 | 8.13 | +2.91 |
| the host's gap before card 1 / card 8 | 0.92 / 0.94 | 2.14 / 2.17 | +1.22 / +1.23 | 0.48 / 0.50 | 0.83 / 0.84 | +0.35 / +0.35 |
| a drafting pass's crossing (each of three) | 1.04 | 2.00 | +0.96 | 0.56 | 0.65 | +0.09 |
| the catch-up pass's crossing | 1.18 | 2.40 | +1.22 | 0.70 | 1.05 | +0.35 |
| the k-pool inputs' `set_input`, µs a step | 363.43 | est. 994 | est. +631 | 222.79 | 583.96 | +361.17 |

- **The saving is the draft's crossings and the server's interval before the verify pass.** At 49,152 the crossings
  take 5.40 ms less and the server's interval 1.22 ms less. That is card 0's upload, est. 1.23 in R64.
- **The verify pass's rest grows the same with or without the switch.** R64 put est. 4.41 ms of its growth on the
  uploads to cards 1 to 8 holding the switches' shared links while the crossings need them. With the tables on the
  cards those uploads shrink to the mask, and the rest still grows 4.44 ms, so something else grows there:
  [Q1356](../../../quants/experiments.md) (R109) splits it with C405's step trace at depth.
- **The host's launch** shrinks from 18.68 to 8.13 ms at 49,152. It runs ahead of the cards (their GPU time is
  130.43 ms), so it is not on the token's path.
- **Each draft pass evaluates once more on card 8 with the switch:** a split before the embedding card's, which
  writes the update into the copy (30 to 60 µs of GPU time). Counting from the update's entry, a drafting pass's lead
  is 0.08 ms longer than its crossing, against 0.02 to 0.04 without the switch. The draft's turns before its last pass grow
  0.50 ms more from 16,384 to 49,152 with the switch than without. [Q1357](../../../quants/experiments.md) (C771)
  folds the update into the draft layers' split.
- **`set_input`** counts the k-pool input's own host time, which still builds the mask. With the switch, the rows'
  update is built in the memory's `apply()`, which this line does not count. The 49,152 value of `host` is from the
  per-micro-batch medians: 206 µs for each of the two four-token passes and 194 for each of the three one-token passes.
  Its 367 k-pool lines did not match the 366 passes of the profile one to one.

## What ran

One deep-lane run on one rig, nothing else on the rig during the loads, 12:12:22 to 13:36:47 (84.4 minutes), then
`test-backend-ops` on card 0, 13:36:58 to 13:37:01. Each load came up in 224 to 226 s:

| Load | Arm | Start | Its own variables and flags |
|---|---|---|---|
| `warm` | the control | 12:12:22 | [the v3 preset](../../../release/flash-next-v3/settings.env) with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (R64's line) |
| `dev` | the switch | 12:27:08 | + `LLAMA_QSA_KPOOL_DEV=1` |
| `check` | the switch, checked | 12:41:56 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |
| `host` | the control, host profile | 13:04:59 | + `LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/host.csv`, `-lv 4` |
| `hostdev` | the switch, host profile | 13:20:53 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$OUT/hostdev.csv`, `-lv 4` |

- **Each load:** the client's warm-up, then at 0, 16,384 and 49,152 tokens a prime and two decodes of 256 greedy
  tokens, with a 4 s pause before every request (`PAUSE=4`), so the host profile cuts into one piece per request.
  The primes read 16,260 tokens on 60 cached, then 33,284 on 15,804.
- **`warm` is a timed control:** unlike R64, it ran at all three depths (`WARM_DEPTHS="0 16384 49152"`), so its
  depth-0 reply is there to compare. The deep lane's own tables ([table.md](table.md)) skip it and take `dev` as their
  control.
- **The build:** the night merge's rig tarball (md5 `736363d6ca12e6c81ff5f5b684ef31b3`) in place of C100's own
  tarball. Every other branch's switch is off in every load, and with every switch off this build computes what
  `pascal-p106-v4` does ([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)).
- **The switch on the rig:** the server log of `check` and `hostdev` shows the tables kept on nine cards for the
  model (16,448 rows, 2.87 MiB on the cards, 0.31 MiB on the host) and on one card for the draft (0.32 MiB). When the
  memory is freed it logs the updates: 561 micro-batches with 1 full update and 31,084 of 3,074,944 table rows sent
  (55.4 a micro-batch) for the model, and 2,196 with 1 full and 87,929 of 12,059,712 rows (40.0 a micro-batch) for
  the draft ([kpool_lines.tsv](kpool_lines.tsv)).
- **The cards** held 5,061, 5,097, 4,965, 5,113, 4,875, 4,981, 4,383, 4,993 and 4,647 MiB after loading with the
  switch, and 5,059, 5,095, 4,963, 5,111, 4,875, 4,979, 4,381, 4,991 and 4,647 MiB without. Of 6,144 MiB, the least
  free in any load was 1,043, 1,007, 1,139, 991, 1,231, 1,123, 1,721, 1,111 and 1,437 MiB (sampled every 10 s,
  [cards.tsv](cards.tsv)). Card 8's 4,647 MiB includes the night build's 80.04 MiB `h_nextn` buffer (R57).
- **The tables ran on a build machine's CPU** (`nice -n 19`, seconds each), never on the rig.

## How the host profile is cut with the switch

`host_depth.py` (R64's, unchanged) ends a draft pass at each evaluation on the draft's last card. With the switch,
each draft pass evaluates on card 8 once before its embedding card: the update's own split. The cut would therefore
count every pass twice. [run/fold_updates.py](run/fold_updates.py) gives each such evaluation card 4's number before
the cut. An evaluation is folded when it is on card 8, the next one in time is on card 4, and it took under 200 µs of
GPU time. A draft pass is then the update, the embedding card and the last card again:
- its lead starts at the update's entry;
- its crossing is still card 4's exit to card 8's entry;
- the update's GPU time counts with the embedding card's.

It folded 2,188 of the profile's 11,628 evaluations. The 9,432 evaluations of `host` stay as they are (0 folded).
With the fold, `hostdev` has the same main passes and typical steps per request as `host`, and its draft passes are
the same to within one ([host_pieces_hostdev.tsv](host_pieces_hostdev.tsv)).

## Files

| File | What it holds |
|---|---|
| [verdict.tsv](verdict.tsv) | per depth and pass: the step of the four speed loads, the two differences, the arms' means, the spread and the class |
| [steps.tsv](steps.tsv) | per load, depth and pass: tokens, accepted of drafted, steps, generation ms, ms a step, tok/s, the reply's md5s |
| [replies.tsv](replies.tsv) | per depth, pass and load: the reply's token ids' and text's md5 |
| [kpool_lines.tsv](kpool_lines.tsv) | per server log: each k-pool line kind (allocation, the check's counts, the counts at free, `LLAMA_QSA_INPUT_STATS`), its count, first and last line |
| [cards.tsv](cards.tsv) | per load and card: MiB used and free after loading, the most used over the load |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [host_depth_host.md](host_depth_host.md), [host_depth_hostdev.md](host_depth_hostdev.md) | `host_depth.py`'s report per profiled load: the cut, the k-pool lines by micro-batch size, the step by part per decode (means and medians), the growth from 16,384 to 49,152 |
| `host_summary_*.tsv`, `host_steps_*.tsv`, `host_pieces_*.tsv`, `qsa_*.tsv` | the same per load: each part's mean and median per decode, every generation step's parts, the cut into pieces, every k-pool line |
| [host.csv.gz](host.csv.gz), [hostdev.csv.gz](hostdev.csv.gz) | the host profiles, whole (9,432 and 11,628 evaluations; `hostdev` before the fold) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's line per request and per load (profile paths written as `$OUT`) |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables (control `dev`; `warm` skipped) |
| [setrows.tsv](setrows.tsv) | `test-backend-ops -o SET_ROWS` on one P106-090: every case and its result |
| [run/r74_tables.py](run/r74_tables.py) | the step, the verdict, the replies, the k-pool lines and the memory from the deep lane's folder |
| [run/fold_updates.py](run/fold_updates.py) | a `hostdev` profile made readable by `host_depth.py` |

## Commands

```bash
# from a machine that reaches the rig over ssh (RIG: its ssh alias; BIN: the night merge tarball's bin/ on the rig;
# MODEL, MODEL_DRAFT: paths on the rig; WORK: a folder on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz)
RIG=$RIG WORK=$WORK BIN=$BIN/llama-server MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL TAG='^t1swift/' \
  WD_ENV=MIN_AVAIL_MB=60 ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 \
  ARMS="sparse dev check host hostdev" \
  ARM_ENV_dev="LLAMA_QSA_KPOOL_DEV=1" \
  ARM_ENV_check="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_DEV_CHECK=1" ARM_ARGS_check="-lv 4" \
  ARM_ENV_host="LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/host.csv" ARM_ARGS_host="-lv 4" \
  ARM_ENV_hostdev="LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_INPUT_STATS=1 GGML_CUDA_PROFILE_HOST=$WORK/out/hostdev.csv" ARM_ARGS_hostdev="-lv 4" \
  LOADS="warm:sparse dev:dev check:check host:host hostdev:hostdev" DEPTHS="0 16384 49152" WARM_DEPTHS="0 16384 49152" PAUSE=4 \
  bash bench/deep_lane/remote.sh start
RIG=$RIG WORK=$WORK bash bench/deep_lane/remote.sh wait
ssh $RIG "cd $WORK && CUDA_VISIBLE_DEVICES=0 $BIN/test-backend-ops -o SET_ROWS -b CUDA0 > setrows.txt 2>&1"
RIG=$RIG WORK=$WORK OUT=$OUT CLEAN=0 bash bench/deep_lane/remote.sh collect

# on a build machine, CPU only, from this repository's root (OUT: the collected folder; RES: the tables)
R=docs/data
D=$R/2026-11-11_r74_c100_p106/run
nice -n 19 python3 -I $D/r74_tables.py $OUT --out $RES > $RES/r74.md
nice -n 19 python3 -I $R/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
H="nice -n 19 python3 -I $R/2026-11-10_step_parts_depth/run/host_depth.py \
  --step-split $R/2026-11-10_served_step_split/run/step_split.py"
for l in host hostdev; do
  nice -n 19 python3 -I $D/fold_updates.py $OUT/$l.csv $RES/$l.csv   # host: 0 folded
  mkdir -p $RES/$l
  $H --host $RES/$l.csv --requests $OUT/requests.csv --load $l --ref-load warm --grow 16384:49152 \
    --log $OUT/server-$l.log --out $RES/$l > $RES/$l/host_depth.md
done
grep -a -E 'kpool_dev_check|k-pool tables' $OUT/server-check.log   # the check's counts; no "differs" line
```

## Limits

- **One load per arm and setting.** The class pairs an unprofiled and a profiled load per arm. Their difference
  includes the profile's cost, which makes the spread conservative. Each value is one request.
- **The loads ran in one order** (`warm`, `dev`, `check`, `host`, `hostdev`). `warm` was the rig's first load after
  idle (Q1055): its depth-0 step on pass 2, 124.43 ms, is 2.39 to 2.77 ms above the other three speed loads'.
- **`set_input` is not the switch's whole host cost.** The update's rows are built in the memory's `apply()`, which
  no line here times. The step holds it.
- **What grows in the verify pass's rest is not split here** (Q1356).

## Provenance and licence

Measured on 2026-11-11 on one rig of nine P106-090 cards, with the Miner Llama fork's v4 night merge (llama.cpp, MIT).
The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they and the replies are not part of this folder (`requests.csv` and the tables hold only checksums).
