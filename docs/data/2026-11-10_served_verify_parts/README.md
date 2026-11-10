# The served verify pass, part by part (R49, Q1211, 2026-11-10)

What this measured: where the verify pass of a served speculative step spends its GPU time on the Miner Llama v3.0
release line (`pascal-p106-v3-b12076-ladder5` `1551f5fe1`, the v3 preset exactly: UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three tokens), launch group by launch group, beside a plain
pass of the same four tokens ([Q1211](../../../quants/experiments.md)). The step around it was split the same morning
([2026-11-10_served_step_split](../2026-11-10_served_step_split/README.md), S37): its verify pass is 106.88 ms, and
its GPU time is 16.53 ms above the plain four-token pass that
[2026-11-10_release_profile](../2026-11-10_release_profile/README.md) (R41) profiled (85.36 ms of GPU time). S37's
host profile gives each card's total only. This run gives the parts, from generation: one load of the preset under the
fork's profiler in group mode, answering a greedy request twice.

## Result

GPU time per verify pass, ms, the profiler's 7.6 µs subtracted per launch group. `plain` is R41's 82 plain prompt
passes of four tokens (no draft loaded, no output row, depth 0 to 336); `r1` and `r2` are the typical verify passes of
the two greedy requests of the one profiled load (81 and 80 passes, depth 79 to 335).

### Where the difference goes

[accounting.tsv](accounting.tsv); the first rule that matches a launch group wins (`run/verify_parts.py`, `BUCKETS`).

| Cards 0 to 7 | plain | r1 | r2 | r1 − plain | r2 − plain |
|---|---:|---:|---:|---:|---:|
| `GATED_DELTA_NET` (35 launch groups in each) | 3.68 | 8.66 | 8.65 | +4.97 | +4.97 |
| `GDN_CONV` (35) | 0.47 | 0.72 | 0.74 | +0.25 | +0.27 |
| copies of the PLE conv history (`cache_ple_r`: 1 a plain pass, 4 a verify pass) | 0.05 | 0.19 | 0.19 | +0.15 | +0.15 |
| routed experts | 19.27 | 19.54 | 19.52 | +0.27 | +0.25 |
| shared expert, router, hyper-connections | 19.29 | 19.35 | 19.34 | +0.06 | +0.05 |
| everything else | 39.72 | 39.66 | 39.68 | −0.06 | −0.04 |
| **cards 0 to 7** | **82.48** | **88.13** | **88.13** | **+5.64** | **+5.65** |

| Card 8 (the last card) | plain | r1 | r2 | r1 − plain | r2 − plain |
|---|---:|---:|---:|---:|---:|
| the output head's product (`output`, Q5_K, 2,560 × 248,320) | 0.00 | 8.05 | 8.06 | +8.05 | +8.06 |
| the sampling chain after the product | 0.00 | 0.68 | 0.68 | +0.68 | +0.68 |
| the output's hyper-connection products | 0.00 | 0.11 | 0.11 | +0.11 | +0.11 |
| `GATED_DELTA_NET` and `GDN_CONV` | 0.12 | 0.27 | 0.27 | +0.15 | +0.15 |
| routed experts | 0.41 | 0.85 | 0.86 | +0.45 | +0.45 |
| shared expert, router, hyper-connections | 0.58 | 0.93 | 0.94 | +0.35 | +0.35 |
| everything else | 1.77 | 1.96 | 1.96 | +0.20 | +0.19 |
| **card 8** | **2.88** | **12.86** | **12.87** | **+9.98** | **+9.99** |

- **Card 8's +9.98 and +9.99 ms:**
  - the head's product for four rows, 8.05 and 8.06 ms;
  - the sampling chain after it, 0.68 ms in both: a `PAD` of the logits 0.22 ms, four `TOP_K` launches of 0.08 ms
    each (0.33), and the top-p, argmax and row gathers of the four rows 0.13 ms
    ([last_card_groups.tsv](last_card_groups.tsv));
  - 1.10 ms in both requests for the last layer's feed-forward and the output's hyper-connection over the four rows.
    A prompt pass with no output row skips them, because the last layer keeps only the rows that have an output. On
    card 8 the plain pass runs one layer's feed-forward (one routed-expert pair, one router, three shared-expert
    products); the verify pass runs two (two pairs, two routers, six products) ([labels.tsv](labels.tsv));
  - 0.15 ms of rollback snapshots in its one recurrent layer.
- **Cards 0 to 7's +5.64 and +5.65 ms:**
  - the rollback snapshots, 5.37 and 5.39 ms. The `GATED_DELTA_NET` groups take +4.97 ms in both requests. Their
    count is the same, but each group runs longer: with a draft of three, the model keeps three snapshots of the
    recurrent state. `GDN_CONV` adds +0.25 and +0.27, and three more copies of the PLE conv history add +0.15 and +0.15;
  - the routed experts, +0.27 and +0.25 ms. The verify passes run on generated text, the plain passes on agent text,
    so other experts are picked (est.);
  - the rest, +0.01 in both.
- **Over the whole step** (S37's median, 131.57 ms):
  - the head's product with its sampling chain is 8.73 and 8.74 ms (6.6 %);
  - the snapshots are 5.52 and 5.54 ms on all nine cards (4.2 %).

### What the code rows could save at most

| Row | What it removes | At most, ms a verify pass |
|---|---|---:|
| [Q663](../../../quants/experiments.md) (C123), a shortlist head for verify batches, parked for exactness | the head's product and the sampling chain | 8.73 and 8.74 (est. 7.0 with a 32,768-row shortlist read at the full head's rate: 1.06 ms of the product left) |
| [Q1159](../../../quants/experiments.md), replay instead of rollback snapshots, upstream | the snapshots on all nine cards | 5.52 and 5.54 (the replay's own cost on a rejected draft not counted) |
| [Q1222](../../../quants/experiments.md) (C690), the head's four-row product at its one-row cost, exact | the product above one row's 2.86 ms (R41) | est. 5.19 and 5.20 |
| [Q1223](../../../quants/experiments.md) (C691), one `TOP_K` launch for the four rows, no `PAD` | part of the sampling chain | est. 0.5 of the 0.68 |

- **The head's product at four rows is not limited by reading the weights.** The head is about 417 MiB of Q5_K. One
  output row costs 2.86 ms (R41: the output layer's part in two single passes), so the weights are read at about
  153 GB/s. Four rows cost 2.82 times as long (8.05 and 8.06 ms).
- **The k-curve bench found the same at this shape**, measured with `test-backend-ops perf` on a P106
  ([2026-11-04_kcurve_p106/perf_dense.tsv](../2026-11-04_kcurve_p106/perf_dense.tsv)):
  - four columns: 8,545.5 µs on the earlier kernels and 11,563.8 µs on the k-column kernel at its old launch shape;
  - one column: 3,109.7 µs, so four columns take 2.75 times as long;
  - with the 32 KB budget: est. 8,103.3 µs at four columns and est. 2,264.5 µs at one
    ([2026-11-04_kshape/ggml_est.tsv](../2026-11-04_kshape/ggml_est.tsv)).
- **Which kernel the preset takes for it is not established by the profile.** The preset sets
  `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`, which leaves the choice to the k-column kernel's launch shape.

### Beside S37's figures

- **Both sides now come from the group profile.** S37's +10.07 (card 8) and +6.46 ms (cards 0 to 7) set its host
  profile's per-card medians (101.89 ms in all) against R41's group profile (85.36 ms). Here both sides use the group
  profile: +9.98 and +9.99 on card 8, +5.64 and +5.65 on cards 0 to 7.
- **The two profiles differ by about 1 ms on the same verify pass.** S37's host profile put its GPU time at 102.10 ms
  (the mean of 82 steps; median 101.91). The group profile here puts it at 100.99 and 101.00 ms (the means of 81 and
  80 passes, [per_card.tsv](per_card.tsv)). The host profile times each evaluation between two events without
  synchronizing; the group profile synchronizes after every evaluation and times each launch group. This run does not
  split the difference.

### The replies and the profile's own cost

[replies.md](replies.md): the server's own timings per reply.

| Load | Request | Prompt tokens read | Drafted | Accepted | Steps (server) | Generation, ms | ms a step |
|---|---|---:|---:|---:|---:|---:|---:|
| `warm` (no profile) | 1 | 79 | 255 | 170 | 86 | 12,418.25 | 144.40 |
| `warm` | 2 | 4 | 247 | 172 | 84 | 10,410.46 | 123.93 |
| `prof` (group mode) | 1 | 79 | 255 | 170 | 86 | 14,860.08 | 172.79 |
| `prof` | 2 | 4 | 247 | 172 | 84 | 13,248.75 | 157.72 |

- **All four replies have the same text** (the reasoning and the answer).
- **The second request is not a copy of the first.** It restored a context checkpoint of 75 of the 79 prompt tokens
  and read 4. It then drafted 247 tokens and accepted 172, against 255 and 170, the same in both loads.
- **The full-head drafting passes:** the first request ran 42 of them (14 steps of three), the second none
  ([census.tsv](census.tsv): card 8's 76-group draft graph). The first request of a load also pays one-off launches.
- **What the profile does to the step:** under it, from verify pass to verify pass, a typical step takes 156.47 and
  159.36 ms ([cut.tsv](cut.tsv)), against S37's 131.57 ms unprofiled. The group profile synchronizes after every
  evaluation, so its step times are not the preset's. Its GPU times are what this folder uses.

## What ran

One detached script on the rig ([run/verify_prof.sh](run/verify_prof.sh)), two loads, 12 minutes (13:34:54 to
13:46:54), nothing else on the rig:

| Load | What | Profile |
|---|---|---|
| `warm` | the preset | none: warms the cards and gives the server's timings without the profiler |
| `prof` | the preset | `GGML_CUDA_PROFILE=<file> GGML_CUDA_PROFILE_MODE=group` |

- **The line:** `release/flash-next-v3/settings.env` exactly (`PRESET_ENV`, `PRESET_ARGS`, the MTP draft). Both
  loads were started through `bench/agent_check/rig_server.sh` with its monitors and RAM watchdog (floor 60 MiB), as
  S37's loads were.
- **The requests:** the greedy request, twice per load, four seconds apart
  ([greedy_request.json](../2026-11-04_draft_one_card/run/greedy_request.json): temperature 0, 256 tokens).
- **The host's memory:** the lowest MemAvailable during the profiled load's requests was 303 MiB.
- **The cards:** after each load's requests they ran at 1,860 to 1,911 MHz, at 31 to 35 °C.
- **The tables ran on a build machine's CPU**, never on the rig.

## How the passes are cut

`run/verify_parts.py` (its header has the definitions):
- **The draft's evaluations:** a group-mode trace names no graph. An evaluation with a launch group labelled `mtp_`
  is the draft's: 677 on card 4 (its token row) and 677 on card 8.
- **The main passes:** every card runs 173 evaluations of the model's graph. The j-th on each card is main pass j;
  under this profiler the cards run one after another, and no pass is out of order.
- **The requests:** pauses of more than 3 s cut the trace into the load and the two requests (88 and 84 main passes).
- **A step:** a main pass and the draft passes before the next one. A request's first passes are its prompt (above
  1.5 times the median GPU time). A typical step has four draft passes and a main pass within 12 % of the median.
  The first three steps of a request are left out, and so is its last pass. This keeps 81 and 80 verify passes; on
  card 8 all of them run the same 203-group graph.
- **The parts:** R41's estimator (`kpass_parts.py`'s `parts_per_pass`, the profiler's 7.6 µs subtracted per span),
  per card group, on the kept passes and on R41's kept plain passes.
- **The last card by position:**
  - before the first launch group labelled ` output:`;
  - that group, the head's product;
  - what follows it, the sampling chain.
- **The self-test** (`--self-test`, 46 checks) runs the whole cut on a synthetic trace with known parts.

## Files

| File | What it holds |
|---|---|
| [verify_parts.md](verify_parts.md) | the full report: the cut per request, parts per card group, where the difference goes, by position, by first op, card 8's launch groups after the head, per card |
| [accounting.tsv](accounting.tsv) | per trace and card group: ms a pass in each bucket of "where the difference goes" |
| [parts.tsv](parts.tsv) | per trace and card group (cards 0 to 7, card 8, all cards): ms a pass by R41's parts; card 8 by position |
| [by_op.tsv](by_op.tsv), [labels.tsv](labels.tsv) | per trace and card group: launch groups a pass and ms a pass by first op and by label |
| [last_card_groups.tsv](last_card_groups.tsv) | card 8's launch groups in graph order, with their position (before, the head, after) and ms |
| [per_card.tsv](per_card.tsv) | per trace and card: ms a pass (profiler cost subtracted) and the median raw GPU time |
| [passes.tsv](passes.tsv) | every main pass of the trace: piece, request, kind (prompt, step, last), typical or not, draft passes after it, host times, raw GPU time per card |
| [cut.tsv](cut.tsv) | per request: main passes, prompt passes, steps, typical steps, from pass to pass under the profiler |
| [census.tsv](census.tsv) | the graphs each card ran (R41's `trace_census.py`): evaluations, launch groups, median and summed GPU time |
| [replies.md](replies.md), [replies.tsv](replies.tsv) | the server's timings per reply, the text's checksum |
| [summaries/](summaries/) | `prof_summary.py`'s report per trace, on all cards and on card 8 |
| [run/verify_prof.sh](run/verify_prof.sh) | the rig run: two loads of the preset, the greedy request twice each |
| [run/verify_parts.py](run/verify_parts.py) | the trace cut into requests, steps and verify passes, the parts beside a plain pass, the self-test |
| [run/replies.py](run/replies.py) | the replies' table |

## Commands

```bash
# on the rig, idle, detached (about 12 minutes; BIN: the release build's bin/, MODEL: the Swift Q2_0 first shard,
# DRAFT: the MTP draft, S: a folder with this repository's files)
BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  OUT=$OUT setsid nohup bash $S/docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh > console.out 2>&1 < /dev/null &

# on a build machine, CPU only, from this repository's root. TRACE: $OUT/prof/prof-verify.csv copied there (24.8 MB,
# md5 92a891f032d1aa9c6278f0e59acf34ce); PLAIN: release-k4.csv, R41's kept plain four-token passes, which
# 2026-11-10_release_profile's release_parts.py writes with "--tag release" (md5 3fca230b351b0f092bb991bb44dc27ce)
R=docs/data/2026-11-10_served_verify_parts/run
V="nice -n 19 python3 -I $R/verify_parts.py --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py \
  --prof-summary scripts/prof_summary.py"
$V --self-test
$V --event-us 7.6 --plain $PLAIN --out $RES $TRACE > $RES/report.md
nice -n 19 python3 -I docs/data/2026-11-10_release_profile/run/trace_census.py $TRACE --out $RES/census.tsv --gap-ms 3000
nice -n 19 python3 -I $R/replies.py $OUT --loads warm,prof --tsv $RES/replies.tsv > $RES/replies.md
```

## Limits

- **One profiled load.** The two requests are two samples of every part from that load, and they agree within
  0.02 ms in every part.
- **The verify passes are at depth 79 to 335, on one short greedy reply.** At 2,760 tokens of context S37's verify
  pass was 7.32 ms longer; this run does not split that growth.
- **Plain and verify passes ran on different text:** the plain passes on agent text, the verify passes on the reply.
  The routed experts' +0.27 and +0.25 ms on cards 0 to 7 is read as routing; that reading is not checked.
- **The profiler's 7.6 µs a launch group is an estimate:** C140's figure, the one R41 used. Card 8's verify graph
  has 203 launch groups, 119 more than a plain pass, so about 0.90 ms of profiler cost was subtracted for them.
- **Which kernel runs the head's four-row product is not checked.**

## Provenance and licence

Measured on 2026-11-10 on one rig of nine P106-090 cards, with the release build of the Miner Llama fork (llama.cpp,
MIT). The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The replies are not part of this folder.
