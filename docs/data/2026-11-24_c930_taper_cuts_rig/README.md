# C930: Flash-Next's prompt pieces planned around the checkpoint cuts, timed on a P106 rig (2026-11-24)

Registry row [Q1668](../../../quants/experiments.md), the rig check of [C905's `cuts=1`](../2026-11-21_c905_taper_cuts/README.md)
(Q1638). Flash-Next's v5 preset reads a prompt call of up to 1,400 tokens in micro-batches of 64
(`--ubatch-taper short=1400:64`, [R225](../2026-11-20_r225_flash_taper1400/README.md)). With in-call checkpoints the
server also cuts each call 516 and 4 tokens before its end, after the taper has planned it, so some calls kept an
extra micro-batch, five of them a piece of only 4 to 6 tokens. `cuts=1` plans each stretch between the cuts on its
own: on R225's replay it changes 14 of a pass's 73 requests, each by one micro-batch fewer. This run times it on nine
P106 cards with R225's replay.

**`cuts=1` read every timed pass faster than every pass without it: 199.71, 199.85 and 200.06 s against 205.21,
201.37, 201.78 and 201.75 s (−2.66 s, −1.31 %). By the rule fixed before the run that is negligible, because the
first control pass (205.21 s, still warming up) sets a spread of 3.83 s; it is slower in no size bucket and a gain on
the turns of 600 to 1,023 tokens.** The whole difference sits on the 14 turns whose plan it changes (each of them
faster in every `cuts=1` pass than in every control pass); the other 48 turns read the same. The greedy replies are
the control's on 62 of 62 turns in every pass. The preset is not changed here: on this class `cuts=1` can go into it
on the owner's word, after its quality check (C981, [Q1714](../../../quants/experiments.md)).

## The run

| | |
|---|---|
| Rig | the second rig, nine P106-090 cards, with its soaked clock profile ([R234](../2026-11-19_r209_oc_v5/README.md#the-profile-again-on-the-boot-of-2026-11-24)) and the kit's watchdog on, as found |
| Engine | `pascal-p106-v5` `af00bb13b` (C905 landed there, merge `7ad7608b7`), `llama-server` md5 `f5cca34ea6bd4f3712cd6193bf0605be` |
| Model | UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as R225 |
| Preset | [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it is now (md5 `b8e950f2de8c7906c156d582416731bf`), its whole line and environment; the line holds `--ubatch-taper short=1400:64`. Only the taper's setting changes, per request (`"ubatch_taper"`) |
| Replay | C222's driver [run/c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh) unchanged (md5 `f87da14737d71f6d287d8d21ac84dfc1`) with `OUTN=c930` and the passes below: [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (`06fa1375d22aa8971b48561931b4fd30`, with its card log since C873) and [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`) on the agent check's [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`): per task the shared 1,549-token prefix, then each turn's new tokens and 8 greedy tokens; the same tokens in every pass |
| Passes | one load, eight passes: a warm-up pass with the control, then `short=1400:64` (control) and `short=1400:64,cuts=1` alternated, control first and last (passes 1, 3, 5, 7 control; 2, 4, 6 `cuts=1`). The driver cannot mark a pass as unmeasured, so the warm-up is pass 0 and every table leaves it out (`c222_tables.py --drop-pass 0`); [passes_with_warmup.tsv](passes_with_warmup.tsv) has it |
| Time | started 20:40:27, server up after 226 s, replay ended 21:18:17 (pass wall times 288, 255, 249, 250, 249, 251, 249 and 251 s); nothing else ran on the rig |

**Every call ran as C905's plan says** ([plan_check.tsv](plan_check.tsv), [run/c930_cuts.py](run/c930_cuts.py)): the
server's `ubatch taper:` lines ([taper_lines.txt](taper_lines.txt)) match the plan C905 computed on a processor
([plans.tsv](../2026-11-21_c905_taper_cuts/plans.tsv), its `before` column for the control and `after` for
`cuts=1`) on 73 of 73 calls in each of passes 1 to 7, and on 72 of 72 in pass 0, whose 73rd call is the load's
first, cold 1,549-token prefix (a size later passes read from the cache). For example the 577-token call reads
`61 5 62 64x6 61 4` with the control and `61 64x8 4` with `cuts=1`. Every pass read the same 36,793 tokens for 36,227
planned. No graph reallocation in any of the 584 requests. Lowest free host memory 267 MiB (the replay's watchdog
line is 60). The clock watchdog never tripped.

## Results

Prompt seconds a pass over the 62 turns, by the size of a turn's new prompt ([passes.tsv](passes.tsv),
[buckets.tsv](buckets.tsv)):

| Pass | Taper | All | Under 600 | 600–1,023 | 1,024–1,999 | 2,000+ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `short=1400:64` (warm-up, not measured) | 217.42 | | | | |
| 1 | `short=1400:64` (control) | 205.21 | 71.71 | 26.74 | 76.15 | 30.62 |
| 2 | `short=1400:64,cuts=1` | 199.71 | 70.04 | 25.15 | 75.13 | 29.40 |
| 3 | `short=1400:64` (control) | 201.37 | 70.34 | 26.11 | 75.52 | 29.41 |
| 4 | `short=1400:64,cuts=1` | 199.85 | 70.09 | 25.14 | 75.20 | 29.41 |
| 5 | `short=1400:64` (control) | 201.78 | 70.49 | 26.15 | 75.70 | 29.44 |
| 6 | `short=1400:64,cuts=1` | 200.06 | 70.13 | 25.18 | 75.28 | 29.47 |
| 7 | `short=1400:64` (control) | 201.75 | 70.48 | 26.14 | 75.67 | 29.46 |
| | turns, new tokens | 62, 36,227 | 42, 8,288 | 6, 4,834 | 11, 15,480 | 3, 7,625 |

### The verdict

The class is the mean of the `cuts=1` passes minus the mean of the four control passes, against the control passes'
largest difference (the spread): gain below minus the spread, loss above it, negligible between
([decision.md](decision.md)).

| Setting | All (spread 3.83 s) | Under 600 (1.37) | 600–1,023 (0.63) | 1,024–1,999 (0.63) | 2,000+ (1.20) |
|---|---|---|---|---|---|
| `short=1400:64,cuts=1` | **negligible**: −2.66 s, −1.31 % | negligible, −0.67 | gain, −1.13 | negligible, −0.55 | negligible, −0.30 |

- **Slower in no bucket**, by this rule and by R60's (the larger of the control's and the setting's own spread; the
  `cuts=1` passes differ by 0.35 s at most). So `cuts=1` meets the condition for the preset: a gain, or negligible and
  slower in no bucket. The preset change is the owner's, after C981's quality check.
- **Said plainly, the fixed rule's spread is mostly warm-up, not noise.** One warm-up pass was not enough on this
  load: pass 1, the first measured control, was 3.58 s slower than the mean of the other three controls (201.37,
  201.78 and 201.75 s, which agree within 0.41 s), most of it on the two longest calls (2,308 and 3,313 tokens, +0.67
  and +0.52 s). R225's warm-up absorbed this; here pass 0 carried 15.79 s and pass 1 the rest. Every `cuts=1` pass
  is faster than the fastest control (201.37 s). Leaving pass 1 out as well, a choice made after seeing the data and so
  not the verdict, it would read as a **gain**: −1.76 s, −0.87 %, against a spread of 0.41 s, a gain in three
  buckets and negligible at 2,000+ ([buckets_without_pass1.tsv](buckets_without_pass1.tsv),
  [decision_without_pass1.md](decision_without_pass1.md)). A second warm-up pass for this replay is code row 23d
  (C1084, [Q1817](../../../quants/experiments.md)).
- **Replies:** the same greedy reply as the control's on 62 of 62 turns in every `cuts=1` pass; the four measured
  control passes agree on 62 of 62 ([replies.tsv](replies.tsv)). (The driver's own end line, "identical to off's on 0
  of 124 turns", compares with a pass named `off`, which this run does not have.)

### Where the time goes: the 14 changed turns

Prompt seconds a pass over the 14 turns whose plan `cuts=1` changes, and over the other 48
([changed_sum.tsv](changed_sum.tsv)):

| Pass | Taper | 14 changed turns | 48 other turns |
|---:|---|---:|---:|
| 1 | control | 66.65 | 138.56 |
| 2 | `cuts=1` | 63.73 | 135.98 |
| 3 | control | 65.46 | 135.92 |
| 4 | `cuts=1` | 63.75 | 136.09 |
| 5 | control | 65.59 | 136.19 |
| 6 | `cuts=1` | 63.84 | 136.22 |
| 7 | control | 65.53 | 136.23 |

On the changed turns `cuts=1` is 2.04 s a pass faster than the four controls' mean (65.81 s against 63.77 s); the
other 48 turns read 136.10 s with `cuts=1` and 136.73 s with the control, 136.11 s without pass 1, the same.

Each changed turn, prompt milliseconds by pass ([changed_calls.tsv](changed_calls.tsv)); the five calls marked * had a
4- to 6-token piece:

| Call tokens | Plan, control → `cuts=1` | Control p1, p3, p5, p7 | `cuts=1` p2, p4, p6 | Difference to the controls' mean |
|---:|---|---|---|---:|
| 569 | `53 11 64x7 53 4` → `53 64x8 4` | 3403.0, 3401.6, 3400.9, 3396.1 | 3352.4, 3359.8, 3363.5 | −41.8 |
| 577 * | `61 5 62 64x6 61 4` → `61 64x8 4` | 3678.9, 3336.9, 3345.9, 3338.3 | 3234.9, 3235.0, 3239.2 | −188.6 |
| 583 * | `63 4 61 64x6 67 4` → `67 64x8 4` | 3608.9, 3600.1, 3613.4, 3606.8 | 3472.1, 3478.5, 3484.6 | −128.9 |
| 599 | `64 19 45 64x7 19 4` → `64 19 64x8 4` | 3562.8, 3556.5, 3566.1, 3566.5 | 3474.9, 3472.9, 3475.4 | −88.6 |
| 610 | `64 30 34 64x7 30 4` → `64 30 64x8 4` | 3538.3, 3488.8, 3488.8, 3492.2 | 3412.0, 3397.0, 3403.9 | −97.7 |
| 775 * | `64x3 63 4 61 64x6 67 4` → `64x3 67 64x8 4` | 4300.4, 4305.3, 4305.3, 4304.5 | 4000.0, 4004.6, 4005.0 | −300.7 |
| 803 | `64x4 31 33 64x7 31 4` → `64x4 31 64x8 4` | 4552.5, 4196.3, 4205.8, 4198.1 | 4106.7, 4106.7, 4113.2 | −179.3 |
| 815 | `64x4 43 21 64x7 43 4` → `64x4 43 64x8 4` | 4388.1, 4381.0, 4390.3, 4385.4 | 4318.2, 4325.0, 4334.6 | −60.3 |
| 896 * | `64x5 60 5 63 64x6 60 4` → `64x5 60 64x8 4` | 4798.3, 4675.7, 4690.5, 4696.4 | 4547.5, 4541.6, 4544.5 | −170.7 |
| 936 | `64x6 36 28 64x7 36 4` → `64x6 36 64x8 4` | 5157.5, 5059.1, 5071.7, 5067.3 | 4764.2, 4769.5, 4779.8 | −317.7 |
| 1,086 * | `64x8 58 6 64x7 58 4` → `64x8 58 64x8 4` | 5654.7, 5660.4, 5671.2, 5658.2 | 5432.9, 5440.5, 5443.8 | −222.1 |
| 1,130 | `64x9 38 26 64x7 38 4` → `64x9 38 64x8 4` | 5902.5, 5902.5, 5922.5, 5906.0 | 5867.3, 5863.6, 5871.0 | −41.1 |
| 1,300 | `64x12 16 48 64x7 16 4` → `64x12 16 64x8 4` | 6899.4, 6895.1, 6924.3, 6891.9 | 6773.9, 6788.8, 6790.0 | −118.4 |
| 1,359 | `64x13 11 53 64x7 11 4` → `64x13 11 64x8 4` | 7206.5, 6997.3, 6997.9, 7017.5 | 6975.8, 6969.3, 6989.5 | −76.6 |

All 14 turns are faster in every `cuts=1` pass than in every control pass. The five calls that lost a 4- to 6-token
piece saved 188.6, 128.9, 300.7, 170.7 and 222.1 ms by this table's means; two calls without such a piece saved as
much (936: 317.7 ms, 803: 179.3 ms), so the micro-batch saved counts more than the kind of piece removed.

A line through the turns of 512 to 1,999 new tokens ([fit.tsv](fit.tsv)): the control 1.47 s + 3.76 ms a token,
`cuts=1` 1.25 s + 3.87 ms. `cuts=1` changes only some calls in that range, so its line describes no single plan.

## Files

| File | What |
|---|---|
| [passes.tsv](passes.tsv), [passes_with_warmup.tsv](passes_with_warmup.tsv) | per pass: the setting, turns, new and read tokens, prompt seconds and tok/s, the requests' wall seconds, turns whose read tokens differ from the plan, graph reallocations, calls logged, the largest micro-batch of a call of 601 to 2,048 tokens; without and with the warm-up pass |
| [buckets.tsv](buckets.tsv), [decision.md](decision.md) | per setting and size bucket: seconds of each pass, against the four control passes, the spread, the class (also by R60's rule); the classes in words: **the verdict** |
| [buckets_without_pass1.tsv](buckets_without_pass1.tsv), [decision_without_pass1.md](decision_without_pass1.md) | the same with pass 1 left out too (three controls): a reading after the fact, not the verdict |
| [turns.tsv](turns.tsv) | every replayed turn: task, turn, new tokens, the tokens its call read, prompt milliseconds in each measured pass |
| [replies.tsv](replies.tsv) | per setting: turns whose greedy reply is the control's in every pass |
| [fit.tsv](fit.tsv) | per setting: the fixed seconds a turn and milliseconds a token over the turns of 512 to 1,999 new tokens |
| [plans.tsv](plans.tsv), [taper_lines.txt](taper_lines.txt) | the micro-batches the server logged for every call (`ubatch taper:` lines, in request order); per pass and call size the largest and the most common |
| [plan_check.tsv](plan_check.tsv) | per pass: calls logged, calls whose size C905's plan covers, those read as planned for the pass's setting, and any difference |
| [changed_calls.tsv](changed_calls.tsv), [changed_sum.tsv](changed_sum.tsv) | every turn whose plan `cuts=1` changes, milliseconds by pass; per pass the seconds of the changed turns and of the others |
| [requests/load-A.tsv](requests/load-A.tsv) | every request as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/c930_cuts.py](run/c930_cuts.py) | the plan check and the changed-turn tables |

The driver and the tables script are C222's: [run/c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh),
[run/c222_tables.py](../2026-11-20_c222_flash_microbatch/run/c222_tables.py) (md5 `6d10266320e02c561a1ee3bb2f061494`).

## Commands

A folder on the rig (`WORK`) holds `rig_taper.sh`, `taper_replay.py`, `turns.csv` and the v5 preset's
`settings.env` (the md5s above). The engine's rig tarball is unpacked in `$BUILDS/$BUILD`, the model's first shard is
in `MODEL` (shards 2 and 3 beside it) and the MTP draft in `DRAFT`. From a machine that only runs ssh:

```bash
export RIG=<ssh alias> WORK=<folder on the rig> BUILDS=<folder of the builds> \
  BUILD=llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2 MODEL=<first shard> DRAFT=<MTP draft> OUTN=c930 \
  PASSES="short=1400:64 short=1400:64 short=1400:64,cuts=1 short=1400:64 short=1400:64,cuts=1 short=1400:64 short=1400:64,cuts=1 short=1400:64"
DRY=1 bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh    # prints the rig command
bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh
# done when $WORK/c930/DONE exists; then copy $WORK/c930/load-A off the rig
C=docs/data/2026-11-20_c222_flash_microbatch; R=docs/data/2026-11-24_c930_taper_cuts_rig
python3 -I $C/run/c222_tables.py --load <copy>/load-A --control short=1400:64 --drop-pass 0 --out tables
python3 -I $C/run/c222_tables.py --load <copy>/load-A --control short=1400:64 --out tables_all          # passes_with_warmup.tsv
python3 -I $C/run/c222_tables.py --load <copy>/load-A --control short=1400:64 --drop-pass 0 --drop-pass 1 --out tables_drop01
python3 -I $R/run/c930_cuts.py --load <copy>/load-A --plans docs/data/2026-11-21_c905_taper_cuts/plans.tsv \
  --setting short=1400:64 --out cuts
```

## Provenance and licence

The engine is llama.cpp's licence (MIT). The replay's text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's measurements.
