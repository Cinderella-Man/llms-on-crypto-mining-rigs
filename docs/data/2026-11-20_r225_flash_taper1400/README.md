# R225: Flash-Next's prompt calls of 601 to 1,400 tokens in micro-batches of 64 (2026-11-20)

Registry row [Q1637](../../../quants/experiments.md), from [C222's replay](../2026-11-20_c222_flash_microbatch/README.md)
(Q775). Nine P106 cards run Qwen3.8-Flash-Next as a pipeline, so a prompt call ends after the sum of its
micro-batches plus about eight times the largest one; smaller micro-batches shorten a mid-sized call and lengthen a
long one. C222 found that micro-batches of 64 read every call of 610 to 1,359 tokens faster than the v5 preset's 512s
and the calls of 1,690 tokens or more slower, and composed from its passes what a size limit between the two would do.
This replay measures the two limits it pointed to.

**Reading calls of up to 1,400 tokens in micro-batches of 64 (`--ubatch-taper short=1400:64`) is a gain: 208.30 and
207.38 s a pass against the v5 preset's 220.24, 220.41 and 220.55 s (−12.56 s, −5.70 %), and it loses in no size
bucket.** `short=1023:64` is a gain too (210.77 and 210.95 s, −9.54 s, −4.33 %) and also loses in no bucket; 1,400 is
3.02 s a pass better. Both match C222's composition (est. −12.60 and −9.54 s). The greedy replies are the control's on
62 of 62 turns in every pass. Each pass replays the agent check's 62 recorded prompts (12 tasks), about 1 s of prompt
reading saved a task with `short=1400:64` (est. 0.8 % of a task); its agent pair, the measure release settings are
chosen on, is the next row (R227, [Q1641](../../../quants/experiments.md)).

## The run

| | |
|---|---|
| Rig | the second rig, nine P106-090 cards, with its soaked clock profile ([R219](../2026-11-19_r209_oc_v5/README.md#the-profile-again-on-the-boot-of-2026-11-20)) and the kit's watchdog on |
| Engine | `pascal-p106-v5` `602691153` (upstream `b12483` merged), `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`, as C222 |
| Model | UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft, as C222 |
| Preset | [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it is now (md5 `4cac44b57bd1c6465390dfaab6cd1847`): its 40 switches, among them the sparse attention from 6,153 cells (`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75`, [R220](../2026-11-20_r220_flash_ratio075/README.md)), which C222 ran without; its whole line (`-b 4096 -ub 512`, in-call checkpoints, the server's taper `short=600:64`). Every pass has the same preset; only the taper changes |
| Replay | C222's driver [run/c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh) unchanged (md5 `f87da14737d71f6d287d8d21ac84dfc1`), with `OUTN=r225` and the passes below: [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (`353ec109ef8b5806c38e7c14a1c23dfb`) and [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`) on the agent check's [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`): per task the shared 1,549-token prefix, then each turn's new tokens of Python standard-library source and 8 greedy tokens; the same tokens in every pass |
| Passes | one load, eight passes, the taper set per request (`"ubatch_taper"`): a warm-up pass with the control (`short=600:64`), then `short=600:64`, `short=1400:64`, `short=1023:64`, `short=600:64`, `short=1023:64`, `short=1400:64`, `short=600:64`. The driver cannot mark a pass as unmeasured, so the warm-up is pass 0 and every table leaves it out (`c222_tables.py --drop-pass 0`); [passes_with_warmup.tsv](passes_with_warmup.tsv) has it |
| Time | started 15:35:32, server up after 227 s, passes 15:39 to 16:15:08 (299, 270, 258, 261, 270, 261, 257 and 270 s of wall time); nothing else ran on the rig |

**The warm-up was needed again:** pass 0 read the replay in 230.60 s, 10.20 s slower than the mean of the three
measured controls, 4.45 s of it on the load's first measured turn (the 896-token turn 1 of JavaScript
`complex-numbers`, 10.72 s against 6.26, 6.29 and 6.28 s). C222's first pass did the same and set its spread. Its
replies are the control's on all 62 turns.

**Every pass ran as set** (the server's `ubatch taper:` lines, [taper_lines.txt](taper_lines.txt), counted by
[run/r225_plans.py](run/r225_plans.py) in [plans_by_limit.tsv](plans_by_limit.tsv)): the largest micro-batch of a
call, by the call's tokens and pass,

| Call tokens (calls a pass) | Control (passes 0, 1, 4, 7) | `short=1023:64` (3, 5) | `short=1400:64` (2, 6) |
|---|---:|---:|---:|
| 9 to 600 (42) | 67 | 67 | 67 |
| 601 to 1,023 (6) | 512 | 67 | 67 |
| 1,024 to 1,400 (5) | 512 | 512 | 64 |
| 1,401 to 2,048 (7) | 512 | 512 | 512 |
| over 2,048 (2) | 512 | 512 | 512 |

For example the 1,028-token call reads as `512x2 4` with the control and `short=1023:64`, as `64x16 4` with
`short=1400:64`; the 1,359-token call as `512 331 512 4` and as `64x13 11 53 64x7 11 4` (a 67 is a 64 that took in a
piece under the taper's 9-token floor; the 4 to 6 are the in-call checkpoint cuts, [Q1638](../../../quants/experiments.md)).
Every pass read the same 36,793 tokens for 36,227 planned (as C222). No graph reallocation in any of the 584 requests.
Lowest free host memory 269 MiB (the replay's watchdog line is 60). The clock watchdog never tripped.

## Results

Prompt seconds a pass over the 62 turns, by the size of a turn's new prompt ([passes.tsv](passes.tsv),
[buckets.tsv](buckets.tsv)):

| Pass | Taper | All | Under 600 | 600–1,023 | 1,024–1,999 | 2,000+ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `short=600:64` (warm-up, not measured) | 230.60 | | | | |
| 1 | `short=600:64` (control) | 220.24 | 71.94 | 36.01 | 81.67 | 30.61 |
| 2 | `short=1400:64` | 208.30 | 71.93 | 26.93 | 78.92 | 30.52 |
| 3 | `short=1023:64` | 210.77 | 72.04 | 26.68 | 81.56 | 30.49 |
| 4 | `short=600:64` (control) | 220.41 | 72.09 | 36.09 | 81.59 | 30.65 |
| 5 | `short=1023:64` | 210.95 | 72.01 | 26.67 | 81.54 | 30.73 |
| 6 | `short=1400:64` | 207.38 | 71.89 | 26.68 | 78.22 | 30.59 |
| 7 | `short=600:64` (control) | 220.55 | 72.01 | 36.34 | 81.58 | 30.63 |
| | turns, new tokens | 62, 36,227 | 42, 8,288 | 6, 4,834 | 11, 15,480 | 3, 7,625 |

### The verdict

The class is the mean of a setting's passes minus the mean of the control's three, against the control passes'
largest difference (the spread): gain below minus the spread, loss above it, negligible between
([decision.md](decision.md)).

| Setting | All (spread 0.31 s) | Under 600 (0.15) | 600–1,023 (0.32) | 1,024–1,999 (0.10) | 2,000+ (0.04) |
|---|---|---|---|---|---|
| `short=1400:64` | **gain**: −12.56 s, −5.70 % | negligible, −0.10 | gain, −9.34 | gain, −3.04 | gain, −0.08 |
| `short=1023:64` | **gain**: −9.54 s, −4.33 % | negligible, +0.01 | gain, −9.47 | negligible, −0.06 | negligible, −0.02 |

- **Neither setting loses in any bucket**, by this rule and by R60's (which takes the larger of the control's and the
  setting's own spread: 0.92 s for `short=1400:64`, 0.18 for `short=1023:64`). `short=1400:64` is the better of the
  two by 3.02 s a pass, all of it on the turns of 1,024 to 1,400 tokens that `short=1023:64` leaves to 512s.
- **Said plainly, `short=1400:64`'s "gain" at 2,000+ is not the taper's:** those three calls (2,007, 2,308 and 3,313
  tokens) are read in exactly the control's micro-batches in every pass, and the bucket's spread (0.04 s over three
  turns) is narrower than the drift between passes (`short=1023:64`'s two passes there differ by 0.24 s). Read it as
  no change. The same holds for both settings under 600, where the calls are split as the control's.
- **Replies:** the same greedy reply as the control's on 62 of 62 turns in every pass of both settings; the four
  control passes (the warm-up included) agree on 62 of 62 ([replies.tsv](replies.tsv)).

### Against C222's composition

C222 composed each setting turn by turn from its passes (est., assuming a turn's time does not depend on how the
turns before it were split). The measured passes bear it out ([calls.tsv](calls.tsv),
[run/r225_calls.py](run/r225_calls.py)):

| Setting | Measured, s a pass | Against the control | C222's est. | Turns changed |
|---|---|---:|---:|---:|
| `short=1023:64` | 210.77, 210.95 | −9.54 s | −9.54 s | 6 |
| `short=1400:64` | 208.30, 207.38 | −12.56 s | −12.60 s | 11 |

Each call of 601 to 2,048 tokens, the mean of its passes against the mean of the three controls (seconds), beside
C222's warm controls and its `short=2048:64` passes:

| Call tokens | 610 | 775 | 803 | 815 | 896 | 936 | 1,028 | 1,086 | 1,130 | 1,300 | 1,359 | 1,534 | 1,567 | 1,578 | 1,690 | 1,747 | 1,985 | 2,007 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 5.86 | 5.87 | 5.99 | 5.93 | 6.28 | 6.22 | 6.68 | 6.72 | 6.88 | 7.37 | 7.25 | 7.48 | 7.81 | 7.88 | 7.74 | 7.59 | 8.19 | 8.52 |
| `short=1023:64` | −2.31 | −1.46 | −1.71 | −1.48 | −1.44 | −1.07 | −0.04 | −0.06 | +0.02 | −0.05 | +0.02 | −0.00 | +0.03 | +0.01 | +0.03 | +0.01 | −0.03 | −0.02 |
| `short=1400:64` | −2.32 | −1.44 | −1.67 | −1.42 | −1.42 | −1.06 | −1.23 | −0.67 | −0.80 | −0.22 | −0.08 | −0.03 | +0.01 | +0.04 | −0.02 | +0.02 | −0.06 | −0.08 |
| C222, 64s | −2.40 | −1.46 | −1.68 | −1.43 | −1.48 | −1.09 | −1.24 | −0.79 | −0.76 | −0.22 | −0.05 | +0.47 | +0.17 | −0.02 | +0.79 | +1.15 | +1.81 | +1.71 |

Every call the limit moves to 64s is faster, by about what C222 measured; every call it leaves to 512s reads as the
control's, within 0.08 s. The 1,534-token call, the one call under 1,600 tokens that C222's 64s read slower
(+0.47 s), stays in 512s under `short=1400:64`.

A line through the turns of 512 to 1,999 new tokens ([fit.tsv](fit.tsv)): the control 2.82 s + 3.21 ms a token
(C222's warm controls: 2.83 s + 3.20 ms). The two settings' lines mix 64s and 512s within that range, so their
1.52 s + 4.00 ms and 1.42 s + 3.97 ms describe no single size.

## Files

| File | What |
|---|---|
| [passes.tsv](passes.tsv), [passes_with_warmup.tsv](passes_with_warmup.tsv) | per pass: the setting, turns, new and read tokens, prompt seconds and tok/s, the requests' wall seconds, turns whose read tokens differ from the plan, graph reallocations, calls logged, the largest micro-batch of a call of 601 to 2,048 tokens (512 in every pass: the calls of 1,401 to 2,048 tokens stay in 512s); without and with the warm-up pass |
| [buckets.tsv](buckets.tsv), [decision.md](decision.md) | per setting and size bucket: seconds of each pass, against the three control passes, the spread, the class (also by R60's rule); the classes in words |
| [turns.tsv](turns.tsv) | every replayed turn: task, turn, new tokens, the tokens its call read, prompt milliseconds in each measured pass |
| [replies.tsv](replies.tsv) | per setting: turns whose greedy reply is the control's in every pass |
| [fit.tsv](fit.tsv) | per setting: the fixed seconds a turn and milliseconds a token over the turns of 512 to 1,999 new tokens |
| [calls.tsv](calls.tsv) | each setting measured against C222's est., and every call of 601 to 2,048 tokens against this run's controls and C222's |
| [plans.tsv](plans.tsv), [plans_by_limit.tsv](plans_by_limit.tsv), [taper_lines.txt](taper_lines.txt) | the micro-batches the server logged for every call (`ubatch taper:` lines, in request order); per pass and call size the largest and the most common, and split at 1,023 and 1,400 tokens (every request, the warm-up and the unmeasured prefix requests included) |
| [requests/load-A.tsv](requests/load-A.tsv) | every request as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/r225_calls.py](run/r225_calls.py) | the comparison with C222's composition |
| [run/r225_plans.py](run/r225_plans.py) | the micro-batch check by call size |

The driver and the tables script are C222's: [run/c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh),
[run/c222_tables.py](../2026-11-20_c222_flash_microbatch/run/c222_tables.py).

## Commands

A folder on the rig (`WORK`) holds `rig_taper.sh`, `taper_replay.py`, `turns.csv` and the v5 preset's
`settings.env` (the md5s above). The engine's rig tarball is unpacked in `$BUILDS/$BUILD`, the model's first shard is
in `MODEL` (shards 2 and 3 beside it) and the MTP draft in `DRAFT`. From a machine that only runs ssh:

```bash
export RIG=<ssh alias> WORK=<folder on the rig> BUILDS=<folder of the builds> \
  BUILD=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 MODEL=<first shard> DRAFT=<MTP draft> OUTN=r225 \
  PASSES="short=600:64 short=600:64 short=1400:64 short=1023:64 short=600:64 short=1023:64 short=1400:64 short=600:64"
DRY=1 bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh    # prints the rig command
bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh
# done when $WORK/r225/DONE exists; then copy $WORK/r225/load-A off the rig
C=docs/data/2026-11-20_c222_flash_microbatch; R=docs/data/2026-11-20_r225_flash_taper1400
python3 -I $C/run/c222_tables.py --load <copy>/load-A --control short=600:64 --drop-pass 0 --out tables
python3 -I $C/run/c222_tables.py --load <copy>/load-A --control short=600:64 --out tables_all   # passes_with_warmup.tsv
python3 -I $R/run/r225_calls.py --r225 tables/turns.tsv --c222 $C/turns.tsv > calls.tsv
python3 -I $R/run/r225_plans.py --load <copy>/load-A > plans_by_limit.tsv
```

## Provenance and licence

The engine is llama.cpp's licence (MIT). The replay's text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's measurements.
