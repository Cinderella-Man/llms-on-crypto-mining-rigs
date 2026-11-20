# C222: Flash-Next's whole prompt calls in micro-batches of 128 and 64 (2026-11-20)

Registry row [Q775](../../../quants/experiments.md) (Flash-Next's arm; with Q777, Q1235 and Q1128 merged in). Nine
P106 cards run Qwen3.8-Flash-Next as a pipeline, so a prompt call ends after the sum of its micro-batches plus about
eight times the largest one ([the taper's arithmetic](../2026-11-04_prompt_taper/README.md#what-the-pipeline-arithmetic-says-est)).
The v5 preset already reads calls of at most 600 tokens in micro-batches of 64 (`--ubatch-taper short=600:64`,
[R60](../2026-11-12_r60_taper_v4/README.md)); this replay tries every call of up to 2,048 tokens in micro-batches of
128 and of 64, where [the v5 review of the prompt path](../../research/notes/v5_review_prompt_path_2026-11.md) priced a
fixed est. 4.3 s on every turn of 512 to 1,999 new tokens.

**Turns of 600 to 1,023 new tokens read about a quarter faster with either size; the gain does not hold for every
turn size, so neither setting goes into the preset as it is.** Each pass replays the agent check's 62 recorded
prompts (12 tasks) on the v5 preset; only the taper changes. The control (`short=600:64`) read them in 230.86, 220.09
and 220.28 s, `short=2048:128` in 216.09 and 213.98 s, `short=2048:64` in 213.78 and 213.49 s, with the same greedy
replies on 62 of 62 turns in every pass. By the rule set before the run (the control passes' difference as the
spread, 10.77 s) both settings are negligible over the whole pass. That spread is the load's first pass: it read
10.77 and 10.58 s slower than the two later controls, which differ by 0.19 s, and against those two both settings are
faster in all four of their passes (−5.15 and −6.55 s a pass, −2.34 and −2.98 %). By the size of the turn:

- **600 to 1,023 new tokens** (6 turns): 36.21 and 36.25 s a pass with the control, 28.08 and 27.41 with 128s,
  26.75 and 26.62 with 64s, a gain for both (the 610-token turn 5.93 s → 3.54 s).
- **1,024 to 1,999** (11 turns): the 128s gain (78.31 and 77.33 s against 81.39 and 81.54); the 64s read them in
  82.91 and 82.66 s. By the call (one turn resumes from a checkpoint and reads 1,747 tokens for 1,231 new), every
  call of 1,028 to 1,359 tokens is faster with either size, and the calls of 1,690 tokens or more are slower with the
  64s (by 0.79 to 1.81 s) and of 1,747 or more with the 128s.
- **2,000 and more** (3 turns): only the 2,006-token turn changes (the other two are calls over 2,048 tokens), and it
  is slower with either size: 8.48 and 8.41 s with the control, 9.23 and 9.25 with 128s, 10.15 and 10.15 with 64s.
- **Under 600** (42 turns): the 64s read these calls exactly as the control does (71.84 and 71.98 s against 71.97 and
  72.04); the 128s take them out of the 64s and lose 6.13 s, on turns of 100 to 499 tokens.

**What the data points to (est.):** the 64s up to about 1,400 tokens, `short=1400:64`, one setting and no code: est.
207.58 s a pass against the warm controls' 220.19 (−12.60 s, −5.72 %), about 1 s of prompt reading a task; it changes
11 turns, and each of them is faster in both 64 passes. It is composed turn by turn from the measured passes
([rules.tsv](rules.tsv)) and is not yet measured as such: a replay of it with a warm-up pass first (R225,
[Q1637](../../../quants/experiments.md)), then its agent pair if it gains. Two code changes to the taper would add
less: a second size rule (64 up to 600 tokens, 128 up to 1,600) est. 0.67 s a pass more
([Q1639](../../../quants/experiments.md)), and the 9-token floor applied after the in-call checkpoint cuts, which now
leave micro-batches of 4 to 6 tokens in some calls, the preset's own included ([Q1638](../../../quants/experiments.md)).

## The run

| | |
|---|---|
| Rig | the second rig, nine P106-090 cards, with its soaked clock profile ([R219](../2026-11-19_r209_oc_v5/README.md#the-profile-again-on-the-boot-of-2026-11-20)) and the kit's watchdog on |
| Engine | `pascal-p106-v5` `602691153` (upstream `b12483` merged), `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b` |
| Model | UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with its MTP draft |
| Preset | [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env) as it was at the start (md5 `90081fd2c14138552a7f2d77c815857e`): its 40 switches and its whole line (`-b 4096 -ub 512`, in-call checkpoints, the server's taper `short=600:64`). Later that day the preset took the sparse attention from 6,153 cells (`GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=0.75`, [R220](../2026-11-20_r220_flash_ratio075/README.md)); this run has 1.0, and only one replayed turn ends past 6,153 cells (6,210) |
| Replay | [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (md5 `353ec109ef8b5806c38e7c14a1c23dfb`) with [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`) on the agent check's [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`): per task the shared 1,549-token prefix, then each turn's new tokens of Python standard-library source and 8 greedy tokens; the same tokens in every pass |
| Passes | one load, seven passes, the taper set per request (`"ubatch_taper"`): `short=600:64`, `short=2048:128`, `short=2048:64`, `short=600:64`, `short=2048:64`, `short=2048:128`, `short=600:64`; no fresh prompts |
| Time | started 14:44:35, server up after 226 s, passes 14:48 to 15:19:56 (299, 264, 263, 270, 263, 262 and 270 s of wall time); nothing else ran on the rig |

**How the taper splits a call.** With `-b 4096` and in-call checkpoints each replayed turn is one llama-server call
(511 requests, 511 taper lines), started with nothing in flight and waited for, so `short=N:S` reads a call of N
tokens or fewer in micro-batches of S and leaves longer calls to the plain split (`-ub 512`); the checkpoints inside a
call add their own cuts (the 5 and 4 below). A spec holds one short rule, so `short=2048:128` also reads the calls
under 600 tokens in 128s. The server's own log lines ([taper_lines.txt](taper_lines.txt), summed in
[plans.tsv](plans.tsv)) show that every pass ran as set:

| Call | Control (`short=600:64`) | `short=2048:128` | `short=2048:64` |
|---|---|---|---|
| 33 tokens | `29 4` | `29 4` | `29 4` |
| 610 tokens | `94 512 4` | `94 34 128x3 94 4` | `64 30 34 64x7 30 4` |
| 896 tokens | `380 512 4` | `128x2 124 5 127 128x2 124 4` | `64x5 60 5 63 64x6 60 4` |
| 1,985 tokens | `512x2 445 512 4` | `128x11 61 67 128x3 61 4` | `64x22 61 5 62 64x6 61 4` |
| 3,313 tokens | `512x5 237 512 4` | the same | the same |
| largest micro-batch of a call of 601 to 2,048 tokens | 512 | 131 | 67 |

Where an in-call checkpoint cut (516 tokens before a call's end) lands a few tokens short of a planned micro-batch
end, the plan leaves a piece of 4 to 6 tokens, below the taper's 9-token floor, which the plan applies before the cuts:
in the control on the 577- and 583-token calls (`61 5 62 64x6 61 4`), with 128s on the 775-, 896- and 1,534-token
calls, with 64s on the 775-, 896-, 1,086-, 1,534- and 1,985-token calls. Pieces of 1 to 8 tokens run the
matrix-vector kernels (the taper's [one-token batches](../2026-11-04_prompt_taper/README.md#one-token-batches)). The
1,534-token call is the one call under 1,600 tokens that both sizes read slower (+0.34 and +0.47 s).

Every pass read the same tokens: 36,793 a pass for 36,227 planned (50 turns read the last cached token again, one
resumes from a checkpoint 516 tokens back, the same in every pass). No graph reallocation in any of the 511 requests.
Lowest free host memory 274 MiB (the replay's watchdog line is 60). The clock watchdog never tripped.

## Results

Prompt seconds a pass over the 62 turns, by the size of a turn's new prompt ([passes.tsv](passes.tsv),
[buckets.tsv](buckets.tsv)):

| Pass | Taper | All | Under 600 | 600–1,023 | 1,024–1,999 | 2,000+ |
|---:|---|---:|---:|---:|---:|---:|
| 0 | `short=600:64` (control, the load's first pass) | 230.86 | 75.48 | 41.17 | 83.25 | 30.96 |
| 1 | `short=2048:128` | 216.09 | 78.33 | 28.08 | 78.31 | 31.38 |
| 2 | `short=2048:64` | 213.78 | 71.84 | 26.75 | 82.91 | 32.28 |
| 3 | `short=600:64` (control) | 220.09 | 71.97 | 36.21 | 81.39 | 30.52 |
| 4 | `short=2048:64` | 213.49 | 71.98 | 26.62 | 82.66 | 32.22 |
| 5 | `short=2048:128` | 213.98 | 77.95 | 27.41 | 77.33 | 31.28 |
| 6 | `short=600:64` (control) | 220.28 | 72.04 | 36.25 | 81.54 | 30.44 |
| | turns, new tokens | 62, 36,227 | 42, 8,288 | 6, 4,834 | 11, 15,480 | 3, 7,625 |

### The verdict

The class is the mean of a setting's passes minus the mean of the control's, against the control passes' largest
difference (the spread): gain below minus the spread, loss above it, negligible between.

| Setting | By the rule set before the run (all three controls) | Without the load's first pass (controls 3 and 6) |
|---|---|---|
| `short=2048:128` | **negligible**: −8.71 s, −3.89 % (spread 10.77); under 600 loss (+4.97, spread 3.52), 600–1,023 gain (−10.13, 4.96), 1,024–1,999 gain (−4.24, 1.86), 2,000+ loss (+0.69, 0.51) | gain: −5.15 s, −2.34 % (spread 0.19); under 600 loss (+6.13), 600–1,023 gain (−8.48), 1,024–1,999 gain (−3.65), 2,000+ loss (+0.85) |
| `short=2048:64` | **negligible**: −10.11 s, −4.52 % (spread 10.77); under 600 negligible (−1.25), 600–1,023 gain (−11.19), 1,024–1,999 negligible (+0.72), 2,000+ loss (+1.61) | gain: −6.55 s, −2.98 % (spread 0.19); under 600 −0.09 (the control's own splits: noise), 600–1,023 gain (−9.54), 1,024–1,999 loss (+1.32), 2,000+ loss (+1.77) |

- **The rule's spread measured the load's warm-up, not the noise.** The control went first so that it would also be
  last, but the first pass of a load reads slower (R60's first pass was 11.98 s slower than its last): here 4.27 s of
  it is the load's first measured turn (the 896-token turn 1 of JavaScript `complex-numbers`, 10.59 s against 6.30 and
  6.33 s in passes 3 and 6). A replay should start with an unmeasured pass. The warm view leaves pass 0 out
  ([buckets_warm.tsv](buckets_warm.tsv), [decision_warm.md](decision_warm.md)); its spread comes from two passes.
- **Neither setting is a preset candidate as it is:** each loses in a size bucket under either view (R60's rule:
  no bucket lost), on turns that the taper should leave to 512s or 64s.
- **Replies:** the same greedy reply as the control's on 62 of 62 turns in every pass of both settings; the three
  control passes agree on 62 of 62 ([replies.tsv](replies.tsv)).

### The fixed cost of a turn

A least-squares line through the turns of 512 to 1,999 new tokens, every pass of a setting together
([fit.tsv](fit.tsv), [fit_warm.tsv](fit_warm.tsv)): the control 2.83 s + 3.20 ms a token (the warm passes; 3.05 s +
3.11 ms with pass 0), the 128s 1.15 s + 4.20 ms, the 64s 0.80 s + 4.72 ms. Small micro-batches take away most of the
fixed cost and charge 1.00 and 1.52 ms more a token, so they pay up to about 1,680 tokens (128s) and 1,340 tokens (64s)
on the fit, and the 64s beat the 128s below about 670. The review's fixed est. 4.3 s came from v4.0's agent runs (the
two buckets' means); on this replay the control's line holds 2.83 s.

### Other settings, composed from these passes (est.)

A turn is one call, and the taper picks a call's splits from the call's length alone (the tokens it reads, the
server's `prompt_n`), so under `short=N:64` a call of 601 to N tokens is read exactly as in the `short=2048:64` passes
and every other call as in the control's. Summing, turn by turn, the means of the passes whose splits a setting would
give ([rules.tsv](rules.tsv), [run/c222_rules.py](run/c222_rules.py); the warm controls, 220.19 s a pass). Est.: it
assumes a turn's time does not depend on how the turns before it were split.

| Setting | est. s a pass | est. against the control | turns changed |
|---|---:|---:|---:|
| `short=1023:64` | 210.64 | −9.54 s, −4.33 % | 6 |
| `short=1200:64` | 207.86 | −12.33 s, −5.60 % | 9 |
| `short=1300:64` | 207.64 | −12.55 s, −5.70 % | 10 |
| `short=1400:64` | 207.58 | −12.60 s, −5.72 % | 11 |
| `short=1600:64` | 208.21 | −11.98 s, −5.44 % | 14 |
| `short=2048:64` (measured: 213.78, 213.49) | 213.67 | −6.52 s, −2.96 % | 18 |
| two rules, 64 up to 600 and 128 up to 1,600 (needs code) | 206.92 | −13.27 s, −6.03 % | 14 |

Each call of 601 to 2,048 tokens against the warm controls (seconds; the end of [rules.tsv](rules.tsv)):

| Call tokens | 610 | 775 | 803 | 815 | 896 | 936 | 1,028 | 1,086 | 1,130 | 1,300 | 1,359 | 1,534 | 1,567 | 1,578 | 1,690 | 1,747 | 1,985 | 2,007 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 5.93 | 5.86 | 5.97 | 5.92 | 6.31 | 6.24 | 6.68 | 6.63 | 6.85 | 7.32 | 7.23 | 7.48 | 7.81 | 7.92 | 7.74 | 7.61 | 8.20 | 8.44 |
| 128s | −2.39 | −1.23 | −1.39 | −1.19 | −1.21 | −1.07 | −1.59 | −1.18 | −1.21 | −0.31 | −0.24 | +0.34 | −0.22 | −0.37 | −0.04 | +0.18 | +1.00 | +0.80 |
| 64s | −2.40 | −1.46 | −1.68 | −1.43 | −1.48 | −1.09 | −1.24 | −0.79 | −0.76 | −0.22 | −0.05 | +0.47 | +0.17 | −0.02 | +0.79 | +1.15 | +1.81 | +1.71 |

`short=1400:64` stands at the fit's crossover for the 64s and leaves the calls from 1,534 tokens on to 512s;
`short=1300:64` differs from it only on the 1,359-token call (−0.05 s with the 64s). Under 600 tokens
the 64s are the right size: the 128s lose 3.92 s a pass on the 7 turns of 100 to 299 tokens and 2.29 s on the 10 of
300 to 499, and win 0.65 s on the 4 of 500 to 599.

## Files

| File | What |
|---|---|
| [passes.tsv](passes.tsv) | per pass: the setting, turns, new and read tokens, prompt seconds and tok/s, the requests' wall seconds, turns whose read tokens differ from the plan, graph reallocations, calls logged, the largest micro-batch of a call of 601 to 2,048 tokens |
| [buckets.tsv](buckets.tsv), [decision.md](decision.md) | per setting and size bucket: seconds of each pass, against the three control passes, the spread, the class (also by R60's rule, which takes the larger of the control's and the setting's own spread); the classes in words |
| [buckets_warm.tsv](buckets_warm.tsv), [decision_warm.md](decision_warm.md), [fit_warm.tsv](fit_warm.tsv) | the same without pass 0 |
| [turns.tsv](turns.tsv) | every replayed turn: task, turn, new tokens, the tokens its call read, prompt milliseconds in each of the seven passes |
| [replies.tsv](replies.tsv) | per setting: turns whose greedy reply is the control's in every pass |
| [fit.tsv](fit.tsv) | per setting: the fixed seconds a turn and milliseconds a token over the turns of 512 to 1,999 new tokens |
| [plans.tsv](plans.tsv), [taper_lines.txt](taper_lines.txt) | the micro-batches the server logged for every call (`ubatch taper:` lines, in request order), and per pass and call size the largest and the most common |
| [rules.tsv](rules.tsv) | other settings composed from the passes (est.), the turns under 600 tokens by finer size, and every call of 601 to 2,048 tokens against the warm controls |
| [requests/load-A.tsv](requests/load-A.tsv) | every request as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/c222.sh](run/c222.sh) | starts the load on the rig, detached, with the v5 preset and the seven passes |
| [run/c222_tables.py](run/c222_tables.py) | every table above from `requests.tsv` and the server's log |
| [run/c222_rules.py](run/c222_rules.py) | the composed settings |

## Commands

A folder on the rig (`WORK`) holds `rig_taper.sh`, `taper_replay.py`, `turns.csv` and the v5 preset's
`settings.env` (the md5s above). The engine's rig tarball is unpacked in `$BUILDS/$BUILD`, the model's first shard is
in `MODEL` (shards 2 and 3 beside it) and the MTP draft in `DRAFT`. From a machine that only runs ssh:

```bash
RIG=<ssh alias> WORK=<folder on the rig> BUILDS=<folder of the builds> \
  BUILD=llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483 MODEL=<first shard> DRAFT=<MTP draft> \
  DRY=1 bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh    # prints the rig command
RIG=... (the same) bash docs/data/2026-11-20_c222_flash_microbatch/run/c222.sh
# done when $WORK/c222/DONE exists; then copy $WORK/c222/load-A off the rig
python3 -I docs/data/2026-11-20_c222_flash_microbatch/run/c222_tables.py --load <copy>/load-A --control short=600:64 --out tables
python3 -I docs/data/2026-11-20_c222_flash_microbatch/run/c222_tables.py --load <copy>/load-A --control short=600:64 --drop-pass 0 --out tables_warm
python3 -I docs/data/2026-11-20_c222_flash_microbatch/run/c222_rules.py tables/turns.tsv > rules.tsv
```

`c222_tables.py` checked on R60's published load first: it gives R60's pass seconds (492.68, 486.04, 473.82, 469.17,
480.70) and replies (61 of 62 for `tail=512:128,short=600:64`) again.

## Provenance and licence

The engine is llama.cpp's licence (MIT). The replay's text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's measurements.
