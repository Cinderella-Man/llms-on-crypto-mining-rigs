# R202b: a ladder of core soaks on the second rig, all nine cards clean for 90 minutes with a core offset on every card (2026-11-18)

**The answer in plain words.** On one fresh boot of the second nine-card P106-090 rig, with nobody at the console, every
card's memory sat at the use profile that had held its soak ([R201](../2026-11-18_r201_soak_ladder/README.md)) while
the core of six cards went up one climb step every 30 minutes, to the value two steps under where each had failed in
[R202](../2026-11-18_r202_core_climb_rig2/README.md)'s core climb ([Q1588](../../../quants/experiments.md), the kit's
[soak ladder](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check)).
Cards 0, 4 and 8 kept their core values throughout. **No card ever differed from stock:** 465 rounds on each of the
nine cards, 155 a rung, 4,185 checks, every one bit-identical to the stock answer. The watchdog never tripped, no card
reported an error to the driver, no card slowed down, and the hottest card reached 46 °C. **Every card now has a core
offset that held 30 minutes at its top value:** the new use profile, [use-profile.json](use-profile.json), is the top
rung, +256, +256, +269, +231, +206, +206, +282, +256 and +231 MHz on cards 0 to 8, each with R201's memory value.

**The setting for the next boot** ([use-profile.json](use-profile.json), made by [run/make_use.py](run/make_use.py)
from [soaked.json](soaked.json)): each card's highest clean rung, but no higher than two climb steps (four of the
driver's 12.66 MHz core steps) under the lowest core offset at which it ever failed, here or in
[R200](../2026-11-17_r200_climb_rig2/README.md) and R202. The memory did not change between rungs, so each card keeps
R201's memory value, which is already two steps under its memory failures.

| Card | Memory (R201's use profile) | Highest clean rung (core) | Lowest core failure | Cap (two climb steps under it) | **Use** |
|---|---|---|---|---|---|
| 0 | +1,300 MT/s | +256 MHz (rung 3) | +307 MHz, R200's climb (memory +1,400) | +256 | **+1,300 MT/s, +256 MHz** |
| 1 | +1,000 | +256 (rung 3) | +307, R202's climb | +256 | **+1,000, +256** |
| 2 | +1,000 | +269 (rung 3) | +320, R202's climb | +269 | **+1,000, +269** |
| 3 | +1,100 | +231 (rung 3) | +282, R202's climb | +231 | **+1,100, +231** |
| 4 | +1,100 | +206 (all rungs) | +282, R200's climb (memory +1,400) | +231 | **+1,100, +206** |
| 5 | +1,200 | +206 (rung 3) | +256, R202's climb | +206 | **+1,200, +206** |
| 6 | +900 | +282 (rung 3) | +332, R202's climb | +282 | **+900, +282** |
| 7 | +900 | +256 (rung 3) | +307, R202's climb | +256 | **+900, +256** |
| 8 | +1,300 | +231 (all rungs) | +282, R200's climb (memory +1,400) | +231 | **+1,300, +231** |

Card 4 stays at +206 MHz, under its cap: it is the only core value it ran here. Its +231 would sit under the cap too,
but R200's first soak failed with card 4 at +1,300 MT/s and +231 MHz together, which R201 counted against the memory;
+231 with the memory at +1,100 has not run. Cards 2 and 3's R200 core failures (+16 and +104 MHz with the memory at
+1,400 MT/s) were memory errors, as R202 showed, and set no cap.

**What it means.**

- **The kit's margins held on every card.** Each card's memory sat at R201's use profile, at most two steps under its
  memory failures, and its core rose to two climb steps under its core failure; nothing differed in 90 minutes. A
  two-step margin is evidence, not a guarantee: in R201's ladder card 6 erred at +1,100 MT/s, two steps under its
  R200 failure at +1,300.
- **Thirty clean minutes at the top rung is still a weak proof.** No error in 155 rounds shows only that each of the
  six cards that rose errs less often than about once in 10 minutes at its top value (95 % confidence); cards 0, 4 and
  8 ran their values for all 465 rounds, so less often than about once in 30 minutes. Daily use needs clean hours and
  the served model with the profile: Ornith's whole-memory check and its agent runs, on a fresh boot.
- **The peak clock for the same offset moves.** In rung 3 seven of the nine cards peaked one driver step
  (12.66 MHz) lower than at the same offsets in R202 (cards 1 and 6 the same). Cards 0, 4 and 8, whose offsets never
  changed, peaked lower in rungs 2 and 3 than in rung 1: one step on cards 0 and 4, two on card 8 (2,138 to
  2,113 MHz). The offset is what the driver is given; the peak it
  reaches is not a fixed property of a card.
- **With a core offset on all nine cards, every card had power readings above 80 W:** 134 half-second samples in the
  90 minutes, 8 to 21 a card, never two in a row, against the watchdog's limit of 80 W for 5 s. The mean power of each
  card in a rung was 44.59 to 47.84 W.

## What was done

The second nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off),
booted at 18:46:57 after R202, with nothing run on it before: every offset 0, no load, 0 MiB on every card, no Xid
this boot, no step-down in the kit's ledger, 3,180 MiB of host memory available; the kit's start check passed. The kit
as published on this branch ([run/MD5SUMS](run/MD5SUMS): every kit file, the settings, the rungs, the input profile,
the check binaries and the server binary, checked on the rig), with R200's settings ([run/env.sh](run/env.sh)) and this
run's in [run/env-r202b.sh](run/env-r202b.sh). The engine build was `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`.

`soak_chain.sh` ran the whole boot detached, with no served model (`MODEL` empty):

1. **The start check** (18:52:23): PASS.
2. **The watchdog, then `climb.py apply` with rung 1** (18:52:27 to 18:57:54): every card's stock check, the offsets
   set and read back, and one check round at rung 1: every card passed the greedy check and the 31 kernel checks.
3. **The ladder** (`soak.py --minutes 30 --keep-going --ladder`, 18:57:54 to 20:30:47): one check server per card with
   the check model (Granite 3.1 3B-A800M Instruct, Q4_K_M), ready at 19:00:20, then every round each card's greedy
   check (two prompts, 256 tokens each, temperature 0, every token's top-10 log-probabilities hashed) against its stock
   hash. After each 30-minute rung the cards that had not differed went up to the next rung's values, read back. A card
   that differed would have stayed where it was and gone on being checked; a card that differed three times, or whose
   check failed or ran past its 45-second limit, would have been set to 0 alone. None did.
4. **The end:** the chain ended with "the soak passed; no model reference (skipped): the offsets stay set with the
   watchdog" (20:30:47). Every offset of every card was then set to 0 and read back (`climb.py reset-all`, 20:33:22 to
   20:33:32), the watchdog stopped after 0 trips (20:33:32), and the state read back: every offset 0, P8, no load, no
   Xid this boot, 3,135 MiB of host memory available, 1,911/4,004 MHz maximum on all nine in `nvidia-smi`.

The rungs ([run/make_rungs.py](run/make_rungs.py), [run/r202b-rung1.json](run/r202b-rung1.json) to
[run/r202b-rung3.json](run/r202b-rung3.json)) take every card's memory from R201's use profile and the climbing
cards' core values from R202's own climb steps ([R202's tables/steps.csv](../2026-11-18_r202_core_climb_rig2/tables/steps.csv)):
rung 3 is R202's suggested core, two climb steps under each card's first failure, and rungs 1 and 2 are two and one
climb steps under it. Core in MHz on cards 0 to 8:

| Rung | Core | Memory (MT/s, every rung) |
|---|---|---|
| 1 | +256, +206, +218, +180, +206, +155, +231, +206, +231 | +1,300, +1,000, +1,000, +1,100, +1,100, +1,200, +900, +900, +1,300 |
| 2 | +256, +231, +244, +206, +206, +180, +256, +231, +231 | the same |
| 3 | +256, +256, +269, +231, +206, +206, +282, +256, +231 | the same |

## Results

### Every card and rung ([tables/rungs.csv](tables/rungs.csv), [tables/cards.csv](tables/cards.csv))

Rung 1 ran 19:00:20 to 19:30:30, rung 2 19:30:35 to 20:00:38, rung 3 20:00:43 to 20:30:44. Each cell is the card's
core offset, its rounds and its differing checks in that rung; peak core clock from the watchdog's samples.

| Card | Rung 1 | Rung 2 | Rung 3 | Highest clean rung | Peak core clock, rungs 1, 2, 3 |
|---|---|---|---|---|---|
| 0 | +256: 155 rounds, 0 | +256: 155, 0 | +256: 155, 0 | 3 | 2,126, 2,113, 2,113 MHz |
| 1 | +206: 155, 0 | +231: 155, 0 | +256: 155, 0 | 3 | 2,100, 2,113, 2,138 |
| 2 | +218: 155, 0 | +244: 155, 0 | +269: 155, 0 | 3 | 2,100, 2,113, 2,138 |
| 3 | +180: 155, 0 | +206: 155, 0 | +231: 155, 0 | 3 | 2,050, 2,062, 2,088 |
| 4 | +206: 155, 0 | +206: 155, 0 | +206: 155, 0 | 3 | 2,113, 2,100, 2,100 |
| 5 | +155: 155, 0 | +180: 155, 0 | +206: 155, 0 | 3 | 2,050, 2,062, 2,088 |
| 6 | +231: 155, 0 | +256: 155, 0 | +282: 155, 0 | 3 | 2,113, 2,126, 2,151 |
| 7 | +206: 155, 0 | +231: 155, 0 | +256: 155, 0 | 3 | 2,100, 2,113, 2,138 |
| 8 | +231: 155, 0 | +231: 155, 0 | +231: 155, 0 | 3 | 2,138, 2,113, 2,113 |

All 4,185 checks gave the stock hash `60ee0f178cfd67d5` on every card ([tables/soak.csv](tables/soak.csv)). The
slowest check took 14.5 s of its 45 s limit. The stock peaks read on this boot in `apply`'s stock round were 1,873,
1,898, 1,885, 1,873, 1,898, 1,898, 1,885, 1,898 and 1,898 MHz on cards 0 to 8 (R202's boot read 1,873 on card 6 and
1,911 on card 8), so rung 3's target peaks were 2,126, 2,151, 2,151, 2,101, 2,101, 2,101, 2,164, 2,151 and 2,126 MHz.

### Apply's check, streaming reads ([tables/apply.csv](tables/apply.csv))

Every card passed the greedy check and the 31 kernel checks at stock (18:56:52) and at rung 1 (18:57:44). Streaming
reads in GB/s, cards 0 to 8, at stock: 179.18, 179.14, 179.38, 179.21, 179.08, 179.23, 179.37, 178.97, 179.32; at
rung 1: 206.82, 199.41, 199.20, 202.04, 201.92, 204.62, 197.55, 197.66, 206.90.

### Heat and power ([tables/heat_power.csv](tables/heat_power.csv))

From the watchdog's half-second samples over the three rungs:

| Card | Peak temperature | Peak power | Samples above 80 W | Mean power, rungs 1, 2, 3 |
|---|---|---|---|---|
| 0 | 44 °C | 81.46 W | 15 | 47.26, 47.19, 47.25 W |
| 1 | 42 | 83.93 | 21 | 47.44, 47.77, 47.84 |
| 2 | 39 | 81.98 | 16 | 45.58, 45.56, 45.65 |
| 3 | 37 | 82.70 | 10 | 45.04, 44.97, 45.06 |
| 4 | 43 | 83.53 | 20 | 45.69, 45.72, 45.71 |
| 5 | 39 | 81.45 | 8 | 44.59, 44.76, 45.00 |
| 6 | 46 | 82.90 | 11 | 45.57, 46.04, 45.83 |
| 7 | 41 | 83.71 | 12 | 44.99, 45.19, 45.41 |
| 8 | 39 | 82.72 | 21 | 46.08, 45.83, 46.03 |

No reading above 80 W was followed by another; no card was in a thermal or hardware slowdown in any sample. The
highest reading of the boot was 84.57 W, card 3 in `apply`, a single sample. The watchdog's start and stop lines are
in [tables/watchdog.txt](tables/watchdog.txt) (0 trips).

## The commands

The rungs, from public files (`python3 -I`, any machine):

```bash
D=docs/data
python3 -I $D/2026-11-18_r202b_core_soak/run/make_rungs.py $D/2026-11-18_r201_soak_ladder/use-profile.json \
  $D/2026-11-18_r202_core_climb_rig2/tables/steps.csv $D/2026-11-18_r202_core_climb_rig2/profile.json OUT_DIR
```

In the kit folder on the freshly booted rig, as root, with the three rungs and `env-r202b.sh` copied there:

```bash
md5sum -c MD5SUMS                     # run/MD5SUMS
. ./env.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/r202b && secs=$(( $(date -d 21:55 +%s) - $(date +%s) ))     # a backstop to END_BY=21:45
ENV_FILE=./env-r202b.sh setsid nohup timeout ${secs}s bash soak_chain.sh > /var/tmp/r202b/chain.console 2>&1 < /dev/null &
cat /var/tmp/r202b/CHAIN_DONE         # when it exists
. ./env.sh && export WORK=/var/tmp/r202b && python3 climb.py reset-all && python3 climb.py watchdog stop && python3 climb.py state
```

The chain started at 18:52:23 and ended by itself at 20:30:47. `env-r202b.sh` is passed as `ENV_FILE` because
`soak_chain.sh` otherwise sources `./env-chain.sh`, which on this rig still held R201's settings (with Ornith as
`MODEL`). A `soak_chain.sh` whose soak passes leaves the offsets set with the watchdog running; here they were
set to 0 at once, since the rig's next use is a fresh boot.

The tables and the use profile, from the work folder copied off the rig:

```bash
python3 -I run/tables.py WORK_COPY tables
python3 -I run/make_use.py WORK_COPY/soak/soaked.json ../2026-11-18_r201_soak_ladder/use-profile.json \
  ../2026-11-18_r202_core_climb_rig2/tables/steps.csv use-profile.json
```

## Files

| File | What |
|---|---|
| [use-profile.json](use-profile.json) | the setting for the next boot: each card's memory from R201's use profile, its core at its highest clean rung within its cap (md5 `b32956cd064d8b449bdd6ac2b8a6d4f7`) |
| [soaked.json](soaked.json) | `soak.py`'s result: each card's highest clean rung and its per-rung rounds and checks (md5 `d5011030b37f8f7fde7e44104eeab3d2`) |
| [tables/cards.csv](tables/cards.csv) | one row per card: each rung's offsets and differing checks, the value `soaked.json` holds |
| [tables/rungs.csv](tables/rungs.csv) | one row per rung and card: offsets, minutes, rounds, passes, differing checks |
| [tables/soak.csv](tables/soak.csv) | one row per round and card: hash, seconds, verdict, rung, offsets (4,185 rows) |
| [tables/apply.csv](tables/apply.csv) | `apply`'s stock round and its check round at rung 1: peaks, kernel checks, streaming reads |
| [tables/heat_power.csv](tables/heat_power.csv) | per phase and card: temperature, power, samples above 80 W, slowdowns, peak core and memory clocks |
| [tables/watchdog.txt](tables/watchdog.txt) | the watchdog's start and stop lines |
| [run/env.sh](run/env.sh), [run/env-r202b.sh](run/env-r202b.sh), [run/MD5SUMS](run/MD5SUMS) | the rig's settings as sourced, and the md5 of every file that ran |
| [run/r202b-rung1.json](run/r202b-rung1.json), [rung2](run/r202b-rung2.json), [rung3](run/r202b-rung3.json) | the three rungs |
| [run/make_rungs.py](run/make_rungs.py), [run/make_use.py](run/make_use.py), [run/tables.py](run/tables.py) | the scripts that made the rungs, the use profile and the tables |

## Provenance and licence

Measured on the rig on 2026-11-18 with the engine build `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(Miner Llama v4's release candidate 2) and the kit as published on this branch. The check model is IBM's
Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; nothing of it is published here, only its name and
the hashes of its answers. The tables are this repository's licence.
