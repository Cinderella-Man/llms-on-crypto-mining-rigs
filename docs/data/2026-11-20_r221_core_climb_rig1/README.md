# R221: the first rig's core climb from its soaked memory, step 3 of its own profile (2026-11-20)

**The answer in plain words.** On a fresh boot of the first nine-card P106-090 rig, every card's memory went straight
to the value that held step 2's soak ([R218](../2026-11-20_r218_soak_rig1/README.md)), all nine passed one check
there, and then every card's core climbed together, aimed at the same peak clock, until each gave its first wrong
answer ([Q1566](../../../quants/experiments.md), the kit's
[`climb.py --from-profile`](../../../bench/mem_offset/README.md#a-fair-core-climb-climbpy---from-profile)). **Eight
cards climbed far:** their last clean core offsets are +294, +282, +282, +231, +231, +256, +307 and +256 MHz on cards
0, 1, 2, 3, 4, 5, 7 and 8, where they reached peaks of 2,151, 2,126, 2,164, 2,126, 2,100, 2,164, 2,214 and
2,126 MHz.
**Card 6 failed at its first core step,** +28 MHz, a 1,898 MHz peak, below the 1,911 MHz that four other cards of this
rig reach at stock, so it keeps core 0 for now. Every failure was a wrong answer except card 7's at the climb's top step, +332 MHz
(2,240 MHz), where the check hung and was stopped by its time limit. No driver error (0 Xid), no kernel check failed,
the watchdog never tripped, and every card ended at 0. The sitting took 15 minutes 27 seconds. The hottest card was
45 °C (card 1) and the highest power 81.19 W in a half-second sample (card 6, in the check round with its core at 0; two
samples in a row above 80 W).

**The result** ([profile.json](profile.json)): every card's memory at R218's value, and its core one climb step
(two driver steps, 25.32 MHz) under its last clean step, so two climb steps under its first failure: +269, +256,
+256, +206, +206, +231, 0, +282 and +231 MHz on cards 0 to 8. These are the top of step 4's core soak ladder, not
settings: one check a step shows where a card fails at once, and the soaks of both rigs failed one or two steps under
a climb's last clean step.

**What it means.**

- **The first rig's cores go about as far as the second rig's.** On the second rig the six cards that climbed failed
  at target peaks of 2,152 and 2,202 MHz ([R202](../2026-11-18_r202_core_climb_rig2/README.md)); here eight cards
  failed at 2,139 (card 4), 2,164 (cards 1, 3 and 8), 2,189 (cards 0 and 5), 2,214 (card 2) and 2,240 MHz (card 7).
  On this rig the memory offset does not order them: card 7 at +1,100 MT/s went highest, card 4 at the same +1,100
  failed first, and cards 0 and 8, both at +900, failed at 2,189 and 2,164 MHz.
- **Card 6 needs a second look.** Its memory, +1,000 MT/s, held R218's rungs at +1,000 and +1,100 for 30 minutes
  each with no differing check (three differed at +1,200), and it passed this sitting's check round and the first
  core step at core 0. Then, two driver steps above its stock peak, prompt 1's text parted from stock at token 178
  while prompt 2 stayed bit-identical. One check cannot say whether that was the core or a rare memory error at
  +1,000 MT/s. The kit's rule takes it as a core failure, so card 6's core stays 0; step 4's ladder keeps card 6 at
  +1,000 MT/s with core 0 for its 90 minutes, which tests the memory half, and a second core sitting for card 6 alone
  is a planned row ([Q1636](../../../quants/experiments.md)).
- **One card hung instead of answering wrong.** Card 7's check at +332 MHz got no answer: the card sat at 2,240 MHz
  and 36.78 W (median; its previous check drew a median 62.24 W, up to 72.89 W) for 36 seconds until the climb's time limit (3 × its 12.2 s stock
  check, 37 s) failed it, set it to 0 and stopped its server. The card then freed its memory and went idle with no
  driver error. This is the hang the time limit was added for ([Q1572](../../../quants/experiments.md)); card 7's
  value is two climb steps under that step, as for any failure.
- **The probability-hashed greedy check is again the only check that saw a wrong answer.** In 2 of the 8 wrong
  answers (cards 0 and 1) the text of both prompts was identical to stock and only the token probabilities moved; in
  3 one prompt was bit-identical (cards 0, 1 and 6); the kernel checks passed 3,503 of 3,503 on the 113 clean checks
  ([tables/greedy_failures.csv](tables/greedy_failures.csv)).

**Each card** ([tables/cards.csv](tables/cards.csv), [tables/steps.csv](tables/steps.csv)). A target peak is the clock
the climb aimed at; the peak reached is the highest core clock the card showed during that step's check. The streaming
read is `mmvq_pascal`'s, at stock and at the memory offset in the check round.

| Card | Stock peak | Memory (R218) | Core: last clean step (target, reached) | Core: first failure (target, reached) | How it failed (check time, limit) | Result core (target, reached) | Streaming read, stock / memory offset |
|---|---|---|---|---|---|---|---|
| 0 | 1,873 MHz | +900 MT/s | +294 MHz (2,164, 2,151) | +320 MHz (2,189, 2,176) | wrong answer: text identical, prompt 1 bit-identical (8.1 s, 37 s) | +269 MHz (2,139, 2,126) | 179.17 / 197.06 GB/s |
| 1 | 1,860 MHz | +1,000 MT/s | +282 MHz (2,139, 2,126) | +307 MHz (2,164, 2,151) | wrong answer: text identical, prompt 1 bit-identical (10.1 s, 37 s) | +256 MHz (2,113, 2,100) | 179.50 / 199.25 GB/s |
| 2 | 1,911 MHz | +1,100 MT/s | +282 MHz (2,189, 2,164) | +307 MHz (2,214, 2,189) | wrong answer: prompt 1's text differs from token 238 (6.5 s, 35 s) | +256 MHz (2,164, 2,138) | 179.29 / 202.12 GB/s |
| 3 | 1,911 MHz | +1,200 MT/s | +231 MHz (2,139, 2,126) | +256 MHz (2,164, 2,151) | wrong answer: both prompts' text differs (11.5 s, 37 s) | +206 MHz (2,113, 2,100) | 178.85 / 204.48 GB/s |
| 4 | 1,885 MHz | +1,100 MT/s | +231 MHz (2,113, 2,100) | +256 MHz (2,139, 2,126) | wrong answer: both prompts' text differs (11.2 s, 37 s) | +206 MHz (2,088, 2,075) | 179.43 / 201.98 GB/s |
| 5 | 1,911 MHz | +1,300 MT/s | +256 MHz (2,164, 2,164) | +282 MHz (2,189, 2,189) | wrong answer: both prompts' text differs (9.5 s, 37 s) | +231 MHz (2,139, 2,138) | 179.41 / 206.84 GB/s |
| 6 | 1,885 MHz | +1,000 MT/s | 0 (1,885, 1,885) | +28 MHz (1,911, 1,898) | wrong answer: prompt 1's text differs from token 178, prompt 2 bit-identical (12.2 s, 37 s) | 0 (1,885, 1,885) | 179.12 / 199.46 GB/s |
| 7 | 1,911 MHz | +1,100 MT/s | +307 MHz (2,214, 2,214) | +332 MHz (2,240, 2,240) | hung: no answer within the limit (37 s), no Xid | +282 MHz (2,189, 2,189) | 179.21 / 201.86 GB/s |
| 8 | 1,885 MHz | +900 MT/s | +256 MHz (2,139, 2,126) | +282 MHz (2,164, 2,164) | wrong answer: prompt 2's text differs at token 82 (12.9 s, 37 s) | +231 MHz (2,113, 2,113) | 179.20 / 197.53 GB/s |

Every wrong answer came well inside its time limit (the slowest, card 8's, in 12.9 s against 37 s). Cards with a
stock peak above a target stay at core 0 until the target passes it, so cards 2, 3, 5 and 7 got their first core offset
at the 1,936 MHz target. The stock peaks were read on this boot: cards 0 and 2 one driver step above R217's readings
(1,873 against 1,860, and 1,911 against 1,898 MHz), the others the same.

**Heat and power** ([tables/heat_power.csv](tables/heat_power.csv), the watchdog's samples every 0.5 s):

| Card | Peak °C, core climb | Peak W, core climb | Samples above 80 W, whole sitting (longest run) | Highest core clock in the sitting |
|---|---|---|---|---|
| 0 | 41 | 77.31 | 0 | 2,176 MHz |
| 1 | 45 | 80.03 | 1 (1) | 2,151 MHz |
| 2 | 38 | 80.55 | 1 (1) | 2,189 MHz |
| 3 | 42 | 79.39 | 0 | 2,151 MHz |
| 4 | 38 | 76.72 | 0 | 2,126 MHz |
| 5 | 38 | 77.45 | 0 | 2,189 MHz |
| 6 | 36 | 74.97 | 2 (2), both in the check round, 81.19 W | 1,898 MHz |
| 7 | 41 | 77.07 | 0 | 2,240 MHz |
| 8 | 39 | 78.82 | 0 | 2,164 MHz |

No card was ever in a thermal or hardware slowdown, and the watchdog's power limit, 80 W for 5 s, was never reached:
the longest run above 80 W on any card was two samples (one second). Host memory with the nine check servers loaded:
1,854 MiB available.

## Step 4's rungs

The core soak ladder (step 4 of the [five steps](../../../bench/mem_offset/README.md#a-rigs-profile-the-same-five-steps-on-every-rig))
runs on the next fresh boot: every card's memory at R218's value, each card's core two climb steps, one climb step and
zero steps under its result core, three rungs of 30 minutes. R202b's
[make_rungs.py](../2026-11-18_r202b_core_soak/run/make_rungs.py) writes them from the climb's own steps, and stops on
a card without two clean core values under its result core: card 6 has none. [run/step4_rungs.py](run/step4_rungs.py)
runs it unchanged with card 6 marked as not climbed, so card 6 keeps the use profile's core, 0, in all three rungs, and
labels the files for this rig ([rungs/](rungs/)):

| Card | Memory | Rung 1 core | Rung 2 core | Rung 3 core |
|---|---|---|---|---|
| 0 | +900 MT/s | +218 MHz | +244 MHz | +269 MHz |
| 1 | +1,000 MT/s | +206 | +231 | +256 |
| 2 | +1,100 MT/s | +206 | +231 | +256 |
| 3 | +1,200 MT/s | +155 | +180 | +206 |
| 4 | +1,100 MT/s | +155 | +180 | +206 |
| 5 | +1,300 MT/s | +180 | +206 | +231 |
| 6 | +1,000 MT/s | 0 | 0 | 0 |
| 7 | +1,100 MT/s | +231 | +256 | +282 |
| 8 | +900 MT/s | +180 | +206 | +231 |

Each rung file passes the kit's profile check (`climb.py rungs` reads it, as `apply` would). The ladder's failures
file, the one `soak.py --failures` reads for the use profile, is [rungs/failures/rung1.json](rungs/failures/rung1.json),
written by `climb.py rungs` from this climb's profile and R218's `use.json`: all 20 failures of steps 1 to 3 (R217's
and R218's memory, this climb's core). Its other files, `rung2.json` to `rung4.json` and `rungs-choice.md`, are the
kit's memory-first rungs from the same inputs, kept for reference; step 4 does not use them.

## What was done

The rig: nine P106-090 6 GB cards (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), each on PCIe 1.0
x1, a 2-core Celeron and 4 GB of host memory. It was booted at about 11:52 for this sitting; the previous boot had run
R218's soak and ended with every offset at 0. The kit ran as published (`bench/mem_offset/`, `scripts/nvml_watchdog.py`;
every file's md5 in [run/MD5SUMS](run/MD5SUMS), the same as this branch's), with the rig's [run/env.sh](run/env.sh)
(`WORK` set to a fresh folder on the command line):

- **The check model:** Granite 3.1 3B-A800M Instruct, Q4_K_M (2.02 GB), one copy served on each card, as in R217 and
  R218.
- **The input profile:** R218's [use.json](../2026-11-20_r218_soak_rig1/use.json) (md5
  `88c5d386f927b38fca6cfb6c23aa0de9`): memory +900, +1,000, +1,100, +1,200, +1,100, +1,300, +1,000, +1,100, +900 MT/s
  on cards 0 to 8, core 0 on all nine.
- **The sitting:** the stock rounds (each card's reference hash, the time limits and its stock peak, over 96 s), every
  card's memory set straight to the profile's value and read back, one check round on all nine, then the core climb
  for all nine by target peak, two driver steps (25.32 MHz) a step from 1,885 to 2,240 MHz, each card's offset from its
  own stock peak and its memory held at the profile's value.
- **Each check:** the greedy check (two fixed prompts, 256 tokens each, temperature 0, every token's top-10
  log-probabilities hashed) against the card's own stock hash, then `mmvq_pascal`'s 31 kernel checks and a streaming
  read, each with a time limit of 3 × its stock time and at least 30 s. A card that fails is set to 0 at once and read
  back, and leaves the climb.
- **The watchdog** on all nine cards throughout: it logs clocks, power, temperature and both offsets every 0.5 s and
  sets every card to 0 above 70 °C, above 80 W for 5 s or on an Xid (an Xid naming one card trips that card alone).
  Its four trip tests passed first.

The timeline: 11:55:53 the servers loading (all nine ready by 11:58:48); the stock rounds to 12:00:24; the check round
12:00:30 to 12:01:16, all nine PASS; the core climb 12:01:18 to 12:11:13 (fifteen target steps, the last card failing
at the top one); every offset 0 and the profile written at 12:11:20. Every card ended at 0 and P8, read back by the kit
and by `nvidia-smi` (1,911/4,004 MHz maximum on all nine), the watchdog stopped after 0 trips, 0 Xid on the boot. A card
whose offset went down needs a fresh boot before any offset goes up again, so step 4 starts after a reboot.

## The commands

In the kit folder on the freshly booted rig, as root:

```bash
. ./env.sh && export WORK=/var/tmp/r221 P=/var/tmp/r218/soak/use.json   # P: md5 88c5d386f927b38fca6cfb6c23aa0de9
python3 climb.py state && python3 climb.py start-check
python3 climb.py --from-profile $P plan          # read-only: every card climbs the core
python3 climb.py trip-test && python3 climb.py watchdog start
python3 climb.py --from-profile $P climb         # detached; ended by itself after 15 min 27 s
cat $WORK/CLIMB_DONE; tail -n 15 $WORK/out/climb.log
python3 climb.py state && python3 climb.py watchdog stop
python3 climb.py rungs $WORK/out/profile.json --failures $P --out $WORK/failures
```

The tables and step 4's rungs, from the work folder copied off the rig (any machine, no GPU):

```bash
python3 -I run/tables.py WORK_COPY tables
python3 -I run/greedy_diff.py WORK_COPY/out tables/greedy_failures.csv
python3 -I run/step4_rungs.py ../2026-11-18_r202b_core_soak/run/make_rungs.py \
  ../2026-11-20_r218_soak_rig1/use.json WORK_COPY/out/steps.csv profile.json rungs --hold 6
```

## Files

| File | What |
|---|---|
| [profile.json](profile.json) | the sitting's result: each card's memory (R218's), core (two climb steps under its first failure; 0 for card 6), the stock peaks, the failures |
| [tables/cards.csv](tables/cards.csv) | one row per card: stock peak, memory, the check round, the last clean and the failing core step (offset, target, peak reached, reason, check time and limit, Xid), the result core, streaming reads at stock and at the memory offset |
| [tables/steps.csv](tables/steps.csv) | every check of every card as `climb.py` wrote it (122 checks: 9 stock, 9 in the check round, 104 in the core climb) |
| [tables/core_rungs.csv](tables/core_rungs.csv) | the result core per card with its target and reached peak and where it comes from |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | each wrong answer against the card's stock result, by prompt: text identical or not, the first differing token, tokens that differ, the first token whose probabilities differ (card 7's hung check gave no answer, so it has no row) |
| [tables/heat_power.csv](tables/heat_power.csv) | per phase and card: temperature, power, samples above 80 W, slowdowns, peak core and memory clocks |
| [tables/watchdog.txt](tables/watchdog.txt) | the watchdog's start and stop lines (0 trips) |
| [rungs/step4-rung1.json](rungs/step4-rung1.json), [2](rungs/step4-rung2.json), [3](rungs/step4-rung3.json) | step 4's three rungs |
| [rungs/failures/](rungs/failures/) | `climb.py rungs`' output; its `rung1.json` is step 4's failures file |
| [run/env.sh](run/env.sh), [run/MD5SUMS](run/MD5SUMS) | the rig's settings as sourced, and the md5 of every kit file that ran, the input profile, the server binary and the check model |
| [run/env-step4.sh](run/env-step4.sh), [run/MD5SUMS-step4](run/MD5SUMS-step4) | step 4's settings for `soak_chain.sh` (the rungs, the failures file, Ornith's whole-memory check), staged in the rig's kit folder with the rungs |
| [run/tables.py](run/tables.py), [run/greedy_diff.py](run/greedy_diff.py) | the scripts that built the tables (R202's, `tables.py` also writing the result core of a card that climbed to the top or failed the check round) |
| [run/step4_rungs.py](run/step4_rungs.py) | step 4's rungs: R202b's `make_rungs.py` with cards held at the use profile's core |

## Provenance and licence

Measured on the rig on 2026-11-20 with the engine build `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(Miner Llama v4's release candidate 2) and the kit as published on this branch. The check model is IBM's
Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; nothing of it is published here, only its name and
the hashes of its answers. The tables are this repository's licence.
