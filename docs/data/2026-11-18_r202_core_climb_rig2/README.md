# R202: a fair core climb for six cards of the second rig, each card's memory at a value that held (2026-11-18)

**The answer in plain words.** On a fresh boot of the second nine-card P106-090 rig, every card's memory went straight
to the use profile that had held a 30-minute soak ([R201](../2026-11-18_r201_soak_ladder/README.md)), all nine passed
one check there, and then only the core climbed, on the six cards that had no core value: cards 1, 2, 3, 5, 6 and 7
([Q1588](../../../quants/experiments.md), the kit's
[`climb.py --from-profile`](../../../bench/mem_offset/README.md#a-fair-core-climb-climbpy---from-profile)). **All six
climbed far: their last clean core offsets are +282, +294, +256, +231, +307 and +282 MHz**, where they reached
peaks of 2,164, 2,176, 2,126, 2,113, 2,176 and 2,164 MHz. Cards 2 and 3, which in the first sitting failed the core at
+16 and +104 MHz with their memory at +1,400 MT/s, now went to +294 and +256: those failures were memory errors charged
to the core, as R200's review read them. Every failure was a wrong answer and nothing else: no driver error (0 Xid), no
check past its time limit, no kernel check failed, the watchdog never tripped. The sitting took 11 minutes 55 seconds.
The hottest card was 48 °C (card 6) and the highest power 83.47 W in a single half-second sample (card 1).

**What it means.**

- **Every card of this rig now has a core value found with its memory at a value that holds.** Two climb steps under
  each card's first failure (the result profile, [profile.json](profile.json)): +256, +269, +231, +206, +282 and
  +256 MHz on cards 1, 2, 3, 5, 6 and 7, with cards 0, 4 and 8 keeping the use profile's +256, +206 and +231
  ([tables/core_rungs.csv](tables/core_rungs.csv)). These are the top of the next soak ladder, not settings: one check
  a step shows where a card fails at once, and R200's soaks failed one or two steps under the climb's last clean step.
- **The core edge is each card's own.** Cards 3 and 5 failed at a 2,152 MHz target peak, cards 1, 2, 6 and 7 at
  2,202 MHz, and cards 0, 4 and 8 failed at 2,177 MHz in the first sitting
  ([R200](../2026-11-17_r200_climb_rig2/README.md)). In this sitting the cards with less memory offset went higher:
  the four at +900 and +1,000 MT/s failed at 2,202 MHz, the two at +1,100 and +1,200 at 2,152. Memory alone does not
  set the edge, though: cards 0, 4 and 8 reached the 2,177 MHz target in R200 with their memory at +1,400.
- **The probability-hashed greedy check is again the only check that saw anything.** In 3 of the 6 failed checks the
  generated text of both prompts was identical to stock and only the token probabilities had moved; in 2 one prompt
  was bit-identical to stock and the other's text differed; in 1 both prompts' text differed
  ([tables/greedy_failures.csv](tables/greedy_failures.csv)). The kernel checks passed 2,666 of 2,666 on the clean
  steps.
- **The served speed with cores on all nine is not measured yet.** With the use profile (cores on cards 0, 4 and 8
  only) Ornith's coding agent ran 3.76 % faster a task ([R203](../2026-11-18_r203_ornith_offsets/README.md)); what
  the six new cores add is measured after their soak.

**Each card** ([tables/cards.csv](tables/cards.csv), [tables/steps.csv](tables/steps.csv)). A target peak is the clock
the climb aimed at; the peak reached is the highest core clock the card showed during that step's check.

| Card | Stock peak | Memory (use profile) | Core: last clean step (target, reached) | Core: first failure (target, reached) | How it failed (check time, limit) | Suggested core, two steps under (target, reached) | First sitting (R200) |
|---|---|---|---|---|---|---|---|
| 0 | 1,873 MHz | +1,300 MT/s | not climbed: the use profile's +256 MHz passed the check round (2,126, 2,126) | — | — | +256 MHz (2,126, 2,126) | failed at +307 (2,177 target), memory +1,400 |
| 1 | 1,898 MHz | +1,000 MT/s | +282 MHz (2,177, 2,164) | +307 MHz (2,202, 2,189) | wrong answer: prompt 1 bit-identical, prompt 2 one token differs (8.1 s, 37 s) | +256 MHz (2,152, 2,138) | not climbed (memory failed at +1,200) |
| 2 | 1,885 MHz | +1,000 MT/s | +294 MHz (2,177, 2,176) | +320 MHz (2,202, 2,202) | wrong answer: prompt 1 bit-identical, prompt 2's text differs from token 5 (9.5 s, 38 s) | +269 MHz (2,152, 2,151) | failed at +16 with memory +1,400 |
| 3 | 1,873 MHz | +1,100 MT/s | +256 MHz (2,126, 2,126) | +282 MHz (2,152, 2,138) | wrong answer: text identical, probabilities differ (9.1 s, 38 s) | +231 MHz (2,101, 2,100) | failed at +104 with memory +1,400 |
| 4 | 1,898 MHz | +1,100 MT/s | not climbed: the use profile's +206 MHz passed the check round (2,101, 2,113) | — | — | +206 MHz (2,101, 2,113) | failed at +282 (2,177 target), memory +1,400 |
| 5 | 1,898 MHz | +1,200 MT/s | +231 MHz (2,126, 2,113) | +256 MHz (2,152, 2,151) | wrong answer: both prompts' text differs (10.6 s, 38 s) | +206 MHz (2,101, 2,100) | not climbed (memory failed at +1,400) |
| 6 | 1,873 MHz | +900 MT/s | +307 MHz (2,177, 2,176) | +332 MHz (2,202, 2,202) | wrong answer: text identical, probabilities differ (11.0 s, 38 s) | +282 MHz (2,152, 2,151) | not climbed (memory failed at +1,300) |
| 7 | 1,898 MHz | +900 MT/s | +282 MHz (2,177, 2,164) | +307 MHz (2,202, 2,189) | wrong answer: text identical, probabilities differ (12.5 s, 37 s) | +256 MHz (2,152, 2,151) | not climbed (memory failed at +1,400) |
| 8 | 1,911 MHz | +1,300 MT/s | not climbed: the use profile's +231 MHz passed the check round (2,139, 2,126) | — | — | +231 MHz (2,139, 2,126) | failed at +282 (2,177 target), memory +1,400 |

Every failing check had 0 Xid lines and ended well inside its time limit (the slowest, card 7's, in 12.5 s against
37 s), so none was a hang.
Cards 0, 4 and 8 keep the use profile's cores, two steps under R200's failures for cards 0 and 8 and three for card 4;
R200 found those failures with the memory at +1,400 MT/s, so a ladder may raise them too.

**Heat and power** ([tables/heat_power.csv](tables/heat_power.csv), the watchdog's samples every 0.5 s):

| Card | Peak °C, core climb | Peak W, core climb | Samples above 80 W (longest run) | Highest core clock in the sitting |
|---|---|---|---|---|
| 1 | 43 | 83.47 | 2 (1) | 2,189 MHz |
| 2 | 39 | 83.35 | 3 (1) | 2,202 MHz |
| 3 | 37 | 82.39 | 1 (1) | 2,138 MHz |
| 5 | 39 | 77.96 | 0 | 2,151 MHz |
| 6 | 48 | 78.73 | 0 | 2,202 MHz |
| 7 | 43 | 79.92 | 0 | 2,189 MHz |

Cards 0, 4 and 8 were at 0 and idle during the core climb; in the check round they peaked at 42, 39 and 38 °C and
75.02, 71.54 and 81.05 W (card 8: one sample above 80 W). No card was ever in a thermal or hardware slowdown. The
watchdog's power limit, 80 W for 5 s, was never reached: the longest run above 80 W on any card was one sample.

## What was done

The rig: nine P106-090 6 GB cards (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), each on PCIe 1.0
x1, a 2-core Celeron and 4 GB of host memory. It was booted at 18:15:43 for this sitting; earlier boots had run R201's
soaks and R203's agent checks, and the last offsets of the previous boot were reset before that boot ended. The kit
ran as published (`bench/mem_offset/` with C875's `climb.py --from-profile`, `scripts/nvml_watchdog.py`; every file's
md5 in [run/MD5SUMS](run/MD5SUMS)), with the rig's [run/env.sh](run/env.sh) (`WORK` set to a fresh folder on the
command line):

- **The check model:** Granite 3.1 3B-A800M Instruct, Q4_K_M (2.02 GB), one copy served on each card, as in R200 and
  R201.
- **The input profile:** R201's use profile ([use-profile.json](../2026-11-18_r201_soak_ladder/use-profile.json),
  md5 `af7735752572f91a045e1e2608aa705c`): memory +1,300, +1,000, +1,000, +1,100, +1,100, +1,200, +900, +900,
  +1,300 MT/s on cards 0 to 8, core +256, +206 and +231 MHz on cards 0, 4 and 8. It held a 30-minute soak of all nine
  cards on a second boot and served Ornith for 30 minutes with answers identical to stock.
- **The sitting:** the stock rounds (each card's reference hash, the time limits and its stock peak), every card's
  memory and kept core set straight to the profile's values and read back, one check round on all nine, cards 0, 4
  and 8 set to 0 with their servers stopped, then the core climb for cards 1, 2, 3, 5, 6 and 7 by target peak, two
  driver steps (25.32 MHz) a step from 1,898 MHz to 2,250 MHz, each card's offset from its own stock peak and its
  memory held at the profile's value.
- **Each check:** the greedy check (two fixed prompts, 256 tokens each, temperature 0, every token's top-10
  log-probabilities hashed) against the card's own stock hash, then `mmvq_pascal`'s 31 kernel checks and a streaming
  read, each with a time limit of 3 × its stock time and at least 30 s. A card that fails is set to 0 at once and read
  back, and leaves the climb.
- **The watchdog** on all nine cards throughout: it logs clocks, power, temperature and both offsets every 0.5 s and
  sets every card to 0 above 70 °C, above 80 W for 5 s or on an Xid (an Xid naming one card trips that card alone).
  Its four trip tests passed first.

The timeline: 18:19:22 the servers loading (all nine ready by 18:21:53, 1,857 MiB of host memory available); the stock
rounds to 18:23:29; the check round 18:23:36 to 18:24:21, all nine PASS; the core climb 18:24:28 to 18:31:11; every
offset 0 and the profile written at 18:31:17. Every card ended at 0 and P8, read back by the kit and by `nvidia-smi`
(1,911/4,004 MHz maximum on all nine). A card whose offset went down needs a fresh boot before any offset goes up again,
so the next sitting starts after a reboot.

## The commands

In the kit folder on the freshly booted rig, as root:

```bash
. ./env.sh && export WORK=/var/tmp/r202
P=./r201-use.json                     # R201's use profile, md5 af7735752572f91a045e1e2608aa705c
python3 climb.py state && python3 climb.py start-check
python3 climb.py --from-profile $P --core-cards "1 2 3 5 6 7" plan     # read-only
python3 climb.py trip-test && python3 climb.py watchdog start
python3 climb.py --from-profile $P --core-cards "1 2 3 5 6 7" climb    # detached; ended by itself after 11 min 55 s
cat $WORK/CLIMB_DONE; tail -n 3 $WORK/out/climb.log
python3 climb.py watchdog stop && python3 climb.py state
```

The tables, from the work folder copied off the rig:

```bash
python3 -I run/tables.py WORK_COPY tables
python3 -I run/greedy_diff.py WORK_COPY/out tables/greedy_failures.csv
```

## Files

| File | What |
|---|---|
| [profile.json](profile.json) | the sitting's result: each card's memory (the use profile's), core (two climb steps under the first failure for the six climbers, the use profile's for cards 0, 4 and 8), the stock peaks, the failures |
| [tables/cards.csv](tables/cards.csv) | one row per card: stock peak, memory, the check round, the last clean and the failing core step (offset, target, peak reached, reason, check time and limit, Xid), the result core, streaming reads at stock and at the profile's memory |
| [tables/steps.csv](tables/steps.csv) | every check of every card as `climb.py` wrote it (92 checks: 9 stock, 9 in the check round, 74 in the core climb) |
| [tables/core_rungs.csv](tables/core_rungs.csv) | the suggested core per card for the next soak ladder, with its target and reached peak and where it comes from |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | each failed greedy check against the card's stock result, by prompt: text identical or not, the first differing token, tokens that differ, the first token whose probabilities differ |
| [tables/heat_power.csv](tables/heat_power.csv) | per phase and card: temperature, power, samples above 80 W, slowdowns, peak core and memory clocks |
| [tables/watchdog.txt](tables/watchdog.txt) | the watchdog's start and stop lines (0 trips) |
| [run/env.sh](run/env.sh), [run/MD5SUMS](run/MD5SUMS) | the rig's settings as sourced, and the md5 of every kit file that ran, the input profile and the server binary |
| [run/tables.py](run/tables.py), [run/greedy_diff.py](run/greedy_diff.py) | the scripts that built the tables (`greedy_diff.py` is R200's, unchanged) |

## Provenance and licence

Measured on the rig on 2026-11-18 with the engine build `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(Miner Llama v4's release candidate 2) and the kit as published on this branch. The check model is IBM's
Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; nothing of it is published here, only its name and
the hashes of its answers. The tables are this repository's licence.
