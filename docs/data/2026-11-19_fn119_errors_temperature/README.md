# FN119: do a card's wrong answers under a clock offset come when the card is warmer? (2026-11-19)

**The answer in plain words.** No sign of it, but these soaks could only have shown a strong effect. Every round of
the second nine-card rig's seven soaks was matched with the watchdog's half-second readings of that card over that
round. Of the 8 wrong answers, the 4 in the soaks of the first day ([R200](../2026-11-17_r200_climb_rig2/README.md))
say nothing either way: each of those soaks stopped at its wrong answer, so the answer is by design the soak's last
round, the card's latest at that value and, as the cards were still warming, one of its warmest. **The other 4
([R201](../2026-11-18_r201_soak_ladder/README.md), which went on counting) came at their cards' usual temperature:**
37.75, 37.79, 39.78 and 46.33 °C over the round. On a scale where 0 is the card's coolest round at that value and 1 its
warmest, they sat at 0.138, 0.243, 0.406 and 0.501, 0.322 on average, where an even spread gives 0.500; as warm or
warmer comes by chance 88.7 % of the time.

**What it means.**

- **Nothing changes in how a profile is checked:** count every round of a rung, as
  [FN118](../2026-11-19_fn118_errors_after_change/README.md) found and the kit does
  ([bench/mem_offset](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check)).
- **The late lean FN118 saw is the same stopping rule, not heat.** By minutes since the card's own change the four
  R200 answers sit at 1.000, 0.804, 1.000 and 1.000 because each ended its soak; the four R201 answers sit at 0.427 on
  average, where an even spread gives 0.500 (as late or later by chance 68.7 %).
- **Once warm, a card hardly changes temperature in a soak.** In the first minute of a soak the cards ran 6.64 °C below
  where they settled, 3.13 °C below in the second minute, 1.15 °C below over minutes 2 to 5, and from 5 minutes on
  within 0.36 °C. After that a card held its temperature within a fraction of a degree (over the 305 R201 rounds at the
  values that erred, a standard deviation of 0.14 °C for card 2 and 0.26 °C for cards 3 and 6), while the sensor reads
  whole degrees. Which card it is matters far more: in R202b card 3 settled at 35.9 °C and card 6 at 45.0 °C.
- **What these soaks cannot tell.** With so little change in temperature inside a soak, 4 wrong answers would show
  even a strong effect rarely: if each degree tripled a card's chance of a wrong answer, the test would flag it 19.1 %
  of the time, and 27.8 % if each degree multiplied it by 5. The coolest round averaged 26.03 °C and the warmest
  46.87 °C; nothing here says what a card does at 55 °C in a hot room. A soak held deliberately warmer is a parked idea
  ([Q1616](../../../quants/experiments.md)).
- **Late and warm cannot be told apart in this data.** Within the strata that erred, temperature follows time
  closely (Spearman 0.788 or more between a round's temperature and its minutes since the card's change; 0.529 over
  every stratum of 20 or more rounds), so a heat effect and a time effect would look the same here.

## What was done

At a desk, on data already copied off the rig; nothing ran on a rig or a GPU. The soaks, their rounds, offsets and
wrong answers are FN118's ([tables/rounds.csv.gz](../2026-11-19_fn118_errors_after_change/tables/rounds.csv.gz):
12,438 card-rounds, 8 wrong answers). The kit's watchdog ([bench/mem_offset](../../../bench/mem_offset/README.md))
ran beside every soak and wrote every card's state every half second: core and memory clock, watts, temperature and
the offsets it read back.

**A card-round's window.** All cards' checks of a round start together; the kit then waits on each card in turn and
writes its row with the seconds since the round began, to the whole second. A card-round's window runs from the row's
time + 0.5 s minus those seconds to the row's time + 0.5 s; it held 21 to 31 watchdog readings (23 the median). Over
each window: the card's mean and peak temperature, its mean and peak watts, its mean core and memory clock, and the
mean temperature of all nine cards. The offsets the watchdog read back agree with FN118's for 12,437 of the 12,438
card-rounds; the other one's window (R201 card 0, round 303) touched the next rung's raise.

**The test.** As in FN118, a stratum is one card at one memory and core value, pooled over every soak in which it
ran. If a card's chance of a wrong answer at a value were the same in every round, the rounds holding its wrong answers
would be rounds drawn at random from all its rounds at that value. Each wrong answer's place among them by a measure
(0 the lowest, 1 the highest, ties shared) is averaged over the wrong answers, and the average is compared with
100,000 random draws (seed 119). The main measure, chosen before the data was read, is the mean temperature over the
round among the card's rounds at that value in every soak. The same is given by peak temperature, watts, core clock,
the whole rig's temperature and FN118's minutes since a change, and among the card's rounds at that value in the same
soak only. That is 14 places for each set of wrong answers: one under 5 % by chance is expected for every 20, so only
the main measure answers the question, and a second measure would need under 0.36 % (5 % over 14) to count.

**The two sets of wrong answers.** R200's four tries stopped at the first wrong answer on any card, so each of their
answers is its soak's last round by design and its place by time or by warm-up heat is forced. They are shown with
the others, and the answer rests on the four in R201, whose soak went on to its end.

**How much heat could have shown.** Each informative wrong answer's round drawn again at random from its stratum with
weight b to the power of its temperature (each degree multiplies the chance of a wrong answer by b), 20,000 times for
each b; the share of draws whose average place reaches the top 5 % of the even spread (0.737) is how often the test
would see an effect of that size.

## Results

### The eight wrong answers ([tables/errors.csv](tables/errors.csv))

| Soak | Card | Round | Memory, core | Mean temperature (peak) | Watts | Core clock | The card's rounds at that value: temperature lowest, median, highest | Place by mean temperature | Place by minutes since own change | Soak stopped at it |
|---|---|---|---|---|---|---|---|---|---|---|
| R200 try 1 | 4 | 13 | +1,300 MT/s, +231 MHz | 41.83 °C (42) | 46.55 | 2,126 MHz | 13: 32.43, 39.75, 41.83 | 1.000 | 1.000 | yes |
| R200 try 2 | 2 | 11 | +1,300, 0 | 36.83 (37) | 45.95 | 1,879 | 24: 28.24, 35.06, 37.42 | 0.870 | 0.804 | yes |
| R200 try 3 | 7 | 108 | +1,200, 0 | 41.71 (42) | 44.33 | 1,895 | 132: 32.66, 41.57, 41.83 | 0.824 | 1.000 | yes |
| R200 try 4 | 7 | 25 | +1,100, 0 | 41.61 (42) | 44.70 | 1,882 | 25: 32.55, 40.29, 41.71 | 0.958 | 1.000 | yes |
| R201 | 3 | 330 | +1,300, 0 | 37.75 (38) | 44.34 | 1,860 | 305: 36.58, 37.87, 38.74 | 0.138 | 0.086 | no |
| R201 | 3 | 384 | +1,300, 0 | 37.79 (38) | 45.18 | 1,859 | 305: 36.58, 37.87, 38.74 | 0.243 | 0.263 | no |
| R201 | 2 | 461 | +1,200, 0 | 39.78 (40) | 45.38 | 1,873 | 438: 28.24, 39.83, 40.26 | 0.406 | 0.664 | no |
| R201 | 6 | 467 | +1,100, 0 | 46.33 (47) | 44.30 | 1,860 | 462: 36.07, 46.33, 46.87 | 0.501 | 0.694 | no |

### The average place ([tables/tests.csv](tables/tests.csv))

Among the card's rounds at that value in every soak (even spread: 0.500):

| Measure | All 8: average place; as high or higher by chance | The 4 in R201: average place; chance band (2.5 % to 97.5 %); as low or lower; as high or higher |
|---|---|---|
| **temperature, mean over the round** | 0.618; 13.1 % | **0.322; 0.220 to 0.779; 11.3 %; 88.7 %** |
| temperature, peak over the round | 0.675; 2.2 % | 0.550; 0.287 to 0.675; 82.0 %; 51.6 % |
| watts, mean | 0.526; 40.2 % | 0.483; 0.220 to 0.782; 45.5 %; 54.5 % |
| core clock, mean | 0.514; 45.0 % | 0.654; 0.231 to 0.775; 86.5 %; 14.2 % |
| all nine cards' temperature, mean | 0.642; 8.9 % | 0.425; 0.220 to 0.779; 30.8 %; 69.2 % |
| minutes since the card's own change (FN118) | 0.689; 3.5 % | 0.427; 0.219 to 0.780; 31.3 %; 68.7 % |
| minutes since any card's change (FN118) | 0.572; 24.7 % | 0.194; 0.219 to 0.782; 1.6 %; 98.4 % |

All 8's places by peak temperature and by minutes since the card's own change are high only through R200's four,
each its soak's last round. Among the 4 in R201, the last row is FN118's own question asked of the two answers that
raised it (cards 2 and 6, 6 and 12 rounds after the ladder's last raise), so it is no new evidence.

Among the card's rounds at that value in the same soak only, the 4 in R201 sit at 0.193 by mean temperature (as low
or lower by chance 1.4 %) and 0.738 by core clock (as high or higher 4.8 %), the lowest chances among the 28 places
computed for them, and above the 0.36 % a second measure needs. Neither is heat: within R201 a card's temperature
followed the morning (Spearman with time 0.910 for card 3, 0.895 for card 2 and 0.820 for card 6), each raise paused
the cards for a round (card 3 read 36.58 °C in the first round after rung 3's raise against 37.58 just before it), and
card 3's two answers came in rung 3's first 16 minutes. What separates "cooler" rounds there is a few tenths of a
degree from a sensor that reads whole degrees, and what separates "faster" ones a few MHz of the core's boost clock.

### How much heat could have shown ([tables/power.csv](tables/power.csv))

| Each degree multiplies the chance of a wrong answer by | 1 (no effect) | 1.25 | 1.5 | 2 | 3 | 5 |
|---|---|---|---|---|---|---|
| The 4 in R201 reach the top 5 % of the even spread | 5.3 % | 7.6 % | 9.9 % | 13.3 % | 19.1 % | 27.8 % |

### The warm-up ([tables/warmup.csv](tables/warmup.csv))

All seven soaks together, by minutes since the soak's first round began. "Below settled" is each card's distance
under its own median from 15 minutes on in the same soak (only the soaks that ran past 15 minutes: R200 try 3, R201,
R201b and R202b).

| Minutes | Card-rounds | Mean temperature | Below settled | Wrong answers |
|---|---|---|---|---|
| 0 to 1 | 315 | 34.29 °C | 6.64 °C | 0 |
| 1 to 2 | 324 | 37.81 | 3.13 | 0 |
| 2 to 5 | 729 | 39.77 | 1.15 | 3 |
| 5 to 10 | 909 | 40.46 | 0.36 | 0 |
| 10 to 15 | 909 | 40.64 | 0.20 | 0 |
| 15 to 30 | 2,349 | 40.69 | 0.19 | 1 |
| 30 to 60 | 2,745 | 40.81 | 0.22 | 0 |
| 60 to 90 | 2,743 | 41.12 | −0.10 | 2 |
| 90 and later | 1,415 | 42.40 | −0.57 | 2 |

The three wrong answers at 2 to 5 minutes are R200's tries 1, 2 and 4, which stopped there. R201 warmed by 1.24 °C
between its 15 to 30 minutes (41.18 °C) and its rounds from 90 minutes on (42.42 °C), over which the ladder raised
every card's memory and then three cards' cores; R202b held 40.23 °C at 15 to 30 minutes and 40.36 °C at 60 to 90.

### Each card's heat ([tables/cards.csv](tables/cards.csv))

Each card's median temperature and watts over the rounds from 15 minutes on:

| Card | R200 try 3 | R201 | R201b | R202b | Wrong answers |
|---|---|---|---|---|---|
| 0 | 43.5 °C, 48.2 W | 44.3, 46.9 | 43.7, 48.2 | 42.8, 47.3 | 0 |
| 1 | 40.8, 46.9 | 42.8, 47.1 | 41.2, 47.0 | 40.8, 47.7 | 0 |
| 2 | 37.9, 45.9 | 39.8, 45.4 | 38.8, 45.8 | 37.9, 45.6 | 2 |
| 3 | 35.9, 44.2 | 37.8, 44.3 | 35.9, 44.0 | 35.9, 45.1 | 2 |
| 4 | 42.8, 45.8 | 44.5, 44.9 | 43.6, 45.7 | 42.7, 45.8 | 1 |
| 5 | 37.5, 43.8 | 39.1, 43.7 | 38.0, 44.0 | 37.9, 44.8 | 0 |
| 6 | 45.3, 44.9 | 46.4, 45.1 | 44.9, 44.1 | 45.0, 45.9 | 1 |
| 7 | 41.7, 45.9 | 41.8, 44.7 | 41.0, 45.3 | 40.4, 45.4 | 2 |
| 8 | 39.3, 46.7 | 40.2, 46.1 | 39.6, 46.5 | 38.6, 46.0 | 0 |

The rig's coolest card (3) erred twice and its warmest (6) once; each card ran its own values, so this compares
nothing across cards. Card 6 is the warmest and card 3 the coolest in every soak.

### Late and warm ([tables/confound.csv](tables/confound.csv))

Spearman correlation between a round's mean temperature and its minutes since the card's own change, within each
wrong answer's stratum: R200's 1.000, 0.987, 0.891 and 0.999; R201's card 3 0.910, card 2 0.814 and card 6 0.788
(over every soak; 0.895 and 0.820 within R201 alone). Over every stratum of 20 or more rounds (12,425 card-rounds),
0.529, weighted by rounds.

## Files

| File | What it holds |
|---|---|
| [tables/heat.csv.gz](tables/heat.csv.gz) | Every card-round (12,438): its window, the watchdog readings in it, the card's mean, peak and first temperature, mean and peak watts, mean core and memory clock, the offsets the watchdog read, the mean temperature of all nine cards |
| [tables/errors.csv](tables/errors.csv) | Each wrong answer: its heat, its stratum's rounds and temperatures, its place by every measure among the card's rounds at that value in every soak and in the same soak, whether its soak stopped at it |
| [tables/tests.csv](tables/tests.csv) | The average place by every measure for all 8 and for the 4 in R201, in every soak and in the same soak: the chance band and the chance of a place as low or as high |
| [tables/power.csv](tables/power.csv) | How often the 4 in R201 would reach the top 5 % if each degree multiplied the chance of a wrong answer by 1, 1.25, 1.5, 2, 3 or 5 |
| [tables/warmup.csv](tables/warmup.csv) | Card-rounds, mean temperature, distance below the card's settled temperature and wrong answers by minutes since each soak's first round, per soak and in all |
| [tables/confound.csv](tables/confound.csv) | Within each wrong answer's stratum: the rounds, the Spearman correlation of temperature with minutes since the card's own change, the temperature's standard deviation; and over every stratum of 20 or more rounds |
| [tables/cards.csv](tables/cards.csv) | Each card's median temperature and watts per soak from 15 minutes on, and its wrong answers |
| [run/errors_temperature.py](run/errors_temperature.py) | `build` makes `heat.csv.gz` from each soak's `soak.csv` and the watchdog's `wd.csv.gz`; `table` makes the rest from `heat.csv.gz` and FN118's `rounds.csv.gz` |

## Commands

On any machine, with `python3 -I` (standard library only). The tables from the published files:

```bash
python3 -I run/errors_temperature.py table \
  ../2026-11-19_fn118_errors_after_change/tables/rounds.csv.gz tables/heat.csv.gz tables
```

`heat.csv.gz` from the work folders as the kit left them on the rig and as they were copied off it: each soak's
`soak/soak.csv` and the watchdog's `wd.csv.gz` (R200's try 1 kept one watchdog file over two boots):

```bash
python3 -I run/errors_temperature.py build --out tables \
  --run try1=R200_TRY1/soak/soak.csv,R200_TRY1/wd-both-boots.csv.gz --run try2=R200_TRY2/soak/soak.csv,R200_TRY2/wd.csv.gz \
  --run try3=R200_TRY3/soak/soak.csv,R200_TRY3/wd.csv.gz --run try4=R200_TRY4/soak/soak.csv,R200_TRY4/wd.csv.gz \
  --run r201=R201_WORK/soak/soak.csv,R201_WORK/wd.csv.gz --run r201b=R201B_WORK/soak/soak.csv,R201B_WORK/wd.csv.gz \
  --run r202b=R202B_WORK/soak/soak.csv,R202B_WORK/wd.csv.gz
```

A later soak adds one more `--run NAME=SOAK_CSV,WD_CSV_GZ` here and one more `--kit` in FN118's `build`.

## Provenance and licence

Computed on 2026-11-19 from the second rig's soaks of 2026-11-17 and 2026-11-18 (engine build
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`, the `mem_offset` kit and its watchdog as published with each
run). The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0); nothing of it is here. The tables are this
repository's licence.
