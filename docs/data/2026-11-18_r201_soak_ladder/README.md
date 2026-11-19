# R201: a ladder of soaks on the second rig, every card's errors counted for two hours in one boot, then the use profile on a second boot (2026-11-18)

**The answer in plain words.** One fresh boot, nobody at the console, no reboot: all nine cards of the second rig ran
the kit's soak for two hours, 608 rounds, while their memory offsets went up one step (100 MT/s) every 30 minutes, and
in the last half hour three cards also got a core offset ([Q1587](../../../quants/experiments.md), the kit
[bench/mem_offset](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check),
[Q1591](../../../quants/experiments.md)). **Six cards never differed from stock** (0, 1, 4, 5, 7 and 8, each at its
highest rung), **three differed four times in all**: card 3 twice at +1,300 MT/s, cards 2 and 6 once each at +1,200
and +1,100. Every one of the four errors spoiled one answer and left the next one bit-identical to stock. The core
offsets on cards 0, 4 and 8 held all 153 rounds of their rung. Because a card differed, the chain set every offset
back to 0 and did not serve Ornith with the offsets on that boot. The hottest card reached 47 °C; the watchdog never
tripped and no card reported an error to the driver.

**The second boot** ([below](#the-second-boot-the-use-profile-held-and-ornith-generated-442--faster), tables in
[use-profile/](use-profile/)): after a reboot, the use profile made from this ladder **held on all nine cards** for a
30-minute soak (152 rounds each, no differing check), and **Ornith served with it gave the same answers as at stock in
all 53 rounds while generating 4.42 % faster**: 16.29 against 15.60 tok/s on both prompts of the kit's check, on the
same boot. That is a single-request diagnostic; the agentic speed with the profile is
[Q1592](../../../quants/experiments.md).

**The setting for the next boot** ([use-profile.json](use-profile.json), made by [run/make_use.py](run/make_use.py)
from [soaked.json](soaked.json)): each card's highest clean rung, but no higher than two memory steps under the lowest
offset at which it ever failed, here or in [R200](../2026-11-17_r200_climb_rig2/README.md):

| Card | Highest clean rung: memory, core | First differed here | The cap and why | **Use** |
|---|---|---|---|---|
| 0 | +1,300 MT/s, +256 MHz | never | none: no memory failure of its own | **+1,300 MT/s, +256 MHz** |
| 1 | +1,000, 0 | never | +1,000: R200's climb failed at +1,200 | **+1,000, 0** |
| 2 | +1,200, 0 | rung 4, at +1,200 | +1,000: this run, at +1,200 | **+1,000, 0** |
| 3 | +1,200, 0 | rung 3, at +1,300 | +1,100: this run, at +1,300 | **+1,100, 0** |
| 4 | +1,200, +206 | never | +1,100: R200's first soak differed at +1,300 (core +231) | **+1,100, +206** |
| 5 | +1,200, 0 | never | +1,200: R200's climb failed at +1,400 | **+1,200, 0** |
| 6 | +1,100, 0 | rung 4, at +1,100 | +900: this run, at +1,100 | **+900, 0** |
| 7 | +1,000, 0 | never | +900: R200's soaks differed at +1,200 and +1,100 | **+900, 0** |
| 8 | +1,300, +231 | never | none: no memory failure of its own | **+1,300, +231** |

Cards 0, 3 and 8 failed R200's core climb with their memory at +1,400 MT/s. Cards 0 and 8 failed at the 2,177 MHz
target together with card 4, which reads as the core's edge, so those failures cap no memory. Card 3's failure at a
low core clock was likely its memory; its own failure here at +1,300 now sets its cap. On a second fresh boot the use
profile held a 30-minute soak and Ornith's 30 minutes ([run/env-use.sh](run/env-use.sh), the second boot below); the
agentic speed with it is [Q1592](../../../quants/experiments.md).

**What it means.**

- **The ladder does in one boot what took four yesterday.** R200 needed four reboots to learn one card's failure each;
  here every card had two hours, and each card's highest clean value came out of one boot with nobody at the console.
- **Errors near the edge are rare.** Cards 2 and 6 erred once in the 305 rounds (an hour) they ran at their rung-3
  values, card 3 twice in 305 rounds at +1,300. Both of cards 2 and 6's errors came after 30 clean minutes at the
  same value, so a 30-minute clean rung shows only that a card errs less often than about once in 10 minutes
  (95 % confidence), as the kit's README says.
- **`soaked.json` counts a value as held although the card differed at it later.** Cards 2 and 6 held their rung-3
  values clean for 30 minutes and differed at the same values in rung 4; `soaked.json` still gives +1,200 and +1,100,
  because its rule takes the highest rung a card held clean. The use profile's rule caps them two steps under; the
  kit should not offer such a value (C883 in [Q1595](../../../quants/experiments.md)).
- **An error passes, again.** In all four failed checks one of the check's two requests differed and the other was
  bit-identical to stock; in three the generated text was identical too and only the token probabilities moved
  ([tables/greedy_failures.csv](tables/greedy_failures.csv)). Counting instead of stopping was right.
- **Two of the four errors came right after a raise.** Card 2 differed in the 6th round of rung 4 and card 6 in the
  12th, at values they had held for all 152 rounds of rung 3, and both were clean for the rest of rung 4. At each raise
  every card's core clock drops for a moment, also on cards whose offsets do not change (1,354 MHz on all nine at
  11:47:42 in the watchdog's samples), and in rung 4 cards 0, 4 and 8 started running at higher core clocks. The
  watchdog shows no change in cards 2 and 6's own clocks, power or temperature at those rounds. If each card's one
  error fell at random among its 305 rounds, both landing in the first 12 rounds of a rung would happen about once in
  650 (est.). R200's first two soaks also failed early, in rounds 13 and 11 after `apply`. Whether a clock change
  makes a nearby error more likely is not known; it is [Q1596](../../../quants/experiments.md).
- **The same core offset peaked lower than in R200.** Card 0 at +256 MHz peaked at 2,113 MHz (2,126 in all four of
  R200's soaks), card 4 at +206 at 2,088 (2,113) and card 8 at +231 at 2,113 (2,138): one and two driver steps
  (12.66 MHz each) lower. Neither the stock peak nor the peak at an
  offset is a fixed property of a card; the offset is what the driver is given.

## What was done

The second nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off),
freshly booted at about 09:18, with nothing run on it before. The kit as published on this branch (`soak.py`,
`soak_chain.sh`, `model_check.sh`, `climb.py`, `greedy_check.py`; [run/MD5SUMS](run/MD5SUMS)), with R200's kit folder
and settings ([R200's run/env.sh](../2026-11-17_r200_climb_rig2/run/env.sh)) and this run's settings in
[run/env-chain.sh](run/env-chain.sh). The engine build was `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`.

`soak_chain.sh` ran the whole boot detached:

1. **The start check** (every offset 0, no load, no Xid this boot, 3,178 MiB of host memory available), then
   **Ornith-1.5-122B Pascal N2P GGUF at stock** with its v4 release settings
   ([release/ornith-v4](../../../release/ornith-v4/settings.env)) on all nine cards, the kit's greedy check repeated:
   loaded in 543 s, then 51 rounds in 30 minutes (09:38:40 to 10:09:20).
2. **The watchdog, then `climb.py apply` with rung 1** (10:09:24 to 10:14:38): every card's stock check, the offsets
   set and read back, and one check round at rung 1: every card passed.
3. **The ladder** (`soak.py --minutes 30 --keep-going --ladder`, 10:14:43 to 12:17:53): one check server per card with
   the check model (Granite 3.1 3B-A800M Instruct, Q4_K_M), and every round each card's greedy check (two prompts,
   256 tokens each, temperature 0, every token's top-10 log-probabilities hashed) against its stock hash. After each
   30-minute rung the cards that had not differed went up to the next rung's values, read back; a card that differed
   stayed where it was and went on being checked. A card that differed three times, or whose check failed or ran past
   its 45-second limit, would have been set to 0 alone; none did.
4. **Ornith with the offsets** for 30 minutes, which runs only if no card differed: skipped.

The rungs ([run/make_rungs.py](run/make_rungs.py), [run/r201-rung1.json](run/r201-rung1.json) to
[run/r201-rung4.json](run/r201-rung4.json)), memory in MT/s on cards 0 to 8:

| Rung | Memory | Core |
|---|---|---|
| 1 | +1,100, +800, +1,000, +1,100, +1,000, +1,000, +900, +800, +1,100 | 0 |
| 2 | +1,200, +900, +1,100, +1,200, +1,100, +1,100, +1,000, +900, +1,200 | 0 |
| 3 | +1,300, +1,000, +1,200, +1,300, +1,200, +1,200, +1,100, +1,000, +1,300 | 0 |
| 4 | as rung 3 | +256, +54, +206 and +231 MHz on cards 0, 3, 4 and 8 |

Rung 3 is R200's last values with card 7 at +1,000; rung 4 adds the core offsets that held R200's soaks.

## Results

### Every card and rung ([tables/rungs.csv](tables/rungs.csv), [tables/cards.csv](tables/cards.csv))

Rounds a rung: 151, 152, 152 and 153. Each cell is the card's offsets (memory MT/s / core MHz) and its differing
checks in that rung.

| Card | Rung 1 | Rung 2 | Rung 3 | Rung 4 |
|---|---|---|---|---|
| 0 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,300 / 0: 0 | +1,300 / +256: 0 |
| 1 | +800 / 0: 0 | +900 / 0: 0 | +1,000 / 0: 0 | +1,000 / 0: 0 |
| 2 | +1,000 / 0: 0 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,200 / 0: **1** (round 461) |
| 3 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,300 / 0: **2** (rounds 330, 384) | +1,300 / 0, held: 0 |
| 4 | +1,000 / 0: 0 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,200 / +206: 0 |
| 5 | +1,000 / 0: 0 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,200 / 0: 0 |
| 6 | +900 / 0: 0 | +1,000 / 0: 0 | +1,100 / 0: 0 | +1,100 / 0: **1** (round 467) |
| 7 | +800 / 0: 0 | +900 / 0: 0 | +1,000 / 0: 0 | +1,000 / 0: 0 |
| 8 | +1,100 / 0: 0 | +1,200 / 0: 0 | +1,300 / 0: 0 | +1,300 / +231: 0 |

Rung 4 started after round 455. Card 3 differed in the 27th and 81st round of rung 3 and stayed at +1,300 with no
core offset for rung 4, where it ran all 153 rounds clean.

### The four failed checks ([tables/greedy_failures.csv](tables/greedy_failures.csv))

| Card | Round | Prompt 1 | Prompt 2 |
|---|---|---|---|
| 2 | 461 | text identical; probabilities differ from token 137 | bit-identical to stock |
| 3 | 330 | bit-identical to stock | text identical; probabilities differ from token 143 |
| 3 | 384 | text differs from token 238 (18 of 256 tokens); probabilities from token 205 | bit-identical to stock |
| 6 | 467 | bit-identical to stock | text identical; probabilities differ from token 178 |

### Ornith at stock ([tables/ornith_stock.csv](tables/ornith_stock.csv))

Ornith filled 5,603, 5,635, 5,549, 5,635, 5,243, 5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8, with 1,521 MiB of
host memory left. All 51 rounds gave the same greedy hash, `adf47784f9b02d4b`, the hash R200 recorded on another
boot. Generation in round 1: 14.54 and 15.24 tok/s (prompts 1 and 2); the median of the 51 rounds: 15.63 and
15.63 tok/s; prompt processing in round 2: 26.88 and 29.62 tok/s. The step was meant to be two rounds: `soak_chain.sh`
passes its environment's `MINUTES` (the 30 minutes a rung) to `model_check.sh`'s stock call, which then repeats for
30 minutes (C883 in [Q1595](../../../quants/experiments.md)). It became a 30-minute stock control on the whole memory
of every card, with no difference.

### Apply's check, streaming reads ([tables/apply.csv](tables/apply.csv))

Every card passed the greedy check and the 31 kernel checks at stock and at rung 1. Streaming reads in GB/s, cards 0
to 8, at stock: 179.15, 179.09, 179.17, 179.21, 179.23, 179.21, 179.18, 179.29, 179.31; at rung 1: 202.14, 195.03,
199.44, 201.96, 199.65, 199.54, 197.56, 195.36, 201.90.

### Heat, power and clocks ([tables/heat_power.csv](tables/heat_power.csv))

From the watchdog's half-second samples: the hottest card was 47 °C (in each rung); no card was in a thermal or
hardware slowdown in any sample; the highest power reading was 84.91 W (card 6, in `apply`), and readings above 80 W
were single samples, never two in a row (one in each of rungs 1 to 3, card 8; 18 in rung 4, cards 0, 4 and 8, the
three with core offsets), against the watchdog's limit of 80 W for 5 s. Peak core clocks in rung 4: 2,113, 2,088 and
2,113 MHz on cards 0, 4 and 8. The watchdog never tripped, and the boot logged no Xid.

### The end

At 12:17:53 the soak ended (`exit 1 rounds 608 differs 4`), wrote [soaked.json](soaked.json), and the chain set
every offset of every card to 0 (12:17:54 to 12:18:04). Then the watchdog was stopped and the state read back: every
offset 0, P8, no load on any card, no Xid this boot, 3,133 MiB of host memory available.

## The second boot: the use profile held, and Ornith generated 4.42 % faster

**In plain words.** The rig was rebooted (at about 12:41) and the chain ran once more with no ladder: Ornith at stock,
`apply` with the use profile, a 30-minute soak of all nine cards at it, then Ornith served with the offsets set for 30
minutes. **Every card held:** 152 rounds each with no differing check, the watchdog never tripped and the boot logged
no Xid. **Ornith with the offsets answered exactly as at stock** in all 53 rounds (greedy hash `adf47784f9b02d4b`, the
hash of the first boot and of R200) and **generated 4.42 % faster on both prompts**: a median of 16.29 tok/s against
15.60 at stock on the same boot. This is the first number for the served model with offsets on all nine cards; it is
the kit's single-request check, a diagnostic, and the agentic speed with the profile is
[Q1592](../../../quants/experiments.md). Thirty clean minutes show only that each card errs less often than about
once in 10 minutes (95 % confidence); permanent use still needs a pass after another reboot and longer clean runs
([Q1587](../../../quants/experiments.md)).

### What was done

The same rig and kit as the first boot ([run/MD5SUMS](run/MD5SUMS), all sums checked on the rig again), with
[use-profile.json](use-profile.json) copied to the kit folder as `r201-use.json` (md5
`af7735752572f91a045e1e2608aa705c`, checked there) and this boot's settings in [run/env-use.sh](run/env-use.sh):
`soak_chain.sh` with that profile as `PROFILE`, no `LADDER`, 30 minutes for the soak and 30 for Ornith. The start
check passed on the fresh boot: every offset 0, no load, no Xid, 3,187 MiB of host memory available.

1. **Ornith at stock** (12:45:23): loaded in 658 s, then 50 rounds in 30 minutes (12:57:01 to 13:26:27). It again ran
   30 minutes instead of two rounds (C883 in [Q1595](../../../quants/experiments.md)), a second 30-minute stock
   control with no difference.
2. **The watchdog, then `climb.py apply` with the use profile** (13:26:31 to 13:31:36): every card's stock check and
   its check at the profile passed.
3. **The soak** (`soak.py --minutes 30 --keep-going`): check servers ready at 13:34:05, 152 rounds to 14:04:08,
   `exit 0 rounds 152 differs 0`.
4. **Ornith with the offsets** (14:04:09): loaded in 661 s, then 53 rounds to 14:45:48, each compared with stock's
   round 1: all the same. The chain ended with "done: apply, the soak and the model with the offsets all passed".
5. **The end** (14:47:40 to 14:47:50): every offset set to 0 (`climb.py reset-all`, read back), the watchdog stopped
   after 0 trips, no load, no Xid this boot, 3,163 MiB of host memory available.

### Every card ([use-profile/cards.csv](use-profile/cards.csv), [use-profile/rungs.csv](use-profile/rungs.csv))

Peaks from the watchdog's half-second samples, in the soak and in Ornith's 53 rounds. The memory clock is the
driver's reading, 4,006 MHz at stock.

| Card | Use profile: memory, core | Soak rounds, differing | Peak core clock (MHz): soak, Ornith | Peak memory clock (MHz) | Peak temperature (°C): soak, Ornith | Peak power (W): soak, Ornith |
|---|---|---|---|---|---|---|
| 0 | +1,300 MT/s, +256 MHz | 152, 0 | 2,126, 2,126 | 4,657 | 45, 43 | 83.03, 79.77 |
| 1 | +1,000, 0 | 152, 0 | 1,898, 1,898 | 4,498 | 42, 40 | 79.06, 80.24 |
| 2 | +1,000, 0 | 152, 0 | 1,898, 1,898 | 4,498 | 39, 37 | 78.44, 77.87 |
| 3 | +1,100, 0 | 152, 0 | 1,885, 1,885 | 4,551 | 36, 35 | 76.71, 78.36 |
| 4 | +1,100, +206 | 152, 0 | 2,113, 2,113 | 4,551 | 44, 41 | 83.39, 80.05 |
| 5 | +1,200, 0 | 152, 0 | 1,911, 1,898 | 4,608 | 39, 36 | 76.51, 72.42 |
| 6 | +900, 0 | 152, 0 | 1,885, 1,885 | 4,455 | 46, 43 | 78.00, 75.15 |
| 7 | +900, 0 | 152, 0 | 1,898, 1,911 | 4,455 | 42, 39 | 79.83, 78.11 |
| 8 | +1,300, +231 | 152, 0 | 2,138, 2,126 | 4,657 | 40, 38 | 83.82, 81.59 |

### Ornith at stock and with the offsets ([use-profile/ornith.csv](use-profile/ornith.csv))

Ornith-1.5-122B Pascal N2P GGUF on all nine cards with its v4 release settings
([release/ornith-v4](../../../release/ornith-v4/settings.env)), the engine build
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`, the kit's greedy check (two prompts, 256 tokens each,
temperature 0), both runs on the same boot.

| | Stock | With the use profile | Change |
|---|---|---|---|
| Rounds; the same as stock's round 1 | 50; 49 (round 1 is the reference) | 53; 53 | |
| Generation tok/s, median of all rounds (prompts 1, 2) | 15.60, 15.60 | 16.29, 16.29 | +4.42 %, +4.42 % |
| Generation tok/s in round 1 (prompts 1, 2) | 14.39, 15.19 | 14.85, 15.80 | |
| Prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.74, 29.54 | 27.30, 30.15 | +2.09 %, +2.06 % |
| Load time | 658 s | 661 s | |

Both loads filled 5,603, 5,635, 5,549, 5,635, 5,243, 5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8, with 1,518 and
1,509 MiB of host memory left.

### Apply's check, heat and power ([use-profile/apply.csv](use-profile/apply.csv), [use-profile/heat_power.csv](use-profile/heat_power.csv))

Every card passed the greedy check and the 31 kernel checks at stock and at the profile. Streaming reads in GB/s,
cards 0 to 8, at stock: 179.06, 179.06, 179.06, 179.11, 179.26, 179.29, 179.24, 179.00, 179.12; at the profile:
206.82, 199.60, 199.42, 201.84, 201.92, 204.70, 197.25, 197.51, 206.92.

The hottest card was 46 °C (card 6, in `apply` and the soak), and no card was in a thermal or hardware slowdown in any
sample. Power went above 80 W only in single samples, or two in a row once (card 6 in `apply`), never near the
watchdog's limit of 80 W for 5 s: the highest reading was 84.28 W (card 2, in `apply`); in the soak 20, 7 and 9 single
samples on cards 0, 4 and 8, the three with core offsets; in Ornith's rounds 1, 2 and 4 on cards 1, 4 and 8.

The same core offsets peaked at R200's clocks again: 2,126, 2,113 and 2,138 MHz on cards 0, 4 and 8 in the soak,
where the first boot's rung 4 gave 2,113, 2,088 and 2,113 MHz. The peak at a given offset moves by one or two driver
steps between boots.

### What it means

- **The rule for the use profile worked on its first test.** Each card two memory steps under its lowest failure
  anywhere gave a profile all nine cards held, on the check model and on Ornith's whole memory, one boot after the
  ladder that found it.
- **The offsets buy Ornith 4.42 % on generation** with the same tokens: memory on all nine cards and the core on
  three. Generation reads the weights from every card's memory for each token, and gains about twice what prompt
  processing gains (+2.09 % and +2.06 %).
- **Six cards have no core offset yet** (1, 2, 3, 5, 6, 7): their core climb from this profile is
  [Q1588](../../../quants/experiments.md).

## Files

| File | What it holds |
|---|---|
| [soaked.json](soaked.json) | `soak.py`'s profile: each card at the highest rung it held clean, with every rung's offsets and differing checks |
| [use-profile.json](use-profile.json) | The setting for the next boot, by the rule above, as `climb.py apply` reads it |
| [tables/cards.csv](tables/cards.csv) | One row per card: each rung's offsets and differing checks, the soaked value, the first difference, the use value |
| [tables/rungs.csv](tables/rungs.csv) | `soak.py`'s rows per rung and card: offsets, rounds, passes, differing checks, the first differing round |
| [tables/soak.csv](tables/soak.csv) | Every card's every round: hash, seconds, verdict, rung and offsets |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | Each failed check against the card's stock run, by prompt (made with [R200's run/greedy_diff.py](../2026-11-17_r200_climb_rig2/run/greedy_diff.py)) |
| [tables/ornith_stock.csv](tables/ornith_stock.csv) | Ornith's 51 stock rounds: hash, prompt and generation tok/s of both prompts |
| [tables/apply.csv](tables/apply.csv) | `apply`'s stock round and its round at rung 1: peaks, the greedy and kernel checks, streaming reads |
| [tables/heat_power.csv](tables/heat_power.csv) | Each phase (apply, the soak's server start, each rung, each raise) and card from the watchdog's samples |
| [run/env-chain.sh](run/env-chain.sh) | This run's settings for `soak_chain.sh`: work folder, the rungs, Ornith and its settings, the minutes |
| [run/env-use.sh](run/env-use.sh) | The next boot's settings: the use profile, a 30-minute soak and Ornith's 30 minutes |
| [run/r201-rung1.json](run/r201-rung1.json) … [run/r201-rung4.json](run/r201-rung4.json) | The four rungs, as `apply` and `soak.py --ladder` read them |
| [run/make_rungs.py](run/make_rungs.py) | Writes the rungs from R200's last profile |
| [run/profile-try4.json](run/profile-try4.json) | R200's last profile (try 4's soak), the input of `make_rungs.py`; `climb.py rungs` makes the rungs from it by the kit's rule |
| [run/make_use.py](run/make_use.py) | Writes the use profile from `soaked.json` and prints the table above |
| [run/failures.json](run/failures.json) | Rig 2's failures before this run (R200's), as `soak.py --failures` reads them: with them `soak.py --use-from soaked.json` gives every card the values of `use-profile.json` |
| [run/tables.py](run/tables.py) | Builds the tables from the work folder |
| [run/MD5SUMS](run/MD5SUMS) | The md5 sums of the kit files and settings copied to the rig, checked there before each boot's run |
| [use-profile/cards.csv](use-profile/cards.csv) | The second boot, one row per card: the profile's offsets, the soak's rounds, passes and differing checks, peak temperature, power and clocks in the soak and in Ornith's run |
| [use-profile/rungs.csv](use-profile/rungs.csv), [use-profile/soak.csv](use-profile/soak.csv) | The second boot's soak as `soak.py` wrote it: one row per card, and one per card and round |
| [use-profile/ornith.csv](use-profile/ornith.csv) | Ornith's 50 stock rounds and 53 rounds with the offsets: hash, prompt and generation tok/s of both prompts, the same as stock or not |
| [use-profile/apply.csv](use-profile/apply.csv) | `apply`'s stock round and its round at the use profile: peaks, the greedy and kernel checks, streaming reads |
| [use-profile/heat_power.csv](use-profile/heat_power.csv) | Each phase of the second boot (apply, the soak's server start, the soak, Ornith's load and rounds) and card from the watchdog's samples |
| [run/tables_use.py](run/tables_use.py) | Builds the second boot's tables and prints Ornith's medians and their change |

## Commands

On the rig, as root, in the kit's folder (R200's, with this branch's `soak.py`, `soak_chain.sh`, `model_check.sh`,
`climb.py`, `greedy_check.py` and this folder's `run/` files beside it), on a fresh boot:

```bash
md5sum -c MD5SUMS
. ./env-chain.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/r201 && setsid nohup bash soak_chain.sh > /var/tmp/r201/chain.console 2>&1 < /dev/null &
cat /var/tmp/r201/CHAIN_DONE; tail -n 3 /var/tmp/r201/soak/soak.log     # a look, at most every 30 minutes
# when CHAIN_DONE exists: every offset 0, the watchdog stopped, the state read back; copy /var/tmp/r201 and
# /var/tmp/r201-ornith off the rig
python3 climb.py reset-all; python3 climb.py watchdog stop; python3 climb.py state
```

The tables and the use profile, on any machine, from the copied folders:

```bash
python3 -I run/make_use.py WORK/soak/soaked.json use-profile.json
# or by the kit's own rule, the same values on every card:
python3 -I ../../../bench/mem_offset/soak.py --use-from WORK/soak/soaked.json --failures run/failures.json --use-out use-again.json
# the rungs by the kit's rule (two steps under each failure): cards 2, 3, 4 and 7 one step lower than run/r201-rung*.json;
# with --cap-steps 1, R201's own rule, the four files card for card
python3 -I ../../../bench/mem_offset/climb.py rungs run/profile-try4.json --failures run/failures.json --out rungs-again --prefix r201-rung
python3 -I run/tables.py WORK ORNITH_OUT use-profile.json tables
python3 -I ../2026-11-17_r200_climb_rig2/run/greedy_diff.py WORK/soak tables/greedy_failures.csv WORK/out r201-ladder
```

The second boot, on the rig after a reboot, with `use-profile.json` in the kit folder as `r201-use.json`:

```bash
md5sum r201-use.json && md5sum -c MD5SUMS       # af7735752572f91a045e1e2608aa705c
. ./env-use.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/r201b && ENV_FILE=./env-use.sh setsid nohup bash soak_chain.sh > /var/tmp/r201b/chain.console 2>&1 < /dev/null &
cat /var/tmp/r201b/CHAIN_DONE; tail -n 3 /var/tmp/r201b/soak/soak.log /var/tmp/r201b-ornith/model.log   # a look about every 25 minutes
# when CHAIN_DONE exists: every offset 0, the watchdog stopped, the state read back; copy /var/tmp/r201b and
# /var/tmp/r201b-ornith off the rig
. ./env-use.sh && python3 climb.py reset-all; python3 climb.py watchdog stop; python3 climb.py state
```

Its tables, on any machine, from the copied folders:

```bash
python3 -I run/tables_use.py WORK ORNITH_OUT use-profile
```

## Provenance and licence

Measured on the rig on 2026-11-18, on two boots (09:29 to 12:19 and 12:45 to 14:48), with the engine build `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(Miner Llama v4's release candidate 2) and the kit as published on this branch (`99421051`). The check model is
IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; Ornith-1.5-122B Pascal N2P GGUF and its licence
are described in [release/ornith-v3](../../../release/ornith-v3/README.md). Neither model is published here, only the
hashes of their answers. The tables are this repository's licence.
