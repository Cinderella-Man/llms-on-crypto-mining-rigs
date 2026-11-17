# R200: every card of a second rig finds its own clock offsets, then 30-minute soaks (2026-11-17)

**The answer in plain words.** The climb kit ([Q1573](../../../quants/experiments.md),
[bench/mem_offset/climb.py](../../../bench/mem_offset/README.md)) ran for the first time on real cards: a second
nine-card P106-090 rig whose cards had never had an offset. All nine climbed together, memory first, then the core,
in 16 minutes 49 seconds with no help, and it gave every card its own profile, one step under the card's last clean
step. After a reboot `apply` set the profile and every card passed its check. **But no profile passed a 30-minute
soak of all nine cards.** Four tries, each after a reboot and each with the last failing card one step lower, ended on
one card each: card 4 in round 13, card 2 in round 11, card 7 in round 108 and card 7 again in round 25 (at memory
+1,100 MT/s). **A stock control then ran 148 rounds, 30 minutes, with every offset 0, and no card differed once**, so the
check does not fail on its own: each failure was the card's offset. Every failure, in the climb and in the soaks,
was caught by the probability-hashed greedy check alone: no kernel check failed (3,968 of 3,968), no card reported an
error to the driver, no check ran past its time limit, and the watchdog never tripped. In 6 of the 13 failed checks
the generated text was identical to stock and only the token probabilities had moved. The hottest card all evening
was card 0 at 52 °C, in the climb.

**What it means.**

- **One check a step finds where a card fails at once, not where it never fails.** Cards 2 and 7 failed the soak one
  and two memory steps under their last clean climb step, after 10, 107 and 24 clean rounds. Near the edge a memory
  error is rare rather than constant, so the climb's single check per step passes a setting that a longer run does
  not. Card 7 needs memory +1,000 MT/s or less. The other eight cards' last values held 157 rounds over four
  boots (cards 0, 1, 3, 5, 6 and 8), 144 over three (card 4) and 133 over two (card 2) ([tables/soak.csv](tables/soak.csv)).
- **A card's stock clock moves between boots, and a stored offset moves with it.** Card 4 peaked at 1,898 MHz at stock
  on the climb's boot and at 1,911 on the next; card 0 at 1,873 and then 1,860. The profile stores a core offset, so
  card 4's +231 MHz gave a measured 2,138 MHz in the first soak, the peak of its last clean climb step rather than one
  step under it, and card 4 failed there. A profile should store each card's target peak clock instead
  ([Q1586](../../../quants/experiments.md)).
- **One rig's cards are not another's.** The first rig's card 0 failed memory at +1,400 MT/s and held the core to a
  2,151 MHz peak ([Q133 and Q135](../2026-11-17_q133_q135_clock_offsets/README.md)); its card 4 failed at +1,200 MT/s
  with a 2,138 MHz peak ([R194](../2026-11-17_r194_nine_cards_soak/README.md)). Here card 1 failed memory at +1,200,
  card 6 at +1,300, cards 5 and 7 at +1,400, and five cards passed +1,400 in the climb.
- **The core edge looks shared.** The three cards that reached the top of the core climb (0, 4 and 8) passed the
  2,151 MHz target and failed at the 2,177 MHz target, as the first rig's cards passed 2,151 and its card 2 failed at
  2,202 (R193). Cards 2 and 3 failed the core climb at far lower clocks with memory held at +1,400, the step cards 5
  and 7 had just failed: memory failures charged to the core.
- **The probability-hashed greedy check is the test that matters** ([Q1564](../../../quants/experiments.md)). It
  caught every failure; the kernel checks caught none, and a check of the generated text alone would have missed six
  of the thirteen ([tables/greedy_failures.csv](tables/greedy_failures.csv)).
- **Streaming reads rise in step with memory** ([tables/stream_gbs.csv](tables/stream_gbs.csv)): card 0 read
  179.31 GB/s at stock, 202.02 at +1,100, 207.37 at +1,300 and 209.32 at +1,400.

**The climb's profile** ([profile.json](profile.json), [tables/cards.csv](tables/cards.csv)), and what the soaks
changed ([tables/profiles.csv](tables/profiles.csv)):

| Card | Stock peak core clock | Memory: last clean, failed at | Core: last clean, failed at (expected peak) | The climb's profile: memory, core (expected peak) | After the soaks |
|---|---|---|---|---|---|
| 0 | 1,873 MHz | +1,400, — | +282 (2,152), +307 (2,177) | +1,300 MT/s, +256 MHz (2,126 MHz) | held 157 rounds |
| 1 | 1,898 MHz | +1,100, +1,200 | not climbed | +1,000 MT/s, 0 (1,898 MHz) | held 157 rounds |
| 2 | 1,885 MHz | +1,400, — | 0, +16 (1,898) | +1,300 MT/s, 0 (1,885 MHz) | failed (try 2); +1,200 held 133 rounds |
| 3 | 1,873 MHz | +1,400, — | +79 (1,949), +104 (1,974) | +1,300 MT/s, +54 MHz (1,924 MHz) | held 157 rounds |
| 4 | 1,898 MHz | +1,400, — | +256 (2,151), +282 (2,177) | +1,300 MT/s, +231 MHz (2,126 MHz) | failed (try 1); +1,200 and +206 held 144 rounds |
| 5 | 1,898 MHz | +1,300, +1,400 | not climbed | +1,200 MT/s, 0 (1,898 MHz) | held 157 rounds |
| 6 | 1,873 MHz | +1,200, +1,300 | not climbed | +1,100 MT/s, 0 (1,873 MHz) | held 157 rounds |
| 7 | 1,898 MHz | +1,300, +1,400 | not climbed | +1,200 MT/s, 0 (1,898 MHz) | failed (try 3); +1,100 failed (try 4) |
| 8 | 1,898 MHz | +1,400, — | +256 (2,151), +282 (2,177) | +1,300 MT/s, +231 MHz (2,126 MHz) | held 157 rounds |

A soak stops at the first card that differs, so the rounds a card held are the rounds its soaks lasted; none has had
the full 30 minutes in one boot. Rig 2 is left at stock. The next try, card 7 at +1,000 MT/s with the rest as they
held, and the whole-memory check with Ornith that is ready for the first soak that passes, are
[Q1587](../../../quants/experiments.md); the core climb for the cards that got none is
[Q1588](../../../quants/experiments.md).

## What was done

A nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off) that had run
other rows since its morning boot, all at stock, and had never had a clock offset on any card. The owner sat at the
rig's console. The kit ran as published (`climb.py`, `greedy_check.py`, `nvml_watchdog.py`, `memoc_units.json` and the
`mem_offset` kit's `nvml_memoc.py` and `mmvq_pascal`), with this rig's settings in [run/env.sh](run/env.sh):

- **The check model:** Granite 3.1 3B-A800M Instruct, Q4_K_M (2.02 GB), one copy served on each card. A larger file
  covers more of each card's memory, but no GGUF of 2 to 4.5 GB that runs alone was on this rig, and this was the one
  at hand on the machine that holds the project's files.
- **The steps:** memory +200, +400, +600, +800, +900, +1,000, +1,100, +1,200, +1,300, +1,400 MT/s with the core at 0
  (the owner chose the climb to +1,400). Then the core, for the cards that never failed, in target peaks 2 driver
  steps (25.3 MHz) apart from the lowest stock peak, 1,873 MHz, up to 2,250 MHz, each card's offset from its own stock
  peak; each card's memory stays at its last clean step.
- **Each step, on every card at once:** the greedy check (two fixed prompts, 256 tokens each, temperature 0, every
  token's top-10 log-probabilities hashed) against the card's own stock hash, then `mmvq_pascal`'s Q8_0 `attn_q` checks
  (31 kernel checks and a streaming read). Each check had a time limit of 3 × its stock time, at least 30 s: 37 s
  (card 5: 38 s) for the greedy check, 30 s for the kernel checks.
- **A card that fails** is set back to 0 at once, both offsets read back, its server stopped, and it leaves the climb
  while the others go on.
- **The watchdog** ([scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py)) on every card throughout: every
  0.5 s it logged clocks, power, temperature and both offsets, and would have set every card to 0 above 70 °C, above
  80 W for 5 s, or on any Xid; an Xid naming one card would have tripped that card only.

The order on the rig: start check, plan, the watchdog's four trip tests, the watchdog, the climb; afterwards the
results copied off the rig, the watchdog stopped, a reboot at the console, and `apply` with the profile. Then, on the
owner's order, four tries of a 30-minute soak, each after a reboot and an `apply`, and a stock control (below).

## Results

### Before any offset

- **Start check:** no load process, 0 MiB, the default 75 W power limit, both offsets 0 and P8 on all nine cards, no
  Xid this boot, no step down in the kit's ledger. 3,126 MiB of host memory available.
- **Trip tests, all four passed** on all nine cards with stand-in processes: a 15 °C temperature limit, a kernel-log
  test line naming no card, and a 1 W power limit each set every card to 0 and killed every stand-in. A kernel-log line
  naming card 0 killed only card 0's stand-in, left card 1's running and tripped nothing else. This was the first test
  of the one-card trip on real cards.
- **Loading:** the nine check servers loaded one after another in 2 minutes 30 seconds (cards 0 to 8: 15, 12, 22, 15, 15, 15, 15, 18 and 22 s). With all
  nine loaded, 1,832 MiB of the 4 GB host's memory was available; the lowest reading in the whole climb was 640 MiB
  (19:16:25, in the memory climb).
- **Stock, two rounds:** every card gave the same greedy hash, `60ee0f178cfd67d5`, in both rounds. The greedy check
  took 12.2 s on eight cards and 12.7 s on card 5; the kernel checks 7.0 s (card 8: 6.5 s). Streaming reads at stock
  in GB/s, cards 0 to 8: 179.31, 179.26, 179.27, 178.95, 179.44, 179.34, 179.37, 179.20, 178.97.

### The climb, 19:08:09 to 19:24:58

| Time | Step | What happened |
|---|---|---|
| 19:13:09–19:17:27 | memory +200 to +1,000 | all nine pass every step |
| 19:18:37 | memory +1,200 | **card 1 fails** (greedy output differs) |
| 19:19:20 | memory +1,300 | **card 6 fails** |
| 19:20:04, 19:20:06 | memory +1,400 | **cards 5 and 7 fail**; cards 0, 2, 3, 4 and 8 pass |
| 19:20:35 | core, target 1,898 MHz | **card 2 fails** at +16 MHz (measured peak 1,898) |
| 19:21:48 | core, target 1,974 MHz | **card 3 fails** at +104 MHz (measured peak 1,961) |
| 19:24:47–19:24:50 | core, target 2,177 MHz | **cards 0, 4 and 8 fail** at +307, +282 and +282 MHz (measured peaks 2,151, 2,176 and 2,164) |
| 19:24:52–19:24:58 | end | every offset of every card set to 0 and read back 0; the profile written |

Every step's every check is in [tables/steps.csv](tables/steps.csv): the offsets tested, the expected and measured
peak core clock, the greedy verdict and time, the kernel checks and the streaming read.

- **The core often runs a step or two under its expected peak.** The expected peak is the stock peak plus the
  offset's whole driver steps of 12.66 MHz. Card 4 reached it at 8 of its 12 core steps and was one step under at the
  other four; card 0 was one step under from its first core step and two from the 2,101 MHz target on; card 8 one
  step under at every step. Pascal's boost drops a step as a card warms, and card 0 was the warmest. So by measured clock card 0 passed 2,126 and failed at 2,151 MHz,
  card 4 passed 2,138 and failed at 2,176, and card 8 passed 2,138 and failed at 2,164.
- **How the failures look** ([tables/greedy_failures.csv](tables/greedy_failures.csv)): at stock and at every clean
  step each card's output was bit-identical to its stock run, so any difference is an error. In the failed checks
  the chosen tokens' log-probabilities moved before any token changed, by as little as 0.0126 (card 6) and as much as
  0.188 (card 8). Cards 1, 3 and 7's
  generated text was identical to stock in both prompts; on the other six the text parted at token 55, 62, 82, 129 or
  194 of a prompt.
- **Heat and power** ([tables/heat_power.csv](tables/heat_power.csv)): the hottest card was card 0 at 52 °C, in the
  core climb. The highest half-second power reading was 85.72 W (card 7, in the memory climb). No card had two
  readings in a row above 80 W, and none showed a thermal or hardware slowdown.

### After the climb: apply, the soaks and the tries

The rig was rebooted at the console after the climb (19:31:13), the kit's start check passed, and `climb.py apply`
set the profile: every card's two offsets set going up from 0 and read back as set, then one greedy check on every
card against its stock result on that boot. **It passed on all nine cards (19:37:32).** Every card's stock hash on
the new boot was again `60ee0f178cfd67d5`.

The owner then asked for a soak: the profile held for 30 minutes with all nine cards loaded at once, every card's
greedy check repeated every round, against its own stock hash ([run/soak.py](run/soak.py): one check server per card,
each check under a 45 s limit; a card that differs, a check that runs late, a trip of the watchdog or of one card
sets every offset of every card to 0 and stops). The slowest check of any soak took 14.7 s; try 3 ran 108 rounds
in 21 minutes 7 seconds. A failure means the rig needs a reboot before the next try (offsets only go up in a boot), and the card
that failed goes one step lower: a card with a core offset loses a core step first, a card without one a memory step
(the owner's choice for card 4: both).

| Try | Boot | What changed | Soak | Failed card, at | How it failed |
|---|---|---|---|---|---|
| 1 | 19:31:13 | the climb's profile | 19:40:45–19:43:08, 13 rounds | **card 4, round 13**: memory +1,300, core +231 (measured peak 2,138 MHz) | probabilities only, from token 244 of prompt 2; the text identical |
| 2 | 19:46:49 | card 4: memory +1,200, core +206 | 19:56:44–19:58:44, 11 rounds | **card 2, round 11**: memory +1,300, core 0 | probabilities only, from token 150 of prompt 1 |
| 3 | 20:00:15 | card 2: memory +1,200 | 20:10:03–20:31:10, 108 rounds | **card 7, round 108**: memory +1,200, core 0 | probabilities only, from token 86 of prompt 2 |
| 4 | 20:36:02 | card 7: memory +1,100 | 20:55:24–21:00:23, 25 rounds | **card 7, round 25**: memory +1,100, core 0 | the text parted at token 123 of prompt 1 (131 tokens differ) |

Each try's `apply` passed: one check per card is not enough to see these failures. Nothing else saw them either:
no Xid, no trip, no check past its time limit, and no card above 47 °C (card 6, try 4) in any soak. The highest power
reading in a soak was 83.85 W (card 0, try 3), and never five seconds in a row above 80 W
([tables/soak.csv](tables/soak.csv)).

**Why card 4 failed at its profile.** A card's stock peak core clock moved one driver step between boots: card 4
peaked at 1,898 MHz on the climb's boot and at 1,911 on the first apply's, card 0 at 1,873 and then 1,860. The
profile stores a core offset, so the same +231 MHz put card 4 at a measured 2,138 MHz in the soak: the measured peak
of its last clean climb step, not one step under it. A profile should store each card's target peak and `apply`
should set the offset from the card's stock peak on its own boot ([Q1586](../../../quants/experiments.md)).

**Why one step under the edge was not enough for cards 2 and 7.** Both failed in the memory dimension with no core
offset, at one step under their last clean climb step (card 2: +1,300 against +1,400; card 7: +1,200 against
+1,300), after 10 and 107 clean rounds. A memory error at the edge is rare, not constant: a single check passes a step
that a long run does not. The climb's one check a step finds where a card fails at once, not where it never fails.

### Ornith on every card's whole memory (try 4)

The check model covers about 2 GB of each card's 6 GB. Ornith-1.5-122B Pascal N2P GGUF with its v4 release settings
([release/ornith-v4](../../../release/ornith-v4/settings.env)) fills 5,603, 5,635, 5,549, 5,635, 5,243, 5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8
([run/ornith_check.sh](run/ornith_check.sh)): on try 4's boot it was loaded at stock first (in 544 s) and gave the same
greedy hash in two rounds, `adf47784f9b02d4b` (every token's top-10 probabilities included), then stopped; `apply`
set try 4's profile and the soak ran. The soak failed (card 7), so Ornith with the profile set did not run: the
script and the stock reference are ready for the first soak that passes ([run/r200_chain.sh](run/r200_chain.sh) runs all
four steps in one boot).

### A stock control, 21:02 to 21:35

Card 7 failed at +1,200 and then at +1,100, after 107 and 24 clean rounds, so the soak itself had to be checked: on
try 4's boot, with every offset 0 and read back 0, the same soak ran for 30 minutes and counted every difference
instead of stopping (`soak.py --keep-going`). **148 rounds, 1,332 checks, none differed** on any card: the check does
not fail without an offset, and card 7's failures were its memory offset. The hottest card was card 6 at 48 °C, the
highest power reading 78.86 W (card 7), the slowest check 17.4 s. The control ran on a boot whose cards had been set
back to 0 nine times (a memory step down leaves a card slower until a reboot, Q1563), so it shows the check is
reliable on such a boot; a control on a fresh boot was not run.

## Files

| File | What it holds |
|---|---|
| [tables/cards.csv](tables/cards.csv) | One row per card: stock peak core clock, the memory climb's last clean step and failure, the core climb's last clean offset and peak and failure, the reason, the profile, apply's verdict |
| [tables/steps.csv](tables/steps.csv) | Every check of every card at every step, as `climb.py` wrote it, then apply's rows |
| [tables/stream_gbs.csv](tables/stream_gbs.csv) | Each card's streaming read at stock and at each memory step |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | Each failed greedy check, in the climb and the soaks, against the card's stock run, by prompt: text identical or not, the first token that differs, how many differ, the first token whose probabilities differ, the largest change before the text parts |
| [tables/heat_power.csv](tables/heat_power.csv) | Each phase and card from the watchdog's half-second samples: temperature, power, readings above 80 W, slowdowns, peak clocks |
| [tables/watchdog.txt](tables/watchdog.txt) | The watchdog's start and stop lines (the climb and the first apply) |
| [tables/soak.csv](tables/soak.csv) | One row per try (and the stock control) and card: the offsets apply set, the card's stock peak on that boot, apply's check, the soak's rounds, passes and failed round, and from the watchdog the soak's peak temperature, power, readings above 80 W and peak clocks |
| [tables/profiles.csv](tables/profiles.csv) | Each try's memory and core value per card |
| [profile.json](profile.json) | The profile `climb.py` wrote, as `apply` reads it |
| [run/env.sh](run/env.sh) | This rig's settings for the kit: work folder, engine build, check model, memory steps |
| [run/status.sh](run/status.sh) | The short status read used every 10 minutes during the climb (read-only) |
| [run/memwatch.sh](run/memwatch.sh) | Host memory every 2 s while the climb ran |
| [run/tables.py](run/tables.py) | Builds the tables from the climb's work folder |
| [run/greedy_diff.py](run/greedy_diff.py) | Builds `greedy_failures.csv` |
| [run/soak.py](run/soak.py) | The soak: one check server per card at the offsets `apply` set, every card's greedy check every round; `--keep-going` for a stock control |
| [run/soak_tables.py](run/soak_tables.py) | Builds `soak.csv` |
| [run/ornith_check.sh](run/ornith_check.sh) | Ornith's greedy check on all nine cards with its release settings, at stock (the reference) or with the profile set |
| [run/r200_chain.sh](run/r200_chain.sh) | One boot without a person in the loop: Ornith at stock, `apply`, the soak, Ornith with the profile |
| [run/env-ornith.sh](run/env-ornith.sh) | The settings `ornith_check.sh` and `r200_chain.sh` read: the Ornith file, its release settings, the hash folder |

## Commands

On the rig, as root, in the kit's folder (the `mem_offset` kit unpacked with `climb.py`, `greedy_check.py`,
`memoc_units.json`, `nvml_watchdog.py` and this folder's `run/` files beside it), with the owner at the console:

```bash
. ./env.sh
python3 climb.py state && python3 climb.py start-check && python3 climb.py plan
python3 climb.py trip-test && python3 climb.py watchdog start
setsid nohup sh memwatch.sh > /var/tmp/r200-mem.log 2>&1 < /dev/null &
python3 climb.py climb                  # detached; ends with /var/tmp/climb/CLIMB_DONE
bash status.sh                          # any time: the step, failures, hottest card, host memory
# after CLIMB_DONE: copy /var/tmp/climb off the rig, then
python3 climb.py watchdog stop
# reboot at the console, then
python3 climb.py start-check && python3 climb.py watchdog start
python3 climb.py apply /var/tmp/climb/out/profile.json    # detached; ends with /var/tmp/climb/APPLY_DONE
python3 soak.py --minutes 30          # after APPLY_DONE says "exit 0"; ends with /var/tmp/climb/SOAK_DONE
# a failed soak has set every offset to 0: copy the work folder off, lower the failing card one step in a copy of
# the profile, reboot, and apply it in a fresh work folder (WORK=...), or run the whole boot unattended:
PROFILE=<profile.json> setsid nohup bash r200_chain.sh > chain.console 2>&1 < /dev/null &
# the stock control, every offset 0 (on the boot of a failed soak):
python3 soak.py --minutes 30 --keep-going
```

The tables, from the copied work folders: `python3 run/tables.py WORK tables` (the first apply's folder, which also
holds the climb), `python3 run/greedy_diff.py WORK/out OUT.csv` for the climb and
`python3 run/greedy_diff.py WORK/soak OUT.csv WORK/out tryN-soak` for a soak, concatenated, and
`python3 run/soak_tables.py tables/soak.csv try1=WORK1 try2=WORK2 ... control-stock=WORK4`.

## Provenance and licence

Measured on the rig on 2026-11-17 with the engine build `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`
(Miner Llama v4's release candidate 2) and the kit as published on this branch. The check model is IBM's
Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; nothing of it is published here, only its name and
the hashes of its answers. Ornith-1.5-122B Pascal N2P GGUF and its licence are described in
[release/ornith-v3](../../../release/ornith-v3/README.md); only its hashes are published here. The tables are this
repository's licence.
