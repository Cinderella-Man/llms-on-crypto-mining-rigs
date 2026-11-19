# R217: the first rig's memory climb, a map for its soak ladder (2026-11-19)

**The answer in plain words.** The climb kit ([bench/mem_offset/climb.py](../../../bench/mem_offset/README.md)) ran
the memory half of its climb on the first nine-card P106-090 rig: every card's memory clock went up together from
+200 to +1,400 MT/s with the core at stock, and every card was checked at every step. It took 12 minutes 45 seconds
with no help. **Card 8 failed first, at +1,100 MT/s; cards 2 and 7 at +1,300; cards 0, 1, 3 and 6 at +1,400; cards 4
and 5 passed every step to +1,400.** As on the second rig, only the probability-hashed greedy check saw the failures:
all 2,697 kernel checks passed, no card reported an error to the driver, no check ran past its time limit and the
watchdog never tripped. In five of the seven failed checks the generated text was identical to stock and only the
token probabilities had moved. The hottest card was card 1 at 47 °C. The core was not climbed: it climbs later, with
each card's memory at a value that has held a soak ([Q1578](../../../quants/experiments.md)).

**What it means.**

- **This is a map, not a setting.** One check a step shows where a card fails at once, not where it never fails: on
  the second rig three cards failed a 30-minute soak one and two steps under the last step they had passed in its
  climb ([R200](../2026-11-17_r200_climb_rig2/README.md)). The settings come from the soak ladder, whose three rungs
  this climb gives ([rungs/](rungs/rungs-choice.md), below).
- **Card 0 fails where it failed two days earlier.** On 2026-11-17, alone and on another boot, card 0 failed at
  +1,400 MT/s with the core at 0, seen only by the same check ([Q133](../2026-11-17_q133_q135_clock_offsets/README.md));
  here it failed at +1,400 again and passed +1,300.
- **Card 4 passed +1,400 MT/s with the core at stock.** On 2026-11-17 it gave wrong kernel results and then hung at
  +1,200 MT/s with a +275 MHz core offset, a 2,138 MHz peak ([R194](../2026-11-17_r194_nine_cards_soak/README.md)).
  Tonight one check a step passed its memory to +1,400 at stock core, so the core is the likelier cause of that
  failure; one check a step does not prove it. Card 2 also held +1,200 MT/s in R194 (with +249 MHz) and here failed
  at +1,300.
- **The two rigs' cards differ.** Here one card failed at +1,100, two at +1,300, four at +1,400 and two passed
  +1,400; on the second rig one card failed at +1,200, one at +1,300, two at +1,400 and five passed +1,400. The
  check model and engine gave both rigs' cards the same stock hash, `60ee0f178cfd67d5`.
- **Streaming reads rise with the memory clock** ([tables/stream_gbs.csv](tables/stream_gbs.csv)): card 4 read
  179.23 GB/s at stock, 199.63 at +1,000, 204.86 at +1,200 and 209.32 at +1,400; card 5 179.12, 199.77, 204.88 and
  209.25.

**The climb's map** ([profile.json](profile.json), [tables/cards.csv](tables/cards.csv)) **and tomorrow's rungs**
([rungs/](rungs/rungs-choice.md)):

| Card | Stock peak core clock | Memory: last clean, failed at | Profile (one step under the last clean) | Memory cap for the ladder | Rung 1 | Rung 2 | Rung 3 |
|---|---|---|---|---|---|---|---|
| 0 | 1,860 MHz | +1,300, +1,400 | +1,200 MT/s | +1,200 (2 steps under +1,400) | +1,000 | +1,100 | +1,200 |
| 1 | 1,860 MHz | +1,300, +1,400 | +1,200 MT/s | +1,200 (2 steps under +1,400) | +1,000 | +1,100 | +1,200 |
| 2 | 1,898 MHz | +1,200, +1,300 | +1,100 MT/s | +1,100 (2 steps under +1,300) | +900 | +1,000 | +1,100 |
| 3 | 1,911 MHz | +1,300, +1,400 | +1,200 MT/s | +1,200 (2 steps under +1,400) | +1,000 | +1,100 | +1,200 |
| 4 | 1,885 MHz | +1,400, — | +1,300 MT/s | none | +1,100 | +1,200 | +1,300 |
| 5 | 1,911 MHz | +1,400, — | +1,300 MT/s | none | +1,100 | +1,200 | +1,300 |
| 6 | 1,885 MHz | +1,300, +1,400 | +1,200 MT/s | +1,200 (2 steps under +1,400) | +1,000 | +1,100 | +1,200 |
| 7 | 1,911 MHz | +1,200, +1,300 | +1,100 MT/s | +1,100 (2 steps under +1,300) | +900 | +1,000 | +1,100 |
| 8 | 1,885 MHz | +1,000, +1,100 | +900 MT/s | +900 (2 steps under +1,100) | +700 | +800 | +900 |

Every rung keeps the core at 0; with no core value in the profile there is no core rung. The rig's profile starts
from this map alone: the earlier sittings on its cards (Q133, R193 and R194) are set aside, so the caps come only
from this climb's failures, and no earlier failure limits a card's memory here or its core in the later core climb.

## What was done

The first nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off, the
stock 75 W power limit), booted at 09:19 and never given a clock offset on that boot; it had run other rows at stock
that day and was idle from 19:43. The kit ran as published at this repository's `v5` (`climb.py`, `greedy_check.py`,
`nvml_memoc.py`, `memoc_units.json`, [scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py)), with
`mmvq_pascal` from the kit's tarball, every file checked by MD5 on the rig. This rig's settings are in
[run/env.sh](run/env.sh):

- **The check model:** Granite 3.1 3B-A800M Instruct Q4_K_M (2.02 GB, `bartowski/granite-3.1-3b-a800m-instruct-GGUF`
  revision `be9a36f0`, Apache-2.0, MD5 `c8e8b8bc7890057906d99643dbf8063b`), the second rig's check model, downloaded
  onto this rig for the climb. One copy served on each card with a 1,024-token context, by the engine build the
  second rig's climb used (`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`).
- **The steps:** memory +200, +400, +600, +800, +900, +1,000, +1,100, +1,200, +1,300, +1,400 MT/s on every card still
  climbing, the core at 0.
- **No core phase:** `CORE_PEAK_MAX=0` puts the core climb's top below every card's stock peak, so it has no step.
  Before the run, `climb.py plan` with that setting listed no core step, and a dry run of the whole climb
  (`--dry-run --core-peak-max 0 climb`, a simulated rig) set the memory up to +1,400 and then every card to 0, with
  no core offset set at any point. The profile marks cards 4 and 5, which never failed, as having taken part in the
  core climb; their core values are 0, and no core offset was ever set.
- **Each step, on every card at once:** the greedy check (two fixed prompts, 256 tokens each, temperature 0, every
  token's top-10 log-probabilities hashed) against the card's own stock hash, then `mmvq_pascal`'s Q8_0 `attn_q`
  checks (31 kernel checks and a streaming read), at most two cards at a time. Each check had a time limit of 3 × its
  stock time, at least 30 s: 38 s (card 3: 37 s) for the greedy check, 30 s for the kernel checks.
- **A card that fails** is set back to 0 at once, both offsets read back, its server stopped, and it leaves the climb
  while the others go on.
- **The watchdog** on every card throughout: every 0.5 s it logged clocks, power, temperature and both offsets, and
  would have set every card to 0 above 70 °C, above 80 W for 5 s, or on an Xid that names no card; an Xid naming one
  card would have tripped that card only.

The order on the rig: the kit's read-only state and start check, `plan`, the dry run, the watchdog's four trip tests,
the watchdog, the climb; afterwards every offset read back, the watchdog stopped, the results copied off. Then, at a
desk, `climb.py rungs` wrote the soak ladder's three rungs from the climb's profile and its own failures.

## Results

### Before any offset

- **Start check:** no load process, 0 MiB, the default 75 W power limit, both offsets 0 and P8 on all nine cards, no
  Xid this boot, no step down in the kit's ledger; 3,142 MiB of host memory available.
- **Trip tests, all four passed** on all nine cards with stand-in processes: a 15 °C temperature limit, a kernel-log
  test line naming no card, and a 1 W power limit each set every card to 0 and killed every stand-in; a kernel-log line
  naming card 0 killed only card 0's stand-in and tripped nothing else.
- **Loading:** the nine check servers loaded one after another in 2 minutes 59 seconds (cards 0 to 8: 15, 31, 15, 12,
  18, 26, 19, 21 and 22 s). With all nine loaded, 1,836 MiB of the 4 GB host's memory was available.
- **Stock, two rounds over 102 s:** every card gave the same greedy hash, `60ee0f178cfd67d5`, in both rounds. The
  greedy check took 12.7 s on eight cards and 12.2 s on card 3; the kernel checks 7.6, 7.2, 6.6, 7.1, 6.6, 6.6, 6.7,
  7.0 and 6.5 s on cards 0 to 8 ([tables/steps.csv](tables/steps.csv)). Streaming reads at stock in GB/s, cards 0 to 8: 179.41, 179.29, 179.50,
  179.14, 179.23, 179.12, 179.31, 179.15, 179.41. Stock peak core clocks, cards 0 to 8: 1,860, 1,860, 1,898, 1,911,
  1,885, 1,911, 1,885, 1,911 and 1,885 MHz; R193's two days earlier read cards 2 and 8 one driver step higher (1,911
  and 1,898).

### The climb, 20:03:05 to 20:15:50

| Time | Step | What happened |
|---|---|---|
| 20:07:46–20:13:01 | memory +200 to +1,000 | all nine pass every step |
| 20:13:19 | memory +1,100 | **card 8 fails** (greedy output differs) |
| 20:13:50–20:14:35 | memory +1,200 | cards 0 to 7 pass |
| 20:14:51, 20:14:52 | memory +1,300 | **cards 2 and 7 fail** |
| 20:15:28–20:15:33 | memory +1,400 | **cards 0, 1, 3 and 6 fail**; cards 4 and 5 pass |
| 20:15:42–20:15:50 | end | every offset of every card set to 0 and read back 0; the profile written |

The memory climb itself took 7 minutes 55 seconds for its ten steps (20:07:46 to 20:15:41). Every check of
every card at every step is in [tables/steps.csv](tables/steps.csv).

- **How the failures look** ([tables/greedy_failures.csv](tables/greedy_failures.csv), made by
  [run/greedy_diff.py](run/greedy_diff.py)): at stock and at every clean step each card's output was bit-identical to
  its stock run, so any difference is an error. In six of the seven failed checks only one of the two prompts
  differed; card 1 at +1,400 differed in both. Cards 0, 3, 6, 7 and 8 generated text identical to stock in both
  prompts, only the probabilities moved, by at most 0.0707, 0.0897, 0.0528, 0.203 and 0.0469 in the chosen tokens'
  log-probabilities. Card 1's text parted at token 62 of prompt 1 (190 tokens differ); card 2's changed one token,
  token 82 of prompt 2.
- **No other check saw anything:** 2,697 of 2,697 kernel checks passed (a card that fails the greedy check runs no
  kernel check at that step), no Xid, no check past its limit, 0 watchdog trips.
- **Heat and power** ([tables/heat_power.csv](tables/heat_power.csv)): the hottest card was card 1 at 47 °C, in the
  memory climb. The highest half-second power readings were 84.14 W (card 4) and 82.53 W (card 8); no card had two
  readings in a row above 80 W, and none showed a thermal or hardware slowdown.

### At the end

Every memory and core offset of every card read back 0; no llama, check or watchdog process left; 0 MiB on every
card. The kit's ledger holds nine step downs for this boot (seven failures, and cards 4 and 5 at the end), so the rig
needs a fresh boot before any offset; the ladder starts on the next boot.

## The soak ladder's rungs

`climb.py rungs` (file work, no rig) wrote three rung profiles from [profile.json](profile.json) and the climb's own
failures, the only ones counted (the earlier sittings are set aside). By the kit's rule each card's top
rung is its profile value, lowered to two steps under the lowest memory value it ever failed at; rungs 1 and 2 are
two steps and one step under it ([the rule](../../../bench/mem_offset/README.md#the-ladders-rungs-climbpy-rungs)).
The files: [rungs/rung1.json](rungs/rung1.json), [rung2.json](rungs/rung2.json), [rung3.json](rungs/rung3.json), and
[rungs/rungs-choice.md](rungs/rungs-choice.md) with each card's caps and their source. The ladder is
`PROFILE=rung1.json LADDER="rung2.json rung3.json"` for `soak_chain.sh` on a fresh boot.

The ladder (R218) runs on the next boot with `soak_chain.sh` and [run/env-chain.sh](run/env-chain.sh), as the second
rig's ladder did ([R201](../2026-11-18_r201_soak_ladder/README.md)): Ornith-1.5-122B Pascal N2P GGUF at stock with its v4
settings (the reference), `apply` of rung 1, 30 minutes a rung with every card's differing checks counted, and, if no
card differed, Ornith again for 30 minutes with the offsets; est. 2 h 30 min. The rung files, the failures, the
settings and Ornith's settings file are staged in the kit's folder ([run/MD5SUMS-r218](run/MD5SUMS-r218)):

```bash
cd /var/tmp/oc/mem_offset-kit-20261024 && md5sum -c MD5SUMS-r218
. ./env-chain.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/r218 && setsid nohup bash soak_chain.sh > /var/tmp/r218/chain.console 2>&1 < /dev/null &
cat /var/tmp/r218/CHAIN_DONE     # when it exists: what happened, in one line
```

## Files

| File | What |
|---|---|
| [profile.json](profile.json) | the climb's profile: each card at one step under its last clean memory step, core 0 |
| [rungs/](rungs/rungs-choice.md) | the ladder's three rung profiles and each card's caps |
| [tables/cards.csv](tables/cards.csv) | one row per card: stock peak core clock, last clean memory step, the step that failed and why, the profile |
| [tables/steps.csv](tables/steps.csv) | every check of every card at every step (offsets, expected and measured peak core clock, greedy verdict and time, kernel checks, streaming read, Xid, verdict) |
| [tables/stream_gbs.csv](tables/stream_gbs.csv) | each card's streaming read at stock and at each memory step (GB/s) |
| [tables/heat_power.csv](tables/heat_power.csv) | per phase and card, from the watchdog's samples: temperature, power, readings above 80 W, slowdowns, peak clocks |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | per failed check and prompt: text identical or not, the first token and first probability that differ |
| [tables/watchdog.txt](tables/watchdog.txt) | the watchdog's start and stop lines |
| [run/env.sh](run/env.sh) | this rig's settings for the kit in this step; for the later steps the rig's `env.sh` is the same without `CORE_PEAK_MAX=0`, which step 3's core climb must not have ([the runbook](../../../bench/mem_offset/README.md#on-a-new-rig-every-command-from-a-fresh-install)) |
| [run/failures.json](run/failures.json) | this climb's seven failures, as `soak.py --failures` reads them: the ladder's `FAILURES` |
| [run/env-chain.sh](run/env-chain.sh), [run/MD5SUMS-r218](run/MD5SUMS-r218) | the ladder's settings for `soak_chain.sh` on the next boot (R218), and the MD5s of the files staged for it |
| [run/tables.py](run/tables.py), [run/greedy_diff.py](run/greedy_diff.py) | the tables from the climb's work folder (R200's scripts) |

## Commands

On the rig, as root, the kit staged in `/var/tmp/oc/mem_offset-kit-20261024` (the tarball's `bin/mmvq_pascal`, the
`v5` kit files and `scripts/nvml_watchdog.py` copied over it, [run/env.sh](run/env.sh) beside them):

```bash
cd /var/tmp/oc/mem_offset-kit-20261024 && . ./env.sh
python3 climb.py state && python3 climb.py start-check && python3 climb.py plan        # read-only; plan lists no core step
python3 climb.py --dry-run --work /var/tmp/r217-dryrun/w --ledger /var/tmp/r217-dryrun/ledger.log \
  --check-model /var/tmp/r217-dryrun/check.gguf --core-peak-max 0 climb                 # simulated; sets nothing
python3 climb.py trip-test
python3 climb.py start-check && python3 climb.py watchdog start && python3 climb.py climb   # detached
cat /var/tmp/r217/CLIMB_DONE && python3 climb.py state      # exit 0; every offset 0
python3 climb.py watchdog stop
```

At a desk, with the work folder copied off the rig to `WORK`:

```bash
python3 -I docs/data/2026-11-19_r217_climb_rig1/run/tables.py $WORK docs/data/2026-11-19_r217_climb_rig1/tables
python3 -I docs/data/2026-11-19_r217_climb_rig1/run/greedy_diff.py $WORK/out docs/data/2026-11-19_r217_climb_rig1/tables/greedy_failures.csv
python3 -I bench/mem_offset/climb.py rungs docs/data/2026-11-19_r217_climb_rig1/profile.json \
  --out docs/data/2026-11-19_r217_climb_rig1/rungs --overwrite
```

Provenance: measured on the rig on 2026-11-19 by the project. The check model is Apache-2.0; nothing of it is
published here. The tables are this project's own measurements.
