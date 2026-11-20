# R223: the first rig's core soak ladder, step 4 of its own profile (2026-11-20)

**The answer in plain words.** On a fresh boot of the first nine-card P106-090 rig, with nobody at the console, every
card's memory sat at the value that held step 2's soak ([R218](../2026-11-20_r218_soak_rig1/README.md)) while the
core of eight cards went up one climb step every 30 minutes, to the value two climb steps under where each had failed
in step 3's core climb ([R221](../2026-11-20_r221_core_climb_rig1/README.md); [Q1566](../../../quants/experiments.md),
the kit's [soak ladder](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check)).
Card 6, whose core had failed at its very first step, kept its core at stock and its memory at +1,000 MT/s. **Seven
cards never differed from stock** (1, 2, 3, 4, 5, 7 and 8, each at its top rung). **Card 0 differed once** at its top
core, +269 MHz, in round 112 of the last rung. **Card 6 differed three times in its first 91 rounds** with its core at
stock and its memory at +1,000 MT/s, the value it had held for 30 clean minutes this morning, and left the soak with
both offsets at 0. All 4,031 other checks were bit-identical to the stock answer. Because a card differed, the chain set
every offset to 0 and did not serve Ornith with the offsets on this boot. The watchdog never tripped, no card reported
an error to the driver, no card slowed down, and the hottest card reached 46 °C.

**The rig's profile** ([use.json](use.json), written by `soak.py` itself, each card's reason in
[use-choice.md](use-choice.md)): each card's highest clean rung, but no higher than two steps under the lowest value at
which that offset ever failed on that card, in steps 1 to 4
([the rule](../../../bench/mem_offset/README.md#a-rigs-profile-the-same-five-steps-on-every-rig)):

| Card | Memory | Core: rungs 1 / 2 / 3 (MHz) | Differing checks | Highest clean rung | Core cap and why | **Use (MT/s, MHz)** |
|---|---|---|---|---|---|---|
| 0 | +900 MT/s | +218 / +244 / +269 | 1, at +269 (rung 3, round 438) | +900, +244 | +218: this soak, at +269 | **+900, +218** |
| 1 | +1,000 | +206 / +231 / +256 | 0 | +1,000, +256 | +256: R221's climb failed at +307 | **+1,000, +256** |
| 2 | +1,100 | +206 / +231 / +256 | 0 | +1,100, +256 | +256: R221, at +307 | **+1,100, +256** |
| 3 | +1,200 | +155 / +180 / +206 | 0 | +1,200, +206 | +206: R221, at +256 | **+1,200, +206** |
| 4 | +1,100 | +155 / +180 / +206 | 0 | +1,100, +206 | +206: R221, at +256 | **+1,100, +206** |
| 5 | +1,300 | +180 / +206 / +231 | 0 | +1,300, +231 | +231: R221, at +282 | **+1,300, +231** |
| 6 | +1,000 | 0 (held) | **3, at +1,000 MT/s and core 0** (rung 1, rounds 43, 55, 91); left | none | memory +800 (this soak, at +1,000); core 0 (R221, at +28) | **0, 0** |
| 7 | +1,100 | +231 / +256 / +282 | 0 | +1,100, +282 | +282: R221, at +332 (a hang) | **+1,100, +282** |
| 8 | +900 | +180 / +206 / +231 | 0 | +900, +231 | +231: R221, at +282 | **+900, +231** |

Every memory value is R218's use profile, unchanged on the eight cards that stayed. Card 6 ends at stock: it held no
rung clean, and the kit gives a card that held nothing 0 for both offsets. Its memory cap is now +800 MT/s. This
profile is what step 5 sets on the next fresh boot, with Ornith's whole-memory check.

**What it means.**

- **Card 6's memory is the weak part, not its core.** In step 3 card 6 failed at its first core step, +28 MHz, with its
  memory at +1,000 MT/s, and one check could not say which offset caused it
  ([Q1636](../../../quants/experiments.md)). Here, with its core at stock and the same memory, it gave wrong answers
  in rounds 43, 55 and 91 of the first rung, about once every 30 rounds (5 to 6 minutes). So +1,000 MT/s errs on its
  own, and step 3's failure at +28 MHz says nothing sure about card 6's core.
- **The same card at the same setting held 30 minutes on one boot and failed within 9 minutes on the next.** This
  morning card 6 ran 151 rounds at +1,000 MT/s with core 0 and then 151 at +1,100 without a differing check
  (R218). This afternoon, at +1,000 MT/s with core 0 again, it differed in its 43rd round, 8 minutes 26 seconds into
  the rung. It was not hotter: 37 °C at
  most in this rung against 43 °C in R218's first rung. Thirty clean minutes show only that a card errs less often
  than about once in 10 minutes; card 6's +1,000 sat near that line. This is why the kit keeps every value two steps
  under any failure, and why step 5 checks the whole profile again on the served model.
- **Card 0's top core was one step too many.** It ran +218 and +244 MHz clean for an hour and differed once at +269,
  the value two climb steps under its climb failure (+320): one wrong answer in 112 rounds. Its use value is +218 MHz,
  two climb steps under +269. In every one of the four failed checks one of the two prompts was bit-identical to stock;
  in three the other prompt's text was identical too, and only the token probabilities moved
  ([tables/greedy_failures.csv](tables/greedy_failures.csv)).
- **The other seven cards held their full core margin.** Each ended at its top rung, two climb steps under its first
  core failure in R221, with no differing check in 493 rounds. The second rig's core ladder
  ([R202b](../2026-11-18_r202b_core_soak/README.md)) ended the same way on all nine cards.

## What was done

The first nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), freshly booted at
about 12:34, with nothing run on it before: every card at 405 MHz memory and 0 MiB used, the start check (12:36:33)
showing every offset 0, no load, no Xid this boot, no offset lowered since the boot, 3,164 MiB of host memory
available. The kit as published on this branch (`soak.py`, `soak_chain.sh`, `climb.py`, `greedy_check.py`,
`model_check.sh`, `nvml_memoc.py`, md5-identical to [bench/mem_offset](../../../bench/mem_offset/README.md)), the
first rig's kit settings ([R221's run/env.sh](../2026-11-20_r221_core_climb_rig1/run/env.sh)) and this run's settings in
[run/env-step4.sh](run/env-step4.sh), every staged file checked by [run/MD5SUMS-step4](run/MD5SUMS-step4) on the rig
before the start, and the failures file's md5 (`1fdfb6a84237f1893a8fe969b91f8011`) checked too. The engine build was
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`.

`soak_chain.sh` ran the whole boot detached (12:36:48 to 14:28:35):

1. **The start check, then Ornith-1.5-122B Pascal N2P GGUF at stock** with its v4 release settings
   ([release/ornith-v4](../../../release/ornith-v4/settings.env)) on all nine cards, the reference for a check with
   the offsets: loaded in 672 s, two rounds of the kit's greedy check (12:48:44 and 12:49:20), the same hash both
   times. The start check passed again after it.
2. **The watchdog, then `climb.py apply` with rung 1** (12:49:28 to 12:55:04): every card's stock check, the offsets
   set and read back, and one check round at rung 1: every card passed.
3. **The ladder** (`soak.py --minutes 30 --keep-going --ladder`, with steps 1 to 3's 20 failures in `--failures`,
   12:55:07 to 14:28:24): one check server per card with the check model (Granite 3.1 3B-A800M Instruct, Q4_K_M),
   ready by 12:57:49, and every round each card's greedy check (two prompts, 256 tokens each, temperature 0, every
   token's top-10 log-probabilities hashed) against its stock hash. After each 30-minute rung the cards that had not
   differed went up to the next rung's values, read back; a card that differed stayed where it was and went on being
   checked. A card that differed three times was set to 0 alone and left (card 6, at 13:15:37).
4. **Ornith with the offsets** for 30 minutes, which runs only if no card differed: skipped. The chain set every
   offset of every card to 0 (14:28:25 to 14:28:35) and ended.

The rungs ([R221's rungs/](../2026-11-20_r221_core_climb_rig1/rungs/), written by R221's
[step4_rungs.py](../2026-11-20_r221_core_climb_rig1/run/step4_rungs.py) from R218's use profile and R221's climb, card
6 held at core 0), core in MHz on cards 0 to 8, the memory as in the table above:

| Rung | Core |
|---|---|
| 1 | +218, +206, +206, +155, +155, +180, 0, +231, +180 |
| 2 | +244, +231, +231, +180, +180, +206, 0, +256, +206 |
| 3 | +269, +256, +256, +206, +206, +231, 0, +282, +231 |

Rung 3 is R221's result profile: each card one climb step under its last clean core step there.

## Results

### Every card and rung ([tables/rungs.csv](tables/rungs.csv), [tables/cards.csv](tables/cards.csv))

Rounds a rung: 159, 167 and 167 (rungs 2 and 3 were quicker with card 6 gone). Each cell is the card's core offset
and its differing checks in that rung.

| Card | Rung 1 | Rung 2 | Rung 3 | Rounds checked |
|---|---|---|---|---|
| 0 | +218: 0 | +244: 0 | +269: **1** (round 438) | 493 |
| 1 | +206: 0 | +231: 0 | +256: 0 | 493 |
| 2 | +206: 0 | +231: 0 | +256: 0 | 493 |
| 3 | +155: 0 | +180: 0 | +206: 0 | 493 |
| 4 | +155: 0 | +180: 0 | +206: 0 | 493 |
| 5 | +180: 0 | +206: 0 | +231: 0 | 493 |
| 6 | 0 (memory +1,000): **3** (rounds 43, 55, 91), left | (left) | (left) | 91 |
| 7 | +231: 0 | +256: 0 | +282: 0 | 493 |
| 8 | +180: 0 | +206: 0 | +231: 0 | 493 |

Every differing check with its round in the rung and the rounds the card had run at that value before it is in
[tables/differs.csv](tables/differs.csv). All 4,031 other checks gave the stock hash `60ee0f178cfd67d5`, the same on
every card and on both rigs; the slowest check took 14.4 s against the 45 s limit.

### The four failed checks ([tables/greedy_failures.csv](tables/greedy_failures.csv))

| Card | Round | Prompt 1 | Prompt 2 |
|---|---|---|---|
| 0 | 438 | bit-identical to stock | text identical; probabilities differ from token 96 |
| 6 | 43 | text parts from stock at token 62 (189 of 256 tokens differ); probabilities from token 22 | bit-identical to stock |
| 6 | 55 | bit-identical to stock | text identical; probabilities differ from token 163 |
| 6 | 91 | text identical; probabilities differ from token 150 | bit-identical to stock |

### Ornith at stock ([tables/ornith.csv](tables/ornith.csv), [tables/ornith_load.csv](tables/ornith_load.csv))

Ornith filled 5,603, 5,635, 5,549, 5,635, 5,243, 5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8, with 1,498 MiB of
host memory left. Both rounds gave the greedy hash `adf47784f9b02d4b`, R218's and the second rig's. Generation in
round 1: 14.22 and 14.86 tok/s (prompts 1 and 2); in round 2: 15.65 and 15.63 tok/s; prompt processing in round 2:
26.83 and 29.64 tok/s. With a card differing, the run with the offsets did not follow, so this boot has no offset
speed for Ornith; step 5 measures it.

### Apply's check, streaming reads and peaks ([tables/apply.csv](tables/apply.csv))

Every card passed the greedy check and the 31 kernel checks at stock and at rung 1. Streaming reads in GB/s, cards 0
to 8, at stock: 179.26, 179.47, 179.34, 179.23, 179.26, 179.24, 179.37, 179.32, 179.17; at rung 1: 197.53, 199.58,
201.98, 204.62, 202.25, 207.02, 199.25, 202.15, 197.36. Peak core clock at stock: 1,860, 1,860, 1,911, 1,911, 1,885,
1,911, 1,885, 1,911, 1,885 MHz; at rung 1: 2,075, 2,062, 2,100, 2,050, 2,037, 2,088, 1,885, 2,138, 2,062 MHz.

### Heat, power and clocks ([tables/heat_power.csv](tables/heat_power.csv))

From the watchdog's half-second samples (12:49:29 to 14:34:08): the hottest card was 46 °C (card 1 in rung 3); no
card was in a thermal or hardware slowdown in any sample; the highest power reading was 83.68 W (card 7 in rung 3).
Readings above 80 W were single samples, never two in a row: 1 in `apply` (card 8) and 87 in the rungs, on every card
but 6 and 8, against the watchdog's limit of 80 W for 5 s. Each card's peak core clock in rungs 1, 2 and 3:

| Card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|
| Rung 1 (MHz) | 2,075 | 2,062 | 2,113 | 2,062 | 2,037 | 2,088 | 1,885 | 2,138 | 2,062 |
| Rung 2 | 2,088 | 2,075 | 2,126 | 2,075 | 2,050 | 2,113 | (left) | 2,164 | 2,075 |
| Rung 3 | 2,113 | 2,100 | 2,151 | 2,100 | 2,062 | 2,138 | (left) | 2,189 | 2,100 |

The memory clock the driver reported peaked at 4,657 MHz (card 5 at +1,300 MT/s). The watchdog stopped after 0 trips,
and the boot logged no Xid.

### The end

At 14:28:24 the soak ended (`exit 1 rounds 493 differs 4 left 6`) and wrote [soaked.json](soaked.json),
[use.json](use.json) and [use-choice.md](use-choice.md); the chain set every offset of every card to 0 (14:28:25 to
14:28:35). Then the watchdog was stopped and the state read back (14:34): every offset 0, P8, 0 MiB on every card, no
load, no Xid this boot, nine offsets lowered since the boot (so the next step needs a reboot), 3,182 MiB of host
memory available.

## Files

| File | What it holds |
|---|---|
| [use.json](use.json) | The rig's profile, by the rule above, as `climb.py apply` reads it; its `failures` are steps 1 to 4's (22) |
| [use-choice.md](use-choice.md) | Each card's highest clean rung, its memory and core caps and where they come from, and the value to use, as `soak.py` wrote it |
| [soaked.json](soaked.json) | `soak.py`'s profile: each card at the highest rung it held clean, with every rung's offsets and differing checks |
| [tables/cards.csv](tables/cards.csv) | One row per card: each rung's offsets and differing checks, the rounds checked, the rung it left in, the soaked value, the first difference, the use value |
| [tables/differs.csv](tables/differs.csv) | Every differing check: time, round, card, rung, offsets, its round in the rung, the card's rounds at that value before it |
| [tables/rungs.csv](tables/rungs.csv) | `soak.py`'s rows per rung and card: offsets, rounds, passes, differing checks, the first differing round, whether it left |
| [tables/soak.csv](tables/soak.csv) | Every card's every round: hash, seconds, verdict, rung and offsets |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | Each failed check against the card's stock run, by prompt (made with [R217's run/greedy_diff.py](../2026-11-19_r217_climb_rig1/run/greedy_diff.py)) |
| [tables/ornith.csv](tables/ornith.csv) | Ornith's rounds (two at stock here): hash, whether it equals stock round 1, prompt and generation tok/s of both prompts |
| [tables/ornith_load.csv](tables/ornith_load.csv) | Ornith's load: seconds to ready, MiB per card, host memory left, rounds |
| [tables/apply.csv](tables/apply.csv) | `apply`'s stock round and its round at rung 1: peaks, the greedy and kernel checks, streaming reads |
| [tables/heat_power.csv](tables/heat_power.csv) | Each phase (apply, the soak's server start, each rung, each raise) and card from the watchdog's samples |
| [run/env-step4.sh](run/env-step4.sh) | This run's settings for `soak_chain.sh`: work folder, the rungs, steps 1 to 3's failures, Ornith and its settings, the minutes |
| [run/MD5SUMS-step4](run/MD5SUMS-step4) | The md5 sums of the rungs and settings staged on the rig (`ornith-v4.env` is [release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)), checked there before the start |
| [run/tables.py](run/tables.py) | Builds the tables from the work folder and Ornith's folder |

The rungs and the failures file are R221's published files: [rungs/step4-rung1.json](../2026-11-20_r221_core_climb_rig1/rungs/step4-rung1.json)
to [step4-rung3.json](../2026-11-20_r221_core_climb_rig1/rungs/step4-rung3.json) and
[rungs/failures/rung1.json](../2026-11-20_r221_core_climb_rig1/rungs/failures/rung1.json).

## Commands

On the rig, as root, in the kit's folder (the first rig's, with the kit files of this branch, `env.sh`, R221's staged
rungs and `env-step4.sh` beside them, `ornith-v4.env` a copy of Ornith's v4 settings), on a fresh boot:

```bash
md5sum -c MD5SUMS-step4 && md5sum <step 3's failures/rung1.json> && . ./env-step4.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/oc-step4 && ENV_FILE=./env-step4.sh setsid nohup bash soak_chain.sh > /var/tmp/oc-step4.console 2>&1 < /dev/null &
cat /var/tmp/oc-step4/CHAIN_DONE      # a look, at most every 30 minutes
# when CHAIN_DONE exists: the chain already set every offset to 0 because a card differed; stop the watchdog and
# read the state back, then copy /var/tmp/oc-step4 and /var/tmp/oc-step4-model off the rig
python3 climb.py watchdog stop; python3 climb.py state
```

The tables, on any machine, from the copied folders:

```bash
python3 -I run/tables.py WORK ORNITH_OUT tables
python3 -I ../2026-11-19_r217_climb_rig1/run/greedy_diff.py WORK/soak tables/greedy_failures.csv WORK/out r223-ladder
# the profile again by the kit's rule (byte-identical to use.json):
python3 -I ../../../bench/mem_offset/soak.py --use-from soaked.json \
  --failures ../2026-11-20_r221_core_climb_rig1/rungs/failures/rung1.json --use-out use-again.json
```

The next step on this rig, the profile on the served model, on the next fresh boot (the kit README's
[five steps](../../../bench/mem_offset/README.md#a-rigs-profile-the-same-five-steps-on-every-rig), step 5), with
R209's [apply_check.sh](../2026-11-19_r209_oc_v5/run/apply_check.sh) and Ornith's v5 settings copied into the kit
folder `$K`:

```bash
KIT=$K BIN=<the v5 engine's bin folder> MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  SETTINGS=$K/ornith-v5.env PROFILE=/var/tmp/oc-step4/soak/use.json WORK=/var/tmp/oc-step5 OUT=/var/tmp/oc-step5-model \
  setsid -f nohup bash $K/apply_check.sh > /var/tmp/oc-step5.console 2>&1 < /dev/null
cat /var/tmp/oc-step5/R209C_DONE      # "done 0": set and checked; the offsets stay set, the watchdog runs
```

## Provenance and licence

Measured on the rig on 2026-11-20, 12:36 to 14:34, with the engine build
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828` (Miner Llama v4's release candidate 2) and the kit as
published on this branch. The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M;
Ornith-1.5-122B Pascal N2P GGUF and its licence are described in
[release/ornith-v3](../../../release/ornith-v3/README.md). Neither model is published here, only the hashes of their
answers. The tables are this repository's licence.
