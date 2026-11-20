# R218: the first rig's memory soak ladder, step 2 of its own profile (2026-11-20)

**The answer in plain words.** On one fresh boot, with nobody at the console, all nine cards of the first rig ran the
kit's soak for an hour and a half: three rungs of 30 minutes, every card's memory one step (100 MT/s) higher each
rung, the core at stock, every card's greedy check against its stock answer every round (3,980 checks in all). The
rungs came from that rig's memory climb the evening before
([R217](../2026-11-19_r217_climb_rig1/README.md)), its step 1. **Five cards never differed from stock** (2, 3, 5, 7
and 8, each at its top rung). **Four did, nine times in all:** card 0 twice at +1,100 MT/s, cards 1 and 6 three times
each at +1,200, so both left the soak with their offsets set back to 0, and card 4 once at +1,300. In every one of
the nine failed checks one of the two prompts was bit-identical to stock; in eight the generated text was identical
too, and only the token probabilities moved. Because a card differed, the chain set every offset to 0 and did not
serve Ornith with the offsets on that boot. The hottest card reached 52 °C; the watchdog never tripped and no card
reported an error to the driver.

**The setting for the next step** ([use.json](use.json), written by `soak.py` itself, each card's reason in
[use-choice.md](use-choice.md)): each card's highest clean rung, but no higher than two memory steps under the lowest
offset at which it ever failed, in R217's climb or in this soak
([the rule](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check)):

| Card | Highest clean rung (soaked.json) | First differed here | The cap and why | **Use** |
|---|---|---|---|---|
| 0 | +1,000 MT/s | rung 2, at +1,100 (round 167) | +900: this soak, at +1,100 | **+900** |
| 1 | +1,100 | rung 3, at +1,200 (round 311) | +1,000: this soak, at +1,200 | **+1,000** |
| 2 | +1,100 | never | +1,100: R217's climb failed at +1,300 | **+1,100** |
| 3 | +1,200 | never | +1,200: R217's climb failed at +1,400 | **+1,200** |
| 4 | +1,200 | rung 3, at +1,300 (round 464) | +1,100: this soak, at +1,300 | **+1,100** |
| 5 | +1,300 | never | none: it never failed (R217 passed +1,400) | **+1,300** |
| 6 | +1,100 | rung 3, at +1,200 (round 319) | +1,000: this soak, at +1,200 | **+1,000** |
| 7 | +1,100 | never | +1,100: R217's climb failed at +1,300 | **+1,100** |
| 8 | +900 | never | +900: R217's climb failed at +1,100 | **+900** |

The core is 0 on every card. Step 3, the core climb from this memory (`climb.py --from-profile use.json climb`), starts
on the next fresh boot; `use.json` lists every failure it used under `failures`, so it also carries steps 1 and 2's
failures to step 4.

**What it means.**

- **The climb's one check a step overstated four cards by one or two steps.** Card 0 was clean to +1,300 MT/s in
  R217's climb and differed at +1,100 here; cards 1 and 6 were clean to +1,300 and differed at +1,200; card 4 passed
  the climb's top step, +1,400, and differed at +1,300. The second rig showed the same
  ([R200](../2026-11-17_r200_climb_rig2/README.md), [R201](../2026-11-18_r201_soak_ladder/README.md)): a climb is a
  map, and the soak sets the values.
- **How often a card errs tells how far over its edge it is.** Cards 1 and 6 differed three times in 42 and 30 rounds
  at +1,200 MT/s, about once every 10 to 14 rounds. Card 0 differed twice in 151 rounds of rung 2 at +1,100, stayed
  there (a card that differed is not raised) and then ran all 170 rounds of rung 3 at the same +1,100 clean: two
  errors in 321 rounds. Card 4 differed once in 170 rounds at +1,300, in round 162 of the rung. Thirty clean minutes
  would not have found card 0's or card 4's errors with any certainty, which is why the use profile keeps two steps
  under each failure.
- **An error passes, again.** In all nine failed checks one prompt was bit-identical to stock, and in eight the text
  of the other was identical too ([tables/greedy_failures.csv](tables/greedy_failures.csv)): the model in each card's
  memory stayed intact and one answer's probabilities moved. Only card 6's check in round 322 changed one token, token
  82 of 256 in prompt 2.
- **Five cards end at their top rung,** which was already their cap from the climb (cards 2, 3, 7 and 8) or the
  climb's last step (card 5). The ladder could not have offered them more.

## What was done

The first nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), freshly booted at
about 09:31, with nothing run on it before: every card at 405 MHz memory and 0 MiB used, the start check showing every
offset 0, no load, no Xid this boot, no offset lowered since the boot, 3,170 MiB of host memory available. The kit as
published on this branch (`soak.py`, `soak_chain.sh`, `climb.py`, `greedy_check.py`, `model_check.sh`,
`nvml_memoc.py`, md5-identical to [bench/mem_offset](../../../bench/mem_offset/README.md)), R217's kit folder and
settings ([R217's run/env.sh](../2026-11-19_r217_climb_rig1/run/env.sh)) and this run's settings in
[run/env-chain.sh](run/env-chain.sh), every file checked by [run/MD5SUMS-r218](run/MD5SUMS-r218) on the rig before
the start. The engine build was `llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`.

`soak_chain.sh` ran the whole boot detached (09:37:26 to 11:29:11):

1. **The start check, then Ornith-1.5-122B Pascal N2P GGUF at stock** with its v4 release settings
   ([release/ornith-v4](../../../release/ornith-v4/settings.env)) on all nine cards, the reference for a later check
   with the offsets: loaded in 672 s, two rounds of the kit's greedy check (09:49:23 and 09:49:59), the same hash
   both times. The start check passed again after it.
2. **The watchdog, then `climb.py apply` with rung 1** (09:50:07 to 09:55:43): every card's stock check, the offsets
   set and read back, and one check round at rung 1: every card passed.
3. **The ladder** (`soak.py --minutes 30 --keep-going --ladder`, with R217's failures in `--failures`, 09:55:48 to
   11:29:00): one check server per card with the check model (Granite 3.1 3B-A800M Instruct, Q4_K_M), ready by
   09:58:34, and every round each card's greedy check (two prompts, 256 tokens each, temperature 0, every token's
   top-10 log-probabilities hashed) against its stock hash. After each 30-minute rung the cards that had not differed
   went up to the next rung's values, read back; a card that differed stayed where it was and went on being checked.
   A card that differed three times was set to 0 alone and left (cards 6 and 1, at 11:04:52 and 11:07:08).
4. **Ornith with the offsets** for 30 minutes, which runs only if no card differed: skipped. The chain set every
   offset of every card to 0 (11:29:02 to 11:29:11) and ended.

The rungs ([run/rung1.json](run/rung1.json) to [run/rung3.json](run/rung3.json), written by `climb.py rungs` from
R217's profile and failures alone), memory in MT/s on cards 0 to 8, the core 0:

| Rung | Memory |
|---|---|
| 1 | +1,000, +1,000, +900, +1,000, +1,100, +1,100, +1,000, +900, +700 |
| 2 | +1,100, +1,100, +1,000, +1,100, +1,200, +1,200, +1,100, +1,000, +800 |
| 3 | +1,200, +1,200, +1,100, +1,200, +1,300, +1,300, +1,200, +1,100, +900 |

Rung 3 is each card's top value: one step under R217's last clean step, capped two steps under its climb failure.

## Results

### Every card and rung ([tables/rungs.csv](tables/rungs.csv), [tables/cards.csv](tables/cards.csv))

Rounds a rung: 151, 151 and 170 (rung 3's rounds were quicker once two cards had left). Each cell is the card's
memory offset and its differing checks in that rung.

| Card | Rung 1 | Rung 2 | Rung 3 | Rounds checked |
|---|---|---|---|---|
| 0 | +1,000: 0 | +1,100: **2** (rounds 167, 242) | +1,100, held: 0 | 472 |
| 1 | +1,000: 0 | +1,100: 0 | +1,200: **3** (rounds 311, 330, 344), left | 344 |
| 2 | +900: 0 | +1,000: 0 | +1,100: 0 | 472 |
| 3 | +1,000: 0 | +1,100: 0 | +1,200: 0 | 472 |
| 4 | +1,100: 0 | +1,200: 0 | +1,300: **1** (round 464) | 472 |
| 5 | +1,100: 0 | +1,200: 0 | +1,300: 0 | 472 |
| 6 | +1,000: 0 | +1,100: 0 | +1,200: **3** (rounds 319, 322, 332), left | 332 |
| 7 | +900: 0 | +1,000: 0 | +1,100: 0 | 472 |
| 8 | +700: 0 | +800: 0 | +900: 0 | 472 |

Every differing check with its round in the rung and the rounds the card had run at that value before it is in
[tables/differs.csv](tables/differs.csv): card 0 in rounds 16 and 91 of rung 2; card 1 in rounds 9, 28 and 42 of
rung 3; card 6 in rounds 17, 20 and 30; card 4 in round 162. All 3,971 other checks gave the stock hash
`60ee0f178cfd67d5`, the same on every card and on both rigs.

### The nine failed checks ([tables/greedy_failures.csv](tables/greedy_failures.csv))

| Card | Round | Prompt 1 | Prompt 2 |
|---|---|---|---|
| 0 | 167 | text identical; probabilities differ from token 0 | bit-identical to stock |
| 0 | 242 | bit-identical to stock | text identical; probabilities differ from token 45 |
| 1 | 311 | bit-identical to stock | text identical; probabilities differ from token 233 |
| 1 | 330 | text identical; probabilities differ from token 192 | bit-identical to stock |
| 1 | 344 | bit-identical to stock | text identical; probabilities differ from token 49 |
| 4 | 464 | bit-identical to stock | text identical; probabilities differ from token 186 |
| 6 | 319 | text identical; probabilities differ from token 57 | bit-identical to stock |
| 6 | 322 | bit-identical to stock | one token differs (82 of 256); probabilities from token 46 |
| 6 | 332 | bit-identical to stock | text identical; probabilities differ from token 86 |

### Ornith at stock ([tables/ornith_stock.csv](tables/ornith_stock.csv))

Ornith filled 5,603, 5,635, 5,549, 5,635, 5,243, 5,351, 5,513, 5,601 and 5,009 MiB on cards 0 to 8, with 1,506 MiB of
host memory left. Both rounds gave the greedy hash `adf47784f9b02d4b`, the hash of the second rig in R200 and R201.
Generation in round 1: 14.22 and 14.76 tok/s (prompts 1 and 2); in round 2: 15.63 and 15.62 tok/s; prompt processing
in round 2: 26.79 and 29.56 tok/s. With a card differing, the run with the offsets did not follow, so this boot has no
offset speed for Ornith.

### Apply's check, streaming reads ([tables/apply.csv](tables/apply.csv))

Every card passed the greedy check and the 31 kernel checks at stock and at rung 1. Streaming reads in GB/s, cards 0
to 8, at stock: 179.32, 179.26, 179.37, 179.23, 179.32, 179.37, 179.08, 179.15, 179.29; at rung 1: 199.50, 199.41,
197.43, 199.60, 201.96, 201.90, 199.31, 197.43, 192.81.

### Heat, power and clocks ([tables/heat_power.csv](tables/heat_power.csv))

From the watchdog's half-second samples (09:50:07 to 11:38:05): the hottest card was 52 °C (card 1 in rung 3, before
it left); no card was in a thermal or hardware slowdown in any sample; the highest power reading was 84.86 W (card 6,
in `apply`). Readings above 80 W were single samples, never two in a row: 3 in `apply` (cards 0, 6 and 8) and 9 in
the rungs (cards 5 and 7), against the watchdog's limit of 80 W for 5 s. The memory clock the driver reported peaked
at 4,657 MHz (cards 4 and 5 at +1,300 MT/s; 4,006 MHz at stock under load). The watchdog stopped after 0 trips, and
the boot logged no Xid.

### The end

At 11:29:00 the soak ended (`exit 1 rounds 472 differs 9 left 1,6`) and wrote [soaked.json](soaked.json),
[use.json](use.json) and [use-choice.md](use-choice.md); the chain set every offset of every card to 0 (11:29:02 to
11:29:11). Then the watchdog was stopped and the state read back (11:38): every offset 0, P8, 0 MiB on every card, no
load, no Xid this boot, nine offsets lowered since the boot (so the next step needs a reboot), 3,182 MiB of host
memory available.

## Files

| File | What it holds |
|---|---|
| [use.json](use.json) | The setting for the next step, by the rule above, as `climb.py apply` and `climb.py --from-profile` read it; its `failures` are steps 1 and 2's |
| [use-choice.md](use-choice.md) | Each card's highest clean rung, its cap and where the cap comes from, and the value to use, as `soak.py` wrote it |
| [soaked.json](soaked.json) | `soak.py`'s profile: each card at the highest rung it held clean, with every rung's offsets and differing checks |
| [tables/cards.csv](tables/cards.csv) | One row per card: each rung's offsets and differing checks, the rounds checked, the rung it left in, the soaked value, the first difference, the use value |
| [tables/differs.csv](tables/differs.csv) | Every differing check: time, round, card, rung, offset, its round in the rung, the card's rounds at that value before it |
| [tables/rungs.csv](tables/rungs.csv) | `soak.py`'s rows per rung and card: offsets, rounds, passes, differing checks, the first differing round, whether it left |
| [tables/soak.csv](tables/soak.csv) | Every card's every round: hash, seconds, verdict, rung and offsets |
| [tables/greedy_failures.csv](tables/greedy_failures.csv) | Each failed check against the card's stock run, by prompt (made with [R217's run/greedy_diff.py](../2026-11-19_r217_climb_rig1/run/greedy_diff.py)) |
| [tables/ornith_stock.csv](tables/ornith_stock.csv) | Ornith's two stock rounds: hash, prompt and generation tok/s of both prompts |
| [tables/apply.csv](tables/apply.csv) | `apply`'s stock round and its round at rung 1: peaks, the greedy and kernel checks, streaming reads |
| [tables/heat_power.csv](tables/heat_power.csv) | Each phase (apply, the soak's server start, each rung, each raise) and card from the watchdog's samples |
| [run/env-chain.sh](run/env-chain.sh) | This run's settings for `soak_chain.sh`: work folder, the rungs, R217's failures, Ornith and its settings, the minutes |
| [run/rung1.json](run/rung1.json), [run/rung2.json](run/rung2.json), [run/rung3.json](run/rung3.json) | The three rungs, as `apply` and `soak.py --ladder` read them (R217's `rungs/`) |
| [run/failures.json](run/failures.json) | R217's seven climb failures, the soak's `--failures` |
| [run/MD5SUMS-r218](run/MD5SUMS-r218) | The md5 sums of the rungs, failures and settings staged on the rig (`ornith-v4.env` is [release/ornith-v4/settings.env](../../../release/ornith-v4/settings.env)), checked there before the start |
| [run/tables.py](run/tables.py) | Builds the tables from the work folder |

## Commands

On the rig, as root, in the kit's folder (R217's, with the kit files of this branch, `env.sh`, and this folder's
`run/` files beside them, `ornith-v4.env` a copy of Ornith's v4 settings), on a fresh boot:

```bash
md5sum -c MD5SUMS-r218 && . ./env-chain.sh && python3 climb.py state && python3 climb.py start-check
mkdir -p /var/tmp/r218 && setsid nohup bash soak_chain.sh > /var/tmp/r218/chain.console 2>&1 < /dev/null &
cat /var/tmp/r218/CHAIN_DONE      # a look, at most every 30 minutes
# when CHAIN_DONE exists: the watchdog stopped and the state read back (the chain already set every offset to 0
# because a card differed); copy /var/tmp/r218 and /var/tmp/r218-ornith off the rig
python3 climb.py watchdog stop; python3 climb.py state
```

The tables, on any machine, from the copied folders:

```bash
python3 -I run/tables.py WORK ORNITH_OUT tables
python3 -I ../2026-11-19_r217_climb_rig1/run/greedy_diff.py WORK/soak tables/greedy_failures.csv WORK/out r218-ladder
# the use profile again by the kit's rule, the same values on every card:
python3 -I ../../../bench/mem_offset/soak.py --use-from soaked.json --failures run/failures.json --use-out use-again.json
```

The next step on this rig, the core climb from this memory, on the next fresh boot (the kit README's
[five steps](../../../bench/mem_offset/README.md#a-rigs-profile-the-same-five-steps-on-every-rig), step 3):

```bash
. ./env.sh && export WORK=<a fresh folder> P=/var/tmp/r218/soak/use.json
python3 climb.py start-check && python3 climb.py --from-profile $P plan     # every card climbs the core
python3 climb.py trip-test && python3 climb.py watchdog start
python3 climb.py --from-profile $P climb       # detached; when $WORK/CLIMB_DONE says exit 0:
python3 climb.py state && python3 climb.py watchdog stop
```

## Provenance and licence

Measured on the rig on 2026-11-20, 09:37 to 11:38, with the engine build
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828` (Miner Llama v4's release candidate 2) and the kit as
published on this branch. The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M;
Ornith-1.5-122B Pascal N2P GGUF and its licence are described in
[release/ornith-v3](../../../release/ornith-v3/README.md). Neither model is published here, only the hashes of their
answers. The tables are this repository's licence.
