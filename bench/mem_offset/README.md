# mem_offset: testing a memory overclock on one P106-090, with proof that the output stays bit-identical

A P106-090's GDDR5 runs at 8,008 MT/s (`nvidia-smi` shows 4,004 MHz). Decode speed on this rig depends partly on
memory bandwidth, so a memory clock offset can buy a few percent: est. +3–5 % decode at +10 % memory clock, because only
about half of a token's time streams weights ([hardware tuning note](../../docs/research/notes/hardware_tuning.md),
sections 1–2). The cards have no ECC, though. A memory clock past a card's margin can corrupt weights silently instead
of crashing, so "it didn't crash" proves nothing.

This kit tests one card at a time and accepts an offset only if the model's output is unchanged bit for bit. It follows
the protocol of the note's section 5.2:
1. find the units of the driver's offset;
2. record stock references;
3. raise the offset in +200 MT/s steps, with a correctness and bandwidth check at each step;
4. at the best step, run a memory pattern test, an exact output check and a one-hour soak.

The offset goes back to 0 at the end, on any failure and on Ctrl-C. Nothing persists across a reboot or driver reload.

For a whole rig, use `climb.py` (next section): it finds every card's memory and core offsets in one sitting.

## Every card in one sitting: `climb.py`

**In plain words.** Every card has its own limit. On one nine-card rig, card 2 crashed at a 2,202 MHz core peak, card 4
gave wrong answers and then hung at memory +1,200 MT/s with a 2,138 MHz peak, and cards 0 to 3 passed both
([R193](../../docs/data/2026-11-17_r193_nine_cards_soak/README.md),
[R194](../../docs/data/2026-11-17_r194_nine_cards_soak/README.md)). A card whose offset goes down stays about 8 % slow
until the next boot, so testing one card after another would cost a reboot for every failure. `climb.py` climbs all
cards together: each step raises every card that is still climbing, checks every card, and a card that fails goes back
to 0 and drops out while the others go on. At the end it writes a profile with each card's settings one step under the
last step that card passed. After a reboot, `climb.py apply` sets the profile and checks every card once.

### The rules it enforces

- **Offsets only go up, and only after a fresh boot.** The climb and `apply` start only when every offset reads 0 and no
  offset went down since the last boot. Every step down (a failure, a stop, the end of the climb) is written to a small
  ledger, `/var/tmp/mem_offset_stepdowns.log`, tagged with the boot it happened in. `apply` also refuses to run on the
  boot the climb ran on.
- **A reboot after any step down.** A card whose memory offset went down ran about 8 % slower at stock until the next
  boot ([Q1563](../../quants/experiments.md)). The climb ends by setting every card to 0, so the rig is rebooted
  before `apply`.
- **The watchdog always on** ([scripts/nvml_watchdog.py](../../scripts/nvml_watchdog.py)). The climb and `apply`
  refuse to start without it and stop everything if it stops. Above 70 °C on any card, or above 80 W for 5 s, it sets
  every offset of every card to 0 and kills every load. With `--per-card-xid`, which the kit uses, a driver error (Xid)
  that names one card by its PCI address sets only that card to 0 and kills only that card's processes; an Xid that
  names no card still stops everything. It also holds every card open, which keeps the offsets in place.

### One step of the climb

1. Every card still climbing gets its next offset, read back from a new process.
2. Every card runs its checks at the same time:
   - **the greedy check** on that card's own `llama-server`: two prompts, 256 tokens each, temperature 0, hashed with
     every token's top-10 probabilities ([greedy_check.py](greedy_check.py)). It must equal the card's own stock hash.
     This is the check that caught a memory error that `cuda_memtest` and all 516 kernel checks missed
     ([Q1564](../../quants/experiments.md));
   - then **a short `mmvq_pascal` run**: 31 kernel checks against a CPU reference and the streaming read (GB/s), at most
     two cards at a time to spare the host's 4 GB.
3. **Every check has a time limit:** 3 × the time the same check took on that card at stock, and at least 30 s. A check
   past its limit fails its card. This catches a card that gives wrong answers and then hangs without any error, as
   card 4 did in R194 ([Q1572](../../quants/experiments.md)).
4. An Xid in the kernel log fails the card it names.
5. A card that fails is set to 0 and read back first, then its checks and its server are stopped. It leaves the climb;
   its last clean step is kept.

### The order

- **Stock round, twice:** every card's greedy hash (the second run must give the same hash, or the check is not
  deterministic and the climb refuses), every check's time (the limits), and every card's stock peak core clock, read
  from the watchdog's samples.
- **The stock peak is the highest reading of a longer window:** from the start of the first stock round to the end of
  the last, and the stock rounds go on (each must give the stock hash again) until they cover at least
  `--stock-peak-secs` (90 s). Nine cards' two rounds took 98, 98, 103, 101 and 101 s on the second rig's five boots,
  so only a smaller rig runs more. A card reads its top step mostly in the first seconds of the first round, while it
  is still cool; the second round alone read a driver step low. On the second rig's five boots it read card 8 at
  1,898 MHz every time, where both rounds read 1,911 MHz, the stock peak that card 8's soak peak implies (2,138 MHz
  at +231 MHz, 18 steps); both rounds read cards 0, 4 and 8 at those implied peaks on all five boots
  ([R200](../../docs/data/2026-11-17_r200_climb_rig2/README.md), the test below). The reading is better, not exact:
  cards 1, 3, 5 and 7 still differ by one step between boots, which is why `apply` keeps the offset (next section).
- **Memory, with the core at 0:** +200, +400, +600, +800 MT/s, then +100 at a time up to +1,400. It stops at +1,400 or
  when every card has failed.
- **Core, for the cards that never failed:** each card's memory stays at its last clean step (lowering it would be a
  step down). The core climbs by target peak clock: every card is aimed at the same peak, using its own stock peak
  ([Q1570](../../quants/experiments.md)), two of the driver's ~12.66 MHz steps (25 MHz) at a time, up to a 2,250 MHz
  peak. A card whose stock peak is above the target stays at 0 until the target passes it. `climb.py plan` prints
  every step's offsets.
- **End:** every card to 0, read back; the servers stopped; the profile.

A card that failed in the memory climb gets no core value in this sitting (it was stepped down). With card 0 of rig 1
failing at +1,400, many cards may fail there; `--mem-steps "200 400 600 800 900 1000 1100 1200"` stops the memory
climb at +1,200 and leaves more cards for the core climb. Such a card gets its core value in a second sitting,
`climb.py --from-profile` (below).

**What the climb's result is, and is not.** It is a quick map of where each card fails at once: 17 minutes for nine
cards on the second rig. It is not a setting: its one check a step passes offsets that fail a longer run, and in the
core phase each card's memory sits at its last clean step, where memory errors begin, so a memory error there is
charged to the core (the likely reading of two of R200's five core failures, at +16 and +104 MHz). The settings come from the soak ladder below, started two or three
steps under the climb's values, and a fair core value from a core climb with the memory at a soaked value.

**What stops everything** (every offset of every card to 0, read back; the servers stopped; the profile written as far
as the climb got): the watchdog's trip or its absence, an Xid that names no climbing card, a check or a server that
does not stop when killed, less than 300 MiB of host memory free, a reset that does not read back 0, a step down
(a bug), `climb.py stop`, and Ctrl-C or SIGTERM.

### The check model

Each card serves its own copy of one GGUF that fits a 6 GB card with room to spare: about 2 to 4.5 GB, for example a
7B model at Q4_0 (3.8 GB). Bigger covers more of the card's memory. The servers load one after another at stock
(each card's weights go in before any offset), with a 1,024-token context. Host memory decides how many `mmvq_pascal`
runs fit beside the nine servers: the climb reads it after the servers load, lowers `--mmvq-jobs` to fit, and refuses
to start (before any offset) if not even one fits.

### The profile

`out/profile.json`, public-ready (no host names; the boot is a hash). Per card:

| Field | What |
|---|---|
| `memory_mts`, `core_mhz` | the settings: one step under the card's last clean step (memory in MT/s, core in MHz as passed to NVML) |
| `core_steps`, `expected_peak_mhz` | the core offset in the driver's steps, and the stock peak plus those steps |
| `stock_peak_mhz` | the card's peak core clock at stock, measured |
| `memory`, `core` | the last clean step, the step that failed (if any), whether the card took part in the core climb |
| `failure` | the phase, offsets, reason and time of the card's failure |
| `needs_reboot_before_use` | true for a card that failed in this sitting (it was set back to 0 then) |

`stock_peak_window_s`, at the top, is how many seconds of stock rounds the stock peaks were read over.
`out/steps.csv` has one row per card and step (offsets, expected and measured peak clock, each check's result, time and
limit, Xid count, verdict and reason); `out/climb.log` is the plain log; `out/climb.json` the state, from which
`climb.py profile` rewrites the profile.

### `apply`: the profile's offsets, never more

`apply` sets each card's offsets exactly as the profile gives them. It does not work the core offset out again from
the stock peak it reads on its own boot: that reading moves a driver step between boots and even inside one, while a
given offset gives the same peak on every boot (on the second rig, card 0 at +256 MHz peaked at 2,126 MHz in all four
soaks, card 8 at +231 at 2,138 and card 4 at +206 at 2,113), so an offset worked out from one boot's low reading
would put the card a step too high. For each card with a core offset the log shows the offset it set, the stock peak
read on this boot and the one in the profile.

Before any offset, and before it detaches (the refusal shows in the terminal and `APPLY_DONE` says `exit 1`, so
`soak_chain.sh` ends instead of waiting), `apply` refuses a profile with a value that is not a whole number or is
below 0, a memory value above the top of `--mem-steps`, a core above `--core-cap`, a card without its
PCI address and name, or a `core_steps` that its `core_mhz` does not give (a hand edit of one and not the other: fix
it). A profile without `core_steps` gets the steps its `core_mhz` gives. Every profile published in this repository
passes.

### Commands

Run as root on the rig, with someone at the console (a hung card needs a reboot, and the reboot needs the disk
passphrase). The kit's tarball has `bin/mmvq_pascal`; copy `climb.py`, `greedy_check.py`, `memoc_units.json` and
`scripts/nvml_watchdog.py` next to it. `BIN` is the llama.cpp build's bin folder, `CHECK_MODEL` the check model. The
kit and the work folder go under `/var/tmp`, which survives the reboot between the climb and `apply` (`/tmp` is
usually emptied at boot).

```bash
cd /var/tmp/oc/mem_offset-kit-<date>
export BIN=/path/to/llama.cpp/bin CHECK_MODEL=/path/to/check-model.gguf WORK=/var/tmp/climb
python3 climb.py state                # read-only
python3 climb.py start-check          # read-only: every offset 0, no step down this boot, 0 MiB, default power limit
python3 climb.py plan                 # read-only: the steps (stock peaks of rig 1 until the climb measures its own)
python3 climb.py trip-test            # the watchdog's four trips with stand-in processes, before any offset
python3 climb.py watchdog start
python3 climb.py climb                # detached; about 50 min for nine cards (est.)
tail -f /var/tmp/climb/out/climb.log
python3 climb.py stop                 # only to stop early: every offset to 0, the profile as far as it got
cat /var/tmp/climb/CLIMB_DONE /var/tmp/climb/out/profile.json
python3 climb.py watchdog stop
python3 climb.py rungs /var/tmp/climb/out/profile.json   # the soak ladder's rungs, next to it (below; file work only)
# reboot at the console, then:
python3 climb.py start-check && python3 climb.py watchdog start
python3 climb.py apply /var/tmp/climb/out/profile.json   # detached; about 15 min (est.)
cat /var/tmp/climb/APPLY_DONE                             # exit 0: the offsets are set; keep the watchdog running
```

### Dry run and tests (any Linux machine, no GPU)

`--dry-run` prints every command a run would send and sets nothing: no root, no GPU, no NVML. It runs on a simulated
rig with a simulated clock (nine cards with rig 1's stock peaks; a full climb takes a second). `--scenario FILE` makes
simulated cards fail at chosen offsets.

```bash
python3 bench/mem_offset/climb.py --dry-run --check-model /models/check.gguf climb
python3 -I bench/mem_offset/test/test_climb.py
```

[test/test_climb.py](test/test_climb.py) checks:
- **R193 and R194 replayed** through the climb's own set, check and failure code, from their recorded benchmark output
  ([test/fixtures/](test/fixtures/), built by [test/make_replay_fixtures.py](test/make_replay_fixtures.py) from the
  runs' raw output). R193: cards 0 and 1 pass (516 checks, 204.70 and 204.80 GB/s); card 2 fails on its Xid 31, which
  names it by PCI address, and is set to 0 while cards 0 and 1 keep their offsets. R194: cards 0 to 3 pass; card 4
  fails on its two wrong kernel results and on the time limit, which ends its hang 210 s after it started (the recorded
  hang was stopped by hand after 383 s).
- **A full climb with the recorded failures** as simulated cards: card 0's memory at +1,400 (only the greedy check
  sees it), card 2's core at a 2,202 MHz peak (an Xid), card 4's memory at +1,200 (wrong results, then a hang). Each
  fails in the right phase for the right reason, the others reach the top of both climbs, no offset ever goes down
  before a card's reset, and the profile holds one step under each last clean step.
- the stops that end everything, `apply`'s refusals (the climb's boot, a step down this boot) and its reset on a
  failure, and that `--dry-run` starts no process.
- **`--from-profile`** on a simulated rig with R201's rung-4 values: each card's memory is set once, to the profile's
  value; the check round runs before any climbing core is set; a card whose memory fails the check round, and a card
  whose kept core fails it, go to 0 alone with nothing in the profile; a card with an Xid in the core climb keeps its
  memory and gets one step under its last clean core; the others reach the top; `apply` reads the result. Also its
  refusals before any offset (another rig's card, a card missing from the profile, a value above a cap, the climb's
  own boot, no watchdog, an offset left set), a watchdog trip in the core climb, and `--core-cards`.
- **The stock peak on R200's five boots** ([test/fixtures/r200_stock.csv.gz](test/fixtures/r200_stock.csv.gz): every
  watchdog sample of the two stock rounds, cut from the raw logs by `make_replay_fixtures.py --stock-windows`): the
  second round alone gives exactly what the kit read then, on every card of every boot; both rounds read card 8 at
  1,911 MHz on all five boots (round 2 alone: 1,898 on all five), card 0 at 1,873 (1,860 on three) and card 4 at
  1,911 (1,898 on two), the peaks their soak peaks imply. On the simulated rig the window starts with the first round,
  three cards get a third stock round, nine do not, and a stock round that differs is refused.
- **`apply`** sets the profile's core and memory on boots whose stock peaks read a step low, a step high and as the
  profile's; it refuses before any offset a `core_steps` its core does not give, a core above `--core-cap`, a memory
  above the cap, a value that is not a whole number and a card without its PCI address, and before it detaches with
  `APPLY_DONE` "exit 1" over an earlier "exit 0"; every published profile passes.
- **`rungs`** (below): R201's rungs from R200's last profile, a fresh climb's profile (its rungs at its own values),
  the caps, the floor at 0, no core rung when no card has a core, its refusals, and that `soak.py`'s ladder check,
  `soak.py --failures` and `apply` read the rung files.

## Holding a profile: `soak.py`, the ladder and the whole-memory check

**In plain words.** The climb's one check a step shows where a card fails at once. It does not show where a card
never fails: on the second rig, three cards failed a 30-minute soak one and two steps under the last step they had
passed in the climb ([R200](../../docs/data/2026-11-17_r200_climb_rig2/README.md)). A profile is settled by soaks,
and the soaks are built on these facts:

- **An error near the edge is rare and it passes.** In 6 of R200's 13 failed checks the first of the check's two
  requests went wrong and the second, run right after it on the same loaded server, was bit-identical to stock again.
  The model in the card's memory was intact; one answer was spoiled. So a soak can count a card's differing checks
  over the whole time instead of ending at the first, and one card's error need not end the other cards' soak.
- **Thirty clean minutes are a weak proof.** No error in 30 minutes shows only that errors come less often than about
  once in 10 minutes (95 % confidence). Daily use needs clean hours.
- **The first minutes after a change are not special.** On the second rig's soaks none of 8 wrong answers came in
  the first 5 rounds after its card's own offset changed, and counted from any card's change they fell as an even
  spread would put them ([FN118](../../docs/data/2026-11-19_fn118_errors_after_change/README.md)): a rung counts
  every round. What limits a soak is its length: a card that erred once in 438 rounds passes a 30-minute rung of 152
  rounds clean 70.7 % of the time, which is why the setting to use keeps two steps under each failure.
- **Offsets only go up in a boot, so the search goes up too.** A soak that starts at the edge and steps a failing
  card down costs a reboot for every step. `soak.py --ladder` starts low and raises the clean cards a rung at a time:
  one boot gives every card its highest rung that held.
- **One thing changes per rung.** Memory rungs first with the core where `apply` set it, then a rung that adds the
  core, so a card that starts to differ names the offset that did it.

**The rule for the setting to use:** each offset at the highest rung the card held clean, but at least two steps
under the lowest value at which that offset ever failed on that card, in a climb or a soak. `soak.py` applies it at
the end of a soak and writes `soak/use.json`, the profile for the next boot (below).

### The three modes of `soak.py`

| Mode | What happens when a card's check differs | Ends with |
|---|---|---|
| plain | every offset of every card to 0, the soak ends | `SOAK_DONE` "exit 1 card N round R: …", or "exit 0 rounds N" |
| `--keep-going` | it is counted and the soak goes on for the full time. A card that differs three times at an offset (`--max-differs`), or whose check fails or runs past its limit, is set to 0 alone (`climb.py reset-card`) and leaves; the others go on | the offsets as they are; "exit 0 rounds N differs 0" only if nothing differed and no card left |
| `--ladder P2.json P3.json …` | as `--keep-going`; after each rung the cards that have not differed go up to the next profile's values (read back), a card that differed stays where it is | the same, and `soak/soaked.json` and `soak/use.json` |

With every offset 0, `--keep-going` is the stock control: it shows that the check itself never differs on that rig.
`soak/rungs.csv` has one row per card and rung (offsets, rounds, passes, differing checks, the first differing
round), `soak/soak.csv` one row per card and round. A ladder is refused before any server starts if a rung would step
a card down or go above `--mem-cap` / `--core-cap`.

Two profiles come out of a soak with `--profile` or `--ladder`; `climb.py apply` reads both:

- **`soak/soaked.json`: what held.** Each card at the highest rung it held for the full minutes with no differing
  check, from the first rung on, and never at the offsets of a rung where it differed or left: a card the ladder did
  not raise can hold a value for a whole rung and differ at it in the next (cards 2 and 6 in
  [R201](../../docs/data/2026-11-18_r201_soak_ladder/README.md)), and that value is not offered. 0 for a card with no
  such rung.
- **`soak/use.json`: the setting to use, by the rule above.** Each card's memory and core start at its `soaked.json`
  values; each is lowered to two steps under the lowest value at which that offset failed on the card, if that is
  lower. Two memory steps are 200 MT/s (`--mem-step-mts`, 100); two core steps are four of the driver's core steps,
  about 51 MHz (`--core-step-bins` 2 and `--core-bin-mhz` 12.66, as the climb's). `soak/use-choice.md` gives each
  card's reason.

The failures it counts are the soak's own and those of earlier runs on the same rig, given with `--failures`
(`FAILURES` in the chain). A rung in which a card differed or left counts against the offset that went up into that
rung, since one thing changes per rung; with the core at 0 against the memory, with the memory at 0 against the core,
and against both when it cannot be told (the first rung, or a rung where both or neither changed). An earlier failure
counts against the offset its entry names, both if it names both:

```json
{"failures": [{"card": 1, "memory_mts": 1200, "where": "the climb, 2026-11-17"},
              {"card": 4, "memory_mts": 1300, "where": "a soak at +1,300 MT/s and +231 MHz, read as a memory error"},
              {"card": 0, "core_mhz": 307, "where": "the core climb, 2026-11-17"}]}
```

`use.json` lists every failure it used under `failures`, so it is the next soak's `--failures` as it stands. A
failures file that cannot be read is refused before any server starts. To make the use profile again from a finished
soak (other failures, or a `soaked.json` from before this rule), on any machine, with nothing set or started:

```bash
python3 -I soak.py --use-from WORK/soak/soaked.json --failures earlier.json --use-out use.json
```

On R201's and R202b's published `soaked.json` and the failures before them
([R201's](../../docs/data/2026-11-18_r201_soak_ladder/run/failures.json),
[R202b's](../../docs/data/2026-11-18_r202b_core_soak/run/failures.json)) this gives every card the values of the use
profiles made by hand for those runs.

### The ladder's rungs: `climb.py rungs`

A ladder needs one profile per rung. `climb.py rungs` writes them from a profile (a climb's `profile.json`, or any
profile `apply` reads) and each card's known failures. It sets and runs nothing and needs no rig.

- **Memory first, core 0.** Rung 3 is each card's top value: the profile's memory, lowered to two steps under the
  lowest memory value the card ever failed at, if that is lower. Rungs 1 and 2 are two steps and one step under it
  (100 MT/s a step, never below 0).
- **Then the core.** Rung 4 keeps rung 3's memory and adds the profile's core, lowered to two climb steps (four of the
  driver's steps, about 51 MHz) under the lowest core value the card ever failed at, if that is lower. When no card
  has a core value there is no rung 4.
- **Why two steps under a failure:** it is the margin of the setting to use (the rule above). A rung above it might
  hold, but `use.json` could never pick that value, so its 30 minutes would buy nothing.

The failures are the profile's own (a card's memory failure in the climb counts against its memory, a core failure
against its core) and those in `--failures` (`FAILURES`), the file `soak.py --failures` reads; a `use.json` works as
either. Each rung file lists the failures it used under `failures`, and `rungs-choice.md` gives each card's caps and
where they come from. `rungs` refuses a profile `apply` would refuse, a failures file that does not read, and rung files
that are already there (`--overwrite` replaces them).

```bash
# anywhere the files are, as any user; writes rung1.json ... rung4.json and rungs-choice.md next to the profile
python3 -I climb.py rungs /var/tmp/climb/out/profile.json --failures rig-failures.json
# the soak chain's settings for that ladder (next section):
#   PROFILE=rung1.json LADDER="rung2.json rung3.json rung4.json"
```

On R201's inputs, R200's last profile ([profile-try4.json](../../docs/data/2026-11-18_r201_soak_ladder/run/profile-try4.json))
and the failures before R201 ([failures.json](../../docs/data/2026-11-18_r201_soak_ladder/run/failures.json)), it
gives the rungs made by hand for cards 0, 1, 5, 6 and 8, and every card's core. Cards 2, 3, 4 and 7 come out one step
(100 MT/s) lower in every rung: R201's rungs were made one step under each card's lowest failure, and `--cap-steps 1`
gives R201's four files card for card. Two of the four values the kit leaves out differed in R201: card 3 at
+1,300 MT/s, twice in rung 3, and card 2 at +1,200 MT/s, once in rung 4. Cards 4 and 7 held +1,200 and +1,000 MT/s,
values the use profile then could not take: it gave them +1,100 and +900, two steps under their failures at +1,300
and +1,100 ([R201](../../docs/data/2026-11-18_r201_soak_ladder/README.md)).

### One boot without a person in the loop: `soak_chain.sh`

```bash
# in the kit folder on a freshly booted rig, as root; WORK is a fresh folder, BIN and CHECK_MODEL as for climb.py
export WORK=/var/tmp/soak1 MODEL=/path/to/served-model.gguf SETTINGS=/path/to/release/settings.env OUT=/var/tmp/soak1-model
PROFILE=rung1.json LADDER="rung2.json rung3.json rung4.json" \
  setsid nohup bash soak_chain.sh > /var/tmp/soak1.console 2>&1 < /dev/null &
cat /var/tmp/soak1/CHAIN_DONE          # when it exists: what happened, in one line
```

1. The start check, then [model_check.sh](model_check.sh) at stock: the model the rig serves, on all cards with its
   release settings, answers the greedy check twice (the reference, and each round's generation tok/s). Two rounds,
   a few minutes: `MINUTES` is the rungs' and does not reach this step. A longer stock control is its own setting,
   `STOCK_MINUTES` (0); a stock control that would end after `END_BY` does not run, and step 4 then has no reference.
2. The watchdog, then `climb.py apply` with the lowest rung.
3. `soak.py --keep-going` with the ladder, `MINUTES` (30) a rung, and the earlier failures in `FAILURES` if set.
4. If no card differed: the served model again with the offsets set, for `MODEL_MINUTES` (30), every round against
   the stock reference. The climb's check model covers about 2 GB of a 6 GB card; this covers what the rig really
   loads, and since the output is the same token for token, its tok/s against step 1's is what the offsets buy on
   that model.

A soak in which a card differed or left ends with every offset at 0 and `soak/use.json` as the next fresh boot's
profile. When everything passes, the offsets stay set while the watchdog runs. No step starts unless it ends by
`END_BY` (21:50).

### Tests (any Linux machine, no GPU)

```bash
python3 -I bench/mem_offset/test/test_soak.py     # ~2 min
bash bench/mem_offset/test/test_chain.sh
```

[test/test_soak.py](test/test_soak.py) runs the real `soak.py`, `greedy_check.py`, `model_check.sh` and
`climb.py reset-card` against a stand-in rig in a temporary folder (a `llama-server` that answers wrongly once, every
time or exits from a chosen memory offset on; an `nvml_memoc.py` that keeps the offsets in a file). It checks a
four-rung ladder (clean cards go up, memory then core; a card that differs once stays; a card that differs three
times and a card whose server dies are set to 0 alone; `soaked.json` holds each card's highest clean rung, and both
`apply` and `climb.py --from-profile` read it; `use.json` applies the rule with an earlier failure given, and `apply`
reads it), a card that differs at a value it held a rung before (not offered it), plain mode's reset of every card,
the stock control, the refusal of a ladder that steps down and of a failures file that cannot be read, and
`--use-from`. Its pure tests check which offset a failure counts against, the caps, and that the rule gives R201's and
R202b's hand-made use profiles from their published data. [test/test_chain.sh](test/test_chain.sh) runs
`soak_chain.sh` against stand-ins for its three programs: every ending, the `END_BY` rule, the stock reference's two
rounds when an env file exports the rungs' `MINUTES`, `STOCK_MINUTES`, and `FAILURES`.

## A fair core climb: `climb.py --from-profile`

**In plain words.** In the first sitting each card's core climbs with its memory at its last clean step, where memory
errors begin, so a memory error there counts against the core. On the second rig two of the five core failures came at
+16 and +104 MHz with the memory at +1,400 MT/s, the step two other cards had just failed
([R200](../../docs/data/2026-11-17_r200_climb_rig2/README.md)): most likely memory errors. `--from-profile` gives the
core its own sitting: the memory goes straight to a value that has held a soak, and only the core climbs.

In one fresh boot:
1. **The stock rounds** as in a climb: each card's reference hash, the time limits and its stock peak clock.
2. **No memory climb.** Every card's memory goes straight to the profile's value, read back. A card that keeps the
   profile's core value gets that core too.
3. **One check round** for every card at once at those values, with the climb's checks and limits. A card that fails
   is set to 0 alone and gets nothing: 0 and 0 in the result.
4. **The cards that keep their core are then done:** set to 0 (read back) and their servers stopped. Their values
   go into the result as checked.
5. **The core climb** for the other cards, as in a climb (by target peak clock, 25 MHz of peak a step, to 2,250 MHz),
   with each card's memory held at the profile's value. A card climbs from 0 even if the profile gives it a core
   value.
6. **The end** as in a climb: every card to 0, read back; the servers stopped; `out/profile.json`.

The input is a climb's `profile.json` or `soak.py`'s `soaked.json`; the cards are the profile's (or `--cards`). By
default the cards whose core value in the profile is 0 climb the core; `--core-cards "1 2 3"` names them instead. The
result is a profile `apply` and `soak.py --profile` read. Each card's `memory_mts` is the profile's value if the card
passed the check round, and 0 if it failed. Its `core_mhz` is one step under its last clean step if it climbed, and
the profile's value, checked in step 3, if it did not. `from_profile` names the input; `core_climb_cards` lists the
cards that climbed.

The climb's rules all hold: the start check (every offset 0, no step down this boot), the watchdog, a time limit on
every check, a failing card set to 0 alone, and the stops that end everything. A profile with a card missing, another
card at a card's PCI address, or a value above `--mem-steps`' top or `--core-cap` is refused before any offset, and
before the climb detaches, so the refusal shows in the terminal. Run `plan` first: read-only, it prints each card's
values, which cards climb, and the core steps.

R202, the second sitting on the second rig, after R201's ladder: the six cards without a fair core value (1, 5, 6
and 7 failed the memory climb; 2 and 3 failed the core climb with the memory at +1,400 MT/s) climb the core with the
memory at the values R201 held. Add to `--core-cards` any card R201's ladder left without a core value; `plan`
lists them. It ran on 2026-11-18 from R201's use profile in 11 min 55 s: all six cards climbed to +231 MHz or more, and
cards 2 and 3 went to +294 and +256 MHz ([R202](../../docs/data/2026-11-18_r202_core_climb_rig2/README.md)). On the
next boot a three-rung core ladder (`soak_chain.sh`, the memory fixed, the cores rising one climb step a rung to two
steps under each failure) ran 90 minutes with no card differing
([R202b](../../docs/data/2026-11-18_r202b_core_soak/README.md)).

```bash
# in the kit folder on a freshly booted rig, as root; BIN and CHECK_MODEL as for climb.py, a fresh WORK
export WORK=/var/tmp/r202
P=/var/tmp/r201/soak/soaked.json      # or the use profile, if it held its soak: its memory sits further from every failure
python3 climb.py start-check
python3 climb.py --from-profile $P --core-cards "1 2 3 5 6 7" plan     # read-only
python3 climb.py trip-test && python3 climb.py watchdog start
python3 climb.py --from-profile $P --core-cards "1 2 3 5 6 7" climb    # detached; about 15 min for nine cards (est.)
tail -f /var/tmp/r202/out/climb.log
cat /var/tmp/r202/CLIMB_DONE /var/tmp/r202/out/profile.json
python3 climb.py watchdog stop
```

**What it is not.** Like the climb, it is a quick map: one check a step shows where each card's core fails at once,
now with its memory at a value that holds. The core values it gives are the base of the next soak ladder (after a
reboot, `apply` sets the memory with the cores a few steps lower, and the ladder raises the cores), not settings.

## Overclocked at every boot: `boot/`

**In plain words.** An offset lasts only until the next reboot, so a rig with a soaked profile runs overclocked only
after someone runs `apply` on a fresh boot. The boot service does that at every boot. It checks that the cards are the
ones the profile was made for, sets the profile with `apply`'s own per-card check, and keeps the watchdog running for
as long as the offsets are set. If anything is off, every card stays at stock, or goes back to it. It is installed off
and stays off until the owner turns it on for that rig, and a rig without its own soaked profile never gets it. On the
second rig, the profile it sets made the coding agent take 1.90 minutes a task instead of 2.00 with Ornith (19.26
against 17.87 tok/s) and 2.26 instead of 2.36 with Flash-Next (30.69 against 27.82 tok/s), with the same answers
([R209](../../docs/data/2026-11-19_r209_oc_v5/README.md)).

### First, a soaked profile for this rig

Every card has its own limits, and so every rig needs its own profile; one rig's profile is never set on another.
The profile comes from the steps above, on that rig: the climb (`climb.py climb`), the ladder (`climb.py rungs`, then
`soak_chain.sh`), and for the core the climb from a profile and a core ladder. What the service sets is the ladder's
`soak/use.json`: each card at the highest rung it held, and at least two steps under any value it ever failed at.
The second rig's came from four runs: [R200](../../docs/data/2026-11-17_r200_climb_rig2/README.md),
[R201](../../docs/data/2026-11-18_r201_soak_ladder/README.md),
[R202](../../docs/data/2026-11-18_r202_core_climb_rig2/README.md) and
[R202b](../../docs/data/2026-11-18_r202b_core_soak/README.md), whose
[use-profile.json](../../docs/data/2026-11-18_r202b_core_soak/use-profile.json) R209 set.

### What it does at each boot

1. **Waits** for the NVIDIA driver and the files the config names, at most `DRIVER_WAIT` (180 s).
2. **Refuses, with no offset touched**, when: the config, the kit or the profile is missing or does not read; the
   cards present are not the profile's (the same card numbers, and each card's PCI address, name and VBIOS as the
   profile has them); with `GPU_SET_ID` set, the cards are not the set the config was made on (identical rigs share
   PCI addresses, names and VBIOS: this catches a profile and config copied from another rig, and a changed card);
   the driver is not `DRIVER` or the profile's; the model check is on and its files are missing; or the kit's
   `start-check` fails (a load runs, a card's memory is in use, a power limit was changed, an offset is set, or an
   offset went down since the boot).
3. **Starts the watchdog**, then runs the kit's `apply` of the profile in the foreground (`climb.py run-apply`, which
   is `apply` without detaching): each card set and read back, then each card's greedy check against its own stock
   result. Nine cards took 5 min 18 s on the second rig (R209's run 3). `APPLY_LIMIT` (2,400 s) ends a hung apply.
4. **Optional, off by default (`MODEL_CHECK=1`):** the served model's whole-memory check (`model_check.sh`): two
   rounds on all cards with its release settings, each equal to the stock hash stored in `MODEL_REF`. R209's rig
   runs used it as the gate. Ornith's load alone took 658 s there, so this adds about 12 minutes (est.) to the
   boot's check.
5. **Reads every offset back** against the profile.
6. **Ready**: `overclocked: yes` in the state file, and the unit active. From then on the service is the unit's main
   process: a watchdog trip, or the end of the watchdog, sets every card to 0 and ends the unit; a trip of one card
   (an Xid that names it) leaves that card at 0, the others set, and says `overclocked: partly`. Stopping the unit
   sets every card to 0 while the watchdog still holds them, then stops the watchdog.

Any failure from step 3 on sets every offset of every card to 0, reads it back and stops the watchdog. Every command
runs under a time limit, so a hung card cannot hang the service or the boot, and the boot does not wait for the
service at all: the unit has no default dependencies, so the rest of the boot goes on while it checks. **Start
nothing on the cards until the state file says `yes` or `no`:** `apply`'s check servers use every card.

Each step goes to the journal and to `WORK/boot.log`; `WORK/out/climb.log` is `apply`'s own log. The state file,
`/run/miner-llama/clocks.state`, is gone at every reboot, so it always tells about this boot:

```
overclocked: yes
phase: overclocked
reason: apply passed; the model check is off; the watchdog holds the cards
time: 2026-11-19 09:31:02
since: 2026-11-19 09:31:02
boot: 3f2a9c41d0b7
profile: /etc/miner-llama/clocks-profile.json
profile_md5: …
cards: 0:+1300/+256 1:+1000/+256 2:+1000/+269 …
card_trips: 0
model_check: off
watchdog: running (pid 1234)
work: /var/lib/miner-llama/clocks/work
```

`overclocked` is `yes`, `no`, `partly` (a card tripped) or `unknown` (a reset that did not read back 0: power the rig
off at the console). Each start gets a fresh work folder; the previous ones are kept beside it, the last
`KEEP_WORKS` (5). The watchdog's sample log is kept under `WD_LOG_MAX_MB` (64).

### Install, turn on, check, turn off

On the rig, as root, from a checkout of this repository or an unpacked kit tarball (which has `bin/mmvq_pascal`):

```bash
bash bench/mem_offset/boot/install.sh      # MMVQ_BIN=<kit>/bin/mmvq_pascal from a checkout; never enables anything
python3 /opt/miner-llama/mem_offset/boot/clocks_at_boot.py gpu-set-id     # read-only: this rig's card-set id
install -m 0644 <this rig's use.json> /etc/miner-llama/clocks-profile.json
${EDITOR:-nano} /etc/miner-llama/clocks.env    # BIN, CHECK_MODEL, DRIVER and GPU_SET_ID at least
```

`install.sh` copies the kit's scripts to `/opt/miner-llama/mem_offset`, the unit to
`/etc/systemd/system/miner-llama-clocks.service` and the example config to `/etc/miner-llama/clocks.env` (only if
there is none), then runs `systemctl daemon-reload`. Machine paths live only in that config: `BIN` and `CHECK_MODEL`
are the engine and check model the climb and soak used. Then, on a fresh boot:

```bash
systemctl start miner-llama-clocks         # once, now: returns when it is set (or has failed)
cat /run/miner-llama/clocks.state          # or: python3 /opt/miner-llama/mem_offset/boot/clocks_at_boot.py status
journalctl -u miner-llama-clocks -b        # every step of this boot
systemctl enable miner-llama-clocks        # at every boot from now on
```

To turn it off: `systemctl disable miner-llama-clocks` (the next boots stay at stock); `systemctl stop
miner-llama-clocks` sets every card to 0 now, and since an offset that went down leaves its card slower until the next
boot, reboot before a timed run. In `/opt/miner-llama/mem_offset`,
`WORK=/var/lib/miner-llama/clocks/work python3 climb.py state` shows every offset and sees the service's watchdog.
After a new driver, the units need measuring again and the rig a new climb and soak (until then the service refuses
the driver, since it is not `DRIVER`); after a new engine, model or settings, the model check needs a new stored hash.

**The model check's stored hash**, once, at stock (the service not started in this boot), with the engine, model and
settings the rig serves:

```bash
cd /opt/miner-llama/mem_offset && BIN=<engine>/bin MODEL=<served>.gguf SETTINGS=<release settings.env> \
  OUT=/var/lib/miner-llama/model-ref TAG=stock ROUNDS=2 bash model_check.sh
```

then `MODEL_CHECK=1`, `MODEL`, `MODEL_BIN` and `SETTINGS` in the config. A stale hash fails safe: the check differs
and every card goes back to stock.

### Tests (any Linux machine, no GPU)

```bash
python3 -I bench/mem_offset/test/test_boot.py     # ~3 min
```

[test/test_boot.py](test/test_boot.py) runs the real `boot/clocks_at_boot.py` as the unit runs it (its own process, a
stop signal, the notify socket), driving the real `climb.py` (`start-check`, `watchdog start` and `stop`,
`run-apply`, `state`, `reset-all`), `greedy_check.py` and `model_check.sh` against a stand-in rig like
`test_soak.py`'s; the only stand-in in `climb.py` itself is its root check. It checks the profile applied and held
(READY, `yes` with the profile's md5, every card at its value, the watchdog's log kept small), the stop (every card
0, the watchdog stopped), a second start in the same boot refused; a card whose check fails, a hung check past
`APPLY_LIMIT` and a stop during `apply` (every card 0); thirteen refusals with no offset touched (among them another
PCI address, VBIOS, driver or card set); a watchdog trip, the watchdog's end and one card's trip while held; the main
process killed (`stop-post` resets); the model check passing and differing; `systemd-analyze verify` on the unit;
`install.sh` into a test root from a checkout and from the tarball's layout, never enabling anything.

## What is in it

| File | What it does |
|---|---|
| `nvml_memoc.py` | Reads and sets one card's memory (or, with `--core`, graphics) clock offset through NVML, from Python's `ctypes` on the driver's `libnvidia-ml.so.1`: no install, no Xorg or Coolbits change. `probe` is read-only. |
| `memoc_steps.sh` | The test, stage by stage or all at once (below). |
| `exact_check.sh` | The exact output check on Flint: one-token llama-perplexity statistics and a greedy-decoding hash against stock. |
| `build.sh` | Builds `mmvq_pascal` and `cuda_memtest` for compute capability 6.1 in Docker and packs the kit into a tarball with MD5 sums. |
| `test/dryrun.sh` | Runs `memoc_steps.sh` against stub programs on any Linux machine (no GPU, no root) to check its control flow: a clean run, a failed exact check (step back), an Xid, an interrupt, and a memory test that cannot run. |
| `climb.py` | Every card's memory and core offsets in one sitting, the profile, and `apply`; with `--from-profile`, the core climb alone with each card's memory at a profile's value; `rungs`, the soak ladder's rung profiles from a profile and the known failures (sections above). |
| `soak.py` | Holds the offsets `apply` set with every card loaded: plain, counting (`--keep-going`) or a ladder of profiles, then writes what held (`soaked.json`) and the setting to use by the rule (`use.json`); `--use-from` makes the use profile again from a finished soak (section above). |
| `soak_chain.sh` | One fresh boot without a person in the loop: the served model's stock reference, `apply`, the soak or ladder, the served model with the offsets. |
| `model_check.sh` | The whole-memory check: the served model's greedy output on all cards, at stock and with the offsets, with each round's tok/s. |
| `greedy_check.py` | The greedy check: a running llama-server's output at temperature 0, hashed with every token's top-10 probabilities, against a reference. |
| `memoc_units.json` | NVML's memory offset unit, measured on a P106-090 with driver 580.178.04 (MT/s); measure it again with `nvml_memoc.py units` on another driver. |
| `test/test_climb.py`, `test/make_replay_fixtures.py`, `test/fixtures/` | The climb's tests, and the replays of R193 and R194 from their recorded output. |
| `test/test_soak.py`, `test/test_chain.sh` | The soak's, the ladder's and the chain's tests against a stand-in rig. |
| `boot/clocks_at_boot.py`, `boot/miner-llama-clocks.service` | The boot service: the checks, the watchdog, `apply`, the optional model check and the reset, as the main process of a systemd unit (section above). |
| `boot/install.sh`, `boot/clocks.env.example` | Installs the service on a rig, never enabled; the per-rig config, with every machine path. |
| `test/test_boot.py` | The boot service's tests against a stand-in rig, the unit's `systemd-analyze verify` and the installer. |

The tarball adds `bin/mmvq_pascal` ([bench/mmvq_pascal](../mmvq_pascal/README.md): streaming-read bandwidth, plus
correctness checks and timing of the decode kernels) and `bin/cuda_memtest`
([cuda_memtest](https://github.com/ComputationalRadiationPhysics/cuda_memtest): walking ones on the address lines,
own-address, moving inversions, random patterns and a stress test). Both are built with sm_61 code, compute_61 PTX,
host code without AVX and the CUDA runtime linked statically, so they need only the NVIDIA driver.

## Building

On any x86-64 Linux machine with Docker and the `nvidia/cuda:12.9.2-devel-ubuntu24.04` image (CUDA 13 can no longer
target Pascal):

```bash
bench/mem_offset/build.sh /path/to/workdir      # JOBS=8, DOCKER_ARGS="--label …" if the build machine wants them
```

It clones cuda_memtest at a pinned commit, builds both programs (a few minutes) and writes
`workdir/mem_offset-kit-<date>.tar.gz` with its `.md5`. `BUILDINFO.txt` in the tarball shows that neither binary
contains AVX instructions, the highest glibc symbol they need (2.38; Ubuntu 24.04 has 2.39) and their CUDA code.

## Running on the rig

Run as root, with nothing else on the card under test, and with someone near the rig's keyboard: a card that hangs
needs a reboot, and on this rig a reboot needs the disk passphrase typed at the console.

```bash
K=mem_offset-kit-20261024
scp $K.tar.gz rig:/tmp/ && ssh rig "mkdir -p /tmp/oc && tar xzf /tmp/$K.tar.gz -C /tmp/oc && cd /tmp/oc/$K && md5sum -c MD5SUMS"

# 1. read-only, ~1 min: do the offset calls exist on this card and driver, and in what range? Also the PCIe link and
#    the CPU frequency settings
ssh rig "python3 /tmp/oc/$K/nvml_memoc.py probe"

# 2. the test on GPU0 (~1 h 50 min), detached so that a dropped ssh connection does not stop it
ssh rig "cd /tmp/oc/$K && setsid nohup ./memoc_steps.sh all > console.log 2>&1 < /dev/null &"
ssh rig "tail -f /tmp/oc/$K/out/steps.log"
# stop early and cleanly (at the next check):  ssh rig "touch /tmp/oc/$K/out/STOP"
# stop at once (the trap still resets):       ssh rig "pkill -TERM -f memoc_steps.sh"

# 3. results, then clean up (the KLD base file is ~0.6 GB)
ssh rig "cat /tmp/oc/$K/out/summary.txt"
scp -r rig:/tmp/oc/$K/out ./mem_offset-gpu0 && ssh rig "rm -rf /tmp/oc /tmp/$K.tar.gz"
```

If `probe` reports `Not Supported` for both offset APIs on the card, route A is closed: stop there. The Coolbits
route (section 1.3 of the note) needs an Xorg reconfiguration.

### Stages

`memoc_steps.sh STAGE` runs one stage. The stages share their state in `out/`. `memoc_steps.sh all` runs them in
order:

| Stage | What | Pass | Time (est.) |
|---|---|---|---|
| `units` | NVML does not document the offset's unit. Sets a raw +100, samples the memory clock under a short mmvq load against offset 0, then sets 0 again. A clock 50 MHz higher means the raw unit is MT/s; 100 MHz higher means the reported MHz (2 MT/s each). | the clock moves by one of the two | 1 min |
| `base` | At offset 0: a `cuda_memtest` smoke test on 64 MB; writes the KLD base file (Flint UD-Q4_K_XL, 4,096 one-token steps on cards 0–5), reruns against it for the reference statistics, and hashes a 128-token greedy completion | the smoke test is clean; the stock rerun shows the same top token everywhere (the run is deterministic) | 15 min (3 model loads) |
| `sweep` | +200, +400, … up to `MAX` (1,200) MT/s; at each step the mmvq sweep (Q8_0, Q5_0, Q4_1, Q5_K and IQ4_XS kernels, each checked against a double-precision CPU reference, then timed; streaming-read GB/s) and an Xid scan of the kernel log | the mmvq checks all pass, no Xid, and the streaming read gains ≥ `MIN_RISE` (0.5 %) over the step before; the first step that fails ends the sweep, and the step before it is BEST | 1–1.5 min per step |
| `memtest` | `cuda_memtest` at BEST, one pass of tests 0–5, 7, 8 and 10, or `MEMTEST_MIN` (10) minutes, whichever ends first. Test 6 (moving inversions with 32-bit patterns) is left out: it takes ~70 % of a pass, est. ~18 min on 6 GB of a P106 (measured on 64 MB of an RTX 3090 and scaled by size and bandwidth); `MEMTEST_ARGS="--disable_all --enable_test 6"` runs it alone | no block reports errors, no Xid; if the program fails without reporting a memory error, the run stops rather than stepping back | 8–10 min |
| `exact` | At BEST: the KLD run against the stock base file, and the greedy hash | statistics identical to the stock rerun, hash identical | 10 min |
| `soak` | At BEST: llama-bench repeating a 512-token prompt plus 128 generated tokens on cards 0–5 for `SOAK_MIN` (60) minutes, checking every minute for an Xid, the process and the card's temperature (`TEMP_MAX`, 70 °C); then the KLD run again | the hour completes, no Xid, temperature under the limit, statistics identical afterwards | 65 min |

In `all`, a failed `memtest` or `exact` at BEST steps back 200 MT/s and tests again (+20 min per step back).
`out/summary.txt` collects the results:
- `sweep.csv` holds GB/s and the memory clock under load at each step;
- `smi.csv` holds the card's clocks, power, temperature and clock event reasons every 500 ms;
- `smi-cards.csv` holds the same for all six cards every 2 s;
- the logs of every run sit next to them.

Every value can be changed through the environment: `GPU`, `STEP`, `MAX`, `MIN_RISE`, `MMVQ_ARGS`, `MEMTEST_MIN`,
`MEMTEST_ARGS`, `SOAK_MIN`, `TEMP_MAX`, `OUT`, and for the llama.cpp runs `B` (bin directory of the build), `M` (model), `WIKI`,
`DEVS` (the cards; must include the card under test and hold the model) and `E` (extra environment,
`LLAMA_NO_MOE_DOWN_RESCALE=tg` for UD-Q4_K_XL). The defaults are the v2 base build
`/opt/llama-builds/llama.cpp-20261023-032b130f4-pascal-p106-v2/bin`, `/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf`
and wikitext-2 under `/opt/llama-builds/data/`.

### Reading the result

- **The units line** fixes how NVML's raw value maps to the clock on this card and driver. Every offset in the logs
  is in MT/s, the units of `nvidia-settings` and of the note: +800 is +10 %.
- **Streaming GB/s against the memory clock** gives the fraction of the clock gain the card turns into bandwidth.
  The kernels' GB/s at each step, divided by the clock ratio, gives the fraction f of section 2.1.
- **BEST is the last clean step, not the setting to keep.** Rare errors near the edge can take hours to show, so the
  setting for service is one step below BEST. Once every card is done, check it with the nine-card exact check and
  llama-bench against stock (stages 5 and 6 of the note).
- **The offset is gone after every stage and at the end.** Read it back with
  `python3 nvml_memoc.py get 0`; `python3 nvml_memoc.py reset 0` forces 0.

## The exact check

The fork's decode is deterministic: the same build, model, card set and input give byte-identical logits. With stable
memory, an overclocked run therefore gives exactly the output of a stock run.
- **Why not "KLD = 0".** llama-perplexity stores the base file's log-probabilities in 16 bits, so even identical
  logits show a small KLD: `Mean KLD` prints 0.000000, while the maximum is ~1e-5 (0.000043–0.000062 on two RTX 3090s
  with Flint G3 and bit-identical logits).
- **What the test is.** `exact_check.sh kld` requires the run's whole statistics block to equal a stock rerun's,
  digit for digit: perplexity, KLD mean, maximum and percentiles, the Δp statistics and "same top p" at 100 %.
  A changed logit anywhere in the 4,096 steps changes that block.
- **The greedy hash** is a second, coarser check through a different binary (`llama-completion`).

Each llama.cpp run loads the 25.1 GB model again (~2.5 min) and then takes ~2.5 min for its 4,096 one-token steps on
six cards. GPU0 holds the first layers of the six-card split, so its weights take part in every token.

## Safety notes

- **What a reset clears.** Offset 0 restores stock at once, and any reboot or driver reload clears an offset. The
  kit changes nothing persistent: no Xorg configuration, no files outside its work folders. The one exception is the
  boot service, which `boot/install.sh` installs only when the owner runs it, and which stays off until the owner
  enables it ([above](#overclocked-at-every-boot-boot)).
- **A hung card** (Xid 79, or a CUDA process that never returns) needs a reboot, after which every card is at stock.
- **Run on a quiet rig.** Nothing else should use the card under test. `memoc_steps.sh` refuses to start if a compute
  process is on it (`FORCE=1` overrides), and the soak uses cards 0–5.
- **Measured on a P106-090 on 2026-11-17** (Q133/Q135, [data](../../docs/data/2026-11-17_q133_q135_clock_offsets/README.md)):
  - `set` and `units` work with no X server on driver 580.178.04; the raw unit is MT/s (raw +100 moved the memory clock
    under load from 4,006 to 4,050 MHz); the ranges are −2,000 to +2,000 (memory) and −200 to +1,200 (core);
  - an offset set by one process stays after it exits while another process holds the card (with persistence mode off
    and no X server, that sitting kept an NVML watchdog running for the purpose);
  - `cuda_memtest` on the card's 6 GB: tests 0–5, 7, 8 and 10 in 79 s, test 6 alone in about 5 min.
- **The kernel checks and `cuda_memtest` are not enough to pass a step.** On the card tested, +1,400 MT/s passed all of
  `mmvq_pascal`'s checks and a `cuda_memtest` pass, yet changed Flint's token probabilities (not its text). The
  sitting judged each step with a greedy run hashed with every token's top-10 probabilities
  ([greedy_check.py](greedy_check.py)); `climb.py` runs it on every card at every step, `memoc_steps.sh` does not
  (Q1564).
- **Step only upward.** After the offset went down from +800 to 0, that card ran slower at the lower memory clocks with
  every offset at 0 (Q1563).
