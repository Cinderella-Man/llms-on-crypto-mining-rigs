# R226: which card of the first rig spoils the served model, found by halves (2026-11-20)

**The answer in plain words.** On a fresh boot of the first nine-card P106-090 rig, the kit
set the rig's own clock profile on cards 0 to 4 only and left cards 5 to 8 at stock. Every card passed its own quick
check. Then Ornith-1.5-122B Pascal N2P GGUF, which fills all nine cards, answered the kit's two fixed prompts for 30
minutes: **56 rounds out of 56 gave the stock answer token for token**, where the full profile had differed in round
22 ([R224](../2026-11-20_r224_apply_rig1/README.md)). Generation ran 4.21 % faster than stock (17.34 against 16.64
tok/s on both prompts). The watchdog never tripped, no card went above 80 W for more than one sample, and the boot
logged no GPU error. At the end every offset went back to 0.

**What it means.**

- **Cards 0 to 4 are the less likely home of the fault, so the next half sets cards 5, 7 and 8** (card 6 stays at
  stock, as in the profile). If cards 0 to 4 erred as often as the whole profile did in R224, once in 22 rounds, 56
  clean rounds in a row would come about 7 % of the time (est., from that one difference).
- **A clean half is not proof that its cards are sound:** 30 clean minutes show only that the half errs less often
  than about once in half an hour. If cards 5, 7 and 8 also hold 30 minutes, the fault needs both halves at once or is
  rarer than either check can see; the search then names no card, and what follows is the owner's choice among
  Q1640's other options.
- **The speed is a single-request diagnostic.** Five cards with their offsets gave +4.21 %; all eight cards with
  offsets gave +6.85 % in R224.

## Why by halves

In step 5 of the first rig's own profile ([R224](../2026-11-20_r224_apply_rig1/README.md)) every card took its
offsets and passed its own check, then Ornith-1.5-122B Pascal N2P GGUF, which spreads its layers over all nine cards,
gave its stock answer 21 rounds in a row and a different answer in round 22, 11 minutes in. One round cannot say which
card caused it. The search sets the profile on some cards and leaves the others at stock, one fresh boot each, and runs
the same 30-minute check ([Q1640](../../../quants/experiments.md), option (b)):

| Half | Cards at the profile's values | Cards at stock | Boot |
|---|---|---|---|
| a | 0, 1, 2, 3, 4 | 5, 6, 7, 8 | 2026-11-20, ~16:35 |
| b | 5, 7, 8 (half a held) | 0, 1, 2, 3, 4, 6 | the next |

Card 6 runs at stock in every half: its profile values are 0 and 0 (step 4,
[R223](../2026-11-20_r223_core_soak_rig1/README.md)). A clean 30 minutes is about 56 rounds at R224's pace (22 rounds
in 709 s), against R224's one difference in 22 rounds. A clean half shows only that its cards err less often than
about once in half an hour; it points at the other half, it does not clear its own cards.

## The profile, card by card

R223's [use.json](../2026-11-20_r223_core_soak_rig1/use.json) (md5 `ae91f969c8ddc782698c8fd9d570734d`), memory MT/s
and core MHz:

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memory | +900 | +1,000 | +1,100 | +1,200 | +1,100 | +1,300 | 0 | +1,100 | +900 |
| core | +218 | +256 | +256 | +206 | +206 | +231 | 0 | +282 | +231 |

## Half a: cards 0 to 4 (2026-11-20, 16:40 to 17:29)

The first nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), freshly booted at
about 16:35 with nothing run on it before. At 16:40 the staged files' md5 sums were checked on the rig
([run/MD5SUMS-a](run/MD5SUMS-a)), the half profile passed the kit's own profile check on the rig
(`half_profile.py --check`), and the kit's start check passed: every offset 0, no load process, no GPU Xid this boot,
no offset lowered since the boot, 3,169 MiB of host memory available.

[run/half_check.sh](run/half_check.sh) ran the boot detached in the kit's folder (16:40:19 to 17:28:50), with the v5
engine (`llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483`) and Ornith's v5 settings
([release/ornith-v5](../../../release/ornith-v5/settings.env)), as R224:

1. **The stock reference:** R224's stock round 1 (`adf47784f9b02d4b`) copied into a fresh output folder and its hash
   checked. No stock rounds on this boot.
2. **The start check, the trip test** (16:40:20 to 16:40:55: the watchdog's four trips with stand-in processes, each
   PASS) **and the watchdog** (16:40:58: a sample every 0.5 s; a card trips above 70 °C, above 80 W held for 5 s, or
   on its own Xid).
3. **`apply` of the half profile** ([half-a.json](half-a.json), 16:41:02 to 16:46:49): a check server on every card
   with the kit's check model (Granite 3.1 3B-A800M Instruct, Q4_K_M, about 2 GB of each card) on the kit's engine
   (`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`), the stock rounds, every card's offsets set 16:45:49 to
   16:45:54 and read back as set, then one round with the offsets: every card PASS at 16:46:40.
4. **Ornith with the offsets**, rounds for 30 minutes after the load, each against the stored stock answer: loaded
   in 673 s (ready 16:58:07), 56 rounds from 16:58:44 to 17:28:33, every one the stock hash.
5. **The reset:** every card's offsets to 0 (17:28:40 to 17:28:50), then the watchdog stopped.

### Ornith's whole-memory check ([halves.md](halves.md); every round in [a/model_rounds.tsv](a/model_rounds.tsv))

| | R224 stock | R224, the whole profile | **half a: cards 0 to 4** | half a against stock |
|---|---|---|---|---|
| load, s | 673 | 675 | **673** | |
| host memory available when loaded, MiB | 1,496 | 1,502 | **1,489** | |
| rounds; the stock answer | 2 (round 1 is the reference) | 22; 21 of 22, round 22 differed (11.7 min after the load) | **56; 56 of 56** in 1,831 s | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 16.04, 16.72 | **15.71, 16.46** | +4.59 %, +4.44 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.79, 17.78 | **17.35, 17.34** | +4.27 %, +4.21 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.78, 17.78 (21 rounds) | **17.34, 17.34** (55 rounds) | **+4.21 %, +4.21 %** |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.86, 29.67 (round 2) | 28.66, 31.61 | **28.13, 31.02** | +4.73 %, +4.55 % |

- From round 2 on, prompt 1 generated 17.34 tok/s in 36 rounds and 17.35 in 19; prompt 2 17.34 in 47, 17.33 in 4 and
  17.35 in 4.
- Both loads used the same memory as R224's: 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on cards 0 to 8.

### `apply`, card by card ([a/apply.md](a/apply.md); the kit's own table [a/apply_steps.csv](a/apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +900 | +218 | 179.34 | 197.42 | +10.08 % | 1,873 | 2,075 | same as stock | PASS |
| 1 | +1,000 | +256 | 179.20 | 199.69 | +11.43 % | 1,860 | 2,113 | same as stock | PASS |
| 2 | +1,100 | +256 | 179.23 | 202.10 | +12.76 % | 1,911 | 2,151 | same as stock | PASS |
| 3 | +1,200 | +206 | 179.27 | 204.90 | +14.30 % | 1,911 | 2,113 | same as stock | PASS |
| 4 | +1,100 | +206 | 179.21 | 201.90 | +12.66 % | 1,885 | 2,075 | same as stock | PASS |
| 5 | 0 | 0 | 179.17 | 179.37 | +0.11 % | 1,911 | 1,911 | same as stock | PASS |
| 6 | 0 | 0 | 179.21 | 179.20 | −0.01 % | 1,898 | 1,885 | same as stock | PASS |
| 7 | 0 | 0 | 179.11 | 179.09 | −0.01 % | 1,911 | 1,911 | same as stock | PASS |
| 8 | 0 | 0 | 179.24 | 179.38 | +0.08 % | 1,885 | 1,885 | same as stock | PASS |

Every card's offsets read back as set; every card's check model gave its stock hash (`60ee0f178cfd67d5` on all
nine) and passed the 31 kernel checks at stock and with the offsets. Cards 0 to 4 read within 0.26 GB/s of R224's
streaming reads with the same offsets.

### Power and clocks ([a/model_cards.md](a/model_cards.md), the watchdog's 3,663 samples a card over Ornith's rounds; [a/watchdog.md](a/watchdog.md), its 5,746 samples a card from 16:40:55 to 17:28:50)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median over the rounds, MHz | 2,075 | 2,100 | 2,151 | 2,100 | 2,075 | 1,911 | 1,885 | 1,911 | 1,885 |
| memory clock median over the rounds, MHz | 4,455 | 4,498 | 4,551 | 4,608 | 4,551 | 4,006 | 4,006 | 4,006 | 4,006 |
| highest W over the rounds | 77.67 | 82.28 | 81.59 | 83.76 | 78.00 | 73.83 | 74.89 | 75.07 | 72.43 |
| samples above 80 W over the rounds, of 3,663 | 0 | 4 | 4 | 6 | 0 | 0 | 0 | 0 | 0 |

Power went above 80 W only in single samples, never two in a row, against the watchdog's 80 W held for 5 s; the
highest was 83.76 W on card 3. No sample tripped, and the boot logged no GPU Xid.

### The end

At 17:30:58 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, five offsets lowered since the boot (cards 0 to
4), 3,155 MiB of host memory available. The next half needs a fresh boot.

## Files

| File | What it holds |
|---|---|
| [halves.md](halves.md) | One row per half: the load, the rounds, how many gave the stock answer, the first differing round, generation tok/s against stock |
| [half-a.json](half-a.json) | Half a's profile: cards 0 to 4 at use.json's values, cards 5 to 8 at 0 |
| [a/apply.md](a/apply.md), [a/apply_steps.csv](a/apply_steps.csv) | Half a's `apply`, card by card: offsets, streaming reads, peaks, the greedy and kernel checks (the kit's own table) |
| [a/model_rounds.tsv](a/model_rounds.tsv) | Half a's every round: time, hash, tokens, against stock's round 1, prompt and generation tok/s of both prompts |
| [a/load.tsv](a/load.tsv) | Half a's load: when the server was ready, seconds, MiB per card, host memory, the end line |
| [a/model_cards.md](a/model_cards.md) | Each card over half a's rounds: power, samples above 80 W, SM and memory clocks, offsets |
| [a/watchdog.md](a/watchdog.md) | Each card over all the watchdog's samples of half a, `apply`'s check servers included |
| [run/half_profile.py](run/half_profile.py) | Makes a half's profile from use.json and runs the kit's own profile check on it (the one `apply` runs) |
| [run/half_check.sh](run/half_check.sh) | One half on a fresh boot: start check, trip test, watchdog, `apply`, Ornith for 30 minutes against the stored stock answer, then every offset to 0 |
| [run/halves.py](run/halves.py), [run/tables.sh](run/tables.sh) | Build the tables from the folders copied off the rig |
| [run/MD5SUMS-a](run/MD5SUMS-a) | The md5 sums of the files staged on the rig for half a, checked there before the start |

## Commands

On the rig, as root, in the kit's folder `$K` (the first rig's, with R224's `ornith-v5.env` in it), on a fresh boot,
from a checkout of this branch at `$CHECKOUT` (`$D` = this folder):

```bash
python3 -I $D/run/half_profile.py $CHECKOUT/bench/mem_offset/climb.py \
  $CHECKOUT/docs/data/2026-11-20_r223_core_soak_rig1/use.json $K/r226-half-a.json --stock 5 6 7 8 --tag a
cp $D/run/half_check.sh $D/run/half_profile.py $K/
cd $K && md5sum -c $D/run/MD5SUMS-a          # half_check.sh, half_profile.py, the half profile, ornith-v5.env
. ./env.sh && python3 half_profile.py climb.py r226-half-a.json --check && python3 climb.py state && python3 climb.py start-check
KIT=$K BIN=<the v5 engine's bin folder> MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  SETTINGS=$K/ornith-v5.env PROFILE=$K/r226-half-a.json REF=<R224's OUT folder> REF_HASH=adf47784f9b02d4b \
  WORK=/var/tmp/r226a OUT=/var/tmp/r226a-model MINUTES=30 \
  setsid -f nohup bash $K/half_check.sh > /var/tmp/r226a.console 2>&1 < /dev/null
cat /var/tmp/r226a/HALF_DONE                 # "done 0": every round the stock answer; either way every offset is 0
export WORK=/var/tmp/r226a && python3 climb.py state      # every offset 0, the watchdog stopped
```

R224's OUT folder holds its stock round 1 (`greedy-stock1.sha` and `greedy-stock1.json`), which `half_check.sh`
copies into a fresh OUT and checks against `REF_HASH`, so every half is compared with the same stock answer. The next
half repeats these commands with its own `--stock` cards, `--tag`, profile name, `WORK` and `OUT`. The tables, on any
machine, from the two folders copied off the rig:

```bash
bash run/tables.sh a <copy of /var/tmp/r226a> <copy of /var/tmp/r226a-model> .
```

## Provenance and licence

Measured on the first rig on 2026-11-20 (half a: 16:40 to 17:31) with the engine builds named above and the kit as published
on this branch. The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; Ornith-1.5-122B
Pascal N2P GGUF and its licence are described in [release/ornith-v3](../../../release/ornith-v3/README.md). Neither
model is published here, only the hashes of their answers. The tables are this repository's licence.
