# R226: which card of the first rig spoils the served model, found by halves (2026-11-20)

**Then the next runs' profiles (Q1760, 2026-11-24).** `run/mem_down.py` now lowers the core offsets too, so the
next first-rig tests have exact profiles that the kit accepts: the memory alone (every core at stock), the cores one
and two steps lower (every memory at stock), and memory and core each one step lower
([below](#the-next-profiles-q1760-2026-11-24)). No rig ran them yet.

**Then the cores alone, in plain words (R235, 2026-11-24).** On a fresh boot every card's memory ran at stock and only
the core offsets were set. Every card passed its own check, but Ornith gave the stock answer only 4 rounds in a row and
**differed in round 5, 2.8 minutes after the load** ([below](#the-cores-alone-r235-2026-11-24-1838-to-1859)). So the
core offsets alone bring the error back: they cause it, or a share of it. Whether the memory offsets are clean is not
yet known. Generation ran 3.55 % faster than stock while it held (17.23 against 16.64 tok/s on both prompts).

**Then every memory one step lower, in plain words (R231, 2026-11-24).** On a fresh boot the profile ran with every
card's memory offset 100 MT/s lower and the cores kept. Every card passed its own check, but Ornith gave the stock
answer only 21 rounds in a row and **differed in round 22, 11.8 minutes after the load**
([below](#every-memory-one-step-lower-r231-2026-11-24-1359-to-1429)). One memory step bought no clear margin, so the
core offsets are now the first suspect; the first rig stays at stock. Generation ran 6.37 % faster than stock while it
held (17.70 against 16.64 tok/s on prompt 1).

**Then the whole profile for 90 minutes, in plain words (R228).** On one more fresh boot the whole profile was set
again, every card passed its own check, and Ornith gave the stock answer **136 rounds in a row, then differed in round
137, 72.5 minutes after the load** ([below](#whole-profile-90-minutes-r228-2026-11-20-1904-to-2035)). With R224 that is
two differences in 159 rounds of the whole profile, est. once in about 80 rounds or 42 minutes of steady generation.
So R224's difference in round 22 came early, and each half's clean 30 minutes is what so rare an error leaves about
half the time: the halves clear no card. The profile is not exact for Ornith, and the first rig runs at stock.
Generation ran 6.91 % faster than stock while the profile held (17.79 against 16.64 tok/s on both prompts).

**The halves, in plain words.** The first nine-card P106-090 rig's clock profile was split in two
halves, each set alone on its own fresh boot with the other cards at stock. Every card passed its own quick check each
time. Then Ornith-1.5-122B Pascal N2P GGUF, which fills all nine cards, answered the kit's two fixed prompts for 30
minutes: **with cards 0 to 4 overclocked, 56 rounds out of 56 gave the stock answer token for token; with cards 5, 7
and 8 overclocked, 55 out of 55 did.** The full profile had differed in round 22
([R224](../2026-11-20_r224_apply_rig1/README.md)). Neither half reproduced that difference, so the search names no
card. Generation ran 4.21 % faster than stock with the first half and 2.40 % with the second (17.34 and 17.04 against
16.64 tok/s on both prompts). The watchdog never tripped, no card went above 80 W for more than one sample, and
neither boot logged a GPU error. At the end of each half every offset went back to 0.

**What it means.**

- **No single card is shown to be the cause.** If one card erred as often as the whole profile did in R224, once in 22
  rounds, its half would have given a clean 30 minutes only about 7.4 % (half a, 56 rounds) or 7.7 % (half b, 55
  rounds) of the time (est., from that one difference).
- **Two readings fit all three runs.** Either the error is rarer than R224's single sample suggests: if one card causes
  it, that card was overclocked in R224's 22 rounds and in its half's 55 or 56, one difference in about 77 rounds, est.
  once in about 42 minutes of rounds, and at that rate a clean 30-minute half comes about half the time (est. 49 %),
  so neither clean half clears its cards. Or the error needs cards of both halves overclocked at once, and then both
  halves holding is what one would expect.
- **Rig 1 stays at stock.** [Q1640](../../../quants/experiments.md)'s option (a), the whole profile again for 90
  minutes, ran next and differed in round 137
  ([below](#whole-profile-90-minutes-r228-2026-11-20-1904-to-2035)); option (c), every memory one step lower, differed
  in round 22 ([below](#every-memory-one-step-lower-r231-2026-11-24-1359-to-1429)); the cores alone differed in round 5
  ([below](#the-cores-alone-r235-2026-11-24-1838-to-1859), with the options after it).
- **The speed is a single-request diagnostic.** Cards 0 to 4 with their offsets gave +4.21 %, cards 5, 7 and 8 gave
  +2.40 %; all eight cards with offsets gave +6.85 % in R224.

## Why by halves

In step 5 of the first rig's own profile ([R224](../2026-11-20_r224_apply_rig1/README.md)) every card took its
offsets and passed its own check, then Ornith-1.5-122B Pascal N2P GGUF, which spreads its layers over all nine cards,
gave its stock answer 21 rounds in a row and a different answer in round 22, 11 minutes in. One round cannot say which
card caused it. The search sets the profile on some cards and leaves the others at stock, one fresh boot each, and runs
the same 30-minute check ([Q1640](../../../quants/experiments.md), option (b)):

| Half | Cards at the profile's values | Cards at stock | Boot |
|---|---|---|---|
| a | 0, 1, 2, 3, 4 | 5, 6, 7, 8 | 2026-11-20, ~16:35 |
| b | 5, 7, 8 (half a held) | 0, 1, 2, 3, 4, 6 | 2026-11-20, ~17:44 |

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

## Half b: cards 5, 7 and 8 (2026-11-20, 17:50 to 18:38)

The same rig, freshly booted at about 17:44 with nothing run on it before. At 17:49:50 the files staged for half a
were checked on the rig ([run/MD5SUMS-b](run/MD5SUMS-b)), the half profile was made there from the rig's own copy of
use.json (`half_profile.py --stock 0 1 2 3 4 6 --tag b`, the same md5 as from this repository's copy) and passed the
kit's own profile check, and the kit's start check passed: every offset 0, no load process, no GPU Xid this boot, no
offset lowered since the boot, 3,172 MiB of host memory available.

[run/half_check.sh](run/half_check.sh) ran the boot detached (17:50:06 to 18:38:44) with the same engine, settings and
stock reference as half a:

1. **The stock reference:** R224's stock round 1 (`adf47784f9b02d4b`) copied into a fresh output folder and its hash
   checked.
2. **The start check, the trip test** (17:50:06 to 17:50:42: the watchdog's four trips with stand-in processes, each
   PASS) **and the watchdog** (17:50:45).
3. **`apply` of the half profile** ([half-b.json](half-b.json), 17:50:49 to 17:56:44): the check servers, the stock
   rounds, every card's offsets set 17:55:46 to 17:55:51 and read back as set, then one round with the offsets: every
   card PASS at 17:56:36.
4. **Ornith with the offsets**, rounds for 30 minutes after the load, each against the stored stock answer: loaded
   in 672 s (ready 18:08:01), 55 rounds from 18:08:39 to 18:38:28, every one the stock hash.
5. **The reset:** every card's offsets to 0 (18:38:35 to 18:38:44), then the watchdog stopped.

### Ornith's whole-memory check ([halves.md](halves.md); every round in [b/model_rounds.tsv](b/model_rounds.tsv))

| | R224 stock | R224, the whole profile | half a: cards 0 to 4 | **half b: cards 5, 7, 8** | half b against stock |
|---|---|---|---|---|---|
| load, s | 673 | 675 | 673 | **672** | |
| host memory available when loaded, MiB | 1,496 | 1,502 | 1,489 | **1,502** | |
| rounds; the stock answer | 2 (round 1 is the reference) | 22; 21 of 22, round 22 differed (11.7 min after the load) | 56; 56 of 56 in 1,831 s | **55; 55 of 55** in 1,832 s | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 16.04, 16.72 | 15.71, 16.46 | **15.28, 16.06** | +1.73 %, +1.90 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.79, 17.78 | 17.35, 17.34 | **17.04, 17.02** | +2.40 %, +2.28 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.78, 17.78 (21 rounds) | 17.34, 17.34 (55 rounds) | **17.04, 17.04** (54 rounds) | **+2.40 %, +2.40 %** |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.86, 29.67 (round 2) | 28.66, 31.61 | 28.13, 31.02 | **27.37, 30.25** | +1.90 %, +1.95 % |

- From round 2 on, prompt 1 generated 17.04 tok/s in 45 rounds, 17.03 in 7, 17.02 and 17.05 in one each; prompt 2
  17.04 in 28, 17.03 in 25 and 17.02 in 1.
- The load used the same memory as R224's and half a's: 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on
  cards 0 to 8.

### `apply`, card by card ([b/apply.md](b/apply.md); the kit's own table [b/apply_steps.csv](b/apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | 0 | 0 | 179.18 | 179.34 | +0.09 % | 1,873 | 1,860 | same as stock | PASS |
| 1 | 0 | 0 | 179.01 | 179.06 | +0.03 % | 1,860 | 1,860 | same as stock | PASS |
| 2 | 0 | 0 | 179.23 | 179.23 | +0.00 % | 1,911 | 1,911 | same as stock | PASS |
| 3 | 0 | 0 | 179.41 | 179.08 | −0.18 % | 1,911 | 1,911 | same as stock | PASS |
| 4 | 0 | 0 | 179.12 | 179.27 | +0.08 % | 1,885 | 1,873 | same as stock | PASS |
| 5 | +1,300 | +231 | 179.15 | 206.96 | +15.52 % | 1,911 | 2,138 | same as stock | PASS |
| 6 | 0 | 0 | 179.14 | 179.34 | +0.11 % | 1,898 | 1,885 | same as stock | PASS |
| 7 | +1,100 | +282 | 179.20 | 201.88 | +12.66 % | 1,911 | 2,189 | same as stock | PASS |
| 8 | +900 | +231 | 179.26 | 197.73 | +10.30 % | 1,898 | 2,113 | same as stock | PASS |

Every card's offsets read back as set; every card's check model gave its stock hash (`60ee0f178cfd67d5` on all
nine) and passed the 31 kernel checks at stock and with the offsets. Cards 5, 7 and 8 read within 0.26 GB/s of R224's
streaming reads with the same offsets.

### Power and clocks ([b/model_cards.md](b/model_cards.md), the watchdog's 3,665 samples a card over Ornith's rounds; [b/watchdog.md](b/watchdog.md), its 5,761 samples a card from 17:50:42 to 18:38:44)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median over the rounds, MHz | 1,860 | 1,847 | 1,898 | 1,898 | 1,873 | 2,138 | 1,885 | 2,189 | 2,113 |
| memory clock median over the rounds, MHz | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 | 4,657 | 4,006 | 4,551 | 4,455 |
| highest W over the rounds | 75.34 | 76.00 | 75.87 | 76.68 | 70.95 | 79.33 | 75.39 | 81.15 | 78.77 |
| samples above 80 W over the rounds, of 3,665 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 2 | 0 |

Power went above 80 W only in single samples, never two in a row (card 7 twice over the rounds, card 1 once during
`apply`'s checks, highest 81.54 W), against the watchdog's 80 W held for 5 s. No sample tripped, and the boot logged
no GPU Xid.

### The end

At 18:40:42 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, three offsets lowered since the boot (cards 5, 7
and 8), 3,161 MiB of host memory available. Any further test needs a fresh boot.

## Whole profile, 90 minutes (R228, 2026-11-20, 19:04 to 20:35)

Neither half alone reproduced R224's difference, so the whole profile ran again with a longer check
([Q1640](../../../quants/experiments.md), option (a)): all eight overclocked cards at their values, card 6 at stock,
and Ornith's rounds for 90 minutes instead of 30. The same rig, freshly booted at about 19:00 with nothing run on it
before. At 19:04:19 the files on the rig were checked ([run/MD5SUMS-full90](run/MD5SUMS-full90); the profile is
R223's [use.json](../2026-11-20_r223_core_soak_rig1/use.json), md5 `ae91f969c8ddc782698c8fd9d570734d`), the profile
passed the kit's own profile check (`half_profile.py --check`: cards 0 to 5, 7 and 8 set, card 6 stock), and the kit's
start check passed: every offset 0, no load process, no GPU Xid this boot, no offset lowered since the boot, 3,168 MiB
of host memory available.

[run/half_check.sh](run/half_check.sh) ran the boot detached (19:04:32 to 20:35:15) with `MINUTES=90` and the same
engine, settings and stock reference as the halves:

1. **The stock reference:** R224's stock round 1 (`adf47784f9b02d4b`) copied into a fresh output folder and its hash
   checked.
2. **The start check, the trip test** (19:04:32 to 19:05:08: the watchdog's four trips with stand-in processes, each
   PASS) **and the watchdog** (19:05:11).
3. **`apply` of the whole profile** (19:05:15 to 19:11:10): the check servers, the stock rounds, every card's offsets
   set 19:10:08 to 19:10:14 and read back as set, then one round with the offsets: every card PASS at 19:11:00.
4. **Ornith with the offsets**, rounds for 90 minutes after the load, each against the stored stock answer: loaded in
   671 s (ready 19:22:25), 137 rounds from 19:23:02 to 20:34:58. **Rounds 1 to 136 gave the stock hash; round 137
   gave `0e9263fcf6d12ba1`**, so the check stopped there, 72.5 minutes after the load.
5. **The reset:** every card's offsets to 0 (20:35:04 to 20:35:14), then the watchdog stopped (20:35:15).

### Ornith's whole-memory check ([halves.md](halves.md), row `full90`; every round in [full90/model_rounds.tsv](full90/model_rounds.tsv))

| | R224 stock | R224, the whole profile | half a: cards 0 to 4 | half b: cards 5, 7, 8 | **the whole profile, 90 minutes** | against stock |
|---|---|---|---|---|---|---|
| load, s | 673 | 675 | 673 | 672 | **671** | |
| host memory available when loaded, MiB | 1,496 | 1,502 | 1,489 | 1,502 | **1,506** | |
| rounds; the stock answer | 2 (round 1 is the reference) | 22; 21 of 22, round 22 differed (11.7 min after the load) | 56; 56 of 56 in 1,831 s | 55; 55 of 55 in 1,832 s | **137; 136 of 137, round 137 differed** (72.5 min after the load; 4,358 s) | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 16.04, 16.72 | 15.71, 16.46 | 15.28, 16.06 | **15.95, 16.83** | +6.19 %, +6.79 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.79, 17.78 | 17.35, 17.34 | 17.04, 17.02 | **17.80, 17.80** | +6.97 %, +6.97 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.78, 17.78 (21 rounds) | 17.34, 17.34 (55 rounds) | 17.04, 17.04 (54 rounds) | **17.79, 17.79** (136 rounds) | **+6.91 %, +6.91 %** |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.86, 29.67 (round 2) | 28.66, 31.61 | 28.13, 31.02 | 27.37, 30.25 | **28.74, 31.71** | +7.00 %, +6.88 % |

- From round 2 on, prompt 1 generated 17.79 tok/s in 84 rounds, 17.80 in 49, 17.78 in 2 and 17.77 in 1; prompt 2
  17.79 in 98, 17.78 in 12, 17.80 in 25 and 17.77 in 1. Round 137 generated 17.80 on both prompts, as fast as the
  rounds before it.
- The load used the same memory as R224's and the halves': 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB
  on cards 0 to 8.

### How round 137 differs ([full90/ornith_differs.csv](full90/ornith_differs.csv))

| | prompt 1 | prompt 2 |
|---|---|---|
| round 137 (this run) | bit-identical to stock | **the same text and the same 256 tokens**; the top-10 log-probabilities part from stock at token 202 and differ on every token after it (54 tokens), the chosen token's by up to 0.138 (token 226) |
| round 22 (R224) | the text parts from stock at token 184 (72 of 256 tokens differ); the log-probabilities from token 167 | bit-identical to stock |

Both differences start late in one prompt and run to its end, while the other prompt stays bit-identical: one wrong
value partway through a reply, carried forward by everything computed after it. They fall on different prompts at
different tokens, so they are not one fixed place in the model or the prompts.

### `apply`, card by card ([full90/apply.md](full90/apply.md); the kit's own table [full90/apply_steps.csv](full90/apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +900 | +218 | 179.12 | 197.23 | +10.11 % | 1,873 | 2,088 | same as stock | PASS |
| 1 | +1,000 | +256 | 179.34 | 199.48 | +11.23 % | 1,860 | 2,113 | same as stock | PASS |
| 2 | +1,100 | +256 | 179.31 | 202.00 | +12.65 % | 1,911 | 2,164 | same as stock | PASS |
| 3 | +1,200 | +206 | 179.23 | 204.58 | +14.14 % | 1,911 | 2,113 | same as stock | PASS |
| 4 | +1,100 | +206 | 179.31 | 202.15 | +12.74 % | 1,885 | 2,088 | same as stock | PASS |
| 5 | +1,300 | +231 | 179.24 | 206.92 | +15.44 % | 1,911 | 2,138 | same as stock | PASS |
| 6 | 0 | 0 | 179.37 | 179.20 | −0.09 % | 1,898 | 1,885 | same as stock | PASS |
| 7 | +1,100 | +282 | 179.26 | 201.96 | +12.66 % | 1,911 | 2,189 | same as stock | PASS |
| 8 | +900 | +231 | 179.29 | 197.43 | +10.12 % | 1,898 | 2,113 | same as stock | PASS |

Every card's offsets read back as set; every card's check model gave its stock hash and passed the 31 kernel checks
at stock and with the offsets. Every overclocked card's streaming read came within 0.32 GB/s of its half's with the same
offsets.

### Power and clocks ([full90/model_cards.md](full90/model_cards.md), the watchdog's 8,716 samples a card over Ornith's rounds; [full90/watchdog.md](full90/watchdog.md), its 10,804 samples a card from 19:05:08 to 20:35:14)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median, MHz | 2,075 | 2,100 | 2,151 | 2,100 | 2,075 | 2,138 | 1,885 | 2,189 | 2,113 |
| memory clock median, MHz | 4,455 | 4,498 | 4,551 | 4,608 | 4,551 | 4,657 | 4,006 | 4,551 | 4,455 |
| highest W | 78.21 | 80.93 | 81.94 | 83.32 | 78.64 | 79.66 | 75.94 | 80.59 | 78.94 |
| samples above 80 W, of 8,716 | 0 | 10 | 7 | 12 | 0 | 0 | 0 | 4 | 0 |

Power went above 80 W only in single samples, never two in a row, against the watchdog's 80 W held for 5 s; the
highest was 83.32 W on card 3. No sample tripped, and the boot logged no GPU Xid.

### What the two whole-profile runs and the halves allow

- **The whole profile errs on Ornith, rarely:** two differences in 159 rounds (R224's 22 and this run's 137), 5,067 s
  of rounds, est. once in about 80 rounds or 42 minutes of steady generation. One sample of two events gives a rough
  rate only.
- **R224's one difference in 22 rounds came early:** at that rate 136 clean rounds in a row would come only 0.18 % of
  the time (est.).
- **The halves clear no card:** at the whole profile's rate a clean 30-minute half (55 or 56 rounds) comes about half
  the time (est. 49 %) even when that half holds the cause. The error may come from one card or need several together;
  these runs cannot tell.
- **Memory or core:** each card's memory and core sit two steps under every failure of steps 1 to 4, and both
  differences are one wrong value carried to a reply's end; these runs cannot say which clock causes it.
- **What a next run needs:** at this rate a 90-minute check catches the error with est. 88 %, a 120-minute check with
  est. 94 %, so a clean 30 minutes says little about any change.

### Options after it ([Q1640](../../../quants/experiments.md))

Each needs a fresh boot; the rigs stop at 22:00.

- **(c) every card's memory one step (100 MT/s) lower, core kept**, then step 5 with 120 minutes of rounds
  ([run/mem_down.py](run/mem_down.py) makes that profile, md5 `67d8cb5c5685c74d7c461696eea42820` from R223's
  use.json): one boot, est. 2 h 20 min (90 minutes: est. 1 h 50 min).
- **(c′) every card's memory and core each one step lower**, then the same 120-minute check: a margin on both clocks,
  since neither run can say which one errs; `mem_down.py` needs a core step first.
- **(b″) the halves again, 120 minutes each:** if one card causes it, its half shows it with est. 94 %; two boots, est.
  2 h 20 min each.
- **(d) a per-card check model that fills most of each card**, so the kit's own soaks read the memory Ornith uses
  (code work first).
- **(e) the profile as it is:** +6.91 % generation on a single request, with Ornith's answer parting from stock est.
  once in about 42 minutes of steady generation; step 5 asks for the served model's stock answer in every
  round.

### The end

At 20:40:29 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, eight offsets lowered since the boot (every card
but card 6), 3,155 MiB of host memory available. Any further test needs a fresh boot.

## Every memory one step lower (R231, 2026-11-24, 13:59 to 14:29)

[Q1640](../../../quants/experiments.md)'s option (c): the whole profile with every card's memory offset one step
(100 MT/s) lower and every core offset kept, card 6 at stock, then Ornith's rounds for 120 minutes. The same rig,
freshly booted at about 13:46 with nothing run on it before. At 13:59:07 the files on the rig were checked
([run/MD5SUMS-mem120](run/MD5SUMS-mem120)), [run/mem_down.py](run/mem_down.py) made the profile on the rig from R223's
[use.json](../2026-11-20_r223_core_soak_rig1/use.json) ([mem-down.json](mem-down.json), md5
`67d8cb5c5685c74d7c461696eea42820`), the profile passed the kit's own profile check (cards 0 to 5, 7 and 8 set, card 6
stock), and the kit's start check passed: every offset 0, no load process, no GPU Xid this boot, no offset lowered since
the boot, 3,165 MiB of host memory available.

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memory, MT/s | +800 | +900 | +1,000 | +1,100 | +1,000 | +1,200 | 0 | +1,000 | +800 |
| core, MHz | +218 | +256 | +256 | +206 | +206 | +231 | 0 | +282 | +231 |

[run/half_check.sh](run/half_check.sh) ran the boot detached (13:59:22 to 14:29:25) with `MINUTES=120` and the same
engine, settings and stock reference as the runs above:

1. **The stock reference:** R224's stock round 1 (`adf47784f9b02d4b`) copied into a fresh output folder and its hash
   checked.
2. **The start check, the trip test** (13:59:22 to 13:59:58: the watchdog's four trips with stand-in processes, each
   PASS) **and the watchdog** (14:00:01).
3. **`apply` of the profile** (14:00:05 to 14:06:01): the check servers, the stock rounds, every card's offsets set
   14:05:02 to 14:05:08 and read back as set, then one round with the offsets: every card PASS at 14:05:53.
4. **Ornith with the offsets**, rounds planned for 120 minutes after the load, each against the stored stock answer:
   loaded in 675 s (ready 14:17:20), 22 rounds from 14:17:57 to 14:29:07. **Rounds 1 to 21 gave the stock hash; round
   22 gave `86422e3068f83d65`**, so the check stopped there, 11.8 minutes after the load.
5. **The reset:** every card's offsets to 0 (14:29:14 to 14:29:24), then the watchdog stopped (14:29:25).

### Ornith's whole-memory check ([halves.md](halves.md), row `mem120`; every round in [mem120/model_rounds.tsv](mem120/model_rounds.tsv))

| | R224 stock | the whole profile, 90 minutes (R228) | **every memory one step lower** | against stock |
|---|---|---|---|---|
| load, s | 673 | 671 | **675** | |
| host memory available when loaded, MiB | 1,496 | 1,506 | **1,500** | |
| rounds; the stock answer | 2 (round 1 is the reference) | 137; 136 of 137, round 137 differed (72.5 min after the load) | **22; 21 of 22, round 22 differed** (11.8 min after the load; 712 s) | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 15.95, 16.83 | **15.82, 16.66** | +5.33 %, +5.71 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.80, 17.80 | **17.71, 17.70** | +6.43 %, +6.37 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.79, 17.79 (136 rounds) | **17.70, 17.69** (21 rounds) | **+6.37 %, +6.31 %** |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.86, 29.67 (round 2) | 28.74, 31.71 | **28.58, 31.56** | +6.40 %, +6.37 % |

- From round 2 on, prompt 1 generated 17.70 tok/s in 12 rounds, 17.69 in 7 and 17.71 in 2; prompt 2 17.70 in 10, 17.69
  in 10 and 17.68 in 1. Round 22 generated 17.70 and 17.68, as fast as the rounds before it.
- The load used the same memory as every run above: 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on
  cards 0 to 8.

### How round 22 differs ([mem120/ornith_differs.csv](mem120/ornith_differs.csv))

| | prompt 1 | prompt 2 |
|---|---|---|
| round 22 (this run) | bit-identical to stock | **the same text and the same 256 tokens**; the top-10 log-probabilities part from stock at token 135 and differ on every token after it (121 tokens), the chosen token's by up to 0.219 (token 206) |
| round 137 (R228) | bit-identical to stock | the same text and the same 256 tokens; the log-probabilities part from stock at token 202 (54 tokens differ), the chosen token's by up to 0.138 |
| round 22 (R224) | the text parts from stock at token 184; the log-probabilities from token 167 | bit-identical to stock |

The same kind of difference as before: one wrong value partway through one prompt's reply, carried to its end, the
other prompt bit-identical, at a different token each time.

### `apply`, card by card ([mem120/apply.md](mem120/apply.md); the kit's own table [mem120/apply_steps.csv](mem120/apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +800 | +218 | 179.15 | 195.25 | +8.99 % | 1,873 | 2,075 | same as stock | PASS |
| 1 | +900 | +256 | 179.27 | 197.58 | +10.21 % | 1,873 | 2,100 | same as stock | PASS |
| 2 | +1,000 | +256 | 179.20 | 199.65 | +11.41 % | 1,911 | 2,151 | same as stock | PASS |
| 3 | +1,100 | +206 | 179.11 | 201.84 | +12.69 % | 1,911 | 2,100 | same as stock | PASS |
| 4 | +1,000 | +206 | 179.38 | 199.63 | +11.29 % | 1,885 | 2,075 | same as stock | PASS |
| 5 | +1,200 | +231 | 179.11 | 204.70 | +14.29 % | 1,911 | 2,138 | same as stock | PASS |
| 6 | 0 | 0 | 179.12 | 179.06 | −0.03 % | 1,885 | 1,885 | same as stock | PASS |
| 7 | +1,000 | +282 | 179.27 | 199.67 | +11.38 % | 1,911 | 2,189 | same as stock | PASS |
| 8 | +800 | +231 | 179.12 | 195.10 | +8.92 % | 1,898 | 2,113 | same as stock | PASS |

Every card's offsets read back as set; every card's check model gave its stock hash and passed the 31 kernel checks
at stock and with the offsets.

### Power and clocks ([mem120/model_cards.md](mem120/model_cards.md), the watchdog's 1,426 samples a card over Ornith's rounds; [mem120/watchdog.md](mem120/watchdog.md), its 3,523 samples a card from 13:59:58 to 14:29:24)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median, MHz | 2,075 | 2,100 | 2,151 | 2,100 | 2,075 | 2,138 | 1,873 | 2,189 | 2,113 |
| memory clock median, MHz | 4,404 | 4,455 | 4,498 | 4,551 | 4,498 | 4,608 | 4,006 | 4,498 | 4,404 |
| highest W | 78.25 | 81.60 | 81.25 | 83.83 | 77.69 | 80.42 | 74.00 | 78.74 | 77.51 |
| samples above 80 W, of 1,426 | 0 | 2 | 2 | 1 | 0 | 1 | 0 | 0 | 0 |

Power went above 80 W only in single samples, never two in a row; the highest was 83.83 W on card 3. No sample
tripped, and the boot logged no GPU Xid.

### What it allows

- **One memory step does not remove the error.** If this profile erred as often as the whole profile (two differences
  in 159 rounds), a difference within 22 rounds would come est. 24 % of the time, so this early difference does not say
  the rate rose; it does say the step bought no clear margin. Had the step cut the rate tenfold, a difference this
  early would come est. 3 % of the time.
- **The core offsets, kept here, are now the first suspect;** a memory error that needs more than one step is the other
  reading. These runs still cannot tell one card from several.
- **Two of the three differences came in round 22, 11.7 and 11.8 minutes after the load** (R224 and this run); R228's
  came in round 137. Three events are too few to say whether that is chance.
- **Speed while it held:** +6.37 % and +6.31 % generation on a single request, against +6.91 % for the whole profile.

### Options after it ([Q1640](../../../quants/experiments.md))

Each needs a fresh boot; the rigs stop at 22:00.

- **(c′) every card's memory and core each one step lower**, then 120 minutes of rounds: `mem_down.py` needs a core
  step first (Q1760).
- **The cores alone:** every memory offset at 0 with the profile's cores (`mem_down.py --steps 13` makes it, no code
  needed), then 120 minutes of rounds: if it differs, the cores cause the error; if it holds, the memory needs more than
  one step (Q1759).
- **(b″) the halves again, 120 minutes each**, **(d) a per-card check model that fills most of each card**, and
  **(e) the profile as it is**, as listed after R228 above.

### The end

At 14:30:01 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, eight offsets lowered since the boot (every card
but card 6), 3,165 MiB of host memory available. Any further test, and any speed run at stock, needs a fresh boot: a
card whose memory offset was lowered runs slower at stock until the next power-on (Q1563).

## The cores alone (R235, 2026-11-24, 18:38 to 18:59)

[Q1759](../../../quants/experiments.md): every card's memory offset at 0 and every core offset as in the profile, card 6
at stock, then Ornith's rounds for 120 minutes. If it differs, the core offsets cause the error or share it; if it
holds, the memory needs more than one step. The same rig, freshly booted at about 18:36 with nothing run on it before.
At 18:38:28 the files on the rig were checked ([run/MD5SUMS-cores120](run/MD5SUMS-cores120)),
[run/mem_down.py](run/mem_down.py) with `--steps 13` (13 memory steps take the largest offset, card 5's +1,300 MT/s,
to 0) made the profile on the rig from R223's [use.json](../2026-11-20_r223_core_soak_rig1/use.json)
([cores.json](cores.json), md5 `c12d1573989cab8ab8b3f0ae7b85e48f`), the profile passed the kit's own profile check
(every memory +0, the cores set on cards 0 to 5, 7 and 8, card 6 stock), and the kit's start check passed: every offset
0, no load process, no GPU Xid this boot, no offset lowered since the boot, 3,160 MiB of host memory available.

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| memory, MT/s | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| core, MHz | +218 | +256 | +256 | +206 | +206 | +231 | 0 | +282 | +231 |

[run/half_check.sh](run/half_check.sh) ran the boot detached (18:38:45 to 18:59:21) with `MINUTES=120` and the same
engine, settings and stock reference as the runs above:

1. **The stock reference:** R224's stock round 1 (`adf47784f9b02d4b`) copied into a fresh output folder and its hash
   checked.
2. **The start check, the trip test** (18:38:45 to 18:39:21: the watchdog's four trips with stand-in processes, each
   PASS) **and the watchdog** (18:39:21).
3. **`apply` of the profile** (18:39:27 to 18:44:56): the check servers, the stock rounds, every card's offsets set
   18:43:58 to 18:44:03 and read back as set, then one round with the offsets: every card PASS at 18:44:48.
4. **Ornith with the offsets**, rounds planned for 120 minutes after the load, each against the stored stock answer:
   loaded in 675 s (ready 18:56:15), 5 rounds from 18:56:53 to 18:59:04. **Rounds 1 to 4 gave the stock hash; round 5
   gave `6f45f3ece7f7d06a`**, so the check stopped there, 2.8 minutes after the load.
5. **The reset:** every card's offsets to 0 (18:59:11 to 18:59:20), then the watchdog stopped (18:59:21).

### Ornith's whole-memory check ([halves.md](halves.md), row `cores120`; every round in [cores120/model_rounds.tsv](cores120/model_rounds.tsv))

| | R224 stock | the whole profile, 90 minutes (R228) | every memory one step lower (R231) | **the cores alone** | against stock |
|---|---|---|---|---|---|
| load, s | 673 | 671 | 675 | **675** | |
| host memory available when loaded, MiB | 1,496 | 1,506 | 1,500 | **1,490** | |
| rounds; the stock answer | 2 (round 1 is the reference) | 137; 136 of 137, round 137 differed (72.5 min after the load) | 22; 21 of 22, round 22 differed (11.8 min) | **5; 4 of 5, round 5 differed** (2.8 min after the load; 174 s) | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 15.95, 16.83 | 15.82, 16.66 | **15.56, 16.18** | +3.60 %, +2.66 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.80, 17.80 | 17.71, 17.70 | **17.23, 17.24** | +3.55 %, +3.61 % |
| generation tok/s, median of rounds 2 on (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.79, 17.79 (136 rounds) | 17.70, 17.69 (21 rounds) | **17.23, 17.23** (4 rounds) | **+3.55 %, +3.55 %** |
| prompt processing tok/s, median of rounds 2 on (prompts 1, 2) | 26.86, 29.67 (round 2) | 28.74, 31.71 | 28.58, 31.56 | **28.545, 31.50** | +6.27 %, +6.17 % |

- Rounds 2 to 4 generated 17.23 tok/s on prompt 1 in each round, and 17.24, 17.23 and 17.23 on prompt 2. Round 5
  generated 16.67 on prompt 1, whose text differed (R224's round 22, whose prompt 1 text also differed, generated 17.25
  against 17.78 in its other rounds), and 17.23 on prompt 2.
- The cores alone gave about half the generation gain of the whole profile (+3.55 % against +6.91 %) and nearly all of
  its prompt processing gain (+6.27 % and +6.17 % against +7.00 % and +6.88 %).
- The load used the same memory as every run above: 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on
  cards 0 to 8.

### How round 5 differs ([cores120/ornith_differs.csv](cores120/ornith_differs.csv))

| | prompt 1 | prompt 2 |
|---|---|---|
| round 5 (this run) | **the text parts from stock at token 181** (75 of 256 tokens differ); the top-10 log-probabilities part from token 70, the chosen token's by up to 0.098 before the text parts | bit-identical to stock |
| round 22 (R231) | bit-identical to stock | the same text and the same 256 tokens; the log-probabilities part from stock at token 135 |
| round 137 (R228) | bit-identical to stock | the same text and the same 256 tokens; the log-probabilities part from stock at token 202 |
| round 22 (R224) | the text parts from stock at token 184; the log-probabilities from token 167 | bit-identical to stock |

The same kind of difference as in every run above: one wrong value partway through one prompt's reply, carried to its
end, the other prompt bit-identical.

### `apply`, card by card ([cores120/apply.md](cores120/apply.md); the kit's own table [cores120/apply_steps.csv](cores120/apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +0 | +218 | 179.35 | 179.14 | −0.12 % | 1,860 | 2,075 | same as stock | PASS |
| 1 | +0 | +256 | 179.32 | 179.06 | −0.14 % | 1,860 | 2,113 | same as stock | PASS |
| 2 | +0 | +256 | 179.23 | 179.15 | −0.04 % | 1,911 | 2,151 | same as stock | PASS |
| 3 | +0 | +206 | 179.35 | 179.12 | −0.13 % | 1,911 | 2,100 | same as stock | PASS |
| 4 | +0 | +206 | 179.11 | 179.11 | +0.00 % | 1,885 | 2,075 | same as stock | PASS |
| 5 | +0 | +231 | 179.44 | 179.26 | −0.10 % | 1,911 | 2,138 | same as stock | PASS |
| 6 | +0 | +0 | 179.29 | 179.21 | −0.04 % | 1,885 | 1,873 | same as stock | PASS |
| 7 | +0 | +282 | 179.34 | 178.89 | −0.25 % | 1,911 | 2,189 | same as stock | PASS |
| 8 | +0 | +231 | 179.11 | 179.21 | +0.06 % | 1,885 | 2,100 | same as stock | PASS |

Every card's offsets read back as set; every card's check model gave its stock hash and passed the 31 kernel checks
at stock and with the offsets. With the memory at stock the streaming reads stay at stock, as expected.

### Power and clocks ([cores120/model_cards.md](cores120/model_cards.md), the watchdog's 350 samples a card over Ornith's rounds; [cores120/watchdog.md](cores120/watchdog.md), its 2,391 samples a card from 18:39:21 to 18:59:20)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median, MHz | 2,062 | 2,100 | 2,151 | 2,100 | 2,075 | 2,138 | 1,873 | 2,189 | 2,100 |
| memory clock median, MHz | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 | 4,006 |
| highest W over the rounds | 75.22 | 80.13 | 66.93 | 68.95 | 75.05 | 77.33 | 74.13 | 76.63 | 72.98 |
| samples above 80 W, of 350 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

Over the whole boot power went above 80 W only in single samples, never two in a row (cards 1, 3, 5 and 6: 1, 1, 2
and 1 of 2,391; the highest 83.92 W on card 6, which ran at stock). No sample tripped, and the boot logged no GPU Xid.

### What it allows

- **The core offsets alone are enough to bring the error back.** With every memory clock at stock, Ornith differed in
  round 5. So the cores cause the error, or a share of it; the memory offsets are not needed for it.
- **The memory offsets are not cleared.** This run says nothing about the memory alone: it may err too.
- **Round 5 is early, but not out of line.** If the cores alone erred at the whole profile's rate (two differences in
  159 rounds), a difference within 5 rounds would come est. 6.1 % of the time. Over every run with the profile's cores
  (R224, R228, R231 and this one) that is four differences in 186 rounds, est. once in about 47 rounds or 26 minutes of
  rounds. Four events give a rough rate only.
- **The per-card check does not catch it.** Every card passed `apply`'s check (the check model's stock answer and 31
  kernel checks) at these cores, and so did step 4's soak (R223), yet Ornith differed within three minutes: the check
  does not load the cores the way Ornith does.
- **Speed while it held:** +3.55 % generation and +6.27 % prompt processing on a single request.

### Options after it ([Q1640](../../../quants/experiments.md))

Each needs a fresh boot; the rigs stop at 22:00.

- **The memory alone** (Q1795): every core offset at 0 with the profile's memory offsets, card 6 at stock, then 120
  minutes of rounds. If it holds, the first rig can run with its memory offsets (a smaller gain than the whole
  profile's, est. about half of it) while the cores are searched; if it differs, both clocks err. Its profile comes
  from Q1760's `mem_down.py --core-steps`, every core taken to 0 and no memory step.
- **The cores lower:** every memory at 0 and every core one or more steps lower (Q1760's `--core-steps`), 120 minutes
  each, until a step holds.
- **A per-card check that loads the cores** (Q1796): the kit's `apply` and soak check each card with work that drives
  the cores as Ornith does, so a core offset that spoils Ornith fails there, on any rig.
- (c′), (b″), (d) and (e), as listed after R228 above.

### The end

At 18:59:52 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, eight offsets lowered since the boot (the cores of
every card but card 6; no memory offset was set), 3,153 MiB of host memory available. Any further test needs a fresh
boot: the kit's start check refuses a boot with a lowered offset.

## The next profiles (Q1760, 2026-11-24)

The memory alone and the cores lower need profiles that change the core offsets, which `run/mem_down.py` could not do.
It now takes `--core-steps N`: N of the kit's core steps (2 driver steps of 12.66 MHz each, `climb.py`'s
`CORE_STEP_BINS` and `CORE_BIN_MHZ`) off every card's `core_steps`, never below 0, with `core_mhz` set from the new
step count by the kit's own rule (`climb.py`'s `core_offset()`, the offset the climb sets for that many steps), so the
two fields agree. `--core-steps all` takes every core offset to 0, `--steps all` every memory offset, and `--steps 0`
keeps the memory. A profile whose `core_mhz` and `core_steps` already disagree is refused. The calls without
`--core-steps` give the same bytes as before: rerun from R223's use.json, they give [mem-down.json](mem-down.json)
(`67d8cb5c…`) and [cores.json](cores.json) (`c12d1573…`) byte for byte. Nothing here ran on a rig.

Each profile below passed the kit's own profile check (`half_profile.py climb.py <profile> --check`, the check
`climb.py apply` runs before it sets anything: memory cap +1,400 MT/s, core cap +400 MHz, 12.66 MHz a driver step;
the kit's `env.sh` changes none of these), and each is made from R223's
[use.json](../2026-11-20_r223_core_soak_rig1/use.json) (`ae91f969c8ddc782698c8fd9d570734d`):

| Profile | For | `mem_down.py` options | md5 |
|---|---|---|---|
| [mem-alone.json](mem-alone.json) | the memory alone (Q1795): every core at 0, the memory offsets kept | `--steps 0 --core-steps all` | `52be6a07347964e074478bdd45537fb5` |
| [cores-down1.json](cores-down1.json) | the cores one step lower, every memory at 0 | `--steps all --core-steps 1` | `37df8e545706e8b9816b47d83fda747c` |
| [cores-down2.json](cores-down2.json) | the cores two steps lower, every memory at 0 | `--steps all --core-steps 2` | `f3eb02712f4251dd7ace42345056298d` |
| [mem-core-down1.json](mem-core-down1.json) | Q1640's (c′): memory and core each one step lower | `--core-steps 1` | `8ba42479e24212d150d6eef4e9a8278a` |

Memory MT/s and core MHz (driver steps in brackets) that `apply` would set on each card; card 6 stays at stock in
every profile:

| Card | use.json | mem-alone.json | cores-down1.json | cores-down2.json | mem-core-down1.json |
|---|---|---|---|---|---|
| 0 | +900 / +218 (17) | +900 / 0 | 0 / +193 (15) | 0 / +168 (13) | +800 / +193 (15) |
| 1 | +1,000 / +256 (20) | +1,000 / 0 | 0 / +231 (18) | 0 / +206 (16) | +900 / +231 (18) |
| 2 | +1,100 / +256 (20) | +1,100 / 0 | 0 / +231 (18) | 0 / +206 (16) | +1,000 / +231 (18) |
| 3 | +1,200 / +206 (16) | +1,200 / 0 | 0 / +180 (14) | 0 / +155 (12) | +1,100 / +180 (14) |
| 4 | +1,100 / +206 (16) | +1,100 / 0 | 0 / +180 (14) | 0 / +155 (12) | +1,000 / +180 (14) |
| 5 | +1,300 / +231 (18) | +1,300 / 0 | 0 / +206 (16) | 0 / +180 (14) | +1,200 / +206 (16) |
| 6 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| 7 | +1,100 / +282 (22) | +1,100 / 0 | 0 / +256 (20) | 0 / +231 (18) | +1,000 / +256 (20) |
| 8 | +900 / +231 (18) | +900 / 0 | 0 / +206 (16) | 0 / +180 (14) | +800 / +206 (16) |

Each run repeats the commands below on its own fresh boot with its profile as `PROFILE` and `MINUTES=120`; the commands
that make and check each profile on the rig are at the end of [Commands](#commands).

## Files

| File | What it holds |
|---|---|
| [halves.md](halves.md) | One row per run (the halves a and b, `full90`, the whole profile for 90 minutes, `mem120`, every memory one step lower, and `cores120`, the cores alone): the load, the rounds, how many gave the stock answer, the first differing round, generation tok/s against stock |
| [half-a.json](half-a.json) | Half a's profile: cards 0 to 4 at use.json's values, cards 5 to 8 at 0 |
| [a/apply.md](a/apply.md), [a/apply_steps.csv](a/apply_steps.csv) | Half a's `apply`, card by card: offsets, streaming reads, peaks, the greedy and kernel checks (the kit's own table) |
| [a/model_rounds.tsv](a/model_rounds.tsv) | Half a's every round: time, hash, tokens, against stock's round 1, prompt and generation tok/s of both prompts |
| [a/load.tsv](a/load.tsv) | Half a's load: when the server was ready, seconds, MiB per card, host memory, the end line |
| [a/model_cards.md](a/model_cards.md) | Each card over half a's rounds: power, samples above 80 W, SM and memory clocks, offsets |
| [a/watchdog.md](a/watchdog.md) | Each card over all the watchdog's samples of half a, `apply`'s check servers included |
| [half-b.json](half-b.json) | Half b's profile: cards 5, 7 and 8 at use.json's values, cards 0 to 4 and 6 at 0 |
| [b/](b/) | Half b's tables, the same six files as half a's |
| [full90/](full90/) | The whole profile for 90 minutes (R228): the same six tables, plus `ornith_differs.csv`, round 137 against stock prompt by prompt |
| [mem-down.json](mem-down.json) | R231's profile: use.json with every card's memory one step (100 MT/s) lower, cores kept (`run/mem_down.py`, made on the rig; md5 `67d8cb5c5685c74d7c461696eea42820`) |
| [mem120/](mem120/) | Every memory one step lower, 120 minutes planned (R231): the same six tables, plus `ornith_differs.csv`, round 22 against stock prompt by prompt |
| [cores.json](cores.json) | R235's profile: use.json with every memory offset at 0, cores kept (`run/mem_down.py --steps 13`, made on the rig; md5 `c12d1573989cab8ab8b3f0ae7b85e48f`) |
| [cores120/](cores120/) | The cores alone, 120 minutes planned (R235): the same six tables, plus `ornith_differs.csv`, round 5 against stock prompt by prompt |
| [run/half_profile.py](run/half_profile.py) | Makes a half's profile from use.json and runs the kit's own profile check on it (the one `apply` runs) |
| [run/half_check.sh](run/half_check.sh) | One half (or the whole profile) on a fresh boot: start check, trip test, watchdog, `apply`, Ornith for `MINUTES` (30; 90 for `full90`) against the stored stock answer, then every offset to 0 |
| [run/halves.py](run/halves.py), [run/tables.sh](run/tables.sh) | Build the tables from the folders copied off the rig |
| [mem-alone.json](mem-alone.json), [cores-down1.json](cores-down1.json), [cores-down2.json](cores-down2.json), [mem-core-down1.json](mem-core-down1.json) | The next runs' profiles from use.json (Q1760): the memory alone, the cores one and two steps lower, memory and core each one step lower ([above](#the-next-profiles-q1760-2026-11-24)) |
| [run/mem_down.py](run/mem_down.py) | use.json with every card's memory (`--steps`, 100 MT/s each; Q1640's option (c)) and/or core (`--core-steps`, 2 × 12.66 MHz each; Q1760) lower, never below 0, `all` for 0; check it with `half_profile.py --check` |
| [run/MD5SUMS-a](run/MD5SUMS-a), [run/MD5SUMS-b](run/MD5SUMS-b), [run/MD5SUMS-full90](run/MD5SUMS-full90), [run/MD5SUMS-mem120](run/MD5SUMS-mem120), [run/MD5SUMS-cores120](run/MD5SUMS-cores120) | The md5 sums of the files on the rig for each run, checked there before the start |

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
copies into a fresh OUT and checks against `REF_HASH`, so every half is compared with the same stock answer. Half b
repeated these commands on its own fresh boot with `--stock 0 1 2 3 4 6 --tag b`, `r226-half-b.json`
([run/MD5SUMS-b](run/MD5SUMS-b)), `WORK=/var/tmp/r226b` and `OUT=/var/tmp/r226b-model`. The whole profile for 90
minutes (`full90`) repeated them on its own fresh boot with R223's use.json itself as `PROFILE` (the rig's copy from
step 4, the same md5; [run/MD5SUMS-full90](run/MD5SUMS-full90)), no half profile made, `MINUTES=90`,
`WORK=/var/tmp/q1640a` and `OUT=/var/tmp/q1640a-model`. Every memory one step lower (`mem120`) made its profile on
the rig with option (c)'s commands below and repeated them on its own fresh boot with `PROFILE=$K/q1640-c.json`
([run/MD5SUMS-mem120](run/MD5SUMS-mem120)), `MINUTES=120`, `WORK=/var/tmp/q1640c` and `OUT=/var/tmp/q1640c-model`.
The cores alone (`cores120`) made its profile on the rig the same way with `--steps 13` (below) and repeated them on
its own fresh boot with `PROFILE=$K/q1640-cores.json` ([run/MD5SUMS-cores120](run/MD5SUMS-cores120)), `MINUTES=120`,
`WORK=/var/tmp/r235` and `OUT=/var/tmp/r235-model`.
The tables, on any machine, from the two folders copied off the rig:

```bash
bash run/tables.sh a <copy of /var/tmp/r226a> <copy of /var/tmp/r226a-model> .
bash run/tables.sh b <copy of /var/tmp/r226b> <copy of /var/tmp/r226b-model> .
bash run/tables.sh full90 <copy of /var/tmp/q1640a> <copy of /var/tmp/q1640a-model> .
bash run/tables.sh mem120 <copy of /var/tmp/q1640c> <copy of /var/tmp/q1640c-model> .
bash run/tables.sh cores120 <copy of /var/tmp/r235> <copy of /var/tmp/r235-model> .
```

Option (c)'s profile, every card's memory one step lower with the core kept (`67d8cb5c5685c74d7c461696eea42820` from
R223's use.json), and the kit's check of it:

```bash
python3 -I $D/run/mem_down.py $CHECKOUT/docs/data/2026-11-20_r223_core_soak_rig1/use.json $K/q1640-c.json
cd $K && . ./env.sh && python3 half_profile.py climb.py q1640-c.json --check
```

The cores alone, every memory offset at 0 with the cores kept (`c12d1573989cab8ab8b3f0ae7b85e48f` from R223's
use.json; 13 steps take the largest memory offset, +1,300 MT/s, to 0):

```bash
python3 -I $D/run/mem_down.py $CHECKOUT/docs/data/2026-11-20_r223_core_soak_rig1/use.json $K/q1640-cores.json --steps 13
cd $K && . ./env.sh && python3 half_profile.py climb.py q1640-cores.json --check
```

The next profiles (Q1760): `mem_down.py` finds the kit's `climb.py` beside itself (copied into `$K`) or in the
checkout, and gives the md5s in [the table above](#the-next-profiles-q1760-2026-11-24):

```bash
U=$CHECKOUT/docs/data/2026-11-20_r223_core_soak_rig1/use.json
python3 -I $D/run/mem_down.py $U $K/q1795-mem.json --steps 0 --core-steps all      # the memory alone, 52be6a07…
python3 -I $D/run/mem_down.py $U $K/q1760-cores1.json --steps all --core-steps 1   # cores one step lower, 37df8e54…
python3 -I $D/run/mem_down.py $U $K/q1760-cores2.json --steps all --core-steps 2   # cores two steps lower, f3eb0271…
python3 -I $D/run/mem_down.py $U $K/q1640-c2.json --core-steps 1                   # (c'), 8ba42479…
cd $K && . ./env.sh && for f in q1795-mem q1760-cores1 q1760-cores2 q1640-c2; do python3 half_profile.py climb.py $f.json --check; done
```

## Provenance and licence

Measured on the first rig on 2026-11-20 (half a: 16:40 to 17:31; half b: 17:49 to 18:41; the whole profile for 90 minutes: 19:04 to 20:40) and 2026-11-24 (every memory one step lower: 13:59 to 14:30; the cores alone: 18:38 to 18:59) with the engine builds named above and the kit as published
on this branch. The next runs' four profiles (Q1760) were made and checked on a laptop with the kit as published on
this branch, without a rig. The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; Ornith-1.5-122B
Pascal N2P GGUF and its licence are described in [release/ornith-v3](../../../release/ornith-v3/README.md). Neither
model is published here, only the hashes of their answers. The tables are this repository's licence.
