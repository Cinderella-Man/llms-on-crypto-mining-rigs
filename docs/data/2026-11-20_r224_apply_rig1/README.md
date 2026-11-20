# R224: the first rig's profile on the served model, step 5 of its own profile (2026-11-20)

**The answer in plain words.** On a fresh boot of the first nine-card P106-090 rig, the kit set the profile that steps
1 to 4 gave this rig ([R223](../2026-11-20_r223_core_soak_rig1/README.md)'s [use.json](../2026-11-20_r223_core_soak_rig1/use.json))
and checked it on the model the rig serves, Ornith-1.5-122B Pascal N2P GGUF, which fills all nine cards. Every card
took its offsets, read them back as set and gave its stock answer on its own. Ornith then answered the kit's two fixed
prompts with the offsets: **21 rounds were the stock answer token for token, and round 22, 11 minutes in, was not.**
One prompt's reply parted from stock at token 184 of 256; the other was bit-identical. The script set every offset
back to 0 at once and stopped the watchdog: **the step failed, and the rig ran its later work at stock.** Ornith
spreads its layers over all nine cards, so one differing answer cannot say which card caused it. While the offsets
held, Ornith generated 6.85 % faster than at stock on the same boot (17.78 against 16.64 tok/s on both prompts). The
watchdog never tripped and the boot logged no GPU error.

**What it means.**

- **The profile is not proven for daily use on this rig.** Each card's values had passed 30 minutes or more of the
  kit's per-card check in steps 2 and 4, two steps under every failure seen there. The served model found an error
  within 11 minutes. Thirty clean minutes show only that a card errs less often than about once in 10 minutes
  ([the kit](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check));
  an error after 11 minutes fits that. The kit's check model fills about 2 GB of each card; Ornith filled 5,009 to 5,635 MiB.
- **The second rig's profile passed the same check:** 58 of 58 rounds in 30 minutes on the same engine, preset and
  script ([R209](../2026-11-19_r209_oc_v5/README.md#the-full-profiles-apply-and-orniths-whole-memory-check-2026-11-19)),
  and both rounds on each of two later boots.
- **The error spoiled one answer.** Prompt 2 of the same round was bit-identical to stock, as in every failed check of
  steps 2 and 4 on this rig, where one prompt stayed bit-identical. The check ends at the first differing round, so
  this run does not show whether the next round would have been clean again.
- **What the next boot does is the owner's choice** ([Q1640](../../../quants/experiments.md)): step 5 again with the
  same profile, the profile on half the cards at a time to find the card, or every card one step lower.

## What was done

The first nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, persistence mode off), freshly booted at
about 14:54 with nothing run on it before: every card at 405 MHz memory and 0 MiB used. At 14:56 the staged files'
md5 sums were checked on the rig ([run/MD5SUMS-step5](run/MD5SUMS-step5)) and the kit's start check passed: every
offset 0, no load process, no GPU Xid this boot, no offset lowered since the boot, 3,179 MiB of host memory available.

R209's [apply_check.sh](../2026-11-19_r209_oc_v5/run/apply_check.sh) ran the boot detached in the kit's folder
(14:57:32 to 15:39:08), with the v5 engine (`llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483`) and Ornith's v5
settings ([release/ornith-v5](../../../release/ornith-v5/settings.env), copied into the kit as `ornith-v5.env`):

1. **The start check, then Ornith at stock** (`model_check.sh` with `ROUNDS=2 MINUTES=0`): loaded in 673 s, two
   rounds at 15:09:32 and 15:10:06, both the hash `adf47784f9b02d4b`. Each round is the kit's greedy check: two
   prompts of 256 tokens at temperature 0, every token's top-10 probabilities hashed.
2. **The start check again, the watchdog** (15:10:17: a sample every 0.5 s; a card trips above 70 °C, above 80 W held
   for 5 s, or on its own Xid) **and `apply`** of the profile (15:10:21 to 15:15:44): a check server on every card
   with the kit's check model (Granite 3.1 3B-A800M Instruct, Q4_K_M, about 2 GB of each card) on the kit's engine
   (`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`), two stock rounds, every card's offsets set
   15:14:46 to 15:14:52 and each read back as set, then one round with the offsets: every card PASS at 15:15:37.
3. **Ornith with the offsets**, the same engine, settings and output folder as step 1, for up to 30 minutes, each
   round compared with stock's round 1: loaded in 675 s, rounds from 15:27:42; round 22 (15:38:50) differed and the
   check ended (22 rounds in 709 s).
4. **The reset:** every card's offsets to 0 (15:38:57 to 15:39:06), then the watchdog stopped (15:39:08).

## Results

### Ornith's whole-memory check ([model_check.md](model_check.md); every round in [model_rounds.tsv](model_rounds.tsv))

| | stock | with the offsets | change |
|---|---|---|---|
| load, s | 673 | 675 | |
| host memory available when loaded, MiB | 1,496 | 1,502 | |
| rounds; the stock hash | 2; `adf47784f9b02d4b` in both | 22; `adf47784f9b02d4b` in rounds 1 to 21, **`07c5e62d210d9220` in round 22** | |
| generation tok/s in round 1 (prompts 1, 2) | 15.02, 15.76 | 16.04, 16.72 | +6.79 %, +6.09 % |
| generation tok/s in round 2 (prompts 1, 2) | 16.64, 16.64 | 17.79, 17.78 | +6.91 %, +6.85 % |
| generation tok/s, median of rounds 2 to 21 (prompts 1, 2) | 16.64, 16.64 (round 2) | 17.78, 17.78 (20 rounds) | **+6.85 %, +6.85 %** |
| prompt processing tok/s in round 2 (prompts 1, 2) | 26.86, 29.67 | 28.71, 31.67 | +6.89 %, +6.74 % |

- **Round 1 is each load's first request** and runs slower in both checks. From round 2 on the check barely moves:
  of the 20 clean rounds with the offsets, prompt 1 read 17.78 tok/s in 11 and 17.79 in 9; prompt 2 read 17.78 in 17,
  17.77 in 2 and 17.65 in 1. So stock's round 2 is the reference. Round 22 read 17.25 and 17.78.
- **Both loads used the same memory:** 5,603 5,635 5,549 5,635 5,243 5,351 5,513 5,601 5,009 MiB on cards 0 to 8, the
  same as on the second rig.
- **Stock here is the second rig's stock:** 16.64 tok/s in round 2 against 16.66 there (R209), with the same engine,
  settings and hash. The gain with the offsets is smaller than the second rig's 8.04 %; on this rig card 6 runs at
  stock and cards 0 and 8 at +900 MT/s.

### The differing round ([ornith_differs.csv](ornith_differs.csv))

| Prompt | Against stock's round 1 |
|---|---|
| 1 | the text parts at token 184 of 256 (72 tokens differ); the probabilities differ from token 167; before the text parts the chosen token's log-probability moved by at most 0.125 |
| 2 | bit-identical |

### `apply`, card by card ([apply.md](apply.md); the kit's own table [apply_steps.csv](apply_steps.csv))

| card | memory MT/s | core MHz | streaming read at stock, GB/s | with the offsets | change | peak core MHz at stock | with the offsets | greedy check | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
| 0 | +900 | +218 | 179.20 | 197.68 | +10.31 % | 1,860 | 2,075 | same as stock | PASS |
| 1 | +1,000 | +256 | 179.23 | 199.44 | +11.28 % | 1,860 | 2,113 | same as stock | PASS |
| 2 | +1,100 | +256 | 179.38 | 202.12 | +12.68 % | 1,911 | 2,151 | same as stock | PASS |
| 3 | +1,200 | +206 | 179.08 | 204.94 | +14.44 % | 1,911 | 2,100 | same as stock | PASS |
| 4 | +1,100 | +206 | 179.15 | 201.98 | +12.74 % | 1,885 | 2,088 | same as stock | PASS |
| 5 | +1,300 | +231 | 179.43 | 206.80 | +15.25 % | 1,911 | 2,138 | same as stock | PASS |
| 6 | 0 | 0 | 179.24 | 179.37 | +0.07 % | 1,885 | 1,885 | same as stock | PASS |
| 7 | +1,100 | +282 | 179.27 | 202.12 | +12.75 % | 1,911 | 2,189 | same as stock | PASS |
| 8 | +900 | +231 | 179.11 | 197.47 | +10.25 % | 1,885 | 2,113 | same as stock | PASS |

Every card's offsets are the profile's, read back as set; every card's check model gave its stock hash
(`60ee0f178cfd67d5` on all nine) and passed the 31 kernel checks at stock and with the offsets.

### Power and clocks ([model_cards.md](model_cards.md), the watchdog's 1,420 samples a card over Ornith's rounds with the offsets; [watchdog.md](watchdog.md), its 3,461 samples a card from 15:10:15 to 15:39:07)

| card | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SM clock median over the rounds, MHz | 2,075 | 2,100 | 2,151 | 2,100 | 2,075 | 2,138 | 1,885 | 2,189 | 2,113 |
| memory clock median over the rounds, MHz | 4,455 | 4,498 | 4,551 | 4,608 | 4,551 | 4,657 | 4,006 | 4,551 | 4,455 |
| highest W over the rounds | 76.69 | 85.75 | 84.03 | 77.17 | 78.09 | 71.19 | 75.03 | 80.72 | 60.16 |
| samples above 80 W over the rounds, of 1,420 | 0 | 4 | 5 | 0 | 0 | 0 | 0 | 1 | 0 |

Power went above 80 W only in single samples, never two in a row, against the watchdog's 80 W held for 5 s; the
highest was 85.75 W on card 1. No sample tripped, and the boot logged no GPU Xid.

### The end

At 16:00:36 the kit's `state` read every offset 0 on all nine cards (and `nvidia-smi` 1,911 / 4,004 MHz maximum), P8,
0 MiB, no load process, the watchdog stopped, no GPU Xid this boot, eight offsets lowered since the boot (card 6 had
none to lower), 3,172 MiB of host memory available. Another step needs a fresh boot.

## Files

| File | What it holds |
|---|---|
| [model_check.md](model_check.md) | Ornith at stock and with the offsets: load, host memory, rounds and hashes, tok/s in rounds 1 and 2 and the median from round 2 on (all 21 rounds, the differing one included) |
| [model_rounds.tsv](model_rounds.tsv) | Every round: tag, time, hash, tokens, against stock's round 1, prompt and generation tok/s of both prompts |
| [ornith_differs.csv](ornith_differs.csv) | The differing round against stock's round 1, by prompt (R217's [greedy_diff.py](../2026-11-19_r217_climb_rig1/run/greedy_diff.py)) |
| [apply.md](apply.md), [apply_steps.csv](apply_steps.csv) | `apply`'s stock rounds and its round with the offsets, card by card: offsets, streaming reads, peaks, the greedy and kernel checks |
| [model_cards.md](model_cards.md) | Each card over Ornith's rounds with the offsets: power, samples above 80 W, SM and memory clocks, offsets (the watchdog started after the stock check) |
| [watchdog.md](watchdog.md) | Each card over all the watchdog's samples, `apply`'s check servers included: power and samples above 80 W |
| [run/MD5SUMS-step5](run/MD5SUMS-step5) | The md5 sums of the script, the settings and the profile staged in the kit's folder, checked on the rig before the start |
| [run/tables.sh](run/tables.sh) | Builds these tables from the two folders copied off the rig |

## Commands

On the rig, as root, in the kit's folder `$K` (the first rig's, as in R223, with `apply_check.sh` and Ornith's v5
settings copied in), on a fresh boot:

```bash
cp $CHECKOUT/docs/data/2026-11-19_r209_oc_v5/run/apply_check.sh $K/           # md5 8b8627e80ab312b00c540d8324ce92d0
cp $CHECKOUT/release/ornith-v5/settings.env $K/ornith-v5.env                    # md5 4b8dac36b6b9866090de3887c39b02ff
cd $K && md5sum apply_check.sh ornith-v5.env /var/tmp/oc-step4/soak/use.json      # use.json ae91f969c8ddc782698c8fd9d570734d
. ./env.sh && python3 climb.py state && python3 climb.py start-check
KIT=$K BIN=<the v5 engine's bin folder> MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  SETTINGS=$K/ornith-v5.env PROFILE=/var/tmp/oc-step4/soak/use.json WORK=/var/tmp/oc-step5 OUT=/var/tmp/oc-step5-model \
  setsid -f nohup bash $K/apply_check.sh > /var/tmp/oc-step5.console 2>&1 < /dev/null
cat /var/tmp/oc-step5/R209C_DONE      # a look, at most every 30 minutes; here "done profile check rc 1"
export WORK=/var/tmp/oc-step5 && python3 climb.py state      # every offset 0, the watchdog stopped
```

`apply_check.sh` is the version with the one-character fix of 2026-11-20 that lets bash parse its argument checks.
The tables, on any machine, from the two folders copied off the rig:

```bash
bash run/tables.sh <copy of /var/tmp/oc-step5> <copy of /var/tmp/oc-step5-model> .
```

## Provenance and licence

Measured on the rig on 2026-11-20, 14:56 to 16:00, with the engine builds named above and the kit as published on this
branch. The check model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0) quantized to Q4_K_M; Ornith-1.5-122B
Pascal N2P GGUF and its licence are described in [release/ornith-v3](../../../release/ornith-v3/README.md). Neither
model is published here, only the hashes of their answers. The tables are this repository's licence.
