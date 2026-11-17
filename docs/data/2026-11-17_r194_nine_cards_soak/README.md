# R194: all nine P106-090 cards at memory +1,200 MT/s and the same core peak clock (2026-11-17)

**The answer in plain words.** No: not every card holds memory +1,200 MT/s with its core peaking at about
2,151 MHz. Each card got its own core offset, chosen from its own stock peak clock so that every card peaks at the
clock cards 0 and 1 held in R193. The cards were set and checked one at a time, going up only. Cards 0, 1, 2 and 3 passed every
check: all 516 kernel checks, and Flint's greedy output identical to stock down to every token's probabilities.
Card 2 had crashed in R193 at a 2,202 MHz peak and passed here at 2,138 MHz. **Card 4 failed:** two of its first 116
kernel checks gave wrong answers, and then the benchmark hung on the card. There was no crash report (no Xid), no
overheating (36 °C at most) and no power spike, so the watchdog had nothing to trip on. The hang was stopped by hand
after six minutes, and every offset of every card was set back to 0 and read back. The 30-minute soak of all nine
cards, `cuda_memtest` and the exact check did not run, and cards 5 to 8 were never offset. The hottest card all run was
card 1 at 51 °C. While each card ran its own check it drew 58.47 W (card 0), 60.41 W (card 1), 59.36 W (card 2) and
59.82 W (card 3) on average at the stock 75 W cap; the highest half-second reading was 92.79 W (card 3), never two
readings in a row above 80 W.

**What it means.** Card 4 fails with its core at the same 2,138 MHz peak that card 2 passed at, so this is not one
clock limit shared by every card. Either card 4's memory does not take +1,200 MT/s, or its core does not take
2,138 MHz; which one is not known. A setting for all nine cards at once does not hold at these offsets, so each card
needs its own climb ([Q1565](../../../quants/experiments.md)). Card 2 passing at a lower core clock with the same memory
offset that it crashed with in R193 fits the core having been card 2's limit there, but card 2 has had one sweep here,
not a soak.

**A failure the watchdog cannot see.** The watchdog trips on heat, power and the driver's error reports. Card 4 gave
wrong results and then stalled, with none of the three, so only the benchmark's own checks caught it, and a hang
stops those checks from ever finishing. A time limit on each check would catch a hang without a person
([Q1572](../../../quants/experiments.md)).

Registry rows: [Q1571](../../../quants/experiments.md) (this run), [Q1570](../../../quants/experiments.md) (core offsets
per card from the stock peak clock), [Q1565](../../../quants/experiments.md) (every card's own climb) and
[Q1572](../../../quants/experiments.md) (a time limit on each check). The run this one repeats with per-card offsets:
[R193](../2026-11-17_r193_nine_cards_soak/README.md); the one-card sitting, its kit and its checks:
[Q133 and Q135](../2026-11-17_q133_q135_clock_offsets/README.md).

## What was done

A nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off), freshly
rebooted after R193 and untouched since (Q1563: a step down in memory clock leaves a card slow until a reboot, so
offsets only go up after a fresh start). The owner sat at the rig's console for the whole run. The run is R193's with
one memory and one core offset per card ([run/r194.sh](run/r194.sh)):

- **The offsets, written down before any card was set.** Memory +1,200 MT/s on every card. Core offset = 300 −
  (the card's stock peak − 1,860 MHz), from R193's measured stock peaks:

  | Cards | Stock peak (R193) | Core offset | Steps of ~12.65 MHz | Expected peak (est.) |
  |---|---|---|---|---|
  | 0, 1 | 1,860 MHz | +300 MHz | 23 | 2,151 MHz |
  | 4, 6 | 1,885 MHz | +275 MHz | 21 | 2,151 MHz |
  | 8 | 1,898 MHz | +262 MHz | 20 | 2,151 MHz |
  | 2, 3, 5, 7 | 1,911 MHz | +249 MHz | 19 | 2,151 MHz |

  The driver moves the core clock in whole steps of about 12.65 MHz and drops the remainder of an offset: in the
  one-card sitting's climb, +25 MHz raised the peak one step (13 MHz), +300 raised it 23 steps (291 MHz). So these
  offsets give every card the same peak, the 2,151 MHz cards 0 and 1 held in R193. The script refuses anything above
  +1,200 MT/s or +300 MHz.
- **Start check:** no model or benchmark process, 0 MiB used, the stock 75 W power limit, both offsets 0 and P8 on all
  nine cards, no Xid this boot.
- **The watchdog on all nine cards** ([scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py), `--gpus all`):
  every 0.5 s it logs every card's clocks, power, temperature, clock event reasons and both offsets, and trips on any
  card above 70 °C, any card above 80 W for 5 s, or any new Xid line in the kernel log; a trip sets both offsets of
  every card to 0, reads them back and kills the model and benchmark processes. Before any offset all three trips were
  tested on all nine cards with lowered limits and stand-in processes (15 °C, a kernel-log test line, 1 W for 5 s);
  each tripped, read back 0 everywhere and killed the stand-ins.
- **No new stock measurements.** R193's stock references were reused: the greedy hash `200fe90f478c1933` (the same in
  the one-card sitting and in R193, across a reboot), R193's stock streaming reads and stock peak clocks, and its exact
  check's stock statistics block ([run/stock_kld.stats](run/stock_kld.stats)). The model server was loaded with every
  offset at 0 before step 1, as in R193.
- **Step 1, card by card, cards 0 to 8 in turn:** the card's memory and core offsets set (both going up from 0) and read
  back from a new process; the full `mmvq_pascal` sweep on that card (the decode kernels of Q8_0, Q5_0, Q4_1, Q5_K and
  IQ4_XS at Flint's shapes, each checked against a double-precision CPU reference, plus the streaming read); the greedy
  check against the served model, which by then ran with every card up to this one at its offsets; an Xid scan. The
  first failure sets every offset to 0 and ends the run.
- **Planned after step 1, not run:** a 30-minute soak with one `mmvq_pascal` loop per card and a greedy loop all at
  once; then the greedy check, `cuda_memtest` on every card (tests 0 to 5, 7, 8 and 10, then test 6) and the exact
  check. R193's stock base file for the exact check was not kept, so the script writes the base file at the offsets and
  reruns against it; it passes only if that rerun's statistics block equals R193's stock block digit for digit, which
  needs both runs to give stock's logits.

## Results

### Per card ([tables/cards.csv](tables/cards.csv))

| Card | Offsets read back (MT/s, MHz) | Peak core clock (MHz) | Kernel checks | Streaming read (GB/s), R193 stock → here | Greedy | Xid | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | +1,200, +300 | 2,151 | 516 of 516 | 179.15 → 204.76 (+14.3 %) | identical | 0 | **pass** |
| 1 | +1,200, +300 | 2,151 | 516 of 516 | 179.34 → 204.60 (+14.1 %) | identical | 0 | **pass** |
| 2 | +1,200, +249 | 2,138 | 516 of 516 | 179.24 → 204.76 (+14.2 %) | identical | 0 | **pass** |
| 3 | +1,200, +249 | 2,151 | 516 of 516 | 179.26 → 204.62 (+14.1 %) | identical | 0 | **pass** |
| 4 | +1,200, +275 | 2,138 | 116 run: 114 passed, **2 wrong**, then a hang | 179.47 → 204.6 (+14.0 %) | not run | 0 | **fail** |
| 5 | not set | — | — | — | — | — | not tested |
| 6 | not set | — | — | — | — | — | not tested |
| 7 | not set | — | — | — | — | — | not tested |
| 8 | not set | — | — | — | — | — | not tested |

The 516 checks of a full sweep are its 448 kernel checks plus 68 per-token totals, counted as in R193. Card 4's count
is of the kernel checks its benchmark printed before it hung (the benchmark writes its table only at the end). Its
streaming read is the benchmark's opening line. The peak core clock is the highest half-second reading during the
card's own check. Card 2 peaked one step below the expected 2,151 MHz; card 3, with the same stock peak and offset,
reached 2,151. Memory +1,200 MT/s showed as 4,608 MHz on every card set (stock 4,006 under load).

### Each tested card during its own check ([tables/own_check.csv](tables/own_check.csv))

| Card | Window | Mean power | Highest reading | Readings above 80 W (longest run) | Readings at the power cap | Peak / median core clock (MHz) | Peak / mean °C |
|---|---|---|---|---|---|---|---|
| 0 | 15:24:42–15:26:17 | 58.47 W | 84.05 W | 5 (1) | 64 of 190 | 2,151 / 1,923 | 46 / 40.1 |
| 1 | 15:26:17–15:27:49 | 60.41 W | 83.17 W | 5 (1) | 63 of 184 | 2,151 / 1,911 | 51 / 43.4 |
| 2 | 15:27:49–15:29:20 | 59.36 W | 83.34 W | 3 (1) | 67 of 182 | 2,138 / 1,911 | 43 / 37.2 |
| 3 | 15:29:20–15:30:53 | 59.82 W | 92.79 W | 5 (1) | 62 of 186 | 2,151 / 1,904 | 49 / 41.8 |
| 4 | 15:30:53–15:37:16 | 36.24 W | 75.92 W | 0 (0) | 4 of 766 | 2,138 / 2,138 | 36 / 34.5 |

Card 4's window is mostly the hang: from 15:31:05, the benchmark's last printed line, to 15:37:16 the card sat at P0
with its core at 2,138 MHz in every sample and drew 35.38 W on average, while its benchmark used a full CPU core. No
card reported a thermal or hardware slowdown at any time ([tables/phases.csv](tables/phases.csv)); cards 5 to 8, never
set, peaked at 28, 28, 30 and 27 °C.

### Card 4's failure ([tables/failed_checks.csv](tables/failed_checks.csv))

| Kernel check | Order in the sweep | Error against the CPU reference (× tolerance) |
|---|---|---|
| IQ4_XS, the attention query projection (`attn_q`), variant `p_r2c1` | 55th | 549.358 |
| Q8_0, the attention output projection (`attn_output`), variant `llama` | 115th | 727.012 |

Every passing check on cards 0 to 4 had an error of at most 0.016 of the tolerance. The variant after the second
failure passed; the one after that never finished. At 15:37 the run was stopped by hand: the step-1 driver and the benchmark killed, every offset of
every card set to 0 (read back 0 on all nine), the model server stopped. The kernel log shows no Xid this boot, and
the watchdog logged no trip. The card answered NVML throughout and idled in P8 with 0 MiB used at the end.

### What the run leaves

- **Cards 0 to 3 hold memory +1,200 MT/s with the core peaking at 2,138 or 2,151 MHz** for one full sweep and the
  greedy check each. None has had the soak, `cuda_memtest` or the exact check at these offsets.
- **Card 4 does not hold memory +1,200 MT/s with the core at a 2,138 MHz peak.** Which of the two is past its limit is
  not known; card 4's memory alone and its core alone were not tested.
- **Cards 5 to 8 were not tested.**
- **The offsets stepped down at the stop,** so the rig needs a reboot before its next timed run (Q1563).

## Files

| File | What it holds |
|---|---|
| [tables/cards.csv](tables/cards.csv) | One row per card: R193's stock peak and streaming read, the offsets set and read back, the expected and measured peak core clock, step 1's checks, streaming read, greedy, Xid and verdict; the soak and `cuda_memtest` columns stayed empty |
| [tables/own_check.csv](tables/own_check.csv) | Each tested card during its own check, from the watchdog's samples: power, readings above 80 W and at the cap, peak and median core clock, memory clock, temperature |
| [tables/phases.csv](tables/phases.csv) | Per phase (the server load at offset 0, step 1) and card: peak and mean temperature and power, readings and the longest run above 80 W, readings at the power cap or in a thermal or hardware slowdown, peak core and memory clocks, the offsets seen |
| [tables/failed_checks.csv](tables/failed_checks.csv) | Every failed kernel check: card, shape, type, variant, its bandwidth, its error against the CPU reference and how many checks ran before it |
| [tables/greedy.csv](tables/greedy.csv) | Every greedy check's hash against the reference |
| [tables/watchdog.txt](tables/watchdog.txt) | The watchdog's start and stop lines and its sample count |
| [run/r194.sh](run/r194.sh) | The run: start check, the offset plan, trip test, watchdog, the server at offset 0, step 1, soak and after-checks, every failure path setting all offsets to 0 |
| [run/tables.py](run/tables.py) | Builds `tables/` from the run's output folder, the watchdog's log and R193's `cards.csv` |
| [run/memoc_units.json](run/memoc_units.json) | NVML's memory offset unit, measured on card 0 in the one-card sitting (raw value = MT/s) |
| [run/stock_kld.stats](run/stock_kld.stats) | R193's exact-check stock block, the reference for the planned exact check |
| [../../../scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py) | The watchdog |
| [../2026-11-17_q133_q135_clock_offsets/run/greedy_check.py](../2026-11-17_q133_q135_clock_offsets/run/greedy_check.py) | The probability-hashed greedy check |
| [../../../bench/mem_offset/](../../../bench/mem_offset/README.md) | The kit: `nvml_memoc.py`, `exact_check.sh`, `mmvq_pascal` and `cuda_memtest` builds |

## Commands

On the rig, as root, with the [memory-offset kit](../../../bench/mem_offset/README.md) unpacked in `$KIT` and, in one
folder, `run/r194.sh`, `run/memoc_units.json`, `run/stock_kld.stats`, `scripts/nvml_watchdog.py` and the sitting's
`run/greedy_check.py`. `BIN` is the llama.cpp build's `bin` folder (Miner Llama v4.0's engine), `MODEL` Flint Pascal
Q5 GGUF, `WIKI` wikitext-2's `wiki.test.raw`; an `env.sh` next to `r194.sh` may set them. `MEMS` and `CORES` give one
offset per card (the defaults are this run's).

```bash
export WORK=/tmp/r194 KIT=/tmp/r194/mem_offset-kit-20261024 BIN=... MODEL=... WIKI=...
bash r194.sh state start_state.txt      # read-only: processes, cards, offsets, Xid, host memory
bash r194.sh start-check                # read-only: no process, 0 MiB, 75 W, every offset 0
bash r194.sh plan                       # read-only: each card's offsets against the caps, its expected peak
bash r194.sh trip-test                  # the watchdog's three trips on all cards, before any offset
bash r194.sh watchdog                   # detached, all cards: 70 C, 80 W for 5 s, any Xid
bash r194.sh serve                      # detached: the server on all cards, every offset 0
bash r194.sh cards                      # detached: step 1, card by card (here it stopped at card 4)
bash r194.sh soak 30                    # detached: step 2 (not run here)
bash r194.sh post                       # detached: greedy, cuda_memtest on every card, exact check (not run here)
bash r194.sh hottest 10                 # each card's peak temperature and power over the last 10 minutes
bash r194.sh reset-all                  # every offset of every card to 0, read back
bash r194.sh stop-watchdog; bash r194.sh state end_state.txt
python3 run/tables.py $WORK/out $WORK/wd.csv tables ../2026-11-17_r193_nine_cards_soak/tables/cards.csv
```

Each detached step writes `$WORK/out/<STEP>_DONE` with its exit status; a failure writes one line to
`$WORK/out/FAILED`. Here card 4's benchmark hung, so step 1 never reached its verdict; it was stopped by hand with
`kill` on the step-1 driver, `bash r194.sh reset-all`, `pkill mmvq_pascal` and `bash r194.sh server stop`.

**Provenance and licence.** Measured on the rig on 2026-11-17 with the kit's `mmvq_pascal` built for sm_61, Miner Llama
v4.0's engine (build of fork commit `eb312b7bc`) and Flint Pascal Q5 GGUF. The tables are this project's measurements
(the repository's licence).
