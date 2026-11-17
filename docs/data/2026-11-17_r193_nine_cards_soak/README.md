# R193: all nine P106-090 cards at memory +1,200 MT/s and core +300 MHz (2026-11-17)

**The answer in plain words.** No: not every card holds memory +1,200 MT/s and core +300 MHz together. The cards were
set and checked one at a time, going up only. Cards 0 and 1 passed every check: all 516 kernel checks, and Flint's
greedy output identical to stock down to every token's probabilities. Card 2 crashed about 40 seconds into its check.
The driver reported a memory-access fault (Xid 31) in the benchmark's kernels at 38 °C. The watchdog set every
offset on every card back to 0 at once and stopped the load. The run stopped there, as planned, so the 30-minute
soak of all nine cards, `cuda_memtest` and the exact check at the offsets did not run, and cards 3 to 8 were never
offset. The hottest card all run was card 1 at 51 °C. While each card ran its own check it drew 59.22 W (card 0), 59.03 W
(card 1) and 65.17 W (card 2) on average at the stock 75 W cap; the highest half-second reading was 88.95 W (card 0),
never two readings in a row above 80 W.

**Most likely the core, not proven.** At stock the nine cards boost to different peak clocks: 1,860 MHz (cards 0 and
1), 1,885 (cards 4 and 6), 1,898 (card 8) and 1,911 (cards 2, 3, 5 and 7). A core offset adds the same MHz to each,
so at +300 cards 0 and 1 peaked at 2,151 MHz and card 2 at 2,202 MHz. The fault was an address read from an invalid
page in a dense kernel whose addresses are computed on the card, which a core running past its limit can produce.
Card 2's memory at +1,200 MT/s streamed 204.7 GB/s and passed 285 kernel checks before the fault, but its memory was
not tested alone. Cards 3, 5 and 7 boost like card 2 and would also reach about 2,200 MHz at +300 (Q1570).

Registry rows: [Q1569](../../../quants/experiments.md) (this run), [Q1565](../../../quants/experiments.md) (every
card's own offsets) and [Q1570](../../../quants/experiments.md) (core offsets set per card from its stock peak clock).
The one-card sitting this run follows, its kit and its checks: [Q133 and Q135's data](../2026-11-17_q133_q135_clock_offsets/README.md).

## What was done

A nine-card P106-090 rig (VBIOS 86.06.5B.00.0C, driver 580.178.04, no X server, persistence mode off), freshly
rebooted (Q1563: a step down in memory clock leaves a card slow until a reboot, so offsets only go up after a fresh
start). The owner sat at the rig's console for the whole run. The checks are the one-card sitting's, applied to every
card ([run/r193.sh](run/r193.sh)):

- **Start check:** no model or benchmark process, 0 MiB used, every memory and core offset 0, the stock 75 W power
  limit, 20 to 27 °C and P8 on all nine cards, no Xid this boot.
- **The watchdog on all nine cards** ([scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py) with `--gpus all`,
  new for this run): every 0.5 s it logs every card's clocks, power, temperature, clock event reasons and both offsets.
  It trips on any card above 70 °C, any card above 80 W for 5 s, or any new Xid line in the kernel log. A trip sets
  both offsets of every card to 0, reads them back and kills the model and benchmark processes. Before any offset all
  three trips were tested with lowered limits and stand-in processes: 15 °C, a kernel-log line with a test pattern,
  and 1 W for 5 s. Each one tripped on all nine cards, read back 0 everywhere and killed the stand-ins.
- **Stock references, every offset at 0:**
  - the exact one-token check's base file (Flint Pascal Q5 GGUF over wikitext-2, 4,096 one-token steps on all nine
    cards), then a stock rerun against it, whose statistics block is the reference. It came out identical, digit for
    digit, to the one-card sitting's stock rerun of the same morning, before the reboot;
  - Flint Pascal Q5 GGUF served on all nine cards and the probability-hashed greedy check (two fixed prompts, 256
    tokens each, every token's top-10 probabilities hashed; Q1564): hash `200fe90f478c1933`, the sitting's stock hash;
  - each card's streaming read (`mmvq_pascal`, 256 MB, plain loads).
- **Step 1, card by card, cards 0 to 8 in turn:** memory +1,200 MT/s and core +300 MHz set on the card (both going up
  from 0) and read back from a new process; then the full `mmvq_pascal` sweep on that card (the decode kernels of Q8_0,
  Q5_0, Q4_1, Q5_K and IQ4_XS at Flint's shapes, 516 variants and shapes, each checked against a double-precision CPU
  reference, plus the streaming read); then the greedy check against the served model, which by then ran with every
  card up to this one at the offsets; then an Xid scan. The first failure sets every offset to 0 and ends the run.
- **Planned after step 1, not run:** a 30-minute soak with one `mmvq_pascal` loop per card and a greedy loop all at
  once; then the greedy check, `cuda_memtest` on every card (tests 0 to 5, 7, 8 and 10, then test 6) and the exact
  check at the offsets.

## Results

### Per card ([tables/cards.csv](tables/cards.csv))

| Card | Stock streaming read (GB/s) | Peak core clock at stock (MHz) | Offsets read back (MT/s, MHz) | Kernel checks | Streaming read at the offsets (GB/s) | Greedy | Xid | Verdict |
|---|---|---|---|---|---|---|---|---|
| 0 | 179.15 | 1,860 | +1,200, +300 | 516 of 516 | 204.70 (+14.3 %) | identical | 0 | **pass** |
| 1 | 179.34 | 1,860 | +1,200, +300 | 516 of 516 | 204.80 (+14.2 %) | identical | 0 | **pass** |
| 2 | 179.24 | 1,911 | +1,200, +300 | 285 passed, then the fault | 204.7 | not run (the server was stopped by the trip) | 1 (Xid 31) | **fail** |
| 3 | 179.26 | 1,911 | not set | — | — | — | — | not tested |
| 4 | 179.47 | 1,885 | not set | — | — | — | — | not tested |
| 5 | 179.31 | 1,911 | not set | — | — | — | — | not tested |
| 6 | 179.15 | 1,885 | not set | — | — | — | — | not tested |
| 7 | 179.40 | 1,911 | not set | — | — | — | — | not tested |
| 8 | 179.21 | 1,898 | not set | — | — | — | — | not tested |

The peak core clock at stock is the highest half-second reading during the stock references (exact check, server
load, greedy check and the streaming reads; [tables/phases.csv](tables/phases.csv)). Card 2's streaming read at the
offsets comes from its benchmark log, which printed it before the fault; the benchmark writes its table only at the
end.

### Each tested card during its own check (the watchdog's half-second samples)

| Card | Window | Mean power | Highest reading | Readings above 80 W (longest run) | Readings at the power cap | Peak / median core clock (MHz) | Memory clock (MHz) | Peak / mean °C |
|---|---|---|---|---|---|---|---|---|
| 0 | 14:47:27–14:48:59 | 59.22 W | 88.95 W | 4 (1) | 63 of 184 | 2,151 / 1,923 | 4,608 | 46 / 40.3 |
| 1 | 14:48:59–14:50:31 | 59.03 W | 83.66 W | 2 (1) | 59 of 184 | 2,151 / 1,866 | 4,608 | 51 / 43.7 |
| 2 | 14:50:31–14:51:09 | 65.17 W | 84.53 W | 4 (1) | 22 of 77 | 2,202 / 2,050 | 4,608 | 39 / 35.2 |

Memory +1,200 MT/s showed as 4,608 MHz on every card set (stock 4,006 under load). No card reported a thermal or
hardware slowdown at any time; the cards that were not under test idled at 23 to 30 °C.

### Card 2's fault

At 14:51:08 the kernel logged Xid 31 for card 2, raised by the benchmark: an MMU fault, a read through an invalid page
directory entry by the graphics engine. The benchmark stopped with "an illegal memory access was encountered" during
the Q8_0 kernels of the shared expert's fused gate and up projections, after 285 of its checks had passed. The
watchdog tripped at 14:51:13.9: every offset of every card read back 0 and the model server was killed. The driver's
own recovery from the fault paused the watchdog's samples for 4.5 s (14:51:09.4 to 14:51:13.9), so it reacted about
6 s after the kernel line. The card itself kept working: it answered NVML at once afterwards and idled in P8 at the
end with 0 MiB used.

### What the run leaves

- **Cards 0 and 1 hold +1,200 MT/s and +300 MHz** for one full benchmark sweep and the greedy check each. Neither has
  had the soak, `cuda_memtest` or the exact check at these offsets.
- **Card 2 does not hold +1,200 MT/s and +300 MHz together.** Which of the two is past its limit is not known.
- **The stock references reproduce across a reboot**: the exact check's statistics block and the greedy hash were
  identical to the morning's, before the owner's reboot. A later run can reuse the sitting's stock hash.
- **A core offset is not one clock across cards.** The same +300 MHz gives each card its own stock peak plus 300 MHz,
  51 MHz apart between cards 0 and 2. Setting each card's core offset from its own stock peak, so that every card
  peaks at a clock already proven, is Q1570.
- **The offsets stepped down at the trip**, so the rig needs a reboot before its next timed run (Q1563).

## Files

| File | What it holds |
|---|---|
| [tables/cards.csv](tables/cards.csv) | One row per card: stock streaming read, step 1 (offsets read back, kernel checks, streaming read, greedy, Xid, verdict); the soak and `cuda_memtest` columns stayed empty |
| [tables/phases.csv](tables/phases.csv) | Per phase and card, from the watchdog's samples: peak and mean temperature, peak and mean power, readings and the longest run above 80 W, readings at the power cap or in a thermal or hardware slowdown, peak core and memory clocks, the offsets seen |
| [tables/greedy.csv](tables/greedy.csv) | Every greedy check's hash against the reference |
| [tables/exact_check.txt](tables/exact_check.txt) | The exact check's stock rerun: the reference statistics block |
| [tables/watchdog.txt](tables/watchdog.txt) | The watchdog's start, trip and stop lines and its sample count |
| [run/r193.sh](run/r193.sh) | The run: state record, trip test, watchdog, stock references, step 1, soak and after-checks, every failure path setting all offsets to 0 |
| [run/tables.py](run/tables.py) | Builds `tables/` from the run's output folder and the watchdog's log |
| [run/memoc_units.json](run/memoc_units.json) | NVML's memory offset unit, measured on card 0 in the one-card sitting (raw value = MT/s), used to set every card |
| [../../../scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py) | The watchdog (`--gpus` added for this run) |
| [../2026-11-17_q133_q135_clock_offsets/run/greedy_check.py](../2026-11-17_q133_q135_clock_offsets/run/greedy_check.py) | The probability-hashed greedy check |
| [../../../bench/mem_offset/](../../../bench/mem_offset/README.md) | The kit: `nvml_memoc.py`, `exact_check.sh`, `mmvq_pascal` and `cuda_memtest` builds |

## Commands

On the rig, as root, with the [memory-offset kit](../../../bench/mem_offset/README.md) unpacked in `$KIT` and, in one
folder, `run/r193.sh`, `run/memoc_units.json`, `scripts/nvml_watchdog.py` and the sitting's `run/greedy_check.py`.
`BIN` is the llama.cpp build's `bin` folder (Miner Llama v4.0's engine), `MODEL` Flint Pascal Q5 GGUF, `WIKI`
wikitext-2's `wiki.test.raw`; an `env.sh` next to `r193.sh` may set them.

```bash
export WORK=/tmp/r193 KIT=/tmp/r193/mem_offset-kit-20261024 BIN=... MODEL=... WIKI=...
bash r193.sh state start_state.txt      # read-only: processes, cards, offsets, Xid, host memory
bash r193.sh trip-test                  # the watchdog's three trips on all cards, before any offset
bash r193.sh watchdog                   # detached, all cards: 70 C, 80 W for 5 s, any Xid
bash r193.sh pre                        # detached: exact check base and stock rerun, server, stock greedy, streaming
bash r193.sh cards                      # detached: step 1, card by card (here it stopped at card 2)
bash r193.sh soak 30                    # detached: step 2 (not run here)
bash r193.sh post                       # detached: greedy, cuda_memtest on every card, exact check (not run here)
bash r193.sh hottest 10                 # each card's peak temperature and power over the last 10 minutes
bash r193.sh reset-all                  # every offset of every card to 0, read back
bash r193.sh stop-watchdog; bash r193.sh state end_state.txt
python3 run/tables.py $WORK/out $WORK/wd.csv tables     # on any machine, from the copied output
```

Each detached step writes `$WORK/out/<STEP>_DONE` with its exit status; a failure writes one line to
`$WORK/out/FAILED`.

**Provenance and licence.** Measured on the rig on 2026-11-17 with the kit's `mmvq_pascal` and `cuda_memtest` built for
sm_61, Miner Llama v4.0's engine (build of fork commit `eb312b7bc`) and Flint Pascal Q5 GGUF; wikitext-2 test set for
the exact check. The tables are this project's measurements (the repository's licence).
