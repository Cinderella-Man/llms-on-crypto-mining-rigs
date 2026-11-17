# Q133 and Q135: memory and core clock offsets on one P106-090 (2026-11-17)

**The answer in plain words.** A P106-090 takes a memory and a core clock offset at runtime through NVML, with no X
server and nothing written to the card or the system. On the card tested, the memory ran clean up to +1,200 MT/s
(+15 %, 4,608 MHz in `nvidia-smi`). At +1,400 it went wrong **without any error message**: every kernel check passed,
but the model's token probabilities changed (its text did not). The core ran clean up to +300 MHz, the highest step
tried (2,138 MHz peak). The settings kept, one step below each highest clean step, are **memory +1,000 MT/s and core
+275 MHz**. They passed a 30-minute soak with every check, `cuda_memtest` and the exact one-token check
(statistics identical to stock, digit for digit). The card never went above 53 °C. On the bench, at the stock 75 W
power cap, the two settings together cut the matrix-vector time of a Flint token on this card by 6.2 % to 6.8 %
against the same card at stock, depending on the weight type. What that buys in served speed needs every card set (the
tuned column, Q1520). One surprise: after the memory offset went down from +800 to 0, the card ran about 8 % slower at
stock with every offset at 0, until the owner rebooted the rig.

Registry rows: [Q133](../../../quants/experiments.md) (memory clock offset), [Q135](../../../quants/experiments.md)
(core clock offset), with Q1517 answered below. Background and the protocol: [hardware tuning
note](../../research/notes/hardware_tuning.md), sections 1, 3 and 5, and the [memory-offset
kit](../../../bench/mem_offset/README.md).

## What was done

One card (card 0, NVML index 0, the first in PCI bus order) of a nine-card P106-090 rig, VBIOS 86.06.5B.00.0C, driver
580.178.04, no X server (the rig boots to a text console), persistence mode off. The owner sat at the rig's console for
the whole sitting, because a hung card needs a reboot and this rig's disk passphrase is typed at its keyboard.

**First, read-only:** NVML reports an offset range on all nine cards: memory −2,000 to +2,000, core −200 to +1,200.
A +0 set of each on card 0 was accepted with no X server and read back 0. Application clocks are not supported, and
the only memory clocks the card lists are 4,004 and 405 MHz, so the offsets are the only route.

**NVML's memory unit is MT/s** (transfer-rate units, as in `nvidia-settings`): a raw +100 moved the memory clock under
load from 4,006 to 4,050 MHz. Every memory offset below is in MT/s; +800 is +10 % of the stock 8,008 MT/s. Core
offsets are in MHz.

**Guards, set before any offset:**
- **A watchdog** ([scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py)), started detached before the first
  offset: every 0.5 s it samples all nine cards through NVML (clocks, power, power limit, temperature, memory, clock
  event reasons, card 0's offsets) and logs each sample; it reads the kernel log as it grows. It trips on any card above
  70 °C, card 0 above a power level for 5 s in a row, or any new NVIDIA Xid line. A trip sets both offsets of card 0
  to 0 at once, kills the bench and model processes and writes a marker file. Holding every card open, it also keeps
  power limits and offsets in place between processes with persistence mode off.
- **The trip was tested before any offset**, with lowered thresholds and stand-in processes: temperature (15 °C), card
  0's power (1 W) and a kernel-log line with a test pattern each tripped, set card 0's offsets to 0 (read back) and
  killed the stand-ins.
- **Power cap:** 60 W on all nine cards for the first part (the lowest the firmware allows; stock is 75 W), with the
  watchdog's power rule at 55 W. At offset 0 the bench alone held card 0 at the cap (54.3 to 62.5 W in half-second
  samples, the firmware's power-cap flag on, 34 °C), so the 55 W rule tripped at stock and killed it; the owner set
  the rule to 65 W. From 12:42 the owner restored the stock 75 W cap, with the rule at 80 W, so that the core offset is
  tested at the clocks it reaches in normal use. The 70 °C and Xid rules never changed.
- **Small steps, card 0 only:** memory +200 MT/s a step, core +25 MHz a step. The climbs were first capped at +800 MT/s
  (+12 % of stock) and +150 MHz; once both were clean the owner raised the caps to +2,000 MT/s (the driver's top) and
  +300 MHz. The first error, mismatch or trip ends a climb, and the setting kept is one step below the highest clean
  step (rare errors near the edge can take hours to show).

**Checks at every step** ([run/offset_climb.sh](run/offset_climb.sh)):
- the [mmvq_pascal](../../../bench/mmvq_pascal/README.md) sweep on card 0 alone: the decode kernels of Q8_0, Q5_0, Q4_1,
  Q5_K and IQ4_XS at Flint's shapes, 516 variants and shapes each checked against a double-precision CPU reference,
  then timed, plus a plain streaming read (GB/s);
- greedy output of Flint Pascal Q5 GGUF served on all nine cards, loaded once at stock: two fixed prompts, 256 generated
  tokens each, with the top-10 probabilities of every token and no prompt cache, hashed and compared with the hash at
  offset 0 ([run/greedy_check.py](run/greedy_check.py)); one changed logit anywhere changes the hash;
- the offset read back from a new process, an Xid scan of the kernel log, and the watchdog's peaks over the step.

**At the settings:** soaks that repeat the step's checks back to back; `cuda_memtest` on card 0 (6 GB, tests 0 to 8 and
10); the exact one-token check (4,096 one-token steps of Flint Pascal Q5 GGUF over wikitext-2 on the nine cards,
whose statistics block must equal a stock rerun's, digit for digit; [bench/mem_offset/exact_check.sh](../../../bench/mem_offset/exact_check.sh)).

## Results

### The memory climb

| Offset (MT/s) | Memory clock under load (MHz) | Streaming read (GB/s) | Bench checks | Greedy | Power cap | Peak °C | Verdict |
|---|---|---|---|---|---|---|---|
| 0 | 4,006 | 179.06 | 516 of 516 | reference | 60 W | 43 | clean |
| +200 | 4,104 | 183.86 | 516 of 516 | identical | 60 W | 47 | clean |
| +400 | 4,201 | 185.64 | 516 of 516 | identical | 60 W | 47 | clean |
| +600 | 4,303 | 190.25 | 516 of 516 | identical | 60 W | 47 | clean |
| +800 | 4,404 | 195.05 | 516 of 516 | identical | 60 W | 47 | clean; soak 25 min (15 rounds), all clean |
| +1,000 | 4,498 | 199.63 | 516 of 516 | identical | 60 W | 47 | clean |
| +1,200 | 4,608 | 204.46 | 516 of 516 | identical | 60 W | 47 | clean; soak 22 min (12 rounds), all clean; again at 75 W: 204.84, clean |
| +1,400 | 4,704 | 209.42 | 516 of 516 | **differs** | 75 W | 53 | **fails** |

**How +1,400 failed:** the first prompt's output was identical; in the second, the token probabilities differed on 50 of
the 256 generated tokens, from token 206 on, though every chosen token, and so the text, stayed the same
(log-probability of the top token −0.049855 against −0.04769 at the first difference). The bench's 516 checks passed
and the kernel logged no Xid. Back at offset 0 the same request gave the reference hash again, so the weights stored on
the card were intact. A check of the text alone would have passed +1,400.

### The core climb

| Offset (MHz) | Peak core clock (MHz) | Power cap | Bench checks | Greedy | Peak °C | Verdict |
|---|---|---|---|---|---|---|
| 0 | 1,847 | 60 W | 516 of 516 | identical | 47 | clean |
| +25 | 1,860 | 60 W | 516 of 516 | identical | 47 | clean |
| +50 | 1,885 | 60 W | 516 of 516 | identical | 46 | clean |
| +75 | 1,911 | 60 W | 516 of 516 | identical | 47 | clean |
| +100 | 1,936 | 60 W | 516 of 516 | identical | 47 | clean |
| +125 | 1,961 | 60 W | 516 of 516 | identical | 48 | clean |
| +150 | 1,987 | 60 W, then 75 W | 516 of 516 | identical | 47; 51 | clean at both caps |
| +175 | 2,012 | 75 W | 516 of 516 | identical | 52 | clean |
| +200 | 2,037 | 75 W | 516 of 516 | identical | 52 | clean |
| +225 | 2,062 | 75 W | 516 of 516 | identical | 52 | clean |
| +250 | 2,088 | 75 W | 516 of 516 | identical | 52 | clean |
| +275 | 2,113 | 75 W | 516 of 516 | identical | 52 | clean |
| +300 | 2,138 | 75 W | 516 of 516 | identical | 52 | clean |

The core climb was run with the memory at 0; the streaming read does not depend on the core clock (164.87 to
166.27 GB/s at every core step). No core limit was found up to +300 MHz.

### The settings: memory +1,000 MT/s and core +275 MHz together

- **Soak, 30 min at 75 W** (13:01 to 13:31): the step and 20 rounds, every round 516 of 516 checks, greedy identical,
  no Xid, 199.14 to 199.88 GB/s, peak core clock 2,113 MHz, 52 °C.
- **`cuda_memtest`** on card 0 (6.36 GB): tests 0 to 5, 7, 8 and 10 in 79 s, then test 6 alone in about 5 min, no
  errors, no Xid. A pass of tests 0 to 5, 7, 8 and 10 at the failed +1,400 MT/s (core 0) found no error either, so
  `cuda_memtest` would not have found this card's limit.
- **The exact one-token check:** pass. The statistics block equals the stock rerun's, digit for digit (mean KLD
  −0.000000, maximum 0.000050, same top token 100 %), no Xid ([tables/exact_check.txt](tables/exact_check.txt)).

### What the offsets do to the decode kernels (bench, card 0 alone)

The fastest variant's time for one Flint token of matrix-vector products, per weight type, in µs
([tables/kernels.csv](tables/kernels.csv)):

| Step | Cap | Q8_0 | Q5_0 | Q4_1 | Q5_K | IQ4_XS |
|---|---|---|---|---|---|---|
| stock (first run, 11:21) | 60 W | 17,596.0 | 12,123.3 | 10,418.1 | 13,802.4 | 13,233.9 |
| memory +800 | 60 W | 17,438.2 | 12,099.6 | 10,370.5 | 13,884.4 | 13,218.7 |
| memory +1,200 | 60 W | 17,277.9 | 12,010.7 | 10,182.1 | 13,870.4 | 13,380.4 |
| stock, card 0 slowed (12:42; see below) | 75 W | 18,947.9 | 12,817.9 | 11,080.4 | 13,921.7 | 12,782.6 |
| core +300, card 0 slowed | 75 W | 18,184.6 | 12,406.4 | 10,651.9 | 13,202.8 | 12,057.5 |
| memory +1,200 | 75 W | 16,001.3 | 11,070.4 | 9,548.6 | 12,587.3 | 12,049.9 |
| memory +1,000 and core +275 (soak round 20) | 75 W | 15,612.1 | 10,738.1 | 9,358.2 | 12,075.9 | 11,362.6 |
| stock, after the reboot (14:00) | 75 W | 16,750.1 | 11,450.0 | 9,992.2 | 12,920.0 | 12,166.4 |

- **At the 60 W cap the cap takes the memory gain:** the bench holds the card at its cap, and a faster memory leaves
  less power for the core, so +800 MT/s changed the kernels' time by −0.9 % to +0.6 %.
- **At 75 W, against the stock run after the reboot:** the settings −6.8 % (Q8_0), −6.2 % (Q5_0), −6.3 % (Q4_1),
  −6.5 % (Q5_K), −6.6 % (IQ4_XS); memory +1,200 alone −4.5 %, −3.3 %, −4.4 %, −2.6 %, −1.0 %. The memory clocks of
  +800 and up were not touched by the slowdown below, so these rows compare fairly.
- **The core alone at 75 W,** against the stock row of 12:42 with the memory in the same slowed state: +300 MHz −4.0 %,
  −3.2 %, −3.9 %, −5.2 %, −5.7 %.
- **These are kernel times on one card at full load, not served speed.** In serving, a busy card draws 35 to 40 W, below
  either cap, and only card 0 was changed: the served greedy generation on nine cards read 31.86 to 32.34 tok/s at every
  step. The served gain needs the offsets on every card (Q1520).

### Card 0 ran slower at the lower memory clocks after the first step down

After the memory went down for the first time (+800 to 0, at 11:59), card 0 streamed about 8 % slower at stock and at
+200 than it had before, while +800 and above were unchanged:

| Setting | Before (11:21 to 11:31) | After (11:59 to 12:43) |
|---|---|---|
| stock | 179.06 GB/s | 165.40; 165.27 and 165.55; after a drop to idle (P8) 165.51 and 165.39; 165.29 and 165.20; at 75 W 164.94 |
| +200 | 183.86 | 169.30 and 169.23 |
| +800 | 195.05 | 194.92 and 195.28 (and 194.88 to 195.32 through its soak) |

The stock kernels slowed with it (Q8_0 17,596.0 against 19,576.5 µs, both at 60 W). With every offset back at 0 at the
end of the sitting, card 0 still streamed 165.29 GB/s at stock while cards 1 and 2, never offset, gave 179.15 and
179.40, and its fastest attn_q kernel took 68.36 µs against 58.06 and 57.10.

| What was tried, offsets at 0 | Card 0's streaming read at stock |
|---|---|
| a drop to idle (P8) | 165.51 and 165.39 GB/s |
| the 75 W cap instead of 60 W | 164.94 |
| the driver letting go of the card (no process held it, persistence mode off), then a new process | 165.39 and 165.12 |
| an underclock to −400 MT/s (156.78 GB/s at 3,802 MHz), then back up to 0 | 165.17 and 165.08 |
| **a reboot of the rig** (by the owner at the console) | **179.15** (cards 1 and 2: 179.06 and 179.41; fastest attn_q 57.624 µs against 57.952 and 57.112) |

The first step down (+800 to 0) is the only event that matches: the climb only went up until then, and +100 to 0
(the unit measurement) stayed within the lower clocks. Something on the card kept a setting from the high clocks that
only a power-on setup clears; which setting is not known (Q1563). Correctness was never affected: every greedy check at
stock matched.

### The base clock between bursts (Q1517)

A core offset does not lift the 1,354 MHz base clock the cards fall to between bursts: in the half-second samples of a
step, card 0 read 1,354 MHz 31 times at +0 and 29 times at +150. Under load the offset lifts the whole curve (peaks
2,113 MHz at +275). So a core offset cannot replace Flint's keep-alive.

### Power and temperature

- Under the bench card 0 sits at its cap with the firmware's power-cap flag on. Half-second readings overshoot the cap
  briefly: up to 85.62 W at the 60 W cap and 93.00 W at 75 W. The longest run above the cap was 10 samples (5 s) at
  60 W; at 75 W, 54 samples read above 80 W, never more than 2 in a row. The watchdog's sustained rule (65 W, then 80 W,
  for 5 s) never tripped after the first stock run.
- Card 0 was the hottest card throughout: 48 °C at most at 60 W, 53 °C at 75 W. The other eight cards idled at 21 to
  39 °C.
- The watchdog took 21,140 samples of all nine cards ([tables/watchdog.txt](tables/watchdog.txt)).

## What changes

- **The offsets are a real lever and NVML is the route,** with no X server and nothing persistent. The tuned column
  (Q1520) can use memory +1,000 MT/s and core +275 MHz on this card.
- **Every other card needs its own climb.** Each card's memory has its own margin; this one failed at +1,400.
- **A check of text is not enough.** The memory's first error showed only in token probabilities; the greedy check
  with probabilities, or the exact one-token check, is the test.
- **Lowering a memory offset at runtime can leave a card slower until a reboot** (above). Apply offsets once, going
  up, after a reboot, and reboot after a sitting that stepped down.

## Files

| File | What it holds |
|---|---|
| [tables/steps.csv](tables/steps.csv) | One row per step and soak round: offset, read-back, bench checks, streaming GB/s, greedy, Xid, the watchdog's peaks (clocks, power, temperature), samples above the cap |
| [tables/kernels.csv](tables/kernels.csv) | Per step: the fastest variant's per-token time per weight type (µs) |
| [tables/watchdog.txt](tables/watchdog.txt) | The watchdog's start, trip and stop lines |
| [tables/exact_check.txt](tables/exact_check.txt) | The exact one-token check: stock rerun and the settings' run, statistics blocks |
| [tables/greedy-ref.json.gz](tables/greedy-ref.json.gz), [tables/greedy-mem1400.json.gz](tables/greedy-mem1400.json.gz) | The greedy responses (text and every token's top-10 probabilities) at stock and at +1,400 MT/s |
| [run/guards.sh](run/guards.sh) | The guards: state record (start and end), the watchdog started or its power rule changed, the power cap on every card, and the watchdog's trip test |
| [run/offset_climb.sh](run/offset_climb.sh) | One step (set, read back, bench, greedy, Xid, the watchdog's peaks), the soak, the server |
| [run/greedy_check.py](run/greedy_check.py) | The greedy check against the served model |
| [run/greedy_diff.py](run/greedy_diff.py) | Where two greedy runs differ: tokens with other probabilities, other tokens, the text |
| [run/final_checks.sh](run/final_checks.sh) | The long checks: the exact check's stock reference, `cuda_memtest` at the settings (tests 0–5, 7, 8, 10, then 6), a pass at a failed step, the exact check |
| [run/short_bench.sh](run/short_bench.sh) | A 7-second streaming check of one card or several, or of one card at a given memory offset (diagnostics) |
| [run/tables.py](run/tables.py) | Builds `tables/` from the run's output and the watchdog log |
| [../../../scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py) | The watchdog |
| [../../../bench/mem_offset/nvml_memoc.py](../../../bench/mem_offset/nvml_memoc.py) | Reads and sets one card's offsets through NVML; `probe` is read-only |

## Commands

On the rig, as root, with the [memory-offset kit](../../../bench/mem_offset/README.md) unpacked in `$KIT`, this
folder's `run/` and the watchdog in `$WORK`, `BIN` the llama.cpp build's `bin` folder (Miner Llama v4.0) and `MODEL`
Flint Pascal Q5 GGUF:

```bash
export WORK=/tmp/q133 KIT=/tmp/q133/mem_offset-kit-20261024 BIN=... MODEL=... WIKI=.../wiki.test.raw
bash run/guards.sh state start_state.txt                            # read-only: processes, cards, offsets
python3 $KIT/nvml_memoc.py probe                                    # read-only: ranges, clocks, power, the link
bash run/guards.sh trip-test                                        # the watchdog's three trips, before any offset
bash run/guards.sh watchdog 65                                      # this sitting: 55, then 65, then 80 at the 75 W cap
bash run/guards.sh cap 60                                           # this sitting: 60, then the stock 75
setsid nohup bash run/final_checks.sh reference > /dev/null 2>&1 < /dev/null &   # the exact check's stock reference
bash run/offset_climb.sh server start
bash run/offset_climb.sh step mem 0 ref                             # the greedy reference
OUT=$WORK/out GPU=0 FORCE=1 bash $KIT/memoc_steps.sh units          # NVML's memory unit
MEM_CAP=2000 bash run/offset_climb.sh step mem 200                  # ... +200 a step until the first failure
python3 run/greedy_diff.py $WORK/out/greedy-ref.json $WORK/out/greedy-mem1400.json   # where a failed step differs
CORE_CAP=300 bash run/offset_climb.sh step core 25                  # ... +25 a step, the memory at 0
python3 $KIT/nvml_memoc.py --core set 0 275
MEM_CAP=2000 bash run/offset_climb.sh soak mem 1000 30              # both offsets, 30 min
bash run/offset_climb.sh server stop
bash run/final_checks.sh memtest settings                           # cuda_memtest at the settings
bash run/final_checks.sh memtest-at 1400 mem1400                    # does cuda_memtest see the failed step?
python3 $KIT/nvml_memoc.py --cap 1200 set 0 1000; python3 $KIT/nvml_memoc.py --core set 0 275
bash run/final_checks.sh exact combined                             # the exact check at the settings
python3 $KIT/nvml_memoc.py reset 0; python3 $KIT/nvml_memoc.py --core reset 0
bash run/guards.sh stop; bash run/guards.sh state end_state.txt
bash run/short_bench.sh check 0 1 2                                 # card 0 against two cards never offset
python3 run/tables.py $WORK/out $WORK/wd.csv tables
```

The sitting itself ran these steps by hand and from small one-off scripts; the scripts in `run/` are those steps
collected, with the paths in the environment.

**Provenance and licence.** Measured on the rig on 2026-11-17 with the kit's `mmvq_pascal` and `cuda_memtest`
(cuda_memtest commit `e94e1ee5`, its own licence) built for sm_61, Miner Llama v4.0's engine (build of fork commit
`eb312b7bc`, the v4.0 code) and Flint Pascal Q5 GGUF; wikitext-2 test set for the exact check. The tables are this project's measurements (the repository's
licence).
