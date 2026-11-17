# The driver's clock controls and its clock field on a P106 (R186, R188; 2026-11-17)

What this is: two checks on one of the 4 GB rigs (nine P106-090 cards, driver 580.178.04, persistence mode off, no X
server). R188 tries the three run-time clock controls the driver documents on an idle card, then times the one it
accepts. R186 looks at what `nvidia-smi`'s clock field shows across the end of a decode. It also measures the size of
the NVIDIA kernel modules, the GPU interrupts a token and the host's memory with the cards empty and full. Registry rows
Q1553 (R188) and Q1551 (R186) in [quants/experiments.md](../../../quants/experiments.md). The questions come from
[the clock-hold note](../../research/notes/clock_hold_by_driver.md) (FN105) and
[the driver-branch note](../../research/notes/driver_branch_pascal.md) (FN107).

## The answers

**No driver control holds a P106 at its boost clock.** Locked clocks and the CUDA clock override are refused. PowerMizer
mode 3 (lock to base) is refused too. Mode 1 is accepted with no X server, but it keeps a card awake at its 1,354 MHz
base clock and does not stop it falling to base between requests. A card that starts a reply at boost decodes at
8.84 ms a token in either mode, and one that starts at base or asleep is slower. The keep-alive kernel stays the only
way to hold boost.

**On driver 580.178.04 the clock field shows a card that slowed down.** After a decode each card's SM clock falls in
steps: from its busy value to 1,354 MHz within 5 s, to about 800 MHz at 17 s, then asleep at 139 MHz by 24 s. Power
falls with it (card 0: 31.9, 27.9, 25.8, 5.5 W), so the field follows the clock the card actually runs. A log of
`clocks.sm`, the P-state and power every 0.5 s or faster sees every drop. A 10 s log misses the ramp at the start of
a reply, which lasts under a second.

### R188 stage 1: three controls on idle card 0 ([r188_stage1.tsv](r188_stage1.tsv))

| Control | Command | Result |
|---|---|---|
| Locked clocks | `nvidia-smi -i 0 -lgc 1911,1911` | **refused**: "Setting locked GPU clocks is not supported" (exit code 0, "treating as warning"); the card stays in P8 at 139 MHz |
| CUDA clock override | `nvidia-smi -i 0 -cc 1` | **refused**: "Overriding CUDA clock limit is not supported for this device"; no change |
| PowerMizer mode 1, "prefer maximum performance" | `nvmlDeviceSetPowerMizerMode_v1`, no X server ([run/r188_stage1.py](run/r188_stage1.py)) | **accepted**, read back as 1: the idle card goes to P0 at 1,354 MHz SM and 4,006 MHz memory, 25.9 W, no clock-event reason |
| PowerMizer mode 3, "lock to GPU base clocks" | the same call, mode 3 | **refused**: `NVML_ERROR_INVALID_ARGUMENT`; the card lists modes 0, 1 and 2 only (mask `0b0111`) |

- **The default mode through NVML is 0 (adaptive)** on all nine cards, not 2 (auto). Mode 2, set as the planned undo,
  let the card sleep again: P0 at 1,354 MHz for 1.8 s, then 772 MHz with the idle reason, then P8 at 177 MHz 4.9 s
  after the set. The card was then set back to 0.
- **A mode lasts only while a process holds the card.** With persistence mode off, the driver tears a card down when
  its last client exits. A fresh process right after the test read mode 0 on card 0, not the 2 the test had set. So a
  mode must be set while a server or another process keeps the cards open ([run/pm_mode.py](run/pm_mode.py) can hold
  them with `HOLD_S`).

### R188 stage 2: mode 1 on a served decode ([requests_clocks.tsv](requests_clocks.tsv), [arms.tsv](arms.tsv))

Flint Pascal Q8 GGUF on its released preset (`release/flint-v4/settings.env`, keep-alive off), the v4.0 build. In one
load, 256-token replies with mode 0 (the default) and mode 1 on all nine cards, set through NVML between requests in
the order 0 1 1 0 0 1 1 0, about 3 s apart. In a second load, the same preset plus
`GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1`, four replies. nvidia-smi logged every 100 ms. Every reply did
the same work: 272 pipelined guesses, 206 right and 17 wrong. So the cards are busier than in a plain one-token pass:
206 of 256 tokens came from Flint's pipelined guesses.

| Request | Mode | ms a token | Slowest card just before it | Slowest card in its decode | Samples at 1,800 MHz or more |
|---:|---|---:|---|---:|---:|
| 1 | 0 | 9.697 | P8, 139 MHz | 1,354 | 34.9 % |
| 2 | 1 | 8.836 | P0, 1,847 MHz | 1,847 | 100 % |
| 3 | 1 | 8.854 | P0, 1,847 MHz | 1,847 | 100 % |
| 4 | 0 | 8.844 | P0, 1,847 MHz | 1,847 | 100 % |
| 5 | 0 | 8.841 | P0, 1,847 MHz | 1,847 | 100 % |
| 6 | 1 | 9.423 | P0, 1,354 MHz | 1,354 | 40.7 % |
| 7 | 1 | 8.840 | P0, 1,847 MHz | 1,847 | 100 % |
| 8 | 0 | 8.859 | P0, 1,847 MHz | 1,847 | 100 % |
| keep-alive, 4 replies | 0 | 9.790, 9.087, 9.099, 9.098 | P0, 1,847 or 1,860 MHz | 1,847 | 100 % |

- **In the load without the keep-alive, what sets the speed is the clock a reply starts at, not the mode.** Requests
  started at boost ran at 8.836, 8.854 and 8.840 ms in mode 1 and 8.844, 8.841 and 8.859 ms in mode 0. Request 1 started asleep after a minute of idle and ran at 9.697 ms. Request 6 started
  at base: 2.7 s after request 5 ended, card 0 dropped to 1,354 MHz with mode 1 set, and the reply ran at 9.423 ms.
- **Mode 1 does not hold the clock** by Q1553's rule. The rule needs 1,800 MHz or more in every sample. Every card read
  below that in some of mode 1's decode samples, and cards 0, 1, 4 and 8 read 1,354 MHz ([clocks.tsv](clocks.tsv):
  81.2 to 90.6 % of a card's samples at 1,800 MHz or more). Its token times are within the rule's 1 ms of the
  keep-alive's. Setting the mode changed no clock: right after each set every card still read its boost value (the
  readback in `pm_mode.py`'s output). The drops came 2–5 s after the last work, in either mode.
- **The keep-alive holds every card at 1,847 MHz or more in every sample,** but its replies take 9.790, 9.087, 9.099
  and 9.098 ms a token, against 8.84 without it. The first ran at 9.790 ms although every card started at boost. That is one load each, so it is a note, not a class: R165 classed the keep-alive a
  decode loss on Flint at depth (−3.0 %,
  [2026-11-16_r165_flint_stack_split](../2026-11-16_r165_flint_stack_split/README.md)).
- **For Q1554** (mode 1 as the idle setting between agent turns): the same 256-token reply ran at 9.697 ms from sleep,
  9.423 ms from base and 8.84 ms from boost. Mode 1 keeps a card at base instead of asleep: est. 0.27 ms a token on a
  reply's first 256 tokens, about 70 ms a turn, for about +20 W a card while tools run.

### R186 (a): the clock field across the end of a decode ([r186a_after.tsv](r186a_after.tsv), [r186a_timeline.tsv](r186a_timeline.tsv))

nvidia-smi every 500 ms for one minute (`clocks.sm`, `clocks.mem`, `pstate`, power, `clocks_event_reasons.active`,
utilization). The minute had 15 s idle, one 512-token decode (7.8 s) and 37 s idle.

- **During the decode no card read below its busy value** (1,860 to 1,911 MHz, a card's own maximum, 14 samples a
  card). Card 6 read 1,873 and 1,885 MHz.
- **After the decode every card read lower, in steps** (card 0: 1,860 MHz to +2.3 s, 1,607 to +5.0 s, 1,354 to
  +16.7 s, 784 with the idle reason to +19.3 s, P8 at 177 MHz, then 139 MHz from +22.0 s). The other cards are the
  same within a second. Cards 7 and 8 held their busy value to +18.3 and +18.8 s, then went straight to about 800 MHz.
- **Power moves with the reading,** so the field reports the clock the card runs, not the one before idle slowdowns:
  card 0 at 0 % load drew 31.9 W at 1,860 MHz, 27.9 W at 1,607, 25.8 W at 1,354 and 5.5 W in P8.
- **How long a card holds boost after work varies.** In the 15 s before the decode, all nine cards stayed at their
  busy clock at 0 % load (the previous request had ended 0.2 s before the log began). After the decode they began to
  fall at 1.8–2.4 s. Between the 256-token replies above, cards held boost through most 3 s gaps and dropped once at
  2.7 s.
- **The reasons field** shows `GpuIdle` (0x1) only below base and in P8, and `SwPowerCap` (0x4) in 4 samples of card
  8, at 60.5–70.7 W with its clock unchanged. Card 8 holds the output layer and reached 74.3 W of its 75 W limit.

### R186 (b), (c), (d): modules, interrupts, host memory

| Measure | Value | File |
|---|---|---|
| NVIDIA kernel modules (`lsmod`) | `nvidia` 105,373,696 bytes (100.49 MiB), `nvidia_uvm` 1.98 MiB, `nvidia_modeset` 1.61 MiB, `nvidia_drm` 0.13 MiB: 104.2 MiB, 2.7 % of the host's 3,836 MiB | [modules.tsv](modules.tsv) |
| The driver's run-time memory, empty cards against Flint loaded | unreclaimable slab 87 → 143 MiB, vmalloc 197 → 224 MiB | [memory.tsv](memory.tsv) |
| GPU interrupts, nine cards together, Flint loaded and idle | 22.97 and 25.38 a second (about 2.5 a card) | [interrupts.tsv](interrupts.tsv) |
| GPU interrupts in a 256-token decode, above the idle rate | 0.626 and 0.190 a token (236 in 3.03 s, 114 in 2.67 s) | [interrupts.tsv](interrupts.tsv) |
| Host memory, cards empty | 3,201 MiB available; the largest process 40 MiB | [memory.tsv](memory.tsv) |
| Host memory, Flint loaded | 1,431 and 1,437 MiB available (two loads); llama-server 1,528 MiB resident (1,746 after the requests), 374 MiB shared, 268 MiB in swap; lowest usable (available minus free CMA) 1,129 and 1,207 MiB | [memory.tsv](memory.tsv) |
| Cards' memory, Flint loaded | 4,793 to 5,923 MiB a card | private raw logs |

- **There is no desktop session to measure.** The rigs boot to `multi-user.target` with no X server or GNOME, so
  nothing on the host draws on the driver's display memory. Q1551 (d) was written for a desktop install. The host's
  memory is what llama-server holds itself, and it came back to 3,159 MiB available after the stop.
- **Interrupts are no cost on decode:** under one interrupt a token for all nine cards, because CUDA's wait polls.
  Q1552 (the driver's interrupt locking mode, parked) would reopen only at a rate worth a millisecond a step. At these
  counts it has nothing to win.

## Files

| File | What |
|---|---|
| [r188_stage1.tsv](r188_stage1.tsv) | R188 stage 1: each command, its exit code and message, card 0 before, during and after |
| [requests.tsv](requests.tsv) | every request: rig-clock start and end, prompt and reply tokens and times, ms a token |
| [arms.tsv](arms.tsv) | ms a token by arm, each request listed |
| [requests_clocks.tsv](requests_clocks.tsv) | each request of the two 100 ms logs: the cards' clocks just before it and in its decode |
| [clocks.tsv](clocks.tsv) | each card's SM clock in the decode windows, by arm: samples, lowest, median, highest, share at 1,800 MHz or more, P-states, reasons |
| [r186a_timeline.tsv](r186a_timeline.tsv) | R186 (a): every 500 ms poll, card by card (P-state, SM clock, reasons), in seconds from the decode's end |
| [r186a_after.tsv](r186a_after.tsv) | R186 (a): each card's busy clock and every change of its reading after the decode |
| [interrupts.tsv](interrupts.tsv) | the NVIDIA interrupt lines over two idle windows and two decodes |
| [memory.tsv](memory.tsv) | `/proc/meminfo` and resident memory in six snapshots (empty, loaded, after the requests, stopped, the second load, the end) |
| [modules.tsv](modules.tsv) | the NVIDIA modules' sizes from `lsmod` |
| [run/r188_stage1.py](run/r188_stage1.py) | stage 1, on the rig: holds NVML open, sends the commands, reads the card after each |
| [run/pm_mode.py](run/pm_mode.py) | reads or sets each card's PowerMizer mode through NVML; `HOLD_S` keeps the cards open |
| [run/run.sh](run/run.sh) | R186 and stage 2: two loads driven from the client machine |
| [run/rig_r186.sh](run/rig_r186.sh) | its rig side: snapshots, interrupt reads, the server with `scripts/ram_watchdog.sh`, the nvidia-smi logs |
| [run/decode_req.py](run/decode_req.py) | one timed decode request to `/completion` |
| [run/tables.py](run/tables.py) | every table above from `run.sh`'s output |

## Commands

```bash
# R188 stage 1, on the rig, nothing on the cards
ssh "$RIG" 'CARD=0 python3 -I -' < run/r188_stage1.py > r188_stage1.log
# every card's mode afterwards, and card 0 back to its mode before the test (0)
ssh "$RIG" 'python3 -I -' < run/pm_mode.py
ssh "$RIG" 'CARDS=0 MODE=0 python3 -I -' < run/pm_mode.py

# R186 and stage 2, from the client machine (two loads, est. 20 min; each load of the 40.4 GB file took 581 and 580 s)
RIG=rig WORK=/tmp/r186-r188 BIN=<the v4.0 build>/bin MODEL="$MODELS/flint-pascal-Q8.gguf" OUT=out bash run/run.sh
# the tables, on any machine with Python 3
python3 -I run/tables.py out
```

## Provenance

- Rig: one of the two 4 GB rigs, Ubuntu 24.04 booted to `multi-user.target`, driver 580.178.04, persistence mode
  off, every card idle (0 MiB, P8) before and after.
- Build: Miner Llama v4.0 (`pascal-p106-v4` at the tag `miner-llama-v4.0`, `llama-server` md5
  `e7afd26db01f4c671549ef12c504ada3`). Model: Flint Pascal Q8 GGUF (`p065`, 40,431,236,544 bytes, md5
  `8387b2e585a7b9cc10f40cb6b4531694`).
- The NVML struct for the PowerMizer calls (`nvmlDevicePowerMizerModes_v1_t`: current mode, mode to set, supported
  mask) is from the `nvml.h` that NVIDIA's nvidia-settings 580.95.05 ships; CUDA 12.9's `nvml.h` does not have it.
  The rig's `libnvidia-ml.so.1` exports both calls.
- Times: requests in the client's clock, shifted to the rig's by the offset `run.sh` logs first (0.03 s); nvidia-smi
  in the rig's local time.
- Every setting was undone and checked: card 0's mode back to 0, all nine cards read mode 0, P8 at 139 MHz and 0 MiB
  at the end. Nothing was reloaded, rebooted or kept.
- Licence: the tables and scripts are this repository's (MIT).
