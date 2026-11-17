# C832: every card's clocks, power, temperature and throttle reasons in a run's log (2026-11-17)

**What a reader can now see, in plain words.** The agent check, the deep lane and `scripts/bench.sh` now keep a record
of every card twice a second through the whole run: its clock, its memory clock, its power, its temperature, its power
state, how busy it was, and the reasons the driver held its clock back. The record only reads; it sets nothing on the
cards. A table script turns it into one line per card and lists every stretch of 2 seconds or more in which a working
card ran below 1,800 MHz. So when a run has a slow stretch, like the 20 turns that ran 306 ms or more slower in three
of R43's replay passes ([Q1464](../../../quants/experiments.md)), its timings can be laid beside what each card was
doing at that moment.

**The test on R194's log.** The table script ran on the watchdog log of R194 (nine cards every 0.5 s for 20 minutes,
the same afternoon). Where its numbers overlap R194's own tables they agree: peaks of 84.05 W and 46 °C on card 0,
92.79 W on card 3 and 51 °C on card 1, and 64, 63, 67, 62 and 4 samples at the power cap on cards 0 to 4. That log has
no utilization column, so "busy" there falls back to "not asleep and no idle reason". That rule counts a card that a
process merely holds open at its 1,354 MHz base clock: in R194's step 1, cards 5 to 8 (never offset, only held open by
the run's server) read 100 % of their "busy" samples below 1,800 MHz. The new log records utilization, the share of
time a kernel ran, and that is what tells a working card from one held open.

Registry: [Q1464](../../../quants/experiments.md) (this row), [Q1551](../../../quants/experiments.md) (R186: on driver
580.178.04 the SM clock field shows the clock a card runs, so samples every 0.5 s or faster see a card slow down),
[Q1550](../../../quants/experiments.md) (the driver point release that fixes frequent `nvidia-smi` and NVML PCI
queries), [Q1518](../../../quants/experiments.md) (joules a token, which this log's power column allows) and
[Q1576](../../../quants/experiments.md) (the drivers not wired yet). R186's data:
[2026-11-17_r186_r188_clocks](../2026-11-17_r186_r188_clocks/README.md); R194's:
[2026-11-17_r194_nine_cards_soak](../2026-11-17_r194_nine_cards_soak/README.md).

## The log

[scripts/card_log.py](../../../scripts/card_log.py) is one Python process for the whole run. It talks to the driver's
NVML library directly (ctypes on `libnvidia-ml.so.1`: no install, no root, no `nvidia-smi` process per sample) and
appends one CSV row every 0.5 s:

| Column | What it is |
|---|---|
| `epoch`, `time` | the sample's time: Unix seconds, and local time to the millisecond |
| `took_ms` | how long the sample took to read |
| `g<i>_pstate` | card i's power state (P0 working or boosting, P8 asleep) |
| `g<i>_sm_mhz`, `g<i>_mem_mhz` | its SM clock (the clock it runs, R186) and memory clock |
| `g<i>_w`, `g<i>_c` | its power draw (W) and temperature (°C) |
| `g<i>_used_mib` | its memory in use |
| `g<i>_util` | its utilization: the share of the driver's last sample period in which a kernel ran |
| `g<i>_reasons` | the reasons its clock is held back, a bit mask: `0x1` idle, `0x2` applications clocks, `0x4` software power cap, `0x8` hardware slowdown, `0x10` sync boost, `0x20` software thermal, `0x40` hardware thermal, `0x80` power brake, `0x100` display clock; `0x0` none |

Cards are numbered as `nvidia-smi` numbers them (PCI bus order). The per-card names are those of
[scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py)'s log, so the table script reads both. Lines starting
with `#` are events: the start (driver version, cards, each card's power limit and both clock offsets as read) and the
stop (why it stopped, samples, the process's CPU seconds, how long sampling took, samples that started more than one
interval late, and the settings again).

It calls only NVML functions that read (`card_log_test.sh` checks the list), and never asks for a card's PCI fields,
UUID or serial number: NVIDIA's 580.196.07 fixes "GPU fallen off bus" errors from frequent PCI queries by `nvidia-smi`
and NVML (FN116), and the rigs run 580.178.04. It replaces three `nvidia-smi` polls:

| Driver | Before | Now |
|---|---|---|
| [bench/agent_check/rig_server.sh](../../../bench/agent_check/rig_server.sh) (the agent check, the release check, the race) | `gpu.csv`: temperature, power, SM clock, memory, utilization every 10 s | `cards.csv` every 0.5 s, from the first server start to `end`; the logger is embedded in the script (it is read over ssh), its copy on tmpfs; a safety stop after 15 hours |
| [bench/deep_lane/deep_lane.sh](../../../bench/deep_lane/deep_lane.sh) (R121, R191, R192 and every deep-lane run) | `gpu.csv` every 10 s | `OUT/cards.csv` every 0.5 s for the whole driver; `remote.sh` copies `card_log.py` beside it; it stops when the driver ends, however it ends |
| [scripts/bench.sh](../../../scripts/bench.sh) (llama-bench per card set) | `smi.csv` every 0.5 s through `nvidia-smi` | `cards.csv` every 0.5 s, embedded like the agent check's |

A log that does not come up (no NVML, a driver error) is reported in the driver's console and the run goes on without
it; its reason is in `card_log.out`.

## What to look at in a run

```bash
python3 -I scripts/card_table.py cards.csv --out cards.md --tsv cards.tsv
python3 -I scripts/card_table.py cards.csv --since 2026-11-17T15:20:00+01:00 --until 2026-11-17T15:35:00+01:00
```

- **The header:** the samples, the median gap between them and any gap over 2 s (time the log missed).
- **Per card:** seconds at each power state; busy seconds; the SM clock's median and lowest while busy; how many busy
  samples ran below 1,800 MHz (a P106-090 boosts to about 1,860 to 1,911 MHz at stock, R186); the memory clock while
  busy; peak power and temperature; the most memory used; the throttle reasons by sample count, over all samples and
  over busy samples. `power cap` while busy means the card hit its 75 W limit; a `thermal` or `hardware slowdown`
  count means it slowed for heat or a hardware signal.
- **Stretches:** every run of busy samples below 1,800 MHz lasting 2 s or more, with its card, start, end, lowest and
  median clock, power states and reasons. A reply's ramp from a resting clock lasts under a second on these cards
  (R186), so a stretch is a card that ran slow, not one waking up.
- **The log's own cost:** the stop line's CPU seconds and late samples, on a 2-core host.
- **One load of the deep lane:** `--since` that load's `started` and `--until` the next load's, both from `loads.tsv`.

## Files

| File | What it holds |
|---|---|
| [tables/r194_cards.md](tables/r194_cards.md), [tables/r194_cards.tsv](tables/r194_cards.tsv) | `card_table.py` on R194's whole watchdog log (2,480 samples, 15:17:01 to 15:37:44): per card, and the first 50 of 62 stretches |
| [tables/r194_cards_step1.md](tables/r194_cards_step1.md) | the same over R194's step 1 alone (15:24:42 to 15:37:16), the window of R194's `phases.csv` |
| [tables/card_log_test.txt](tables/card_log_test.txt) | `card_log_test.sh`'s output: 26 of 26 checks pass |
| [../../../scripts/card_log.py](../../../scripts/card_log.py) | the logger |
| [../../../scripts/card_table.py](../../../scripts/card_table.py) | the table script |
| [../../../scripts/card_log_test.sh](../../../scripts/card_log_test.sh) | its self-test: the embedded copies match, the logger calls only reading NVML functions, and the logger against a stand-in for NVML written in Python (no card touched) with `card_table.py` on its log |

## Commands

The self-test, on any machine (no card or NVIDIA library is touched); `SECTIONS=real` runs the logger on the machine's
own cards for `REAL_S` seconds:

```bash
bash scripts/card_log_test.sh
```

The tables, on a copy of R194's watchdog log (`wd.csv.gz`, kept with the run's raw logs, not published):

```bash
python3 -I scripts/card_table.py wd.csv.gz --out tables/r194_cards.md --tsv tables/r194_cards.tsv
python3 -I scripts/card_table.py wd.csv.gz --since "2026-11-17 15:24:42" --until "2026-11-17 15:37:16" \
  --out tables/r194_cards_step1.md
```

## Not yet known

- **The logger has not run on a rig.** Its NVML calls are the watchdog's and R186's, but the logger itself was checked
  only against the stand-in. The next timed run on either rig is its first real run: its `cards.csv` should have a
  value in every field, a row every 0.5 s, and a stop line with its CPU seconds.
- **Its cost on the 2-core host** is not measured: about 72 NVML reads every 0.5 s (8 per card), est. a few ms of CPU
  each time. The stop line gives it.
- **What "busy" by utilization counts** during a layer-split decode, where each card works for a slice of every token,
  is not checked against the old fallback rule.
- **Not every driver logs yet:** `bench/release_matrix/run_matrix.sh` polls `nvidia-smi` every second, the four
  `bench/pipespec/rig_*.sh` scripts every second, and `scripts/server_bench.sh` and the runs' own `run/` scripts log
  nothing ([Q1576](../../../quants/experiments.md)).

**Provenance and licence.** The tables summarise R194's watchdog log, measured on a nine-card P106-090 rig on
2026-11-17 (driver 580.178.04). The scripts and tables are this project's (the repository's licence).
