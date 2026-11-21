# C942: where the card log's CPU goes, and a cheaper sample (2026-11-21)

**In plain words.** On the rig, the read-only record of every card twice a second
([scripts/card_log.py](../../../scripts/card_log.py)) uses 35.0 ms of the host's CPU for each sample: 7 % of one of
the Celeron's two cores. Almost none of that is the script's own Python work. It goes into the 72 questions it asks
NVIDIA's library each sample (8 for each of 9 cards), which the driver answers on the host's CPU. So the only real
savings are fewer questions or fewer samples. The logger now has three options for that, all off unless asked for:
`CARD_LOG_FAST=1` writes the same rows with less Python work; `CARD_LOG_SLOW_EVERY=4` also asks for the memory clock,
temperature and memory used on only one sample in four, which saves 28 % of the questions; `CARD_LOG_PROFILE=1`
measures what each question costs. A short rig run (C943, commands below) will measure all three. Until then,
everything runs as before.

Registry: [Q1682](../../../quants/experiments.md) (this row), from [Q1576](../../../quants/experiments.md)
([C873](../2026-11-21_c873_card_log_drivers/README.md), which found the cost); new rows
[Q1683](../../../quants/experiments.md) (the rig check, C943) and [Q1684](../../../quants/experiments.md) (the
driver's own sample buffers, C944).

## Where the 35 ms goes

**The rig's own logs.** The logger's stop line gives its CPU seconds and the median time a sample took
(`sampling … ms median`, the wall clock). They match, so a sample is CPU work in the logger's own process, not
waiting:

| Run (nine cards, 0.5 s) | Samples | CPU | CPU a sample | Sampling, median |
|---|---|---|---|---|
| [R53](../2026-11-20_r53_flash_draft_prompt/README.md) (deep lane) | 3,675 | 128.56 s | 34.98 ms | 39.2 ms |
| [R175](../2026-11-20_r175_flash_shortlist/README.md) (deep lane) | 3,425 | 120.29 s | 35.12 ms | 36.5 ms |
| [R69](../2026-11-20_r69_sparse_switch/README.md) (deep lane) | 6,272 | 220.54 s | 35.16 ms | 37.8 ms |
| [R227](../2026-11-20_r227_flash_taper1400_agent/README.md) arm b (agent check) | 3,078 | 91.04 s | 29.58 ms | 29.3 ms |

Each row's `took_ms` column, split by what the cards were doing in that row
([run/took_by_state.py](run/took_by_state.py) on the three deep-lane logs):

| Run | All 9 asleep (P8) | Awake, utilization 0 | Some busy |
|---|---|---|---|
| R53 | 33.5 ms median (3 rows) | 31.4 ms (45 rows) | 39.3 ms (3,627 rows) |
| R175 | 32.1 ms (3 rows) | 33.4 ms (56 rows) | 36.6 ms (3,366 rows) |
| R69 | 31.6 ms (3 rows) | 30.3 ms (85 rows) | 37.9 ms (6,184 rows) |

Most of the cost is there even when no card works; busy cards add a few milliseconds.

**The Python side, on a laptop.** [run/prof_today.py](run/prof_today.py) runs today's sample with every NVML call
replaced by a real ctypes call into a C function that does nothing (libc's `abs()` on a null handle, which returns 0,
NVML's "success"). So every Python and ctypes cost of a real call is paid, and NVML's own work is not. On a Ryzen 9
8940HX with Python 3.14.4, 9 cards, 3,000 samples, two runs ([prof_today.txt](prof_today.txt)):

| Part of a sample | Run 1 | Run 2 |
|---|---|---|
| The whole sample: 72 calls, the row, the line, the write | 64.8 µs | 52.8 µs |
| the 9 cards' calls and fields alone | 54.5 µs | 41.8 µs |
| the line (time stamp, formatting) | 2.4 µs | 1.8 µs |
| the write, one `write()` a row | 1.4 µs | 1.0 µs |
| the write into a 64 KiB buffer | 0.2 µs | 0.1 µs |
| the parent and stop-file checks | 4.0 µs | 2.9 µs |
| one call as today's code makes it (lookup, fresh buffer, conversion): P-state | 0.5 µs | 0.4 µs |
| the same: a clock (with its clock-type argument) | 0.9 µs | 0.7 µs |
| the same: power | 0.7 µs | 0.5 µs |
| the same: memory (a structure) | 0.6 µs | 0.4 µs |

Python's start-up and imports (`python3 -I card_log.py --help`: 0.03 s in each of three runs) happen once a run.
**Est.:** if this Celeron 3865U (1.8 GHz, no boost, an older core) runs the same Python five times slower than the
laptop, the Python side is est. 0.3 ms of the rig's 35.0 ms. The other est. 34.7 ms, 99 %, is inside the NVML
library and the kernel driver answering the 72 calls. The stand-in cannot measure that part, and `CARD_LOG_PROFILE=1`
on the rig will.

**Asking for several fields in one call does not help here.** `nvmlDeviceGetFieldValues` takes a list of field ids.
The newest NVML header on the laptop (CUDA 13.4's `nvml.h`, newer than the rig's 580 driver) has a field for power
(`NVML_FI_DEV_POWER_INSTANT`, 186, "supported on all architectures"). It has none for the P-state, the SM or memory
clock, the die temperature (only its limits, 193 to 196, and the memory's temperature, 82), memory used, utilization
or the throttle reasons as a mask (only time counters, 74 to 81 and 269 to 271). The header adds that fields that one
driver call fills come from one call, which none of ours share. So that path would swap one call for one call, and it
is not built.

## The three options

All three are in [scripts/card_log.py](../../../scripts/card_log.py) and in every copy of it (the agent check's
`rig_server.sh`, `scripts/bench.sh`, `scripts/server_bench.sh`, R43's replay driver `rig_taper.sh`, and the copies
beside `bench/pipespec/` and `bench/release_matrix/`). Each driver passes the environment through to the logger
unchanged, so setting a variable where the driver starts is enough. With none set, the log is made exactly as before.

| Option | What changes | Rows |
|---|---|---|
| `CARD_LOG_FAST=1` (`--fast`) | each NVML function looked up once, each card's arguments and buffers made once, the row built in one pass; rows go to the file in blocks: the header and start line, the first row at once (the drivers wait for it), then every 5 s and at the stop. A logger killed with SIGKILL loses up to 5 s of rows, but SIGTERM, the stop file and the parent's end write everything. A second `#` line after the start line says it is on | the same NVML calls in the same order, the same rows |
| `CARD_LOG_SLOW_EVERY=N` (`--slow-every N`, turns on `--fast`) | a card's memory clock, temperature and memory used are asked for on one sample in N, and on any sample where its P-state changed; in between, the row repeats the value last read. P-state, SM clock, power, utilization and throttle reasons are read on every sample | the same columns; those three fields can lag a change by up to N − 1 samples (1.5 s at N = 4 and 0.5 s), except at a P-state change |
| `CARD_LOG_PROFILE=1` (`--profile`) | times each NVML call on the thread's CPU clock; one more `#` line just before the stop line: CPU a sample (user and system), the part inside NVML, and for each function its calls a sample and CPU ms a call, the costliest first | unchanged |

A slower interval cuts the cost in proportion. The drivers already take it from `CARD_LOG_S` (for example
`CARD_LOG_S=1`), apart from `scripts/bench.sh`, which keeps 0.5 s.

The fast sample's Python side, the same laptop and harness ([run/prof_fast.py](run/prof_fast.py),
[prof_fast.txt](prof_fast.txt), two runs, µs of CPU a whole sample):

| Sample | Run 1 | Run 2 |
|---|---|---|
| today's | 66.6 | 63.1 |
| `CARD_LOG_FAST=1` | 34.9 | 25.6 |
| `CARD_LOG_SLOW_EVERY=4` | 28.1 | 20.7 |

**Est. cost on the rig, a sample (the stand-in is not a P106 driver):**
- `CARD_LOG_FAST=1`: est. 34.8 ms against 35.0. It saves est. 0.2 ms of Python work, so on its own it is a tidier
  logger, not a cheaper one.
- `CARD_LOG_SLOW_EVERY=4`: a card makes 5 calls on three samples of four and 8 on the fourth, 5.75 on average against
  8, or 71.9 % of today's calls. If every call costs alike, that is est. 25.0 ms a sample, 5.0 % of one core at 0.5 s.
  The profile will show whether the three calls it skips cost more or less than the average.
- `CARD_LOG_SLOW_EVERY=4` with `CARD_LOG_S=1`: est. 12.5 ms a second, 2.5 % of one core.

## Tests (laptop, on the stand-in for NVML; no rig, no GPU)

- [scripts/card_log_test.sh](../../../scripts/card_log_test.sh) has a new section, `fast`, with 13 checks (the first with 8 parts). Over 40
  samples of 3 cards from `scripts/card_log_standin.py`, with its slow stretch and a memory offset set at sample 20,
  `--fast` makes the same 960 NVML calls in the same order and the same 40 rows, field for field, as today's sample.
  The rows are byte for byte the same apart from `epoch`, `time` and `took_ms`. `--slow-every 4` reads the
  temperature on samples 0, 4, 8, … and at card 0's P-state changes (10 and 15); the other five fields match on every
  sample, and it makes 696 calls in place of 960. From the command line: each variable turns on its option and
  `CARD_LOG_FAST=0` does not; the first row reaches the file at once; after SIGTERM every sampled row is in the file;
  the profile line comes just before the stop line and names the 7 sampled functions; `CARD_LOG_SLOW_EVERY` set to
  `x`, `0` or `-2` stops the logger with status 2 and a message. The fake and driver sections check that a log made
  with an option says so, and one made without has only the start and stop lines.
- The whole self-test, three ways: **78 of 78** with nothing set, **79 of 79** with `CARD_LOG_FAST=1`, **80 of 80**
  with `CARD_LOG_SLOW_EVERY=4 CARD_LOG_PROFILE=1`. That covers the six copies, the stand-in runs read by
  `card_table.py` (card 1's slow stretch found, none on card 0), and the sixteen driver dry runs. With an option set,
  every driver's log carries the fast line, so the setting reaches the logger through each driver.
- `bench/release_matrix/test/dryrun.sh`: **22 of 22** with nothing set and **22 of 22** with `CARD_LOG_FAST=1`.
- [run/rig_cost_dryrun.sh](run/rig_cost_dryrun.sh): the rig check's script on the stand-in and a stub server,
  **13 of 13**.

## The rig check (C943, not run yet)

[run/rig_cost.sh](run/rig_cost.sh) runs on the rig in a quiet window, about 30 min:

1. **Idle, no model loaded:** the logger alone for 120 s an arm, the arms today, fast and slow4 in turn, twice, then
   today and slow4 once more with the profile.
2. **Busy:** one load of a release preset, then three rounds of the arms off (no log), today, fast and slow4 in turn,
   each arm 4 requests of 384 tokens from one fixed prompt at temperature 0, then one arm of today's sample with the
   profile.

`summary.md` gives, for each arm, the logger's CPU a sample from its stop line, the host's busy share from
`/proc/stat`, and each request's generation and prompt speed by value. The profile lines say which NVML calls cost
the most.

```bash
scp scripts/card_log.py docs/data/2026-11-21_c942_card_log_cost/run/rig_cost.sh release/flint-v4/settings.env <rig>:/var/tmp/c943/
ssh <rig> 'cd /var/tmp/c943 && timeout 50m env BIN=<engine folder> MODELS=<model folder> PRESET=/var/tmp/c943/settings.env bash rig_cost.sh'
```

## Files

| File | What |
|---|---|
| [prof_today.txt](prof_today.txt) | today's sample, part by part, on the laptop's ctypes stand-in (two runs) |
| [prof_fast.txt](prof_fast.txt) | today's sample against `--fast` and `--slow-every 4`, the same harness (two runs) |
| [run/prof_today.py](run/prof_today.py), [run/prof_fast.py](run/prof_fast.py) | the harnesses: `python3 -I run/prof_today.py scripts/card_log.py 9 3000` |
| [run/took_by_state.py](run/took_by_state.py) | `took_ms` by what the cards did: `python3 -I run/took_by_state.py <cards.csv.gz> …` |
| [run/rig_cost.sh](run/rig_cost.sh), [run/rig_cost_dryrun.sh](run/rig_cost_dryrun.sh) | the rig check and its dry run (`bash run/rig_cost_dryrun.sh`) |

The rig logs read here are the published runs' own `cards.csv`; this folder adds no rig data. Licence: as the
repository.
