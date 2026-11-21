# C873: the card log in the remaining drivers, and the watchdog's clock (2026-11-21)

**What changed, in plain words.** The record of every card twice a second (clock, memory clock, power, temperature,
power state, memory used, how busy, and why the driver held the clock back) now runs in the rest of the scripts that
time things on the rig: the prompt replay that showed slow stretches in some passes, `scripts/server_bench.sh`, the
release matrix and four of the pipelined-speculation scripts. Where they used to start `nvidia-smi` once a second, they
now use the same read-only logger as the agent check and the deep lane, and they log the same way: it starts before
the first load, writes `cards.csv` beside the script's other outputs, and stops when the run ends, also when the run is
killed. Everything else the scripts do is unchanged. Separately, the clock-offset watchdog's log now gives each row's
time from one clock reading and adds the monotonic clock, so its rows can no longer appear to step back in time.

Registry: [Q1576](../../../quants/experiments.md) (this row), from [Q1464](../../../quants/experiments.md) (C832,
[2026-11-17_c832_card_logs](../2026-11-17_c832_card_logs/README.md)); new rows
[Q1681](../../../quants/experiments.md) (the rig scripts that still log nothing) and
[Q1682](../../../quants/experiments.md) (the logger's CPU cost on the rig, below).

## The drivers

| Driver | How it carries the logger | Its log | Off with | Before |
|---|---|---|---|---|
| R43's prompt replay, [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (what [c222.sh](../2026-11-20_c222_flash_microbatch/run/c222.sh), `r43_run.sh` and `r135_load.sh` start on the rig) | embedded byte for byte: the launchers copy this one script to the rig | `OUT/cards.csv` | `CARD_LOG=0` | nothing through the run |
| [scripts/server_bench.sh](../../../scripts/server_bench.sh) | embedded (it is read over ssh or copied alone) | `/tmp/server_bench/<TAG>/cards.csv` | `CARD_LOG=0` | nothing through the run |
| [bench/release_matrix/run_matrix.sh](../../../bench/release_matrix/run_matrix.sh) | `card_log.py` beside it (a copy of the folder carries it) | `cards-<block>.csv`, one per block | — | `nvidia-smi -lms 1000` to `smi-<block>.csv` |
| [bench/pipespec/](../../../bench/pipespec/) `rig_run.sh`, `rig_streams.sh`, `rig_hostloop.sh`, `rig_prefill.sh` | `card_log.sh` and `card_log.py` beside them | `out/<TAG>/cards.csv` | `CARD_LOG=0` (`GPU_LOG=0` for `rig_run.sh`) | `nvidia-smi -l 1` to `gpu.csv` |

- **Start and stop** as in the agent check's [rig_server.sh](../../../bench/agent_check/rig_server.sh): the driver waits
  up to 10 s for the first sample and says in its console (`run.log` for the replay, `matrix.log` for the matrix)
  either `card log: every card every 0.5 s to cards.csv` or `card log: not running: …` (then `card_log.out` beside the
  log says why, and the run goes on without it). It passes its own process as `--parent` and a stop file
  (`cards.csv.stop`), so the logger ends within 0.5 s of the driver even when the driver is killed with SIGKILL, and a
  session can end it by creating that file. At the run's end the driver sends SIGTERM (the logger writes its stop line:
  samples, CPU seconds, sampling time) and waits up to 3 s, never on a logger that hangs.
- **The release matrix's "highest memory clock"** (`meta.tsv` and `tables.md`) is now read from the block's card log,
  over the cards in `GPUS`, as it was read from `smi-<block>.csv`.
- **Nothing reads the old files.** No table script in the repository reads the matrix's `smi-<block>.csv` or the
  pipespec scripts' `gpu.csv`; the readers of the older `gpu.csv` name the agent check's and the deep lane's, which
  C832 replaced. The launchers of the next replay runs copy `rig_taper.sh` and read only `env.txt`, `run.log` and
  `DONE`; their one-off `nvidia-smi` checks (compute processes, memory before a run) stay, as do the replay's own
  one-off files (`quiet_check.txt`, `memory_loaded.txt`, `memory_after.txt`).

## The watchdog's clock

[scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py) wrote each row's time as the second from one clock call
and the milliseconds from a second call, so a row written across a second boundary read about 1 s early (3 of R194's
2,480 rows). Each row's `time` now comes from one reading, and a last column, `mono_s`, holds the monotonic clock at
the same moment: it never steps back, so it orders the rows and times the gaps even when the wall clock is set. The
power, trip-spacing, `--duration` and sampling timers run on it too. A log that was started without `mono_s` is
appended to without it, so its columns stay the same. Every reader of the watchdog's log in this repository finds its
columns by name or counts them from the start, so the new last column changes none of them.
[scripts/card_table.py](../../../scripts/card_table.py) orders a log with `mono_s` by it, placed on the wall clock by
the median offset between the two (per boot). [watchdog_step.txt](watchdog_step.txt): on a 3.1 s stand-in log whose
wall clock was set back 2 s halfway, the table counts card 0 at P0 for 3.1 s with `mono_s` and for 2.0 s without it.

The watchdog that runs on the rigs is the copy in the memory-offset kit (`bench/mem_offset/build.sh` packs it), so the
rigs keep the old time stamps until the kit is rebuilt and installed.

## The logger's cost on the rig (from the logs it already wrote)

Since C832 the agent check and the deep lane carry the logger. Its stop lines on 2026-11-20 give its own CPU on the
2-core Celeron host, nine cards every 0.5 s:

| Run | Length | Samples | Logger CPU | Of one core | CPU a sample | Sampling time, median and max |
|---|---|---|---|---|---|---|
| R53 (deep lane) | 1,837.1 s | 3,675 | 128.56 s | 7.00 % | 35.0 ms | 39.2 ms, 197.2 ms |
| R175 (deep lane) | 1,712.5 s | 3,425 | 120.29 s | 7.02 % | 35.1 ms | 36.5 ms, 289.7 ms |
| R227, arm b (agent check) | 1,538.7 s | 3,078 | 91.04 s | 5.92 % | 29.6 ms | 29.3 ms, 427.0 ms |

C832 had estimated a few milliseconds a sample. The 72 reading calls a sample (eight per card) take about 0.5 ms each
on this host, so the logger takes about 7 % of one core for the whole run, and no sample started late. Whether that
slows a timed run, and how to make a sample cheaper, is [Q1682](../../../quants/experiments.md). A comparison inside
one run (arms in the same load, as the replay's passes) has the logger under every arm alike.

## The tests

No rig and no GPU: the laptop, `python3 -I`, every driver run in a scratch folder with stub programs and the logger on
[scripts/card_log_standin.py](../../../scripts/card_log_standin.py), a stand-in for NVML (the one C832's self-test
uses, as a reusable script: no card and no NVIDIA library is touched; the laptop's own `nvidia-smi` is replaced by a
stub that records its calls).

- [selftest.txt](selftest.txt): [scripts/card_log_test.sh](../../../scripts/card_log_test.sh), **63 of 63** in 31.8 s.
  C832's 26 checks; every embedded and copied logger equal to `scripts/card_log.py` (6); no `nvidia-smi` poll left in
  the ten drivers that log (10); the watchdog's clocks (7); and the drivers' dry runs (16): the replay driver (a normal
  run: the log starts before the load, 26 rows, stops with SIGTERM at the end, `DONE` written, its loads, files and
  md5 line as before; NVML missing: reported, the run goes on; `CARD_LOG=0`: no log; killed with SIGKILL: the log
  stops by itself, `parent … is gone`), the four pipespec scripts from a copy of their folder (plus `GPU_LOG=0`), and
  `server_bench.sh` (a normal run with the log starting before the load, and a server that dies while loading: the
  log still ends).
- [matrix_dryrun.txt](matrix_dryrun.txt): [bench/release_matrix/test/dryrun.sh](../../../bench/release_matrix/test/dryrun.sh),
  **22 of 22** in 3 min 25 s: its 16 checks as before (one widened from "no server left" to "no server or logger
  left"), and 6 new: the memory clocks read from the card logs (4,004 MHz untuned and 4,304 MHz tuned, the stand-in
  following the stub offset), the log started in both blocks, a log per block with rows for 3 cards, each stopped with
  its block, no `nvidia-smi` poll, and no logger left after SIGTERM.
- Every changed shell script passes `bash -n`.

## What to check in each driver's first run on a rig

1. The console says `card log: every card every 0.5 s to cards.csv` (or `cards-<block>.csv`) before the first load.
2. `cards.csv`'s first `#` line: `driver 580.178.04`, `9 cards (NVIDIA P106-090)`, each card's limit and offsets as
   the rig is set (its overclock profile, if one is applied).
3. About two rows a second, every field filled for all nine cards.
4. The last `#` line `card_log stop (signal): …`: CPU seconds near 7 % of the run's length, as above; many late
   samples would be new.
5. `python3 -I scripts/card_table.py cards.csv --since <pass start> --until <pass end>` beside a slow stretch of a pass:
   any stretch of 2 s or more below 1,800 MHz, and with which reasons.

## Files

| File | What |
|---|---|
| [selftest.txt](selftest.txt) | `bash scripts/card_log_test.sh`, 63 of 63 |
| [matrix_dryrun.txt](matrix_dryrun.txt) | `bash bench/release_matrix/test/dryrun.sh`, 22 of 22 |
| [watchdog_step.txt](watchdog_step.txt) | `card_table.py` on a watchdog log whose wall clock stepped back, with and without `mono_s` |

The tools are in the repository: [scripts/card_log.py](../../../scripts/card_log.py),
[scripts/card_log_standin.py](../../../scripts/card_log_standin.py), [scripts/card_log_test.sh](../../../scripts/card_log_test.sh),
[scripts/card_table.py](../../../scripts/card_table.py), [scripts/nvml_watchdog.py](../../../scripts/nvml_watchdog.py),
[bench/pipespec/card_log.sh](../../../bench/pipespec/card_log.sh).

## Commands

```bash
bash scripts/card_log_test.sh                       # sync, static, fake, watchdog, drivers: no GPU, about 30 s
KEEP=1 SECTIONS=drivers bash scripts/card_log_test.sh   # keeps the dry runs' logs and names the folder
bash bench/release_matrix/test/dryrun.sh            # about 3.5 min, most of it the script's own sleeps
# watchdog_step.txt: the self-test's watchdog runner (the WDRUN block of card_log_test.sh) saved as wd_run.py, then
python3 -I wd_run.py scripts/card_log_standin.py scripts/nvml_watchdog.py wd.csv TRIPPED 3 2
grep -v '^#' wd.csv | sed 's/,[^,]*$//' > wd_nomono.csv
python3 -I scripts/card_table.py wd.csv --tsv with.tsv; python3 -I scripts/card_table.py wd_nomono.csv --tsv without.tsv
# the logger's cost: the "card_log stop" line of each run's cards.csv (R53 and R175: the deep lane's OUT/cards.csv;
# R227: the agent check's rig/cards.csv)
```

## Provenance and licence

The scripts are this repository's. The cost table reads the stop lines of three card logs from 2026-11-20's rig runs
(their data folders: [R53](../2026-11-20_r53_flash_draft_prompt/README.md),
[R175](../2026-11-20_r175_flash_shortlist/README.md), [R227](../2026-11-20_r227_flash_taper1400_agent/README.md)).
