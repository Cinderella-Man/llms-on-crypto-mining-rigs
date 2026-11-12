# The deep lane runs a load again after the RAM watchdog stops it (C778, Q1366, 2026-11-12)

What this changed: [bench/deep_lane](../../../bench/deep_lane/README.md)'s driver, `deep_lane.sh`, no longer ends a run
when the RAM watchdog stops a load. With `RETRY_STOPPED=1`, the default, it sets the stopped load aside, goes on with
the next load and starts the stopped load again after the others. Registry row
[Q1366](../../../quants/experiments.md).

Why: on the 4 GB rigs the watchdog ([scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh)) stops llama-server
when the host's free memory falls below its floor. Until now one such stop ended the whole run. The remaining loads
ran only after someone resumed the run by hand, with the stopped load's files set aside and the load moved to the
end. That happened in three runs:

- the threshold run of [2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md) (the first `m2` load at 4,096
  tokens, 41 MiB);
- [2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md) (`off1` at 32,768 tokens, 54 MiB);
- [2026-11-11_r92_upstream_defaults](../2026-11-11_r92_upstream_defaults/README.md) (`host2` at 49,152 tokens,
  57 MiB against a floor of 60).

Each stopped load ran through when it was started again. With this change the run itself starts it again, so the
rig's time is not lost waiting for someone to resume it.

Nothing here is a measurement: the checks run the driver against the mock server on a CPU, with no model and no GPU.

## What the driver does on a stop

When a load fails, the driver first asks whether the RAM watchdog stopped it: the watchdog's `watchdog:` line in the
server log, or its `KILL` line in its samples. If the server is already gone, the driver gives the watchdog up to 3 s
to write those lines and end. A failure the watchdog did not cause ends the run, as before. On a stop:

1. The load's files move to `OUT/stopped/<load>-<n>/` (`n` counts that load's stops: 1, then 2): `server-<load>.log`,
   `ram_watchdog-<load>.tsv`, `requests-<load>.csv` and `<load>.jsonl` as far as the load got, `quiet_<load>.txt`,
   `fit_<load>.txt`. `requests.csv` is rebuilt without the stopped requests.
2. `loads.tsv` gets a line for the stop: the load's own columns, with `stopped` in the `timed` column. Its `n`-th stop
   line goes with the folder `stopped/<load>-<n>`.
3. The run goes on with the next load.
4. After the other loads, each stopped load starts again, in the order they stopped. A stopped warm-up load is not
   started again.
5. A second stop of the same load in one run ends the run (`failed`), with the second stop's files in
   `stopped/<load>-2/`.

With `RETRY_STOPPED=0` the first stop ends the run and leaves its files in place, as before this change.
`remote.sh` passes the variable on to the rig.

`deep_table.py` reads the stop lines apart from the loads. A load listed more than once counts with its last start,
and the tables list the loads in the order they ran. The table's last check lists every stop: the load, the last
request it answered (or "while loading"), the watchdog's reason, its folder and whether the load ran again. A run
without a stop writes no `stopped/` folder and no stop line, and its table has no such check.

One more fix in the same change: when the watchdog stopped the server at its very first sample, the driver could
report that the watchdog "did not come up". It now reads the sample first.

## The checks

[selftest.sh](../../../bench/deep_lane/selftest.sh) runs the driver against
[mock_server.py](../../../bench/agent_check/mock_server.py), with the real `ram_watchdog.sh` reading a stand-in for
`/proc/meminfo`. The stand-in shows 3,900 MiB available, and 50 MiB (below a floor of 60) on the starts a case names:
from the watchdog's first sample, or from the moment a load has written a given request. The `host` arm decodes at
40 ms a token instead of 0.2, so that a stop in the middle of its requests lands seconds before its last request.

| Case | What it checks |
|---|---|
| `stop1` | `ARMS="base host"`, two loads each. The watchdog stops `host1` after its decode pass 1 at depth 512, and `base2` at its first sample. The run is done; the loads start in the order warm, base1, host1, base2, host2, then host1 and base2 again; `loads.tsv` has a `stopped` line for each; `stopped/host1-1` holds the server log with the watchdog's line, the watchdog's samples with its `KILL`, 5 requests, and the quiet and fit files; `stopped/base2-1` holds no requests and no fit file; the loads started again have files of their own; `requests.csv` holds 51 requests, each load once; the table lists the loads in the order they ran and both stops; `host` is classed on its two loads |
| `stop2` | the watchdog stops `host1` on both starts: the run ends at the second stop, after `base2` and `host2` ran between the two; both stops are set aside and listed |
| `stop0` | `RETRY_STOPPED=0`: the first stop ends the run with the message it had before this change, its files in place, no `stopped/` folder |
| `rig` | one check added to the existing case: no stop, so no `stopped/` folder and no stop line |

Every case that was there before still passes. All 49 checks pass with this change, and the version before it passes
its 36 ([selftest.txt](selftest.txt), [selftest_before.txt](selftest_before.txt)).

A run without a stop behaves as before. [run/test_c778.sh](run/test_c778.sh) runs the self-test of the version before
the change and of this one in the same folder, so that every path in their files is the same, and compares every case
both ran file by file ([compare.txt](compare.txt)). Before comparing, it takes out the clock and the host's free
memory: ISO times, the console's HH:MM:SS, `loads.tsv`'s `started` and `load_s`, `requests.csv`'s `wall_s` and
`time`, the `.jsonl` files' `time` and `wall_s`, a server's load seconds, and `MemAvailable` and `SwapFree`.

The result: the 7 case folders both versions wrote (`rig`, which `resume` and `other` reuse, `busy`, `ptx`, `fresh`,
`short`, `ctx` and `table`) hold the same files, 76 of them in `rig`, with the same content, and the consoles of all
8 cases are the same. `table.md`, `classes.tsv`, `prompts.json` and `settings.md5` are the same byte for byte.

| Self-test | Checks | Time on one core at `nice -n 19` |
|---|---|---|
| before the change (`406a939`) | 36 of 36 | 40 s |
| with it | 49 of 49 | 121 s (the stop cases' slower arm) |

The table of `stop1`, as the driver wrote it: [stop1_table.md](stop1_table.md).

## Files

| File | What it is |
|---|---|
| [selftest.txt](selftest.txt) | the self-test's output with this change: 49 checks |
| [selftest_before.txt](selftest_before.txt) | the self-test's output before it: 36 checks |
| [compare.txt](compare.txt) | the cases both versions ran, compared file by file |
| [stop1_table.md](stop1_table.md) | `deep_table.py`'s table of the `stop1` case (mock speeds, not measurements) |
| [run/test_c778.sh](run/test_c778.sh) | the run: both self-tests and the comparison |

## Commands

On a Linux machine with bash, curl and Python 3.9 or newer (here Python 3.14, at `nice -n 19`). `NEW` and `OLD` are
two copies of `bench/deep_lane`, `bench/agent_check/mock_server.py`, `release/flash-next-v3/settings.env` and
`scripts/ram_watchdog.sh`: `NEW` from this change, `OLD` from the public `v4` before it (`406a939`):

```bash
git archive 406a939 bench/deep_lane bench/agent_check/mock_server.py release/flash-next-v3/settings.env \
  scripts/ram_watchdog.sh | tar -C old -xf -
NEW=new OLD=old OUT=<results folder> nice -n 19 bash docs/data/2026-11-12_c778_lane_retry/run/test_c778.sh
```

The self-test alone: `bash bench/deep_lane/selftest.sh <work folder>`.

Provenance and licence: the self-test writes its own recorded text from a fixed word list, so nothing here comes from
another source. The files are this repository's, under its licence.
