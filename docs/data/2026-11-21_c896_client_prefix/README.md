# Agent checks that share one client: containers named per run (C896; 2026-11-21)

Result: registry [Q1619](../../../quants/experiments.md) (follow-up Q1679).

The agent check's client ([bench/agent_check/client_arm.sh](../../../bench/agent_check/client_arm.sh)) runs the logging
proxy, pi and the tests in Docker containers. It named them `PREFIX-ARM-*` with `PREFIX=acrel` by default, and removed
every `PREFIX-ARM-*` container when an arm started and ended. Every Flash-Next run through R156's driver has the arm
`v4r`, so two such runs from two rigs on one client shared the names: on 2026-11-19 one run's client start removed the
other run's proxy in the middle of a task, and that run had to be repeated.

What changed:

- `client_arm.sh` names a run's containers after its results folder, `acrel-<folder>-<CRC-32 of its full path>-<arm>-*`,
  keeps the name in `<OUT>/<arm>.prefix`, and removes only those. A `PREFIX` given still wins. Before it touches
  anything it refuses to start, with the cause named and no rc file (so the driver following it stops too), when
  another `client_arm.sh` runs the same arm in the same folder, when another run's proxy holds the same container
  name (one `PREFIX` given to two runs; the proxy carries the label `ac-out=<OUT>`), or when its `PROXY_PORT` is in use.
- The drivers ([release_check.sh](../../../bench/agent_check/release_check.sh),
  [release_check_s12.sh](../2026-11-03_flash_next_mtp/run/release_check_s12.sh), which R156's driver and the later
  Flash-Next and Flint launchers call, and [race_rigs.sh](../../../bench/agent_check/race_rigs.sh)):
  - work the same name out from `CLIENT_OUT`, pass it to the client and print it (`client v4r: containers
    acrel-r215a-c0a73319-v4r-*`), so a client folder prepared before this change names its containers per run too;
  - remove the client's containers by that name when they are interrupted. Before, release_check_s12.sh removed
    `acrel-<arm>-*`, which never matched a run with its own `PREFIX`, and an interrupt in the second the client
    started left the client running on its own;
  - check every port before the relay starts (`RELAY_PORT` here, `CLIENT_PORT` and `PROXY_PORT` on the client) and
    stop with `nothing started: ...` when one is in use. An ssh forward whose port another check holds does not fail
    the run: with `ExitOnForwardFailure` it exits and is retried every 3 s, while the client's `/health` answers through
    the other check's forward, so the run would send its attempts to the other rig. release_check.sh and
    release_check_s12.sh also stop when the relay's log shows that a forward could not take its port (two runs started
    in the same second).

## Checks

Run on a machine with Docker and the sandbox image, its CPU only (`--cpu-shares 256`, nice 19), no GPU:

| Check | Result |
|---|---|
| `selftest_prefix.sh`, part 1: `client_arm.sh` as the drivers start it, against the mock | **16 of 16 pass** ([selftest_prefix.txt](selftest_prefix.txt)): one run alone exits 0 with every attempt scored, the same files as before plus `same.prefix` (`acrel-st-one-<8 hex>-same`), its proxy and attempt containers named with it and none left; A (3 tasks) and B (1 task) with the same arm and D with `PREFIX=acpfx` at once all finish with every attempt scored and no infrastructure try, A's and B's proxies ran at the same time and A's was up when B ended; C with A's `PROXY_PORT` and E with D's `PREFIX` and arm were refused before they started anything (the cause named, no rc file, no container) |
| `selftest_prefix.sh`, part 2: the drivers against a fake rig, `CLIENT=local` as on the laptop | **18 of 18 pass** (the same file; the whole test 06:06:43–06:11:53): release_check_s12.sh A2 (6 tasks) and B2 (1 task) with the same arm `v4r`, and release_check.sh R2 (an `ssh -R` relay, arm `rel`) at once all finish with every attempt scored and container names of their own, as printed in `run.log`; C2 with A2's `RELAY_PORT` stopped with `nothing started: RELAY_PORT ... is in use here` (no rig log folder, no client); D2, whose `RELAY_PORT` was bound but not listening, stopped with `the relay could not take its port (bind [127.0.0.1]:...: Address already in use)` before its client started; E2, stopped with TERM as its client started, and F2, stopped in the middle of an attempt, removed their own containers and stopped their rig server and relay; A2's proxy was up after each; no container and no port left |
| The same with `AC_CLIENT_BENCH`: part 2 with a client folder prepared on 2026-11-14 (its `client_arm.sh` names containers `PREFIX-ARM-*` with `PREFIX=acrel` by default) | **33 of 33 pass** ([selftest_prefix_oldclient.txt](selftest_prefix_oldclient.txt); 06:11:53–06:16:48): part 1 (16) as above; part 2 (17, all of the above but the check of this tree's `<arm>.prefix`): with the old `client_arm.sh` the containers still carried the names the drivers passed (`acrel-st-a2-<8 hex>-v4r-*` and the like), two runs with the same arm finished side by side, and the runs stopped with TERM removed only their own |
| `selftest.sh` with the Docker sandbox: the runner, resume, the turn cap, replay, the token mark, the reports | **passed**, 40 checks, 74 s ([selftest.txt](selftest.txt)) |
| `run/c896_compare.sh`: one run of `client_arm.sh` alone before and after, arm `same`, the first 6 bugfix tasks, the same mock (some tasks failed and some first streams dropped on purpose) | **The same on every task** ([compare.txt](compare.txt)): both exit 0 in 19 s; status, success, try and infrastructure tries equal on all 6 tasks (5 passed, `javascript-triangle` failed as the mock intends, `javascript-space-age` scored on try 2 after its dropped first stream), 5 turns each; the same files in `OUT` but the new `same.prefix`; containers `acrel-same-*` before, `acrel-new-<8 hex>-same-*` after; none left |

## Files

| File | What |
|---|---|
| `selftest_prefix.txt` | [selftest_prefix.sh](../../../bench/agent_check/selftest_prefix.sh)'s output with this tree's client (folder paths shortened to `<ROOT>`) |
| `selftest_prefix_oldclient.txt` | the same with `AC_CLIENT_BENCH`, a client folder prepared on 2026-11-14 |
| `selftest.txt` | [selftest.sh](../../../bench/agent_check/selftest.sh)'s output with the Docker sandbox |
| `compare.txt` | `run/c896_compare.sh`'s output: one run of `client_arm.sh` before and after, the same mock and settings |
| `run/chain.sh` | the three checks one after another (`bash run/chain.sh <ROOT>`) |
| `run/c896_compare.sh` | the single run before and after |

## Commands

```bash
# ROOT/new holds this repository's bench/agent_check, bench/task_results.py, bench/paired_tasks.py,
# scripts/realloc_counts.py and docs/data/2026-11-03_flash_next_mtp/run/release_check_s12.sh at their paths;
# ROOT/old the same paths from public v5 before C896 (git archive <commit> <paths>)
bash docs/data/2026-11-21_c896_client_prefix/run/chain.sh $ROOT

# the driver part with a client folder prepared earlier (agent_check/, task_results.py, paired_tasks.py,
# realloc_counts.py as a release check's prep leaves them)
AC_CLIENT_BENCH=$CLIENT_DIR AC_CPU_SHARES=256 nice -n 19 bash $ROOT/new/bench/agent_check/selftest_prefix.sh $ROOT/runs/prefix_old
```

## Provenance

Everything here is this repository's code against its own mock server (`bench/agent_check/mock_server.py`) and the
bugfix tasks built from Exercism's MIT-licensed exercises (`bench/agent_check/tasks/`); no model, no rig. Raw logs
are not published.
