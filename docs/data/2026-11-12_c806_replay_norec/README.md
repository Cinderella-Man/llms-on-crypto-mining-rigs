# A replay names the cases its set does not hold (C806, Q1412, 2026-11-12)

What this is: a fix to the native check tool, not to the engine. A check set is recorded once, on a build machine, with
one build, and replayed on a P106 with `test-backend-ops` ([C372](../2026-11-04_replay_checks/README.md)). A later
build of the same line can carry more test cases than the set: a branch merged after the recording adds cases, and the
same filter then selects them too. The replay found no record for those cases and counted them as one failure of the
run. So [`scripts/replay_checks.sh`](../../../scripts/replay_checks.sh) marked the run `WRONG` even when every
recorded case passed. Twice on 2026-11-12 this cost a rerun and a check by hand:

- [R112](../2026-11-12_r112_v4day_replay/README.md): the `v4day` set, recorded with `pascal-p106-v4-b12246-day`, on the
  `-kern` build. Its 14 `dmt` runs pass all 315 recorded cases each, but `-kern` also selects C728's 30 K-tile cases
  (IQ4_XS and Q4_K at K = 6,144, 1-5 tokens). All 14 runs read `WRONG`, and the suite ran again on `-day`.
- [R73 part 1](../2026-11-12_r73_qsa_prompt_replay/README.md): the `qsa-prompt` set on `-kern`. Its 9 passing runs
  pass all 72 recorded cases each, but `-kern` also selects C103's 50 cases, which another set holds. All 9 read
  `WRONG`.

Registry row [Q1412](../../../quants/experiments.md). The fix has two parts: the replay names those cases, and the
script tells them apart from failing ones. Neither changes a verdict on a recorded case. On a P106
([R129](#on-a-p106-rig-row-r129)) the `v4day` set replays on the new build in one run: R112's 14 `dmt` runs read
`ok, 30 unrecorded` with the 30 named, and a run with failing cases still reads `WRONG`.

## The change

**`test-backend-ops`** (fork branch `pascal-p106-v4-b12246-c806`, one commit on `pascal-p106-v4` `c96561876`, merged
into `pascal-p106-v4`; only `tests/test-backend-ops.cpp` changes):

- Each case the filter selects that the record does not hold gets its own verdict, `NOREC`. The output line is
  `<case>: NOREC (not recorded) [<the record's meta file>]`, and the replay log (`--replay-log`, the per-case table
  `replay_checks.sh` writes as `<run>.tsv`) gets the line `<case id>\t<case>\tNOREC\t` (no errors).
- That happens with and without `--replay-allow-missing`. Before, the flag left them out silently.
- The summary line and the exit status stay as they were: without the flag, a run with unrecorded cases still ends with
  `N selected by the filter have no record (FAIL: the record does not cover this filter)` and exit 1. A script that
  calls `test-backend-ops` with `GGML_TEST_REPLAY` directly still sees that the set does not cover its filter.
- Builds before it print `<case>: not recorded [<meta>]`, and nothing under `--replay-allow-missing`.

**`scripts/replay_checks.sh`**:

- It reads the unrecorded cases' count from the replay summary and their names and ids from their lines (both forms).
- It sets apart the summary's FAIL line for them. All other FAIL lines count as before.

| The run | Before | Now |
|---|---|---|
| must pass; every recorded case passes; no unrecorded case | `ok` | `ok` |
| must pass; every recorded case passes; N unrecorded | `WRONG` | `ok, N unrecorded`, the N named |
| must pass; a recorded case fails (with or without unrecorded ones) | `WRONG` | `WRONG` (`, N unrecorded` added) |
| must pass; unrecorded cases and none replayed | `WRONG`, or `ok` under `ALLOW_MISSING` | `WRONG` (`NO CASE REPLAYED`) |
| fault run; a recorded case fails | `ok` | `ok` |
| fault run; no recorded case fails, N unrecorded | `ok` (the no-record line counted as the fault) | `WRONG` |
| known failures (the run list's seventh field) and N unrecorded | `WRONG` | `ok, N unrecorded` when only the known cases fail |

- Every run with no unrecorded case gets the verdict it got before: the same rule, line for line.
- `NOREC=wrong` makes any unrecorded case `WRONG`. Use it for a replay on the build that recorded the set, where every
  selected case must have its record.
- Output:
  - The console line ends in `ok, 30 unrecorded`. The names follow it, once for each list (`    NOREC <case>`); a run
    with the same list says `    the same 30 unrecorded cases as dmt-on`. Two cases can share a case string (their ids
    differ), so a name can show twice.
  - `results.tsv` gains a tenth column, `unrecorded`, and `unrecorded.tsv` lists each run's unrecorded cases (run, id,
    case).
  - The last console line adds the number of runs with unrecorded cases.
  - The verdict column of `results.tsv` stays `ok` or `WRONG`.
- Table scripts that count the lines of a `<run>.tsv` as replayed cases should count the `OK` and `FAIL` lines: the
  `NOREC` lines are cases that did not run.

No switch: this is test code and a script, and the engine is `c96561876`'s (below).
[docs/switches.md](../../switches.md) lists `--replay` among the tools.

## Checks on a build machine's CPU

Driver: [run/checks.sh](run/checks.sh), with every build and test in a container without GPUs or network, at a low CPU
share, under one lock. The cards were not used.

| Check | Result |
|---|---|
| The diff | Against `c96561876`, only `tests/test-backend-ops.cpp` changes (17 lines in, 8 out). No file under `ggml/src/ggml-cuda` differs from `-c805` `dd755a09c`. |
| The rig tarball | Built (`TESTS=1`, sm_61): md5 `fca33ab691240ca5d1c292e4aaa1a337`, 0 AVX, GLIBC_2.38, `test-backend-ops` md5 `2d2d8225aafb4f53cc49682caa0333d9`, the `-kern` build's 34 warning lines. |
| GPU code | `llama-server`'s SASS and PTX, cut into 8,540 functions: the same dump as the checked `-kern` build's (`d097d6d88`) and `-c805`'s. |
| ctest | 70 of 70, static and shared. Not run, as in C565: the tests that download a model or need Python with torch, and the one that needs the network. |
| C372's replay tests on the new build | t1 to t7 pass. t5 counts "not recorded" lines, and the NOREC lines keep those words. |
| The replay on a CPU (`n0` to `n3`) | A record of 49 cases, replayed with a filter that selects 53. **New build:** 4 NOREC lines, the 4 q5_0 cases; the replay log has 49 `OK` lines and 4 `NOREC`; the summary's FAIL line; exit 1. With `--replay-allow-missing`: the same 4 named, exit 0. **The base:** 4 "not recorded" lines and no log line for them; nothing under the flag. A made-up value failure (one case's references doubled plus one, its md5 updated in the meta): one `ERR > limit` FAIL line, exit 1. |
| The script on a CPU (`n4`, `n5`) | 15 runs, for each of the new and old script on each of the new and old build ([tool/verdicts.tsv](tool/verdicts.tsv)). The new script gives every run the verdict written for it, on both builds. The old script gives the same verdict to every run without unrecorded cases. The two differ only on `a-wide` and `b-known` (`WRONG` → `ok`) and on `a-fault` and `a-only-am` (`ok` → `WRONG`). With `NOREC=wrong`, a run with unrecorded cases is `WRONG`. |
| The script on R112's and R73's P106 logs (a stub replays them) | **The stub is faithful:** the old script gives each saved `results.tsv` again, byte for byte (R112 on `-kern`: 71 runs; on `-day`: 14; R73: 11). **R112 on `-kern`, new script:** the 14 `dmt` runs go from `WRONG` to `ok` with 30 unrecorded each, the same 30 in each: IQ4_XS and Q4_K at K = 6,144 and 1-5 tokens, none of them among the ids the replay ran. Every other column of every run is the saved one's. **On `-day`:** 14 `ok`, 0 unrecorded, as before. **R73:** the 9 passing runs go from `WRONG` to `ok`, 50 unrecorded each, the 50 R73 published; the two fault runs stay `ok`. |
| Made-up failures on those logs | `dmt-on` with one case turned failing: `WRONG, 30 unrecorded`. `qp-stock` the same: `WRONG, 50 unrecorded`. `qp-fault2` as it ran: `ok`. `qp-fault2` with its 59 failing cases turned OK, so the fault is not caught: `WRONG` (the old script says `ok`). |
| The rig row on a CPU | The `v4day` set, replayed by the tip's CPU build with `rig_runs.tsv`'s `dmt-on` and the two `dmt-q3f` runs: 295 and 462 cases replayed, all passing, and 30 unrecorded in each, R112's 30. A CPU declines the 20 and 23 large-activation cases (`big=250000`) that a card runs, so on a P106 the counts are 315 and 485. The fault switch acts only on a card. |
| The merge | `pascal-p106-v4` `3ab5c876b`: its tree is `-c806`'s, its parents `c96561876` and `74881101b`. |

Two lines of [checks.tsv](checks.tsv) read FAIL. Both were errors in the checks themselves, fixed and run again:
- `n4 names` compared a de-duplicated list of names with one that was not. Two of the 4 unrecorded cases share one case
  string but have different ids (the id counts the earlier cases with the same string). The rerun compares ids.
- `setcpu` first expected 315 and 485 cases, but a CPU declines the large-activation ones.

## On a P106 (rig row R129)

**The replay reads right on a card: every run gets the verdict written for it, and the console ends with
`15 runs as expected, 1 WRONG`** (the one is `dmt-q3f-pass`, which must read `WRONG`).

The set `v4day` (R112's, recorded with `pascal-p106-v4-b12246-day`) replayed on card 0 of a P106-090 rig on 2026-11-12,
17:58:31–18:02:37 (4 min 6 s), with:

- the rig tarball `llama.cpp-20261112-74881101b-pascal-p106-v4-b12246-c806.tar.gz` (678,328,540 B, md5
  `fca33ab691240ca5d1c292e4aaa1a337`, sm_61, no AVX; `test-backend-ops` md5 `2d2d8225aafb4f53cc49682caa0333d9`), md5
  checked on both ends of the copy;
- [`scripts/replay_checks.sh`](../../../scripts/replay_checks.sh) as C806 left it (md5 `c723029fe001f5544780b2612be38f05`);
- [run/rig_c806.sh](run/rig_c806.sh) on the 16 runs of [run/rig_runs.tsv](run/rig_runs.tsv):
  - R112's 14 `dmt` runs, as the set's `runs.tsv` has them;
  - `dmt-q3f-pass` and `dmt-q3f-fail`: the `dmt` filter joined with the `q3kmt` suite's, under
    `GGML_CUDA_MMVQ_PASCAL_Q3K_MT_FAULT=1`. That gives the same 30 unrecorded cases beside recorded cases that fail,
    expected to pass and to fail.

| Runs | Cases replayed | Pass | Fail | Unrecorded (`NOREC`) | FAIL lines, exit | Verdict |
|---|---|---|---|---|---|---|
| each of the 14 `dmt` runs | 315 | 315 | 0 | 30 | 3, 1 | `ok, 30 unrecorded` |
| `dmt-q3f-pass` (expect pass) | 485 | 382 | 103 | 30 | 106, 1 | `WRONG, 30 unrecorded` |
| `dmt-q3f-fail` (expect fail) | 485 | 382 | 103 | 30 | 106, 1 | `ok, 30 unrecorded` |

- **The 30 are named:** in full under `dmt-on`, then `the same 30 unrecorded cases as dmt-on` under each of the other 15
  runs ([rig/console.txt](rig/console.txt)). In every run they are R112's 30, C728's K-tile cases: IQ4_XS and Q4_K at
  K = 6,144 and 1-5 tokens, with m = 333 and 4,097, and m = 129 with `k_v` = 6,656. Each run's per-case table has
  exactly the 30 `NOREC` lines that `unrecorded.tsv` lists for it, and none of the 30 ids is among the ids it replayed
  ([rig/counts.tsv](rig/counts.tsv)).
- **The FAIL lines:** 3 in each `dmt` run. Those are the replay summary's no-record line and the two lines a failing
  run ends with, which the script sets apart. The `dmt-q3f` runs have the same 3 plus their 103 failing cases. Those
  103 are q3_K cases, the same 103 in both runs ([rig/failing.tsv](rig/failing.tsv)).
- **Against R112 on `-kern`:** `-kern` has the same GPU code (8,540 of 8,540 functions), and every case of every run
  here has R112's verdict and R112's errors to every printed digit:
  - each `dmt` run's 315 cases, against R112's run of the same name;
  - in the two `dmt-q3f` runs, the 315 IQ4_XS and Q4_K cases against R112's `dmt-on`, and the 170 q3_K cases against
    R112's `q3kmt-fault` (its 103 failing cases included);
  - 0 differences in 5,380 replayed cases ([rig/vs_r112.tsv](rig/vs_r112.tsv)).
- The card read 0 MiB before and after the run. The kernel log had no Xid line before or after.

```bash
# on the rig, with the tarball unpacked in $D and the set in $SET (R112's v4day)
SET=$SET BIN=$D/llama.cpp-20261112-74881101b-pascal-p106-v4-b12246-c806/bin OUT=$D/out \
  CHECKS=<repo>/scripts/replay_checks.sh DEV=0 bash <repo>/docs/data/2026-11-12_c806_replay_norec/run/rig_c806.sh
# then, on any machine with the output copied off the rig and R112's -kern replay output at hand (CPU only)
RIG_OUT=<the run's out/replay> R112_OUT=<R112's -kern replay output> \
  R112_UNREC=docs/data/2026-11-12_c806_replay_norec/saved/unrecorded_r112_dmt.tsv OUT=<the tables' folder> \
  SET_DIR=<the set's folder on the rig> RUN_DIR=<$D> bash docs/data/2026-11-12_c806_replay_norec/run/r129_tables.sh
```

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check's line, in the order run (row, check, verdict, detail) |
| [rig_bin_md5.txt](rig_bin_md5.txt) | md5 of each file in the rig tarball's `bin/` |
| [tool/norec_tests.tsv](tool/norec_tests.tsv) | the CPU tests of the replay and the script (`run/norec_tests.sh`) |
| [tool/runlist.tsv](tool/runlist.tsv), [tool/verdicts.tsv](tool/verdicts.tsv) | the 15 runs (set, switches, filter, expect, known failures, the verdict written for them) and each script and build's verdict |
| [tool/console_A.txt](tool/console_A.txt), [tool/console_B.txt](tool/console_B.txt), [tool/console_A_old_build.txt](tool/console_A_old_build.txt) | the new script's console on both sets, and on set A with the base's build (unrecorded cases named except under `ALLOW_MISSING`) |
| [tool/n1_new_lines.txt](tool/n1_new_lines.txt), [tool/n1_old_lines.txt](tool/n1_old_lines.txt), [tool/n1_new_replay_log.tsv](tool/n1_new_replay_log.tsv) | `test-backend-ops`' own lines for the unrecorded cases, new and base, and the new replay log |
| [saved/saved_logs.tsv](saved/saved_logs.tsv) | the script on R112's and R73's logs (`run/saved_logs.sh`) |
| `saved/compare_<replay>.tsv` | per run: the saved verdict, the new one, the unrecorded count, whether the other columns are the saved ones |
| `saved/console_<replay>.txt`, `saved/results_<replay>.tsv` | the new script's console and `results.tsv` on each replay's logs |
| [saved/unrecorded_r112_dmt.tsv](saved/unrecorded_r112_dmt.tsv) | the 30 cases R112's `dmt` runs selected that `v4day` does not hold (id, case) |
| [saved/console_fake.txt](saved/console_fake.txt), [saved/console_fake_old_script.txt](saved/console_fake_old_script.txt) | the made-up failures, new and old script |
| [setcpu_console.txt](setcpu_console.txt), [setcpu_results.tsv](setcpu_results.tsv) | the rig row's run list on a CPU |
| [run/checks.sh](run/checks.sh) | the driver on the build machine |
| [run/norec_tests.sh](run/norec_tests.sh), [run/saved_logs.sh](run/saved_logs.sh) | the CPU tests and the saved-log tests |
| [run/rig_c806.sh](run/rig_c806.sh), [run/rig_runs.tsv](run/rig_runs.tsv) | the rig row's script and run list |
| [rig/console.txt](rig/console.txt), [rig/results.tsv](rig/results.tsv), [rig/unrecorded.tsv](rig/unrecorded.tsv) | R129 on a P106: `replay_checks.sh`'s console (the rig's folders written as `$SET` and `$D`), its per-run table and its list of each run's unrecorded cases (run, id, case) |
| [rig/counts.tsv](rig/counts.tsv) | per run: the verdict, the per-case table's `OK`, `FAIL` and `NOREC` lines, and whether the `NOREC` lines are the listed 30 and R112's 30 |
| [rig/vs_r112.tsv](rig/vs_r112.tsv), [rig/vs_r112_diff.tsv](rig/vs_r112_diff.tsv) | each run against R112's replay on `-kern`, case by case (same verdict and errors, differing, missing), and the differing cases (none) |
| [rig/failing.tsv](rig/failing.tsv) | the 103 failing cases of each `dmt-q3f` run, with their errors |
| [rig/cases.tsv.gz](rig/cases.tsv.gz) | every case line of every run's per-case table (run, id, case, verdict, errors) |
| [run/r129_tables.sh](run/r129_tables.sh) | the script that made the `rig/` tables from the run's output and R112's |

## Commands

On a build machine (CPU only; the paths in the environment):

```bash
# M: a work folder; REPO: this repository; OLD_CHECKS: scripts/replay_checks.sh as it was before C806
# (git show b5b7e6d:scripts/replay_checks.sh); SEED_SRC: a llama.cpp clone with the fork as remote "fork";
# KERN_TREE and C805_TREE: the unpacked -kern and -c805 rig tarballs; R112_OUT, R112_DAY_OUT, R73_OUT: R112's and
# R73's replay output folders; V4DAY_TAR: the v4day set's tar; RIG_ROOT: the rig's model disk, as the saved
# consoles name it (written as $SET and $B in the tables)
M=... REPO=... LOCK=... OLD_CHECKS=... SEED_SRC=... KERN_TREE=... C805_TREE=... R112_OUT=... R112_DAY_OUT=... \
  R73_OUT=... V4DAY_TAR=... RIG_ROOT=... JOBS=8 STEPS='rig diff sass cpu tool saved setcpu c565 tables' \
  bash docs/data/2026-11-12_c806_replay_norec/run/checks.sh
# after the merge into pascal-p106-v4
M=... REPO=... LOCK=... SEED_SRC=... STEPS=merged MERGE=<the merge commit> bash docs/data/2026-11-12_c806_replay_norec/run/checks.sh
```

## Provenance and licence

- Code: the fork `Cinderella-Man/miner-llama` (MIT, as llama.cpp).
- The saved replay logs are this project's own runs, R112 and R73.
- Tables and scripts here: MIT.
