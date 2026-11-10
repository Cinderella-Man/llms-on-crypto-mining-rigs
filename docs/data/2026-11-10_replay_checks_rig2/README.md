# C441 / Q230, Q931, Q875: the `ladder3-rest` checks replayed natively on a P106-090 with the v3.0 release build (2026-11-10)

The first replay of a recorded `test-backend-ops` suite on a mining rig
([how record and replay work](../2026-11-04_replay_checks/README.md)). The suite is `ladder3-rest`: 1,916 recorded
cases and 63 runs, which hold the sparse decode attention for Pascal cards (registry Q230) with its three fault
codes, the grid types' and IQ1_M's k-column and multi-token kernels in six settings, the one-token grid-type kernels'
last runs, and 45 memcheck runs. It ran on one P106-090 (card 0 of a nine-card rig) with the test binary of the
release itself: Miner Llama v3.0, fork commit `1551f5fe1` (`pascal-p106-v3-b12076-ladder5`), a native `sm_61` build.
Until this run the sparse attention kernel had only run through the build's compute 6.1 PTX on newer cards.

Result pages: registry rows Q230 (the sparse kernel), Q931 (record and replay) and Q875 (llama.cpp's MMQ with
repeated expert ids) in [quants/experiments.md](../../../quants/experiments.md).

## Result

**Every sparse decode attention case passes on the P106.** All 63 runs took 947 seconds (15 min 47 s).

| Runs | What | Result on the P106 |
|---|---|---|
| `qsa-sparse1`, `qsa-sparse2`, `qsa-sparse0` | FLASH_ATTN_EXT with the `n_kv_max` hint, the switch `GGML_CUDA_FATTN_SPARSE_PASCAL` at 1, at 2 with up to 128 query tokens, and off | 48 of 48 cases pass in each |
| `qsa-fault1` | fault 1, the output times 1.01 | 48 of 48 pass, as the suite expects: the fault stays under the test's limit (largest error 0.000142832306 against 0.0005) |
| `qsa-fault2`, `qsa-fault3` | fault 2 (mask values ignored) and fault 3 (every split's last list entry dropped): both must fail cases | 29 of 48 cases fail in each, the same 29 (smallest failing error 0.00316041362 and 0.000840011764) |
| `iqkcol-on`, `-kcol_r2`, `-cap1`, `-rb1` | the grid types and IQ1_M in the k-column and multi-token kernels: defaults and three launch settings | 1,855 of 1,855, 861 of 861, 994 of 994 and 994 of 994 pass |
| `iq2xs-on-fusion`, `-on-shared`, `-skip-mm`, `-skip-mmid`, `-skip-fusion`, `-skip-shared` | the one-token grid-type kernels' last runs and their skip arm | 20 of 20, 6 of 6, 403 of 403, 223 of 223, 171 of 171 and 21 of 21 pass |
| `iqkcol-off`, `iqkcol-skip` | the same cases with the fork's kernels skipped, so llama.cpp's own code computes them | each stops at its 891st case with a CUDA illegal memory access; the 890 cases before it pass (below) |
| 45 `memcheck-*` runs | single cases under `compute-sanitizer --tool memcheck` | every case passes (1 to 3 a run), but the sanitizer does not work on this card, so there is no memcheck verdict (below) |

- **The sparse cases** are listed by name in [sparse_cases.tsv](sparse_cases.tsv) and as a table of their parameters in
  [sparse_cases.md](sparse_cases.md). The suite records 61 FLASH_ATTN_EXT cases; the CUDA backend declines 13 of them
  on any card (K heads of 128 and V heads of 96 values, [unmatched.tsv](unmatched.tsv)), so 48 run.
- **Which cases the sparse kernel computed:** a case's error against the stored CPU reference differs from the
  switch-off run's in 21 of the 48 cases with the switch at 1 and in 35 with it at 2 and up to 128 query tokens, so
  another kernel than the dense ones computed those. The 29 cases that faults 2 and 3 fail are among the 35, and the
  21 are among the 29. The largest error among the 21 is 0.0000444350045 (head size 256, 12 query heads per KV head, a
  cache of 32,768 cells, `n_kv_max` 2,051, F16), among the 35 it is 0.0000472637789; the limit is 0.0005.
- **The same verdicts as through the PTX.** On an RTX 3090 through the rig build's compute 6.1 PTX the six sparse runs
  gave 48, 48, 48, 48, 19 and 19 passing cases ([sets.tsv](../2026-11-04_replay_checks/sets.tsv)), as here.
- **The stored references are the rig's own.** `CPU_CHECK=200` ran 6 small cases of `iqkcol-on` on the rig's CPU
  as well: all 6 references are identical to the stored ones, value for value.
- **The kernel log** held no Xid line before the run and none after the six sparse runs.

### The two stock arms stop in llama.cpp's own code (Q875)

`iqkcol-off` and `iqkcol-skip` turn the fork's grid-type kernels off, so llama.cpp's stock path computes every case.
Both replay 890 cases, all passing, and stop at the next one,
`MUL_MAT_ID(type_a=iq2_xxs,type_b=f32,n_mats=28,n_used=10,b=0,m=256,n=6,k=2560,m_v=0,ids_offset=0,skip=0,dup=1)`,
with `CUDA error: an illegal memory access was encountered` (exit status 134). The kernel log shows Xid 13 (out of
range address) and Xid 31 (an MMU fault on a read) for the card at both stops. That case repeats an expert id within
a token at six tokens, which sends it to llama.cpp's MMQ kernels; upstream keeps its own repeated-id cases disabled
for CUDA, and no model's router repeats an expert within a token. The same two runs stopped the same way through the
PTX on an RTX 3090, so the fault is not the PTX's: a P106 shows it natively. Each fault ended only its own process,
and the 51 runs after them ran normally. The 965 recorded cases behind the stop were not reached in these two runs
(they pass in `iqkcol-on` with the fork's kernels); rerunning the two arms without the repeated-id cases is Q1190.

### compute-sanitizer does not work on a P106-090

Each of the 45 memcheck runs printed `Error: GPU debugging features are disabled. Please consult the list of
supported CUDA devices for more details.` and then ran its cases without the sanitizer. So these runs check the cases
once more (all pass) and say nothing about memory errors. `scripts/replay_checks.sh` marks them `WRONG`, because it
finds no `ERROR SUMMARY: 0 errors` line; in [runs.tsv](runs.tsv) the `log` column holds the sanitizer's message.
Memory checks of the P106 kernels therefore stay with the newer cards and the PTX, where the same 45 cases gave 0
errors (2026-11-04). Whether the card or a setting of the rig turns the debugging features off is Q1189.

### Cases the record does not hold

The release build's test file holds cases that were added after the suite was recorded on 2026-11-04. The runs'
filters select them too: 212 in `iqkcol-on`, `-cap1` and `-rb1`, 13 in `iq2xs-on-fusion`, 64 in `iq2xs-skip-mmid`, 38
in `iq2xs-skip-fusion` and 1 in each of 21 memcheck runs (`not_recorded` in [runs.tsv](runs.tsv)). With
`GGML_TEST_REPLAY_ALLOW_MISSING=1` they are left out instead of failing the run, so they are not checked here; every
recorded case was replayed. Recording them means running `scripts/record_checks.sh` again on the release tree, which
keeps the existing records.

## Files

| File | What |
|---|---|
| [runs.tsv](runs.tsv) | one line per run: switches, what passing means, recorded cases the filter selects, cases replayed, passed and failed, selected records not replayed, cases without a record, exit status, `replay_checks.sh`'s verdict, seconds, and the log's message where a run stopped or the sanitizer refused |
| [sparse_cases.tsv](sparse_cases.tsv) | the 48 sparse decode attention cases by name: OK or FAIL in each of the six runs and each compared node's error |
| [sparse_cases.md](sparse_cases.md) | the same 48 cases as a table of their parameters, with the error at switch 1, 2 and off and whether it differs from the switch-off run's |
| [cases.tsv.gz](cases.tsv.gz) | every replayed case of every run (7,679 lines: 7,621 OK and 58 FAIL, the 58 in the two fault runs): run, record id, case, verdict, errors |
| [unmatched.tsv](unmatched.tsv) | the 13 recorded cases no run replayed (the CUDA backend declines them) |
| [run/run_replay.sh](run/run_replay.sh) | the run on the rig: the six sparse runs first, then the other 57, two calls of [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) |
| [run/tables.py](run/tables.py) | builds the tables above from the replay's output folders |
| [run/relay_split.sh](run/relay_split.sh), [run/split_lists.sh](run/split_lists.sh), [run/recv_split.sh](run/recv_split.sh) | the suite onto a rig that has no single disk with room for it: one pass of the tar over the LAN, unpacked onto three disks, md5 checked, joined by symlinks |

## Method

**The suite on the rig.** The suite's tar is 13,218,580,480 bytes, and the rig's three disks had 6.5, 8.6 and 4.1 GB
free. `run/split_lists.sh` splits the tar's members into three lists (5.60, 5.20 and 2.41 GB, a case's three files
together). `run/relay_split.sh` sends the tar once over the LAN; on the rig `run/recv_split.sh` takes the stream's
md5 and three `tar` readers unpack one list each. The md5 on arrival equalled the tar's own
(`e40f70841a0895ab81e48114609aa653`), the cases on the second and third disk were symlinked into the first disk's
`cases/` folder (5,748 entries, none dangling), and each replayed case checks its two blobs' md5 again. The relay took
12 minutes; the slowest disk, a USB stick, set the pace. The run used a copy of `relay_split.sh` with the hosts and
paths written in.

```bash
CLIENT='ssh build' RIG='ssh rig' RIG_ADDR=<the rig's LAN address> TAR=<sets>/ladder3-rest.tar WORK=<scratch> \
  D1=/mnt/models/replay D2=<a folder on the second disk> D3=<a folder on the third disk> \
  C1=5600000000 C2=5200000000 bash docs/data/2026-11-10_replay_checks_rig2/run/relay_split.sh
```

**The run**, detached on the rig, with nothing else running on it (no llama process, 0 MiB used on every card):

```bash
W=/mnt/models/replay
SET=$W/ladder3-rest BIN=/opt/llama-builds/llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5/bin \
  D=$W/out-ladder3-rest CHECKS=$W/replay_checks.sh DEV=0 CPU_CHECK=200 \
  setsid nohup bash $W/run_replay.sh > $W/out-ladder3-rest.log 2>&1 < /dev/null &
```

`run_replay.sh` runs the six sparse runs first, on a card nothing else had touched since boot, then the other 57 in
the suite's order, all with `GGML_TEST_REPLAY_ALLOW_MISSING=1`. The binary is the release folder's own
`bin/test-backend-ops` (md5 `2c7f4054190d271eae3696792d487dd2`); the record and replay code is part of `1551f5fe1`.
The suite was recorded on 2026-11-04 by a CPU-only build of fork commit `a89d6b281` with the rig build's CPU flags.

**The tables**, on the build machine's CPU from a copy of the output folder:

```bash
OUT="<out>/qsa <out>/rest" SET=<the suite's folder: runs.tsv, manifest.tsv> DST=tables \
  nice -n 19 python3 docs/data/2026-11-10_replay_checks_rig2/run/tables.py
```

Seconds per run come from the modification times of the runs' logs: the six sparse runs took 89 s, the twelve
grid-type runs 758 s (`iqkcol-on` 175 s with the CPU check) and the 45 memcheck runs 100 s.

## Provenance and licence

The tables are this project's own measurements, published under the repository's licence. The cases are llama.cpp's
`test-backend-ops` cases (MIT) and the fork's additions to them; their inputs are random draws stored in the record.
The raw logs (each run's output and the kernel log's lines) are not published.
