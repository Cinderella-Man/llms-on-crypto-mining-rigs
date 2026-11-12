# The v4 day merge's kernel cases on a P106 (R112, Q1359, 2026-11-12)

The replay set `v4day` holds the CUDA cases of the seven kernel suites merged into the v4 line on 2026-11-11
([the day merge](../2026-11-11_v4_day_merge/README.md#the-replay-set-for-the-rigs), Q1359), recorded once on a CPU with
the merged build: 1,512 `test-backend-ops` cases with their CPU references, 71 runs. Here the set ran on one P106-090
(card 0 of a nine-card rig) with `pascal-p106-v4` at `d097d6d88`, the `-kern` rig tarball. That build holds the day
merge, four later host switches and [the kernel merge](../2026-11-11_v4_kernel_merge/README.md), every switch off by
default. It is the first time C726's, C750's, C754's, C757's and C767's kernels ran on a P106 against their recorded
references, and it stands for the replay step of each of those branches' rig rows.

| Suite | Branch, switch | Runs | Rig row |
|---|---|---:|---|
| `dmt` | C722: the IQ4_XS and Q4_K products of 1-5 tokens (`GGML_CUDA_MMVQ_PASCAL_DMT`), at C758's R = 1 default | 14 | R79 |
| `q5kmt` | C690: the Q5_K products of 1-4 tokens (`GGML_CUDA_MMVQ_PASCAL_Q5K_MT`), at C766's R = 1 default | 10 | R75 |
| `q3kmt` | C726: the Q3_K products of 2-8 tokens (`GGML_CUDA_MMVQ_PASCAL_Q3K_MT`) | 12 | R103 |
| `bf16x` | C750: the exact BF16 hyper-connection products (`GGML_CUDA_MMVF_PASCAL_EXACT`) | 9 | R97 |
| `f32x` | C754: the F32 products of 4-8 tokens on `mul_mat_vec_f` (`GGML_CUDA_MMVF_F32_MT`) | 6 | R99 |
| `glumt` | C757: one launch for the shared expert's gate, up and SWIGLU (`GGML_CUDA_MMVQ_PASCAL_GLU_MT`) | 9 | R107 |
| `q5k1r` | C767: the one-token Q5_K products on C690's kernel (`GGML_CUDA_MMVQ_PASCAL_Q5K_1R`) | 11 | R111 |

Result page: registry row Q1359 in [quants/experiments.md](../../../quants/experiments.md) (and Q1325, Q1330, Q1285,
Q1333, Q1348 for the five branches).

## Result

**Every case of the seven suites passes on the P106 in every run that must pass, and every fault run catches its
fault.** In the 64 runs that must pass, 14,200 case replays, all OK. All 7 fault runs fail cases. Every one of the set's
1,512 cases ran and passed in at least one run that must pass.

| Suite | Runs that must pass | Cases a run | Passed | Fault runs | Cases the fault fails |
|---|---:|---|---|---|---|
| `dmt` | 14 | 315 | 315 of 315 in each (4,410) | none | |
| `q5kmt` | 10 | 124 | 124 of 124 in each (1,240) | none | |
| `q3kmt` | 11 | 170 | 170 of 170 in each (1,870) | `q3kmt-fault` | 103 of 170 |
| `bf16x` | 7 | 152 in 6 runs, 8 in `bf16x-fault-off` | all (920) | `bf16x-fault`, `bf16x-fault2` | 8 of 8, 8 of 8 |
| `f32x` | 4 | 115 in 3 runs, 37 in `f32x-fault-off` | all (382) | `f32x-fault`, `f32x-fault2` | 37 of 37, 37 of 37 |
| `glumt` | 8 | 636 | 636 of 636 in each (5,088) | `glumt-fault` | 28 of 28 |
| `q5k1r` | 10 | 29 | 29 of 29 in each (290) | `q5k1r-fault` | 11 of 29 |

- **`dmt` and `q5kmt` at the R = 1 defaults** (`dmt-defaults`: `_DMT=1` alone; `q5kmt-defaults`: `_Q5K_MT=1` alone),
  and every other setting of the two suites, pass every case.
- **The `-fault-off` runs** set a fault switch without its kernel's switch, so the fault's code never runs: 8 of 8 and
  37 of 37 pass, as the run list expects.
- **The 33 large-activation products** (`big=250000`), which the CPU backend does not take, so that the CPU replay of
  the set could not judge them, pass in every run that must pass and selects them
  ([big_cases.tsv](big_cases.tsv)). That is 10 IQ4_XS and 10 Q4_K products in all 14 `dmt` runs, 10 Q5_K in the 10
  `q5kmt` runs (the 3 of one token in the 11 `q5k1r` runs too) and 3 Q3_K in the 11 `q3kmt` runs. Two of the Q3_K
  ones fail under `q3kmt-fault`, as they must.
- **The run's verdicts** from `scripts/replay_checks.sh`: 57 of 71 runs as expected with the `-kern` build. The other
  14 are the `dmt` runs, which pass all 315 recorded cases but each select 30 cases the set has no record of (below).
  With the `-day` build that recorded the set, the `dmt` suite gives 14 of 14 runs as expected.
- **The rig**: card 0 at 0 MiB before and after; no Xid line in the kernel log since the rig's boot, before or after.

### The `dmt` runs on `-kern`: 30 cases the set has no record of

The set was recorded with the `-day` build's `test-backend-ops` (`18d0fa309`, md5
`25ffc9b368c59545109504ec2e283005`). The kernel merge, which `-kern` adds on top, also adds C728's cases to
`test-backend-ops`: IQ4_XS and Q4_K products at K = 6,144 with 333 rows, with 4,097 rows, and with 129 rows that are
views of 6,656-wide rows, each at 1-5 tokens. That makes 2 × 3 × 5 = 30 cases. The `dmt` runs' filter
`type_a=(iq4_xs|q4_K),(type_b=f32,)?m=[0-9]+,n=[1-5],` selects them. The replay finds no record for them and counts
that as one failure of the run ("the record does not cover this filter"). So `replay_checks.sh` marks each `dmt` run
`WRONG`, though all 315 recorded cases pass in each. No filter of any other suite selects a case the set lacks.

- **The `dmt` suite again on `-day`** (`replay_checks.sh` with `RUNS` its 14 runs and `BIN` the `-day` build, whose
  `test-backend-ops` is the recording one): 14 of 14 runs as expected, 315 of 315 cases in each, none unrecorded
  ([results_day.tsv](results_day.tsv)).
- **The 30 cases on `-kern`**: they are cases of the set `v4kern`
  ([replay_cases.tsv](../2026-11-11_v4_kernel_merge/replay_cases.tsv)) and passed on this same build and card in R120a,
  in every `dmtkt` run that must pass (345 of 345 each, `dmtkt-on` with `_DMT=1` alone among them;
  [replay.tsv](../2026-11-12_r120a_kern_exact/replay.tsv)).

So there is no fault in the merge or in the kernels: the set predates those 30 cases. Registry Q1412 has the replay
name each case it holds no record of and give such a run its own verdict, so that a set replays on a later build of its
line without a second run.

## Files

| File | What |
|---|---|
| [suites.tsv](suites.tsv) | per suite, with the `-kern` build: runs, the runs that must pass and how many ended as expected, the cases each replayed, their case replays and OKs, the unrecorded cases a run, the fault runs and the cases each failed |
| [results.tsv](results.tsv) | `scripts/replay_checks.sh`'s table, one line per run: run, switches, what passing means, cases passed, FAIL lines, exit status, memcheck errors (none of these runs uses the sanitizer), verdict |
| [verdicts_by_run.tsv](verdicts_by_run.tsv) | per run: what passing means, the verdict, and its cases by their own verdict (OK, FAIL) |
| [console.txt](console.txt) | the script's console with test-backend-ops' own summary per run (the rig's folders written as `$SET` and `$B`) |
| [big_cases.tsv](big_cases.tsv) | each large-activation product: in how many runs it replayed, with what verdict, and the runs where it was not OK |
| [coverage.tsv](coverage.tsv) | per op: the set's cases, how many a run that must pass replayed and passed |
| [failing_cases.tsv](failing_cases.tsv) | every case not OK in a run that must pass: none |
| [unreplayed.tsv](unreplayed.tsv) | the set's cases no run that must pass replayed: none |
| [cases.tsv.gz](cases.tsv.gz) | every replayed case of every run: run, case id, case, verdict, errors |
| [suites_day.tsv](suites_day.tsv), [results_day.tsv](results_day.tsv), [verdicts_by_run_day.tsv](verdicts_by_run_day.tsv), [console_day.txt](console_day.txt) | the same for the `dmt` suite with the `-day` build |
| [run/run_replay.sh](run/run_replay.sh) | the run on the rig: [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) copied over and started detached, the cards and the kernel log recorded before and after |
| [run/suites.sh](run/suites.sh) | builds `suites.tsv` from `verdicts_by_run.tsv` and the runs' logs |
| [run/r112_tables.sh](run/r112_tables.sh) | builds every table above from the two output folders, with [the v4night replay's tables.sh](../2026-11-11_v4night_replay/run/tables.sh) and `suites.sh` |

## Method

**The builds on the rig.** The `-kern` rig tarball `llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern.tar.gz`
(md5 `37a4c3731f76455dce14013210b2fac7`; `test-backend-ops` md5 `1b2e6a7b426c8062a2bdee9acb761148`) and the `-day`
one `llama.cpp-20261111-18d0fa309-pascal-p106-v4-b12246-day.tar.gz` (md5 `ad766d3f8198be0c77f9f41ebf9c69d5`;
`test-backend-ops` md5 `25ffc9b368c59545109504ec2e283005`, the one in the set's `record.txt`), both already unpacked on
the rig. Both are native `sm_61` builds of CUDA 12.9.

**The set on the rig.** `v4day.tar` (5,473,720,320 bytes) streamed once over the LAN into the rig's data disk, its md5
taken on the way in: `f23c05d9365e55b70df90401eb064ca4`, the tar's own. 4,536 case files (three a case). The set's
`runs.tsv` is [set_runs.tsv](../2026-11-11_v4_day_merge/set_runs.tsv) byte for byte (md5
`f8c604d68ae26400e5a2b34f902d45bc`), its `manifest.tsv` md5 `14c746f4ffac5da891390d85e2dce7e3`.

**The runs**, detached on the rig with nothing else running on it, every run on card 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`),
no live CPU check (`CPU_CHECK=0`): the CPU references in the set judge every case. With `-kern`, every run of the run
list in its order, 14:01:40-14:27:51; with `-day`, the 14 `dmt` runs, 14:29:11-14:32:32. From the repository root:

```bash
SET=<the unpacked v4day on the rig> BIN=<the -kern build>/bin W=<a folder on the rig for the run> \
  bash docs/data/2026-11-12_r112_v4day_replay/run/run_replay.sh start   # then the same with wait, then with end
TAG=-day RUNS='dmt-on dmt-nopf dmt-r1 dmt-r2 dmt-r1-nopf dmt-r2-nopf dmt-rule dmt-nw1 dmt-nw2 dmt-defaults dmt-off dmt-skip-q4k dmt-skip-iq4xs dmt-skip' \
  SET=<the same> BIN=<the -day build>/bin W=<the same> \
  bash docs/data/2026-11-12_r112_v4day_replay/run/run_replay.sh start   # then wait, then end
```

The first runs on the rig

```bash
SET=$SET BIN=$BIN OUT=$W/out DEV=0 CPU_CHECK=0 RUNS='' setsid nohup bash replay_checks.sh > nohup.log 2>&1 < /dev/null &
```

and the second the same with `OUT=$W/out-day` and its `RUNS`.

**The tables**, on the build machine's CPU from a copy of the two output folders:

```bash
REPO=<this repository> OUT=<copy of $W/out> OUT_DAY=<copy of $W/out-day> SET=<the set's folder: manifest.tsv> \
  DEST=docs/data/2026-11-12_r112_v4day_replay RIG_W=$W RIG_SET=$SET RIG_B=<the rig's builds folder> \
  nice -n 19 bash docs/data/2026-11-12_r112_v4day_replay/run/r112_tables.sh
```

## Provenance and licence

The tables are this project's own measurements, published under the repository's licence. The cases are llama.cpp's
`test-backend-ops` cases (MIT) and the fork's additions to them
([Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama), MIT); their inputs are random draws stored
in the record. The raw logs (each run's output) are not published.
