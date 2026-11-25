# Upstream's 16 new `IQ4_XS` cases in the P106 record sets (C1038, Q1771, 2026-11-25)

Registry: [Q1771](../../../quants/experiments.md). It follows the P106 check of upstream b12674
([2026-11-24_c912_p106_check](../2026-11-24_c912_p106_check/README.md)).

**Recorded.** The P106 check replays two sets of the fork's kernel cases, `v4kern` and `ro8kt`, against processor
results stored on a build machine, because a rig's 2-core Celeron is far too slow to compute them. Upstream's
[#31998](https://github.com/ggml-org/llama.cpp/pull/31998) added 16 `IQ4_XS` matrix-vector cases to `test-backend-ops`
that the sets did not hold, so the last check ran them on the card with nothing to judge them by
([its unrecorded.tsv](../2026-11-24_c912_p106_check/unrecorded.tsv)). Both sets are now recorded again with the
b12674 build's own `test-backend-ops`. They hold all 16 cases, and a replay with that build finds no selected case
missing.

## What was recorded

The build is `pascal-p106-v5` at `eee615cdb` (upstream b12674 merged), its rig tarball
`llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz` (md5 `575ef3c702f69f14d7d3022c1287c38f`), whose
`test-backend-ops` (md5 `7e6731c4d42a98cfd6a90cf4bdeb0cc5`) is built without AVX (SSE4.2 at most), so the stored
results are what a rig's own processor would compute. [scripts/record_checks.sh](../../../scripts/record_checks.sh)
ran it in a container without GPUs at a low CPU share, on a build machine's processor, as
[R153 recorded the sets](../2026-11-13_c831_upstream/README.md#on-a-p106-r153-q1463): `v4kern` from its three filters,
then `ro8kt` with every case hard-linked from `v4kern` (each of its cases is one of `v4kern`'s). 2026-11-25
12:26:57–12:28:04, [run/record_sets.sh](run/record_sets.sh).

| Set | Cases | R153's | New | Bytes (inputs and results) | Runs in its list | Files |
|---|---:|---:|---:|---:|---:|---:|
| `v4kern`: C796's three kernel suites and all three switches together | 738 | 722 | 16 | 4,227,398,278 | 47 | 2,220 |
| `ro8kt`: C794's K tiles at 5–8 tokens | 210 | 202 | 8 | 725,677,616 (all hard links into `v4kern`) | 16 | 636 |

- **The new cases are exactly the 16 the last check listed** ([new-cases.tsv](new-cases.tsv)): `IQ4_XS` products at
  m = 1,023 with k = 4,096 and at m = 4,096 with k = 1,024, each at 1 to 8 columns. All 16 are in `v4kern`; the 8 at
  5 to 8 columns are in `ro8kt` too. No case of R153's sets is missing ([sets.tsv](sets.tsv)).
- **Every other case is R153's case, drawn again** ([vs-r153.tsv](vs-r153.tsv)): the same 722 and 202 case strings
  and ids, the same input and result sizes, new input data. `test-backend-ops` seeds its random inputs from the
  system on every run, so a new record draws new tensors and computes new results for them. The sets are the b12674
  build's own record, replacing R153's for checks of b12674 and later builds.
- Both sets together: 2,856 files, 4,228,443,702 bytes on disk (hard links counted once). [files.md5](files.md5) has
  one md5 per file, paths relative to the folder that holds `v4kern/` and `ro8kt/`; the list's own md5 is
  `88fab2d8e805fbfae598860114bebb36`. The case lists are [manifest-v4kern.tsv](manifest-v4kern.tsv) and
  [manifest-ro8kt.tsv](manifest-ro8kt.tsv): id, case, input bytes and md5, result bytes and md5.

## The replay: no selected case is missing

A P106 replay ([scripts/replay_checks.sh](../../../scripts/replay_checks.sh)) runs each line of a set's run list
with its switches; the switches change what the card runs, not which cases are selected. So
[run/cpu_replay.sh](run/cpu_replay.sh) replays each distinct case filter of the run list once, with the same
`test-backend-ops`, on the processor backend (`-b CPU --replay`), in the same container. Each replay looks every
selected case up in the set and compares the processor's result with the stored one. 12:28:05–12:28:21.

| Set | Filter (`-o MUL_MAT -p ...`) | Runs using it | Cases replayed | Pass | Fail | Selected, not recorded | Declined by the processor backend (all recorded) |
|---|---|---:|---:|---:|---:|---:|---:|
| `v4kern` | `type_a=q6_K,...,n=[1-5],` (`q6kmt`) | 13 | 130 | 130 | 0 | **0** | 10 |
| `v4kern` | `type_a=(iq4_xs\|q4_K),...,n=[1-5],` (`dmtkt`) | 12 | 336 | 336 | 0 | **0** | 20 |
| `v4kern` | `type_a=(iq4_xs\|q4_K\|q5_K),...,n=[5-8],` (`ro8`) | 13 | 295 | 295 | 0 | **0** | 3 |
| `v4kern` | all three (`kern-all` and the three faults with all switches on) | 9 | 705 | 705 | 0 | **0** | 33 |
| `ro8kt` | `type_a=(iq4_xs\|q4_K),...,n=[5-8],` | 16 | 208 | 208 | 0 | **0** | 2 |

([cpu-replay.tsv](cpu-replay.tsv) has the full filters.) The combined filter covers every case of `v4kern`
(705 + 33 = 738) and `ro8kt`'s filter every case of `ro8kt` (208 + 2 = 210).

- **0 selected cases without a record, in every filter.** The 16 new cases are among the replayed ones: 10 in each
  `dmtkt` replay, 8 in each `ro8` and `ro8kt` replay, 16 in the combined one, the counts the last check found
  missing.
- **The declined cases are the large-activation products** (`big=250000`, 33 shapes of `IQ4_XS`, `Q4_K`, `Q5_K` and
  `Q6_K`). Their test class runs only on a GPU backend: the processor backend reports them "not supported" before
  any record is looked up, so this replay cannot count them. Each one is in the set's manifest, and a card replays
  them as R153's did.
- Every replayed case passes against its stored result, so the records read back whole.

## On a rig

The sets go to a rig whole: one `tar` stream of the folder that holds `v4kern/` and `ro8kt/` keeps the hard links
(4.2 GB instead of 4.9 GB), and `md5sum -c files.md5` in that folder on the rig checks every file. The next P106
check replays them with the check's [rig_suite.sh](../2026-11-17_v5_cand_p106_check/run/rig_suite.sh), `SETS` set to
that folder. A card's replay counts no unrecorded case then, and judges the 16 cases there.

## Files

| File | What |
|---|---|
| [sets.tsv](sets.tsv) | per set: cases, R153's cases, new cases, cases only R153's set held, bytes, runs, the `test-backend-ops` md5, files, how many of the last check's 16 unrecorded cases it now holds |
| [new-cases.tsv](new-cases.tsv) | the 16 new cases: id, case, input and result bytes and md5, which sets hold it, whether the last check listed it unrecorded |
| [vs-r153.tsv](vs-r153.tsv) | the cases both records hold: same case string, same input and result sizes, same md5s |
| [cpu-replay.tsv](cpu-replay.tsv) | the replay per set and filter: the filter, the runs using it, cases replayed, pass, fail, not recorded, declined by the processor backend and how many of those the set holds, `test-backend-ops`' replay summary, exit status |
| [manifest-v4kern.tsv](manifest-v4kern.tsv), [manifest-ro8kt.tsv](manifest-ro8kt.tsv) | the sets' case lists (`manifest.tsv` of each set, with a header) |
| [files.md5](files.md5) | one md5 per file of both sets |
| [run/record_sets.sh](run/record_sets.sh) | the record: `v4kern`, then `ro8kt` linking it, then `files.md5` |
| [run/cpu_replay.sh](run/cpu_replay.sh) | the replay on the processor backend, once per distinct filter of a set's run list |
| [run/tables.py](run/tables.py) | this folder's tables, from the sets' small files, R153's manifests and the last check's `unrecorded.tsv` |

## Commands

```bash
# on the build machine (Docker, no GPU needed): $T is the unpacked rig tarball, $W a work folder, $REPO this repository
OUT=$W/b12674 BIN=$T/bin REPO=$REPO JOBS=8 nice -n 19 bash docs/data/2026-11-25_c1038_iq4xs_records/run/record_sets.sh
for s in v4kern ro8kt; do
  SET=$W/b12674/$s BIN=$T/bin OUT=$W/replay/$s JOBS=8 nice -n 19 bash docs/data/2026-11-25_c1038_iq4xs_records/run/cpu_replay.sh
done
# the tables ($IN: each set's manifest.tsv, runs.tsv and record.txt, R153's manifests as r153-<set>.tsv, each replay's
# summary.tsv, filters.tsv and filter-<n>.log under replay-<set>/, and files.md5; see the script's header)
python3 -I docs/data/2026-11-25_c1038_iq4xs_records/run/tables.py $IN docs/data/2026-11-25_c1038_iq4xs_records \
  docs/data/2026-11-24_c912_p106_check/unrecorded.tsv
# to a rig: one stream, then every file's md5 there ($D: the rig's folder for the sets)
tar cf - -C $W b12674 | ssh rig "mkdir -p $D && tar xf - -C $D"
ssh rig "cd $D/b12674 && md5sum -c --quiet files.md5 && echo all files match"
```

## Provenance and licence

The cases are `test-backend-ops`' own (llama.cpp, MIT licence), with random inputs; the blobs themselves (4.2 GB of
random tensors and their results) are not published, as anyone can record them again with the commands above. The
tables are this project's.
