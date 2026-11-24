# A kept card-suite record of MUL_MAT and MUL_MAT_ID (C1023, Q1756, 2026-11-24)

What this is: registry row [Q1756](../../../quants/experiments.md), point 25 of the GPU checks of the night's branches
(registry [Q1687](../../../quants/experiments.md)). Every landing and branch check runs `test-backend-ops` on a build
machine's cards with the fork's Pascal kernels forced on. Each case there is computed twice, on the card and on the
processor, and the processor's half is the slow one: in [the landing of 14 branches](../2026-11-24_land_v5_batch2/README.md)
the live `MUL_MAT_ID` share took 38 minutes on one card while its references used 12 processor cores, as
[C968](../2026-11-23_c968_pascal_mmvq_suite/README.md) had found. `test-backend-ops --record` writes each case's inputs
and the processor's answer to a folder, and `--replay` restores them so that only the card computes, with the same
comparison and the same verdicts. This makes that record once for `MUL_MAT` and `MUL_MAT_ID`, keeps it, and checks that
replaying it gives the live run's verdicts on two builds. Correctness only: the times below are the wall time of each
step, not a speed of anything the rig runs.

## Result

**In plain words:** the kept record works. Replayed, the two matrix-product shares pass every case the live check
passes, on the build that made the record and on the build before it, in 145 s or less a share where recording
`MUL_MAT_ID` took 2,084 s. The two builds give every case the same error, bit for bit, and the processor's answers recomputed for a
sample match the stored ones exactly, so the record is still good for this code.

| Share | Cases | Record (live check while recording) | Size | Replay on `af00bb13b` | Replay on `8a9629942` | Replay on `af00bb13b` with `--cpu-check 25` |
|---|---:|---|---:|---|---|---|
| `MUL_MAT_ID` | 2,370 | **2,370 of 2,370** pass, 2,084 s | 29.5 GB | **2,370 of 2,370**, 145 s | **2,370 of 2,370**, 68 s | **2,370 of 2,370**, 77 s; 84 references recomputed, 84 identical |
| `MUL_MAT` | 5,125 | **5,125 of 5,125** pass, 393 s | 18.0 GB | **5,125 of 5,125**, 81 s | **5,125 of 5,125**, 45 s | **5,125 of 5,125**, 41 s; 204 references recomputed, 204 identical |

([steps.tsv](steps.tsv), [verdict.tsv](verdict.tsv), [record.tsv](record.tsv))

- **Every case the live check passed passes in every replay, and no case is missing from the record** (0 cases without
  a record in every replay, run without `--replay-allow-missing` so that a missing case would have failed the run). The
  live counts are the landing's: `MUL_MAT` 5,125 of 5,125 and `MUL_MAT_ID` 2,370 of 2,370 with the kernels forced.
  `MUL_MAT` lists 521 cases and `MUL_MAT_ID` 8 as not supported on the card, in the record step and in every replay
  alike.
- **The record is 47.5 GB:** `MUL_MAT_ID` 2,370 cases in 7,110 files, 29,517,090,436 bytes; `MUL_MAT` 5,125 cases in
  15,375 files, 18,021,068,767 bytes (an input, a reference and a metadata file per case). Every file's md5 is in
  [manifest-mul_mat_id.md5.gz](manifest-mul_mat_id.md5.gz) and [manifest-mul_mat.md5.gz](manifest-mul_mat.md5.gz)
  (md5 of the two lists: `758db38bd4df6eb24059d860acfd9514` and `3b8c6931855cfd00f8396358b0d22345`; three passes over
  the record gave the same lists).
- **The replays take seconds where the record step took minutes:** `MUL_MAT_ID` 145 s, 68 s and 77 s against
  2,084 s; `MUL_MAT` 81 s, 45 s and 41 s against 393 s. They ran right after the record was written, so the files were read
  from the operating system's cache, not the disk. The disk read the largest file (674 MB) at 211 MB/s once its cached
  pages were dropped, so a replay days later reads the record for est. 2 min 20 s (`MUL_MAT_ID`) and est. 1 min 25 s
  (`MUL_MAT`) on top of the card's work, est. 6 minutes for both shares against 38 minutes for `MUL_MAT_ID` alone live.
- **The two builds compute every case the same:** each case's largest normalized error is the same number in the replay
  on `af00bb13b`, on `8a9629942` and in the `--cpu-check` replay ([cases.tsv.gz](cases.tsv.gz), 7,495 cases). So the
  landing of 14 branches changed no result of these products with the kernels forced at their defaults, and a replay
  repeats a build's result exactly.
- **The processor's answers still match the record.** `--cpu-check 25` recomputes about one case in 25 (inputs under
  32 MB) on this machine's processor before the replay and compares the result with the stored reference bit for bit:
  84 `MUL_MAT_ID` and 204 `MUL_MAT` references identical, 0 verdicts differ. A reference that differs is the sign that
  the processor's code for these products has changed (an upstream merge can do that), and the record then needs to
  be made again.

## Replaying it

A record holds each case's inputs and the processor's answer, never a card's result, so one record serves any build
of the same line and any kernel setting. The record step is the expensive part and is done once; every later landing
and branch check replays.

With `test-backend-ops` directly, per share (`MUL_MAT` and `mul_mat`, `MUL_MAT_ID` and `mul_mat_id`), the fork's
kernels forced with a P106's launch shapes (`run/run.sh`'s `ON`):

```bash
env GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 \
    GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 \
    <build>/bin/test-backend-ops test -b CUDA0 -o MUL_MAT_ID -p '^(?!.*dup=1)' \
    --replay $REC/mul_mat_id --replay-log mul_mat_id-replay.tsv --replay-allow-missing --cpu-check 25
```

Or with this folder's driver, one card, every share, one or more builds:

```bash
env WORK=<work folder> NEW=<build>/bin REC=<record folder> STEPS=replay ALLOW_MISSING=1 \
    BUILDS='head=<build>/bin base=<base build>/bin' GPU=0 MOUNTS='<folders holding the builds>' \
    setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
```

Reading the result:

| What the replay prints | Meaning | What to do |
|---|---|---|
| `N/N tests passed`, `0 selected by the filter have no record`, `references identical` = cases checked | the build passes the share | nothing |
| a case `FAIL` | the card's result is wrong for that case | the branch's fault, as in a live run; the record replays the same inputs as often as needed |
| a case `NOREC` (with `--replay-allow-missing`; without it the run fails) | `test-backend-ops` has a case the record lacks: its case list grew, or a case's parameters changed | record the new cases into the same folder: the record step with `--record $REC/<share>` keeps every case already there and records only the rest (`run.sh` with `STEPS=record RESUME=1`), then replay again |
| `replay cpu-check: ... N verdicts differ`, or fewer references identical than checked | the processor's code for these products changed since the record | a new record in a new folder with the new build (`STEPS='record replay manifest'`) |

Other rules:
- The record covers the card suite's filter, every case but the repeated-expert-id ones (`-p '^(?!.*dup=1)'`). Other
  filters replay the cases they share with it; the `dup=1` cases are not in it.
- Before trusting a record that has sat for a while, `md5sum -c` of the manifests (from the record's folder:
  `zcat manifest-mul_mat_id.md5.gz | md5sum -c --quiet`) reads every file once, est. 4 minutes at 211 MB/s.
- A replay uses the record's draw of the inputs. A live run draws new ones each time, so a live run now and then
  still checks cases the record never drew; the record is the fast check of every landing.

## What ran where

| | |
|---|---|
| Builds | the record and its first replay: `pascal-p106-v5` at `af00bb13b` (the landing of 14 branches), rig tarball md5 `b51ad2740d5822e7e14af656c413ec22`, `test-backend-ops` `65fd4d6b430b6f326bedc3ca2233db06`. The second replay: `8a9629942`, the branch before that landing, its `TESTS=1` rig tarball md5 `59ca8336b87367a6522e63c6fcc1381b` ([C984](../2026-11-24_c984_tests_build_tools/README.md)), `test-backend-ops` `f089e54a68e7a4051fc126abf5f41439` |
| Card | one RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT; a CUDA preflight passed first; each step in its own container at low processor priority |
| Settings | the fork's kernels forced with a P106's launch shapes in the record and every replay: `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`. Recording with them forced matters: 278 large-activation cases need the backend feature `MMVQ_PASCAL` and are not supported, so not recorded, without it |
| Times | 2026-11-24: record 14:54–15:36 (`MUL_MAT_ID` then `MUL_MAT`), replays 15:36–15:43, manifest 15:43–15:44, again 15:45–15:46 and 15:47–15:48 |
| Record | kept on the build machine's bulk hard disk (ext4), outside this repository: 47.5 GB of random test data. Its folder is passed as `REC` |

The record, the replays and the first manifest ran in one invocation of `run/run.sh`. Its read of the largest file
for the disk's speed used `dd iflag=direct`, which this disk refuses (`EINVAL`, also for whole MiB in a second
`STEPS=manifest` run), so the manifest step ran a third time alone with the read done after dropping that file's
cached pages: [run/run.sh](run/run.sh) as committed, md5 `b43a558dc035c600b4ed573f9e35265f`. All three gave the same
two lists.

## Files

| File | What it holds |
|---|---|
| [steps.tsv](steps.tsv) | every `test-backend-ops` step: share, step, exit code, the summary line, OK, FAIL, NOREC, not supported, illegal access, the record's or replay's own line (with the cpu-check line), wall seconds |
| [verdict.tsv](verdict.tsv) | the pass check per share and replay: cases passing live while recording, passing in the replay, failing, without a record, and live passes not passing in the replay |
| [record.tsv](record.tsv) | per share: cases, files, bytes and the md5 of its manifest; the disk read of the largest file |
| [cases.tsv.gz](cases.tsv.gz) | every recorded case: share, record id, case, the live verdict while recording, and each replay's verdict and largest normalized error |
| [manifest-mul_mat_id.md5.gz](manifest-mul_mat_id.md5.gz), [manifest-mul_mat.md5.gz](manifest-mul_mat.md5.gz) | the md5 of every file of the record, paths relative to its folder (`md5sum -c`) |
| [run/run.sh](run/run.sh) | the driver: record, replays and manifest on one card |
| [run/tables.py](run/tables.py) | the tables from a copy of the driver's `bops/` folder (standard library, `python3 -I`) |

## Commands

On the build machine, with both rig tarballs unpacked, `run/` copied into a work folder, ~50 GB free on a bulk disk
for the record, and one card free:

```bash
env WORK=<work folder> NEW=<af00bb13b build>/bin REC=<record folder> GPU=<card> \
    BUILDS='af00bb13b=<af00bb13b build>/bin 8a9629942=<8a9629942 build>/bin' \
    MOUNTS='<folders holding the builds>' DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=<CUDA check, run as $PREFLIGHT -g <card>> \
    setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the manifest step again, alone (run twice, see above)
env WORK=<work folder> NEW=<af00bb13b build>/bin REC=<record folder> GPU=<card> STEPS=manifest \
    MOUNTS='<folders holding the builds>' DOCKER_ARGS='--cpu-shares=256' \
    setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the tables, on any machine, from a copy of <work folder>/bops
python3 -I run/tables.py <copy of bops> <this folder> af00bb13b 8a9629942 af00bb13b-cpucheck
```

Provenance and licence: the cases, inputs and references are `test-backend-ops`' own random test data from the
builds named above (the fork of llama.cpp, MIT); the tables and scripts here are under this repository's licence.
