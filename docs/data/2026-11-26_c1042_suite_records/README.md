# Kept card-suite records of FLASH_ATTN_EXT and the fused cases (C1042, Q1775, 2026-11-26)

What this is: registry row [Q1775](../../../quants/experiments.md), point 27 of the GPU checks of the night's branches
(registry [Q1687](../../../quants/experiments.md)). Every landing and branch check runs `test-backend-ops` on a build
machine's cards. Each case there is computed twice, on the card and on the processor, and the processor's half is the
slow one. [C1023](../2026-11-24_c1023_suite_record/README.md) made a kept record of the two matrix-product shares
(`MUL_MAT`, `MUL_MAT_ID`), which landings now replay. This does the same for the card suite's other two slow shares:
`FLASH_ATTN_EXT`, which every landing runs live three times (unset, with the Pascal tile kernel, with every Pascal
attention switch), and the fused cases, which the fuller checks run live
([C968](../2026-11-23_c968_pascal_mmvq_suite/README.md)). `test-backend-ops --record` writes each case's inputs and the
processor's answer to a folder, and `--replay` restores them so that only the card computes, with the same comparison
and the same verdicts. Correctness only: the times below are the wall time of each step, not a speed of anything the
rig runs.

## Result

**In plain words:** both records work. Replayed, the attention share and the fused cases pass every case the live
check passes, on the build that made the records and on the build before it, under every setting the checks use, with
the same error on every case on both builds. Together with C1023's record, a landing's whole card suite but the rest
share can now be replayed. For the fused cases this saves most of the time: the four replays from the cache took 186,
115, 112 and 112 s, the live check 1,473 s. `FLASH_ATTN_EXT`'s live runs take about four minutes each (233, 233 and
237 s in the last landing); its six replays from the cache took 98, 50, 59, 50, 69 and 56 s.

| Share | Cases | Record (live check while recording) | Size | Replays on `2d839da1d` | Replays on `eee615cdb` | Replay on `2d839da1d` with `--cpu-check 25` |
|---|---:|---|---:|---|---|---|
| `FLASH_ATTN_EXT` | 4,679 | **4,679 of 4,679** pass, every Pascal attention switch, 294 s | 19.5 GB | **4,679 of 4,679** with every switch (98 s), the tile kernel alone (50 s) and no switch (59 s) | **4,679 of 4,679** with every switch (50 s), the tile kernel alone (69 s) and no switch (56 s) | **4,679 of 4,679**, 198 s; 186 references recomputed, 186 identical |
| fused cases (55 operators) | 5,424 | **5,424 of 5,424** pass, every compute-6.1-only path forced, 1,473 s | 44.6 GB | **5,424 of 5,424** with every path forced (186 s) and the matrix-vector kernels alone (115 s) | **5,424 of 5,424** with every path forced (112 s) and the matrix-vector kernels alone (112 s) | **5,424 of 5,424**, 120 s; 200 references recomputed, 200 identical |

([steps.tsv](steps.tsv), [verdict.tsv](verdict.tsv), [errors.tsv](errors.tsv), [record.tsv](record.tsv))

- **Every case the live check passed passes in every replay, and no case is missing from the records** (0 cases
  without a record in all 12 replays, 0 failures). The live counts match the checks the records replace: the last
  landing's three live `FLASH_ATTN_EXT` runs passed 4,679 of 4,679 each, and C968's 43 fused-case operators hold
  5,329 cases in this build (C968: 5,274 on an older test file, one failing on a router tie). That `TOPK_MOE_TAIL` case
  passes here, 55 of 55, because [C992](../2026-11-24_c992_topk_tail_tie/README.md)'s fix of the test is in `2d839da1d`.
  `FLASH_ATTN_EXT` lists 1,429 cases and the fused share 231 as not supported on the card, in the record step and in
  every replay alike.
- **The fused share is 55 operators, not C968's 43.** The list step (`run.sh` `STEPS=list`, a dry pass over every case
  of the build in 48 s) finds every operator name test mode runs that is not a ggml operation
  ([list_ops.tsv](list_ops.tsv)): C947's 43, 9 more with 95 cases the card checks (`ADD_RMS_NORM_MUL`, `CONV_1D_DW`,
  `MUL_MAT_HADAMARD`, `MUL_MAT_ID_W4A4`, `MUL_MAT_ID_W4A8`, `MUL_MAT_PREC_SRC1_BF16`, `MUL_MAT_PREC_SRC1_BF16_SRC0`,
  `MUL_MAT_W4A4`, `MUL_MAT_W4A8`) and 3 whose cases the card does not support (`MUL_MAT_PREC_SRC1_F32`,
  `MUL_MAT_ID_PREC_SRC1_F32`, `MUL_MAT_ID_PREC_SRC1_BF16`). `-o` matches operator names exactly, so none of the 9 ran in
  any share of the card suite's split before this record; all 95 cases pass.
- **The records are 64.1 GB:** `FLASH_ATTN_EXT` 4,679 cases in 14,037 files, 19,463,260,559 bytes; fused 5,424 cases in
  16,272 files, 44,609,015,825 bytes (an input, a reference and a metadata file per case). Every file's md5 is in
  [manifest-flash_attn_ext.md5.gz](manifest-flash_attn_ext.md5.gz) and [manifest-fused.md5.gz](manifest-fused.md5.gz)
  (md5 of the two lists: `678878c114c54a8fbfec9986603449cd` and `84498f7e16747b413de482b849902cb6`).
- **Times against live.** The last landing's live `FLASH_ATTN_EXT` runs took 233, 233 and 237 s (no switch, the tile
  kernel, every switch; one card, beside three other shares on the other cards), and its record step here 294 s. The
  fused share's live check took 1,473 s here (C968's live fused run, beside two other shares, 47 minutes). The replays
  above read the records from the operating system's cache, right after they were written. The disk read the largest
  file (2.1 GB) at 163 MB/s once its cached pages were dropped, so a replay days later first reads the record for
  est. 2 min (`FLASH_ATTN_EXT`) and est. 4 min 30 s (fused): est. 5 minutes for a landing's three `FLASH_ATTN_EXT` replays
  against 703 s live, est. 6 min 30 s for the fused share against 1,473 s. The processor computes no reference in a
  replay, so the shares beside it on the other cards no longer compete for it.
- **The two builds compute every case the same:** under each setting, each case's largest normalized error is the same
  number on `2d839da1d` and on `eee615cdb` ([errors.tsv](errors.tsv), [cases.tsv.gz](cases.tsv.gz)): the landing
  between them changed no result of these operators. The `--cpu-check` replay gives every case the same error as the
  replay without it.
- **Where the Pascal kernels act** ([errors.tsv](errors.tsv)): against no switch, the Pascal tile kernel alone changes
  the result of 173 `FLASH_ATTN_EXT` cases and every attention switch 227, so these are the cases the fork's attention
  kernels compute; every one passes. In the fused share, forcing every compute-6.1-only path gives every case the same
  error as forcing the matrix-vector kernels alone: the attention block, the persistent MoE kernel and the tile kernel
  change no fused case.
- **The processor's answers still match the records.** `--cpu-check 25` recomputed 186 `FLASH_ATTN_EXT` and 200 fused
  references on this machine's processor, all identical bit for bit to the stored ones, 0 verdicts differ.

## Replaying them

A record holds each case's inputs and the processor's answer, never a card's result, so one record serves any build
of the same line and any setting. Whether a `FLASH_ATTN_EXT` case is supported on the card does not depend on any
Pascal switch (the CUDA backend asks only whether llama.cpp has a kernel for it), so the record made with every Pascal
attention switch on holds every case the other two settings run. The fused share was recorded with every
compute-6.1-only path of the fork forced, the widest setting.

With `test-backend-ops` directly, per share and setting:

```bash
# FLASH_ATTN_EXT, every Pascal attention switch (the landings' third run); the first two runs: no switch, and
# GGML_CUDA_FATTN_TILE_PASCAL=2 alone
env GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 \
    GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4 \
    <build>/bin/test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p '^(?!.*dup=1)' \
    --replay $REC/flash_attn_ext --replay-log flash_attn_ext-replay.tsv --replay-allow-missing --cpu-check 25
# the fused cases, every compute-6.1-only path forced (scripts/pascal_forced.sh)
. scripts/pascal_forced.sh
env $PASCAL_FORCED <build>/bin/test-backend-ops test -b CUDA0 -o "$(paste -sd, run/fused-ops.txt)" -p '^(?!.*dup=1)' \
    --replay $REC/fused --replay-log fused-replay.tsv --replay-allow-missing --cpu-check 25
```

Or with this folder's driver, both shares side by side on two cards, every setting, one or more builds:

```bash
env WORK=<work folder> NEW=<build>/bin REC=<record folder> STEPS=replay ALLOW_MISSING=1 GPUS=0,1 \
    BUILDS='head=<build>/bin base=<base build>/bin' MOUNTS='<folders holding the builds>' \
    setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
```

`ARMS_FA` and `ARMS_FUSED` pick the settings (default `all tile plain` and `forced mmv`), `CPU_CHECK=0` drops the
cpu-check replay. Reading the result is as for C1023's record:

| What the replay prints | Meaning | What to do |
|---|---|---|
| `N/N tests passed`, `0 selected by the filter have no record`, `references identical` = cases checked | the build passes the share | nothing |
| a case `FAIL` | the card's result is wrong for that case | the branch's fault, as in a live run; the record replays the same inputs as often as needed. An F16 or BF16 case failing with an error near 0.5 right after its sibling is the CUDA graph cache's reuse across cases ([C1111](../2026-11-26_c1111_bf16_intermittent/README.md)): rerun with `GGML_CUDA_DISABLE_GRAPHS=1` |
| a case `NOREC` | `test-backend-ops` has a case the record lacks: its case list grew, or a case's parameters changed | record the new cases into the same folder (`run.sh` with `STEPS=record RESUME=1`), then replay again |
| `replay cpu-check: ... N verdicts differ`, or fewer references identical than checked | the processor's code for these operators changed since the record | a new record in a new folder with the new build |
| the list step names operators `FUSED_OPS` lacks | `test-backend-ops` gained fused-case operators | add them to `run/fused-ops.txt` and record them (`STEPS=record RESUME=1`) |

Other rules:
- The records cover the card suite's filter, every case but the repeated-expert-id ones (`-p '^(?!.*dup=1)'`).
- Before trusting a record that has sat for a while, `md5sum -c` of the manifests (from the record's folder:
  `zcat manifest-fused.md5.gz | md5sum -c --quiet`) reads every file once.
- A replay uses the record's draw of the inputs. A live run draws new ones each time, so a live run now and then
  still checks cases the record never drew; the records are the fast check of every landing.

## What ran where

| | |
|---|---|
| Builds | the records and the first replays: `pascal-p106-v5` at `2d839da1d` (the landing of 2026-11-24's eight branches), rig tarball md5 `7d774f05c29460a6993b19d38796047c`, `test-backend-ops` `65048f96fa48b8722c7ae9f03638b6b0`. The second replays: `eee615cdb`, the branch before that landing (upstream b12674 merged), its `TESTS=1` rig tarball md5 `575ef3c702f69f14d7d3022c1287c38f`, `test-backend-ops` `7e6731c4d42a98cfd6a90cf4bdeb0cc5` |
| Cards | two RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT: `FLASH_ATTN_EXT` on one, the fused cases on the other, side by side; a CUDA preflight passed first; each step in its own container at low processor priority, beside another session's builds on the same processor |
| Settings | `FLASH_ATTN_EXT` recorded with every Pascal attention switch (`GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_LDS=1 GGML_CUDA_FATTN_TILE_PASCAL_KDP=1 GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4`), replayed with that, with `GGML_CUDA_FATTN_TILE_PASCAL=2` alone and with no switch; the fused cases recorded with `PASCAL_FORCED` ([scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh): the matrix-vector kernels at a P106's shapes, the attention block, the persistent MoE kernel, the tile kernel), replayed with that and with `PASCAL_FORCED_MMV` (the matrix-vector kernels alone, C968's and C1023's set) |
| Times | 2026-11-26: the list step 04:02–04:03, the records 04:03–04:08 (`FLASH_ATTN_EXT`) and 04:03–04:28 (fused), replays 04:08–04:18 and 04:28–04:38, manifest 04:38–04:44 |
| Records | kept on the build machine's bulk hard disk (ext4), outside this repository: 64.1 GB of random test data. Its folder is passed as `REC` |

A first `STEPS=list` run at 03:58, with a list step that counted only the cases the card checks, gave the same 52
operator names with cases and the same 27,590 cases; the run above repeats the list step with the committed script.

## Files

| File | What it holds |
|---|---|
| [steps.tsv](steps.tsv) | every `test-backend-ops` step: share, step, setting, exit code, the summary line, OK, FAIL, NOREC, not supported, illegal access, the record's or replay's own line (with the cpu-check line), wall seconds |
| [verdict.tsv](verdict.tsv) | the pass check per share and replay: cases passing live while recording, passing in the replay, failing, without a record, and live passes not passing in the replay |
| [errors.tsv](errors.tsv) | per share and setting: cases whose largest normalized error is the same on both builds; per setting against no switch (attention) or the matrix-vector kernels alone (fused): cases whose error the setting changes |
| [record.tsv](record.tsv) | per share: cases, files, bytes and the md5 of its manifest; the disk read of the largest file |
| [list_ops.tsv](list_ops.tsv) | the list step: every operator name test mode runs, the cases the card checks and their record bytes, the cases it does not support, and whether the name is a ggml operation or a fused-case name |
| [cases.tsv.gz](cases.tsv.gz) | every recorded case: share, record id, case, the live verdict while recording, and each replay's verdict and largest normalized error |
| [manifest-flash_attn_ext.md5.gz](manifest-flash_attn_ext.md5.gz), [manifest-fused.md5.gz](manifest-fused.md5.gz) | the md5 of every file of the records, paths relative to their folder (`md5sum -c`) |
| [run/run.sh](run/run.sh) | the driver: list, records, replays and manifest on one or more cards |
| [run/fused-ops.txt](run/fused-ops.txt) | the fused share's 55 operators |
| [run/tables.py](run/tables.py) | the tables from a copy of the driver's `bops/` folder (standard library, `python3 -I`) |

## Commands

On the build machine, with both rig tarballs unpacked, `run/` and `scripts/pascal_forced.sh` copied into a work folder,
~70 GB free on a bulk disk for the records, and two cards free:

```bash
env WORK=<work folder> NEW=<2d839da1d build>/bin REC=<record folder> GPUS=<card>,<card> \
    STEPS='list record replay manifest' ALLOW_MISSING=1 CPU_CHECK=25 \
    BUILDS='2d839da1d=<2d839da1d build>/bin eee615cdb=<eee615cdb build>/bin' \
    MOUNTS='<folders holding the builds>' DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=<CUDA check, run as $PREFLIGHT -g <cards>> \
    setsid nohup bash run/run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the tables, on any machine, from a copy of <work folder>/bops
python3 -I run/tables.py <copy of bops> <this folder> 2d839da1d eee615cdb
```

The driver's own timeout per `test-backend-ops` run is `TMO` (3 hours); the whole invocation ran under `timeout 23400`.

Provenance and licence: the cases, inputs and references are `test-backend-ops`' own random test data from the
builds named above (the fork of llama.cpp, MIT); the tables and scripts here are under this repository's licence.
