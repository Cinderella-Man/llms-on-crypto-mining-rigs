# What `GGML_CUDA_TOP_K_SORT=1` costs on a P106 (C997, Q1730; 2026-11-24)

What this is: the P106 half of registry row Q1730 in [quants/experiments.md](../../../quants/experiments.md).
[C985](../2026-11-24_c985_topk_order/README.md) added the switch `GGML_CUDA_TOP_K_SORT` (off by default, in
`pascal-p106-v5` since `af00bb13b`). With it, one small kernel runs after the card's radix top-k and sorts each row's k
candidates, so a sampled reply no longer changes between loads with a fixed seed. This folder times that kernel on one
P106-090 of the second rig, every card at its soaked clock profile. The row also asked for Flint Pascal Q8 GGUF's
decode step with the switch unset and set. That half did not run: the model's file (40.4 GB) is not on that rig and
does not fit on its disks. It is a row of its own now (Q1816).

## Result in plain words

- **On a P106 the switch adds 6.14 and 8.20 µs of card time to the sampler's top-k** (one row of 151,936 columns,
  k = 20: Flint's shape is 150,144 columns and k = 20). That is the sort kernel's own time: the fork's profiler timed
  the whole top-k on the card with the switch unset and set, two runs each, alternated.
- **On Flint that is est. 4.24 and 5.66 µs a token,** against 18.09 ms a token (0.023 % and 0.031 %), the size of
  the spread between two repeated Flint runs (R168: 4,631.065 and 4,631.941 ms over the same 176 passes, 5.0 µs a pass
  apart).
  So est. negligible; the step itself is not measured here.
- **The sort is not cheap for the sparse-attention top-k:** with k = 2,048 of 131,072 columns (the indexers of
  Flash-Next's QSA layers, which want no order) it adds 68.60 and 78.85 µs a call. The switch sorts every radix top-k on
  the card, so Flash-Next would pay that in every QSA layer of every token. New row Q1815: sort only the sampler's
  top-k.

## What ran

- **Machine:** the second rig, one card (`CUDA_VISIBLE_DEVICES=0`, NVIDIA P106-090, compute 6.1), every card at its
  soaked clock profile with the watchdog running, as found. Nothing else ran on the rig; no llama process before or
  after.
- **Build:** `pascal-p106-v5` `af00bb13b` (rig tarball md5 `b51ad2740d5822e7e14af656c413ec22`), `test-backend-ops`
  md5 `65fd4d6b430b6f326bedc3ca2233db06` ([perf/](perf/) holds every console).
- **Pass 1, wall time** ([run/kernel.sh](run/kernel.sh), 20:21:56–20:24:21): `test-backend-ops perf -o TOP_K -b CUDA0`
  on the perf cases whose row holds a sampler's vocabulary (151,936 or 248,320 columns) or a sparse-attention indexer's
  (131,072 columns), alternated unset 1, set 1, unset 2, set 2. Each line is the op's wall time a run: the host
  launches every op (about ten kernels each) one at a time, so it holds launch time as well as card time.
- **Pass 2, card time** ([run/kernel_prof.sh](run/kernel_prof.sh), 20:25:43–20:26:13): three of those cases again under
  the fork's profiler (`GGML_CUDA_PROFILE`, group mode: every launch group, here one top-k op, timed with CUDA events
  on the card), alternated the same way. The median of a run's 16,384 (8,192 for one case) timed ops is the card-bound
  op; the mean also holds the ops where the card waited for the host's launches. The profiler's own events slow the
  wall time (its consoles in [perf/](perf/)), which is why only its event times are read.
- **The switch's log line** ("GGML_CUDA_TOP_K_SORT=1: the radix top-k's columns sorted…") is in every set run's console
  and in no unset run's.
- Tables on the operator's machine: [run/kernel_table.py](run/kernel_table.py) and
  [run/kernel_prof_table.py](run/kernel_prof_table.py).

## The sort kernel's own time (pass 2, card time, µs per op)

[kernel_profile.md](kernel_profile.md), from [perf/gp-*.csv.gz](perf/):

| case (columns, rows, k) | unset 1 median (mean) | set 1 median (mean) | unset 2 median (mean) | set 2 median (mean) | set 1 − unset 1 | set 2 − unset 2 |
|---|---:|---:|---:|---:|---:|---:|
| 151,936 × 1, k = 20 (a sampler row, Flint's size) | 78.85 (111.50) | 84.99 (119.92) | 77.82 (111.34) | 86.02 (120.88) | +6.14 | +8.20 |
| 248,320 × 1, k = 20 (a sampler row, Flash-Next's and Ornith's size) | 80.90 (113.53) | 87.04 (122.14) | 80.90 (114.06) | 89.09 (123.28) | +6.14 | +8.19 |
| 131,072 × 1, k = 2,048 (a sparse-attention indexer row) | 80.90 (116.87) | 149.50 (150.62) | 71.68 (110.36) | 150.53 (150.87) | +68.60 | +78.85 |

The card's event clock on a P106 moves in steps of about 1.02 µs, so each median is good to about a microsecond. The
two unset runs of the sampler rows differ by 1.03 µs (151,936) and 0.00 µs (248,320); the sort's cost is beyond that.

## The op's wall time (pass 1, µs a run)

[kernel_perf.md](kernel_perf.md) holds all 19 cases. The sampler rows:

| case (columns, rows, k) | unset 1 | set 1 | unset 2 | set 2 | set 1 − unset 1 | set 2 − unset 2 |
|---|---:|---:|---:|---:|---:|---:|
| 151,936 × 1, k = 20 | 80.24 | 87.71 | 80.42 | 87.20 | +7.47 | +6.78 |
| 151,936 × 1, k = 40 | 73.49 | 111.31 | 74.44 | 111.83 | +37.82 | +37.39 |
| 248,320 × 1, k = 20 | 76.13 | 115.96 | 76.28 | 115.03 | +39.83 | +38.75 |
| 248,320 × 2, k = 20 | 128.62 | 134.92 | 128.55 | 135.28 | +6.30 | +6.73 |
| 248,320 × 4, k = 20 | 205.20 | 211.42 | 206.28 | 211.74 | +6.22 | +5.46 |

- At Flint's shape the wall time grows as much as the card time (+7.47 and +6.78 µs against +6.14 and +8.20).
- **Not explained:** one row of 248,320 columns grows 39.83 and 38.75 µs in wall time (151,936 columns with k = 40:
  37.82 and 37.39) while its card time grows only 6.14 and 8.19 µs (k = 40 was not profiled). Both repeats agree, so it is not noise;
  it is outside the card, in how the host's launches of the perf loop line up. A server whose decode runs as captured
  CUDA graphs pays the card time; whether the server's step sees anything of the wall-time jump is what the step half
  (Q1816) measures.
- The indexer rows (131,072 columns, k = 2,048) grow more with more rows: +45.89 and +46.25 µs a run at one row,
  +205.06 and +208.14 at 32 rows.

## What it prices

- **Flint, est.:** R176 ([data](../2026-11-17_r176_flint_step_map/README.md)) prices a millisecond off a pass at
  0.69 ms a token at depth 0 and 0.64 at 8,192, because right guesses come nearly free. The sort adds 6.14 and 8.20 µs to
  the last card's part of every pass, so est. 4.24 and 5.66 µs a token at depth 0 (0.023 % and 0.031 % of 18.09 ms),
  3.93 and 5.25 at 8,192. If every evaluation's top-k sat on the critical path (2.06 a token at depth 0, 2.85 at
  8,192), at most 12.65 and 16.89 µs a token at depth 0 (0.070 % and 0.093 %).
- **Ornith and Swift Flash-Next, sampler only, est.:** 6.14 and 8.19 µs a pass against steps of about 64 and 49 ms,
  under 0.02 % on either.
- **Swift Flash-Next with its QSA indexers:** 68.60 and 78.85 µs a call at 131,072 columns; times the indexer calls a
  token, that is a measurable loss. Its preset should not take the switch until Q1815 keeps the indexers unsorted.

## Class

- **The kernel: a loss of 6.14 and 8.20 µs a sampler call on a P106,** beyond the 1.03 µs between the unset repeats.
- **Flint's step: est. negligible, not measured.** The est. 4.24 and 5.66 µs a token is about the spread between two
  repeated Flint runs (5.0 µs a pass). The class on the step, and whether the switch goes into the presets, waits on
  Q1816. Until then it is not in the stacked line or in the negligible group.
- **Flash-Next with the switch as it is: est. loss** (the indexers' sort), until Q1815.

## New rows

- **Q1815 (C1082):** sort only the sampler's top-k. The switch sorts every radix top-k on the card, including the
  sparse-attention indexers' (k = 2,048: +68.60 and +78.85 µs a call on a P106), which need no order. Sort only where the
  order matters: the sampling chain's top-k (`LLAMA_SAMPLING_TOP_K_ROWS`'s and the chain's own) and the staircase
  drafter's, for example as an op parameter those callers set, or below a k limit; then Flash-Next's preset can take the
  switch too.
- **Q1816 (C1083):** the step half of this row on the first rig, which holds Flint Pascal Q8 GGUF: the v5 preset with
  the switch unset and set, two loads each, alternated, depths 0 and 8,192 on the deep lane, the same greedy replies
  ([run/step.sh](run/step.sh), written for it and not yet run). Est. 50 min (four Flint loads of about ten minutes).

## Files

- `README.md`: this page.
- `kernel_profile.md`, `kernel_perf.md`: the two tables in full.
- `perf/`: every console (`perf-u1.txt` … `perf-s2.txt` pass 1, `perf-pu1.txt` … `perf-ps2.txt` pass 2), the card's clock
  and power at the start of each pass-1 run (`times.txt`), the profiler's files (`gp-*.csv.gz`).
- `run/kernel.sh`, `run/kernel_prof.sh`: the two passes (env `BIN`, the build's bin folder, and `O`, the output
  folder); `run/kernel_table.py`, `run/kernel_prof_table.py`: the tables (`python3 -I run/kernel_table.py <folder>`);
  `run/step.sh`: the step half for Q1816 (R176's deep lane, arms `unset` and `sort`).
