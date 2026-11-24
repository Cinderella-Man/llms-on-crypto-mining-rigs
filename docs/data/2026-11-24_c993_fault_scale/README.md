# A size for the Pascal prompt kernel's fault switch (C993, Q1726; 2026-11-24)

What this is: registry row Q1726 in [quants/experiments.md](../../../quants/experiments.md), its code and its check.
To know how small an attention error a quality check can see, a test switch injects a known one:
`GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1` multiplies the output of the fork's compute-6.1 attention kernels (the prompt
kernel and, when it is on, the decode kernel) by a fixed 1.05. With one size only, the
[per-stretch KLD check on Ornith](../2026-11-24_c989_attn_error_floor/README.md) could only estimate its smallest
visible error (est. 3.43 % against the F16 cache, and est. 4.4 % in the
[self-referenced form](../2026-11-24_c994_attn_fault_selfref/README.md)).

The fork branch `pascal-p106-v5-fault-scale` (two commits on `pascal-p106-v5` at `af00bb13b`: `da4507e32`, and
`ceaf710a1`, which names the warning's function; `ggml/src/ggml-cuda/fattn-tile-pascal.cu` and `.cuh` only) adds
**`GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=<factor>`**, the factor fault 1 multiplies the output by:

- default 1.05, so `_FAULT=1` alone is unchanged; a value that is not a finite number gets a warning and 1.05;
- read once, and copied to each device's constant memory beside the fault switch the first time the kernel runs there
  with the fault on; with `_FAULT` unset nothing is copied and the kernels never read it;
- one INFO line names the fault and the factor the cards use (for 1.01: `output x 1.00999999`, the nearest float).

Built for the rig (`TESTS=1`) and checked on a build machine with RTX 3090s, which run the rig build's compute 6.1
PTX; the machine code compared is the compute 6.1 code a P106 runs. Correctness only: nothing here is timed.

## Result in plain words

- **With the switch unset nothing changes.** In the rig build's compute 6.1 code, 9,007 of 9,018 kernels keep the
  base's bytes. The other 11 (the decode kernel and the 10 prompt-kernel variants) keep the base's instructions in the
  base's order; only the fault's multiplies differ, 361 lines: the fixed 1.05 (`FMUL32I.FTZ R, R,
  1.0499999523162841797`) is now read from constant memory (`FMUL.FTZ R, R, c[0x3][0x4]`), 320 plain and 41
  predicated ([sass_kernels.tsv](sass_kernels.tsv), [sass.tsv](sass.tsv)). On the 367 head-256 flash-attention cases,
  replayed on the same stored inputs, the branch gives the base's error on every case to the last digit, with the
  decode kernel off and on ([identity.tsv](identity.tsv)).
- **The factor acts as set.** The test's error is `sum((a - b)^2) / sum(a^2)` with `a` the card's output, so an output
  scaled by f against an exact reference gives ((f − 1)/f)^2. On the 88 cases the prompt kernel takes, the median
  error is 0.002268 at × 1.05 (expected 0.002268), 0.0008490 at × 1.03 (0.0008483), 0.0003854 at × 1.02 (0.0003845)
  and 0.00009926 at × 1.01 (0.00009803); the cases' own rounding (median 1.422e-6 unset, largest 0.0001263) is the
  rest ([arms.tsv](arms.tsv)).
- **The test's verdicts:** unset, all 367 pass. `_FAULT=1` alone fails exactly the 88 prompt-kernel cases, with the
  base's errors. `_FAULT_SCALE=1.0` passes all, with the unset errors. **× 1.01 passes all 367** and × 1.02 fails one
  case, while × 1.03 fails all 88. The test's band (5e-4) is crossed at f = 1/(1 − √5e-4) = 1.0229, so an attention
  output wrong by a uniform 2 % passes `test-backend-ops`; the case that × 1.02 fails is the one whose own rounding is
  largest (0.0001263 unset, 0.0005148 at × 1.02). (In the first run, on another random draw, × 1.02 failed two.)
- **The other site, the decode kernel:** with `GGML_CUDA_FATTN_DEC_PASCAL=4 GGML_CUDA_FATTN_TILE_PASCAL_MIN=1`, the 15
  cases it takes (Q8_0, up to 4 query tokens, no sinks) give a median 0.002268 at × 1.05 (the base) and 0.0008483 at
  × 1.03 (the branch), each ((f − 1)/f)^2 to a relative 4e-5; the 98 cases the prompt kernel takes in those arms the
  same ([cases.tsv](cases.tsv)).
- **The rest of the switch:** `_FAULT_SCALE=x1.03` prints `GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=x1.03 is not a
  number: 1.05 is used` and gives `_FAULT=1`'s errors; `_FAULT_SCALE=1.03` without `_FAULT` gives the unset errors and
  no fault line ([logs.tsv](logs.tsv)). Every arm's log names the prompt kernel at head 256 with F16 and Q8_0 K/V, and
  the decode arms the decode kernel.
- **The replay is the live test:** two arms checked every case with inputs under 32 MB (309 of 367) live first: the
  processor's reference was the stored one bit for bit and every verdict the same ([live.tsv](live.tsv)).
- **The build:** rig tarball `llama.cpp-20261124-ceaf710a1-pascal-p106-v5-fault-scale.tar.gz`, 1,385,293,679 B, md5
  `76b179182497c2c5766c508f3558a74b`; `test-backend-ops` md5 `6e4aa3e9eb682ddb445b1864aa3cb607`. It has the base's
  21 binaries and web page (b12483), the same 32 compiler warning lines (none new), no AVX instructions and GLIBC_2.38
  at most. Every binary's md5 differs from the base's because each one carries the commit id.

For the KLD check on a model, `GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1
GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=1.02` (at `-lv 4` the log shows the factor) puts a 2 % error on every prompt
pass that the kernel takes.

## Files

| File | What it holds |
|---|---|
| [arms.tsv](arms.tsv) | per arm (binary and switches, all with the prompt kernel forced): cases, passes, failures, the cases the fork's kernels take, their verdicts, and their error against ((f − 1)/f)^2 |
| [identity.tsv](identity.tsv) | the arm pairs that must give the same error on every case (unset against the base, the default factor, 1.0, a non-number, the factor without the fault), and that they do |
| [cases.tsv](cases.tsv) | every case: which kernel takes it with the fault on, and its verdict and error in every arm |
| [logs.tsv](logs.tsv) | per arm, the kernel, fault and warning lines its log printed |
| [live.tsv](live.tsv) | the two arms with a live processor check before each replay: cases checked, verdicts, references identical |
| [sass_kernels.tsv](sass_kernels.tsv) | the 11 kernels whose compute 6.1 code differs from the base's: instructions in each, lines that differ |
| [sass.tsv](sass.tsv) | each of the 361 differing lines, base and branch |
| [run/run.sh](run/run.sh) | the driver: sources, the rig build, the machine-code comparison, the checks on one card, the tables |
| [run/c993_tables.py](run/c993_tables.py) | the kernel comparison and the tables from the driver's outputs |

## Commands

```bash
# on the build machine, detached; SEED is a llama.cpp clone holding the base, BASE the base's rig build (TESTS=1)
W=<work folder> SEED=<clone> REPO=<copy of this repository> BASE=<base rig build folder> BASE_LOG=<its build log> \
  RIG_CCACHE=<compiler cache to seed> JIT_SEED=<PTX JIT cache to seed> GPU=2 PREFLIGHT='<CUDA check, run as $PREFLIGHT -g 2>' \
  LOCK=<build lock file> JOBS=8 setsid nohup bash run/run.sh > <work folder>/logs/run.out 2>&1 < /dev/null &
# the tables again from a copy of the work folder (check/, checks/sass/)
python3 -I run/c993_tables.py tables <copy of the work folder> <this folder>
# one arm by hand on any CUDA card: the head-256 cases with the prompt kernel forced and a 3 % fault
GGML_CUDA_FATTN_TILE_PASCAL=2 GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE=1.03 \
  test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT -p '^hsk=256,hsv=256,'
```

The run recorded the cases once with the base build and no fault (`--record`, 6.8 GB of inputs and processor
references), then replayed that record in every arm (`--replay`, `--replay-log`), so every arm scores the same inputs.
`sass_kernels.tsv` was added to the table script after the run and written from the fetched outputs; the run's own
tables are byte for byte the ones the script writes now.

Provenance: the inputs are `test-backend-ops`' own random draws (`std::random_device`), stored in the record, which is
not kept here (6.8 GB). Code MIT (llama.cpp's licence).
