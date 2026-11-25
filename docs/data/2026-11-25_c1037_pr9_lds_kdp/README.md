# Q1770: pull request 9's `_LDS` and `_KDP` switches at Ornith-1.5-122B's prompt shape on one P106-090 (2026-11-25)

Since 2026-11-24 Ornith-1.5-122B Pascal N2P GGUF's v5 preset runs a Q8_0 cache with two of pull request 9's switches,
`GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` and `GGML_CUDA_FATTN_DEC_PASCAL=4` ([Q1706](../../../quants/experiments.md)). That
made decoding faster, but its prompts read more slowly: the Pascal tile kernel ran a 512-token prompt micro-batch 23.4 %
slower with the Q8_0 cache than with F16 at depth 256
([2026-11-24_r232a_ornith_q8_kernels](../2026-11-24_r232a_ornith_q8_kernels/README.md)), and the agent check read its
19 same-prompt requests 2.4 % more slowly ([2026-11-24_r232b_ornith_q8_agent](../2026-11-24_r232b_ornith_q8_agent/README.md)).
The pull request has two more switches for exactly those calls ([docs/switches.md](../../switches.md)):
- `GGML_CUDA_FATTN_TILE_PASCAL_LDS=1` decodes the Q8_0 K and V as they are stored to shared memory, not as they are
  read from it ([Q1702](../../../quants/experiments.md)).
- `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` keeps Q8_0 K as int8 at head 256 and scores it with `dp4a` against the query,
  which is quantized to int8 in blocks of 32 values. Its V loads are decoded at store, whether or not `_LDS` is set
  ([Q1704](../../../quants/experiments.md)).

This folder times both at Ornith's exact attention shape on a P106, each on top of the preset's switches, after a
correctness pass.

**Result: every case passes. `_KDP=1` makes the Q8_0 cache's prompt attention about 39 % faster than the preset's,
and now about 19 % faster than the F16 cache it replaced.**
- **Correctness:** 367 of 367 of the suite's head-256 `FLASH_ATTN_EXT` cases and 64 of 64 Ornith-shaped cases pass in
  each Q8_0 arm (the preset's switches, `+_LDS=1`, `+_KDP=1`, `+both`), each logging the kernel its switch selects
  ([tests.tsv](tests.tsv)).
- **512-token prompt micro-batch, one layer's call at 8,192 tokens:**
  - the preset: 179,079.53 and 178,610.57 µs;
  - with `_LDS=1`: 152,053.01 and 151,796.54;
  - **with `_KDP=1`: 107,861.90 and 107,920.41**;
  - the F16 cache (nothing set): 132,757.47 and 133,148.76.

  So `_KDP=1` is 39.67 % faster than the preset and 18.85 % faster than F16 there. From 4,096 to 49,152 tokens the
  change stays within a point: 38.96 % (4,096) to 39.67 % (8,192) faster than the preset and 17.96 % (4,096) to
  19.10 % (32,768) faster than F16, each depth in the tables below. At depth 256 it is 30.76 % faster than the preset and 14.39 % faster than F16.
- **8-token calls** (a drafted or pipelined pass) at 8,192 tokens:
  - the preset: 2,964.77 and 2,964.17 µs;
  - **with `_KDP=1`: 1,790.60 and 1,784.94**;
  - F16 (llama.cpp's stock kernel): 2,239.58 and 2,266.07.

  That is 39.69 % faster than the preset and 20.64 % faster than F16.
- **`_LDS=1` alone** is 9.08 % (512 tokens at depth 256) to 16.52 % (8 tokens at 49,152) faster than the preset, but
  still 9.51 % (8 tokens at 4,096) to 14.69 % (512 tokens at 16,384) slower than F16, except at 8 tokens and depth
  256. At head 256 `_KDP=1` already includes its loads, so `_LDS=1` adds nothing on top:
  `_KDP=1` with and without it runs the same kernel and differed by at most 0.24 % ([floor.tsv](floor.tsv)).
- **What this could mean for Ornith (est.):** Ornith has 12 attention layers. One 512-token micro-batch at 8,192
  tokens would take est. 0.85 s less attention than with the preset (est. 1.29 s against 2.15 s) and est. 0.30 s less
  than with F16.
  - The agent check's 19 same-prompt requests took est. 4.25 s longer with the preset than with F16 (the means of two
    runs each). If all of that is this kernel, and it scales as at 8,192 tokens, `_KDP=1` would read those requests
    in est. 6.6 s less than the preset (est. 2.3 s less than F16 did).
  - That is est. 0.5 % of a run's prompt and decode seconds.
  - Decoding (1 to 4 tokens a call) runs the decode kernel and is not changed.
  - A new row ([Q1835](../../../quants/experiments.md)) runs the agent check with `_KDP=1` against the preset.
    Another ([Q1836](../../../quants/experiments.md)) checks its quality on LLMRig first.
- **The preset's own numbers repeat R232a's** on a newer tree and another boot. A 512-token micro-batch at depth 256
  took 6,511.65 and 6,512.01 µs with the Q8_0 cache, against R232a's 6,517.63 and 6,505.36. With F16 it took 5,257.75
  and 5,274.71, against 5,278.34 and 5,274.28. R232a's single pilot run at 4,096 tokens (88,955.03 µs Q8_0, 66,596.81
  F16) agrees with this run's two (89,598.74 and 89,797.68; 66,482.02 and 67,007.18).

Each class below uses the spread of the repeats and the same-kernel floor
([verdict.tsv](verdict.tsv)):

| Arm (on top of the preset's `_MIN=1` and `_DEC_PASCAL=4`) | Against the preset (Q8_0) | Against the F16 cache (nothing set) |
|---|---|---|
| `GGML_CUDA_FATTN_TILE_PASCAL_LDS=1` | **gain** in all 12 cases: −9.08 % (512 tokens, depth 256) to −16.52 % (8 tokens, 49,152) | **loss** in 11 cases: +9.51 % (8 tokens, 4,096) to +14.69 % (512 tokens, 16,384); **gain** at 8 tokens and depth 256 (−13.76 %) |
| `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` | **gain** in all 12 cases: −30.76 % (512 tokens, depth 256) to −40.59 % (8 tokens, 49,152) | **gain** in all 12 cases: −14.39 % (512 tokens, depth 256) to −34.33 % (8 tokens, depth 256) |
| both | **gain** in all 12 cases, the same kernel as `_KDP=1` | **gain** in all 12 cases |

The preset alone against F16 (the reason for this row): **loss** in 11 of 12 cases (+29.54 % for 8 tokens at 4,096 to
+34.52 % for 512 tokens at 8,192; +23.65 % for 512 tokens at depth 256), **gain** at 8 tokens and depth 256 (−3.44 %).

## What ran

| | |
|---|---|
| Build | Miner Llama's v5 candidate on upstream b12674 at `eee615cdb` (rig tarball `llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674.tar.gz`, md5 `575ef3c702f69f14d7d3022c1287c38f`, sm_61, CUDA 12.9); `test-backend-ops` md5 `7e6731c4d42a98cfd6a90cf4bdeb0cc5`. Its `fattn-tile-pascal.cu` is the same file as in `af00bb13b` (R232a's tree), and its `test-backend-ops` reads R232a's case files unchanged, so one tree serves every arm |
| Card | one P106-090 (card 0 of the second rig), driver 580.178.04, with the second rig's clock profile set ([2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md); card 0 at +1,300 MT/s memory and +256 MHz core, max clocks 2,164 / 4,654 MHz in every reading) and its watchdog running. Nothing else ran on the rig. Before and after: every card at its value, 0 Xid, 0 step-downs. Clocks before and after each run: [clocks.tsv](clocks.tsv) |
| Time | 2026-11-25 11:30:19–14:10:12: correctness 11:30–12:14, timings 12:13:59–14:10:12; every run exit 0 |
| Cases | `run/make_cases.py` (R232a's, unchanged): Ornith's attention node as llama.cpp builds it for its 12 gated-attention layers (head 256, 32 query heads over 2 KV heads, the query a permuted view, K and V permuted views of a 49,152-cell cache, an F16 mask, scale 1/16, F32 accumulation). Correctness: R232a's 64 cases (Q8_0 and F16; n_kv 256 to 49,152; 1, 2, 3, 4, 8 and 512 tokens). Timing: 8 and 512 query tokens (a drafted or pipelined pass, and the preset's `-ub 512` micro-batch) at n_kv 256 (depth 0; llama.cpp pads n_kv to 256), 4,096, 8,192, 16,384, 32,768 and 49,152; Q8_0 for the four Q8_0 arms, F16 for the F16 arm |

### The arms and the kernel each one runs

Each arm is a fresh process (the switches are read once). The kernel per case follows from the routing in
`fattn-tile-pascal.cu` and `fattn.cu`. Every timing run logged the lines its switches select
([perf_runs.tsv](perf_runs.tsv)). The `_KDP` arm logs the tile kernel's plain-scores line and then the line saying K
is scored in int8: the kernel it runs is the int8-scoring one, which decodes V at store, as the identical times of the
`kdp` and `ldskdp` arms show.

| Arm | Set | Timed on | 8 and 512 tokens |
|---|---|---|---|
| `base` | nothing (the preset's F16 cache before 2026-11-24) | F16 cases | 8: llama.cpp's stock kernel; 512: the Pascal tile kernel, F16 |
| `q8` | `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1 GGML_CUDA_FATTN_DEC_PASCAL=4` (the v5 preset's switches) | Q8_0 cases | the Pascal tile kernel, Q8_0 decoded as it is read |
| `lds` | `q8` + `GGML_CUDA_FATTN_TILE_PASCAL_LDS=1` | Q8_0 cases | the Pascal tile kernel, Q8_0 decoded as it is stored |
| `kdp` | `q8` + `GGML_CUDA_FATTN_TILE_PASCAL_KDP=1` | Q8_0 cases | the Pascal tile kernel, K scored in int8 with `dp4a`, V decoded as it is stored |
| `ldskdp` | `q8` + both | Q8_0 cases | the same kernel as `kdp` |

Correctness was run in the four Q8_0 arms; R232a ran it with nothing set on the same cases.

## Correctness ([tests.tsv](tests.tsv))

`test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` against the CPU, one run per arm and case set. The suite is trimmed
to the model's head size (`-p 'hsk=256,hsv=256'`, 367 cases, about 9 minutes a run on the rig's 2-core CPU), as in
R232a. The whole suite passes with each switch through the compute 6.1 PTX on an RTX 3090
([2026-11-23_pr9_pr10_review](../2026-11-23_pr9_pr10_review/README.md)).

| Arm | Suite, head 256 | Ornith cases | Pascal paths logged |
|---|---|---|---|
| `q8` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 and F16; the decode kernel |
| `lds` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 decoded at store, and F16; the decode kernel |
| `kdp` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 K in int8 with `dp4a`, and F16; the decode kernel |
| `ldskdp` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 decoded at store with K in int8 with `dp4a`, and F16; the decode kernel |

The suite's error check (normalized mean squared error against the CPU) passes with `_KDP=1`, but it does not show
how much the int8 query moves a real model's output. A Q8_0 cache's quality was checked without `_KDP`
([Q1707](../2026-11-23_c976_ornith_q8_cache_kld/README.md)). Pull request 9's decode kernel, already in the preset,
also scores an int8 query with `dp4a`, and passed that check. The same check with `_KDP=1` is a new row
([Q1836](../../../quants/experiments.md)).

## Times ([perf.tsv](perf.tsv): every case, arm and repeat)

`test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT --test-file <cases>` runs each case back to back for at least a
second and about 32 GB of traffic, and reports µs per call. There were two rounds in ABBA order (`base q8 lds kdp
ldskdp`, then the reverse), so every arm has two repeats in separate processes.

512-token prompt micro-batches, µs per call (one layer), each repeat by value:

| n_kv | F16, nothing set | Q8_0, the preset | preset + `_LDS=1` | preset + `_KDP=1` | preset + both |
|---:|---|---|---|---|---|
| 256 | 5,257.75, 5,274.71 | 6,511.65, 6,512.01 | 5,929.46, 5,911.60 | **4,511.02, 4,505.97** | 4,509.33, 4,505.50 |
| 4,096 | 66,482.02, 67,007.18 | 89,598.74, 89,797.68 | 76,235.55, 76,113.39 | **54,866.95, 54,643.26** | 54,761.82, 54,657.73 |
| 8,192 | 132,757.47, 133,148.76 | 179,079.53, 178,610.57 | 152,053.01, 151,796.54 | **107,861.90, 107,920.41** | 108,162.97, 108,138.45 |
| 16,384 | 264,082.41, 264,399.45 | 354,807.60, 355,207.25 | 303,006.34, 303,118.34 | **214,876.32, 214,508.41** | 214,827.07, 215,134.32 |
| 32,768 | 529,909.92, 529,528.00 | 707,655.57, 706,905.72 | 606,433.66, 604,613.69 | **428,157.91, 428,942.10** | 428,790.92, 429,177.96 |
| 49,152 | 794,392.53, 793,804.17 | 1,063,396.88, 1,061,487.98 | 910,594.38, 908,178.88 | **643,452.22, 644,569.29** | 644,822.15, 644,245.28 |

8-token calls, µs per call (one layer), each repeat by value:

| n_kv | F16, nothing set | Q8_0, the preset | preset + `_LDS=1` | preset + `_KDP=1` | preset + both |
|---:|---|---|---|---|---|
| 256 | 128.96, 129.90 | 124.53, 125.43 | 111.65, 111.59 | **84.95, 85.05** | 85.29, 85.03 |
| 4,096 | 1,149.63, 1,165.87 | 1,499.99, 1,499.60 | 1,271.17, 1,264.48 | **914.11, 911.22** | 912.32, 912.76 |
| 8,192 | 2,239.58, 2,266.07 | 2,964.77, 2,964.17 | 2,494.43, 2,481.70 | **1,790.60, 1,784.94** | 1,788.39, 1,787.97 |
| 16,384 | 4,403.90, 4,447.82 | 5,872.30, 5,870.73 | 4,928.46, 4,900.89 | **3,515.17, 3,503.50** | 3,512.48, 3,514.18 |
| 32,768 | 8,783.69, 8,847.67 | 11,725.58, 11,723.43 | 9,813.76, 9,773.51 | **7,005.73, 6,984.20** | 7,000.61, 7,001.80 |
| 49,152 | 13,077.48, 13,155.71 | 17,573.27, 17,566.13 | 14,699.69, 14,635.19 | **10,446.09, 10,428.91** | 10,445.57, 10,443.55 |

Change of the mean, `_KDP=1` against the preset and against F16 (from [verdict.tsv](verdict.tsv)):

| n_kv | 512 tokens, against the preset | 512 tokens, against F16 | 8 tokens, against the preset | 8 tokens, against F16 |
|---:|---:|---:|---:|---:|
| 256 | −30.76 % | −14.39 % | −31.99 % | −34.33 % |
| 4,096 | −38.96 % | −17.96 % | −39.15 % | −21.17 % |
| 8,192 | −39.67 % | −18.85 % | −39.69 % | −20.64 % |
| 16,384 | −39.52 % | −18.75 % | −40.23 % | −20.71 % |
| 32,768 | −39.41 % | −19.10 % | −40.34 % | −20.65 % |
| 49,152 | −39.38 % | −18.90 % | −40.59 % | −20.43 % |

The pull request's own table (one GTX 1070, one layer, 128 query rows) has `_KDP` 20.2 % faster than an F16 cache at
39k keys (24,951 against 31,250 µs). Here, on a P106 at Ornith's shape with 512 rows, it is 19.10 % faster at 32,768
and 18.90 % at 49,152.

### How the classes were set ([floor.tsv](floor.tsv))

The rule was fixed in `run/summarize.py` before any timing run ended. A **gain** needs three things: every repeat of
the arm faster than every repeat of the reference, the means further apart than the larger repeat spread, and a change
larger than the same-kernel floor. A **loss** is the same the other way. Anything else is **negligible**.

The floor is the larger of two numbers:
- R232a's same-kernel difference between arms: 1.71 % at 1 to 8 tokens, 2.69 % at 512 tokens;
- this run's own largest difference between `kdp` and `ldskdp`, which run the same kernel: 0.19 % at 8 tokens and
  0.24 % at 512 tokens.

So R232a's floor applied. The smallest change classed is 9.08 % (`_LDS=1` against the preset, 512 tokens at depth 256),
and the smallest loss is 9.51 % (`_LDS=1` against F16, 8 tokens at 4,096). No result is near the floor.

## Files

| File | What it holds |
|---|---|
| [tests.tsv](tests.tsv) | each correctness run: arm, switches, case set, exit code, passed of run, failed, skipped, Pascal paths logged |
| [perf.tsv](perf.tsv) | each Ornith case and arm: the kernel, µs per call in each repeat, mean, spread |
| [verdict.tsv](verdict.tsv) | each Q8_0 arm per case against the preset and against F16: repeats, means, ratio, larger repeat spread, class, floor used, and the est. 12-layer difference from the preset in ms a call |
| [floor.tsv](floor.tsv) | `ldskdp` against `kdp` (the same kernel) per case |
| [perf_runs.tsv](perf_runs.tsv) | the order of the timing runs and the Pascal paths each logged |
| [clocks.tsv](clocks.tsv) | card 0's P-state, clocks and memory before and after each run |
| `run/make_cases.py` | writes the Ornith cases as `test-backend-ops --test-file` lines (R232a's) |
| `run/cases.txt`, `run/cases_perf_q8.txt`, `run/cases_perf_f16.txt` | the cases as run (correctness: 64; timing: 12 Q8_0 and 12 F16) |
| `run/c1037.sh` | the driver: correctness per arm, then timing in ABBA rounds |
| `run/summarize.py` | the tables from the driver's output, with the class rule |

## Commands

On the rig (the build unpacked in `BIN`, the run folder in `OUT`):

```bash
python3 -I run/make_cases.py > cases.txt
python3 -I run/make_cases.py --types q8_0 --rows 8 --prompt-depths 256,4096,8192,16384,32768,49152 > cases_perf_q8.txt
python3 -I run/make_cases.py --types f16 --rows 8 --prompt-depths 256,4096,8192,16384,32768,49152 > cases_perf_f16.txt
BIN=/path/to/llama.cpp-20261124-eee615cdb-pascal-p106-v5-cand-b12674/bin OUT=$PWD/out CASES=$PWD/cases.txt \
  PCASES_Q8=$PWD/cases_perf_q8.txt PCASES_F16=$PWD/cases_perf_f16.txt DEV=0 REPEATS=2 bash run/c1037.sh
```

Anywhere, on the copied output: `python3 -I run/summarize.py out .`

Provenance and licence: every number here was measured for this folder on the build named above, except R232a's,
which come from its own folder, and the pull request's GTX 1070 table, which is quoted from the pull request. The
scripts are this repository's. No model file is read: the cases fill the tensors with random values.
