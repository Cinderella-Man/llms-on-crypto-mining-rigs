# Q1706 (a): pull request 9's attention switches at Ornith-1.5-122B's shape on one P106-090 (2026-11-24)

Pull request 9 added two switches to the fork's compute 6.1 attention code
([Q1703](../../../quants/experiments.md)): `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` lets the Pascal tile kernel take calls
of 1 or 2 query tokens (by default it starts at 3 with a Q8_0 cache and 9 with F16), and
`GGML_CUDA_FATTN_DEC_PASCAL=N` sends calls of up to N tokens with a Q8_0 cache at head 256 to a new decode kernel.
[Q1707](../2026-11-23_c976_ornith_q8_cache_kld/README.md) showed a Q8_0 cache keeps Ornith-1.5-122B Pascal N2P GGUF's
quality within noise of the F16 cache its v5 preset uses. This folder is the P106 half: does the attention suite pass
on a P106 with each switch unset and set, and how long does one attention call take at Ornith's exact shape, depth 0
to the preset's 49,152 tokens. Part (b), Ornith's agent check with the Q8_0 cache, is a separate run.

**Result: every case passes, and with a Q8_0 cache the two switches together make Ornith's one-token attention call
3.9 to 4.5 times faster than the stock kernel from 4,096 tokens on (3.4 at depth 0) and 1.8 to 1.9 times faster than
the F16 cache the preset uses today.**
- **Correctness:** 367 of 367 of the suite's head-256 `FLASH_ATTN_EXT` cases and 64 of 64 Ornith-shaped cases pass in
  each of the four arms (nothing set, `_MIN=1`, `_DEC_PASCAL=4`, both), the decode kernel's log line present exactly
  in the arms that set it ([tests.tsv](tests.tsv)).
- **One token (decode), Q8_0 cache, one layer's call:** 1,455.09 and 1,455.88 µs at 8,192 tokens with nothing set
  (llama.cpp's vector kernel), 780.62 and 780.19 with `_MIN=1` (the Pascal tile kernel), **335.23 and 334.99 with
  both** (the decode kernel); at 49,152 tokens 7,464.84 and 7,447.15 against **1,894.25 and 1,895.69**. The F16 cache
  with nothing set, the preset's path today: 619.89 and 619.59 µs at 8,192, 3,660.10 and 3,629.50 at 49,152.
- **Per decoded token** (Ornith's 12 attention layers, est. from one card's times): the Q8_0 cache with both switches
  takes est. 3.42 ms less than the F16 cache at 8,192 tokens and est. 21.00 ms less at 49,152
  ([q8_vs_f16.tsv](q8_vs_f16.tsv)). R176 put Ornith's whole one-token step at 64.41 and 64.42 ms at 8,192 tokens (v4.0, the same rig).
- **Where the Q8_0 cache is slower than F16:** 4-token calls (the decode kernel: +7.1 % at 8,192, +10.5 % at 49,152),
  8-token calls (the Pascal tile kernel: +31.5 % at 8,192, +33.6 % at 49,152) and prompts (a 512-token micro-batch:
  +23.4 % at depth 256 over two repeats each; +33.6 % at depth 4,096 in one pilot run each). Ornith's preset decodes
  one token a pass, so decode gains; its prompts may lose, which part (b) will show.
- **F16 cache with `_MIN=1`** (the preset's cache, one switch): one-token calls 7.8 % faster at 8,192 tokens and 9.7 %
  at 49,152, but 9.9 % slower at depth 256; two-token calls 30.4 % faster at 8,192. A separate row
  ([Q1769](../../../quants/experiments.md)) tries it in the preset.

Verdict per switch (each class with the spread of the repeats and the same-kernel floor below; [verdict.tsv](verdict.tsv)):

| Switch | Q8_0 cache | F16 cache |
|---|---|---|
| `GGML_CUDA_FATTN_DEC_PASCAL=4` with `_MIN=1` (against `_MIN=1` alone) | **gain** at 1 and 2 tokens, every depth: 335.11 against 780.40 µs (means) at 8,192, 1 token; 631.22 against 786.84 at 2 | never reached (Q8_0 only) |
| `GGML_CUDA_FATTN_DEC_PASCAL=4` alone | **gain** at 3 and 4 tokens, every depth (3 tokens 895.24 against 1,520.17 µs at 8,192; 4 tokens 1,212.28 against 1,525.01); 1 and 2 tokens never reach it without `_MIN=1` | never reached |
| `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` | **gain** at 1 and 2 tokens, every depth (the tile kernel instead of the vector kernel: 780.40 against 1,455.49 µs at 8,192, 1 token); 3 tokens and more unchanged | 1 token: **loss** at depth 256 (34.95 against 31.80 µs), **gain** from 4,096 on (571.57 against 619.74 at 8,192); 2 tokens **gain** at every depth; 3 and 4 tokens **negligible**; 8 tokens **gain** (−3.4 % at 8,192) |
| Both (against nothing set) | **gain** at 1 to 4 tokens, every depth: 1 token 0.2302 of base at 8,192, 0.2542 at 49,152 | as `_MIN=1` |

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v5` at `af00bb13b` (rig tarball `llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2.tar.gz`, md5 `b51ad2740d5822e7e14af656c413ec22`, sm_61, CUDA 12.9), which holds pull request 9 with the decode kernel's mask fix; `test-backend-ops` md5 `65fd4d6b430b6f326bedc3ca2233db06` |
| Card | one P106-090 (card 0 of the second rig), driver 580.178.04, with the second rig's clock profile set ([2026-11-19_r209_oc_v5](../2026-11-19_r209_oc_v5/README.md); card 0 at +1,300 MT/s memory and +256 MHz core, max clocks 2,164 / 4,654 MHz) and its watchdog running; nothing else ran on the rig. Clocks before and after each run: [clocks.tsv](clocks.tsv) |
| Time | 2026-11-24 14:41:19–16:01:29: correctness 14:41–15:24, timings 15:24:51–16:01:29; every run exit 0 |
| Cases | `run/make_cases.py`: Ornith's attention node as llama.cpp builds it for its 12 gated-attention layers (head 256, 32 query heads over 2 KV heads, the query a permuted view, K and V permuted views of a 49,152-cell cache, an F16 mask, scale 1/16, F32 accumulation); K/V Q8_0 and F16; n_kv 256 (depth 0; llama.cpp pads n_kv to 256), 4,096, 8,192, 16,384, 32,768 and 49,152; 1, 2, 3, 4 and 8 query tokens (decode, pipelined and drafted passes) and 512 (the preset's `-ub 512`) at n_kv 256, plus 4,096 for correctness only |

### The arms and the kernel each one runs

Each arm is a fresh process (the switches are read once). The kernel per case follows from the routing in
`fattn-tile-pascal.cu` and `fattn.cu`; the logs show each Pascal path's line in the arms expected
([perf_runs.tsv](perf_runs.tsv)).

| Arm | Set | Q8_0, 1–2 tokens | Q8_0, 3–4 | Q8_0, 8 and 512 | F16, 1–8 | F16, 512 |
|---|---|---|---|---|---|---|
| `base` | nothing | llama.cpp's vector kernel | Pascal tile | Pascal tile | llama.cpp's stock kernel | Pascal tile |
| `min1` | `GGML_CUDA_FATTN_TILE_PASCAL_MIN=1` | Pascal tile | Pascal tile | Pascal tile | Pascal tile | Pascal tile |
| `dec4` | `GGML_CUDA_FATTN_DEC_PASCAL=4` | llama.cpp's vector kernel | Pascal decode | Pascal tile | llama.cpp's stock kernel | Pascal tile |
| `both` | the two | Pascal decode | Pascal decode | Pascal tile | Pascal tile | Pascal tile |

## Correctness ([tests.tsv](tests.tsv))

`test-backend-ops test -b CUDA0 -o FLASH_ATTN_EXT` against the CPU, one run per arm and case set. The rig's 2-core CPU
computes each reference, so the suite was trimmed to the model's head size (`-p 'hsk=256,hsv=256'`, 367 cases; 9 min a
run); the whole suite (4,679 cases) passes in each arm through the compute 6.1 PTX on an RTX 3090
([2026-11-23_pr9_pr10_review](../2026-11-23_pr9_pr10_review/README.md)).

| Arm | Suite, head 256 | Ornith cases | Pascal paths logged |
|---|---|---|---|
| `base` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 and F16 |
| `min1` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 and F16 |
| `dec4` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 and F16; the decode kernel |
| `both` | 367 / 367 | 64 / 64 | prompt kernel, Q8_0 and F16; the decode kernel |

## Times ([perf.tsv](perf.tsv): every case, arm and repeat)

`test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT --test-file cases_perf.txt`, which runs each case back to back for
at least a second and about 32 GB of traffic and reports µs per call. Two rounds in ABBA order (`base min1 dec4
both`, then `both dec4 min1 base`), so every arm has two repeats in separate processes.

One-token calls, µs per call, each repeat by value:

| n_kv | F16, nothing set (the preset) | Q8_0, nothing set | Q8_0, `_MIN=1` | Q8_0, `_DEC_PASCAL=4` | Q8_0, both |
|---:|---|---|---|---|---|
| 256 | 31.83, 31.76 | 78.22, 78.57 | 45.82, 45.75 | 78.65, 78.58 | **23.03, 23.09** |
| 4,096 | 317.37, 316.90 | 811.50, 814.93 | 398.81, 398.67 | 815.93, 813.98 | **179.43, 179.33** |
| 8,192 | 619.89, 619.59 | 1,455.09, 1,455.88 | 780.62, 780.19 | 1,458.90, 1,458.59 | **335.23, 334.99** |
| 16,384 | 1,227.02, 1,225.02 | 2,720.62, 2,721.71 | 1,537.32, 1,535.93 | 2,726.15, 2,725.12 | **643.11, 642.73** |
| 32,768 | 2,444.28, 2,435.03 | 5,294.13, 5,288.07 | 3,012.63, 3,010.44 | 5,297.87, 5,298.06 | **1,267.79, 1,264.86** |
| 49,152 | 3,660.10, 3,629.50 | 7,464.84, 7,447.15 | 4,497.93, 4,492.33 | 7,446.84, 7,440.51 | **1,894.25, 1,895.69** |

The decode kernel reads the Q8_0 cache at about 27 GB/s at 8,192 tokens (8.9 MB in 335 µs) and 28 GB/s at 49,152,
the F16 path at about 27 GB/s too (16.8 MB in 620 µs): both far below the card's ~190 GB/s, so a kernel that reads at
card speed (Q702) would still have room.

The Q8_0 cache with both switches against the F16 cache with nothing set, by tokens per call (means, from
[q8_vs_f16.tsv](q8_vs_f16.tsv)):

| Tokens | 4,096 | 8,192 | 16,384 | 32,768 | 49,152 |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5656 | 0.5407 | 0.5244 | 0.5191 | 0.5199 |
| 2 | 0.7993 | 0.7613 | 0.7537 | 0.7613 | 0.7667 |
| 3 | 0.7974 | 0.7955 | 0.8032 | 0.8320 | 0.8345 |
| 4 | 1.0939 | 1.0714 | 1.0777 | 1.0980 | 1.1053 |
| 8 | 1.2889 | 1.3152 | 1.3293 | 1.3341 | 1.3361 |

Prompt micro-batches (512 tokens, the same Pascal tile kernel in every arm): at depth 256 the Q8_0 cache took 6,517.63
and 6,505.36 µs, the F16 cache 5,278.34 and 5,274.28 (nothing set). One pilot run each at depth 4,096 (same build and
card, before the main run): Q8_0 88,955.03 µs, F16 66,596.81. Pull request 9's other two switches (`_LDS`, `_KDP`)
work on exactly these calls and were not timed here ([Q1770](../../../quants/experiments.md)).

### How the classes were set ([floor.tsv](floor.tsv))

The rule fixed before the run: a gain needs every repeat of the arm faster than every repeat of the base, with the
means further apart than the larger repeat spread; a loss the same the other way; anything else negligible. That rule
was too narrow: pairs of arms whose process runs the same kernel for a case (for example `_DEC_PASCAL=4` on an F16
case) differed by up to 1.71 % at 1 to 8 tokens (−1.71 % on F16 at depth 256 with 4 tokens; −1.51 % on F16 at 32,768
with 2 tokens), more than their own repeat spread, and by up to 2.69 % on the 512-token prompt case. So the final class
also asks the change to exceed 1.71 %. That moves 14 F16 results at 3 and 4 tokens (`_MIN=1` and both, −1.66 % to
+0.33 %) to negligible, 12 from gain and 2 from loss, and none of the Q8_0 results, which are changes of 18 % to 82 %.

## Files

| File | What it holds |
|---|---|
| [tests.tsv](tests.tsv) | each correctness run: arm, case set, exit code, passed of run, failed, skipped, Pascal paths logged |
| [perf.tsv](perf.tsv) | each Ornith case and arm: µs per call in each repeat, mean, spread |
| [verdict.tsv](verdict.tsv) | each switch arm against `base` per case: kernels, means, difference, ratio, larger repeat spread, class by the rule fixed before the run, final class |
| [floor.tsv](floor.tsv) | every pair of arms that runs the same kernel for a case, and their difference |
| [q8_vs_f16.tsv](q8_vs_f16.tsv) | the Q8_0 cache with both switches against the F16 cache with nothing set, with the 12-layer difference per pass (est.) |
| [perf_runs.tsv](perf_runs.tsv) | the order of the timing runs and the Pascal paths each logged |
| [clocks.tsv](clocks.tsv) | card 0's P-state, clocks and memory before and after each run |
| `run/make_cases.py` | writes the Ornith cases as `test-backend-ops --test-file` lines |
| `run/cases.txt`, `run/cases_perf.txt` | the cases as run (correctness: 64; timing: 62, without the 512-token case at 4,096) |
| `run/r232a.sh` | the driver: correctness per arm, then timing in ABBA rounds |
| `run/summarize.py` | the tables from the driver's output |

## Commands

On the rig (the build unpacked in `BIN`, the run folder in `OUT`):

```bash
python3 -I run/make_cases.py > cases.txt
python3 -I run/make_cases.py --prompt-depths 256 > cases_perf.txt
BIN=/path/to/llama.cpp-20261124-af00bb13b-pascal-p106-v5-land-batch2/bin OUT=$PWD/out CASES=$PWD/cases.txt \
  PCASES=$PWD/cases_perf.txt DEV=0 REPEATS=2 bash run/r232a.sh
```

Anywhere, on the copied output: `python3 -I run/summarize.py out .`

Provenance and licence: every number here was measured for this folder on the build named above; the scripts are this
repository's. No model file is read: the cases fill the tensors with random values.
