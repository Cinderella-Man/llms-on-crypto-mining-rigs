# The shared expert's Q4_0 and Q5_0 downs on the multi-column kernel: the check harness (C755, Q1331, 2026-11-13)

What this is: the code part of registry row [Q1331](../../../quants/experiments.md). It adds a mode to
`bench/mmvq_pascal/verify/multitoken_bitid`, builds it for a P106 rig and runs the checks a build machine's CPU can run.
The rig run is R150 (Q1456): [the rig check](#the-rig-check-r150).

**The answer, on a P106 (R150):** the switch is a **gain** and exact. With it, the 23 products of a four-token verify
pass take 0.646, 0.643, 0.646 and 0.649 ms (two loads, two requests each) instead of 0.857, 0.862, 0.858 and 0.860. That
is 0.213 ms less (spread 0.005), and the whole verify pass is 0.174 ms shorter (spread 0.127). They are then bit for bit
the one-token products, which the k-column kernel's Q5_0 products are not, even on ordinary inputs. At 5–8 tokens the
switch is a loss (llama.cpp's kernel is slower than the k-column kernel there), so Q1458 is needed.

**Background** ([C731](../2026-11-11_c731_shared_expert/README.md)). In Qwen3.8-Flash-Next's release file (Swift
GSQ-RCO Q2_0), 16 of the shared expert's down projections are Q4_0 and 7 are Q5_0 (640 → 2,560). In a four-token
verify pass they take 0.595 and 0.285 ms, 0.880 ms together (R41). The preset sets `GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0`,
so the k-column kernel runs them at 2–8 tokens. Its block arithmetic is round(I · d) · d8, while the one-token kernel's
is d · round(d8 · I) (I is a block's integer sum). The two agree while |I| < 2^13. `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0`
needs no engine change: it gives these products to `mul_mat_vec_q_pascal_mc`, which runs the one-token kernel's
arithmetic for each token. Q824's table timed that kernel faster than the k-column kernel there at 2–4 tokens.

**The answer, on the CPU:**
- **The mode works.** `multitoken_bitid shdown` on ggml's CPU backend: 72 comparisons, 0 with differences. The CPU
  computes each column alone, so this checks the harness itself, not a GPU kernel.
- **The "big" inputs reach the blocks where the kernels part.** In each of the 24 "big" products, every block's integer
  sum is past 2^13 (est.; for example 204,800 of 204,800 blocks for Q4_0 at 2,560 × 640 and 4 tokens). With standard
  normal activations no Q4_0 block gets there, and 33 to 144 of 85,248 to 327,760 Q5_0 blocks do. So on a GPU, the
  k-column arm must differ in the "big" products.
- **The switch moves only the 23 shared-expert downs.** The served files hold 1,258 tensors (the three splits of the
  main model and the MTP draft). The only Q4_0 or Q5_0 ones are the 16 + 7 `ffn_down_shexp.weight`, 640 × 2,560; the
  draft has none. The switch is read in two places. `mmvq-pascal-kcol.cu` is the k-column hook, which declines the
  listed types. `mmvq-pascal-glu.cu` is C757's gate/up launch, which follows the same list; no gate or up of this model
  is Q4_0 or Q5_0.
- **One side effect, found in the dispatch code:** `mul_mat_vec_q_pascal_mc` takes at most 4 tokens
  (`MMVQP_MAX_COLS`). So at 5–8 tokens the skip moves these products from the k-column kernel to llama.cpp's
  `mul_mat_vec_q`. That kernel's sums are not the one-token sums either, and nothing has timed it there. A four-token
  verify pass never runs 5–8 tokens, but a short prompt chunk does. R150 times 5–8 tokens too. A form of the switch
  that keeps 5–8 tokens on the k-column kernel is new row Q1458 (C828).

**What ran on the rig instead of the build machine.** Nothing in this row's code part ran on a GPU. No GPU of the
build machine was free for our jobs that day, so the mode's runs through the compute_61 PTX on a newer card were
skipped. The four GPU arms, two of them fault arms, ran natively on a P106 as R150's first step.

## How the switch routes a product

`ggml_cuda_mul_mat_vec_q` asks the k-column hook first for a plain product (no expert ids, no fusion).
`ggml_cuda_mmvq_pascal_kcol` returns at once for a type in `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP`. The product then goes to
the Pascal kernels of `mmvq-pascal.cu`:

| Tokens | The preset (`KCOL_TABLE=0`) | The preset with `KCOL_SKIP=q4_0,q5_0` | The default table (`KCOL_TABLE=1`) |
|---|---|---|---|
| 1 | the one-token Pascal kernel | the one-token Pascal kernel | the one-token Pascal kernel |
| 2–4 | k-column | `mul_mat_vec_q_pascal_mc` | `mul_mat_vec_q_pascal_mc` (the table declines Q4_0 and Q5_0 at 2–4 tokens, K ≤ 4,096) |
| 5–8 | k-column | llama.cpp's `mul_mat_vec_q` | k-column |

Every switch is read once per process. [skip_readers.txt](skip_readers.txt) lists every line that names the switch in
the build's tree (`65ac9a47d`). Two of them read it: the k-column hook and C757's gate/up launch. The rest are comments;
`mmvq-pascal-q3kmt.cuh` follows the hook's answer for Q3_K only.

## The mode

`multitoken_bitid shdown` checks Q4_0 and Q5_0 at four shapes:
- 2,560 × 640, the shared expert's down;
- 4,097 × 640, a partial last row block;
- 640 × 2,560;
- 333 × 4,096.

Each product of 2, 3 and 4 tokens is compared, bit for bit, with the same columns run one token at a time. The one-token
products are the reference. Each shape and token count gets three inputs:

| Input | Weights | Activations | Blocks with \|I\| ≥ 2^13 |
|---|---|---|---|
| `MUL_MAT` | normal, standard deviation 0.02 | standard normal | none for Q4_0; 33 to 144 for Q5_0 |
| `MUL_MAT/unif` | the same | uniform on [−1, 1] | none or 1 for Q4_0; 577 to 2,487 for Q5_0 |
| `MUL_MAT/big` | one sign per column for every row, magnitudes 0.01 to 0.02 | that column's sign times 0.75 to 1 | every block |

After each product line comes the count of its blocks with |I| ≥ 2^13. The count is an estimate: the activations are
quantized on the CPU as `quantize_q8_1` does on the GPU. It is a bound on the blocks where the two arithmetics can part;
C731 counted the blocks where they do.

At the start, for each type, the mode prints whether the k-column hook takes it (`ggml_cuda_mmvq_pascal_kcol_on`).
`BITID_CPU=1` runs any mode on ggml's CPU backend instead of CUDA0.

## The checks on the build machine's CPU

[summary.tsv](summary.tsv), by [run/checks.sh](run/checks.sh):

| Check | Verdict | What |
|---|---|---|
| build | OK | `multitoken_bitid` linked against the static ggml of the `pascal-p106-v4` rig build `65ac9a47d` (the merge of C813), image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61 and compute_61 PTX, 0 warnings: **md5 `f1f61714c977a0c2adde28494625f382`**, 91,810,952 bytes |
| cpu | OK | `BITID_CPU=1 multitoken_bitid shdown` in that image without a GPU (the CUDA driver library is the toolkit's stub, never called): 72 product lines, 72 comparisons, 0 with differences ([cpu_table.tsv](cpu_table.tsv), by [run/cpu_table.py](run/cpu_table.py)) |
| route | OK | [types.tsv](types.tsv), every tensor of the served files by [run/gguf_types.py](run/gguf_types.py) (headers only): 1,258 tensors; Q4_0 and Q5_0 only in `ffn_down_shexp.weight`, 16 and 7 ([route.tsv](route.tsv), per file and type [types_count.tsv](types_count.tsv)) |

Between `65ac9a47d` and the line's tip of that day (`2144cf760`), only C822's F32 host code changed. The k-column hook,
`mul_mat_vec_q_pascal_mc` and the one-token kernel are the same.

## What the rig run must show (R150)

[run/rig_c755.sh](run/rig_c755.sh), on a P106 rig, the rig idle, on the `pascal-p106-v4` tarball and the
`multitoken_bitid` above (md5 `f1f61714c977a0c2adde28494625f382`). It loads the rig's own CUDA 12.9 runtime libraries,
as the tarball's binaries do.

1. **bitid:** `multitoken_bitid shdown` in four arms:

   | Arm | Switches | Kernel at 2–4 tokens | Done when |
   |---|---|---|---|
   | `on` | the preset's `KCOL_SMEM=32768 KCOL_TABLE=0`, `KCOL_SKIP=q4_0,q5_0` | `mul_mat_vec_q_pascal_mc` | exit 0: 72 comparisons, 0 differences |
   | `table` | none (the default dispatch table) | `mul_mat_vec_q_pascal_mc` | exit 0: 0 differences |
   | `off` (fault) | the preset's alone | k-column | exit 1: the "big" products differ |
   | `mc0` (fault) | the skip and `GGML_CUDA_MMVQ_PASCAL_MC=0` | llama.cpp's `mul_mat_vec_q` | exit 1: products differ |

   The `off` arm's counts for normal and uniform inputs also show how often the k-column kernel parts on such data.
2. **perf:** `test-backend-ops perf -o MUL_MAT -p 'type_a=(q4_0|q5_0),type_b=f32,m=2560,n=[1-8],k=640,'`, the arms
   `kcol` (the preset's settings) and `mc` (with the skip) alternated, two passes. 1 token runs the one-token kernel in
   both arms, so it gives the spread. 2–4 tokens compare `mul_mat_vec_q_pascal_mc` with the k-column kernel. 5–8 tokens
   compare llama.cpp's kernel with the k-column kernel (for Q1458).
3. **served:** R49's profile of a served verify pass ([verify_prof.sh](../2026-11-10_served_verify_parts/run/verify_prof.sh))
   on the preset with the MTP draft, the skip off and on (`off on on off`, loads `warm prof`). Then the two products'
   ms a four-token pass by value against 0.880 (Q4_0 0.595, Q5_0 0.285), from the traces as C731 read R41's
   ([shexp_times.sh](../2026-11-11_c731_shared_expert/run/shexp_times.sh)). Also the step time from the warm loads'
   server timings.

Done when: the four bitid arms end as the table says; perf gives each type and token count's µs a call per arm and
pass; and the served pair puts the switch in one class: gain, negligible or loss.

## The rig check (R150)

**Where and on what.** One P106 rig, idle, on 2026-11-13 from 12:55:58 to 13:33:51 ([status.txt](rig_check/status.txt)):
bitid 12:55:58–12:56:07, perf 12:56:08–12:57:29, served 12:57:29–13:33:51. The engine was the fork's
`pascal-p106-v4-b12246-c812` build `d679a6d9d`: the `pascal-p106-v4` line plus C812's host switch, which is off by
default. Its k-column kernel, `mul_mat_vec_q_pascal_mc` and the one-token kernel are the same code as in `65ac9a47d`,
the build the harness was linked against. Binaries: `llama-server` md5 `b8365b25864cf1aab0b81b47ec2f40a7`,
`test-backend-ops` md5 `fbeeec6685fa1ad2c24a37207b0177a3`, `multitoken_bitid` md5 `f1f61714c977a0c2adde28494625f382`.
The release candidate `v4-rc1` (`da2920d45`) changes none of these kernels. Its one change in `mmvq.cu` renames the L2
prefetch helper and changes the arguments of `quantize_row_q8_1_cuda`.

**1. bitid: every arm ended as required** ([bitid.tsv](rig_check/bitid.tsv)). Each arm made 72 comparisons, 12 for
each type and input.

| Arm | Exit | Comparisons with differences | Where |
|---|---:|---:|---|
| `on` (the skip) | 0 | 0 | none |
| `table` (no switch) | 0 | 0 | none |
| `off` (the preset alone, the k-column kernel; fault) | 1 | 44 | all 24 "big" products, and 20 others: Q5_0 normal 8 of 12 (23 values), Q5_0 uniform 12 of 12 (676 values); Q4_0 normal and uniform 0 |
| `mc0` (the skip with `MC=0`, llama.cpp's kernel; fault) | 1 | 72 | every product |

So on a P106 the k-column kernel's Q5_0 products part from the one-token products on ordinary inputs too. Under the
skip they are the one-token products bit for bit.

**2. perf: faster at 2–4 tokens, slower at 5–8** ([perf.tsv](rig_check/perf.tsv), [perf_pass.tsv](rig_check/perf_pass.tsv),
raw [perf_us_run.tsv](rig_check/perf_us_run.tsv)). µs a call at m = 2,560, k = 640, pass 1 and pass 2. The est. ms a
pass is (mc − kcol) × 16 for Q4_0 plus × 7 for Q5_0, the served file's layers.

| Tokens | Q4_0 kcol | Q4_0 mc | Q5_0 kcol | Q5_0 mc | est. ms a pass, both types |
|---:|---|---|---|---|---|
| 1 (the same kernel in both arms) | 14.73, 13.97 | 14.01, 14.01 | 14.23, 14.31 | 14.32, 14.34 | −0.011, +0.001 |
| 2 | 28.71, 28.89 | 15.46, 15.43 | 31.97, 32.17 | 21.65, 21.62 | −0.284, −0.289 |
| 3 | 29.89, 30.08 | 17.99, 17.98 | 33.55, 33.74 | 26.32, 26.35 | −0.241, −0.245 |
| 4 | 31.10, 31.31 | 19.34, 19.34 | 34.76, 34.99 | 28.87, 28.82 | −0.229, −0.235 |
| 5 (llama.cpp's kernel under the skip) | 31.72, 31.93 | 42.26, 42.45 | 34.91, 35.13 | 46.96, 47.01 | +0.253, +0.251 |
| 6 | 32.97, 33.19 | 46.68, 46.87 | 35.61, 35.87 | 53.34, 53.47 | +0.343, +0.342 |
| 7 | 37.48, 38.31 | 55.46, 55.75 | 37.73, 38.00 | 60.79, 60.67 | +0.449, +0.438 |
| 8 | 56.03, 56.36 | 64.78, 64.90 | 61.83, 62.24 | 76.68, 76.70 | +0.244, +0.238 |

At 5–8 tokens llama.cpp's `mul_mat_vec_q` is slower than the k-column kernel for both types at every token count, in
both passes. A prompt chunk of 5–8 tokens pays that under the skip, so the skip is wanted at 2–4 tokens only. That is
[Q1458](../../../quants/experiments.md).

**3. served: a gain** ([served.tsv](rig_check/served.tsv), [products.tsv](rig_check/products.tsv),
[step.tsv](rig_check/step.tsv), [replies.tsv](rig_check/replies.tsv)). Four loads of the preset with the MTP draft,
the skip `off on on off`. Each load answered the greedy request twice under the profiler; 81 and 80 typical verify
passes were kept per load. The products' ms come from R49's
[verify_parts.py](../2026-11-10_served_verify_parts/run/verify_parts.py) `labels.tsv` (launch groups
`MUL_MAT ffn_down_shexp:q4_0` and `:q5_0`, both card groups summed; 16 and 7 a pass in every load), read by
[run/r150_tables.py](run/r150_tables.py). C731's `shexp_times.sh` reads R41's prompt-pass cut, which a served trace does
not have. ms a four-token verify pass, all cards, in the order load 1 off (r1, r2), load 4 off (r1, r2) against
load 2 on (r1, r2), load 3 on (r1, r2):

| Part | R41 (C731) | off | on | on − off (means) | spread | class |
|---|---:|---|---|---:|---:|---|
| Q4_0 downs | 0.595 | 0.580, 0.586, 0.581, 0.581 | 0.387, 0.383, 0.389, 0.385 | −0.196 | 0.006 | gain |
| Q5_0 downs | 0.285 | 0.276, 0.276, 0.277, 0.280 | 0.258, 0.261, 0.258, 0.263 | −0.017 | 0.006 | gain |
| both | 0.880 | 0.857, 0.862, 0.858, 0.860 | 0.646, 0.643, 0.646, 0.649 | −0.213 | 0.005 | gain |
| the rest of the pass | | 99.684, 99.710, 99.723, 99.748 | 99.698, 99.775, 99.728, 99.821 | +0.040 | 0.124 | negligible |
| the verify pass | | 100.540, 100.571, 100.581, 100.608 | 100.343, 100.418, 100.375, 100.470 | −0.174 | 0.127 | gain |

The server's own timings (warm loads, no profiler), ms a step, off loads 1 and 4 against on loads 2 and 3:
- request 1: 131.36 and 129.23 off, 127.06 and 126.92 on: −3.31 (spread 2.13).
- request 2: 121.67 and 121.64 off, 121.16 and 121.25 on: −0.45 (spread 0.09).

Both are a gain by the rule. Under the profiler the step moved by −0.28 (spread 1.59) and +0.39 (spread 1.81):
negligible. All 16 replies had the same text (md5 `e460d038a1f6…`), with 2.98 and 3.05 tokens a step in every load.
The server's step moved by more than the profiled GPU time of the two products. The profile does not show where the
rest comes from.

**Verdict:** `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q4_0,q5_0` is a **gain** for Flash-Next's verify pass and makes these
products exact. In its present form it is a **loss** at 5–8 tokens. Q1458 restricts it to 2–4 tokens, which keeps the
gain without the loss.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the three checks with their verdicts |
| [cpu_table.tsv](cpu_table.tsv) | the CPU run: each type, shape, token count and input, its result, and the blocks with \|I\| ≥ 2^13 (est.) |
| [types.tsv](types.tsv) | every tensor of the served files: name, type, shape, file |
| [types_count.tsv](types_count.tsv) | tensors per file and type, counted from types.tsv |
| [route.tsv](route.tsv) | the Q4_0 and Q5_0 tensors, per file and name |
| [skip_readers.txt](skip_readers.txt) | every line of the build's ggml that names `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` |
| [run/checks.sh](run/checks.sh) | the build, the CPU run and the routing dump |
| [run/gguf_types.py](run/gguf_types.py) | lists a GGUF file's tensors from its header (standard library only) |
| [run/cpu_table.py](run/cpu_table.py) | the CPU run's output as cpu_table.tsv |
| [run/rig_c755.sh](run/rig_c755.sh) | R150 on a P106 rig: bitid, perf, served |
| [run/r150_tables.py](run/r150_tables.py) | R150's bitid, perf, served and step tables (standard library only) |
| [rig_check/r150.md](rig_check/r150.md) | all of R150's tables in one report |
| [rig_check/bitid.tsv](rig_check/bitid.tsv) | bitid: each arm, input and type, its comparisons, those with differences and the values differing |
| [rig_check/perf.tsv](rig_check/perf.tsv), [perf_pass.tsv](rig_check/perf_pass.tsv) | perf: µs a call per arm and pass, mc − kcol, est. ms a pass |
| [rig_check/perf_us_run.tsv](rig_check/perf_us_run.tsv) | perf, raw: arm, pass, type, m, k, tokens, µs a call |
| [rig_check/served.tsv](rig_check/served.tsv) | served: the two products, the rest and the verify pass per load and request, on − off, spread, class |
| [rig_check/products.tsv](rig_check/products.tsv) | served: the one-op weight products by type, ms a pass ([served_table.py](../2026-11-11_r79_c722_p106/run/served_table.py)) |
| [rig_check/replies.tsv](rig_check/replies.tsv), [step.tsv](rig_check/step.tsv) | served: the server's timings of every request; the step time's class |
| [rig_check/status.txt](rig_check/status.txt) | the run's status log |

Also changed for this row: `bench/mmvq_pascal/verify/multitoken_bitid.cpp` (the mode `shdown`, `BITID_CPU=1`).

## Commands

On the build machine (a rig build of `pascal-p106-v4` in `SRC`, its `build/` kept):

```bash
M=<work dir> REPO=<this repository> LOCK=<flock file> SRC=<rig build tree> \
  MODEL_FILES="<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf>" DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> \
  bash docs/data/2026-11-13_c755_shexp_downs/run/checks.sh
```

On the rig (R150):

```bash
BIN=<tarball>/bin OUT=<out> BITID=<multitoken_bitid> MODEL=<model> DRAFT=<draft> \
  PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
  REQ=docs/data/2026-11-04_draft_one_card/run/greedy_request.json \
  VERIFY_PROF=docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
  setsid -f nohup bash docs/data/2026-11-13_c755_shexp_downs/run/rig_c755.sh > console.txt 2>&1 < /dev/null
```

R150's tables, on any CPU, from the rig's `out/` and the four `served/<k>-<arm>/prof/prof-verify.csv` traces in `TR`:

```bash
for d in 1-off 2-on 3-on 4-off; do mkdir -p tables/served/$d
  python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
    --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py \
    --event-us 7.6 --out tables/served/$d $TR/$d/prof/prof-verify.csv > tables/served/$d/report.md; done
python3 -I docs/data/2026-11-11_r79_c722_p106/run/served_table.py out/served --parts tables/served --out tables/served \
  --types q4_0,q5_0 > tables/served/served.md
python3 -I docs/data/2026-11-13_c755_shexp_downs/run/r150_tables.py --perf out/perf/us_run.tsv --parts tables/served \
  --replies tables/served/replies.tsv --bitid out/bitid --out tables > tables/r150.md
```

Provenance and licence: the tables are this repository's own output. The tensor list comes from the headers of UkisAI's
Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and of the MTP draft named in the preset: names, types and shapes only.
