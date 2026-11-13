# The shared expert's Q4_0 and Q5_0 downs on the multi-column kernel: the check harness (C755, Q1331, 2026-11-13)

What this is: the code part of registry row [Q1331](../../../quants/experiments.md). It adds a mode to
`bench/mmvq_pascal/verify/multitoken_bitid`, builds it for a P106 rig and runs the checks a build machine's CPU can run.
The rig run is [R150](#what-the-rig-run-must-show-r150) (Q1456). It has not run yet.

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

**What waited for the rig, and why.** Nothing in this row ran on a GPU. No GPU of the build machine was free for our
jobs that day, so the mode's runs through the compute_61 PTX on a newer card were skipped. The four GPU arms and the
two fault arms are R150's first step, natively on a P106.

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
| build | OK | `multitoken_bitid` linked against the static ggml of the `pascal-p106-v4` rig build `65ac9a47d` (the merge of C813), image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, sm_61 and compute_61 PTX, 0 warnings: **md5 `00ecfab07d32021d060dff1c5b6960a9`**, 91,806,832 bytes |
| cpu | OK | `BITID_CPU=1 multitoken_bitid shdown` in that image without a GPU (the CUDA driver library is the toolkit's stub, never called): 72 product lines, 72 comparisons, 0 with differences ([cpu_table.tsv](cpu_table.tsv)) |
| route | OK | [types.tsv](types.tsv), every tensor of the served files by [run/gguf_types.py](run/gguf_types.py) (headers only): 1,258 tensors; Q4_0 and Q5_0 only in `ffn_down_shexp.weight`, 16 and 7 ([route.tsv](route.tsv), per file and type [types_count.tsv](types_count.tsv)) |

Between `65ac9a47d` and the line's tip of that day (`2144cf760`), only C822's F32 host code changed. The k-column hook,
`mul_mat_vec_q_pascal_mc` and the one-token kernel are the same.

## What the rig run must show (R150)

[run/rig_c755.sh](run/rig_c755.sh), on a P106 rig, the rig idle, on the `pascal-p106-v4` tarball and the
`multitoken_bitid` above (md5 `00ecfab07d32021d060dff1c5b6960a9`). It loads the rig's own CUDA 12.9 runtime libraries,
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

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | the three checks with their verdicts |
| [cpu_table.tsv](cpu_table.tsv) | the CPU run: each type, shape, token count and input, its result, and the blocks with \|I\| ≥ 2^13 (est.) |
| [types.tsv](types.tsv) | every tensor of the served files: name, type, shape, file |
| [types_count.tsv](types_count.tsv) | tensors per file and type |
| [route.tsv](route.tsv) | the Q4_0 and Q5_0 tensors, per file and name |
| [skip_readers.txt](skip_readers.txt) | every line of the build's ggml that names `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP` |
| [run/checks.sh](run/checks.sh) | the build, the CPU run and the routing dump |
| [run/gguf_types.py](run/gguf_types.py) | lists a GGUF file's tensors from its header (standard library only) |
| [run/rig_c755.sh](run/rig_c755.sh) | R150 on a P106 rig: bitid, perf, served |

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

Provenance and licence: the tables are this repository's own output. The tensor list comes from the headers of UkisAI's
Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and of the MTP draft named in the preset: names, types and shapes only.
