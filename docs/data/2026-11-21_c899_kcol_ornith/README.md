# The read-once kernel only where a P106 ran it faster: `GGML_CUDA_MMVQ_PASCAL_MT=2` (C899, Q1625; 2026-11-21)

What this is: a new value of an existing switch in the v5 engine, built and checked on a build machine's CPU. R197
([2026-11-19_r197_ornith_passes](../2026-11-19_r197_ornith_passes/README.md)) timed Ornith-1.5-122B Pascal N2P GGUF
(`valid1`) on nine P106-090 cards with `GGML_CUDA_MMVQ_PASCAL_MT=1` and `GGML_CUDA_MMVQ_PASCAL_KCOL_SKIP=q5_0,q8_0`,
which send every dense Q5_0 and Q8_0 product of 2 to 4 tokens to `mul_mat_vec_q_pascal_mt`, the kernel that decodes
each weight once for all tokens. That kernel was faster on some of those products and slower on others. With
`GGML_CUDA_MMVQ_PASCAL_MT=2` it takes only the products where R197 timed it faster; every other product runs the kernel
it runs with the switch unset. Registry row Q1625 in [quants/experiments.md](../../../quants/experiments.md); the
switch's row in [docs/switches.md](../../switches.md).

## Result in plain words

- **The rule picks R197's faster kernel for every one of Ornith's 13 dense Q5_0 and Q8_0 products at 2, 3 and 4
  tokens.** Priced on R197's own timings, a verify pass gets est. 3.63, 4.20 and 0.78 ms cheaper at 2, 3 and 4 tokens
  than with the switch unset. `MT=1` with the skip was 3.22 and 4.20 ms cheaper at 2 and 3 tokens and 3.33 ms dearer at
  4 (the 13 products' sums, [rule.md](rule.md)). A P106 has not run the new value yet; that is the rig check below.
- **Nothing changes with the switch unset, `0` or `1`.** A CPU test checks that both places that read the switch
  decide every type, row width, row count and token count as before. The GPU code is unchanged: all 8,836 kernels of the rig build's `llama-server`, and the 176 `kq_fp32` ones, are the base's byte for byte.
- **The build passes the base's checks:** no new warning, and ctest gives the base's result test by test (the same four known failures) plus the new test, which passes 112,991,050 checks with none failed.

## The rule

Which kernel each product runs, from R197's per-product timings at depth 0 (R197's `compare.tsv.gz`, rows `label`;
the dense products take the same time at any depth). A product goes to the read-once kernel at a width where both of
R197's loads timed it faster there than on its kernel with the switch unset, each by more than both arms' spread
between their two loads (R197's own class rule):

| Weights | Values in | Rows | 2 tokens | 3 tokens | 4 tokens | Ornith's products (values in × rows) | Measured against |
|---|---|---|---|---|---|---|---|
| Q5_0 | more than 4,096 | any | read-once | read-once | read-once | `ssm_out`, `attn_output` (8,192 × 3,072) | the k-column kernel |
| Q5_0 | 2,049 to 4,096 | up to 4,096 | read-once | read-once | as unset | `ffn_gate_shexp`, `ffn_up_shexp` (3,072 × 1,024), `ssm_alpha`, `ssm_beta` (3,072 × 64) | `mul_mat_vec_q_pascal_mc` |
| Q5_0 | 2,049 to 4,096 | more than 4,096 | as unset | read-once | as unset | `attn_gate` (× 8,192), `attn_qkv` (× 12,288), `attn_q` (× 16,384), `output` (× 248,320) | `mul_mat_vec_q_pascal_mc` |
| Q5_0 | up to 2,048 | any | as unset | read-once | as unset | `ffn_down_shexp` (1,024 × 3,072) | `mul_mat_vec_q_pascal_mc` |
| Q8_0 | up to 4,096 | any | read-once | read-once | read-once | `attn_k`, `attn_v` (3,072 × 512) | the k-column kernel |

- With the switch unset, Ornith's Q8_0 products and its Q5_0 products wider than 4,096 values run the k-column kernel
  at 2 to 8 tokens; its other Q5_0 products run `mul_mat_vec_q_pascal_mc` at 2 to 4 tokens, because the k-column
  kernels' dispatch table (Q824) leaves Q5_0 rows up to 4,096 wide to it. So the rule sits in both places: the
  k-column kernels leave the rule's products to the read-once kernel, and `mmvq-pascal.cu` runs it only for them.
- Where R197 timed the two kernels level (`ssm_alpha` at 2 tokens, `ssm_beta` and `ffn_down_shexp` at 4), the rule
  follows the rest of its row.
- Every product per width with both arms' loads, the class and the kernel each setting runs: [rule.md](rule.md),
  [rule.tsv](rule.tsv).
- **The bounds between and around the measured shapes are est.** The 4,096 bound follows the k-column table's; the
  2,048 bound and the 4,096-row bound sit between shapes R197 measured. R197 timed Ornith's file only, so another
  file's products in these ranges are not measured.
- **Not in the rule:** other types, products of 1 token or 5 to 8 tokens, and expert products (`MUL_MAT_ID`). They run
  as with the switch unset. The read-once kernel is not retuned here: its launch shapes for the 3,072-wide products at 2
  and 4 tokens, where it still loses, are new row Q1660.

## The code

Fork branch `pascal-p106-v5-kcol-ornith`, one commit `37c05b76a` on `pascal-p106-v5` at `3dfad876f`:

- `ggml/src/ggml-cuda/mmvq-pascal-pick.h` (new): the rule, and the k-column kernels' dispatch table (Q824) moved from
  `mmvq-pascal-kcol.cu` unchanged, with the switch's parsing. It needs no GPU, so a CPU test reads the same code the
  engine runs.
- `mmvq-pascal-kcol.cu`: with `MT=2`, the k-column kernels decline the rule's products (in their mode 1, the default,
  and not for a product `GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT` keeps).
- `mmvq-pascal.cu`: its choice of `mul_mat_vec_q_pascal_mt` goes through the header; the value is read once.
- `tests/test-cuda-mmvq-pascal-pick.cpp` (new, in ctest): `MT` on or off for every value exactly as before, only `2`
  selecting the rule; unset and `1` decide as the code before over every type, 20 row widths, 13 row counts, 0 to 9
  tokens, plain and expert products, 7 channel counts and 2 sample counts; the moved table the old one for every type,
  width and token count; `2` changing nothing outside plain Q5_0 and Q8_0 products of 2 to 4 tokens; and, in a model of
  the dispatch order, each of Ornith's 13 products at 1 to 8 tokens running R197's faster kernel with `2`, the kernels
  R197 read from the source unset, and the read-once kernel at 2 to 4 tokens with R197's switches.

`MT` is read as `atoi` reads it, so it is on or off for every value as before; only `2` changes from "every product" to
the rule. A build before C899 reads `2` as `1`.

## Checks on a build machine's CPU

Run 2026-11-21 00:51–01:02 by [run/run.sh](run/run.sh) against the base `3dfad876f` (`pascal-p106-v5`), whose rig
build, build log, ctest results and CPU warnings come from C908's run
([2026-11-21_c908_v5_merged_build](../2026-11-21_c908_v5_merged_build/README.md)). Every container ran on the CPU, at
low priority, without network.

| Check | Result |
|---|---|
| The new test, compiled alone with `g++ -std=c++17 -Wall -Wextra -Wpedantic -fsyntax-only` | no error, no warning |
| Rig build (`scripts/build_llamacpp_docker.sh` with `TESTS=1`, CUDA 12.9.2, sm_61, `HF_UI_VERSION=b12483`) | rc 0; 38 warning lines, the base's set ([warnings.tsv](warnings.tsv)); 0 AVX instructions, GLIBC_2.38 at most |
| The rig build's `test-cuda-mmvq-pascal-pick` (on the CPU, the driver stub loaded) | 112,991,050 checks, 0 failed; the kernel per product and token count in [unit.txt](unit.txt) |
| CPU build of every target (native, shared libraries) | rc 0, 0 error lines; 26 warning lines, the base's set ([warnings.tsv](warnings.tsv)) |
| ctest without network ([ctest.tsv](ctest.tsv), [cpu_summary.tsv](cpu_summary.tsv)) | 72 of 76 pass; the four failures are the base's (`test-kv-cache` aborts, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` fail), test by test the same as the base's 75; the one new test, `test-cuda-mmvq-pascal-pick`, passes |
| `cuobjdump -sass` of `llama-server` against the base's ([sass_summary.txt](sass_summary.txt)) | 8,836 of 8,836 kernels identical, none missing, none new; the 176 `kq_fp32` kernels identical |

What [unit.txt](unit.txt) shows, in the model of the dispatch order: with the switch unset, Ornith's Q5_0 products
wider than 4,096 values and its Q8_0 products run the k-column kernel (`kcol`) at 2 to 8 tokens, its other Q5_0
products `mul_mat_vec_q_pascal_mc` (`mc`) at 2 to 4 tokens and `kcol` at 5 to 8, and every product the one-token
kernel at 1 token, as R197 read them from the source. With `MT=2` the read-once kernel (`mt`) takes exactly the
cells of the rule table above, and every other cell keeps its kernel. With R197's switches every product runs `mt` at
2 to 4 tokens.

The driver's console counted 26 CPU warning lines as new. That count compared lists sorted in different locales and
with each run's own work folder in one CMake line; sorted together, the two sets are the same 26 lines. The committed
`run/run.sh` sorts every list with `LC_ALL=C` and writes the CPU tree's folder as `<tree>`.

**The rig tarball:** `llama.cpp-20261120-37c05b76a-pascal-p106-v5-kcol-ornith.tar.gz`, 841,777,567 bytes, md5
`ade8ec94ae8db18ca3a2ff28aaf0e6e1`; `bin/llama-server` md5 `4fa3ce45b17da8b621bee0b77cfb9dbc`. Beside the base's
binaries it holds `test-cuda-mmvq-pascal-pick`.

## The rig check (not run yet)

R197's run again on the C899 build, with the rule arm in place of R197's switches: [run/c899_rig.sh](run/c899_rig.sh),
13 loads, est. 3 h 25 min (R197 took 3 h 21 min). R197's [run/tables.sh](../2026-11-19_r197_ornith_passes/run/tables.sh)
and [run/compare.py](../2026-11-19_r197_ornith_passes/run/compare.py) read it unchanged (their `ro` column is the
rule arm), and [run/rule.py](run/rule.py) prices each product again. What it should show, fixed before the run:

- **The release file's identity:** with the switch unset the C899 build writes R197's replies: md5 of the 256 token
  ids `b8279751` (one token and widths 2, 3 at depth 0), `eea8bef3` (one token, 8,192), `dacfc86f` (widths 2, 3 at
  8,192), `69f664b1` and `b5bc3400` (width 4 at 0 and 8,192).
- **The verify pass:** the rule arm against the switch unset, classed by R197's rule (a gain when both paired
  differences are below zero and each is larger than both arms' spreads). Est. −3.63, −4.20 and −0.78 ms at 2, 3 and
  4 tokens at depth 0; R197's spreads were at most 0.15 ms, so each is larger than the error.
- **Per product:** the rule arm's 13 dense products as fast as R197's faster arm for each, and the products the rule
  leaves alone level with the switch unset.
- The rule arm's replies differ from the unset arm's (the k-column kernel's sums are not the read-once kernel's); each
  load of an arm should repeat its own reply, and each reply should read coherently.

## Commands

```bash
# the build and the checks, on a build machine (Docker, CPU only); paths in the environment
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> BASE_RIG=<3dfad876f's rig build folder> \
  BASE_RIG_LOG=<its build log> BASE_CTEST=<its ctest per test, tsv> BASE_CPU_WARN=<its CPU build's warnings> \
  REPO=<folder with build_llamacpp_docker.sh cpu_tests.sh sass_compare.py sass_classify.py run.sh> \
  LOCK=<flock file> bash docs/data/2026-11-21_c899_kcol_ornith/run/run.sh

# the rule and its price from R197's timings, on any machine with Python 3
python3 -I docs/data/2026-11-21_c899_kcol_ornith/run/rule.py \
  docs/data/2026-11-19_r197_ornith_passes/compare.tsv.gz docs/data/2026-11-21_c899_kcol_ornith

# the rig check, from the repository root on a machine that runs only ssh and tar
RIG=<rig alias> WORK=<run folder on the rig> BIN=<the C899 build on the rig>/bin/llama-server \
  MODEL=<models>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  TEXT_LOCAL=<the release check's proxy.jsonl.bodies.gz> DRY=1 STEP=start \
  bash docs/data/2026-11-21_c899_kcol_ornith/run/c899_rig.sh
# then the same without DRY=1, then STEP=wait, then STEP=collect OUT=<folder>; then
S=<this repository> OUT=<folder> RES=<tables> bash docs/data/2026-11-19_r197_ornith_passes/run/tables.sh
python3 -I docs/data/2026-11-21_c899_kcol_ornith/run/rule.py <tables>/compare.tsv.gz <tables>
```

## Files

| File | What it holds |
|---|---|
| [rule.md](rule.md), [rule.tsv](rule.tsv) | `rule.py` on R197's timings: every dense Q5_0 and Q8_0 product per width, both arms' loads, the paired differences and spreads, the class, the kernel unset and with `MT=2`, and the sums |
| [unit.txt](unit.txt) | the rig build's `test-cuda-mmvq-pascal-pick`: the kernel each of Ornith's 13 products runs at 1 to 8 tokens, unset, with `MT=2` and with R197's switches, and the check count |
| [ctest.tsv](ctest.tsv), [cpu_summary.tsv](cpu_summary.tsv) | ctest per test, the base against C899; `cpu_tests.sh`'s summary |
| [warnings.tsv](warnings.tsv) | warning lines of both builds, the base against C899 |
| [sass_summary.txt](sass_summary.txt) | the machine-code comparison of `llama-server` |
| [run/run.sh](run/run.sh) | the build and the checks (steps `src syntax rig unit cpu sass`) |
| [run/rule.py](run/rule.py) | the rule from R197's timings, and its price |
| [run/c899_rig.sh](run/c899_rig.sh) | the rig check (R197's run with the rule arm) |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), the v5 engine. The timings the rule comes from are R197's, measured on a Manli
M-P106L9-N6G (nine P106-090) with Ornith-1.5-122B by ornith-ai, quantized for this project. The tables here hold
kernel names, counts, times and md5s.
