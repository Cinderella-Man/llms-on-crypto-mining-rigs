# Q1245, first step: the sparse Pascal attention at 256 and 512 query tokens through the compute_61 PTX (2026-11-11)

Miner Llama's sparse attention below Turing ([Q230](../2026-11-04_qsa_sparse_attn/README.md),
`GGML_CUDA_FATTN_SPARSE_PASCAL`) reads only the cells Qwen3.8-Flash-Next's QSA layers keep. Its recorded cases had
passed natively on a P106 up to 128 query tokens a call. Prompt micro-batches of 256 and 512 tokens through it
(registry Q1245) need it to handle calls of up to 1,024 columns of query token × KV head: 512 query tokens over the
model's 2 KV heads. [C705](../2026-11-11_c705_sparse_prompt_rule/README.md) added those cases to test-backend-ops and
recorded them in its check set `qsa-prompt`. This folder holds that set's run on one RTX 3080 of the build machine
with the rig build itself, whose GPU code is sm_61 machine code and compute_61 PTX only, so the card can run nothing
but the PTX that a P106 gets compiled ahead of time. It is a correctness check only: no timing on this card means
anything for a P106.

**Result: every case passes, the 256- and 512-token cases included, and the sparse kernel ran on them.** Both runs
pass all 72 cases the card takes. In `qp-m2`, which sends every call the kernel can take to it, the two cases at 256
query tokens and the two at 512 (F16 and Q8_0 K and V each, Flash-Next's QSA call on 4,096 cells) give other errors
than the dense kernels in `qp-stock`, all within the test's limit. The two cases at 129 query tokens behave the same.
The native run on a P106 is `qp-pnq512` of the rig row (registry Q1272); the timing of 256- and 512-token
micro-batches is Q1245's deep-lane run.

## What ran

| | |
|---|---|
| Build | Miner Llama `pascal-p106-v4-b12246-c705` `baac08803`, the rig tarball `llama.cpp-20261110-baac08803-pascal-p106-v4-b12246-c705.tar.gz` (647,364,343 B, md5 `6a5239a02ffa53d327b955f0ea240db9`), its `test-backend-ops` (md5 `77bf0ba5e598d58226805ff21799de6f`, the one that recorded the set) |
| GPU code in it ([archs.txt](archs.txt), `cuobjdump`) | 187 sm_61 machine-code modules and 187 compute_61 PTX modules, nothing else |
| Card | one RTX 3080 (compute 8.6, driver 595.99.02) in a container that sees only that card; the CUDA preflight passed first |
| Check set | `qsa-prompt` (85 cases; the tar 2,125,137,920 B, md5 `9a498be37790ab86555e0050084af11d`, checked before unpacking): 72 run on the card; CUDA declines the other 13 on any card (head 128 for K and 96 for V, 5,003 cells, 3 query tokens, 13 type pairs) |
| Runs | from the set's [runs-ptx.tsv](../2026-11-11_c705_sparse_prompt_rule/runs-ptx.tsv): `qp-stock` (`GGML_CUDA_FATTN_SPARSE_PASCAL=0`, the dense kernels) and `qp-m2` (`=2 _PROMPT=1 _PROMPT_NQ=512`: mode 2 takes calls at any cache size, so a newer card uses the sparse kernel too, and prompt calls of up to 512 query tokens go to it) |
| Settings in every run | the fork's PTX mode: `CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` (the fork's Pascal kernel paths with a P106's 5 SMs, so the sparse kernel picks a P106's split counts) |
| Time | the replay 43 s (16:42:46–16:43:29), the driver's JIT compile included |

## The replay

From [replay/results.tsv](replay/results.tsv) and [replay/console.txt](replay/console.txt):

| Run | Switches | Must | Cases passing | FAIL lines | Exit | Verdict |
|---|---|---|---:|---:|---:|---|
| `qp-stock` | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` | pass | 72 / 72 | 0 | 0 | ok |
| `qp-m2` | `GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1 GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT_NQ=512` | pass | 72 / 72 | 0 | 0 | ok |

`CPU_CHECK=200` drew no case (about one case in 200 with inputs under 32 MB is checked against the host's CPU too).

**Which kernel took which case**, from each case's errors (every node's error as printed, in the per-case files
[replay/qp-stock.tsv](replay/qp-stock.tsv) and [replay/qp-m2.tsv](replay/qp-m2.tsv); [cases.tsv](cases.tsv), made by
[run/cases.py](run/cases.py)). A kernel that ran gives other errors than the one it replaced.

| Query tokens a call | Cases | Pass `qp-stock` | Pass `qp-m2` | Other errors in `qp-m2` |
|---:|---:|---:|---:|---:|
| 1 | 20 | 20 | 20 | 14 |
| 2 | 5 | 5 | 5 | 4 |
| 3 | 3 | 3 | 3 | 2 |
| 4 | 4 | 4 | 4 | 3 |
| 5 | 2 | 2 | 2 | 2 |
| 7 | 2 | 2 | 2 | 2 |
| 8 | 2 | 2 | 2 | 2 |
| 9 | 4 | 4 | 4 | 4 |
| 16 | 2 | 2 | 2 | 2 |
| 32 | 2 | 2 | 2 | 2 |
| 64 | 13 | 13 | 13 | 9 |
| 67 | 1 | 1 | 1 | 1 |
| 128 | 6 | 6 | 6 | 6 |
| 129 | 2 | 2 | 2 | 2 |
| 256 | 2 | 2 | 2 | 2 |
| 512 | 2 | 2 | 2 | 2 |
| all | 72 | 72 | 72 | 59 |

The 13 cases with the same errors in both runs stay on the dense kernels, as the switch's limits say: the 9 cases
with heads of 512 and 576 (the MLA shapes), two decode cases with one query head per KV head (Q4_1 and Q8_0), a Q4_0
decode case and a 4-token head-128 case. Every case with Flash-Next's head shape and F16 or Q8_0 K and V changes.

**The cases of more than 128 query tokens** (Flash-Next's QSA call: head 256, 2 KV heads × 12 query heads each,
4,096 cells, 2,051 kept cells per query token, K and V of one type). The error is the test's normalized mean squared
error of the attention's output against the stored CPU reference; FLASH_ATTN_EXT passes below 5·10⁻⁴.

| Query tokens | K and V | Columns | `qp-stock` | `qp-m2` | Error, dense kernels | Error, sparse kernel |
|---:|---|---:|---|---|---:|---:|
| 129 | F16 | 258 | OK | OK | 4.43601053·10⁻⁵ | 4.43601012·10⁻⁵ |
| 129 | Q8_0 | 258 | OK | OK | 1.56408099·10⁻⁶ | 1.52835747·10⁻⁶ |
| 256 | F16 | 512 | OK | OK | 4.40232851·10⁻⁵ | 4.40232826·10⁻⁵ |
| 256 | Q8_0 | 512 | OK | OK | 1.66010664·10⁻⁶ | 1.62120132·10⁻⁶ |
| 512 | F16 | 1,024 | OK | OK | 4.42814649·10⁻⁵ | 4.42814806·10⁻⁵ |
| 512 | Q8_0 | 1,024 | OK | OK | 1.56391085·10⁻⁶ | 1.52770901·10⁻⁶ |

The sparse kernel's errors are of the same size as the dense kernels' at every size, up to the kernel's 1,024 columns;
the largest, 4.43601012·10⁻⁵ (F16, 129 query tokens), is 8.9 % of the limit.

## Files

| File | What |
|---|---|
| [run/ptx_check.sh](run/ptx_check.sh) | the run: the preflight, the set unpacked and its md5 checked, then one container with the card: the GPU code listed and the replay |
| [run/cases.py](run/cases.py) | the per-size table and the line per case of more than 128 query tokens |
| [steps.txt](steps.txt) | the run's steps with times (paths as `$SET`, `$SET_TAR`, `$BIN`) |
| [archs.txt](archs.txt) | the GPU code in `test-backend-ops` (`cuobjdump --list-elf`, `--list-ptx`) |
| [replay/console.txt](replay/console.txt), [replay/results.tsv](replay/results.tsv) | `scripts/replay_checks.sh`'s verdict per run |
| [replay/qp-stock.tsv](replay/qp-stock.tsv), [replay/qp-m2.tsv](replay/qp-m2.tsv) | per case: id, case string, verdict, each node's error |
| [cases.tsv](cases.tsv) | `run/cases.py`'s output: per number of query tokens, then the cases of more than 128, then the failing cases (none) |

## Commands

On a build machine with Docker and one free card (`C` the folder with the c705 branch's tarball, its unpacked tree
and `sets/qsa-prompt.tar` from [C705's commands](../2026-11-11_c705_sparse_prompt_rule/README.md#commands); `W` a
scratch folder; the card by CDI name):

```bash
W=<scratch> C=<c705 work folder>
mkdir -p $W/repo/scripts $W/out && cp scripts/replay_checks.sh $W/repo/scripts/ \
  && cp docs/data/2026-11-11_fn63_sparse_256_512/run/ptx_check.sh docs/data/2026-11-11_fn63_sparse_256_512/run/cases.py $W/
OUT=$W/out SET=$W/qsa-prompt SET_TAR=$C/sets/qsa-prompt.tar SET_MD5=9a498be37790ab86555e0050084af11d \
  BIN=$C/llama.cpp-20261110-baac08803-pascal-p106-v4-b12246-c705/bin REPO=$W/repo \
  GPU="--device nvidia.com/gpu=<card>" MOUNTS="<every folder the paths above lie on>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<cuda_preflight.sh> NVCACHE=$W/nvcache CPU_CHECK=200 \
  RUNS="qp-stock qp-m2" setsid -f nohup bash $W/ptx_check.sh > $W/out/driver.out 2>&1 < /dev/null
# when $W/out/done exists:
python3 -I $W/cases.py $W/out/replay > cases.tsv
```

`ptx_check.sh` runs itself again inside the container (`INSIDE=1`); `MOUNTS` are mounted at the same paths, so every
path means the same inside and out. Leave `RUNS` empty for every run of `runs-ptx.tsv` (the faults and `qp-m2-off`
as well).

Provenance: the fork branch's own test cases (random inputs, no model, no data set). Licence: as the repository.
