# Q1316: the union kernel of the sparse Pascal attention through the compute_61 PTX on a newer card (2026-11-11)

The union kernel ([Q643](../2026-11-11_c103_union_gather/README.md), `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION`, Miner Llama
`pascal-p106-v4-b12246-c103` `4a3423134`) reads the union of several query tokens' kept cells once per block. It had
been checked only on a CPU. This folder holds its first run on a GPU: the rig build itself, whose GPU code is sm_61
machine code and compute_61 PTX, on one RTX 3080 of the build machine, where the driver can run nothing but the PTX.
The P106 gets the same PTX compiled ahead of time, so a wrong result here would be wrong there too. It is a correctness
check only: no timing on this card means anything for a P106.

**Result: every check passes.** The check set `qsa-union` replays as designed in all 9 of its PTX runs: every run that
must pass passes all 122 cases the card takes, and the three faults that must fail cases fail them. The union kernel
ran: the multi-token cases it takes give other errors than the per-token kernel's, and the fault that breaks only the
union kernel fails exactly those cases. `fasp_check --gpu --nsm 5` passes 58 of 58 on the card. The native check and
the timing on a P106 are the next step (registry Q1315).

## What ran

| | |
|---|---|
| Build | the rig tarball `llama.cpp-20261111-4a3423134-pascal-p106-v4-b12246-c103.tar.gz` (649,847,005 B, md5 `c83704625726bdd9246796acebca0fc0`), its `test-backend-ops` |
| GPU code in it ([archs.txt](archs.txt), `cuobjdump`) | `test-backend-ops`: 187 sm_61 machine-code modules and 187 compute_61 PTX modules, nothing else; `fasp_check`: 2 sm_61 modules, 1 PTX module |
| Card | one RTX 3080 (compute 8.6, driver 595.99.02) in a container that sees only that card; the CUDA preflight passed first |
| Check set | `qsa-union` (135 cases, the tar 4,463,237,120 B, md5 `4f7fa912d83889c50cb84a353abecea1`, checked before unpacking): 122 run on the card; CUDA declines the other 13 on any card (head 128 for K and 96 for V, 5,003 cells) |
| Runs | the set's [runs-ptx.tsv](../2026-11-11_c103_union_gather/runs-ptx.tsv): `qu-stock` and the eight mode-2 runs (mode 1 leaves a newer card on its own kernels, mode 2 forces the sparse kernels at every size) |
| Settings in every run | the fork's PTX mode: `CUDA_FORCE_PTX_JIT=0 GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` (the fork's Pascal kernel paths with a P106's 5 SMs, so the sparse kernel picks a P106's split counts) |
| Then | `CUDA_FORCE_PTX_JIT=1 fasp_check --gpu --nsm 5` (md5 `eb92db897d645252db1920586cfad9c2`) |
| Time | the replay 3 min 25 s (16:20:52–16:24:17), `fasp_check` 2 min 57 s, the driver's JIT cache included |

## The replay

From [replay/results.tsv](replay/results.tsv) and [replay/console.txt](replay/console.txt):

| Run | Switches beyond `=2 _PROMPT=1 _PROMPT_NQ=512` | Must | Cases passing | FAIL lines | Exit | Verdict |
|---|---|---|---:|---:|---:|---|
| `qu-stock` | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` (the dense kernels) | pass | 122 / 122 | 0 | 0 | ok |
| `qu-m2` | — (the per-token kernel) | pass | 122 / 122 | 0 | 0 | ok |
| `qu-m2-union` | `_UNION=1` | pass | 122 / 122 | 0 | 0 | ok |
| `qu-m2-union-s1` | `_UNION=1 _SPLITS=1` | pass | 122 / 122 | 0 | 0 | ok |
| `qu-m2-union-s32` | `_UNION=1 _SPLITS=32` | pass | 122 / 122 | 0 | 0 | ok |
| `qu-fault1` | `_UNION=1 _FAULT=1` | pass | 122 / 122 | 0 | 0 | ok |
| `qu-fault2` | `_UNION=1 _FAULT=2` | fail | 13 / 122 | 111 | 1 | ok |
| `qu-fault3` | `_UNION=1 _FAULT=3` | fail | 36 / 122 | 88 | 1 | ok |
| `qu-fault4` | `_UNION=1 _FAULT=4` | fail | 27 / 122 | 97 | 1 | ok |

A failing run's FAIL lines are its failing cases plus the two lines a failing run ends with. Fault 1 changes the result
by less than the test's limit, so its run passes by design.

**Which kernel took which case**, from each case's errors (every node's error as printed; [errors.tsv](errors.tsv),
made by [run/errors.py](run/errors.py) from the per-case files in [replay/](replay/)). Cells: cases with the same errors
as the reference run / cases compared, one-token calls (20) and calls of two or more query tokens (102):

| Run | as `qu-stock`, 1 token | as `qu-stock`, 2+ | as `qu-m2`, 1 token | as `qu-m2`, 2+ | as `qu-m2-union`, 1 token | as `qu-m2-union`, 2+ |
|---|---:|---:|---:|---:|---:|---:|
| `qu-m2` | 6 / 20 | 7 / 102 | — | — | 20 / 20 | 7 / 102 |
| `qu-m2-union` | 6 / 20 | 7 / 102 | 20 / 20 | 7 / 102 | — | — |
| `qu-m2-union-s1` | 6 / 20 | 7 / 102 | 8 / 20 | 7 / 102 | 8 / 20 | 37 / 102 |
| `qu-m2-union-s32` | 6 / 20 | 7 / 102 | 8 / 20 | 7 / 102 | 8 / 20 | 8 / 102 |
| `qu-fault1` | 6 / 20 | 7 / 102 | 6 / 20 | 7 / 102 | 6 / 20 | 7 / 102 |
| `qu-fault4` | 6 / 20 | 7 / 102 | 20 / 20 | 7 / 102 | 20 / 20 | 7 / 102 |

- **13 cases stay on the dense kernels in every run** (6 one-token, 7 multi-token): their errors are `qu-stock`'s in all
  nine runs, faults included: the 9 cases with heads of 512 and 576 (the MLA shapes), two decode cases with one query
  head per KV head (Q4_1 and Q8_0), a Q4_0 decode case and a 4-token head-128 case.
- **The sparse kernels take the other 109** (14 one-token, 95 multi-token). Fault 2 fails exactly the cases whose
  errors in `qu-m2` differ from `qu-stock`'s, those 109, and passes the 13 dense ones.
- **The union kernel takes the 95 multi-token cases and nothing else.** With `_UNION=1` every one-token case keeps
  `qu-m2`'s errors bit for bit (20 of 20: decode calls stay on the per-token kernel), and every one of the 95 changes.
  Fault 4, which corrupts only the union kernel (each token scored with the next token's mask row), fails exactly the
  cases whose errors `_UNION=1` changes, those 95 (2 to 512 query tokens, verify and prompt calls alike), and passes
  all 20 one-token cases with `qu-m2`'s errors.
- **The split count reaches the union kernel:** with one split 37 of the 102 multi-token cases keep the default run's
  errors, with 32 splits 8 do (the 7 dense cases and 4 tokens on 49,152 cells), and every case passes in both.
- Fault 3 fails 86 cases (14 one-token, 72 multi-token).

## `fasp_check` on the card

[fasp_gpu.txt](fasp_gpu.txt): `bench/fattn_sparse`'s cases (26 for the per-token kernel, 32 for the union kernel)
launched on the card through the PTX with the lists built on the host, each kernel launched twice (its merge counters
must be back at zero after a launch), against FP64 and against ggml's CPU kernel.

| | Per-token kernel, 26 cases | Union kernel, 32 cases |
|---|---|---|
| Pass | 26 / 26 | 32 / 32 |
| Largest error against FP64, relative to the largest output | 1.04·10⁻⁶ (one split, F16) | 2.18·10⁻⁶ (the same-cells case at 128 tokens, F16) |
| Largest NMSE against ggml | 1.23·10⁻⁵ (Flash-Next decode, F16) | 4.55·10⁻⁵ (a 4-token verify whose rows share 90 % of their pools, F16) |

In every case the card's errors are of the same size as the host model's of the same kernel (the `model` columns). The
`vs_tok` and `bits` columns compare the union model with the per-token model on the host, as in the CPU check.

## Files

| File | What |
|---|---|
| [run/ptx_check.sh](run/ptx_check.sh) | the run: the preflight, the set unpacked and its md5 checked, then one container with the card: the GPU code listed, the replay, `fasp_check` |
| [run/errors.py](run/errors.py) | the table of which run gives which case the same errors, and the failing cases |
| [steps.txt](steps.txt) | the run's steps with times (paths as `$SET`, `$BIN`, `$FASP`) |
| [archs.txt](archs.txt) | the GPU code in `test-backend-ops` and `fasp_check` (`cuobjdump --list-elf`, `--list-ptx`) |
| [replay/console.txt](replay/console.txt), [replay/results.tsv](replay/results.tsv) | `scripts/replay_checks.sh`'s verdict per run |
| `replay/<run>.tsv` | per case: id, case string, verdict, each node's error |
| [errors.tsv](errors.tsv) | the comparison table, then every failing case of the fault runs (run, query tokens, case string) |
| [fasp_gpu.txt](fasp_gpu.txt) | `fasp_check --gpu --nsm 5` |

## Commands

On a build machine with Docker and one free card (`C` the folder with the branch's tarball, its tree, `fasp_check` and
`sets/qsa-union.tar` from [Q643's commands](../2026-11-11_c103_union_gather/README.md#commands); `W` a scratch folder;
the card by CDI name):

```bash
W=<scratch> C=<c103 work folder>
mkdir -p $W/repo/scripts && cp scripts/replay_checks.sh $W/repo/scripts/ && cp docs/data/2026-11-11_fn74_c103_ptx/run/ptx_check.sh $W/
OUT=$W/out SET=$W/qsa-union SET_TAR=$C/sets/qsa-union.tar SET_MD5=4f7fa912d83889c50cb84a353abecea1 \
  BIN=$C/llama.cpp-20261111-4a3423134-pascal-p106-v4-b12246-c103/bin FASP=$C/fasp_check REPO=$W/repo \
  GPU="--device nvidia.com/gpu=<card>" MOUNTS="<every folder the paths above lie on>" \
  DOCKER_ARGS="--label <job>=1 --cpu-shares=256" PREFLIGHT=<cuda_preflight.sh> NVCACHE=$W/nvcache CPU_CHECK=200 \
  setsid -f nohup bash $W/ptx_check.sh > $W/out/driver.out 2>&1 < /dev/null
# when $W/out/done exists:
python3 -I docs/data/2026-11-11_fn74_c103_ptx/run/errors.py $W/out/replay $W/qsa-union/runs-ptx.tsv > errors.tsv
```

`ptx_check.sh` runs itself again inside the container (`INSIDE=1`); `MOUNTS` are mounted at the same paths, so every
path means the same inside and out. `CPU_CHECK=200` drew no case in this run (about one case in 200 with inputs under
32 MB is checked against the host's CPU as well).

Provenance: the fork branch's own test cases (random inputs, no model, no data set) and `bench/fattn_sparse`'s cases.
Licence: as the repository.
