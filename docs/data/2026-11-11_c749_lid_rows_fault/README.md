# Q1324 (C749): the in-place indexer's check fault takes a list longer than the cache (2026-11-11)

C704 (Q1244) added `ggml_lightning_indexer_rows`: the lightning indexer scores the K rows a list names, so that the
QSA layer reads each pooled key in its cell of the cache instead of from a gathered copy. Its check suite
`indexer-rows` has a fault run, `lid-rows-fault`, that sets `GGML_CUDA_LID_ROWS_FAULT=1` (checks only): the CUDA
backend then scores other rows than the listed ones, and the run must fail every one of the 113 in-place cases. On a
P106-090 the run failed the first 104 and then stopped the process
([the v4night replay](../2026-11-11_v4night_replay/README.md#lid-rows-fault-stops-in-the-faults-own-code-path)): the
fault scored K rows 0 to n−1 and asserted that the list was no longer than the cache, and the 105th case lists 1,100
entries over 1,000 cells. The kernel path was not affected.

The fix is fork commit `be5d25629` on `pascal-p106-v4-b12246-c749`, one commit on the merged night branch
`pascal-p106-v4-b12246-night` `c408dabcb`. With the switch on, the CUDA backend now writes the list `i mod n_cells` on
the card (a small kernel of its own) and scores through the list kernels, so a longer list stays inside the cache. For
a list no longer than the cache it scores the same rows 0 to n−1 as before. The kernels and every path without the
switch are unchanged.

Result page: registry row Q1324 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**Checked on a build machine's CPU (no GPU): the fault run now reaches all 113 in-place cases and fails every one,
the GPU code of every kernel is the night build's, and ctest passes. The CUDA path of the fault runs first on a P106
(below).**

| Check | Result |
|---|---|
| The branch's GPU code against `c408dabcb`'s ([code.tsv](code.tsv)): `cuobjdump -sass` and `-ptx` of `llama-server` and `test-backend-ops` | in each of the four listings all 8,428 functions of the night build (8,847 copies) are the same; one is new, the fault's list kernel `lightning_indexer_rows_fault_list` |
| `lid-rows-fault` replayed on the CPU, the rig build's `test-backend-ops` with `GGML_CUDA_LID_ROWS_FAULT=1` | 113 of 113 pass: the CPU backend ignores the CUDA switch |
| the same with a CPU stand-in of the fault (below), switch off | 113 of 113 pass |
| the same with the CPU stand-in, `GGML_CUDA_LID_ROWS_FAULT=1` | **113 of 113 fail**, and the run ends with its summary (`0/113 tests passed`, exit status 1) |
| ctest, static CPU build of `be5d25629` | 70 of 70 pass |

- **The cases the old fault never reached** (the 105th to the 113th: the two 1,100-entry lists over 1,000 cells, at 4
  and 9 tokens, Flash-Next's four shapes with 65,536 cells, and the three two-stream cases) fail with the CPU stand-in
  like the others ([cases.tsv](cases.tsv), lines 106 to 114).
- **How they fail:** the 110 cases that also compute the gathered copy's scores (`same=1`) fail with error `inf`,
  the list op's half no longer equal to the gathered half, as on the card; the three two-stream cases (`same=0`)
  fail with normalized errors of 1.23776941, 1.30901038 and 1.74239991 against a limit of 1e-6.
- **Why a CPU stand-in:** the fault is CUDA code, so no CPU run reaches it. [run/cpu_fault.patch](run/cpu_fault.patch)
  gives the CPU's list op the same rule under the same switch (row `i mod n_cells` for entry `i`), in the list op
  only, as on the card. Replayed against the record, it shows the verdicts the card's run must give: no case's list
  agrees with `i mod n_cells` closely enough to pass. The patch is applied to a CPU-only build for this check and is
  in no branch.
- **The rig tarball:** `llama.cpp-20261111-be5d25629-pascal-p106-v4-b12246-c749.tar.gz`, 656,365,069 B, md5
  `6e818988faf1653ccea419ad478e4eea` (`TESTS=1`, sm_61, image `nvidia/cuda:12.9.2-devel-ubuntu24.04`, 0 AVX
  instructions, GLIBC_2.38).

### What the rig run must show

The CUDA path runs first on a card: with the tarball unpacked on the rig and the `v4night` set in place
([the v4night replay](../2026-11-11_v4night_replay/README.md)), the one run `lid-rows-fault` on card 0:

```bash
SET=<rig data disk>/v4night BIN=/opt/llama-builds/llama.cpp-20261111-be5d25629-pascal-p106-v4-b12246-c749/bin \
  OUT=<rig data disk>/c749-replay DEV=0 RUNS=lid-rows-fault bash scripts/replay_checks.sh
```

Right when all 113 in-place cases fail, the run ends with its summary and `results.tsv` marks it `ok`. On the night
build the first 104 cases failed with error `inf` and the process stopped at the 105th.

## Files

| File | What |
|---|---|
| [results.tsv](results.tsv) | the three CPU replays of `lid-rows-fault`: binary, switch, what passing means, cases, passed, failed, exit status, test-backend-ops' summary |
| [cases.tsv](cases.tsv) | every in-place case in the order it runs: its record id, the case, its verdict in each replay, and the CPU stand-in's error per compared node |
| [code.tsv](code.tsv) | the GPU code comparison, one line per listing: the base build's functions and copies, the branch's, how many are the same, differ, are missing or new, the new ones, the verdict |
| [run/chain.sh](run/chain.sh) | the checks on the build machine: the rig tarball, the GPU code comparison, ctest, the CPU stand-in build, the three replays and the tables |
| [run/cpu_fault.patch](run/cpu_fault.patch) | the CPU stand-in of the fault, for the check only |

## Method

On a build machine with no GPU in use, CPU only, every build and test under one lock file at a low CPU share
(`--cpu-shares=256`, `nice -n 19`, 8 compile jobs), 2026-11-11 10:23–10:37:

```bash
WORK=<work folder> PUB=<this repository> \
  BASE=<folder>/llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night.tar.gz SET=<folder>/v4night \
  LOCK=<lock file> JOBS=8 DOCKER_ARGS='--label <label>' \
  nice -n 19 bash docs/data/2026-11-11_c749_lid_rows_fault/run/chain.sh
```

- **build:** `TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c749`, the rig tarball with
  `test-backend-ops`. The run seeded its work folder with the night build's clone and compiler cache (`SEED_SRC`,
  `SEED_CCACHE`), so it compiled only what the commit changes.
- **code:** `cuobjdump -sass` and `-ptx` of both builds' `llama-server` and `test-backend-ops`, compared function by
  function by C103's [funcs.py](../2026-11-11_c103_union_gather/run/funcs.py). `BASE` is the night build's tarball
  (md5 `736363d6ca12e6c81ff5f5b684ef31b3`).
- **ctest:** C565's [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), steps `tree build ctest`,
  `SHARED=OFF`. It leaves out the tests that need a downloaded model, Python's torch or the network, as in every
  v4 check. The commit changes only a CUDA file, so this CPU build compiles the same code as `c408dabcb`'s.
- **emu, replay:** the commit's tree with [run/cpu_fault.patch](run/cpu_fault.patch), built for the CPU with the rig
  build's CPU flags (SSE4.2, no AVX, static). Then `test-backend-ops -b CPU -o LIGHTNING_INDEXER -p n_cells=`, the
  filter of `lid-rows-fault` in the set's run list, with `--replay` on the `v4night` set (md5 of its tar
  `95a1ac635a761619e1f4fe6790424b85`, the set the rig holds), three times: the rig build with the switch, the CPU
  stand-in without and with it.

## Provenance and licence

Code: the fork's branch `pascal-p106-v4-b12246-c749`
([Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama/tree/pascal-p106-v4-b12246-c749), MIT, as
llama.cpp). The cases are `test-backend-ops` cases with random inputs stored in the `v4night` record; no model file or
user data is involved. The tables are this project's own output, published under the repository's licence; the raw
logs are not published.
