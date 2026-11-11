# Q1321 (R89, step 2): the night's check set replayed natively on a P106-090 with the merged build (2026-11-11)

The replay set `v4night` holds the CUDA cases of the seven check suites written on the night of 2026-11-10/11, recorded
once on a CPU when the night's 17 engine branches were merged into one branch
([the merge](../2026-11-11_v4_night_merge/README.md), Q1319): 967 cases and 52 runs. Here it ran on one P106-090 (card 0
of a nine-card rig) with the merged build's own `test-backend-ops`: fork commit `c408dabcb`
(`pascal-p106-v4-b12246-night`), a native `sm_61` build. It is the first time any of these kernels ran on a GPU, and it
stands for the replay step of each branch's own rig row.

| Suite | Branch, switch | Runs | Rig row |
|---|---|---|---|
| `qsa-prompt` | C705: the sparse decode attention's prompt rule (`GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT`) | `qp-*`, 11 | R73 |
| `qsa-union` | C103: one list for a group of query heads (`GGML_CUDA_FATTN_SPARSE_PASCAL_UNION`) | `qu-*`, 13 | R88 |
| `q5kmt` | C690: the Q5_K multi-token kernel (`GGML_CUDA_MMVQ_PASCAL_Q5K_MT`) | `q5kmt-*`, 10 | R75 |
| `dmt` | C722: the IQ4_XS and Q4_K multi-token kernel (`GGML_CUDA_MMVQ_PASCAL_DMT`) | `dmt-*`, 12 | R79 |
| `indexer-rows` | C704: the indexer reading a list of rows in place | `lid`, `lid-rows-fault` | R77 |
| `range-mask` | C719: `GGML_OP_RANGE_MASK` | `range-mask`, `range-mask-fault` | R78 |
| `top-k-rows` | C691: `TOP_K` over long rows and `TOP_K_ROWS` | `tk`, `tk-rows` | R83 |

Result page: registry row Q1321 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**Every case passes on the P106 in every run that must pass, and every fault is caught.** 51 of the 52 runs ended as
the run list expects. The other one, `lid-rows-fault`, failed every case it reached, as it must, then stopped on an
assertion in the fault's own code path (below). The 52 runs took 48 min 20 s.

| Runs | What | Result on the P106 |
|---|---|---|
| 8 `qp-*` and 9 `qu-*` without a fault | FLASH_ATTN_EXT with the `n_kv_max` hint: switch off, at 1 and at 2, with the prompt rule and its settings, with the union and its settings, with one and with 32 splits | 122 of 122 cases pass in each |
| `qp-fault1`, `qu-fault1` | fault 1, the output times 1.01, which stays under the test's limit | 122 of 122 pass in each, as the run list expects |
| `qp-fault2`, `qp-fault3` | prompt rule with faults 2 and 3: must fail cases | 109 and 95 of 122 fail |
| `qu-fault2`, `qu-fault3`, `qu-fault4` | union with faults 2, 3 and 4: must fail cases | 109, 87 and 95 of 122 fail |
| 10 `q5kmt-*` | Q5_K products at 1 to 4 tokens: the kernel at its defaults, without the prefetch, with one and two warps, R forced to 1, without the preset's k-column settings, off, and Q5_K left out | 112 of 112 pass in each |
| 12 `dmt-*` | IQ4_XS and Q4_K products at 1 to 5 tokens: the kernel at its defaults, without the prefetch, R forced to 1 and 2, one and two warps, without the preset, off, each type left out, and both types' one-token kernels off | 315 of 315 pass in each |
| `lid` | every LIGHTNING_INDEXER case of the set | 257 of 257 pass |
| `lid-rows-fault` | `GGML_CUDA_LID_ROWS_FAULT=1` over the 113 in-place cases: must fail cases | 104 of 104 reached fail; the 105th stops the process (below) |
| `range-mask`, `range-mask-fault` | every RANGE_MASK case, then `GGML_CUDA_RANGE_MASK_FAULT=1` | 22 of 22 pass; with the fault 22 of 22 fail |
| `tk`, `tk-rows` | TOP_K over 248,320-entry rows, TOP_K_ROWS | 39 of 39 and 75 of 75 pass |

- **The 30 large-activation products** (`MUL_MAT(...,big=250000.000000)`, Q5_K, IQ4_XS and Q4_K at 1 to 4 tokens),
  which the CPU backend does not take, so the CPU replay of the set could not judge them: each passes in every run that
  selects it, the 10 Q5_K ones in all 10 `q5kmt-*` runs and the 20 IQ4_XS and Q4_K ones in all 12 `dmt-*` runs
  ([big_cases.tsv](big_cases.tsv)).
- **Every case of the set has a verdict on one backend or the other.** 942 of the 967 cases ran on the P106, all
  passing ([coverage.tsv](coverage.tsv)). The CUDA backend declines the other 25 on any card: 13 FLASH_ATTN_EXT cases
  with V heads of 96 values and 12 LIGHTNING_INDEXER cases with IQ4_NL keys, a type the indexer kernels do not take
  ([unreplayed.tsv](unreplayed.tsv)). The CPU replay of the set passed 937 cases: these 25 among them, the 30 large
  products not.
- **Every replayed case was checked against its record** first (case string, tensor layout, the md5 of each blob), and
  none of the 51 runs that reached their summary selected a case the set does not hold.
- **The kernel log** held no Xid line from the rig's boot to the end of the run.

### `lid-rows-fault` stops in the fault's own code path

With `GGML_CUDA_LID_ROWS_FAULT=1` the indexer scores K's rows 0 to n−1 instead of the n listed ones, so the scores
change and the case fails against its record. In 104 cases it did so (normalized error `inf` on every one). The 105th
in-place case is `LIGHTNING_INDEXER(hsk=128,nh=4,n_cells=1000,kv=1100,nb=4,ns=1,type_K=f16,same=1)`: a list of 1,100
entries over a cache of 1,000 cells, which the suite holds on purpose (every seventh entry repeats one). There the
fault's path asserts that the list is no longer than the cache
(`ggml-cuda/lightning-indexer.cu:713: GGML_ASSERT(r->ne[0] <= k->ne[2]) failed`), and the process ends with SIGABRT
(exit status 134) before its summary line, so `scripts/replay_checks.sh` marks the run `WRONG`. The 8 in-place cases
after it (the 1,100-entry list at 9 tokens, Flash-Next's four shapes and the three two-stream cases) were not reached
with the fault on.

The kernel path without the fault computes that case correctly: the same case and the 8 after it pass in `lid`. The
assertion belongs to the stand-in that only the check uses. Making the stand-in take a list longer than the cache, so
that the fault run reaches all 113 cases, is Q1324.

## Files

| File | What |
|---|---|
| [results.tsv](results.tsv) | `scripts/replay_checks.sh`'s table, one line per run: run, switches, what passing means, cases passed, FAIL lines, exit status, memcheck errors (none of these runs uses the sanitizer), verdict |
| [console.txt](console.txt) | the script's console: the same per run with test-backend-ops' own summary (the rig's data-disk folder written as `$W`) |
| [verdicts_by_run.tsv](verdicts_by_run.tsv) | per run: what passing means, the verdict, and its cases by their own verdict (OK, FAIL) |
| [big_cases.tsv](big_cases.tsv) | the 30 large-activation products: how many runs replayed each, and with what verdict |
| [coverage.tsv](coverage.tsv) | per op: the set's cases, how many a run that must pass replayed, and how many passed there |
| [unreplayed.tsv](unreplayed.tsv) | the 25 cases of the set no run replayed on the card |
| [failing_cases.tsv](failing_cases.tsv) | every case not OK in a run that must pass: none (the header only) |
| [cases.tsv.gz](cases.tsv.gz) | every replayed case of every run: run, record id, case, verdict, each compared node's error |
| [run/run_replay.sh](run/run_replay.sh) | the run on the rig: [scripts/replay_checks.sh](../../../scripts/replay_checks.sh) copied over and started detached |
| [run/tables.sh](run/tables.sh) | builds the tables above from the replay's output folder and the set's `manifest.tsv` |

## Method

**The set and the build on the rig.** The merged build's rig tarball (md5 `736363d6ca12e6c81ff5f5b684ef31b3`) unpacked
in `/opt/llama-builds/`, and the set's tar streamed once over the LAN into the rig's data disk, its md5 taken on the way
in: `95a1ac635a761619e1f4fe6790424b85`, the tar's own (9,669,294,080 B; 2,901 files, three a case). The set's
`runs.tsv` is [set_runs.tsv](../2026-11-11_v4_night_merge/set_runs.tsv) byte for byte (md5
`2ddbe430c5a0f9a9c321dc7483b56f4f`). The set was recorded with `24ae0aa83`'s `test-backend-ops`, whose test code is the
same as `c408dabcb`'s (that commit changes only `ggml-rpc.h`).

**The run**, detached on the rig with nothing else running on it (no llama process, 0 MiB used on card 0):

```bash
RIG='ssh rig' W=<a folder on the rig holding v4night/> \
  BIN=/opt/llama-builds/llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night/bin \
  bash docs/data/2026-11-11_v4night_replay/run/run_replay.sh
```

which runs on the rig

```bash
SET=$W/v4night BIN=$BIN OUT=$W/v4night-replay DEV=0 CPU_CHECK=0 setsid nohup bash replay_checks.sh > nohup.log 2>&1 < /dev/null &
```

every run of the set's run list in its order on card 0 (`CUDA_DEVICE_ORDER=PCI_BUS_ID`). The binary is the tarball's
own `bin/test-backend-ops` (md5 `42c963bdd4b69f5a0c5adabb98fb3cb1`). It started at 08:29:05 and wrote its end at
09:17:25.

**The tables**, on the build machine's CPU from a copy of the output folder:

```bash
OUT=<copy of $W/v4night-replay> SET=<the set's folder: manifest.tsv> DEST=docs/data/2026-11-11_v4night_replay \
  nice -n 19 bash docs/data/2026-11-11_v4night_replay/run/tables.sh
```

## Provenance and licence

The tables are this project's own measurements, published under the repository's licence. The cases are llama.cpp's
`test-backend-ops` cases (MIT) and the fork's additions to them
([Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama), MIT); their inputs are random draws stored
in the record. The raw logs (each run's output) are not published.
