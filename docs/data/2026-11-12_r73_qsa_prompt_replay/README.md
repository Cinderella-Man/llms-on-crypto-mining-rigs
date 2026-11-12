# The sparse prompt rule's check set on a P106 (R73 part 1, Q1272, 2026-11-12)

[C705](../2026-11-11_c705_sparse_prompt_rule/README.md) (Q1246) gave prompt micro-batches a rule of their own in
Miner Llama's sparse Pascal attention (`GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1`): calls of 9 to `_PROMPT_NQ` query
tokens go to the sparse kernel once the cache holds `_PROMPT_MIN_RATIO` × `n_kv_max` cells, while decode and verify
calls keep their own rule. Its check set `qsa-prompt` (85 `FLASH_ATTN_EXT` cases with their CPU references, 11 runs)
had passed through the compute_61 PTX on an RTX 3080 ([FN63](../2026-11-11_fn63_sparse_256_512/README.md)). Here it
ran natively on one P106-090 (card 0 of a nine-card rig), with the build that the prompt ladder's sparse-prompt rung
(R123b) runs: `pascal-p106-v4` at `d097d6d88`, the `-kern` rig tarball, which holds C705's rule. This is part (1) of
registry row Q1272. Part (2), the timing on the deep lane, is a separate run.

Result page: registry row Q1272 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**The rule holds on a P106, and every recorded case passes.**

- **Every run that must pass passes all 72 cases the card takes**, in all nine of them. CUDA declines the set's other
  13 cases (head 128 for K and 96 for V) on any card, so 72 is the number FN63 saw.
- **Both fault runs that must fail do:** `qp-fault2` fails 59 cases, `qp-fault3` 49. `qp-fault1` (a ×1.01 error,
  inside the test's limit) passes 72 of 72, as it must.
- **Every case sits on the kernel the rule names, in every run** ([rule_check.tsv](rule_check.tsv)). Where the rule
  sends a case to the sparse kernel, its errors differ from `qp-stock`'s (all dense kernels). Where the rule keeps it
  on the dense kernels, the errors are `qp-stock`'s exactly. That holds for all 72 cases in each of the 10 runs that
  are not `qp-stock`, with 0 exceptions. Every case a fault run fails is one the rule sends to the kernel.
- **The 24 prompt cases follow C705's table in all 7 runs** ([prompt_cases.tsv](prompt_cases.tsv)). The table, the
  rule as computed here and the errors against `qp-stock` agree in all 168 cells, and every case is `OK`. The cases of
  129, 256 and 512 query tokens go to the kernel only in `qp-pnq512` and `qp-m2`, and pass there.
- **Decode and verify calls (1 to 8 query tokens) keep `pascal-p106-v4`'s rule.** `qp-prompt` sends the same 21 of
  them to the kernel as `qp-sparse1` (the prompt rule off). `qp-r100`'s lower verify ratio adds 2 more (23).
- **Against FN63's run through the PTX** ([fn63_side.tsv](fn63_side.tsv)): in `qp-m2`, all 59 cases the sparse kernel
  takes give the same errors on the P106 as on the RTX 3080, to every printed digit. The dense kernels' errors in
  `qp-stock` differ between the two cards in the last digits in 70 of 72 cases, all within the limit.
- **`replay_checks.sh` marks the nine runs that must pass `WRONG`, and no case is at fault.** The `-kern` build's
  `test-backend-ops` also selects 50 cases that the set does not hold (below). The replay counts "the record does not
  cover this filter" as one failure of the run. In each of those nine runs the only FAIL lines are that line and the
  two lines a failing run ends with: 3 FAIL lines and exit 1, against 72 of 72 cases passing. The two fault runs are
  `ok`.

So the sparse-prompt rung of the prompt ladder (R123b) may run on this build.

### The replay

From [results.tsv](results.tsv) and [replay/console.txt](replay/console.txt). Card 0, 14:43:00 to 14:47:24. Card 0 was
at 0 MiB before and after, and the kernel log had no Xid line.

| Run | Switches (`GGML_CUDA_FATTN_SPARSE_PASCAL` …) | Must | Cases passing | Unrecorded | FAIL lines | Exit | `replay_checks.sh` |
|---|---|---|---:|---:|---:|---:|---|
| `qp-stock` | `=0` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-sparse1` | `=1` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-prompt` | `=1 _PROMPT=1` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-r100` | `=1 _MIN_RATIO=1.0 _PROMPT=1` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-pratio` | `=1 _PROMPT=1 _PROMPT_MIN_RATIO=2.5` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-pnq512` | `=1 _PROMPT=1 _PROMPT_NQ=512` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-m2` | `=2 _PROMPT=1 _PROMPT_NQ=512` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-m2-off` | `=2` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-fault1` | `=2 _PROMPT=1 _PROMPT_NQ=512 _FAULT=1` | pass | 72 / 72 | 50 | 3 | 1 | WRONG (unrecorded only) |
| `qp-fault2` | `=2 _PROMPT=1 _PROMPT_NQ=512 _FAULT=2` | fail | 13 / 72 | 50 | 62 | 1 | ok |
| `qp-fault3` | `=2 _PROMPT=1 _PROMPT_NQ=512 _FAULT=3` | fail | 23 / 72 | 50 | 52 | 1 | ok |

`CPU_CHECK=200` drew no case in `qp-stock` (about one case in 200 with inputs under 32 MB is also checked against the
rig's CPU).

### Which kernel took which case

The sparse kernel can take 59 of the 72 cases: those whose errors differ from `qp-stock`'s in FN63's `qp-m2` through
the PTX, which sends every call the kernel can take to it. The other 13 stay on the dense kernels by the kernel's own
limits (FN63 lists them). The rule (`fasp_rule_takes` in `ggml/src/ggml-cuda/fattn-sparse-pascal-impl.cuh`) then
decides per run. Counted here from [cases.tsv](cases.tsv); in every cell, the cases the rule sends to the kernel are
exactly the cases whose errors differ from `qp-stock`'s on the P106:

| Run | Decode and verify calls (1-8 query tokens; 29 of 38 the kernel can take) | Prompt calls (9-512 query tokens; 30 of 34) | The 24 prompt cases C705 added |
|---|---:|---:|---:|
| `qp-sparse1` | 21 | 0 | 0 |
| `qp-prompt` | 21 | 24 | 18 |
| `qp-r100` | 23 | 24 | 18 |
| `qp-pratio` | 21 | 17 | 12 |
| `qp-pnq512` | 21 | 30 | 24 |
| `qp-m2` | 29 | 30 | 24 |
| `qp-m2-off` | 29 | 0 | 0 |
| `qp-fault1`, `qp-fault2`, `qp-fault3` | 29 | 30 | 24 |

**The faults** ([faults.tsv](faults.tsv)). Every case each fault run fails is one the kernel takes:

| Run | Fault | Cases failing | Largest error (limit 5·10⁻⁴) |
|---|---|---:|---:|
| `qp-fault1` | ×1.01 on the output | 0 of 72 (all 59 kernel cases give other errors, within the limit) | — |
| `qp-fault2` | must fail | 59 of 72 (all 59 kernel cases) | 0.00446692 |
| `qp-fault3` | must fail | 49 of 72 (10 kernel cases pass with other errors) | 0.0498893 |

**The cases of more than 128 query tokens**: Flash-Next's QSA call, head 256, 2 KV heads × 12 query heads each, 4,096
cells, 2,051 kept cells per query token. Each value is the test's normalized mean squared error against the stored CPU
reference.

| Query tokens | K and V | `qp-stock`, P106 | `qp-stock`, PTX (FN63) | `qp-m2`, P106 | `qp-m2`, PTX (FN63) |
|---:|---|---:|---:|---:|---:|
| 129 | F16 | 4.43601229·10⁻⁵ | 4.43601053·10⁻⁵ | 4.43601012·10⁻⁵ | 4.43601012·10⁻⁵ |
| 129 | Q8_0 | 1.56407711·10⁻⁶ | 1.56408099·10⁻⁶ | 1.52835747·10⁻⁶ | 1.52835747·10⁻⁶ |
| 256 | F16 | 4.40232997·10⁻⁵ | 4.40232851·10⁻⁵ | 4.40232826·10⁻⁵ | 4.40232826·10⁻⁵ |
| 256 | Q8_0 | 1.66010530·10⁻⁶ | 1.66010664·10⁻⁶ | 1.62120132·10⁻⁶ | 1.62120132·10⁻⁶ |
| 512 | F16 | 4.42814645·10⁻⁵ | 4.42814649·10⁻⁵ | 4.42814806·10⁻⁵ | 4.42814806·10⁻⁵ |
| 512 | Q8_0 | 1.56391085·10⁻⁶ | 1.56391085·10⁻⁶ | 1.52770901·10⁻⁶ | 1.52770901·10⁻⁶ |

### The 50 cases the set does not hold

The set was recorded with C705's own `test-backend-ops` (`baac08803`). The `-kern` build also holds C103's union gather
([2026-11-11_c103_union_gather](../2026-11-11_c103_union_gather/README.md)), whose cases match the set's filter
`-o FLASH_ATTN_EXT -p 'n_kv_max=[1-9]'`. So the build selects 122 cases on a card, and the set holds records for 72 of
them. The other 50 ([unrecorded.tsv](unrecorded.tsv)) are exactly the cases of C103's set `qsa-union` (135 cases, a
superset of `qsa-prompt`'s 85) that `qsa-prompt` lacks. The 50 are decode and verify shapes of 2 to 8 query tokens on
4,096 to 49,152 cells, 128-token calls on 4,096 to 49,152 cells, and head-128 shapes of 7, 8 and 13 query tokens. The
replay skips them; C103's native check (R88) replays them from `qsa-union`. Registry Q1412 has the replay name each
case it holds no record of and give such a run its own verdict.

## Files

| File | What |
|---|---|
| [results.tsv](results.tsv) | per run: `replay_checks.sh`'s line, plus the cases replayed, the unrecorded ones and the CPU check |
| [rule_check.tsv](rule_check.tsv) | per run: cases replayed, OK and FAIL; how many the rule sends to the kernel (S) or keeps dense (d); S with other errors than `qp-stock`'s, d with the same, and the two kinds of exception (none) |
| [prompt_cases.tsv](prompt_cases.tsv) | the 24 prompt cases × 7 runs: C705's table, the rule as computed, `=` or `x` against `qp-stock`'s errors, the verdict |
| [cases.tsv](cases.tsv) | every case's shape, and per run its verdict, `=` or `x` against `qp-stock`'s errors and the rule's kernel |
| [fn63_side.tsv](fn63_side.tsv) | per case: the first error of `qp-stock` and `qp-m2` on the P106 and through the PTX (FN63), and whether they are identical |
| [faults.tsv](faults.tsv) | per fault run: cases failing, how many of them the kernel takes, the largest error |
| [unrecorded.tsv](unrecorded.tsv) | the 50 cases the build selects that the set holds no record of (id, case) |
| [replay/console.txt](replay/console.txt), [replay/results.tsv](replay/results.tsv) | `scripts/replay_checks.sh`'s output (the rig's folders written as `$W`, `$SET` and `$B`) |
| `replay/<run>.tsv` | per run and case: id, case, verdict, each node's error |
| [run/run_replay.sh](run/run_replay.sh) | the run: copies `replay_checks.sh` to the rig, records the cards, starts the replay detached, waits, records the cards again |
| [run/r73_tables.py](run/r73_tables.py) | the tables above, from the replay's output, C705's `runs.tsv` and `set.tsv`, and FN63's `replay/` |

## Commands

The set, relayed from the build machine (`CLIENT`; `$WORK` its C705 work folder) through the operator's machine to the
rig (`$W` a folder on the rig's data disk), with its md5 compared on arrival
(`9a498be37790ab86555e0050084af11d`, the tar 2,125,137,920 B):

```bash
ssh $CLIENT "cat $WORK/sets/qsa-prompt.tar" | ssh $RIG "mkdir -p $W && cd $W && tee >(md5sum > qsa-prompt.tar.md5.got) | tar xf -"
```

The replay with `-kern`'s `bin/` (`test-backend-ops` md5 `1b2e6a7b426c8062a2bdee9acb761148`), from the repository's
root:

```bash
SET=$W/qsa-prompt BIN=<builds>/llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern/bin W=$W \
  bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh start   # SET, BIN, OUT=$W/out, DEV=0, CPU_CHECK=200
SET=... BIN=... W=$W bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh wait    # returns when the replay ends
SET=... BIN=... W=$W bash docs/data/2026-11-12_r73_qsa_prompt_replay/run/run_replay.sh end     # the cards afterwards
```

The tables, on any machine's CPU, with `$W/out` copied to `out`:

```bash
python3 -I docs/data/2026-11-12_r73_qsa_prompt_replay/run/r73_tables.py out \
  docs/data/2026-11-11_c705_sparse_prompt_rule docs/data/2026-11-11_fn63_sparse_256_512/replay tables
```

`unrecorded.tsv` is the set's `manifest.tsv` against `qsa-union`'s (`tar xOf qsa-union.tar qsa-union/manifest.tsv`):
the ids only `qsa-union` holds, with their case strings.

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `pascal-p106-v4` `d097d6d88`; the set was recorded with
`pascal-p106-v4-b12246-c705` `baac08803`. The check's inputs are random tensors drawn by `test-backend-ops`; no model or
text is in the set.
