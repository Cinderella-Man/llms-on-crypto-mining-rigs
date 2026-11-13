# Q1315 part (2a): the union gather's decode step on a P106 (2026-11-13)

The deep lane's three decode arms for the union kernel of the sparse Pascal attention
([2026-11-11_c103_union_gather](../2026-11-11_c103_union_gather/README.md), switch
`GGML_CUDA_FATTN_SPARSE_PASCAL_UNION`), timed on one 9 × P106-090 rig on Miner Llama `v4-rc2` (`pascal-p106-v4`
`eb82916d3`; the rig build's `llama-server` reports commit `eb312b7bc`, branch `pascal-p106-v4-b12360-c828`, whose
tree is `v4-rc2`'s). The kernel was proved correct on a P106 first
([part 1](../2026-11-11_c103_union_gather/README.md#the-native-check-on-a-p106-r88-part-1)). The three prompt arms
(`pr128`, `pu128`, `pu512`) are part (2b), a separate run. Result page: registry row Q1315 in
[quants/experiments.md](../../../quants/experiments.md).

**Result: the union kernel makes the decode step slower at every depth with context.** With `_UNION=1` the
repeated step is 3.26 ms slower than the preset's at 4,096 tokens and 9.52 ms slower at 8,192, and 3.31 and 16.98 ms
slower than with `_MIN_RATIO=1.0`. A coding agent's turns run at those depths. The rule the folder of the union
kernel set (`u150`'s step at most `r100`'s at every depth and below it at 4,096) does not hold: `u150` is above `r100`
at four of the six depths and within the spread at the other two.

## The arms

Flash-Next's preset ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)) with the MTP
draft, `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5` added, 256 tokens a decode request at temperature 0, each request twice
a load; one untimed warm-up load, then two loads an arm in the order `ub512`, `r100`, `u150`, `ub512`, `r100`, `u150`.

| Arm | Switch (`GGML_CUDA_FATTN_SPARSE_PASCAL…`) | What it is |
|---|---|---|
| `ub512` | `=1` | the preset (control) |
| `r100` | `=1 _MIN_RATIO=1.0` | per-token lists; a 4-token verify on the sparse kernel from 8,204 cells |
| `u150` | `=1 _UNION=1` | union lists; a 4-token verify on the union kernel from 3,077 cells |

## The repeated step (pass 2), ms

A step is one pass of the main model (its own token plus the draft tokens it accepts); pass 2 repeats pass 1's
request on the same cache. Each load by value; the class is [deep_table.py](../../../bench/deep_lane/deep_table.py)'s
on pass 2: gain or loss when the means differ by more than the larger of the two arms' spreads between their loads,
else negligible.

| Depth (tokens) | `ub512` | `r100` | `u150` | `u150` against `ub512` | `u150` against `r100` | `r100` against `ub512` |
|---:|---|---|---|---|---|---|
| 0 | 121.22, 121.19 | 121.54, 121.17 | 121.34, 121.29 | +0.11 ms (+0.1 %), spread 0.05: loss | −0.04 ms, spread 0.37: negligible | +0.15 ms, spread 0.37: negligible |
| 4,096 | 135.21, 135.18 | 135.29, 135.01 | 139.50, 137.42 | **+3.26 ms (+2.4 %)**, spread 2.08: loss | **+3.31 ms (+2.4 %)**, spread 2.08: loss | −0.04 ms, spread 0.28: negligible |
| 8,192 | 149.04, 149.10 | 141.79, 141.43 | 157.86, 159.30 | **+9.52 ms (+6.4 %)**, spread 1.44: loss | **+16.98 ms (+12.0 %)**, spread 1.44: loss | −7.46 ms (−5.0 %), spread 0.36: gain |
| 16,384 | 148.38, 148.37 | 148.75, 148.37 | 151.12, 151.02 | +2.69 ms (+1.8 %), spread 0.11: loss | +2.51 ms (+1.7 %), spread 0.38: loss | +0.19 ms, spread 0.38: negligible |
| 32,768 | 164.76, 164.62 | 166.99, 164.49 | 166.49, 166.06 | +1.58 ms (+1.0 %), spread 0.43: loss | +0.53 ms (+0.3 %), spread 2.50: negligible | +1.05 ms, spread 2.50: negligible |
| 49,152 | 180.83, 180.56 | 181.11, 180.67 | 182.51, 181.96 | +1.54 ms (+0.9 %), spread 0.56: loss | +1.34 ms (+0.7 %), spread 0.56: loss | +0.20 ms, spread 0.44: negligible |

At depth 0 the cache holds at most 320 cells, under every arm's threshold, so no call runs a sparse kernel there; the
+0.11 ms is a loss by the rule because neither arm's two loads differ by more than 0.05 ms.

Generation over all depths (every depth's tokens over their time, pass 2): `ub512` 22.39 and 22.41 tok/s, `r100`
22.52 and 22.64 (+0.8 %, gain), `u150` 21.97 and 22.02 (−1.8 %, loss).

The whole tables (pass 1, tok/s, prompt reading, the draft's counters, the checks): [table.md](table.md).

## Checks

- Every load started on a rig checked quiet (no llama process, at most 100 MiB on any card); the RAM watchdog stopped
  no load; no request reallocated the target's graph.
- The same prompt ids at every depth on every load and pass; 256 tokens out on every timed request.
- The greedy reply at depth 0 is the same on every pass, load and arm. From 4,096 on, `u150`'s reply differs from
  `ub512`'s (the union kernel takes the verify passes there, and its results differ from the dense kernel's in the
  last bits: the native check's class `U`), and at 8,192 `r100`'s does (the per-token sparse kernel takes the verify
  there, class `S`). Each arm's two loads write the same replies, pass for pass. At 8,192 `u150`'s pass 1 and pass 2
  write different replies in both loads; its pass 2 writes `r100`'s reply. The draft's acceptance follows the reply:
  at 8,192 on pass 2, 3.66 tokens a step for `ub512`, 3.71 for `r100`, 3.76 for `u150`.

## Files

| File | What |
|---|---|
| [table.md](table.md) | `deep_table.py`'s tables, control `ub512`, classed on pass 2 |
| [classes.tsv](classes.tsv) | the classes against `ub512` per metric, depth and arm, each load's value |
| [classes_r100.tsv](classes_r100.tsv) | the same against `r100` |
| [requests.csv](requests.csv) | `deep_client.py`'s line per request: prompt and generation times, draft counters, reply md5s |
| [run/r88_2a.sh](run/r88_2a.sh) | the run: start, wait, status, collect |

## Commands

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) with the command of
[2026-11-11_c103_union_gather](../2026-11-11_c103_union_gather/README.md#commands), cut to its three decode arms, on
the `v4-rc2` build. The preset's line has carried `--spec-draft-n-ubatch 128` since v3; upstream #32078 makes that
flag necessary for the MTP draft on this build ([2026-11-13_c831_upstream](../2026-11-13_c831_upstream/README.md)).

```bash
RIG=<rig> WORK=<run folder on the rig> BIN=<v4-rc2 tree on the rig>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz DRY=1 bash docs/data/2026-11-13_r88_union_deep/run/r88_2a.sh
# then the same without DRY=1 (about 1 h 45 min: 16.5 min a load), then STEP=wait and STEP=collect OUT=<folder>
```

The tables, on any machine with Python 3:

```bash
python3 -I bench/deep_lane/deep_table.py <folder> --control ub512 --class-on pass2 --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py <folder> --control r100 --class-on pass2 --out table_r100.md --tsv classes_r100.tsv
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2`. Model: Swift-1.5 GSQ-RCO Qwen3.8-Flash-Next Q2_0
([ukisai](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF), Swift Open License 1.0 and
qwen-community-1.0), as retyped by [Q218](../2026-11-03_retype/README.md), with its MTP draft file; the prompts are
the deep lane's recorded agent text ([bench/deep_lane](../../../bench/deep_lane/README.md)). The tables and
`requests.csv` hold times, counts and md5s, no text.
