# Flash-Next's attention call timed alone on a P106: dense, per-token and union kernels (C839, Q1473, 2026-11-26)

What this measured: the time of one attention call of Qwen3.8-Flash-Next (head 256, 2 KV heads of 12 query heads, the
QSA hint of 2,051 kept cells a token, F16 and Q8_0 caches in the cache layout) on one P106-090 card, at the verify
passes of 1 and 4 query tokens over 4,096 to 131,072 cells, for each of the three kernels the engine can run it on: the
dense kernels over every cell, the per-token sparse kernel (each token reads its own list) and the union kernel (one
list for the 4 tokens, read once; `GGML_CUDA_FATTN_SPARSE_PASCAL_UNION`). Then each sparse kernel at every split
count from 1 to 32. The question was what the union kernel costs alone, so that the change that has it score only each
token's own cells (C746, registry [Q1318](../../../quants/experiments.md)) could be priced. R121 had named the
attention as most of the verify pass's growth with context: 9.18 and 9.18 ms of 10.20 and 10.29 ms from depth 0 to
4,096 tokens ([R121](../2026-11-17_r121_flash_step_map/README.md)). Registry row [Q1473](../../../quants/experiments.md).

Each value is one round; the three values in a cell are the three rounds in run order.

## Result

**The union kernel is the slowest of the three at every size measured, and it would stay behind the per-token kernel
even if it scored only its own cells: its time follows the cells each warp scores, not the cells it reads.** On a P106 the
sparse kernels are not bound by their memory reads, so reading a cell once instead of four times buys almost nothing.

**The run found a larger, simpler gain instead: the per-token kernel splits a four-token verify into the wrong number
of pieces.** The engine picks 3 splits a list at 4 tokens (24 blocks for a card with room for 20 at once, so a second
wave of 4). With 5 splits (two full waves of 20) the call takes 590.4, 589.9 and 590.2 µs instead of 968.9, 969.0 and
968.7 at 4,096 cells, and 625.7, 625.8 and 625.9 instead of 1,003.5, 1,003.4 and 1,004.0 at 8,192. That is 4.54 and
4.53 ms less over a verify pass's 12 attention calls (est.). At 5 splits the per-token kernel is also faster than the
dense kernels from est. 3,935 cells, so the four-token switch could take verifies from there instead of from 6,153
cells. New rows C1175 (try it with today's switches) and C1176 (the code fix); C746 is parked.

### The three kernels (F16, the engine's split counts)

µs a call ([table.md](table.md) has every size, Q8_0 too, and [calls.tsv](calls.tsv) every timed call):

| Cells | Tokens | dense | per-token | union | union list (est. cells) |
|---:|---:|---|---|---|---:|
| 4,096 | 1 | 374.8, 333.7, 333.7 | 215.2, 215.2, 214.9 | - | - |
| 4,096 | 4 | 609.3, 612.9, 613.5 | 968.3, 968.7, 967.2 | 1,305.7, 1,305.7, 1,299.5 | 3,841 |
| 8,192 | 1 | 646.9, 650.8, 651.1 | 216.2, 216.1, 215.8 | - | - |
| 8,192 | 4 | 1,196.5, 1,201.7, 1,203.2 | 1,003.4, 1,003.7, 1,002.1 | 1,851.4, 1,852.1, 1,849.1 | 5,603 |
| 16,384 | 4 | 2,358.3, 2,368.8, 2,367.7 | 1,038.4, 1,039.0, 1,037.7 | 2,251.7, 2,252.2, 2,248.8 | 6,784 |
| 32,768 | 4 | 4,699.8, 4,717.9, 4,712.9 | 1,059.3, 1,060.5, 1,059.0 | 2,423.9, 2,424.9, 2,421.5 | 7,460 |
| 131,072 | 4 | 18,738.0, 18,858.7, 18,781.9 | 1,132.7, 1,132.5, 1,131.7 | 2,804.7, 2,805.0, 2,801.0 | 8,008 |

- Each time is the operation as the engine runs it: the mask compaction (which builds the lists) and the kernel.
- The dense value at 4,096 cells and 1 token in round 1 (374.8) is the run's first timed call, taken while the card's
  clock was still rising (its lowest sample in that run was 1,493 MHz). Rounds 2 and 3 give 333.7 and 333.7.
- The union list's length is the expected size of the union of the 4 tokens' kept cells: test-backend-ops draws each
  token's 2,051 cells independently.

### The union kernel and C746

- **Its time grows with the union list, as each of its 12 warps scores every cell of the list.** Over the six sizes and
  three rounds it fits 340.5 µs per 1,000 cells of the list (F16, least squares, largest residual 113.6 µs;
  [union_fit.tsv](union_fit.tsv)). At 4,096 cells its warps score est. 46,092 cells in all against the per-token
  kernel's 24,594 (1.87 times), and at 5 splits each it takes 2.13 times as long (1,257.1 against 590.2 µs, the rounds'
  means). At 8,192 cells the ratio is est. 2.73 in cells and 2.85 in time (1,781.5 against 625.8 µs).
- **The reads it saves are small next to the call.** The per-token kernel at 5 splits reads each token's cells:
  16.8 MB a call (4 tokens, 2 KV heads, a 512-byte K row and a 512-byte V row a cell), in 590.2 µs at 4,096 cells and
  625.8 at 8,192 (the rounds' means): 28.4 and 26.8 GB/s, where the card streams 173 GB/s at stock
  ([mining_rig.md](../../mining_rig.md)).
- **C746 priced:** scoring only each token's own cells leaves each warp 2,051 cells, the per-token kernel's work, read
  from one list. By the fit the union kernel at 2,051 cells takes est. 663.3 µs at its engine split count, against the
  per-token kernel's 590.2 and 625.8 µs at 5 splits. **Est. no gain over the per-token kernel at 5 splits:** at most the
  reads it saves, which overlap the arithmetic, and a large code change. Parked; it reopens if a card is found where
  the sparse kernels are bound by their reads.

### The split counts

F16, the kernel alone at each `GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS`, µs a call; ★ marks the engine's count at 5 SMs:

| Kernel | Cells | Tokens | 2 | 3 | 4 | 5 | 6 | 10 | 20 |
|---|---:|---:|---|---|---|---|---|---|---|
| per-token | 4,096 | 1 | 825.2, 825.4, 821.9 | 562.6, 562.4, 562.5 | 439.5, 439.5, 439.6 | 348.9, 348.7, 348.8 | 300.0, 299.9, 300.0 | ★ 215.2, 215.2, 215.2 | 258.5, 258.5, 258.6 |
| per-token | 4,096 | 4 | 653.0, 652.7, 651.8 | ★ 968.9, 969.0, 968.7 | 736.9, 737.0, 737.1 | **590.4, 589.9, 590.2** | 767.1, 766.8, 766.9 | 736.3, 735.6, 736.5 | 769.0, 770.5, 769.6 |
| per-token | 8,192 | 4 | 711.2, 711.2, 711.4 | ★ 1,003.5, 1,003.4, 1,004.0 | 771.6, 771.5, 771.3 | **625.7, 625.8, 625.9** | 800.7, 800.6, 800.7 | 736.2, 736.4, 736.0 | 766.3, 766.9, 766.6 |
| union | 4,096 | 4 | 1,508.7, 1,508.2, 1,508.2 | 2,011.2, 2,010.7, 2,011.2 | 1,523.2, 1,522.6, 1,523.1 | **1,257.1, 1,256.9, 1,257.1** | 1,547.5, 1,547.5, 1,547.7 | ★ 1,306.0, 1,305.5, 1,305.7 | 1,403.1, 1,403.1, 1,403.2 |
| union | 8,192 | 4 | 2,199.3, 2,199.2, 2,199.8 | 2,830.0, 2,828.9, 2,830.1 | 2,200.6, 2,200.0, 2,200.9 | **1,781.6, 1,781.1, 1,781.7** | 2,171.4, 2,170.8, 2,171.7 | ★ 1,851.7, 1,851.6, 1,851.9 | 1,918.1, 1,918.0, 1,918.4 |

Splits 1, 8, 16 and 32 are in [table.md](table.md); 1-token passes at 8,192 cells match 4,096's.

- **Why:** a block of the per-token kernel is 3 warps of 158 registers a thread, so a P106's 5 SMs hold 20 blocks at once;
  a block of the union kernel is 12 warps, so they hold 5. A call takes as many waves as its blocks need, and each wave
  as long as a block's share of the list. At 4 tokens the per-token kernel has 8 columns (4 tokens × 2 KV heads):
  3 splits make 24 blocks (two waves, the second of 4 blocks, each 22 tiles of 32 cells), 5 splits make 40 (two full
  waves of 13 tiles). At 1 token 10 splits make 20 blocks: one wave, already the best. The union kernel's 2 columns at
  10 splits make 20 blocks: four waves of 5.
- **The engine's rule** (`fasp_pick_splits`: about 4 blocks a SM in all, `4 × nsm / columns` rounded up) assumes 4
  resident blocks a SM and ignores the waves.
- **What 5 splits are worth** at 4 tokens, est., this card's clocks: 378.7 µs a call at 4,096 cells and 377.8 at
  8,192, 4.54 and 4.53 ms a verify pass of 12 calls. From 6,153 cells (where Flash-Next's v5 preset takes four-token
  verifies onto the per-token kernel, `_MIN_RATIO=0.75`) up, every verify pass gains about that.
- **Below 6,153 cells** the preset runs the dense kernels. At 5 splits the per-token kernel beats them from est. 3,935
  cells: the dense call grows 0.1437 µs a cell at 4 tokens (611.9 µs at 4,096, 1,200.5 at 8,192, the rounds' means)
  and the per-token one 0.0087 (590.2, 625.8). At 6,153 cells est. 907.5 against 608.0 µs, 3.6 ms a verify pass. That
  crossover is `_MIN_RATIO` est. 0.48.
- **With today's switches** (C1175): `GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=10` gives 5 splits at 4 tokens and 20 at 1
  token (258.5, 258.5 and 258.6 µs against 215.2, 215.2 and 215.2: +43.3 µs a one-token call). `_SPLITS=5` gives 5 to
  every call (348.9, 348.7 and 348.8 µs at 1 token). The code fix (C1176) picks the count from the kernel's resident
  blocks and the SM count, for any card.

### Against R121's verify pass

R121 timed the attention inside the server on Miner Llama v4.0 at stock clocks: `FLASH_ATTN_EXT` over a four-token
verify pass's 12 calls took 0.97 ms at depth 0, 10.15 at 4,096 tokens (the dense kernels) and 12.66 at 8,192 (the
per-token kernel), in verify passes of 83.22, 93.42 and 96.80 ms (the nine cards' GPU time). One-token passes took
2.90 ms at 4,096 and 2.91 at 8,192. Here, 12 calls, ms:

| Cells | Tokens | The preset's kernel here | R121 | per-token at 5 splits | its gain against the preset's | as a share of R121's verify pass |
|---:|---:|---|---:|---|---:|---:|
| 4,096 | 1 | per-token: 2.58, 2.58, 2.58 | 2.90 | (10 splits already) | - | - |
| 4,096 | 4 | dense: 7.31, 7.35, 7.36 | 10.15 | 7.08, 7.08, 7.08 | est. 0.26 | 0.3 % |
| 8,192 | 4 | per-token: 12.04, 12.04, 12.02 | 12.66 | 7.51, 7.51, 7.51 | est. 4.53 | 4.7 % |

- **The two runs differ in the clock, the engine and the cache size.** This card ran with the second rig's soaked
  profile (the median SM clock of each run's busy samples 2,113 MHz in 73 runs and 2,126 MHz in 2, memory 4,657 MHz
  in all 75; [clocks.tsv](clocks.tsv)), R121's at stock. One-token calls here take 0.89 of R121's time (215.2 against 241.7 µs), about
  the clock ratio.
- **The server's cache at R121's depths holds the prompt's tail and the reply too**, 64 to 320 cells more than the
  depth, padded. R121's group profile also adds a small fixed cost to every launch group.
- **Depth 0 is not timed here.** test-backend-ops has no case under 4,096 cells, and the sparse kernels never take a
  call there: the cache holds at most 320 cells, under every switch point. R121's 0.97 ms a pass on the dense kernels
  stands.

## How it ran

- **The rig:** a nine-card P106-090 rig (Manli M-P106L9-N6G, the stock 4 GB host), nothing else running. Card 0 only,
  with the second rig's soaked clock profile set that morning (+1,300 MT/s memory, +256 MHz core on card 0;
  [its check](../2026-11-19_r209_oc_v5/README.md)) and the kit's watchdog running. Every run was on the same boot and
  the same card.
- **The engine:** Miner Llama `pascal-p106-v5-cand-b12674` `eee615cdb` (the build carries its own
  `test-backend-ops`). Its sparse kernel header `fattn-sparse-pascal-impl.cuh` is byte for byte the one in v4.0
  (`eb82916d3`) and in `pascal-p106-v5` `af00bb13b` (md5 `f32a91d2cf504f57f82336b062b8f641`). R88's union kernel and
  R121's per-token kernel are therefore the code timed here.
- **The cases:** test-backend-ops' perf set for Flash-Next's attention call (`make_test_cases_perf`: head 256, 12 query
  heads a KV head, 2 KV heads, permute `[0,2,1,3]`, K and V as views of a cache, the hint 2,051), 4,096 to 131,072
  cells, 1 and 4 tokens, F16 and Q8_0. The 4 tokens keep 2,051, 2,050, 2,049 and 2,048 random cells
  (`init_tensor_kq_mask_sparse`), each drawn independently of the others. perf mode runs the operation repeatedly for at least a second and reports µs a call.
- **The arms** ([run/c839.sh](run/c839.sh)), each a process of its own, in this order per round:

  | Arm | Variables | Calls |
  |---|---|---|
  | dense | `GGML_CUDA_FATTN_SPARSE_PASCAL=0` | 24 |
  | per-token | `=2 _NSM=5` (2: any cache size) | 24 |
  | union | `=2 _NSM=5 _UNION=1` | 12 (4 tokens) |
  | splits, per-token | `=2 _NSM=5 _SPLITS=S`, S 1, 2, 3, 4, 5, 6, 8, 10, 16, 20, 32 | 4 each (F16, 4,096 and 8,192, 1 and 4 tokens) |
  | splits, union | `=2 _NSM=5 _UNION=1 _SPLITS=S`, the same S | 2 each (F16, 4 tokens) |

  Three rounds, 378 timed calls, 10:11:15 to 10:38:05; every process exited 0. Each process also sampled card 0's
  clocks every second.
- **Why test-backend-ops:** the bench in [bench/fattn_sparse](../../../bench/fattn_sparse/README.md) times the kernels
  alone with QSA-like selections, but it runs its CPU checks first, which take hours on the rig's Celeron (Q1473). Its
  bench-only build was written but not built for this run. test-backend-ops is part of the engine build, so it needed
  no compiler.

## Limits

- **The selections are independent random cells.** Real verify tokens pick pools of 4 cells and may share many of
  them. That changes the union kernel's list, which the fit covers, and not the per-token kernel's work.
- **One card, overclocked.** At stock the times are est. about 1/0.89 longer. The rig's other eight cards carry their
  own offsets of the same profile.
- **The kernel alone.** Gains in a verify pass are est. from 12 calls; the server's step, and the agent check, decide
  (C1175).

## Commands

```bash
# from the repository root on a machine that runs only ssh; BIN is the engine build's bin folder on the rig
RIG=<rig alias> WORK=<folder on the rig> BIN=<build>/bin STEP=start bash docs/data/2026-11-26_c839_union_attn_rig2/run/c839.sh
RIG=<rig alias> WORK=<folder on the rig> STEP=status bash docs/data/2026-11-26_c839_union_attn_rig2/run/c839.sh  # about 27 min
RIG=<rig alias> WORK=<folder on the rig> OUT=<results> STEP=collect bash docs/data/2026-11-26_c839_union_attn_rig2/run/c839.sh
RIG=<rig alias> WORK=<folder on the rig> STEP=clean bash docs/data/2026-11-26_c839_union_attn_rig2/run/c839.sh

# the tables, on any machine with Python 3
python3 -I docs/data/2026-11-26_c839_union_attn_rig2/run/tables.py <results> docs/data/2026-11-26_c839_union_attn_rig2
```

One call by hand, the per-token kernel at 4 tokens and 8,192 cells with 5 splits:

```bash
CUDA_VISIBLE_DEVICES=0 GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 \
  GGML_CUDA_FATTN_SPARSE_PASCAL_SPLITS=5 <build>/bin/test-backend-ops perf -b CUDA0 -o FLASH_ATTN_EXT \
  -p 'nr23=\[12,1\],kv=8192,nb=4,.*type_K=f16,.*permute=\[0,2,1,3\].*n_kv_max=2051'
```

## Files

| File | What it holds |
|---|---|
| [table.md](table.md) | every arm, size and type by round; a verify pass's 12 calls; the union fit; every split count; the clocks |
| [calls.tsv](calls.tsv) | every timed call: arm, split count (0: the engine's), type, cells, tokens, round, µs a call, perf mode's runs |
| [union_fit.tsv](union_fit.tsv) | the union kernel's time against its expected list length, F16 and Q8_0 |
| [clocks.tsv](clocks.tsv) | card 0's SM and memory clock in each run's busy (P0) samples |
| [run/c839.sh](run/c839.sh) | the run: start, status, stop, collect, clean |
| [run/tables.py](run/tables.py) | the tables from the runs' SQL output |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `eee615cdb`, and its `test-backend-ops`. No model and no text: the inputs are
test-backend-ops' random tensors. Measured on a Manli M-P106L9-N6G (nine P106-090), one card.
