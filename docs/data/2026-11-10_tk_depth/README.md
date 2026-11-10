# T(k) at agent depth, the sparse Pascal attention off and without its threshold (C313, Q873)

Registry rows: [Q873](../../../quants/experiments.md) (this run), [Q1212](../../../quants/experiments.md) (the sparse
kernel's threshold for verify passes, which this run gates), [Q643](../../../quants/experiments.md) (one gather over
a verify's union of cells), [Q870](../../../quants/experiments.md) (attention for 2–8 query tokens). The line is
Q583's k-token llama-bench ([2026-11-03_verify_cost](../2026-11-03_verify_cost/README.md)), with depth added, on Miner
Llama v3.0's Flash-Next build and environment ([release/flash-next-v3](../../../release/flash-next-v3/README.md)).

A speculative step verifies four tokens in one pass of the main model. On the served line the sparse Pascal attention
(Q230's kernel) made a step 40 % faster at 49,152 tokens of context, but at 8,192 only 2.62 ms faster
([2026-11-10_deep_lane](../2026-11-10_deep_lane/README.md)). By its own threshold, mode 1 of the kernel sends a call
with k query tokens to the dense kernels while the cache holds fewer than 1.5 × 2,051 × k cells, so a four-token
verify under 12,306 cells never reaches it. The coding agent's turns end at 2.8k to 10k tokens, inside that band.
This run measures a pass of k = 1 to 8 tokens at three depths, with the switch off and at 2 (the kernel for every call
it can take, no threshold). It answers whether the sparse kernel wins at four query tokens on 8,192 cells at all.

## Result: at 8,192 cells the sparse kernel is faster at every k from 1 to 8, so a lower threshold is worth timing

At four query tokens and 8,192 tokens of context a pass takes **129.71 ms with the switch at 2 against 136.17 ms off**
(settled values; repetitions 3 to 5: 129.57, 129.71 and 130.18 against 136.17, 134.91 and 136.70): −6.46 ms
(−4.7 %), a gain. Mode 1 sends that call to the dense kernel. At 8,192 the switch at 2 is a gain at every k, by 6.46
to 17.09 ms. At depth 0 it costs nothing, and at 32,768 it is 42.78 to 100.27 ms faster from k = 2 on.

Settled T(k) in ms per pass (the median of repetitions 3 to 5 of one load an arm; [tables.md](tables.md) lists all five
repetitions of every test):

| k | depth 0, off | depth 0, =2 | class | 8,192, off | 8,192, =2 | =2 − off | class | 32,768, off | 32,768, =2 | =2 − off | class |
|---:|---:|---:|---|---:|---:|---:|---|---:|---:|---:|---|
| 1 | 69.51 | 68.81 | negligible | 83.01 | 74.93 | −8.08 | gain | 156.21 | 125.91 | −30.30 | negligible |
| 2 | 81.70 | 82.77 | negligible | 105.53 | 95.30 | −10.22 | gain | 191.10 | 148.31 | −42.78 | gain |
| 3 | 97.81 | 96.67 | negligible | 124.44 | 115.78 | −8.66 | gain | 243.26 | 161.74 | −81.52 | gain |
| 4 | 107.50 | 106.69 | negligible | 136.17 | 129.71 | −6.46 | gain | 250.15 | 177.23 | −72.92 | gain |
| 5 | 126.66 | 125.91 | negligible | 162.90 | 145.81 | −17.09 | gain | 285.42 | 201.46 | −83.96 | gain |
| 6 | 140.43 | 138.49 | negligible | 175.08 | 167.00 | −8.08 | gain | 333.57 | 233.30 | −100.27 | gain |
| 7 | 154.01 | 152.84 | gain (−1.18) | 189.64 | 180.73 | −8.92 | gain | 332.35 | 240.42 | −91.93 | gain |
| 8 | 177.78 | 176.33 | negligible | 211.70 | 202.63 | −9.08 | gain | 336.19 | 259.37 | −76.83 | gain |

The repetitions at 8,192, in order (the full table, with the rep-by-rep differences, is in [tables.md](tables.md)):

| k | off (=0) | =2 |
|---:|---|---|
| 1 | 150.32, 101.87, 88.50, 81.86, 83.01 | 144.29, 93.85, 80.52, 74.91, 74.93 |
| 2 | 124.65, 110.52, 106.01, 105.53, 105.13 | 114.57, 100.94, 95.30, 95.04, 95.46 |
| 3 | 143.51, 129.27, 124.05, 124.80, 124.44 | 134.23, 121.45, 116.24, 115.78, 115.54 |
| 4 | 151.77, 143.12, 136.17, 134.91, 136.70 | 146.08, 135.86, 129.57, 129.71, 130.18 |
| 5 | 180.98, 170.99, 162.08, 163.58, 162.90 | 162.73, 151.15, 146.48, 145.81, 145.19 |
| 6 | 193.33, 182.86, 175.08, 176.06, 174.93 | 184.80, 175.71, 167.00, 167.02, 166.75 |
| 7 | 206.46, 195.25, 188.67, 189.64, 190.29 | 197.60, 187.18, 178.73, 181.23, 180.73 |
| 8 | 228.45, 216.90, 212.84, 209.97, 211.70 | 218.50, 207.39, 202.63, 201.41, 203.07 |

How to read it:

- **The settled value and the class.** Each test makes a new context, restores the depth's saved state and runs five
  passes. The cards ramp up over the first two. At depth 0 and 8,192, repetition 1 is the slowest of the five in all
  32 tests, by 3.6 % to 92.6 % over repetition 5, and repetition 2 is the second slowest in 30 of them. So the
  comparison uses repetitions 3 to 5. Their median is the settled value. A class is **gain** when every settled
  repetition of one arm is faster than every one of the other, and **negligible** when the ranges overlap. At 8,192 the
  =2 arm is faster in each of the 24 settled repetitions paired by index (by 5.19 to 17.77 ms).
- **Two loads, one an arm per run, in sequence.** Depth 0 measures how far apart two loads are with the kernel
  barely in play: within 1.93 ms at every k. The 6.46 to 17.09 ms at 8,192 is beyond that.
- **Which kernel mode 1 picks** (from the source of `1551f5fe1`; the cache view grows in steps of 256 cells, so 8,192
  tokens and k query tokens are 8,448 cells): at 8,448 cells mode 1 takes k = 1 and 2 to the sparse kernel and k ≥ 3
  to the dense kernels. At 33,024 cells it takes every k ≤ 8. So at 8,192 the =2 column for k ≥ 3 is what a lower
  threshold would give, and the off column is what mode 1 runs today. With `_MIN_RATIO` at 1.0 a four-token call
  needs 8,204 cells, at 0.5 4,102 (est. from the same rule).
- **Random tokens, last logits only.** llama-bench's passes use random token ids, which spread over more experts than
  an answer's tokens, and ask for the last token's logits only (Q583). The numbers are T(k) of the main model, not a
  served step. R39a's served step at 32,768 (241.65 against 165.86 ms) fits this run's k = 4 difference there
  (−72.92 ms).

### 32,768: measured with the host short of memory

llama-bench processes a depth once and keeps the state on the host to restore it before every later repetition. At
32,768 tokens that state is est. 1.2 GB (the fall in available memory plus what went to swap when it was saved). After
the model loaded, the 4 GB host had 807, 782, 878 and 881 MiB available in the four loads. In the first run, where the
state of 8,192 was still held, the watchdog stopped llama-bench in both arms as it saved the state of 32,768 (658 and
637 MiB written to swap within 10 s). The 0 and 8,192 tests had finished before that, with swap use unchanged (21 and
27 MiB). In the second run, 32,768 alone, both arms finished with 121 and 57 MiB available at the lowest and 547 and 464
MiB in swap ([host_memory.tsv](host_memory.tsv)). Every repetition there restores the state through swap. That is
why the 32,768 tests spread far more: repetition 1 is the slowest in only 7 of 16 tests, and k = 1 off ran 561.53,
282.12, 251.10, 155.67 and 156.21 ms. So 32,768 gives the direction (=2 faster at every k, in each of the 24 settled
pairs) and a size with that caveat. A served line holds no such copy.

### What the depth adds, and the slope

Settled T(k) at the depth minus settled T(k) at depth 0, in ms:

| k | off, 8,192 | =2, 8,192 | off, 32,768 | =2, 32,768 |
|---:|---:|---:|---:|---:|
| 1 | 13.50 | 6.11 | 86.70 | 57.10 |
| 2 | 23.83 | 12.54 | 109.40 | 65.55 |
| 3 | 26.63 | 19.11 | 145.45 | 65.08 |
| 4 | 28.67 | 23.02 | 142.66 | 70.54 |
| 5 | 36.24 | 19.90 | 158.75 | 75.55 |
| 6 | 34.65 | 28.50 | 193.15 | 94.80 |
| 7 | 35.63 | 27.89 | 178.34 | 87.58 |
| 8 | 33.93 | 26.29 | 158.42 | 83.03 |

As ms per 1,000 cells of cache (the slope of the settled values): off, 1.648 at k = 1 and 3.500 at k = 4 from 0 to
8,192, and 2.979 and 4.638 from 8,192 to 32,768. With =2: 0.746 and 2.810 from 0 to 8,192, then 2.075 and 1.933 from
8,192 to 32,768 (the k = 1 value at 32,768 is the noisiest of the run). [tables.md](tables.md) has every k.

- **With the switch at 2 the cost of depth still grows**, by 1.933 ms per 1,000 cells at k = 4 from 8,192 to 32,768.
  The kernel reads a fixed 2,051 cells per query token, so the growth is in work that covers every cell: the
  indexer's scores and the selection. That is R51's question (what still grows with depth with the switch on).
- **Per query token:** at 8,192 the depth's cost with =2 grows from 6.11 ms at one token to 23.02 ms at four (×3.8),
  and the dense kernels' from 13.50 to 28.67 (×2.1). Each query token reads its own list, which is the part Q643's
  union gather would read once. At 32,768 the =2 line's cost of depth barely grows with k from k = 2 on (65.55 at
  two, 70.54 at four), so there it is mostly per call, not per query token.
- **The dense kernels at five query tokens on 8,192 cells:** off goes from 136.17 ms at k = 4 to 162.90 at k = 5
  (+26.73), against +19.16 at depth 0 and +16.10 with =2. A kernel change at five query tokens shows up here. It is
  outside a four-token verify and is noted for Q870.

## For the next rows

- **R50 (Q1212):** yes. At 8,192 and four query tokens the sparse kernel wins (129.71 against 136.17 ms, −6.46 ms),
  so a lower `_MIN_RATIO` is worth timing. At 8,448 cells a ratio of 1.0 would take k = 4 to the sparse kernel, and
  0.5 every k ≤ 8.
- **Q643:** at 8,192 the sparse line's cost of depth is 23.02 ms at four query tokens against 6.11 at one, the share
  a union gather acts on. At 32,768 it grows by 4.99 ms from two to four tokens.
- **Q870:** where the sparse kernel runs, it is already the 2–8-query-token attention kernel and beats the dense ones
  from 8,192 cells. Q870's attention part matters below the cache size where R50 puts the threshold.

## Files

| File | What it holds |
|---|---|
| [tables.md](tables.md) | `tk_table.py`'s output: every test's five repetitions and settled value per depth and arm, the difference rep by rep and the class, the slope between depths, what each depth adds, mode 1's kernel per depth and k |
| [tk.csv](tk.csv) | per arm, depth and k: the cells, every repetition in ms, llama-bench's own mean and standard deviation, the settled value with its range, and the run it came from |
| [host_memory.tsv](host_memory.tsv) | per run and arm: the host's available memory after the model loaded and at its lowest, swap used, the most written to swap in 10 s, memory pressure, what the watchdog did |
| [run/tk_depth.sh](run/tk_depth.sh) | the run on the rig: one llama-bench an arm after a logged quiet check, under the host-memory watchdog, stopping an arm if a depth would be processed twice |
| [run/tk_table.py](run/tk_table.py) | the tables from llama-bench's JSONL (standard library only) |

## Commands

On the rig, with nothing else running, `run/` holding `tk_depth.sh`, [scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh)
and the preset's [settings.env](../../../release/flash-next-v3/settings.env):

```
ssh $RIG "BIN=<release build>/bin MODEL=<Swift Q2_0 first shard> OUT=<work dir>/out \
  setsid nohup bash <work dir>/run/tk_depth.sh > <work dir>/tk_depth.out 2>&1 < /dev/null &"
# 32,768 alone, in a fresh process per arm (the first run's state of 8,192 left no room for it):
ssh $RIG "BIN=<release build>/bin MODEL=<Swift Q2_0 first shard> OUT=<work dir>/out2 DEPTHS=32768 \
  setsid nohup bash <work dir>/run/tk_depth.sh > <work dir>/tk_depth2.out 2>&1 < /dev/null &"
```

Each arm is one llama-bench:

```
env $PRESET_ENV GGML_CUDA_FATTN_SPARSE_PASCAL=0 llama-bench -m $MODEL -ngl 99 -fa on -b 2048 -ub 512 \
  -ts 6/6/6/6/5/6/5/6/3 -lm dio --no-host 1 -p 1,2,3,4,5,6,7,8 -n 0 -d 0,8192,32768 -r 5 --progress -o jsonl -oe md
env $PRESET_ENV GGML_CUDA_FATTN_SPARSE_PASCAL=2 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 llama-bench <the same>
```

Then, on another machine: `python3 -I run/tk_table.py <dest> <work dir>/out <work dir>/out2` (a later folder's test
replaces an earlier one's).

## Provenance and licence

On 2026-11-10, on the second of my identical rigs (nine P106-090 cards, Celeron 3865U, one 4 GB SO-DIMM), headless,
15:45 to 16:26. The quiet checks before each arm (no llama process, every card at 0 MiB) are in the run's console.
Miner Llama v3.0's build `1551f5fe1` (`pascal-p106-v3-b12076-ladder5`, llama-bench build 12358). The model: Swift-1.5
GSQ-RCO Q2_0 from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
the release preset's file; its weights are under the Swift Open License 1.0 and qwen-community-1.0. The tables hold
timings only. The scripts are under this repository's licence.
