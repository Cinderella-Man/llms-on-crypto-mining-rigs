# The taper's settings on v4's whole line, by prompt size (R60, Q1235, 2026-11-12)

What this measured: which setting of the micro-batch taper reads a coding agent's prompts fastest on v4's line without
losing on any prompt size, on nine P106-090 cards. The taper cuts a prompt call into smaller micro-batches where the
nine-card pipeline fills and drains ([2026-11-04_prompt_taper](../2026-11-04_prompt_taper/README.md)). `on` means
`head=256:64,tail=512:128,short=600:64`. On R85's line the taper gained over the agent check's 62 prompts but lost on
turns of 1,024 to 1,999 new tokens ([2026-11-11_r85_c217_p106](../2026-11-11_r85_c217_p106/README.md)). R90's agent
check read those turns slower than v3.0 did ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md#the-agent-check)).
Registry row [Q1235](../../../quants/experiments.md).

The build is `pascal-p106-v4` `d097d6d88`, the `-kern` rig tarball (the v4 line with the kernel merge, C796). The line
is R90's `gains` line: Flash-Next's v3 preset with Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 and its MTP draft, the
switches every rig proved a gain on 2026-11-11, and R70's five. One more load added `GGML_CUDA_MMVQ_PASCAL_RO8=1`
(C729: the 5 to 8 token products on the read-once kernels).

## Result

**The pick is `short=600:64`, the taper without its head and tail rules.** It reads the 62 prompts in 469.17 s a
pass, against 492.68 and 480.70 s with the taper off. That is −17.52 s (−3.60 %) against a spread of 11.98 s, a gain,
with no loss in any size bucket. It gives the same greedy replies as the taper off on 62 of 62 turns.

- **`on` loses at 512 or more new tokens.** It gains 14.70 s a pass on turns under 512 new tokens and loses 5.60,
  7.03 and 1.43 s in the three larger buckets, so over all 62 prompts it is negligible (−0.65 s). The head rule is
  what loses: without it, `tail+short` and `short` lose nowhere.
- **`tail=512:128,short=600:64` is also a gain, but smaller:** 473.82 s (−12.87 s, −2.64 %). Its tail rule changes
  the greedy reply of one turn.
- **The taper is not what makes v4 slow on long prompts.** With the taper off, this line reads turns of 1,024 to 1,999
  new tokens at 79.3 and 81.1 tok/s (two passes), the speed R90's agent check measured on it (80.1). R85's line, the
  night build with only the checkpoint switches set, read the same turns in 112.12 s a pass against 192.99 s here. A
  fresh 2,049-token prompt takes 23.36, 22.94, 23.32 and 22.97 s here against 12.27, 12.38, 12.41 and 12.14 there.
- **What the logs show:** every request of this line reallocates the target's graph (373 of 373 requests, 1,547
  reallocations in load `a`). R85's two loads had none in 600 requests. A reallocation makes the scheduler wait for
  every card before it moves the buffers, so the pipeline loses the overlap of the micro-batches around it. Which
  switch causes it is Q1405's question.
- **`RO8` gains nothing on prompts.** Its `on` pass took 497.08 s against `on`'s 486.04 (+11.04 s, within the spread
  of 11.98; it was its load's first pass). Its `off` pass took 483.93 s, between load `a`'s two `off` passes. Its
  replies are `on`'s on 61 of 62 turns. The one other reply is the one `tail+short` also gives on that turn.

### The replay, a pass at a time

Two loads one after the other on one rig, nothing else running: `a` (10:10 to 11:07) and `ro8` (11:08 to 11:36). Each
replays the agent check's 62 prompts (12 bug-fix tasks, 36,793 prompt tokens a pass as the server counts them, 8
greedy tokens a turn), the taper set per request.

| Load | Pass | Taper | Prompt time, s | tok/s | Against `off` (492.68, 480.70) | Class | Greedy replies as `off`'s |
|---|---:|---|---:|---:|---|---|---|
| `a` | 0 | off | 492.68 | 74.68 | — (the reference) | — | 62 of 62 |
| `a` | 1 | `on` | 486.04 | 75.70 | −0.65 s (−0.13 %) | negligible | 62 of 62 |
| `a` | 2 | `tail=512:128,short=600:64` | 473.82 | 77.65 | −12.87 s (−2.64 %) | gain | 61 of 62 |
| `a` | 3 | `short=600:64` | 469.17 | 78.42 | −17.52 s (−3.60 %) | **gain** | **62 of 62** |
| `a` | 4 | off | 480.70 | 76.54 | — | — | 62 of 62 |
| `ro8` | 0 | `on`, `RO8=1` | 497.08 | 74.02 | +10.39 s (+2.14 %) | negligible | 61 of 62 |
| `ro8` | 1 | off, `RO8=1` | 483.93 | 76.03 | −2.76 s (−0.57 %) | negligible | 62 of 62 |

- **The class:** a pass is compared with the mean of `a`'s two `off` passes (486.69 s). The spread is their
  difference, 11.98 s; a setting with one pass has no spread of its own.
- **The pass order** puts `off` first and last, so the first pass's colder page cache falls on the reference.
  `ro8`'s first pass is its `on`.
- **The reply that differs** is turn 2 of `bugfix-python-list-ops` (1,533 new tokens). `tail+short` and RO8's `on`
  write the same other reply there (`d8ef7c5d21` against `168611855a`), so these are two replies the turn takes with
  differently rounded micro-batch sums.

### By prompt size

Each turn is binned by its new tokens in R90's buckets (`run/prompt_buckets.py`), in seconds a pass
([buckets.tsv](buckets.tsv), [prompt_buckets.tsv](prompt_buckets.tsv)):

| New tokens | Turns | Tokens | off | `on` | `tail+short` | `short` | `on`, RO8 | off, RO8 | Spread |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| under 512 | 38 | 5,963 | 106.61, 100.96 | 89.08 | 88.39 | 88.36 | 94.30 | 101.72 | 5.65 |
| 512–1,023 | 10 | 7,159 | 93.92, 92.62 | 98.87 | 94.29 | 93.37 | 100.52 | 92.56 | 1.30 |
| 1,024–1,999 | 11 | 15,480 | 195.11, 190.87 | 200.02 | 194.66 | 190.76 | 203.73 | 193.01 | 4.24 |
| 2,000 and more | 3 | 7,625 | 97.04, 96.26 | 98.07 | 96.48 | 96.68 | 98.54 | 96.64 | 0.78 |
| all | 62 | 36,227 | 492.68, 480.70 | 486.04 | 473.82 | 469.17 | 497.08 | 483.93 | 11.98 |

Against `off`, by bucket (the difference in seconds a pass, then the class):

| New tokens | `on` | `tail+short` | `short` | `on`, RO8 | off, RO8 |
|---|---|---|---|---|---|
| under 512 | −14.70 gain | −15.40 gain | −15.42 gain | −9.49 gain | −2.07 negligible |
| 512–1,023 | +5.60 **loss** | +1.02 negligible | +0.09 negligible | +7.25 **loss** | −0.71 negligible |
| 1,024–1,999 | +7.03 **loss** | +1.67 negligible | −2.23 negligible | +10.74 **loss** | +0.02 negligible |
| 2,000 and more | +1.43 **loss** | −0.16 negligible | +0.03 negligible | +1.89 **loss** | −0.01 negligible |

- **Prompt tok/s at 1,024–1,999 new tokens** ([prompt_buckets.tsv](prompt_buckets.tsv)): off 79.3 and 81.1, `on` 77.4,
  `tail+short` 79.5, `short` 81.1, RO8's `on` 76.0 and `off` 80.2.
- **The median turn at 1,024–1,999** ([bins.tsv](bins.tsv)): off 18.21 s (both passes), `on` 19.58, `tail+short`
  19.52, `short` 18.18.
- **The tokens:** the buckets count a turn's planned new tokens (36,227 a pass), so the same turns pair across
  passes. The server reads 566 more a pass, the previous turn's 8-token reply with each turn's new tokens.

### The prompt graph is reallocated on this line

From the server logs (`-lv 5`): each request's `print_timing` block counts the target's graph reallocations since
the load, with the graph's size unchanged and changed ([reallocs.tsv](reallocs.tsv), `run/realloc_counts.py`).
R85's two loads (the night merge `c408dabcb` with `--ctx-checkpoints-vram 4` and `--ctx-checkpoints-incall`, nothing
else set) are beside this run's:

| Load | Requests | With a reallocation | Same size | Changed size | Prompt time, s |
|---|---:|---:|---:|---:|---:|
| R85 `v` | 300 | 0 | 0 | 0 | 1,876.4 |
| R85 `i` | 300 | 0 | 0 | 0 | 1,652.5 |
| `a` | 373 | 373 | 802 | 745 | 2,931.7 |
| `ro8` | 154 | 154 | 365 | 307 | 1,318.0 |

- **One turn, both lines:** the 1,690-token turn 2 of `bugfix-javascript-phone-number`, the first taper-off pass, read
  in 13,501.91 ms with 0 reallocations on R85's `i` and in 21,624.69 ms with 6 on `a` (4 at the same size, 2
  changed).
- **The reallocations this count sees are not backend changes:** the callback skips those. The graph's node count or
  its tensors' sizes change between micro-batches while every card assignment stays the same. So the cause is
  something that builds the graph, not a switch that only picks a CUDA kernel (no `GGML_CUDA_` switch of the line is
  read where the graph is built). The line's graph-building switches are `LLAMA_QSA_KPOOL_DEV=1` (the
  indexer's k-pool on the cards; how many pools a micro-batch re-pools changes with the micro-batch),
  `LLAMA_QSA_INDEXER_ROWS=1` and `LLAMA_SAMPLING_TOP_K_ROWS=1`. The build itself can be the cause too: R85 ran the
  night merge, this run the `-kern` tarball. Q1405 finds which, by counting reallocations with one switch off at a
  time.
- **Decoding is faster on this line all the same:** the same turn's 8 tokens took 40.73 ms a token here against 62.70
  on R85's `i`.

### Fresh 2,049-token prompts

Nothing cached, four prompts a load, each with the taper off and `on` ([fresh.tsv](fresh.tsv)), in seconds:

| Load | Taper off | Taper `on` |
|---|---|---|
| `a` | 23.36, 22.94, 23.32, 22.97 | 23.27, 23.80, 23.33, 23.55 |
| `ro8` | 23.34, 23.07, 23.22, 23.06 | 23.33, 23.65, 23.51, 23.50 |
| R85 `i` (another line, [fresh.tsv](../2026-11-11_r85_c217_p106/fresh.tsv)) | 12.27, 12.38, 12.41, 12.14 | 12.07, 12.23, 12.30, 12.24 |

### Checkpoints and memory

- **Checkpoints:** 124 a pass, all on the cards (112.6 MiB), 0.3 ms each (median) of host bookkeeping; one restore a
  pass ([passes.tsv](passes.tsv), [loads.tsv](loads.tsv)).
- **Cards:** both loads used the same memory to the MiB, loaded (card 3 the fullest at 5,113 of 6,144 MiB) and after
  the load (card 3 at 5,201).
- **Host:** the lowest MemAvailable was 434 MiB (`a`) and 421 MiB (`ro8`); the watchdog's floor was 60 MiB. Swap held
  784 and 558 MiB after the loads.

## What it decides

- **R60's rule picks `short=600:64`:** it goes into R123's base and R72's line in place of `--ubatch-taper on`.
- **RO8 stays out of the prompt path:** its pass gains nothing (R116's served loads with RO8 run only on a gain).
- **The loss at 1,024 or more new tokens against v3.0 stays open,** and its likely cause is the graph reallocations
  (Q1405). R123's ladder starts from this line, so every rung carries them until the cause is found.

## Files

| File | What |
|---|---|
| [buckets.tsv](buckets.tsv) | per load, setting and size bucket: seconds of each pass, against `off`, the spread, the class |
| [decision.md](decision.md) | the classes in words and the pick by R60's rule (`run/r60_buckets.py`'s output) |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per pass and bucket: turns, tokens, prompt seconds and tok/s; the last column is each request's wall time (prompt plus 8 greedy tokens), as the replay streams no first token |
| [turns.csv](turns.csv) | every replayed turn of every pass in the agent check's `turns.csv` form, the input of `prompt_buckets.py` |
| [replies.tsv](replies.tsv) | per load and setting: turns whose greedy reply is `off`'s (and, for `ro8`, `a`'s with the same taper) |
| [reallocs.tsv](reallocs.tsv) | the target graph's reallocations per size bucket, this run's two loads and R85's two |
| [passes.tsv](passes.tsv), [bins.tsv](bins.tsv), [fresh.tsv](fresh.tsv), [loads.tsv](loads.tsv), [versus.tsv](versus.tsv), [versus_bins.tsv](versus_bins.tsv), [summary.md](summary.md) | R43's tables (`lane_tables.py`) for both loads, with `a`'s `off` as the base |
| [requests/a.tsv](requests/a.tsv), [requests/ro8.tsv](requests/ro8.tsv) | every request of each load as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/r60_loads.sh](run/r60_loads.sh) | starts one load on the rig with R90's `gains` line (`LOAD=a` or `ro8`) |
| [run/r60_buckets.py](run/r60_buckets.py) | the size buckets, classes, replies and pick |
| [run/realloc_counts.py](run/realloc_counts.py) | the reallocation counts from the server logs |

The rig scripts are R43's and C218's, unchanged:
- [r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh) (md5 `b79bb3d224ad5f7ace4bbf4e3505cb51`) and
  [lane_tables.py](../2026-11-10_prompt_lane/run/lane_tables.py);
- [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (`e82a80981c9e88e6a235cc476b993de8`);
- [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`);
- [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`);
- the v3 preset's [settings.env](../../../release/flash-next-v3/settings.env) (`0c94ef0531708c73a94350b9ad73d4ff`).

## Commands

A folder on the rig (`WORK`) holds those five files. The build's rig tarball is unpacked in `$BUILDS/$BUILD`
(`llama-server` md5 `9bf23c162646c9d1356d0beafcce0afa`), the model's first shard is in `MODEL` (shards 2 and 3 beside
it) and the MTP draft in `DRAFT`. Each load ran detached and ended with `$WORK/<load>/DONE`, and nothing else ran on
the rig meanwhile.

```bash
BUILD=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern
RIG=$RIG WORK=$WORK BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT LOAD=a bash run/r60_loads.sh
# once $WORK/r60a/DONE exists
RIG=$RIG WORK=$WORK BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT LOAD=ro8 bash run/r60_loads.sh
```

The server line of each load (`rig_taper.sh` adds everything up to `-c` and from `-np` on; the repeated
`--spec-draft-n-ubatch` keeps its last value, 512, and the server logs a warning):

```
llama-server -m $MODEL -ngl 99 -fa on -c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT \
  --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host \
  --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 \
  --ctx-checkpoints-incall --ubatch-taper on \
  -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5
# environment (the server's "switches" line lists 14 set in a, 15 in ro8):
#   the preset's GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0
#     GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode;
#   R90's GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0
#     GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1
#     LLAMA_SAMPLING_TOP_K_ROWS=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1;
#   ro8 also GGML_CUDA_MMVQ_PASCAL_RO8=1. MIN_AVAIL_MB=60.
# passes: a "off on tail=512:128,short=600:64 short=600:64 off", ro8 "on off"; then 4 fresh 2,049-token prompts
```

The tables, on any machine with the two load folders (each with its `server.log`) and R85's two server logs:

```bash
python3 -I ../2026-11-10_prompt_lane/run/lane_tables.py --base a --load a=r60a/load-A --load ro8=r60ro8/load-A --out .
python3 -I run/r60_buckets.py --load a=r60a/load-A --load ro8=r60ro8/load-A --base a --out .
python3 -I ../2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py turns.csv a:p0:off a:p1:on a:p2:tail+short \
  a:p3:short a:p4:off ro8:p0:on ro8:p1:off > prompt_buckets.tsv
python3 -I run/realloc_counts.py r85v=c217v/load-A/server.log r85i=c217i/load-A/server.log \
  a=r60a/load-A/server.log ro8=r60ro8/load-A/server.log > reallocs.tsv
```

`r60_buckets.py` was checked on R85's published requests first: it gives R85's 126.93 against 112.12 s a pass at
1,024–1,999 new tokens.

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements, and R85's server
logs for the reallocation counts.
