# Four more prompt settings as one ladder on an agent session's prompts (R123b, Q1401, 2026-11-12)

What this measured: how much four more prompt-side settings of v4's line speed up the reading of a coding agent's own
prompts on nine P106-090 cards, each added on top of the ones before, on the line the first half of this row kept
([2026-11-12_r123a_prompt_ladder](../2026-11-12_r123a_prompt_ladder/README.md)). They are:

- `LLAMA_DRAFT_EMBD_DEV=layer` (C762), in place of the line's `-otd token_embd=CUDA4`: the MTP draft's token table
  (644.14 MiB, Q8_0) goes to its layer's card, card 8, so each catch-up pass sends card 8 its token ids instead of
  F32 rows from card 4 through the host ([2026-11-11_c762_draft_rows](../2026-11-11_c762_draft_rows/README.md));
- `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1` (C705): prompt micro-batches of 9 to 128 tokens go through the sparse
  attention once the cache holds 2,051 cells ([2026-11-11_c705_sparse_prompt_rule](../2026-11-11_c705_sparse_prompt_rule/README.md),
  exact on a P106 in [2026-11-12_r73_qsa_prompt_replay](../2026-11-12_r73_qsa_prompt_replay/README.md));
- `CUDA_SCALE_LAUNCH_QUEUES=2x`: CUDA's launch queue at twice its size, for the host's stalls on a full queue that
  [2026-11-10_prompt_timeline](../2026-11-10_prompt_timeline/README.md) saw (R62);
- `-b 4096`: a logical batch of 4,096 tokens instead of 2,048, so a prompt of 2,049 to 4,096 new tokens is one call
  instead of two (R63).

The workload is R43's replay of the agent check's 62 prompts
([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md)), read by the size of each turn's new prompt, as in
R123a. Registry row [Q1401](../../../quants/experiments.md) (its second half).

The build is `pascal-p106-v4` `d097d6d88`, the `-kern` rig tarball. The model is Swift Qwen3.8-Flash-Next GSQ-RCO
Q2_0 with its MTP draft. Every load starts from R123a's kept line:

- the 2026-11-12 line: R90's `gains` line ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md)) with
  `--ubatch-taper short=600:64` (R60), without `LLAMA_SAMPLING_TOP_K_ROWS=1` (FN84), plus C720's and C719's k-pool
  switches;
- R122's eight host switches of the decode step ([2026-11-12_r122_host_ladder](../2026-11-12_r122_host_ladder/README.md));
- R123a's `GGML_CUDA_GRAPHS_FORCE=1`, `LLAMA_MTP_KV_ONLY=1` and `LLAMA_MTP_CATCHUP_OVERLAP=1`, without
  `LLAMA_NGRAM_PREFETCH=1`.

## Result

**The draft's table on card 8, the sparse prompt rule and `-b 4096` go into the stacked line; the doubled launch
queue does not, for its host memory.** Each of the four reads the 62 prompts within the spread of the two `base`
loads, with `base`'s greedy replies on all 62 turns. Together they are a gain: 243.95 and 233.12 s a pass against
`base`'s 247.52 and 240.27, which is −5.36 s, −2.20 %. `CUDA_SCALE_LAUNCH_QUEUES=2x` takes 222 MiB off the host's
lowest available memory (171 against 393 MiB) for −0.80 s, so it is a memory loss and goes back to its own row.

| Rung | Setting added | Seconds a pass | Against the rung before | Class | Replies as `base` | Lowest MemAvailable |
|---|---|---|---|---|---|---|
| `base` | — | 247.52, 240.27 | — | reference | 62 of 62 | 470 MiB |
| `embd` | `LLAMA_DRAFT_EMBD_DEV=layer`, no `-otd token_embd=CUDA4` | 246.53, 238.34 | −1.46 s (−0.60 %) | negligible | 62 of 62 | 388 MiB |
| `sparse` | `GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1` | 246.01, 238.31 | −0.27 s (−0.11 %) | negligible | 62 of 62 | 393 MiB |
| `queues` | `CUDA_SCALE_LAUNCH_QUEUES=2x` | 247.49, 235.23 | −0.80 s (−0.33 %) | negligible in time, a loss in host memory | 62 of 62 | 171 MiB |
| `b4096` | `-b 4096` | 243.95, 233.12 | −2.83 s (−1.17 %) | negligible (a gain at 2,000+) | 62 of 62 | 183 MiB |
| `base2` | — (the line again) | 249.17, 246.16 | +3.77 s against `base` (the spread) | reference | 62 of 62 | 470 MiB |

A load's value is the mean of its two passes. A rung is classed against the rung before it, with the spread being
the difference between the two `base` loads' means: 3.77 s over all 62 turns, and 0.18, 0.07, 2.73 and 0.79 s in the
four size buckets. Most of the spread is `base2`'s second pass at 1,024–1,999 new tokens, 91.57 s against its first
pass's 87.80.

- **The draft's token table on card 8** is faster in every bucket by little: −0.28, −0.35, −0.51 and −0.32 s.
  The table moved as intended: "token_embd.weight (644.14 MiB q8_0) in CUDA8 (LLAMA_DRAFT_EMBD_DEV)". Card 4 holds
  652 MiB less after loading (4,175 against 4,827 MiB) and card 8 730 MiB more (5,545 against 4,815). It costs the
  host 82 MiB at its lowest (388 against 470 MiB). The draft's context reserves a pinned host compute buffer of
  86.41 MiB instead of 21.60, and 278.39 MiB instead of 195.48 on card 8 ([buffers.tsv](buffers.tsv)). Which
  tensor that is, and whether it can shrink, is its own row (Q1420).
- **The sparse prompt rule** moves the whole by −0.27 s: −0.21 under 512 new tokens, +0.23 at 512–1,023 (a loss
  against that bucket's spread of 0.07), −0.30 at 1,024–1,999 and +0.01 at 2,000+. Its rule takes only prompt
  micro-batches of 9 to 128 tokens on a cache of 2,051 cells or more: here, the taper's 64-token micro-batches of
  short prompts deep in a session. The server logs no line for those calls.
- **The doubled launch queue** takes −0.80 s over all: −0.19, −0.19, −0.69 and +0.27 s by bucket, and pass 0 is
  slower than `sparse`'s (247.49 against 246.01 s) while pass 1 is faster (235.23 against 238.31). The driver
  honours the variable: after loading the host has 520 MiB available against 847, 327 MiB less, or 36 MiB a card.
  The lowest MemAvailable falls from 393 to 171 MiB, 111 MiB above the watchdog's floor of 60. No reserved buffer
  changes. A time within the spread for 222 to 327 MiB of a 4 GB host is a loss, so it stays out of the line and its
  own row (R62, Q1239) keeps it. Q693 (the queue at `0.5x` and `0.25x` for host memory) is the other direction.
- **`-b 4096` acts only on the three turns of 2,000 or more new tokens.** They are now read in one call each, and
  they read −2.86 s faster (−8.15 %, spread 0.79) at 236.4 tok/s against 217.1. In the other buckets the time moves
  by 0.05 s or less. The server logged 146 taper lines instead of 150, and "0 catch-ups ran after the next call was
  queued" instead of 4, so with one call there is nothing left for R123a's overlapped catch-up to overlap. The host
  holds no more than with the queue alone (183 against 171 MiB at the lowest). Card 8 holds 160 MiB more after loading
  (5,705 against 5,545 MiB), outside the scheduler's buffers, and 5,787 after the replay, 357 MiB under its 6,144.

Every rung up to each one, against `base`:

| Line | Under 512 | 512–1,023 | 1,024–1,999 | 2,000+ | All 62 |
|---|---|---|---|---|---|
| `base`, s a pass (mean) | 64.84 | 56.93 | 86.96 | 35.16 | 243.89 |
| + table on card 8 | −0.28 (−0.43 %) | −0.35 (−0.61 %) | −0.51 (−0.59 %) | −0.32 (−0.92 %) | −1.46 (−0.60 %) |
| + sparse prompt rule | −0.48 (−0.74 %) | −0.12 (−0.22 %) | −0.81 (−0.94 %) | −0.31 (−0.89 %) | −1.73 (−0.71 %) |
| + launch queue `2x` | −0.68 (−1.05 %) | −0.31 (−0.55 %) | −1.50 (−1.72 %) | −0.04 (−0.13 %) | −2.54 (−1.04 %) |
| + `-b 4096` (all four) | −0.63 (−0.97 %) | −0.37 (−0.64 %) | −1.46 (−1.68 %) | −2.91 (−8.27 %) | −5.36 (−2.20 %) |
| all four, tok/s | 92.9 | 126.6 | 181.1 | 236.4 | 151.9 |
| `base`, tok/s | 92.0 | 125.8 | 178.0 | 216.8 | 148.5 |

The line that goes on, without the launch queue, was not run as one load. Est. from the rungs: −1.46, −0.27 and
−2.83 s, so −4.56 s a pass (−1.87 %), with `-b 4096`'s part measured on top of the doubled queue.

**Time to the first token.** The replay asks for 8 greedy tokens and streams none, so the request's wall time
(prompt plus 8 tokens) stands in for it, next to the server's own prompt time. Medians and p90 over the 62 turns:

| Load | Pass | Wall median | Wall p90 | Prompt median | Prompt p90 |
|---|---|---|---|---|---|
| `base` | 0, 1 | 3.15, 2.96 s | 8.67, 8.62 s | 2.75, 2.66 s | 8.35, 8.24 s |
| `embd` | 0, 1 | 3.13, 2.95 | 8.70, 8.51 | 2.74, 2.64 | 8.30, 8.18 |
| `sparse` | 0, 1 | 3.17, 3.00 | 8.63, 8.52 | 2.74, 2.64 | 8.29, 8.15 |
| `queues` | 0, 1 | 3.19, 2.95 | 8.63, 8.44 | 2.75, 2.59 | 8.24, 8.08 |
| `b4096` | 0, 1 | 3.45, 2.95 | 8.60, 8.32 | 2.74, 2.58 | 8.22, 8.00 |
| `base2` | 0, 1 | 3.15, 2.98 | 8.82, 9.18 | 2.76, 2.67 | 8.47, 8.79 |

**What else held in every load:**

- **Replies:** the same 8 greedy tokens as `base` on 62 of 62 turns in both passes, and both passes agree.
- **Graph reallocations:** 0 in all 146 requests of each load (124 turns and 22 unmeasured prefix requests), each
  rung against the one before ([realloc_marks.tsv](realloc_marks.tsv)).
- **Host memory** ([memory.tsv](memory.tsv), [host.tsv](host.tsv)): the lowest MemAvailable was 470, 388, 393, 171,
  183 and 470 MiB, in ladder order. The watchdog's floor (60) never acted.
- **Cards** ([cards.tsv](cards.tsv)): the highest card held 5,163 MiB after the replay in `base` and `base2` (card 3),
  5,627 in `embd`, `sparse` and `queues` and 5,787 in `b4096` (card 8).
- **Checkpoints and taper:** 315 checkpoint lines in every load; 150 taper lines, 146 in `b4096`.
- **The settings acted:** 26, 27, 28, 28, 28 and 26 switches set; the table's placement line in every load from
  `embd` on; `CUDA_SCALE_LAUNCH_QUEUES=2x` in `queues`' and `b4096`'s environment; `n_batch = 4096` in both of
  `b4096`'s contexts ([switch_lines.txt](switch_lines.txt)).
- **The line is R123a's:** `base` reads 243.89 s a pass where R123a's kept line read 243.57 (247.11 and 240.02).

**Est. what it changes for an agent:** prompt reading was 33 % of an agent task in R90's agent check, so the est.
−1.87 % of it is est. up to 0.6 % of a task. R72's agent check on the stacked line `v4c2` measures it.

## How it ran

One load a rung, each a fresh server with two passes of R43's replay, every request carrying the line's taper
(`short=600:64`), `FRESH=0`, as R123a. Nothing else ran on the rig during a load, and each load was copied off before
the next started.

| Load | Start | Server up after | Done | Load took |
|---|---|---|---|---|
| `base` | 16:26:52 | 226 s | 16:41:06 | 14 min 14 s |
| `embd` | 16:41:39 | 226 s | 16:55:50 | 14 min 11 s |
| `sparse` | 16:56:19 | 226 s | 17:10:29 | 14 min 10 s |
| `queues` | 17:10:55 | 226 s | 17:25:05 | 14 min 10 s |
| `b4096` | 17:25:37 | 226 s | 17:39:44 | 14 min 7 s |
| `base2` | 17:40:13 | 226 s | 17:54:34 | 14 min 21 s |

## Files

| File | What |
|---|---|
| [decision.md](decision.md) | each rung's class in words, against the rung before and against `base`, and where it goes by the rule |
| [ladder.tsv](ladder.tsv) | per load and size bucket: turns, tokens, seconds of each pass, the mean, tok/s, the rung before's mean (and `base`'s), the difference, the spread and the class |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per load, pass and bucket: turns, tokens, prompt seconds and tok/s, the median wall time (R90's `prompt_buckets.py`) |
| [first.tsv](first.tsv) | per load and pass: median and p90 of the wall time and of the server's prompt time |
| [replies.tsv](replies.tsv) | per load: turns whose greedy reply is `base`'s and `base2`'s in both passes, and whose two passes agree |
| [reallocs.tsv](reallocs.tsv) | the target graph's reallocations per load and bucket of prompt tokens |
| [realloc_marks.tsv](realloc_marks.tsv) | the same per load against the rung before (`scripts/realloc_counts.py --marks ladder`) |
| [memory.tsv](memory.tsv) | per load: the lowest MemAvailable and when |
| [host.tsv](host.tsv) | per load: the host's used, shared, buff/cache and available memory after loading and after the replay |
| [cards.tsv](cards.tsv) | per load: each card's memory after loading and after the replay |
| [buffers.tsv](buffers.tsv) | per load: every model, KV and compute buffer the server reserved, by device |
| [loads.tsv](loads.tsv) | per load: start, server start, replay end, done, switches set, checkpoint and taper lines, the replay's own line |
| [switch_lines.txt](switch_lines.txt) | per load: the server's switch list, the `CUDA_` variables, the line's `-otd` rule, the contexts' batch sizes, where the draft's token table went, and the lines the kept switches print |
| [turns.csv](turns.csv) | every replayed turn of every pass of every load in the agent check's `turns.csv` form, the input of `prompt_buckets.py` |
| [requests/](requests/) | every request of each load as `taper_replay.py` wrote it (`reset` rows are the unmeasured prefix requests) |
| [run/r123b_loads.sh](run/r123b_loads.sh) | starts one load on the rig: the line, and each rung's settings |
| [run/r123b_logs.sh](run/r123b_logs.sh) | the per-load facts from the run logs, card and host memory and server logs |

The tables come from R123a's reader, [r123a_ladder.py](../2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py),
unchanged (md5 `cd13ac0ade46bafb95f3f8cc22fd0503`).

The rig scripts are R43's and C218's, unchanged (md5 checked on the rig):
- [r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh) `b79bb3d224ad5f7ace4bbf4e3505cb51`, for `base` and
  `base2`. From `embd` on, the load runs `r43_run_noemb.sh`, which the launcher makes on the rig from `r43_run.sh`
  with only ` -otd token_embd=CUDA4` removed (md5 `ce687e4fedc61fec13e6984fb6cceef7`);
- [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) `e82a80981c9e88e6a235cc476b993de8`;
- [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) `329df66616395a0162065530a0f6108e`;
- [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) `a3c7dc91660a507cd415694d47bf7ced`;
- the v3 preset's [settings.env](../../../release/flash-next-v3/settings.env) `0c94ef0531708c73a94350b9ad73d4ff`.

`llama-server` md5 `9bf23c162646c9d1356d0beafcce0afa`.

## Commands

A folder on the rig (`WORK`) holds those five files. The build's rig tarball is unpacked in `$BUILDS/$BUILD`, the
model's first shard is in `MODEL` (shards 2 and 3 beside it) and the MTP draft in `DRAFT`. Each load runs detached
and ends with `$WORK/r123b_<rung>/DONE`.

```bash
export RIG WORK BUILDS MODEL DRAFT BUILD=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern
for rung in base embd sparse queues b4096 base2; do
  RUNG=$rung bash run/r123b_loads.sh
  ssh "$RIG" "while [ ! -e $WORK/r123b_$rung/DONE ]; do sleep 30; done"
  ssh "$RIG" "tar cf - -C $WORK r123b_$rung r123b_$rung.out r123b_$rung.launch" | tar xf - -C data/
done
```

The server line of `base` (`rig_taper.sh` adds everything up to `-c` and from `-np` on; the repeated
`--spec-draft-n-ubatch` keeps its last value, 512):

```
llama-server -m $MODEL -ngl 99 -fa on -c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT \
  --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host \
  --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 \
  --ctx-checkpoints-incall --ubatch-taper short=600:64 \
  -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5
# environment, base (26 switches set):
#   the v3 preset's GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0
#     GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode;
#   R90's GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5
#     GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1;
#   C720's LLAMA_QSA_KPOOL_STAGE=1 and C719's LLAMA_QSA_POOL_MASK_DEV=1;
#   R122's LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1
#     LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1;
#   R123a's GGML_CUDA_GRAPHS_FORCE=1 LLAMA_MTP_KV_ONLY=1 LLAMA_MTP_CATCHUP_OVERLAP=1.
# each rung adds its setting to the one before: embd LLAMA_DRAFT_EMBD_DEV=layer and no -otd token_embd=CUDA4,
#   sparse GGML_CUDA_FATTN_SPARSE_PASCAL_PROMPT=1, queues CUDA_SCALE_LAUNCH_QUEUES=2x, b4096 -b 4096 at the end of the
#   line (the last -b wins). MIN_AVAIL_MB=60.
# passes: "short=600:64 short=600:64", FRESH=0
```

The tables, on any machine with the six load folders in `data/`:

```bash
L="--load base=data/r123b_base/load-A"
for r in embd sparse queues b4096 base2; do L="$L --load $r=data/r123b_$r/load-A"; done
python3 -I ../2026-11-12_r123a_prompt_ladder/run/r123a_ladder.py $L --base base --base2 base2 --out .
python3 -I ../2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py turns.csv base:p0 base:p1 embd:p0 embd:p1 \
  sparse:p0 sparse:p1 queues:p0 queues:p1 b4096:p0 b4096:p1 base2:p0 base2:p1 > prompt_buckets.tsv
S=$(for r in base embd sparse queues b4096 base2; do printf '%s=data/r123b_%s/load-A/server.log ' $r $r; done)
python3 -I ../2026-11-12_r60_taper_v4/run/realloc_counts.py $S > reallocs.tsv
OUT=. REPO=../../.. bash run/r123b_logs.sh $(for r in base embd sparse queues b4096 base2; do printf '%s=data/r123b_%s ' $r $r; done)
```

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
