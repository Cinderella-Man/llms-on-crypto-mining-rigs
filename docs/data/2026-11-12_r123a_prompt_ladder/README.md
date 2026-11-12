# The prompt switches as one ladder on an agent session's prompts (R123a, Q1401, 2026-11-12)

What this measured: how much four prompt-side switches of v4's line speed up the reading of a coding agent's own
prompts on nine P106-090 cards, each switch added on top of the ones before. They are:

- `GGML_CUDA_GRAPHS_FORCE=1` (C702): prompt micro-batches are captured and replayed as CUDA graphs;
- `LLAMA_MTP_KV_ONLY=1` (C620 with C748's fix): the MTP draft's catch-up over a prompt writes only its caches;
- `LLAMA_MTP_CATCHUP_OVERLAP=1` (C703): the draft's catch-up runs while the target's next call is queued;
- `LLAMA_NGRAM_PREFETCH=1` (C761): a helper thread reads the n-gram rows of a call's later micro-batches ahead.

Until now they were timed only on a fresh 4,097-token prompt or on the deep lane's 8,128-token chunks
([2026-11-11_r82_c702_p106](../2026-11-11_r82_c702_p106/README.md),
[2026-11-12_r84a_c703_identity](../2026-11-12_r84a_c703_identity/README.md)). The workload here is R43's replay of
the agent check's 62 prompts ([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md)), read by the size of
each turn's new prompt. Registry row [Q1401](../../../quants/experiments.md) (its first half).

The build is `pascal-p106-v4` `d097d6d88`, the `-kern` rig tarball. The model is Swift Qwen3.8-Flash-Next GSQ-RCO
Q2_0 with its MTP draft. Every load starts from the line every rig row of 2026-11-12 starts from:

- R90's `gains` line ([2026-11-11_r90_v4_vs_v3](../2026-11-11_r90_v4_vs_v3/README.md));
- `--ubatch-taper short=600:64` (R60, [2026-11-12_r60_taper_v4](../2026-11-12_r60_taper_v4/README.md));
- without `LLAMA_SAMPLING_TOP_K_ROWS=1`, which reallocates the prompt graph on every request (FN84,
  [2026-11-12_fn84_graph_realloc](../2026-11-12_fn84_graph_realloc/README.md));
- C720's and C719's k-pool switches;
- R122's eight host switches of the decode step ([2026-11-12_r122_host_ladder](../2026-11-12_r122_host_ladder/README.md)).

## Result

**Graphs, the KV-only catch-up and the overlapped catch-up go into the stacked line; the n-gram prefetch does not.**
Together the first three read the 62 prompts in 247.11 and 240.02 s a pass against `base`'s 305.38 and 295.68. That
is −56.96 s, −18.95 %, a gain in every size bucket, with `base`'s greedy replies on all 62 turns.

| Rung | Switch added | Seconds a pass | Against the rung before | Class | Replies as `base` |
|---|---|---|---|---|---|
| `base` | — | 305.38, 295.68 | — | reference | 62 of 62 |
| `graphs` | `GGML_CUDA_GRAPHS_FORCE=1` | 259.42, 252.74 | −44.45 s (−14.79 %) | gain | 62 of 62 |
| `kvonly` | `LLAMA_MTP_KV_ONLY=1` | 249.92, 243.01 | −9.62 s (−3.76 %) | gain | 62 of 62 |
| `overlap` | `LLAMA_MTP_CATCHUP_OVERLAP=1` | 247.11, 240.02 | −2.90 s (−1.17 %) | negligible (a gain at 2,000+) | 62 of 62 |
| `ngram` | `LLAMA_NGRAM_PREFETCH=1` | 248.03, 249.28 | +5.09 s (+2.09 %) | loss | 62 of 62 |
| `base2` | — (the line again) | 307.51, 299.82 | +3.14 s against `base` (the spread) | reference | 62 of 62 |

A load's value is the mean of its two passes. A rung is classed against the rung before it, with the spread being
the difference between the two `base` loads' means: 3.14 s over all 62 turns, and 0.05, 0.50, 1.97 and 0.61 s in the
four size buckets.

- **The prompt graphs carry most of it.** `graphs` is faster in every bucket: −10.60 s under 512 new tokens, −6.57
  at 512–1,023, −18.36 at 1,024–1,999 and −8.91 at 2,000 and more. Turns of 1,024–1,999 new tokens read at 170.2
  tok/s against 141.6.
- **The KV-only catch-up adds a gain in every bucket**: −1.96, −1.55, −4.14 and −1.97 s.
- **The overlapped catch-up acts only where a prompt is cut into several calls.** That happens on the three turns of
  2,000 or more new tokens, which read −2.83 s faster (−7.48 %, spread 0.61). In the other buckets the time moves
  by 0.05 s or less. The server logged "4 catch-ups ran after the next call was queued" in the whole load. Over all
  62 turns that is within the spread: negligible with the same replies, so by the row's rule it joins the line.
- **The n-gram prefetch loses at 1,024 or more new tokens:** +3.75 s at 1,024–1,999 (spread 1.97) and +1.27 s at
  2,000 and more (spread 0.61). Below 1,024 it changes nothing (+0.04 and +0.02). Its second pass is the slow one.
  At 1,024–1,999 it took 92.68 s against its own first pass's 88.38. Every other load's second pass is as fast as its
  first or faster there (`overlap` 86.69 then 86.86). The helper issued 954,976 n-gram rows and `set_input` 436,848,
  0 differed, and the helper reached its 48 MiB cap. Why it loses is its own row (R106, Q1351), with its trace.

Every rung up to each one, against `base`:

| Line | Under 512 | 512–1,023 | 1,024–1,999 | 2,000+ | All 62 |
|---|---|---|---|---|---|
| `base`, s a pass (mean) | 77.36 | 65.08 | 109.33 | 48.76 | 300.53 |
| + graphs + KV-only | −12.56 (−16.23 %) | −8.12 (−12.47 %) | −22.51 (−20.59 %) | −10.89 (−22.32 %) | −54.07 (−17.99 %) |
| + overlap (the line kept) | −12.57 (−16.25 %) | −8.12 (−12.47 %) | −22.55 (−20.63 %) | −13.72 (−28.13 %) | −56.96 (−18.95 %) |
| + n-gram prefetch | −12.53 (−16.19 %) | −8.10 (−12.44 %) | −18.80 (−17.19 %) | −12.45 (−25.53 %) | −51.87 (−17.26 %) |
| the line kept, tok/s | 92.0 | 125.7 | 178.4 | 217.6 | 148.7 |
| `base`, tok/s | 77.1 | 110.0 | 141.6 | 156.4 | 120.5 |

**Time to the first token.** The replay asks for 8 greedy tokens and streams none, so the request's wall time
(prompt plus 8 tokens) stands in for it, next to the server's own prompt time. Medians and p90 over the 62 turns:

| Load | Pass | Wall median | Wall p90 | Prompt median | Prompt p90 |
|---|---|---|---|---|---|
| `base` | 0, 1 | 3.98, 3.71 s | 12.75, 12.60 s | 3.46, 3.45 s | 12.33, 12.32 s |
| `graphs` | 0, 1 | 3.26, 3.11 | 9.12, 9.07 | 2.87, 2.79 | 8.84, 8.58 |
| `kvonly` | 0, 1 | 3.15, 2.97 | 8.69, 8.62 | 2.75, 2.66 | 8.36, 8.20 |
| `overlap` | 0, 1 | 3.15, 2.97 | 8.65, 8.63 | 2.75, 2.67 | 8.34, 8.17 |
| `ngram` | 0, 1 | 3.14, 2.96 | 8.90, 9.36 | 2.73, 2.65 | 8.57, 8.90 |
| `base2` | 0, 1 | 4.00, 3.71 | 12.85, 12.45 | 3.47, 3.45 | 12.41, 12.16 |

**What else held in every load:**

- **Replies:** the same 8 greedy tokens as `base` on 62 of 62 turns in both passes, and both passes agree.
- **Graph reallocations:** 0 in all 146 requests of each load (124 turns and 22 unmeasured prefix requests). FN84's
  cause is off.
- **Host memory:** the lowest MemAvailable was 468, 467, 464, 472, 470 and 465 MiB, in ladder order. The KV-only
  catch-up's 66 and 93 MiB in R76 came at deep-lane depths; these contexts stay under about 7,000 tokens. The
  watchdog's floor (60) never acted.
- **Cards:** the highest card held 5,075 MiB after loading and 5,163 after the replay in every load. With the
  overlapped catch-up, card 8 (the draft's) held 80 MiB more after loading (4,815 against 4,735), as R84 (a)
  estimated.
- **Checkpoints and taper:** 315 checkpoint lines and 150 taper lines in every load.
- **The switches acted:** each load's switch line has 23, 24, 25, 26, 27 and 23 switches. Each rung's own line is in
  its server log ([switch_lines.txt](switch_lines.txt)).

**Est. what it changes for an agent:** prompt reading was 33 % of an agent task in R90's agent check, so −18.95 % of
it is est. up to 6.3 % of a task. R72's agent check on the stacked line `v4c2` measures it.

Seen on the way, on another rig and not classed: R60's line, which still reallocated the prompt graph on every
request, read the same 62 prompts with the same taper in 469.17 s a pass on rig 1. That is the line `base` has before
FN84's fix and R122's switches.

## How it ran

One load a rung, each a fresh server with two passes of R43's replay, every request carrying the line's taper
(`short=600:64`). No fresh prompts: the replay runs those with the taper off and on, not the line's. Nothing else ran
on the rig during a load, and each load was copied off before the next started.

| Load | Start | Server up after | Done | Load took |
|---|---|---|---|---|
| `base` | 14:34:32 | 227 s | 14:51:04 | 16 min 32 s |
| `graphs` | 14:51:41 | 226 s | 15:06:22 | 14 min 41 s |
| `kvonly` | 15:06:43 | 226 s | 15:21:01 | 14 min 18 s |
| `overlap` | 15:21:14 | 226 s | 15:35:27 | 14 min 13 s |
| `ngram` | 15:35:45 | 226 s | 15:50:08 | 14 min 23 s |
| `base2` | 15:59:16 | 226 s | 16:15:54 | 16 min 38 s |

`base2` started 9 minutes after `ngram` ended, as the machine that drives the loads restarted in between. No load was
running while it restarted.

## Files

| File | What |
|---|---|
| [decision.md](decision.md) | each rung's class in words, against the rung before and against `base`, and where it goes by the rule |
| [ladder.tsv](ladder.tsv) | per load and size bucket: turns, tokens, seconds of each pass, the mean, tok/s, the rung before's mean (and `base`'s), the difference, the spread and the class |
| [prompt_buckets.tsv](prompt_buckets.tsv) | per load, pass and bucket: turns, tokens, prompt seconds and tok/s, the median wall time (R90's `prompt_buckets.py`) |
| [first.tsv](first.tsv) | per load and pass: median and p90 of the wall time and of the server's prompt time |
| [replies.tsv](replies.tsv) | per load: turns whose greedy reply is `base`'s and `base2`'s in both passes, and whose two passes agree |
| [reallocs.tsv](reallocs.tsv) | the target graph's reallocations per load and bucket of prompt tokens |
| [memory.tsv](memory.tsv) | per load: the lowest MemAvailable and when |
| [cards.tsv](cards.tsv) | per load: each card's memory after loading and after the replay |
| [loads.tsv](loads.tsv) | per load: start, server start, replay end, done, switches set, checkpoint and taper lines, the replay's own line |
| [switch_lines.txt](switch_lines.txt) | per load: the server's switch list and the lines the ladder's switches print |
| [turns.csv](turns.csv) | every replayed turn of every pass in the agent check's `turns.csv` form, the input of `prompt_buckets.py` |
| [requests/](requests/) | every request of each load as `taper_replay.py` wrote it (`reset` rows are the unmeasured prefix requests) |
| [run/r123a_loads.sh](run/r123a_loads.sh) | starts one load on the rig: the line, and each rung's switches |
| [run/r123a_ladder.py](run/r123a_ladder.py) | the ladder's tables and classes; R60's `r60_buckets.py` for a ladder of loads instead of taper settings in one load |
| [run/r123a_logs.sh](run/r123a_logs.sh) | the per-load facts from the run logs, card memory and server logs |

The rig scripts are R43's and C218's, unchanged (md5 checked on the rig):
- [r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh) `b79bb3d224ad5f7ace4bbf4e3505cb51`;
- [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) `e82a80981c9e88e6a235cc476b993de8`;
- [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) `329df66616395a0162065530a0f6108e`;
- [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) `a3c7dc91660a507cd415694d47bf7ced`;
- the v3 preset's [settings.env](../../../release/flash-next-v3/settings.env) `0c94ef0531708c73a94350b9ad73d4ff`.

`llama-server` md5 `9bf23c162646c9d1356d0beafcce0afa`.

## Commands

A folder on the rig (`WORK`) holds those five files. The build's rig tarball is unpacked in `$BUILDS/$BUILD`, the
model's first shard is in `MODEL` (shards 2 and 3 beside it) and the MTP draft in `DRAFT`. Each load runs detached
and ends with `$WORK/r123a_<rung>/DONE`.

```bash
export RIG WORK BUILDS MODEL DRAFT BUILD=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern
for rung in base graphs kvonly overlap ngram base2; do
  RUNG=$rung bash run/r123a_loads.sh
  ssh "$RIG" "while [ ! -e $WORK/r123a_$rung/DONE ]; do sleep 30; done"
  ssh "$RIG" "tar cf - -C $WORK r123a_$rung r123a_$rung.out" | tar xf - -C data/
done
```

The server line of every load (`rig_taper.sh` adds everything up to `-c` and from `-np` on; the repeated
`--spec-draft-n-ubatch` keeps its last value, 512):

```
llama-server -m $MODEL -ngl 99 -fa on -c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT \
  --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host \
  --spec-draft-n-ubatch 512 --spec-draft-sampling probabilistic --spec-draft-p-min 0 --ctx-checkpoints-vram 4 \
  --ctx-checkpoints-incall --ubatch-taper short=600:64 \
  -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5
# environment, base (23 switches set):
#   the v3 preset's GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0
#     GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode;
#   R90's GGML_CUDA_FATTN_SPARSE_PASCAL=1 GGML_CUDA_FATTN_SPARSE_PASCAL_MIN_RATIO=1.0 GGML_CUDA_FATTN_SPARSE_PASCAL_NSM=5
#     GGML_CUDA_MMVQ_PASCAL_DMT=1 GGML_CUDA_MMVQ_PASCAL_Q5K_MT=1 LLAMA_QSA_INDEXER_ROWS=1 LLAMA_QSA_KPOOL_DEV=1;
#   C720's LLAMA_QSA_KPOOL_STAGE=1 and C719's LLAMA_QSA_POOL_MASK_DEV=1;
#   R122's LLAMA_DRAFT_PREFETCH=1 LLAMA_VERIFY_PREPARE=2 LLAMA_MTP_GRAPH_KEEP=1 LLAMA_KV_SEQ_RM_BOUND=1
#     LLAMA_SERVER_YIELD=2 LLAMA_MTP_EMBD_FIRST=1 LLAMA_MTP_DRAFT_CAP=1 LLAMA_MTP_SKIP_LAST_ROW=1.
# each rung adds its switch to the one before: graphs GGML_CUDA_GRAPHS_FORCE=1, kvonly LLAMA_MTP_KV_ONLY=1,
#   overlap LLAMA_MTP_CATCHUP_OVERLAP=1, ngram LLAMA_NGRAM_PREFETCH=1. MIN_AVAIL_MB=60.
# passes: "short=600:64 short=600:64", FRESH=0
```

The tables, on any machine with the six load folders in `data/`:

```bash
L=""; for r in base graphs kvonly overlap ngram base2; do L="$L --load $r=data/r123a_$r/load-A"; done
python3 -I run/r123a_ladder.py $L --base base --base2 base2 --out .
python3 -I ../2026-11-11_r90_v4_vs_v3/run/prompt_buckets.py turns.csv base:p0 base:p1 graphs:p0 graphs:p1 \
  kvonly:p0 kvonly:p1 overlap:p0 overlap:p1 ngram:p0 ngram:p1 base2:p0 base2:p1 > prompt_buckets.tsv
python3 -I ../2026-11-12_r60_taper_v4/run/realloc_counts.py \
  $(for r in base graphs kvonly overlap ngram base2; do printf '%s=data/r123a_%s/load-A/server.log ' $r $r; done) > reallocs.tsv
OUT=. bash run/r123a_logs.sh $(for r in base graphs kvonly overlap ngram base2; do printf '%s=data/r123a_%s ' $r $r; done)
```

`r123a_ladder.py` was checked on R60's published requests first. With R60's taper-off passes as its base it gives
R60's 492.68 and 480.70 s a pass, and 192.99 s at 1,024–1,999.

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
