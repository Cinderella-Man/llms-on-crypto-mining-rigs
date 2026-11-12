# The KV-only draft pass with the k-pool tables on the cards, on a P106 (R91, Q1323, 2026-11-12)

What this is: the rig run of C748 ([the change and its CPU check](../2026-11-11_c748_kvonly_kpool/README.md),
[Q1320](../../../quants/experiments.md)), where two switches of the v4 line meet in the MTP draft's catch-up passes.
C620's `LLAMA_MTP_KV_ONLY=1` ([R76's rig run](../2026-11-11_r76_c620_p106/README.md)) cuts a draft pass with no output
row down to the writes of the draft layer's caches. C100's `LLAMA_QSA_KPOOL_DEV=1` keeps the QSA k-pool tables on each
card and sends each micro-batch only the rows that changed ([R74](../2026-11-11_r74_c100_p106/README.md)), and C720's
`LLAMA_QSA_KPOOL_STAGE=1` puts the five small k-pool inputs into that update
([R80](../2026-11-11_r80_c720_p106/README.md)). Before C748 the cut pass left the card's copy of the tables behind, so
the pass after it sent both tables whole. C748 makes the cut pass update the tables the way the whole pass does.

One deep-lane run ([bench/deep_lane](../../../bench/deep_lane/README.md)) on nine P106-090 cards with Miner Llama's
`pascal-p106-v4` at `d097d6d88`, which holds C748's code; no other switch of the v4 line was set. Depths 0, 16,384 and
49,152 tokens. Eight loads: the preset's control as the warm-up, then two loads each of the KV-only pass alone (`kv`),
the tables on the cards alone (`dev`) and both together (`kvdev`), and last `kvdev` with C100's exactness check
(`check`). Registry [Q1323](../../../quants/experiments.md).

## Result

**Exact on the cards, and the two switches stack.** With both on, the step on the repeated request is 1.83 ms
shorter than with the KV-only pass alone at 16,384 tokens and 3.84 ms shorter at 49,152 (gains). The two prompt
chunks read 11.18 % faster than with the tables alone (104.65 and 104.27 against 93.81 and 94.10 tok/s, a gain).
Each switch keeps the gain it has on its own.

- **Exact.** The `check` load ran C100's check with both switches on. It compared the tables' copies with the host's
  tables, byte for byte, on all 2,196 micro-batches of the MTP draft (card 8) and all 561 of the main model (nine
  cards). Every copy was identical and nothing aborted. The log has no line that reports a difference.
- **The tables stay current through the cut pass.** When the draft memory is freed it logs 2,196 micro-batches, 1 full
  rewrite and 87,929 of 12,059,712 table rows sent (40.0 a micro-batch). These are the counts R80's `check` load
  logged with the whole draft pass ([kpool_counts.tsv](kpool_counts.tsv)). The one full rewrite is the first
  micro-batch. The load generated 457 steps and its draft ran 2,196 micro-batches, so a full rewrite after every
  catch-up would have shown as hundreds. The main model's tables: 561 micro-batches, 1 full rewrite, 31,084 of
  3,074,944 rows.
- **The same replies and drafts.** All eight loads wrote the same greedy reply (token ids' md5) on both passes at
  every depth: `08569d4a…` at 0, `41cfc791…` at 16,384 and `3a311cdd…` at 49,152 ([replies.tsv](replies.tsv)), the
  replies of R74 and R80. Every load accepted the same draft tokens: 178 of 229 and 181 of 220 at 0, 184 of 210 at
  16,384, 182 of 218 at 49,152.
- **The step on pass 2** (the repeated request at a depth) is classed against each arm's two loads:

  | Depth | `kv` (kv1, kv2) | `dev` (dev1, dev2) | `kvdev` (kvdev1, kvdev2) | `kvdev` − `kv` | spread | class | `kvdev` − `dev` | spread | class |
  |---:|---|---|---|---:|---:|---|---:|---:|---|
  | 0 | 120.63, 120.62 | 121.95, 121.69 | 120.35, 120.40 | −0.25 | 0.05 | **gain** | −1.44 | 0.26 | **gain** |
  | 16,384 | 147.34, 147.48 | 146.63, 146.54 | 145.67, 145.47 | −1.83 | 0.20 | **gain** | −1.01 | 0.20 | **gain** |
  | 49,152 | 178.75, 178.80 | 176.12, 178.00 | 176.41, 173.47 | −3.84 | 2.93 | **gain** | −2.12 | 2.93 | negligible |

  ms a step, one request a cell ([verdict.tsv](verdict.tsv)). At 49,152 the two `kvdev` loads are 2.94 ms apart and the
  two `dev` loads 1.88 ms, so the KV-only pass's −2.12 ms on top of the tables stays within the spread there.
- **Prompt reading** (each depth's prime: 16,320 tokens on an empty cache, then 33,284 on 15,804 cached):

  | Depth | `kv` | `dev` | `kvdev` | `kvdev` against `kv` | class | `kvdev` against `dev` | class |
  |---:|---|---|---|---:|---|---:|---|
  | 16,384 | 141.44, 141.21 | 128.97, 131.12 | 141.71, 142.59 | +0.58 % (spread 0.88 tok/s) | negligible | +9.31 % (spread 2.16) | **gain** |
  | 49,152 | 92.80, 92.75 | 82.75, 82.66 | 92.76, 92.12 | −0.36 % (spread 0.63) | negligible | +11.78 % (spread 0.63) | **gain** |

  tok/s ([prompt.tsv](prompt.tsv)). Both chunks together (49,604 tokens) took 473.983 and 475.747 s with both switches,
  474.044 and 474.413 s with `kv`, 528.787 and 527.122 s with `dev`, and 522.288 s in the warm-up load. The tables on
  the cards do not change prompt reading, and the KV-only pass keeps all of its gain with them (R76: +10.0 % on its
  four chunks).
- **The two savings add up (est.).** Against the warm-up load (one load, the preset with the sparse attention only),
  pass 2's step changes by −1.23 ms with `kv`, −2.06 with `dev` and −3.07 with `kvdev` at 16,384 (the sum of the first
  two: −3.28), and by −1.91, −3.62 and −5.74 at 49,152 (sum −5.52). At depth 0: −1.03, +0.16 and −1.28 (sum −0.88).
- **Pass 1** (the first request at a new depth) is negligible at 16,384 and 49,152 in both pairs: the host swapped
  during some of these requests (up to 128 MiB moved to swap in `kvdev1` at 16,384), and each arm's two loads are up
  to 41.40 ms apart ([verdict.tsv](verdict.tsv), [host_memory.tsv](host_memory.tsv)). At depth 0 `kvdev` against `dev`
  is a gain (124.61 and 124.58 against 126.57 and 125.59 ms, spread 0.97) and against `kv` negligible (124.82 and
  124.53, spread 0.29).
- **Generation** on pass 2: 24.41 and 24.44 tok/s with both switches at 16,384, against 24.13 and 24.11 (`kv`) and
  24.25 and 24.26 (`dev`); at 49,152 19.61 and 19.94 against 19.35 and 19.35, and 19.64 and 19.44.
- **Memory.** After loading, `kvdev` holds on every card what `dev` holds, and `kv` what the warm-up load holds. With
  the tables on the cards, cards 1, 3 and 6 hold 16 MiB more than without them (C720's larger compute buffers, R80),
  cards 0, 2, 5, 7 and 8 2 MiB more and card 4 the same. Card 8, the draft's, peaks 18 MiB lower with the KV-only pass:
  4,691 MiB in `kvdev` and 4,689 in `kv`, against 4,709 in `dev` and 4,707 in the warm-up ([cards.tsv](cards.tsv)).
  The least free on any card was 977 MiB (card 3, in `dev`, `kvdev` and `check`). The host's lowest MemAvailable per load: 453 MiB in the warm-up, 300
  and 467 in `kv`, 90 and 87 in `dev`, 462 and 115 in `kvdev`, 81 in `check`. The loads that moved memory to swap
  (`kv1` up to 864 MiB in use, `kv2` 1,027, `kvdev1` 1,096, the warm-up 468) have the higher minima. The RAM watchdog
  (floor 60 MiB) never acted.

**So `LLAMA_MTP_KV_ONLY=1` with the tables on the cards (`LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1`) is exact on
a P106 and keeps both gains.** The KV-only rung can go on top of the tables in the prompt ladder (R123a) and in the
stacked line's agent check (R72).

## What ran

One deep-lane run on one rig, nothing else on the rig during the loads, 10:10:52 to 12:13:11 (122.3 minutes). Each
server came up in 225, 227, 224, 227, 224, 227, 226 and 225 s, in this order:

| Load | Arm | Start | Minutes | Its own variables and flags |
|---|---|---|---:|---|
| `warm` | the preset's control (the warm-up) | 10:10:52 | 14.8 | [the v3 preset](../../../release/flash-next-v3/settings.env) with `GGML_CUDA_FATTN_SPARSE_PASCAL=1` (R64's line) |
| `kv1` | `kv` | 10:25:40 | 13.9 | + `LLAMA_MTP_KV_ONLY=1` |
| `dev1` | `dev` | 10:39:36 | 14.8 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| `kvdev1` | `kvdev` | 10:54:23 | 14.1 | + `LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| `kvdev2` | `kvdev` | 11:08:27 | 13.9 | the same |
| `dev2` | `dev` | 11:22:19 | 14.8 | + `LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1` |
| `kv2` | `kv` | 11:37:06 | 14.0 | + `LLAMA_MTP_KV_ONLY=1` |
| `check` | `kvdev`, checked | 11:51:07 | 22.1 | + `LLAMA_MTP_KV_ONLY=1 LLAMA_QSA_KPOOL_DEV=1 LLAMA_QSA_KPOOL_STAGE=1 LLAMA_QSA_KPOOL_DEV_CHECK=1`, `-lv 4` |

- **Each load:** the client's warm-up, then at 0, 16,384 and 49,152 tokens a prime and two decodes of 256 greedy
  tokens, with a 4 s pause before every request (`PAUSE=4`). The warm-up load ran at all three depths
  (`WARM_DEPTHS="0 16384 49152"`, as in R74 and R80), so its replies are there to compare.
- **Two loads an arm, in mirrored order.** The design ([C748](../2026-11-11_c748_kvonly_kpool/README.md#what-the-rig-run-must-show-r91))
  named one load of each arm. A class needs the spread of repeated loads, so `kv`, `dev` and `kvdev` ran twice, the
  second round in reverse order, with `check` last.
- **The build:** `pascal-p106-v4` `d097d6d88` (llama-server `build 12661`), the rig tarball of C796's kernel merge
  (md5 `37a4c3731f76455dce14013210b2fac7`), in place of C748's own tarball: it holds C748's code. Every server log lists
  the switches set at load (C770): the preset's six and the sparse attention's, plus this load's own, and none that the
  build does not read.
- **The switches on the cards:** the `check` log has `LLAMA_MTP_KV_ONLY`'s own line, the stage switch's line for the
  model and the draft, and the tables kept on nine cards for the model (16,448 rows, 2.87 MiB on the cards, 0.31 MiB
  on the host) and on one card for the draft (0.32 MiB) ([kpool_lines.tsv](kpool_lines.tsv)). The check logged 12
  progress lines: the first micro-batch of each memory rewritten in full, every later one "in step".
- **The tables ran on a build machine's CPU** (`nice -n 19`), never on the rig. The deep lane's own tables
  ([table.md](table.md), [classes.tsv](classes.tsv)) take `kv` as their control and skip the warm-up load; rerun on
  the build machine they are the same bytes.

## Files

| File | What it holds |
|---|---|
| [verdict.tsv](verdict.tsv) | per measure (the step, prompt reading), pair (`kvdev` against `kv`, against `dev`), depth and pass: each load by value, the arms' means, the difference in ms and %, the spread and the class |
| [steps.tsv](steps.tsv) | per load, depth and pass: tokens, accepted of drafted, steps, generation ms, ms a step, tok/s, the reply's md5s |
| [prompt.tsv](prompt.tsv) | per load and depth: the prime's tokens sent, cached, read, its ms and tok/s |
| [replies.tsv](replies.tsv) | per depth, pass and load: the reply's token ids' and text's md5 |
| [kpool_counts.tsv](kpool_counts.tsv) | the `check` load's two exit lines as numbers: micro-batches, full rewrites, rows sent and checked, with the load's generation steps |
| [kpool_lines.tsv](kpool_lines.tsv) | per server log: each k-pool line kind (the switches at load, the KV-only line, the stage line, allocation, the check's counts, the counts at free), its count, first and last line |
| [cards.tsv](cards.tsv) | per load and card: MiB used and free after loading, the most used over the load |
| [host_memory.tsv](host_memory.tsv) | the host's memory per load and request from the RAM watchdog ([host_memory.py](../2026-11-10_deep_lane/run/host_memory.py)) |
| [requests.csv](requests.csv), [loads.tsv](loads.tsv) | the deep lane's line per request and per load (the binary written as `$BIN/llama-server`) |
| [table.md](table.md), [classes.tsv](classes.tsv) | the deep lane's own tables (control `kv`; the warm-up load skipped) |
| [run/r91_run.sh](run/r91_run.sh) | the run: start, wait and collect through `bench/deep_lane/remote.sh` |
| [run/r91_tables.py](run/r91_tables.py) | the step, both verdicts, prompt reading, the replies, the k-pool lines and counts and the memory from the deep lane's folder (R80's `r80_tables.py` with arms in place of fixed loads) |

## Commands

```bash
# from a machine that reaches the rig over ssh, at this repository's root (RIG: its ssh alias; BIN: the build's bin/
# on the rig; MODEL, MODEL_DRAFT: paths on the rig; TEXT_LOCAL: the release check's recorded requests,
# proxy.jsonl.bodies.gz; OUT: a results folder here)
D=docs/data/2026-11-12_r91_c748_rig/run
RIG=$RIG BIN=$BIN MODEL=$MODEL MODEL_DRAFT=$MODEL_DRAFT TEXT_LOCAL=$TEXT_LOCAL STEP=start bash $D/r91_run.sh
RIG=$RIG STEP=wait bash $D/r91_run.sh
RIG=$RIG OUT=$OUT STEP=collect bash $D/r91_run.sh

# on a build machine, CPU only, from this repository's root (OUT: the collected folder; RES: the tables)
nice -n 19 python3 -I $D/r91_tables.py $OUT --out $RES > $RES/r91.md
nice -n 19 python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py $OUT > $RES/host_memory.tsv
nice -n 19 python3 -I bench/deep_lane/deep_table.py $OUT | cmp - $OUT/table.md   # the rig's own tables
grep -a -E 'kpool_dev_check|k-pool tables|go up in the k-pool|switches: |LLAMA_MTP_KV_ONLY: ' $OUT/server-check.log
grep -a -c -i 'differ' $OUT/server-check.log                                       # 0
```

## Limits

- **The unfixed code is not in this run.** The build before C748 (`pascal-p106-v4-b12246-night`) with both switches
  would have sent both tables whole after every catch-up; C748 estimated that at 0.54 and 1.59 ms a step at 16,384 and
  49,152 tokens from the bytes. This run shows that the fixed line is exact and keeps both switches' gains; the
  saving against the unfixed line is not timed.
- **Two loads an arm, one order** (`kv`, `dev`, `kvdev`, then `kvdev`, `dev`, `kv`). Each value is one request. At
  49,152 tokens pass 2's loads of one arm were up to 2.94 ms apart, wider than in R80's run (0.24).
- **The sum of the two savings** is taken against the warm-up load, which is one load and the run's first; it is an
  estimate.

## Provenance and licence

Measured on 2026-11-12 on one rig of nine P106-090 cards, with the Miner Llama fork's `pascal-p106-v4` (llama.cpp,
MIT). The tables are this project's own measurements. The models are UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft; neither is republished. The prompts are cut from this project's recorded agent
requests; they and the replies are not part of this folder (`requests.csv` and the tables hold only checksums).
