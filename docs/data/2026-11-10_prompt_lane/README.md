# The prompt lane on the release line, switches only (R43, Q1155, 2026-11-10)

What this measured: how much of a coding agent's prompt reading on the Miner Llama v3.0 release line goes into the
context checkpoints the server takes, and what the switches already in the release build change, before any code is
written for the prompt path ([Q1155](../../../quants/experiments.md); the checkpoints' drains are
[Q770](../../../quants/experiments.md), the device stores [Q440](../../../quants/experiments.md), the prompt taper
[Q771](../../../quants/experiments.md)). The line is the v3 preset exactly: `pascal-p106-v3-b12076-ladder5`
`1551f5fe1`, UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, the MTP draft at three
tokens, the preset's environment ([settings.env](../../../release/flash-next-v3/settings.env)), `-c 65536`.

Flash-Next is a hybrid model: most of its layers keep a recurrent state, which cannot be rolled back, so llama-server
copies that state (112.6 MiB over the nine cards) into a checkpoint twice in every prompt, 516 and 4 tokens before
its end, to be able to resume there. Each copy waits for the nine-card pipeline to empty first.

## Result

**The checkpoints cost 30.72 % of prompt time on the release line**, and `--ctx-checkpoints-vram 4` wins half of that
back with no code. The replay of the agent check's 62 prompts takes 509.16 and 499.93 s a pass with the preset's
checkpoints, 349.57 s with none (load B, a diagnostic: without checkpoints an edit re-reads the whole prompt), and
437.54 and 416.17 s with the newest four checkpoints kept on the cards. That is above the 15 % line, so the
drain-free checkpoint (C217, through C663's port) is the first engine code for prompts. With the device stores on,
the drains still cost 18.10 % of prompt time (426.85 against 349.57 s), which is above that line too.

### The replay, a pass at a time

The agent check's 62 prompts (12 bug-fix tasks, Q218's `turns.csv`), 36,793 new prompt tokens a pass, each turn asking
for 8 greedy tokens ([the method](../2026-11-04_prompt_taper/README.md#on-the-rig-next-c218r)). The taper is set per
request, so each load runs both arms; load A's and the device stores' passes go off, on, on, off.

| Load | Checkpoints | Taper | Prompt time a pass, s | tok/s a pass | Against A with the taper off | Class | Greedy replies as A off's |
|---|---|---|---|---|---|---|---|
| A (the preset) | 4, on the host | off | 509.16, 499.93 | 72.26, 73.60 | — (the base) | — | 62 of 62 |
| A | 4, on the host | on | 434.46, 430.50 | 84.69, 85.47 | −72.06 s (−14.28 %) | gain | 61 of 62 |
| B | none (`--ctx-checkpoints 0`) | off | 349.57 | 108.21 | −154.98 s (−30.72 %) | diagnostic | 61 of 62 |
| B | none | on | 318.08 | 118.92 | −186.47 s (−36.96 %) | diagnostic | 62 of 62 |
| vram4 | 4, all on the cards (`--ctx-checkpoints-vram 4`) | off | 437.54, 416.17 | 84.09, 88.41 | −77.69 s (−15.40 %) | gain | 62 of 62 |
| vram4 | 4, all on the cards | on | 358.94, 349.97 | 102.50, 105.13 | −150.09 s (−29.75 %) | gain | 62 of 62 |

- **The class** compares the mean of a load's passes with the base's (504.55 s) against the larger of the two arms'
  own spread between their first and last pass: 9.23 s for the base, 21.37 s for the device stores with the taper
  off. One load each, so the spread is the passes', not the loads'.
- **Load B reads 1,033 tokens more a pass** (37,826): one task's first turn (`bugfix-python-affine-cipher`) is sent
  cold with the 1,549-token prefix, and where load A resumes from a checkpoint at 1,033 tokens, load B reads all
  2,780 (19.87 against 16.34 s). Load B's gain is the drains' cost less that turn.
- **The greedy replies** of the device stores are the base's on every turn. One turn's 8 tokens differ in the taper's
  arm of load A and in load B's taper-off arm, the rounding of a different micro-batch split.

### Where the drains' cost goes

| Per pass (the 62 measured turns) | A, host checkpoints | vram4, device stores |
|---|---|---|
| checkpoints created | 124 | 124 |
| their copies, as the server logs them | 79.95 and 79.61 s | 5.70 and 2.74 s |
| checkpoints restored | 1 (0.62 s) | 1 (0.00 s) |
| a checkpoint's copy, median over the load (600 copies) | 641.5 ms | 17.6 ms |
| a restore, median over the load (47 restores) | 622.1 ms | 2.6 ms |

- **The copies are half of the drains' cost:** 79.78 s of the 154.98 s a pass (mean of the two passes), two copies a
  turn. The device stores take them to 5.70 and 2.74 s, and the pass gains 77.69 s.
- **The other half is the drain itself:** the device stores' line still takes 77.28 s a pass more than load B, of
  which the device copies are 4.22 s (the mean of 5.70 and 2.74). A checkpoint makes the server stop the prompt's
  calls 516 and 4 tokens before its end and wait for the pipeline to empty, so the 512 tokens before the last 4, and
  then the last 4, each cross the nine cards alone. No switch removes that; a checkpoint written by a stream-ordered
  copy inside the pipelined calls does (Q770).

### By prompt size

The turns' new tokens in the agent check's size bins; the median turn over every pass, in seconds
([bins.tsv](bins.tsv); every turn's time in [requests/](requests/)).

| New tokens | Turns | A off | A on | B off | B on | vram4 off | vram4 on |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1–64 | 18 | 2.41 | 2.41 | 1.04 | 1.03 | 1.12 | 1.12 |
| 65–255 | 8 | 3.62 | 3.10 | 2.60 | 1.68 | 2.47 | 1.84 |
| 256–511 | 12 | 6.93 | 5.02 | 5.61 | 3.78 | 5.76 | 3.77 |
| 512–1,023 | 10 | 10.79 | 8.38 | 6.96 | 5.88 | 9.48 | 7.14 |
| 1,024–1,999 | 11 | 15.94 | 14.59 | 9.14 | 11.38 | 14.32 | 13.49 |
| 2,000 and more | 3 | 19.09 | 18.82 | 18.51 | 16.00 | 17.77 | 17.76 |

- **A short turn's fixed cost is the two copies:** a turn of at most 64 new tokens takes 2.41 s with the preset and
  1.12 s with the device stores (1.04 s without checkpoints).
- **The drain's own cost is largest from 512 tokens on,** where the prompt is long enough for the last 512 tokens to
  run alone: 9.48 against 6.96 s and 14.32 against 9.14 s with the device stores against none.
- **The taper pays from 65 to 1,023 new tokens** (its `short` rule and the 512-token call after the first cut in
  64-token micro-batches) and hardly at 64 or fewer. Without checkpoints it is slower at 1,024–1,999 (11.38 against
  9.14 s).

### Fresh 2,049-token prompts

Nothing cached, four prompts a load, each with the taper off and on ([fresh.tsv](fresh.tsv)), in seconds:

| Load | Taper off | Taper on |
|---|---|---|
| A | 16.27, 15.96, 16.11, 15.97 | 16.71, 16.91, 16.61, 17.09 |
| B | 12.78, 12.98, 12.76, 12.42 | 11.61, 11.91, 11.60, 11.96 |
| vram4 | 14.98, 14.78, 14.90, 14.71 | 15.51, 15.71, 15.40, 15.88 |

- **The drains cost a fresh 2k prompt 3.27 s** (medians 16.04 and 12.77 s; Q770 had predicted at least 5 s). The two
  copies are 1.28 s of it with host checkpoints and 0.06 s with device stores.
- **On a lone fresh prompt the taper is a loss while checkpoints are on** and a gain without them: its micro-batches
  of 64 at the start of a long, pipelined call cost more than they save there (Q1235).

### Memory

| Load | Lowest MemAvailable, MiB | Swap used after the load, MiB | Fullest card after the load |
|---|---:|---:|---|
| A | 119 | 1,044 | card 3, 5,159 of 6,144 MiB |
| B | 312 | 1,054 | — |
| vram4 | 148 | 1,029 | card 3, 5,201 of 6,144 MiB |

The watchdog's floor was 60 MiB (the preset's first long request left the host at 75, 70, 96 and 86 MiB in S37's
four loads); it never fired. Swap was at 20 MiB before run 1. The four device stores (112.57 MiB each over the nine cards) add 48 MiB
on card 0 and 12 MiB on card 8 to what the host-checkpoint load used at its end.

## What goes into the preset

`--ctx-checkpoints-vram 4` (a gain alone, the same greedy replies) and the taper (`--ubatch-taper on`, a gain on the
replay alone and on top of the device stores) go into the Flash-Next preset, together −29.75 % of prompt time on the
replay, once the agent check confirms it on agentic speed (R59, Q1234). `LLAMA_SERVER_TAIL_CKPT_MIN=4096` (Q263) was
not run: it is R58 (Q1233), now on top of the device stores.

## Files

| File | What |
|---|---|
| [versus.tsv](versus.tsv) | per load and taper arm: prompt time a pass, against the base, the spread, the class, the greedy replies |
| [versus_bins.tsv](versus_bins.tsv) | the same by size bin, with each bin's share of the base's prompt time |
| [passes.tsv](passes.tsv) | per load, arm and pass: turns, tokens, seconds, tok/s, the checkpoints created and restored and their logged time |
| [bins.tsv](bins.tsv) | per load, arm and size bin: turns, the median turn, the sum per pass |
| [fresh.tsv](fresh.tsv) | every fresh 2,049-token prompt, with its checkpoints' time |
| [loads.tsv](loads.tsv) | per load: checkpoint lines (count, median and total time, bytes on the cards), lowest MemAvailable |
| [summary.md](summary.md) | the classes in words and `taper_replay.py`'s own summary per load |
| [requests/A.tsv](requests/A.tsv), [requests/B.tsv](requests/B.tsv), [requests/vram4.tsv](requests/vram4.tsv) | every request of each load as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/r43_run.sh](run/r43_run.sh) | one run on the rig: the preset's environment and line around `rig_taper.sh` |
| [run/lane_tables.py](run/lane_tables.py) | the tables from the load folders, with the checkpoint time of each request from the server log |

The rig script and the replay are unchanged from C218's folder:
[rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (md5 `e82a80981c9e88e6a235cc476b993de8`) and
[taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`), with
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`).

## Commands

A folder on the rig (`WORK`) with `r43_run.sh`, `rig_taper.sh`, `taper_replay.py`, `turns.csv` and the preset's
`settings.env`; the release build in `$BUILDS/$BUILD`, the model's first shard in `MODEL` (shards 2 and 3 beside it),
the MTP draft in `DRAFT`. Each run is detached and ends with `$OUT/DONE`; nothing else ran on the rig meanwhile.

```bash
# run 1: loads A (--ctx-checkpoints 4) and B (--ctx-checkpoints 0), 13:51 to 15:00
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/run1 LOADS='A B' \
  setsid nohup bash $WORK/r43_run.sh > $WORK/run1.out 2>&1 < /dev/null &"
# run 2: load A with the newest four checkpoints on the cards, 15:01 to 15:41
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/run2 LOADS=A \
  ARGS_ADD='--ctx-checkpoints-vram 4' setsid nohup bash $WORK/r43_run.sh > $WORK/run2.out 2>&1 < /dev/null &"
# run 3 (not run; R58): load A with the end-of-prompt checkpoint only 4,096 tokens past the newest one
ssh $RIG "cd $WORK && LLAMA_SERVER_TAIL_CKPT_MIN=4096 BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT \
  OUT=$WORK/run3 LOADS=A setsid nohup bash $WORK/r43_run.sh > $WORK/run3.out 2>&1 < /dev/null &"
```

The server line each load ran (`rig_taper.sh` adds everything up to `-c` and from `-np` on):

```
llama-server -m $MODEL -ngl 99 -fa on -c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT \
  --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host [--ctx-checkpoints-vram 4] \
  -np 1 -bs --ctx-checkpoints 4|0 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5
# environment: GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0
#              GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode; MIN_AVAIL_MB=60
```

The tables, on any machine with the load folders (`run1/load-A`, `run1/load-B`, `run2/load-A`, each with its
`server.log`):

```bash
python3 -I run/lane_tables.py --base A --load A=run1/load-A --load B=run1/load-B --load vram4=run2/load-A --out .
```

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the MTP
draft as the release lists them. The prompts' text is the Python standard library's source on the rig (PSF licence),
tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
