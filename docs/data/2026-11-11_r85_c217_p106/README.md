# C217's checkpoints inside the calls on a P106 (R85, Q1306, 2026-11-11)

What this measured: whether `--ctx-checkpoints-incall`
([C217, Q770](../2026-11-11_c217_incall_checkpoints/README.md)) keeps everything the server computes on nine P106-090
cards, and what it gains on a coding agent's prompts. The switch takes a prompt's context checkpoints inside the
decode call instead of cutting the call there, so the nine-card pipeline no longer empties at each checkpoint. On the
v3.0 line those drains cost 77.28 s a pass of the agent check's 62 prompts even with the checkpoints on the cards
([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md), R43). Registry row
[Q1306](../../../quants/experiments.md).

The build is the v4 night merge `pascal-p106-v4-b12246-night` `c408dabcb` (C217's branch and 16 others, every switch
off by default; [2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), with only this switch set.
`LLAMA_MTP_CATCHUP_OVERLAP` (C703) was unset in both loads. Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 with its MTP draft,
the v3 preset's line and environment, `-c 65536`, as R43 ran it.

## Result

**A gain, with every reply, checkpoint and restore unchanged.** With the taper off, the checkpoints inside the calls
read the 62 prompts in 338.96 and 331.09 s a pass against 416.56 and 410.60 s with the same checkpoints taken
between calls. That is −78.55 s (−18.99 %) against a spread of 7.87 s. Replies, checkpoints and restores:
- the greedy replies are the same on 62 of 62 turns;
- the server took its checkpoints at the same token counts, in the same order, on all 300 requests, and 85 of the 124
  checkpoints a pass went inside a call;
- it restored the same checkpoints on all 300 requests;
- prompt, cache and reply token counts are the same on every request.

The taper adds to it: with the taper on in both loads, 321.40 and 321.09 s against 344.00 and 343.60 s (−22.55 s,
−6.56 %, a gain).

### The replay, a pass at a time

Two loads one after the other on one rig, nothing else running: `v` with `--ctx-checkpoints-vram 4` (the four newest
checkpoints in device stores, R43's run 2 on this build) and `i` with `--ctx-checkpoints-vram 4
--ctx-checkpoints-incall`. Each load replays the agent check's 62 prompts (12 bug-fix tasks, 36,793 new prompt tokens
a pass, 8 greedy tokens a turn) in four passes. The taper is set per request, so the passes go off, on, on, off.

| Load | Taper | Prompt time a pass, s | tok/s a pass | Against `v` with the taper off | Class | Greedy replies as `v` off's |
|---|---|---|---|---|---|---|
| `v` | off | 416.56, 410.60 | 88.33, 89.61 | — (the base) | — | 62 of 62 |
| `v` | on | 344.00, 343.60 | 106.95, 107.08 | −69.78 s (−16.87 %) | gain | 61 of 62 |
| `i` | off | 338.96, 331.09 | 108.55, 111.13 | −78.55 s (−18.99 %) | **gain** | **62 of 62** |
| `i` | on | 321.40, 321.09 | 114.48, 114.59 | −92.33 s (−22.32 %) | gain | 62 of 62 |

- **The class:** the mean of an arm's two passes is compared with the base's (413.58 s). The spread is the larger of
  the two arms' own differences between their passes: 5.96 s for the base, 7.87 s for `i` off.
- **The taper on both sides:** `i` on against `v` on is −22.55 s (−6.56 %) a pass, against a spread of 0.40 s:
  a gain. Within `i`, the taper takes 13.78 s (4.11 %) off, against 69.78 s (16.87 %) within `v`.
- **The one reply that differs** is `v`'s with the taper on: turn 3 of `bugfix-python-pig-latin`, in both of its
  passes. The taper splits that prompt into other micro-batches, and the rounding changes 8 greedy tokens. With the
  switch, a joined call has one taper plan, and its reply is the untapered one.

### The checkpoints and restores

From the two server logs (`-lv 5`), request by request ([ckpt_summary.tsv](ckpt_summary.tsv),
[ckpt_requests.tsv](ckpt_requests.tsv)). A checkpoint inside a call counts when it joins its batch (`goes inside the
call`); one before a call counts when it is created.

| Requests | Count | Same token counts, same order | Checkpoints `v` / `i` | Of them inside a call (`i`), all filled | Same restores | Restores `v` / `i` |
|---|---:|---:|---|---:|---:|---|
| turns, taper off (2 passes) | 124 | 124 | 248 / 248 | 170 | 124 | 2 / 2 |
| turns, taper on (2 passes) | 124 | 124 | 248 / 248 | 170 | 124 | 2 / 2 |
| the unmeasured prefix requests, off / on | 22 / 22 | 22 / 22 | 44 / 44 each | 23 / 22 | 22 / 22 | 21 / 21, 22 / 22 |
| fresh 2,049-token prompts, off / on | 4 / 4 | 4 / 4 | 8 / 8 each | 4 / 4 | 4 / 4 | 0 / 0 |

- **Which checkpoints go inside a call** (the first taper-off pass): a turn of at most 516 new tokens takes one, the
  checkpoint 4 tokens before the end (38 turns). Its other checkpoint is where the new tokens start, before any call.
  A turn of 517 to 2,048 new tokens takes both inside (22 turns). Of the two turns over 2,048, one takes both; the
  other (2,308 new tokens) takes one. There, the part after the other checkpoint would have run past `-b 2048`, so
  the call stops at it, as without the switch (joining past the batch's room is C736, Q1303).
- **Nothing was dropped:** no `was not taken inside the call` line, no in-call checkpoint erased, none moved to the
  host. No error lines appear in either log, and the warning lines are the same three start-up lines in both. The
  server logged the switch's start-up line once. C703's catch-up overlap logged nothing, so it was off.
- **The checkpoints' cost in the log:** 0.18 s a pass of host bookkeeping in `i` (median 0.3 ms a checkpoint, the
  copy runs in the call), against 3.66 and 3.68 s of device copies in `v` (median 22.3 ms). The restores are 2.6 ms
  (median) in both.

### By prompt size

The turns' new tokens in the agent check's size bins: the sum of a bin's turns a pass (the mean of two passes) and the
median turn, in seconds ([bins.tsv](bins.tsv), [versus_bins.tsv](versus_bins.tsv); every request in
[requests/](requests/)).

| New tokens | Turns | `v` off sum | `i` off sum | Change | `v` off median | `i` off median | `v` on median | `i` on median |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1–64 | 18 | 21.01 | 18.84 | −2.17 | 1.11 | 1.00 | 1.11 | 0.99 |
| 65–255 | 8 | 20.26 | 19.27 | −0.99 | 2.44 | 2.32 | 1.82 | 1.70 |
| 256–511 | 12 | 65.12 | 63.71 | −1.41 | 5.57 | 5.45 | 3.63 | 3.53 |
| 512–1,023 | 10 | 95.72 | 69.25 | −26.47 | 9.39 | 6.71 | 6.98 | 6.22 |
| 1,024–1,999 | 11 | 152.24 | 112.12 | −40.12 | 13.83 | 8.71 | 13.04 | 12.43 |
| 2,000 and more | 3 | 59.23 | 51.84 | −7.39 | 17.05 | 17.06 | 17.02 | 16.99 |

- **The gain is where a prompt has the 512 tokens before its last checkpoint:** 66.59 of the 78.55 s are at 512 to
  1,999 new tokens. There the whole prompt now runs as one call through the pipeline. Shorter turns gain the drain
  before the 4-token tail and little else, because their other checkpoint sits at the start.
- **With the checkpoints inside the calls, the taper loses on turns of 1,024–1,999 new tokens:** 126.93 s a pass
  against 112.12 (median 12.43 against 8.71). It still gains from 65 to 1,023 (Q1235, R60). R43 saw the same without
  checkpoints: 11.38 against 9.14 s at 1,024–1,999.
- For scale, est. (another build): R43's load without checkpoints on v3.0 took 6.96 and 9.14 s (median) at 512–1,023 and
  1,024–1,999, beside `i`'s 6.71 and 8.71.

### Fresh 2,049-token prompts

Nothing cached, four prompts a load, each with the taper off and on ([fresh.tsv](fresh.tsv)), in seconds:

| Load | Taper off | Taper on |
|---|---|---|
| `v` | 14.38, 14.51, 14.33, 14.33 | 15.18, 15.19, 15.32, 15.44 |
| `i` | 12.27, 12.38, 12.41, 12.14 | 12.07, 12.23, 12.30, 12.24 |

- With the checkpoints inside the calls, a fresh 2k prompt is 2.03 s faster (medians 12.33 and 14.36 s). Its
  checkpoint at 1,533 tokens goes inside the call. The one at 2,045 does not: the 4 tokens after it would take the
  call to 2,049, past `-b 2048` (C736).
- With the checkpoints inside the calls, the taper no longer loses on a fresh prompt (R43's and `v`'s loss came with
  the drains).

### Memory

Both loads used the same card memory to the MiB, loaded (card 3 the fullest at 5,111 of 6,144 MiB) and after the load
(card 3 at 5,201). The host's lowest MemAvailable was 468 MiB (`v`) and 453 MiB (`i`); the watchdog's floor was 60 MiB.
Swap held 10 and 11 MiB after the loads.

## What goes into the preset

`--ctx-checkpoints-incall` beside `--ctx-checkpoints-vram 4` joins the v4 preset candidate (Q1254). It is a gain alone
and on top of the taper, with the same greedy replies, checkpoints and restores. It goes into the night's gains
together (R90, Q1322), which run on `pascal-p106-v4-b12246-c735` (`9e4bbe897`). That build also keeps the in-call
checkpoints with C703's catch-up overlap on; Q1327 times that pair on a P106. The taper's loss at 1,024–1,999 new
tokens on this line is for R60 (Q1235).

## Files

| File | What |
|---|---|
| [versus.tsv](versus.tsv) | per load and taper arm: prompt time a pass, against `v` off, the spread, the class, the greedy replies |
| [versus_bins.tsv](versus_bins.tsv) | the same by size bin, with each bin's share of the base's prompt time |
| [passes.tsv](passes.tsv) | per load, arm and pass: turns, tokens, seconds, tok/s, the checkpoints created and restored and their logged time |
| [bins.tsv](bins.tsv) | per load, arm and size bin: turns, the median turn, the sum per pass |
| [fresh.tsv](fresh.tsv) | every fresh 2,049-token prompt, with its checkpoints' logged time |
| [loads.tsv](loads.tsv) | per load: checkpoint lines (count, median and total time, bytes on the cards), lowest MemAvailable |
| [summary.md](summary.md) | the classes in words and `taper_replay.py`'s own summary per load |
| [ckpt_summary.tsv](ckpt_summary.tsv), [ckpt_summary.txt](ckpt_summary.txt) | the checkpoints and restores of `i` against `v` per request kind and taper arm; the log-wide counts and the warning lines |
| [ckpt_requests.tsv](ckpt_requests.tsv) | every request: the token counts of its checkpoints and restores in both loads, those inside a call |
| [requests/v.tsv](requests/v.tsv), [requests/i.tsv](requests/i.tsv) | every request of each load as `taper_replay.py` wrote it (the unmeasured prefix requests are `reset`) |
| [run/ckpt_compare.py](run/ckpt_compare.py) | the checkpoint and restore comparison from the two server logs |

The rig scripts are unchanged from R43 and C218:
- [r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh) and [lane_tables.py](../2026-11-10_prompt_lane/run/lane_tables.py);
- [rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (md5 `e82a80981c9e88e6a235cc476b993de8`);
- [taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`);
- [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`).

## Commands

A folder on the rig (`WORK`) with `r43_run.sh`, `rig_taper.sh`, `taper_replay.py`, `turns.csv` and the v3 preset's
`settings.env` ([release/flash-next-v3/settings.env](../../../release/flash-next-v3/settings.env)). It also needs the
night merge's rig tarball unpacked in `$BUILDS/$BUILD`, the model's first shard in `MODEL` (shards 2 and 3 beside it)
and the MTP draft in `DRAFT`. The two loads ran detached, one after the other, and each ended with `$OUT/DONE`. `v` ran
13:39 to 14:18 and `i` 14:18 to 14:53, and nothing else ran on the rig meanwhile.

```bash
BUILD=llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night
# v: the device stores, the switch off
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c217v LOADS=A \
  ARGS_ADD='--ctx-checkpoints-vram 4' setsid nohup bash $WORK/r43_run.sh > $WORK/c217v.out 2>&1 < /dev/null &"
# i: the same with the checkpoints inside the calls, once c217v/DONE exists
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c217i LOADS=A \
  ARGS_ADD='--ctx-checkpoints-vram 4 --ctx-checkpoints-incall' setsid nohup bash $WORK/r43_run.sh > $WORK/c217i.out 2>&1 < /dev/null &"
```

The server line of each load (`rig_taper.sh` adds everything up to `-c` and from `-np` on):

```
llama-server -m $MODEL -ngl 99 -fa on -c 65536 -b 2048 -ub 512 -ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio -md $DRAFT \
  --spec-type draft-mtp --spec-draft-n-max 3 -otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host \
  --ctx-checkpoints-vram 4 [--ctx-checkpoints-incall] \
  -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --no-webui -lv 5
# environment: GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 LLAMA_HOP_CAST=q8_0
#              GGML_CUDA_HOP_STAGED=1 LLAMA_DRAFT_SHORTLIST=32768,8192,2 GGML_CUDA_KEEPALIVE=decode; MIN_AVAIL_MB=60
```

The tables, on any machine with the two load folders (each with its `server.log`):

```bash
python3 -I ../2026-11-10_prompt_lane/run/lane_tables.py --base v --load v=c217v/load-A --load i=c217i/load-A --out .
python3 -I run/ckpt_compare.py --a v=c217v/load-A --b i=c217i/load-A --out .
```

The taper-on comparison (`i` on against `v` on, and each load's taper on against off) is the per-pass sums of
`requests/*.tsv`'s `prompt_ms` over the 62 turns: `v` on 344.0045 and 343.5999 s, `i` on 321.4031 and 321.0930 s,
`v` off 416.5568 and 410.6017 s, `i` off 338.9638 and 331.0906 s.

## Provenance and licence

The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next and the
MTP draft as the v3 release lists them. The prompts' text is the Python standard library's source on the rig (PSF
licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements.
