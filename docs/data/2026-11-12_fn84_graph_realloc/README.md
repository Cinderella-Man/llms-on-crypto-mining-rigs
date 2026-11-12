# What reallocates the prompt graph on v4's `gains` line (FN84, Q1405, 2026-11-12)

What this measured: which switch makes v4's `gains` line reallocate the main model's compute graph on every request.
R60 found it ([2026-11-12_r60_taper_v4](../2026-11-12_r60_taper_v4/README.md#the-prompt-graph-is-reallocated-on-this-line)):
373 of 373 requests on that line, against 0 of 600 on R85's line. A reallocation makes the scheduler wait for all nine
cards before it re-plans the buffers, so the nine-card pipeline empties partway through a prompt. That line read turns
of 1,024 to 1,999 new tokens in 192.99 s a pass against R85's 112.12. Registry row [Q1405](../../../quants/experiments.md).

This run counts reallocations; it is not a timing run. The build, model and line are R60's load `a`:
`pascal-p106-v4` `d097d6d88` (the `-kern` rig tarball), Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 with its MTP draft on
nine P106-090 cards, the v3 preset plus R90's `gains` switches and flags. It replays one task of R43's replay with the
taper off, plus one fresh 2,049-token prompt, with the server at `-lv 5`.

## Result

**The cause is `LLAMA_SAMPLING_TOP_K_ROWS=1` (C691, one top-k over the output rows).** Without it, and with everything
else on R90's line unchanged, the main model's graph reallocates 0 times in 8 requests. In the same 6 requests, R60's
load had 26 reallocations: 15 at an unchanged graph size and 11 at a changed one. The greedy replies and the context
checkpoints of those 6 requests are R60's, request for request.

| Request | New tokens | Reallocations, R60 (same / changed) | Reallocations, without the switch | Reply as R60's | Checkpoints as R60's |
|---|---:|---:|---:|---|---|
| prefix (unmeasured) | 1,549 | 3 / 1 | 0 / 0 | yes | yes |
| turn 1 | 896 | 2 / 2 | 0 / 0 | yes | yes |
| turn 2 | 3,313 | 7 / 2 | 0 / 0 | yes | yes |
| turn 3 | 302 | 1 / 2 | 0 / 0 | yes | yes |
| turn 4 | 30 | 1 / 2 | 0 / 0 | yes | yes |
| turn 5 | 92 | 1 / 2 | 0 / 0 | yes | yes |
| fresh prompt, taper off | 2,049 | — | 0 / 0 | (same reply with the taper on) | — |
| fresh prompt, taper on | 2,049 | — | 0 / 0 | | |

**Why this switch.** With the switch on, the sampling part of the graph pads the logits by one row only when the
micro-batch has no outputs
([`src/llama-graph.cpp` line 5431 at `d097d6d88`](https://github.com/Cinderella-Man/miner-llama/blob/d097d6d88a3ee3fb2e3dbdb90819d320557d9809/src/llama-graph.cpp#L5431)).
Without the switch, the pad is always there. So with the switch on, a prompt's middle micro-batches build a graph of
6,944 nodes, and the micro-batches with outputs (a prompt's last one, every verify pass) build 6,943. Each change in
the node count makes ggml's allocator re-plan the graph's buffers, after it has synchronized every card. The new plan
fits only that graph's sizes, so the next larger micro-batch of the same node count does not fit it either. That is
why most reallocations happen at an unchanged size (R60's server log lists the node count of each one, in
[requests_reallocs.tsv](requests_reallocs.tsv)'s `target_reallocs`: `6944c` is a changed size). The switch gains on
decoding because it drops that pad from verify passes (R83: −0.72 ms a step, −0.6 %,
[2026-11-11_r83_c691_p106](../2026-11-11_r83_c691_p106/README.md)).

The code showed which switch changes the node count, so the first load dropped that switch. Its count fell to 0 and
its replies and checkpoints matched R60's, so no other load ran. The other suspects (`LLAMA_QSA_KPOOL_DEV`,
`LLAMA_QSA_INDEXER_ROWS`) stayed on in this load, and their graphs reallocate nothing.

**Prompt times, one request each (a diagnostic; this load was not run for timing).** The 3,313-token turn took
21.14 s against R60's 42.49 s for the same text on the same rig and build. Turn 1 took 11.53 s against 11.81, and the
1,549-token prefix 22.24 s against 23.42. The fresh 2,049-token prompt took 11.72 s with the taper off and 11.42 s
with it on. R60's four fresh prompts on the full line took 23.36, 22.94, 23.32 and 22.97 s with the taper off (other
text of the same length), and R85's took 12.27, 12.38, 12.41 and 12.14.

**The draft's graph reallocates too, on every line.** The MTP draft's own graph (213 nodes) reallocates 2 to 3 times
a request at an unchanged size: 25 times in this load's 8 requests, 617 in R60's 373 and 964 in each of R85's two
loads of 300. That is older than v4's switches and separate from this finding (Q1407).

What follows:
- **Q1406 (C802):** C691 with a graph whose node count stays the same in every micro-batch, so the line keeps its
  decoding gain without the reallocations. Its prompt time is then measured with R43's replay on a P106.
- **Q1407 (C803):** where and how much the draft's reallocations cost.
- **Q1408 (C804):** every switch's own rig check counts the graph's reallocations per request. C691's check (R83)
  measured verify passes only, where every graph has outputs, so the cost showed up two runs later.

## Files

| File | What |
|---|---|
| [compare.tsv](compare.tsv) | per request: new tokens, prompt ms, reallocations (same / changed) in R60's load `a` and here, the node counts of R60's reallocations, the reply's md5 and the checkpoints' md5 in both |
| [requests_reallocs.tsv](requests_reallocs.tsv) | `run/fn84_requests.py`'s output for R60's first 6 requests (`r60a`) and this load (`topk`): reallocations per request, the target graph's node count at each, the draft's reallocations, the checkpoints |
| [reallocs.tsv](reallocs.tsv) | this load's reallocations per size bucket (R60's [`run/realloc_counts.py`](../2026-11-12_r60_taper_v4/run/realloc_counts.py), the form of R60's [reallocs.tsv](../2026-11-12_r60_taper_v4/reallocs.tsv)) |
| [requests.tsv](requests.tsv) | every request of this load as `taper_replay.py` wrote it (R60's are [requests/a.tsv](../2026-11-12_r60_taper_v4/requests/a.tsv), pass 0's first 6 rows) |
| [run/fn84_loads.sh](run/fn84_loads.sh) | starts one load on the rig: R60's line, one task, taper off, one fresh prompt; `DROP` removes switches or flags |
| [run/fn84_requests.py](run/fn84_requests.py) | the reallocations, their node counts and the checkpoints per request, from a server log |

The rig scripts are R43's and C218's, unchanged (md5s in the load's `md5.txt`, the same as R60's):
[r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh),
[rig_taper.sh](../2026-11-04_prompt_taper/run/rig_taper.sh) (`e82a80981c9e88e6a235cc476b993de8`),
[taper_replay.py](../2026-11-04_prompt_taper/run/taper_replay.py) (`329df66616395a0162065530a0f6108e`),
the agent check's [turns.csv](../2026-11-03_retype_agent/agent/turns.csv) (`a3c7dc91660a507cd415694d47bf7ced`) and
the v3 preset's [settings.env](../../../release/flash-next-v3/settings.env).

## Commands

As R60: a folder on the rig (`WORK`) holds those five files, the `-kern` tarball is unpacked in `$BUILDS/$BUILD`
(`llama-server` md5 `9bf23c162646c9d1356d0beafcce0afa`), the model's first shard is `MODEL` and the MTP draft `DRAFT`.
The load ran detached from 11:50:07 to 11:57:08 (up after 314 s) and ended with `$WORK/fn84_topk/DONE`. Nothing
else ran on the rig meanwhile.

```bash
BUILD=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern
RIG=$RIG WORK=$WORK BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT \
  NAME=topk DROP=LLAMA_SAMPLING_TOP_K_ROWS bash run/fn84_loads.sh
```

The server line is R60's (its README, "Commands"), with the taper off per request and the environment without
`LLAMA_SAMPLING_TOP_K_ROWS=1`. The server's `switches:` line lists 13 set (R60's load `a`: 14). The replay ran
`taper_replay.py --passes off --tasks bugfix-javascript-complex-numbers --fresh 1 --fresh-n 2048 --gen 8`. With one
task, the task's text is R60's pass 0's (it is the replay's first task), so its replies compare with R60's.

The tables, from the two loads' server logs and request files:

```bash
python3 -I run/fn84_requests.py r60a=<R60 load a>/load-A/server.log:6 topk=<this load>/load-A/server.log \
  > requests_reallocs.tsv
python3 -I ../2026-11-12_r60_taper_v4/run/realloc_counts.py topk=<this load>/load-A/server.log > reallocs.tsv
cp <this load>/load-A/requests.tsv requests.tsv
# compare.tsv joins requests_reallocs.tsv with both loads' requests.tsv (gen_md5) by request number
```

## Provenance and licence

Measured on one Manli M-P106L9-N6G mining rig (nine P106-090 cards, 4 GB host) on 2026-11-12; the tables were made on
a build machine's CPU. The engine is llama.cpp's licence (MIT); the model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of
Qwen3.8-Flash-Next and the MTP draft as the v3 release lists them. The prompts' text is the Python standard library's
source on the rig (PSF licence), tokenized there and not stored; the request sizes are the agent check's, from
[turns.csv](../2026-11-03_retype_agent/agent/turns.csv). The tables are this run's own measurements, and R60's server
log of load `a` for the reference.
