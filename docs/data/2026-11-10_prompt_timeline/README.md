# One long prompt's per-card timeline (R42, Q1156, 2026-11-10)

What this measured: how busy each of the nine cards is while the Miner Llama v3.0 release line reads one long
prompt, and where the host waits ([Q1156](../../../quants/experiments.md)). The line is the v3 preset exactly:
`pascal-p106-v3-b12076-ladder5` `1551f5fe1`, UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090
cards, the MTP draft at three tokens, the preset's environment and arguments. The prompts are fresh (nothing cached)
4,097-token and 451-token texts, each read with the micro-batch taper off (the preset) and on. The
[prefill note](../../research/notes/flash_next_prefill.md) (section 3) had put the cards at est. 25 to 40 % busy in a
long prompt, cause unknown; this is the per-card timeline it asked for.

## Result

**In the 4,097-token prompt, 1.42 of the nine cards are busy on average** (the sum of the nine cards' busy shares;
1.49 in the second fresh prompt of the same load), **1.94 with the taper. The 451-token prompt: 0.79, and 1.86 with
the taper.** The busiest card's GPU time in the 4,097-token prompt is 5,631.0 ms of the prompt's 29,983.0 ms. The
rest is the host waiting: for the cards to empty after each of the prompt's four calls (13,760.4 ms), and inside a
call's third queued micro-batch on cards 1 to 7 (and its fourth on card 7: 8,851.0 ms in 15 splits), most likely
because each card's CUDA launch queue is full.

**The decision rule** (R42: under half the cards busy in a long prompt → the multi-GPU fork's overlap commits, C663,
before the taper's agent arm, R59): the condition is met, so the overlap commits come first.

### Busy share per card

GPU time of the prompt's micro-batches on each card, and that time's share of the prompt's window (from the first
micro-batch on card 0 to the first evaluation after the last micro-batch). The window is from the graph-profile load,
which reads the prompt at the unprofiled speed; the GPU time per micro-batch is from the host-profile load of the
same prompt, which has every one of them (see "How it was measured"). The first fresh prompt of each load (requests
7 and 8).

| Card | 4,097 tokens, taper off: GPU ms | share | taper on: GPU ms | share | 451 tokens, taper off: GPU ms | share | taper on: GPU ms | share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5,522.6 | 18.42 % | 7,740.4 | 25.97 % | 609.4 | 10.47 % | 1,274.1 | 26.28 % |
| 1 | 5,631.0 | 18.78 % | 7,680.8 | 25.77 % | 595.8 | 10.23 % | 1,189.0 | 24.52 % |
| 2 | 5,106.7 | 17.03 % | 7,036.7 | 23.61 % | 556.6 | 9.56 % | 1,091.6 | 22.51 % |
| 3 | 5,425.8 | 18.10 % | 7,416.6 | 24.88 % | 567.3 | 9.74 % | 1,131.9 | 23.34 % |
| 4 | 4,302.7 | 14.35 % | 5,938.9 | 19.92 % | 472.4 | 8.11 % | 950.7 | 19.61 % |
| 5 | 5,014.7 | 16.73 % | 6,752.0 | 22.65 % | 547.4 | 9.40 % | 1,069.8 | 22.06 % |
| 6 | 4,461.2 | 14.88 % | 5,889.1 | 19.76 % | 458.1 | 7.87 % | 885.8 | 18.27 % |
| 7 | 5,053.8 | 16.86 % | 6,672.3 | 22.38 % | 542.0 | 9.31 % | 1,044.8 | 21.55 % |
| 8 | 1,928.3 | 6.43 % | 2,574.7 | 8.64 % | 239.7 | 4.12 % | 387.5 | 7.99 % |
| **cards busy on average** | | **1.416** | | **1.936** | | **0.788** | | **1.861** |
| micro-batches | 9 | | 27 | | 2 | | 8 | |
| window, ms | 29,983.0 | | 29,807.6 | | 5,823.2 | | 4,849.0 | |

The second fresh prompt of each load (requests 9 and 10): 4,097 tokens 1.486 cards with the taper off (window
28,618.6 ms) and 1.937 with it (29,699.8 ms); 451 tokens 1.854 with the taper (4,862.8 ms). The 451-token prompt
without the taper has no window in its graph-profile load: its one large micro-batch was the last evaluation of its
graph, which that profile never writes. Card 8 holds 3 of the layer split's 49 shares (`-ts 6,6,6,6,5,6,5,6,3`), so
its share is lowest in every prompt.

With the taper the cards are busier but the 4,097-token prompt is not faster (29,939.12 and 29,906.29 ms against
30,420.33 and 28,806.65 ms in the graph-profile load): its 64-token micro-batches cost more GPU time per token (card
0: 7,740.4 against 5,522.6 ms for the same prompt).

### Where the prompt's time goes

The graph-profile load, the same requests, ms. The parts add up to the window. "Blocked inside a split" counts the
host's time before a split on cards 1 to 8 when it is over 20 ms; the 2 to 10 ms of an unblocked split (its input
copies and launches) are "other splits".

| Part | 4,097, taper off | 4,097, taper on | 451, taper off | 451, taper on |
|---|---:|---:|---:|---:|
| the host waits for the cards after a call (from its last queued split to the next evaluation) | 13,760.4 | 5,672.3 | 4,815.5 | 1,310.1 |
| the host blocked inside a split | 8,851.0 (15 splits) | 15,483.3 (59) | 0.0 | 2,195.7 (16) |
| the draft's catch-up over the call's rows (cards 4 and 8 only) | 2,944.1 | 2,944.0 | 288.2 | 288.1 |
| after the catch-up until the next call (checkpoint saves, then the next call's first micro-batch) | 2,671.7 | 1,650.2 | 667.1 | 662.4 |
| of it, the checkpoint saves (the server's log) | 1,352.9 | 1,335.0 | 632.3 | 632.8 |
| the host before card 0 inside a call (graph build, the CPU split, inputs) | 1,379.0 | 3,245.4 | 0.0 | 202.8 |
| other splits | 376.8 | 812.4 | 52.4 | 189.8 |
| **window** | **29,983.0** | **29,807.6** | **5,823.2** | **4,849.0** |

Before card 0 the host spends 231.8, 243.7, 237.0, 383.2 and 283.3 ms on the 512-token micro-batches inside a call
(median 243.7), and 32.5 to 34.7 ms on each 64-token one (median 34.3): about half a ms per token, so most of it
scales with the tokens (the CPU split gathers each token's embedding and n-gram rows) rather than with the graph.

**The 4,097-token prompt, taper off, step by step** (ms from the first micro-batch; graph-profile load, request 7).
`-b 2048` and the two checkpoints (at 4 + 512 and 4 tokens before the end) cut it into calls of 2,048, 1,533, 512
and 4 tokens. After each call the MTP draft reads the call's hidden rows in passes of 128 (`--spec-draft-n-ubatch
128`) on cards 4 and 8, so the host first waits for the last card to finish the call.

| Step | Starts | Takes |
|---|---:|---:|
| call 1: 2,048 tokens, 4 micro-batches queued (the third blocks on cards 1 to 7) | 0.0 | 5,515.8 |
| the host waits for the cards | 5,515.8 | 5,256.4 |
| the draft's catch-up over 2,048 rows (16 passes) | 10,772.2 | 1,433.9 |
| graph build, CPU split | 12,206.1 | 588.2 |
| call 2: 1,533 tokens, 3 micro-batches (the third blocks on cards 1 to 7) | 12,794.3 | 4,998.2 |
| the host waits for the cards | 17,792.5 | 2,395.6 |
| the draft's catch-up over 1,533 rows (12 passes) | 20,188.1 | 1,155.9 |
| the checkpoint at 3,581 tokens (636.0 ms), graph build, CPU split | 21,344.0 | 1,316.4 |
| call 3: 512 tokens, 1 micro-batch | 22,660.5 | 71.1 |
| the micro-batch passes the nine cards one after another (5,700.8 ms of GPU time) | 22,731.6 | 5,988.7 |
| the draft's catch-up over 512 rows (4 passes) | 28,720.2 | 354.3 |
| the checkpoint at 4,093 tokens (716.9 ms), graph build, CPU split | 29,074.5 | 767.1 |
| call 4: 4 tokens, 1 micro-batch, queued and through the nine cards | 29,841.5 | 141.4 |
| the prompt's output is on the host | 29,983.0 | |

### The host's time before each split

ms from the previous evaluation (any card) until the host had queued this split; graph-profile load. Micro-batch
0's card 0 value lies before the window. The 4,097-token prompt, taper off (request 7):

| micro-batch | card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 444.1 | 4.7 | 4.6 | 4.8 | 4.0 | 4.5 | 4.3 | 4.6 | 3.2 |
| 1 | 231.8 | 6.6 | 6.7 | 6.6 | 6.0 | 6.7 | 6.2 | 6.4 | 4.9 |
| 2 | 243.7 | 589.0 | 570.2 | 728.3 | 257.1 | 819.7 | 429.2 | 736.8 | 8.0 |
| 3 | 237.0 | 7.2 | 8.0 | 6.9 | 6.1 | 10.1 | 8.0 | 527.2 | 6.5 |
| 4 | 588.2 | 7.5 | 7.3 | 7.5 | 6.7 | 7.4 | 7.1 | 7.3 | 5.9 |
| 5 | 383.2 | 9.5 | 9.4 | 9.5 | 8.8 | 9.7 | 9.1 | 9.4 | 7.7 |
| 6 | 283.3 | 471.6 | 562.9 | 818.8 | 264.1 | 845.8 | 498.5 | 731.8 | 8.3 |
| 7 | 1,316.4 | 9.3 | 9.1 | 9.4 | 8.4 | 9.2 | 8.9 | 9.2 | 7.6 |
| 8 | 767.0 | 3.0 | 3.0 | 3.0 | 2.5 | 2.8 | 2.7 | 2.8 | 1.9 |

The 451-token prompt, taper on (request 8: seven micro-batches of 64 tokens and 63, then the last 4):

| micro-batch | card 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 68.0 | 4.0 | 3.8 | 3.9 | 3.2 | 3.7 | 3.4 | 3.7 | 1.9 |
| 1 | 34.6 | 5.2 | 5.1 | 5.2 | 4.5 | 5.1 | 4.8 | 5.1 | 5.5 |
| 2 | 34.7 | 153.2 | 140.1 | 175.2 | 61.4 | 208.1 | 106.3 | 181.5 | 3.2 |
| 3 | 32.5 | 4.0 | 3.8 | 4.0 | 3.2 | 5.1 | 4.9 | 107.5 | 3.2 |
| 4 | 34.1 | 5.2 | 5.1 | 5.6 | 4.6 | 5.1 | 4.9 | 117.5 | 3.1 |
| 5 | 34.3 | 5.6 | 87.4 | 186.2 | 62.1 | 208.3 | 114.3 | 183.1 | 3.2 |
| 6 | 32.6 | 3.9 | 3.7 | 4.0 | 3.1 | 5.1 | 4.9 | 103.5 | 3.2 |
| 7 | 662.4 | 2.8 | 2.6 | 2.8 | 2.2 | 2.6 | 2.5 | 2.7 | 1.7 |

**Why the third micro-batch blocks.** The host queues each of a call's first two micro-batches on all nine cards
soon after its card 0 (34.7 and 50.2 ms from card 0 to card 8 in call 1, 56.8 and 73.2 in call 2), before card 1 has
started on the first; on the third it stops at every card from 1
to 7 until that card is part way through its first micro-batch, whatever the micro-batch's size (512 or 64 tokens).
Every main-model evaluation of the prompts ran without a CUDA graph ("eager" in the graph profile), so each split is
its launch groups launched one by one (327 on card 0, 239 to 304 on cards 1 to 7, 109 on card 8 for a 512-token
micro-batch), and a card's queue holds about two splits of them. The pattern is that of a CUDA launch queue of fixed
length (est.; the fork's profiles do not see the queue itself): card 8, with the fewest layers, never blocks, and card
4 (5 layers and the draft's embeddings) blocks for less (257.1 and 264.1 ms) than its neighbours. When the queue has
room, a split's launches take the host a median 3.21 ms on cards 0 to 7 and 0.88 ms on card 8 (the host profile's
entry to exit in its fresh 4,097-token prompts). Blocked, the host cannot queue the next micro-batch on card 0, so
card 0 idles from the end of its third micro-batch until the host is through card 7 (in call 1, est. from 2.0 s to
4.9 s). The multi-GPU fork names a full launch queue as one of its three causes; `CUDA_SCALE_LAUNCH_QUEUES` (CUDA's
launch queue size) and CUDA graphs for prompt micro-batches are the two levers, now Q1239 and Q1240.

### Graph rebuilds and reallocations

| What | Per request (every fresh prompt, every load) |
|---|---|
| main model: graphs reused (the server's count) | 4 |
| main model: compute-buffer reallocations the build logs (same size / changed size) | 0 / 0 |
| draft: compute-buffer reallocations ("graph reallocation" lines; changed size among them) | 34 (4) at 4,097 tokens, 7 (4) at 451 |
| main model's prompt evaluations: CUDA graph replays | 0 (every written one eager; the whole graph-profile load at 4,097 tokens: 494 eager, 36 captures, 9 replays, all captures and replays in generation) |

The build logs a compute-buffer reallocation (with its time) unless it follows a change of backend assignments, and
graph reuse only as the count per request, so a rebuild per micro-batch is not visible in the log. The main model
logs no reallocation at 4,097 tokens: the multi-GPU fork's first cause (reallocations that synchronize every backend
as the context grows) does not show here. The draft reallocates its compute buffer on about every catch-up pass (33
passes in the 4,097-token prompt). Per-micro-batch rebuilds need C663's trace commit `1567041ef` ported (Q1238).

### What it changes

- **C663's port list, ranked by these numbers** (the 4,097-token prompt, taper off): the waits for the cards after
  each call (13,760.4 ms, 45.9 % of the window; the 512-token call between the two checkpoints alone 5,988.7 ms,
  since one micro-batch cannot overlap), the blocked splits (8,851.0 ms, 29.5 %), the draft's catch-up (2,944.1 ms,
  9.8 %), the host before card 0 (1,379.0 ms, 4.6 %) and the checkpoint saves (1,352.9 ms, 4.5 %). A fix for the
  reallocations is not needed for the main model at this length.
- **Why R43's drains cost more than their saves** (30.72 % of prompt time in R43, against 1,352.9 ms of saves here):
  the two checkpoints also cut the prompt into calls, and each call ends in a wait for the cards and a catch-up of
  the draft; the call of 512 tokens between them runs on one card at a time.
- **New rows:** Q1238 (the per-micro-batch trace), Q1239 (`CUDA_SCALE_LAUNCH_QUEUES=2x` on the prompt), Q1240 (CUDA
  graphs for prompt micro-batches), Q1241 (`-b 4096`: one call and one wait fewer per 2,048 tokens), Q1242 (the
  draft's catch-up during the next call's fill).

## How it was measured

Five loads of one server each, in this order, on the idle rig (15:48:57 to 16:29:12): `warm4096` (unprofiled: the
warm-up load and the unprofiled reference), `graph4096`, `graph450` (`GGML_CUDA_PROFILE=<file>
GGML_CUDA_PROFILE_MODE=graph`), `host4096`, `host450` (`GGML_CUDA_PROFILE_HOST=<file>`). Each load is one call of
R43's wrapper ([2026-11-10_prompt_lane/run/r43_run.sh](../2026-11-10_prompt_lane/run/r43_run.sh), with
[rig_taper.sh and taper_replay.py](../2026-11-04_prompt_taper/run/)): the preset's environment and line with
`--ctx-checkpoints 4`, `-lv 5`, the replay of one agent-check task with the taper off (`TASKS=bugfix-javascript-triangle
PASSES_A=off`, 5 turns and its reset), then `FRESH=2` fresh prompts of `FRESH_N` tokens, each with the taper off and
on (requests 7 to 10: fresh0 off, fresh0 on, fresh1 on, fresh1 off).

The two profiles are the fork's (`ggml/src/ggml-cuda/ggml-cuda.cu`), and each misses something:
- **Graph mode** puts an event pair around each graph evaluation and nothing waits. It reads an evaluation when the
  same graph is next evaluated on its card, so an evaluation still running then has no GPU time ("not ready": 424 of
  the 1,463 lines of `graph4096`, nearly all prompt micro-batches queued ahead), and the last evaluation of each graph
  is never written (Q826). Its host clock (when the host had queued each evaluation) is complete apart from those
  last evaluations, and it does not slow the prompt: 30,420.33 and 28,806.65 ms with the taper off against 30,213.68
  and 28,981.42 unprofiled, 29,939.12 and 29,906.29 with it against 28,911.51 and 29,827.05.
- **The host profile** has every evaluation's GPU time, but before an evaluation it waits for the card's previous
  one to end, so the host never queues ahead and the prompt slows: 35,821.69 and 35,463.35 ms with the taper off,
  39,471.63 and 39,378.47 with it (451 tokens: 6,378.40 and 5,961.19 off, 6,457.05 and 6,536.32 on, against
  6,207.82 and 5,968.14, 4,904.43 and 4,933.81 under the graph profile).

So the busy share takes each micro-batch's GPU time from the host-profile load and the window from the graph-profile
load, request by request (`--pair`). The two agree where both have a GPU time: in request 7 at 4,097 tokens the
graph profile's sum over a card's micro-batches is between 0.9894 and 0.9987 of the host profile's on each of the
nine cards. The trace's clock is set to the
server log's by the draft's "graph reallocation" lines, each logged within a ms of a draft evaluation (149 of 162 lines
matched in `graph4096`, 155 of 162 in `host4096`).

## Files

| File | What |
|---|---|
| [busy.tsv](busy.tsv) | per load, request and card: micro-batches, GPU time, window, busy share; `gpu_from` names the load the GPU time is from (the paired rows have `graph_over_host_gpu`) |
| [waits.tsv](waits.tsv) | per load and request: the parts of the window above, the host's time before card 0, the blocked splits |
| [phases.tsv](phases.tsv) | per request: the runs of main micro-batches and of draft evaluations, with their times |
| [splits.tsv](splits.tsv) | every split of every prompt: micro-batch, card, time from the window's start, the host's time before it, GPU time, CUDA graph use |
| [requests.tsv](requests.tsv) | every request of every load: prompt time, the server's calls, checkpoint saves, graphs reused, reallocations, the taper's plan, micro-batches, window |
| [counts.tsv](counts.tsv) | every evaluation of each profiled load by model and CUDA graph use |
| [summary.txt](summary.txt) | the table script's alignment and pairing lines |
| [run/r42_run.sh](run/r42_run.sh) | the five loads, on the rig |
| [run/timeline.py](run/timeline.py) | the tables, from the five loads' folders |

## Commands

On the rig, detached, from a folder `$R43` that holds R43's `r43_run.sh`, `rig_taper.sh`, `taper_replay.py`, the
agent check's `turns.csv` ([2026-11-03_retype_agent/agent/turns.csv](../2026-11-03_retype_agent/agent/turns.csv)) and
the preset's `settings.env` ([release/flash-next-v3](../../../release/flash-next-v3/)):

```bash
ssh $RIG "cd $WORK && R43=$R43 BUILDS=$BUILDS BUILD=llama.cpp-20261105-1551f5fe1-pascal-p106-v3-b12076-ladder5 \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/out \
  setsid nohup bash $WORK/r42_run.sh > $WORK/r42_run.out 2>&1 < /dev/null &"
```

The tables, on any machine with Python 3 (seconds):

```bash
python3 -I timeline.py --load warm4096=out/warm4096 --load graph4096=out/graph4096 --load graph450=out/graph450 \
  --load host4096=out/host4096 --load host450=out/host450 --pair graph4096=host4096 --pair graph450=host450 --out tables
```

## Provenance and licence

Measured on one rig with nine P106-090 cards (`docs/mining_rig.md`), lowest MemAvailable on the 4 GB host 173, 113,
517, 70 and 506 MiB in the five loads. The prompts are Python standard-library source (PSF licence) tokenized on the
rig; no text is in these files, only counts and times. Tables and scripts are this repository's licence.
