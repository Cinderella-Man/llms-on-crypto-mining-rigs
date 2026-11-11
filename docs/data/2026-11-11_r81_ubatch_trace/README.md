# R81 (Q1295): the micro-batch trace on a P106, one long prompt step by step (2026-11-11)

What this measured: C701's micro-batch trace (`LLAMA_UBATCH_TRACE=1`, [the lines](../2026-11-11_c701_microbatch_trace/README.md#the-lines))
in [R42's driver](../2026-11-10_prompt_timeline/README.md) on one P106 rig, to explain three things R42 left open about
a fresh 4,097-token prompt: what the host does before card 0 in each micro-batch, why the host stops inside each call's
third micro-batch on cards 1 to 7, and why the MTP draft plans its compute buffers again 34 times a prompt. The
registry row is Q1295 in [quants/experiments.md](../../../quants/experiments.md). It is a diagnostic: the trace adds a
log line per evaluation, and the build is not R42's, so the prompt's speed is not compared with R42's.

The build is Miner Llama's v4 night merge, `pascal-p106-v4-b12246-night` at `c408dabcb`
([2026-11-11_v4_night_merge](../2026-11-11_v4_night_merge/README.md)), which holds C701's trace. No other switch it
adds was set, so it computes what `pascal-p106-v4` computes. The model and line are the v3 preset's: UkisAI's
Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next on nine P106-090 cards, `-b 2048 -ub 512`, the MTP draft at three tokens
with its token rows on card 4 (`-otd token_embd=CUDA4`) and catch-up passes of 128 rows. One load, R42's `warm4096`:
the replay of one agent-check task (its reset and five turns, eight tokens each), then two fresh 4,097-token prompts,
each with the micro-batch taper off and on. Request 7 is the first fresh prompt with the taper off (R42's request 7);
request 10 is the second prompt with the taper off, read for the second time (request 9 read it with the taper on).

## Result

- **Before card 0, a 512-token micro-batch of a fresh prompt holds the host for a median 291.3 ms (8 micro-batches of
  request 7), and two thirds of it is the host setting the graph's inputs:** a median 198.3 ms in `inputs`, then
  card 0's input copies (59.9 ms), the CPU split (22.1 ms), the graph's build (4.7 ms) and allocation (6.4 ms). When
  the same prompt is read a second time, `inputs` falls to a median 11.4 ms (request 10) and the copies stay at 60.0 ms.
  The eight requests that read their prompt for the first time spend 308.7 to 648.4 µs a token in `inputs` (each in
  the table below), the two that read it again 24.4 and 22.3. Est.: it is the n-gram table's reads from the SSD,
  which the n-gram input issues as `madvise(MADV_WILLNEED)` on each token's rows when it sets the input. The CPU
  split, which gathers the rows, takes 22.1 ms.
- **The third micro-batch of calls 1 and 2 blocks in `l`, the launch, on cards 1 to 7:** 15 splits hold the host
  8,872.2 ms in `l` and 0.257 ms in `w` (request 10: 9,351.8 ms and 0.287), the same 15 splits R42 found blocked.
  The host waits in the launch itself, not for an event or a card: the pattern of a full CUDA launch queue, so prompt
  CUDA graphs (C702, R82) and a larger queue (R62) are the levers.
- **The MTP draft plans again 34 times in each fresh 4,097-token prompt, in all four.** 30 times the allocator's plan
  is too small for a tensor sized by the draft's cache length, which grows 128 cells a catch-up pass: a host input of
  one element per four cells (`leaf_1`, 17 times with `leaf_8` once) and a FILL of one F16 per cell on card 8
  (`node_69`, 13 times). The other 4 follow the drafting steps: the step's hidden row is on the host and the
  catch-up's on card 8, so the backends change at `mtp_h_input` and change back. The drains before the re-plans sum
  to 1,491.8 ms in request 7, but the draft's scheduler has one input copy, and a pass without a re-plan waits as long
  in card 8's `w` (71.9, 75.7 and 70.6 ms in the three such passes). What the re-plans add is their planning, 6.2 ms a
  prompt.
- **The main model never plans again:** none of the load's 186 re-plans is the main model's, in prompts, replay turns
  or generation. C717 (the re-plan without a drain) stays parked.
- **In a verify pass, R86's 5.610 ms before card 0's launch is mostly the CPU split, not card 0's input copies.** In
  the ten four-token verify passes of requests 2 to 7 (the agent turns and the first fresh prompt) the evaluation's
  time to card 0's launch is a median 5.269 ms: the CPU split (the token embedding's and n-gram table's rows)
  4.379 ms, card 0's input copies 0.857 ms, its wait 0.029 ms. In the three of requests 8 to 10 the CPU split takes 0.076, 0.077 and 0.076 ms. Est.: the split
  reads the drafted tokens' n-gram rows from the SSD unless they are cached. Prefetching each drafted token's rows
  as it is drafted (C615) acts on the 4.379 ms; the k-pool rows (C100, C720, C719) act on the copies, which are small
  at this depth.

### Request 7's micro-batches, before card 0

ms; `card0` is from the micro-batch's start to card 0's launch and holds the columns before it. Every main-model
micro-batch is a new graph: the first of a call and the last call because none was allocated (`not_allocated`),
call 3 because its size differs from the micro-batch before (`params`), the rest because the previous micro-batch is
still on the cards (`in_flight`, as Miner Llama does under pipeline parallelism).

| Call, micro-batch | Tokens | Why rebuilt | pre | build | alloc | inputs | CPU split | card 0 wait | card 0 copies | **card0** |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1, 1 | 512 | not_allocated | 0.6 | 4.7 | 6.4 | 212.6 | 23.5 | 0.0 | 57.0 | **304.8** |
| 1, 2 | 512 | in_flight | 0.4 | 4.9 | 6.4 | 197.5 | 15.7 | 0.0 | 59.4 | **284.4** |
| 1, 3 | 512 | in_flight | 0.4 | 4.6 | 6.6 | 196.6 | 15.9 | 0.0 | 60.0 | **284.3** |
| 1, 4 | 512 | in_flight | 0.4 | 4.8 | 6.4 | 79.6 | 20.6 | 0.0 | 59.2 | **171.1** |
| 2, 1 | 512 | in_flight | 0.7 | 4.6 | 13.2 | 254.7 | 26.8 | 0.0 | 59.8 | **359.9** |
| 2, 2 | 512 | in_flight | 0.6 | 4.7 | 6.5 | 199.2 | 25.0 | 0.0 | 62.2 | **298.2** |
| 2, 3 | 509 | in_flight | 0.5 | 4.8 | 6.4 | 114.7 | 17.2 | 0.0 | 62.2 | **205.9** |
| 3, 1 | 512 | params | 0.9 | 4.4 | 6.4 | 204.3 | 24.0 | 0.0 | 61.6 | **301.7** |
| 4, 1 | 4 | not_allocated | 0.6 | 4.7 | 6.5 | 0.8 | 5.3 | 0.0 | 0.9 | **18.8** |

The eight micro-batches of 509 and 512 tokens: `card0` 2,210.3 ms, of it `inputs` 1,459.1, card 0's copies 481.5, the
CPU split 168.8, `alloc` 58.4, `build` 37.5. Request 10, the same arm on a prompt read before: `inputs` 11.1, 10.3,
11.6, 12.7, 12.6, 13.8, 10.7 and 8.6 ms, card 0's copies a median 60.0 ms, `card0` 849.5 ms in all.

Card 0 takes 19 inputs, the other cards 15; at the link's ~186 MB/s (R64) its 57.0 to 62.2 ms of copies are est.
10.6 to 11.6 MB a micro-batch, about what 512 tokens' F32 embedding rows (5.2 MB) and n-gram rows (~10 KB a token,
Q447) make. Cards 1 to 8 copy theirs in 1.7 to 9.0 ms.

**`inputs` per request**, the main model's micro-batches of more than 8 tokens ([read07/prompt_inputs.tsv](read07/prompt_inputs.tsv)):

| Request | What | Tokens | `inputs`, ms | µs a token |
|---:|---|---:|---:|---:|
| 1 | the replay's reset | 1,545 | 722.5 | 467.6 |
| 2 | turn 1 | 426 | 268.2 | 629.7 |
| 3 | turn 2 | 1,355 | 878.5 | 648.4 |
| 4 | turn 3 | 254 | 137.3 | 540.4 |
| 5 | turn 4 | 27 | 8.3 | 308.7 |
| 6 | turn 5 | 67 | 33.9 | 505.5 |
| 7 | fresh prompt 0, taper off | 4,093 | 1,459.1 | 356.5 |
| 8 | fresh prompt 0 again, taper on | 4,093 | 99.9 | 24.4 |
| 9 | fresh prompt 1, taper on | 4,093 | 1,278.2 | 312.3 |
| 10 | fresh prompt 1 again, taper off | 4,093 | 91.4 | 22.3 |

### The blocked micro-batches

`l` of each card's split, ms, request 7 ([read07/prompt_splits.tsv](read07/prompt_splits.tsv)). `w` is 15 to 19 µs in
each of these splits; an unblocked split's `l` on cards 1 to 7 is 2.3 to 3.1 ms.

| Call, micro-batch | card 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1, 3 | 455.5 | 566.9 | 718.4 | 253.9 | 812.1 | 425.3 | 733.2 | 1.3 |
| 1, 4 | 3.0 | 2.9 | 3.0 | 2.4 | 3.0 | 2.5 | 615.5 | 1.2 |
| 2, 3 | 592.7 | 556.8 | 816.1 | 257.8 | 847.9 | 490.9 | 729.4 | 1.3 |

### The draft's re-plans

Per request ([read07/replan_counts.tsv](read07/replan_counts.tsv), every line in [read07/replans.tsv](read07/replans.tsv)):

| Request | Re-plans | Allocator: host input (`leaf_1`, `leaf_8`) | Allocator: card 8's FILL (`node_69`) | Backends changed (`mtp_h_input`) | Drains, ms | Plans, ms |
|---:|---:|---:|---:|---:|---:|---:|
| 1 (reset) | 12 | 7 | 4 | 1 | 498.8 | 2.2 |
| 2 | 9 | 1 | 0 | 8 | 71.5 | 1.6 |
| 3 | 13 | 7 | 0 | 6 | 324.8 | 2.4 |
| 4 | 6 | 2 | 0 | 4 | 71.9 | 1.1 |
| 5 | 5 | 1 | 0 | 4 | 4.7 | 0.9 |
| 6 | 5 | 1 | 0 | 4 | 2.3 | 1.0 |
| 7 | 34 | 17 | 13 | 4 | 1,491.8 | 6.2 |
| 8 | 34 | 17 | 13 | 4 | 1,489.8 | 6.3 |
| 9 | 34 | 17 | 13 | 4 | 1,489.1 | 6.3 |
| 10 | 34 | 17 | 13 | 4 | 1,493.8 | 6.3 |

In request 7 the triggers alternate pass by pass: `leaf_1 ... ne = [192] needs 768 bytes in CUDA_Host, planned 512`,
then `node_69 (FILL) ne = [768] needs 1536 bytes in CUDA8, planned 1024`, up to `ne = [1088]` and `[3584]`. Each
re-plan sizes the buffers to the pass in hand, so the next pass outgrows them. A plan made once for the full context
(C712's first option) removes the 30; the drafting step's hidden row on the draft's card, or a scheduler per draft
shape (C619's `LLAMA_MTP_GRAPH_KEEP`, R95), removes the 4 a step (est.).

The catch-up passes in request 7 ([read07/prompt_sums.tsv](read07/prompt_sums.tsv)): 16 over the 2,048-token call,
12 over the 1,533 and 4 over the 512. In each pass card 8's split waits a median 8.199 ms (2,048-token call) and
8.200 ms (1,533) in `w` after the drain, est. the pass's 128 token rows coming from card 4 (1.31 MB of F32 rows at
~160 MB/s); card 4's own split takes a median 20.5 and 22.0 µs to launch. The host waits for the target's rows before the
three catch-ups 4,820.4, 2,228.7 and 5,898.6 ms (`decode-trace`'s `wait`).

### Before card 0 in a verify pass

Every main-model evaluation of up to 8 tokens after a prompt ([read07/steps_detail.tsv](read07/steps_detail.tsv);
eight tokens a request, so these are the first steps after each prompt and most rebuild their graph). The
four-token verify passes, µs:

| Request | Graph | Evaluation start → card 0's launch | CPU split | card 0 wait | card 0 copies |
|---:|---|---:|---:|---:|---:|
| 2 | rebuilt (params) | 7,399 | 6,579 | 27 | 782 |
| 2 | reused | 5,346 | 4,447 | 33 | 856 |
| 2 | reused | 4,068 | 3,248 | 27 | 782 |
| 3 | rebuilt (params) | 5,912 | 5,040 | 29 | 834 |
| 3 | reused | 6,275 | 5,382 | 27 | 857 |
| 4 | rebuilt (params) | 4,392 | 3,478 | 29 | 876 |
| 5 | rebuilt (params) | 5,192 | 4,311 | 29 | 842 |
| 5 | reused | 5,906 | 4,994 | 29 | 873 |
| 6 | rebuilt (params) | 3,529 | 2,578 | 27 | 914 |
| 7 | rebuilt (params) | 4,112 | 3,198 | 36 | 869 |
| 8 | rebuilt (params) | 1,118 | 76 | 27 | 1,010 |
| 9 | rebuilt (params) | 980 | 77 | 28 | 870 |
| 10 | rebuilt (params) | 970 | 76 | 27 | 862 |

The CPU split is 4 nodes (the token embedding's and the n-gram table's rows) and the same graph in every row, so its
2,578 to 6,579 µs in requests 2 to 7 against 76 and 77 µs in requests 8 to 10 depend on the data: est. whether the
tokens' n-gram rows are in the page cache. Requests 7 and 8 return the same reply; request 7 reads its rows first. A
rebuilt verify graph also costs the host `build` and `alloc` before the evaluation (4,793 and 6,470 µs, medians of the
nine rebuilt passes, against 59, 88, 63 and 61 µs and no allocation in the four reused).

## What it changes

- **C712 (Q1264):** the trigger is known; a plan once at the full context removes 30 of 34 re-plans a prompt, but
  with one input copy that saves the planning (6.2 ms a prompt), not the drains. The drains matter only once the
  draft's passes can queue ahead (C703, `LLAMA_MTP_CATCHUP_OVERLAP`, or more input copies for the draft).
- **C717 (Q1269):** stays parked; no main-model re-plan in any request.
- **C702 (Q1240) and R82, R62 (Q1239):** confirmed as the levers for the 8,872.2 ms of blocked launches.
- **C738 (Q1307):** of R86's 5.610 ms only card 0's copies (0.857 ms) can move into the last drafting pass's wait with
  every input but the ids; its upper case shrinks from 6.50 to est. ~1.75 ms a step. The CPU split needs the ids.
- **C615 (Q1100):** the v4 line's n-gram input already prefetches each micro-batch's rows when it sets the input;
  what is left is to prefetch each drafted token's rows as it is drafted, est. up to the 4.379 ms CPU split of a
  verify pass whose rows are not cached.
- **New:** Q1339 (the prompt's n-gram prefetch off the host's path before card 0: est. 0.46 to 1.4 s of a fresh
  4,097-token prompt), Q1340 (the draft's catch-up token rows from card 4, est. 8.2 ms a 128-row pass) and Q1341
  (this trace with the prompt graphs on, beside R82).

## How it was measured

One load of R42's driver ([run/r42_run.sh](../2026-11-10_prompt_timeline/run/r42_run.sh), R43's wrapper and the
preset's environment and line, `R42_LOADS=warm4096`) with `LLAMA_UBATCH_TRACE=1` exported, on the idle rig: start
11:20:42, the server up after 309 s, done 11:29:06. The server's log (1,304,086 bytes) holds the trace of 171 decode
calls and 362 micro-batches with 1,564 splits and 186 re-plans; every evaluation belongs to a micro-batch. [run/split_requests.py](run/split_requests.py) cuts the log at each request's
`launch_slot_ ... processing task` line, C701's [trace_table.py](../2026-11-11_c701_microbatch_trace/run/trace_table.py)
makes its tables of each request and of the whole log, and [run/r81_read.py](run/r81_read.py) makes the readouts.
The prompt times, unprofiled: 28,029.24 and 27,306.99 ms with the taper off (requests 7 and 10), 26,313.00 and
26,902.38 ms with it on; the lowest MemAvailable in the load was 196 MiB and 269 MiB were in swap at its end.

## Files

| File | What it is |
|---|---|
| [tables/requests.tsv](tables/requests.tsv) | per request: lines, prompt and generated tokens and times, prompt calls, trace lines of each kind |
| [tables/all/](tables/all/), [tables/req00/](tables/req00/) to [tables/req10/](tables/req10/) | `trace_table.py` on the whole log and on each request: `calls.tsv`, `ubatch.tsv`, `splits.tsv`, `replan.tsv`, `summary.txt` (req00 is the load and its warm-up) |
| [read07/](read07/) | `r81_read.py` with request 7: `prompt_calls.tsv`, `prompt_ubatch.tsv`, `prompt_splits.tsv`, `prompt_sums.tsv`; and the load's `prompt_inputs.tsv`, `replans.tsv`, `replan_counts.tsv`, `steps.tsv`, `steps_all.tsv`, `steps_detail.tsv` |
| [read10/](read10/) | the same with request 10 |
| [load_requests.tsv](load_requests.tsv) | R43's wrapper's table of the load's requests: tokens, times, reply md5 |
| [run/split_requests.py](run/split_requests.py) | cuts llama-server's log into its requests |
| [run/r81_read.py](run/r81_read.py) | the readouts above from `trace_table.py`'s tables per request |

## Commands

On the rig, detached, with R42's driver in `$R42`, R43's files in `$R43` and the night build unpacked in `$BUILDS`
(as R42's README says):

```bash
ssh $RIG "cd $R42 && LLAMA_UBATCH_TRACE=1 R42_LOADS=warm4096 R43=$R43 BUILDS=$BUILDS \
  BUILD=llama.cpp-20261111-c408dabcb-pascal-p106-v4-b12246-night \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$R42/out-r81 \
  setsid nohup bash $R42/r42_run.sh > $R42/r42_r81.out 2>&1 < /dev/null &"
```

When `$R42/out-r81/DONE` exists, the log off the rig, then on any machine with Python 3 (seconds), from this folder
with C701's script beside these:

```bash
L=out-r81/warm4096/load-A/server.log
python3 -I run/split_requests.py $L tables
python3 -I ../2026-11-11_c701_microbatch_trace/run/trace_table.py $L tables/all
for f in tables/req*.log; do n=$(basename $f .log); mkdir -p tables/$n; \
  python3 -I ../2026-11-11_c701_microbatch_trace/run/trace_table.py $f tables/$n; done
python3 -I run/r81_read.py tables read07 --prompt-req 7
python3 -I run/r81_read.py tables read10 --prompt-req 10
```

The per-request logs (`tables/req*.log`) are raw server output and are not published.

Provenance: the prompts are R43's fresh prompts and the agent check's recorded turns of `bugfix-javascript-triangle`
([2026-11-03_retype_agent/agent/turns.csv](../2026-11-03_retype_agent/agent/turns.csv)); nothing was downloaded.
