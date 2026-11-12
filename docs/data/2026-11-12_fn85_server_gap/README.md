# What a served reply does outside its profiled steps, on a P106 (FN85, Q1410, 2026-11-12)

What this is: a reading of [R122's](../2026-11-12_r122_host_ladder/README.md) raw files, on a CPU, with no new rig
time. In R122 each reply's generation time, as the server reports it (`predicted_ms`), was longer than the sum of
its profiled decode steps after the first. On a server's second request the gap was about 1.25 s in nine of the
fourteen loads and about 2.26 s in the other five, whatever the rung. On its first request it was about 3.6 s up to
rung `e`, and rung `f` (C771, C744, C742) cut it to 1,692.4 and 1,668.3 ms. This folder names that time. Registry
[Q1410](../../../quants/experiments.md); the code row it leads to is Q1417 (C811).

## Result

**Almost all of it is waiting for the keep-alive.** `GGML_CUDA_KEEPALIVE=decode` runs one small block on each card
until no graph has been computed on that card for 1,000 ms (`GGML_CUDA_KEEPALIVE_MS`). Under CUDA 12's lazy module
loading, the first launch of a kernel that is not loaded yet waits until every kernel running on its card has ended
(C590, Q1075). A decode or verify graph's first call runs eagerly, kernel by kernel. Only a capture stops the card's
block first. So when such a call is the first to need a kernel, it waits for the block to time out, about one
second. All 97 waits found in the 28 requests end between 1,002.6 and 1,031.7 ms after their card's last beat, the
controller's signal right after each evaluation ([waits.tsv](waits.tsv)).

- **The second request: ~1.25 s = one wait + ~0.24 s of ordinary work.** In every load the draft's first passes after
  the first token wait on card 8, the draft's card: 1,003.5 to 1,008.0 ms. The rest is the reply's last verify pass,
  the last catch-up and drafting passes and step 0's tail: 236.1 to 267.5 ms in the nine shorter loads.
- **Why five loads show ~2.26 s: one more wait, on card 4, in the reply's last step.** Each server runs one heavier
  draft evaluation on card 4 (where the draft's token embeddings are) exactly once, in the second request's last
  step ([card4.md](card4.md)). In nine loads it takes 7.15 to 9.63 ms of GPU time and does not wait. In `base`, `e`,
  `e-2` and `f-2` it takes 23.5 to 24.0 ms, so it runs at another size. Its first launch waits for card 4's block
  there: 1,019.6, 1,022.3, 1,019.0 and 1,022.3 ms, most of it as host idle before the evaluation begins. In `d-2` it
  waits 904.6 ms inside the evaluation (1,014.3 ms in all). The long request samples at temperature 1.0, so how its
  last step ends differs from load to load. That is why the split is not tied to a rung.
- **The first request: three waits up to rung `e`, one with `f`.** Card 8 waits 1,007.4 to 1,007.6 ms in the first
  drafting pass after the first token. Card 0 waits 1,032.6 to 1,034.9 ms in the reply's last verify pass, in all
  fourteen loads: that pass is not one of the profiled steps, so this wait is all "outside". Card 8 waits
  1,002.4 to 1,003.5 ms in the catch-up after that pass. With `f`, both of card 8's waits are gone. Card 0's stays.
- **Two shorter stretches are left unnamed, both in the first request only.** A host idle of 201.8 to 276.7 ms
  before its first verify pass, and one of 246.9 to 252.9 ms after its last verify pass's wait, with no evaluation
  running on any card. The second request has neither. They are Q1418 (FN86).
- **The same waits in the prompt time.** Every second request's prompt has one card-8 wait of 1,025.1 to 1,025.3 ms
  in the draft's catch-up over the prompt. Every first request's prompt has two up to rung `e` and one with `f`, which
  is `f`'s 994.5 and 973.2 ms shorter prompt in R122.

Per load and request (ms; "the rest" is R122's: the profiled steps after step 0; each wait as far as it lies inside
the generation time; [summary.md](summary.md), from [requests.tsv](requests.tsv) and [waits.tsv](waits.tsv)):

### The first request (79 prompt tokens, 256 generated, temperature 0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting pass | idle before the first verify pass | other, head | wait, card 0, last verify pass | idle after it | wait, card 8, last catch-up | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 13427.7 | 9753.6 | **3674.1** | 1007.6 | 246.7 | 134.2 | 1033.3 | 248.2 | 1002.4 | 0.5 |
| a | 13054.3 | 9386.4 | **3667.8** | 1007.6 | 241.8 | 129.6 | 1033.5 | 251.4 | 1002.6 | 0.5 |
| b | 13018.5 | 9339.0 | **3679.5** | 1007.5 | 253.9 | 129.5 | 1032.6 | 252.0 | 1002.6 | 0.5 |
| c | 12908.7 | 9229.1 | **3679.6** | 1007.6 | 255.2 | 125.8 | 1034.0 | 252.9 | 1002.5 | 0.5 |
| d | 12866.9 | 9188.0 | **3679.0** | 1007.4 | 259.5 | 124.6 | 1033.7 | 249.2 | 1003.4 | 1.2 |
| e | 12811.9 | 9189.8 | **3622.2** | 1007.6 | 201.8 | 124.8 | 1034.6 | 248.7 | 1003.5 | 1.1 |
| f | 10856.6 | 9164.2 | **1692.4** | — | 276.7 | 130.3 | 1034.0 | 246.9 | — | 4.4 |
| base-2 | 13416.3 | 9737.8 | **3678.5** | 1007.4 | 248.7 | 133.7 | 1033.9 | 250.7 | 1002.8 | 0.5 |
| a-2 | 12993.0 | 9362.0 | **3631.0** | 1007.6 | 204.3 | 129.8 | 1034.2 | 250.9 | 1002.7 | 0.5 |
| b-2 | 12996.1 | 9357.4 | **3638.7** | 1007.6 | 210.4 | 129.2 | 1034.9 | 252.6 | 1002.7 | 0.5 |
| c-2 | 12891.2 | 9227.4 | **3663.8** | 1007.5 | 240.8 | 125.6 | 1034.4 | 251.3 | 1002.6 | 0.5 |
| d-2 | 12869.6 | 9207.2 | **3662.4** | 1007.6 | 242.9 | 124.9 | 1033.8 | 248.6 | 1003.4 | 1.2 |
| e-2 | 12830.8 | 9196.4 | **3634.4** | 1007.6 | 215.9 | 124.7 | 1033.6 | 248.0 | 1003.4 | 1.2 |
| f-2 | 10824.8 | 9156.5 | **1668.3** | — | 251.4 | 130.2 | 1032.9 | 249.4 | — | 4.4 |

"Other, head" is step 0's verify pass and the draft's passes between the first token and step 1; "other, tail" is
what the last catch-up adds beyond its wait.

### The second request (2,760 prompt tokens, 256 generated, temperature 1.0)

| load | generation, ms | the rest, ms | outside, ms | wait, card 8, first drafting passes | other, head | wait, card 4, last step | other, tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| base | 13511.2 | 11241.0 | **2270.2** | 1007.9 | 31.2 | 1019.6 | 211.5 |
| a | 12133.3 | 10876.1 | **1257.2** | 1007.9 | 25.5 | — | 223.8 |
| b | 10193.7 | 8941.5 | **1252.1** | 1007.9 | 23.5 | — | 220.7 |
| c | 11223.6 | 9970.8 | **1252.9** | 1007.9 | 24.3 | — | 220.6 |
| d | 11237.1 | 9993.1 | **1244.0** | 1007.9 | 22.2 | — | 213.9 |
| e | 12043.5 | 9777.4 | **2266.1** | 1007.9 | 22.4 | 1022.3 | 213.4 |
| f | 10624.5 | 9380.3 | **1244.2** | 1004.1 | 21.8 | — | 218.3 |
| base-2 | 11569.8 | 10294.4 | **1275.4** | 1007.9 | 31.8 | — | 235.7 |
| a-2 | 11566.3 | 10295.5 | **1270.8** | 1008.0 | 26.0 | — | 236.8 |
| b-2 | 11616.6 | 10353.3 | **1263.4** | 1007.9 | 23.2 | — | 232.3 |
| c-2 | 10834.7 | 9584.1 | **1250.6** | 1007.9 | 23.8 | — | 218.9 |
| d-2 | 12072.5 | 9835.5 | **2237.0** | 1007.9 | 22.9 | 1014.3 | 191.9 |
| e-2 | 12717.2 | 10450.7 | **2266.5** | 1007.7 | 22.5 | 1019.0 | 217.3 |
| f-2 | 12863.7 | 10602.8 | **2260.9** | 1003.5 | 21.9 | 1022.3 | 213.1 |

"Other, tail" is the reply's last verify pass and the evaluations after it up to the last acceptance: the catch-up,
drafting passes and the card-4 evaluation ([timeline/a_long.md](timeline/a_long.md)).

## Why a first call waits, from the code

On Miner Llama's `pascal-p106-v4` at `d097d6d88` (the build R122 ran); `c96561876`, the branch today, changes none of
it:

- **The block.** `ggml/src/ggml-cuda/keepalive.cu` launches one thread per card on a low-priority stream of its own.
  It polls a word in mapped host memory until the host ends it or no beat has come for `GGML_CUDA_KEEPALIVE_MS`. The
  controller (`keepalive-policy.h`) beats the card after every graph computed on it. It stops the block before a
  capture (`graph_begin`), a free, a host allocation or a device-wide synchronize (`quiesce`). Its own comment says
  the timeout also bounds the wait of any synchronizing call the controller is not told about.
- **The eager first call.** `ggml_backend_cuda_graph_compute` runs a graph directly on its first call and after any
  change of its properties, and captures it only on the second unchanged call (the warm-up). `GGML_CUDA_GRAPHS_FORCE`
  (C702, in R72's `v4c2`) captures at the first call, but only for graphs of more than 8 tokens: prompt micro-batches
  and a draft's catch-up over a prompt (`graph-force.h`). Decode and verify graphs keep the warm-up on purpose: their
  first call makes state that must exist outside a capture.
- **So a decode-path kernel's first launch meets a running block.** Within a decode, every card computes a graph about
  every 112 ms, so its block is always running. A first launch then waits until the block ends after 1,000 ms
  without a beat. The waits above end between 2.6 and 31.7 ms after that point. When the host itself blocks (the
  launch inside the evaluation, or work before it), the profile shows a long host span or a host idle. When it does
  not, the card's GPU time is about one second.
- **Where the first calls are** (est., from where the waits fall): the draft's first drafting pass after a prompt; a
  narrower verify pass at a reply's end, where the server cuts the draft at the reply's 256-token cap; the catch-up
  over that pass; and an evaluation at a size the server has not run before. The second request's card-8 wait comes
  although the first request had run every drafting shape. Est.: the draft first needs some kernel at the longer
  context (the first request never passed 335 tokens).

## Which of `f`'s three switches removes the first request's share

R122 adds the three together, so its data cannot tell them apart. Est., from the code:

- **C771 (`LLAMA_MTP_EMBD_FIRST=1`) is the one that changes the passes where the waits fall.** Without it the draft's
  k-pool update is its own split on card 8 before the token-row gather on card 4 (`src/models/qwen4exp.cpp`). Every
  draft pass is then two graphs on card 8, a 0.03 to 0.1 ms one and the layer's, and both waits were in the
  layer's. With it, each pass is one graph on card 8, and the first request's card-8 waits are gone: both in the
  generation time, and one of the two in the prompt's catch-up.
- **C744 (`LLAMA_MTP_DRAFT_CAP=1`) acts once, at the greedy reply's last step:** its closing line says 1 drafting
  pass was left out in 1 draft. That is the step whose catch-up waited without `f`, so C744 may be the one that
  removes that wait. It has no hand in the first drafting pass. It acts only where the draft keeps no distributions
  (`!dp.result_q` in `common/speculative.cpp`). Its line counts 1 draft in each `f` load: the greedy reply's last.
- **C742 (`LLAMA_MTP_SKIP_LAST_ROW=1`) only drops the last drafting pass's copy of its hidden row** to the host. That
  is the third pass of a step, not the first, where the head's wait falls.

Rig row R128 (Q1419) settles it with the three one at a time. C811 removes the waits whichever switch shapes them.

## What it changes

- **C811 (Q1417), code: no first launch waits for a keep-alive block.** Stop the card's block before every
  evaluation that is not a replay: a graph's first call, a call after its properties changed, and any graph that is
  never captured. This is what the controller already does before a capture. Do the same before a copy or cast the
  scheduler launches onto a card outside a graph when its size is new, since card 4's wait came before its evaluation
  began. Most decode evaluations replay a warm graph, so the stops come only at the eager calls after a change. Est.
  one second for each first
  launch in the decode path. R122's line at rung `f` meets two or three of them in a server's first two replies, plus
  two in their prompts. `GGML_CUDA_GRAPHS_FORCE` in `v4c2` may already remove the prompt ones, since those graphs are
  batched.
- **R128 (Q1419), rig: the waits without code, and `f`'s switches apart.** On R122's driver and line:
  `GGML_CUDA_KEEPALIVE_MS=250` (each wait est. at most 250 ms; the block still covers a 112 ms step);
  `CUDA_MODULE_LOADING=EAGER` (no lazy loading, so no wait, at the cost of card and host memory, which the load
  measures); rung `e` with each of C771, C744 and C742 alone; and one load without the host profiler, to see the
  first reply's 1.9 s in the server's own times too.
- **FN86 (Q1418), CPU: the stretches left unnamed.** The first request's host idles of about 250 ms around its first
  and last verify passes. Also, every request's slot starts 444.1 to 559.1 ms before its first evaluation (the
  offset below), and that is prompt time. R86's step trace has the marks to name them
  ([its data](../2026-11-11_r86_step_trace/README.md)).
- **C590 (Q1075), for `decode`:** the 1 s block does cost llama-server about one second at each of these first calls
  (est. lazy loading, the row's mechanism; R128's `EAGER` load tests it).
- **C810 (Q1416):** R120b's 1,067 ms pause near a first reply's last main pass looks like the same kind of wait as card
  0's here (est.). R120b's traces are still that row's to read.

## Method

- **The steps** are S37's `step_split.py`
  ([served step split](../2026-11-10_served_step_split/README.md)): it cuts the host profile into requests, finds the
  main passes (one evaluation on each of the nine cards in order) and the steps between them. "The rest" is the sum
  of the generation steps after step 0, as in R122's `ladder.md`.
- **The two clocks.** The server's prompt time runs from the slot's start to the first token's sample (T_s), and its
  generation time from there to the acceptance after the last verify pass (T_end)
  (`tools/server/server-common.h`, `server_slot_stats`). The log and the profile have different zeros. `fn85.py`
  places the slot's start `delta` ms before the request's first evaluation, using the server's 3-second progress
  lines. Each prints `n_gen` and `tg` right after a step's acceptance, and in a decode step the server accepts right
  after the draft's catch-up over the verified tokens. Every line of a request must land at the end of a catch-up
  for the same `delta` (within 2 ms), with T_end no earlier than the request's last evaluation. That holds in all 28
  requests, for a `delta` of 444.1 to 559.1 ms. With it, the last acceptance falls 0.4 to 16.8 ms after the request's
  last evaluation in 19 requests, 92.4 ms after it in `f`'s second request, and 0.7 to 0.9 ms before it in eight
  first requests, where the last evaluation's end is its GPU time read from events. In seven second requests more
  than one `delta` fits. The fits differ by about a step, and the largest is used. A smaller one would put the last
  acceptance a step or more after the request's last evaluation, and in the requests with one fit it lies within
  16.8 ms of it, except in `f`'s second request.
- **The waits.** An evaluation ends at its entry plus the longer of its host span and its GPU time. It waited for a
  block when it ends 1,000 to 1,060 ms after its card's last beat: the latest exit of an evaluation on that card
  before it ends. Its wait runs from its entry, or from the start of a host idle of 100 ms or more right before it,
  to its end. Waits shorter than 300 ms are not counted (none came near).

## What ran

- **Input:** R122's raw rig output, its fourteen loads (`base a b c d e f`, then the same seven again), each a host
  profile (`GGML_CUDA_PROFILE_HOST`) and a server log at verbosity 3. That run used nine P106-090 cards and the stock
  4 GB host, Miner Llama `pascal-p106-v4` at `d097d6d88`, and Swift GSQ-RCO Q2_0 with the MTP draft
  ([R122's README](../2026-11-12_r122_host_ladder/README.md)).
- **The reading** ran on a build machine's CPU at `nice -n 19`, nothing on a GPU: `run/tables.sh`.

## Files

| File | What it holds |
|---|---|
| [summary.md](summary.md) | the two tables above |
| [fn85.md](fn85.md), [requests.tsv](requests.tsv) | every request: `delta`, prompt and generation time, the rest, outside, head and tail and their parts, the waits in generation and in the prompt, the anchor |
| [waits.tsv](waits.tsv) | every wait for a block (97): card, evaluation, where (prompt, head, steps, tail), its start against T_s and its end against T_end, the host idle before it, host span, GPU time, wait, time after the card's last beat, the part inside the generation time |
| [card4.md](card4.md) | each request's draft evaluations on card 4 with 5 ms or more of GPU time |
| `timeline/<load>_<request>.md` | five requests' head and tail, evaluation by evaluation, against T_s and T_end: `base` and `f` first requests, `base`, `a` and `d-2` second requests |
| [run/fn85.py](run/fn85.py) | the anchor, the parts and the waits |
| [run/timeline.py](run/timeline.py), [run/card4_passes.py](run/card4_passes.py), [run/summary.py](run/summary.py) | the timelines, the card-4 evaluations, the summary |
| [run/tables.sh](run/tables.sh) | all of it from R122's output |

## Commands

```bash
# on a build machine, CPU only (OUT: R122's rig output, <load>/host_profile.csv and <load>/server-<load>.log;
# REPO a checkout of this repository)
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_fn85_server_gap/run/tables.sh $OUT tables
```

## Limits

- **Every load ran with the host profiler on.** The profiler waits for each card's previous evaluation before it
  records the next, and the waits above do not depend on it. Still, no unprofiled load was read. In R122 the
  server's own generation time shows the first reply's ~1.9 s, so the waits are in what the server counts; R128
  includes an unprofiled load.
- **The kernel behind each first call is not named.** The profile shows evaluations, not kernels. "First launch
  under lazy loading" is the mechanism C590 found with resident blocks in this project's bench. That every wait ends
  1,002.6 to 1,031.7 ms after the card's last beat ties the waits to the block's timeout. The kernel's name would
  take a trace of module loads, or `CUDA_MODULE_LOADING=EAGER` (R128) making the waits disappear.
- **`delta` is fitted, not logged.** The profile's zero is not in the server log at its default verbosity (as in
  R86, Q1329). Where several values fit, the choice moves a reply's head and tail by about one step in opposite
  directions and leaves "outside" and the waits unchanged.
- **Two requests a server.** An agent session's later turns may meet more first launches (new sizes, longer
  contexts) or fewer. How many is R128's and R72's to count, with `fn85.py` on their profiles.

## Provenance and licence

Read on 2026-11-12 from R122's measurements of the same day, on one rig of nine P106-090 cards with the Miner Llama
fork's `pascal-p106-v4` (llama.cpp, MIT). The tables are this project's own. No model output is in this folder.
Licence: the repository's.
