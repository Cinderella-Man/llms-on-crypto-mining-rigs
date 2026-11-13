# No first kernel launch waits for a keep-alive block (C811, Q1417, 2026-11-12)

What this is: a change to Miner Llama's CUDA backend, behind a new switch, `GGML_CUDA_KEEPALIVE_EAGER` (off by
default), its checks on a build machine's CPU, and its check on nine P106 cards (R131). It removes the waits that
[FN85](../2026-11-12_fn85_server_gap/README.md) found in served replies: a decode or verify graph's first call waited
about one second for its card's keep-alive block to time out. On the rig it removes every one of them, with the same
greedy reply and the same step. A server's first two requests take 3,861.9 and 2,733.1 ms less in all. The one cost is
in a long prompt: its first chunk at sizes new to the cards loses its overlap across the cards, +1,851.1 and
+1,819.5 ms on a 2,760-token prompt (Q1426). Registry [Q1417](../../../quants/experiments.md); switch list
[docs/switches.md](../../switches.md).

## The problem

`GGML_CUDA_KEEPALIVE=decode` (Q1021) runs one small block on each card. It polls a word in mapped host memory until
the host ends it, or until no graph has been computed on the card for `GGML_CUDA_KEEPALIVE_MS` (1,000 ms by default).
While it runs, the driver keeps the card at its boost clock. The controller ends a card's block before a CUDA graph
capture, a free, a host allocation and a device-wide synchronize, because each of those would wait for it.

Under CUDA 12's lazy module loading, the first launch of a kernel on a card also waits until every kernel running on the
card has ended (C590, Q1075). A decode or verify graph's first call, and its first call after its shapes change, runs
kernel by kernel: llama.cpp captures a graph only on its second unchanged call. Such a call is often the first to need a
kernel. It then met a running block and waited for its timeout. FN85 found 97 such waits in R122's 28 served requests,
each ending 1,002.6 to 1,031.7 ms after its card's last beat: about 1.25 s of every reply's generation time, and up to
3.6 s of a server's first reply.

## The change

Fork branch `pascal-p106-v4-b12246-c811`, one commit `cfc657de5` on `pascal-p106-v4` `c96561876`, host code only:
`ggml/src/ggml-cuda/keepalive-policy.h`, `keepalive.cu`, `keepalive.cuh`, `ggml-cuda.cu`,
`tools/server/server-context.cpp` and `tests/test-cuda-keepalive.cpp`.

| `GGML_CUDA_KEEPALIVE_EAGER` | What it does (with `GGML_CUDA_KEEPALIVE` on) |
|---|---|
| unset (default) | nothing changes |
| `1` | Before an evaluation that runs kernel by kernel (not a capture, not a replay) at shapes the card has not run before, the card's block is ended, as before a capture. The same before a set, get, memset or copy the backend runs on a card outside a graph at a size new to the card. The block's next start then waits on the card's stream for the next evaluation's work, so no kernel that evaluation launches finds it running |
| `2` | as `1`, and before every evaluation that runs kernel by kernel, new shapes or not |
| `0` (or any other value) | no stop; only the counts |

- **Which evaluations.** `ggml_backend_cuda_graph_compute` knows whether an evaluation is a replay, a capture or a run
  kernel by kernel. For the last kind, the switch computes one number from the graph's nodes: their ops, types, shapes,
  strides and op parameters. That number tells which kernels the evaluation launches. A number the card has not seen
  ends the block. A graph that llama reuses as it was (the same `uid`) keeps its number, so a decode that replays its
  graphs pays nothing. A graph that is never captured pays only when its shapes are new. In a decode, a card's graphs
  change shape when the attention's KV view grows past a 256-cell step, so a card stops its block once per 256 tokens of
  context at most. Level `2` covers a kernel chosen by something the number does not hold (a pointer's alignment).
- **Work outside a graph.** The scheduler's copies between cards, the inputs it sets, the outputs it reads and the
  memsets run outside any graph, and the driver's own copy and memset kernels may load lazily too. FN85's card-4 wait
  came before its evaluation began, as host idle, in four of its five loads. The switch keys each such call by its kind
  (set, get, memset, copy on one card, the two ends of a copy between cards) and size, per card.
- **The restart after the evaluation.** Without the switch, the block starts again right after the evaluation's work
  is queued (the beat at its end). The first launch's wait could still meet that new block if the driver orders the
  load on the card. So a start that the switch holds waits on an event recorded on the evaluation's stream: the block
  begins when the evaluation's work is done. The card is busy until then.
- **The counts line.** The block reports are logged when a backend is freed, which llama-server's exit does not reach
  (C590). Whenever the switch is set (`0` included), llama-server prints one line after each reply, at its default
  verbosity, with the counts since it started:
  `keep-alive = <runs> runs, <n> ended by the timeout (by card <card>:<n> ...), <n> by the host, <n> without a report,
  <n> running; EAGER=<level>: <n> evaluations kernel by kernel, <n> acted on (<n> stops), <n> new sizes outside a
  graph (<n> stops), <n> starts held`. A block that has ended is read before the line is written. The CUDA registry
  exports it as `ggml_backend_cuda_keepalive_counts`.

## Results on a CPU

All on a build machine, CPU only, 2026-11-12: the branch 17:36–17:56, the merged tree 17:56–17:58 ([checks.md](checks.md)); nothing ran on a GPU.

| Check | Result |
|---|---|
| The rig tarball (`TESTS=1`, `EXTRA_TARGETS=test-cuda-keepalive`; sm_61, no AVX) | built: `llama.cpp-20261112-cfc657de5-pascal-p106-v4-b12246-c811.tar.gz`, 741,524,473 B, md5 `fcba7b406d934d102b6f7936e1982738`; 0 AVX instructions, highest symbol GLIBC_2.38; the base build's 34 warning lines |
| The switch list (C770) | 861 names, `GGML_CUDA_KEEPALIVE_EAGER` among them and in `bin/llama-server`, with the registry name `ggml_backend_cuda_keepalive_counts` |
| `llama-server`'s GPU code against `d097d6d88`'s | the same: 8,540 functions, SASS and PTX, each dump the same file (`c96561876` adds C802 and C805, host code only) |
| `test-cuda-keepalive` (the controller, a thread per card) | 162 checks, 0 failed, in each of three runs. The new ones: the switch's values; unset and at `0` the controller does what it did before; a held start waits for the evaluation's stream, once; shapes and sizes are new per card; level `2` acts on every evaluation; request mode's kick; the counts through a timeout; five threads at once with no run lost |
| `test-cuda-keepalive gpu` without a card | exits 1 with "no CUDA device", as it must |
| `test-llama-archs` against `d097d6d88`'s logits | 146 of 146 the same with the switch unset, and at `0`, `1` and `2` with `GGML_CUDA_KEEPALIVE=decode` (a CPU build; the CUDA switches must change nothing there) |
| C565's set, static and shared | ctest 70 of 70, the network test 1 of 1, archs 146 written and compared, the nine hand tests pass |
| The merged `pascal-p106-v4` (`e27947287`: `-c811` on C803's `d21820340`, 17:56:48–17:58:09) | a static CPU build of every target; `test-cuda-keepalive` 162 of 162; `test-llama-archs` 146 of 146 the same as `d097d6d88` with every switch off and with the switch at `1` |

## The rig check (R131)

On nine P106-090 cards, 2026-11-12 18:10:00–18:29:53, with the rig tarball above (relayed with its md5 checked on both
ends; `bin/llama-server` md5 `b9e64b34c0619b501c0efedc6370e360`, `bin/test-cuda-keepalive` md5
`62c42e4562ad1ccbf2596543a0d2ed3b`). The private repository's `c811_rig.sh` (relay, push, start, wait, collect, tables,
clean) runs [run/rig_c811.sh](run/rig_c811.sh) on the rig:

1. `test-cuda-keepalive gpu` on card 0, with `GGML_CUDA_KEEPALIVE=decode` and `GGML_CUDA_KEEPALIVE_MS=2000`, three
   processes: the switch unset, at `1`, and `CUDA_MODULE_LOADING=EAGER` with the switch unset. Each runs a small graph
   four times (its block running), then a graph of kernels not launched before (`sqr`, `sin`) right after a beat, then a
   memset and a set at sizes new to the card, and times each. At `1` none may take a quarter of the timeout. The other
   two show whether a first launch waits for the block under lazy loading and whether eager loading removes it.
2. R122's loads at rung `f` ([R122](../2026-11-12_r122_host_ladder/README.md): the eight host switches on R122's line,
   the host profile on, both of S37's requests), four loads: `f0` and `f0-2` with the switch at `0` (the counts only),
   `f1` and `f1-2` with it at `1`.

Read with [run/rig_tables.sh](run/rig_tables.sh) on a CPU: FN85's `fn85.py` (every wait for a block, per request and
part), S37's `step_split.py` and `arms.py` (each load's step against `f0`'s and `f0-2`'s, the replies against `f0`'s) and
the counts lines. It passes when `f1` and `f1-2` have no wait in their generation time, the greedy reply is `f0`'s, and
the step is within `f0` and `f0-2`'s spread. [run/rig_prompt_chunks.sh](run/rig_prompt_chunks.sh) then shows how the
host queued the long prompt's chunks. R128 (Q1419) tests two settings that may remove part of the same waits without
code (`GGML_CUDA_KEEPALIVE_MS=250`, `CUDA_MODULE_LOADING=EAGER`).

### Result

**It passes all three conditions, and `GGML_CUDA_KEEPALIVE_EAGER=1` joins the stacked line (`v4c2`) beside
`GGML_CUDA_KEEPALIVE=decode`.** Neither load with the switch has a wait for a keep-alive block anywhere: not in
generation, and not in the prompts. The greedy reply is `f0`'s text, and the step is within the reference loads'
spread. Taken together, the server's two requests are 3,861.9 and 2,733.1 ms shorter. A server's first long prompt pays
for it: +1,851.1 and +1,819.5 ms on 2,760 tokens (below; Q1426, C817).

| Condition | `f1` | `f1-2` |
|---|---|---|
| No wait in generation ([waits.tsv](rig/waits.tsv), `in_gen_ms`) | none (and none in either prompt) | none (and none in either prompt) |
| The greedy reply `f0`'s ([arms.md](rig/arms.md)) | the same text, 170 of 254 drafts accepted | the same text, 170 of 254 |
| The step within `f0` and `f0-2`'s spread (112.31 and 111.91 ms; spread 0.40) | 112.13 ms, +0.02 against their mean: negligible | 112.23 ms, +0.12: negligible |

**The card test** ([ka_gpu.txt](rig/ka_gpu.txt); card 0, `GGML_CUDA_KEEPALIVE=decode`, `GGML_CUDA_KEEPALIVE_MS=2000`;
every arm 0 of 65,536 values wrong). It confirms on a P106 what C590 assumed: under lazy module loading, the first launch
of a kernel that is not loaded yet waits for the running block.

| Arm | Graph A's four calls, ms | Graph B's first call (kernels not launched before), ms | Memset, set at a new size, ms | Graph B's second call, ms | Checks |
|---|---|---:|---|---:|---|
| the switch unset | 15.38, 0.70, 0.08, 0.11 | **2,000.36**: it waited for the block's 2,000 ms timeout | 0.03, 10.01 | 0.12 | 1 of 1 |
| `GGML_CUDA_KEEPALIVE_EAGER=1` | 15.41, 0.45, 0.06, 0.09 | **18.30** | 0.06, 10.20 | 0.18 | 5 of 5 |
| `CUDA_MODULE_LOADING=EAGER`, the switch unset | 2.49, 0.41, 0.06, 0.07 | **0.16** | 0.06, 9.97 | 0.21 | 1 of 1 |

**The four loads** (the server's own timings, ms; each wait found by `fn85.py`, with its card and length; step = the
mean of the two requests' median typical steps; [fn85.md](rig/fn85.md), [arms.md](rig/arms.md), [vm.tsv](rig/vm.tsv)):

| Load | Switch | First request (79 prompt tokens, greedy): prompt | its generation | waits in its generation | Second request (2,760 prompt tokens, temperature 1.0): prompt | its generation | waits in its generation | waits in the two prompts | Step | Lowest MemAvailable, MiB |
|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|
| `f0` | 0 | 7,673.0 | 10,829.0 | card 0, 1,036.4 (the last verify pass) | 26,460.0 | 12,277.9 | card 8, 1,020.3 (the first drafting pass); card 4, 1,018.7 (the last step) | card 8, 1,020.5 and 1,025.2 | 112.31 | 508 |
| `f1` | 1 | **4,726.8** | **10,091.0** | none | **28,311.1** | 10,249.1 | none | none | 112.13 | 478 |
| `f0-2` | 0 | 7,505.4 | 10,835.6 | card 0, 1,033.4 | 26,447.5 | 11,890.7 | card 8, 1,003.8; card 4, 1,020.6 | card 8, 1,020.4 and 1,025.1 | 111.91 | 496 |
| `f1-2` | 1 | **4,771.3** | **10,108.4** | none | **28,267.0** | 10,799.4 | none | none | 112.23 | 489 |

Paired by order (`f1` against `f0`, `f1-2` against `f0-2`):

- **The first request's prompt: −2,946.2 and −2,734.1 ms.** It is the server's first prompt, and its waits are gone.
- **The first request's generation: −738.0 and −727.2 ms** (25.27 and 25.23 tok/s against 23.55 and 23.53). Card 0's
  wait in the last verify pass is gone. The reply's tail after its profiled steps is 255.6 and 254.8 ms against
  1,289.0 and 1,286.4: the last verify pass now runs straight through, about 30 ms on each card
  ([timeline/f1_greedy.md](rig/timeline/f1_greedy.md)). The 248.2 and 248.5 ms of idle that followed card 0's wait
  (FN86's second stretch, Q1418) went with that wait. The head grew: the host idles 452.5 and 459.7 ms before the first
  verify pass against 268.5 and 273.2 (FN86's first stretch), and the other head work takes 199.5 and 204.0 ms against
  129.8 and 130.5.
- **The second request's generation: −2,028.8 and −1,091.3 ms.** It samples at temperature 1.0, so its text and step
  count differ (84 and 90 steps against 87 and 85). Its time outside the profiled steps is the fairer figure: 658.7 and
  533.3 ms against 2,272.8 and 2,260.1. Card 4's heavier draft evaluation still runs once (24.53 and 18.92 ms of GPU
  time, [card4.md](rig/card4.md)) and no longer waits.
- **The second request's prompt: +1,851.1 and +1,819.5 ms**, all of it in the first 756 tokens. At the server's progress
  line, `f1` and `f1-2` had taken 9.23 and 9.25 s for those tokens against 7.17 and 7.20. The other 2,004 tokens took
  19.08 and 19.02 s against 19.29 and 19.25.
- **Both requests together: 53,378.0 against 57,239.9 ms (−3,861.9), and 53,946.1 against 56,679.2 (−2,733.1).**
- The step's parts are the same within 0.2 ms (verify pass on the cards 91.84 and 91.83 ms against 91.80 and 91.65;
  the draft's span 16.62 and 16.64 against 16.69 and 16.53). No graph was reallocated in any request
  ([reallocs.tsv](rig/reallocs.tsv)).

**Where the long prompt loses its time** ([prompt_chunks.md](rig/prompt_chunks.md): each card's host entry minus card
0's for the prompt's chunks, cards 0–3 and 5–7). With the switch at 0, the host queues the first chunk on every card
within 43.7 and 43.1 ms, and the cards pass it along while card 0 starts the next chunk. With the switch at 1, it
queues the first chunk one card at a time, each card about one GPU time after the card before. In `f1` the gaps from
card 0 to card 3 are 671.5, 677.6 and 668.9 ms (card 0's evaluation took 670.7 ms on its GPU), then 1,231.1 from card 3
to card 5 (over card 4), 655.4 and 569.5; in `f1-2` 672.7, 676.9, 669.6, 1,234.0, 659.2 and 570.5. Card 7 enters
4,474.0 and 4,482.9 ms after card 0, so card 0 starts the second chunk 5,927.8 and 6,068.3 ms into the prompt instead of
1,252.7 and 1,275.1. From the third chunk on, all four loads queue alike. Chunks 3 and 6 go onto every card within
27 ms of card 0 in every load; chunks 2, 4, 5 and 7 go card by card in every load, as a card's launch queue fills at the third
queued micro-batch (Q1240). The est. mechanism: after a stop, a card's next block start is held on an
event after that card's next evaluation. The copy to the next card is at a size new to both cards, so it stops the
source card's block again. That stop has to wait for the held block, and so for the evaluation just queued.
[C817](../2026-11-12_c817_keepalive_held/README.md) (Q1426) is the fix: a stop at a held start that does not wait. Its rig check R140 passed: the first chunk queued on seven cards within 43.5 and 44.0 ms, and the 2,760-token prompt 1,925.32 and 1,321.11 ms shorter with the switch than at 0.

**The counts lines** ([keepalive_counts.tsv](rig/keepalive_counts.tsv), since the server started). Switch at 0: 39 of
100 blocks ended by the timeout after the first reply, and 111 of 210 and 111 of 214 after the second. Switch at 1: 0 of
139 after the first reply. After the second, 19 of 277 and 19 of 288 ended by the timeout (2 on each card, 3 on card
8). None of them delayed an evaluation (`fn85.py` finds none), so they are est. blocks that ran out while the server
waited between the two requests. The stops by then: 66 before evaluations run kernel by kernel (210 and 219 such
evaluations, 188 and 198 at shapes new to the card) and 112 and 122 before work outside a graph (901 and 932 new
sizes), with 197 and 208 starts held.

## Commands

```bash
# the checks, on a build machine (CPU only; M a work folder, REPO a checkout of this repository, LOCK a flock file,
# BASE_TREE an unpacked rig tarball of pascal-p106-v4 d097d6d88, BASE_CPU the bin/ of its static CPU build, SEED_SRC a
# llama.cpp clone with the fork as remote "fork")
M=$M REPO=$REPO LOCK=$LOCK BASE_TREE=$BASE_TREE BASE_CPU=$BASE_CPU SEED_SRC=$SEED_SRC JOBS=8 \
    bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/checks.sh
# after the merge into pascal-p106-v4: the merged tree's "every switch off" check
M=$M ... STEPS=merged bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/checks.sh

# the rig check, ON the rig (paths of the tarball's bin/, the model, the MTP draft, the n-gram table's shard, R122's
# preset and the C615 driver; S37's two requests)
BIN=... MODEL=... DRAFT=... TABLE=... PRESET=.../2026-11-12_r122_host_ladder/run/preset.env \
    RIG_SERVER=.../bench/agent_check/rig_server.sh VM_SAMPLER=.../2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
    RIG_LOADS=.../2026-11-11_c615_draft_prefetch/run/rig_loads.sh REQ_GREEDY=... REQ_LONG=... OUT=out \
    bash run/rig_c811.sh
# its tables, on any CPU (rig_prompt_chunks.sh after rig_tables.sh: it reads the long request's timeline)
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_tables.sh out tables
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_prompt_chunks.sh out tables
```

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: the rig tarball, the switch list, `llama-server`'s GPU code against the base's, a static CPU build, `test-cuda-keepalive`, `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |
| [run/rig_c811.sh](run/rig_c811.sh) | the rig check R131 (on the rig) |
| [run/rig_tables.sh](run/rig_tables.sh) | R131's tables (any CPU) |
| [run/rig_prompt_chunks.sh](run/rig_prompt_chunks.sh) | R131's prompt-chunk table: how the host queued the long prompt's chunks on the cards (any CPU) |
| [rig/fn85.md](rig/fn85.md), [rig/requests.tsv](rig/requests.tsv), [rig/waits.tsv](rig/waits.tsv) | R131: each request's time outside the profiled steps, part by part, and every wait for a keep-alive block (FN85's `fn85.py`) |
| [rig/arms.md](rig/arms.md), [rig/arms.tsv](rig/arms.tsv), [rig/arms_class.tsv](rig/arms_class.tsv), [rig/arms_paired.tsv](rig/arms_paired.tsv) | R131: the server's timings per reply, the step and its parts per load, each load against `f0` and `f0-2`, the greedy replies against `f0`'s (S37's `arms.py`) |
| [rig/split_f0.md](rig/split_f0.md), [rig/split_f1.md](rig/split_f1.md), [rig/split_f0-2.md](rig/split_f0-2.md), [rig/split_f1-2.md](rig/split_f1-2.md) | R131: each load's step split (S37's `step_split.py`) |
| [rig/prompt_chunks.md](rig/prompt_chunks.md) | R131: the long prompt's chunks, each card's host entry against card 0's |
| [rig/timeline/](rig/timeline/) | R131: every evaluation around the head and the tail of both requests of `f0` and `f1`, and of the long request of `f0-2` and `f1-2` (FN85's `timeline.py`) |
| [rig/ka_gpu.txt](rig/ka_gpu.txt) | R131: the three `test-cuda-keepalive gpu` runs |
| [rig/keepalive_counts.tsv](rig/keepalive_counts.tsv), [rig/card4.md](rig/card4.md), [rig/reallocs.tsv](rig/reallocs.tsv), [rig/vm.tsv](rig/vm.tsv) | R131: the counts line after each reply, card 4's heavier draft evaluation, graph reallocations, the lowest MemAvailable |

The build script [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) takes
`EXTRA_TARGETS="..."` since this change, which puts `test-cuda-keepalive` into the rig tarball's `bin/`.

## Limits

- **Two requests per server, two servers per arm.** R131 measures a server's first two requests: the ones where FN85
  found the waits. How often an agent session meets new shapes and new copy sizes later on, in its replies and its
  prompts, is for the agent check of the stacked line to show.
- **The long prompt's cost has an est. mechanism.** The host profile shows the first chunk queued one card at a time.
  That a stop at a held start waits for the evaluation behind it is read from the code, not traced (Q1426).
- **The first-launch mechanism is confirmed on one card only.** `test-cuda-keepalive gpu` shows the 2,000 ms wait with
  the switch unset and none with `CUDA_MODULE_LOADING=EAGER`. In the served loads FN85 tied the waits to the block's
  timeout, and none is left with the switch.
- **A kernel picked by something outside the shape number** (a pointer's alignment, data read on the host) still meets
  a running block at level `1`. Level `2` stops before every evaluation that runs kernel by kernel.

## Provenance and licence

Code and checks of 2026-11-12, on the Miner Llama fork of llama.cpp (MIT). The CPU checks ran on a build machine with no
GPU in use. R131 ran on one nine-card P106-090 rig on 2026-11-12, with Flash-Next's served file, its MTP draft and the
n-gram table as R122 ran them; its tables were made on the build machine's CPU. No model output is in this folder (the
replies are compared by `arms.py`, not published). Licence: the repository's.
