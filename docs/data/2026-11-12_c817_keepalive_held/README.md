# A stop at a held keep-alive start does not wait for the evaluation behind it (C817, Q1426, 2026-11-12)

What this is: a change to Miner Llama's CUDA backend inside [C811](../2026-11-12_c811_keepalive_wait/README.md)'s
switch `GGML_CUDA_KEEPALIVE_EAGER` (off by default; no new switch), its checks on a build machine's CPU and its check on
nine P106 cards (R140). C811 removed
every wait for a card's keep-alive block from served replies, but on nine P106 cards (R131) it made a server's first
long prompt 1,851.1 and 1,819.5 ms longer: the host queued the prompt's first chunk one card at a time. The cause is a
stop that waited for a block which had not begun, and so for the evaluation the block was queued behind. Now such a stop
does not wait. On a CPU, in a model of the controller with a thread for each card, a stop at a held start took 0.0 ms
in each of three runs, where C811's controller waited 300.1, 300.0 and 300.1 ms for the evaluation it is held behind; a
first chunk over four cards was queued in 0.6, 0.5 and 0.5 ms, where C811's controller took 181.1, 180.5 and 180.6 ms (one
60 ms evaluation a card). It is merged off by default into `pascal-p106-v4` as `e572eb1e6`. On nine P106 cards (R140,
below) the host queues the first chunk on seven cards within 43.5 and 44.0 ms with the switch at 1, as it does at 0,
and the 2,760-token prompt is now 1,925.32 and 1,321.11 ms shorter with the switch than without it. Registry [Q1426](../../../quants/experiments.md); switch list [docs/switches.md](../../switches.md).

## The problem

`GGML_CUDA_KEEPALIVE=decode` (Q1021) runs one small block on each card that polls a word in mapped host memory, so the
driver keeps the card at its boost clock. `GGML_CUDA_KEEPALIVE_EAGER=1` (C811) ends a card's block before an evaluation
that runs kernel by kernel at shapes new to the card, and before a copy, set, get or memset outside a graph at a size new
to the card: under lazy module loading, the first launch of a kernel waits for every kernel running on the card, the
block included. The block's next start is then held: it is launched on its own stream behind an event recorded after the
card's next evaluation, so it begins only when that evaluation is done.

In R131 the 2,760-token prompt of a server's second request took 28,311.1 and 28,267.0 ms with the switch at 1, against
26,460.0 and 26,447.5 at 0. All of the difference was in its first 756 tokens, the first chunk at sizes new to the cards.
There the host entered each card about one GPU time after the card before (671.5, 677.6 and 668.9 ms from card 0 to card
3; card 0's evaluation took 670.7 ms on its GPU), against 6 to 7 ms with the switch at 0. Card 7 was entered 4,474.0 and
4,482.9 ms after card 0, so card 0 started the second chunk about 4.7 s late, and the cards lost their overlap for the
first two chunks.

The controller's code shows why. On each card in turn, the scheduler queued the evaluation (its block's start held behind
it), then the copy of its output to the next card. That copy's size was new to the card, so the switch stopped the card's
block. The controller writes the stop's word and then waits for the block to end, by waiting for the block's stream. The
block had not begun: it was still queued behind the evaluation just queued. So the stop waited for that evaluation, and
the host entered the next card only after it.

## The change

Fork branch `pascal-p106-v4-b12246-c817`, two commits on `pascal-p106-v4` `b9a1c1fa7` (`8f6916133` the change and its
tests; `5f8dd491b` one test check that assumed a timing), host code only: `ggml/src/ggml-cuda/keepalive-policy.h`,
`keepalive.cu` and `tests/test-cuda-keepalive.cpp`.

- **A stop at a held start that has not begun does not wait.** After writing the stop's word, the controller asks
  whether the card's newest block has begun. A held start records an event on the evaluation's stream and makes the
  keep-alive stream wait for it right before the block's launch, so while that event is not done, the newest block
  (held, or launched later behind a held one) has not begun (`cudaEventQuery`, which does not wait). Such a block is left
  to end by itself: when the evaluation is done it begins, reads a word without its own run's number (the stop's 0, or a
  later run's), and ends at its first poll without running. Because the word is written before the question is asked, a
  block found not begun is sure to read it. The host goes on at once.
- **Everything else is as before.** A block that has begun is stopped and waited for as C811 does (it ends within one
  poll). The same rule applies when every block is ended before a free, a host allocation or a device-wide synchronize:
  those wait for the card's streams anyway, and the block ends right after the evaluation.
- **Only at levels 1 and 2.** No start is held otherwise, and the question is never asked: with the switch unset or at
  `0` the controller does what it did before.
- **The counts line** (llama-server after each reply, whenever the switch is set) ends with
  `N starts held, N blocks ended before they began`. Such a block counts as ended by the host.

## Results on a CPU

All on a build machine, CPU only, 2026-11-12: the branch `5f8dd491b` 21:31:15–22:14:44, the base's rig tarball in a
first run 20:48–21:15, the merged tree 22:15:18–22:18:37 ([checks.md](checks.md)). Nothing ran on a GPU.

| Check | Result |
|---|---|
| The rig tarball (`TESTS=1`, `EXTRA_TARGETS=test-cuda-keepalive`; sm_61, no AVX) | built: `llama.cpp-20261112-5f8dd491b-pascal-p106-v4-b12246-c817.tar.gz`, 741,714,237 B, md5 `8b651102279353d33c35260c1b825d56`; 0 AVX instructions, highest symbol GLIBC_2.38; `bin/test-cuda-keepalive` md5 `719c24782620821a45253b9cd405b02a`, `bin/llama-server` md5 `fbc82b14f36576b89ae611bffac5b1ac`; the base build's 34 warning lines |
| The base's rig tarball (`b9a1c1fa7`, the same build) | built: `llama.cpp-20261112-b9a1c1fa7.tar.gz`, 741,689,679 B, md5 `262917966d1c0a597f2b5e5a2adc1d91` |
| The switch list (C770) | 867 names, `GGML_CUDA_KEEPALIVE_EAGER` among them and in `bin/llama-server`, with the registry name `ggml_backend_cuda_keepalive_counts` |
| `llama-server`'s GPU code against the base's | the same: 8,540 functions, SASS and PTX, each dump the same file |
| `test-cuda-keepalive` (the controller, a thread per card) | 237 checks, 0 failed, in each of three runs. The new ones: unset and at `0` the new question is never asked and nothing ends unbegun; a stop at a held start returns at once, and the block it leaves begins after its evaluation, ends at its first poll without running and counts as ended by the host; the next held start runs after the next evaluation; a first chunk over four cards; four threads at once with busy cards (484, 468 and 468 runs, 303, 297 and 297 of them ended before they began, none lost, no block that ran launched over) |
| `test-cuda-keepalive gpu` without a card | exits 1 with "no CUDA device", as it must |
| `test-llama-archs` against the base's logits (C811's static CPU build) | 146 of 146 the same with the switch unset, and at `0`, `1` and `2` with `GGML_CUDA_KEEPALIVE=decode` (a CPU build; the CUDA switches must change nothing there) |
| C565's set, static and shared | ctest 71 of 71, the network test 1 of 1, archs 146 written and compared, the nine hand tests pass |
| The merged `pascal-p106-v4` (`e572eb1e6`: `-c817` on C815's `14710289b`, 22:15:18–22:18:37) | a static CPU build of every target; `test-cuda-keepalive` 237 of 237; `test-llama-archs` 146 of 146 the same as the base's with every switch off and with the switch at `1` |

**The mechanism, timed** (`test-cuda-keepalive`, which models the controller with a thread for each card's block: a held
block waits until its stream's queued work is done, and a card's blocks run in order). The same scenario runs with C811's
controller (`cancel_held` off in the test; the build has no way to turn it off) and with C817's, three runs each:

| Case | C811's controller | C817 |
|---|---|---|
| `test_held_stop`: on one card, an evaluation kernel by kernel at a new shape (300 ms of the card's time), its block's start held behind it; then a copy at a size new to the card stops that block. The time the stop takes | 300.1, 300.0 and 300.1 ms: it waited for the evaluation | 0.0, 0.0 and 0.0 ms |
| `test_held_chunk`: a first chunk over four cards, as R131's prompt: on each card an evaluation at a new shape (60 ms, starting when the previous card's ends), then the copy to the next card at a size new to both. When the host queued each card, ms after card 0 | 0.0, 60.4, 120.6, 181.0; 0.0, 60.2, 120.3, 180.5; 0.0, 60.2, 120.4, 180.5: one evaluation a card | 0.0, 0.3, 0.4, 0.6; 0.0, 0.3, 0.3, 0.5; 0.0, 0.2, 0.3, 0.5 |

C811's column is R131's pattern: card k entered one evaluation after card k − 1. A first run of the checks on
`8f6916133` (20:48–21:15) gave the same timings (300.0, 300.1 and 300.1 ms; card 3 at 180.4, 180.5 and 180.9 ms against
0.4, 0.4 and 0.6) and failed one check in each run. That check assumed that the card's first block had polled once
before the switch ended it ahead of the evaluation, which a stop microseconds after its start need not allow.
`5f8dd491b` checks only what the case is about, that the held block never ran; the controller did not change.

## The rig check (R140)

On nine P106-090 cards, with the rig tarball above: [R131](../2026-11-12_c811_keepalive_wait/README.md#the-rig-check-r131)'s
driver [run/rig_c811.sh](../2026-11-12_c811_keepalive_wait/run/rig_c811.sh) as it ran there, `test-cuda-keepalive gpu` on
card 0 (unset, at `1`, `CUDA_MODULE_LOADING=EAGER`; C817's binary adds the held-start set: a set at a new size right after
a graph of 400 in-place `cos` on 8 Mi floats, whose first call runs kernel by kernel, is queued), then R122's rung `f` as
four loads `f0 f1 f0-2 f1-2` (the switch at 0 and 1), read with
[run/rig_tables.sh](../2026-11-12_c811_keepalive_wait/run/rig_tables.sh) and
[run/rig_prompt_chunks.sh](../2026-11-12_c811_keepalive_wait/run/rig_prompt_chunks.sh). It passes when:

- `f1` and `f1-2` queue the long prompt's first chunk a few ms a card (`prompt_chunks.md`), and their long prompt takes
  `f0`'s and `f0-2`'s time;
- there is still no wait for a keep-alive block in their generation (`waits.tsv`), and the greedy reply is `f0`'s;
- `test-cuda-keepalive gpu` passes at `1`: the set did not wait for the graph (a quarter of its time at most).

R140 ran on 2026-11-13 08:21:05–08:40:42, with the rig tarball relayed and its md5 checked on both ends (`bin/llama-server`
md5 `fbc82b14f36576b89ae611bffac5b1ac`, `bin/test-cuda-keepalive` md5 `719c24782620821a45253b9cd405b02a`) and R131's five
scripts unchanged (the same md5s as R131's).

### Result

**It passes every condition. C817 removes the one cost C811's switch had: with `GGML_CUDA_KEEPALIVE_EAGER=1` a
server's first 2,760-token prompt is now 1,925.32 and 1,321.11 ms shorter than with the switch at 0, where R131 found it
1,851.1 and 1,819.5 ms longer.** The host queues the prompt's first chunk on every card at once again, and the switch's
gains in the first prompt and in generation stay.

| Condition | `f1` | `f1-2` |
|---|---|---|
| The first chunk a few ms a card ([prompt_chunks.md](rig/prompt_chunks.md): cards 1, 2, 3, 5, 6, 7, entry minus card 0's) | 7.0, 13.7, 20.3, 31.9, 37.9, 43.5 ms (`f0`: 7.7, 14.4, 20.8, 32.3, 38.2, 43.7) | 7.0, 13.9, 20.6, 32.3, 38.3, 44.0 ms (`f0-2`: 7.1, 13.9, 20.6, 32.3, 38.2, 43.7) |
| The long prompt `f0`'s and `f0-2`'s time ([arms.md](rig/arms.md)) | 24,712.63 ms against `f0`'s 26,637.95: 1,925.32 shorter | 24,744.13 ms against `f0-2`'s 26,065.24: 1,321.11 shorter |
| No wait in generation ([waits.tsv](rig/waits.tsv), `in_gen_ms`) | none (and none in either prompt) | none (and none in either prompt) |
| The greedy reply `f0`'s ([arms.md](rig/arms.md)) | the same text, 170 of 254 drafts accepted | the same text, 170 of 254 |
| The step within `f0` and `f0-2`'s spread (111.95 and 111.37 ms; spread 0.58) | 111.63 ms, −0.02 against their mean: negligible | 111.75 ms, +0.09: negligible |

Both long prompts with the switch are shorter than either reference load's, which differ by 572.71 ms: a **gain**.

**The card test** ([ka_gpu.txt](rig/ka_gpu.txt); card 0, `GGML_CUDA_KEEPALIVE=decode`, `GGML_CUDA_KEEPALIVE_MS=2000`;
every arm 0 of 65,536 values wrong in graph B and 0 of 1,024 in graph C). Graph C is 400 in-place `cos` on 8 Mi floats,
whose first call runs kernel by kernel; the set at a new size is queued right after it, the card's next block start held
behind graph C.

| Arm | Graph A's four calls, ms | Graph B's first call, ms | Memset, set at a new size, ms | Graph C queued, ms | The set right after graph C, ms | Graph C done after it was queued, ms | Graph C's second and third calls, ms | Blocks ended before they began | Checks |
|---|---|---:|---|---:|---:|---:|---|---:|---|
| the switch unset | 21.35, 1.99, 0.09, 0.13 | **2,000.34** | 0.03, 10.00 | 2.16 | **2,000.23** | 2,184.60 | 191.87, 186.59 | 0 | 2 of 2 |
| `GGML_CUDA_KEEPALIVE_EAGER=1` | 15.19, 0.44, 0.07, 0.09 | 17.87 | 0.08, 10.20 | 2.98 | **0.29** | 225.81 | 199.60, 187.97 | 1 | 8 of 8 |
| `CUDA_MODULE_LOADING=EAGER`, the switch unset | 2.52, 0.39, 0.06, 0.07 | 0.16 | 0.04, 10.01 | 3.14 | 0.05 | 187.38 | 192.21, 186.59 | 0 | 2 of 2 |

At `1` the set returned in 0.29 ms, where graph C's third call took 187.97 ms: the stop did not wait for graph C, and
the block it left ended before it began (the count of 1). With the switch unset, graph C's first call waited for the
running block's 2,000 ms timeout under lazy module loading (done 2,184.60 ms after it was queued), and the set took
2,000.23 ms. With eager loading neither waited.

**The four loads** (the server's own timings, ms; each wait found by `fn85.py`, with its card and length; step = the
mean of the two requests' median typical steps; [fn85.md](rig/fn85.md), [arms.md](rig/arms.md), [vm.tsv](rig/vm.tsv)):

| Load | Switch | First request (79 prompt tokens, greedy): prompt | its generation | waits in its generation | Second request (2,760 prompt tokens, temperature 1.0): prompt | its generation (tokens) | waits in its generation | waits in the two prompts | Step | Lowest MemAvailable, MiB |
|---|---:|---:|---:|---|---:|---:|---|---|---:|---:|
| `f0` | 0 | 7,742.09 | 10,811.48 | card 0, 1,034.7 (the last verify pass) | 26,637.95 | 3,812.15 (75: the reply ended there) | card 8, 1,012.7 (the first drafting pass) | card 8, 1,020.3 and 1,025.2 | 111.95 | 495 |
| `f1` | 1 | **4,054.56** | **10,019.34** | none | **24,712.63** | 9,813.13 (256) | none | none | 111.63 | 506 |
| `f0-2` | 0 | 7,467.70 | 10,739.64 | card 0, 1,033.9 | 26,065.24 | 10,177.75 (256) | card 8, 1,003.7 | card 8, 1,020.3 and 1,025.1 | 111.37 | 495 |
| `f1-2` | 1 | **4,009.26** | **10,096.05** | none | **24,744.13** | 10,392.16 (256) | none | none | 111.75 | 497 |

Paired by order (`f1` against `f0`, `f1-2` against `f0-2`):

- **The second request's prompt: −1,925.32 and −1,321.11 ms.** At the server's progress line the first 756 tokens took
  6.18 and 6.21 s against 7.13 and 7.12 (R131 with C811 alone: 9.23 and 9.25 against 7.17 and 7.20). The other 2,004
  tokens took 18.53 and 18.53 s against 19.51 and 18.95; the reference loads' include card 8's wait in the prompt,
  1,025.2 and 1,025.1 ms.
- **Where the first 756 tokens gain** ([prompt_chunks.md](rig/prompt_chunks.md)): the first chunk is queued on every
  card at once in all four loads, and in the second chunk card 0's evaluation takes 444.6 and 449.0 ms of GPU time with
  the switch against 1,386.6 and 1,387.7 at 0 (R131: 432.9 and 436.1 against 1,389.8 and 1,390.2). Est.: at 0 that
  evaluation's first launch at a new shape waits for card 0's block, which the switch ends first. Card 0 starts the
  third chunk 7,035.5 and 7,057.7 ms into the prompt against 9,048.0 and 8,807.7; in R131 the switch made it 10,096.9
  and 10,257.1 against 8,884.6 and 8,920.2.
- **The first request's prompt: −3,687.53 and −3,458.44 ms.** It is the server's first prompt, and its waits are gone.
- **The first request's generation: −792.14 and −643.59 ms** (25.45 and 25.26 tok/s against 23.59 and 23.74). Card 0's
  wait in the last verify pass is gone: the reply's tail after its profiled steps is 292.4 and 292.4 ms against 1,284.6
  and 1,283.8. The head grew, as in R131: 582.7 and 658.0 ms against 399.2 and 340.7, of which the host idles 383.6 and
  460.0 against 269.0 and 211.1.
- **The second request's generation** samples at temperature 1.0, so its text and length differ (`f0`'s reply ended
  after 75 tokens). Its time outside the profiled steps is 528.5 and 516.2 ms against 1,386.8 and 1,249.9; the
  reference loads' include card 8's wait in the first drafting pass, 1,012.7 and 1,003.7 ms.
- **The two prompts together: 28,767.19 against 34,380.04 ms (−5,612.85), and 28,753.39 against 33,532.94
  (−4,779.55).** Both requests of `f1-2` against `f0-2`: 49,241.60 against 54,450.33 ms (−5,208.73); `f0`'s second
  reply was shorter, so `f1` has no such pair.
- The step's parts are the same within 0.2 ms (verify pass on the cards 95.04 and 94.91 ms against 94.90 and 94.84,
  means of the medians; the draft's span 16.52 and 16.50 against 16.59 and 16.42). No graph was reallocated in any
  request ([reallocs.tsv](rig/reallocs.tsv)).

**The counts lines** ([keepalive_counts.tsv](rig/keepalive_counts.tsv), since the server started). Switch at 0: 39 of
100 blocks ended by the timeout after the first reply, and 102 of 199 and 102 of 205 after the second. Switch at 1: 0 of
139 after the first reply, with 96 starts held and 42 blocks ended before they began. After the second, 18 of 281 and
18 of 299 ended by the timeout (2 on each card), with 200 and 213 starts held and 78 and 70 blocks ended before they
began. The stops by then: 59 and 82 before evaluations run kernel by kernel (210 and 224 such evaluations, 190 and 204
acted on) and 129 and 112 before work outside a graph (932 and 901 new sizes).

## Commands

```bash
# the checks, on a build machine (CPU only; M a work folder, REPO a checkout of this repository, LOCK a flock file,
# BASE_CPU the bin/ of a static CPU build whose test-llama-archs writes the base's logits, SEED_SRC a llama.cpp clone
# with the fork as remote "fork"; without BASE_TREE the base's rig tarball is built too)
M=$M REPO=$REPO LOCK=$LOCK BASE_CPU=$BASE_CPU SEED_SRC=$SEED_SRC JOBS=8 \
    bash $REPO/docs/data/2026-11-12_c817_keepalive_held/run/checks.sh
# after the merge into pascal-p106-v4: the merged tree's "every switch off" check
M=$M ... STEPS=merged bash $REPO/docs/data/2026-11-12_c817_keepalive_held/run/checks.sh

# the controller's checks alone, on any CPU (the timed cases print their test_held_* lines)
build/bin/test-cuda-keepalive

# the rig check R140, ON the rig: C811's driver as R131 ran it, with BIN the bin/ of this rig tarball (paths of the
# model, the MTP draft, the n-gram table's shard, R122's preset and the C615 driver; S37's two requests)
BIN=... MODEL=... DRAFT=... TABLE=... PRESET=.../2026-11-12_r122_host_ladder/run/preset.env \
    RIG_SERVER=.../bench/agent_check/rig_server.sh VM_SAMPLER=.../2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
    RIG_LOADS=.../2026-11-11_c615_draft_prefetch/run/rig_loads.sh REQ_GREEDY=... REQ_LONG=... OUT=out \
    bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_c811.sh
# its tables, on any CPU (rig_prompt_chunks.sh after rig_tables.sh: it reads the long request's timeline)
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_tables.sh out tables
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_prompt_chunks.sh out tables
```

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: the rig tarball, the switch list, the base's rig tarball, `llama-server`'s GPU code against the base's, a static CPU build, `test-cuda-keepalive`, `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |
| [rig/fn85.md](rig/fn85.md), [rig/requests.tsv](rig/requests.tsv), [rig/waits.tsv](rig/waits.tsv) | R140: each request's time outside the profiled steps, part by part, and every wait for a keep-alive block (FN85's `fn85.py`) |
| [rig/arms.md](rig/arms.md), [rig/arms.tsv](rig/arms.tsv), [rig/arms_class.tsv](rig/arms_class.tsv), [rig/arms_paired.tsv](rig/arms_paired.tsv) | R140: the server's timings per reply, the step and its parts per load, each load against `f0` and `f0-2`, the greedy replies against `f0`'s (S37's `arms.py`) |
| [rig/split_f0.md](rig/split_f0.md), [rig/split_f1.md](rig/split_f1.md), [rig/split_f0-2.md](rig/split_f0-2.md), [rig/split_f1-2.md](rig/split_f1-2.md) | R140: each load's step split (S37's `step_split.py`) |
| [rig/prompt_chunks.md](rig/prompt_chunks.md) | R140: the long prompt's chunks, each card's host entry against card 0's |
| [rig/timeline/](rig/timeline/) | R140: every evaluation around the head and the tail of both requests of `f0` and `f1`, and of the long request of `f0-2` and `f1-2` (FN85's `timeline.py`) |
| [rig/ka_gpu.txt](rig/ka_gpu.txt) | R140: the three `test-cuda-keepalive gpu` runs |
| [rig/keepalive_counts.tsv](rig/keepalive_counts.tsv), [rig/card4.md](rig/card4.md), [rig/reallocs.tsv](rig/reallocs.tsv), [rig/vm.tsv](rig/vm.tsv) | R140: the counts line after each reply, card 4's heavier draft evaluation, graph reallocations, the lowest MemAvailable |

R140's driver and table scripts are C811's, in [its folder](../2026-11-12_c811_keepalive_wait/README.md#files):
`run/rig_c811.sh`, `run/rig_tables.sh` and `run/rig_prompt_chunks.sh`.

## Limits

- **The cause is shown in a model of the controller, on a CPU.** The test's threads stand in for the cards and their
  streams; the CPU timings above are that model's. On a P106 R140 shows the effect (the first chunk queued on every
  card at once, the set right after a long graph not waiting) and the count of blocks ended before they began; the stop
  itself is not timed in the served loads.
- **Two requests per server, two servers per arm,** as in R131. The second request samples at temperature 1.0, so its
  generation is not paired; `f0`'s reply ended after 75 tokens.
- **A stop can still wait in one case.** The question is answered for the event of the card's last held start. If the
  card's keep-alive stream also waits for an earlier held start on another stream (two contexts on one card, as the
  target and the MTP draft on the last card) whose evaluation is still running, the newest block is reported as begun
  and the stop waits as before. It is never the other way round: a block reported as not begun has not begun.
- **The ended block's own report is not read.** A later block on the card writes its report after it, and the counts
  take the ended one as ended by the host.

## Provenance and licence

Code and checks of 2026-11-12, on the Miner Llama fork of llama.cpp (MIT). The CPU checks ran on a build machine with no
GPU in use. R140 ran on one nine-card P106-090 rig on 2026-11-13, with Flash-Next's served file, its MTP draft and the
n-gram table as R122 and R131 ran them; its tables were made on the build machine's CPU. No model output is in this
folder (the replies are compared by `arms.py`, not published). Licence: the repository's.
