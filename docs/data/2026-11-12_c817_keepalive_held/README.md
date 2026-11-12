# A stop at a held keep-alive start does not wait for the evaluation behind it (C817, Q1426, 2026-11-12)

What this is: a change to Miner Llama's CUDA backend inside [C811](../2026-11-12_c811_keepalive_wait/README.md)'s
switch `GGML_CUDA_KEEPALIVE_EAGER` (off by default; no new switch), and its checks on a build machine's CPU. C811 removed
every wait for a card's keep-alive block from served replies, but on nine P106 cards (R131) it made a server's first
long prompt 1,851.1 and 1,819.5 ms longer: the host queued the prompt's first chunk one card at a time. The cause is a
stop that waited for a block which had not begun, and so for the evaluation the block was queued behind. Now such a stop
does not wait. On a CPU, in a model of the controller with a thread for each card, a stop at a held start took 0.0 ms
in each of three runs, where C811's controller waited 300.1, 300.0 and 300.1 ms for the evaluation it is held behind; a
first chunk over four cards was queued in 0.6, 0.5 and 0.5 ms, where C811's controller took 181.1, 180.5 and 180.6 ms (one
60 ms evaluation a card). It is merged off by default into `pascal-p106-v4` as `e572eb1e6`; the rig check is R140, not
run yet. Registry [Q1426](../../../quants/experiments.md); switch list [docs/switches.md](../../switches.md).

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

Not run yet.

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
```

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: the rig tarball, the switch list, the base's rig tarball, `llama-server`'s GPU code against the base's, a static CPU build, `test-cuda-keepalive`, `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |

## Limits

- **The cause is shown in a model of the controller, on a CPU.** The test's threads stand in for the cards and their
  streams; the timings above are that model's. On a P106 R140 shows whether the first chunk is queued card by card no
  longer.
- **A stop can still wait in one case.** The question is answered for the event of the card's last held start. If the
  card's keep-alive stream also waits for an earlier held start on another stream (two contexts on one card, as the
  target and the MTP draft on the last card) whose evaluation is still running, the newest block is reported as begun
  and the stop waits as before. It is never the other way round: a block reported as not begun has not begun.
- **The ended block's own report is not read.** A later block on the card writes its report after it, and the counts
  take the ended one as ended by the host.

## Provenance and licence

Code and checks of 2026-11-12, on the Miner Llama fork of llama.cpp (MIT). The CPU checks ran on a build machine with no
GPU in use. Licence: the repository's.
