# No first kernel launch waits for a keep-alive block (C811, Q1417, 2026-11-12)

What this is: a change to Miner Llama's CUDA backend, behind a new switch, `GGML_CUDA_KEEPALIVE_EAGER` (off by
default), and its checks on a build machine's CPU. It removes the waits that
[FN85](../2026-11-12_fn85_server_gap/README.md) found in served replies: a decode or verify graph's first call waited
about one second for its card's keep-alive block to time out. The check on nine P106 cards is the rig row R131 (below);
it has not run yet. Registry [Q1417](../../../quants/experiments.md); switch list [docs/switches.md](../../switches.md).

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

## The rig check (R131, not run)

On nine P106-090 cards, from the private repository's `c811_rig.sh` (relay, push, start, wait, collect, tables, clean),
which runs [run/rig_c811.sh](run/rig_c811.sh) on the rig:

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
the step is within `f0` and `f0-2`'s spread. Est.: one second less for each first call in the decode path, two or three
in a server's first two replies at rung `f`. R128 (Q1419) tests two settings that may remove part of the same waits
without code (`GGML_CUDA_KEEPALIVE_MS=250`, `CUDA_MODULE_LOADING=EAGER`); both rows run on a rig.

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
# its tables, on any CPU
REPO=$REPO nice -n 19 bash $REPO/docs/data/2026-11-12_c811_keepalive_wait/run/rig_tables.sh out tables
```

## Files

| File | What it holds |
|---|---|
| [checks.md](checks.md) | every check's line from the build machine (`summary.tsv`) |
| [run/checks.sh](run/checks.sh) | the checks: the rig tarball, the switch list, `llama-server`'s GPU code against the base's, a static CPU build, `test-cuda-keepalive`, `test-llama-archs` against the base's logits, C565's ctest set static and shared, the merged tree |
| [run/rig_c811.sh](run/rig_c811.sh) | the rig check R131 (on the rig) |
| [run/rig_tables.sh](run/rig_tables.sh) | R131's tables (any CPU) |

The build script [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) takes
`EXTRA_TARGETS="..."` since this change, which puts `test-cuda-keepalive` into the rig tarball's `bin/`.

## Limits

- **Not yet on a P106.** The CPU checks show that nothing changes with the switch unset and that the controller does
  what it should with a thread standing in for each card. Whether the stops remove FN85's waits, and what they cost a
  step, is R131's to show.
- **The mechanism is est.** FN85 tied every wait to the block's timeout; "a first launch under lazy loading" is C590's
  mechanism, not a traced kernel load. `test-cuda-keepalive gpu` with the switch unset and with
  `CUDA_MODULE_LOADING=EAGER` tests it on one card.
- **A kernel picked by something outside the shape number** (a pointer's alignment, data read on the host) still meets
  a running block at level `1`. Level `2` stops before every evaluation that runs kernel by kernel.

## Provenance and licence

Code and checks of 2026-11-12, on the Miner Llama fork of llama.cpp (MIT). The CPU checks ran on a build machine with no
GPU in use. No model output is in this folder. Licence: the repository's.
