# seq_rm bounded by the cells it can change (C740, Q1309, 2026-11-11)

What this is: a switch in Miner Llama, `LLAMA_KV_SEQ_RM_BOUND=1` (off by default), that stops the KV cache's
`seq_rm` from visiting every cell of the cache on every call. The server calls it twice in every speculative step:
once to take the drafted tokens out of the draft context's memory, and once to take the rejected tokens out of the
target's and the draft's memories after the acceptance. At the presets' 65,536-token context each call walked 65,536
cells per cache to change at most a handful. [R86](../2026-11-11_r86_step_trace/README.md) timed the two places on a
P106 rig: 0.393 ms (`s.drf.end → s.ckpt`, before the verify pass, on the token's path) and 0.779 ms
(`s.acc.spec → s.rm`, after the catch-up is queued). The registry row is Q1309 in
[quants/experiments.md](../../../quants/experiments.md).

Fork branch `pascal-p106-v4-b12246-c740` at `f023e94bf`, five commits on `pascal-p106-v4-b12246-c748` (`5b329df16`):
`c2b3e1ba3` the switch, the check and the timing, then four to the test code alone (`8644c4881` the timing's rollback,
`ec869ffed` and `0d901af37` no copy of a recurrent sequence with a rollback pending, `f023e94bf` a test that aborts
can be left out; see [The check](#the-check)). Not merged into another branch. Everything here ran on a build
machine's CPU; nothing ran on a card.

## Result

**With the switch, the two stages of a speculative step that hold `seq_rm` take 0.0070 ms instead of 0.8270 on a
CPU, at 65,536 tokens of context, with the same replies, drafts and caches.** In llama-server's MTP path on a build
machine's CPU (the tiny random qwen4exp model with one MTP block, the draft at three tokens, every draft rejected),
`s.drf.end → s.ckpt` (the draft context's call) takes 0.0020 ms instead of 0.2740 and 0.2730 (two loads each, medians
of 202 steps), and `s.acc.spec → s.rm` (the call on both memories after the acceptance) 0.0050 ms instead of 0.5520
and 0.5520. At 3,015 filled cells of 65,536, the end of R86's long request, one call that removes a step's 3 rejected
tokens takes 0.870 µs instead of 137.098 on llama's one KV cache, and 3.120 µs instead of 276.236 on qwen4exp's memory
(the recurrent memory, the attention cache and the indexer cache), medians of 200 calls on the same CPU
([below](#the-host-time-of-a-call)).

**It is exact.** `test-llama-archs --kv-seq-rm` runs every architecture the test builds (139 of them, Dense and MoE,
one, four and four split sequences: 436 tests) through 400 random decodes and memory edits in two contexts, the switch
off and on, and compares after every edit the memory's whole host state (the cells, the heads, each sequence's index,
the changed-cell list for the device copies, the QSA tables), the return values and every decode's logits: all the
same bytes, also with the device-side mask and its check on, with the four QSA device-table switches and their checks
on, and on the rig build's binary. Its test hook that leaves the cells unsorted fails 10 of 12 tests, so the check sees
the order. With every switch off the branch gives `5b329df16`'s logits on all 146 architectures, and ctest passes.

On a P106 R86's sizes give an est. 0.67 ms a step on the token's path (0.39 in the gap before the verify pass, up to
0.28 after the catch-up) and est. 1.16 ms a step of the Celeron's time in all; R101 measures it.

## The change

`llama_kv_cache::seq_rm(seq_id, p0, p1)` removes `seq_id` from every cell whose position is in `[p0, p1)`. Without the
switch it tests all cells of the stream. A cell outside the range or without the sequence is skipped by its first two
tests and nothing else.

The cells already keep, per sequence, an ordered index of `(position, cell)` pairs (`seq_pos`, used for the
sequence's lowest and highest position). A cell holds sequence `s` at position `p` exactly when `(p, cell)` is in
`s`'s index, and every change to a cell's position or sequences updates the index (`seq_pos_inc`, `seq_pos_dec`,
`seq_pos_rm`, `seq_pos_add`). With the switch:

- **One sequence (`seq_id >= 0`, every call the server makes):** the cells are read from the index between
  `(p0, 0)` and the first entry at `p1` or above: exactly the cells the loop would not skip. They are visited in
  increasing cell order, as the loop visits them, so every step is the same: `kq_cells_mark` lists the changed cells
  for the copies of the cells that the mask built on the cards reads (`LLAMA_GPU_KQ_MASK`) in the same order, and the
  first freed cell (the new `head`) and the freed cell of the lowest position (`LLAMA_KV_HEAD_TAIL`'s `head`) are the
  same cells. The index's order is position order; after a removal and a refill, or a wrap of the cache, a sequence's
  cells can be out of cell order, and then up to 64 cells are sorted. Past 64 out-of-order cells (a whole sequence of
  a fragmented cache), the call visits the used range instead (below).
- **Every sequence (`seq_id < 0`):** the loop runs from the first used cell to the last (`used_min()`,
  `used_max_p1()`), since every cell outside that range is empty and an empty cell is in no range.

Nothing else in `seq_rm` changes, so the cells, `head`, the list of changed cells and every table built from them
afterwards (the QSA k-pool layout and tables of C100 and C720, the pool mask of C719, the indexer's staleness) are the
same as without the switch. The QSA memory (`llama_memory_hybrid_idx`) removes from its recurrent memory first, then
from the indexer's cache and the attention cache, both of which are `llama_kv_cache`s and take the bound. The recurrent
memory's own `seq_rm` loops over its few cells and is unchanged.

`LLAMA_KV_SEQ_RM_BOUND=2` is a test hook: the index's cells in position order, unsorted. It must make the check fail.

For the check, `llama-ext.h` gains two test functions: `llama_memory_debug_state(mem)`, the host state of a memory as
bytes (every KV cache's heads; every non-empty cell's position, shift, extension and sequences; the used set; each
sequence's index and cell bits; the changed-cell list for the device copies; and, for a QSA memory, the staleness of
each sequence's pools, the pool layout, the k-pool tables the host holds for the device copies and their counters),
and `llama_memory_seq_rm_stats(mem)`, which counts the bounded calls and the cells they visited.

## The check

`test-llama-archs --kv-seq-rm` builds each architecture's random model and, per configuration, two contexts from it:
one without the switch and one with it (the switch is read when a cache is created). Configurations: one sequence in
256 cells; four sequences sharing 256 cells; four sequences in four streams of 128 cells. Small caches, so that
removals and refills leave holes and a sequence's cells fall out of position order. Both contexts take the same 400
random steps:

| Step | Weight | What |
|---|---:|---|
| decode | 36 | 1–8 tokens of one sequence at its next positions, sometimes with 1–4 tokens of a second sequence in the same batch; when the cache is full, a sequence is removed |
| rollback | 16 | 0–4 tokens off the tail right after a decode (a verify step's rejected drafts), or 1–4 off any sequence |
| checkpoint | 6 | models with a recurrent memory: the server's checkpoint (`llama_state_seq_get_data_ext`, partial), a decode, the restore and the positions past it removed |
| range | 10 | a random range of one sequence, up to 24 positions |
| sequence | 6 | a whole sequence |
| any range, everything | 4, 1 | a range of every sequence, every sequence |
| copy | 7 | as the server copies a slot: the destination cleared, the whole source copied |
| keep | 2 | one sequence kept, the rest removed |
| shift | 6 | models that can shift: a sequence's positions from a random one on moved by ±1–3 |

After every step: each removal's return value, `llama_memory_debug_state` of both memories (the same bytes), and each
decode's return code and logits (the same bytes); every 25 steps and at the end, the whole state write
(`llama_state_get_data`). For mamba, mamba2, deepseek4 and deepseek41, whose memories the debug state does not cover
(a recurrent memory alone, DeepSeek V4's cache), the state write is compared after every step instead. A removal
that a recurrent memory refuses (a partial removal it holds no snapshot for, a removal for every sequence) must be
refused in both. A recurrent sequence is not copied while it holds a rollback not yet applied: the recurrent memory
cannot write a state that two sequences share with two rollback indices (it aborts), and the server never copies a
slot in the middle of a step.

`run/bound_loop.sh` runs the arms. A test that aborts the process is run again for its architecture with the switch
off in both contexts (`LLAMA_ARCHS_SEQ_RM_BOUND_ON=0`); when it aborts at the same test there too, the abort is the
architecture's, the test is left out (`LLAMA_ARCHS_SKIP`) and the arm runs again.

| Check | Verdict |
|---|---|
| CPU build of `f023e94bf`, every target | OK (3 warnings, all of the base) |
| `--kv-seq-rm`, every architecture, seed 1 | **436 of 436 tests the same bytes**; 2 left out, aborting without the switch too (plamo2, Dense, 4 sequences and 4 streams: `GGML_ASSERT(view_src == NULL ...)` in `ggml.c`) |
| the same on llama, qwen4exp, qwen3next, qwen35moe, gemma3, mamba2, deepseek2, deepseek4, seeds 2 and 3 | 27 of 27 and 27 of 27 |
| every architecture with `LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1` (the mask built from the device copies, its changed-cell list in use), seed 1 | **435 of 435**; 3 left out, aborting without the switch too (plamo2's two, and minimax-01 MoE in 4 streams: the device mask's shape assert in `ggml.c`) |
| qwen4exp with C100's, C720's and C719's switches and their checks (`LLAMA_QSA_KPOOL_DEV=1`, `LLAMA_QSA_KPOOL_STAGE=1`, `LLAMA_QSA_POOL_MASK_DEV=1`, `LLAMA_QSA_INDEXER_ROWS=1`, `LLAMA_QSA_KPOOL_DEV_CHECK=1`, `LLAMA_QSA_POOL_MASK_DEV_CHECK=1`) and the device mask, seeds 1–3 | 3 of 3, 3 of 3, 3 of 3 |
| the control: `LLAMA_KV_SEQ_RM_BOUND=2` (the cells unsorted) on llama, qwen4exp, gemma3 | **10 of 12 fail**, each on the memory's host state after a range removal, as it must |
| llama-server, MTP draft at 3 tokens, `-c 4096`: the base `5b329df16`, the switch off, on; plain, the head's shortlist, a draft cut at 0.102 | 36 of 36 the same (3 replies and the spec dump each) |
| llama-server at `-c 65536` with the step trace: off, on, off, on | 16 of 16 the same; the stages in [rm_stages.md](rm_stages.md) |
| every switch off: `test-llama-archs -b CPU -s 1` against `5b329df16`'s logits | 146 of 146 the same |
| ctest (without the tests that need a download, the network or torch) | 70 of 70 |
| the rig tarball (`TESTS=1 scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c740`, sm_61) and `test-llama-archs` from its build tree | OK: `llama.cpp-20261111-f023e94bf-pascal-p106-v4-b12246-c740.tar.gz`, 656,329,805 B, md5 `0bd35603a3ce65a6182c47b53073ea18`, 0 AVX instructions, GLIBC_2.38 at most |
| `--kv-seq-rm` with the rig build's binary on the CPU (its CUDA backend against the stub driver): the eight architectures, seeds 1–3; qwen4exp with the four QSA switches and their checks, seed 1 | 27 of 27, 27 of 27, 27 of 27; 3 of 3 |

Over the 436 tests of the first arm, the bounded path ran 90,971 removals of one sequence (6,795 of them sorted their
cells, 104 visited the used range) and 21,361 removals for every sequence, visiting 615,915 cells in all; the loop
without the switch visits 256 cells per stream every call.

llama-server prints no library information lines at its default verbosity, so the switch's own log line does not show
in the server logs; that the server read it shows in the 65,536-token loads' stage times above, from the same binary
and the same environment variable as the 4,096-token arms.

Four commits after the switch's own changed only the test: the timing's rollback is now a 4-token verify batch less its
3 drafts (a recurrent memory keeps the snapshots of an n-token batch's last n − 1 tokens), a recurrent sequence with a
rollback pending is not copied (the first all-architecture runs stopped in clover_moe's four-sequence test on the
recurrent memory's own abort, "cannot write shared recurrent state with different rollback indices", which the
switch's code does not reach), and a test that aborts can be left out by name.

## The host time of a call

From [time.tsv](time.tsv) (`test-llama-archs --kv-seq-rm-time -s 1`): one context of 65,536 cells with the switch off
and one with it on, both holding one sequence of the given length, the calls alternating between them. "Removes 0":
`seq_rm(0, n, -1)` with nothing past n, a step whose drafts were all accepted (2,000 calls). "Removes 3": after a
decode of 4 tokens (the sampled token and 3 drafts), the call that removes the 3 drafts (200 calls; the context grows
by one token a call). µs, median; the mean is in the file.

| Model (its memory) | Cells filled | Call | Off | On | Cells visited a call, on |
|---|---:|---|---:|---:|---:|
| llama (one KV cache) | 79 | removes 0 | 137.053 | 0.040 | 0 |
| | 79 | removes 3 | 138.423 | 0.510 | 3 |
| | 335 | removes 0 | 134.173 | 0.070 | 0 |
| | 335 | removes 3 | 134.883 | 0.590 | 3 |
| | 2,760 | removes 0 | 134.243 | 0.090 | 0 |
| | 2,760 | removes 3 | 135.783 | 0.830 | 3 |
| | 3,015 | removes 0 | 134.263 | 0.100 | 0 |
| | 3,015 | removes 3 | 137.098 | 0.870 | 3 |
| | 16,384 | removes 0 | 134.903 | 0.120 | 0 |
| | 16,384 | removes 3 | 145.954 | 8.120 | 3 |
| qwen4exp (recurrent, attention and indexer caches) | 79 | removes 0 | 268.436 | 0.180 | 0 |
| | 79 | removes 3 | 274.876 | 1.710 | 6 |
| | 335 | removes 0 | 268.476 | 0.250 | 0 |
| | 335 | removes 3 | 275.556 | 2.080 | 6 |
| | 2,760 | removes 0 | 268.406 | 0.370 | 0 |
| | 2,760 | removes 3 | 276.151 | 3.085 | 6 |
| | 3,015 | removes 0 | 268.427 | 0.360 | 0 |
| | 3,015 | removes 3 | 276.236 | 3.120 | 6 |
| | 16,384 | removes 0 | 269.767 | 0.450 | 0 |
| | 16,384 | removes 3 | 284.532 | 10.541 | 6 |

Without the switch a call costs the same at 79 cells filled as at 16,384: it is the walk over all 65,536. With it, a
call costs the index lookup and the cells it removes. At 16,384 cells the removing call is slower on both (on: 8.120
µs against 0.120 for "removes 0"; off: 145.954 against 134.903): the decode before it has pushed the index's tree out
of the CPU's caches, and each removed cell updates two ordered sets. 79 and 335 are the context at R86's greedy
request's first and last step, 2,760 and 3,015 at its long request's.

In the server ([rm_stages.md](rm_stages.md), `run/rm_stages.py` over the step traces of four loads at `-c 65536`, each
three requests, 202 steps; every draft rejected, since the draft is random):

| Load | `s.drf.end → s.ckpt`, median ms | `s.acc.spec → s.rm`, median ms | both, median ms |
|---|---:|---:|---:|
| off | 0.2740 | 0.5520 | 0.8270 |
| on | 0.0020 | 0.0050 | 0.0070 |
| off, second load | 0.2730 | 0.5520 | 0.8270 |
| on, second load | 0.0020 | 0.0050 | 0.0070 |

`s.drf.end → s.ckpt` holds the draft context's call and the checkpoint bookkeeping; `s.acc.spec → s.rm` holds the
call on the target's memory (its recurrent memory, attention cache and indexer cache) and on the draft's, and the
prompt's tokens updated. The trace has a microsecond's resolution.

## What it changes on the rig (est., from R86's sizes)

R86 measured the two stages on a P106 rig with the night merge: `s.drf.end → s.ckpt` 0.393 ms and
`s.acc.spec → s.rm` 0.779 ms (medians). On the CPU the switch takes 0.2720 of 0.2740 ms (99.3 %) out of the first
and 0.5470 of 0.5520 (99.1 %) out of the second. In the same proportions on the rig, est.:
- `s.drf.end → s.ckpt` 0.393 → 0.003 ms. It is in the gap between the last drafting pass and the verify pass, on the
  token's path: est. 0.39 ms a step.
- `s.acc.spec → s.rm` 0.779 → 0.007 ms. It runs after the catch-up is queued; the host reaches the next drafting pass
  0.28 ms after the catch-up's work ends (R86), so est. up to 0.28 ms a step on the token's path, and the rest is
  Celeron time freed while the card computes.
- In all est. 0.67 ms a step on the token's path (R86's estimate for this row), and est. 1.16 ms a step of host time.
  Deeper in an agent session the off cost stays (it is the walk over 65,536 cells), and the on cost grows slowly with
  the index (the 16,384-cell rows above).

## What is left for C739

C739 (Q1308) moves the draft context's `seq_rm` from before the verify pass's launch to after it. With this switch on,
that call takes most of the 0.0020 ms left of `s.drf.end → s.ckpt` on the CPU: est. 0.003 ms on the rig, so C739 would
save est. at most 0.003 ms a step on top of this switch. Without this switch C739 is still worth est. up to 0.39 ms.

## What the rig run must show (R101, Q1344)

On rig 1, untimed, R86's driver ([rig_trace.sh](../2026-11-11_c405_server_gap/run/rig_trace.sh) with `LOAD_ENV`) with
the c740 tarball and the v3 preset. Six loads in one run, each answering S37's greedy request and then the agent's
first-turn request: `trace` and `tracebound` (the step trace with the switch off and on), then `prof`, `bound`,
`prof-2` and `bound-2` (the host profile alone: off, on, off, on). Est. 40 min of loads (R86's two loads took 13.4
min), plus the relay. The run is right when the six loads give R86's greedy reply text and draft counts, the switch off
and on. The verdict:
- [run/rm_stages.py](run/rm_stages.py) on `trace` and `tracebound`: `s.drf.end → s.ckpt` and `s.acc.spec → s.rm`,
  est. 0.393 → 0.003 ms and 0.779 → 0.007 ms;
- `bound` and `bound-2` against `prof` and `prof-2` on the step (S37's `arms.py`, its class against the spread of the
  two reference loads), and `step_gap.py`'s gap before the verify pass in `tracebound` against `trace`.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict, as `run/checks.sh` wrote it |
| [bound.tsv](bound.tsv) | every test of `test-llama-archs --kv-seq-rm`: build, arm, seed, architecture, configuration, edits, decodes, refused removals, the bounded calls and the cells they visited, status |
| [time.tsv](time.tsv) | `test-llama-archs --kv-seq-rm-time`: the host time of one `seq_rm` call off and on, per architecture, filled cells and call |
| [compare4k.txt](compare4k.txt), [compare65k.txt](compare65k.txt) | `server_smoke.py compare`: each reply's tokens and draft counts and the spec dump against the group's first arm, and the draft's graph builds; at 4,096 tokens of context (the base, the switch off and on, three variants) and at 65,536 (off, on, off, on) |
| [rm_stages.md](rm_stages.md) | `run/rm_stages.py` over the 65,536-token server's step traces: the two stages that hold `seq_rm`, off, on, off, on |
| [run/checks.sh](run/checks.sh) | the driver of every check (CPU builds, the tests, the servers, the rig tarball) |
| [run/bound_loop.sh](run/bound_loop.sh) | `test-llama-archs --kv-seq-rm` over the arms in one container, with the switch-off rerun of a test that aborts |
| [run/rm_stages.py](run/rm_stages.py) | the two stages that hold `seq_rm` in `LLAMA_STEP_TRACE` files |

The server checks use C619's [run/server_smoke.py](../2026-11-11_c619_draft_graph_reuse/run/server_smoke.py), and the
rig build C620's [run/checks.sh](../2026-11-11_c620_draft_kv_catchup/run/checks.sh) (its `build` step).

## Commands

On a build machine, CPU only (each build and test in a container at low CPU priority, one at a time under a lock):

```bash
# WORK: a work folder; SRC: a llama.cpp clone in $WORK/rig/src with the fork as remote "fork" (https fetch URL), which
# scripts/build_llamacpp_docker.sh checks out and cleans; BASE_BIN: bin/ of 5b329df16's static native CPU build with
# test-llama-archs and llama-server; REPO: this repository
WORK=$WORK REPO=$REPO SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c740 BASE_BIN=$BASE_BIN LOCK=$WORK/lock JOBS=8 \
  bash docs/data/2026-11-11_c740_seq_rm_bound/run/checks.sh
```

The check and the timing by hand, with a CPU build of the branch:

```bash
test-llama-archs --kv-seq-rm -s 1                       # every architecture, the switch off and on
LLAMA_ARCHS_SEQ_RM_BOUND_ON=2 test-llama-archs --kv-seq-rm -a '^(llama|qwen4exp|gemma3)$' -s 1   # must fail
test-llama-archs --kv-seq-rm-time -s 1                  # the host time of one call
```

## Limits

- Every number here is from a build machine's CPU (a desktop Ryzen), a diagnostic: the rig's two-core Celeron has
  less cache and a slower memory, and R101 measures the stages there.
- The check compares the switch off and on in one process on the CPU backend, with the device-side mask and the QSA
  device tables in their CPU form. On the cards the same host state drives the same uploads; R101 checks the replies.
- `seq_cp`, `seq_keep`, `seq_add` and `seq_div` still visit every cell; the server calls them only to copy or clear a
  slot and to shift the context, not in every step.

## Provenance and licence

The models are `test-llama-archs`'s random fixtures (this fork's test code). The engine is Miner Llama (MIT, as
llama.cpp). The tables are this project's measurements.
