# A drafted token's n-gram rows read ahead as it is drafted (C615, Q1100, 2026-11-11)

What this is: a change on the v4 line that starts the disk reads of a speculative step's verify pass while the MTP
draft is still drafting. Flash-Next's 28.8 GB n-gram table stays on the SSD, read through the page cache (`--lazy-mode`:
the table is mapped and its rows read on demand). Each token adds 16 rows of 90 bytes from it to the first layer. The
n-gram input asks for a micro-batch's rows (`llama_prefetch_rows`: `madvise(MADV_WILLNEED)` on each row) when it sets
its input, just before the CPU split that gathers them and before card 0 starts. In a verify pass those rows belong to
tokens the drafter chose a few milliseconds earlier, so they are rarely in the page cache.
[R81](../2026-11-11_r81_ubatch_trace/README.md#before-card-0-in-a-verify-pass) timed ten four-token verify passes on a
P106 rig (the v4 night merge `c408dabcb`): the CPU split before card 0 took a median 4.379 ms, against 0.076, 0.077 and
0.076 ms in the passes whose rows were already cached. That is the largest host cost left in a decode step
([R86](../2026-11-11_r86_step_trace/README.md): 7.512 ms between the last drafting pass and the verify pass).

The ids are known earlier than that. The verify pass of a step reads the rows of the token sampled at the end of the
last step (`id_last`) and of each drafted token; at three drafted tokens, `id_last` is known three drafting passes
ahead, and each drafted token one to two passes ahead. The change issues each token's rows as soon as the drafter has
it, so the reads run while the draft's card works.

The change is on the fork branch `pascal-p106-v4-b12246-c615` (`08c2d98c6`, two commits on
`pascal-p106-v4-b12246-c748` `5b329df16`), behind `LLAMA_DRAFT_PREFETCH`, off by default. This page is the change and its
check on a CPU; the rig run that times it is [Q1354](../../../quants/experiments.md) (below).

## Result

**With `LLAMA_DRAFT_PREFETCH=1` llama-server gives the same tokens, top probabilities, draft counts and verify steps as
the base build on every request, and every one of the 406 verify passes finds all its n-gram rows already issued by
the drafter (6,448 of 6,448 rows), from four prefetch calls a step: `id_last` and the three drafted tokens. With the cap
at 0 nothing is issued and nothing is found ahead, with the same replies. With the variable unset the branch gives
`5b329df16`'s logits on every model.** The checks, on a build machine's CPU ([summary.tsv](summary.tsv),
[server.tsv](server.tsv), [prefetch.tsv](prefetch.tsv), [server_verdict.txt](server_verdict.txt)):

- **llama-server with a random-weight `qwen4exp` (Flash-Next's architecture) model with one MTP block and its n-gram
  table mapped (`-lm mmap -lzm on`), draft-mtp at three tokens with `p_min` 0
  ([run/server_check.sh](run/server_check.sh)):** the switch unset, `=1`, `=1` with its counts, `=1` with a zero cap
  and the counts alone each give the base build's (`5b329df16`) tokens, top-3 probabilities, drafted and accepted
  counts, and all 406 verify steps of `LLAMA_SPEC_DUMP` (the draft, the drafter's candidates and probabilities, the
  target's top tokens) on four requests (two greedy, two at temperature 1.0 with the preset's top-k 20 and top-p 0.95);
  so does `=1` with the backend sampler (`-bs`, as the preset) against the base with `-bs`.
- **Every row ahead:** in each arm with the switch on and its counts, each of the 406 evaluations that follow a draft
  finds every one of its rows in the drafter's window (6,448 of 6,448). Each step made 4 calls (1,624 in all) that
  issued the rows of 4 tokens: 398 four-token verify passes, and 4 three-token and 4 two-token ones at the replies' ends,
  where the server cut the draft below three tokens after the drafter had issued all three (the only rows issued and
  not read). 6,420 rows issued, 76 skipped as already issued in the same window, no token over the cap; at most 81,920
  bytes of pages in one window (this model has 4 rows a token; Flash-Next has 16). The calls took 7.085 ms in all, 4.36
  µs a call on this machine with the pages cached. The other evaluations of 2 to 8 tokens are the prompts' last
  micro-batches (4 tokens each, which no draft precedes).
- **The control:** with `LLAMA_DRAFT_PREFETCH_KB=0` the 1,624 calls find every token over the cap, issue nothing, and
  no evaluation finds a row ahead. With the counts alone (switch unset) no call is made and the 429 evaluations are
  counted.
- **The rig build:** the rig tarball's own `llama-server` on the CPU (its CUDA backend loads against the stub driver
  and finds no card; its CPU code has no AVX) gives with the switch on and its counts the same tokens, probabilities,
  draft counts and 406 verify steps as with it unset, and every verify pass's rows ahead.
- **With C701's trace:** `LLAMA_UBATCH_TRACE=1` with the switch off and on gives the base's replies and verify steps;
  the rig row's readout ([run/c615_read.py](run/c615_read.py)) pairs each of the 429 `draft-prefetch` lines with its
  evaluation's `sched-trace` and `ubatch-trace` lines ([trace_summary.tsv](trace_summary.tsv)). On this machine every
  table page is in the page cache (the 31 MB model file), so `resident` is 5,739 of 5,739 pages in both arms: the page
  cache's part is the rig's to show.
- **A test merge with C761** (`pascal-p106-v4-b12246-c761` `cbc253131`, the n-gram rows of a prompt call's later
  micro-batches issued by a helper thread; `git merge-tree`, tree `cc25dfdd8`, no conflict): its CPU build gives the
  base's replies and verify steps with nothing set, with this switch on, and with both switches on, every verify pass's
  rows ahead in both; C761's own `test-llama-archs --ngram-prefetch` passes at seeds 1, 2 and 3 with this switch and
  its counts on. The branch also merges with C738 (`b45b4afd1`) and C741 (`be9db2460`) without a conflict
  (`git merge-tree`; not built).
- **With the variable unset, `=1`, and `=1` with its counts** (`test-llama-archs`'s random models keep their tables
  resident, so nothing calls the prefetch), `test-llama-archs -b CPU -s 1` gives `5b329df16`'s logits on 146 of 146
  random models. **ctest: 70 of 70 pass** (the static CPU build; the tests that need a download, the network or torch
  are left out, as in C565).
- **The rig tarball builds:** `llama.cpp-20261111-08c2d98c6-pascal-p106-v4-b12246-c615.tar.gz`, 656,437,007 B, md5
  `fab9f5c56fe4df9bead0c62191fcaa41` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the
  tree's `bin/`).

## What it changes on the rig (est.)

From R81's and R86's sizes, before the rig row measures it:

- **Est. up to 4.3 ms a step:** R81's CPU split before card 0 in a verify pass was a median 4.379 ms with the rows read
  from the SSD and 0.076, 0.077 and 0.076 ms with them cached; if every verify pass finds its rows cached, a step saves
  the difference (4.379 − 0.076 = 4.303 ms). `id_last`'s rows get three drafting passes of lead and the drafted tokens'
  two and one; the last drafted token's 16 rows get the host's stages between the last drafting pass and the verify
  pass's CPU split (R86: 1.483 ms from the last drafting pass's wait to the inputs set, then the decode up to the
  split). Read in parallel on the SATA SSD (152 µs a 4 KiB read one at a time,
  [Q214](../2026-11-03_table_storage/README.md)), est. they arrive in time.
- **The calls' cost:** four calls a step, each the row hash of 16 rows and one `madvise` per new page range. Three of
  them run while a drafting pass computes, where the host waits for its sampling anyway; the fourth runs after the last
  pass, doing the `madvise` work the verify pass's n-gram input did there before (it now finds those pages cached).
- **Memory:** est. 0.26 MB of pages a step, the pages the verify pass reads a few ms later anyway (below).

## How it works

- **`llama_draft_prefetch(ctx, seq_id, pos0, tokens, n_tokens, i_first)`** (`src/llama-ext.h`, new file
  `src/llama-draft-prefetch.{h,cpp}`): `tokens` are the tokens a later decode of `ctx` will put at positions `pos0`,
  `pos0 + 1`, … of `seq_id`; the rows of `tokens[i_first…]` are issued (through `llama_prefetch_rows`), the earlier
  ones only serve as predecessors. A token's rows depend on the token and its two predecessors (Flash-Next's n-grams of
  2 and 3 tokens, 8 heads each); the predecessors before `pos0` come from the context's attention cells, as the n-gram
  input reads them (`seq_pos_tok_le`). The row hash is the input's own loop, line for line. Supported by `qwen4exp`
  with its table mapped (lazy); `llama_draft_prefetch_enabled(ctx)` is false otherwise or with the switch unset, and the
  drafters then call nothing.
- **draft-mtp** keeps per sequence `id_last` and the tokens drafted so far, in the verify pass's order. `id_last`'s rows
  are issued once the first drafting pass is queued, each drafted token's once the next drafting pass is queued, and
  the last drafted token's after the last pass. A queued pass leaves the host waiting for its sampling, so the
  `madvise` calls run while the draft's card works rather than on the step's path; only the last token's calls stand
  where the verify pass's own `madvise` stood before.
- **The other drafters** (draft-simple, EAGLE3, DFlash): `common_speculative_draft` issues a finished draft's rows
  (`id_last` and the draft) once the draft is known, when the target context has the table and the drafter did not
  issue them itself.
- **The window:** the rows issued since the last evaluation whose n-gram input set the context's rows. A row issued in
  the window is not issued again, and at most `LLAMA_DRAFT_PREFETCH_KB` (default 1024) KiB of pages are issued in one
  window: a token whose new pages would pass the cap is not issued, nor any later token of that call (the verify pass
  then reads it as before). The n-gram input closes the window each time it sets rows.
- **The bytes read are the same:** the n-gram input still computes, prefetches and uploads its own rows, and the graph
  gathers them as before. Only when the reads are issued changes.
- **Counts:** with `LLAMA_DRAFT_PREFETCH_STATS=1` or C701's `LLAMA_UBATCH_TRACE`, the n-gram input logs one line per
  evaluation (the switch off too), before that micro-batch's `sched-trace` and `ubatch-trace` lines. One from the CPU
  check (the random model has 4 rows a token):

  ```
  draft-prefetch: n_tokens = 4 rows = 16 ahead = 16 pages = 16 resident = 16 | calls = 4 tokens = 4 issued = 16 dup = 0 capped = 0 bytes = 65536 call_us = 19 | us = 19
  ```

  `rows` the evaluation's rows, `ahead` how many of them the window had issued, `pages` their distinct pages and
  `resident` how many of those were in the page cache when the input had set its rows (`mincore`, for evaluations of up
  to 64 tokens; `-1` otherwise); then the window's prefetch calls, tokens issued, rows issued, rows already issued,
  tokens over the cap, page bytes and the calls' time; `us` is the counting's own time. When the context is freed, a
  `C615:` line gives the totals.

### The 4 GB host's memory

The reads go into the page cache as clean, file-backed pages that the kernel can drop at any time; nothing is pinned or
allocated on the heap but a few hundred row ids. At three drafted tokens a step issues the rows of four tokens: 64 rows
of 90 bytes, so at most 64 pages plus the few rows that straddle a page edge (est. 2.2 % of rows), est. 0.26 MB a step,
and less where rows are cached already. Every one of them is a row the verify pass reads a few milliseconds later
anyway, since the verify pass takes every drafted token, accepted or not; the only extra reads are the drafted tokens
the server cuts from a draft at the end of a reply (its draft cap). So the switch moves when those pages enter the page
cache, not how many do. The cap (1 MiB a window by default) bounds a window even for a drafter with a long draft.

## The checks

All on a build machine's CPU, in containers at low priority, one at a time ([run/checks.sh](run/checks.sh)):

| Check | What | Verdict |
|---|---|---|
| `cpu` | a static native CPU build of the branch, every target | OK (3 warning lines, as `5b329df16`'s) |
| `server` | llama-server, 8 arms ([server.tsv](server.tsv), [prefetch.tsv](prefetch.tsv)) | PASS |
| `off` | `test-llama-archs -b CPU -s 1` against `5b329df16`'s CPU build: unset, `=1`, `=1` with the counts | 146/146 the same logits, three times |
| `ctest` | ctest in the static CPU tree | 70/70 |
| `merge` | `git merge-tree` with `-c761` `cbc253131`, its CPU build, 3 server arms, `--ngram-prefetch` at seeds 1–3 with this switch on | no conflict; OK; PASS; 3/3 |
| `build` | the rig tarball, `TESTS=1` | OK, md5 `fab9f5c56fe4df9bead0c62191fcaa41` |
| `rsrv` | the rig build's `llama-server` on the CPU, unset and `=1` with the counts | PASS |
| `trace` | `LLAMA_UBATCH_TRACE=1` off and on, then `c615_read.py` on both logs ([trace_summary.tsv](trace_summary.tsv)) | PASS; 429 evaluations paired in each |

The test merge's `--ngram-prefetch` log shows none of the engine's lines (the test keeps them out of its output, C761's
own too); the test's verdict is its own comparison of logits and rows.

## The rig row (Q1354, R108)

Not run yet. One untimed run on a P106 rig with the c615 tarball (below), R86's method with the switch as the variable:
[run/rig_loads.sh](run/rig_loads.sh) starts the v3 preset (`release/flash-next-v3/settings.env`, the MTP draft at three
tokens) through `bench/agent_check/rig_server.sh` with the host profile (`GGML_CUDA_PROFILE_HOST`), six loads in this
order, each answering S37's greedy request (79 prompt tokens, 256 generated) and then the agent's first-turn request
(2,760 prompt tokens, 256 generated):

| Load | Adds |
|---|---|
| `trace` | C405's step trace (`LLAMA_STEP_TRACE`) and C701's `LLAMA_UBATCH_TRACE=1` (which turns this switch's counts on): the switch off |
| `tracedp` | the same with `LLAMA_DRAFT_PREFETCH=1` |
| `prof`, `dp`, `prof-2`, `dp-2` | the profile alone: off, on, off, on |

Before every load the driver drops the n-gram table's file from the page cache (`dd iflag=nocache count=0`: that
file's clean pages only) and logs its resident bytes with `fincore`, so that every load reads its rows from the SSD as
a fresh session's new n-grams are; otherwise the second load would find the first load's rows (the requests are the
same). C761's [vm_sampler.sh](../2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh) logs MemAvailable, the page cache,
swap and the major faults every 0.5 s through each load.

```bash
BIN=<the tarball tree's bin/> MODEL=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> \
  DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> TABLE=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00003-of-00003.gguf> \
  PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
  VM_SAMPLER=docs/data/2026-11-11_c761_ngram_prefetch/run/vm_sampler.sh \
  REQ_GREEDY=<S37's greedy_request.json> REQ_LONG=<S37's long_request.json> OUT=<dir> \
  LOADS='trace tracedp prof dp prof-2 dp-2' \
  LOAD_ENV='trace:LLAMA_UBATCH_TRACE=1 tracedp:LLAMA_UBATCH_TRACE=1,LLAMA_DRAFT_PREFETCH=1 dp:LLAMA_DRAFT_PREFETCH=1 dp-2:LLAMA_DRAFT_PREFETCH=1' \
  setsid nohup bash docs/data/2026-11-11_c615_draft_prefetch/run/rig_loads.sh > console.out 2>&1 < /dev/null &
```

Then on a build machine: [run/c615_read.py](run/c615_read.py) on the two trace loads' server logs (per verify pass:
the CPU split's time, the evaluation's time to card 0's launch, card 0's copies, the rows issued ahead and the pages
resident); R86's `align_t0.py` and C405's `step_gap.py` on them (the gap before each verify pass); S37's `step_split.py`
on every load and its `arms.py` with `prof` and `prof-2` as the reference (the greedy request draws nothing at random,
so its steps pair up); and per load the lowest MemAvailable, the most swap and the major faults from `vm.tsv`.

What it must show: every load's greedy reply equal to `prof`'s; in `tracedp` the verify passes' rows issued ahead
(`ahead` = `rows`) and the `C615:` line; the verify passes' CPU split in `tracedp` against `trace` (R81: a median 4.379
ms with the rows on the disk, 0.076 cached) and the share of their pages resident when the input set its rows; the
paired greedy step of `dp` and `dp-2` against `prof` and `prof-2`, classed against the spread of the two `prof` loads
(gain, negligible or loss); MemAvailable and swap no worse with the switch on. Est. up to 4.3 ms a step.

The tarball: `llama.cpp-20261111-08c2d98c6-pascal-p106-v4-b12246-c615.tar.gz`, 656,437,007 B, md5
`fab9f5c56fe4df9bead0c62191fcaa41` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in its `bin/`).
Est. 50 minutes of rig time (six loads of about 6.7 minutes, as R86's two took 13.4, and the relay).

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail, as the run wrote them |
| [server.tsv](server.tsv) | `server_check.sh`: per arm and request, prompt and predicted tokens, drafted and accepted, the reply's tokens |
| [prefetch.tsv](prefetch.tsv) | `server_check.sh`: per arm, the `draft-prefetch` lines summed (the server, merge and trace runs) |
| [server_verdict.txt](server_verdict.txt) | the server runs' verdict lines |
| [trace_summary.tsv](trace_summary.tsv) | `c615_read.py` on the CPU's trace arms (switch off and on) |
| [run/checks.sh](run/checks.sh) | the build machine's checks, step by step (header: usage) |
| [run/server_check.sh](run/server_check.sh) | the server arms and requests (header: usage) |
| [run/c615_read.py](run/c615_read.py) | the readout of a load's log: each evaluation's prefetch counts beside its CPU split and card 0's copies |
| [run/rig_loads.sh](run/rig_loads.sh) | the rig row's driver: R86's loads with the table dropped from the page cache before each |

## Commands

On a build machine with Docker (no GPU used): a clone of llama.cpp with the fork as remote `fork` in `$WORK/rig/src`, the
base's static native CPU build (`5b329df16`, every target) in `$BASE`, this repository in `$WORK/repo`:

```bash
WORK=<work dir> REPO=$WORK/repo SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c615 \
  BASE_CPU=$BASE/bin BASE_SRV=$BASE/tree/build/bin LOCK=<flock file> JOBS=8 \
  DOCKER_ARGS='--label pascal-quants=1' bash $WORK/repo/docs/data/2026-11-11_c615_draft_prefetch/run/checks.sh
```

The CPU image is Ubuntu 24.04 with git, cmake, build-essential, ccache, python3, curl and jq.

Provenance: the models are `test-llama-archs`'s random-weight models (no download); the server's requests are token
lists from a fixed generator (in `server_check.sh`). Licence: the repository's.
