# The fork's KV removal on upstream's walk (C881, Q1590, 2026-11-21)

What this is: Miner Llama's `llama_kv_cache::seq_rm`, the call that removes positions of a sequence from the KV
cache (every speculative step makes it to drop rejected draft tokens, the server makes it to drop what a cancelled or
replaced request left), rebuilt on upstream llama.cpp's own walk. Upstream
[#32380](https://github.com/ggml-org/llama.cpp/pull/32380), merged into the fork with upstream's tag b12483, gave the
cells a `llama_kv_cells::seq_rm(seq_id, p0, p1)` that walks the sequence's index of (position, cell) pairs instead of
every cell of the cache, the walk the fork already did behind its switch `LLAMA_KV_SEQ_RM_BOUND`
([C740](../2026-11-11_c740_seq_rm_bound/README.md)). The fork kept its own loop, because it does two things around
each removal that upstream does not, and so it failed upstream's new test `test-kv-cache`. The registry row is Q1590
in [quants/experiments.md](../../../quants/experiments.md).

Fork branch `pascal-p106-v5-kv-seq-rm` at `003bf4023`, two commits on `pascal-p106-v5` (`6644c2bc6`): `492287ced`
the change and the test's one-line adjustment, `003bf4023` a trace in `test-llama-archs --kv-seq-rm` (test code only).
Not merged into `pascal-p106-v5`: that follows a check on a P106 rig. Everything here ran on a build machine's CPU,
in Docker, with no network; nothing ran on a card.

## Result

**Upstream's `test-kv-cache` passes, and the rework keeps every cell, head, changed-cell list and logit of the
engine it came from.** Over C740's whole check (every architecture's random model, two contexts each, 400 random
decodes and memory edits per configuration), a hash of each context's memory after every edit and of its logits after
every decode is the same, line for line, in the two builds: 253,112 lines on the host mask, 252,533 with the mask built
from the device copies of the cells (the changed-cell list in use), 15,691 and 15,588 at seeds 2 and 3. llama-server
with random models gives the base's tokens on requests that remove KV ([below](#llama-server)).

**What failed the test was not the removal loop but the fork's `LLAMA_KV_HEAD_TAIL` rule**, on unless set to 0: after
the tail of a sequence is removed, the next token goes back into the cell of the first removed position, so that a
re-decoded tail takes the cells it had and attention sums its keys in the same order (pipelined speculation relies on
it for identical logits). Upstream's test expects the lowest freed cell instead. The current engine passes upstream's
test unchanged when run with `LLAMA_KV_HEAD_TAIL=0`, and aborts at that one assertion with the rule on. The fork's
behaviour is kept; the test now expects the fork's cell with the rule and upstream's with `LLAMA_KV_HEAD_TAIL=0`, and
passes both ways.

**Fewer fork lines.** Against upstream's b12483, the fork's lines in the KV cache's three files go from 783 added and
11 removed to 751 added and 12 removed, and in `seq_rm` itself from 106 added to 75. The one-sequence path now holds
upstream's own call, so an upstream change to that walk reaches the fork through a merge instead of a rewrite.

**No GPU code changed**, so no rig build was needed for these checks: the change is host code in `llama-kv-cache.cpp`
and `llama-kv-cache.h` plus two tests, and the device copies of the cells receive the same changed-cell list, byte for
byte (part of the traced memory). A P106 check before landing is listed under [Limits](#limits).

## The change

`llama_kv_cache::seq_rm` of one sequence (`seq_id >= 0`) now calls upstream's walk with or without the switch, and
keeps only what the fork adds around it:

| The fork's part | Before | Now |
|---|---|---|
| the removal | its own loop: every cell without `LLAMA_KV_SEQ_RM_BOUND`; with it, the sequence's index entries in cell order (sorted up to 64 out of order, past that the used range) | upstream's `llama_kv_cells::seq_rm(seq_id, p0, p1)`, which also returns the lowest freed cell |
| the changed-cell list of the device copies (`kq_cells_mark`, used by `LLAMA_GPU_KQ_MASK`) | each changed cell marked in cell order during the loop | read from the same index entries before upstream's walk, sorted into cell order, marked after it; only while the list is tracked, and at most 33 entries (past 32 changes the copies go stale whatever the order, as before) |
| `LLAMA_KV_HEAD_TAIL`'s cell | the freed cell of the lowest position, found during the loop | the first entry in position order whose cell holds no other sequence, read before the walk, only for a tail removal |
| `LLAMA_KV_HEAD_EMPTY` and `LLAMA_KV_HEAD_TAIL`'s head rules | as before | unchanged |

`LLAMA_KV_SEQ_RM_BOUND=1` now bounds only the removal from every sequence (`seq_id < 0`) to the used cells, as
before, and counts the one-sequence calls for `llama_memory_seq_rm_stats` (the count of calls that scanned the used
range stays 0: there is no scan any more). Its test hook `=2` leaves the changed-cell list in position order, which
the check must catch. The presets set the switch; with or without it the one-sequence removal is now the same code.

## The check

Two CPU builds of every target with the tests (`run/build.sh`): the reference, `pascal-p106-v5` `6644c2bc6` with only
the test's trace commit added (tree `736fe72a3`, made with `git merge-tree`), and the branch at `003bf4023`. Image
Ubuntu 24.04, gcc 13.3.0, cmake 3.28.3, `-DGGML_NATIVE=ON -DLLAMA_BUILD_TESTS=ON -DBUILD_SHARED_LIBS=OFF`, 8 jobs.

| Check | Reference | C881 |
|---|---|---|
| build | 0 errors, 30 warning lines, 135 s | 0 errors, the same 30 warning lines, 22 s (ccache) |
| ctest, 82 tests | 71 pass; 11 fail | **72 pass; 10 fail, all among the reference's 11** |
| `test-kv-cache` (upstream's, adjusted at one line on the branch) | aborts: `test-kv-cache.cpp:164: GGML_ASSERT(add(0, 6, 1) == 0)` | **passes** |
| `LLAMA_KV_HEAD_TAIL=0 test-kv-cache` | passes (the unchanged test) | passes |
| `test-llama-archs -s 1`, every architecture's random model on the CPU | 153 of 153 | 153 of 153; **the log the same bytes** but its first line (a timestamp) |
| `--kv-seq-rm -s 1`, every architecture (C740's check: the switch off and on in one process) | 442 of 442 | 442 of 442 |
| the same with `LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1` | 441 of 441 | 441 of 441 |
| the same on eight architectures, seeds 2 and 3 | 27 of 27, 27 of 27 | 27 of 27, 27 of 27 |
| the control, `LLAMA_ARCHS_SEQ_RM_BOUND_ON=2` with the device mask on llama, qwen4exp, gemma3 (must fail) | 10 of 12 fail | 9 of 12 fail, each on the memory's host state |
| **the traces of the four arms above, the reference against C881** | | **the same, line for line** (253,112, 252,533, 15,691 and 15,588 lines) |
| llama-server, random models, requests that remove KV | see [below](#llama-server) | |

The 10 ctest failures of both builds are the build setup's, not the code's: no network in the container
(`test-download-model`, `test-eval-callback-download-model`, `test-arg-parser`'s download stage; three tests Not Run
after them), no `numpy` in the image (`test-conversion-fp8`, `test-kv-cache-scales`, `test-jinja-py`), and two
template files of `test-jinja` at a path made relative by ccache. The reference's eleventh is `test-kv-cache`.

The tests left out of the `--kv-seq-rm` arms abort without the switch too and are the architectures' own (C835):
plamo2's two four-sequence tests, and with the device mask minimax-01's MoE test in four streams. Both builds left out
the same tests, so the traces cover the same runs.

The trace (`LLAMA_ARCHS_SEQ_RM_TRACE=<file>`, the branch's second commit) writes, after every edit, an FNV-1a hash of
each context's `llama_memory_debug_state` (each KV cache's heads, cells, per-sequence indexes and changed-cell list;
a QSA memory's pool tables), or of the whole state write where that does not cover the memory, and after every decode
a hash of each context's logits. The two columns are the switch off and on, so one file pair checks both against the
reference's two paths: the old loop over every cell (off) and C740's bounded walk (on).

## llama-server

`run/server_check.sh`, in a container with no network: random models saved by `test-llama-archs` with GPT-2's
tokenizer, each a target (stdev 0.1) and a drafter (0.13); llama-server with the drafter (`--spec-type draft-simple
--spec-draft-n-max 4`, probabilistic draft sampling), backend sampling (`-bs`, as the presets), `-c 4096`, six
threads. Two servers per arm:

- **llama** (attention only: every rejected draft token is removed from the KV cache), two slots in one stream
  (`-np 2 -kvu`), each request pinned to a slot, the prompt cache on: A1 slot 0 at temperature 0, 64 tokens; A2 slot 1,
  48 tokens, its cells after slot 0's; A3 a streamed request on slot 0 that the client leaves after 12 chunks (a cancel);
  A4 slot 0 with A1's prompt extended: the cached prompt kept (65 tokens cached, 11 processed), the rest of slot 0
  removed; A5 slot 1 at temperature 0.7 under a JSON tool-call grammar, its earlier tail removed.
- **qwen4exp** (Flash-Next's hybrid architecture: the server keeps no rollback snapshots with this draft type, so it
  saves a checkpoint before each verify pass and restores it after a rejection; 52 restores logged per arm): Q1
  temperature 0, 48 tokens; Q2 a cancel as A3; Q3 Q1's prompt extended after the cancel; Q4 temperature 0.7 under the
  grammar; Q5 temperature 0 under the grammar.

Arms: the reference twice (is the check repeatable?), the reference with `LLAMA_KV_SEQ_RM_BOUND=1`, C881, C881 with
`LLAMA_KV_SEQ_RM_BOUND=1`.

| Request | Tokens | Drafted / accepted | The five arms' token lists ([tokens.txt](tokens.txt)) |
|---|---:|---|---|
| A1 | 64 | 110 / 34 | **the same** (`264dcb870fb6`) |
| A2 | 48 | 82 / 25 | **the same** (`b3002b8815a5`) |
| A4, after the cancel | 48 | 68 / 29 | **the same** (`168e66664240`) |
| A5 | 34 (the grammar ended it) | 132 / 0 | **the same** (`05f3673d5f13`) |
| Q1 | 48 | 45 / 27 | **the same** (`e509d9ac9d8b`) |
| Q3, after the cancel | 48 | 42 / 33 | **the same** (`e6dbc6355589`) |
| Q4 | 39 | 76 / 0 | **the same** (`d8c4df591d6e`) |
| Q5 | 37 | 72 / 0 | **the same** (`410f2e74f726`) |

Every arm also drafted and accepted the same counts, cached the same prompt lengths ([server.tsv](server.tsv)), and
saw both cancels (12 chunks read, one cancelled task logged by each server).

## Files

| File | What |
|---|---|
| [verdicts.tsv](verdicts.tsv) | every check's verdict line, as `run/checks.sh` and `run/seqrm_loop.sh` wrote it |
| [tokens.txt](tokens.txt) | the server check: per request, the md5 of each arm's token list (first 12 hex digits) |
| [server.tsv](server.tsv) | the server check: per arm and request, tokens, cached tokens, drafted and accepted tokens, time |
| [run/build.sh](run/build.sh) | the two CPU builds with the tests and ctest, one container each under a build lock |
| [run/checks.sh](run/checks.sh) | `test-kv-cache` both ways, `test-llama-archs`, the `--kv-seq-rm` arms of both builds and the comparison of their traces |
| [run/seqrm_loop.sh](run/seqrm_loop.sh) | the `--kv-seq-rm` arms of one build with their traces (after C740's `bound_loop.sh`: an abort is rerun with the switch off and left out when it aborts there too) |
| [run/server_check.sh](run/server_check.sh) | llama-server with random models: the requests and the comparison of the tokens |

## Commands

On a build machine, CPU only, in Docker with no network (each build under a lock, the tests at low CPU priority):

```bash
# WORK: a work folder; $WORK/src: a llama.cpp clone with the fork (https://github.com/Cinderella-Man/miner-llama)
# fetched as remote "fork"; REPO: this repository
git -C $WORK/src fetch fork pascal-p106-v5 pascal-p106-v5-kv-seq-rm
B=$(git -C $WORK/src rev-parse fork/pascal-p106-v5); N=$(git -C $WORK/src rev-parse fork/pascal-p106-v5-kv-seq-rm)
t=$(git -C $WORK/src merge-tree --write-tree --merge-base $N~1 $B $N)       # the reference: the base + the trace commit
git -C $WORK/src update-ref refs/heads/c881-base $(git -C $WORK/src commit-tree $t -p $B -m reference)
R=$REPO/docs/data/2026-11-21_c881_kv_seq_rm/run
WORK=$WORK LOCK=$WORK/lock JOBS=8 bash $R/build.sh base=c881-base:ctest c881=fork/pascal-p106-v5-kv-seq-rm:ctest
WORK=$WORK bash $R/checks.sh
docker run --rm --network none -v $WORK:$WORK -e WORK=$WORK -e C=$WORK/srv <image with curl and jq> bash $R/server_check.sh
```

By hand, with a CPU build of the branch:

```bash
test-kv-cache                                      # the fork's head rules
LLAMA_KV_HEAD_TAIL=0 test-kv-cache                 # upstream's placement
LLAMA_ARCHS_SEQ_RM_TRACE=c881.trace test-llama-archs --kv-seq-rm -s 1   # then the same with the reference; cmp the files
```

## Limits

- Everything ran on a desktop CPU. The rework changes no GPU code and the device copies receive the same changed-cell
  list, but the branch lands in `pascal-p106-v5` only after a check on a P106 rig: a rig build of the branch, its
  `test-llama-archs --kv-seq-rm` with the device mask, and one agent-check pair against the current engine (Q1654).
- `--kv-seq-rm-time` (the host time of one call) was not run: both builds' one-sequence removal now walks the index,
  as C740's switch did, so the time is C740's "on" column whether the switch is set or not.
- The cancels of the server check are left out of the comparison: how many tokens a cancelled request makes depends
  on when the server sees the client hang up.

## Provenance and licence

The models are `test-llama-archs`'s random fixtures (this fork's test code) with the GPT-2 tokenizer file from the
llama.cpp tree. The engine is Miner Llama (MIT, as upstream llama.cpp). The tables and scripts here are this
repository's.
