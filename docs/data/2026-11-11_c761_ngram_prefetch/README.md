# The prompt's n-gram row reads off the host's path (C761, Q1339, 2026-11-11)

What this is: a change on the v4 line that issues the n-gram table's row reads of a prompt call's later micro-batches
from a helper thread, while the earlier micro-batches compute, instead of just before card 0's launch.
[R81](../2026-11-11_r81_ubatch_trace/README.md) traced Flash-Next's prompt micro-batches on a P106 with the v4 night
merge (`c408dabcb`): a fresh 512-token micro-batch held the host a median 291.3 ms before card 0's launch, 198.3 ms of
it in setting the graph's inputs (`inputs`), against 11.4 ms when the same prompt was read again. In the v4 source the
n-gram input (`llm_graph_input_qwen4exp_ple::set_input`) calls `llama_prefetch_rows`, a `madvise(MADV_WILLNEED)` on each
token's rows of the 28.8 GB table, which stays on the disk and is read through the page cache (`--lazy-mode`): 16 rows of
90 bytes a token, nearly every one on its own page, so est. 8,400 page reads for a 512-token micro-batch, issued and
waited for right before the cards get the micro-batch.

The change is on the fork branch `pascal-p106-v4-b12246-c761` (`cbc253131`, three commits on
`pascal-p106-v4-b12246-c748` `5b329df16`), behind `LLAMA_NGRAM_PREFETCH`, off by default. This page is the change and
its checks on a CPU; the rig run that times it is R106 ([Q1351](../../../quants/experiments.md), below).

## Result

**With `LLAMA_NGRAM_PREFETCH=1` the helper thread issues the n-gram rows of every prompt call's later micro-batches, and
everything the model computes stays the same: `test-llama-archs --ngram-prefetch` gives the same logits, byte for byte,
and per decode call the same rows (count and digest) and the same distinct pages as the inline path, at three seeds;
llama-server gives the base build's tokens, top probabilities, draft counts and 406 verify steps; no helper row ever
differed from the n-gram input's own, and the helper never had more than the cap issued ahead. With the switch off the
branch gives `5b329df16`'s logits on every model.** The checks, on a build machine's CPU ([summary.tsv](summary.tsv),
[npf.tsv](npf.tsv), [npf_calls.tsv](npf_calls.tsv), [server_calls.tsv](server_calls.tsv)):

- **`test-llama-archs --ngram-prefetch`** ([npf.tsv](npf.tsv), [npf_calls.tsv](npf_calls.tsv)): the random-weight
  `qwen4exp` model (Flash-Next's architecture; its table has 4 rows a token) written to a file and loaded with the table
  mapped (`--lazy-mode on`), then a server's calls in micro-batches of 16: a one-token warm-up, a 120-token prompt (8
  micro-batches), a 70-token continuation (5), three one-token steps, a 40-token call (3) and a second sequence's
  50-token prompt (4); every token's logits kept. The switch off (the inline path, counted with
  `LLAMA_NGRAM_PREFETCH_STATS=1`) against three arms: the default cap (48 MiB), 64 KiB (a few tokens' rows of this
  table) and 0. At seeds 1, 2 and 3, every arm gives the off run's logits byte for byte; each of the 8 calls has the off
  run's micro-batches, rows (1,136 in all), distinct pages (147, 148 and 145 at the three seeds) and row digest; the 4
  calls of several micro-batches are planned; 0 rows differed. At the default cap the helper issues 864 of the 1,136
  rows (every micro-batch after each call's first), at 64 KiB 228, 220 and 236 (the window full at 65,536 of 65,536
  bytes), at 0 none.
- **llama-server** ([run/server_check.sh](run/server_check.sh), [server_calls.tsv](server_calls.tsv)): the random
  `qwen4exp` model with one MTP block, its table mapped, draft-mtp at three tokens with `p_min` 0, `-b 512 -ub 64`, four
  requests (a 200-token and a 600-token prompt, each greedy and at temperature 1.0 with top-k 20 and top-p 0.95). The
  switch unset, on, on at 64 KiB, and the trace with it off and on, each give the base build's (`5b329df16`) tokens,
  top-3 probabilities, drafted and accepted counts, and all 406 verify steps of `LLAMA_SPEC_DUMP`. With the trace, the
  off and on loads log the same 425 calls with the same micro-batches, rows (12,868), distinct pages (6,083) and digest
  per call; on the on load the helper issued 4,128 of the rows in the 4 planned calls (the server sends the prompts as
  calls of 132, 64 and 4 tokens and of 512, 20, 64 and 4: the 132- and 512-token calls have several micro-batches),
  set_input 8,740, none differed, and the
  helper was at most 7,757,824 bytes ahead of a 50,331,648-byte cap. At 64 KiB the helper issued 248 rows and was at
  most 65,536 bytes ahead.
- **With the switch off, on or counting** (nothing in that test maps a table), `test-llama-archs -b CPU -s 1` gives
  `5b329df16`'s logits on 146 of 146 random models: the n-gram input's rows now come from the function the helper
  shares, and they are the same. **ctest: 70 of 70 pass** (the static CPU build; the tests that need a download, the
  network or torch are left out, as in C565).
- **The rig tarball builds:** `llama.cpp-20261111-cbc253131-pascal-p106-v4-b12246-c761.tar.gz`, 656,490,506 B, md5
  `188297ade26e157ca0361d9011c27c1d` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the
  tree's `bin/`). Its own `test-llama-archs --ngram-prefetch` passes on the CPU at the three seeds as the CPU build's
  does (the `rig` rows of [npf.tsv](npf.tsv)).

## How it works

- **When the call is known.** `llama_context::decode` hands the helper the call's tokens once the memory module has
  split it into micro-batches, with the first micro-batch's size and the n-gram's predecessors of the call's first
  token, read from the attention cells as the input reads them (`seq_pos_tok_le`). The call is planned when it is a
  token batch of one sequence at consecutive positions with more than one micro-batch, in a context whose n-gram input
  has run once (the server's warm-up), on a model whose table is mapped.
- **The helper.** It waits until the call's first micro-batch has issued its own rows in `set_input` (so that the second
  micro-batch's reads do not share the disk with the first's, which the cards wait for), then takes the call's other
  tokens in order, 32 at a time: computes each token's 16 rows with the function the input now calls too
  (`llama_ngram_prefetch::token_rows`), claims the tokens, and issues their rows with `llama_prefetch_rows`.
- **The window.** The helper claims a token only while the page spans of the rows it has issued for tokens that
  `set_input` has not reached yet, plus the token's own, fit in `LLAMA_NGRAM_PREFETCH_MB` (default 48 MiB). A
  Flash-Next micro-batch of 512 tokens needs est. 32.7 MiB (512 × 16 rows × 1.022 pages a 90-byte row × 4 KiB), so
  the default holds the next micro-batch and part of the one after it, under a quarter of R81's lowest MemAvailable
  (196 MiB). The pages are clean file pages that the kernel can drop; the cap keeps the helper from reading so far ahead
  that they push the server out to swap or are dropped before use.
- **The same reads.** `set_input` still computes every row and uploads them for `ggml_get_rows`. For its micro-batch it
  compares the rows of the tokens the helper claimed with its own, issues the rows of the tokens the helper had not
  claimed (the call's first micro-batch, and whatever the cap left), and issues any token whose rows differ (none in any
  check; counted). The bytes the model reads come from the same mapped file, so the switch changes only when the reads
  are issued.
- **The trace** (`LLAMA_UBATCH_TRACE=1`): each `ubatch-trace` line ends with `inputs_by = <input>:<us>,...`, every
  input type's time in `set_input` (`qwen4exp_ple` is the n-gram input), and `ngram = <us>/<rows>/<helper rows>`, the
  n-gram input's own prefetch call, the rows it issued there and the rows the helper had issued; each main-model call
  gets an `ngram-prefetch: call` line (its rows, helper and inline, the rows that differed, its distinct pages and bytes,
  the bytes passed to madvise, the row digest, the most bytes the helper had ahead and the cap). The trace counts the
  rows with the switch off too, so the two runs compare call by call ([run/ngram_calls.py](run/ngram_calls.py)).
  Counting costs the trace runs a sort of each micro-batch's pages inside the n-gram input (est. under 1 ms a 512-token
  micro-batch on the rig's Celeron); without the trace and the switch nothing is created and the input prefetches as
  before. With the switch on the context logs its cap when it is made and a closing `C761:` line with the calls planned
  and the rows each side issued.

## What it changes on the rig (est.)

- R81's fresh 4,097-token prompt (taper off: calls of 2,048, 1,533, 512 and 4 tokens) has 8 micro-batches of about 512
  tokens; 5 of them are not a call's first (3 in the 2,048-token call, 2 in the 1,533). If their reads finish while the
  cards work, each loses most of its `inputs` (R81's median 198.3 ms): est. up to 5 × 198.3 = 991.5 ms of the host's
  time before card 0 in that prompt (R81: 1,459.1 ms of `inputs` in request 7, 1,278.2 in request 9). In an agent
  turn's prompt call each token past the call's first micro-batch saves est. up to its first-read cost, which R81
  measured at 648.4 µs a token in turn 2 (1,355 tokens) and 308.7 in turn 4 (27 tokens).
- The first micro-batch of each call still reads inline; the 512-token call and the 4-token call have nothing to move.
  Those first micro-batches are where the cards wait for the host (est.: a call ends with the cards drained), so their
  reads are worth the most: reading call 2's and call 3's first rows during the call before needs the server to pass the
  prompt's later tokens down, which is its own row ([Q1353](../../../quants/experiments.md)).
- Whether the reads finish in time depends on how far the host runs ahead of the cards: R81 saw the host hold 8,872.2 ms
  in the third micro-batch's launch on cards 1 to 7 (a full launch queue), time the helper can read in. The table and the
  swap file share the rig's SSD ([C752](../2026-11-11_c752_long_request_stalls/README.md)), so the rig row records
  MemAvailable, swap and paging per load.

## The rig row (R106, Q1351)

R81's driver ([R42's `r42_run.sh`](../2026-11-10_prompt_timeline/run/r42_run.sh), `R42_LOADS=warm4096`: the preset with
the MTP draft, the replay cut to one task, then two fresh 4,097-token prompts with the taper off and on) twice on rig 1,
the switch off then on, both under `LLAMA_UBATCH_TRACE=1`, through [run/rig_c761.sh](run/rig_c761.sh). Before each
load the table's file leaves the page cache (`posix_fadvise(POSIX_FADV_DONTNEED)` on that file alone), so that both
loads read their fresh prompts' rows from the disk; [run/vm_sampler.sh](run/vm_sampler.sh) logs MemAvailable, swap and
paging every 0.5 s. Est. 2 × 8.5 min plus the relay.

```bash
# relay the tarball and unpack it next to the other builds
ssh "$QUANT" "cat $WORK/rig/llama.cpp-20261111-cbc253131-pascal-p106-v4-b12246-c761.tar.gz" | \
  ssh "$RIG" "cat > $BUILDS/c761.tar.gz && tar -xzf $BUILDS/c761.tar.gz -C $BUILDS && rm $BUILDS/c761.tar.gz"
# this folder's run/ scripts to the rig, then the driver, detached
ssh "$RIG" "mkdir -p $RIGWORK/r106" && for f in rig_c761.sh vm_sampler.sh; do ssh "$RIG" "cat > $RIGWORK/r106/$f" < run/$f; done
timeout 15 ssh "$RIG" "cd $RIGWORK/r106 && R42=$RIGWORK/r42 R43=$RIGWORK/r43 BUILDS=$BUILDS \
  BUILD=llama.cpp-20261111-cbc253131-pascal-p106-v4-b12246-c761 MODEL=<Swift Q2_0 first shard> DRAFT=<MTP draft> \
  TABLE=<the shard that holds the table> OUT=$RIGWORK/r106/out setsid nohup bash rig_c761.sh > $RIGWORK/r106/run.out 2>&1 < /dev/null &"
# when out/DONE exists: the tables on a build machine
python3 -I run/c761_read.py <tables> off=out/off/r42/warm4096/load-A/server.log,out/off/vm.tsv \
  on=out/on/r42/warm4096/load-A/server.log,out/on/vm.tsv
```

[run/c761_read.py](run/c761_read.py) writes per arm and request the prompt's calls, the sums and medians of `inputs`
and of the n-gram input's time over its micro-batches of at least 128 tokens (each call's first apart from the later
ones), the rows the helper and set_input issued, and per arm the lowest MemAvailable and the swap at the start, highest
and end with the swap traffic. Read requests 7 and 9 (the two first reads of a fresh prompt): `qwen4exp_ple`'s share of
`inputs` with the switch off confirms that `inputs` is the n-gram input; with it on, the later micro-batches' `inputs`
and the prompt's time say what moved. The switch is a gain if the later micro-batches' `inputs` drop and the prompt
times with it on are below the off load's, with the swap no higher.

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail |
| [npf.tsv](npf.tsv) | `test-llama-archs --ngram-prefetch`, one row per seed and arm, with the CPU build (`tip`) and the rig build on the CPU (`rig`) |
| [npf_calls.tsv](npf_calls.tsv) | the same runs, one row per decode call and arm: tokens, micro-batches, planned, rows (helper, inline, differed), distinct pages, madvise bytes, digest, the most bytes ahead |
| [server_calls.tsv](server_calls.tsv) | llama-server with the trace, switch off and on: one row per call with both loads' `ngram-prefetch` fields |
| [server_verdict.txt](server_verdict.txt) | the server check's verdict lines |
| [run/checks.sh](run/checks.sh) | the build-machine checks (builds, `--ngram-prefetch`, server, the off identity, ctest, the rig tarball) |
| [run/server_check.sh](run/server_check.sh) | the llama-server arms and their verdict |
| [run/ngram_calls.py](run/ngram_calls.py) | two traced server logs compared call by call |
| [run/rig_c761.sh](run/rig_c761.sh), [run/vm_sampler.sh](run/vm_sampler.sh) | the rig row's driver and its memory sampler |
| [run/c761_read.py](run/c761_read.py) | the rig row's tables from its server logs and memory samples |

## Commands

On a build machine with Docker, as one job under a lock shared with other builds (the paths come from the environment):

```bash
WORK=<work dir> REPO=<this repository> SRC=<llama.cpp clone, the fork as remote "fork"> \
  BRANCH=pascal-p106-v4-b12246-c761 BASE_CPU=<5b329df16's static CPU build bin/> \
  BASE_SRV=<5b329df16's llama-server dir> LOCK=<lock file> JOBS=8 \
  bash docs/data/2026-11-11_c761_ngram_prefetch/run/checks.sh
```

Every container runs with `--cpu-shares=256` and the builds and tests at `nice -n 19`, without a GPU. The models are
random weights made by `test-llama-archs`; nothing is downloaded. Licence: the scripts and tables are this repository's.
