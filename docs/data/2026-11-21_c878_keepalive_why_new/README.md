# Why the reply's last pass stops the keep-alive at `=3`, and a debug line that names it (C878, Q1585; 2026-11-21)

What this is: the follow-up to [R195](../2026-11-17_r195_first_request_fix/README.md). With
`GGML_CUDA_KEEPALIVE_EAGER=3` ([C866](../2026-11-17_c866_keepalive_key/README.md)) the keep-alive ends a card's
clock-holding block only before an evaluation whose key is new to the card, and the key leaves the cache's length out.
In R195 the main model's last pass of a reply still ended the blocks on all nine cards at 6,144 tokens, which looked
like a size the key should have left out. This row reads why, from R195's own server log and profile, and adds a debug
switch, `GGML_CUDA_KEEPALIVE_TRACE`, that names the node and the field that make a key new. Fork branch
`pascal-p106-v5-keepalive-why-new` (`1d177d31e`, one commit on `pascal-p106-v5` `3dfad876f`), built and checked on a
build machine's CPU, then loaded once on nine P106-090 cards from `pascal-p106-v5` `af00bb13b`, the landing that carries
it ([P106 load](#p106-load-c926-2026-11-25), C926). Registry row Q1585 in
[quants/experiments.md](../../../quants/experiments.md).

## Result in plain words

- **No size is missed: the key worked as designed.** The reply's last pass was not the same pass at both depths. R195
  limits every reply to 256 tokens, so near the end the server drafts only as many tokens as are left: at 4,096 the
  last pass was two tokens (one token and one draft), at 6,144 one token (no room for a draft). The key keeps the token
  count, so these are two different keys, and each had run only once before, at the end of the warm-up request, at 93
  and 94 cells of cache. There the indexer's top-k runs another kernel and the sparse attention does not take them, so
  at 4,349 and 6,398 cells their picks are new on every card, and the stop is the one C866 meant to keep.
- **So the 117 ms R195 estimated is not there to take from the key.** These short passes come only when a reply is cut
  by its token limit, once a server for each width and pick; what removes their stops is loading those kernels before
  the first turn (C868, registry Q1567) or remembering per card which picks it has run (C869, Q1568).
- **The debug line is built.** With `GGML_CUDA_KEEPALIVE_TRACE=1` each key new to a card gets one log line: the
  card's earlier key that differs in the fewest nodes, how many nodes differ, the first one (index, name, op, type,
  sizes) and the field that differs there with its value before and now, then up to four more. Unset, it costs
  nothing: the key is computed by the same code as before. All checks on the CPU passed, and the card code is the
  base's kernel for kernel.
- **On a P106 the line says what this reading said.** At 4,096 tokens the reply's two-token last pass is new on each of
  the nine cards against the warm-up's two-token pass, and the differences are only the indexer's top-k path and the
  sparse attention's pick; at 6,144 the same holds for the one-token pass against the warm-up's one-token pass. The
  replies are R195's, token for token, and no line warns that the trace misses a field.
- **One size the key keeps, on the last card only.** At 6,144 the MTP draft's store of hidden rows on card 8 grows from
  4,097 to 8,194 rows (its second half is made when the first prompt longer than a batch arrives, here the 6,144-token
  prompt). Every graph on card 8 that writes or reads that store is then new once: four keys at 6,144 differ from an
  earlier one only by that row count. The draft's shortlist head adds such keys too, each time its list of further ids
  grows (183 to 222 ids in this load). Both are new ideas (Q1869, Q1870).

## What R195's log and profile show

[run/short_passes.py](run/short_passes.py) reads a server log at `-lv 5` (the slot's "max possible draft",
"generate_draft" and "decode" lines) and the graph-mode profile of the same load, lists every generation pass shorter
than a full verify pass (one token and three drafts), and matches card 0's evaluations of rarely run graphs to them by
time. R195's `=3` load ([short_passes-k1.tsv](short_passes-k1.tsv)):

| Request | Pass | Tokens | Cache cells before it | Card 0: graph, how, launch groups, GPU ms |
|---|---|---:|---:|---|
| warm-up (32 tokens) | third to last | 3 | 92 | not written (its graph never ran again) |
| warm-up | second to last | 2 | 93 | graph 4, eager, 311, 21.06 |
| warm-up | last | 1 | 94 | graph 5, eager, 310, 25.36 |
| 4,096, first request | last | 2 | 4,349 | **graph 4, eager, 311, 21.39** |
| 4,096, repeat | last | 2 | 4,349 | not written (the profiler writes an evaluation when its graph runs again) |
| 6,144, first request | last | 1 | 6,398 | **graph 5, eager, 310, 20.95** |
| 6,144, repeat | last | 1 | 6,398 | not written |

- The two clocks line up to within 25 ms for all four written evaluations (the log's clock is the profile's plus
  284.180 s). R195's `=1` load gives the same passes and graphs ([short_passes-e1.tsv](short_passes-e1.tsv)).
- At 8,192 both requests end with a full verify pass: no shorter pass runs, as R195 found.
- Graph 4 (two tokens) and graph 5 (one token) differ in their token count and by one launch group on card 0; the full
  verify pass's graph also has 311.
- The repeats end with a pass of the same width as their first requests and stop no block (R195's counts), because
  their keys were made a few seconds before.

## The debug switch

`GGML_CUDA_KEEPALIVE_TRACE` (fork `ggml/src/ggml-cuda/keepalive-trace.h`, read with `GGML_CUDA_KEEPALIVE` on and
`GGML_CUDA_KEEPALIVE_EAGER` at `1`, `2` or `3`):

| Value | What |
|---|---|
| unset, empty, `0`, not a number | off: the key is computed as before, nothing is recorded |
| `1` | a line for each key new to a card; up to 16 MiB of recorded fields kept a card |
| `N` > 1 | the same, up to N MiB a card (at most 4,096) |

- **How:** `ggml_cuda_keepalive_shapes` became `ggml_cuda_keepalive_shapes_impl<trace>`. Both forms mix the same
  fields in the same order: per node its op and type, sizes, strides and op parameters, and its sources' types, sizes
  and strides; at `=3` in the ops with sizes along the cache's length those sizes as one mark, packed strides as
  another, and what the length picks in the attention and the top-k. The traced form also records each field with what
  it is, node by node. With the switch unset only the untraced form runs (the switch is read once).
- **The line:** a new key's record is compared node by node with each earlier key's on the same card; the one with
  the fewest differing nodes is named, then the first differing node and its first differing field. For example, from
  the CPU test: `key 0x… new at EAGER=3 (the card's new key 3, 8 nodes, 4 recorded): nearest earlier key 0x… (new key
  1, 8 nodes, 4 recorded), 1 nodes differ; first node 3: the top-k's path 0 -> 2`; on a card the node is named as
  `node <index> '<name>' (<op> <type> [<ne0>, <ne1>, <ne2>, <ne3>])`. A size the key leaves out at `=3` shows as
  "(along the cache's length, left out)", so it can never be the difference.
- **A check of itself:** each record is mixed again into a key; if that is not the key the keep-alive used, the line
  is preceded by a warning that the trace misses a field.
- **Memory:** past the budget the oldest keys keep only one hash a node: they can still be the nearest key and name
  the node, but not the field. At most 4,096 keys a card.

## Build checks

| Check (C878 `1d177d31e` against base `3dfad876f`) | Result |
|---|---|
| Rig build (`scripts/build_llamacpp_docker.sh`, `TESTS=1`, sm_61, `HF_UI_VERSION=b12483`) | rc 0; 38 warning lines on both, the same set; highest glibc symbol GLIBC_2.38 |
| `test-cuda-keepalive` of the rig builds, driver stub, no card ([keepalive_tests.txt](keepalive_tests.txt)) | base 252 checks, 0 failed; C878 294 checks, 0 failed (the 42 new ones check the switch's values, the nearest key, the first differing field and its name, a node count change, the budget and the key from a record) |
| Native CPU build of every target ([cpu_summary.tsv](cpu_summary.tsv)) | rc 0; 26 warning lines, none in the new code |
| ctest, 75 tests, no network ([ctest.tsv](ctest.tsv)) | 71 pass, test by test the base's result (C908's table); `test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` fail on both (the base's known four) |
| `cuobjdump -sass` of `llama-server` ([sass_summary.txt](sass_summary.txt)) | 8,836 of 8,836 kernels identical, none new or missing: the two per-kernel tables are the same file (9,510 rows, the 176 `kq_fp32` kernels included) |

Rig tarball `llama.cpp-20261121-1d177d31e-pascal-p106-v5-keepalive-why-new.tar.gz`, 841,574,915 bytes, md5
`35cbaa370246abc38879d5c886c148ff` (`bin/llama-server` md5 `5df53fdaa3c6a21aacb510d5b67be266`,
`bin/test-cuda-keepalive` `3a9912e6a7bcacaab256cae90f2a5a3a`).

## P106 load (C926, 2026-11-25)

One untimed load of [run/c878.sh](run/c878.sh) on nine P106-090 cards at stock clocks: R195's deep lane with
`GGML_CUDA_KEEPALIVE_EAGER=3` and `GGML_CUDA_KEEPALIVE_TRACE=1`, Flash-Next's v4 preset unchanged (UkisAI's Swift-1.5
GSQ-RCO Q2_0 file with its MTP draft), a warm-up load, then the first request and its repeat at 4,096 then 6,144 tokens,
each after its prime, temperature 0, 256 tokens. Engine: `pascal-p106-v5` `af00bb13b` (rig tarball md5
`b51ad2740d5822e7e14af656c413ec22`, `llama-server` md5 `f5cca34ea6bd4f3712cd6193bf0605be`), which carries this
branch's commit since the landing of 2026-11-24 ([2026-11-24_land_v5_batch2](../2026-11-24_land_v5_batch2/README.md)).

The checks set before the load:

| Check | Result |
|---|---|
| No warning that a record misses a field | none in either load's log; 101 trace lines in the traced load |
| At 4,096, the reply's two-token last pass: on each card the warm-up's two-token key as nearest, a pick as the difference | **9 of 9 cards**: nearest the warm-up's two-token key (93 cells), first difference the indexer's top-k path (1 -> 2), then the sparse attention's pick by the cache's length |
| At 6,144, the one-token last pass: the same against the warm-up's one-token key | **9 of 9 cards**: nearest the warm-up's one-token key (94 cells), first difference the top-k path (1 -> 4), then the attention's pick; on card 8 one more node, the MTP store's row count (below) |
| The same replies as R195's `=3` load | **all ten requests** of both loads give R195's text and token ids (the md5s of [p106_requests.csv](p106_requests.csv) and R195's [requests.csv](../2026-11-17_r195_first_request_fix/requests.csv)), with the same drafted and accepted counts |

The last pass's lines card by card ([p106_trace_lines.tsv](p106_trace_lines.tsv), column `last_of_reply`):

| Card | 4,096: nodes that differ | 6,144: nodes that differ | Beyond the top-k path and the attention's pick |
|---:|---:|---:|---|
| 0, 2, 4, 5, 7 | 2 | 2 | none |
| 1, 3, 6 | 4 | 4 | none (two indexed attention layers on these cards, so each pick differs twice) |
| 8 | 2 | 3, and a second line | at 6,144: `h_nextn_dev` (`SET_ROWS`), rows 4,097 -> 8,194; the second line is the draft's catch-up graph (72 nodes), whose one difference from the warm-up's one-token key is the same store read by a `GET_ROWS` (source rows 4,097 -> 8,194) |

- **The store's row count is the one size the key keeps.** `h_nextn_dev` holds the target's hidden rows for the MTP
  draft on the output card, in halves of `n_batch` + 1 rows; with `LLAMA_MTP_HROW_LAZY=1` (the preset's,
  [C813](../2026-11-12_c813_hrow_half/README.md)) the second half is made only when the first prompt longer than a batch
  arrives. Here that is the 6,144-token prime (6,020 new tokens, two calls at `-b 4096`). From then on every card-8
  graph that writes or reads the store has a new key once: four keys at 6,144 differ from an earlier key only by that
  row count (the prime's call, the catch-up graph beside it, the first verify pass, and the catch-up graph of the last
  pass). The count changes once in a server's life, and only on the output card. Registry row Q1869.
- **The draft's shortlist head adds keys as its list grows.** Its further ids (beyond the first 32,768 token ids) are
  one `MUL_MAT_ID` whose row count is the list's length; that went 183 -> 184 at 4,096 and 184 -> 221 -> 222 at 6,144,
  and three of the draft's keys on card 8 differ from an earlier one only by that count. Registry row Q1870.
- **The repeats stop nothing:** no trace line falls in either repeat, as R195's counts showed.
- **Where all 101 lines come from:** the warm-up request's prompt (30: each card's first keys) and its passes of 4, 3, 2
  and 1 tokens (1, 10, 10, 10: the token count), its draft steps (3); at 4,096 the prompt's 4-token tail piece (9: the
  top-k path 1 -> 2 on every card), the draft (3), the first verify pass on card 8 (1) and the last pass (9); at 6,144
  the prime's call on card 8 (2: the store), the draft (2), the first verify pass on card 8 (1: the store) and the last
  pass (10).

Card 0's short passes, from this load's server log and profile with [run/short_passes.py](run/short_passes.py)
([p106_short_passes-k1.tsv](p106_short_passes-k1.tsv)), are R195's: the same passes, the same graphs (4 and 5, eager)
and the same launch groups (311 and 310). Their GPU times were logged with the trace and `-lv 5` on and are not a
speed result.

## Files

| File | What |
|---|---|
| [short_passes-k1.tsv](short_passes-k1.tsv), [short_passes-e1.tsv](short_passes-e1.tsv) | R195's `=3` and `=1` loads: every pass shorter than a verify pass, and card 0's evaluations of rarely run graphs matched to them |
| [keepalive_tests.txt](keepalive_tests.txt) | `test-cuda-keepalive`'s result lines, base and C878 |
| [ctest.tsv](ctest.tsv), [cpu_summary.tsv](cpu_summary.tsv) | ctest per test, the base's and C878's side by side; the CPU test set's verdicts |
| [sass_summary.txt](sass_summary.txt) | the card code comparison |
| [run/run.sh](run/run.sh) | the build machine's check (steps src, rig, ka, cpu, sass) |
| [run/short_passes.py](run/short_passes.py) | the short passes of a load from its server log and profile |
| [run/c878.sh](run/c878.sh) | the rig load with the trace on |
| [p106_trace_lines.tsv](p106_trace_lines.tsv) | the P106 load's 101 trace lines, each with its request, step (prompt, draft or main pass), the pass's width and cells, whether it is the reply's last pass, the card, the key, the nearest earlier key and where that key was first made, how many nodes differ, the first field and the line's node text |
| [p106_requests.csv](p106_requests.csv) | the P106 load's requests (warm-up load and traced load): prompt and generation times, drafted and accepted tokens, the reply's md5s |
| [p106_short_passes-k1.tsv](p106_short_passes-k1.tsv) | the traced load's short passes and card 0's evaluations of rarely run graphs, as for R195 |
| [p106_gp-k1.csv.gz](p106_gp-k1.csv.gz) | the traced load's graph-mode profile (every card's evaluations) |
| [run/trace_lines.py](run/trace_lines.py) | the trace lines of a load placed in their requests and steps |

## Commands

On the build machine, from a folder holding the scripts (`REPO`: `run.sh`, `scripts/build_llamacpp_docker.sh`,
[2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh), C861's
`sass_compare.py` and C854's `sass_classify.py`), with the base's clone, rig build, build log and ctest table of
[2026-11-21_c908_v5_merged_build](../2026-11-21_c908_v5_merged_build/README.md):

```bash
W=<work dir> SEED=<llama.cpp clone holding 3dfad876f> BASE_RIG=<3dfad876f's rig build folder> \
  BASE_RIG_LOG=<its build log> BASE_CTEST=<its ctest table> REPO=<scripts> LOCK=<flock file> JOBS=8 bash run.sh
```

Every container carried `--cpu-shares=256` and `--memory=16g`; everything that ran the built code ran without network.
The short-pass tables, from R195's raw server logs (not published) and its published profiles:

```bash
python3 -I run/short_passes.py <R195's server-k1.log.gz> ../2026-11-17_r195_first_request_fix/gp-k1.csv.gz > short_passes-k1.tsv
python3 -I run/short_passes.py <R195's server-e1.log.gz> ../2026-11-17_r195_first_request_fix/gp-e1.csv.gz > short_passes-e1.tsv
```

The rig load, from the repository's root on a machine with ssh to the rig:

```bash
RIG=<rig> WORK=<run folder on the rig> BIN=<C878's llama-server on the rig> \
  MODEL=<Swift Q2_0's first shard on the rig> MODEL_DRAFT=<the MTP draft on the rig> TEXT_LOCAL=<recorded agent text> \
  STEP=start bash docs/data/2026-11-21_c878_keepalive_why_new/run/c878.sh
# STEP=wait, then STEP=collect OUT=<here>; the trace lines: zgrep ggml_cuda_keepalive_trace <OUT>/server-k1.log.gz
```

The P106 load ran that way with `BIN` the `af00bb13b` tree's `llama-server` (`DEPTHS` and the preset as in the
script). Its tables, from the collected folder (`OUT`):

```bash
python3 -I run/trace_lines.py <OUT>/server-k1.log <OUT>/requests-k1.csv > p106_trace_lines.tsv
python3 -I run/short_passes.py <OUT>/server-k1.log <OUT>/gp-k1.csv > p106_short_passes-k1.tsv
cp <OUT>/requests.csv p106_requests.csv; gzip -9nc <OUT>/gp-k1.csv > p106_gp-k1.csv.gz
```

## Provenance

The fork's commit is public on [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama). The short-pass
tables come from R195's load on nine P106-090 cards (its README has the build, file and settings), the `p106_` files
from the P106 load above (C926, a Manli M-P106L9-N6G with nine P106-090, at stock clocks). The base's rig build, build
log and ctest table are C908's. Raw logs are not published.
