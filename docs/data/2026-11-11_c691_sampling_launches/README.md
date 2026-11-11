# C691 (Q1223): the verify pass's sampling chain in fewer launches (2026-11-11, the night's code rows)

Miner Llama serves Qwen3.8-Flash-Next with backend sampling (`-bs`): the card that holds the output head (card 8 of
nine P106-090) also picks the next tokens, so the 248,320 logits of a row never cross the slow link. In a served
speculative step the verify pass outputs four rows (the token and three drafted ones), and each row runs its own
sampler chain. R49 profiled that chain at 0.68 ms of GPU time per verify pass, in both requests: a `PAD` of the logits
0.22 ms, four `TOP_K` launches 0.33 ms, and 20 smaller launch groups a row after each, 0.13 ms
([2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md)).

This folder holds the change that takes the four rows' top-k in one op and drops the `PAD`, behind a new switch on
Miner Llama's v4 line, and its CPU checks. The registry row is Q1223 in
[quants/experiments.md](../../../quants/experiments.md); the switch is listed in [docs/switches.md](../../switches.md).
The check on a P106 and the time it saves are the rig row Q1300.

**Result:** with `LLAMA_SAMPLING_TOP_K_ROWS=1` the chains of a verify pass share one `TOP_K` over the output rows and
one gather of their logits, and read the logits without the padded copy. On a build machine's CPU the sampler gives
the same output with the switch off and on, bit for bit: every sampled token, probability, logit and candidate of 85
outputs across five pairs of chains, among them the presets' order. The switch-off graph has the base branch's node
lines, and the switch-on graph has no `PAD` and one `TOP_K` where the chains had one each. All 682 operator cases
pass. The rig tarball builds, and 114 CUDA cases are recorded for a P106. Not run on a card yet: whether the verify
pass saves its est. 0.39 ms is rig row Q1300.

## Where the 0.68 ms goes

`llm_graph_context::build_sampling` (in `src/llama-graph.cpp`) builds the sampling part of every graph that has
backend samplers:

- **The `PAD`.** It first copies the logits into a tensor one row longer (`ggml_pad(t_logits, 0, 1)`): four rows of
  248,320 F32 values (3.97 MB) read and five rows (4.97 MB) written. The extra row gives every chain a row to view even
  in a micro-batch without any output, so that the graph keeps one shape whatever rows a micro-batch outputs.
- **A chain per output.** Every sequence's sampler builds `n_outputs_max_per_seq` chains (four with the preset's
  three drafted tokens), each on a 1-D view of its row. With the preset's settings (`--top-k 20 --top-p 0.95`, and a
  greedy or a sampled request) a chain is: three samplers that do nothing at their settings (penalties, DRY,
  top-n-sigma), a top-k (one `TOP_K` over the 248,320 values, then a `GET_ROWS` of their logits), a top-p (an argsort,
  two gathers, a softmax, a cumulative sum and the mask's 9 small ops), the temperature and the final pick.
- **One `TOP_K` launch group is ten kernels.** At this width CUDA takes the radix select (`top_k_radix_cuda`): an init,
  four histogram and select passes of 8 bits, and a gather. For one row, about 0.08 ms; R49's four were 0.085, 0.082,
  0.079 and 0.082 ms.

## The change

Branch `pascal-p106-v4-b12246-c691`, two commits on `pascal-p106-v4` (`84f7bd3d2`): `e39723391` (the change and its
tests) and `d18c71761` (an include the CPU build of the test needs).

| File | What |
|---|---|
| `src/llama-cparams.h`, `src/llama-context.cpp` | `sampling_top_k_rows`, read from `LLAMA_SAMPLING_TOP_K_ROWS` when a context is made; a log line when it is on |
| `src/llama-sampler.{h,cpp}` | `llama_sampler_chain_backend_top_k`: the k of a chain whose first backend sampler that adds to the graph is a top-k (empty samplers before it are skipped), and the index of the sampler after it; `llama_sampler_chain_backend_apply_from`: the chain's backend samplers from that index on |
| `src/llama-graph.cpp` | `build_sampling`: with the switch on and every chain starting with a top-k, one `TOP_K` per k over all the output rows and one `GET_ROWS` of their logits; each chain gets views of its row of the two and runs the rest of its samplers; the logits are not padded unless the micro-batch has no output |
| `tests/test-backend-ops.cpp` | `TOP_K` cases at 248,320 and 151,936 columns over 1 to 8 rows; `test_top_k_rows` (`TOP_K_ROWS`); perf cases of 1, 2 and 4 rows and of the `PAD` |
| `tests/test-backend-sampler.cpp` | `test_top_k_rows_switch`: the same decodes with the switch off and on, compared bit for bit |

**The switch:** `LLAMA_SAMPLING_TOP_K_ROWS=1`; unset or `0` is the default. With it off `build_sampling` builds
`pascal-p106-v4`'s graph node for node (the same `PAD`, the same views, each chain's own `backend_apply`).

**When it applies.** Every sequence's chain must start with a top-k on the backend, with only empty samplers before it
(the server's default order at the presets' settings), and the logits must be one contiguous F32 tensor. Otherwise
the whole graph keeps each chain's own top-k and the `PAD`. Chains of different k share nothing: one `TOP_K` per
distinct k, each over all the rows.

**Why every row gets the same candidates and logits:**

- **Each row of a multi-row `GGML_OP_TOP_K` is computed on its own.** The CPU sorts each row (`std::partial_sort`). On
  CUDA at a row of more than 4,096 columns, with CUDA 12.9's CUB (no `DeviceTopK`, which needs CCCL 3.4.3), one row
  and several rows both take the radix select: the same 64 blocks a row, the same histograms, the same threshold, and
  the same tied columns at the k-th place (the lowest). So each row's set of k indices is the one its own top-k gives.
  Within a row, the order of the values above the threshold comes from an atomic counter and varies from run to run,
  with one row as with several.
- **What follows depends on the set, not its order.** The `GET_ROWS` copies each candidate's logit as it is. Top-p
  sorts the candidates by logit (an argsort) before its softmax and cumulative sum; the temperature's and greedy's
  argmax and the draw read that sorted list. Only candidates with exactly equal logits can change places, and they
  already do from run to run without the switch.
- **Nothing after a top-k writes to the logits.** The two samplers that write into their input in place (the penalties'
  and the logit bias's `SET_ROWS`) only exist before a top-k, and a chain that starts with one keeps the old graph. So
  the chains can read the logits directly; the `PAD` stays only for a micro-batch without outputs (the views need a
  row there), where no chain is computed.
- **Chains without a row of the micro-batch** view row 0 and are not computed, as before.
- **The one op's two tensors are graph outputs.** A chain whose later samplers leave the candidates or the logits as
  they are (a top-k alone on the backend, or a draw right after it) returns views of them; flagged as outputs, the
  allocator keeps them to the end of the graph.

At a row of 4,096 columns or fewer CUDA takes one row's top-k by a sort and several rows' by the radix select: the same
set wherever the values are distinct, but with ties at the k-th place the two may keep different tied columns. Real
vocabularies are far wider; the test model's vocabulary is 150,144.

**What it saves (est.):** est. 0.39 ms of the 0.68 a verify pass, est. 0.3 % of S37's 131.57 ms step. The `PAD`'s 0.22
ms goes. One four-row `TOP_K` reads the four rows five times (19.9 MB, est. 0.115 ms at the rig's measured 173 GB/s)
on top of the launches that one row's select also pays (est. 0.05 ms of its 0.08), est. 0.165 ms against the four
launches' 0.33. Three `GET_ROWS` launches of the four go too (R49: 0.008 ms in all). A candidate for the "Negligible
gains" group if the rig cannot separate it from the spread.

## The checks

All on a build machine's CPU, with no GPU, in containers at the lowest CPU priority, one at a time behind the
machine's shared lock, on 2026-11-10 and 2026-11-11 from 23:29 to 01:14 ([checks.tsv](checks.tsv)):

| Check | Result |
|---|---|
| the rig build (`TESTS=1`) | builds: `llama.cpp-20261110-d18c71761-pascal-p106-v4-b12246-c691.tar.gz`, 647,609,710 B, md5 `777e3e858347099e2ae79b3f9459fcf5`, 0 AVX instructions, GLIBC_2.38 at most |
| `test-backend-ops -b CPU -o TOP_K,TOP_K_ROWS`, the rig build's CPU code against itself | 682 of 682 OK: 607 `TOP_K`, 75 `TOP_K_ROWS` ([cpu_cases.tsv](cpu_cases.tsv)) |
| `test_top_k_rows_switch`, off against on | 5 pairs of chains, 17 outputs each: the same bytes in every output ([switch_test.txt](switch_test.txt)) |
| `test-backend-sampler`, every test in a process of its own | 21 of 22 pass with the switch unset and 21 of 22 with it on for the whole process; the base passes 20 of 21. The one failure is `test_logit_bias` in all three, on the same token: it expects a +10 bias to win, which it does not on random weights ([sampler_tests.tsv](sampler_tests.tsv)) |
| the graphs of `test_multi_output_cpu_suffix` (`GGML_SCHED_DEBUG=2`) | the base and the switch off: the same 4,491 node lines; on: 4,455, with 18 `TOP_K` against 27, no `PAD` against 18, 171 `GET_ROWS` against 180 ([graph_nodes.tsv](graph_nodes.tsv)) |
| the recorded set, replayed on the CPU | 39 of 39 `TOP_K` and 75 of 75 `TOP_K_ROWS` OK, none missing ([cpu_replay_cases.tsv](cpu_replay_cases.tsv)) |

**`test_top_k_rows_switch`** (in `tests/test-backend-sampler.cpp`) makes two contexts per pair of chains, the first
with the switch at `0` and the second at `1` (it is read when a context is made), each with two sequences, up to four
outputs a sequence, and the same five batches: 6 and 4 outputs, 1 and 3, a batch with no output, 4, then 1 and 2 with
the second sequence first in the batch (17 outputs). It compares each output's sampled token, probabilities, logits
and candidates byte for byte, and the token that `llama_sampler_sample` returns. The five pairs:

| Pair | Sequence 0 | Sequence 1 | What it covers |
|---|---|---|---|
| 0 and 1 | the server's order at the presets' settings: empty penalties and top-n-sigma, top-k 20, empty typical, top-p 0.95, empty min-p, temperature 0.7, a draw | top-k 40, top-p 0.95, temperature 0, greedy (a greedy request) | two k, so two `TOP_K` ops; the empty samplers before the top-k |
| 0 and 0 | the same | the same | one `TOP_K` for both sequences |
| 2 and 3 | top-k 8 on the backend, then typical-p and a draw on the host | top-k 20, a draw | the candidates and logits returned as views of the one op; the host's part of a chain |
| 3 and 2 | the same, swapped | | |
| 0 and 4 | the presets' order | a logit bias, top-k 20, a draw | a chain that starts with another sampler: the graph keeps each chain's own top-k |

The model is a tiny Flint with random weights (Q111's, 6 layers of 256, the 150,144-token vocabulary), so the
logits are spread wide and ties are rare; the test runs on the CPU, where each row's top-k is the same code with one
row or several. **On a card, the identity of the rows is the recorded set's job:**

- **`TOP_K_ROWS`** (`test_top_k_rows` in `tests/test-backend-ops.cpp`) takes the top-k of n rows in one op and each
  row's own top-k on the tested backend, sorts both (the order inside a row is not specified) and outputs the
  difference, which is zero when every row picks the same columns. The reference is zero too, so any difference fails.
  It runs at 248,320 and 151,936 columns, 1 to 8 rows, k = 1, 20 and 40, with distinct values and with runs of seven
  equal values that straddle the k-th place (so the op has to choose among tied columns); and at k = 512, and at
  4,096 columns without ties.
- **`TOP_K`** at the same widths, rows and k against the CPU (the indices as sets; with ties, their values).

**The recorded set** (`scripts/record_checks.sh SUITE=top-k-rows`, made with the rig build's own `test-backend-ops`,
md5 `e8959071c177a8806c809175f0c77a27`): 114 cases, the 39 `TOP_K` cases at 248,320 columns and all 75 `TOP_K_ROWS`
cases, 371,482,360 B of files; the tar is 371,793,920 B, md5 `185aca24175a7a998955483c2389327f`. Two runs, both must
pass ([set_runs.tsv](set_runs.tsv)); the cases, their sizes and md5s are in [set_manifest.tsv](set_manifest.tsv).

## The rig row (Q1300)

On one rig, est. 75 minutes (pad: 90): the relay est. 10 (a 648 MB tarball and the 372 MB set), the replay est. 5,
the operator times est. 5, R49's served profile with the switch off, on, on, off, each a warm and a profiled load,
est. 48, collecting est. 5. Paths are variables: `RIG` (the rig's ssh alias), `BUILD` (the build machine's), `W` (the
build folder there), `R` (a folder on the rig), `RES` (a results folder on the build machine). The model, the draft,
the preset, the server script and the request are R49's. From a machine that only runs ssh, at the repository's root:

```bash
# 1. relay the tarball, the set and the scripts to the rig (through this machine)
T=llama.cpp-20261110-d18c71761-pascal-p106-v4-b12246-c691
ssh $BUILD "cat $W/w/$T.tar.gz" | ssh $RIG "mkdir -p $R/set && tee $R/$T.tar.gz | md5sum && tar xzf $R/$T.tar.gz -C $R"
ssh $BUILD "cat $W/sets/top-k-rows.tar" | ssh $RIG "tar xf - -C $R/set"
tar -cf - scripts/replay_checks.sh docs/data/2026-11-11_c691_sampling_launches/run/rig_c691.sh \
  docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh | ssh $RIG "tar -C $R -xf -"

# 2. on the rig, detached: the replay (every run "ok"), TOP_K and PAD times, the served verify pass off/on/on/off
ssh $RIG "cd $R && BIN=$R/$T/bin OUT=$R/out SET=$R/set/top-k-rows REPLAY=$R/scripts/replay_checks.sh \
  MODEL=<Swift Q2_0 first shard> DRAFT=<the MTP draft> PRESET=<release/flash-next-v3/settings.env> \
  RIG_SERVER=<bench/agent_check/rig_server.sh> REQ=<2026-11-04_draft_one_card/run/greedy_request.json> \
  VERIFY_PROF=$R/docs/data/2026-11-10_served_verify_parts/run/verify_prof.sh \
  setsid -f nohup bash $R/docs/data/2026-11-11_c691_sampling_launches/run/rig_c691.sh > $R/console.out 2>&1 < /dev/null"
ssh $RIG "tail --pid=\$(cat $R/out/PID) -s 15 -f /dev/null; cat $R/out/status.txt"

# 3. on the build machine, CPU only: each profiled load's trace cut as R49's (run/verify_parts.py of
#    2026-11-10_served_verify_parts, with R41's plain passes as PLAIN), the replies compared across the loads
ssh $RIG "cd $R/out/served && tar cf - */prof/prof-verify.csv" | ssh $BUILD "mkdir -p $RES && tar -C $RES -xf -"
V="nice -n 19 python3 -I docs/data/2026-11-10_served_verify_parts/run/verify_parts.py \
  --kpass-parts docs/data/2026-11-04_kpass_profile/run/kpass_parts.py --prof-summary scripts/prof_summary.py"
for d in 1-off 2-on 3-on 4-off; do $V --event-us 7.6 --plain $PLAIN --out $RES/$d $RES/$d/prof/prof-verify.csv > $RES/$d/report.md; done
```

What it decides:

- **The replay passes, or the change stops there.** Run `tk-rows` is the one that shows each row of one op is the
  row's own top-k on a P106.
- **The `on` loads' server logs** have the switch's line (`rig_c691.sh` counts it), and every load gives the greedy
  request's reply text (temperature 0).
- **The sampling chain per verify pass** (`verify_parts.py`'s part after the head's product): `on` against R49's 0.68
  and against the `off` loads, each listed by value; the profiled groups show one `TOP_K` and no `PAD`.
- **`perf/us_run.tsv`** gives one four-row `TOP_K` against four one-row ones and the `PAD` on the card itself.
- **The class:** the warm loads' step times, `on` against the two `off` loads' spread. A saving inside the spread
  (est. 0.3 % of a step) makes it **negligible**: it is listed in `QUEUES.md`'s "Negligible gains", kept in the
  cumulative build behind its switch and measured with that group (C534). A gain goes to the v4 preset candidate.

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | every check: command, build, setting, result |
| [cpu_cases.tsv](cpu_cases.tsv) | `test-backend-ops -b CPU -o TOP_K,TOP_K_ROWS`: the 682 cases and their verdicts |
| [sampler_tests.tsv](sampler_tests.tsv) | `test-backend-sampler`, each test's exit code, the branch with the switch unset and on, and the base |
| [switch_test.txt](switch_test.txt) | `test_top_k_rows_switch`'s lines |
| [graph_nodes.tsv](graph_nodes.tsv) | the node counts of one test's graphs: the base, the switch off and on |
| [set_manifest.tsv](set_manifest.tsv) | the recorded set: per case its id, case string, input and reference bytes and md5s |
| [set_runs.tsv](set_runs.tsv) | the set's two runs for `scripts/replay_checks.sh` |
| [cpu_replay_cases.tsv](cpu_replay_cases.tsv) | the recorded set replayed on the CPU: 114 cases and their verdicts |
| [run/chain.sh](run/chain.sh) | the build machine's chain: build, CPU builds, operator cases, sampler tests, graphs, record, replay |
| [run/rig_c691.sh](run/rig_c691.sh) | the rig run: replay, the operator times, the served verify pass off and on |

## Commands

On the build machine (CPU only; no GPU), from a checkout of this repository, with the branch on the fork remote of
`$WORK/w/src` (a llama.cpp clone with an https fetch URL for the fork):

```bash
WORK=<a folder of its own> PUB=<this repository> MODEL=<a small GGUF, here a tiny Flint with random weights> \
  LOCK=<a flock file shared with the machine's other jobs> DOCKER_ARGS="--label <your label>" \
  nice -n 19 bash docs/data/2026-11-11_c691_sampling_launches/run/chain.sh
```

[run/chain.sh](run/chain.sh) builds the rig tarball (`TESTS=1 scripts/build_llamacpp_docker.sh
fork/pascal-p106-v4-b12246-c691`, `-j 8`, CPU shares 256, `nice -n 19`), static native CPU builds of
`test-backend-sampler` and `test-backend-ops` for the branch and for `pascal-p106-v4` from the same clone, runs
`test-backend-ops -b CPU -o TOP_K,TOP_K_ROWS`, every sampler test on its own for the branch (switch unset and on) and
the base, one test's graphs for the three, records the set with `OUT=$WORK/sets SUITE=top-k-rows BIN=<tarball>/bin bash
scripts/record_checks.sh` and replays it on the CPU. `STEPS` picks steps (default all, in order); the sampler step
ran a second time with `STEPS=sampler` after the first run had all tests in one process, where `test_logit_bias`'s
failure ended the rest.

## Provenance

Code: the fork's branch `pascal-p106-v4-b12246-c691` (MIT, as llama.cpp). The operator checks use random tensors
generated by `test-backend-ops`. `test-backend-sampler` ran on a tiny Flint with random weights (6 layers,
256-dimensional, Flint's 150,144-token vocabulary, 167 MB; made for Q111's tests); no trained weights or user data
are involved. The R49 numbers quoted above are from
[2026-11-10_served_verify_parts](../2026-11-10_served_verify_parts/README.md).
