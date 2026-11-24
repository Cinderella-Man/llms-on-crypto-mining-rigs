# The card's top-k candidates in a fixed order (C985, Q1718; 2026-11-24)

What this is: the fix that registry row Q1718 in [quants/experiments.md](../../../quants/experiments.md) asked for, and
its correctness check. [C962](../2026-11-23_c962_sampled_reply_gpu/README.md) found why a sampled reply changed between
loads with a fixed seed: with backend sampling (`-bs`), the chain top-k → temperature → dist sums the candidates'
probabilities in the order ggml's CUDA radix top-k writes them, and that order changes from call to call. The fix is a
new switch, `GGML_CUDA_TOP_K_SORT`, off by default, on the fork branch `pascal-p106-v5-topk-order` (`931d777f5`, one
commit on `pascal-p106-v5` at `8a9629942`). It was built and checked on a build machine's three RTX 3090s, which run the
rig build's compute 6.1 code through the driver's compiler. Correctness only: nothing here is timed. The cost on a
P106 is row Q1730.

## Result in plain words

- **With the switch, one seed gives one reply.** Flint Pascal Q8 GGUF's v5 preset at top-p 1 gave the same reply to
  C962's request in 15 of 15 rounds on three loads. Without it, the same build gave C950's two replies again (the
  second in 4 of 15 rounds). The card's top-k itself gave one order in 2,000 calls of every test row, always the sorted
  one; without the switch it gave 6 orders on the row of C962's 24th token, and three different tokens from one draw.
- **The reply with the switch is the sorted order's pick, token for token.** With the dist step's probabilities
  returned, all 256 tokens of the reply are the token the draw picks when the candidates come in descending order (at all
  3 positions where the order decides). Those probabilities are now the same bits in every round (6 rounds on two
  loads); without the switch they differ in the last bit between rounds of one load, at 37, 32 and 32 of 256 positions.
- **It is a different reply from before, and as random as before.** Where the order decides, the sorted order can
  pick another token than the card's usual order did (here at the 171st token), so sampled replies at top-p 1 change
  once, then stay fixed. Each candidate is still drawn with its own probability.
- **Nothing else changes.** Requests at top-p 0.95 (the presets' and the agent check's setting) and greedy requests gave
  the same replies with and without the switch. With the switch unset, the build is the base: every kernel copy of the
  card code is the base's, the logits of 148 random models are the same bits, ctest is the same test by test, and no
  warning is new.
- **One test needed it:** `test-backend-sampler`'s check of `LLAMA_SAMPLING_TOP_K_ROWS` compares two runs' candidates in
  order, so on a card it aborts without the switch and passes with it.
- **Its cost is not measured here.** It adds one small kernel launch for each top-k the card runs: each step's
  sampling and each pipelined guess's. Its time on a P106 is row Q1730.

## The change

With `GGML_CUDA_TOP_K_SORT=1`, a small kernel runs right after the card's radix top-k. One block per row sorts the
row's k candidates in shared memory: descending logit, as the host sampler sorts its candidates, and equal logits by
the lower token id. It uses a bitonic network over a strict order, so the result does not depend on the
order the radix gather wrote. The same sort follows CUB's DeviceTopK, whose order is unspecified too; that path exists
only in builds with CCCL 3.4.3 or later, not in the rig builds (CUDA 12.9), so it is not exercised here. The bitonic
and argsort paths of the top-k, which narrow rows take, already return their candidates sorted. Unset or `0`, nothing
changes.

The sort went into the kernel, not into `llama_sampler_top_k_backend_apply`, for three reasons:

- **One place covers every path.** The release presets set `LLAMA_SAMPLING_TOP_K_ROWS=1`, whose one top-k over all
  output rows (`src/llama-graph.cpp`) replaces each sampling chain's own top-k. A sort in the chain's sampler would miss
  the presets' path and would be needed twice. The kernel also covers the staircase drafter's top-k.
- **It is smaller.** One kernel launch per top-k and about 90 lines in one file (`ggml/src/ggml-cuda/top-k.cu`). In the
  sampler it would be an argsort of the k logits and gathers of the ids and logits, about six graph nodes per chain.
- **Ties come out by token id.** The card's argsort is a bitonic network, which is not stable, so equal logits would
  still follow the radix order unless the ids were sorted first. The kernel compares (logit, id) pairs.

The cost of this choice: with the switch on, every radix top-k on the card is sorted, including the sparse-attention
indexers' (Flash-Next's QSA layers), which take more time there. Their graph treats that top-k as an unordered set.
That their results stay the same is not checked here.

A row whose k candidates do not fit in a block's 48 KB of shared memory (k above 4,096) stays unsorted, with one
warning. `test-backend-ops` has one such case (k = 9,999); sampling uses far smaller k.

The commit also adds `tests/test-top-k-sort`, built but run by hand. It runs `ggml_top_k` on each device many times on
fixed rows and counts the orders it gets.

## The checks

All on a build machine (Ryzen 9 5900X, three RTX 3090s, driver 595, CUDA 12.9 in the containers) with the rig
build, whose card code is compute 6.1 SASS and PTX: the 3090s run the PTX through the driver's compiler. Base
`8a9629942`, head `931d777f5`.

### With the switch unset: the base's build

[checks.tsv](checks.tsv) has these rows as a file.

| Check | Base `8a9629942` | Head `931d777f5` | Verdict |
|---|---|---|---|
| Rig build: warning lines | 32 | 32, with `test-top-k-sort` and `test-backend-sampler` built too | none new |
| CPU build of every target: warning lines | 2 | the same 2 (the CMake line names the work folder) | none new |
| Card code: SASS of `llama-server`, one SHA-1 per kernel copy | 9,518 | 9,519 | every base copy identical; the new one is `top_k_sort_rows` |
| Card code: PTX of `llama-server`, one SHA-1 per function | 9,518 | 9,519 | every base function identical; the new one and its shared-memory declaration |
| `test-llama-archs -b CPU -s 1` of the rig builds | writes 148 logits files, 153 of 153 pass | 153 of 153 pass, plain and comparing | 148 of 148 files bit for bit |
| ctest, 75 tests, no network (CPU build) | 71 pass; `test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales` fail | the same | the same, test by test |

The base's rows come from the [C880 landing](../2026-11-24_c880_land/README.md), which built `8a9629942` with the same
script and settings; the logits files were written by its rig build in this run. ctest leaves out the seven tests every
run of `cpu_tests.sh` leaves out (C880's README lists them).

### The top-k on one card, the switch unset and set

`test-top-k-sort` ran 2,000 calls of each case on one RTX 3090, as one row and as four rows (the several output rows of
`LLAMA_SAMPLING_TOP_K_ROWS`), plain and with the row written before every call ([topk_sort.tsv](topk_sort.tsv)). The
cases are C962's three probe rows (Flint's 150,144-token vocabulary, eight candidates with the probabilities the server
reported, the draw the fork keys by seed and position) and two rows with exact ties: three equal pairs among the top
eight, and an eighth value shared by five columns.

| Case (one row, plain) | Unset: orders in 2,000 calls | Unset: tokens the draw picked | Set: orders | Set: calls in the sorted order | Set: token |
|---|---|---|---|---|---|
| C962's session context, 24th token (u 0.3042) | 6 | 14 in 1,523, 65618 in 469, 11844 in 8 | 1 | 2,000 | 14 |
| C962's fresh context, 24th token (u 0.3042) | 11 | 14 in 2,000 | 1 | 2,000 | 14 |
| C962's forced Pascal kernels, token index 110 (u 0.0285) | 5 | 1002 in 2,000 | 1 | 2,000 | 1910 |
| three equal pairs in the top eight | 6 | — | 1 | 2,000 | — |
| the eighth value shared by five columns | 7 | — | 1 | 2,000 | — |

With four rows, or with the row written before every call, the switch gives the same: one order, the sorted one, in
every case. Unset, four rows give many different outputs in 2,000 calls (750 on C962's fresh row, plain; each output
holds the four rows' orders).
Every call kept the right candidates (no set error), and the processor's top-k, run alongside, gave one order in every
case (its own, which it does not sort). At token index 110 the sorted order picks 1910 where the card's usual order
picks 1002: the draw lands in another candidate's share when the shares come in another order.

[ops.tsv](ops.tsv): `test-backend-ops -o TOP_K` passed 607 of 607 cases with the switch unset and set. It compares
the set of candidates, not their order. One case has k = 9,999, which stays unsorted with the warning.

[sampler.tsv](sampler.tsv): `test-backend-sampler --device gpu`, every test in a process of its own, on a tiny Flint
with random weights:

| Switch | Exit 0 | Not 0 |
|---|---|---|
| unset | 21 of 23 | `test_logit_bias` (134), `test_top_k_rows_switch` (134) |
| `1` | 22 of 23 | `test_logit_bias` (134) |

`test_logit_bias` fails with both, as in [C691](../2026-11-11_c691_sampling_launches/README.md) and
[C802](../2026-11-12_c802_topk_fixed_graph/README.md): it expects a +10 bias to win, which it does not on random weights.
`test_top_k_rows_switch` checks that `LLAMA_SAMPLING_TOP_K_ROWS=1` gives each chain the candidates and logits its own
top-k gives, in order. Unset, the card's order differs between the two runs, so its check of the logits aborts; with the
switch it passes. Its earlier runs were on a processor.

### C962's request in the session's own context, three cards

Flint Pascal Q8 GGUF with its v5 preset (`--pipe-depth 4`, `-bs`, `LLAMA_SAMPLING_TOP_K_ROWS=1`), the split changed to
the three cards (`-ts 1,1,1`; the file is 40.4 GB, more than one card holds). Six loads alternating set and unset; each
ran C950's session seven times: the requests before R14 (nine greedy ones and R3, which C950 sends at temperature 0.7
with a JSON grammar), then R14 at temperature 0.7, top-k 8, top-p 1, seed 7 on the prompt cache those built: five times
as C950 sent it, once with top-p 0.95, once greedy. Then three more loads (set, unset, set) ran it three times each with
`n_probs` 8 and `post_sampling_probs`, which keeps the same draws and returns the dist step's probabilities
([rounds.tsv](rounds.tsv), [lists.tsv](lists.tsv), [draws.tsv](draws.tsv), [probs_bits.tsv](probs_bits.tsv)).

| Request | Unset | `GGML_CUDA_TOP_K_SORT=1` |
|---|---|---|
| R14 as C950 sent it | 15 rounds on 3 loads: C950's first reply (`fe36bf356f32`) 11, its second (`de544dad17e0`) 4 | 15 rounds on 3 loads: one reply (`e4b9ba7e039e`) 15 |
| R14 with `n_probs` | 3 rounds on 1 load: `fe36bf356f32` 3 | 6 rounds on 2 loads: `e4b9ba7e039e` 6 |
| R14 at top-p 0.95 | 3 on 3 loads: `fe36bf356f32` | 3 on 3 loads: `fe36bf356f32` |
| R14 greedy | 3 on 3 loads: `fe36bf356f32` | 3 on 3 loads: `fe36bf356f32` |
| the nine greedy requests before R14 | the same replies in all 24 rounds | the same replies in all 27 rounds, and the same as unset |
| R3 (sampled, with a grammar) | one reply in 24 of 24 rounds (`45f26934f6df`) | one reply in 27 of 27 rounds (`bce8d0e84421`) |

The `n_probs` rounds, checked token by token with the draws the fork keys by seed and position (C962's method):

| Switch | Tokens | Positions where the order of the candidates decides the token | Tokens that are the descending order's pick | Pairs of rounds with probabilities that differ in any bit |
|---|---|---|---|---|
| unset | 256 in each of 3 rounds | 2 | 255 (1 of the 2 deciding positions) | 3 of 3 pairs, at 37, 32 and 32 of 256 positions (at most 2.38e-07) |
| `1` | 256 in each of 6 rounds | 3 | 256 (3 of 3) | 0 of 15 pairs |

R14 with the switch first leaves C950's first reply at the 171st token (index 170): 13209 holds 0.9961 and 694 holds
0.0021 of the probability, and the draw there is 0.9978, past 13209's share and inside 694's in the descending order,
so that order picks 694. R3 also
takes another reply with the switch, one in every round; its probabilities were not returned, so which order it follows
is not checked.

## Files

| File | What |
|---|---|
| [topk_sort.tsv](topk_sort.tsv) | `test-top-k-sort`: every case per switch, mode and device: calls, distinct orders, calls in the reference order, set errors, the draw and the tokens it picked |
| [ops.tsv](ops.tsv) | `test-backend-ops -o TOP_K` per switch |
| [sampler.tsv](sampler.tsv) | `test-backend-sampler --device gpu`, each test's exit code per switch |
| [rounds.tsv](rounds.tsv) | the session: every round of every load: whether its nine greedy requests gave the same replies as in every other round, R3's reply and R14's (md5 of the token ids) |
| [lists.tsv](lists.tsv) | the session per switch, for each kind of R14 request and for R3: rounds, loads, the replies and how many rounds gave each |
| [draws.tsv](draws.tsv) | R14's `n_probs` rounds: tokens, positions where the order of the candidates decides the token, tokens that are the descending order's pick |
| [probs_bits.tsv](probs_bits.tsv) | the same rounds compared in pairs per switch: positions whose dist-step probabilities differ in any bit, the largest difference |
| [checks.tsv](checks.tsv) | every check with the switch unset: base, head, verdict |
| [run/run.sh](run/run.sh) | the driver: the steps below |
| [run/ptx_functions.py](run/ptx_functions.py) | compares two builds' PTX function by function |
| [run/c985_tables.py](run/c985_tables.py) | the tables from the run's output |

## Commands

On a build machine with Docker and NVIDIA cards, from a copy of this repository at `REPO` (with this folder, C962's
`run/session_r14.py` and `run/probe/`, C950's `run/session.py`, `release/flint-v5/settings.env`,
`scripts/build_llamacpp_docker.sh`, [cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh) and
[sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py)), detached:

```bash
W=<work dir> SEED=<llama.cpp clone holding 8a9629942> REPO=<that copy> \
  BASE_RIG=<8a9629942's rig build folder> BASE_RIG_LOG=<its build log> \
  BASE_CTEST=<8a9629942's ctest table> BASE_CPU_LOG=<its CPU build log> \
  FLINT=<Flint Pascal Q8 GGUF> MODEL_SMALL=<a small GGUF with Flint's vocabulary> MOUNTS='<model folders>' \
  RIG_CCACHE=<ccache dir> CPU_CCACHE=<ccache dir> PREFLIGHT=<cuda check> LOCK=<flock file> JOBS=8 \
  setsid nohup bash "$REPO/docs/data/2026-11-24_c985_topk_order/run/run.sh" > run.out 2>&1 < /dev/null &
```

Steps (`STEPS`, default all, in this order): `src` (a clone, the branch fetched and checked against its commit),
`rig` (the rig tarball with `TESTS=1` and the extra targets `test-llama-archs test-cuda-keepalive test-top-k-sort
test-backend-sampler`, its md5 and warnings), `units` (on one card: `test-top-k-sort`, `test-backend-ops -o TOP_K`,
`test-backend-sampler`, each with the switch unset and set), `session` (C962's R14 on three cards, the arms alternating
on and off), `archs` (the rig builds' `test-llama-archs` logits), `sass` (the card code), `cpu` (a CPU build of every
target and ctest). The base's rig build, CPU build log and ctest table are the ones the
[C880 landing](../2026-11-24_c880_land/README.md) made of `8a9629942` with the same script and settings. Images:
`nvidia/cuda:12.9.2-devel-ubuntu24.04` for the rig build, the GPU steps, the logits and `cuobjdump`; `ubuntu:24.04` with
`git cmake build-essential ccache python3 python3-jinja2` for the CPU step. The run took 32 minutes (02:16 to 02:48):
the rig build 1 minute 21 seconds from a warm compiler cache, the unit checks 2 minutes, the session 21, the card-code
dumps 6.

The `n_probs` rounds are a second run of the session step, in a work folder of its own whose `rig/` links the first
one's, with its own container label, so it ran beside the first run's processor steps (02:40 to 02:48):

```bash
env <the same variables> W=<second work dir> NAME=c985p PORT=18986 STEPS=session \
  ARMS="on-d off-d on-e" VARIANTS="probs probs probs" bash run.sh
python3 -I run/c985_tables.py <this folder> <a folder whose session/ holds both runs' session/ folders>
```

## Provenance and licence

Code: the fork's branch (MIT, as llama.cpp). The model replies are Flint Pascal Q8 GGUF's (released by this project) to
C950's synthetic coding task; `test-backend-sampler` ran on a tiny Flint with random weights (6 layers, 256-dimensional,
Flint's 150,144-token vocabulary); `test-top-k-sort`'s rows are C962's probe inputs (the candidates and probabilities
the server reported) on a fixed pseudo-random background, and two made-up rows with ties. `test-backend-ops` uses random
tensors, `test-llama-archs` random-weight models. No download, trained weights other than Flint's, or private data.
