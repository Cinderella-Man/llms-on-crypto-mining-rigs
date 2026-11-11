# The verify pass prepared during the last drafting pass (C738, Q1307, 2026-11-11)

What this is: a change on the v4 line that moves the host's work for a speculative step's verify pass off the token's
path. [R86](../2026-11-11_r86_step_trace/README.md) traced the step on a P106 with the v4 night merge
(`c408dabcb`). Of the 7.512 ms between the last drafting pass and the verify pass (median), the host waits 2.825 ms in
that pass's `synchronize` (the drafted id is computed on the MTP draft's card), then spends 0.893 ms in
`llama_context::decode` before the verify pass's inputs are set (`v.dec → v.ub.in`), and 5.610 ms pass between the
inputs set and card 0's launch, where the scheduler copies card 0's inputs. Of all the verify pass's inputs, only the
token ids need the drafted ids, and only the last of them comes out of that last drafting pass.

The change is on the fork branch `pascal-p106-v4-b12246-c738` (`b45b4afd1`, eight commits on
`pascal-p106-v4-b12246-c748` `5b329df16`), behind `LLAMA_VERIFY_PREPARE`, off by default. This page is the change and
its check on a CPU; the rig run that times it is [Q1345](../../../quants/experiments.md) (below).

## Result

**With `LLAMA_VERIFY_PREPARE=1` or `=2` llama-server gives the same tokens, top probabilities, draft counts and verify
steps as the base build on every request, and prepares 398 of the 406 verify passes during the last drafting pass.
`test-llama-archs --verify-prepare` gives the plain decode's outputs and end state, byte for byte, on every model it
prepares (98 models a run, 196 of 272 rows), at three seeds, and fails as it should when the prepared call leaves its
stand-in token in the KV cells. With the variable unset the branch gives `5b329df16`'s logits on every model.** The
checks, on a build machine's CPU ([summary.tsv](summary.tsv), [vprep.tsv](vprep.tsv), [server.tsv](server.tsv)):

- **llama-server, 11 arms, with a random-weight `qwen4exp` (Flash-Next's architecture) model with one MTP block,
  draft-mtp at three tokens with `p_min` 0 ([run/server_check.sh](run/server_check.sh)):** the switch unset, `=1` and
  `=2` each give the base build's (`5b329df16`) tokens, top-3 probabilities, drafted and accepted counts, and all 406
  verify steps of `LLAMA_SPEC_DUMP` (the draft, the drafter's candidates and probabilities, the target's top tokens) on
  four requests (two greedy, two at temperature 1.0 with the preset's top-k 20 and top-p 0.95); so do the same three
  with the backend sampler (`-bs`, as the preset) against the base with `-bs`. In every switched arm 398 of the 406
  verify passes are prepared during the last drafting pass; the other 8 are the replies' last steps, where the slot caps
  the draft below three tokens (C744), and run as before.
- **The uploads on a device split:** with the RPC build every layer runs on an RPC device served from the same machine,
  so the verify pass has a CPU split and a device split whose graph inputs are copied from the host, as card 0's are on
  the rig. `=1` and `=2` give the RPC arm's replies and verify steps with the switch unset; with `=2` each prepared pass
  uploaded 13 graph inputs ahead (5,174 in all), and each of the 398 evaluations skipped all 13 (`:u13` in its
  `LLAMA_UBATCH_TRACE` line).
- **`test-llama-archs --verify-prepare`** ([vprep.tsv](vprep.tsv)): for every architecture with a memory, a 40-token
  prompt, then nine rounds of a server step's calls: the verify pass (four tokens, every one an output) prepared with a
  stand-in for its last id (the true id every fourth round), the rejected tail removed (0 to 3 drafted tokens kept, so
  in some rounds the prepared token stays in the memory), and every third round a one-token call. Its logits, or a
  backend top-k and dist sampler's tokens, probabilities and candidates, and the sequence's saved state at the end, are
  compared with the same calls run plainly before and after it. 272 rows a run (136 models × 2), of which 196 prepare
  every one of their 9 verify passes and 76 (38 architectures) are refused before anything runs (their memories are
  other types: sliding-window caches and their hybrids, DSA, MSA, DeepSeek-V4),
  and 20 rows of 10 recurrent-only models are skipped (their memory cannot remove a rejected draft). **All 272 pass at
  seeds 1, 2 and 3, with `LLAMA_VERIFY_PREPARE` unset and `=2`; none of the plain run pairs differed.** The same six
  runs with the rig tarball's own `test-llama-archs` on the CPU (its CUDA backend loads against the stub driver) pass
  the same way.
- **The test fails where it must:** with `LLAMA_VERIFY_PREPARE_FAULT=1` (the stand-in id left in the KV cells), both
  `qwen4exp` rows fail on outputs and state (the PLE n-grams read the cells' ids), and 24 rows of models whose cells keep
  ids for M-RoPE or PLE fail on their state; 26 fail in all.
- **With the variable unset, and with it set to `1` or `2`** (nothing in the test calls the prepare), `test-llama-archs
  -b CPU -s 1` gives `5b329df16`'s logits on 146 of 146 random models. **ctest: 70 of 70 pass** (the static CPU build;
  the tests that need a download, the network or torch are left out, as in C565). llama-server's own pytest suite was
  not run: it downloads its models.
- **The rig tarball builds:** `llama.cpp-20261111-b45b4afd1-pascal-p106-v4-b12246-c738.tar.gz`, 656,460,204 B, md5
  `48fcc2bea15eccd88af370edd9571a18` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the
  tree's `bin/`).

## What it changes on the rig (est.)

From R86's sizes, before the rig row measures it:

- **`LLAMA_VERIFY_PREPARE=1`: est. 0.89 ms a step.** The verify pass's host part up to its inputs (0.893 ms, median)
  runs inside the last drafting pass's 2.825 ms wait. What stays after the wait is setting the token ids: the batch,
  the ubatch and the KV cells' copies of the last id, then the inputs that read ids (the token input, Flash-Next's PLE
  n-gram rows, the backend sampler's draws), a few µs on a desktop CPU.
- **`LLAMA_VERIFY_PREPARE=2`: est. 1.75 ms a step.** It also sends card 0's graph inputs during the wait (13 of the
  RPC device split's 20 to 21 inputs in the CPU check). [R81](../2026-11-11_r81_ubatch_trace/README.md) split
  R86's 5.610 ms on the same build: the verify pass's card 0 copies take 0.857 ms (median of 10 four-token passes) and
  the CPU split before them 4.379 ms (the drafted tokens' rows of the n-gram table, C615 and C761, which need the ids).
  So the uploads add 0.857 ms, and 0.893 + 0.857 = 1.75 ms fits in the 2.825 ms wait with 1.08 ms to spare: no
  upload moves the drafted id's read later. (R86's "up to 6.50 ms" counted all of the 5.610 ms as card 0's copies.)
  Each upload is a synchronous copy: the host would carry any that did not fit in the wait. One asynchronous staged
  copy instead (C765, Q1346) is parked while they fit.

## How it works

`llama_decode_prepare` (`src/llama-ext.h`) runs `llama_context::decode` for a batch whose last token ids are not known
yet (a stand-in id), and stops after the graph's inputs that read no token id are set:

1. **Before anything of the call runs, it refuses** what it cannot hold: a memory other than the KV cache, the
   recurrent memory and the two hybrids (their exact types: these can take token ids after `apply()`), embeddings or
   state rows, more than one sequence, more than one ubatch, a call in flight, the shortlist head with its bound (it
   reads the ids), cuts or store fills inside the call, pipelined slots, a reserve still to run. A refused call returns
   1 and the decode runs as before. (`deepseek4`'s memory showed why: when its prologue ran twice, once in a prepare
   that gave up later and once in the decode, the decode's results changed.)
2. **Prepare** (decode in mode 1): the batch allocator, the pending memory updates, the memory's slots
   (`init_batch`), the output buffer, the h_nextn half and its wait for the draft's reads, `apply()` (the cells take the
   stand-in id), the graph reused or built and allocated, and the inputs whose class says it reads no token id
   (`sets_without_tokens`: positions, output rows, the KV indices and masks with C100's device cells, the recurrent
   state copies, qwen4exp's k-pool tables; each class answers for its exact type). With `=2` the scheduler then copies
   the first device split's graph inputs (`ggml_backend_sched_preupload_inputs`) and keeps a copy of their bytes.
   Nothing is counted in the context's timings yet, and the backend samplers' transaction does not start.
3. **Submit** (the next decode, mode 2): the batch must be the prepared one but for token ids, or the process aborts.
   The changed ids go into the batch allocator, the ubatch and the cells `apply()` filled (`set_tokens`: the attention
   and indexer caches keep each token's id for the PLE n-grams and M-RoPE). Then the samplers' transaction starts, the
   other inputs are set (the tokens, the PLE rows, the samplers' draws), and the graph runs. The scheduler skips each
   pre-uploaded input whose host bytes are still the ones sent, and copies any other one as usual.

llama-server (`LLAMA_VERIFY_PREPARE=1|2`) calls it from draft-mtp's new `last_pass` hook: after the drafter queues its
last drafting pass and before it waits for it, when that pass adds exactly one token (`p_min` 0, as the preset drafts).
The server prepares only when the step's batch is that slot's sampled token and whole draft and nothing before the
decode reads or changes the target context: one slot at work, a generating completion, no draft cap below the
draft (a reply's last steps: 8 of 406 in the check), no adaptive or forced drafts, no checkpoint of the target in the
step, no screened or bounded head. It sets the batch's LoRA, embeddings, taper and taper hint first, which the step
sets again later to the same values. A verify pass prepared and not decoded by the step's end (an error path) is
dropped: its sequence is cleared and the slot forgets its cached prompt.

## The checks

All on a build machine's CPU, in containers at low priority, one at a time ([run/checks.sh](run/checks.sh)):

| Check | What | Verdict |
|---|---|---|
| `cpu` | static native CPU builds of the branch: every target, and llama-server with `ggml-rpc-server` (`GGML_RPC=ON`) | OK, OK (3 warning lines each, as `5b329df16`'s) |
| `vprep` | `test-llama-archs --verify-prepare`, seeds 1–3, switch unset and `=2` | 6/6 runs, 272/272 rows each |
| `fault` | the same at seed 1 with `LLAMA_VERIFY_PREPARE_FAULT=1` | fails as expected: both `qwen4exp` rows, 26 in all |
| `server` | llama-server, 11 arms ([server.tsv](server.tsv), [server_verdict.txt](server_verdict.txt)) | PASS |
| `off` | `test-llama-archs -b CPU -s 1` against `5b329df16`'s CPU build, switch unset, `=1`, `=2` | 146/146 the same logits, three times |
| `ctest` | ctest in the static CPU tree | 70/70 |
| `build` | the rig tarball, `TESTS=1` | OK, md5 `48fcc2bea15eccd88af370edd9571a18` |
| `vrig` | `vprep` with the rig tarball's `test-llama-archs` on the CPU (CUDA stub driver) | 6/6 runs, 272/272 rows each |
| `table` | [vprep.tsv](vprep.tsv) rebuilt from the 13 logs | 3,796 rows |

The control rows: `LLAMA_VERIFY_PREPARE_FAULT=1` leaves the stand-in ids in the KV cells. Every prepared `qwen4exp` row
(PLE) fails on its outputs and state, and the models whose cells keep ids (M-RoPE, PLE) fail on their state: 26 rows,
where the models that keep no ids pass. Two plain runs around each prepared one are the test's control of a model's own
spread: none differed.

## The rig row (Q1345, R102)

Not run yet. One untimed run on a P106 rig with the c738 tarball (below), R86's method with the switch as the
variable: [run/rig_loads.sh](run/rig_loads.sh) starts the v3 preset (`release/flash-next-v3/settings.env`, the MTP
draft at three tokens) through `bench/agent_check/rig_server.sh` with the host profile (`GGML_CUDA_PROFILE_HOST`), six
loads in this order, each answering S37's greedy request (79 prompt tokens, 256 generated) and then the agent's
first-turn request (2,760 prompt tokens, 256 generated):

| Load | Adds |
|---|---|
| `prof-off` | nothing: the reference |
| `prof-vp1` | `LLAMA_VERIFY_PREPARE=1` |
| `prof-vp2` | `LLAMA_VERIFY_PREPARE=2` |
| `trace-off`, `trace-vp1`, `trace-vp2` | the same three with C405's `LLAMA_STEP_TRACE` |

```bash
BIN=<the tarball tree's bin/> MODEL=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> \
  PRESET=release/flash-next-v3/settings.env RIG_SERVER=bench/agent_check/rig_server.sh \
  REQ_GREEDY=<S37's greedy_request.json> REQ_LONG=<S37's long_request.json> OUT=<dir> \
  setsid nohup bash docs/data/2026-11-11_c738_verify_prepare/run/rig_loads.sh > console.out 2>&1 < /dev/null &
```

Then on a build machine: S37's `step_split.py` on every load and its `arms.py` with `prof-off` as the reference (the
greedy request draws nothing at random, so its steps pair up), and C405's `step_gap.py` on the trace loads with R86's
`align_t0.py`. What it must show: the greedy replies of every load equal to `prof-off`'s; each `vp` server's closing
line (`C738: N decodes prepared ahead of their last token ids, ...`) with N near the request's verify steps and, for
`vp2`, inputs uploaded; the paired greedy step shorter by about 0.89 ms (`vp1`) and 1.75 ms (`vp2`), classed against
the spread of S37's two `prof` loads (−0.68 ms on the same measure); in the trace loads, `v.dec → v.ub.in` down from
0.893 ms to the ids' share, the prepare's stages (`p.*`) inside the last drafting pass's wait, and with `vp2` card 0's
copies gone from the time to card 0's launch.

The tarball: `llama.cpp-20261111-b45b4afd1-pascal-p106-v4-b12246-c738.tar.gz`, 656,460,204 B, md5
`48fcc2bea15eccd88af370edd9571a18` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in its
`bin/`). Est. 30 minutes of rig time.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail, as the run wrote them |
| [vprep.tsv](vprep.tsv) | `test-llama-archs --verify-prepare`: per build, arm, seed, model and sampling: verify passes prepared, outputs, state, status |
| [server.tsv](server.tsv) | `server_check.sh`: per arm and request, prompt and predicted tokens, drafted and accepted, passes prepared, inputs uploaded ahead |
| [server_verdict.txt](server_verdict.txt) | its verdict lines |
| [run/checks.sh](run/checks.sh) | the build machine's checks, step by step (header: usage) |
| [run/server_check.sh](run/server_check.sh) | the server arms and requests (header: usage) |
| [run/rig_loads.sh](run/rig_loads.sh) | the rig row's driver: R86's loads with the switch off, `=1` and `=2` |

## Commands

On a build machine with Docker (no GPU used): a clone of llama.cpp with the fork as remote `fork` in `$WORK/rig/src`, the
base's static native CPU build (`5b329df16`, every target) in `$BASE`, this repository in `$WORK/repo`:

```bash
WORK=<work dir> REPO=$WORK/repo SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c738 \
  BASE_CPU=$BASE/bin BASE_SRV=$BASE/tree/build/bin LOCK=<flock file> JOBS=8 \
  DOCKER_ARGS='--label pascal-quants=1' bash $WORK/repo/docs/data/2026-11-11_c738_verify_prepare/run/checks.sh
```

The steps ran as `STEPS='cpu vprep fault server'`, then `STEPS='off ctest build vrig'`, then `STEPS=table`. The CPU
image is Ubuntu 24.04 with git, cmake, build-essential, ccache, python3, curl and jq.

Provenance: the models are `test-llama-archs`'s random-weight models (no download); the server's requests are token
lists from a fixed generator (in `server_check.sh`). Licence: the repository's.
