# Why some checkpoints stop Ornith's long prompts, and a fill that does not wait (C907, Q1644; 2026-11-21)

**In short:** Ornith's server saves a copy of the model's running state (its context checkpoints) at a few points in a
long prompt. Most copies are queued on the cards behind the work already running there and cost nothing. But when such
a point falls at the very end of a 4,096-token batch, the copy is taken between two batches, and it first waits until
all nine cards have finished everything queued: that wait was the 8.20, 11.85 and 12.37 s of the 47,012-token prompt in
[R183's Ornith arm](../2026-11-20_r183_ornith_recall/README.md). Miner Llama's new switch `LLAMA_SERVER_CKPT_NOWAIT=1`
(off by default) queues that copy behind the work as the others are. On a build machine's CPU it gives the same
replies, calls and checkpoints on R183's own prompts; the time it saves on the rig is the next check.

This is registry row Q1644 in [quants/experiments.md](../../../quants/experiments.md). The fork branch is
`pascal-p106-v5-ckpt-stall` (`0aebc4a57`, one commit on `pascal-p106-v5` `3dfad876f`).

## What R183 showed

Ornith's v5 preset ([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)) reads a prompt in calls
of up to 4,096 tokens (`-b 4096 -ub 512`) and keeps four checkpoints of the recurrent state on the cards
(`--ctx-checkpoints 4 --ctx-checkpoints-vram 4`), taken inside the calls where it can (`--ctx-checkpoints-incall`).
In R183 every prompt past 8,077 tokens also took one to three checkpoints between two calls, each behind a call of one
small-talk exchange (36, 43, 36, 31, 39 and 36 tokens), and those rows took 7.44 s (16,075-token prompt), 7.06 and
7.99 s (32,025), 8.20, 11.85 and 12.37 s (47,012). The server's own counter of checkpoint save time (`ckpt_save_ms`)
was 7.34, 14.73 and 31.67 s for the three prompts.

## Why: the code, read at `3dfad876f`

1. **Where the checkpoints fall.** A checkpoint is taken at a user message at least `--checkpoint-min-step` tokens past
   the last one: 8,192 by default ([common/common.h:714](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/common/common.h#L714)), exactly two batches of 4,096. So once one checkpoint sits at
   a batch's start, the next one falls in the last message of the batch after next. In R183 the batch before each
   between-calls checkpoint ended 14, 39, 1, 16, 28 and 22 tokens short of 4,096, at that message.
2. **Why the next call is one message long.** The prompt loop decides where a batch stops by comparing each user
   message with the newest checkpoint in the list ([tools/server/server-context.cpp:7183](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L7183)), but the checkpoint at the
   batch's own start is added only after the loop ([server-context.cpp:7315-7319](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L7315-L7319)). So a batch that starts at a
   checkpoint stops again at the very next user message, one exchange later. Upstream llama.cpp has the same test
   (b12483's `server-context.cpp:4421-4428`); it is upstream's behaviour.
3. **Why it is not taken inside the call.** With `--ctx-checkpoints-incall` the server joins the next part into the
   current call only where it fits whole in the batch's room ([server-context.cpp:7231](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L7231) takes back a part the room
   cut short; [:7329](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L7329) goes on only while room is left), so that every call boundary, and with it every number, stays
   as it is without the switch ([C217](../2026-11-11_c217_incall_checkpoints/README.md)). The one-exchange part never
   fits in the 14, 39, 1, 16, 28 and 22 tokens left, so its checkpoint falls back to the old path: taken before the next
   call, between two calls.
4. **What the save waits on.** That path is `create_checkpoint` ([server-context.cpp:4253](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L4253)) → `update_tgt_store`
   ([:4286](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L4286)) → `llama_state_seq_get_data_store` ([common/common.cpp:2862](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/common/common.cpp#L2862)), whose first step is a full
   `synchronize()` of the context ([src/llama-context.cpp:8894-8897](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/src/llama-context.cpp#L8894-L8897), [:1454](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/src/llama-context.cpp#L1454)). A prompt call's `llama_decode`
   returns without waiting for its own end (upstream's commented-out `synchronize()`, [llama-context.cpp:5504](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/src/llama-context.cpp#L5504)), so
   when the server builds the next batch the last call's micro-batches are still moving through the nine cards. The
   save waits for all of it. It is not a copy to the host (the four newest checkpoints stay on the cards, and the copy
   is card to card), not a re-plan (the counter times only the fill, [server-context.cpp:4280-4302](https://github.com/Cinderella-Man/miner-llama/blob/3dfad876f48a41ab2792b23152f5ddd3418ba953/tools/server/server-context.cpp#L4280-L4302)) and not the
   pinned buffer: the in-call checkpoints copy the same bytes behind the work, and took 0.01 ms in all for the
   8,077-token prompt's four.
5. **Why it grows with depth.** The wait is the last call's unfinished work, and calls get slower as the context
   grows. Each save against the call before it: 7.34 s after a 14.15 s call, 7.06 after 14.41, 7.99 after 21.04, 8.20
   after 20.60, 11.85 after 24.09, 12.37 after 25.90.

**What it costs, est.:** part of each wait is work the cards had to do anyway. Through the same point, 12,389 and
12,371 tokens into the prompt, the 16,075- and 32,025-token prompts (one stall each) took 46.40 and 46.15 s, the
47,012-token prompt (no stall there, through 12,361) 42.14 s: est. 4.26 and 4.01 s lost at the stall near 8,200
tokens. The rest of each wait would have been spent anyway.

## The change

- **`llama_state_seq_get_data_store_nowait`** (new in `include/llama.h`): where work is queued on the devices, the
  fill that an in-call checkpoint uses (`state_seq_fill_store_at`): the state as the host holds it now, each row copied
  on the stream of the card that holds it, behind the queued work and ahead of the next call's, with an event recorded
  after. With nothing queued it is the waiting fill. A host copy of the store waits for those events, and so does a
  restore (`llama_state_seq_set_data_store` now waits for a store's events before reading it).
- **`LLAMA_SERVER_CKPT_NOWAIT=1`** (llama-server, off by default): a prompt's checkpoint taken between two calls into
  a device store uses it. The calls, the checkpoints and their contents stay the same. `timings.ckpt_nowait_n` counts
  them, and the checkpoint's trace line (`-lv 4`) says "no wait". Checkpoints on the host and those taken while
  generating are unchanged.
- **`test-llama-context`**: new cases `store_nw` and `store_nw_p` (whole and partial state): the no-wait fill behind
  a call whose work is still queued returns the waiting fill's blob byte for byte; the store keeps that call's end after
  a later call has changed the live state (its host copy equals the host state taken then); its restore continues as
  the host state's, bit for bit; with nothing queued it fills as the waiting fill.

## Checks on a build machine's CPU

Fork branch `pascal-p106-v5-ckpt-stall` at `0aebc4a57` against its base `3dfad876f`, 2026-11-21 07:43–08:01. No GPU
and no rig: the CUDA build's binaries run their CPU backend against the driver stub, and the GPU code is read with
`cuobjdump`. Every container ran at a low CPU share without network.

| Check | Result |
|---|---|
| build warnings | rig build 38, CPU build 26 distinct, the base's sets; none new |
| ctest, 75 tests without network | **71 pass, test by test the base's** ([ctest.tsv](ctest.tsv)); the same 4 fail on both, the base's known failures (`test-kv-cache`, `test-jinja`, `test-jinja-py`, `test-kv-cache-scales`) |
| `test-llama-archs` | 153 of 153 on both; the branch's logits the base's, bit for bit, in 148 of 148 files, natively and with the rig build against the base's rig build |
| hand-run tests | all pass on both ([cpu_summary.tsv](cpu_summary.tsv)) |
| `test-llama-context`, the store cases | on random qwen35moe (Ornith's architecture), qwen35, qwen4exp and mamba models: 24 runs, 0 failed; `store_nw` and `store_nw_p` pass on all four ([store_cases.tsv](store_cases.tsv)) |
| `test-llama-context`, every case | 104 runs on the four models: 93 pass, 11 skip (a case for DeepSeek V4, a NaN fill a hybrid model cannot take, a rollback that mamba's cache lacks), 0 fail ([context_cases.tsv](context_cases.tsv)) |
| GPU code | all 8,836 kernels identical, the 176 `kq_fp32` variants too ([sass_summary.txt](sass_summary.txt)) |
| llama-server, R183's prompts | **with the switch unset, everything the base's; set, the same replies, calls, checkpoints and restore, and every checkpoint between calls filled without a wait** (below) |

### R183's prompts on the CPU

The server check replays R183's own four prompts ([prompts](../2026-11-20_r183_ornith_recall/prompts/)) on a random
qwen35moe model built with Ornith's tokenizer (the Qwen3.5 vocabulary, 248,320 tokens) and Ornith's own chat template,
read from its GGUF, with the v5 preset's prompt line and server switches. So the server sees what Ornith's server saw:
**the same token counts as on the rig (8,077, 16,075, 32,025 and 47,012) and the same calls.** The CPU's progress lines
start once a prompt has run 3 s, so they show the 32,025- and 47,012-token prompts' later calls only, and those start
where R183's did: 20,558, 24,653, **24,689** (after the 36-token cut call), 28,785; and 20,552, 24,632, **24,663**,
28,759, 32,827, **32,866**, 36,962, 41,036, **41,072**, 45,168 ([calls.tsv](calls.tsv)). After the four prompts a
restore pair follows: the 16,000-token prompt behind a new system message (a whole read), then the same with a user
message at 62 % of the list changed, which restores the checkpoint taken between calls at 8,234 tokens and reads the
remaining 7,858.

| Run | Build | Setting | Result ([server.tsv](server.tsv)) |
|---|---|---|---|
| `base-off` | `3dfad876f` | the preset | the reference |
| `head-off` | `0aebc4a57` | the preset | the base's in all six requests: replies, each token's top-3 log-probabilities, counters, calls, the 33 checkpoints and the restore |
| `head-on` | `0aebc4a57` | plus `LLAMA_SERVER_CKPT_NOWAIT=1` | `head-off`'s in all six requests; **all 7 checkpoints taken between calls filled without a wait**: 1 at 16,075 tokens, 2 at 32,025, 3 at 47,012 (at 24,632, 32,827 and 41,036, the three cut calls of R183), 1 in the restore pair's first request, and the second request restored that one (at 8,234) |

The CPU computes each call to its end before `llama_decode` returns, so nothing is queued and nothing waits there:
`ckpt_save_ms` is 0.077 ms for the 47,012-token prompt with the switch unset. The time the switch saves can only be
measured on the rig.

### Checkpoints 7,680 tokens apart instead (C945, Q1686; not part of the verdict)

A fourth run, `head-cms`, leaves the switch unset and sets `--checkpoint-min-step 7680`, so that the checkpoints at user
messages fall about 500 tokens before a batch's end, where the one-exchange part fits and the checkpoint goes inside
the call. With no code change, it takes 4 checkpoints between calls in the six requests instead of 7 (the 32,025-token
prompt none instead of 2, the 47,012-token prompt 1 instead of 3), with the same replies on this model. But the calls
change, and so does the list of four: the restore pair's second request found no checkpoint before its change and read
all 16,092 tokens again, where the default restored the one at 8,234 and read 7,858.

## The rig check (not run)

[run/rig_check.sh](run/rig_check.sh): R183's Ornith arm again on one rig with this branch's build, one load per arm
(`off`: the preset; `on`: plus `LLAMA_SERVER_CKPT_NOWAIT=1`; optionally `cms`: plus `--checkpoint-min-step 7680`), the
8,000-token control first, then 16,000, 32,000 and 47,000 tokens; each arm's replies against the first's, and the
prompt seconds, `ckpt_save_ms` and checkpoint counts side by side. It passes if every reply is the same; est. it saves
4.26 and 4.01 s at the stall near 8,200 tokens (above), and more at the deeper ones.

## Files

| File | What it is |
|---|---|
| [server.tsv](server.tsv) | `compare.py`: per pair and request, whether the reply, log-probabilities, counters, calls, checkpoints and restore are the same, the checkpoints between calls, those filled without a wait, `ckpt_nowait_n`; the verdicts; the `head-cms` table |
| [calls.tsv](calls.tsv) | each run's and request's call starts, from the server's progress lines |
| [checkpoints.tsv](checkpoints.tsv) | each checkpoint in order: between calls or inside one, its position, whether it was filled without a wait, its save time and size (host and devices) |
| [server/](server/) | `client.py`'s reply files, one per run (gzipped JSON lines: each reply's text, tokens with top-3 log-probabilities, timings) |
| [store_cases.tsv](store_cases.tsv), [context_cases.tsv](context_cases.tsv) | `test-llama-context`'s verdicts per model and case: the store cases (the fail-fast step), every case |
| [ctest.tsv](ctest.tsv), [cpu_summary.tsv](cpu_summary.tsv), [sass_summary.txt](sass_summary.txt) | ctest test by test, base and branch; the CPU test script's verdicts; the GPU code comparison |
| [run/run.sh](run/run.sh) | the checks' steps (`src quick rig cpu context gen server archs sass`), as they ran |
| [run/client.py](run/client.py), [run/compare.py](run/compare.py) | the server check's client and comparison (`compare.py` later gained the device-size column of `checkpoints.tsv`, re-run on the same files) |
| [run/rig_check.sh](run/rig_check.sh) | the rig check above |

## Commands

```bash
# on a build machine with Docker, CPU only; SEED: a llama.cpp clone holding 3dfad876f; BASE_RIG and BASE_RIG_LOG: the
# base's rig build (scripts/build_llamacpp_docker.sh fork/pascal-p106-v5 at 3dfad876f) and its build log; VOCAB: a
# vocabulary-only GGUF with the Qwen3.5 tokenizer; TPL: Ornith's chat template (tokenizer.chat_template of its GGUF)
W=<work dir> SEED=<clone> BASE_RIG=<dir> BASE_RIG_LOG=<file> VOCAB=<vocab.gguf> TPL=<template.jinja> \
  PROMPTS=docs/data/2026-11-20_r183_ornith_recall/prompts REPO=<folder with build_llamacpp_docker.sh, cpu_tests.sh,
  sass_compare.py, client.py, compare.py> LOCK=<flock file> JOBS=8 bash docs/data/2026-11-21_c907_ckpt_stall/run/run.sh
python3 -I docs/data/2026-11-21_c907_ckpt_stall/run/compare.py <W>/checks/server > server.tsv

# the rig check, from the machine that runs the client
RIG=rig BIN=<this branch's bin folder on the rig> MODEL=<folder>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  OUT=<results> STOP_AT=21:50 bash docs/data/2026-11-21_c907_ckpt_stall/run/rig_check.sh
```

`cpu_tests.sh` is [docs/data/2026-11-05_c565_ladder4fix/run/cpu_tests.sh](../2026-11-05_c565_ladder4fix/run/cpu_tests.sh),
`sass_compare.py` [docs/data/2026-11-17_c861a_fp32_scale/run/sass_compare.py](../2026-11-17_c861a_fp32_scale/run/sass_compare.py),
`build_llamacpp_docker.sh` [scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh).

## Provenance and licence

The code is Miner Llama's (the fork's licence, MIT, as upstream llama.cpp). The prompts are R183's, this project's
own text. The model in the server check has random weights; its replies mean nothing and are published only so that
the comparison can be rerun. The rig numbers quoted are R183's. Raw build and server logs are not published.
