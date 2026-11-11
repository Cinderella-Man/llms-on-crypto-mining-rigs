# C217 (Q770): context checkpoints inside a prompt's decode call (2026-11-11, the night's code rows)

Qwen3.8-Flash-Next is a hybrid model: its recurrent state (Gated DeltaNet) cannot be rolled back, so llama-server saves
context checkpoints of it while it reads a prompt, and restores one when a later prompt diverges. To take one, the
server cuts the prompt's decode call at the checkpoint, waits for the call to end, saves the state, and starts the next
call. On nine cards in a pipeline every cut empties the pipeline: the last micro-batch before the cut crosses the
remaining cards with nothing behind it, and the next call fills them again. With the MTP draft, the end of every call
also waits for the target's last hidden row (the draft's pending row), so a cut costs a drain even when the save itself
waits for nothing.

On rig 1 with the v3.0 release preset the checkpoints cost 30.72 % of prompt time. With `--ctx-checkpoints-vram 4` the
copies leave (17.6 ms each on the cards against 641.5 ms on the host), and the drain is what is left: 77.28 s a pass of
the agent check's 62 prompts, 18.10 % of that line's prompt time
([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md), R43).

This folder holds the change that takes a checkpoint without cutting the call, behind a new switch on Miner Llama's v4
line, and its CPU checks. The registry row is Q770 in [quants/experiments.md](../../../quants/experiments.md); the
switch is listed in [docs/switches.md](../../switches.md). The rig run is Q1306.

**Result:** with `--ctx-checkpoints-incall` a checkpoint inside a prompt is taken inside the decode call, and on a
build machine's CPU everything the server computes stays as it was. A state stored inside one call at a cut is the
state of the calls cut there, byte for byte, on all three random hybrid and recurrent models (`test-llama-context`,
`store_at` and `store_at_p`, with and without rollback snapshots), and the check would see a wrong split: one call
without the cuts ends in another state in all 6 runs without snapshots, and the cuts without the snapshots' rule in all
4 runs with them. llama-server gives the base's tokens, top probabilities, prompt and cache counts and restored
checkpoints on every request, with checkpoints at the base's token counts in the base's order, in four switch-on
arms: 9, 8, 9 and 21 checkpoints taken inside calls (at the preset's 4 and 4 too, and with every other checkpoint
moved to the host). With the switch off, 146 of 146 architectures give the base's logits bit for bit. The rig tarball
builds.

**On nine P106-090 cards (Q1306, R85, the v4 night merge `c408dabcb` with only this switch set):** a gain of 18.99 %
of prompt time with every reply, checkpoint and restore unchanged. With the taper off, R43's replay of the agent
check's 62 prompts took 338.96 and 331.09 s a pass against 416.56 and 410.60 s with `--ctx-checkpoints-vram 4` alone.
The greedy replies are the same on 62 of 62 turns. The checkpoints are at the same token counts, in the same order,
with the same restores, on all 300 requests; 85 of the 124 checkpoints a pass go inside a call. With the taper on: 321.40
and 321.09 s against 344.00 and 343.60 s ([2026-11-11_r85_c217_p106](../2026-11-11_r85_c217_p106/README.md)).

## The change

`--ctx-checkpoints-incall` on the fork's branch `pascal-p106-v4-b12246-c217`, off by default, works with
`--ctx-checkpoints-vram` (the checkpoints' device stores). Three parts:

1. **llama-server joins the calls.** When a prompt's batch stops for a checkpoint (at a user message, at a
   `--ctx-checkpoints-spread` point, or at the two end-of-prompt points 4 + `-ub` and 4 tokens before the end), it goes
   on with the call that would have followed, as one more part of the same batch. A part is kept only when it fits
   whole into the batch's room (`-b`): one that would have run on past the room is taken back, and the batch ends at
   the checkpoint, as without the switch. Each part makes its own checkpoint decision, as its call would, and takes
   its place in the slot's list and its device store at once (so a later part's checkpoint may evict it before the
   call runs, as the base creates and then evicts it; it then costs no copy). A part goes on only when a store will be
   free for it: always when `--ctx-checkpoints-vram` covers `--ctx-checkpoints` (making room in a full list frees one),
   otherwise when one is free already.
2. **The call cuts its micro-batches where the joined calls would have ended** (`llama_set_ubatch_cuts`). Each part is
   split from its own start, and the rule of the rollback snapshots that keeps a call's last `n_rs_seq + 1` tokens in
   one micro-batch (for the MTP draft's verify steps) applies to each part as to a call. So the joined call computes
   the cut calls' micro-batches, each of them.
3. **The state goes into the checkpoint's device store inside the call** (`llama_state_seq_store_at`), right after
   the micro-batch that ends at the checkpoint has been queued. The blob is the host's view of the memory at that point
   (what a call that ended there leaves), and each recurrent row is copied on the stream of the card that computes it:
   behind that micro-batch's work, ahead of the next one's. Nothing waits for the cards. An event recorded after the
   copies makes any host copy of the store wait for them (a checkpoint moved to the host, a slot saved to a file); a
   restore synchronizes the context first, as it did.

The MTP draft follows: its catch-up pass over the joined call takes the same cuts and the same fills, so the draft's
part of a checkpoint is what it holds at the cut (Flash-Next's draft keeps an empty recurrent part whose partial state
records its cell's position), read back in the host format. The pending row stashed with a checkpoint (the target's
hidden row before it) is read from the call's rows instead of being carried between calls
(`common_speculative_get_state_at`).

**With C703** (`LLAMA_MTP_CATCHUP_OVERLAP`, the MTP draft's catch-up of a prompt call deferred until the next call is
queued; its own branch): both branches add to `common/speculative.cpp`'s base class at the same place, and keeping both
merges them. With both switches on, a deferred catch-up would run after the draft's cuts are cleared, so the
cumulative branch passes the cuts with the deferred pass (Q1302).

**Where it applies:** one slot's batch (no other slot's tokens in it; other slots join the next batch), token prompts
(no images), a model without sliding-window layers, and no speculation or the MTP draft with one head. A request with
an aLoRA, a shared prompt prefix for several completions, a decision request, or a cold `--prefix-cache` start cuts
its calls as before. A checkpoint at the first token of a call is taken before the call, as before: nothing is in
flight there. A switch whose conditions do not hold logs why at start-up and stays off.

**What stays the same:** every micro-batch, so with `--ubatch-taper` off every number the server computes; the
checkpoints' positions, order and stores. With the taper on, a joined call has one taper plan, with its head and tail
rules at its own ends only, so the micro-batches inside it are the taper's for one call, not for the cut calls.

| Code | What |
|---|---|
| `src/llama-batch.{h,cpp}` | `set_ubatch_cuts`: a split stops at the next cut and treats it as the batch's end, `n_keep_tail` included |
| `src/llama-context.{h,cpp}` | `llama_set_ubatch_cuts`, `llama_state_seq_store_at` and `_get`: the fill after the micro-batch that ends at a cut, the copies on each card's stream, the events; a host copy of a store waits for them |
| `common/speculative.{h,cpp}` | `common_speculative_incall_ok`, `common_speculative_get_state_at` (the MTP draft's pending row at a cut) |
| `tools/server/server-context.cpp` | the batch in parts, `create_checkpoint_incall`, `ckpt_incall_begin` and `_finish`; `timings.ckpt_incall_n` |
| `tests/llama-context/cases-rs.cpp` | the `store_at` and `store_at_p` cases |

## The checks

On a build machine's CPU, in containers without a GPU, under one lock shared with the night's other jobs (CPU shares
256, `nice -n 19`); branch `68ea07e60`, base `84f7bd3d2`. The verdicts are in [summary.tsv](summary.tsv).

| Check | Result |
|---|---|
| Rig tarball (`TESTS=1`, sm_61, no AVX) | builds: `llama.cpp-20261111-68ea07e60-pascal-p106-v4-b12246-c217.tar.gz`, 647,648,320 B, md5 `3cc6110f021c197b24e83b1d0dec4b6f`, 0 AVX instructions, GLIBC_2.38; `test-llama-context` and `test-llama-archs` built in its tree |
| CPU builds (static, native) | the branch (`test-llama-archs`, `test-llama-context`, `llama-server`) and the base (`test-llama-archs`, `llama-server`) |
| `test-llama-context` on random qwen4exp, qwen35 and mamba models | `store_at` and `store_at_p` OK on all three; 72 case runs, 63 OK, 9 SKIP (cases for DeepSeek V4, mamba's cases that need rollback snapshots, the NaN fill that would reach an attention cache), 0 FAIL ([context.tsv](context.tsv)) |
| The controls of `store_at` | one call without the cuts ends elsewhere than the cut calls in 6 of 6 runs without rollback snapshots and in 0 of 4 with them; the cuts without the snapshots' rule end elsewhere in 4 of 4 ([control.txt](control.txt)) |
| Switch off: `test-llama-archs -b CPU -s 1` | 146 of 146 architectures give the base's logits bit for bit (151 of 151 cases pass in both builds) |
| llama-server, base against branch | PASS: every switch-on arm = its base arm ([server_verdict.txt](server_verdict.txt)) |

**What `store_at` checks.** A prompt of 60 tokens in micro-batches of 16, the reference decoded as three calls cut at
17 and 34. One call of all 60 tokens, with a fill at each cut, ends in the reference's end state and last logits byte
for byte; each fill's host copy is the reference's state at its cut, byte for byte, with the same position range; the
first fill, restored, continues as the reference restored from its host state at that cut does; the cuts alone give
the same end; a fill's result is gone after the next call. Without rollback snapshots the cut calls take one-token
micro-batches ([16, 17) and [33, 34)), which round the recurrent state differently from longer ones: that is why one
uncut call ends elsewhere, and so the check sees the split. With snapshots for 3 drafted tokens the cut calls take
13 + 4 and 13 + 4 tokens, which round as an uncut call's 16 + 16 do; there the check sees the rule instead: the cuts
without it (one-token micro-batches again) end elsewhere in every run. Whole states (`store_at`) and partial states
(`store_at_p`, what the server saves) alike; mamba has no snapshots, so its runs are the first kind only.

**The server check** ([run/server_check.sh](run/server_check.sh), the random qwen4exp model, `-b 512 -ub 128`, greedy,
token prompts of 3,000 to 3,566 tokens: a fresh prompt, two that resume from a checkpoint, the live state continued,
an edit after a checkpoint; [server_summary.tsv](server_summary.tsv), [server_checkpoints.tsv](server_checkpoints.tsv)):

| Arm | Flags | Checkpoints | Inside a call (filled) | Against |
|---|---|---|---|---|
| `off` | `--ctx-checkpoints-vram 8` | 10 | 0 | `base`: the same |
| `incall` | `--ctx-checkpoints-vram 8 --ctx-checkpoints-incall` | 10 | 9 (9) | `base`: the same |
| `incall2` | `--ctx-checkpoints-vram 2 --ctx-checkpoints-incall` (4 moved to the host) | 10 | 8 (8) | `base`: the same |
| `incall4` | `--ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall` | 10 | 9 (9) | `base4`: the same |
| `sincall` | `--ctx-checkpoints-vram 8 --ctx-checkpoints-spread 200 --ctx-checkpoints-incall` | 30 | 21 (16) | `sbase`: the same |

"The same" is the tokens, the top 3 probabilities, `prompt_n`, `cache_n` and the restored checkpoint of all five
requests, and the checkpoints' token counts in the order taken. Two of the three restores in `incall` and `incall4`
resume from a checkpoint taken inside a call; in `incall2` one resumes from the host copy of a store filled inside a
call (moved to the host in the background), and in `incall4` the list is full from the third request on. A checkpoint
at the first token of a request is taken before the call, as before; so is one where no store was free (`incall2`'s
at 3,562 in the third request) and one where the joined call would not have fit whole (the parts `sincall` took
back). In `sincall` 5 of the 21
were evicted by a later checkpoint of the same batch before their call ran, so they were never copied; the base copies
them and then evicts them. `ckpt_save_ms` of a checkpoint taken inside a call is the host's bookkeeping after the call
(0.001 ms); its copy runs in the call. The CPU times are not speeds: the model is a 19 MB random one.

Two earlier runs of the same driver (`cff8d7f59`, `9fb51d28f`) passed too, except that the spread arm of the second
logged fewer `created` lines than its base: in-call checkpoints evicted before their call were never created. Since
`68ea07e60` the server logs each in-call checkpoint when it joins its batch, which the check compares. The first run's
spread arm took fewer checkpoints inside calls (7) because a part went on only when a store was free before room was
made; since `9fb51d28f` a full list counts the store that making room frees.

## What the CPU cannot check, and the rig row

- **The cards.** On the CPU the copies are synchronous, so the CPU checks cover the positions, the bytes and the
  splits, not the stream order on the cards (a copy queued behind its micro-batch's work, ahead of the next one's). The
  rig row checks it the way that shows: the greedy replies, the checkpoints' positions and the restores against the
  switch off, in one build. On nine P106s all three are the switch-off load's (Q1306).
- **The MTP draft.** The random models have no MTP block, so the draft's cuts and fills and its pending row at a
  checkpoint run only on the rig, where the replies and restores above cover them. A one-card run of the `store_at`
  cases on the build machine (Q1301) checks the fills' stream order on a CUDA card directly. It passes: both cases on one
  RTX 3080 through the compute 6.1 PTX, with the models and the state stores on the card
  ([FN72](../2026-11-11_fn72_c217_ptx/README.md)).
- **The speed.** Measured on the rig: 78.55 s a pass (−18.99 %) of the agent check's prompts with the taper off. The
  micro-batches stay the cut calls' (the 4-token one at a prompt's end too).

## The rig row (Q1306)

R43's replay ([2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md), its `run/r43_run.sh` with C218r's
`rig_taper.sh` and `taper_replay.py`): the agent check's 62 prompts, fresh text of their sizes, 8 greedy tokens a turn,
the taper off and on per request, two passes a load, and four fresh 2,049-token prompts. Two loads of the c217 tarball,
one after the other, each detached, nothing else on the rig:

```bash
# v: the device stores, the switch off (R43's run 2 on this build)
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c217v LOADS=A \
  ARGS_ADD='--ctx-checkpoints-vram 4' setsid nohup bash $WORK/r43_run.sh > $WORK/c217v.out 2>&1 < /dev/null &"
# i: the same with the checkpoints inside the calls
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c217i LOADS=A \
  ARGS_ADD='--ctx-checkpoints-vram 4 --ctx-checkpoints-incall' setsid nohup bash $WORK/r43_run.sh > $WORK/c217i.out 2>&1 < /dev/null &"
```

`BUILD` is the c217 tarball's folder, `MODEL` the first shard of Swift Qwen3.8-Flash-Next Pascal Q2 GGUF and `DRAFT`
its MTP draft, as in R43. Read both with R43's `run/lane_tables.py`. Right when, with the taper off, `i` gives `v`'s
greedy replies on all 62 turns, its server log shows the same checkpoints (token counts, in order) with those inside a
call marked `inside the call`, and its restores resume at the same counts. The verdict is `i`'s prompt seconds a pass
against `v`'s two passes (gain, negligible or loss, by the spread). The taper's passes are reported beside them: a
joined call has one taper plan, so its micro-batches differ from the cut calls'. Est. ~40 min a load.

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every step's verdict on the build machine |
| [context.tsv](context.tsv) | `test-llama-context` on the three random models: every case and run |
| [control.txt](control.txt) | the `store_at` cases' controls, per model and state kind |
| [server_summary.tsv](server_summary.tsv) | `server_check.sh`: every arm and request (prompt and cache counts, checkpoints, tokens) |
| [server_checkpoints.tsv](server_checkpoints.tsv) | the token counts of every arm's checkpoints, in the order taken |
| [server_verdict.txt](server_verdict.txt) | its verdict |
| [run/checks.sh](run/checks.sh) | the build machine's driver: rig tarball, CPU builds of the branch and the base, the checks |
| [run/server_check.sh](run/server_check.sh) | the llama-server check |

## Commands

On a build machine with Docker (CPU only; no GPU), from a checkout of this repository, with the fork as remote `fork`
of `$WORK/src` (a llama.cpp clone with an https fetch URL for the fork) and a compiler cache in `$WORK/ccache`:

```bash
WORK=<a folder of its own> BUILD=scripts/build_llamacpp_docker.sh BRANCH=pascal-p106-v4-b12246-c217 BASE_REF=84f7bd3d2 \
  LOCK=<a flock file shared with the machine's other jobs> JOBS=8 THREADS=8 DOCKER_ARGS="--label <your label>" \
  setsid -f nohup bash docs/data/2026-11-11_c217_incall_checkpoints/run/checks.sh > <log> 2>&1 < /dev/null
```

[run/checks.sh](run/checks.sh) takes the lock once and runs, in order: the rig tarball (`TESTS=1
scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c217`, `-j 8`, CPU shares 256, `nice -n 19`); static
native CPU builds (`-DGGML_NATIVE=ON -DGGML_CUDA=OFF`) of the branch (`test-llama-archs`, `test-llama-context`,
`llama-server`) and of the base `84f7bd3d2` (`test-llama-archs`, `llama-server`); the random models
(`test-llama-archs -a '^<arch>$' -s 1 -o <dir>` for qwen4exp, qwen35 and mamba) and `test-llama-context -m <model>`
on each; `test-llama-archs -b CPU -s 1` with `LLAMA_ARCHS_LOGITS` set, the base's run writing the logits and the
branch's comparing; and [run/server_check.sh](run/server_check.sh) on the qwen4exp model. `STEPS` picks steps
(default all, in order); `$WORK/summary.tsv` collects the verdicts. One case by hand:

```bash
<cpu build>/bin/test-llama-context -m <random qwen4exp gguf> -t 8
```

## Provenance

Code: the fork's branch `pascal-p106-v4-b12246-c217` (MIT, as llama.cpp). Every number from the build machine comes from
random tiny models that the tests generate; no model file or user data is involved. The R43 numbers quoted above are
from [2026-11-10_prompt_lane](../2026-11-10_prompt_lane/README.md).
