# C405 (Q960): host timestamps along the served speculative step (2026-11-10, the night's code rows)

On the Miner Llama v3.0 release line a served speculative step of the Flash-Next preset is 131.57 ms (median, greedy
request), and 7.84 ms of it is the server between the draft's last pass and the next verify pass: no card works then,
and it is the one host part above 5 ms ([S37](../2026-11-10_served_step_split/README.md)). The fork's host profile
(`GGML_CUDA_PROFILE_HOST`) cannot split it: it sees only each backend's `graph_compute`, and everything the server,
the drafter and `llama_context` do between two evaluations falls into the same interval.

This folder holds a new diagnostic switch on Miner Llama's v4 line that writes those host steps down, its CPU check,
the script that puts its file beside the host profile, the rig run's commands, and the list of what in the 7.84 ms
could leave the token's path, read from the code. The registry row is Q960 in
[quants/experiments.md](../../../quants/experiments.md); the switch is in [docs/switches.md](../../switches.md).

**Result:** the switch `LLAMA_STEP_TRACE=<file>` writes one line per served step, with a host timestamp at each
stage of the server, the MTP drafter and both contexts, on the clock the host profile now names. On a build machine's
CPU, with a tiny random `qwen4exp` model, llama-server gives the base build's tokens and draft counts in 15 of 15
replies with the trace off, on, and pointed at a path it cannot write. The trace's drafted and accepted counts equal
the server's, and `step_gap.py` splits every step. With the switch off, every architecture's logits equal the base's
(146 of 146). The rig tarball builds. Even on the CPU the trace shows three things:
- the MTP draft context rebuilds its graph twice a step;
- the drafter runs passes the slot then throws away at a reply's end (Q1314);
- every output read waits for the device again.

On a P106 the gap is 7.51 ms on the v4 line, and 5.61 ms of it is the scheduler copying card 0's inputs. The host's own
stages take 1.48 ms (Q1313, [2026-11-11_r86_step_trace](../2026-11-11_r86_step_trace/README.md)). The six moves below
(Q1307–Q1312) were read from the code; that run ranks them.

## The change

Branch `pascal-p106-v4-b12246-c405` at **`ca8725c62`**, three commits on `pascal-p106-v4` (`84f7bd3d2`, the
`-ladder6` tree): the trace (`0d598ab8c`), its accepted count moved to where the server counts it and the tiny MTP
model for the CPU check (`ada988253`), and a fix to the model saver that the tiny model needed (`ca8725c62`):

| File | What |
|---|---|
| `src/llama-step-trace.cpp`, `src/llama-step-trace.h`, `src/llama-ext.h` | `LLAMA_STEP_TRACE=<file>`, read once. `llama_step_trace_begin`, `_mark`, `_count` and `_end` keep a step's marks (a name and the steady clock's time) and write them as one JSON line; `llama_step_trace_role` names a context's marks |
| `src/llama-context.cpp`, `src/llama-context.h` | each context marks the stages of `decode()` and every real wait in `synchronize()`, as `v.<stage>` (a main context) or `d.<stage>` (an MTP context, or the context the server names the draft's) |
| `common/speculative.cpp` | the MTP drafter marks its catch-up (`mtp.cu*`) and each drafting pass: the decode called and returned, the sampled id read, the next batch built (`mtp.dp*`) |
| `tools/server/server-context.cpp` | the server marks its loop: the task post, the slot selection, the draft and its yield, the checkpoints, the batch, the render, `decode()` and its yield, the verify pass's synchronize, the catch-up, the sampling and acceptance, the drafter's accept, the rollback, each token's processing; it counts the batch's tokens, the drafted and the accepted tokens; it names the draft context `d` |
| `ggml/src/ggml-cuda/ggml-cuda.cu` | `GGML_CUDA_PROFILE_HOST`'s log line also gives the steady clock's time of its file's 0 ms |
| `tests/test-llama-archs.cpp` | `-o <dir> --mtp` saves `qwen4exp`'s MoE test model with one MTP block (`qwen4exp-moe-mtp.gguf`), the model the CPU check serves |
| `src/llama-model-saver.cpp` | a per-layer key is written with one value per block, the MTP blocks included, since the loader reads `block_count` values; before, a model with an MTP block saved a file that did not load again. A model without MTP blocks saves the same file as before |

**With the variable unset nothing else changes.** A context checks one flag of its own; the server's and the drafter's
calls return at the first line (a function-local static). No sampler, batch, memory or timing code moves: the marks are
calls added between the existing lines, and the server's own `timings` are computed as before. The saver's fix changes
only files of models with MTP blocks; the CPU check's `archs` step saves and reloads every other architecture's model
and gets the base's logits.

### The file

The first line is a header, `{"trace":"llama-step","clock":"steady","t0_steady_ms":<t>}`: the steady clock's time of
the file's 0 ms. Then one line per `update_slots` call that decoded:

```
{"step":<n>,"write_us":<the previous line's write>,"n":{"tok":4,"drf":3,"acc":2},"t":[["s.upd",<ms>],...,["s.upd.end",<ms>]]}
```

Times are ms since the header's 0, at microsecond resolution, in the order the marks were made. With
`GGML_CUDA_PROFILE_HOST` in the same process, the server log carries "its 0 ms is <t'> ms on the steady clock", so a
profile time p is at p + t' - t on the trace's scale: the profile's per-card entries and GPU times land between the
trace's marks.

A generation line holds, in order: the server's loop start (`s.upd`, `s.post`, `s.shift`, `s.sel`); the draft
(`s.drf`, the yield's start `s.drf.in`, `mtp.drf`, then per drafting pass `mtp.dp`, the draft context's `d.dec` …
`d.ub.comp` and `d.dec.end`, `mtp.dp.ret`, its wait `d.sync`/`d.sync.end`, the sampled id `mtp.dp.smp`, the next batch
`mtp.dp.row`; then `mtp.drf.end`, `s.drf.out`, the yield's end `s.drf.end`); the checkpoints (`s.ckpt`, `s.ckgen`);
the batch (`s.bat`, `s.pre.end`, `s.slots.end`, `s.render`); the verify pass (`s.dec`, `s.dec.y`, `s.dec.in`, `v.dec`,
`v.dec.bal` the batch allocator, `v.dec.mem` the memory slot, `v.dec.out` the outputs, `v.ub.apply`, `v.ub.reuse` or
`v.ub.build`, `v.ub.in` the inputs set, `v.ub.comp` the nine cards' graphs queued, `v.ub.cp` the copies back queued,
`v.dec.end`, `s.dec.ret`, the wait `v.sync`/`v.sync.end`, `s.dec.out`, `s.dec.end`); `s.mpd`; the catch-up (`s.cu.in`,
`mtp.cu`, `mtp.cu.dec`, `d.dec` … `d.dec.end`, `mtp.cu.ret`, `mtp.cu.end`, `s.cu.out`, `s.cu.end`); the acceptance
(`s.pd`, `s.acc`, the sampler's clone `s.acc.cl`, `s.acc.smp`, the drafter's accept `s.acc.spec`, the rollback `s.rm`,
`s.tok` per output token, `s.tg`); `s.upd.end`.

### Reading it: `run/step_gap.py`

`step_gap.py TRACE [--profile HOST_PROFILE.csv --server-log SERVER.log]` prints three tables over the typical
generation steps (the commonest sequence of marks, the first three steps after each prompt left out):
- **waterfall:** every pair of consecutive marks, median, mean, p10 and p90; the rows add up to the line;
- **the gap:** the stages from the last drafting pass's wait (`d.sync.end`) to the verify pass's inputs set
  (`v.ub.in`). With the profile, two rows more: from the end of the last drafting pass's work on card 8 to that wait's
  end (the sampled id's way back and the host's wake), and from `v.ub.in` to card 0's `graph_compute` entry (the
  scheduler's input copies for card 0). Together they are S37's "server between the last draft pass and the next
  verify pass";
- **between lines:** from a line's end to the next line's start (the line's own write, the server's task loop).

`--check` verifies the file (the header, every line, times that never go back, at least one drafting pass per drafted
token, one main decode per line). A run of back-to-back waits of one context is folded into one pair, `d.sync(6)` for
six: each getter of a pass's output synchronizes once.

## What in the 7.84 ms could leave the token's path

Read from the code of the path between the last drafting pass and the verify pass (the v4 line, `84f7bd3d2`). Every
host step there delays the verify pass's launch, while the host is idle for most of the step: about 100 ms in the
verify pass's wait and 2.6 ms in each drafting pass's. Host work placed after the verify pass's launch, or before the
first drafting pass (where the catch-up's 3.63 ms of card time already hides the server's acceptance work, S37's
negative turnaround), costs the step nothing. The rig run (below) gives each stage's size; the rows are ranked by it.

| Row | What moves | From where to where | Exact? |
|---|---|---|---|
| Q1307 | **The verify pass prepared during the last drafting pass.** All of the verify's host work except its token ids: the batch's positions and output flags, the batch allocator, the memory slot (`find_slot`, the recurrent snapshots' cells), `apply`, the graph reuse check, every input but the token ids (positions, the attention mask, the recurrent copies, the backend samplers' inputs) | from after the last sampled id to the 2.6 ms the host now waits in the last drafting pass's `synchronize`; after the sampled id only the four token ids are written and the graphs queued. Needs `llama_context::decode` split in two (prepare a batch shape, then submit its ids); a draft cut short by `p_min` prepares again | yes: the same inputs, written earlier |
| Q1308 | **The draft context's `seq_rm` after the verify pass's launch.** The server removes the draft's drafted positions from the draft context's cache (`llama_memory_seq_rm(ctx_dft, pos_max + 1, -1)`) between the draft and the verify | to after `llama_process(ctx_tgt)` returns, before the verify pass's wait: the draft context is not used again until the catch-up | yes |
| Q1309 | **`seq_rm` bounded by the used cells.** `llama_kv_cache::seq_rm` visits every cell of the cache, 65,536 at the preset's context, on every call: once in the gap (the draft context, Q1308 moves it) and once after the acceptance (the main context, in the catch-up's shadow) | a loop to `used_max_p1` (or from a per-sequence index of positions) instead of the whole cache | yes: the cells past `used_max_p1` are empty |
| Q1310 | **Fewer yields to the task queue.** Three `yield_to_queue` a step (the draft, the verify decode, the catch-up), each a wake of the queue's worker thread and a wait for it to stop; the draft's end and the verify's start are in the gap, on a two-core host | the draft and the catch-up without a yield (they last a few ms and the worker only answers metrics tasks meanwhile), or one yield over the whole step | yes: the same work on the same thread |
| Q1311 | **No hidden row back from the last drafting pass.** Every drafting pass copies its 2,560-value h row to the host (10 KB over PCIe 1.0 x1) before its `synchronize` returns; the last pass's row is never read | the drafter tells the context that a pass is its last (no `h_nextn` output), as the hidden-row carry (`LLAMA_DRAFT_H_CARRY`) already keeps the rows on the card | yes |
| Q1312 | **The server's per-step settings before the draft.** `common_set_adapter_lora`, `llama_set_embeddings`, the taper string check, `metrics_pre_decode`, the screened-head and head-bound decisions and the batch's first token (the last accepted one) depend on nothing the draft returns | from between the draft and the verify to before the first drafting pass | yes |

Not new rows: the drafted ids staying on card 8 and fed to the next pass on the card is Q1148 and Q969 (parked below
5 ms; the trace's sampled-id rows reopen them); the attention mask built on the cards for decode steps is Q261 and Q430
(its copy for card 0 sits in the last row of the gap table, after `v.ub.in`); the draft context's graph built again
for its catch-up and its first drafting pass is Q1104.

One more row came from the CPU check's trace, outside the 7.84 ms. **Q1314: the MTP drafter stops at the slot's draft
cap.** At a reply's end the server caps the draft at the tokens still to generate, but the drafter runs its three
passes anyway and the extra tokens are cut afterwards. That is one or two passes a reply, so its gain is negligible.

## The CPU check

On a build machine's CPU (`run/cpu_check.sh`, no GPU): the branch's static native build of `llama-server` against the
base's (`84f7bd3d2`). The models are `test-llama-archs`'s random `qwen4exp` MoE model and the same model with one MTP
block (`-o --mtp`), seed 1. Each arm answers three requests: greedy 96 tokens, temperature 1.0 with seed 42 48 tokens,
and greedy 64 tokens on a longer prompt. Draft-mtp drafts three tokens with the model's own MTP layer, as the release
preset does; draft-simple uses the plain model as a separate draft model.

| Check | Verdict | Detail |
|---|---|---|
| CPU build | OK | `ca8725c62` |
| identity | OK | 15 of 15 replies give the base's tokens and draft counts: the trace off, on, and on with an unwritable path (draft-mtp, 9 replies), off and on (draft-simple, 6) |
| draft-mtp trace | OK | 212 lines; drafted 597 (the server's timings: 597), accepted 0 (0: a random MTP layer); `step_gap.py --check` passes |
| draft-simple trace | OK | 90 lines; drafted 180 (180), accepted 122 (122); `step_gap.py --check` passes |
| the trace off | OK | no file written |
| an unwritable path | OK | one error line at the start; the server serves as with the trace off |
| every architecture | OK | `test-llama-archs -b CPU -s 1`: 146 of 146 architectures give the base's logits; 151 of 151 tests pass in both builds |

What the CPU traces already show (`cpu_gap_mtp.md`; a tiny model on a desktop CPU, so the sizes are not the rig's):

| Seen in the trace | On the CPU | On the rig |
|---|---|---|
| the MTP draft context builds its graph again for the catch-up and for the first drafting pass of every step; passes 2 and 3 and the verify pass reuse theirs | 0.087 and 0.080 ms a step | Q1104's open question (C619): the catch-up's build is on the token's path, the first drafting pass's runs while the catch-up computes |
| each yield's end (the worker thread stopped) | 0.015 to 0.016 ms, three a step | Q1310 |
| the main context's `seq_rm` after the acceptance | 0.015 ms over 4,096 cells | the preset holds 65,536 cells (Q1309) |
| the draft context's bookkeeping after the draft (`s.drf.end → s.ckpt`) | 0.007 ms | Q1308, Q1309 |
| the host's part of the gap (`d.sync.end → v.ub.in`) | 0.060 ms (draft-mtp), 0.087 (draft-simple) | Q1313 |
| drafting passes beyond the drafted tokens | 9 in 6 of 202 steps, the last two of each reply | Q1314 |
| waits per output read: every getter synchronizes once | runs of 6 waits after each drafting pass and of 24 in the acceptance | on the CUDA backend a wait with nothing in flight sweeps all nine cards unless `LLAMA_INPUT_UPLOAD` is on (C714/Q1266); the trace counts them |
| the trace's own cost | a line's write 50 µs, between steps | the rig run compares the step with and without the trace |

## The rig run (Q1313)

One untimed load of the v3 preset on rig 1 with both files, and one with the host profile alone, each answering S37's
greedy request and the agent's first-turn request (`run/rig_trace.sh`, about 15 minutes of rig time). It ran on the v4
night merge `pascal-p106-v4-b12246-night` (`c408dabcb`), which holds this branch, with every other switch off. This
branch's own rig tarball is `llama.cpp-20261111-ca8725c62-pascal-p106-v4-b12246-c405` (md5
`b257824e744109af446e14609f4c6240`, `TESTS=1`, sm_61, 0 AVX instructions). The run also gives the v4 line's step split
beside S37's v3.0 one. `step_gap.py` splits the gap stage by stage. S37's `step_split.py` on both loads checks that
the trace leaves the step alone, on the greedy request's paired steps (it draws nothing at random). The result, the
tables and the commands are in [2026-11-11_r86_step_trace](../2026-11-11_r86_step_trace/README.md); its server log
lacked the profile's anchor line, so its `run/align_t0.py` finds the profile's offset from the trace.

## Files

| File | What |
|---|---|
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: the branch's native CPU build of `llama-server`, the server smoke with the trace off and on against the base build, every architecture's logits, the verdicts |
| [run/step_gap.py](run/step_gap.py) | the trace's tables, alone or beside the host profile; `--check` |
| [run/rig_trace.sh](run/rig_trace.sh) | the rig run's driver (two loads, two requests each), as S37's `step_loads.sh` |
| [cpu_summary.tsv](cpu_summary.tsv) | the CPU check's verdicts |
| [cpu_gap_mtp.md](cpu_gap_mtp.md), [cpu_gap_sim.md](cpu_gap_sim.md) | `step_gap.py`'s tables of the CPU check's two traces (a tiny random model on a desktop CPU: the format and the order of the stages, not the rig's sizes) |
| [cpu_trace_mtp_head.jsonl](cpu_trace_mtp_head.jsonl) | the first 11 lines of the CPU check's draft-mtp trace (the header, the prompt's lines, the first steps) |

## Commands

On a build machine (`Q` reaches it; `W` its work folder, with a llama.cpp clone in `$W/src` whose remote `fork` is
Miner Llama, and a ccache in `$W/ccache`; `BASE_BIN` a static native CPU build of the base `84f7bd3d2`):

```bash
# the rig tarball (TESTS=1: sm_61 code, no AVX, the test binaries)
$Q "cd $W && env WORK=$W TESTS=1 JOBS=8 DOCKER_ARGS='--cpu-shares 256' nice -n 19 \
    bash scripts/build_llamacpp_docker.sh fork/pascal-p106-v4-b12246-c405"
# the CPU check: a native CPU build of llama-server, then the smoke
$Q "env WORK=$W BRANCH=pascal-p106-v4-b12246-c405 BASE_BIN=$BASE_BIN JOBS=8 THREADS=4 nice -n 19 \
    bash docs/data/2026-11-11_c405_server_gap/run/cpu_check.sh"
```

The rig run, as S37's:

```bash
# on the rig, the rig idle (BIN: this branch's tarball's bin/, MODEL: the Swift Q2_0 first shard, DRAFT: the MTP draft,
# S: a folder with this repository's files, LONG: the first-turn request built by
# BUILD_SSH=<build machine> bash bench/agent_check/first_turn_request.sh long_request.json)
BIN=$BIN MODEL=$MODEL DRAFT=$DRAFT PRESET=$S/release/flash-next-v3/settings.env \
  RIG_SERVER=$S/bench/agent_check/rig_server.sh REQ_LONG=$LONG \
  REQ_GREEDY=$S/docs/data/2026-11-04_draft_one_card/run/greedy_request.json OUT=$OUT \
  setsid nohup bash $S/docs/data/2026-11-11_c405_server_gap/run/rig_trace.sh > console.out 2>&1 < /dev/null &

# on a build machine, CPU only (OUT copied there; RES a new folder; R=docs/data/2026-11-11_c405_server_gap/run,
# P=docs/data/2026-11-10_served_step_split/run)
nice -n 19 python3 -I $R/step_gap.py $OUT/trace/step_trace.jsonl --profile $OUT/trace/host_profile.csv \
  --server-log $OUT/trace/server-trace.log --tsv $RES/gap_steps.tsv > $RES/gap_trace.md
for l in trace prof; do
  nice -n 19 python3 -I $P/step_split.py $OUT/$l/host_profile.csv --load $l --requests greedy,long \
    --tsv $RES/steps_$l.tsv > $RES/split_$l.md
done
```

## Provenance and licence

The code is Miner Llama's (MIT, as llama.cpp). The CPU check's models are random `qwen4exp` models that the branch's
`test-llama-archs` writes (`-a ^qwen4exp$ -s 1`, with and without `--mtp`); no real weights were read. The tables are
this project's measurements.
