# C703 (Q1242): the MTP draft's prompt catch-up runs while the next call starts (2026-11-11, the night's code rows)

llama-server reads a long prompt in calls of up to `-b` tokens (2,048 in the preset), and a context checkpoint inside
the prompt ends a call early. After each call the MTP draft reads that call's tokens with the target's hidden rows (the
draft's catch-up, `process()` of the `draft-mtp` driver in `common/speculative.cpp`), and only then does the server
queue the next call. In R42's timeline of a fresh 4,097-token prompt on the v3.0 release line the catch-up took
2,944.1 ms in 32 passes of 128 rows, 9.8 % of the prompt's window, on cards 4 and 8 while the other seven cards idled;
before it the host waited for the call to leave card 8
([R42](../2026-11-10_prompt_timeline/README.md), Q1156). The next call's first micro-batch crosses cards 0 to 7
before it needs card 8, so the catch-up of call n fits under the start of call n + 1 if the server queues call n + 1
first.

This folder holds that change, behind a new switch on Miner Llama's v4 line, and its CPU check. The registry row is
Q1242 in [quants/experiments.md](../../../quants/experiments.md); the switch is listed in
[docs/switches.md](../../switches.md). It sits on C620's branch (`LLAMA_MTP_KV_ONLY`,
[design](../2026-11-11_c620_draft_kv_catchup/README.md)) and works with that switch on or off.

**Result:** the switch `LLAMA_MTP_CATCHUP_OVERLAP=1` lets the draft catch up on a prompt call once the next call is
queued on the target. On the CPU backend the draft's sequence state and the driver's state at a checkpoint, after the
prompt and at the end, and every draft with its probabilities, are the same bytes with the switch off and on. That holds
in 30 of 30 runs: five configurations (F16 and Q8_0 caches, flash attention on and off, C620's switch on, the driver's
top-k on the backend and on the CPU) at three seeds, each with the rig build's CPU code and with a native CPU build.
With the switch on, four of the test's prompt calls were caught up after the next call was queued; with it off, none.
C620's check passes unchanged on this branch (30 of 30). With the switch off the branch gives the base's logits bit for
bit on all 146 architectures `test-llama-archs` builds. The rig tarball builds (0 AVX instructions). The time it takes
off a prompt on the cards, and the drafts' identity there, are Q1304.

## The change

Branch `pascal-p106-v4-b12246-c703` at **`db253c8a7`**, two commits on C620's `pascal-p106-v4-b12246-c620`
(`ca3055285`): `648b276f5` the change and its check, `db253c8a7` the check's table kept free of the driver's log lines:

| File | What |
|---|---|
| `src/llama-context.cpp`, `src/llama-context.h`, `src/llama-ext.h` | the target's h_nextn device buffer in two halves (`llama_set_embeddings_nextn_dev_halves`), a hold on the last decode's half (`llama_hold_embeddings_nextn`) and its first row (`llama_get_embeddings_nextn_row0`); the buffer's events per half |
| `src/llama-graph.cpp`, `src/llama-graph.h` | the target's graph writes its rows at the half's first row, read when the inputs are set (`h_nextn_row0`), so that graphs are reused across halves |
| `common/speculative.cpp`, `common/speculative.h` | the `draft-mtp` driver keeps a prompt call's catch-up until the next call; `common_speculative_pending`, `common_speculative_flush` and `common_speculative_after_catchup` |
| `tools/server/server-context.cpp` | where llama-server lets a catch-up wait, and its own draft work queued behind it |
| `tests/test-llama-archs.cpp` | `--mtp-catchup-overlap`: the CPU check below |

**The target's rows.** With the MTP draft the target writes each call's hidden rows into a buffer on the output card
(`n_batch + n_seq_max` rows of the model's `n_embd_out` values; 2,049 rows of 10,240 F32 values, 80.0 MiB, in the
preset), and the draft's graph reads its rows there. With the switch the buffer has two such halves. A decode writes
the half the last decode wrote, unless the draft holds that half: then it writes the other one. Each half has its own
pair of events (written by the target, read by the draft), so the draft waits only for the call whose rows it reads
and the target only for the reader of the half it writes. The second half costs the output card 80.0 MiB in the
preset.

**The driver.** `process()` of a batch whose tokens all have ids and none is an output row (a prompt call before the
last) keeps a copy of the batch's tokens and the first row of its half, holds that half and returns. The next
`process()` (the server calls it right after it queued the next call on the target) first runs the kept catch-up, then
handles its own batch. The kept catch-up is the code of the immediate one with the rows offset by the half's first
row: the same batch, the same pending row from the call before, the same passes in the same order. A call with an
output row (the prompt's last call, a verify step) is caught up at once, as before. Every other entry point of the
driver (`begin`, `draft`, `accept`, `get_state`, `set_state`) runs a kept catch-up first.

**The server.** Between two calls llama-server touches the draft context itself in two places, and these wait for the
kept catch-up (`common_speculative_after_catchup`, run in their order right after it):
- the cut of the memory past the next batch's start (`memory_seq_rm [p0, end)`): the target's part now, the draft's
  after the catch-up;
- a checkpoint inside the prompt: the target's part now, the draft's part and the driver's state after the catch-up.

The catch-up stays kept only while one slot reads its prompt on. A turn of the server loop with any other slot busy,
or with the slot generating, runs it first, and so does every task but the loop's own turn and the reads of the
server's state (metrics, the slot list, the LoRA list).

**Exactness.** The draft's context sees the same operations in the same order as without the switch; only their time
relative to the target's next call changes. The target computes the same values; only the rows of its h buffer they
land in change. So the draft's caches, its drafts and the server's checkpoints are the ones the immediate catch-up
gives.

**With the switch off nothing changes:** the buffer has one half, its first row is 0, the events of half 0 are the
buffer's old events, and no catch-up is kept (`common_speculative_flush` finds nothing).

**Limits.**
- llama-server only. Another program that uses the driver and touches the draft context between `process()` calls
  must keep the switch off or run `common_speculative_flush` first.
- C217's checkpoints inside a decode call (`--ctx-checkpoints-incall`, Q770) set the draft's cuts around
  `common_speculative_process` and read them right after it; a kept catch-up would run without them. The two together
  need Q1302 (C735).
- A growth of the draft's compute buffer during the catch-up frees the old buffer, and CUDA's free waits for every
  stream on the card, the target's included (R42 counted 4 size changes a 4,097-token request). Q1305 is the
  follow-up.
- The kept catch-up starts once the server has queued the whole next call. R42 found the host blocked inside a call's
  third queued micro-batch until the cards catch up (est. a full CUDA launch queue), so the catch-up starts later than
  the next call's first micro-batch. Q1106's worker thread would start it while the host is still held there.

## The CPU check

`run/checks.sh` on a build machine, CPU only. It builds the branch without CUDA (static, native: `test-llama-archs`,
and `llama-server` to compile its hooks), runs the checks below with it, builds the branch's rig tarball (`TESTS=1`,
sm_61 code, no AVX) and `test-llama-archs` from the same build tree, and runs the checks again with the rig build's
binary in a container without a GPU (CUDA's stub driver: no CUDA device, the CPU backend runs).

- **overlap:** `test-llama-archs --mtp-catchup-overlap` at seeds 1, 2 and 3.
- **kvonly:** C620's `--mtp-kv-only`, unchanged by this branch.
- **off:** `test-llama-archs -b CPU -s 1` on every architecture the test builds, the logits written by the base's CPU
  build (C620's `ca3055285`) and compared by the branch's: with the switch off the branch computes what the base
  computes.

`--mtp-catchup-overlap` builds a random `qwen4exp` model of the test's fixture (two trunk layers, hyper-connections of
four streams, QSA with blocks of four, MoE) with one MTP block after the trunk, also a QSA layer. It makes a target
context (calls of up to 32 tokens in micro-batches of 8) and an MTP context (micro-batches of 18, so that blocks of
four complete across them) on it, and drives both through the `draft-mtp` driver in llama-server's order, once with
the switch off and once on:

1. a 100-token prompt in four calls (32, 32, 32 and 4 tokens, an output row on the last token only), each call
   through the target and then `common_speculative_process`; before each call the cut past its start
   (`common_memory::seq_rm`, the draft's part through `common_speculative_after_catchup`), after the second call a
   checkpoint's draft part (the draft's sequence state and the driver's, through `common_speculative_after_catchup`);
2. the draft's sequence state and the driver's after the prompt;
3. three draft steps from the target's argmax: the draft (`n_max` 3), its region removed from the draft's memory, the
   verify batch through the target and the driver, and `accept` with 0, 1 and all of the draft;
4. a second turn: the sampled token alone, then 44 prompt tokens in two calls (31 and 13);
5. one more draft step (`accept` 2), and both states at the end.

On the CPU the switch-off run reads the target's rows from host memory (the CPU device has no events, so the
one-half device buffer is not made); the switch-on run reads them from the device buffer's two halves (a CPU device
needs no events there, since a CPU decode is done when it returns). So the check covers the halves, the hold, the
kept catch-up and its queued work against the base's path.

| Config | Cache | Flash attention | `LLAMA_MTP_KV_ONLY` | The driver's top-k |
|---|---|---|---|---|
| `f16 fa` | F16 | on | off | on the backend |
| `f16` | F16 | off | off | on the backend |
| `q8_0 fa` | Q8_0 | on | off | on the backend |
| `f16 fa kv-only` | F16 | on | on | on the backend |
| `f16 fa host` | F16 | on | off | on the CPU sampler |

A configuration passes when the checkpoint's draft state, both states after the prompt, every draft and its
probabilities, both states at the end and the checkpoint's driver state are the same bytes with the switch off and
on, and when no catch-up waited with the switch off and four did with it on (the first prompt's three calls without
an output and the second turn's first call; the second turn's first token waits as well and is not counted).

### Results

| Step | Verdict | Detail |
|---|---|---|
| CPU build | OK | static, native, without CUDA, `db253c8a7`: `test-llama-archs` and `llama-server` |
| rig tarball | OK | `llama.cpp-20261111-db253c8a7-pascal-p106-v4-b12246-c703.tar.gz`, 647,631,545 bytes, md5 `c8c1ee08e9b66e9ff8988ce661a4edac`; sm_61, `TESTS=1`, 0 AVX instructions in `llama-bench`, GLIBC_2.38 at most |
| `test-llama-archs`, rig tree | OK | built from the tarball's CUDA build tree into its `bin/` |
| overlap, rig build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| overlap, CPU build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| kvonly, rig build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| kvonly, CPU build | OK | seeds 1, 2, 3: 5 of 5 configurations each |
| off | OK | the base (`ca3055285`) wrote 146 architectures' logits, the branch compared 146: 0 differ; 151 of 151 test cases pass in both |

[summary.tsv](summary.tsv) holds the verdicts, [overlap.tsv](overlap.tsv) and [kvonly.tsv](kvonly.tsv) one row per
build, seed and configuration. The overlap rows are the same at every seed and in both builds:

| Config | Checkpoint (draft) | After the prompt | Drafts | At the end | Checkpoint (driver) | Kept catch-ups, off / on |
|---|---|---|---|---|---|---|
| `f16 fa` | identical | identical | identical | identical | identical | 0 / 4 |
| `f16` | identical | identical | identical | identical | identical | 0 / 4 |
| `q8_0 fa` | identical | identical | identical | identical | identical | 0 / 4 |
| `f16 fa kv-only` | identical | identical | identical | identical | identical | 0 / 4 |
| `f16 fa host` | identical | identical | identical | identical | identical | 0 / 4 |

The kvonly rows are C620's, node counts included (1,458 / 531 for `f16 fa`, 2,043 / 2,043 with `dist`).

**The first run, on `648b276f5`,** gave the same verdicts with the CPU build (overlap and kvonly 15 of 15, off 146 of
146), but the driver's own log lines split each row of the check's table, so the table could not be read by script;
`db253c8a7` pauses the log during the two runs of a configuration, and every step ran again on it. The first rig
build of `db253c8a7` stopped at once because the build script copied to the machine was not executable; it ran again
after `chmod`.

## How to run it

On a build machine with Docker, from a checkout of this repository, with the fork as remote `fork` of the work
folder's llama.cpp clone (`scripts/build_llamacpp_docker.sh`, its header):

```bash
WORK=<work dir> BUILD=scripts/build_llamacpp_docker.sh BRANCH=pascal-p106-v4-b12246-c703 \
BASE_CPU=<bin/ of the base's static native CPU build> JOBS=8 LOCK=<flock file> \
setsid -f nohup bash docs/data/2026-11-11_c703_catchup_overlap/run/checks.sh > <work dir>/logs/driver.out 2>&1 < /dev/null
```

`STEPS` picks steps (default `cpubuild overlap kvonly off build`); the run here took the lock three times, for
`STEPS="cpubuild overlap kvonly off"`, `STEPS=build` and `STEPS="overlap kvonly"`. Verdicts land in
`$WORK/summary.tsv`, the tables in `$WORK/checks/overlap/overlap.tsv` and `$WORK/checks/kvonly/kvonly.tsv`, and
`$WORK/C703_DONE` marks the end. One configuration by hand:

```bash
<build>/bin/test-llama-archs --mtp-catchup-overlap -s 1
```

## On the rig

**(a) ran on 2026-11-12 ([2026-11-12_r84a_c703_identity](../2026-11-12_r84a_c703_identity/README.md)): the identity
holds on a P106.** With the switch the load gives the same 39 verify steps as both control loads, every field the same,
and logs 20 kept catch-ups; the output card holds 80 MiB more. It ran on `pascal-p106-v4` `d097d6d88`, which holds this
branch, in place of the tarball below. (b) and (c) are not run yet: they are the overlap rung of R123a (Q1401).

Q1304 relays the tarball to a rig and runs three things, each with the `wait` and `collect` steps of
[bench/deep_lane](../../../bench/deep_lane/README.md) or R42's driver.

```bash
# (a) identity, untimed: every verify step's drafted tokens and probabilities the same with the switch as without,
#     wherever draft2 repeats draft1; copy the two dumps off the rig before its work folder is removed
RIG=rig BIN=<c703 build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 DEPTHS="0 32768" ARMS="draft overlap" \
ARM_ENV_draft=LLAMA_SPEC_DUMP=/tmp/deep-lane/dump-draft.jsonl \
ARM_ENV_overlap="LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_SPEC_DUMP=/tmp/deep-lane/dump-overlap.jsonl" \
LOADS="draft1:draft overlap1:overlap draft2:draft" \
bash bench/deep_lane/remote.sh start

# (b) R42's timeline of a fresh 4,097-token prompt, the switch off then on, on the c703 build (on the rig, detached,
#     from a folder $R43 as in R42's README)
ssh $RIG "cd $WORK && R43=$R43 BUILDS=$BUILDS BUILD=<c703 build folder> R42_LOADS='warm4096 graph4096' \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/off \
  setsid nohup bash $WORK/r42_run.sh > $WORK/r42_off.out 2>&1 < /dev/null &"
ssh $RIG "cd $WORK && LLAMA_MTP_CATCHUP_OVERLAP=1 R43=$R43 BUILDS=$BUILDS BUILD=<c703 build folder> \
  R42_LOADS='warm4096 graph4096' \
  MODEL=$MODELS/swift_q2_0/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=$MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=$WORK/on \
  setsid nohup bash $WORK/r42_run.sh > $WORK/r42_on.out 2>&1 < /dev/null &"
python3 -I docs/data/2026-11-10_prompt_timeline/run/timeline.py --load warm4096=off/warm4096 \
  --load graph4096=off/graph4096 --out tables-off
python3 -I docs/data/2026-11-10_prompt_timeline/run/timeline.py --load warm4096=on/warm4096 \
  --load graph4096=on/graph4096 --out tables-on

# (c) the prompt table: R53's primes with the draft, the draft with the switch, and no draft
RIG=rig BIN=<c703 build>/bin/llama-server \
MODEL=<models>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
MODEL_DRAFT=<models>/mtp-Qwen3.8-Flash-Next-draft.gguf \
TEXT_LOCAL=<recorded>/proxy.jsonl.bodies.gz TAG='^t1swift/' WD_ENV=MIN_AVAIL_MB=60 \
ENV_ADD=GGML_CUDA_FATTN_SPARSE_PASCAL=1 N_PREDICT=16 \
ARMS="draft overlap nodraft" ARM_ENV_overlap=LLAMA_MTP_CATCHUP_OVERLAP=1 ARM_DRAFT_nodraft=off \
LOADS="warm:draft draft1:draft overlap1:overlap nodraft1:nodraft draft2:draft overlap2:overlap nodraft2:nodraft" \
bash bench/deep_lane/remote.sh start
python3 -I docs/data/2026-11-10_deep_lane/run/draft_ubatch.py <results folder> --out draft_ubatch.md --tsv draft_ubatch
python3 -I docs/data/2026-11-10_deep_lane/run/host_memory.py <results folder> > host_memory.tsv
```

(b) runs the two driver calls one after the other, the second when the first's `DONE` is there. The second's server
log must have the switch's line (`LLAMA_MTP_CATCHUP_OVERLAP: a prompt call's catch-up runs once the target has the
next call queued`) and at its end the count of kept catch-ups.

**What to expect (est.):** in R42 the 4,097-token prompt's four calls (2,048, 1,533, 512 and 4 tokens) left 2,944.1 ms
of catch-up between calls. The last call's catch-up stays where it was (354.3 ms in R42); the other three's move under
the next call: est. up to 2.6 s of the prompt's 29,983.0 ms (8.7 %), less where the catch-up shares card 8 with the
next call's micro-batches or waits for its launches. With C620's switch on as well the catch-up itself is shorter
(R76), and so is the time this one can take away. The output card holds 80.0 MiB more.

## Files

| File | What |
|---|---|
| [run/checks.sh](run/checks.sh) | the build and the CPU check (steps `cpubuild`, `overlap`, `kvonly`, `off`, `build`), as run |
| [summary.tsv](summary.tsv) | the verdict of every step |
| [overlap.tsv](overlap.tsv) | `--mtp-catchup-overlap`: one row per build, seed and configuration |
| [kvonly.tsv](kvonly.tsv) | C620's `--mtp-kv-only` on this branch: one row per build, seed and configuration |

The check itself is `test-llama-archs --mtp-catchup-overlap` in the fork's `tests/test-llama-archs.cpp` (branch
`pascal-p106-v4-b12246-c703`).

## Provenance and licence

Every number here comes from random models that `test-llama-archs` builds from its seeds; no model file, prompt or
recorded text was read. R42's figures are that folder's. The fork is MIT-licensed, as llama.cpp; these tables and the
script are published with this repository.
