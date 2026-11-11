# C217's in-call checkpoints with C703's catch-up overlap (C735, Q1302, 2026-11-11)

What this is: a fix on the v4 line for two llama-server switches that meet in the MTP draft's pass over a prompt call.
C217's `--ctx-checkpoints-incall` ([2026-11-11_c217_incall_checkpoints](../2026-11-11_c217_incall_checkpoints/README.md))
takes a prompt's checkpoints inside the decode call instead of cutting the call there. For that, the draft's pass over
the call is cut at the same rows and fills the draft's part of each checkpoint into a store. C703's
`LLAMA_MTP_CATCHUP_OVERLAP=1` ([2026-11-11_c703_catchup_overlap](../2026-11-11_c703_catchup_overlap/README.md)) keeps
that pass back until the target has its next call queued.

On `pascal-p106-v4-b12246-c748` (`5b329df16`, [2026-11-11_c748_kvonly_kpool](../2026-11-11_c748_kvonly_kpool/README.md))
the two did not work together. The server set the draft's cuts and fills on the draft context just before the driver's
call, cleared the cuts right after it, and read the draft's part of each checkpoint at once. With both switches on, the
pass had not run yet at that point, so the part was missing. The server then erased the checkpoint and stopped taking
checkpoints inside calls for the rest of its life. In the CPU check below that cost the next request its cache: 3,150
prompt tokens read again instead of 282. The replies were still exact.

The fix is on the fork branch `pascal-p106-v4-b12246-c735` (`9e4bbe897`, two commits on `-c748`). A pass that waits
keeps its own cuts and fills, and the checkpoints' draft parts are read after it. With either switch alone, or both
off, the branch computes what `-c748` computes. This page is the change, the CPU checks, and the rig run still to do
([Q1302](../../../quants/experiments.md)).

## Result

**With both switches on, the draft's part of every checkpoint taken inside a call, the driver's state at it, the
draft's state after the prompt, its drafts and its end state are the same bytes as with the prompt in separate calls.
llama-server gives the same replies, probabilities, cache use, checkpoints and drafts with both switches as with either
alone and with neither. With every switch off the branch gives `-c748`'s logits.** The checks, all on a CPU
([summary.tsv](summary.tsv), [incall.tsv](incall.tsv), [overlap.tsv](overlap.tsv), [server_summary.tsv](server_summary.tsv),
[server_verdict.txt](server_verdict.txt)):

- **`test-llama-archs --mtp-incall-overlap` (new), 3 switch sets × 3 seeds × 5 configurations: 45 of 45 pass** with
  the branch's CPU build, and 45 of 45 with the rig build's binary on the CPU. Each run drives the target and the
  draft-mtp driver in the server's order six ways (next section). All five arms that should match the reference match
  it byte for byte. The switch sets are none, C100's k-pool tables on the cards with C720's stage and C100's check
  (`LLAMA_QSA_KPOOL_DEV=1`), and the same with the draft's tables only (`=2`).
- **The test sees both faults.** The control arm takes the checkpoints in `-c748`'s server order, and with the switch
  on it differs from the reference in all 90 configurations. A build of the branch with one line reverted (the driver
  reads the rows from the buffer's first half, as `-c748` does) fails 5 of 5 configurations, on the driver's state at
  the checkpoints and on nothing else ([summary.tsv](summary.tsv), `nohalf`).
- **llama-server with the MTP draft, the switches alone and together: PASS** (`server_check.sh`, a random-weight
  qwen4exp with one MTP block, five requests with three checkpoint restores). Each arm gives its reference's tokens,
  top-3 probabilities, `prompt_n`, `cache_n`, restored checkpoint and draft counts on every request. It takes its
  checkpoints at the same token counts and writes the same 70 verify steps in `LLAMA_SPEC_DUMP`: drafts, the drafter's
  candidates and probabilities, the target's top tokens. The arms with in-call checkpoints took 8 to 16 of them inside
  calls, and none was dropped. With both switches, 6 to 9 catch-ups a run waited for the next call.
- **The control on the server:** `-c748`'s llama-server with both switches erased its first two in-call checkpoints
  (`the draft's part of the context checkpoint at n_tokens = 2868 was not taken inside the call - erased`). It then
  read Q2's 3,150 and Q4's 3,000 prompt tokens again (282 and 132 with the checkpoints).
- **C703's own check** (`--mtp-catchup-overlap`): 45 of 45. **C620's** (`--mtp-kv-only`): 30 of 30. **C217's**
  (`test-llama-context` on random qwen4exp, qwen35 and mamba models): 72 case runs, 0 FAIL, `store_at` and
  `store_at_p` OK in 6.
- **With every switch off, the branch gives `-c748`'s logits** (`test-llama-archs -b CPU -s 1`, `5b329df16`'s CPU
  build writing): 146 of 146. **ctest: 70 of 70** (static CPU build; the tests that need a download, the network or
  torch are left out, as in C565).
- **The rig tarball builds** (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the tree's
  `bin/`): `llama.cpp-20261111-9e4bbe897-pascal-p106-v4-b12246-c735.tar.gz`, 656,368,001 B, md5
  `accb6c8e12a7922b2a44b196dc9f5b91`.

The server check, request by request: Q2, Q4 and Q5 restore a checkpoint taken inside a call in every in-call arm
([server_summary.tsv](server_summary.tsv)):

| Arms | Switches | Q1 prompt tokens | Q2 prompt / cached | Q4 prompt / cached | Checkpoints inside calls (Q1–Q5) | Catch-ups that waited |
|---|---|---:|---:|---:|---:|---:|
| base, off, ovl | none; overlap only | 3,000 | 282 / 2,868 | 132 / 2,868 | 0 | 0; 15 (ovl) |
| incall, both | in-call; both | 3,000 | 282 / 2,868 | 132 / 2,868 | 9 | 0; 6 |
| incall2, both2 | the same with 2 of 8 checkpoints on the cards | 3,000 | 282 / 2,868 | 132 / 2,868 | 8 | 0; 7 |
| incall4, both4 | the release preset's 4 of 4 | 3,000 | 282 / 2,868 | 132 / 2,868 | 9 | 0; 6 |
| sincall, sboth | a checkpoint every 200 tokens | 3,000 | 550 / 2,600 | 200 / 2,800 | 16 | 0; 9 |
| bboth (the control: `-c748`, both) | both | 3,000 | 3,150 / 0 | 3,000 / 0 | 0 (2 erased) | — |

The arms of a row give the same numbers as their reference (base, base4, sbase), which take every checkpoint between
calls.

## What was wrong on `-c748`

With both switches on, a prompt call with checkpoints inside it went like this:

1. Before the target's call the server set the draft's fills on the draft context, and before the driver's call its
   cuts. The driver kept the pass for later, and the server cleared the cuts. **The pass then ran without them** and
   cut its micro-batches only at the fills.
2. Before the next call the server set that call's fills on the draft context too. **The waiting pass took them**, at
   rows of the wrong call, and the next call's own pass found none.
3. The server read the draft's part of each checkpoint right after the driver's call, **before the pass had run**.
   It found no fill, erased the checkpoint and switched in-call checkpoints off.
4. The driver's state at a checkpoint is a row of the target's hidden-row buffer. **The driver read it from the
   buffer's first half.** With the overlap on, the target writes each call's rows into the other half from the last
   one's, so the read is right only for every other call.

## The change

- **`common/speculative.{h,cpp}`:** `common_speculative_set_cuts(spec, cuts, fills)` gives the driver the cuts and the
  device-store fills of the next batch's pass, in rows of that batch (`common_speculative_store_at` is
  `llama_state_seq_store_at`'s arguments). The draft-mtp driver keeps them with a pass that waits. Right before the
  pass's decode it sets the cuts and fills on the draft context, and it clears the cuts right after. It also keeps the
  first row of the buffer half its last pass read, and `get_state_at` reads from that half. With one half that row is
  0, as before.
- **`tools/server/server-context.cpp`:** the in-call checkpoints' draft fills and cuts go through
  `common_speculative_set_cuts` instead of onto the draft context. After the call the server reads each checkpoint's
  target part at once, as before, because the target's next call replaces its fills. It also keeps its place in the
  list, its device store and the move of older stores to the host where they were. The draft's part and the driver's
  state at the checkpoint (`ckpt_incall_finish_dft`) go through `common_speculative_after_catchup`: right after the
  pass, or at once when no pass waits. The C703 overlap already takes a cut call's checkpoint this way. A store moved
  to the host is copied into a new checkpoint. When its draft part still waits, the move runs the waiting pass first.
- **With either switch alone, or both off,** the draft's pass takes the same cuts and fills at the same point, the
  next decode on the draft context. The server reads the same bytes in the same order. One side effect: a target call
  that fails and is retried no longer leaves its fills on the draft context.
- **The test** (`tests/test-llama-archs.cpp`, `--mtp-incall-overlap`) takes a 100-token prompt with checkpoints after
  20, 50 and 84 tokens on a random qwen4exp with one MTP block, in the server's order. The target uses micro-batches of
  8 tokens and the draft passes of 18. It runs six ways:

  | Arm | Calls | `LLAMA_MTP_CATCHUP_OVERLAP` | Must |
  |---|---|---|---|
  | reference | separate: 20, 30, 14, 20, 12, 4 | off | — |
  | cut on | separate | on | = reference |
  | joined off | joined: 64 cut at 20 and 50, 32 cut at 20, 4 | off | = reference |
  | joined on | joined | on | = reference |
  | old off | joined, the draft's cuts and fills in `-c748`'s server order | off | = reference |
  | old on (the control) | joined, `-c748`'s order | on | differ |

  For each arm it compares each checkpoint's draft part (`update_dft`'s bytes, or the fill's in the same format), the
  driver's state at each checkpoint, the draft's sequence state and the driver's after the prompt, then three draft
  steps' drafts and probabilities, and both states at the end. It also counts the prompt calls whose pass waited: 5 of
  6 separate calls and 2 of 3 joined ones with the switch on, none with it off.

## What the rig run must show (R93, Q1327)

Nothing above ran on a card. On the rig the waiting pass's fills go to the draft card's streams while the target's
next call runs. The draft parts and the driver's state must still be right there, and the overlap must still save
time when the calls are joined. With C217 a 4,097-token prompt runs in fewer and larger calls, so fewer passes can wait
(6 against 15 in the server check). R85's replay ([C217's rig row](../2026-11-11_c217_incall_checkpoints/README.md#the-rig-row-q1306),
R43's `r43_run.sh`) on this tarball, two detached loads one after the other, nothing else on the rig:

```bash
# i: C217 alone (R85's load i on this build)
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c735i LOADS=A \
  LLAMA_SPEC_DUMP=$WORK/c735i.dump.jsonl ARGS_ADD='--ctx-checkpoints-vram 4 --ctx-checkpoints-incall' \
  setsid nohup bash $WORK/r43_run.sh > $WORK/c735i.out 2>&1 < /dev/null &"
# io: both switches
ssh $RIG "cd $WORK && BUILDS=$BUILDS BUILD=$BUILD MODEL=$MODEL DRAFT=$DRAFT OUT=$WORK/c735io LOADS=A \
  LLAMA_MTP_CATCHUP_OVERLAP=1 LLAMA_SPEC_DUMP=$WORK/c735io.dump.jsonl \
  ARGS_ADD='--ctx-checkpoints-vram 4 --ctx-checkpoints-incall' \
  setsid nohup bash $WORK/r43_run.sh > $WORK/c735io.out 2>&1 < /dev/null &"
```

`BUILD` is this tarball's folder, `MODEL` the first shard of Swift Qwen3.8-Flash-Next Pascal Q2 GGUF and `DRAFT` its
MTP draft, as in R43. Read both loads with R43's `run/lane_tables.py`. The run is right when, with the taper off:

- `io` gives `i`'s greedy replies on all 62 turns;
- its server log shows the same checkpoints (token counts, in order), with the same ones marked `inside the call`, no
  `was not taken inside the call` line, and the switch's line with its count of passes that waited;
- its dump without `run`, `step` and `task` (`jq -c 'select(.draft != null) | del(.run, .step, .task)'`) is `i`'s.

The verdict is `io`'s prompt seconds a pass against `i`'s two passes: gain, negligible or loss, by the spread. Est. ~40
min a load, ~1 h 30 min with the relay.

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict (check, verdict, detail) |
| [summary-try1.tsv](summary-try1.tsv) | the server check's first run: every arm matched but its dump comparison kept the task id, a server-internal counter that is lower when calls are joined (22 against 24 at Q2); `server_check.sh` now leaves it out, as it leaves out the run's start time and the step counter |
| [incall.tsv](incall.tsv) | `--mtp-incall-overlap`, one row per build (`tip`: the CPU build; `rig`: the rig binary on the CPU), switch set, seed and configuration |
| [overlap.tsv](overlap.tsv) | C703's `--mtp-catchup-overlap` on this branch, the same way |
| [server_summary.tsv](server_summary.tsv) | the server check: per arm and request, the prompt and cache counts, the checkpoints made, on the devices and inside calls, the draft counts, the times and the reply's tokens |
| [server_verdict.txt](server_verdict.txt) | the server check's verdict lines |
| [run/checks.sh](run/checks.sh) | the whole chain: builds, the tests, the server check, the logits comparison, ctest, the rig tarball |
| [run/server_check.sh](run/server_check.sh) | the server check (made from C217's) |

## Commands

On a build machine with Docker, every step in a container at low CPU priority (`--cpu-shares=256`, `nice -n 19`), one
at a time under a lock; `SRC` is a llama.cpp clone with the fork as remote `fork`, `BASE_CPU` the `bin/` of
`5b329df16`'s static native CPU build (its `test-llama-archs`) and `BASE_SRV` its build's `bin/` (its `llama-server`):

```bash
WORK=<work dir> REPO=<this repository> SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c735 \
  BASE_CPU=<base bin/> BASE_SRV=<base build bin/> LOCK=<lock file> JOBS=8 \
  bash docs/data/2026-11-11_c735_c217_c703/run/checks.sh
```

The run took the default steps in three parts: `cpu incall`, then `nohalf overlap kvonly context server off ctest build
incallrig`, then `server` again after the task id was left out of the dump comparison.

## Provenance and licence

The fork is [Cinderella-Man/miner-llama](https://github.com/Cinderella-Man/miner-llama) (MIT, as llama.cpp), branch
`pascal-p106-v4-b12246-c735`: `566cfa178` the change, `9e4bbe897` the test, on `-c748` `5b329df16`. The models are
random-weight files that `test-llama-archs` makes; the prompts are token lists from a fixed generator. No downloaded
data. The tables and scripts here are this repository's licence.
