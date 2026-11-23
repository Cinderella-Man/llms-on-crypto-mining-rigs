# Ornith's checkpoint saves queued behind the work, on three newer cards (C955, 2026-11-23)

What this is: the GPU half of the check of [C907](../2026-11-21_c907_ckpt_stall/README.md) (registry
[Q1644](../../../quants/experiments.md)), point 9 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). Ornith's server saves a copy of the model's running state (a context
checkpoint) at a few points in a long prompt. When such a point falls between two of the prompt's calls, the save used to
wait until every card had finished all the work already queued: on nine P106 cards that stopped a 47,012-token prompt
for 8.20, 11.85 and 12.37 s ([R183](../2026-11-20_r183_ornith_recall/README.md)). C907's switch
`LLAMA_SERVER_CKPT_NOWAIT=1` (fork branch `pascal-p106-v5-ckpt-stall` `0aebc4a57`, one commit on `pascal-p106-v5`
`3dfad876f`, off by default) queues those copies on each card behind the work instead, and makes anything that reads
the checkpoint later (a restore, or a copy to host memory) wait for them first.

C907 checked the branch on a processor, which runs every copy at once, so the queued copies themselves had never run.
Here they do: on a build machine's three RTX 3090s, through the rig build's compute 6.1 PTX (the same binaries a P106
rig runs), with the real model. Correctness only: nothing here is timed.

## Result

**In plain words:** it passes. With the setting on, the server saved every checkpoint that falls between two of a
prompt's calls without waiting for the cards (12 such saves on the preset's line), and nothing after it changed: every
reply has the same tokens and the same top-3 log-probabilities, the checkpoints sit at the same positions, and the two
restores, each of which reloads one of those saves and reads on from there, give the same replies. The same holds on a
line that moves older checkpoints to host memory, where 12 of the saves made without a wait were moved and two were
restored from there. With the setting unset the branch is the base, field for field.

| Pass criterion (Q1687 point 9) | Result |
|---|---|
| the same replies and checkpoints, the setting off and on | **yes.** `head-on` against `head-off`, all 8 requests: the same reply text, the same top three log-probabilities at each of the 64 reply tokens, the same counters (prompt and cached tokens, checkpoints made, inside a call, on the cards), the same 45 checkpoints at the same positions, the same restores (8,234 and 41,016 tokens) and the same printed calls ([server.tsv](server.tsv)) |
| with the setting, the saves between calls queued, in the server log | **yes.** All 12 checkpoints taken between calls say "no wait" in the server log, and the server's own count (`ckpt_nowait_n`) is 1, 2, 3, 1 and 5 on the requests that take them, the number taken between calls; the in-call checkpoints are unchanged ([checkpoints.tsv](checkpoints.tsv)) |
| (added) the switch unset, the branch is the base | **yes.** `head-off` against `base-off`, all 8 requests: every field above the same; no checkpoint says "no wait" |
| (added) the same with older checkpoints moved to host memory | **yes.** `head-on-spill` against `head-off-spill`: every field above the same, and the same 29 moves to host memory at the same positions ([moves.tsv](moves.tsv)); all 18 checkpoints between calls filled without a wait, 12 of them later moved to the host; both restores read a host copy of one of them. All 40 replies of the five arms carry `base-off`'s tokens and log-probabilities ([replies.tsv](replies.tsv)) |

**Each arm** (one load, the 8 requests below):

| Arm | Checkpoints | Between calls | Of them filled without a wait | Moved to host memory | Restores: position, from | Replies with `base-off`'s tokens and log-probabilities |
|---|---:|---:|---:|---:|---|---:|
| `base-off` | 45 | 12 | 0 | 0 | 8,234 and 41,016, the cards | 8 of 8 |
| `head-off` | 45 | 12 | 0 | 0 | 8,234 and 41,016, the cards | 8 of 8 |
| `head-on` | 45 | 12 | **12** | 0 | 8,234 and 41,016, the cards; both filled without a wait | 8 of 8 |
| `head-off-spill` | 45 | 18 | 0 | 29 | 8,234 and 41,016, host memory | 8 of 8 |
| `head-on-spill` | 45 | 18 | **18** | 29 | 8,234 and 41,016, host memory; both filled without a wait | 8 of 8 |

**The checkpoints taken between calls** (`head-on`; tokens of the prompt at the checkpoint): `kinds-16000` 8,257;
`kinds-32000` 8,232 and 24,653; `kinds-47000` 24,632, 32,827 and 41,036; `rst-a` 8,234; `rst-c` 8,212, 16,405, 24,609,
32,804 and 41,016. The first six are the positions of R183's cut calls on nine P106 cards and of C907's processor run:
where a checkpoint falls is the host's decision, the same on any cards. With only two checkpoints on the cards the spill
line takes one more per prompt between calls, the middle one of the three near a prompt's end (8,073, 16,033, 31,983,
46,970, 16,048, 46,986), in both of its arms alike.

**Card memory** ([cards.tsv](cards.tsv), MiB on cards 0, 1, 2): every arm once up 15,425, 16,641 and 15,605; after
the requests 15,779, 16,997 and 15,909 on the preset's line and 15,727, 16,943 and 15,869 on the spill line, the same
within each pair; back to 15, 97 and 15 after each stop.

**Which kernels ran.** The fork's Pascal matrix-vector kernels turn themselves on only on compute 6.1 cards, so every
arm forced them on with a P106's launch shapes (`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2
GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`), as [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) did; the server's
switch line lists them as set in every arm, so the replies ran on the fork's kernels as a P106 runs them. The
checkpoints' copies do not depend on them.

## How it ran

**The line.** Ornith-1.5-122B Pascal N2P GGUF with its v5 preset
([release/ornith-v5](../../../release/ornith-v5/settings.env)): a 49,152-token context, one slot, prompts read in calls
of up to 4,096 tokens (`-b 4096 -ub 512`), four checkpoints of the recurrent state kept on the cards
(`--ctx-checkpoints 4 --ctx-checkpoints-vram 4`) and taken inside the calls where they fit
(`--ctx-checkpoints-incall`). Two changes: the layer split for the three cards (`-ts 1,1,1`) and the log at verbosity 4
(`-lv 4`), which prints a line for every checkpoint, with "no wait" where the switch filled it without waiting. No
draft, as R183 ran it.

| Arm | Build | Change |
|---|---|---|
| `base-off` | the base `3dfad876f` (C908's rig tarball) | none |
| `head-off` | the branch `0aebc4a57` (C907's rig tarball) | none |
| `head-on` | the branch | `LLAMA_SERVER_CKPT_NOWAIT=1` |
| `head-off-spill` | the branch | `--ctx-checkpoints 8 --ctx-checkpoints-vram 2`: the newest two checkpoints stay on the cards, older ones are copied to host memory |
| `head-on-spill` | the branch | the same, and `LLAMA_SERVER_CKPT_NOWAIT=1` |

The two spill arms are an addition to the row: on the preset's line every checkpoint stays on the cards, so the other
place that must wait for a queued copy, its move to host memory, is reached only with a longer list.

**The requests** ([run/client.py](run/client.py)), one load per arm, in this order, each greedy (temperature 0,
top-k 1, seed 1546) with 64 reply tokens and the top three log-probabilities of each:

| Request | Prompt | Prompt cache | What it reaches |
|---|---|---|---|
| `kinds-8000` | R183's 8,077-token prompt | off | every checkpoint inside a call |
| `kinds-16000` | R183's 16,075-token prompt | off | one checkpoint between calls (8,257 tokens) |
| `kinds-32000` | R183's 32,025-token prompt | off | two (8,232, 24,653) |
| `kinds-47000` | R183's 47,012-token prompt | off | three (24,632, 32,827, 41,036) |
| `rst-a` | the 16,075-token prompt behind a new system message | on | a whole read: one checkpoint between calls (8,234) |
| `rst-b` | `rst-a` with the user message at 62 % of the list changed | on | the server restores the checkpoint at 8,234 tokens and reads the 7,858 after it |
| `rst-c` | the 47,012-token prompt behind another new system message | on | a whole read: five checkpoints between calls (8,212, 16,405, 24,609, 32,804, 41,016) |
| `rst-d` | `rst-c` with the user message at 93 % of the list changed | on | the server restores the checkpoint at 41,016 tokens and reads the 6,014 after it |

So both restores read a checkpoint taken between two calls: with the switch on, one filled without a wait (on the
cards in `head-on`, from host memory in `head-on-spill`).

**What is compared** ([run/compare.py](run/compare.py)), each arm against its reference (`head-off` against
`base-off`, `head-on` against `head-off`, `head-on-spill` against `head-off-spill`), request by request: the reply
(reasoning and answer text), every reply token's top three log-probabilities, the server's counters (prompt tokens,
cached tokens, the checkpoint a restore used, checkpoints made, made inside a call, on the cards), the checkpoints in
order (kind, position), the restores, the moves to host memory, and the prompt's calls the server printed (one line at
each call's start once a prompt has run 3 s, so how many of a prompt's first calls show depends on the cards' speed:
two arms are compared from the later of their first printed calls on). With the switch on, every checkpoint taken
between calls must say "no wait" and be counted in `ckpt_nowait_n`; unset, none.

## Files

| File | What it holds |
|---|---|
| [server.tsv](server.tsv) | per pair and request: reply, log-probabilities, counters, calls, checkpoints, restores and moves the same or not; checkpoints between calls, those filled without a wait, the server's count of them; the verdict per pair |
| [replies.tsv](replies.tsv) | every request of every arm: finish, reply tokens, a SHA-1 of the reply's tokens, the same tokens and log-probabilities as `base-off` or not, prompt and cached tokens, the checkpoint a restore used and what kind it was, the reasoning's first 60 characters |
| [checkpoints.tsv](checkpoints.tsv) | every checkpoint of every arm in order: request, kind (between calls or inside a call), position, filled without a wait, store, size on the host and on the cards |
| [moves.tsv](moves.tsv) | per arm and request, the checkpoints moved from the cards to host memory |
| [calls.tsv](calls.tsv) | per arm and request, the prompt's call starts the server printed |
| [cards.tsv](cards.tsv) | every card's memory use (MiB) before, once up, after the requests and after the stop, per arm |
| [run/run.sh](run/run.sh) | the driver |
| [run/client.py](run/client.py) | the requests |
| [run/compare.py](run/compare.py) | the comparison and the tables |

## Commands

On the build machine, with the branch's rig tarball (`llama.cpp-20261121-0aebc4a57-pascal-p106-v5-ckpt-stall.tar.gz`,
md5 `f652183d16ddb5c14ead6a2ed3e06c32`, built by C907 with `TESTS=1`) and the base's
(`llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off.tar.gz`, md5 `0b5b983e22d7296ff44fe363d55f61bb`)
unpacked, this repository copied to `$REPO`, and the cards checked with a short CUDA allocation and copy on each first:

```bash
env WORK=$WORK NEW=$NEW_TREE/bin REF=$BASE_TREE/bin REPO=$REPO ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
    MOUNTS="<the folders holding the builds and the model>" DOCKER_ARGS="--cpu-shares=256" \
    setsid nohup bash $REPO/docs/data/2026-11-23_c955_ckpt_nowait_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
# it ends with compare.py's tables in $WORK/compare/ and writes $WORK/DONE; again on the output:
python3 -I run/compare.py $WORK/load $WORK/compare > $WORK/compare/server.tsv
```

Every server runs in a `nvidia/cuda:12.9.2-devel-ubuntu24.04` container on all three cards, with the PTX JIT's cache in
`$WORK/jitcache`; the requests and the comparison run on the host.

## Provenance and licence

The model file is Ornith-1.5-122B Pascal N2P GGUF (valid1), the release file of
[release/ornith-v5](../../../release/ornith-v5/settings.env). The prompts are R183's, this project's own
([2026-11-20_r183_ornith_recall](../2026-11-20_r183_ornith_recall/README.md)). The tables are derived from the runs'
server logs and replies and published under this repository's licence.
