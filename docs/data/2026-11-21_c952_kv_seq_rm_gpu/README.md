# The KV removal on upstream's walk, on three newer cards (C952, Q1687, 2026-11-21)

What this is: the GPU half of the check of [C881](../2026-11-21_c881_kv_seq_rm/README.md) (registry
[Q1590](../../../quants/experiments.md)), point 6 of the GPU checks of the night's branches (registry
[Q1687](../../../quants/experiments.md)). C881's branch (`pascal-p106-v5-kv-seq-rm` `003bf4023`, two commits on
`pascal-p106-v5` `6644c2bc6`) makes `llama_kv_cache::seq_rm`, the removal of one sequence's positions from the KV
cache, call upstream llama.cpp's walk over the sequence's (position, cell) index (#32380). Before the walk it reads,
from the same index entries, the cells it will change (the changed-cell list that keeps each card's copy of the cells
current for an attention mask built on the card) and the cell where `LLAMA_KV_HEAD_TAIL` puts the next token after a
tail removal. Its processor check passed; here two release files run on a build machine's three RTX 3090s, through the
rig build's compute 6.1 PTX (the same binaries a P106 rig runs), with requests that make the server remove cells.
Correctness only: nothing here is timed.

## Result

**In plain words:** it passes. On both models, in every setting, the branch gives the base's replies token for token,
reads the same number of prompt tokens on every request, restores the same checkpoints at the same positions, and
drafts and accepts the same number of tokens. Where the cards build the attention mask from their own copy of the
cells, all 7,422 masks the check compared are identical to the host's, on both builds, with the same counts.

Two things the check showed about itself:

- **`LLAMA_KV_HEAD_TAIL=0` changes nothing on one slot.** Its tokens are the default's on both builds and both models.
  With one slot the cache's cells stay in position order, so the fork's rule (the next token goes into the cell of the
  first removed position) and upstream's (the lowest freed cell) pick the same cell. The criterion holds in both
  settings, but on one slot it cannot tell the two rules apart. They differ only when removals leave holes between
  another sequence's cells, which C881's processor check covered (upstream's `test-kv-cache` and two slots in one
  stream).
- **The changed-cell list only fills up with a draft, and only on Ornith.** Flash-Next's attention layers are all
  sparse-attention (QSA) layers with masks of their own, so its cache's mask, and the list, are never used (the
  device-mask arm logged no check line; it ran as the default). On Ornith without a draft every removal here was
  larger than the list's 32 entries or followed a checkpoint restore, after which the cards' copies are uploaded whole,
  so every one of its 4,110 checks had an empty list. A third arm was added for this: Ornith with the MTP layer in its
  own file drafting three tokens a step. Its rejected drafts remove one to three cells at a time: 1,341 of its 3,312
  checks had a non-empty list (3,109 cells in all, at most 3), every mask identical to the host's, the same counts on
  both builds.

| Pass criterion (Q1687 point 6) | Result |
|---|---|
| the same tokens on base and branch, `LLAMA_KV_HEAD_TAIL` at its default and `=0` | **yes**: all 12 replies an arm (6 requests × 2 rounds) the same token ids on both builds, in all 8 model-and-setting pairs ([verdicts.tsv](verdicts.tsv), [compare.tsv](compare.tsv)); the cancelled reply's first 24 chunks too |
| the same restored positions | **yes**: every arm restores 6 checkpoints, at the same positions on both builds; the server's whole list of checkpoint events (40 an arm: created, restored, erased) is the same, position for position ([events.tsv](events.tsv)); every request reads the same number of prompt tokens and uses the same cached tokens |
| the draft's rollbacks | **yes**: Flash-Next's and Ornith's drafted and accepted counts the same on both builds, request by request |
| (added) the cards' attention masks after removals | **yes**: Ornith 4,110 and Ornith with its draft 3,312 checks, all identical to the host's mask, the same counts on both builds ([mask.tsv](mask.tsv)) |

## The setup

**Builds**, each a rig tarball (CUDA 12.9, `sm_61` only, compute 6.1 PTX), unpacked on the build machine:

| Build | Commit | Tarball md5 | `llama-server` md5 |
|---|---|---|---|
| base | `pascal-p106-v5` `3dfad876f` (`6644c2bc6` and one commit that reads `LLAMA_PIPE_GRAMMAR=0` as off; neither preset runs the pipe) | `0b5b983e22d7296ff44fe363d55f61bb` | `5b330eccef892b7dd9f4faf1027319c3` |
| new | `pascal-p106-v5-kv-seq-rm` `003bf4023` | `8efed57f4d241689efb6f28db577f14e` | `f2589439340f243610085c822db5da38` |

The branch's tarball was built for this check with
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) `fork/pascal-p106-v5-kv-seq-rm`,
`TESTS=1`, `JOBS=8`, the web page pinned to upstream's `b12483`.

**Models**, each with its v5 preset, the layer split changed to the three cards (`-ts 1,1,1`), `-lv 4` (the server's
checkpoint lines are verbosity 4), one slot:

| Model in the tables | File | Preset |
|---|---|---|
| `ornith` | Ornith-1.5-122B Pascal N2P GGUF (`valid1`) | [release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env), no draft |
| `fn` | Swift GSQ-RCO Q2_0 (UkisAI's Swift-1.5, three shards, the file the preset names) | [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env), `--spec-type draft-mtp`, three drafted tokens a step; the draft is Swift-1.5's own MTP layer ([2026-11-10_swift_own_mtp_head](../2026-11-10_swift_own_mtp_head/README.md)), a head of the preset's kind (the preset's Gyro draft is kept on the rigs only), as in C947 and C951 |
| `ornmtp` | Ornith-1.5-122B Pascal N2P GGUF | Ornith's preset and `--spec-type draft-mtp --spec-draft-n-max 3` (the MTP layer in the file) |

**Settings:** `def`, every switch as the preset sets it (`LLAMA_KV_HEAD_TAIL` at its default, on; Flash-Next's preset
sets `LLAMA_KV_SEQ_RM_BOUND=1`, Ornith's does not); `tail0`, `LLAMA_KV_HEAD_TAIL=0`; `mask`,
`LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1`: each card builds the attention mask from its copy of the cells,
updated from the changed-cell list, and every mask is compared with the host's (a difference stops the server).
`ornmtp` ran `def` and `mask`. One load per model, setting and build: 16 loads.

**Requests** ([run/session.py](run/session.py)), on one slot, every request but the first reusing the slot's cache, so
the server keeps the shared prefix and removes the rest. These hybrid models' Gated DeltaNet layers cannot be cut
back, so the server restores the newest checkpoint below the first differing token, removes every KV cell after it and
reads the rest of the prompt again. Greedy, 128 tokens, the token ids returned, the whole set twice a load:

| Request | What it does | Prompt tokens read / cached tokens used (Ornith; Flash-Next) |
|---|---|---|
| fresh | the first 15,853 characters of [docs/switches.md](../../switches.md) as of this commit (5,956 tokens with the question), the slot's cache cleared | 5,956 / 0 |
| turn | fresh's prompt, its reply and a follow-up question: the cache reused as it is | 17 / 6,083 |
| edit | a changed history: fresh's prompt, the first third of its reply, a cut marker and another question; a checkpoint restored at 5,440 | 577 / 5,440; 574 / 5,440 |
| cancel | edit's prompt, its reply and a question, streamed, end-of-text ignored, 400 tokens; the client hangs up after 24 chunks | (only the 24 chunks read are compared) |
| after | cancel's prompt, a withdrawal note and a question: the cancelled reply removed; a checkpoint restored at 6,161 (6,158) | 26 / 6,161; 26 / 6,158 |
| back | the document with another question only: a checkpoint restored at 5,440 | 510 / 5,440 |

Every restore is the newest checkpoint below the first differing token, taken inside a prompt's call
(`--ctx-checkpoints-incall`, both presets): 5,440 (position 5,439) and 6,161 (6,158 on Flash-Next's tokens).

**Drafted / accepted tokens** (`def`; the same on both builds and in every setting):

| Model | Round | fresh | turn | edit | after | back |
|---|---|---|---|---|---|---|
| Flash-Next | 1 | 111 / 89 | 101 / 93 | 102 / 93 | 95 / 95 | (an end-of-text at once) |
| Flash-Next | 2 | 111 / 90 | 116 / 88 | 102 / 93 | 96 / 95 | (an end-of-text at once) |
| Ornith with its draft | 1 | 117 / 88 | 141 / 80 | 130 / 83 | 117 / 87 | 112 / 89 |
| Ornith with its draft | 2 | 120 / 87 | 138 / 81 | 130 / 83 | 117 / 87 | 112 / 89 |

**Rounds within one load.** Ornith without a draft repeats every reply and count in round 2. With a draft the counts
move between rounds (the draft carries state from request to request), and on Flash-Next one greedy reply moves with
them: turn's reply differs from its fourth token in round 2, in all six Flash-Next loads, on both builds alike.
Registry Q971 already explains it (a 4-token verify pass and 1-token decodes take different code paths, so a near tie
can flip where the verify passes fall); it is why drafting arms are compared with each other here, round by round.

## Files

| File | What it holds |
|---|---|
| [verdicts.tsv](verdicts.tsv) | per model and setting: requests the same on both builds (tokens, prompt tokens, cached tokens, drafts), the event lists, restores, full re-reads, mask mismatches, the mask counts, the verdict |
| [compare.tsv](compare.tsv) | per model, setting, round and request: base against branch |
| [replies.tsv](replies.tsv) | every reply: prompt tokens read, cached tokens used, generated tokens, the md5 of the token ids (12 hex digits), drafted and accepted, the stop reason |
| [events.tsv](events.tsv) | per arm, the server's checkpoint events in order, positions only (created, created inside the call, restored, erased) |
| [mask.tsv](mask.tsv) | the device-mask check's counts per arm: masks identical to the host's, from the resident copy and after a full upload, steps with a non-empty changed-cell list, cells listed, the largest list, mismatches |
| [settings.tsv](settings.tsv) | each setting's replies against `def`'s, per model and build |
| [run/run.sh](run/run.sh) | the driver: one container and llama-server per arm, the requests, nothing timed |
| [run/session.py](run/session.py) | the requests |
| [run/c952_tables.py](run/c952_tables.py) | the tables from the driver's output; the server's checkpoint lines carry host times, which it cuts off |

## Commands

On the build machine, from a copy of this repository at `$REPO` (the two presets, `docs/switches.md` and this
folder's `run/`), both builds unpacked, detached:

```bash
export WORK REPO NEW=$NEW/bin REF=$REF/bin ORNITH=$MODELS/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
    FN=$MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
    DRAFT=$MODELS/mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf \
    MOUNTS="$MODELS $BUILDS" DOCKER_ARGS="--cpu-shares=256"
setsid nohup bash run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
# once $WORK/DONE exists, the arm with Ornith's own draft:
MODELS_RUN=ornmtp SETTINGS="def mask" setsid nohup bash run/run.sh > $WORK/driver2.log 2>&1 < /dev/null &
python3 -I run/c952_tables.py $WORK/out <this folder>
```

Ran 2026-11-21: the first run 14:35–15:09 (12 loads, Ornith and Flash-Next, the driver before the `ornmtp` case was
added; the same code path for both), the second 15:12–15:20 (4 loads, `ornmtp`). Run file md5s: `run.sh`
`0594b7484e87a60e4d7f62de988c8476` (the second run; the first ran the same file without the `ornmtp` case),
`session.py` `2678d91f9b23ae59afadcb32d40ef1ad`, `release/ornith-v5/settings.env` `4b8dac36b6b9866090de3887c39b02ff`,
`release/flash-next-v5/settings.env` `b8e950f2de8c7906c156d582416731bf`, `docs/switches.md`
`2f5642a25b81738124f37fd76f91f985`; the MTP draft's md5 `e0aa166bbd5328f39026f2f42b458ff6`.

## Provenance and licence

The replies are generated text, summarized here by hashes and counts only. The prompts are this repository's own
`docs/switches.md`. Ornith-1.5-122B Pascal N2P GGUF and Swift-1.5's files are as their pages publish them
(read in place from the build machine's disks). Tables and scripts: this repository's licence.
