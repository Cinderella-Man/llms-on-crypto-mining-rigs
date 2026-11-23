# The KV head rules with two conversations in one cache (C966, Q1696, 2026-11-23)

What this is: point 13 of the GPU checks of the night's branches (registry [Q1696](../../../quants/experiments.md),
under Q1687), on a build machine's three RTX 3090s through the rig build's compute 6.1 PTX (the same binaries a P106
rig runs). Correctness only: nothing here is timed.

[C881](../2026-11-21_c881_kv_seq_rm/README.md) (`pascal-p106-v5-kv-seq-rm` `003bf4023`) makes `llama_kv_cache::seq_rm`,
the removal of one sequence's positions from the KV cache, call upstream llama.cpp's walk over the sequence's
(position, cell) index (#32380). Before the walk it reads, from the same index, the cell where the fork's
`LLAMA_KV_HEAD_TAIL` rule puts the next token after a tail removal: the cell of the first removed position. Upstream's
rule (`LLAMA_KV_HEAD_TAIL=0`) instead moves the head only down, to the lowest freed cell. [C952](../2026-11-21_c952_kv_seq_rm_gpu/README.md)
checked the branch on one slot, where the cells stay in position order and both rules pick the same cell, so it could
not tell them apart. Here two conversations share one cache, and a request is shaped so that one slot's removals land
above a gap the other slot left: there the two rules pick different cells.

## Result

**In plain words:** it passes, and the fork's rule is the one to keep. In every setting the branch gives the base's
replies token for token, reads the same prompt tokens, restores the same checkpoints, drafts the same tokens, and puts
every token into the same cell, in all 19,254 placements a load logged (19,092 with rule 0). With the fork's rule a repeat of the same
two-slot session puts its tokens in the same cells and gives the same replies; with upstream's the cells drift from
round to round and most replies change.

| Pass criterion (Q1696) | Result |
|---|---|
| the same tokens on both builds in each setting | **yes**: all 32 replies a load (16 requests × 2 rounds) the same token ids on base and branch, in all four settings ([verdicts.tsv](verdicts.tsv), [compare.tsv](compare.tsv)); the cancelled replies' chunks too; the same drafted and accepted counts request by request |
| the same restored positions | **yes**: 14 restores a load at the same positions on both builds; the whole list of 130 checkpoint events a load (created, created inside the call, restored, erased) the same, position for position ([events.tsv](events.tsv)); every request the same prompt tokens read and cached tokens used |
| the same mask counts | **yes**: every card-built attention mask identical to the host's, the same counts on both builds: 9,248 masks with the default rule and 9,170 with rule 0, of them 3,789 and 3,755 after removals with one to three changed cells listed, no mismatch ([mask.tsv](mask.tsv)) |
| (added) the same cells | **yes**: the cache's debug line before each placement of a micro-batch (head, used cells, the highest used cell, each slot's lowest and highest position) the same on both builds, request by request, in every setting ([compare.tsv](compare.tsv), column `placements`) |
| whether `=0` changes a reply (recorded, not judged) | **it does**: the rules first part at request 12 of round 1 (head on cell 10,374 with the fork's rule, cell 1,124 with upstream's; [rules.tsv](rules.tsv), [heads.tsv](heads.tsv)); from there 18 of the 32 replies differ from the default's ([settings.tsv](settings.tsv)), the same 18 on both builds |

**Where the rules part.** After slot 0 drops its 6,000-token conversation for a 1,000-token prompt (request 11,
`s0-short`), the cells it freed sit below slot 1's, and the head stays at the end of slot 0's new prompt, cell 1,124.
Request 12 (`s1-back`) removes slot 1's tail from position 4,291 (a checkpoint restore), all of it above the head.
The fork's rule puts the head on that position's cell, 10,374, so slot 1 reads its tail back into the same cells.
Upstream's rule leaves the head at 1,124, because no freed cell lies below it, so slot 1's tail goes into slot 0's
gap. Both builds make the same choice in each setting.

**The case where the two builds' code differs.** The branch finds the fork's cell by walking the sequence's index in
position order (the first entry whose cell holds no other sequence). The base loops over every cell and keeps the
freed cell of the lowest position. The two can only disagree when a slot's cells are out of position order. That
happens here with the default rule: request 15 (`s1-bturn`) appends slot 1's tokens at cell 1,198, in slot 0's gap,
below its earlier cells. Then request 16 (`s1-bedit`) removes slot 1's tail across both parts, where the first removed
position's cell (10,374) is not the lowest freed cell (1,198). Both builds put the head on 10,374, in both rounds.

**Rounds within one load** ([rounds.tsv](rounds.tsv)):

| Setting | Round-2 replies the same as round 1's | Round-2 placements the same as round 1's |
|---|---|---|
| default (`def`, `mask`) | 16 of 16 | 11 of 16: round 2 starts with slot 1's cells in the cache (round 1 started empty), and four requests draft a different number of tokens in round 2 (the draft carries state from request to request, registry Q971), so their placements differ in number |
| `LLAMA_KV_HEAD_TAIL=0` (`tail0`, `mask0`) | 3 of 16 | 0 of 16 |

Attention sums the keys in cell order, so a token in another cell can change the last bits of a score and then a
greedy token at a near tie. The rule-0 replies stay coherent (the same opening sentences, 128 tokens each). The first
request whose prompt is the same in both settings and whose reply differs is `s1-back`, at reply token 29. Most later
requests carry an earlier reply in their prompt, so their prompts differ too once that reply differs.

## The setup

**Builds**, each a rig tarball (CUDA 12.9, `sm_61` only, compute 6.1 PTX), unpacked on the build machine:

| Build | Commit | Tarball md5 | `llama-server` md5 |
|---|---|---|---|
| base | `pascal-p106-v5` `3dfad876f` | `0b5b983e22d7296ff44fe363d55f61bb` | `5b330eccef892b7dd9f4faf1027319c3` |
| new | `pascal-p106-v5-kv-seq-rm` `003bf4023` (C952's build) | `8efed57f4d241689efb6f28db577f14e` | `f2589439340f243610085c822db5da38` |

**Model:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`) with its v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)), changed in four ways:
- the layer split for three cards (`-ts 1,1,1`);
- `-np 1` replaced by `-np 2 -kvu`: two slots and one unified KV cache of 49,152 cells, so both conversations share
  one stream of cells;
- both slots drafting three tokens a step with the MTP layer in the file (`--spec-type draft-mtp --spec-draft-n-max 3`),
  so every rejected draft is a removal too;
- `-lv 5` with `LLAMA_KV_CACHE_DEBUG=1`: before every placement of a micro-batch the cache prints the head, the used
  cells, the highest used cell + 1 and each slot's lowest and highest position (upstream's debug line, the same code in
  both builds). The tables read the head from it.

The fork's Pascal matrix-vector kernels were forced on with a P106's launch shapes (`GGML_CUDA_MMVQ_PASCAL=2
GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`). A newer card runs them only so; on a P106 they run by default. The head rule
is host code and does not depend on them.

**Settings:**
- `def`: every switch as the preset sets it (`LLAMA_KV_HEAD_TAIL` at its default, on);
- `tail0`: `LLAMA_KV_HEAD_TAIL=0`;
- `mask`: `LLAMA_GPU_KQ_MASK=1 LLAMA_GPU_KQ_MASK_CHECK=1`. Each card builds the attention mask from its copy of the
  cells, kept current from the changed-cell list (which the branch fills before upstream's walk), and every mask is
  compared with the host's (a difference stops the server);
- `mask0`: `mask` with `LLAMA_KV_HEAD_TAIL=0`.

One load per setting and build: 8 loads.

**Requests** ([run/session2.py](run/session2.py)), each pinned to its slot (`id_slot`), one at a time. Greedy, 128
tokens, the token ids returned, the whole set twice a load. They are C952's six requests sent to the two slots in turn,
plus a short prompt on slot 0 and four requests that use the gap it leaves. A cancel is followed at once by the same
slot's next request, which removes the cancelled tokens before the other slot writes. That way no cell of the other
slot depends on how far the server ran past the hang-up.

| # | Request | What it does | Prompt tokens read / cached tokens used (round 1, `def`) |
|---|---|---|---|
| 1 | s0-fresh | slot 0: the first 15,853 characters of [docs/switches.md](../../switches.md) as of this commit and a question (5,956 tokens), the slot's cache cleared | 5,956 / 0 |
| 2 | s1-fresh | slot 1: 13,925 characters of the same file from line 201 and the question (4,807 tokens), cache cleared | 4,807 / 0 |
| 3, 4 | s0-turn, s1-turn | the prompt, its reply and a follow-up question: the cache reused as it is | 17 / 6,083; 17 / 4,934 |
| 5, 6 | s0-edit, s1-edit | a changed history inside fresh's reply: a checkpoint restored | 576 / 5,440; 578 / 4,291 |
| 7, 8 | s0-cancel, s0-after | edit's prompt, its reply and a question, streamed, end-of-text ignored, 400 tokens, the client hangs up after 24 chunks; then a withdrawal note and a question | (24 chunks compared); 26 / 6,160 |
| 9, 10 | s1-cancel, s1-after | the same on slot 1 | (24 chunks compared); 26 / 5,013 |
| 11 | s0-short | slot 0: the file's first 3,183 characters and the question (997 tokens), cache cleared: slot 0's cells are freed and only a few refilled, so a gap opens below slot 1's cells | 997 / 0 |
| 12 | s1-back | slot 1's document with another question: slot 1's tail removed, all of it above the head | 510 / 4,291 |
| 13 | s0-sturn | short's prompt, its reply and a question: slot 0 appends | 17 / 1,124 |
| 14 | s0-sedit | a changed history inside short's reply: slot 0 removes its tail | 590 / 481 |
| 15 | s1-bturn | back's prompt, its reply and a question: slot 1 appends where the head is | 16 / 4,928 |
| 16 | s1-bedit | a changed history inside back's reply: slot 1's tail removed across both parts | 578 / 4,291 |

## Files

| File | What it holds |
|---|---|
| [verdicts.tsv](verdicts.tsv) | per setting: requests the same on both builds (tokens, prompt tokens, cached tokens, drafts), the event lists, restores, full re-reads, asserts and mask mismatches, the mask counts, requests with the same placements, the verdict |
| [compare.tsv](compare.tsv) | per setting, round and request: base against branch, with the number of placements each logged |
| [replies.tsv](replies.tsv) | every reply: prompt tokens read, cached tokens used, generated tokens, the md5 of the token ids (12 hex digits), drafted and accepted, the stop reason, placements logged |
| [events.tsv](events.tsv) | per arm, the server's checkpoint events in order with their slot, positions only |
| [mask.tsv](mask.tsv) | the device-mask check's counts per arm: masks identical to the host's, from the resident copy and after a full upload, steps with a non-empty changed-cell list, cells listed, the largest list, mismatches |
| [heads.tsv](heads.tsv) | per build and request, the placements with `LLAMA_KV_HEAD_TAIL=0` against the default (`tail0` against `def`, `mask0` against `mask`): how many have another head or another used-cell count, the values at the first that differs |
| [rules.tsv](rules.tsv) | the same per build and pair: how many requests differ and where the rules first part |
| [settings.tsv](settings.tsv) | each setting's replies against another's, per build, with the reply token where each one first differs |
| [rounds.tsv](rounds.tsv) | per arm, round 2's replies and placements against round 1's |
| [run/run.sh](run/run.sh) | the driver: one container and llama-server per arm, the requests, nothing timed |
| [run/session2.py](run/session2.py) | the requests |
| [run/c966_tables.py](run/c966_tables.py) | the tables from the driver's output; the server's lines carry host times, which it cuts off |

## Commands

On the build machine, with the two rig tarballs unpacked, a copy of this repository and the model on a local disk:

```bash
WORK=/path/to/work NEW=/path/to/llama.cpp-20261121-003bf4023-pascal-p106-v5-kv-seq-rm/bin \
REF=/path/to/llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off/bin REPO=/path/to/this/repo \
ORNITH=/path/to/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf MOUNTS='/path/to /models' \
DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=/path/to/cuda_preflight.sh \
setsid nohup bash docs/data/2026-11-23_c966_kv_head_tail_gpu/run/run.sh > "$WORK/driver.log" 2>&1 < /dev/null &
# when $WORK/DONE exists, with the outputs copied to a machine with Python 3:
python3 -I docs/data/2026-11-23_c966_kv_head_tail_gpu/run/c966_tables.py "$WORK/out" docs/data/2026-11-23_c966_kv_head_tail_gpu
```

A first load with `RUN=smoke SETTINGS=def BUILDS=base ROUNDS=1 SESSION_ARGS='6000 5000 200 1000 24 2'` (two requests)
checked that the debug lines print and how large the log grows (0.8 MB for two requests; 6.3 to 7.8 MB a load in the
run). Each of the 8 loads took about three minutes.

## Provenance and licence

The model file is Ornith-1.5-122B Pascal N2P GGUF (`valid1`), this project's quantization of Ornith-1.5-122B. The
prompts are text of this repository. The tables hold counts, positions, cells and md5 hashes of token ids taken from
the server's replies and log lines: no model weights and no generated text.
