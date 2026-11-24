# One slot's reply and the other slot's history (C987, Q1720, 2026-11-24)

What this is: point 17 of the GPU checks of the night's branches (registry [Q1720](../../../quants/experiments.md),
under Q1687), on a build machine's three RTX 3090s through the rig build's compute 6.1 PTX (the same binaries a P106
rig runs). Correctness only: nothing here is timed.

llama-server can serve several agents from one model, one slot each (`-np N`). By default each slot gets its own
stream of KV cells. With `-kvu` (a unified cache) all slots share one stream: a slot's tokens go into whatever cells
the other slots' histories left free, and attention reads every cell from the first up to the highest used one, with
the other slots' cells masked out. So the same prompt can be summed over the cells in another order, depending on
where it landed. In [C966](../2026-11-23_c966_kv_head_tail_gpu/README.md) the same prompt gave another reply from
token 29 when only the cache's head rule put its tail into another cell. This check asks the direct question: does one
agent's greedy reply change with what another agent did before it?

## Result

**In plain words:** yes, with `-kvu`, and no without it. One fixed prompt on slot 1 gave three different greedy
replies after three different histories on slot 0, each reply repeatable. With one stream of cells per slot it gave the
same reply as a server with one slot, after every history. **Several agents that need reproducible replies run without
`-kvu`.**

| Server | Slot 0's history before the fixed request | Slot 1's 5,062 cells | Fixed reply (md5 of its 256 token ids) | First reply token that differs from one slot's |
|---|---|---|---|---|
| `-np 1` (the preset) | none (the request on slot 0) | cells 0–5,061 | `950183f1880a` | — |
| `-np 2 -kvu` | none | cells 0–5,061 | `950183f1880a` | — |
| `-np 2 -kvu` | long: a 6,000-token conversation, 6,227 cells | cells 6,227–11,288 | `661f3c84af9b` | 9 |
| `-np 2 -kvu` | short: that conversation replaced by a 1,000-token one, 1,268 cells | cells 1,268–6,329 | `c4a5d269cdf0` | 9 |
| `-np 2` | none, long or short | cells 0–5,061 of its own stream | `950183f1880a` | — |

- **Repeatable in each arm:** every arm was loaded twice, and each load sent the fixed request three times (fix-1,
  then fix-2 and fix-p, which re-read the prompt from scratch). All six replies of an arm are the same token ids, in the
  same cells ([compare.tsv](compare.tsv), [cells.tsv](cells.tsv)). So the reply follows the cells, and the cells follow
  the other slot's history.
- **The log-probabilities say the same:** fix-p asked for each step's two most likely tokens. With `-kvu` and a history
  they differ from one slot's at every step, from the first reply token on (largest difference before the replies part:
  0.193 after the long history, 0.0815 after the short), and the replies part at token 9, where one slot's choice led
  the runner-up by 0.41. With `-kvu` and no history, and with separate streams after every history, all 256 steps'
  log-probabilities are bit-identical to one slot's ([probs.tsv](probs.tsv)).
- **The replies stay coherent:** all three start with the same nine tokens and go on reasoning about the question in
  other words. That fits sums taken in another order rather than other values summed: a small difference in a score,
  carried through the layers (Ornith picks 8 of 256 experts a token, and a near tie between two experts can flip),
  until a token near a tie changes.
- **Slot 0's own replies** are the same in every arm (it always starts at cell 0; [replies.tsv](replies.tsv)).
- **Why separate streams fix it:** each slot's cells start at cell 0 of its own stream whatever the other slot did, so
  its attention reads the same cells over the same span. The cache line confirms the modes: 49,152 cells, 1 stream
  (`-np 1`); 49,152 cells, 2 sequences, 1 stream (`-kvu`); 24,576 cells a stream, 2 streams (`-np 2`)
  ([loads.tsv](loads.tsv)). The cost is that each slot then holds at most its share of `-c`.

## The setup

**Build:** the base `pascal-p106-v5` `3dfad876f` (C908's rig tarball, CUDA 12.9, `sm_61` only, compute 6.1 PTX;
tarball md5 `0b5b983e22d7296ff44fe363d55f61bb`, `llama-server` md5 `5b330eccef892b7dd9f4faf1027319c3`).

**Model:** Ornith-1.5-122B Pascal N2P GGUF (`valid1`) with its v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)), changed in three ways:
- the layer split for three cards (`-ts 1,1,1`);
- `-np 1` kept (mode `one`), or replaced by `-np 2 -kvu` (mode `kvu`) or `-np 2` (mode `streams`);
- `-lv 5` with `LLAMA_KV_CACHE_DEBUG=1`: before every placement of a micro-batch the cache prints the head (where the
  search for free cells starts), the used cells, the highest used cell + 1 and each slot's lowest and highest position
  (upstream's debug line). The tables read where each request's cells went from it.

**Kernels:** the fork's Pascal matrix-vector kernels were forced on with a P106's launch shapes
(`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`), as in C966; the server's system line lists
`MMVQ_PASCAL = 1` and `MMVF_PASCAL = 1` in every load. Attention ran llama.cpp's flash-attention kernels as chosen for
compute 6.1 (the build holds only compute 6.1 code). On a P106 the prompt's attention runs the fork's Pascal tile kernel
(`GGML_CUDA_FATTN_TILE_PASCAL`, on by default on compute 6.1, off on these cards unless set to `2`), so a P106 computes
other replies than these. The cause is not in a kernel: every attention kernel reads the cache's view from cell 0 up to
the highest used cell, so where a slot's cells sit changes how its keys are grouped and summed on any card.

**Requests** ([run/slots.py](run/slots.py), derived from C966's [session2.py](../2026-11-23_c966_kv_head_tail_gpu/run/session2.py):
the same documents, questions and greedy settings), each pinned to its slot (`id_slot`), one at a time, greedy
(temperature 0, top-k 1), the token ids returned. One load answers one history and the fixed request:

| Request | Slot | What it does | Prompt tokens read / cached tokens used |
|---|---|---|---|
| s0-fresh (histories long and short) | 0 | the first 15,853 characters of [docs/switches.md](../../switches.md) as of this commit and a question (5,956 tokens), cache_prompt off, 128 tokens | 5,956 / 0 |
| s0-turn (long) | 0 | that prompt, its reply and a follow-up question: slot 0 then holds 6,227 cells | 17 / 6,083 |
| s0-short (short) | 0 | the file's first 3,183 characters and the question (997 tokens), cache_prompt off: slot 0's cells freed, the first few refilled | 997 / 0 |
| s0-sturn (short) | 0 | that prompt, its reply and a follow-up question: slot 0 then holds 1,268 cells | 17 / 1,124 |
| fix-1 | 1 (0 with `-np 1`) | the fixed request: 13,925 characters of the file from line 201 and the question (4,807 tokens), cache_prompt off, 256 tokens | 4,807 / 0 |
| fix-2 | the same | fix-1 again: the slot's cells removed and the prompt read again (the fork's head rule puts it back into the same cells) | 4,807 / 0 |
| fix-p | the same | fix-1 with `n_probs` 2: the server samples on the host for that request and returns each step's two most likely tokens with their log-probabilities | 4,807 / 0 |

**Arms:** `one-none`, `kvu-none`, `kvu-long`, `kvu-short`, `streams-none`, `streams-long`, `streams-short`, each loaded
twice: 14 loads, about two minutes each.

## Files

| File | What it holds |
|---|---|
| [verdicts.tsv](verdicts.tsv) | per mode: loads, distinct fixed replies, loads with one slot's reply, the histories that change it, repeats inside a load and between the two loads |
| [compare.tsv](compare.tsv) | per load: the fixed reply's tokens and md5, against one slot's (and the reply token where they part), fix-2 and fix-p against fix-1, against the arm's other load |
| [cells.tsv](cells.tsv) | per load: where the fixed request's cells went (stream, first cell, the highest used cell + 1 before its last placement, the other slot's cells in that stream), for fix-1, fix-2 and fix-p |
| [probs.tsv](probs.tsv) | per load: fix-p's log-probabilities against one slot's, step by step up to the first token that differs: steps whose chosen token's log-probability differs at all, the largest difference, the smallest gap between the two most likely tokens |
| [replies.tsv](replies.tsv) | every request: slot, prompt tokens read, cached tokens used, generated tokens, the md5 of the token ids (12 hex digits), the stop reason, the cache's placements (how many, stream, head and used cells before the first, highest used cell + 1 before the last) |
| [loads.tsv](loads.tsv) | per load: the KV cache's cells, sequences and streams, the backend features that name the fork's Pascal kernels, asserts, requests |
| [run/run.sh](run/run.sh) | the driver: one container and llama-server per load, the requests, nothing timed |
| [run/slots.py](run/slots.py) | the requests |
| [run/c987_tables.py](run/c987_tables.py) | the tables from the driver's output; the server's lines carry host times, which it cuts off |

## Commands

On the build machine, with the rig tarball unpacked, a copy of this repository and the model on a local disk:

```bash
WORK=/path/to/work BIN=/path/to/llama.cpp-20261120-3dfad876f-pascal-p106-v5-pipe-grammar-off/bin REPO=/path/to/this/repo \
ORNITH=/path/to/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf MOUNTS='/path/to /models' \
DOCKER_ARGS='--cpu-shares=256' PREFLIGHT=/path/to/cuda_preflight.sh \
setsid nohup bash docs/data/2026-11-24_c987_slot_cells_gpu/run/run.sh > "$WORK/driver.log" 2>&1 < /dev/null &
# when $WORK/DONE exists, with the outputs copied to a machine with Python 3:
python3 -I docs/data/2026-11-24_c987_slot_cells_gpu/run/c987_tables.py "$WORK/out" docs/data/2026-11-24_c987_slot_cells_gpu
```

A first load with `RUN=smoke ARMS=kvu-long REPS=1 SLOTS_ARGS='6000 5000 200 1000 32'` (32-token replies) checked that
the debug lines print, where the fixed request's cells go, and the log-probability file.

## Provenance and licence

The model file is Ornith-1.5-122B Pascal N2P GGUF (`valid1`), this project's quantization of Ornith-1.5-122B. The
prompts are text of this repository. The tables hold counts, cells, log-probabilities and md5 hashes of token ids taken
from the server's replies and log lines: no model weights and no generated text.
