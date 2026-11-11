# The MTP draft's token rows gathered on its layer's card (C762, Q1340, 2026-11-11)

What this is: a change on the v4 line that stops the MTP draft's token rows from crossing between two cards through
the host. Under the Flash-Next preset the draft's token embeddings are on card 4 (`-otd token_embd=CUDA4`) and its MTP
layer and head on card 8. Each pass of the draft therefore gathers its token rows on card 4 and sends them, as F32, to
card 8 through host memory, since the cards have no peer-to-peer link. A catch-up pass over a prompt carries 128 rows of
2,560 F32 (1.31 MB). [R81](../2026-11-11_r81_ubatch_trace/README.md) traced a fresh 4,097-token prompt on a P106 with
the v4 night merge (`c408dabcb`): card 8's split waits a median 8.199 ms (2,048-token call) and 8.200 ms (1,533) in `w`
in each catch-up pass, est. that copy, while card 4's own split launches in a median 20.5 and 22.0 µs.

The change is on the fork branch `pascal-p106-v4-b12246-c762` (`950844caf`, three commits on
`pascal-p106-v4-b12246-c748` `5b329df16`), behind `LLAMA_DRAFT_EMBD_DEV=layer`, off by default. This page is the
choice between the ways out, the change and its check on a CPU; the rig run that times it is
[Q1349](../../../quants/experiments.md) (below).

## Result

**Of the three ways out, one keeps the rows exact and fits: the draft's own Q8_0 table on card 8, beside its layer. With
`LLAMA_DRAFT_EMBD_DEV=layer` the draft gathers its token rows on its layer's device, ahead of the preset's `-otd` rule,
and gives the same rows, drafts and replies as without it, byte for byte, on a CPU with two devices laid out as the
preset's cards 4 and 8.** At the preset's 65,536-token line card 8 takes 644.14 MiB more (est. 5,291 MiB after loading,
777 MiB of its 6,068 left) and card 4 645.39 MiB less (est.).

- **The target's rows are not the draft's.** The target (Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0) keeps its token
  embeddings on the host in Q3_K (273,152,000 B), the draft its own in Q8_0 (675,430,400 B), both 2,560 × 248,320
  ([target_embd_head.tsv](target_embd_head.tsv), [draft_tensors.tsv](draft_tensors.tsv), read from the files' headers
  with [run/gguf_tensors.py](run/gguf_tensors.py)). The two are the same matrix quantized twice (row cosine 0.988 mean,
  Q564), not the same bytes: rows taken from the target's host gather would change the draft's numbers, and so its
  drafts. Not exact.
- **A smaller type than Q8_0 is not exact either.** Every type with fewer bits per weight than Q8_0's 8.5 rounds its
  8-bit values to fewer levels, so the rows it gives back differ from the Q8_0 rows. Not pursued.
- **The table on card 8 fits.** On the night merge at the preset's line (R57's load, `-c 65536 -ub 512`, the draft on)
  card 8 holds 4,647 MiB after loading, 1,421 MiB under the 6,068 MiB a P106-090 can use. The table is 644.14 MiB, and
  the rows it gives are the bytes card 4 gives: the same `get_rows` kernel on the same Q8_0 bytes, on the same model of
  card.

The checks of the switch, on a build machine's CPU ([summary.tsv](summary.tsv), [server.tsv](server.tsv),
[placement.tsv](placement.tsv), [hash_nodes.tsv](hash_nodes.tsv), [server_verdict.txt](server_verdict.txt)):

- **llama-server with a separate MTP-only draft, 9 arms** ([run/server_check.sh](run/server_check.sh)). The model is a
  random-weight `qwen4exp` (Flash-Next's architecture) with one MTP block, and the draft file is that block alone with
  its token embeddings in Q8_0, as Flash-Next's MTP draft (`test-llama-archs -o --mtp-draft`, new). Draft-mtp at three
  tokens with `p_min` 0, four requests (two greedy, two at temperature 1.0 with the preset's top-k 20 and top-p 0.95),
  the draft's catch-up passes 64 rows (`--spec-draft-n-ubatch 64`).
  - On the CPU alone, the branch with the switch unset and set gives the base build's (`5b329df16`) tokens, top-3
    probabilities, drafted and accepted counts and all 406 verify steps of `LLAMA_SPEC_DUMP` (the draft, the drafter's
    candidates and probabilities, the target's top tokens). With the switch the table stays on the CPU, where the
    layer is.
  - **The preset's layout on two devices:** with the RPC build, two `ggml-rpc-server`s on the same machine are two
    devices; every layer is offloaded (`-ngl 99 -ngld 99`), so the draft's MTP layer is on the second, and
    `-otd token_embd=<the first>` puts its table on the first, as the preset's `-otd token_embd=CUDA4` with the layer
    on card 8. With `LLAMA_DRAFT_EMBD_DEV=layer` the loader puts the table on the second device and says that the
    override was not applied. The arm with the switch gives the same replies, probabilities, draft counts and 406
    verify steps as the arm without it; so does the pair with the backend sampler (`-bs`, as the preset).
  - **The rows themselves:** with `LLAMA_TENSOR_HASH=mtp_` (new, a check hook) every node of the draft's graph named
    `mtp_*` is hashed once computed, with the buffer that holds it. 7,863 nodes a run, the same names, shapes and bytes
    with the switch and without. Of them 1,661 are the token rows (`mtp_tok_embd`): 24 catch-up passes over prompt
    chunks (64 rows), 404 over a verify step's tokens (4 rows), 11 others (2, 3 and 20 rows), and 1,222 drafting passes
    (1 row). Without the switch every one is gathered on the first device, with it on the second; the target's hidden
    rows (`mtp_h_input`), their norm, the token rows' norm and the `eh_proj` product are the same bytes too.
- **With the variable unset, set to `layer`, and with `LLAMA_TENSOR_HASH` on three node names,** `test-llama-archs -b
  CPU -s 1` gives `5b329df16`'s logits on 146 of 146 random models (none of them is an MTP-only file, so the switch
  places nothing there, and the hash hook changes no logit). **ctest: 70 of 70 pass** (the static CPU build; the tests
  that need a download, the network or torch are left out, as in C565).
- **The rig tarball builds:** `llama.cpp-20261111-950844caf-pascal-p106-v4-b12246-c762.tar.gz`, 656,346,907 B, md5
  `b9d0b06e835f512127ed44320b7c96c6` (`TESTS=1`, sm_61, 0 AVX instructions, GLIBC_2.38, `test-llama-archs` in the
  tree's `bin/`).

## Memory per card at the preset's split (computed from the tensors, est.)

The draft's file holds 2,686,492,672 B: the token embeddings (Q8_0, 644.14 MiB), the output head (Q6_K, 497.31 MiB) and
the MTP layer ([draft_tensors.tsv](draft_tensors.tsv)). Without the switch the table is the draft's whole buffer on
card 4 (R57: `CUDA4 model buffer size = 644.14 MiB`, the rest 1,917.90 MiB on card 8). With it the draft has no tensor
and no split on card 4.

| Card | Night merge, `-c 65536 -ub 512`, after loading (R57), MiB | With the switch, est., MiB | Change, est. |
|---:|---:|---:|---|
| 0–3, 5–7 | 5,059, 5,095, 4,963, 5,111, 4,979, 4,381, 4,991 | the same | none |
| 4 | 4,875 | 4,230 | −644.14 MiB of table, −1.25 MiB of the draft's compute buffer there (the 128 gathered rows) |
| 8 | 4,647 | 5,291 | +644.14 MiB of table; the draft's compute buffer on card 8 holds the gathered rows instead of their copy (est. the same size; on the CPU check the layer's device's draft buffer grew 0.26 MiB, from 6.00 to 6.26, at 64 rows of 256 F32) |

- **At the preset's line card 8 becomes the fullest card,** est. 5,291 MiB after loading and 5,351 at the highest (R39b's
  highest on v3.0, 4,625, +82 MiB of the night merge's hidden-row buffer, +644): 717 MiB left, against card 3's 915.
- **At 131,072 tokens (`-ub 256`, S38's line, measured on v3.0)** card 8 held 4,935 MiB at its highest; with the night
  merge's 82 MiB and the table, est. 5,661: 407 MiB left, then the fullest card (card 3 had 649 left).
- **At 196,608 tokens (`-ub 256`, S40's line, v3.0)** card 8 held 5,393 MiB at its highest; est. 6,119 with both, 51 MiB
  over the 6,068: that line does not fit with the switch unless a layer moves off card 8 (`-ts`).
- The host is unchanged: with `--no-host` the table was on card 4 and is now on card 8; nothing goes back to host memory.

## What the switch does

- `LLAMA_DRAFT_EMBD_DEV=layer` (or `1`) is read when a model loads. A file that holds only its MTP layers (no tensor of
  layer 0, one of the first NextN layer, as Flash-Next's separate MTP draft) gets the buffer types of its first MTP
  layer's device for its token embeddings. The loader then places `token_embd` there, ahead of the input layer's host
  buffer and of any `-otd` rule for it, and logs the rule it did not apply and where the table went
  (`llama_model_loader: token_embd.weight (644.14 MiB q8_0) in CUDA8 (LLAMA_DRAFT_EMBD_DEV)` on the rig, est.). The
  output head that reuses the table in some files, the other input-layer tensors, and every file with its trunk (a
  target, or an MTP context on the target's own weights) are placed as before. Any other value warns and places the
  table as without it.
- The draft's `get_rows` runs where its weight is, so each pass sends card 8 its token ids (512 B for 128 rows) instead
  of 1.31 MB of F32 rows from card 4 through the host, and the draft's graph has no split on card 4.
- It follows the layer's device whatever the split: on another card count the table goes where `-ts` puts the draft's
  layer, with no `-otd` rule to rewrite.
- With it on, `LLAMA_DRAFT_EMBD_CACHE` (Q961's row cache on the layer's card) has nothing left to save.
- `LLAMA_TENSOR_HASH=<prefix>[,<prefix>...]` (new, checks only): every graph node whose name starts with a prefix is
  hashed (FNV-1a) once computed, through the scheduler's eval callback, and logged at the library's INFO level with its
  type, shape and buffer (`tensor-hash: <v|d> <n> <name> ...`, `d` the draft's context). Unused when the caller sets an
  eval callback; while set, the scheduler computes each split in pieces that end at those nodes.

## What it changes on the rig (est.)

- **A fresh prompt's catch-up:** R81's 8.199 ms a pass, est. the rows' crossing, over the 33 catch-up passes of a fresh
  4,097-token prompt: est. ~0.27 s (R81's request 7 read its prompt in 28,029.24 ms), less where the catch-up overlaps
  the target's next call (C703's `LLAMA_MTP_CATCHUP_OVERLAP`).
- **Decode:** each drafting pass and each catch-up over a verify step's tokens stops crossing card 4 too (one 10 KB row,
  or four). Q961's row cache removed the same crossings for batches of up to 16 tokens and measured negligible in the
  greedy request's steps on v3.0 (S37: −0.91 ± 0.17 ms on 82 paired steps, the preset's second load −0.68 ± 0.13), so
  est. under 1 ms a step here; the switch has none of the cache's host copies of rows.

## The rig run (Q1349)

On rig 1, untimed: R81's driver (R42's `r42_run.sh` with `R42_LOADS=warm4096` and `LLAMA_UBATCH_TRACE=1`, the v3 preset
with the MTP draft) on this branch's tarball twice, one load after the other: the switch off, then
`LLAMA_DRAFT_EMBD_DEV=layer`. Then R81's tables per load ([R81's scripts](../2026-11-11_r81_ubatch_trace/README.md)
with C701's `trace_table.py`) and [run/r104_compare.py](run/r104_compare.py) side by side. Read:

- card 8's `w` per 128-row catch-up pass in request 7 (`draft call of 2048 rows: CUDA8 w` and `1533 rows`; R81: 8.199
  and 8.200 ms), and no `CUDA4` row with the switch on;
- the replies' md5 per request (`requests.tsv`): the same with the switch off and on;
- the memory per card after loading (`memory_loaded.txt`) against the table above, and the load's `load_tensors` and
  `sched_reserve` lines for cards 4 and 8;
- the prompt times of the four fresh 4,097-token requests, as a diagnostic (one load each).

## Files

| File | What it is |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict: the builds, the server check, the logits identity, ctest, the rig tarball |
| [server.tsv](server.tsv) | every server arm's requests: prompt and generated counts, draft counts, the tokens |
| [placement.tsv](placement.tsv) | per arm: the switch's log lines, where the table went, the override not applied, the hashed draft nodes and the token rows' buffers |
| [hash_nodes.tsv](hash_nodes.tsv) | the hashed draft nodes of the two hash arms by node and shape: passes, each arm's buffer, how many have the same bytes |
| [server_verdict.txt](server_verdict.txt) | the server check's verdict |
| [draft_tensors.tsv](draft_tensors.tsv) | the MTP draft file's tensors: type, shape, bytes |
| [target_embd_head.tsv](target_embd_head.tsv) | the target's token embeddings and output head: type, shape, bytes |
| [run/checks.sh](run/checks.sh) | the build machine's steps: CPU builds (plain and RPC), the server check, the logits identity, ctest, the rig tarball |
| [run/server_check.sh](run/server_check.sh) | the server check's arms and verdict |
| [run/gguf_tensors.py](run/gguf_tensors.py) | a GGUF file's tensors from its header (no numpy) |
| [run/r104_compare.py](run/r104_compare.py) | the rig run's two loads side by side from R81's tables |

## Commands

On the build machine, CPU only (Docker with `--cpu-shares=256`, `nice -n 19`, `-j 8`, no GPU), the work folder `WORK`
holding a clone of llama.cpp with the fork as remote `fork` (`SRC`), and the base's static CPU builds:

```bash
WORK=$WORK REPO=<this repository> SRC=$WORK/rig/src BRANCH=pascal-p106-v4-b12246-c762 \
  BASE_CPU=<bin/ with 5b329df16's test-llama-archs> BASE_SRV=<dir with 5b329df16's llama-server> \
  LOCK=<a flock file> JOBS=8 bash docs/data/2026-11-11_c762_draft_rows/run/checks.sh
python3 -I docs/data/2026-11-11_c762_draft_rows/run/gguf_tensors.py $MODELS/mtp-Qwen3.8-Flash-Next-draft.gguf > draft_tensors.tsv
python3 -I docs/data/2026-11-11_c762_draft_rows/run/gguf_tensors.py \
  $MODELS/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf '^(token_embd|output)\.' > target_embd_head.tsv
```

The server check alone, on a CPU with `curl` and `jq`, after `test-llama-archs -a '^qwen4exp$' -s 1 -o $OUT/models
--mtp-draft`:

```bash
BASE=<base llama-server dir> BIN=<branch llama-server dir> RPC_BIN=<branch GGML_RPC build's bin> \
  MODEL=$OUT/models/qwen4exp-moe-mtp.gguf DRAFT=$OUT/models/qwen4exp-moe-mtp-draft.gguf OUT=$OUT/server \
  SARGS='--override-kv qwen4exp.context_length=int:16384' bash docs/data/2026-11-11_c762_draft_rows/run/server_check.sh
```

## Provenance and licence

The random test models are written by the branch's own `test-llama-archs`. The headers read are those of the Gyro MTP
draft for Qwen3.8-Flash-Next (`mtp-Qwen3.8-Flash-Next-draft.gguf`) and of UkisAI's Swift-1.5 GSQ-RCO Q2_0 (its first
shard); only tensor names, types, shapes and sizes are published. The tables and scripts here are this repository's
licence.
