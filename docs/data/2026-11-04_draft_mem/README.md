# The MTP draft's memory and its head: shared tensors, a token shortlist, smaller load staging (Q564, Q635, Q592, Q580)

Three engine switches for the Gyro MTP draft of Qwen3.8-Flash-Next on the 4 GB host, each off by default, on the fork
branch `pascal-p106-v3-b12076-draftmem` (`1e0caf230`, one commit on the speed ladder's `pascal-p106-v3-b12076-ladder`).
Result rows: [Q564, Q635, Q592 and Q580 in the registry](../../../quants/experiments.md). What the draft cost before
them: [the MTP draft within the 4 GB host](../2026-11-03_flash_next_mtp/README.md) (S15) and
[what verifying k tokens costs](../2026-11-03_verify_cost/README.md) (S21). Their memory and speed on the rig are the
next measurement; this folder holds the checks that run without a GPU.

| Switch | What it does | Saves |
|---|---|---|
| `LLAMA_DRAFT_SHARE=embd` | the draft reads the main model's token embeddings instead of loading its own | the draft's 644 MiB `token_embd` (Q8_0): host memory, or a card's with `-otd token_embd=CUDA<n>` |
| `LLAMA_DRAFT_SHARE=embd,head` | also the main model's output head | the draft's 497 MiB head (Q6_K) on its card, and a cheaper head per draft step (Q5_K: 0.84× the bytes) |
| `LLAMA_DRAFT_SHORTLIST=<k>[,<m>[,<r>]]` | the draft's head over a token shortlist: the ids below `k`, the special tokens, an optional frequency list (`LLAMA_DRAFT_SHORTLIST_FILE`, one id per line) and the context's tokens, up to `m` beyond `k`; the full head for `r` drafts after a main-model token outside the list | most of the head's 18.5 of 30.9 ms per draft step (S21) |
| `LLAMA_DIO_STAGING=<n>x<MiB>` | the direct-I/O loader's staging: `n` pinned upload buffers of that size (default 4 × 64 MiB), and the bounce buffer of reads into host memory | pinned memory at every buffer load: 256 → 32 MiB at `2x16` |

Drafts never change the output: verification keeps only what the main model itself produces, so the shortlist and a
shared tensor can change only the acceptance.

## The tensors are the same matrices

`run/embd_equal.py` compares the two files on a CPU ([embd_equal.md](embd_equal.md), per block of 8,192 rows in
[embd_equal_rows.tsv](embd_equal_rows.tsv)):

| | draft | main model (Swift Q2_0) | per-row cosine (mean, min) | relative RMS difference (mean, max) | shuffled rows (cosine) |
|---|---|---|---|---|---|
| vocabulary | 248,320 tokens, 247,587 merges | the same, hash for hash | | | |
| `token_embd.weight` | Q8_0, 675.4 MB | Q3_K, 273.2 MB | 0.9884, 0.9798 | 0.153, 0.203 | 0.046 |
| `output.weight` | Q6_K, 521.5 MB | Q5_K, 437.0 MB | 0.9991, 0.9978 | 0.042, 0.066 | 0.172 |

Not byte-identical: the draft keeps them in Q8_0 and Q6_K, the main model in Q3_K and Q5_K. Every row agrees to within
the main model's rounding (no row below 0.97), against ~0.05 for unrelated rows, so both are the same trained matrices.
With `LLAMA_DRAFT_SHARE=embd` the draft's input embedding carries Q3_K's rounding (the same the main model's first
layer sees); with `head`, its logits carry Q5_K's.

## How each switch works

- **Shared tensors** (`llama_model_params::share_src`, `share_tensors`): the draft's loader skips its own
  `token_embd.weight` / `output.weight` (their bytes are never read) and points the draft at the main model's tensors,
  after checking the shape and a spread of token texts. The draft's graph is unchanged: its `get_rows` on the
  embeddings runs where the main model keeps them (the CPU with `--no-host`), its head where the main model's head is
  (the last card, where the draft's layer also is). `qwen4exp` supports it; other architectures log that they load
  their own.
- **Shortlist** (`llama_head_shortlist_*`, `src/llama-head-shortlist.cpp`): the ids below `k` are a plain view of the
  first `k` rows of the head, so they take the usual matrix-vector kernel; the further ids go through `mul_mat_id`
  with one head row per "expert", as the screened head does; a `concat` with a row of −∞ and a `set_rows` give the
  sampler a full 248,320-wide row. In a BPE vocabulary the low ids are the early, frequent merges, and Qwen's special
  tokens (chat markup, tool calls, the end of a turn) sit at the top of the id range, so they join the list by
  attribute. The further ids live on the head's card and change only when the context brings new tokens. Decodes with
  more than one output use the full head.
- **Staging**: `load_all_data` stages each buffer it loads through `n` pinned buffers and allocates them again for every
  buffer; the draft loads after the main context holds its pinned compute buffer, so its staging comes on top (S15).

## Checks without a GPU

**Unit test** `test-draft-mem` (fork `tests/`, CPU backend): 82 checks pass. The shortlist's logits equal the full
head's exactly (largest difference 0) for F32, Q8_0, Q5_K, Q6_K and Q3_K heads, with only the low ids, with further
ids, and with ids added after the device copy exists; every other token is −∞, and the argmax over the list is the
full head's whenever that token is in it. The list's bookkeeping, the fallback policy and both settings parse as
documented.

**The real files on a CPU** (`run/cpu_draft_check.sh`; the rig build in its build image with no GPU, so every tensor
on the CPU; two greedy requests per load, 96 and 53 tokens; [cpu_check.tsv](cpu_check.tsv)):

| Arm | switches | code prompt: accepted / drafted | explanation prompt: accepted / drafted | text = no draft |
|---|---|---|---|---|
| base | none | 70 / 73 | 33 / 57 | yes |
| embd | `LLAMA_DRAFT_SHARE=embd` | 70 / 73 | 32 / 60 | yes |
| share | `embd,head` | 70 / 73 | 33 / 57 | yes |
| short | `LLAMA_DRAFT_SHORTLIST=32768,8192,2` (own head) | 70 / 73 | 30 / 66 | yes |
| shareshort | `embd,head` + the shortlist | 70 / 73 | 31 / 63 | yes |
| shortfile | the same + a frequency list ([token_freq_s12.tsv](token_freq_s12.tsv)) | 70 / 73 | 31 / 63 | yes |

- Every arm returns the main model's own greedy text, and the loader logs the skipped tensors (675,430,400 and
  521,472,000 bytes).
- Sharing both tensors drafts exactly as the draft's own copies on these requests; the embeddings alone cost one
  accepted token of 33.
- The shortlist (32,801 ids: the 32,768 lowest and 33 special tokens) held 5 of the ~150 generated tokens outside it;
  each sent the next two drafts to the full head (36–37 drafts with the shortlist, 10 with the full head). On the
  explanation prompt it accepted 30–31 of 63–66 against 33 of 57: more drafted and fewer accepted, as when a draft
  runs on with an in-list token where the right one is outside the list.
  Two requests are far too few to settle the acceptance; the rig's agent check does.
- The frequency list holds only 805 distinct tokens: an agent check's logged requests contain the model's earlier turns
  (123 distinct, 8,626 tokens), not its reasoning, so it added 121 ids. A useful list needs the streamed outputs
  (Q636).
- Generation speed on this CPU build (no AVX, a shared machine at low priority) is not a measurement and is left out.
- `-lm dio` was not run here: the CPU path reads the whole main model into memory, and the pinned staging exists only
  with a GPU, so the rig measures it.

## Files

| File | What |
|---|---|
| [embd_equal.md](embd_equal.md), [embd_equal_rows.tsv](embd_equal_rows.tsv) | the tensor comparison |
| [cpu_check.tsv](cpu_check.tsv) | the real files on a CPU, per arm and request |
| [token_freq_s12.tsv](token_freq_s12.tsv) | token id and count of the assistant turns in S12's agent-check requests |
| `run/embd_equal.py` | the comparison (gguf-py, numpy) |
| `run/cpu_draft_check.sh` | the switches on the real files, on a CPU |
| `run/token_freq.sh` | a frequency list for `LLAMA_DRAFT_SHORTLIST_FILE` from an agent check's logged requests |

## Commands

```bash
# the comparison (any x86-64 machine with the files; gguf-py from a llama.cpp tree)
DRAFT=<mtp-Qwen3.8-Flash-Next-draft.gguf> TARGET=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> OUT=out \
  PYTHONPATH=<llama.cpp>/gguf-py python3 run/embd_equal.py

# the engine (rig build with tests), then the unit test in the same tree
TESTS=1 JOBS=6 scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-draftmem
cmake --build build --target test-draft-mem && build/bin/test-draft-mem

# the real files on a CPU: inside the build image, no GPU passed in, the toolkit's stub driver standing in for
# libcuda (CUDA then finds no device), curl and jq installed
ln -s /usr/local/cuda/lib64/stubs/libcuda.so /tmp/stub/libcuda.so.1 && export LD_LIBRARY_PATH=/tmp/stub
BIN=<build>/bin MODEL=<main model shard 1> DRAFT=<draft> OUT=out THREADS=16 BODIES=<proxy.jsonl.bodies.gz> \
  ARMS="nodraft base embd share short shareshort shortfile" bash run/cpu_draft_check.sh
```

## Provenance and licence

The main model is Swift-1.5's GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (re-sharded into three files), the draft the Gyro
repository's `mtp-Qwen3.8-Flash-Next-draft.gguf` (sha256 `00818378…391b`). The tables hold statistics of their tensors
and token counts, no weights and no text.
