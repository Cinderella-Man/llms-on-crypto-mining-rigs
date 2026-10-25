# Flint v2 release: everything in one place

The second release of the Pascal engine, tuned and measured on Flint: what ships, what changed since v1, what was
measured, where every file lives, and how to rebuild the model file and the drafter from scratch. Placeholders:
`RIG-TBD` is a number from the 2026-10-25 rig run, `OWNER-TBD` a decision of the project's owner.

## The headline: near-lossless Flint on nine mining cards, at full context, faster than 4-bit

Flint in 8-bit (Unsloth's Q8_0, 42.3 GB) runs on all nine P106-090 cards of a retired mining rig with the full
262,144-token context, and it generates faster than the usual 4-bit file. The 8-bit file is near-lossless: it picks
the original model's next token 93.3 % of the time (KL divergence 0.052 against the BF16 weights), against 87.7 % and
0.120 for Unsloth's UD-Q4_K_XL.

| Nine P106-090 cards, the full 262,144-token context, served chat | Stock llama.cpp | v1 | **v2** |
|---|---|---|---|
| **Unsloth Q8_0** (42.3 GB, 93.3 % same top token) | 14.78 tok/s | 29.55 | **RIG-TBD** |
| Unsloth UD-Q4_K_XL (25.1 GB, 87.7 %) | 14.20 | 27.97 | RIG-TBD |
| Q8_0 with pipelined speculation (copy lookup, store and drafter) | — | — | **RIG-TBD** |

The v1 rows are a five-turn conversation at the full context with the same settings and seeds on both engines
([the chat benchmark](../../docs/benchmarks/2026-10-23-flint-chat.md)). Why 8 bits can beat 4 bits here: on 5-SM
Pascal cards much of a token's time goes into unpacking compressed weights; the Pascal engine decodes the simple 8-bit
format at close to the cards' memory speed, while the elaborate 4-bit formats need more instructions per byte. So the
file that reads 2.70 GB per token (Q8_0) keeps up with, and on v1 beat, the one that reads 2.21 GB (UD-Q4_K_XL).

The second story is the 4-bit work: K2h-r16, this project's own Flint file, reaches better quality than UD-Q4_K_XL
in 29.4 GB and fits six cards; and the trained drafter (below).

## What ships

| Piece | What it is | Where |
|---|---|---|
| **Engine** | llama.cpp fork for compute 6.1, commit `2d53179aa` (rc4: rc3 + the drafter), prebuilt for Ubuntu 24.04+ without AVX, with NVIDIA's CUDA runtime libraries as a second archive | the fork's GitHub release, OWNER-TBD (tag); notes: [release/notes-v2.0.md](../notes-v2.0.md) |
| **Unsloth's Flint Q8_0** | the headline file: near-lossless, 42.3 GB, nine cards at the full context | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (not redistributed) |
| **K2h-r16** | our Flint file: Q5_0 and Q8_0 tensors, KLD 0.105 against BF16, 29.4 GB, fits six cards | Hugging Face [CinderellaMan/Flint-K2h-r16-GGUF](https://huggingface.co/CinderellaMan/Flint-K2h-r16-GGUF); card: [huggingface/flint-K2h-r16](huggingface/flint-K2h-r16/README.md) |
| **The staircase drafter** | two ~0.27 GB side files that guess Flint's next token from the hidden states in flight, for `--pipe-guess staircase`: R2-9 for nine cards, R2-gen for six cards and any other count from 3 to 12 | Hugging Face [CinderellaMan/Flint-Staircase-Drafter](https://huggingface.co/CinderellaMan/Flint-Staircase-Drafter); card: [huggingface/flint-staircase-drafter](huggingface/flint-staircase-drafter/README.md) |
| **Unsloth's UD-Q4_K_XL** | the file most people already have, KLD 0.120 | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (not redistributed) |
| The continuation store | Flint's earlier answers (13.8 M tokens, 110 MB) for `--pipe-store`; the drafter's served numbers were measured with it in front | OWNER-TBD: it is built from Flint's answers, so it waits for the decision on sharing data; it would ship with the drafter files ([UPLOAD.md](huggingface/UPLOAD.md)) |
| The drafter's data | **later, not in this release:** the prompts and Flint's answers with token ids (without the ShareGPT rows), the heads in training format and their training checkpoints | draft card: [huggingface/flint-drafter-data](huggingface/flint-drafter-data/README.md); how the data is made: [reproduce/drafter.md](reproduce/drafter.md) |

Everything this project publishes is under Apache-2.0, like Flint itself, and is checked by SHA-256 (in
[CHECKSUMS](CHECKSUMS) and in each Hugging Face repository; no GPG signatures). Unsloth's files are linked, not
re-uploaded. The upload plan, with the checks before each upload: [huggingface/UPLOAD.md](huggingface/UPLOAD.md). Made
by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X).

How to run each: the [Flint page](../../models/flint/README.md#command-lines).

## What changed since v1

v1 (2026-10-23, `eeaa59660`) doubled Flint's speed over stock llama.cpp with CUDA graphs on Pascal, a launch shape for
one-token products, and Pascal matrix-vector kernels. v2 adds, in the order they help a single conversation:
- **32-bit activation sums,** so llama.cpp's MoE rescale is dropped for one-token steps (+9.6 %), and the shared expert
  fused with its activation (+0.7 %);
- **one persistent kernel per MoE layer** (+5.6–8.8 %), a decode-attention kernel, and less host work per token;
- **pipelined speculation** in llama-server (`--pipe-depth`): guessed tokens on the idle cards of the layer split, with
  the copy lookup, a store of earlier answers and **the trained drafter** as guessers;
- **tool requests sampled on the GPU** (they decoded 12–16 % slower than chat before);
- **prefill:** MoE tiles sized by tokens per expert (+22–29 %), a prefill-attention kernel, the attention mask built
  on each card (decode at 64k +12 %), and a prompt cache for agents;
- **Pascal kernels for Q4_K and Q6_K,** for other models' common downloads.

Every piece has its own switch ([reproduce/engine.md](reproduce/engine.md)) and passed its exactness checks before it
joined: outputs equal the base build's, and pipelined output equals plain decoding's.

## What was measured

[results.md](results.md) lists every number behind the release with its command, build, date and registry row. Beyond
the headline above, nine P106-090 cards:

| | Stock llama.cpp | v1 | **v2** |
|---|---|---|---|
| Generation (`llama-bench`), Unsloth Q8_0 | RIG-TBD | 35.9 (kernels on, before v1's release) | **RIG-TBD** |
| Generation, UD-Q4_K_XL | 16.5 | 32.6 | **39.9** (the v2-next build; release build RIG-TBD) |
| Generation, K2h-r16 | — | — | **41.8** (v2-next; release build RIG-TBD) |
| Served chat, K2h-r16, with the store and the drafter (8k context) | — | — | **50.4** (48.6 with the copy lookup, continuations and store, 33.4 without guessing) |

The write-ups behind the work: [docs/research/notes](../../docs/research/notes/README.md) (the v2 synthesis, release
settings, pipelined speculation, the drafter), [docs/drafter.md](../../docs/drafter.md), and the registry
[quants/experiments.md](../../quants/experiments.md) (every experiment, Q1–Q163, and the decision log).

## Rebuild it yourself

| What | How | Needs |
|---|---|---|
| The engine | [reproduce/engine.md](reproduce/engine.md): `scripts/build_llamacpp_docker.sh`, `scripts/package_release.sh` | Docker on any x86-64 Linux PC |
| K2h-r16 | [reproduce/model.md](reproduce/model.md), `reproduce/build_k2h_r16.sh`: Flint BF16 → llama-quantize with the tensor-type map → GPTQ head → F16 routers → header; checked tensor by tensor against [k2h-r16.tensors.tsv](reproduce/k2h-r16.tensors.tsv) | CPU and ~150 GB of disk; GPUs only to compute the head yourself |
| The drafter | [reproduce/drafter.md](reproduce/drafter.md), `reproduce/drafter_pipeline.sh`: prompts → Flint's answers → hidden-state dump → training → scoring → export | GPUs (days of GPU time), ~1.3 TB of disk |

Big data is not in git. The model files download from Hugging Face; Flint's original weights and Unsloth's importance
matrix are pinned by revision and SHA-256 ([reproduce/inputs.sha256](reproduce/inputs.sha256)); the drafter's training
data (74.66 M generated answer tokens, ~1.3 TB of dumps) is defined by its prompt files, seeds and settings. The one
small dataset the checks need, the 38 held-out Flint transcripts, is in [reproduce/data/q51](reproduce/data/q51/).

## Files and checksums

| File | Size | Produced by | SHA-256 |
|---|---|---|---|
| Engine archives (binaries, CUDA runtime, tests) | ~230 / 554 / 24 MB | `scripts/package_release.sh` on rc4 | RIG-TBD (rc4 not packaged yet; rc3 and rc1 packages exist) |
| `flint-K2h-r16.gguf` (release header) | 29.4 GB | `reproduce/build_k2h_r16.sh` | RIG-TBD |
| `flint-q113-K2h-r16.gguf` (the same tensors, the build header) | 29.4 GB | the project's build | in [CHECKSUMS](CHECKSUMS) |
| `flint-staircase-drafter-9cards-v2.0.gguf` (R2-9), `flint-staircase-drafter-v2.0.gguf` (R2-gen) | ~0.27 GB each | `export_gguf.py` | RIG-TBD (exported for the release) |
| `imatrix_unsloth.gguf` (input) | 225 MB | Unsloth | in [CHECKSUMS](CHECKSUMS) |
| `flint-UD-Q4_K_XL.gguf`, the copy every UD number here was measured with | 25.1 GB | Unsloth | in [CHECKSUMS](CHECKSUMS) |

[CHECKSUMS](CHECKSUMS) holds the hashes known now in `sha256sum -c` format; each file's line is added as it is
produced.

## Gaps

What a stranger cannot yet redo exactly from this repository, and what closes each gap:

1. **The engine's release commit is not public yet.** `2d53179aa` and the release branch wait for the owner's push to
   the fork (OWNER-TBD), and the release archives for the tag.
2. **The GPTQ output head is byte-exact only by copying it.** A head computed from scratch (dumps on GPUs, GPTQ on the
   CPU) is equivalent within noise, not identical: GPU dumps differ in the last bits between GPUs. The calibration
   data is public (wikitext-2 train and the shipped Q51 transcripts).
3. **The drafter heads may still change.** The release heads are R2-9 (nine cards) and R2-gen (six cards and other
   counts), with their exact `train.py` arguments in the guide; the final round (15:00) continues R2-9 on ~50 M answer
   tokens, and if its winner replaces R2-9, its arguments and scores replace R2-9's (RIG-TBD).
4. **The drafter's training data is regenerated, not downloaded.** Publishing it is parked for a later release; its
   draft card leaves out the 8,865 answers to ShareGPT prompts (12.62 M of 74.7 M answer tokens, unclear terms), which
   the release heads were trained with. Answers sampled at temperature 1 on GPUs differ between machines, so a drafter
   rebuilt from regenerated answers is equivalent, not identical.
5. **Parallel orchestration is private.** The project ran generation, dump and training as several copies with
   machine-specific launchers. `reproduce/drafter_pipeline.sh` is the public sequential form of the same commands; it
   has been checked for syntax, not run end to end.
6. **Rig drivers are private.** The scripts that relayed builds to the rig and looped the public bench scripts contain
   machine names; `results.md` gives each number's public command instead (the drafter's served numbers come from
   `bench/pipespec/rig_server.sh` with its `d` configuration).
7. **The store file** (110 MB of Flint's answers) is built by `bench/ngram_replay/datastore.cpp` from the generated
   answers, which gap 4 covers. Whether it ships is OWNER-TBD. The drafter's served numbers were measured with it in
   front; the copy lookup and continuations without it were not measured (RIG-TBD: a rig test of each piece).
8. **Reference data that is large:** the BF16 logits for the KL-divergence checks (`flint-BF16.c512x80.kld`) and the Q51
   dump (`q51-grid.bin`) are regenerated with the commands in the reproduce pages.
9. **How the mix was found:** the per-group measurements that chose K2h-r16's tensor types (Q73, Q77) ran as private job
   files; their results are in the registry, and the type map, not the search, defines the file.

## Acceptance check: everything needed to rebuild every shipped file

Each shipped file, the guide and script that rebuild it, the checksum a rebuild must match, and what still stands in the
way:

| Shipped file | Rebuild guide | Script | Expected checksum | Status |
|---|---|---|---|---|
| `llama-p106-v2.0-bin-ubuntu-cuda12.9-sm61.tar.gz` (engine) | [reproduce/engine.md](reproduce/engine.md) | `scripts/build_llamacpp_docker.sh`, then `scripts/package_release.sh` | RIG-TBD (rc4 package) | **gap 1:** the release commit is not public yet. Packaging is deterministic: the same binaries give the same archive (two runs on rc3's build, identical bytes). A fresh compile is checked by `BUILDINFO.txt`'s commit and `test-backend-ops`, not by its bytes |
| `cudart-llama-p106-v2.0-bin-ubuntu-cuda12.9.tar.gz` (CUDA runtime) | [reproduce/engine.md](reproduce/engine.md) | `scripts/package_release.sh` (from NVIDIA's `cuda:12.9.2-devel-ubuntu24.04` image) | `65b46821ab94c5a46ec762f4fc8896de857ada120770d5a2499695ed3277a3bf` (the same for any engine commit packaged as v2.0) | complete: byte for byte from the same image (two runs identical) |
| `llama-p106-v2.0-tests-ubuntu-cuda12.9-sm61.tar.gz` | [reproduce/engine.md](reproduce/engine.md) | as the engine | RIG-TBD | as the engine |
| `flint-K2h-r16.gguf` | [reproduce/model.md](reproduce/model.md) | `reproduce/build_k2h_r16.sh` | RIG-TBD (release header); every tensor in [k2h-r16.tensors.tsv](reproduce/k2h-r16.tensors.tsv) | verified: the script rebuilt it from BF16 with all 779 tensors equal byte for byte (2026-10-25, head copied); **gap 2:** the output head is byte-exact only when copied |
| `flint-staircase-drafter-9cards-v2.0.gguf`, `flint-staircase-drafter-v2.0.gguf` | [reproduce/drafter.md](reproduce/drafter.md) | `reproduce/drafter_pipeline.sh` (`CARDS=9` for the nine-card head), `export_gguf.py` | RIG-TBD | **gaps 3–5:** regenerated data gives equivalent heads, not the same bytes |
| The continuation store, if shipped | [reproduce/drafter.md](reproduce/drafter.md), section 9 | `bench/ngram_replay/datastore.cpp` | RIG-TBD | **gap 7** |
| Later, not in this release: the drafter's dataset (prompts, answers, heads U/R1/R2-9/R2-gen with checkpoints) | [huggingface/flint-drafter-data](huggingface/flint-drafter-data/README.md) (draft), [reproduce/drafter.md](reproduce/drafter.md) | `prompts.py`, `tool_prompts.py`, `generate.py`, `train.py` | the dataset's CHECKSUMS, when published | **gap 4:** parked; the draft leaves out the 8,865 ShareGPT answers |
| Release notes | [notes-v2.0.md](../notes-v2.0.md) | `scripts/package_release.sh NOTES_VARIANT=rc4` | — | complete once the numbers are in |
| Hugging Face cards | [huggingface/](huggingface/UPLOAD.md) | — | — | complete once the numbers and names are in |
| The Q51 test set (inputs to the checks) | [reproduce/data/q51](reproduce/data/q51/README.md) | `bench/expert_overlap` | in git | complete |

## Not in this release

Qwen3.6-35B-A3B, Ornith-1.5-35B and Ornith-1.5-122B ([models/](../../models/)), the memory clock offset (hardware
tuning, measured separately), the screened output head's side file, and several conversations through one pipeline in
llama-server.
