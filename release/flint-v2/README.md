# Flint v2 release: everything in one place

The second release of the Pascal engine, tuned and measured on Flint: what ships, what changed since v1, what was
measured, where every file lives, and how to rebuild the model file and the drafter from scratch. Placeholders:
`RIG-TBD` is a number from the 2026-10-25 rig run, `OWNER-TBD` a decision of the project's owner.

## What ships

| Piece | What it is | Where |
|---|---|---|
| **Engine** | llama.cpp fork for compute 6.1, commit `5215256bd` (rc4: rc3 + the drafter), prebuilt for Ubuntu 24.04+ without AVX, with NVIDIA's CUDA runtime libraries as a second archive | the fork's GitHub release, OWNER-TBD (tag); notes: [release/notes-v2.0.md](../notes-v2.0.md) |
| **K2h-r16** | our Flint file: Q5_0 and Q8_0 tensors, KLD 0.105 against BF16, 29.4 GB | Hugging Face, OWNER-TBD (account and name); card: [huggingface/flint-K2h-r16](huggingface/flint-K2h-r16/README.md) |
| **The staircase drafter** | a 0.29 GB side file that guesses Flint's next token from the hidden states in flight, for `--pipe-guess staircase` | Hugging Face, OWNER-TBD; card: [huggingface/flint-staircase-drafter](huggingface/flint-staircase-drafter/README.md) |
| **Unsloth's UD-Q4_K_XL** | the no-new-file option: the file most people already have, KLD 0.120 | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (not redistributed) |
| The continuation store | Flint's earlier answers (13.8 M tokens, 110 MB) for `--pipe-store` | OWNER-TBD: ship it or not |

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

[results.md](results.md) lists every number behind the release with its command, build, date and registry row. The
headline, nine P106-090 cards:

| | Stock llama.cpp | v1 | **v2** |
|---|---|---|---|
| Generation, UD-Q4_K_XL | RIG-TBD | RIG-TBD | **RIG-TBD** |
| Served chat, K2h-r16, with the drafter | — | — | **RIG-TBD** |

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
| `flint-staircase-drafter-v2.0.gguf` | ~0.29 GB | `export_gguf.py` | RIG-TBD (the head is picked today) |
| `imatrix_unsloth.gguf` (input) | 225 MB | Unsloth | in [CHECKSUMS](CHECKSUMS) |
| `flint-UD-Q4_K_XL.gguf`, the copy every UD number here was measured with | 25.1 GB | Unsloth | in [CHECKSUMS](CHECKSUMS) |

[CHECKSUMS](CHECKSUMS) holds the hashes known now in `sha256sum -c` format; each file's line is added as it is
produced.

## Gaps

What a stranger cannot yet redo exactly from this repository, and what closes each gap:

1. **The engine's release commit is not public yet.** `5215256bd` and the release branch wait for the owner's push to
   the fork (OWNER-TBD), and the release archives for the tag.
2. **The GPTQ output head is byte-exact only by copying it.** A head computed from scratch (dumps on GPUs, GPTQ on the
   CPU) is equivalent within noise, not identical: GPU dumps differ in the last bits between GPUs. The calibration
   data is public (wikitext-2 train and the shipped Q51 transcripts).
3. **The drafter's release head is not chosen yet.** The final round's exact `train.py` arguments and the head's scores
   come with today's pick (RIG-TBD).
4. **The drafter's training data is regenerated, not downloaded.** Answers sampled at temperature 1 on GPUs differ
   between machines, so a rebuilt drafter is equivalent, not identical. A sanitized manifest of the answer files
   (prompt numbers, writer, engine commit, token counts, SHA-256 per file) is still private (OWNER-TBD: publish it, or
   the answers themselves as a dataset).
5. **Parallel orchestration is private.** The project ran generation, dump and training as several copies with
   machine-specific launchers. `reproduce/drafter_pipeline.sh` is the public sequential form of the same commands; it
   has been checked for syntax, not run end to end.
6. **Rig drivers are private.** The scripts that relayed builds to the rig and looped the public bench scripts contain
   machine names; `results.md` gives each number's public command instead. `bench/pipespec/rig_server.sh`'s drafter
   configuration letter lands in the public repo after today's check.
7. **The store file** (110 MB of Flint's answers) is built by `bench/ngram_replay/datastore.cpp` from the generated
   answers, which gap 4 covers; shipping it is OWNER-TBD.
8. **Reference data that is large:** the BF16 logits for the KL-divergence checks (`flint-BF16.c512x80.kld`) and the Q51
   dump (`q51-grid.bin`) are regenerated with the commands in the reproduce pages.
9. **How the mix was found:** the per-group measurements that chose K2h-r16's tensor types (Q73, Q77) ran as private job
   files; their results are in the registry, and the type map, not the search, defines the file.

## Not in this release

Qwen3.6-35B-A3B, Ornith-1.5-35B and Ornith-1.5-122B ([models/](../../models/)), the memory clock offset (hardware
tuning, measured separately), the screened output head's side file, and several conversations through one pipeline in
llama-server.
