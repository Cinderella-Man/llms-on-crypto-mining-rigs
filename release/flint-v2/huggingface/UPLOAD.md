# Uploading the release's big files to Hugging Face

Nothing here has been uploaded. This page is the plan: which repositories, which files, in what order, and the checks
before each upload. `OWNER-TBD` marks the owner's decisions (the licences, the header's credit line, what the dataset
holds). The account is [CinderellaMan](https://huggingface.co/CinderellaMan).

## Repositories

| Repository | Kind | Files | Folder here |
|---|---|---|---|
| `CinderellaMan/Flint-K2h-r16-GGUF` | model | `flint-K2h-r16.gguf` (29.4 GB), `README.md`, `LICENSE`, `CHECKSUMS` | [flint-K2h-r16/](flint-K2h-r16/) |
| `CinderellaMan/Flint-Staircase-Drafter` | model | `flint-staircase-drafter-9cards-v2.0.gguf` (R2-9) and `flint-staircase-drafter-v2.0.gguf` (R2-gen), ~0.27 GB each, `README.md`, `LICENSE` (OWNER-TBD), `CHECKSUMS` | [flint-staircase-drafter/](flint-staircase-drafter/) |
| `CinderellaMan/flint-drafter-data` (optional) | dataset | `manifest.json`; OWNER-TBD: `prompts/*.jsonl` (~0.3 GB), `answers/gen-*.jsonl` (~0.6 GB) | [flint-drafter-data/](flint-drafter-data/) |

The engine archives go to the fork's GitHub release, not to Hugging Face ([reproduce/engine.md](../reproduce/engine.md)).

## Before uploading K2h-r16

1. Rewrite the header, verifying the tensors on the way:
   `scripts/gguf_release_header.py flint-q113-K2h-r16.gguf flint-K2h-r16.gguf --preset flint --name "Flint K2h-r16" --quantized-by "OWNER-TBD"`.
2. Check the tensors against the manifest:
   `release/flint-v2/reproduce/tensor_manifest.py hash flint-K2h-r16.gguf > t.tsv && release/flint-v2/reproduce/tensor_manifest.py compare release/flint-v2/reproduce/k2h-r16.tensors.tsv t.tsv`
   (0 differences).
3. `sha256sum flint-K2h-r16.gguf`: put the line into `flint-K2h-r16/CHECKSUMS` and into the folder's
   [CHECKSUMS](../CHECKSUMS).
4. Replace the card's `RIG-TBD` numbers with the measured values.
5. One load test of the final file on the rig: the chat command line of the card and one request.

## Before uploading the drafter

1. Export both heads, R2-9 and R2-gen (`export_gguf.py`, [reproduce/drafter.md](../reproduce/drafter.md), step 7),
   and run the checks of step 8 on each.
2. `sha256sum` into `flint-staircase-drafter/CHECKSUMS` and the folder's CHECKSUMS; the same files go into the engine
   release notes through `package_release.sh` (`DRAFTS="flint-staircase-drafter-9cards-v2.0.gguf=… flint-staircase-drafter-v2.0.gguf=…"`).
3. The card's card counts, acceptance and speed from today's rig run; the licence (OWNER-TBD).

## Before publishing the dataset

- The private manifest lists machine paths: copy it with those fields removed (model files by name and SHA-256,
  engines by commit).
- Decide what it holds (OWNER-TBD): the manifest alone, plus the prompt files, plus the answers. The answer files carry
  the prompts' token ids, so the prompt datasets' licences apply to them too.
- Drop the snapshot copies (`*.snap*.jsonl`) and keep one file per writer run.

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Flint-K2h-r16-GGUF flint-K2h-r16.gguf flint-K2h-r16.gguf
hf upload CinderellaMan/Flint-K2h-r16-GGUF release/flint-v2/huggingface/flint-K2h-r16 .
hf upload CinderellaMan/Flint-Staircase-Drafter flint-staircase-drafter-9cards-v2.0.gguf flint-staircase-drafter-9cards-v2.0.gguf
hf upload CinderellaMan/Flint-Staircase-Drafter flint-staircase-drafter-v2.0.gguf flint-staircase-drafter-v2.0.gguf
hf upload CinderellaMan/Flint-Staircase-Drafter release/flint-v2/huggingface/flint-staircase-drafter .
hf upload CinderellaMan/flint-drafter-data release/flint-v2/huggingface/flint-drafter-data . --repo-type dataset
```

A 29.4 GB file takes ~40–80 minutes at 50–100 Mbit/s. Hugging Face shows each file's SHA-256 on its page (the LFS
object ID), which must equal the line in CHECKSUMS; `hf cache verify` checks a download against it.
