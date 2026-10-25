# Uploading the release's big files to Hugging Face

Nothing here has been uploaded. This page is the plan: which repositories, which files, in what order, and the checks
before each upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan); everything is under
Apache-2.0 and signed by "Kamil Skowron (@kamilskowron)". `COUNT-TBD` marks the ShareGPT counts still being made.

## Repositories

| Repository | Kind | Files | Folder here |
|---|---|---|---|
| `CinderellaMan/Flint-K2h-r16-GGUF` | model | `flint-K2h-r16.gguf` (29.4 GB), `README.md`, `LICENSE`, `CHECKSUMS` | [flint-K2h-r16/](flint-K2h-r16/) |
| `CinderellaMan/Flint-Staircase-Drafter` | model | `flint-staircase-drafter-9cards-v2.0.gguf` (R2-9) and `flint-staircase-drafter-v2.0.gguf` (R2-gen), ~0.27 GB each, `README.md`, `LICENSE`, `CHECKSUMS` | [flint-staircase-drafter/](flint-staircase-drafter/) |
| `CinderellaMan/flint-drafter-data` | dataset | `prompts/*.jsonl` and `answers/gen-*.jsonl` without the ShareGPT rows (under 0.3 and 0.6 GB), `manifest.json`, `vocab.npy`, `heads/{U,R1,R2-9,R2-gen}/` (`head.pt`, `ckpt.pt`, `config.json`, `log.jsonl`; ~12 GB), `README.md`, `LICENSE`, `CHECKSUMS` | [flint-drafter-data/](flint-drafter-data/) |

- The engine archives go to the fork's GitHub release, not to Hugging Face ([reproduce/engine.md](../reproduce/engine.md)).
- **Unsloth's Flint files are linked, never re-uploaded:** the headline Q8_0 and UD-Q4_K_XL stay at
  [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF).
- **Integrity:** SHA-256 in `CHECKSUMS` here on GitHub and in each Hugging Face repository; no GPG signatures.

## Before uploading K2h-r16

1. Rewrite the header, verifying the tensors on the way:
   `scripts/gguf_release_header.py flint-q113-K2h-r16.gguf flint-K2h-r16.gguf --preset flint --name "Flint K2h-r16" --quantized-by "Kamil Skowron (@kamilskowron)"`.
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
3. The card's card counts, acceptance and speed from today's rig run.

## Before publishing the dataset

- **Leave out every ShareGPT row:** drop the prompts whose `source` is `Aeala/ShareGPT_Vicuna_unfiltered` from the
  prompt files, and every answer to those prompt numbers from the answer files; put the counts (prompts, answers and
  answer tokens, per writer) into the card's `COUNT-TBD` places.
- The private manifest lists machine paths: copy it with those fields removed (model files by name and SHA-256,
  engines by commit), and with the ShareGPT answers taken out of its counts.
- Each head's `config.json` records the training arguments with machine paths (`feats`, `out`, `vocab`, `flint`,
  `init_from`): replace them with file names. `head.pt` and `ckpt.pt` hold only tensors and counters.
- Drop the snapshot copies (`*.snap*.jsonl`) and keep one file per writer run.
- Search every text file for machine names and paths before the upload.
- Write `CHECKSUMS` over every file: `find . -type f ! -name CHECKSUMS ! -name README.md ! -name LICENSE -print0 | sort -z | xargs -0 sha256sum > CHECKSUMS`.
- Packed training sets and hidden-state dumps are not published (the card says how to recompute them).

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Flint-K2h-r16-GGUF flint-K2h-r16.gguf flint-K2h-r16.gguf
hf upload CinderellaMan/Flint-K2h-r16-GGUF release/flint-v2/huggingface/flint-K2h-r16 .
hf upload CinderellaMan/Flint-Staircase-Drafter flint-staircase-drafter-9cards-v2.0.gguf flint-staircase-drafter-9cards-v2.0.gguf
hf upload CinderellaMan/Flint-Staircase-Drafter flint-staircase-drafter-v2.0.gguf flint-staircase-drafter-v2.0.gguf
hf upload CinderellaMan/Flint-Staircase-Drafter release/flint-v2/huggingface/flint-staircase-drafter .
cp release/flint-v2/huggingface/flint-drafter-data/{README.md,LICENSE} <data folder>/   # then write CHECKSUMS there
hf upload CinderellaMan/flint-drafter-data <data folder> . --repo-type dataset
```

`<data folder>` holds the dataset as its card lists it: `prompts/`, `answers/`, `heads/`, `manifest.json`, `vocab.npy`.

A 29.4 GB file takes ~40–80 minutes at 50–100 Mbit/s. Hugging Face shows each file's SHA-256 on its page (the LFS
object ID), which must equal the line in CHECKSUMS; `hf cache verify` checks a download against it.
