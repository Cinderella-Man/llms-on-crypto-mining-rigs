# Uploading the release's big file to Hugging Face

Nothing here has been uploaded. This page is the plan: which repository, which files, and the checks before the
upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan); the file is under Apache-2.0 and signed
by "Kamil Skowron (@kamilskowron)".

## Repository

| Repository | Kind | Files | Folder here |
|---|---|---|---|
| `CinderellaMan/Flint-K2h-r16-GGUF` | model | `flint-K2h-r16.gguf` (29.4 GB), `README.md`, `LICENSE`, `CHECKSUMS` | [flint-K2h-r16/](flint-K2h-r16/) |

- The engine archives go to the fork's GitHub release, not to Hugging Face ([reproduce/engine.md](../reproduce/engine.md)).
- **Unsloth's Flint files are linked, never re-uploaded:** the headline Q8_0 and UD-Q4_K_XL stay at
  [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF).
- **Integrity:** SHA-256 in `CHECKSUMS` here on GitHub and in the Hugging Face repository; no GPG signatures.
- **Coming later (v2.1):** the trained drafter and the store of Flint's earlier answers
  ([docs/drafter.md](../../../docs/drafter.md)).

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

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Flint-K2h-r16-GGUF flint-K2h-r16.gguf flint-K2h-r16.gguf
hf upload CinderellaMan/Flint-K2h-r16-GGUF release/flint-v2/huggingface/flint-K2h-r16 .
```

A 29.4 GB file takes ~40–80 minutes at 50–100 Mbit/s. Hugging Face shows each file's SHA-256 on its page (the LFS
object ID), which must equal the line in CHECKSUMS; `hf cache verify` checks a download against it.
