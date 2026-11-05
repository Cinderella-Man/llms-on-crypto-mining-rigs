# Uploading Flint Pascal Q8 GGUF to Hugging Face

Nothing here has been uploaded. This page is the plan: which repository, which files, and the checks before the
upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan); the file is under Apache-2.0.

## Repository and files

`CinderellaMan/Flint-Pascal-Q8-GGUF` (model), from [Flint-Pascal-Q8-GGUF/](Flint-Pascal-Q8-GGUF/) and the release
copy:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `flint-pascal-Q8.gguf` | 40,431,236,512 | `da1c33ae6eac089c3d64146ce9e567422e80b78ee334abc9874e4203bab4c695` | the release copy `flint-near-q8-p065.gguf`, renamed (composed again from the recipe, md5 equal to the file the rig ran, importance-matrix paths cut to file names) |
| `README.md`, `CHECKSUMS`, `LICENSE` (Apache-2.0) | | | this folder |

- **Unsloth's Q8_0 is linked, never re-uploaded:** it stays at [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF).

## Before uploading

1. **The gain on the same build.** Compare the card's `TODO(p065)` row (this file on Miner Llama v3) with Unsloth's
   Q8_0 on the same build (45.9 tokens/s), and fill the card's `TODO(p065)` and `TODO(stock)` rows. No meaningful
   gain means no upload: Unsloth's Q8_0 carries Flint's Q8 lane.
2. **The owner's explicit go** for this file.
3. **The header, the owner's call.** The copy's `general.license`, `general.name` and `general.quantized_by` are
   Unsloth's. v2's Flint Pascal Q5 GGUF got its own:

   ```bash
   scripts/gguf_release_header.py flint-near-q8-p065.gguf flint-pascal-Q8.gguf --preset flint --name "Flint-Pascal-Q8" --quantized-by "Kamil Skowron (@kamilskowron)"
   ```

   This changes the file: hash it again and replace the line in `CHECKSUMS`. Without it, only rename the copy
   (`mv flint-near-q8-p065.gguf flint-pascal-Q8.gguf`); its SHA-256 stays the one above.
4. **Check the file:** `sha256sum -c CHECKSUMS` in its folder.
5. **The release page** ([../README.md](../README.md)): add the file to Flint's Q8 lane (`TODO(flintq5)` in
   [FILL-IN.md](../../FILL-IN.md)).

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Flint-Pascal-Q8-GGUF flint-pascal-Q8.gguf flint-pascal-Q8.gguf
hf upload CinderellaMan/Flint-Pascal-Q8-GGUF release/flint-v3/huggingface/Flint-Pascal-Q8-GGUF .
```

40.4 GB takes est. 54 minutes at 100 Mbit/s. Hugging Face shows the file's SHA-256 (the LFS object ID), which must
equal the line in `CHECKSUMS`.
