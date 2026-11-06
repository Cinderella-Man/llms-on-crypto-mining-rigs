# Uploading Flint Pascal Q8 GGUF to Hugging Face

Nothing here has been uploaded. This page is the plan: which repository, which files, and the checks before the
upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan); the file is under Apache-2.0.

## Repository and files

`CinderellaMan/Flint-Pascal-Q8-GGUF` (model), from [Flint-Pascal-Q8-GGUF/](Flint-Pascal-Q8-GGUF/) and the release
copy:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `flint-pascal-Q8.gguf` | 40,431,236,704 | `15c501699c52d8c856cf4064035d9b3b40ae5088d3e17870c6e18c393ece4854` | the release copy `flint-near-q8-p065.gguf` with the release header (step 3; composed again from the recipe, the composed file's md5 equal to the file the rig ran, every tensor checked byte for byte, importance-matrix paths cut to file names) |
| `README.md`, `CHECKSUMS`, `LICENSE` (Apache-2.0) | | | this folder |

- **Unsloth's Q8_0 is linked, never re-uploaded:** it stays at [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF).

## Before uploading

1. **The gain on the same build.** The card's rows are filled: this file at 47.4 tokens/s on Miner Llama v3, 12 of 12,
   against Unsloth's Q8_0 at 45.9 on the same build (+3.3 %). No meaningful gain means no upload: Unsloth's Q8_0
   carries Flint's Q8 lane.
2. **The owner's explicit go** for this file.
3. **The header.** The file carries its own licence, name and quantizer in place of Unsloth's, as v2's Flint Pascal Q5
   GGUF does, and no `general.repo_url`; the `SHA-256` above is the file with this header, written by:

   ```bash
   scripts/gguf_release_header.py flint-near-q8-p065.gguf flint-pascal-Q8.gguf --preset flint --name "Flint-Pascal-Q8" --quantized-by "Kamil Skowron (@kamilskowron)" --delete general.repo_url
   ```

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
