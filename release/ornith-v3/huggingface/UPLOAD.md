# Uploading Ornith-1.5-122B Pascal N2P GGUF to Hugging Face

Nothing here has been uploaded. This page is the plan: which repository, which files, and the checks before the
upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan); the file is under MIT.

## Repository and files

`CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF` (model), from
[Ornith-1.5-122B-Pascal-N2P-GGUF/](Ornith-1.5-122B-Pascal-N2P-GGUF/) and the release copy:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 48,196,722,528 | `ef28114d83df37a534d3b1667ba766de2c808a29d630c11318e75e0e69bbe76a` | the release copy (importance-matrix paths cut to file names, every tensor checked byte for byte) |
| `README.md`, `CHECKSUMS`, `LICENSE` (MIT, with the ornith-ai credit) | | | this folder |
| `LICENSE-QWEN3.5` (Apache-2.0, Ornith's base) | | | Qwen's `LICENSE`, unchanged, fetched in step 3 |

- **Unsloth's UD-Q2_K_XL and importance matrix are linked, never re-uploaded:**
  [unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF).

## Before uploading

1. **The gain on the same build.** Compare the card's preset row (14.9 tokens/s) with `ornithv3ud` (Miner Llama v3 on
   UD-Q2_K_XL), together with the KL divergence advantage, and fill the card's `TODO(ornithud)` and `TODO(stock)`
   rows. No meaningful gain in speed or quality means no upload: UD-Q2_K_XL carries the release.
2. **The owner's explicit go** for this file.
3. **The base's licence,** unchanged from Qwen's repository:

   ```bash
   hf download Qwen/Qwen3.5-122B-A10B LICENSE --revision dc4d348443bc740c68e2d77492492c11606384d5 --local-dir qwen35 && mv qwen35/LICENSE release/ornith-v3/huggingface/Ornith-1.5-122B-Pascal-N2P-GGUF/LICENSE-QWEN3.5
   ```

4. **The header, advisable.** The copy's `general.name` is inherited from ornith-ai's Q8_0
   (`Qwen3.5-122B-A10B-ck2100-v1014-MTP1740`) and it has no `general.license`:

   ```bash
   scripts/gguf_release_header.py Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf ornith-pascal-n2p.gguf --name "Ornith-1.5-122B-Pascal-N2P" --quantized-by "Kamil Skowron (@kamilskowron)" --set general.license=mit
   ```

   Then rename the result back to `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` (the preset's name), hash it again
   and replace the line in `CHECKSUMS`.
5. **Check the file:** `sha256sum -c CHECKSUMS` in its folder.
6. **The release page** ([../README.md](../README.md)): fill its `TODO(upload)` slots (the file row, the checksum).

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf
hf upload CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF release/ornith-v3/huggingface/Ornith-1.5-122B-Pascal-N2P-GGUF .
```

48.2 GB takes est. 64 minutes at 100 Mbit/s. Hugging Face shows the file's SHA-256 (the LFS object ID), which must
equal the line in `CHECKSUMS`.
