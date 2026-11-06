# Uploading Ornith-1.5-122B Pascal N2P GGUF to Hugging Face

The model file is on Hugging Face since 2026-11-06, its SHA-256 there equal to its line in `CHECKSUMS`; the card
files of this folder go up at the release, once their links resolve on `main`. This page is the plan: which repository,
which files, and the checks before each upload. The account is
[CinderellaMan](https://huggingface.co/CinderellaMan); the file is under MIT.

## Repository and files

`CinderellaMan/Ornith-1.5-122B-Pascal-N2P-GGUF` (model), from
[Ornith-1.5-122B-Pascal-N2P-GGUF/](Ornith-1.5-122B-Pascal-N2P-GGUF/) and the release copy:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` | 48,196,722,624 | `0bfa0cf25b93de4370e6634d33b04942ad6e244acf41e9ac1764dd27162f44e1` | the release copy with the release header (step 4; importance-matrix paths cut to file names, every tensor checked byte for byte) |
| `README.md`, `CHECKSUMS`, `LICENSE` (MIT, with the ornith-ai credit) | | | this folder |
| `LICENSE-QWEN3.5` (Apache-2.0, Ornith's base) | | | Qwen's `LICENSE`, unchanged (step 3), in this folder |

- **Unsloth's UD-Q2_K_XL and importance matrix are linked, never re-uploaded:**
  [unsloth/Ornith-1.5-122B-GGUF](https://huggingface.co/unsloth/Ornith-1.5-122B-GGUF).

## Before uploading

1. **The gain on the same build.** Compare the card's preset row (14.9 tokens/s) with `ornithv3ud` (Miner Llama v3 on
   UD-Q2_K_XL), together with the KL divergence advantage. The card's rows: v3 on UD-Q2_K_XL 12.6 tokens/s, 12 of 12;
   stock on it 6.0, 11 of 12 (2026-11-06). No meaningful gain in speed or quality means no upload: UD-Q2_K_XL carries the release.
2. **The owner's explicit go** for this file.
3. **The base's licence,** unchanged from Qwen's repository:

   ```bash
   hf download Qwen/Qwen3.5-122B-A10B LICENSE --revision dc4d348443bc740c68e2d77492492c11606384d5 --local-dir qwen35 && mv qwen35/LICENSE release/ornith-v3/huggingface/Ornith-1.5-122B-Pascal-N2P-GGUF/LICENSE-QWEN3.5
   ```

4. **The header.** The file carries its own `general.name` (in place of ornith-ai's Q8_0 name
   `Qwen3.5-122B-A10B-ck2100-v1014-MTP1740`), `general.quantized_by` and `general.license`; the `SHA-256` above is the file
   with this header, written by:

   ```bash
   scripts/gguf_release_header.py Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf ornith-pascal-n2p.gguf --name "Ornith-1.5-122B-Pascal-N2P" --quantized-by "Kamil Skowron (@kamilskowron)" --set general.license=mit
   ```

   and renamed back to `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf` (the preset's name).
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
