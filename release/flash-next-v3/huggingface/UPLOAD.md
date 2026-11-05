# Uploading Swift Qwen3.8-Flash-Next Pascal Q2 GGUF to Hugging Face

Nothing here has been uploaded. This page is the plan: which repository, which files, and the checks before the
upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan).

## Repository and files

`CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF` (model), from
[Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF/](Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF/) and the release copies:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf` | 10,999,371,328 | `7d4b6cedfdbbefe3d016a0017f0874239f359589d44c2e4d0f211f5299004ce9` | the release copy (importance-matrix paths cut to file names, every tensor checked byte for byte) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf` | 26,755,194,496 | `f20e58f28af29423a731897bc0b58abb7918b19d0628de614eae609e89604965` | the release copy |
| `README.md`, `CHECKSUMS` | | | this folder |
| `LICENSE` (Swift Open License v1.0), `LICENSE-QWEN` (Qwen Community License 1.0), `NOTICE` | | | UkisAI's, unchanged, fetched in step 3 |

- **Shard 3 (the n-gram table) is never uploaded:** users cut it from UkisAI's download (the card's step 4).
- **The licence metadata** stays the Swift Open License (`license: other`, `license_name: swift-open-license-1.0`), so
  its US$1M threshold keeps binding downstream users.

## Before uploading

1. **The gain on the same build.** Compare the card's `t1base` row (20.75 tokens/s) with `t1swift` (Miner Llama v3 on
   UkisAI's Q2_0), and fill the card's `TODO(t1swift)` and `TODO(stock)` rows. No meaningful gain in speed or quality
   means no upload: the release page's UkisAI row carries the release.
2. **The owner's explicit go** for this file.
3. **The licence files,** unchanged from UkisAI's repository at the measured revision:

   ```bash
   hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF LICENSE LICENSE-QWEN NOTICE --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --local-dir release/flash-next-v3/huggingface/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF
   ```

4. **The header, the owner's call.** Shard 1's metadata is UkisAI's (`gguf_compose.py` keeps the base's header).
   Writing the change notice and the quantizer into it is advisable under the Swift licence's §4(b); the header keys
   as copied are in [2026-11-05_night_quant_chain](../../../docs/data/2026-11-05_night_quant_chain/README.md). Any
   rewrite (`scripts/gguf_release_header.py … --quantized-by "Kamil Skowron (@kamilskowron)" --set …`) changes
   shard 1: hash it again and replace its line in `CHECKSUMS`.
5. **Check the copies:** `sha256sum -c CHECKSUMS` in the copies' folder.
6. **The release page** ([../README.md](../README.md)): fill its `TODO(upload)` slots (the file row, the download
   command, the sizes) and add the two lines to its [CHECKSUMS](../CHECKSUMS).

## Commands (once the above is done)

```bash
hf auth login
hf upload CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF <copies>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf
hf upload CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF <copies>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf
hf upload CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF release/flash-next-v3/huggingface/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF .
```

37.8 GB takes est. 50 minutes at 100 Mbit/s. Hugging Face shows each file's SHA-256 (the LFS object ID), which must
equal its line in `CHECKSUMS`.
