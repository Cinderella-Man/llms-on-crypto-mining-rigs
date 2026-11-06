# Uploading Swift Qwen3.8-Flash-Next Pascal Q2 GGUF to Hugging Face

**Not uploaded for v3.0** (decided 2026-11-06): on the release build this file runs at the speed of UkisAI's own Q2_0
(20.75 against 20.40 tokens/s, 2.45 against 2.47 minutes a task at equal work), so Flash-Next's release uses UkisAI's
file as published. The card and this plan stay for a later file that earns an upload: which repository, which files,
and the checks before the upload. The account is [CinderellaMan](https://huggingface.co/CinderellaMan).

## Repository and files

`CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF` (model), from
[Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF/](Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF/) and the release copies:

| File | Bytes | SHA-256 | From |
|---|---:|---|---|
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf` | 10,999,371,968 | `55a0b1c9a2b170be46c544819e4b87486c5ec9324d8c3f66d96955f91aa03fdf` | the release copy with the release header (step 4; every tensor checked byte for byte) |
| `Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00002-of-00003.gguf` | 26,755,194,496 | `f20e58f28af29423a731897bc0b58abb7918b19d0628de614eae609e89604965` | the release copy |
| `README.md`, `CHECKSUMS` | | | this folder |
| `LICENSE` (Swift Open License v1.0), `LICENSE-QWEN` (Qwen Community License 1.0), `NOTICE` | | | UkisAI's, unchanged (step 3), in this folder |

- **Shard 3 (the n-gram table) is never uploaded:** users cut it from UkisAI's download (the card's step 4).
- **The licence metadata** stays the Swift Open License (`license: other`, `license_name: swift-open-license-1.0`), so
  its US$1M threshold keeps binding downstream users.

## Before uploading

1. **The gain on the same build.** The card's rows are filled: this file at 20.75 tokens/s (`t1base`) against UkisAI's
   Q2_0 at 20.40 (`t1swift`), 2.45 against 2.47 minutes a task at equal work (negligible), with agent-text KL
   divergence 0.2792 ± 0.0054 against 0.2866 ± 0.0056. No meaningful gain in speed or quality means no upload: the
   release page's UkisAI row carries the release.
2. **The owner's explicit go** for this file.
3. **The licence files,** unchanged from UkisAI's repository at the measured revision:

   ```bash
   hf download ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF LICENSE LICENSE-QWEN NOTICE --revision 99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d --local-dir release/flash-next-v3/huggingface/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF
   ```

4. **The header.** Shard 1 names the file and its maker and carries the change notice the Swift licence asks for
   (§4(b)); the licence keys stay UkisAI's. The `SHA-256` above is the file with this header, written by:

   ```bash
   scripts/gguf_release_header.py Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf out.gguf --name "Swift-Qwen3.8-Flash-Next-Pascal-Q2" --quantized-by "Kamil Skowron (@kamilskowron)" --set "general.description=Modified copy of UkisAI's Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF (revision 99bb8f7f), made by Kamil Skowron (@kamilskowron) for Miner Llama v3: 168 tensors requantized by llama-quantize from Swift-1.5's Q8_0 with the importance matrix imatrix-swift15-q8_0-agent-train.gguf (quantize.imatrix.* records UkisAI's matrix, used for the other tensors); the router converted from F32 to BF16 (exact); shard 3, the n-gram table, is UkisAI's own file, not included. Not affiliated with or endorsed by UkisAI. Swift Open License v1.0."
   ```

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
