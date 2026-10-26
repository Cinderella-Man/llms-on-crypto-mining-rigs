# Flint v2 release: everything in one place

The second release of the Pascal engine, tuned and measured on Flint: what ships, what changed since v1, what was
measured, where every file lives, and how to rebuild the engine and the model file from scratch. Placeholders:
`RIG-TBD` is a rig number still being measured, `OWNER-TBD` a decision of the project's owner.

## The headline: near-lossless Flint on nine mining cards at the full context, three times stock speed for an agent

Flint in 8-bit (Unsloth's Q8_0, 42.3 GB) runs on all nine P106-090 cards of a retired mining rig with the full
262,144-token context. It is near-lossless: it picks the original model's next token 93.3 % of the time (KL divergence
0.052 against the BF16 weights), against 87.7 % and 0.120 for Unsloth's UD-Q4_K_XL. For a coding agent it runs about
three times as fast as stock llama.cpp and within about 15 % of the 4-bit files. The fastest file on these cards is the
project's own Flint-Pascal-Q5 (recipe K2h-r16, 29.4 GB), closer to the original than UD-Q4_K_XL (88.8 %, KL divergence
0.105) and small enough for six cards.

A coding agent ([bench/agent_check](../../bench/agent_check/README.md): the pi agent fixing twelve small bugs, Python
and JavaScript, real test runs, the context growing every turn) on nine P106-090 cards. Tasks passed, generation in
tokens per second over every request, and the median minutes per task:

| Nine P106-090 cards, a coding agent at work | Stock llama.cpp (65,536-token context) | **v2** (262,144-token context) |
|---|---|---|
| **Unsloth Q8_0** (42.3 GB, 93.3 % same top token) | 8 of 12, 14.4 tok/s, 2.9 min | **12 of 12, 44.6–45.9 tok/s, 1.1–1.4 min** (two runs) |
| **Flint-Pascal-Q5** (29.4 GB, 88.8 %) | RIG-TBD | **12 of 12, 53.8 tok/s, 1.2 min** |
| Unsloth UD-Q4_K_XL (25.1 GB, 87.7 %) | RIG-TBD | 10 of 12, 50.7 tok/s, 1.1 min |

Stock llama.cpp runs at 65,536 tokens because at 262,144 its attention mask, built on the host, does not fit the rig's
4 GB of RAM. Pipelined speculation alone is worth +52 % in this test (29.3 → 44.6 tok/s on Q8_0); the output does not
change. Why 8 bits come this close to 4 bits: on 5-SM Pascal cards much of a token's time goes into unpacking
compressed weights; the Pascal engine decodes the simple 8-bit format at close to the cards' memory speed, while the
elaborate 4-bit formats need more instructions per byte. So the file that reads 2.70 GB per token (Q8_0) stays within
about 15 % of the one that reads 2.21 GB (UD-Q4_K_XL).

## What ships

| Piece | What it is | Where |
|---|---|---|
| **Engine** | llama.cpp fork for compute 6.1, commit `37e717ff6` (rc4 with the release defaults): the Pascal kernels, the persistent MoE kernel, pipelined speculation with guesses from the request itself, GPU sampling for tool requests and the prefill-attention kernel; prebuilt for Ubuntu 24.04+ without AVX, with NVIDIA's CUDA runtime libraries as a second archive | the fork's GitHub release, OWNER-TBD (tag); notes: [release/notes-v2.0.md](../notes-v2.0.md) |
| **Unsloth's Flint Q8_0** | the headline file: near-lossless, 42.3 GB, nine cards at the full context | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (linked, not redistributed) |
| **Flint-Pascal-Q5** | our Flint file: Q5_0 and Q8_0 tensors, KLD 0.105 against BF16, 29.4 GB, fits six cards | Hugging Face [CinderellaMan/Flint-Pascal-Q5-GGUF](https://huggingface.co/CinderellaMan/Flint-Pascal-Q5-GGUF); card: [huggingface/Flint-Pascal-Q5](huggingface/Flint-Pascal-Q5/README.md) |
| **Unsloth's UD-Q4_K_XL** | the file most people already have, KLD 0.120 | [unsloth/flint-GGUF](https://huggingface.co/unsloth/flint-GGUF) (linked, not redistributed) |

**Coming later (v2.1):** a trained guesser for pipelined speculation (the staircase drafter) and a store of Flint's
earlier answers, rebuilt from sources whose answers may be published: [docs/drafter.md](../../docs/drafter.md).

Everything this project publishes is under Apache-2.0, like Flint itself, and is checked by SHA-256 (in
[CHECKSUMS](CHECKSUMS) and in the Hugging Face repository; no GPG signatures). Unsloth's files are linked, not
re-uploaded. The upload plan, with the checks before the upload: [huggingface/UPLOAD.md](huggingface/UPLOAD.md). Made
by Kamil Skowron ([@kamilskowron](https://x.com/kamilskowron) on X).

How to run each: the [Flint page](../../models/flint/README.md#command-lines).

## What changed since v1

v1 (2026-10-23, `eeaa59660`) doubled Flint's speed over stock llama.cpp with CUDA graphs on Pascal, a launch shape for
one-token products, and Pascal matrix-vector kernels. v2 adds, in the order they help a single conversation:
- **32-bit activation sums,** so llama.cpp's MoE rescale is dropped for one-token steps (+9.6 %), and the shared expert
  fused with its activation (+0.7 %);
- **one persistent kernel per MoE layer** (+5.6–8.8 %) and less host work per token;
- **pipelined speculation** in llama-server (`--pipe-depth 4` on nine cards): guessed tokens on the idle cards of the
  layer split, the guesses taken from the request itself (the copy lookup and its continuation counts), the output
  equal to plain decoding's;
- **tool requests sampled on the GPU** (they decoded 12–16 % slower than chat before);
- **prefill:** MoE tiles sized by tokens per expert (+22–29 %) and a prefill-attention kernel (+10 % at 8,192 tokens,
  +3 % at 32,768);
- **Pascal kernels for Q4_K and Q6_K,** for other models' common downloads.

Also in the build but off unless set, because they did not help the coding agent: a Pascal decode-attention kernel,
the input layer on the first card with packed uploads, the attention mask built on each card, and a prompt cache for
agents. Every piece has its own switch ([reproduce/engine.md](reproduce/engine.md)) and passed its exactness checks
before it joined: outputs equal the base build's, and pipelined output equals plain decoding's.

## What was measured

[results.md](results.md) lists every number behind the release with its command, build, date and registry row. The
release settings were chosen on the coding agent's speed (the headline above); the rows below explain where it comes
from. Nine P106-090 cards unless a row says otherwise:

| | Stock llama.cpp | v1 | **v2** |
|---|---|---|---|
| Coding agent, Q8_0: generation without pipelined speculation / `--pipe-depth` 2 / 3 / 4 (tok/s) | 14.4 (64k) | — | 29.3 / 40.1 / 44.6 / **44.6–45.9** |
| Coding agent, Flint-Pascal-Q5: `--pipe-depth` 3 / 4 | — | — | 48.6 / **53.8** |
| Coding agent, Q8_0 at depth 3 with the agent prompt cache (`--prefix-cache --checkpoint-gen`) | — | — | 42.2 against 44.6 without (11 s more per task): off by default |
| Prompt reading in llama-server, Q8_0: a fresh 13,400-token prompt; the next turn's first token | — | — | 485 tok/s; 5.8 s |
| Prefill, Flint-Pascal-Q5, the prefill-attention kernel on against off, at 8,192 / 32,768 tokens | — | — | +10 % / +3 % |
| Generation (`llama-bench`), UD-Q4_K_XL, nine / six cards | 16.5 / 15.2 | 32.6 / 31.2 | 39.9 / 38.2 (v2-next and the persistent build) |
| Generation (`llama-bench`), Flint-Pascal-Q5, nine / six cards | — | — | 41.8 / 40.7 (v2-next and the persistent build) |

The write-ups behind the work: [docs/research/notes](../../docs/research/notes/README.md) (the v2 synthesis, release
settings, pipelined speculation) and the registry [quants/experiments.md](../../quants/experiments.md) (every
experiment, Q1–Q163, and the decision log).

## Rebuild it yourself

| What | How | Needs |
|---|---|---|
| The engine | [reproduce/engine.md](reproduce/engine.md): `scripts/build_llamacpp_docker.sh`, `scripts/package_release.sh` | Docker on any x86-64 Linux PC |
| Flint-Pascal-Q5 | [reproduce/model.md](reproduce/model.md), `reproduce/build_k2h_r16.sh`: Flint BF16 → llama-quantize with the tensor-type map → GPTQ head → F16 routers → header; checked tensor by tensor against [k2h-r16.tensors.tsv](reproduce/k2h-r16.tensors.tsv) | CPU and ~150 GB of disk; GPUs only to compute the head yourself |

Big data is not in git. The model files download from Hugging Face; Flint's original weights and Unsloth's importance
matrix are pinned by revision and SHA-256 ([reproduce/inputs.sha256](reproduce/inputs.sha256)). The one small dataset
the checks need, the 38 held-out Flint transcripts (part of the output head's calibration data), is in
[reproduce/data/q51](reproduce/data/q51/).

## Files and checksums

| File | Size | Produced by | SHA-256 |
|---|---|---|---|
| Engine archives (binaries, CUDA runtime, tests) | 253 / 554 / 24 MB | `scripts/package_release.sh` on `37e717ff6` | in [CHECKSUMS](CHECKSUMS) and the release's `SHA256SUMS` |
| `flint-pascal-Q5.gguf` (release header) | 29.4 GB | `reproduce/build_k2h_r16.sh` | `0163ec541e88a863…` (in [CHECKSUMS](CHECKSUMS)) |
| `flint-q113-K2h-r16.gguf` (the same tensors, the build header) | 29.4 GB | the project's build | in [CHECKSUMS](CHECKSUMS) |
| `imatrix_unsloth.gguf` (input) | 225 MB | Unsloth | in [CHECKSUMS](CHECKSUMS) |
| `flint-UD-Q4_K_XL.gguf`, the copy every UD number here was measured with | 25.1 GB | Unsloth | in [CHECKSUMS](CHECKSUMS) |

[CHECKSUMS](CHECKSUMS) holds the hashes known now in `sha256sum -c` format; each file's line is added as it is
produced.

## Gaps

What a stranger cannot yet redo exactly from this repository, and what closes each gap:

1. **The engine's release commit is not public yet.** `37e717ff6` and the release branch wait for the owner's push to
   the fork (OWNER-TBD), and the release archives for the tag.
2. **The GPTQ output head is byte-exact only by copying it.** A head computed from scratch (dumps on GPUs, GPTQ on the
   CPU) is equivalent within noise, not identical: GPU dumps differ in the last bits between GPUs. The calibration
   data is public (wikitext-2 train and the shipped Q51 transcripts).
3. **Rig drivers are private.** The scripts that relayed builds to the rig and looped the public bench scripts contain
   machine names; `results.md` gives each number's public command instead.
4. **Reference data that is large:** the BF16 logits for the KL-divergence checks (`flint-BF16.c512x80.kld`) are
   regenerated with the commands in [reproduce/model.md](reproduce/model.md).
5. **How the mix was found:** the per-group measurements that chose Flint-Pascal-Q5's tensor types (Q73, Q77) ran as private job
   files; their results are in the registry, and the type map, not the search, defines the file.

## Acceptance check: everything needed to rebuild every shipped file

Each shipped file, the guide and script that rebuild it, the checksum a rebuild must match, and what still stands in the
way:

| Shipped file | Rebuild guide | Script | Expected checksum | Status |
|---|---|---|---|---|
| `llama-p106-v2.0-bin-ubuntu-cuda12.9-sm61.tar.gz` (engine) | [reproduce/engine.md](reproduce/engine.md) | `scripts/build_llamacpp_docker.sh`, then `scripts/package_release.sh` | `f213259e52f11a2893be46e69dc1683bc8b98296f07cfb476ff4c19a681baec1` | **gap 1:** the release commit is not public yet. Packaging is deterministic: the same binaries give the same archive (two runs on rc3's build, identical bytes). A fresh compile is checked by `BUILDINFO.txt`'s commit and `test-backend-ops`, not by its bytes |
| `cudart-llama-p106-v2.0-bin-ubuntu-cuda12.9.tar.gz` (CUDA runtime) | [reproduce/engine.md](reproduce/engine.md) | `scripts/package_release.sh` (from NVIDIA's `cuda:12.9.2-devel-ubuntu24.04` image) | `65b46821ab94c5a46ec762f4fc8896de857ada120770d5a2499695ed3277a3bf` (the same for any engine commit packaged as v2.0) | complete: byte for byte from the same image (two runs identical) |
| `llama-p106-v2.0-tests-ubuntu-cuda12.9-sm61.tar.gz` | [reproduce/engine.md](reproduce/engine.md) | as the engine | `26e0f0633e4ed9967cfbbbf8a2b45bec2c79ae80f526a8ab4afde920f72665fe` | as the engine |
| `flint-pascal-Q5.gguf` | [reproduce/model.md](reproduce/model.md) | `reproduce/build_k2h_r16.sh` | `0163ec541e88a863…` (release header, [CHECKSUMS](CHECKSUMS)); every tensor in [k2h-r16.tensors.tsv](reproduce/k2h-r16.tensors.tsv) | verified: the script rebuilt it from BF16 with all 779 tensors equal byte for byte (2026-10-25, head copied); **gap 2:** the output head is byte-exact only when copied |
| Release notes | [notes-v2.0.md](../notes-v2.0.md) | `scripts/package_release.sh NOTES_VARIANT=rc4` | — | complete once the numbers are in |
| The Hugging Face card of Flint-Pascal-Q5 | [huggingface/](huggingface/UPLOAD.md) | — | — | complete once the numbers are in |
| The Q51 test set (inputs to the checks) | [reproduce/data/q51](reproduce/data/q51/README.md) | `bench/expert_overlap` | in git | complete |

## Not in this release

Qwen3.6-35B-A3B, Ornith-1.5-35B and Ornith-1.5-122B ([models/](../../models/)), the memory clock offset (hardware
tuning, measured separately), the screened output head's side file, and several conversations through one pipeline in
llama-server.
