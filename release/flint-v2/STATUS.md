# Release status (2026-10-25)

Working notes for finishing the v2.0 release; delete this file before the release is published. Line numbers are as
of the commit that adds this file.

## Done

- **This folder** covers what ships: the engine rc4 `2d53179aa`, K2h-r16, and Unsloth's Flint Q8_0 on nine cards at
  262k as the headline (linked, not re-uploaded). The drafter and the store are named once as "coming later (v2.1)";
  their drafts (cards, rebuild guide, pipeline, pre-upload checks) are parked in [../flint-v2.1](../flint-v2.1/README.md).
- **K2h-r16:** the public recipe rebuilt it from BF16 with 779 of 779 tensors equal; the header tool, tensor manifest,
  Apache-2.0 LICENSE and card are ready; the account is CinderellaMan; `--quantized-by "Kamil Skowron (@kamilskowron)"`.
- **Engine packaging:** deterministic (two runs identical); the CUDA runtime archive's SHA-256 is known; rc3 and rc1
  packages exist, rc4 is not packaged yet.
- **Pages:** README, Flint page and notes template lead with Q8_0; `scripts/release_variant.sh rc4` (= rc4 q163-q111
  nodrafter) strips the drafter text and keeps one "coming later" line per page. Links, variants and the privacy scan
  pass.

## Placeholders and what fills them

`RIG-TBD` = a measurement from the rig; `OWNER-TBD` = the owner's decision. Lines tagged `<!-- only:drafter -->` or
inside `begin:drafter` blocks need nothing: the rc4 variant removes them (README.md 36, 89; models/flint/README.md 52,
82, 161, 173–177, 257; release/notes-v2.0.md 24, 72, 74, 102).

| What fills it | Command | Placeholders (file:line) |
|---|---|---|
| Five-turn chat at 262k, Q8_0 and UD-Q4_K_XL, v2 | `scripts/server_bench.sh PROMPTS=chat` (Q8_0 with `-ts 6,6,5,5,6,5,5,6,6`) | README.md 20, 21, 62; models/flint/README.md 17, 78, 79; release/flint-v2/README.md 16, 17; results.md 29; release/notes-v2.0.md 12, 19, 20; huggingface/flint-K2h-r16/README.md 35 |
| Q8_0 served at 262k, plain and `--pipe-depth 3` | `bench/pipespec/rig_server.sh` with `CTX=262144 TS=6,6,5,5,6,5,5,6,6 DEPTHS='0 3' GUESS=cont` | README.md 22; models/flint/README.md 80, 81; release/flint-v2/README.md 18; results.md 28; release/notes-v2.0.md 21 |
| Q8_0 identity, pipelined = plain | `bench/pipespec/rig_run.sh` with `TS=6,6,5,5,6,5,5,6,6 DEPTHS='1 3' N=256` | results.md 26 |
| Q8_0 `llama-bench`, stock / v1 / v2 | `bench/release_matrix` with `MODELS=Q8_0=… BENCH_ARGS='… -ts 6,6,5,5,6,5,5,6,6'` | README.md 31; models/flint/README.md 83; release/flint-v2/README.md 73; results.md 27; release/notes-v2.0.md 22 |
| UD-Q4_K_XL and K2h-r16 `llama-bench`, nine and six cards, stock / v1 / v2 | `bench/release_matrix` (stock-clock block) | README.md 27 (comment), 32, 114 (comment), 123–126; models/flint/README.md 100; release/flint-v2/README.md 74, 75; results.md 36–38; release/notes-v2.0.md 23; huggingface/flint-K2h-r16/README.md 29, 73 |
| Prefill at 512 / 2,048 / 8,192 tokens | `llama-bench -p 512,2048,8192 -n 0 -r 3` (or the matrix) | README.md 33; results.md 40; release/notes-v2.0.md 25 |
| K2h-r16 / UD served at 262k, depth 0 / 3, and the tool request against chat | `bench/pipespec/rig_server.sh` with `CTX=262144 DEPTHS='0 3' GUESS=cont` | README.md 90; models/flint/README.md 61, 115–117, 251; release/flint-v2/README.md 76; results.md 39 |
| Prefill attention (Q163) off / on at 8k and 32k | `GGML_CUDA_FATTN_TILE_PASCAL=0/1 llama-batched-bench` | README.md 92; models/flint/README.md 193, 255; release/flint-v2/README.md 77; results.md 41; release/notes-v2.0.md 64 |
| Prompt cache (Q111) off / on | `bench/agent_cache/rig_replay.sh`, `rig_check.sh` | README.md 93; models/flint/README.md 62, 256; release/flint-v2/README.md 78; results.md 42; release/notes-v2.0.md 59 |
| Agent check (Q131), stock against v2 | `bench/agent_check` | release/flint-v2/README.md 79; results.md 43 |
| Decode-attention kernel gain (Q151) | `GGML_CUDA_FATTN_PASCAL=0/1 llama-bench` | README.md 86; models/flint/README.md 250 |
| Identity on the P106 | `GGML_CUDA_PERSISTENT_MOE_CHECK=1 llama-perplexity … -ub 1`; `bench/pipespec/check_identity.sh` | results.md 35 |
| Quick-start check and `-b 2048 -ub 512` at 262k | `SERVE=1 scripts/p106_check.sh` | models/flint/README.md 55; results.md 44 |
| Streams and scheduler copies (optional) | `bench/pipespec/rig_hostloop.sh`; `llama-bench` with `SCHED_COPIES=4/8` | results.md 45, 46 |
| Memory clock offset (not in this release) | `bench/release_matrix` with `BLOCKS=tuned`, or delete the rows | README.md 128–149; models/flint/README.md 197; results.md 108; release/notes-v2.0.md 27, 28 |
| "Not measured yet" item | delete once the run is in | models/flint/README.md 262 |
| rc4 engine package hashes | `scripts/package_release.sh` (below) | release/flint-v2/README.md 101, 133, 135 |
| K2h-r16 SHA-256 after the header rewrite | `sha256sum flint-K2h-r16.gguf` | release/flint-v2/README.md 102, 136; huggingface/flint-K2h-r16/CHECKSUMS 1 |
| Fork push, release tag and links (OWNER-TBD) | the owner | README.md 45, 171, 176–178, 182; models/flint/README.md 25–27, 39; release/flint-v2/README.md 33, 115; reproduce/engine.md 12; huggingface/flint-K2h-r16/README.md 44 |

If the v2 numbers put Q8_0 below UD-Q4_K_XL or K2h-r16, reword the "faster than 4-bit" sentences (release/flint-v2
README 7–10, README.md 16–17, release/notes-v2.0.md 10–13, the K2h card's "not slower" at 32–35) to "near-lossless at
the full context, at about the 4-bit speed".

## Steps left to publish

1. **Fork (owner):** push branch `pascal-p106-v2-rc4` at `2d53179aa` and tag the release; fill the tag and links above.
2. **Engine archives:** build rc4 with `scripts/build_llamacpp_docker.sh` (or reuse the checked rc4 tarball), then
   `BUILD_DIR=<unpacked rc4> SRC=<llama.cpp tree> OUT=<dir> NOTES_VARIANT=rc4 scripts/package_release.sh - v2.0 2d53179aa`;
   once the numbers are in the template, `NOTES_ONLY=1` with the same `OUT` rewrites `RELEASE_NOTES.md`. Put the three
   archives, `SHA256SUMS` and `RELEASE_NOTES.md` on the fork's GitHub release (owner).
3. **K2h-r16 (owner uploads):** the steps in [huggingface/UPLOAD.md](huggingface/UPLOAD.md): header rewrite, manifest
   compare (0 differences), SHA-256 into both CHECKSUMS, card numbers, one load test on the rig, `hf upload`.
4. **Pages:** fill the placeholders, then `scripts/release_variant.sh rc4 README.md models/flint/README.md` in place;
   check links and privacy.
5. **Public repo (owner approves):** check `origin/v2..v2` for other sessions' unreviewed commits, push `v2`, then merge
   `v2` into `main` for the release. Delete this file first.
