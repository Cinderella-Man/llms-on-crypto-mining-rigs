# v3.0: what is left to fill in before the release

Every slot in the v3 drafts that a run of 2026-11-05 evening or 2026-11-06 morning fills is marked `TODO(<name>)`.
Find them all with `grep -rn 'TODO(' README.md release docs/switches.md docs/book`. Line numbers are as drafted; they
move as slots are filled. Delete this file when every box is ticked.

## Slots

| | Slot | Filled by | Where |
|---|---|---|---|
| [ ] | `TODO(stock)` | stock llama.cpp b12076 (`20a51f24a`) on the agent check: `stockq2` and `stockpmix` (Flash-Next, rig 2), `flintstock` (Flint Q8_0, rig 1), `ornithstock` (Ornith UD-Q2_K_XL, rig 1); tasks, generation and prompt tok/s, first token per turn, the data folder; then the speed-up of each preset over its stock row | `README.md:16`; `release/notes-v3.0.md:22`, `:23`, `:24`, `:44`; `release/flash-next-v3/README.md:20`, `:21`; `release/flint-v3/README.md:16`; `release/ornith-v3/README.md:15`; `docs/book/10-v3.md:129` |
| [ ] | `TODO(preset)` | T1's agent check of Flash-Next's preset on the release build, and its `release/flash-next-v3/settings.env` (then replace the inline block in the page with a link to the file) | `release/notes-v3.0.md:28`; `release/flash-next-v3/README.md:27`, `:102`; `docs/switches.md:47` |
| [ ] | `TODO(t1skip)` | T1's arm `t1skip`: the preset with the expert skip at p = 0.9; into the preset if it is a gain or negligible | `release/flash-next-v3/README.md:30`; `docs/switches.md:41` |
| [ ] | `TODO(t1neg)` | T1's arm `t1neg` (2026-11-06, rig 2): the preset with every switch of the negligible group | `release/flash-next-v3/README.md:31`; `docs/switches.md:42` |
| [ ] | `TODO(identity)` | T0c's greedy check, `1f7215948` against `1551f5fe1`, one per model (`docs/data/2026-11-05_ladder5/run/rig_greedy.sh`): SAME for all three carries the Flint and Ornith runs over to the release commit | `release/notes-v3.0.md:41` |
| [ ] | `TODO(kld)` | Flint's KL divergence against v2.0 (rig 1, 2026-11-05 evening), and Ornith's if it runs: mean KLD and same-top-token share; say whether the greedy drift is only near ties | `release/notes-v3.0.md:125`; `release/flint-v3/README.md:29`; `docs/book/10-v3.md:156` |
| [ ] | `TODO(flintside)` | rig 1, 2026-11-06: Flint one switch at a time against `v3base` (the keep-alive, the re-read fix, the staged hop, the answer store `store`, the negligible group); the winners into `release/flint-v3/settings.env` | `release/notes-v3.0.md:93`; `release/flint-v3/README.md:25`, `:61`; `docs/switches.md:37`, `:39`, `:40`, `:42`, `:44` |
| [ ] | `TODO(ornithka)` | T23's `agent_ornith_ka` (Ornith valid1 with `GGML_CUDA_KEEPALIVE=decode`, running 2026-11-05 evening) | `release/ornith-v3/README.md:19`, `:66`; `docs/switches.md:39` |
| [ ] | `TODO(ornithmtp)` | rig 1, 2026-11-06: Ornith's arms `mtp`, `mtpadapt`, `mtpka` (its own MTP layer as the draft); the best into `release/ornith-v3/settings.env` | `release/ornith-v3/README.md:20`, `:66`; `docs/switches.md:45` |
| [ ] | `TODO(upload)` | the owner's call: `pmix-q2-fast` shards 1–2 on Hugging Face after the check of Swift's licence and the importance matrix's text sources; valid1 the same, after rewriting its `quantize.imatrix.file` key; repository names, the download commands, sizes and SHA-256s (also into `release/flash-next-v3/CHECKSUMS`) | `release/flash-next-v3/README.md:39`, `:58`, `:170`; `release/ornith-v3/README.md:32` |
| [ ] | `TODO(tag)` | the owner's call: the fork tag on `1551f5fe1` and the GitHub release; the release date and the engine archives' sizes | `release/flash-next-v3/README.md:42`; `docs/book/09-the-models.md:240` |

## Before publishing

- [ ] T23 pushes `docs/data/2026-11-05_release_regression/` with its `docs/data/README.md` row: five pages link it.
- [ ] T0c's PTX check of `1551f5fe1` (quant machine, one card) passes, with the 30 runs of the formerly failing case.
- [ ] If a preset changes after its agent check, its numbers come from the run of the new preset, not the old one.
- [ ] `scripts/package_release.sh fork/pascal-p106-v3-b12076-ladder5 v3.0` (the quant machine) writes `RELEASE_NOTES.md`
      from `release/notes-v3.0.md`; after a later fill, `NOTES_ONLY=1` rewrites only the notes.
- [ ] `grep -rn 'TODO(' README.md release docs/switches.md docs/book` prints only this file.
- [ ] `v3` is merged into `main` (the notes link to `main`), then this file is deleted.
