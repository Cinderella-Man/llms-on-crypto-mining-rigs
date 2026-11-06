# v3.0: what is left to fill in before the release

Every slot in the v3 drafts that a run of 2026-11-05 evening or 2026-11-06 morning fills is marked `TODO(<name>)`.
Find them all with `grep -rn 'TODO(' README.md release docs/switches.md docs/book`. Line numbers are as drafted; they
move as slots are filled. Delete this file when every box is ticked.

## Slots

| | Slot | Filled by | Where |
|---|---|---|---|
| [x] | `TODO(stock)` | stock llama.cpp b12076 (`20a51f24a`) on the agent check, then the speed-up of each preset over its stock row. **Filled 2026-11-06 (release-fill) for Flash-Next** (`stockq2`, 8.20 tok/s on UkisAI's Q2_0, the release file) **and Flint** (`flintstock`, 14.4, 10/12, 65,536 tokens). Ornith's `ornithstock` (rig 1, 2026-11-06 12:00): stock tuned (`-b 2048 -ub 256 -fa on`) on Unsloth UD-Q2_K_XL, 11/12, 6.0 tok/s, prompts 55, first token 12.57 s median (v3 on the same file 12.6); in the Ornith page, its ladder, the card, the notes, README and book ch. 10 | — |
| [x] | `TODO(preset)` | the owner, 2026-11-06 ~11:35: no Swift upload, so Flash-Next's preset is `t1swift`'s line on UkisAI's Q2_0 (20.40 tok/s, 12/12, on `1551f5fe1`); `release/flash-next-v3/settings.env` in the `PRESET_*` form of Flint's and Ornith's; the page, notes, README, switch list and book lead with 20.40, and my mix's 20.75 (`t1base`) is stated as negligible against it | — |
| [x] | `TODO(t1skip)` | `t1skip`: negligible (−2.9 % a task, 11/12), out of the preset | — |
| [x] | `TODO(t1neg)` | `t1neg`: a loss (19.05 tok/s, +5.3 % a task), out of the preset | — |
| [ ] | `TODO(identity)` | the greedy check, `1f7215948` against `1551f5fe1`, one per model. **Filled for Flint** (8/8 requests the same, release regression) **and Flash-Next** (both prompts the same, T1; its release row ran on `1551f5fe1` itself). Left: Ornith valid1's pair on the rig (`docs/data/2026-11-05_release_regression/run/greedy_identity.sh` with both builds), queued on rig 1 after the Flint 5-bit rows (QUEUES T3f) | `release/notes-v3.0.md:51` |
| [x] | `TODO(kld)` | Flint's KL divergence against v2.0 (−0.000000, max 0.000056, same top 100.000 %, PPL 24.4293 on both) in the notes' known issues, the Flint page and the book; Ornith's P106 KLD (T3a) beside it, with T3e's verdict as `TODO(t3e)` | — |
| [x] | `TODO(t3e)` | T3e (rig 1, 2026-11-06 12:40): against the Ornith-1.5-122B Q8_0 reference on the P106 (wikitext-2, 80 × 512), `d7a75deb3` 0.175864 ± 0.002767 and `1551f5fe1` 0.175182 ± 0.002745; paired d = −0.00068 ± 0.00069, within noise: neither build is closer to the model; in the notes, the Ornith page and the book | — |
| [ ] | `TODO(flintside)` | Flint one switch at a time against `v3base` (the keep-alive, the re-read fix, the staged hop, the answer store `store`, the negligible group), moved to rig 2 (flint-side-rig2). **Filled for the re-read fix** (negligible: 2.59 against 2.60 minutes a task at equal work, 2026-11-06, [data](../docs/data/2026-11-06_flint_side_rig2/README.md)); the staged hop, the negligible group, the answer store and the keep-alive were not run (stopped after `reread`; QUEUES T2d); the winners into `release/flint-v3/settings.env` | `release/notes-v3.0.md:107`; `release/flint-v3/README.md:30`, `:81`, `:109`, `:121`; `docs/switches.md:37`, `:39`, `:40`, `:42`, `:44` |
| [x] | `TODO(ornithka)` | `valid1-ka`: 16.3 tok/s, 12/12, a gain on generation; in Ornith's preset since 2026-11-06 (the owner: "Add it now"), on its page, the notes, README, book, card and switch table | — |
| [ ] | `TODO(ornithmtp)` | rig 1: Ornith's arms `mtp`, `mtpadapt`, `mtpka` (its own MTP layer as the draft); the best, with the keep-alive (a gain, `valid1-ka`), into `release/ornith-v3/settings.env` | `release/ornith-v3/README.md:20`, `:76`, `:99`, `:113`; `docs/switches.md:39`, `:45` |
| [x] | `TODO(ornithud)` | `ornithv3ud` (rig 1, 2026-11-06 10:53): 12/12, 12.6 tok/s, prompts 69, first token 7.90 s median; in the Ornith page's table and ladder and the card, with its row in `docs/data/2026-11-05_release_regression/` | — |
| [x] | `TODO(t1swift)` | rig 2, 2026-11-06: `t1swift`, 12/12, 20.40 tok/s: Flash-Next's release row | — |
| [x] | `TODO(p065)` | Flint Pascal Q8 GGUF on the release build, 12/12, 47.4 tok/s: in the card, the Flint page's 8-bit lane, the notes and the book (upload decided by the owner, 2026-11-06); its file row waits on `TODO(upload)`, stock on it on `TODO(flintq5)` | — |
| [ ] | `TODO(flintq5)` | below: stock + Flint Pascal Q8 GGUF (T2f) and the 5-bit lane (T2e, T2g) | `release/notes-v3.0.md:24`; `release/flint-v3/README.md:46` |
| [x] | `TODO(upload)` | decided 2026-11-06 ~11:35 (the owner: "Ornith + Flint Q8"); the model files are on Hugging Face since 11:52 (Ornith-1.5-122B Pascal N2P GGUF) and 11:59 (Flint Pascal Q8 GGUF), each file's SHA-256 there equal to its `CHECKSUMS` line; the two cards go up at the release, once their links resolve on `main`; Swift Qwen3.8-Flash-Next Pascal Q2 GGUF is not (its `UPLOAD.md` says so). When each upload ends: the repository link, the download command and the checksum in its file row | `release/ornith-v3/README.md:41`; `release/flint-v3/README.md:53` |
| [ ] | `TODO(tag)` | the owner's call: the fork tag on `1551f5fe1` and the GitHub release; the release date and the engine archives' sizes | `release/flash-next-v3/README.md:50`; `docs/book/09-the-models.md:240` |

## Before publishing

- [x] T23 pushes `docs/data/2026-11-05_release_regression/` with its `docs/data/README.md` row: five pages link it.
- [x] T0c's PTX check of `1551f5fe1` (quant machine, one card): every part OK, experts 4,759 of 4,759, the formerly failing case 30 of 30 ([2026-11-05_ladder5](../docs/data/2026-11-05_ladder5/README.md)).
- [x] T5b's expert-skip gate on Flint Q8_0 and Ornith valid1: above noise on all five tests, so the skip stays out of both presets ([2026-11-05_night_quant_chain](../docs/data/2026-11-05_night_quant_chain/README.md); `docs/switches.md` and both release pages say so).
- [x] T5c's clean copies of the four release files: sha256 equal the Hugging Face cards' `CHECKSUMS` (same folder); each upload still waits for the owner's go (`TODO(upload)`).
- [ ] If a preset changes after its agent check, its numbers come from the run of the new preset, not the old one.
- [ ] `scripts/package_release.sh fork/pascal-p106-v3-b12076-ladder5 v3.0` (the quant machine) writes `RELEASE_NOTES.md`
      from `release/notes-v3.0.md`; after a later fill, `NOTES_ONLY=1` rewrites only the notes.
- [ ] `grep -rn 'TODO(' README.md release docs/switches.md docs/book` prints only this file. The `UPLOAD.md` files name slots in backticks; they are instructions, not slots.
- [ ] `v3` is merged into `main` (the notes link to `main`), then this file is deleted.

- **TODO(flintq5): Flint in two lanes.** `release/flint-v3/README.md`, `release/notes-v3.0.md` and the README each get two Flint tables in the three-row form. **Flint Q8:** stock + Unsloth Q8_0, v3 + Unsloth Q8_0 (45.9), v3 + `p065`. **Flint Q5:** stock + Unsloth UD-Q5_K_XL, v3 + UD-Q5_K_XL, v3 + Flint-Pascal-Q5 (already on Hugging Face since v2). Each row is filled from its own 2026-11-06 rig 1 run.

- **TODO(names): the public names** (the owner, 2026-11-05; the map is in the private BUILDS.md, "Public names"). In every release page, card, the notes, the README and book chapter 10, call my files Flint Pascal Q5 GGUF, Flint Pascal Q8 GGUF (p065), Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (pmix-q2-fast) and Ornith-1.5-122B Pascal N2P GGUF (valid1), with the internal code only in parentheses on first mention. The Hugging Face repos follow the same pattern (`CinderellaMan/Swift-Qwen3.8-Flash-Next-Pascal-Q2-GGUF` and so on); the Swift card adds a line saying it is not affiliated with or endorsed by UkisAI.

- **TODO(grid): the full grid per model pair** (the owner, 2026-11-05). Each lane's table on its release page and Hugging Face card shows both engines, each tuned, on both files. That is stock llama.cpp b12076 and Miner Llama v3, each on the public file (Swift GSQ-RCO Q2_0, Unsloth Q8_0, Unsloth UD-Q5_K_XL, Unsloth UD-Q2_K_XL) and on my file (Swift Qwen3.8-Flash-Next Pascal Q2 GGUF, Flint Pascal Q8 GGUF, Flint Pascal Q5 GGUF, Ornith-1.5-122B Pascal N2P GGUF), and next to each stock row the settings stock was tuned with. Ornith-1.5-122B Pascal N2P GGUF cannot run on stock (its types are this fork's): say so in that cell.
