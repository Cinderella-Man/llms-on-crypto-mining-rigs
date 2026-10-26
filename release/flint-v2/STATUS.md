# Release status (2026-10-26)

Working notes for finishing the v2.0 release; delete this file before the release is published.

## Done

- **This folder** covers what ships: the engine `37e717ff6` (rc4 `2d53179aa` with the release defaults),
  Flint-Pascal-Q5, and Unsloth's Flint Q8_0 on nine cards at 262k as the headline (linked, not re-uploaded). The
  drafter and the store are named once as "coming later (v2.1)"; their drafts (cards, rebuild guide, pipeline,
  pre-upload checks) are parked in [../flint-v2.1](../flint-v2.1/README.md).
- **Release settings,** chosen on the coding agent ([results.md](results.md), "Agentic speed"): `--pipe-depth 4`,
  `-b 512 -ub 128`, no agent prompt cache; decode attention, the input layer on the card, packed uploads and the
  card-built mask off unless set.
- **Flint-Pascal-Q5:** the public recipe rebuilt it from BF16 with 779 of 779 tensors equal; release header
  (`0163ec54…`), tensor manifest, Apache-2.0 LICENSE and the card with speed, quality and how it was made; account
  CinderellaMan, `--quantized-by "Kamil Skowron (@kamilskowron)"`.
- **Engine archives:** packaged from `37e717ff6` as v2.0: no AVX instructions, glibc 2.38 at most, every library found
  with both archives unpacked; the three hashes are in [CHECKSUMS](CHECKSUMS).
- **Pages:** README, the Flint page, the notes template and this folder carry the agentic numbers;
  `scripts/release_variant.sh rc4` (= rc4 q163-q111 nodrafter) strips the drafter text and keeps one "coming later"
  line per page.

## Placeholders left

`RIG-TBD` = a measurement from the rig; `OWNER-TBD` = the owner's decision. Lines tagged `<!-- only:drafter -->` or
inside `begin:drafter` blocks need nothing: the rc4 variant removes them.

| What fills it | How | Where |
|---|---|---|
| Unsloth's UD-Q5_K_XL, the same size as Flint-Pascal-Q5: KL divergence and the coding agent at depth 4 | `llama-perplexity --kl-divergence` against the BF16 base; `release_check.sh` on the release line | the card, results.md, the Flint page. The Flint-Pascal-Q5 upload waits for it (the owner, 2026-10-26) |
| Fork push, release tag and links (OWNER-TBD) | the owner | README.md (engine archives, clone branch); models/flint/README.md (engine rows); README.md of this folder (What ships, gap 1); reproduce/engine.md; the card's install step |

## Steps to publish, in order

1. **Fork (owner):** push branch `pascal-p106-v2-rc4` at `37e717ff6` to the fork and tag it; fill the tag and links.
2. **Release notes:** once the stock rows are in the template,
   `NOTES_ONLY=1 OUT=<the out-v2.0 folder> NOTES_VARIANT=rc4 scripts/package_release.sh - v2.0 37e717ff6` rewrites
   `RELEASE_NOTES.md` next to the archives.
3. **Engine release (owner):** a GitHub release on the fork for the tag, with the three archives, `SHA256SUMS` and
   `RELEASE_NOTES.md` as its text.
4. **Flint-Pascal-Q5 (owner, after the UD-Q5_K_XL comparison):** the card's numbers final, then the two `hf upload`
   commands in [huggingface/UPLOAD.md](huggingface/UPLOAD.md).
5. **Pages:** fill the tag and links, then `scripts/release_variant.sh rc4 README.md models/flint/README.md` in place;
   check links and privacy.
6. **Public repo (owner approves):** delete this file; check `git log origin/v2..v2` for other sessions' unreviewed
   commits; push `v2`, then merge `v2` into `main` for the release.
