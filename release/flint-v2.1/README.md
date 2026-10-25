# Flint v2.1 (in preparation): the staircase drafter and its store

Not released. The v2 release ships the engine, K2h-r16 and Flint Q8_0 on nine cards
([release/flint-v2](../flint-v2/README.md)); the trained drafter and the store of Flint's earlier answers follow in
v2.1, rebuilt from sources whose answers may be published. How the drafter works and every training run so far:
[docs/drafter.md](../../docs/drafter.md).

This folder holds the drafts for that release:

| File | What |
|---|---|
| [reproduce/drafter.md](reproduce/drafter.md) | the rebuild guide: the clean prompt sources and their licences, answers, dump, vocabulary, training, scoring, export, checks, the store |
| [reproduce/drafter_pipeline.sh](reproduce/drafter_pipeline.sh) | the same steps as one sequential script |
| [huggingface/flint-staircase-drafter/](huggingface/flint-staircase-drafter/README.md) | the model card for `CinderellaMan/Flint-Staircase-Drafter` (Apache-2.0) |
| [huggingface/flint-drafter-data/](huggingface/flint-drafter-data/README.md) | the dataset card for `CinderellaMan/flint-drafter-data`, if the data is published |

## Before uploading (v2.1)

- **The drafter heads:** export both (`export_gguf.py`, [reproduce/drafter.md](reproduce/drafter.md), step 7), run the
  checks of step 8 on each, write their SHA-256 into `huggingface/flint-staircase-drafter/CHECKSUMS`, and fill the
  card's file names and numbers from their rig run. The engine's release notes list them through
  `scripts/package_release.sh DRAFTS="<nine-card file>=… <general file>=…"` with the `drafter` variant.
- **The store**, `flint-answers-core-13.8M.pstore` (110 MB, Flint's answers to our own code prompts), ships next to the
  heads if the owner approves it.
- **The data, if published:** only the clean sources' prompts and the answers to them, without the prompts that share
  text with Q51; the manifest without machine paths (model files by name and SHA-256, engines by commit); each head's
  `config.json` with its machine paths (`feats`, `out`, `vocab`, `flint`, `init_from`) replaced by file names (`head.pt`
  and `ckpt.pt` hold only tensors and counters); no snapshot copies (`*.snap*.jsonl`); every text file searched for
  machine names and paths; `CHECKSUMS` over every file:
  `find . -type f ! -name CHECKSUMS ! -name README.md ! -name LICENSE -print0 | sort -z | xargs -0 sha256sum > CHECKSUMS`.

```bash
hf upload CinderellaMan/Flint-Staircase-Drafter <nine-card file> <nine-card file>
hf upload CinderellaMan/Flint-Staircase-Drafter <general file> <general file>
hf upload CinderellaMan/Flint-Staircase-Drafter release/flint-v2.1/huggingface/flint-staircase-drafter .
hf upload CinderellaMan/Flint-Staircase-Drafter flint-answers-core-13.8M.pstore flint-answers-core-13.8M.pstore   # if the store ships
cp release/flint-v2.1/huggingface/flint-drafter-data/{README.md,LICENSE} <data folder>/   # then write CHECKSUMS there
hf upload CinderellaMan/flint-drafter-data <data folder> . --repo-type dataset            # if the data is published
```
