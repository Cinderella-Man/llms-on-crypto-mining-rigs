# The Flash-Next pipe fix on the P106 itself (C934; 2026-11-24)

Registry rows [Q1672](../../../quants/experiments.md) (this check) and [Q1670](../../../quants/experiments.md) (the fault
and its fix). The processor half is [2026-11-21_c932_pipe_logits](../2026-11-21_c932_pipe_logits/README.md), the
newer cards' half [2026-11-21_c948_pipe_fix_gpu](../2026-11-21_c948_pipe_fix_gpu/README.md).

**Passed.** On the P106 the old build shows the fault and the current v5 build does not. With the old build every
pipelined Flash-Next job leaves plain decoding: on a random model its logits, KV state and probe differ while its
tokens stay the same; on the real file every reply turns to noise after its first four words and stops early. With
the current build all 26 pipelined Flash-Next jobs give plain decoding's tokens, logits, state and probe exactly. Flint is
the same on both builds, pipelined and plain, on its random model and on its release file with its preset's depth 4.
Plain decoding gives the same hashes on both builds in every check. Nothing here is timed.

## What ran where

| | |
|---|---|
| Rig | one rig of nine P106-090 6 GB cards (PCIe 1.0 x1 each, a 4 GB host), at stock clocks with every clock offset 0. An earlier clock check that day had stepped eight of its cards down to a lower level until the next power-on; nothing here is timed, so it does not matter |
| New build | `pascal-p106-v5` `af00bb13b`: the landing of 14 checked branches, the Flash-Next pipe fix (C932's commit `80d83fa1a`, merged as `0c83f4a86`) among them ([the landing](../2026-11-24_land_v5_batch2/README.md)). Rig tarball md5 `b51ad2740d5822e7e14af656c413ec22`; `llama-pipespec` `091830d65e87dde7e9118895483f508b`, `test-llama-archs` `bf171d94e7000d14d1933e594eb0d2de` |
| Base build | `pascal-p106-v5` `8a9629942`, the tip before that landing, which has the fault. Rig tarball with the check tools md5 `59ca8336b87367a6522e63c6fcc1381b`; `llama-pipespec` `280ad667f2df869ed190b149d8195cd8`, `test-llama-archs` `eb9dd8a6d59d58f144e7ca7f8ca0582e` |
| Settings | every switch at its default (none set); the base with `LLAMA_GDN_FUSE=0` (the unfused conv window) as the control on the random models |
| The check | [bench/pipespec/check_identity.sh](../../../bench/pipespec/check_identity.sh) (md5 `8cf4fe35bce82ab6f19170d6a663b26d`): one model load runs plain decoding, then every pipelined job (depth 0 with no guesses; at each depth an oracle replaying plain decoding's tokens with 0 % and 50 % of its guesses made wrong on purpose, and four guessers: copy, copy-else-continue, ngram-simple and ngram-map-k). Each job is compared with plain decoding on four hashes: the tokens, every token's logits bit for bit, the KV state after generation, and a probe (a few tokens decoded afterwards, which read every KV cell still visible). Greedy sampling throughout |

The models and settings, as [C932](../2026-11-21_c932_pipe_logits/README.md) wrote them for this check
([run/p106_check.sh](run/p106_check.sh)):

| Step | Model | Cards | Settings |
|---|---|---|---|
| `random` | random-weight Flash-Next (`qwen4exp`) and Flint models made by the new build's `test-llama-archs -s 1` with a 4,096-token vocabulary (md5 `3c1f8f2cc6bd17a424e459ba638650c5` and `8a91d5448c650c93b981d7e89e348110`, the same files as C948's) | card 0 | no chat template, the rewrite prompt, 96 tokens, depths 1 and 3, `-ngl 99 -fa on -c 2048`; base, base with `LLAMA_GDN_FUSE=0`, new |
| `flint` | Flint Pascal Q8 GGUF (`p065`), Flint's release file | all nine, the v5 preset's split `-ts 6,6,5,5,6,5,5,6,6` | the chat template, the code prompt, 256 tokens, depth 4 (the preset's), the preset's batch sizes: `-ngl 99 -fa on -c 4096 -b 512 -ub 128` |
| `fn` | Swift GSQ-RCO Q2_0 (UkisAI's Swift-1.5 file of Qwen3.8-Flash-Next with its n-gram table as the third shard), as Q550 ran it | all nine, `-ts 6,5,5,6,5,5,6,5,6` | the chat template, the rewrite prompt, 128 tokens, depths 1 and 3, `-ngl 99 -fa on -c 4096 -fit off -lm dio` |

## Results

Per check ([summary.tsv](summary.tsv); every job in [checks.tsv](checks.tsv)):

| Step | Model | Build | Pipelined jobs | Jobs that differ from plain | ... with other tokens | Plain decoding's tokens and logits hashes |
|---|---|---|---:|---:|---:|---|
| random | Flash-Next type | base | 13 | **13** | 0 | `4c74b9f8e110cde1` `bf25a5eb24f82b1a` |
| random | Flash-Next type | base, `LLAMA_GDN_FUSE=0` | 13 | 0 | 0 | the same |
| random | Flash-Next type | new | 13 | 0 | 0 | the same |
| random | Flint type | base, base `LLAMA_GDN_FUSE=0`, new | 13 each | 0 | 0 | `7f02660cc77ddd71` `10639afab5fda5c4` on all three |
| flint | Flint Pascal Q8 GGUF | new | 7 | 0 | 0 | `c0dcfb3074f4fef9` `96d96b8aafcce425` |
| flint | Flint Pascal Q8 GGUF | base | 7 | 0 | 0 | the same |
| fn | Swift GSQ-RCO Q2_0 | base | 13 | **13** | **13** | `f045c2e02c346d72` `12a9ae53edfff142` |
| fn | Swift GSQ-RCO Q2_0 | new | 13 | 0 | 0 | the same |

- **The base's fault shows on the P106.** On the random Flash-Next model every pipelined job differs from plain
  decoding in its logits, state and probe, from depth 0 on, while its tokens stay plain decoding's. With the unfused
  conv window (`LLAMA_GDN_FUSE=0`) the base is exact again, which places the fault in the fused window, as C932 found
  on the processor and C948 on three newer cards.
- **On the real file the base breaks the replies.** Plain decoding opens "We need answer user's request: rename
  variable plan to layout ..."; the base's depth 0 writes "We need answer user wants rename plan: Rename plan to layout
  sources plan: ..." and stops after 27 tokens. Every one of the base's 13 pipelined jobs stopped early: 27, 29, 60,
  31, 29, 27, 27, 25, 25, 18, 25, 27 and 27 tokens against 128. The oracle, which replays plain decoding's own tokens,
  had 5 of its 28 guesses accepted at depth 1.
- **With the fix every pipelined job is plain decoding's,** at depths 1 and 3, with 0 % and 50 % wrong guesses, on the
  real file and the random model. On the real file the oracle had 126 of 127 guesses accepted at both depths; with
  half its guesses wrong, 72 accepted and 54 dropped at both depths.
- **Flint is not affected:** its random model and its release file give plain decoding's hashes in every pipelined job
  on both builds, and the same plain hashes on both builds. On the release file the two builds' jobs guessed, accepted
  and dropped the same counts job for job; with half its guesses wrong at depth 4 the oracle guessed 602 tokens and
  dropped 463.
- The random models' plain token hashes are the ones C948 had on three RTX 3090s (`4c74b9f8e110cde1`,
  `7f02660cc77ddd71`); their logits differ there, as other cards add up their sums in another order.
- No CUDA error, no crash, no timeout in any check; every check's exit code is 0 where every job is the same and 1
  where one differs. Every card was back at 0 MiB at the end.

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | per check: step, model, build, exit code, pipelined jobs, jobs differing, jobs with other tokens, plain decoding's logits hash |
| [checks.tsv](checks.tsv) | every job of every check: tokens generated, guesses, accepted, dropped, wasted, and the four comparisons with plain decoding |
| [run/p106_check.sh](run/p106_check.sh) | the check as run on the rig: C932's script with only its header changed to name these two builds |
| [run/tables.py](run/tables.py) | `checks.tsv` from the check's output folder (standard library, `python3 -I`); C948's script, unchanged |

## Commands

On the rig, with both rig builds unpacked, a copy of `bench/pipespec` (`check_identity.sh` and `prompts/`) and the two
model files, the cards free:

```bash
NEW=<af00bb13b build>/bin REF=<8a9629942 build>/bin PS=<copy of bench/pipespec> OUT=<output folder> \
FLINT=<flint-pascal-Q8.gguf> FN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> \
setsid -f nohup timeout 7200 bash run/p106_check.sh > <output folder>/console.txt 2>&1 < /dev/null
# the job table, from a copy of the output folder
python3 -I run/tables.py <copy of output folder> > checks.tsv
```

It ran 17:28–18:07: the random models took two minutes, each load of a real file about ten minutes from the rig's USB
disk. The random models are made fresh by the new build's `test-llama-archs` (seed 1), so the run needs nothing beyond
the builds and the two files. The model files are not part of this folder; the tables hold only hashes and counts, and
the prompts are this repository's own (`bench/pipespec/prompts/`).
