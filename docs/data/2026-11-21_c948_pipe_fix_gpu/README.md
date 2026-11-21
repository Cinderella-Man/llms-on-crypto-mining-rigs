# The Flash-Next pipe fix on three newer cards: the GPU half of its check (C948; 2026-11-21)

Registry rows [Q1687](../../../quants/experiments.md) (the GPU half of the night's branches; this is its second point,
C948) and [Q1670](../../../quants/experiments.md) (the fault and its fix). The processor half is
[2026-11-21_c932_pipe_logits](../2026-11-21_c932_pipe_logits/README.md).

**Passed.** The cards show the fault and the fix removes it. With the base, every pipelined Flash-Next job leaves plain
decoding: on a random model the tokens stay the same but the logits, the KV state and the probe differ; on both real
Flash-Next files the replies turn to noise within a few words. With the fix, all 66 pipelined Flash-Next jobs give
plain decoding's tokens, logits, state and probe exactly. Flint is the same on both builds, pipelined and plain. Plain
decoding gives the same hashes on both builds in every check, so the fix does not touch it. Nothing here is timed: the
speed half is a P106 rig row.

## What ran where

| | |
|---|---|
| New build | fork branch `pascal-p106-v5-pipe-logits` `80d83fa1a`: one commit on the base, "qwen4exp: the fused conv window writes its histories to the ubatch's state group (C932)". Rig tarball md5 `91569ac26e57b1086c4cf1c2b8747844`; `llama-pipespec` `068ad23b5f64f7710216902faeae7d9f`, `test-llama-archs` `abbe4b794ec600e3f602d6a22283b8cb` |
| Base build | `pascal-p106-v5` `3dfad876f` ([its build](../2026-11-21_c908_v5_merged_build/README.md)); rig tarball md5 `0b5b983e22d7296ff44fe363d55f61bb`, `llama-pipespec` `be98d73855ce0869970a18f6af127c11` |
| Cards | three RTX 3090 24 GB on a build machine, running the rig build's compute 6.1 PTX through the driver's JIT, as in [C947](../2026-11-21_c947_v5_merged_gpu/README.md). A CUDA preflight passed on all three first |
| Settings | every switch at its default (none set), as in C932's checks; no clock, power or other card setting changed |
| The check | [bench/pipespec/check_identity.sh](../../../bench/pipespec/check_identity.sh) (md5 `8cf4fe35bce82ab6f19170d6a663b26d`): one model load runs plain decoding, then every pipelined job (depth 0 with no guesses; at each depth an oracle replaying plain decoding's tokens with 0 % and 50 % of its guesses made wrong on purpose, and four guessers: copy, copy-else-continue, ngram-simple and ngram-map-k). Each job is compared with plain decoding on four hashes: the tokens, every token's logits bit for bit, the KV state after generation, and a probe (a few tokens decoded afterwards, which read every KV cell still visible). Greedy sampling throughout |

The models:

| Step | Model | Cards | Settings |
|---|---|---|---|
| `random` | random-weight Flash-Next (`qwen4exp`) and Flint models made by the new build's `test-llama-archs -s 1` with a 4,096-token vocabulary (md5 `3c1f8f2cc6bd17a424e459ba638650c5` and `8a91d5448c650c93b981d7e89e348110`) | the first | no chat template, the rewrite prompt, 96 tokens, depths 1 and 3, `-ngl 99 -fa on -c 2048`; base, base with `LLAMA_GDN_FUSE=0` (the unfused conv window, the control) and new |
| `fnreal` | Swift GSQ-RCO Q2_0: UkisAI's Swift-1.5 file of Qwen3.8-Flash-Next with its n-gram table as the third shard, the Flash-Next v5 preset's file | all three, `-ts 1,1,1` | C932's real-file settings: the chat template, the code prompt, 64 tokens, depth 1, `-ngl 99 -fa on -c 4096 -fit off` |
| `fn` | the same file | all three | the P106 check's settings: the rewrite prompt, 128 tokens, depths 1 and 3 |
| `fnp` | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (`pmix-q2-fast`), the third shard the same n-gram table (md5 `98ce140e641bc5d68af9f31df0f1c048`) | all three | as `fn` |
| `flint` | Flint Pascal Q8 GGUF (`p065`) | all three | the chat template, the code prompt, 256 tokens, depth 4 (its preset's), its preset's batch sizes: `-ngl 99 -fa on -c 4096 -b 512 -ub 128` |

The real files are larger than one card (the Flash-Next files' first two shards 37.7 GB, Flint 40.4 GB), so every
real-file check ran split over the three cards, with the hops between them. No draft model was needed: the guessers
and the oracle make the guesses.

## Results

Per check ([summary.tsv](summary.tsv); every job in [checks.tsv](checks.tsv)):

| Step | Model | Build | Pipelined jobs | Jobs that differ from plain | ... with other tokens | Plain decoding's tokens and logits hashes |
|---|---|---|---:|---:|---:|---|
| random | Flash-Next type | base | 13 | **13** | 0 | `4c74b9f8e110cde1` `07ab7e5b03f196e4` |
| random | Flash-Next type | base, `LLAMA_GDN_FUSE=0` | 13 | 0 | 0 | the same |
| random | Flash-Next type | new | 13 | 0 | 0 | the same |
| random | Flint type | base, base `LLAMA_GDN_FUSE=0`, new | 13 each | 0 | 0 | `7f02660cc77ddd71` `a80fd54f9b84fa1d` on all three |
| fnreal | Swift GSQ-RCO Q2_0 | base | 7 | **7** | **7** | `fc5830f6e0f13ec0` `aa85fb9f8145b66c` |
| fnreal | Swift GSQ-RCO Q2_0 | new | 7 | 0 | 0 | the same |
| fn | Swift GSQ-RCO Q2_0 | base | 13 | **13** | **13** | `1080aa0a831751d6` `e1b3af3c4be3e6bc` |
| fn | Swift GSQ-RCO Q2_0 | new | 13 | 0 | 0 | the same |
| fnp | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | base | 13 | **13** | **13** | `dd31fa51c5f8d214` `2347d1bf176d0b1a` |
| fnp | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF | new | 13 | 0 | 0 | the same |
| flint | Flint Pascal Q8 GGUF | base | 7 | 0 | 0 | `762ba7ad25ca5a25` `b75e5a659077b637` |
| flint | Flint Pascal Q8 GGUF | new | 7 | 0 | 0 | the same |

- **The base's fault shows on the cards as on the processor.** On the random Flash-Next model every pipelined job
  differs in its logits, state and probe from depth 0 on, while its tokens stay plain decoding's; the unfused conv
  window (`LLAMA_GDN_FUSE=0`) makes the base exact again, which places the fault in the fused window, as C932 found.
- **On the real files the base breaks the replies.** With C932's settings the base's seven pipelined jobs all leave
  plain decoding's reply within its first four words (plain decoding opens "We need answer user: ..."; depth 0 writes
  "We need answer user is a Python" and stops), and six of the seven stop early: 9, 41, 9, 25, 9 and 9 tokens against
  64. The oracle, which replays plain decoding's own tokens, had 3 of its 63 guesses accepted. With 128 tokens every
  base job on both Flash-Next files stopped early, after at most 40 tokens ([checks.tsv](checks.tsv) lists each).
- **With the fix every pipelined job is plain decoding's,** at depths 1 and 3, with 0 % and 50 % wrong guesses, on both
  Flash-Next files and the random model. The fix's guesses and accepted guesses with C932's settings are the
  processor's job for job (the oracle 62 of 63, with half its guesses wrong 38 accepted and 24 dropped; copy 20 and 7,
  copy-else-continue 26 and 17, ngram-simple 16 and 2, ngram-map-k 16 and 1), and plain decoding's 64 tokens on the
  cards are the processor's (tokens hash `fc5830f6e0f13ec0` in both; the logits differ, as the cards add up their sums
  in another order).
- **Flint is not affected:** its random model and its release file give plain decoding's hashes in every pipelined job
  on both builds, and the same plain hashes on both builds. On the release file with the preset's depth 4 the jobs
  guessed up to 602 tokens and dropped up to 463.
- No CUDA error, no crash, no timeout in any check; every check's exit code is 0 where every job is the same and 1
  where one differs.

## Files

| File | What it holds |
|---|---|
| [summary.tsv](summary.tsv) | per check: step, model, build, exit code, pipelined jobs, jobs differing, jobs with other tokens, plain decoding's tokens and logits hashes |
| [checks.tsv](checks.tsv) | every job of every check: tokens generated, guesses, accepted, dropped, wasted, and the four comparisons with plain decoding |
| [run/run.sh](run/run.sh) | the driver: each check in its own container on the build machine, the base and then the fix |
| [run/tables.py](run/tables.py) | `checks.tsv` from the driver's output folder (standard library, `python3 -I`) |

## Commands

On the build machine, with the two rig builds unpacked, a copy of `bench/pipespec` and the model files on its disks
(the three cards free, a CUDA preflight first):

```bash
WORK=<work folder> NEW=<80d83fa1a build>/bin REF=<3dfad876f build>/bin PS=<copy of bench/pipespec> \
FN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> \
FNP=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf> FLINT=<flint-pascal-Q8.gguf> \
MOUNTS='<folders holding the builds, models and the bench copy>' DOCKER_ARGS='--cpu-shares=256' \
setsid nohup bash run.sh > <work folder>/driver.log 2>&1 < /dev/null &
# the job table, from a copy of <work folder>/out
python3 -I run/tables.py <copy of work folder>/out > checks.tsv
```

It ran 10:50–11:08. The random models are made fresh by the new build's `test-llama-archs` (seed 1), so the run needs
nothing beyond the builds and the three files. The model files are not part of this folder; the tables hold only
hashes and counts, and the prompts are this repository's own (`bench/pipespec/prompts/`).
