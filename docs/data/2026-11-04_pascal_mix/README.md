# Q880: Pascal mixes of Swift Qwen3.8-Flash-Next (2026-11-04)

For Flint this project made its own quantizations to run faster on the P106 cards. These are the same idea for
Qwen3.8-Flash-Next: keep the makers' routed experts byte for byte and choose every other weight's type for the P106.
The routed experts are ~34 of the ~37.5 GB the Q2_0 file puts on the cards (~43 of ~47 GB in IQ3_XXS), quantized
with the makers' GSQ-RCO method; requantizing them from a Q8_0 would very likely do worse. But a decoded token reads
the routed experts only 10 of 512 at a time: of the 4.2 GB the Swift Q2_0 file reads per token, 3.5 GB are the dense
weights (attention and recurrent projections, hyper-connections, router, shared expert, output layer). Their types
decide most of the decode time, and they cost almost nothing on the cards.

Registry rows Q880–Q889 in [quants/experiments.md](../../../quants/experiments.md). Tool:
[bench/pascal_mix/pmix.py](../../../bench/pascal_mix/pmix.py) (the plan) with [scripts/gguf_retype.py](../../../scripts/gguf_retype.py),
[scripts/gguf_compose.py](../../../scripts/gguf_compose.py) and llama-quantize (the build).

## What is slow on a P106, per weight

One-token decode on nine P106-090s, fork branch `pascal-p106-v3-b12076-ladder3` (the night's Pascal kernels). GB/s
are in-model speeds from [bench/file_choice/efficiency.tsv](../../../bench/file_choice/efficiency.tsv) (R13's
profile, R16, R28); the standalone one-card times of today's kernels are in
[2026-11-04_kcurve_p106](../2026-11-04_kcurve_p106/README.md) (`perf_dense.tsv`, `kcol_best.tsv`).

| Type | Bits a weight | GB/s in the model | µs per million weights | Where Swift Q2_0 uses it |
|---|---:|---:|---:|---|
| Q3_K | 3.44 | 33 (k-column kernel's one-column path; 255–329 µs for a 2,560 × 10,240 product alone) | 12.9 | 56 attention/recurrent matrices (475 MB a token), 34 shared gate/up |
| IQ3_S | 3.44 | 25.5 (52 expected with C2's kernel) | 16.9 | IQ3_XXS file: 15 dense, 40 shared |
| IQ4_NL | 4.50 | 23 (no Pascal one-token kernel) | 24.3 | 7 shared down (IQ3_XXS: 40) |
| IQ4_XS | 4.25 | 87.7 | 6.1 | 45 dense |
| Q4_K | 4.50 | 91.4 | 6.2 | 26 dense |
| Q5_K | 5.50 | 87.1 | 7.9 | 12 dense |
| Q6_K | 6.56 | 88.8 | 9.2 | 10 dense |
| BF16 | 16 | 141.9 (C54's kernel; 144 alone at the router's 2,560 → 512) | 14.1 | hyper-connections, indexer |
| F32 | 32 | 83.5 (llama.cpp's kernel) | 47.9 | the router (252 MB a token) |

Q3_K is the slowest weight type per weight, slower than Q6_K, which has nearly twice its bits. At 2–8 tokens (a
verify pass) the gap narrows but holds: a 2,560 × 10,240 product at four tokens takes 370 µs in IQ4_XS (13.9 MB) and
403 µs in Q3_K (11.3 MB). And the router's F32 holds BF16 values: all 96 `ffn_gate_inp*` tensors of both files have
their low 16 bits zero (checked on every value), so BF16 is exact; C54's kernel reads that shape in 18 µs a layer on one
card, where the F32 copy takes est. 63 at llama.cpp's 83.5 GB/s.

## The three mixes

Every mix keeps byte for byte: the routed experts, the hyper-connections (BF16: on `-ladder3` they beat the Q8_0
retype at every k, [k-curve](../2026-11-04_kcurve_p106/README.md)), the output layer (Q5_K: 144 GB/s; Q6_K's kernel
reads that shape at 94), the token embeddings (host memory), the n-gram table and every norm. The router becomes BF16
(exact). No tensor gets fewer bits than the makers gave it.

| Mix | Base experts | What changes | Tensors | Cards (MiB) | Est. ms a token, timed kernels (expected) | Est. tok/s |
|---|---|---|---:|---:|---:|---:|
| Swift Q2_0 (published) | Q2_0 | — | — | 35,730 | 82.4 (73.0); 82.4 measured on rig 1 | 12.1 |
| **pmix-q2-fast** | Q2_0 | Q3_K → IQ4_XS (dense, shared); IQ4_NL shared down → Q5_0; shared gate/up one type a layer; router BF16 | 168 | 35,735 | **72.4** (72.4) | **13.8** |
| pmix-q2-spine6 | Q2_0 | every dense matrix and shared gate/up below Q6_K → Q6_K; shared down below Q5_0 → Q5_0; router BF16 | 328 | 36,522 | 82.0 (81.9) | 12.2 |
| Swift IQ3_XXS (published) | IQ3_XXS mix | — | — | 44,710 | 98.9 (91.9) | 10.1 |
| **pmix-iq3-fast** | IQ3_XXS mix | IQ3_S → IQ4_XS (dense, shared); IQ4_NL shared down → Q5_0; shared gate/up one type a layer; router BF16 | 142 | 44,628 | **93.4** (88.9) | 10.7 |
| pmix-iq3-spine6 (planned, not built) | IQ3_XXS mix | as q2-spine6 | 277 | 45,075 | 98.6 (94.3) | 10.1 |

The estimate is [bench/file_choice](../../../bench/file_choice/README.md)'s model: the MB a token reads of each type
over that type's GB/s, plus 33.48 ms outside the weight products; it gave 82.4 ms for Swift Q2_0, which rig 1 then
measured on `-ladder3` (82.4). "Expected" puts the not-yet-timed kernels (C4's Q3_K, C2's IQ3_S, C81's short Q8_0) at
their nearest timed relative's speed; the mixes remove those types, so their two columns meet. Not modelled: the shared
expert's gate/up fusion (one launch instead of two in 34 of 48 layers of Q2_0, 30 of IQ3_XXS; est. −0.3 to −0.7 ms)
and the k-curve (measured by [run/rig_time.sh](run/rig_time.sh)).

Shared gate and up fuse into one launch only when they have the same type (`ggml_cuda_should_fuse_mul_mat`), and
beside the routed experts when they also have the experts' type (Swift Q2_0's six layers with both in Q2_0, which
the fast recipe leaves alone). The fast recipe gives each other layer's pair the type of the two with more bits, after
Q3_K and IQ3_S become IQ4_XS; so a pair (Q2_0, Q3_K) becomes IQ4_XS twice, (Q3_K, Q5_K) Q5_K twice.

Per-role tables (MB a token, GB/s, est. ms, base against mix): [design/](design/), one folder per mix with
`summary.tsv`, `plan.tsv` (every tensor: class, layer, shape, old and new type, how, bytes), `types.txt` (the
llama-quantize type file) and `take.txt`.

### Expected quality (the design's reasoning; the scores follow below)

- **pmix-q2-fast:** 1,106 M of the dense weights go from Q3_K (3.44 bits) to IQ4_XS (4.25) and the shared expert
  gains bits; nothing loses any. KLD at most Swift Q2_0's (agent 0.285), lower by however much the dense spine
  contributes; the spine6 file bounds that.
- **pmix-q2-spine6:** every dense matrix at 6.56 bits (from the Q8_0, near lossless): what is left of the KLD is the
  routed experts' and the hyper-connections' (exact) share. It answers whether spending bytes on the dense spine pays
  at Q2_0's experts.
- **pmix-iq3-fast:** 238 M dense weights from IQ3_S to IQ4_XS, the shared expert's IQ4_NL down to Q5_0. KLD at most
  Swift IQ3_XXS's (0.176).
- The retyped tensors come from the Swift-1.5 Q8_0 of the same model (FN1's reference, its dense matrices in Q8_0),
  requantized by llama-quantize with FN4's importance matrix (Swift-1.5 Q8_0 on 200 chunks of 512 tokens of the agent
  training text, [flash_next_quality](../2026-11-03_flash_next_quality/README.md)). Q8_0's own error is about a
  sixteenth of a 4-bit type's, so the double rounding costs little.

### Fit at 64k (nine 6 GB cards, the table on the SSD)

- pmix-q2-fast: +5 MiB against Swift Q2_0, so Swift Q2_0's lines hold: `-c 65536 -b 2048 -ub 512
  -ts 6,5,5,6,5,5,6,5,6` (735 MiB on the fullest card, [fit probe](../2026-11-03_flash_next_fit/README.md)).
- pmix-q2-spine6: +792 MiB; Swift IQ2_XS (+1,466 MiB against Q2_0) still fits 64k at `-ub 512`
  ([file choice](../2026-11-04_file_choice/README.md)), so this one does too, with less margin.
- pmix-iq3-fast: −82 MiB against Swift IQ3_XXS, which fits 64k on whole layers at `-ub 64` with 155 MiB to spare.

## Scores

KL divergence against the Swift-1.5 Q8_0 reference's logits (FN1's bases, made on all seven quant-machine cards),
with FN2's `kld.sh` and split (`-fit on`), on five cards (two RTX 3080, three RTX 3090; the other two cards were out
of service). Each base file is scored again on the same five cards, so a mix is read against its control. The
controls land within 0.002 (Q2_0) and 0.003 (IQ3_XXS's agent text) of FN2's seven-card scores, so the card set
moves these numbers very little. Mean KLD (± its standard error over tokens), the agent text's 99th percentile, and
the share of tokens whose top choice matches the reference's; all columns in [kld.tsv](kld.tsv):

| File | Wiki KLD | Agent KLD | Code KLD | Agent p99 | Same top: wiki / agent / code | Est. ms a token | Cards (MiB) |
|---|---:|---:|---:|---:|---|---:|---:|
| Swift Q2_0, control | 0.4282 ± 0.0059 | 0.2866 ± 0.0056 | 0.1121 ± 0.0040 | 6.32 | 77.85 / 90.67 / 95.97 % | 82.4 | 35,730 |
| **pmix-q2-fast** | 0.4264 ± 0.0059 | 0.2792 ± 0.0054 | 0.1080 ± 0.0038 | 6.17 | 77.71 / 90.70 / 96.12 % | **72.4** | 35,735 |
| **pmix-q2-spine6** | 0.4211 ± 0.0058 | **0.2559** ± 0.0049 | 0.1042 ± 0.0038 | 5.41 | 78.04 / 90.92 / 96.27 % | 82.0 | 36,522 |
| Swift IQ3_XXS, control | 0.2480 ± 0.0039 | 0.1786 ± 0.0039 | 0.0542 ± 0.0023 | 3.69 | 83.35 / 92.24 / 97.41 % | 98.9 | 44,710 |
| **pmix-iq3-fast** | 0.2491 ± 0.0039 | 0.1735 ± 0.0037 | 0.0551 ± 0.0023 | 3.64 | 83.19 / 92.29 / 97.44 % | **93.4** | 44,628 |
| Q8_0 at `-ub 256`, the floor (seven cards) | 0.0170 | 0.0291 | 0.0036 | 0.55 | 95.57 / 97.10 / 99.24 % | | |

- **pmix-q2-fast is Swift Q2_0's quality at est. 10 ms a token less** (12.1 → 13.8 tok/s), +5 MiB on the cards: KLD
  −0.4, −2.6 and −3.7 % on wiki, agent and code, top-token agreement the same (±0.15 points). No tensor lost bits.
  It is the file to time on a rig first.
- **The dense spine holds about a tenth of Q2_0's agent KLD.** With every dense matrix at Q6_K, pmix-q2-spine6 cuts
  the agent KLD by 10.7 % (0.2866 → 0.2559, about five standard errors) and its 99th percentile by 14 %, at the
  published file's est. speed (82.0 ms) and +792 MiB. The rest, ~0.26 on the agent text, is the routed experts'.
  Against pmix-q2-fast it buys 0.023 of agent KLD for est. +9.6 ms (0.0024 a ms); pmix-iq3-fast buys 0.106 for
  +21.0 ms (0.0050 a ms), twice as much a ms, but it fits 64k only at `-ub 64` (or with experts moved) and needs 18
  expert tensors moved to take the MTP draft.
- **pmix-iq3-fast is Swift IQ3_XXS's quality at est. 5.5 ms a token less**, 82 MiB lighter: KLD within one standard
  error on all three texts (−2.9 % agent, +0.4 % wiki, +1.6 % code).
- The standard errors are over tokens of one text, and every file is scored on the same tokens, so a difference
  between two files is known more precisely than the two errors suggest; the per-token values are not kept, so the
  paired error is not given here.

Fit of the built files at 64k, from their headers ([bench/file_choice](../../../bench/file_choice/README.md)'s
`fit.py`; [built/fit.md](built/fit.md) has every context, cache type and draft): pmix-q2-fast `-ub 512` with 737 MiB
free on the fullest card (Swift Q2_0: 737), 548 with the MTP draft (557); pmix-q2-spine6 `-ub 512` with 644 (452 with
the MTP draft); pmix-iq3-fast `-ub 64` on whole layers with 169 (Swift IQ3_XXS: 155), `-ub 256` with 8 expert tensors
moved 395 (387). [built/](built/) also holds the built files' bytes a token by type and role and the est. ms
(`token_bytes.tsv`, `speed.tsv`, `layout.tsv`, `ranking.md`).

## On a rig

[run/rig_time.sh](run/rig_time.sh) runs a mix and the file it was made from on one rig, the same boot and build: a
64k load at the release line with one request (the cards' memory and the host's MemAvailable), then T(1) … T(8) with
the k-curve's settings. For pmix-q2-fast: `MODELS='fast=<its first shard> q2_0=<Swift Q2_0's first shard>'`; for
pmix-iq3-fast add `UB=64 TS=6,6,7,5,5,5,5,5,5` (IQ3_XXS's whole-layer 64k line). Shards 1 and 2 of a mix differ from
the base's; shard 3, the n-gram table, is the same file (link it under the mix's name).

## Build

On a CPU (no GPU), with `nice -n 19 ionice -c3`, pinned to four cores: [run/chain.sh](run/chain.sh) plans each mix
with `pmix.py` and builds it with [run/build.sh](run/build.sh):

1. the base with its router F32 → BF16: `gguf_retype.py retype --rule 'ffn_gate_inp\.weight$=BF16@F32'` (the BF16
   rule added for this, round to nearest even as `ggml_compute_fp32_to_bf16`, with its self-test case), then
   `gguf_retype.py verify`;
2. the tensors to retype taken from the Q8_0: `gguf_compose.py compose --take <each name of take.txt>`;
3. `llama-quantize --allow-requantize --keep-split --imatrix <FN4's> --tensor-type-file types.txt <step 2> <out> Q8_0 4`,
   where `types.txt` names every matrix: the new type for the retyped ones, its own type (copied) for the rest; the
   log must show exactly `take.txt`'s tensors converted;
4. the final file: step 1's shards with `take.txt`'s tensors from step 3 (`gguf_compose.py compose`, then `verify`), so
   every kept tensor is the makers' own bytes whatever llama-quantize did; the table shard stays a link.

```bash
# on the build machine, from a folder holding this repository as tools/ (paths are examples)
export REPO=$PWD/tools DUMPS=<dump.py JSONs: swift-q2_0.json, swift-iq3_xxs.json>
export BASE_Q2=<Swift Q2_0, 3 shards, table alone in shard 3>-00001-of-00003.gguf
export BASE_IQ3=<Swift IQ3_XXS, 3 shards>-00001-of-00003.gguf
export DONOR=<Swift-1.5 Q8_0>-00001-of-00005.gguf IMATRIX=<imatrix.gguf> QUANTIZE=<llama-quantize>
export WORK=<scratch, ~140 GB free> MODELS=<output folder> TABLE=<the table shard the mixes link to>
export CPUS=10,11,22,23 THREADS=4 MOUNTS="<folders mounted at the same path in the container>"
setsid -f nohup bash tools/docs/data/2026-11-04_pascal_mix/run/chain.sh >> chain.log 2>&1 < /dev/null
```

pmix-iq3-fast and pmix-q2-spine6 were scored in place ([run/score_direct.sh](run/score_direct.sh)): the fast disk was busy with
writes, which slowed every KLD run reading its base logits there, so these two files were read from the HDDs, only
the n-gram table from the fast disk. spine6 was built on an HDD for the same reason. Their KLD ran beside small test
jobs on the same cards (under ~2 GB each; a copy of FN2's scripts with [run/wait_shared.sh](run/wait_shared.sh) as its
`wait_gpus`: a free-memory floor a card instead of an empty card), and the first files beside other sessions'
smaller checks on two of the cards; that changes a run's speed, not its KLD.

The scores, on the quant machine (FN1's environment file copied with `GPUS_ALL` set to the five cards and `TAG` and
`WORK` to this run's; the controls are named `<file>-c5` so FN2's seven-card logs, which `score.sh` links in, are not
taken for them):

```bash
export ENV=<that environment file> QUALITY=<2026-11-03_flash_next_quality's run/> BASES=<FN1's kld folder>
export OUT=<score folder> HOT=<fast scratch> TABLE_FAST=<the table shard on the fast disk> SCORE_TAG=pmix
export SCORE=tools/docs/data/2026-11-04_gguf_compose/run/score.sh DONE=<marker folder>
export LIST="pmix-q2-fast <its first shard> <WORK>/done/pmix-q2-fast
swift-q2_0-c5 <Swift Q2_0's first shard>
..."
setsid -f nohup bash tools/docs/data/2026-11-04_pascal_mix/run/score_all.sh >> score.log 2>&1 < /dev/null
python3 -I <2026-11-03_flash_next_quality's run/>table.py <OUT> q8_0-ub256 swift-q2_0-c5 pmix-q2-fast ... > kld.tsv
```

## Files

| File | What it is |
|---|---|
| [design/](design/) | per mix: `summary.tsv`, `plan.tsv`, `types.txt`, `take.txt` (from `pmix.py`) |
| [kld.tsv](kld.tsv) | every score: mean KLD, its error, 99th and 99.9th percentile, same top token, RMS of the probability change, perplexity, per file and text |
| [built/](built/) | the built files from their headers: fit at every context (`fit.md`), bytes a token by role and type, est. ms, layout, ranking |
| [run/chain.sh](run/chain.sh), [run/build.sh](run/build.sh) | the build |
| [run/score_all.sh](run/score_all.sh) | the KLD runs: [gguf_compose's run/score.sh](../2026-11-04_gguf_compose/run/score.sh) one file at a time, resumable (a done marker per file, a STOP file between files), each file after its build marker |
| [run/stage.sh](run/stage.sh) | a built file staged on the fast disk before its KLD run (shards copied and checked against the build's md5, the table shard linked to the fast copy), so the cards wait on no copy |
| [run/score_direct.sh](run/score_direct.sh) | score.sh's interface without its staging: a file scored where it lies, through symbolic links (its table shard to the fast disk's copy), for a fast disk that must take no writes |
| [run/wait_shared.sh](run/wait_shared.sh) | FN2's `wait_gpus` for cards shared with small jobs: a free-memory floor a card (MIN_FREE_MIB) |
| [run/rig_time.sh](run/rig_time.sh) | the rig timing of a mix against its base: a 64k load with one request, then T(1) … T(8) |

## Provenance and licence

The base files are Swift-Qwen3.8-Flash-Next GSQ-RCO Q2_0 and IQ3_XXS as published (the n-gram table moved alone into
shard 3 by [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py)); the donor is ukisai's
`Swift-1.5-Qwen3.8-Flash-Next-GGUF` Q8_0 (revision `0142554c`), FN1's reference. The tables here are derived from their
headers. The model files are not redistributed here; their licences are the models' own.
