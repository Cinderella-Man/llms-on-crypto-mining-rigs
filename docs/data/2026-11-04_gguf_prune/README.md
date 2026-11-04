# Usage-pruned Flash-Next files: Swift IQ3_S with 464, 448 and 416 experts per layer (2026-11-04)

Registry rows Q890 (the tool and the files) and Q891 (their KL divergence) ([quants/experiments.md](../../../quants/experiments.md)).
[2026-11-04_expert_usage](../2026-11-04_expert_usage/README.md) counted how often each of Qwen3.8-Flash-Next's 512
routed experts per layer is picked on agent text and found that keeping each layer's busiest 464 loses 0.22 % of the
routings, and that Swift-1.5 GSQ-RCO IQ3_S, the best file that does not fit one rig's nine 6 GB cards, fits them at 64k
context with 464 experts per layer (448 with the DFlash2 draft, 416 on whole layers). This folder holds the tool that
makes such a file, [scripts/gguf_prune.py](../../../scripts/gguf_prune.py), its checks, the three cut files' sizes and
checksums, and what the cuts cost: a first look on a CPU, then the KL divergence of every file on five GPUs. The cuts and
their checks ran on a CPU; no rig was used.

## Result

**A usage cut keeps agent and code quality and loses general knowledge.** Q891 scored the uncut Swift IQ3_S and its three
cuts against the Swift-1.5 Q8_0 reference's logits on the quality chain's three texts
([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md): held-out agent sessions 12 × 8,192 tokens,
Exercism code 80 × 512, wikitext-2 80 × 512), all four on the same five GPUs with the same flags; Swift IQ3_XXS, today's
one-rig file, is that chain's published score. Mean KL divergence (± its error) and the share of tokens whose top choice
stays the same, [kld.tsv](kld.tsv):

| File | Agent KLD | Same top | Wiki KLD | Same top | Code KLD | Same top |
|---|---|---|---|---|---|---|
| IQ3_S, uncut | 0.118 ± 0.003 | 93.6 % | 0.153 ± 0.003 | 87.0 % | 0.032 ± 0.002 | 98.0 % |
| IQ3_S, 464 experts | 0.123 ± 0.003 | 93.8 % | 0.446 ± 0.006 | 78.0 % | 0.034 ± 0.002 | 98.0 % |
| IQ3_S, 448 experts | 0.122 ± 0.003 | 93.6 % | 0.551 ± 0.007 | 75.9 % | 0.034 ± 0.001 | 98.0 % |
| IQ3_S, 416 experts | 0.131 ± 0.003 | 93.5 % | 0.724 ± 0.009 | 71.7 % | 0.043 ± 0.002 | 97.7 % |
| IQ3_XXS (published score) | 0.176 ± 0.004 | 92.3 % | 0.248 ± 0.004 | 83.3 % | 0.054 ± 0.002 | 97.5 % |

- **On agent and code text the cuts cost almost nothing.** Against the uncut file they add 0.005, 0.004 and 0.012 on
  agent text and 0.002, 0.002 and 0.011 on code, and stay 0.045–0.053 and 0.011–0.020 below IQ3_XXS. The median token
  does not move (agent 0.0008 in all four files, code 0.00004) and the share of unchanged top tokens stays at the uncut
  file's; the added mean sits in the tail, where the 99th percentile rises from 2.40 to 2.57–2.69 on agent text and
  from 0.79 to 0.87–1.03 on code ([kld_stats.tsv](kld_stats.tsv)).
- **On wiki text they cost a lot, and everywhere in the text.** The mean rises from 0.153 to 0.446, 0.551 and 0.724,
  above IQ3_XXS's 0.248 at every cut; the median token's KLD rises from 0.034 to 0.112, 0.147 and 0.222, the unchanged
  top tokens fall from 87.0 % to 78.0, 75.9 and 71.7 %, and the perplexity ratio against the reference rises from 1.04 to
  1.29, 1.42 and 1.66. The experts that agent text rarely picks, which a ranking counted on agent sessions drops, carry
  the general knowledge that the wiki text asks for.
- **What a cut file is for:** a coding-only line. At 464 experts it is 4.74 GB smaller than the uncut file, keeps its
  agent and code scores, and fits the rig's nine cards at 64k by Q232's plan where the uncut file does not
  ([cuts.tsv](cuts.tsv); the rig's own fit is Q902); it is no replacement for IQ3_XXS where the model is asked anything
  else. 448 and 416 save 1.6 and 4.7 GB more and cost more on every text, so 464 is the cut to take further. A ranking
  that includes wiki routings (Q892) and a low-bit tier for rarely used experts instead of dropping them (Q238) are the
  ways to keep the knowledge.
- **The control is the same file on the same cards.** The uncut file scored again here reads 0.118 / 0.153 / 0.032
  against 0.121 / 0.153 / 0.032 when the quality chain scored it on other cards, so the published IQ3_XXS scores are
  comparable to within about 0.003. The cuts rank experts by routing counts alone; whether they keep the experts a tool
  call needs is a separate check (Q897, Q937).

## The cut files

**Three cut copies of Swift IQ3_S, exact and loadable,** in the layout the rig uses (shards 1–2 cut, shard 3 the
unchanged n-gram table). Each keeps every layer's most-routed experts on Q232's agent text ([cuts.tsv](cuts.tsv)):

| Kept per layer | Tensor data, shards 1–2 | Cut | Agent routings kept, all layers (worst layer) | Fits nine cards at 64k (Q232's plan, smallest free card) | CPU KL divergence against the uncut file | Same top token |
|---|---|---|---|---|---|---|
| 512 (uncut) | 54.93 GB | — | 100 % | no | — | — |
| 464 | 50.19 GB | −4.74 GB (8.6 %) | 99.78 % (99.46 %) | experts moved, `-ub 128`: 153 MiB | 0.060 ± 0.015 | 97.5 % |
| 448 | 48.61 GB | −6.32 GB (11.5 %) | 99.61 % (98.89 %) | experts moved, `-ub 256`: 165 MiB; with DFlash2, `-ub 128`: 211 MiB | 0.044 ± 0.012 | 97.4 % |
| 416 | 45.46 GB | −9.48 GB (17.3 %) | 99.07 % (97.20 %) | whole layers, `-ub 256`: 172 MiB; with DFlash2, `-ub 128`: 299 MiB | 0.056 ± 0.014 | 96.9 % |

- **The CPU column is a small sample** ([cpu_kld.tsv](cpu_kld.tsv)): the KL divergence of each cut from the uncut file
  on 4 chunks of 512 tokens of the held-out agent text (1,024 scored tokens), almost all of it in the first chunk
  (0.17–0.23; the later three average 0.004–0.005). The median token does not change (KLD 4–6 × 10⁻⁶); the mean comes
  from the 1 % of tokens above a KLD of about 1. The scores of the full texts above are the measure: on 12 × 8,192 agent
  tokens the cuts add 0.004–0.012 to the uncut file's KLD against the Q8_0 reference, not 0.044–0.060.
- The routings a cut drops (0.22 % at 464) understate what it changes: a token that picked a cut expert in any of 48
  layers gets the next one on its router's list, and on Q232's text that is about one routing a token (10 picks ×
  48 layers × 0.22 %).

## The tool

`gguf_prune.py prune` keeps N experts in every layer of a GGUF, split or not:

- **Which experts:** `--experts` takes a ranking table (columns `layer`, `expert` and a count, by default `routings`:
  the expert usage folder's `experts.tsv.gz`) and keeps each layer's N largest, ties to the lower id, in ascending id
  order; `--ids` takes one line per layer, `<layer><TAB><id>,<id>,...`, kept in the order given.
- **What changes:** in llama.cpp's layout the expert axis is a tensor's last, outermost dimension, so one expert's part
  of `ffn_gate_exps`, `ffn_up_exps`, `ffn_down_exps` (or a fused `ffn_gate_up_exps`) is a run of whole rows, whole
  blocks of any quantized type. Each kept expert's slab moves unchanged, and so does its router row (`ffn_gate_inp`),
  its `exp_probs_b` entry and its `.scale` / `.input_scale` entries where a file has them. `<arch>.expert_count`
  becomes N. Every other key and tensor is copied byte for byte and stays in its shard; the kept ids go to `kept.tsv`.
- **Why the rest of the model is untouched:** qwen4exp routes with a softmax over all experts, takes the top
  `expert_used_count` (10) and renormalizes their weights (`build_moe_ffn` with `norm_w`), so the softmax's
  denominator cancels: a token whose 10 picks are all kept gets the same output as before (to float rounding), and only
  a token that picked a cut expert changes, to the next expert on its router's list. The cut needs N ≥ 10, and one N in
  every layer, since llama.cpp reads one `expert_count`.
- `--headers-only` writes only the headers, each file sized (sparse) as if its data were there, which is what
  [scripts/fit_probe.sh](../../../scripts/fit_probe.sh) needs; `--link-unchanged` makes a shard with nothing to cut
  (Flash-Next's 28.8 GB n-gram table) a link to its source.
- `gguf_prune.py verify OLD... --new NEW... --ids kept.tsv` checks the new shards against the old: the same keys with
  only `expert_count` changed, the same tensors, types and order, N on the sliced tensors' expert axis, every other
  tensor byte for byte, and every kept slab and router row byte-equal to its source.

Standard-library Python, reusing [scripts/gguf_reshard.py](../../../scripts/gguf_reshard.py)'s GGUF reader and writer.

## Checks

| Check | Result |
|---|---|
| `scripts/gguf_prune_test.py`: a three-shard and a one-file synthetic qwen4exp model (8 experts in Q8_0, Q2_0, IQ3_S, Q6_K, F32 and F16, fused and separate gate/up, `exp_probs_b`, `.scale`, `.input_scale`, tensors whose last dimension is 8 but that are not experts) | ok: all 8 kept writes the same bytes; 5 kept by a ranking with ties, and an id list in its own order, write the bytes the test's own writer makes; header copies match; `verify` passes, and fails on a changed slab byte, a changed other tensor and two swapped ids; too few experts, a repeated id and a missing layer are refused |
| The self-test against planted faults in the tool: the router left unsliced, ties to the higher id, the padding after a sliced tensor dropped, the id list's order ignored | each fails the test |
| Random qwen4exp models (`test-llama-archs`: 32 experts, 8 used, 2 layers; F32 and with a Q5_K head), Miner Llama's CPU build (`-routeonce` `b3825c215`) | every id in order: the same bytes; every id in reverse: `verify` ok and the same perplexity and KLD against the model's own logits as the model itself, to all printed digits (perplexity 4100.697819); reversed slabs with the routers in the old order: a different result (the check sees a mismatch); 24 kept by a random ranking: `verify` ok, loads, KLD computed, `test-route-once` 223 checks, 0 failed |
| Q232's ids apply to this file ([router_order.tsv](router_order.tsv)): the IQ3_S's router rows against the Swift-1.5 Q8_0's, on which Q232 counted | cosine 1.0000 for the same id in layers 0, 7, 23, 35 and 47, 0.09–0.45 for the neighbouring id |
| `gguf_prune.py verify` of each cut against the source shards | all three pass: per file 1,032 tensors byte for byte and 192 sliced (48 layers × gate, up, down and router), every kept slab and router row equal to its source; shard 3 is the source table itself ([checksums.md5](checksums.md5)) |
| The share of Q232's routings each kept set carries ([cuts.tsv](cuts.tsv)) | equal to Q232's `summary.tsv` (99.784, 99.608, 99.075 %) |
| Header copies (`--headers-only`, sparse, checked with `verify --headers-only`) | pass for all three; ~3.2 MB a file packed (`tar -S`), the full sizes when unpacked with `tar -xSzf` |
| The same cut twice (once to the NVMe, once to the HDD) | the same tensor data, 50,194,076,160 bytes |

All of it ran on the CPU at the lowest priority, pinned to 8 threads, the Python steps in a container capped at 8 GB so
that ~150 GB of reads and writes per file did not push other jobs' files out of the page cache.

## Files

| File | What |
|---|---|
| [cuts.tsv](cuts.tsv) | per cut: experts kept, the bytes of the three shards and those saved against the uncut file, and the share of Q232's agent routings and activation energy the kept experts carry (all layers, the worst layer) |
| [checks.tsv](checks.tsv) | one verdict line per check of `run/prune.sh` (self-test, random models, cut, verify, header copies, CPU load) |
| [cpu_kld.tsv](cpu_kld.tsv) | the CPU load: each cut's KL divergence, same-top-token share and perplexity ratio against the uncut file's logits on 4 × 512 tokens of the held-out agent text |
| [kld.tsv](kld.tsv) | Q891: each file's KL divergence against the Swift-1.5 Q8_0 reference on the agent, code and wiki texts (`kld`, `err`, `same_top_pct`, `ppl_ratio`), and the difference from the uncut IQ3_S scored in the same run (`d_uncut`) and from Swift IQ3_XXS's published score (`d_iq3_xxs`: agent 0.175513, wiki 0.247651, code 0.054359) |
| [kld_stats.tsv](kld_stats.tsv) | the same runs' distribution: mean, median, 99th and 99.9th percentile and largest KLD, and the unchanged-top-token share |
| [checksums.md5](checksums.md5) | md5 of every written shard (shard 3, the n-gram table, is the unchanged source shard, md5 `98ce140e641bc5d68af9f31df0f1c048`) |
| [kept/](kept/) | `kept_e464.tsv`, `kept_e448.tsv`, `kept_e416.tsv`: each layer's kept expert ids (the `--ids` format) |
| [run/](run/) | `prune.sh` (every step above), `cuts.py` (the cuts table), `kld.sh` (Q891's KL-divergence run on the GPUs), `kld_stats.sh` (its distribution table) |

The tool and its self-test are [scripts/gguf_prune.py](../../../scripts/gguf_prune.py) and
[scripts/gguf_prune_test.py](../../../scripts/gguf_prune_test.py).

## Commands

The tool alone (any machine with the file; reads and writes each kept byte once):

```bash
python3 scripts/gguf_prune.py prune --keep 464 --experts docs/data/2026-11-04_expert_usage/experts.tsv.gz \
  --out-dir <dir> --prefix Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-e464 --link-unchanged <shard 1> <shard 2> <shard 3>
python3 scripts/gguf_prune.py verify <shard 1> <shard 2> <shard 3> --new <dir>/*-e464-0000?-of-00003.gguf --ids <dir>/kept.tsv
python3 scripts/gguf_prune_test.py /tmp/prune-test   # prints "ok"
```

Everything in this folder (CPU only; ~1 h, most of it disk: three ~50 GB files written and read back twice):

```bash
# SHARDS: the IQ3_S's three shards (the n-gram table as shard 3), comma-separated; BIN: a CPU build of Miner Llama
# (llama-perplexity, test-route-once: ../2026-11-04_route_once/run/cpu_checks.sh's build step) and its random/ folder
WORK=<work folder> OUT=<folder for the cut files> BIN=<cpu build>/bin RANDOM_MODELS=<random/> \
TEXT=<agent-heldout.txt> MOUNTS="<folders the containers need>" SHARDS=<s1,s2,s3> \
NAME=Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S TAG=swift_iq3_s KEEP="464 448 416" \
EXPERTS=docs/data/2026-11-04_expert_usage/experts.tsv.gz CPUS=8-11,20-23 THREADS=8 \
PY_IMAGE=python:3.12-slim-bookworm PY_MEMORY=8g bash docs/data/2026-11-04_gguf_prune/run/prune.sh
```

`TEXT` is the held-out agent text of [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md).

Q891 on the GPUs (five cards, each file split across them by `-fit on` with the bases' `gpu_args`; the four files took
75 minutes), after
the quality chain's bases exist ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)).
`kld.files` lists the files in the order they are scored, as `<name>=<first shard>` lines; the uncut file goes in
beside the cuts so that it is scored again on the same cards as their control:

```
swift-iq3_s-e464=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-e464-00001-of-00003.gguf
swift-iq3_s=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00003.gguf
swift-iq3_s-e448=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-e448-00001-of-00003.gguf
swift-iq3_s-e416=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-e416-00001-of-00003.gguf
```

Every path starts with `FAST`, so each file is read where it is (`fn2.sh` copies a file that is outside `FAST` to `HOT`
first), with the cut files' shard 3 linked to the n-gram table so that all four share it in the page cache.

```bash
# the quality chain's environment (PPL, IMAGE, MOUNTS, WIKI, AGENT, CODE) as in 2026-11-03_flash_next_quality, then:
export TAG=prune GPUS=<five cards> BASES=<the chain's bases folder, with base.{agent,code,wiki}.kld and gpu_args> \
       OUT=<out> HOT=<scratch folder> FAST=<path prefix of the model folder> FILES=<kld.files> TEXTS="agent code wiki" \
       PREFLIGHT=scripts/cuda_preflight.sh PREFLIGHT_CACHE=<folder>
bash docs/data/2026-11-04_gguf_prune/run/kld.sh            # writes $OUT/<name>.<text>.log and $OUT/kld.tsv
OUT=$OUT FILES=$FILES bash docs/data/2026-11-04_gguf_prune/run/kld_stats.sh > kld_stats.tsv
```

`kld.tsv` here adds the two difference columns to the run's `$OUT/kld.tsv` (the uncut row's `kld` and Swift IQ3_XXS's
published score subtracted from each row's).

## Provenance

The cut files are made from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
IQ3_S (Swift Open License 1.0 and qwen-community-1.0), resharded so that its n-gram table is a shard of its own
(`gguf_reshard.py split --isolate '^per_layer_token_embd\.'`; that shard is byte for byte the table shard of every Swift
GSQ-RCO file); the files themselves are not published here. The ranking is the expert usage folder's routing counts on
publishable agent sessions. The random models come from Miner Llama's `test-llama-archs` (route once folder). Tables and
scripts of this repository (MIT).
