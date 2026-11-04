# Cold-tier Flash-Next files: the rarely picked experts at fewer bits instead of cut out (2026-11-04)

Registry row Q238 ([quants/experiments.md](../../../quants/experiments.md)). Qwen3.8-Flash-Next picks 10 of 512 routed
experts per layer, and on agent text half of them carry ~91 % of the picks
([2026-11-04_expert_usage](../2026-11-04_expert_usage/README.md)). Cutting the least picked ones out
([2026-11-04_gguf_prune](../2026-11-04_gguf_prune/README.md)) keeps agent and code quality but loses general knowledge:
Swift IQ3_S cut to 464 experts scores a wikitext KL divergence of 0.446 against the uncut file's 0.153 (Q891). A cold
tier keeps those experts, in a lower-bit type: each layer holds its experts in two sets of tensors, the hot ones in the
file's type and the cold ones in their own, and the router still scores all 512.

This folder holds the engine change (Miner Llama's `-coldtier` branch), the tool that writes such files
([scripts/gguf_tier.py](../../../scripts/gguf_tier.py)), their checks, and five donor files for the composed file
(Q930): each is `pmix-q2-fast` (Swift Q2_0's experts, every other weight in the P106's fastest type;
[2026-11-04_pascal_mix](../2026-11-04_pascal_mix/README.md)) with only its experts changed, so its KL divergence
prices one expert lever alone. Everything here ran on a CPU, except a two-minute check on one RTX 3090; no rig.

## Result

**The engine change is exact and the five donors are built.** The KL divergence against the Swift-1.5 Q8_0 on the
agent, code and wiki texts is the next step (on GPUs; the command is below). A CPU preview against the uncut IQ3_S
shows the hot tier working and the IQ1_M cold tier free on agent text.

| File (`pmix-q2-fast` with only the experts changed) | Hot tier | Cold tier | Weights on the cards | Weight bytes a token reads | Cold tiers' share of agent routings |
|---|---|---|---|---|---|
| `pmix-q2-fast` (the base) | 512 × Q2_0 | — | 37.47 GB | 4.160 GB | — |
| `t256iq3` | 256 × IQ3_XXS | 256 × Q2_0 | 41.94 GB | 4.319 GB | 9.24 % |
| `t320iq3` | 320 × IQ3_XXS | 192 × Q2_0 | 43.06 GB | 4.327 GB | 4.54 % |
| `t256iq3s` | 256 × IQ3_S | 256 × Q2_0 | 45.63 GB | 4.451 GB | 9.24 % |
| `t256iq3c1` | 256 × IQ3_XXS | 256: gate/up IQ1_M, down Q2_0 | 39.42 GB | 4.310 GB | 9.24 % |
| `t256c1` | 256 × Q2_0 | 256: gate/up IQ1_M, down Q2_0 | 34.95 GB | 4.151 GB | 9.24 % |
| `pmix-iq3-fast` (reference) | 512 × IQ3_XXS | — | 46.80 GB | 4.721 GB | — |
| Swift IQ3_S cut to 464 (C330) | 464 × IQ3_S | — (cut) | 49.86 GB | 5.261 GB | — |

([bytes.tsv](bytes.tsv); weights on the cards = every tensor but the token embeddings and the n-gram table; a token
reads the dense weights whole and 10 experts per layer, the cold ones as often as Q232 counted them.)

- **The hot 256 carry 90.8 % of the agent routings, so a tier changes the bytes a token reads far less than the file:**
  IQ3_XXS hot experts add 4.47 GB to the cards but only 0.16 GB (3.8 %) to each token, and an IQ1_M cold tier takes
  2.52 GB off the cards for 9 MB a token. `t256iq3` is 4.86 GB lighter than `pmix-iq3-fast` (which fits nine 6 GB
  cards at 64k only at `-ub 64`) and reads 8.5 % less a token; `t256c1` is the smallest file here.
- **The dense weights are most of every token** (3.50 of 4.16 GB in `pmix-q2-fast`): a cold tier is a lever on fit and
  quality, hardly on speed.
- **The CPU preview** (4 × 512 held-out agent tokens against the uncut Swift IQ3_S's logits, C330's base; a small
  sample, and IQ3_S is not the Q8_0 reference): the IQ3_XXS hot tier moves the output a third of the way to IQ3_S, and the IQ1_M cold
  tier costs nothing measurable on agent text ([cpu_kld.tsv](cpu_kld.tsv)):

  | File | KL divergence from the uncut IQ3_S | Same top token |
  |---|---|---|
  | `pmix-q2-fast` | 0.432 ± 0.045 | 90.6 % |
  | `t256iq3` | 0.283 ± 0.034 | 92.2 % |
  | `t256iq3c1` | 0.287 ± 0.033 | 92.5 % |
  | `t256c1` | 0.428 ± 0.045 | 90.6 % |

  Agent text is where the cold experts matter least (9.24 % of its routings); C331's cut lost wiki quality, so the wiki
  text of the GPU run is the test of whether the cold experts' knowledge survives at IQ1_M.
- **As built, a tiered layer costs more than its bytes:** each tier runs all 10 slots and reads its slot 0 (at most
  one more expert per tier and layer: 4.47 against 4.32 GB a token for `t256iq3`). The KL divergence does not depend
  on it; a rig timing should wait for Q941.

## The engine: two tiers in one layer

Miner Llama branch `pascal-p106-v3-b12076-coldtier` (on `-kshape`), off by default: a file without cold tensors loads
and builds the same graph as before (every architecture's logits bit for bit, below).

- **The file:** `blk.N.ffn_{gate,up,down}_exps` (or `ffn_gate_up_exps`) hold the layer's hot experts and
  `blk.N.ffn_{gate,up,down}_exps_cold` the cold ones, each set in its own type, with per-expert `.scale` /
  `.input_scale` per tier where a type has them. `blk.N.ffn_exps_tier` (F32, one value per expert id) maps an expert to
  its slot: `s` for slot `s` of the hot tensors, `n_hot + s` for slot `s` of the cold ones. `<arch>.expert_count` stays
  512, the router `ffn_gate_inp` is unchanged, and each layer has its own split (a layer can stay untiered).
- **The graph** (`build_moe_ffn`): the router's softmax and top-10 run as before. The tier map turns each pick into a
  hot and a cold index; each tier runs the expert FFN over its own tensors, and a select (`GET_ROWS` over
  [hot | cold]) takes each slot's output from its tier before the router weights apply and the slots are summed as
  before. A pick of the other tier reads that tier's slot 0, which the select drops; no output is multiplied by zero, so
  nothing from the unused tier can reach the sum.
- **One expert per column:** llama.cpp's CUDA `MUL_MAT_ID` kernels group a token's columns by expert and assume each
  expert at most once per token (top-k routing guarantees it). Reading slot 0 for several picks breaks that, and the
  first version of the branch made an illegal memory access on an RTX 3090 while the CPU, whose kernels count every
  column, was exact. So a tiered layer's products run each slot as a column of its own with one expert (the token's
  input repeated over its 10 slots), and no id can repeat. `test-moe-ids` checks the property on the CPU.
- **What it costs as built:** each tier runs over all 10 slots, so a tiered layer does twice the expert products of an
  untiered one, and each tier also reads its slot 0. That is fine for KL divergence runs; for a rig timing the kernels
  should skip the other tier's slots instead (Q935).
- Loading: `create_tensor_exps_cold` (qwen4exp calls it; other architectures can), the hot tensors' expert count read
  from the file. The MoE cache skips tiered layers (its slots are keyed by router id), and a tiered layer may not have
  per-expert biases.

## The tool

`gguf_tier.py` (standard-library Python, on `gguf_reshard.py`'s reader and writer):

- `split --experts <usage table>... --hot N|<a>-<b>:N,... [--force-hot ids.tsv]`: each layer's N most picked experts
  are hot (most picked first, so a tier's slot 0 is its busiest expert), the rest cold. Several usage tables (agent,
  wiki and code text) rank by the sum of their shares (`--weights`); a spec gives 8-layer groups their own N or leaves
  layers untiered; `--force-hot` keeps given ids hot (e.g. the tool-call experts of C336).
- `extract --ids cold.tsv --imatrix <imatrix.gguf>`: the cold experts' slabs of a file (here the Swift-1.5 Q8_0) as a
  GGUF of their own, and the importance matrix cut to the same experts, for `llama-quantize --allow-requantize
  --tensor-type ...` (a quantized default type is needed for `--tensor-type` to apply).
- `tier`: the new file. Every tensor of the base byte for byte, but each listed layer's expert tensors cut to the hot
  slabs (from the base or `--hot`, another quant of the same model) with the cold tensors after them (from `--cold`;
  `--cold-only` takes only matching tensors from it, the rest from the base or `--cold-rest`), the tier map after the
  router, and `split.tensors.count` raised. Header copies (`--headers-only`) for
  [scripts/fit_probe.sh](../../../scripts/fit_probe.sh); a shard with nothing to change can stay a link.
- `verify`: each new shard against its sources, slab for slab.

## Checks

| Check | Result |
|---|---|
| `scripts/gguf_tier_test.py`: a three-shard synthetic qwen4exp model (8 experts in IQ3_S, Q8_0, Q6_K, Q2_0, F32 and F16; separate and fused gate/up; `.scale` and `.input_scale`; a `.bias` layer; padding), a synthetic importance matrix | ok: split ranks with ties to the lower id, several tables by summed shares, per-layer counts and forced ids; tier writes the bytes the test's own writer makes, from the base, from an extract, from files in other types and orders, with `--cold-only` / `--cold-rest`; header copies match; verify passes, and fails on one changed byte in a cold slab, a hot slab, a tier map or another tensor; overlapping or incomplete tiers, too few hot experts, expert biases and short id lists are refused |
| The self-test against eight planted bugs in the tool (tier map off by one, cold slabs in id order, ties to the higher id, forced ids ignored, `--cold-only` ignored, `split.tensors.count` not raised, importance counts not cut, verify skipping the slabs) | each fails the test |
| `test-llama-archs -b CPU` on every architecture, this branch against `-ladder3`'s build (no CPU-side change between it and the branch's parent) | the logits of all 143 architectures bit for bit; the same 7 of 148 tests fail in both builds |
| Random qwen4exp models (`test-llama-archs`: 32 experts, 8 used, 2 layers): F32, its Q4_0-expert copy, a sharp one (weights at stdev 0.6), each tiered into two tiers of one type at hot 24, 31 and 8 (= experts used) | `verify` ok; `llama-perplexity` writes the same KL base file byte for byte (every token's log-probabilities over the vocabulary), with `--no-repack` (x86's repacked Q4_0 kernels round by how many tokens an expert gets: perplexity 4104.52 against 4104.34 with them) |
| The negative control: cold slabs taken from the wrong experts | differs: on the sharp model KLD 1.15, the same top token 21 % |
| F32 hot experts with Q4_0 cold ones (extract, llama-quantize, tier) | loads, runs, differs from the F32 and the all-Q4_0 models |
| `test-moe-ids` (this branch's `tests/`) on every random model, untiered and tiered | every `MUL_MAT_ID`'s ids distinct per token and below its expert count (48 nodes, 2,336 one-expert columns per tiered model); the checker flags the first graph's id pattern |
| One RTX 3090, the CUDA build: the sharp model's tiered copy against its own logits on the card, at `-ub 128` (the batched kernels) and `-ub 1` (the one-token kernels) | KLD 8 × 10⁻⁶ (largest 5 × 10⁻⁵) and 0 (largest 1 × 10⁻⁵): rounding; the negative control 1.15; `test-moe-ids` on the card clean; no Xid in the kernel log |
| The first graph (a tier's slot 0 read for every pick of the other tier) on the same card | an illegal memory access in the first batch (an Xid 31 MMU fault); the fix is above |
| `gguf_tier.py verify` of each donor against its sources (`pmix-q2-fast`, the hot and the cold file) | all five pass: every dense tensor identical to `pmix-q2-fast`'s, every hot and cold slab equal to its source slab, the tier maps right; shard 3 is the base's own table ([checksums.md5](checksums.md5)). The first verify of two files was stopped and run again: it read the linked 28.8 GB table twice from a busy disk, which `verify` now skips ([files.tsv](files.tsv)) |
| The real files through the engine on the CPU (one chunk of 512 agent tokens) | `t256iq3`, `t256iq3c1` and `pmix-q2-fast` load and run; a real same-type tier (`pmix-q2-fast` with layers 0–3 tiered into its own Q2_0 experts) gives `pmix-q2-fast`'s perplexity to every printed digit, 2.0186 ± 0.21945 |
| The routers of the source files (Swift Q2_0, IQ3_XXS, IQ3_S and `pmix-q2-fast`), layers 0, 23 and 47 | the same router row for the same expert id in every file (cosine 1.0000; the next id 0.09–0.42): the experts line up, so a tier can take them from any of the files |

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | one verdict line per check of `run/cpu_checks.sh` and `run/cuda_check.sh` |
| [files.tsv](files.tsv) | one verdict line per step of `run/files.sh` (split, extract, quantize, tier, verify, header copies) |
| [bytes.tsv](bytes.tsv) | per file: tensor bytes without the embedding tables, the dense and expert bytes a token reads, the cold tiers' share of the routings |
| [checksums.md5](checksums.md5) | md5 of every written shard |
| [split/](split/) | `hot_h256.tsv`, `cold_h256.tsv`, `hot_h320.tsv`, `cold_h320.tsv`: each layer's tiers (the `--ids` format) |
| [run/](run/) | `build.sh` (the three builds), `cpu_checks.sh`, `cuda_check.sh`, `files.sh` (every file step), `bytes.py` |

The tool and its self-test are [scripts/gguf_tier.py](../../../scripts/gguf_tier.py) and
[scripts/gguf_tier_test.py](../../../scripts/gguf_tier_test.py).

## Commands

The tool alone (any machine with the files; `tier` and `verify` read and write each slab once):

```bash
python3 scripts/gguf_tier.py split --experts docs/data/2026-11-04_expert_usage/experts.tsv.gz --hot 256 --out-dir split
python3 scripts/gguf_tier.py tier --hot-ids split/hot.tsv --cold-ids split/cold.tsv --hot <Swift IQ3_XXS shards> \
  --cold <Swift Q2_0 shards> --out-dir <dir> --prefix <name> --link-unchanged <pmix-q2-fast shards>
python3 scripts/gguf_tier.py verify <pmix-q2-fast shards> --new <dir>/<name>-0000?-of-00003.gguf \
  --hot-ids split/hot.tsv --cold-ids split/cold.tsv --hot <Swift IQ3_XXS shards> --cold <Swift Q2_0 shards>
python3 scripts/gguf_tier_test.py /tmp/tier-test   # prints "ok"
```

A cold tier in a new type: `extract` the cold experts and the importance matrix, quantize them, take gate and up from
the result and keep the base's down projections:

```bash
python3 scripts/gguf_tier.py extract --ids split/cold.tsv --imatrix <imatrix.gguf> --out-dir cold <Swift-1.5 Q8_0 shards>
llama-quantize --allow-requantize --imatrix cold/imatrix.gguf --tensor-type 'ffn_(gate|up)_exps=iq1_m' \
  --tensor-type 'ffn_down_exps=q8_0' cold/experts.gguf cold/experts-c1.gguf Q8_0 8
python3 scripts/gguf_tier.py tier --hot-ids split/hot.tsv --cold-ids split/cold.tsv --cold cold/experts-c1.gguf \
  --cold-in cold/ids.tsv --cold-only 'ffn_(gate|up|gate_up)_exps' --cold-rest <Swift Q2_0 shards> \
  --hot <Swift IQ3_XXS shards> --out-dir <dir> --prefix <name> --link-unchanged <pmix-q2-fast shards>
```

Everything in this folder (CPU, except `cuda_check.sh`; the builds ~1 h with ccache, the checks ~15 min, the files
~2 h on a busy disk):

```bash
WORK=<build folder with a llama.cpp clone in src/, the fork as remote "fork"> REPO=<this repository> \
  LOCK=<build lock> DOCKER_ARGS="--cpu-shares=256" bash docs/data/2026-11-04_cold_tier/run/build.sh
WORK=<work> REPO=<this repository> BIN=<build>/cpu/bin BASE_BIN=<a -ladder3 CPU build>/bin TEXT=<agent-heldout.txt> \
  CPUS=4-7,16-19 THREADS=8 bash docs/data/2026-11-04_cold_tier/run/cpu_checks.sh
WORK=<work> BIN=<build>/sm86/bin TEXT=<agent-heldout.txt> GPU=<card UUID> bash docs/data/2026-11-04_cold_tier/run/cuda_check.sh
REPO=<this repository> WORK=<scratch> OUT=<output folder> QUANTIZE=<build>/cpu/bin/llama-quantize Q8=<Q8_0 shard 1> \
  IMATRIX=<imatrix.gguf> EXPERTS=docs/data/2026-11-04_expert_usage/experts.tsv.gz COLD_REST=<Swift Q2_0 shard 1> \
  FILES="<folder>|<pmix-q2-fast shard 1>|<prefix>|<N>|<cold: c1, base or a file's shard 1>|<hot file's shard 1> ..." \
  REFS="<name>=<shard 1> ..." MOUNTS="<folders>" CPUS=8-11,20-23 bash docs/data/2026-11-04_cold_tier/run/files.sh
```

The KL divergence of the donors (Q930's way, on the bases of
[2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)) uses
[2026-11-04_gguf_prune/run/kld.sh](../2026-11-04_gguf_prune/run/kld.sh) with `PPL` set to this branch's CUDA
`llama-perplexity` (a build without the branch refuses the cold tensors) and `FILES` listing the donors, then
`pmix-q2-fast` itself as the same-engine control.

## Provenance

The donor files are made from
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF)
(Q2_0, IQ3_XXS and IQ3_S experts byte for byte) and ukisai's Swift-1.5 Q8_0 (the IQ1_M cold experts, requantized with
the importance matrix of [2026-11-04_expert_usage](../2026-11-04_expert_usage/README.md)), under the Swift Open
License 1.0 and qwen-community-1.0; the files themselves are not published here. The ranking is that folder's routing
counts on publishable agent sessions. The random models come from Miner Llama's `test-llama-archs`. Tables and scripts
of this repository (MIT).
