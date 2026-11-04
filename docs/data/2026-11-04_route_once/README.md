# Q860: route once per verify block, a switch for its quality and speed tests (2026-11-04)

Registry rows: [Q860](../../../quants/experiments.md) (route once per verify block: the owner's idea, lossy), Q861–Q864
(what follows from it). The switch is on the fork branch `pascal-p106-v3-b12076-routeonce` (`b3825c215`, two commits on
`-ladder3` `7bc1552f4`), off by default. This folder holds the CPU checks of 2026-11-04 (no GPU) and the scripts for
the two GPU runs that come next: the KL divergence by k on the quant machine and a timing on the rig.

## The switch

`LLAMA_MOE_ROUTE_ONCE=<k>[,max=<n>][,hist=<0|1>][,dry=<0|1>]` (or `llama_set_moe_route_once`): the tokens of a
micro-batch form blocks of up to k consecutive positions of one sequence. A block's first token picks its 10 routed
experts as usual; every later token takes the same 10 experts, ordered by its own router scores, with its own weights
(its own probabilities at those experts, normalized as usual). So a pass over a block reads one token's experts per MoE
layer instead of the union of all k tokens' experts.

| Item | What it does |
|---|---|
| `k` | the block length; 0 (the default) is off, 1 takes the switch's path with every token its own block's first (the router's selection exactly) |
| `max=<n>` | only micro-batches of at most n tokens: a server's verify batches, never its prompts |
| `hist=1` | after each micro-batch, the experts read and those independent routing would pick, per MoE layer, read back to the host: the distinct experts per full block and layer, as read and under independent routing, and how many of a later token's own top-10 lie in its block's set. Logged every 65,536 tokens and when the context is freed (INFO lines, shown at `-lv 4`); a synchronous read, so never for timing |
| `dry=1` | the routing as it is, only counted: the histogram of independent routing |

Blocks restart at a sequence change, a position gap and each micro-batch's start; a layer cropped to the outputs takes
each block's first output. Expert groups, GroveMoE and experts chosen by the caller route as usual.

The same switch does both jobs:

- **Quality:** in `llama-perplexity`'s 512-token micro-batches, blocks of k consecutive positions are exactly what a
  k-token verify pass sees at each of its positions, including the cache entries of the tokens it accepts, so the KL
  divergence by k against FN1's bases ([Flash-Next quality chain](../2026-11-03_flash_next_quality/README.md)) prices
  route once.
- **Speed:** in a real k-token verify pass, the multi-token expert kernel (R7) reads each distinct expert once, so with
  route once a pass reads 10 experts per layer instead of the k tokens' union.

In the graph, each token's block-leader index is an input; per MoE layer the switch adds a `get_rows` of the leader's
router scores, their top-10, the token's own scores there, a sort of those 10 and a `get_rows` of the ids, and it keeps
the CUDA backend's fused top-k kernel from matching, so the router's softmax, top-k and weight normalization run as
separate kernels. That graph-level cost is what the rig timing's `k1` arm measures.

## CPU checks (quant machine, no GPU)

`run/cpu_checks.sh` on the branch's CPU build: `test-route-once` (a new test, run by hand on a model file) on random
qwen4exp models from `test-llama-archs` (4,096 tokens, 32 routed experts, 8 used; F32 head and Q5_K head; routers as
they are and scaled by 30), `test-llama-archs` on every architecture against the `-ladder3` base build, and
`llama-perplexity` on Swift-1.5 GSQ-RCO Q2_0.

**Routing, from the graph's own tensors** ([cpu_unit.tsv](cpu_unit.tsv)): `test-route-once` reads each MoE layer's router
probabilities, the experts it reads in slot order and the normalized weights through the eval callback, and checks every
row against its block leader: 223 checks, 0 failed, on each of the four models. On 16 prompts of 8–64 tokens in three
layouts (one sequence; the odd positions as outputs, so the last layer is cropped to them; two sequences in one
micro-batch, so the blocks restart at the second):

| Arm | Logits against off | Later rows | Rows with experts other than the leader's top-10, out of their own order, or with wrong weights | Later rows whose experts differ from their own top-k |
|---|---|---|---|---|
| k = 1, with and without the histogram | identical, every layout and model | 0 | 0 | 0 |
| k = 5, dry run | identical | 0 | 0 | 0 |
| k = 5, `max=4` (no micro-batch that short) | identical | 0 | 0 | 0 |
| k = 2, 3, 5 and 64 (the whole micro-batch) | differ (nmse 2.9e-7 to 7.6e-7: a random model's routed experts add little to its logits) | 255–1,112 per layout | 0 | all of them |

So every token's experts are its block leader's experts, a subset of exactly one token's top-k, in its own order with
its own weights, and the switch changes the logits only when it routes. The histogram's counts on the same runs
([cpu_unit_hist.tsv](cpu_unit_hist.tsv)): every full block reads 8 distinct experts (the leader's 8 of 32), no later
token reads outside its block's set, and independent routing would read 13.95 per block at k = 2 and 24.26 at k = 5
(a random router's union: 32 · (1 − (24/32)^5) = 24.4).

**Every architecture against the `-ladder3` base** ([cpu_identity.tsv](cpu_identity.tsv)): `test-llama-archs -b CPU`
writes each architecture's logits with the base build, and this build compares; all 143 compare.

| `LLAMA_MOE_ROUTE_ONCE` | the test's own experts (most MoE architectures 2, both used) | 8 experts, 2 used |
|---|---|---|
| unset, `0`, `1`, `1,hist=1`, `5,dry=1,hist=1`, `5,max=1` | 0 of 143 differ | 0 of 143 differ (unset, `1`, the dry run) |
| `5` | 1 differs: clover_moe, the one with more experts (4) than it uses | 27 differ, all MoE: llama4, grok, qwen2moe, qwen3moe, qwen3next, qwen3vlmoe, qwen35moe, clover_moe, qwen4exp, phimoe, minicpm, maple, cohere2moe, dbrx, olmoe, arctic, deepseek, granite, granitemoe, bailingmoe, hunyuan-moe, gpt-oss, smallthinker, mistral-large4, maion-coder, mellum, minimax-01 |

The other MoE architectures stay the same at k = 5: GroveMoE and the expert-group ones (beam) route as usual by design,
and in the random models of flint and deepseek2 every token picks the same 2 experts (their dry runs count 2.00
distinct experts per 5-token block), so routing once has nothing to change. (The tests exit 1 in the base run too: architectures that share a logits file compare
against each other there; those lines, repeated word for word, are not counted as differences.)

**Swift-1.5 GSQ-RCO Q2_0 on the CPU** ([real_cpu.tsv](real_cpu.tsv)): `llama-perplexity` over the first 4 × 512 tokens
of the agent text; `off` writes its logits and every other arm is compared with them, so the KL divergence is what the
switch itself adds (12 threads, about a minute an arm):

| Arm | Mean KLD against off | Median KLD | Same top token | Perplexity (off: 1.3594) | Distinct experts per block and layer, read | Under independent routing | A later token's own top-10 inside its block's set |
|---|---|---|---|---|---|---|---|
| k = 1 | 0 | 0 | 100 % | 1.3594 | 10 | 10 | — |
| dry, k = 2 | 0 | 0 | 100 % | 1.3594 | 16.32 | 16.32 | 3.68 |
| dry, k = 3 | 0 | 0 | 100 % | 1.3594 | 21.60 | 21.60 | 3.38 |
| dry, k = 4 | 0 | 0 | 100 % | 1.3594 | 26.42 | 26.42 | 3.07 |
| dry, k = 5 | 0 | 0 | 100 % | 1.3594 | 30.86 | 30.86 | 2.93 |
| on, k = 2 | 0.288 | 0.007 | 89.4 % | 1.614 (× 1.19) | 10 | 16.07 | 3.93 |
| on, k = 3 | 0.531 | 0.027 | 82.4 % | 2.127 (× 1.56) | 10 | 21.04 | 3.61 |
| on, k = 4 | 0.720 | 0.071 | 76.9 % | 2.513 (× 1.85) | 10 | 25.51 | 3.33 |
| on, k = 5 | 0.898 | 0.112 | 74.5 % | 3.081 (× 2.27) | 10 | 29.56 | 3.12 |

k = 1 and the dry runs reach the comparison's floor (maximum KLD 4.7e-5 in each: the base file stores the logits in 16
bits), so k = 1 picks exactly the router's experts on the real model too.

- **Consecutive tokens share few experts.** A token keeps 3.7 of its 10 experts in the previous token's set (13.7 % of
  the pairs share none), and the tokens 1–4 positions after a block's first keep 2.9 on average. So routing once
  replaces 6–7 of every later token's 10 experts.
- **The quality cost is large.** At k = 2 the switch adds a KLD of 0.29 against the same Q2_0 without it, about as much
  as the whole Q2_0 quantization costs against the Q8_0 reference on the agent text (0.285, 90.7 % same top token, in
  the [quality chain](../2026-11-03_flash_next_quality/README.md)), and it grows to 0.90 at k = 5. The median is far
  lower (0.007 at k = 2): most positions barely move, and some move a lot. Here every token of the context is routed in
  blocks, the prompt's included; a server with `max=` routes only its verify batches that way, so a long agent prompt
  keeps its cache exact, but every verified token still loses most of its own experts.
- **The reads it saves.** Independent routing reads 16.3, 21.6, 26.4 and 30.9 distinct experts per layer for 2–5
  consecutive tokens (62 % of k × 10 at k = 5, below the 34–38 estimated in the [MoE speculation note](../../research/notes/moe_speculation_union.md)), and route once
  reads 10: at k = 5 a third of them.

**Strict route once is not usable on Swift.** The softer forms keep more of each token's own experts and read more than
10: a few own experts per token (Q874), a wider set from the block's first token (Q863), a set pooled over the block
(Q864); their KL divergence by k decides, with the rig timing below as the bound on what any of them can save.

## Next: the KL divergence by k and the rig timing

- **The KL divergence on the GPUs** (`run/kld_route_once.sh`): the same arms on FN1's bases (wiki, agent and code
  against the Q8_0 reference) with the sm_86 build, after `test-route-once -d CUDA0`. It checks the CUDA path (`k1`
  within rounding of `off`, since the switch-off graph takes CUDA's fused router kernel and the switch's does not) and
  puts the CPU result on the quality chain's scale; `ARMS="off k1 k2"` covers both, the full curve is the default.
- **The rig timing** (`run/rig_timing.sh`): T(k) for k = 1–5 with the switch off, at k = 1 (the switch's graph alone)
  and on, beside [the k-curve](../2026-11-04_kcurve_p106/README.md). `on` minus `off` is the most any route-once form
  can save (a pass reading 10 experts per layer instead of the union); `k1` minus `off` is the cost of doing it in the
  graph, which a fused router kernel would remove (Q861).

## Files

| File | What it is |
|---|---|
| [cpu_unit.tsv](cpu_unit.tsv) | `test-route-once` per model, router scale, arm and layout (one sequence, odd outputs so the last layer is cropped, two sequences in one micro-batch): logits against off, their nmse, rows checked, later rows, rows whose experts are not the leader's own top-k, out of their own order or with wrong weights, later rows whose experts differ from their own top-k |
| [cpu_unit_hist.tsv](cpu_unit_hist.tsv) | the histogram's counts per model and arm: full blocks × layers, distinct experts read and under independent routing, later token-layers reading outside their block's set |
| [cpu_identity.tsv](cpu_identity.tsv) | `test-llama-archs -b CPU` per arm against the base build's logits: architectures compared and those that differ, with the test's own expert counts and with 8 experts, 2 used |
| [real_cpu.tsv](real_cpu.tsv) | Swift Q2_0 on the CPU, 4 × 512 tokens of the agent text: KL divergence against the switch off, same top token, perplexity, distinct experts per block and layer (read, independent), the later tokens' own experts inside their block's set |
| [run/cpu_checks.sh](run/cpu_checks.sh) | the CPU checks: build, unit, identity, real |
| [run/build_gpu.sh](run/build_gpu.sh) | the rig tarball and `test-route-once` for the P106s; `llama-perplexity` and `test-route-once` for Ampere cards |
| [run/kld_route_once.sh](run/kld_route_once.sh) | the KL divergence by k on the quant machine's GPUs, after a CUDA run of `test-route-once` |
| [run/table_route_once.sh](run/table_route_once.sh) | the table of the KLD logs (or the real step's) |
| [run/rig_timing.sh](run/rig_timing.sh) | T(k) on the rig with the switch off, at k = 1 and on |

## Run

The CPU checks and the builds, on a build machine with Docker (`SRC` a checkout of the fork branch, `BASE_BIN` a CPU
build of `-ladder3` with `test-llama-archs`; every binary runs in a container of the build image):

```bash
WORK=<work dir> SRC=<work dir>/src BASE_BIN=<-ladder3 CPU build>/bin THREADS=12 JOBS=6 \
  STEPS="build unit identity real" CHUNKS=4 MODEL=<Swift Q2_0 first shard> TEXT=<agent-heldout.txt> \
  MOUNTS="<the model's folder> <the text's folder>" bash run/cpu_checks.sh > checks.log 2>&1
WORK=<work dir> REPO=<this repository> SRC=<work dir>/src bash run/build_gpu.sh > build_gpu.log 2>&1
OUT=<work dir>/checks/real bash run/table_route_once.sh > real_cpu.tsv
```

The KL divergence by k, on the quant machine's GPUs after the quality chain (its `kld.sh` bases and environment:
`BASES`, `WIKI`, `AGENT`, `CODE`, `MOUNTS`):

```bash
PPL=<work dir>/cuda86/bin/llama-perplexity TEST=<work dir>/cuda86/bin/test-route-once RANDOM_MODELS=<work dir>/random \
  BASES=<the bases> OUT=<work dir>/kld MODEL=<Swift Q2_0 first shard> GPUS=<the cards> \
  WIKI=... AGENT=... CODE=... MOUNTS=... TAG=routeonce bash run/kld_route_once.sh > kld.log 2>&1
OUT=<work dir>/kld bash run/table_route_once.sh > kld_route_once.tsv
```

The rig timing, on the rig, idle, from a folder with the tarball unpacked, the extra (`extra-<commit>/test-route-once`),
the two random models and S21's `verify_cost.sh` with `mem_guard.sh` and `smoke_request.json` in `run/`:

```bash
BIN=<-routeonce build>/bin MODEL=<Swift Q2_0 first shard> OUT=$PWD/out VC=$PWD/run \
  TEST=$PWD/extra-<commit>/test-route-once RANDOM_MODELS=$PWD/random \
  setsid nohup bash run/rig_timing.sh > rig_timing.out 2>&1 < /dev/null &
python3 -I ../2026-11-04_kcurve_p106/run/tk.py off=out/off/bench-wall.json k1=out/k1/bench-wall.json \
  on=out/on/bench-wall.json > tk_route_once.tsv
```

## Provenance and licence

The random models are written by the branch's `test-llama-archs` (random weights, no data). Swift-1.5 GSQ-RCO Q2_0 is
[ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
under the Swift Open License 1.0 and qwen-community-1.0; the agent text is the held-out sessions' text of
[2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md) (`run/agent_text.py` there). The tables hold only counts,
scores and statistics.
