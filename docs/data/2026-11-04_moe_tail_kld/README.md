# What dropping a token's low-weight routed experts costs Flash-Next: KL divergence by threshold

Registry rows: [Q618](../../../quants/experiments.md) (fewer routed experts on confident tokens), Q660 and Q661 (the
kernels that would turn the cut into fewer bytes read). The switch and its CPU checks:
[2026-11-04_bytes_switches](../2026-11-04_bytes_switches/README.md).

Qwen3.8-Flash-Next routes each token, in each of its 48 layers, to 10 of 512 experts and sums their outputs by the
router's weights. `LLAMA_MOE_TAIL` (fork branch `pascal-p106-v3-b12076-bytesw`, `fd549b754`) gives a slot weight 0
when the slots ranked above it already hold a share p of the token's routed weight (`p=`), or when its own share is
below w (`w=`); `renorm=1` rescales the kept weights to the token's sum. It works at graph level: the dropped experts'
rows are still read, so this run measures the quality the cut costs, and the experts-kept histogram gives the reads a
kernel-side skip (Q660) would save.

## Results so far (2026-11-04, three cards: two RTX 3080 and an RTX 3090)

The controls and the two mildest cuts have run; p = 0.95, 0.9 (renormalized and plain), 0.9 with a 3 % floor, 0.8,
0.7 and the 5 % floor run next, beside a repeat of `off` (the run-to-run spread). Paired change: against `p1`, the
same graph with nothing dropped; the error is the paired standard error over the chunks.

| Arm | `LLAMA_MOE_TAIL` | agent KLD (change) | wiki KLD (change) | code KLD (change) | experts kept (of 10) | routed reads left |
|---|---|---|---|---|---|---|
| `off` | unset | 0.2869 | 0.4289 | 0.1113 | 10 | 100 % |
| `p1` | `p=1` | 0.2868 (−0.0002 ± 0.0002 against `off`) | 0.4289 (0) | 0.1113 (0) | 10.000 | 100 % |
| `p099r` | `p=0.99,renorm=1` | 0.2876 (+0.0008 ± 0.0044) | 0.4289 (−0.0000 ± 0.0005) | 0.1114 (+0.0001 ± 0.0002) | 10.000 | 100.0 % |
| `w002r` | `w=0.02,renorm=1` | 0.2856 (−0.0012 ± 0.0042) | 0.4284 (−0.0005 ± 0.0006) | pending | 9.994 | 99.9 % |

- **The controls hold.** `off` reproduces the quality chain's Swift Q2_0 on seven cards (agent 0.2851, wiki 0.4283,
  code 0.1124) and on five (0.2866, 0.4282, 0.1121) within its error. `p1` equals `off` bit for bit on wiki and code;
  on the agent text, whose chunks run as 16 micro-batches, it is 0.0002 lower, almost all of it in the first chunk.
- **The agent text's paired noise is ~0.004.** p = 0.99 drops a slot in 0.01 % of the (token, layer) pairs, yet moves
  three of the twelve 8,192-token chunks by 0.016–0.037: a changed expert early in a long context carries through
  the attention and recurrent state into every later token. Any cut's agent change is read against that ±0.004; wiki
  and code (80 chunks of 512) resolve ±0.0005.
- **Swift's router is flat.** Even the 10th of a token's experts holds more than 2 % of its routed weight in 99.8 %
  of the pairs, and the slots above it less than 99 % of the weight in 99.99 %: the mild cuts drop nothing that
  matters, in quality or in bytes.

## Slot 2 results (2026-11-04 16:19–17:15, main session; set a on GPUs 0–2, set b on 5–6, each paired against its own `off`)

Full tables: [`kld_by_arm.tsv`](kld_by_arm.tsv) (every arm and text) and [`kept_by_layer.tsv`](kept_by_layer.tsv) (experts kept per layer).

| Arm | Experts kept (of 10) | Routed reads | agent Δ KLD | wiki Δ KLD | code Δ KLD |
|---|---|---|---|---|---|
| w005r (own share < 5 % dropped) | 9.33 | 93 % | −0.002 ± 0.005 | +0.007 ± 0.002 | +0.004 ± 0.001 |
| p095r | 9.74 | 97 % | −0.006 ± 0.006 | −0.001 ± 0.001 | +0.001 ± 0.001 |
| **p090r** | **8.77** | **88 %** | −0.003 ± 0.008 | +0.001 ± 0.002 | +0.0035 ± 0.0012 |
| p080r | 7.27 | 73 % | −0.001 ± 0.007 | +0.025 ± 0.003 | +0.011 ± 0.002 |
| p070r | 5.95 | 60 % | +0.020 ± 0.015 | +0.084 ± 0.004 | +0.028 ± 0.004 |
| p090 (not renormalized, set b) | 8.76 | 88 % | +0.006 ± 0.005 | +0.001 ± 0.001 | +0.000 ± 0.001 |

**Verdict:** p = 0.9 renormalized keeps 8.77 of 10 experts (−12 % of routed reads) at no measurable cost (agent −0.003 ± 0.008, wiki +0.001 ± 0.002, code +0.0035 ± 0.0012); p = 0.8 keeps 7.27 (−27 %) for wiki +0.025 ± 0.003 and code +0.011 ± 0.002 (agent within noise); p = 0.7 keeps 5.95 (−40 %) for wiki +0.084 and code +0.028, too lossy; renormalizing matters (p = 0.9 plain: agent +0.006). The agent text has 12 chunks (its paired noise is ±0.005–0.008); wiki and code have 80 each and are the sharper tests. The saving becomes speed only when the kernels skip the dropped experts' reads (Q660): est. ~1 ms of a one-token pass at p = 0.9 (routed experts are ~8.6 ms) and more on multi-token verify passes, where fewer experts per token shrink the union a pass reads.

## Method

- **Model:** Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (the rig's file), every layer on the GPUs (`-fit on`).
- **Reference:** the Swift-1.5 Q8_0 logits of the Flash-Next quality chain
  ([2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md)) over its three texts: held-out agent
  sessions (12 × 8,192 tokens), wikitext-2 test (80 × 512) and the agent check's Exercism code (80 × 512).
- **Engine:** `llama-perplexity` of `fd549b754` built for sm_86, `-ub 512 -fa on`, `-lv 4` (the histogram lines are
  llama INFO lines).
- **Arms:** `off` (no switch) and `p1` (the switch on at p = 1, nothing dropped) are the controls; the cuts run from
  mildest to strongest. Each arm is one llama-perplexity run per text.
- **Paired change:** llama-perplexity prints each chunk's running mean KLD, so chunk i's own mean is
  i · K_i − (i − 1) · K_(i−1); an arm's change against `off` on the same cards is the mean of the per-chunk differences,
  and its standard error their spread over √chunks. The noise floor: the same file and engine on five against seven
  cards differ by +0.0015 ± 0.0020 (agent), −0.0001 ± 0.0007 (wiki) and −0.0004 ± 0.0004 (code).
- **Experts kept:** the switch's histogram over every token the run decodes (scored and context), per token and layer,
  and per MoE layer.

## Files

| File | What it holds |
|---|---|
| [kld_by_arm.tsv](kld_by_arm.tsv) | per card set, arm and text: the switch's setting, mean KLD and its error, same top token, the paired change against `off` and its error, chunks, experts kept (mean of 10), the routed reads left, the histogram by count |
| [kept_by_layer.tsv](kept_by_layer.tsv) | per card set, arm and text: the mean experts kept in each of the 48 MoE layers |
| [run/kld_tail.sh](run/kld_tail.sh) | the arms (the bytes-switches folder's script, plus a stop time and a stop file so the cards are free when another job needs them) |
| [run/table.sh](run/table.sh) | the two tables from the logs |

## Commands

```bash
# the arms on a set of cards (FN1c's environment gives WIKI, AGENT, CODE, MOUNTS and KLD, the bases' folder)
PPL=<bytesw build>/bin/llama-perplexity BASES=<bases> OUT=<out> MODEL=<Swift Q2_0 first shard> \
  GPUS=<ids> MOUNTS=... WIKI=... AGENT=... CODE=... TAG=moetail [ARMS="off p1 p099r ..."] [STOP_AT=<epoch>] \
  bash run/kld_tail.sh
# the tables
OUT=<out> bash run/table.sh > kld_by_arm.tsv
OUT=<out> LAYERS=1 bash run/table.sh > kept_by_layer.tsv
```

## Provenance and licence

Swift-1.5 GSQ-RCO Q2_0 and Q8_0 of Qwen3.8-Flash-Next and the three texts as in
[2026-11-03_flash_next_quality](../2026-11-03_flash_next_quality/README.md). The tables are this repository's
measurements.
