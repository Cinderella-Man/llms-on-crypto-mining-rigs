# Two bytes-per-token switches for Qwen3.8-Flash-Next: fewer routed experts, a bounded shortlist head

Registry rows: [Q618 and Q619](../../../quants/experiments.md) (fewer routed experts on confident tokens; a shortlist
for the main model's output layer with an exact bound), Q660–Q663 (what follows from them).

Both switches are on the fork branch `pascal-p106-v3-b12076-bytesw` (`fd549b754`, one commit on `-draftmem`), off by
default. This folder holds the CPU checks of 2026-11-04 (no GPU) and the scripts for the GPU runs that come next.

| Switch | What it does | Measured on the CPU |
|---|---|---|
| `LLAMA_MOE_TAIL=p=<p>,w=<w>,renorm=<0\|1>,hist=<0\|1>` (Q618) | after the router's top-k, a slot's weight becomes 0 when the slots ranked above it already hold a share p of the token's routed weight, or its own share is below w; renorm rescales the kept weights to the token's sum; hist logs a histogram of the experts kept per token and layer | p = 1: the same logits bit for bit and the same text; on Swift-1.5 Q2_0, p = 0.9 keeps 8.62 of the 10 experts on average, p = 0.8 keeps 7.02 |
| `LLAMA_HEAD_BOUND=<k_low>[,<m_cap>]` (Q619) | the main model's head over a shortlist (the ids below k_low, the special tokens, `LLAMA_HEAD_BOUND_FILE`, the context's tokens), with an upper bound of every other logit; the sampler keeps the shortlist's logits only if the bound excludes every token it could pick, and otherwise gets the full head's | the same tokens and probabilities as the full head in every sampling tried; on Swift-1.5 Q2_0 the bound never excludes the tokens left out: 99.7 % of the rows fall back to the full head |

## Q618: fewer routed experts on confident tokens

The cut works at graph level: the dropped slots get weight 0, and the expert kernels still read their rows. It
measures what the cut does to quality; the bytes are saved only once the kernels skip the dropped slots (Q660). The
histogram gives the share of the routed experts' reads that would remain.

On a random qwen4exp (32 routed experts, 8 used; [random_moe_tail.tsv](random_moe_tail.tsv)): at p = 1, with and
without renormalization, the logits of 24 prompts (317 positions) equal the switch-off logits bit for bit, and every
token keeps all 8 experts. Below it, the experts kept and the logits change as expected (the routers are scaled by
30, because a random router weighs its experts almost alike and a cut would keep them all):

| Head | p | w | renorm | logits against off | experts kept (of 8) |
|---|---|---|---|---|---|
| F32 | 1 | 0 | 0, 1 | identical | 8.00 |
| F32 | 0.9 | 0 | 0, 1 | max abs diff 7e-5, same top token 100 % | 6.70 |
| F32 | 0.6 | 0 | 1 | max abs diff 2e-4, same top token 100 % | 3.38 |
| F32 | 1 | 0.094 | 1 | max abs diff 1.5e-4, same top token 100 % | 4.20 |
| Q5_K | 1 | 0 | 0, 1 | identical | 8.00 |
| Q5_K | 0.9 | 0 | 0, 1 | max abs diff 1.2e-3, same top token 99.4 % | 6.70 |

On Swift-1.5 GSQ-RCO Q2_0 (four coding prompts, each greedy, with the server's default sampling and with the agent
check's, 64 tokens; [real_server_arms.tsv](real_server_arms.tsv)):

| `LLAMA_MOE_TAIL` | text against off | experts kept (of 10) | routed reads left | kept 9 / 8 / 7 / 6 / ≤ 5 |
|---|---|---|---|---|
| `p=1` | the same in all 12 requests | 10.00 | 100 % | — |
| `p=0.9,renorm=1` | differs in 10 of 12 (greedy: the first 4–151 characters agree) | 8.62 | 86.2 % | 66.4 / 29.8 / 3.2 / 0.5 / 0.2 % |
| `p=0.8,renorm=1` | differs in 9 of 12 | 7.02 | 70.2 % | — / 27.0 / 53.5 / 15.4 / 4.2 % |

Swift's router is not peaked: 90 % of a token's routed weight takes 8–9 of its 10 experts in 96 % of the (token, layer)
pairs, and layer 0 cuts most (7.77 at p = 0.9, the other layers 8.3–8.9). A changed greedy text is not a quality
measure on a 2-bit model, where any perturbation flips near-ties; the KL divergence by threshold decides
(`run/kld_tail.sh`, below). At R13's 28 ms of routed Q2_0 experts per token, p = 0.8 would save at most ~8 ms a token
once the kernels skip the dropped slots, and p = 0.9 ~4 ms.

## Q619: the shortlist head with an exact bound

For a row of shortlist logits, the bound is at least every left-out logit the full head computes: Cauchy–Schwarz on
the head's input row h (as the kernels quantize it: Q8_1 blocks on CUDA, Q8_K on the CPU) and the head's row norms,
plain and about the mean row, read once when the switch is set up, plus 2 % for the kernels' arithmetic. The sampler
keeps the row when the k-th highest candidate is above the bound (k: top-k, plus the tokens penalties or logit biases
may push down; min-p's cut; greedy: 1), after the reasoning budget and, when it applies, the grammar; otherwise it
asks for the full head, computed again from h with the same product. Samplings that read the whole distribution
(top-p or typical-p before any top-k, n_probs, mirostat, XTC, dynamic temperature) always take the full head.

**Exactness.** On random qwen4exp models (4,096 tokens; heads in F32 and Q5_K; [random_head_bound.tsv](random_head_bound.tsv)),
24 prompts × 24 generated tokens per sampling, the shortlist head on and off give the same tokens and the same
probability of every sampled token for both heads, with their rows as they are and zeroed, in all eight samplings
(greedy, the server's default chain, top-k 5, the agent check's settings, min-p 0.2, top-k 20 with repeat and
frequency penalties, top-k 10 with a grammar, top-p 0.9): 32 runs of 576 tokens. A random head's rows are all alike,
so the bound excludes nothing and every row falls back; with the head's rows from k_low on set to zero in both runs,
the bound excludes them and the shortlist's rows stand for greedy, the default chain, top-k, the agent check's
settings, penalties and the grammar (min-p 0.2's cut and top-p still fall back), with the same tokens and
probabilities. On Swift-1.5 Q2_0 the text is the same as the full head's in all 12 requests.

**The bound on the real head** (Swift-1.5 Q2_0's Q5_K head, 2,560 × 248,320; the shortlist: the 32,768 lowest ids,
33 special tokens, 121 ids of [S12's frequency list](../2026-11-04_draft_mem/token_freq_s12.tsv) and the context's
tokens; 655 rows of the 12 requests):

| | value |
|---|---|
| rows whose top logit was above the bound | 0 of 655 |
| rows that fell back to the full head | 653 (99.7 %; the other 2 were never sampled) |
| top logit, mean | 24.3 |
| bound, mean (min of the plain and the mean-centred bound) | 173.7; bound − top logit: mean 149.4, min 80.4 |
| \|h\|, mean; largest \|W_t\|, \|W_t − mu\|, \|mu\| | 155.7; 1.414, 1.632, 0.246 |
| on the fallbacks: rows whose top token was outside the shortlist | 21 of 653 (3.2 %) |
| on the fallbacks: top logit − highest left-out logit, mean | 9.9 |
| on the fallbacks: bound − highest left-out logit, mean | 159.3 |

The shortlist is good (it held the top token in 97 % of the rows, with the highest left-out logit ~10 below the top),
but a Cauchy–Schwarz bound in 2,560 dimensions assumes the input row points along the worst row, and the real logits
are ~0.1 of \|h\| \|W_t\|: the bound overshoots the true left-out maximum by ~160 logits, 16 times the margin it would
need. No sampling can keep a shortlist row with it, so the switch only adds the shortlist's products and a second head
per token; it is parked (Q619), with tighter exact bounds as Q662.

## Files

| File | What |
|---|---|
| [random_head_bound.tsv](random_head_bound.tsv) | random qwen4exp: per head (F32, Q5_K; rows random or zero from k_low) and sampling, tokens the same, probabilities the same, tokens from the shortlist's rows and from the full head |
| [random_moe_tail.tsv](random_moe_tail.tsv) | random qwen4exp: per head and `LLAMA_MOE_TAIL` arm, logits against off, same top token, experts kept |
| [real_server_arms.tsv](real_server_arms.tsv) | Swift-1.5 Q2_0 on a CPU: per arm, prompt and sampling, tokens, tok/s (CPU, a diagnostic) and whether the text is the off arm's |
| `run/cpu_check.sh` | both CPU checks: the random models (test-llama-archs, llama-quantize, test-bytes-switches), then `run/server_arms.sh` on the real model |
| `run/server_arms.sh` | the switches in llama-server on a real model, one load per arm; the same script on the rig |
| `run/table_random.sh` | the two random-model tables from cpu_check.sh's logs |
| `run/kld_tail.sh`, `run/table_tail.sh` | Q618's KL divergence by threshold on the GPUs, and its table |

## Commands

```bash
# the branch, CPU build with the tests (the rig build: scripts/build_llamacpp_docker.sh fork/pascal-p106-v3-b12076-bytesw)
cmake -S . -B build-cpu -DGGML_CUDA=OFF -DLLAMA_BUILD_TESTS=ON && \
  cmake --build build-cpu -j 6 --target test-bytes-switches test-llama-archs llama-quantize llama-server

# the CPU checks (random models, then the real model in llama-server)
BIN=build-cpu/bin OUT=out THREADS=12 MODEL=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> \
  FREQ=../2026-11-04_draft_mem/token_freq_s12.tsv bash run/cpu_check.sh
OUT=out bash run/table_random.sh

# Q618's KLD by threshold (an sm_86 build of the branch; the bases, texts and split of ../2026-11-03_flash_next_quality)
PPL=<build>/bin/llama-perplexity BASES=<that run's kld folder> OUT=out-kld MODEL=<Swift Q2_0 shard 1> GPUS=0,1,2,3,4,5,6 \
  MOUNTS="<model and text folders>" WIKI=<wiki.test.raw> AGENT=<agent text> CODE=<code text> bash run/kld_tail.sh
OUT=out-kld bash run/table_tail.sh > kld_tail.tsv
```

`kld_tail.sh`'s arms: `off` and `p1` (the controls: the same KLD), then p = 0.95, 0.9, 0.8 renormalized, p = 0.9
without renormalization, minimum shares 0.02 and 0.05, p = 0.99 and 0.7 renormalized, p = 0.95 without and p = 0.9
with a 0.03 minimum (`ARMS` picks a subset, `TEXTS` the texts). Each log ends with the experts-kept histogram.

## Provenance and licence

The random models are written by `test-llama-archs` from a seeded generator (no weights of any published model).
The real model is Swift-1.5's GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (re-sharded into three files), the same file as
[../2026-11-04_draft_mem](../2026-11-04_draft_mem/README.md). The frequency list is that folder's
`token_freq_s12.tsv`: token ids and counts of the model's own assistant turns in an agent-check run. The four prompts
were written for this check. The tables hold counts and statistics, no weights and no generated text.
