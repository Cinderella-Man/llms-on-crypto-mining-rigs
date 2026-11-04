# C271 / Q827: T(k) on real agent text on the P106s (2026-11-04)

[Q591](../2026-11-04_kcurve_p106/README.md)'s T(k) and [C266](../2026-11-04_kpass_profile/README.md)'s profile ran on
llama-bench's prompt tokens, which are random token ids, so their routing could differ from text's (Q860 counts 16.3 /
21.6 / 26.4 / 30.9 distinct experts per layer at k = 2–5 on real text). This folder times passes of k **consecutive real
tokens**: 1,680 tokens of held-out coding-agent text (the agent-check sessions' text the quality runs score on), as a
prompt whose calls the `-taper` branch's per-request micro-batch plan cuts into micro-batches of exactly k tokens
(`"ubatch_taper": "short=1000000:k,min=1"`; 1,680 divides by every k up to 8). Swift Qwen3.8-Flash-Next GSQ-RCO Q2_0 on
nine P106-090s, fork branch `pascal-p106-v3-b12076-ladder3` `7bc1552f4`, `-c 8192 -np 1 --cache-ram 0`, the KV cache
not reused between requests.

## T(k), ms per pass of k real tokens

| k | defaults | `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768` | llama-bench, random tokens, defaults (Q591) | the same, `SMEM=32768` (Q591) | pipelining on, defaults |
|---:|---:|---:|---:|---:|---:|
| 1 | 73.2 | **64.9** | 82.4 | 74.9 | 39.4 |
| 2 | 112.8 | **97.9** | 122.1 | 107.7 | 42.9 |
| 3 | 139.3 | **127.9** | 149.4 | 138.5 | 45.6 |
| 4 | 162.0 | **151.6** | 173.0 | 162.8 | 48.8 |
| 5 | 208.1 | **193.9** | 217.1 | 205.3 | |
| 6 | 237.9 | **222.9** | 249.1 | 234.7 | |
| 7 | 273.1 | **249.9** | 280.1 | 261.7 | |
| 8 | 310.1 | **288.5** | 319.1 | 297.3 | |
| T(5) / T(1) | 2.84 | 2.99 | 2.63 | 2.74 | |

- **Real text costs the same as llama-bench's random tokens.** The real-text passes sit a steady 7–11 ms below
  llama-bench's at every k. That difference is the output layer and its logits row: a prompt micro-batch asks for no
  logits, and llama-bench's pass for one row (the Q5_K output layer, 3.1 ms at one row, and ~1 MB of logits to the host,
  est. ~5 ms over PCIe 1.0 x1). So the routed experts cost the same on text although it reads fewer distinct experts
  than uniform routing would; llama-bench's random tokens do not route uniformly either.
- **The 32 KB launch budget is 8–23 ms faster at every k on real text too**, as Q591 found on random tokens.
- **A verify of k drafted tokens costs more than both curves**: it runs the output layer on k rows and copies k rows of
  logits (Q900). Neither file holds that.
- **With llama.cpp's pipelining on**, the cards overlap one prompt's consecutive micro-batches (the next starts on card 0
  while the last runs on card 1), so micro-batches of 1–4 tokens cost 39–49 ms each, about half an isolated pass. A
  verify is a single micro-batch and never overlaps; the timings above therefore run with the pipelining off (any
  tensor override turns it off in `llama_context`; here `-ot token_embd.weight=CPU`, where the embeddings are anyway).

## Where the time goes on real text (1 and 5 tokens, 32 KB budget)

The `prof` arm ran the first 340 real tokens at k = 1 and 5 under `GGML_CUDA_PROFILE_MODE=group` (C140's 7.6 µs per span
subtracted; [C266](../2026-11-04_kpass_profile/README.md)'s method and categories), ms per pass:

| Part | 1 token | 5 tokens | 5 / 1 | C266, random tokens, defaults: 1 → 5 |
|---|---:|---:|---:|---|
| routed experts | 8.62 | 54.46 | 6.32 | 8.69 → 55.49 |
| attention and recurrent projections | 19.12 | 50.04 | 2.62 | 27.13 → 61.77 |
| hyper-connections (BF16) | 10.13 | 22.76 | 2.25 | 10.03 → 22.25 |
| shared expert | 3.14 | 6.70 | 2.13 | 3.24 → 6.36 |
| router | 2.16 | 5.74 | 2.66 | 2.15 → 5.85 |
| Gated DeltaNet state and conv | 3.46 | 5.39 | 1.56 | 3.46 → 5.46 |
| attention (FLASH_ATTN_EXT) | 0.73 | 1.94 | 2.65 | 0.63 → 1.69 |
| other operations without weights | 5.44 | 8.77 | 1.61 | 9.22 → 10.58 |
| output layer | 0.01 | 0.05 | | 3.58 → 3.58 (llama-bench asks for one row) |
| the cards' GPU time | 53.3 | 157.1 | 2.95 | 68.7 → 174.3 |
| the pass, unprofiled | 64.9 | 193.9 | 2.99 | 82.4 → 217.1 |

The routed experts cost the same on real text as on random tokens (6.3–6.4× from one token to five), so their growth
is not a matter of routing; the 32 KB budget's gain is in the projections. Each request also decodes one answer token,
so the 5-token mean holds one 1-token evaluation among its 68 (est. ~1 % low). Files: [parts_real_s32k.tsv](parts_real_s32k.tsv),
[summaries/](summaries/).

## Files

| File | What it is |
|---|---|
| [requests.tsv](requests.tsv) | per arm and k: the prompt's tokens, its processing time, the micro-batches, ms per micro-batch (pipelining off) |
| [requests_pipelined.tsv](requests_pipelined.tsv) | the same with the pipelining on (k = 1–4; the run was stopped there) |
| [run/kreal.sh](run/kreal.sh) | the rig run: one llama-server load per arm, one request per k |
| [run/kreal_req.py](run/kreal_req.py) | the requests: the text tokenized by the server, its first N tokens sent per k |
| [../2026-11-04_kcurve_p106/cost/t_k_q2_0_s32k_real.txt](../2026-11-04_kcurve_p106/cost/t_k_q2_0_s32k_real.txt), [t_k_q2_0_real.txt](../2026-11-04_kcurve_p106/cost/t_k_q2_0_real.txt) | the `LLAMA_SPEC_ADAPT_COST` files from these passes |

## Run

On the rig, idle, the text in a file (here the first 20,000 bytes of the agent text, md5 `312fd205dec2fba4879d4bf421f1e80e`):

```bash
BIN=<-ladder3 build>/bin MODEL=<Swift Q2_0 first shard> TEXT=<text file> OUT=<dir> setsid nohup bash run/kreal.sh > kreal.console 2>&1 < /dev/null &
```

Each arm loads `llama-server -m MODEL -ngl 99 -fa on -c 8192 -np 1 --cache-ram 0 --no-warmup -b 2048 -ub 512
-ts 6,6,6,6,5,6,5,6,3 -fit off -lm dio --no-host -ot token_embd.weight=CPU` (~4 minutes) and sends one `/completion` per k
with the token ids, `n_predict` 1 and `cache_prompt` false; T(k) = the request's `prompt_ms` / (1,680 / k). ~11 minutes
an arm for k = 1–8. `NOPIPE=` (empty) keeps the pipelining.
