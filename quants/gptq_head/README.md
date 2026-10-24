# GPTQ rounding for the output head

Tools to store a model's output head (the `output.weight` matrix that turns the final hidden state into one logit per
vocabulary entry) in Q5_0 with GPTQ rounding instead of llama-quantize's round-to-nearest, and to check the result.

Why the head: on the rig a token's decode time follows the bytes it reads, and Flint's head (150,144 × 2,048) is read
in full for every token. In Q8_0 that is 0.327 GB, ~14 % of a ~2.3 GB mix; in Q5_0 it is 0.211 GB. Round-to-nearest
Q5_0 costs too much quality there, so the question is whether better rounding of the same format closes the gap.

## How it works

1. **Head inputs.** `head_dump/` is a small llama.cpp example (`llama-head-dump`) that runs text through a model and
   writes the graph tensor `result_norm`, the final hidden state after the output norm, for every token. It runs the
   quant whose head is to be replaced, so the inputs are the ones the new head will see.
2. **H.** `gptq_head.py hessian` sums x xᵀ over the dumped rows in float64 (2,048 × 2,048).
3. **Rounding.** `gptq_head.py quant` quantizes the BF16 head row block by row block (rows are independent, so it runs
   on CPU cores in parallel):
   - `--mode rtn`: llama-quantize's Q5_0 without an importance matrix, byte for byte;
   - `--mode imat`: llama-quantize's Q5_0 with an importance matrix holding diag(H) (what `llama-imatrix
     --process-output` would give);
   - `--mode gptq`: GPTQ ([Frantar et al. 2022](https://arxiv.org/abs/2210.17323)) in column order: each column is
     rounded and its error, weighted by the Cholesky factor of H⁻¹ (1 % damping), is pushed onto the columns not yet
     rounded. A Q5_0 block (32 weights, one FP16 scale, 5-bit codes) is one GPTQ group; its scale is set when GPTQ
     reaches the block, from the already-updated weights (`--scale max`: ggml's rule, the largest magnitude maps to
     −16; `--scale search`: ggml's weighted scale search with diag(H) as weights). Options: `--act-order` (columns by
     decreasing diag(H), scales fixed beforehand) and `--refit N` (least-squares block scales under H for the codes
     found).
4. **Offline check.** `gptq_head.py proxy` compares heads on held-out head inputs: the KL divergence and same top
   token of the head's logits against the BF16 head's, which isolates the head's own error without running the model.
5. **GGUF.** `gptq_head.py base` writes the model without its head as the first file of a two-file split GGUF, once;
   `shard` adds a candidate head as the second file (llama.cpp loads `<name>-00001-of-00002.gguf` and finds the rest),
   so trying a head costs 0.2 GB instead of a full copy. `full` writes the model with the new head as one file.

`run.sh` runs the whole recipe for one model. `test_gptq_head.py` checks the tools: Q5_0 and Q4_0 packing against
gguf-py, round-to-nearest byte-identical to llama-quantize on a real tensor, GPTQ with H = I equal to round-to-nearest,
and GPTQ against round-to-nearest under a known H.

## Results: Flint, head Q8_0 → Q5_0 (2026-10-23)

Setup: the head inputs of F3 (a Q73 mix with a Q8_0 head; see [experiments.md](../experiments.md)) on 500 chunks of
512 tokens of wikitext-2 train (255,500 rows) and on Flint's own transcripts from Q51 (38 conversations: chat, code,
edits, agent turns; 81,723 rows). Every tool output was checked before use: the dumped rows times F3's head reproduce
F3's own logits to 0.1–0.3 % (median relative error; llama.cpp quantizes activations to 8 bits for this product), and
one row off they miss by 16–18 %.

H has one dominant input dimension: dimension 1952 of the final hidden state has 285 times the average mean square,
and the same eight dimensions lead in wikitext and in the transcripts. Round-to-nearest Q5_0 spends the same relative
precision on every input dimension, which is why the head is sensitive to it.

**Full model, KL divergence against BF16** (80 × 512 tokens of wikitext-2 test, rescale off, all on the same two
RTX 3090s; F3's body with each head):

| Head (Q5_0 unless noted) | Calibration | KLD | Same top token | PPL vs BF16 |
|---|---|---|---|---|
| Q8_0 (F3 as made) | | 0.1113 | 88.56 % | +3.39 % |
| Round to nearest (llama-quantize) | | 0.1344 | 85.84 % | +6.04 % |
| imat (llama-quantize with diag(H)) | wiki + transcripts | 0.1186 | 87.68 % | +3.67 % |
| GPTQ, max scale | wiki + transcripts | 0.1139 | 88.40 % | +3.60 % |
| **GPTQ, search scale** | wiki + transcripts | **0.1136** | **88.48 %** | **+3.22 %** |
| GPTQ, search scale, one scale refit | wiki + transcripts | 0.1136 | 88.53 % | +3.27 % |
| GPTQ, search scale | wiki only | 0.1133 | 88.50 % | +3.31 % |
| GPTQ, search scale | transcripts only | 0.1145 | 88.43 % | +3.39 % |

The round-to-nearest row is byte for byte the head llama-quantize writes; it reproduces Q73's H1 (F3 with a Q5_0
head: +0.023 KLD on both GPU sets). The head is the last layer, so changing it cannot change any expert routing: the
differences in this table are the head's own effect, without the routing-flip noise that makes whole-mix comparisons
uncertain to ~0.005. GPTQ leaves +0.0023 KLD and −0.08 points of same top token of the +0.023 that round-to-nearest
costs, and perplexity is no worse than with the Q8_0 head.

**The head alone** (logits from F3's head inputs, against the BF16 head's logits; held-out rows only: 40 wikitext
chunks never used for H, and 8 transcripts, 2 per kind, left out of H for this check):

| Head | KLD, wikitext | Same top, wikitext | KLD, transcripts | Same top, transcripts |
|---|---|---|---|---|
| Q8_0 (F3's) | 0.0003 | 99.32 % | 0.0001 | 99.73 % |
| Q5_0 round to nearest | 0.0213 | 92.41 % | 0.0078 | 97.25 % |
| Q5_0 imat | 0.0074 | 95.50 % | 0.0036 | 98.09 % |
| Q5_0 GPTQ, H from wiki | 0.0030 | 97.35 % | 0.0022 | 98.59 % |
| Q5_0 GPTQ, H from the other 30 transcripts | 0.0042 | 96.63 % | 0.0016 | 98.84 % |
| Q5_0 GPTQ, H from wiki + the other 30 transcripts | 0.0031 | 97.29 % | 0.0017 | 98.70 % |

These head-only numbers, minus the Q8_0 head's own, predict the full-model differences above (round to nearest 0.021 →
+0.023, imat 0.0071 → +0.0073, GPTQ 0.0029 → +0.0023), so the offline check is enough to compare heads. A head
calibrated on wikitext keeps most of its gain on Flint's own chat and code (−73 % of round-to-nearest's excess error,
against −81 % with in-domain calibration); calibrating on both is within 0.00015 of the better of the two on each.
Act-order (columns by decreasing diag(H)) gives nothing here (0.00314 in both orders); a scale refit gives less than
0.0001.

**Recipe:** GPTQ with search scales, 1 % damping, column order, H from the model's own head inputs on ~250k tokens of
wikitext-2 train plus ~80k tokens of the model's own outputs; `run.sh` with `GPTQ_ARGS="--scale search"` (pass
`SEQS=` for the model's own outputs). The head shrinks from 0.327 to 0.211 GB, −0.116 GB per token.

**In whole mixes** (same GPUs, same scoring; GB read per token from [token_cost.py](../../scripts/token_cost.py)):

| Mix | Head | KLD | Same top token | PPL vs BF16 | GB per token |
|---|---|---|---|---|---|
| G3 | Q8_0 | 0.1040 | 89.09 % | +2.87 % | 2.327 |
| G3 | GPTQ Q5_0 | 0.1063 | 88.86 % | +2.73 % | 2.212 |
| H3 | Q8_0 | 0.1150 | 88.46 % | +1.64 % | 2.216 |
| H3 | GPTQ Q5_0 | 0.1172 | 88.29 % | +1.47 % | 2.101 |

The GPTQ head costs +0.0021 to +0.0024 KLD in every mix, as on F3. G3 with it reads fewer bytes than H3 at lower
KLD. The head was calibrated on F3's head inputs and reused unchanged for G3 and H3: the mixes differ only in their
bodies, and a head calibrated on another text domain loses only ~0.001 (above).

## Results: a screened head (2026-10-24)

A screened head keeps the exact Q8_0 head in memory but reads it only for a few candidate rows: a cheaper copy of the
head (the screen) ranks all 150,144 tokens, and exact logits are computed for the screen's top M. Greedy decoding
needs the exact top token among the candidates; the makers' sampler (top-k 20, top-p 0.95, T 1.0) needs the exact top
20, since it renormalizes over them. `screen_head.py` measures both on held-out head inputs (the 40 wikitext chunks
and the 8 held-out transcripts, 40,527 rows; screens calibrated without them), with the exact head itself as a control
screen, which scores 1 everywhere.

| Screen | Screen bytes (share of Q8_0) | Full top-20 in top M: M = 64 / 128 / 256 / 512 / 1,024 | M needed, 99.9 % of rows | Sampled token in top 64 |
|---|---|---|---|---|
| GPTQ Q5_0 | 211 MB (64.7 %) | 99.998 / 100 / 100 / 100 / 100 % | 30 | 100 % |
| **GPTQ Q4_0** | **173 MB (52.9 %)** | 99.980 / 99.993 / **100** / 100 / 100 % | 43 | 100 % |
| Q4_0, round to nearest | 173 MB (52.9 %) | 97.76 / 99.59 / 99.93 / 99.985 / 99.995 % | 229 | 99.975 % |
| Q4_1, round to nearest | 192 MB (58.8 %) | 99.10 / 99.84 / 99.97 / 99.995 / 100 % | 145 | 99.986 % |
| rank 512 under H (Q8_0 factors) | 83 MB (25.3 %) | 9.2 / 26.1 / 47.4 / 67.1 / 81.5 % | > 4,096 | 95.7 % |
| rank 256 under H (Q8_0 factors) | 41 MB (12.7 %) | 2.7 / 9.3 / 21.1 / 37.6 / 55.7 % | > 4,096 | 87.1 % |

"Sampled token" is the probability that the token the makers' sampler draws is among the candidates. The exact top
token is among the top 64 of every 4-bit screen on every row (Q4_1: all but one).

- **A GPTQ Q4_0 screen with M = 256 is exact for the makers' sampler on all 40,527 rows**, and so is GPTQ Q5_0 from
  M = 128. Reads per token: 173 MB + 256 rows × 2,176 bytes = 174 MB, 53 % of the Q8_0 head (−0.153 GB per token,
  against −0.116 GB for the GPTQ Q5_0 head alone, which is not exact). Logits outside the candidates come from the
  screen; the head-only KLD of that mix against the exact head is 0.0001 (Q4_0 GPTQ, M = 256), against 0.0029 for
  the GPTQ Q5_0 head alone.
- GPTQ is what makes 4 bits work as a screen: round-to-nearest Q4_0 misses the full top-20 on 0.07 % of rows at
  M = 256 and still on 0.005 % at M = 1,024.
- **Low-rank screens fail.** The head's H-weighted energy is 99.4 % in its top 64 directions, but that energy is the
  few outlier input dimensions, which shift all logits alike; the directions that separate tokens are spread out.
  Rank 512 (a quarter of the bytes) misses the full top-20 on 18.5 % of rows even at M = 1,024, and rank 64 finds the
  exact top token in its top 64 on only 60 % of rows.
- Costs not measured here: the screened head needs both copies in memory (+173 MB on the card holding the head) and
  a top-M selection over 150,144 scores per token on the GPU. Whether that selection plus the gather of M rows costs
  less than reading 0.153 GB (~1 ms at the fork's ~150 GB/s) has to be measured on the rig.

## The screened head in the engine (2026-10-24)

Our llama.cpp fork (branch `pascal-p106-v2-screenhead`) runs the screened head. A model gets it from an extra tensor,
`output_screen.weight` (same shape as `output.weight`, here GPTQ Q4_0), either inside the GGUF or from a side file:

```bash
# build the screen: GPTQ Q4_0 from the BF16 head, H from your quant's head inputs (see "How it works")
python3 gptq_head.py quant flint-BF16.gguf screen.q4_0 --type q4_0 --mode gptq --scale search --hessian H.npy
# either: a copy of the model with the screen inside (checked byte for byte with verifyscreen)
python3 gptq_head.py screen model.gguf screen.q4_0 model-screen.gguf --top-m 256
# or: a 173 MB side file that works with any Flint GGUF whose head is the usual 150,144 x 2,048
python3 gptq_head.py screenfile model.gguf screen.q4_0 flint-output-screen.gguf --top-m 256
```

At run time `LLAMA_SCREEN_HEAD=1` turns it on (`LLAMA_SCREEN_HEAD_FILE=flint-output-screen.gguf` adds the side file,
`LLAMA_SCREEN_HEAD_M` overrides the number of candidates M, 256 by default). For a batch with one output row, the graph
then computes the screen's logits, picks its top M on the GPU (llama.cpp's radix top-k), computes those M rows of the
Q8_0 head through the same matrix-vector kernel as the full head (the head viewed as 150,144 one-row "experts" of a
`mul_mat_id`), and gives every other token −∞. Everything downstream, including backend sampling (`-bs`), sees an
ordinary logits row.

When it is used:
- only for output rows of the plain head product, one at a time (decoding, the last token of a prompt); perplexity,
  batched outputs and speculative verification keep the full head;
- `llama-cli`, `llama-completion` and the other tools keep it only if their sampling settings can't see a difference:
  greedy, or top-k with 1 ≤ k ≤ M before any sampler that reads the normalized distribution (the makers' top-k 20 →
  top-p 0.95 → T 1.0 qualifies); no token probabilities, grammar, positive logit bias, mirostat or raising penalties.
  Otherwise they switch it off with a warning. The reasoning-budget sampler works with it: a token it forces counts
  as the only choice even when the screen left it without a logit;
- `llama-server` decides for every batch: on only if every request with a row in the batch qualifies.

**Checked on the quant machine** (RTX 3090s, `llama-screen-check`; 8 prompts of prose, code, maths, chat and SQL,
128 tokens each; the screen is the GPTQ Q4_0 one, M = 256):

| Model | Kernels | Steps | Candidate logits bit-identical to the full head | Full top-20 among the candidates | Greedy tokens | Seeded makers' sampling |
|---|---|---|---|---|---|---|
| G3 + screen in the file | Pascal one-token kernels (forced) | 1,024 | 262,144 of 262,144 | 1,024 of 1,024 | identical | identical |
| G3 + screen in the file | llama.cpp's own kernels | 1,024 | 262,144 of 262,144 | 1,024 of 1,024 | identical | identical |
| G3 + side file | Pascal one-token kernels (forced) | 1,024 | 262,144 of 262,144 | 1,024 of 1,024 | identical | identical |
| UD-Q4_K_XL + side file | Pascal one-token kernels (forced) | 1,024 | 262,144 of 262,144 | 1,024 of 1,024 | identical | identical |

The exact rows really are the full head's numbers: the candidates' logits match bit for bit, not within a tolerance,
because each row goes through the same kernel with the same arithmetic. The UD-Q4_K_XL row uses a screen calibrated
on another quant's hidden states. `llama-server` with backend sampling (`-bs`) answered 24 requests (the 8 prompts
greedy, sampled with a seed, and sampled with `n_probs`, which needs the full head) byte-identically with and without
`LLAMA_SCREEN_HEAD=1`; its log shows the screen on for the greedy and sampled requests and off for each `n_probs` one
(8 switches each way). `test-backend-ops` passes the screened head's ops against the CPU backend with both kernel
sets: `mul_mat_id` over 32,768 one-row Q8_0 experts, top-256 of 150,144, `set_rows` and `fill`.

### Measuring it on the rig

The screened head saves bytes only if the top-M selection and the few extra kernel launches cost less than reading
0.153 GB (~1 ms per token at the fork's ~150 GB/s on large shapes), so it has to be measured on the P106s. With a
build of the branch in `/opt/llama-builds/<build>` and the side file next to the models:

```bash
# correctness on the Pascal cards themselves: bit-identical candidate logits, same tokens (greedy and seeded)
ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 LLAMA_SCREEN_HEAD_FILE=<dir>/flint-output-screen-gptq-q4_0.gguf \
  /opt/llama-builds/<build>/bin/llama-screen-check -m <dir>/flint-S4L0.gguf -f <dir>/prompts.txt -n 64 -ngl 99 \
  -fa on -c 4096 --temp 1.0 --top-k 20 --top-p 0.95 --min-p 0 --seed 1234"
# speed: tg128 with the screen off and on, same binary, cards and flags
for s in 0 1; do ssh rig "LLAMA_NO_MOE_DOWN_RESCALE=1 LLAMA_SCREEN_HEAD=$s \
  LLAMA_SCREEN_HEAD_FILE=<dir>/flint-output-screen-gptq-q4_0.gguf BIN=/opt/llama-builds/<build>/bin \
  MODEL=<dir>/flint-S4L0.gguf TAG=screen$s GPU_SETS='0,1,2,3,4,5' BENCH_ARGS='-ngl 99 -fa on -p 0 -n 128 -r 3' \
  bash -s" < scripts/bench.sh; done
```

The same pair for UD-Q4_K_XL (without `LLAMA_NO_MOE_DOWN_RESCALE`) and G3 tells whether the gain holds across mixes,
and `GGML_CUDA_PROFILE=<file>` on one tg32 run shows what the screen, the top-M and the exact rows cost on the card
that holds the head. The screen adds 173 MB on that card, so a tight `-ts` split may need one step less there.

## GPTQ for body tensors (Q112, E1, 2026-10-24)

The same rounding for three groups of G3's always-read Q8_0 tensors whose round-to-nearest Q5_0 cost Q77 measured
(on the same GPUs, KLD against BF16, 80 × 512 wikitext-2): attention output in layers 0–8 (+0.0105), the shared
expert in layers 9–16 (+0.0071) and attention k/v in layers 9–16 (+0.0070), against G3's 0.1040.

- **Hessians** from G3's own forward pass (`act_hessian/`, a llama.cpp example: H = Σ x xᵀ of chosen graph tensors,
  accumulated during the pass) on 500 chunks of 512 tokens of wikitext-2 train: `kqv_out-N` (attention output's
  input, 4,096 values), `attn_norm-N` (k/v), `ffn_norm-N` (shared expert gate/up), `ffn_swiglu-N` (its down
  projection, 1,024 values).
- **Rounding** with `gptq_body.py` (gptq_head.py's GPTQ: searched Q5_0 block scales, column order, 1 % damping), and
  the same H-weighted relative output error as for the head, for the tensor as G3 holds it (Q8_0), for
  llama-quantize's round-to-nearest Q5_0 with Unsloth's importance matrix, and for GPTQ Q5_0.
- **Files** with `split_compose.py`: one shared first file (G3 without the 49 tensors of the three groups) and, per
  group, a small second file with that group in GPTQ Q5_0 and G3's own tensors for the other two; each checked tensor
  for tensor.

| Group | Tensors | Error left by GPTQ (share of round-to-nearest's, output-weighted) | per tensor |
|---|---|---|---|
| Attention output 0–8 | 9 | **0.294** | 0.13–0.36 |
| Shared expert 9–16 | 24 | 0.552 | gate/up 0.54–0.57, down 0.61–0.69 |
| Attention k/v 9–16 | 16 | 0.546 | 0.53–0.57 |

Attention output reads the most lopsided inputs (Q56's head showed the same pattern), and GPTQ removes 71 % of
round-to-nearest's error there; on the shared expert and k/v it removes 45 %.

**Full model** (KLD against BF16, 80 × 512 wikitext-2 test, rescale off, the GPUs that scored Q77's references; G3
0.1040 there). Recovery f = (round to nearest − GPTQ) / (round to nearest − G3):

| G3 with the group in Q5_0 | Bytes saved per token | Round to nearest (Q77) | GPTQ | Recovery f | Proxy predicted |
|---|---|---|---|---|---|
| Attention output 0–8 | 28 MB | 0.1145 (+0.0105) | **0.1062 (+0.0022)** | **0.79** | 0.71 |
| Shared expert 9–16 | 19 MB | 0.1111 (+0.0071) | **0.1066 (+0.0026)** | **0.63** | 0.45 |
| Attention k/v 9–16 | 6 MB | 0.1110 (+0.0070) | **0.1078 (+0.0039)** | **0.45** | 0.45 |

- **E1 passes** (rule: f ≥ 0.4 in two groups; kill: f < 0.25 in all three). With a paired noise of about 0.002 on the
  differences, attention output (0.0083 recovered) and the shared expert (0.0045) are clear; k/v (0.0032) is less
  sure.
- The proxy, the head-only error weighted by the captured Hessian, again ranks the groups correctly and is
  conservative: GPTQ recovers at least what it predicts.
- Together the three groups save 53 MB per token at +0.0087 KLD with GPTQ, against +0.0246 with round to nearest.

## GPTQ for K2h's body (Q112 E2/E3, Q114): tools and checks (2026-10-24)

The next steps apply the same rounding to K2h, the lead mix: its 120 always-read tensors that are already Q5_0
(attention q in layers 0–8 and 25–48, k/v/output 33–48, the shared expert 33–45: GPTQ at the same type, so no byte
change), its routed experts (Q5_0 gate/up 9–48 and down 2–45), and, for E3, the always-read tensors it keeps in Q8_0.

- **Per-expert Hessians.** `act_hessian/` has an expert mode (`HESS_EXPERTS=9,25,...`): each token's `ffn_norm` row is
  added to the Hessian of every expert the router chose for it (`ffn_moe_topk`), and each `ffn_moe_swiglu` row to
  the down-projection Hessian of the expert that produced it. Rows are buffered per expert and added in blocks;
  the output keeps one packed upper triangle per expert (4.3 GB per layer for gate/up, float64).
- **Checked against numpy.** `HESS_DUMP` writes the rows and expert ids a short run used, and
  `act_hessian/check_hessians.py` recomputes every Hessian from them. On K2h (4 chunks of 512 tokens, layers 9 and 47,
  CPU build): the worst expert differs by 1.4 × 10⁻⁷ (gate/up) and 1.8 × 10⁻⁷ (down) of the largest entry, row
  counts are exact, and the experts' gate/up Hessians sum to 6 × the all-token Hessian within 7 × 10⁻⁸.
- **The round-to-nearest baseline is llama-quantize's own bytes.** `imat_ggml` in gptq_head.py computes Q5_0 (or Q4_0)
  with an importance matrix in ggml's float32 arithmetic and summation order. It reproduces every row of K2h's Q5_0
  tensors that `test_gptq_head.py` checks (attention, shared expert, and expert slices in layers 9 and 47, with
  Unsloth's importance matrix), so K2h's Q5_0 tensors are exactly llama-quantize's output and the proxy's baseline is
  the mix itself. The float64 version `imat` gives the same error but only 68–95 % of the same rows, because near-ties
  fall either way.
- **Reproducible GPTQ needs a fixed BLAS thread count.** The block updates' summation order changes with the number
  of OpenBLAS threads: 8 instead of 12 threads changes 2.5–4.5 % of the Q5_0 blocks (same proxy error to three
  digits). With 12 threads, `gptq_body.py` reproduces E1's tensors byte for byte. The jobs pin the thread count.
- **`gptq_body.py`** takes named groups (`k2h_q`, `k2h_kvo`, `k2h_sh`, all three as `k2h_q5`; `k2h_q8` for every
  always-read tensor K2h keeps in Q8_0) or a regex, maps each tensor to the graph tensor that feeds it, and runs
  tensors in parallel with `--workers`. The "= mix" column gives the share of rows where round to nearest reproduces
  the mix's bytes (100 % when the mix holds that type).
- **`gptq_experts.py`** is the offline expert check (Q114): for chosen layers and a sample of experts spread over the
  ranking by routed rows, the H-weighted output error of the mix, round to nearest and GPTQ for gate, up and down,
  summed over each expert's rows, so the layer ratio weights experts by use. Experts with fewer rows than inputs get
  the all-token Hessian blended in and 10 × damping. Kill: GPTQ / round-to-nearest error above 0.85.

Planned runs: Hessians of K2h with F16 routers (the release candidate) on the same wikitext-2 train text as E1; per-expert
Hessians for layers 9, 25, 41 and 47 from 1.02 M tokens, about 24,000 rows per expert; the expert check on 64 experts
per layer; K2h with GPTQ on its 120 Q5_0 always-read tensors, scored against K2h on the same GPUs; and the
round-to-nearest/GPTQ ratio for every Q8_0 always-read tensor, which E3 uses to rank demotions.
