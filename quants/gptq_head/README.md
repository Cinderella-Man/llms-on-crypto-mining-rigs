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
