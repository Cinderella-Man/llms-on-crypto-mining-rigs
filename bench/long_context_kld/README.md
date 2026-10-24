# Long-context KLD on agent conversations

Does a quant that is close to the original on 512-token wikitext chunks stay close on what an agent actually reads:
8,000-token conversations full of tool schemas, tool output, reasoning and calls? This folder measures KL divergence
there, against a reference model, with the token-level detail to see where a difference comes from
([research note](../../docs/research/notes/long_context_quality.md), measurements M1 and M2).

## Pieces

- **Material.** Flint's own BFCL v4 multi-turn conversations, from the runs of [bfcl_flint](../bfcl_flint/README.md).
  `multi_turn.py render` replays each stored conversation through BFCL's tool backends with the model's recorded
  outputs, so the tool results are regenerated exactly, and writes the last step's prompt (the whole history with
  every reasoning block, rendered with the GGUF's chat template) plus the last output.
- **`build_eval.py`** picks conversations (per producing model and BFCL category, each task once), cuts each to exactly
  `--ctx` tokens and joins them, so that `llama-perplexity -c <ctx>` scores one conversation per chunk. It checks that
  every rendered prompt has exactly the token count llama-server reported when the model saw it, that every cut
  re-tokenizes to itself, and that the joined file splits exactly at the chunk boundaries.
- **[perplexity-parse-special-kld-dump.diff](../../patches/perplexity-parse-special-kld-dump.diff)** lets
  `llama-perplexity` tokenize chat markup as special tokens (`--parse-special`) and dump one line per scored token
  (`LLAMA_KLD_DUMP=<file>` with `--kl-divergence`). Pass `--no-escape` with chat text: by default `-f` text is
  unescaped, which would turn every `\n` inside the JSON of tool schemas and results into a newline.
- **`analyze.py`** compares dumps chunk by chunk with a reference file: KLD, same top token, the ratio to the reference
  with a bootstrap interval over chunks, by position and by token class (tool schemas, user turns, tool output,
  reasoning, calls, answers, markup, the first token after each `</think>`, and long-range copies: tokens that end an
  8-token run first seen more than 512 tokens earlier, the needles only the global-attention layers can reach).
- **`analyze.py --pooled --diff A-B,...`** adds token-pooled ratios (sums over a group's tokens) for read, written,
  tool-schema, tool-output and user tokens, and paired differences between two files in units of the reference
  file's KLD on each group. Intervals and standard errors come both from chunks and from clusters of chunks:
  conversations of the same scenario share their tool schemas, so chunks whose token prefixes agree into the scored
  half are resampled together (Q106's 24 chunks form 15 such clusters).
- **`stop_decision.py`** reads the same dumps at the positions that decide how long Flint reasons: every `</think>`
  that closes a reasoning block (where the producing model chose to stop) and every hesitation marker in the
  reasoning ("wait", "but", "maybe", "actually", ...: the word list of [answer_time](../answer_time/lengths.py), at
  the word's first token). It reports each file's log-probability of the actual token there, minus the reference
  file's and minus the dumps' reference model's, with bootstrap intervals over chunks, overall and by the model that
  produced the conversation.

## Running it

```bash
# 1. render (BFCL's venv; TEMPLATE = the GGUF's chat template)
python multi_turn.py render TEMPLATE.jinja rendered.jsonl ud=e2-ud.jsonl g3=e2-g3.jsonl --min-tokens 8192
# 2. build: 24 chunks of 8,192 tokens
PYTHONPATH=<llama.cpp>/gguf-py python3 build_eval.py rendered.jsonl flint-Q8_0.gguf <llama.cpp>/build/bin/llama-tokenize agent/
# 3. reference logits, once
llama-perplexity -m flint-Q8_0.gguf -f agent/eval.txt --parse-special --no-escape -c 8192 --chunks 24 \
  -b 2048 -ub 512 -ngl 99 -fa on --kl-divergence-base q8.agent.kld
# 4. every candidate, same flags and GPUs (token ids come from the reference file)
LLAMA_KLD_DUMP=dump-x.tsv llama-perplexity -m x.gguf -f agent/eval.txt -c 8192 -b 2048 -ub 512 -ngl 99 -fa on \
  --kl-divergence-base q8.agent.kld --kl-divergence
# 5. compare
python3 analyze.py --ref ud --chunks agent/chunks.jsonl --model flint-Q8_0.gguf ud=dump-ud.tsv x=dump-x.tsv
python3 stop_decision.py --ref ud --chunks agent/chunks.jsonl --model flint-Q8_0.gguf --base-name Q8_0 \
  ud=dump-ud.tsv x=dump-x.tsv
```

The reference file holds 150,148 16-bit values per scored token: 29.5 GB for 24 chunks of 8,192 tokens. Delete it
when the candidates are done; it is cheaper to make again than to keep.

## Results

### Flint, 2026-10-24

Reference: Unsloth's Flint Q8_0. Material: 24 conversations of 8,192 tokens from Flint's BFCL v4 multi-turn pilot runs
(12 produced by UD-Q4_K_XL, 12 by G3, 3 per category: base, missing function, missing parameter, long context;
98,280 scored tokens at positions 4,097–8,191). Wikitext-2 test at 8,192 tokens (12 chunks) and at 512 (80 chunks)
for comparison. All files on the same two RTX 3090s with the same build and flags (`-b 2048 -ub 512 -fa on`), the
rescale on for UD-Q4_K_XL and off for the others. G3h is G3 with the GPTQ Q5_0 output head
([gptq_head](../../quants/gptq_head/README.md)); K1h and K2h are later mixes with the same head. Ratios are KLD
against UD-Q4_K_XL's on the same tokens, with 95 % bootstrap intervals over chunks.

| File | Agent text, 8k: KLD | ratio | written tokens: ratio | read tokens: ratio | Wikitext 8k: ratio | Wikitext 512: ratio |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 0.478 | 1 | 1 | 1 | 1 (KLD 0.138) | 1 (KLD 0.124) |
| G3 | 0.494 | 1.033 (1.008–1.056) | **0.887 (0.844–0.936)** | 1.037 (1.003–1.086) | 0.986 (0.930–1.037) | 0.902 (0.854–0.951) |
| G3h | 0.495 | 1.035 (1.011–1.059) | 0.957 (0.915–1.009) | 1.038 (1.004–1.084) | 1.004 (0.949–1.056) | 0.925 (0.876–0.973) |
| K2h | 0.494 | 1.032 (1.013–1.051) | 1.013 (0.971–1.059) | 1.046 (1.013–1.100) | 0.978 (0.918–1.030) | 0.935 (0.887–0.984) |
| K1h | 0.498 | 1.042 (1.025–1.058) | 1.038 (0.997–1.079) | 1.051 (1.019–1.100) | 0.992 (0.931–1.045) | 0.950 (0.904–1.001) |

"Written" tokens are the ones the model samples in an agent loop: reasoning, calls and answers (45.6 % of the
scored tokens). "Read" tokens are prefilled text it never samples: tool schemas, user turns and tool output (52.6 %).

- **The 512-token advantage does not carry to 8,192 tokens.** G3's KLD is 10 % below UD-Q4_K_XL's at 512 tokens
  (same Q8_0 reference; 16 % against BF16), equal at 8,192 on wikitext, and 3 % above it on agent text.
- **On agent text the excess sits in what the model reads, not in what it writes.** Most of the KLD is on tool
  schemas (0.97–1.01 nats per token) and tool output (0.70–0.72), and on long-range copies: tokens that repeat an
  8-token run first seen more than 512 tokens earlier (29.6 % of scored tokens, 1.18–1.23 nats). Every file is 3–5 %
  above UD-Q4_K_XL there. On the model's own output G3 stays 11 % below it (reasoning 0.85, answers 0.86, calls
  0.95, the first token after `</think>`, which decides between a call and an answer, 0.69 on 230 tokens), and G3h
  4 % below. K1h and K2h are level with UD-Q4_K_XL on written tokens.
- **Where the KLD is.** KLD falls along the chunk (UD-Q4_K_XL 0.82 at positions 4,096–5,119, 0.20 at 7,168–8,191)
  because tool schemas come first in most conversations; reasoning carries 0.015–0.018 nats per token, a fiftieth of
  the tool schemas'.
- **The reference.** Unsloth's Q8_0 is not a neutral reference here: against BF16 it has KLD 0.27 on agent text
  itself (below). Its tensors also differ from our Q8_0 tensors (only 4 of G3's 321 Q8_0 tensors are byte-identical to
  it), so it is not simply the files' common ancestor. The BF16 check below repeats the comparison without it.
- Same top token on agent text: 87.7–87.9 % for every file; no non-finite values.

### Against BF16 (bias check)

BF16 reference logits on 12 of the 24 chunks (3 per category, 6 produced by each model; experts of layers 21–48 in
system RAM), every file rescored against them with the same flags as above. Ratios to UD-Q4_K_XL with 95 % bootstrap
intervals over the 12 chunks:

| File | KLD | Same top | Ratio | Read tokens | Written tokens | Long-range copies |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 0.316 | 90.32 % | 1 | 1 (KLD 0.815) | 1 (KLD 0.018) | 1 (KLD 0.975) |
| Q8_0 (Unsloth) | 0.267 | 92.08 % | 0.845 (0.794–0.872) | 0.837 (0.789–0.884) | 0.537 (0.505–0.572) | 0.874 |
| G3 | 0.329 | 90.73 % | 1.042 (1.027–1.053) | 1.064 (1.019–1.136) | 0.892 (0.847–0.947) | 1.056 |
| K2h | 0.327 | 90.43 % | 1.036 (1.020–1.054) | 1.039 (0.999–1.081) | 0.999 (0.948–1.049) | 1.046 |

On the same 12 chunks the Q8_0 reference gives G3/UD-Q4_K_XL 1.012 (0.965–1.062), read 1.029, written 0.901.

- **The read-token gap is real, not the reference's doing.** Against BF16 it is larger (G3 +6.4 % on read tokens,
  +5.6 % on long-range copies) than against Q8_0 on the same chunks (+2.9 %). The Q8_0 reference was hiding part of
  it, not creating it.
- **So is G3's edge on what the model writes:** 0.892 against BF16, 0.901 against Q8_0. K2h is level with
  UD-Q4_K_XL on written tokens either way.
- Read tokens are where every quant struggles: even Unsloth's Q8_0 has KLD 0.70 there against BF16, 84 % of
  UD-Q4_K_XL's, while on written tokens it has half of UD-Q4_K_XL's.

### Global-layer attention in Q8_0 (Q107)

G3g and K2hg are G3 and K2h with every attention tensor (q, k, v, output) of Flint's 14 global layers (0, 1, 5, 9,
…, 45, 48: no sliding window, no RoPE, the only layers that see more than 512 tokens back) in Q8_0, quantized from
the same BF16 file (byte-identical to llama-quantize's Q8_0). They test whether the read-token gap comes from those
layers' precision.

| File | GB per token | 512-token KLD (vs BF16) | Agent 8k vs Q8_0: ratio | read | written | Agent 8k vs BF16: ratio | read | written |
|---|---|---|---|---|---|---|---|---|
| G3 | 2.327 | 0.1040 | 1.033 (1.008–1.056) | 1.037 | 0.887 | 1.042 (1.027–1.053) | 1.064 | 0.892 |
| G3g | 2.365 | 0.1031 | 1.032 (1.006–1.058) | 1.037 | 0.865 | 1.035 (1.023–1.046) | 1.074 | 0.883 |
| K2h | 2.140 | 0.1092 | 1.032 (1.013–1.051) | 1.046 | 1.013 | 1.036 (1.021–1.054) | 1.039 | 0.999 |
| K2hg | 2.192 | 0.1089 | 1.049 (1.020–1.073) | 1.055 | 0.999 | 1.050 (1.009–1.070) | 1.073 | 1.005 |

Ratios to UD-Q4_K_XL (24 chunks against Q8_0, 12 against BF16). The 512-token KLD is the 80 × 512 wikitext-2 score
on the same GPUs (G3 0.1040, K2h 0.1092 there).

- **The global layers' attention precision is not the cause.** Restoring it to Q8_0 leaves the read-token and
  long-range-copy ratios where they were (G3g 1.049 and K2hg 1.073 on copies against BF16, G3 1.056, K2h 1.046)
  and costs 38–52 MB per token. It is free at 512 tokens (−0.0009 and −0.0003).
- Written tokens move by less than their noise (G3g 0.883 against G3's 0.892).

### Where reasoning stops (Q123)

Does a quantized file change where Flint ends its reasoning? A file that gives the actual `</think>` less probability
than UD-Q4_K_XL at the points where reasoning ended is less willing to stop; one that gives hesitation markers more
probability re-checks more. Both would lengthen the reasoning, so both are read from the dumps above with
`stop_decision.py`: 230 stops (every reasoning block in the scored halves is non-empty; 112 in conversations
produced by G3, 118 by UD-Q4_K_XL) and 842 markers (maybe 272, but 266, actually 104, perhaps 65, however 53,
wait 43, others 39).

Difference in log-probability of the actual token, file minus UD-Q4_K_XL, in nats, 95 % bootstrap over the 24 chunks:

| File | At `</think>` | At hesitation markers | `</think>` vs Q8_0 | Markers vs Q8_0 |
|---|---|---|---|---|
| UD-Q4_K_XL (mean log p) | (−0.163, p 0.889) | (−0.696, p 0.640) | +0.001 | −0.004 |
| G3 | −0.004 (−0.035 to +0.017) | +0.005 (−0.009 to +0.022) | −0.003 | +0.001 |
| G3h | +0.005 (−0.021 to +0.026) | −0.010 (−0.027 to +0.011) | +0.007 | −0.014 |
| K2h | +0.007 (−0.022 to +0.028) | −0.006 (−0.022 to +0.013) | +0.008 | −0.010 |
| K1h | +0.008 (−0.015 to +0.027) | −0.009 (−0.025 to +0.009) | +0.009 | −0.013 |
| G3g | −0.007 (−0.035 to +0.014) | +0.002 (−0.010 to +0.017) | −0.006 | −0.002 |
| K2hg | +0.001 (−0.026 to +0.022) | −0.011 (−0.022 to +0.005) | +0.003 | −0.015 |

- **No file moves the stop decision.** Every shift is under 0.011 nats, a fifth of the 0.05-nat threshold set in
  advance ([answer time](../../docs/research/notes/answer_time.md)), and every interval covers 0. At the 230 stops
  every file gives the actual `</think>` 0.886–0.894 probability on average. Quant choice is not a lever on reasoning
  length; the measured lengths agree (G3/UD-Q4_K_XL 1.000 on 3,477 single-turn tasks).
- **The producing model shows, but cancels.** Split by producer, files give markers in G3's conversations
  0.007–0.023 nats more probability than UD-Q4_K_XL does and markers in UD-Q4_K_XL's 0.016–0.035 less; at stops the
  signs flip (−0.025 to −0.009 in G3's, +0.010 to +0.024 in UD-Q4_K_XL's). With half the conversations from each
  producer the pooled difference balances it, and every split value stays below 0.05 too.
- **Against BF16** (the 12 chunks of the bias check): at stops UD-Q4_K_XL −0.002, Q8_0 −0.008, G3 −0.015, K2h −0.005
  nats; at markers all within ±0.017. The files' own log-probabilities on those chunks are identical to the 24-chunk
  runs, so the comparisons to UD-Q4_K_XL there are the same numbers on half the chunks. On that half one value's
  interval excludes 0, K2hg at markers (−0.017, −0.029 to −0.004): fewer hesitation markers, and a third of the
  threshold.

### K2 with the Q8_0 head and F16 routers (Q126)

K2-r16 is K2 (Q77's round-5 mix; K2h is the same file with the GPTQ Q5_0 head) with its Q8_0 head and F16 routers.
With the screened head, which is exact, it is the lead file for our engine. All files below ran on the same two V100s with the same sm_70
build and flags, against the same Q8_0 reference; UD-Q4_K_XL was rescored there because the card type alone moves every
file's KLD by 1–3 %. Token-pooled ratios to UD-Q4_K_XL, 95 % bootstrap over the clusters of chunks that share tool
schemas, on the 23 chunks where every file is finite (see the last point):

| File | KLD | Written | Read | Tool schemas | Tool output |
|---|---|---|---|---|---|
| UD-Q4_K_XL | 0.487 | 1 | 1 | 1 | 1 |
| K2-r16 | 0.518 | **0.929 (0.890–0.968)** | 1.068 (1.043–1.094) | 1.074 (1.042–1.109) | 1.045 |
| K2h (GPTQ Q5_0 head, F32 routers) | 0.513 | 1.003 (0.963–1.040) | 1.056 (1.022–1.081) | 1.064 (1.021–1.099) | 1.023 |

- **K2-r16 writes 7 % better than UD-Q4_K_XL and reads tool schemas 7 % worse.** On all 24 chunks: written 0.930
  (0.893–0.965), read 1.068, schemas 1.074.
- **The Q8_0 head takes back what the GPTQ head cost on written tokens:** K2-r16 − K2h is −0.074 ± 0.012 of
  UD-Q4_K_XL's written KLD, the same size as G3h − G3 (+0.073). The F16 routers move read and schema tokens by
  +0.012 ± 0.008 and +0.011 ± 0.010, within the routing-flip noise.
- **The card type moves files together on written tokens but not quite on read tokens.** On the RTX 3090s, UD-Q4_K_XL
  and K2h each have 2.6 % more written-token KLD than on the V100s; their read ratio is 1.034 there and 1.056 here.
  Compare files only on the same cards.
- **With the MoE down-projection rescale off, a file can turn NaN on some inputs.** K2h gave NaN for every scored
  token of one chunk on the V100s (`multi_turn_long_context_10`), and so did G3 with UD-Q4_K_XL's attention (Q127
  below). On that chunk both are finite with the rescale on; K2-r16, whose body differs from K2h's only in the
  router's precision, happened to be finite with it off. The rescale only acts where FP16 would overflow: G3 with it
  on is bit-identical to G3 with it off on all 24 chunks. Run every file with the rescale on.

### Which tensors carry UD-Q4_K_XL's edge on tool schemas (Q127)

Three probes from G3 (Q8_0 head, F32 routers; probes, not candidates), each scored on the same V100s as G3 and
UD-Q4_K_XL. Paired differences to G3 in units of UD-Q4_K_XL's KLD on the group, ± standard error over chunks / over
the clusters of chunks that share tool schemas (G3 + UD's attention on the 23 finite chunks):

| Probe | Tool schemas | Written | Read |
|---|---|---|---|
| G3 + UD-Q4_K_XL's attention (all q, k, v, output: 196 tensors) | +0.010 ± 0.011 / 0.016 | +0.002 ± 0.010 / 0.012 | +0.005 ± 0.010 / 0.015 |
| G3 + UD-Q4_K_XL's routed experts (141 tensors; rescale on) | +0.019 ± 0.011 / 0.018 | **+0.166 ± 0.029 / 0.032** | +0.013 ± 0.011 / 0.017 |
| G3a: G3 + Q8_0 for the 19 sliding-layer attention tensors UD-Q4_K_XL keeps at Q8_0 | +0.003 ± 0.002 / 0.003 | −0.010 ± 0.005 / 0.005 | +0.002 ± 0.002 / 0.003 |
| G3 with the rescale on | 0 | 0 | 0 |

On these cards G3 itself is 1.032 of UD-Q4_K_XL on tool schemas (0.994–1.063 by clusters), 1.025 on read tokens and
0.852 on written tokens.

- **No probe moves tool-schema tokens toward UD-Q4_K_XL.** Its attention, its experts, the attention precision it
  keeps in the sliding layers, and the rescale each leave G3's schema ratio where it was or raise it. The schema gap
  is not attributable to a tensor class, so no bytes should be spent chasing it.
- **G3's advantage on written tokens comes from its routed experts.** With UD-Q4_K_XL's experts (IQ4_XS, Q4_K and Q5_K
  at fewer bytes) G3 writes at UD-Q4_K_XL's level: written ratio 1.018.
- Q8_0 for the 19 sliding-layer attention tensors buys −1 % on written tokens for 60 MB per token.

