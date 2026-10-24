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
