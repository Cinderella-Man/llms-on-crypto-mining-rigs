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
- **A caveat on the reference.** Q8_0 tensors in a candidate are byte-identical to the reference's and add no
  divergence. UD-Q4_K_XL keeps attention q, k, v and output in Q8_0 in 42 of 49 layers (Q5_K in the other 7); G3
  keeps q in Q8_0 only in layers 0–24 (Q5_0 in 25–48) and output in 0–32 (Q5_0 in 33–48). The read-token gap, which
  runs through long-range attention, is the part this flatters most. A run against BF16 on a few chunks settles it.
- Same top token on agent text: 87.7–87.9 % for every file; no non-finite values.
