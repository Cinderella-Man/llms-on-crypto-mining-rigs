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
