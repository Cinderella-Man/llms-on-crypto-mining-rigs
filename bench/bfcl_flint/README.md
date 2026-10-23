# BFCL tool-calling check for Flint

A small, faithful harness for checking whether a quantized Flint file still calls tools correctly: 300 single-turn
tasks from the Berkeley Function Calling Leaderboard v3 (BFCL), scored with BFCL's own AST checker, run against
llama-server. It exists because KL divergence misjudges Flint's makers' Q4_1, which was tuned by distillation after
quantizing (results in [quants/experiments.md](../../quants/experiments.md), Q30).

## How it works

- **Tasks** ([task_ids.txt](task_ids.txt)): 50 simple, 50 multiple, 40 parallel, 40 parallel_multiple, 40 live_simple,
  40 live_multiple, 20 irrelevance, 20 live_irrelevance, a seeded sample per category from the `bfcl-eval==3.0.0` wheel
  (the last release with the v3 data). The per-category counts are those of the terelabs Flint comparison.
- **Prompts** ([prepare.py](prepare.py)): BFCL's own conversion of each task's functions to OpenAI-style tools,
  rendered on the client with the chat template embedded in the GGUF ([gguf_meta.py](gguf_meta.py) extracts it), as
  `apply_chat_template(messages, tools=tools, add_generation_prompt=True)` would.
- **Generation** ([run.py](run.py)): llama-server's native `/completion` endpoint, temperature 0, `cache_prompt false`,
  no BOS. No `--jinja` and no server-side tool parsing.
- **Scoring** ([score.py](score.py)): Flint's `<tool_call>name<arg_key>…</arg_key><arg_value>…</arg_value></tool_call>`
  output is decoded with the tool schema's types and checked with BFCL v3's `ast_checker` and relevance logic, unmodified.
  [selftest.py](selftest.py) renders every task's ground truth through the template and scores it (292 of 300 pass;
  the rest are dataset quirks). [pairs.py](pairs.py) compares two runs task by task (McNemar's test).

## Running it

```bash
pip download --no-deps bfcl-eval==3.0.0 && python -m zipfile -e bfcl_eval-3.0.0-py3-none-any.whl bfcl-v3
export BFCL_ROOT=$PWD/bfcl-v3          # needs numpy, jinja2, requests, tqdm, pyyaml, tenacity, overrides
python gguf_meta.py flint.gguf > template.jinja
python prepare.py template.jinja tasks.jsonl
llama-server -m flint.gguf -c 16384 -np 2 -ngl 99 -fa on --cache-ram 0 &
python run.py http://127.0.0.1:8080/completion tasks.jsonl raw/myfile.jsonl 2
python score.py tasks.jsonl raw out BF16 myfile
```

## Reading the results

Flint reasons in a `<think>` block before answering, and any numeric change (GPU model, batching, quantization)
changes 11–21 % of its calls, so two runs of the same file on two GPU types agree on only ~80 % of calls. With 300 tasks
the 95 % interval is about ±4.5 points; telling files 3 points apart needs ~500 paired tasks, 1.5 points ~2,000.
