# Expert overlap between consecutive tokens (Q51)

How many of Flint's routed experts do neighbouring generated tokens share? A speculative verify step of n tokens reads
every distinct expert its tokens pick, so this decides how much a long draft block costs and whether a kernel that reads
each shared expert once pays ([speculation plumbing](../../docs/research/notes/speculation_plumbing.md), registry row
Q51 in [quants/experiments.md](../../quants/experiments.md)).

## Method

1. [prompts.py](prompts.py) builds 38 prompts in four groups: 10 chat, 10 code writing, 8 whole-file edits of this
   repository's own scripts ("output the complete updated file"), and 10 agent turns with four coding tools (6 first turns,
   4 turns after a `read_file` result).
2. [generate.py](generate.py) renders them with llama-server's `/apply-template` (Flint's GGUF template, `--jinja`),
   tokenizes them, and generates up to 3,072 tokens each at the server's default sampling, which llama.cpp takes from the
   GGUF (the makers' settings: temperature 1.0, top-k 20, top-p 0.95, min-p 0.05), with a fixed seed per prompt. It keeps
   the exact prompt and generated token ids.
3. [route_dump.cpp](route_dump.cpp) (`llama-route-dump`, built inside a llama.cpp tree) runs each sequence through the
   model in batches and records, from the graph's `ffn_moe_topk-<layer>` and `ffn_moe_weights-<layer>` tensors, the six
   experts and their router scores for every position and MoE layer. Every token requests its logits, because llama.cpp
   runs the last layer only for output rows.
4. [analyze.py](analyze.py) (standard library only) counts the distinct experts in every window of n consecutive
   generated tokens, per layer, and compares them with independent random routing, 256 × (1 − (1 − 6/256)^n).

Run on a 7-GPU quant machine with Unsloth's Flint Q8_0 on two V100s: 65,558 generated tokens (chat 17,432, code 21,416,
edits 23,747, agent 2,963) over 47 MoE layers.

## Results

Distinct experts per layer touched by n consecutive generated tokens:

| Tokens | 1 | 2 | 4 | 8 | 16 | Shared by adjacent tokens |
|---|---|---|---|---|---|---|
| Chat | 6 | 10.96 | 19.84 | 35.16 | 59.51 | 1.04 |
| Code | 6 | 11.01 | 20.03 | 35.40 | 59.51 | 0.99 |
| Whole-file edits | 6 | 11.16 | 20.44 | 36.08 | 60.14 | 0.84 |
| Agent turns | 6 | 11.02 | 19.96 | 35.29 | 59.56 | 0.98 |
| All generated | 6 | 11.05 | 20.13 | 35.58 | 59.74 | 0.95 |
| Prompt tokens | 6 | 11.13 | 20.24 | 35.56 | 59.12 | 0.87 |
| Random routing | 6 | 11.86 | 23.17 | 44.24 | 80.84 | 0.14 |

- **Flint has little routing locality.** Eight consecutive tokens touch 80 % of the experts random routing would, and
  adjacent tokens share about one expert of six. On Qwen3-Coder-30B-A3B eight tokens touch 51 % of random.
- **Copying does not help.** Whole-file edits, where the model mostly repeats its input, route as independently as chat.
- **By layer:** layers 17–21 share the most (30–32 distinct experts at 8 tokens, 1.3–1.5 shared by neighbours), layers
  2–6 the least (40–42, 0.2–0.5).
- **Route weights:** Flint's router is a sigmoid with the top six renormalized. The mean weights by rank are 0.31, 0.21,
  0.16, 0.13, 0.11 and 0.10. The lightest expert is under 5 % of its token's route on 6.4 % of token-layers and under 2 % on
  0.3 %, so dropping light experts would remove real signal for little saving.
- **Expert use is spread out:** in a layer, the most-used 10 %, 25 % and 50 % of experts take 23 %, 44 % and 71 % of the
  routed reads, and all but two of the 12,032 (layer, expert) pairs are used.

What it means for speculation on the rig: an n-token verify reads about 0.92 (n = 2) to 0.74 (n = 8) times n tokens'
worth of routed-expert bytes, so a kernel that reads each shared expert once saves only 8–26 % of them, and draft blocks
longer than 7 tokens cannot rely on overlap.

## Running it

```bash
python -I prompts.py prompts.jsonl <source files to edit...>
llama-server -m flint-Q8_0.gguf -c 65536 -np 4 -ngl 99 -fa on --jinja --cache-ram 0 --port 18451 &
python -I generate.py http://127.0.0.1:18451 prompts.jsonl gen.jsonl seqs.txt 4 3072
# copy this folder to <llama.cpp>/examples/route-dump, add add_subdirectory(route-dump) to examples/CMakeLists.txt
ROUTE_IN=seqs.txt ROUTE_OUT=routes.bin llama-route-dump -m flint-Q8_0.gguf -ngl 99 -c 16384 -b 512 -ub 512 -fa on
python -I analyze.py routes.bin gen.jsonl q51.json
```
