# Q51: 38 held-out Flint transcripts

The test set every drafter score is measured on, and part of the GPTQ head's calibration
([reproduce/model.md](../../model.md), step 3; the drafter's guide,
[release/flint-v2.1/reproduce/drafter.md](../../../../flint-v2.1/reproduce/drafter.md), steps 3 and 6).

| File | What |
|---|---|
| `prompts.jsonl` | the 38 prompts: 10 chat, 10 code writing, 8 whole-file edits of this repository's scripts, 10 agent turns with four coding tools |
| `gen.jsonl` | Flint's answers with the exact prompt and answer token ids, 65,558 generated tokens |
| `seqs.txt` | the same sequences in the dump tools' input format |

Made on 2026-10-23 with [bench/expert_overlap](../../../../../bench/expert_overlap/README.md) (`prompts.py`,
`generate.py`): Unsloth's Flint Q8_0 in llama-server, the sampling the GGUF carries (temperature 1.0, top-k 20,
top-p 0.95, min-p 0.05), a fixed seed per prompt, up to 3,072 tokens per answer. None of these prompts or their source
files are in the drafter's training prompts.

The answers are Flint's outputs (Flint is Apache-2.0); the edit prompts quote this repository's own scripts (MIT).
