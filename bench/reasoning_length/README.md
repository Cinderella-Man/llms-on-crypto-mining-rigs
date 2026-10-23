# Reflection words in Flint's reasoning (Q94)

Can a logit bias against self-reflection words shorten Flint's reasoning? NoWait ([arXiv 2506.08343](https://arxiv.org/abs/2506.08343))
masks tokens such as "Wait", "Hmm" and "Alternatively" during reasoning and reports 13–31 % fewer reasoning tokens on
RL-trained models ([unconventional ideas](../../docs/research/notes/unconventional_ideas.md), idea 2). The bias can only
remove what is there, so this counts those words in Flint's own reasoning first (registry row Q94).

## Method

[reflections.py](reflections.py) (standard library only) reads the reasoning block of each output, `<think>` …
`</think>`, or to the end when the output ran out of tokens.

- **Words.** Reflection words: wait, hmm, actually, alternatively, oh, ah, double-check, recheck, verify, reconsider,
  rethink, and phrases such as "but wait", "hold on", "let me double-check", "let me verify". Contrast words, counted
  apart because they also join ordinary clauses: but, however, maybe, perhaps.
- **Two ways of counting.**
  - With token ids, as in Q51's transcripts, it reads Flint's vocabulary from a GGUF header and counts word-initial tokens
    whose text is the word: per 1,000 reasoning tokens, at the start of a sentence or inside one, and by tenth of the
    reasoning.
  - Without token ids (`--text`, e.g. llama-server's `content` from the BFCL runs) it counts per 1,000 words.
- **What a bias aims at.** The reasoning in sentences, and in paragraphs, that a reflection word opens: the re-checks and
  changes of course themselves. A bias that only removes these spans removes at most that share. NoWait's larger cuts come
  from models whose reasoning is made of such loops.

Data:
- **Q51's transcripts** ([expert_overlap](../expert_overlap/README.md)): 38 outputs of Flint Q8_0 at its default sampling
  (temperature 1.0), 43,373 reasoning tokens (66 % of the output): chat, code, whole-file edits, agent turns.
- **Q41's BFCL v4 single-turn runs**: 3,476 tool-call answers, 3,475 of them with a reasoning block (first half of E1,
  attempt 1), from two quantized files (G3 and Unsloth's UD-Q4_K_XL) at the makers' sampling; 317,922 words of reasoning, 61 % of the output text.

## Results

Q51's transcripts, per 1,000 reasoning tokens:

| Output | Reasoning tokens | Reflection words (sentence-initial) | Contrast words (sentence-initial) | Reasoning in sentences a reflection word opens | in paragraphs it opens |
|---|---|---|---|---|---|
| Chat | 7,950 | 2.1 (1.4) | 6.3 (0.6) | 2.1 % | 2.0 % |
| Code | 11,233 | 2.5 (1.5) | 8.8 (2.0) | 3.4 % | 2.2 % |
| Whole-file edits | 21,577 | 1.3 (1.3) | 12.7 (7.5) | 1.1 % | 2.9 % |
| Agent turns | 2,613 | 12.2 (8.4) | 13.4 (2.7) | 11.9 % | 23.2 % |
| All | 43,373 | 2.4 (1.8) | 10.5 (4.5) | 2.5 % | 3.8 % |

BFCL v4 single-turn, per 1,000 reasoning words (Q51's reasoning has 1.62 tokens per word: 4.0 reflection and 17.1
contrast words per 1,000 words):

| BFCL category | Answers | Reasoning words | Reflection words (sentence-initial) | Contrast words (sentence-initial) | Reasoning in sentences a reflection word opens | in paragraphs it opens |
|---|---|---|---|---|---|---|
| simple | 394 | 22,594 | 0.5 (0.3) | 6.0 (1.2) | 0.5 % | 0.0 % |
| multiple | 202 | 11,166 | 1.0 (0.4) | 3.9 (0.7) | 0.7 % | 0.0 % |
| parallel | 416 | 38,124 | 2.3 (1.7) | 8.1 (3.3) | 1.7 % | 1.4 % |
| irrelevance | 224 | 21,429 | 3.0 (2.4) | 16.4 (5.9) | 3.1 % | 4.4 % |
| live | 2,239 | 224,609 | 3.4 (2.6) | 16.3 (6.1) | 3.0 % | 4.7 % |
| All | 3,475 | 317,922 | 3.0 (2.3) | 14.2 (5.2) | 2.6 % | 3.8 % |

- **Flint seldom reflects.** "Wait" occurs 18 times and "Hmm" 18 times in 43,373 reasoning tokens; over the BFCL answers
  "wait" occurs 150 times and "hmm" 94 times in 317,922 words. All reflection words together: 2.4 per 1,000 reasoning
  tokens (Q51) and ~1.9 per 1,000 tokens (BFCL, 3.0 per 1,000 words), below the note's threshold of ~5 per 1,000.
- **Its hedges are "But" and "maybe".** 271 "but" and 167 "maybe" against 61 "actually" (Q51). They start ordinary
  clauses ("But the user wants…", "maybe use argparse"); a bias on them would act on the reasoning's content, not on
  re-checks.
- **Its reasoning style is terse planning.** The most frequent sentence openers are "We", "But", "if", "Let's", "The",
  "We'll" and code keywords; "Actually" ranks 16th (43) and "Hmm" 40th (16).
- **Where:** reflection words lean towards the end of the reasoning, the final check: 44 % of them fall in its last
  three tenths (8, 8, 10, 9, 13, 4, 7, 18, 15, 14 per tenth over all outputs).
- **Agent turns are the exception**: 12 per 1,000 tokens and 12–23 % of their reasoning in re-checks ("Hmm", "Actually",
  "Wait — the user says…"), from the four turns that follow a `read_file` result. The BFCL answers, single-turn tool calls,
  don't show it (0.5–3.4 per 1,000 words).

## Estimate

- A reflection bias aims at 2.5–3.8 % of Flint's reasoning (both datasets): 1.5–2.5 % of the output tokens, ≤ 1.03×
  per answer. NoWait's 13–31 % came from models whose reasoning is full of these loops; Flint's is not, so its kill
  criterion, under 10 % fewer reasoning tokens, is the expected outcome for chat, code, edits and single-turn tool calls.
- **What would still be worth testing:** multi-turn agent sessions, where the after-tool turns re-check heavily (12–23 % of
  their reasoning here, from four turns), in Q41's multi-turn lane with and without the bias.
- **For shorter reasoning in general, the other half of idea 2 is the lever:** self-training on Flint's shortest correct
  samples, which shortens planning, not only re-checks.

## Running it

```bash
python -I reflections.py gen.jsonl flint-Q8_0.gguf --out q94.json      # Q51 transcripts, token ids and vocabulary
python -I reflections.py --text bfcl_raw/*.jsonl --out q94_bfcl.json   # llama-server outputs, per 1,000 words
```
