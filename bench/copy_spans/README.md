# Copy spans: how much of a model's agent output is copied from its context

Speculation that copies from the context (a prompt-lookup guesser, a long copy chain) pays only if a coding agent's
output repeats text it was given: file contents it edits, commands it ran before, error lines it quotes, its own
earlier reasoning. This folder measures that on the request and response logs of the agent check
([agent_check](../agent_check/README.md)): which share of the generated tokens lies in verbatim repeats of the
context, how long the repeated runs are, where they start, which parts of a response they sit in, and what a
prompt-lookup guesser (and a perfect one) would accept on the same transcripts. The result for Flash-Next is in
[docs/data/2026-11-04_copy_spans](../../docs/data/2026-11-04_copy_spans/README.md) (registry rows Q617 and Q624 in
[quants/experiments.md](../../quants/experiments.md)).

Standard library only (Python 3.10+), one external program: `llama-tokenize` from a llama.cpp build that knows the
model's pre-tokenizer. No GPU, no model weights: the scripts read the vocabulary out of a model file and tokenize with
it. The chat template and the tool-call format in `common.py` are Qwen3.8-Flash-Next's (the Qwen3.5/3.6 family format
with `<tool_call><function=…><parameter=…>` blocks); another model needs its own `render_prompt` and `build_output`.

## What is measured

- **Prompt and response, as the server saw them.** From each logged request `render_prompt` rebuilds the text the
  server fed the model (system prompt with the tool list, every earlier turn with its reasoning, tool calls and tool
  results). From the server's parsed reply `build_output` rebuilds what the model generated: reasoning, `</think>`, answer
  text, then each tool call in the model's own format, with every byte labelled by the region it belongs to (reasoning,
  answer, call scaffolding, and each tool argument: `bash.command`, `edit.edits.oldText`, `edit.edits.newText`, …).
- **Checks of the rebuild** (`prepare.py` prints them): the rebuilt prompt has the server's own token count in every
  request, and the rebuilt response is within ±2 tokens of the server's count (the total differs by 3 in 115,794).
  `check_render.py` renders each request with the model's Jinja template (needs the `jinja2` package) and compares the
  text byte for byte.
- **Copied tokens.** Four sources are compared (the `variant` column of every table):
  - `ctx`: the prompt only;
  - `own`: the prompt and the response's own earlier tokens, which is what a guesser sees while decoding;
  - `esc`: the prompt plus the JSON-escaped text of every tool result and user message (inside a JSON string a
    newline is the two characters `\n`, so file text the model writes into an `edits` argument does not match the raw
    text it was read in);
  - `own+esc`: both.

  A generated token is *copied at minimum length L* (4, 8 or 16) when it lies in a run of at least L consecutive
  generated tokens that occurs verbatim earlier in the source. A *copy span* is a maximal run of copied tokens.
  Spans may cross region borders: a `</think>` and a tool call's opening are often part of the span that began in the
  reasoning.
- **Where spans start.** `argument_values.csv` takes every tool-argument value (a `<parameter=…>` value, or a string
  inside the `edits` array) and says whether a copy runs into it, starts at its first token, starts at its second or
  third, starts later, or never starts. `span_starts.csv` has the same by the marker that precedes each span.
- **The guesser.** The simulation follows [ngram_replay](../ngram_replay/README.md): the history is the prompt plus the
  response so far; the longest suffix of it that occurred earlier (newest occurrence on ties, matches of up to 48 tokens
  inside the response, any length inside the prompt) supplies the draft, the `n` tokens that followed; the step keeps the
  longest prefix equal to what the model generated and adds one token. A draft needs a match of at least `m` tokens.
  Draft lengths: 2, 8, 16, 32, and `adapt` (twice the match length, at most 32). The recorded token is the target's
  sample, so acceptance is exact for these transcripts. The **oracle** drafts exactly the rest of the current copy span
  (at most `n` tokens) once `m` of its tokens were generated, always accepted and never in fresh text, with `m = 0` as the
  bound for one that knows where every span starts. The **occurrence oracle** takes, among the newest 400 earlier
  occurrences of the last `m` tokens, the one whose continuation is right for longest and drafts exactly that many: a
  bound on any copy guesser with an `m`-token key, whichever occurrence it prefers.
- **Cost model.** A verify pass of k columns costs `1 + r (k − 1)` one-token steps; `r = 0.40` is what the rig
  measured for Flash-Next before the multi-token kernels (136 ms plus 50–59 ms per extra token, Q583), `r = 0.10` is
  the estimate after them. A DFlash2 step adds a draft pass (0.15 and 0.22 of a one-token step). DFlash2's own figure
  is Q561's: two drafted tokens, 2.56 tokens per step, 3 columns. `dflash_fit.py` splits that figure into text with and
  without copies by a line through the per-request acceptance log.

## Run it

```bash
# 1. get the logs to the machine that runs the analysis (any machine with Python and llama-tokenize)
RIG="ssh <host>" WORK=/path/to/work ./copy_in.sh baseline=/path/to/run1 dflash2=/path/to/run2

# 2. everything else, on that machine
WORK=/path/to/work RUNS="baseline=/path/to/work/in/baseline dflash2=/path/to/work/in/dflash2" \
TOKENIZE=/path/to/llama-tokenize MODEL=/path/to/model-00001-of-0000N.gguf \
ACC=/path/to/draft_acceptance.csv FIT_RUN=dflash2 FIT_ARM=dflash2b ./run.sh
```

`run.sh` writes the tokenizer-only file (`make_vocab_gguf.py`, needed when the build does not know the model's
architecture), runs `selftest.py`, `prepare.py`, `analyze.py` and, with `ACC` set, `dflash_fit.py` followed by a second
`analyze.py` that uses the fitted figure. The whole thing takes about 25 seconds for 419 requests on six cores.
`samples.py` writes the longest and some random copy spans as text, for reading; that file is transcript text.

| File | What it does |
|---|---|
| [common.py](common.py) | log reader, prompt and response rebuild, vocabulary and tokenizer driver, suffix automaton, match profile |
| [make_vocab_gguf.py](make_vocab_gguf.py) | a tokenizer-only GGUF from a model file, so `llama-tokenize` loads an architecture it does not know |
| [prepare.py](prepare.py) | logs to token ids, region labels and markers; the rebuild checks; `texts.jsonl.gz` holds the text (private) |
| [analyze.py](analyze.py) | the tables below |
| [dflash_fit.py](dflash_fit.py) | tokens per step of a DFlash2 request against its copy share |
| [samples.py](samples.py), [check_render.py](check_render.py) | spans as text; the Jinja comparison |
| [selftest.py](selftest.py) | the match code against brute force (1,200 random cases), the span code against its definition, the guesser against a plain re-implementation (1,800 cases) |
| [run.sh](run.sh), [copy_in.sh](copy_in.sh) | the pipeline; the copy step |

Tables written to `$WORK/tables/`: `share_overall.csv` (by run and arm, with 95 % bootstrap intervals over sessions),
`share_per_response.csv`, `share_by_region.csv`, `share_without_scaffolding.csv`, `share_by_task.csv`,
`share_by_turn.csv`, `span_lengths.csv`, `span_starts.csv`, `argument_values.csv`, `continuation_by_match_length.csv`,
`guesser.csv`, `guesser_by_region.csv`, `combined.csv` (copy guesser with DFlash2 as the fallback), `per_request.csv`.

## Limits

- The tokens are the tokenizer's tokens for the text the server parsed back out of the stream. For a reply the
  server's parser has re-serialised (tool-call JSON), the model's own spacing is reconstructed in the template's style;
  the counts agree with the server's to ±2 per response.
- The logs hold one sampled transcript per request (temperature 1.0). Acceptance in the simulation is the acceptance on
  that sample; sessions are the unit of the bootstrap, and the tasks repeat across arms.
- The first generated token of a response comes from the prompt pass, so the simulation starts at the second.
