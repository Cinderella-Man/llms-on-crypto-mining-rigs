# Reasoning length and time to a correct answer

Does a quantized Flint file reason longer than another, and which one gives a correct answer soonest on the rig?
Tokens per second alone can't say: Flint always reasons, about two thirds of its output is reasoning, and a file that
decodes 3 % faster but reasons 3 % longer gains nothing ([research note](../../docs/research/notes/answer_time.md)).

## Pieces

- **[flint_tokens.py](flint_tokens.py)** counts tokens with Flint's own tokenizer (`tokenizer.json` from the published
  checkpoint) in standard-library Python. It implements byte-level BPE, the pre-tokenizer's Unicode classes built from
  `unicodedata`, and the added tokens such as `<think>` and `<tool_call>` counted as one token each. On Q41's outputs
  it reproduces llama-server's `tokens_predicted` exactly, up to the stop token, on 98.9 % of samples.
- **[lengths.py](lengths.py)** reads one or more Q41 data folders: `raw/<lane>-<file>[-a<attempt>-p<i>of<n>].jsonl`
  from [bfcl_flint](../bfcl_flint/) and [livecodebench_flint](../livecodebench_flint/), plus their scored
  `out/*/per_attempt.csv`. For each sample it counts reasoning tokens (`<think>` to the last `</think>`) and answer
  tokens. For multi-turn conversations it does the same per step, per user turn and per conversation, and counts the
  new prompt tokens of every step. Against a reference file, per lane and paired by task, it reports:
  - ratio of means and median per-task ratio of total, reasoning and answer tokens, with cluster-bootstrap
    intervals (medians over equal numbers of attempts);
  - quantile ratios, runaway rates, and where the difference sits (tasks both files solve, both fail, or split on);
  - hesitation-word density ("wait", "but", "maybe", ...), the tokens quantization noise raises in low-bit reasoning
    models;
  - a noise floor: one file's attempt b against its attempt a;
  - with `--time`, seconds per attempt and per correct answer from a cost model of the rig (decode time per token
    from bytes per token, served overhead, host sampling for tool requests, context depth, prompt processing per new
    token and per request), and their ratio to the reference.
  Data folders are pooled per lane unless their (file, task, attempt) keys collide. Samples without a verdict are
  left out unless `--keep-unscored` is given.

## Running it

```bash
python lengths.py --tokenizer tokenizer.json --alias ud-q4kxl=ud --time q41-main/ q41-pilot/ > lengths.md
# other cost assumptions, e.g. prompt processing after the prefill fixes, or measured rig speeds per file
python lengths.py --tokenizer tokenizer.json --time --prefill-ms 2.0 --request-s 1.0 \
  --token-ms k2h=26.2 --token-ms ud=26.84 q41-main/
```

Defaults of the cost model (all estimates for nine P106-090 cards on the fork's v2 build; see the note, section 3):

| Option | Default | Meaning |
|---|---|---|
| `--token-ms FILE=MS` | ud 26.84, g3 27.40, g3h 26.63, k2h 26.16, k1h 25.86 | decode ms per token at depth 0, six cards, llama-bench: UD measured, the others 12.0 ms + GB per token / 151 GB/s |
| `--serve-ms` | 4.5 | served nine-card overhead per token over llama-bench |
| `--host-ms` | 3.3 | logits copy and host sampling per token for requests with tools (single- and multi-turn lanes) |
| `--depth-ms` | 0.3 | extra ms per token per 1,000 tokens of context |
| `--prefill-ms`, `--request-s` | 7.1, 2.2 | ms per new prompt token and seconds per request in llama-server today |

## Results

Q41's runs at 08:07 on 2026-10-24, against UD-Q4_K_XL (95 % intervals). The full tables are in the note.

| Lane | Pair | Total tokens, ratio of means | Median per-task ratio | Noise floor (UD against itself) |
|---|---|---|---|---|
| BFCL v4 single-turn, 3,477 tasks | G3 | 1.000 (0.977–1.023) | 0.995 (0.987–1.003) | 1.027 (0.991–1.064) |
| BFCL v4 single-turn, 3,477 tasks | K2h | 0.990 (0.955–1.032) | 0.991 (0.981–1.000) | |
| BFCL v4 multi-turn, 199 conversations | G3 | 0.957 (0.878–1.053) | 0.941 (0.854–1.006) | 0.985 (0.902–1.082) |
| LiveCodeBench, 56 problems, 32k budget | G3 | 1.033 (0.948–1.124) | 1.003 (0.978–1.111) | — |

No candidate reasons longer than UD-Q4_K_XL beyond attempt-to-attempt noise.
