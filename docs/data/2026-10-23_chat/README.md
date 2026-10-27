# Flint in a real chat: per-turn tables and transcripts (2026-10-23)

The same five-message conversation, run with three Flint files on stock llama.cpp and on Miner Llama, all nine cards,
262,144 tokens of context. The result page is [2026-10-23-flint-chat.md](../../benchmarks/2026-10-23-flint-chat.md);
it has the totals and per-turn tables of the six runs it compares. This folder holds each run's own per-turn table and
the full transcript, plus a seventh run with a lower cap on reply length.

| Folder | Engine | File | Tokens per reply at most |
|---|---|---|---|
| `chat-ud` | Miner Llama (`pascal-p106` at `eeaa59660`) | Unsloth UD-Q4_K_XL | 8,192 |
| `chat-ud-max3072` | Miner Llama | Unsloth UD-Q4_K_XL | 3,072 |
| `chat-s4l0` | Miner Llama | Miner S4-L0 (Q4) | 65,536 |
| `chat-q80` | Miner Llama | Unsloth Q8_0 | 65,536 |
| `chat-stock-ud` | stock llama.cpp `d77256999` | Unsloth UD-Q4_K_XL | 65,536 |
| `chat-stock-s4l0` | stock llama.cpp | Miner S4-L0 (Q4) | 65,536 |
| `chat-stock-q80` | stock llama.cpp | Unsloth Q8_0 | 65,536 |

| File | What |
|---|---|
| `<run>/summary.txt` | [scripts/server_bench.sh](../../../scripts/server_bench.sh)'s summary: load time, memory per card, sampling, then per turn the history, new prompt tokens and their speed, time to the first token, tokens generated (answer tokens in brackets), generation speed, why the reply stopped and the turn's wall time; the last line totals the conversation |
| `<run>/chat.json` | the transcript: sampling, the messages as sent, and per turn the server's full response (answer, `reasoning_content`, token counts and timings) |
| `run/run.sh`, `run/run2.sh` | the commands, run from the repository root against the rig |

**A cap of 3,072 tokens per reply is too low for Flint's reasoning.** In `chat-ud-max3072` the first three replies hit
the cap while Flint was still reasoning (finish reason `length`) and contained no answer at all; the fourth finished at
2,667 tokens and the fifth at 958:

| Turn | History | Generated (answer) | Stopped by |
|---|---|---|---|
| 1 | 69 | 3,072 (0) | length |
| 2 | 107 | 3,072 (0) | length |
| 3 | 135 | 3,072 (0) | length |
| 4 | 163 | 2,667 (447) | end of reply |
| 5 | 627 | 958 (133) | end of reply |

Because an empty answer adds nothing to the history, turns 2–4 started from 107–163 tokens instead of 1,000–2,600.
With 8,192 tokens (`chat-ud`) every reply ended on its own; the longest reply in any run took 7,909 tokens.

Made on 2026-10-23 on the rig:

```bash
RIG=rig bash docs/data/2026-10-23_chat/run/run.sh       # chat-ud
RIG=rig MAX_TOKENS=3072 TAG=chat-ud-max3072 bash docs/data/2026-10-23_chat/run/run.sh   # chat-ud-max3072 (UD part)
RIG=rig bash docs/data/2026-10-23_chat/run/run2.sh      # the other five
```

Each run writes `/tmp/server_bench/<TAG>/` on the rig. Both scripts take the build and model paths from `FORK`,
`STOCK`, `UD`, `L0` and `Q8`; the defaults are the paths used here. Sampling: Flint's GGUF defaults (temperature 1.0,
top-p 0.95, top-k 20), seeds 1001–1005. The server logs are not published.

The prompts are this repository's own (in `scripts/server_bench.sh`, MIT); the replies are Flint's outputs (Flint is
Apache-2.0).
