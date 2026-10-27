# Q55: what tool requests lose without backend sampling (2026-10-24)

One coding-agent request with four tools (bash, read, write, edit) sent to llama-server three ways: `tool_choice` none
with backend sampling on, none with it off, and auto (tools allowed: llama.cpp then builds the tool-call grammar, which
turned backend sampling off in this build); at temperature 0 and at 1.0 with two seeds; nine cards, then six. Flint
UD-Q4_K_XL on the combined v2 build `b3052a4f8` with Q108 off (`GGML_CUDA_MMQ_MOE_LEGACY=1`), the server settings of a
real chat (`-c 262144 -b 512 -ub 128 -np 1 -bs --ctx-checkpoints 4`). Result: registry row Q55 in
[quants/experiments.md](../../../quants/experiments.md); the fix (Q67, tool requests sampled on the GPU) is in
[tool_request_sampling.md](../../research/notes/tool_request_sampling.md) and measured with
[bench/tool_sampling/tool_request_ab.sh](../../../bench/tool_sampling/tool_request_ab.sh).

| File | What |
|---|---|
| `results_9cards.jsonl`, `results_6cards.jsonl` | one line per request: temperature, seed, `tool_choice`, `backend_sampling`, tokens generated, tok/s and ms per token from the server's timings, finish reason |
| `run/q55.sh`, `run/client.py` | the driver and the client (the request is written in `client.py`) |

## What it shows

| Request (tok/s, nine cards / six cards) | T 0, 3,000 tokens | T 1 seed 11 | T 1 seed 12 |
|---|---|---|---|
| `tool_choice` none, backend sampling on | 32.87 / 33.45 | 36.46 / 37.75 (147 tokens) | 34.02 / 34.60 (2,908 tokens) |
| none, backend sampling off | 29.38 / 29.64 | 35.21 / 36.16 | 29.27 / 29.69 (2,721 tokens) |
| auto (tools allowed) | 29.38 / 29.65 | 34.38 / 35.06 (tool call) | 29.58 / 30.00 (tool call after 1,824 tokens) |

A tool request decodes at the host-sampling speed (−10.6 % at T 0 on nine cards): the server logged "backend sampling is
not compatible with grammar, disabling" once per auto request (three per card count). At T 1 the arms can sample
different texts (seed 12: 2,908 against 2,721 tokens), so only the T 0 rows are like for like.

## How it was made

On 2026-10-24 11:23–11:50, nothing else on the rig, the build unpacked in `/opt/llama-builds`
([scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh)) and the model at
`/mnt/models/unsloth/flint-GGUF/flint-UD-Q4_K_XL.gguf`:

```bash
RIG=rig OUT=. docs/data/2026-10-24_rig_q55/run/q55.sh
```

The client sends the nine requests with `cache_prompt: false`, `top_p 0.95, top_k 20, min_p 0` and up to 3,000 tokens.
The server logs are not included. Flint UD-Q4_K_XL is Unsloth's quant of Flint (Apache-2.0); the request was written
for this repository.
