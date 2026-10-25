# Agent prompt cache: replay, exactness checks and rig scripts (Q111, Q132)

An agent re-sends its whole conversation with every request: Flint's BFCL multi-turn sessions send 39 prompt tokens per
generated token. The fork's llama-server (branch `pascal-p106-v2-agentcache`) keeps the server from reading them again:
saved states of the shared system and tool prompt that any slot restores (`--prefix-cache DIR`), one end-of-prompt
checkpoint with the whole sliding-window ring instead of two, checkpoints at `</think>` and `<tool_call>` while
generating (`--checkpoint-gen`), no scheduler reserve per request with backend sampling, and restores that put every
cell where a fresh context has it, so cached and cold answers agree bit for bit. Design and results:
[agent_prompt_cache.md](../../docs/research/notes/agent_prompt_cache.md).

| File | What it does |
|---|---|
| [replay_sessions.py](replay_sessions.py) | Drives recorded multi-turn sessions against a server the way an agent does: each request carries the whole conversation and the server's own last reply goes back into it (reasoning kept, or dropped with `--history drop`); the recorded tool results answer the live calls. Streams, fixed seeds per request, one row per request: time to first token, the server's prompt and cache counters, the reuse class (exact, partial, reply read again, older) and a reply hash. `pick` chooses sessions that share a tool list, `summary` tabulates labels against a reference. Standard library only |
| [export_bfcl.py](export_bfcl.py) | Sessions for the replay from a BFCL v4 multi-turn run of Flint (`bench/bfcl_flint/multi_turn.py run`): every request as a chat-completions client sends it (tools, messages added since the previous reply, the recorded reply, prompt and reply lengths). Needs bfcl_eval |
| [data/bfcl-8.jsonl](data/bfcl-8.jsonl) | Eight sessions from Flint's BFCL E2 run (UD-Q4_K_XL): two tool sets with four sessions each, alternating (each session follows one with the other system and tool prompt, as when sub-agents or several projects share a server), 4-11 requests, first prompts of 3.5k and 4.8k tokens |
| [check_exact.py](check_exact.py) | Every request to the server under test against a cold server (cache_prompt off): the same top log-probabilities, bit for bit. Scenarios: prefix (save, another conversation in between, restore; after a restart from the file), turns (replies sent back), trim (raw token prompts that diverge in the live window, before the end-of-prompt checkpoint, nowhere usable), gen (a divergence after a generation checkpoint), bs (the same replies with and without a reserve per request) |
| [make_tiny_flint.py](make_tiny_flint.py) | A 6-layer Flint with random weights and Flint's tokenizer, template, window and layer pattern (167 MB), for the CPU checks |
| [cpu_check.sh](cpu_check.sh) | The CPU exactness run: a cold server and the server under test with the agent cache, again after a restart, with stock's two tail checkpoints and per-request reserves, and with `-bs` in both reserve modes. Docker, CPU only |
| [gpu_check.sh](gpu_check.sh), [gpu_inner.sh](gpu_inner.sh) | The same with the real Flint on one GPU (a test slot and a cold slot of one server), then a short replay (stock behaviour, the exactness control, the agent cache) for the token counts and the replies. One container under an optional `flock` |
| [rig_replay.sh](rig_replay.sh) | The rig run: the eight sessions per configuration (stock behaviour; stock behaviour with the single tail checkpoint, the exactness control whose replies the agent cache must give; the agent cache; stock and the agent cache with the reasoning dropped from the history; optionally the unmodified base build), K2h-r16 on nine cards, a memory watchdog; writes the replay rows and summaries |

## Exactness, and why the checks run without flash attention on the CPU

Attention adds up the keys in cell order, so two servers that hold the same values in different cells differ in the
last bits. The fork therefore restarts an emptied cache stream at cell 0 and restores a saved state into the cells a
sequence processed from its first token occupies (position p in cell p, p mod size in the sliding-window ring). With
that, a CPU build without llamafile GEMM and with `-fa off` sums every query the same way whatever the batch, and every
cache path must equal a cold server exactly (`CHECK_ARGS="-fa off" bench/agent_cache/cpu_check.sh`). The CPU's flash
attention splits the keys between threads by batch size, and GPU kernels differ between one-token and multi-token
batches, so there only the paths whose reused cells come from the same prompt evaluations as a cold server's (a saved
prefix) are bit-identical; the others report the first token's largest log-probability difference, as stock llama.cpp's
own prompt cache would.

## Running

```
# sessions (quant machine, BFCL venv): 200 sessions, 2,364 requests
python export_bfcl.py RAW.jsonl sessions.jsonl --template TEMPLATE.jinja
python3 replay_sessions.py pick sessions.jsonl --groups 2 --per-group 4 --write data/bfcl-8.jsonl

# CPU exactness (WORK holds out-cpu/build/bin/llama-server, models/tiny-flint.gguf, sessions-e2-ud.jsonl, bench/)
CHECK_ARGS="-fa off" WORK=... bench/agent_cache/cpu_check.sh

# on the rig, from a folder holding replay_sessions.py, rig_replay.sh and bfcl-8.jsonl
BIN=/opt/llama-builds/<agentcache build>/bin MODEL=<flint.gguf> SESSIONS=bfcl-8.jsonl OUT=/tmp/q111/out \
    ARMS='stock agent stock-drop agent-drop' bash rig_replay.sh
```
