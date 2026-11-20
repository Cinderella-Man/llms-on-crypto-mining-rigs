# Recall at depth on Ornith-1.5-122B with the v5 engine (R183's Ornith arm, Q1546; 2026-11-20)

What this checked: whether Ornith-1.5-122B on nine P106-090 cards still gives a right answer deep into a long prompt,
up to the end of its preset's 49,152-token context. Upstream llama.cpp issue
[#32375](https://github.com/ggml-org/llama.cpp/issues/32375) (2026-11-16) reports a Tesla P100 (compute 6.0) that
answers a 60,240-token recall prompt on a hybrid model with a short echo and none of six planted facts, with no error,
while the same binary on the CPU lists all six. Ornith is a hybrid model too, and a P106 is compute 6.1 and runs its
own kernels. This is Ornith's arm of registry row Q1546 in [quants/experiments.md](../../../quants/experiments.md);
Flash-Next's arm (R183 and R190, 2026-11-17) is in
[2026-11-17_r183_recall_depth](../2026-11-17_r183_recall_depth/README.md), and this run uses its method and its
question that worked (R190's).

The test is this project's own version of the issue's ([bench/recall_depth](../../../bench/recall_depth/README.md)):
small talk in which the user states six facts about themselves, planted 48,000, 32,000, 16,000, 4,000, 1,000 and 300
tokens from the end of the prompt, then a question; greedy decoding; a prompt passes when the answer holds all six
facts' marker words. A fact deeper than the prompt goes to its start, so in each prompt below the deepest facts sit in
its first lines: the 47,012-token prompt opens with the cat, 47,009 tokens from the end.

## Result: six of six at every length, up to 47,012 tokens

**Ornith answers right at depth on nine P106 cards: all six facts at 8,077, 16,075, 32,025 and 47,012 tokens,** each
reply ending by itself (finish "stop") in 127 to 269 tokens. Asked "Which pet, city, car, job, film and drink did I
mention in our conversation? Answer in six short lines, one for each, without comments.", the model's greedy
reasoning named each kind with the sentence that holds it (in the 32,025 and 47,012-token prompts it quoted the six
sentences word for word), and the answer followed. Nothing in the run looks like upstream's fault: no echo, no
missing fact, no error in the server's log.

| Prompt | Tokens (server) | Fact depths, tokens from the end | Found | Finish | Reply tokens | Reasoning, characters | Prompt s | Prompt tok/s | Reply tok/s | The answer |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| control, `kinds-8000` | 8,077 | 8,074, 8,037, 7,995, 4,015, 999, 292 | **6/6** | stop | 193 | 565 | 38.7 | 208.66 | 15.31 | "Cat: Pumpernickel / City: Dundee / Car: old green Volvo / Job: lighthouse keeper / Film: Metropolis / Drink: hibiscus tea" |
| `kinds-16000` | 16,075 | 16,072, 16,035, 15,993, 4,015, 999, 292 | **6/6** | stop | 127 | 347 | 72.8 | 220.78 | 13.35 | "Cat: Pumpernickel / City: Dundee / Car: green Volvo / Job: lighthouse keeper / Film: Metropolis / Drink: hibiscus tea" |
| `kinds-32000` | 32,025 | 32,022, 31,985, 15,993, 4,015, 999, 292 | **6/6** | stop | 269 | 884 | 176.6 | 181.38 | 10.53 | "Cat: Pumpernickel / City: Dundee / Car: old green Volvo / Job: lighthouse keeper / Film: Metropolis / Drink: hibiscus tea" |
| `kinds-47000` | 47,012 | 47,009, 31,985, 15,993, 4,015, 999, 292 | **6/6** | stop | 196 | 599 | 278.2 | 168.98 | 8.72 | "Cat: Pumpernickel / City: Dundee / Car: old green Volvo / Job: lighthouse keeper / Film: Metropolis / Drink: hibiscus tea" |

- One try each: every prompt passed the first time, so R183's list question, planned for a prompt that missed a fact,
  was not sent.
- Each fact by value: [table.md](table.md); every request's numbers: [results.tsv](results.tsv) (the facts found in
  the answer and in the reasoning, the prompt's SHA-256, the question, the sampling). The facts are also all six in
  every reasoning.
- The answers are quoted whole, their six lines joined by " / ".
- The 47,012-token prompt left room for a 2,124-token reply in the 49,152-token context (the tool never asks past the
  context); it used 196.
- **Both arms together:** Flash-Next (UkisAI's Swift-1.5 GSQ-RCO Q2_0, Miner Llama v4.0) 6 of 6 at 8,080, 56,443 and
  60,207 tokens; Ornith-1.5-122B Pascal N2P GGUF (v5 engine) 6 of 6 at 8,077, 16,075, 32,025 and 47,012 tokens. On
  P106 cards this test does not show upstream #32375's fault within either preset's context.

### Diagnostics from the same run: long prompts and checkpoints

The speeds are one request each, the whole prompt read from an empty cache (`cache_prompt: false`), on cards at stock
after the evening's reset of their clock offsets: diagnostics, not results. Two things show in them.

- **Generation slows with depth:** 65.32, 74.90, 94.97 and 114.66 ms a token at 8,077, 16,075, 32,025 and 47,012
  tokens of context.
- **Some context checkpoints stop a long prompt for 7 to 12 s.** The server reads a prompt in calls of about 4,096
  tokens and takes a checkpoint of the model's recurrent state on the way. With the preset's
  `--ctx-checkpoints-incall` most checkpoints are taken inside a call. Every prompt past 8,077 tokens also took one to
  three checkpoints the other way: the server cut a call of 31 to 43 tokens, and that call took 7.44 s (16,075-token
  prompt), 7.06 and 7.99 s (32,025), 8.20, 11.85 and 12.37 s (47,012), where a full call of 4,096 tokens took 13.56 to
  27.20 s. The server's own counter of checkpoint save time matches (`ckpt_save_ms`: 7.34, 14.73 and 31.67 s), and the
  number of short calls equals the checkpoints not taken inside a call (5 − 4, 7 − 5 and 9 − 6). So about a tenth of
  each long prompt's time (32.42 of 278.21 s at 47,012 tokens) went to those checkpoints. The agent check's prompts are
  much shorter, so this is a row for long prompts: Q1644 in the registry. The calls one by one:
  [prompt_calls.tsv](prompt_calls.tsv).

## The run

| | |
|---|---|
| Rig | one rig, nine P106-090 cards, every clock offset 0 (reset two hours earlier on the same boot, so the cards may run slightly below their stock speed until a reboot; the run checks answers, not speed); idle before the load (no llama process, 0 MiB on every card) and after the stop |
| Build | the v5 engine: `pascal-p106-v5` at `602691153` (the candidate `pascal-p106-v5-cand-b12483`, upstream b12483 merged); `llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`, checked on the rig before the load |
| Model | Ornith-1.5-122B Pascal N2P GGUF (`valid1`, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`, 48,196,722,592 bytes), the preset's `PRESET_MODEL`; no draft |
| Settings | [release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env) unchanged (md5 `4b8dac36b6b9866090de3887c39b02ff` on both ends): its `PRESET_ENV` and `PRESET_ARGS` (`-c 49152 -np 1`, `--reasoning-format deepseek`, `--ctx-checkpoints 4 --ctx-checkpoints-vram 4 --ctx-checkpoints-incall`), `--host 127.0.0.1 --port 8080`; the RAM watchdog on the server ([scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh)) |
| Requests | as R190's: `/v1/chat/completions`, not streamed, greedy (`temperature` 0, `top_k` 1, `top_p` 1, `min_p` 0), `seed` 1546, `cache_prompt` false, `max_tokens` 8,192 or the room left in the context (2,124 for the 47,012-token prompt) |
| Question | `QUESTION=kinds` (R190's): "Which pet, city, car, job, film and drink did I mention in our conversation? Answer in six short lines, one for each, without comments." |
| On a miss | `ON_MISS=go`: every prompt is sent whatever the one before gave; then each prompt that missed a fact is sent once more with R183's question (`QUESTION=list`). No prompt missed |
| Prompts | `recall_depth.py`, seed 1546, targets 8,000 (the control), 16,000, 32,000 and 47,000, counted by Ornith's own chat template and tokenizer: the question costs 39 tokens, the template's prefix 3, a small-talk exchange 31 to 43, a fact's exchange 32 to 42 |
| Times | server started 19:48:47, listening at 19:57:48 (the model loaded in 541 s); the control 19:58:00–19:58:51; 16,075 tokens 19:58:53–20:00:15; 32,025 tokens 20:00:18–20:03:40; 47,012 tokens 20:03:46–20:08:46; the server stopped 20:08:46, the rig idle at 20:08:55 |
| Host memory | the lowest MemAvailable over the load 1,261 MiB, as the 47,012-token prompt began; swap used at most 322 MiB, during the load (the watchdog's table; no kill) |
| Client | `recall_depth.py` on another machine on the same network, through an ssh tunnel to the server's loopback port |

## Files

| File | What it is |
|---|---|
| [table.md](table.md) | `recall_depth.py table`: one row a request, each fact found or not by value |
| [results.tsv](results.tsv) | the client's table: one row a request (prompt length as counted by the server and by the client, fact depths, facts found in the answer and in the reasoning, finish, reply tokens, prompt and reply speed, wall time, the prompt's SHA-256, the answer's first line, the question, the sampling) |
| [prompt_calls.tsv](prompt_calls.tsv) | each prompt's decode calls from the server's log (tokens before and after, seconds, speed), then each prompt's checkpoint counters from the server's reply (`run/calls.py`) |
| [prompts/](prompts/) | the four prompts as sent: the messages, gzipped JSON (`kinds-<target>.json.gz`) |
| [run/r183o.sh](run/r183o.sh) | the run's steps: `load` (copy, start, wait), `run` (the client, then R183's question on any miss), `end` (stop, collect, clean up), `all` (the four in order, as it ran) |
| [run/calls.py](run/calls.py) | builds `prompt_calls.tsv` from the server's log and the client's reply files |

The tool: [bench/recall_depth](../../../bench/recall_depth/README.md) (`recall_depth.py`, `serve.sh`, `remote.sh`).

## Commands

```bash
export RIG=rig BIN=<the v5 engine's bin folder on the rig> \
  MODEL=<folder>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf OUT=<results here> STOP_AT=21:50
bash docs/data/2026-11-20_r183_ornith_recall/run/r183o.sh all     # TARGETS="8000 16000 32000 47000"; all four passed
OUT=$OUT/client python3 bench/recall_depth/recall_depth.py table > table.md
python3 docs/data/2026-11-20_r183_ornith_recall/run/calls.py $OUT/rig/server.log $OUT/client > prompt_calls.tsv
```

`all` loads the server, opens the tunnel (`ssh -N -L $LPORT:127.0.0.1:8080 "$RIG"`, `LPORT` 18080 by default), runs
the prompts and always stops and collects the server; `STOP_AT` stops the client at that time. The same seed,
question and model give the same prompts: `URL=<server> CTX=49152 python3 bench/recall_depth/recall_depth.py show
47000` on Ornith's server prints the 47,012-token prompt and its SHA-256 (`f89148243b6f9e0b…`, in
[results.tsv](results.tsv)).

## Provenance and licence

The prompts' text (small talk, facts, questions) is this project's own, written for the tool, and is published here.
The model's replies are summarized; the answers are quoted whole. The raw replies with their reasoning, the server's
log and the watchdog's table are not published. Upstream's issue was read only; nothing of its code was run.
