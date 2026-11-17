# Recall at depth on Flash-Next with Miner Llama v4.0 (R183, Q1546, and R190, Q1557; 2026-11-17)

What this checked: whether Flash-Next on nine P106-090 cards still gives a right answer deep into a long prompt.
Upstream llama.cpp issue [#32375](https://github.com/ggml-org/llama.cpp/issues/32375) (2026-11-16) reports a Tesla
P100 (compute 6.0) that answers a 60,240-token recall prompt on a hybrid model with a short echo and none of six
planted facts, with no error, while the same binary on the CPU lists all six; the same prompt family passes at 56,460
tokens. A P106 is compute 6.1 and runs its own kernels, and no rig run of this project had checked an answer past
49,152 tokens. These are Flash-Next's two runs for registry row Q1546 in
[quants/experiments.md](../../../quants/experiments.md): R183 first, then R190 (row Q1557), which ran it again on the
v4 preset's own file with a question the model can answer without listing the conversation. Ornith-1.5-122B's arm is
still to run.

The test is this project's own version of the issue's ([bench/recall_depth](../../../bench/recall_depth/README.md)):
small talk in which the user states six facts about themselves, planted 48,000, 32,000, 16,000, 4,000, 1,000 and 300
tokens from the end of the prompt, then a question; greedy decoding; a prompt passes when the answer holds all six
facts' marker words. One short control prompt (about 8,000 tokens) proves the harness.

## Result (R190): six of six at 56,443 and 60,207 tokens

**On the v4 preset's own file, Flash-Next answers right at depth: all six facts at 56,443 tokens and at 60,207
tokens,** the length at which upstream's P100 answered with none. Asked "Which pet, city, car, job, film and drink did
I mention in our conversation? Answer in six short lines, one for each, without comments.", the model's greedy
reasoning quoted the six sentences that hold the facts, one per kind, and the answer followed. The replies took 209,
239 and 252 tokens, each ending by itself (finish "stop"). The fact 48,004 tokens from the end was recalled as well as the one 292 tokens
from it.

So on nine P106-090 cards, with Miner Llama v4.0 and the v4 preset unchanged, this test does not show upstream
#32375's fault up to 60,207 tokens. R183's reply (below) never answered because its greedy reasoning listed the
messages one by one until it looped; R190 changed both the question and the file, so it does not say which of the two
made the difference. The prompts are this project's own text counted by Flash-Next's tokenizer, not the issue's
prompt.

| Prompt | Tokens (server) | Fact depths, tokens from the end | Found | Finish | Reply tokens | Prompt s | Prompt tok/s | Reply tok/s | Draft tokens accepted | The answer |
|---|---:|---|---:|---|---:|---:|---:|---:|---:|---|
| control, `kinds-8000` | 8,080 | 8,035, 7,998, 7,956, 4,015, 999, 292 | **6/6** | stop | 209 | 43.3 | 186.76 | 26.28 | 148 of 180 | "Pumpernickel / Dundee / Volvo / Lighthouse keeper / Metropolis / Hibiscus tea" |
| `kinds-56460` | 56,443 | 48,004, 31,985, 15,993, 4,015, 999, 292 | **6/6** | stop | 239 | 279.8 | 201.73 | 23.99 | 171 of 204 | "Cat: Pumpernickel / City: Dundee / Car: old green Volvo / Job: lighthouse keeper / Film: Metropolis / Drink: hibiscus tea" |
| `kinds-60240` | 60,207 | 48,004, 31,985, 15,993, 4,015, 999, 292 | **6/6** | stop | 252 | 321.8 | 187.09 | 22.25 | 175 of 228 | "Pumpernickel / Dundee / Volvo / Lighthouse keeper / Metropolis / Hibiscus tea" |

- One try each: every prompt passed the first time, so no prompt was sent again and the sampled fallback (a greedy
  reply that hits the cap with no answer is sent once more with the makers' sampling) was not needed.
- Each fact by value: [table_r190.md](table_r190.md); every request's numbers: [results_r190.tsv](results_r190.tsv)
  (the facts found in the answer and in the reasoning, the prompt's SHA-256, the question and the sampling).
- The answers are quoted whole, their six lines joined by " / ". Each reasoning (682, 745 and 855 characters) lists
  the six kinds in the question's order, the sentence that holds each and the name taken from it.
- At 60,207 tokens in a 65,536-token context a reply had room for 5,313 tokens (the tool never asks past the
  context); it used 252.
- The speeds are one request each, the whole prompt read from an empty cache (`cache_prompt: false`): diagnostics, not
  results. They are not comparable with R183's below, which ran another file.

## R183: the list question, no answer at 56,437 tokens

**The check could not answer the question on Flash-Next: at 56,437 tokens the reply never reached an answer.** It
spent all 8,192 reply tokens in its reasoning and ended at the cap (finish "length") with no answer text: 0 of 6
facts. It did so twice, byte for byte the same reasoning (29,470 characters, md5
`63f16f33c793652576a6527c9f296784` in both tries). The control prompt passed: 6 of 6 at 8,074 tokens.

**This miss is not upstream's failure.** Upstream's P100 ignored the context: it echoed the last small-talk line and
stopped after 17 tokens. Here the model read the start of the prompt correctly. Asked for every fact, it began its
reasoning by listing the conversation's user messages in order, one per line, and its first 11 lines are exactly the
prompt's first 11 user messages, the oldest text in the prompt, about 56,000 tokens back
([reasoning_items_56460.tsv](reasoning_items_56460.tsv)). From line 12 the list no longer follows the prompt: lines 12
to 22 are six small-talk lines in an order the prompt does not have, two of them already alternating from line 15.
From line 23 it alternates between those two lines ("I watched a series last night, an ordinary detective story." and
"I tried cooking a new dish and it came out quite well.") until line 385, where the cap ended the reply. The prompt holds 1,535 user messages; the first fact is user message 229, which the list never
reached. Greedy decoding has no randomness to leave such a loop.

So the run neither shows the upstream fault on a P106 nor rules it out at 56,437 tokens. The 60,240-token prompt, the
length that fails upstream, was not sent: the run stops after a repeated miss. The 85,000-token prompt does not fit
the v4 preset's 65,536-token context and was not built.

| Prompt | Try | Tokens (server) | Fact depths, tokens from the end | Found | Finish | Reply tokens | Prompt s | Prompt tok/s | Reply tok/s | The reply |
|---|---:|---:|---|---:|---|---:|---:|---:|---:|---|
| control, `recall-8000` | 1 | 8,074 | 8,029, 7,992, 7,950, 4,009, 993, 286 | **6/6** | stop | 1,621 | 24.5 | 329.71 | 30.24 | a list of 22 lines, the first "- Has a cat named Pumpernickel": all six facts and the small talk |
| `recall-56460` | 1 | 56,437 | 47,998, 32,014, 15,987, 4,009, 993, 286 | **0/6** | length | 8,192 | 275.6 | 204.81 | 27.51 | no answer: reasoning only, a two-line loop from its 23rd item |
| `recall-56460` | 2 | 56,437 | 47,998, 32,014, 15,987, 4,009, 993, 286 | **0/6** | length | 8,192 | 274.5 | 205.59 | 27.61 | the same reasoning, byte for byte |

- Each fact by value: [table.md](table.md) (the tool's own table, one column a fact), and every request's numbers in
  [results.tsv](results.tsv) (the facts found in the answer and in the reasoning, the prompt's SHA-256 as rendered by
  the server's chat template).
- Fact depths are measured on each final prompt by the server's tokenizer, from the fact's first token to the end.
  In the control the three deepest facts sit at its start, as a fact deeper than the prompt does.
- The speeds are one request each, full prompts read from an empty cache (`cache_prompt: false`): diagnostics, not
  results. The reply speed of the 56,437-token prompt is that of a loop the draft predicts well (draft acceptance
  6,128 of 6,186 and 6,129 of 6,184 tokens).
- The server's own count of each prompt equals the client's measurement (8,074 and 56,437 tokens).

### After R183

R190 (above) ran three of the checks this result called for: a question that does not invite a listing of every
message, the 60,240-token prompt, and the v4 preset's own file, UkisAI's Swift-1.5 GSQ-RCO Q2_0 (R183 ran the Pascal
mix of it). Still open:
- the list question with the reasoning off;
- the makers' sampling (temperature 1.0, top-p 0.95, top-k 20) at several seeds, as an agent runs the model;
- R183's 56,437-token prompt on another device (a CPU reference or a newer card), to see whether its loop is the
  model's under greedy decoding or the cards'.

## R190's run

| | |
|---|---|
| Rig | one rig, nine P106-090 cards; idle before the load (no llama process, 0 MiB on every card) and after the stop |
| Build | Miner Llama v4.0, the same tree and `llama-server` as R183's (md5 `e7afd26db01f4c671549ef12c504ada3`, checked on the rig before the load) |
| Model | the v4 preset's own file, UkisAI's Swift-1.5 GSQ-RCO Q2_0 (`PRESET_MODEL`), re-sharded with the n-gram table alone in the third shard: shard 1 md5 `ac0f5aaf04db9e16263f2a9760133ec8` (10,998,979,680 bytes), shard 2 `0df1578ca02412deb263f11c057e52c0` (26,750,834,816), shard 3 the n-gram table `98ce140e641bc5d68af9f31df0f1c048` (28,800,138,432); all checked on the rig before the load |
| Draft | the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf` (`PRESET_DRAFT`), sha256 `008183784ac5aa2cdcd3d7777e4e6febb0645ae4b8d1da1c065b2149c1d6391b`, checked |
| Settings | [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env) unchanged (md5 `dfc25d073a85961ee17746f7963f6564` on both ends), the same line as R183's |
| Requests | as R183's: greedy (`temperature` 0, `top_k` 1, `top_p` 1, `min_p` 0), `seed` 1546, `cache_prompt` false, `max_tokens` 8,192 or the room left in the context (5,313 for the 60,207-token prompt) |
| Question | `QUESTION=kinds`: "Which pet, city, car, job, film and drink did I mention in our conversation? Answer in six short lines, one for each, without comments." |
| On a miss | `ON_MISS=sample`: an answer that misses a fact stops the run; a greedy reply that hits the cap with no answer is sent once more with the preset's sampling (temperature 1.0, top-p 0.95, top-k 20, min-p 0) and seed 1546, and the run stops there. Neither happened |
| Prompts | `recall_depth.py`, seed 1546: the control (target 8,000), then targets 56,460 and 60,240. The small talk is drawn as in R183, but the shorter question changes the counts, so the prompts are not R183's (their SHA-256 are in [results_r190.tsv](results_r190.tsv)) |
| Times | server started 10:07:33, answering at 10:11:20 (the model loaded after 224 s); the control 10:11:31–10:12:23; the 56,443-token prompt 10:12:30–10:17:23; the 60,207-token prompt 10:17:32–10:23:08; the server stopped 10:23:27 |
| Host memory | the lowest MemAvailable over the load, 351 MiB, during the 60,207-token prompt (the watchdog's table; no kill) |
| Client | `recall_depth.py` on another machine on the same network, through an ssh tunnel to the server's loopback port |

## R183's run

| | |
|---|---|
| Rig | one rig, nine P106-090 cards; idle before the load (no llama process, 0 MiB on every card) and after the stop |
| Build | Miner Llama v4.0: the tag `miner-llama-v4.0` (`eb82916d3`); the rig tree is C828's (`eb312b7bc`, `pascal-p106-v4-b12360-c828`, the tag commit's second parent, the same tree); `llama-server` md5 `e7afd26db01f4c671549ef12c504ada3`, checked on the rig before the load |
| Model | Swift Qwen3.8-Flash-Next Pascal Q2 GGUF (`pmix-q2-fast`): my Pascal type mix of UkisAI's Swift-1.5 GSQ-RCO Q2_0, shard 1 md5 `dcbe9ae703ec77ddf0e1329852b7d7b9` (10,999,371,360 bytes), shard 2 `ab9aeccd468d88272defa43009f207bd` (26,755,194,496), shard 3 the n-gram table `98ce140e641bc5d68af9f31df0f1c048` (28,800,138,432); all checked on the rig before the load. The v4 preset's own file is UkisAI's unmixed Q2_0; the mix changes the dense and shared-expert tensor types only ([release/flash-next-v3](../../../release/flash-next-v3/README.md)) |
| Draft | the MTP draft `mtp-Qwen3.8-Flash-Next-draft.gguf`, sha256 `008183784ac5aa2cdcd3d7777e4e6febb0645ae4b8d1da1c065b2149c1d6391b`, checked |
| Settings | [release/flash-next-v4/settings.env](../../../release/flash-next-v4/settings.env) unchanged (md5 `dfc25d073a85961ee17746f7963f6564` on both ends): its `PRESET_ENV` and `PRESET_ARGS`, `-c 65536 -np 1`, `-lm dio`, the draft by `-md`, `--host 127.0.0.1 --port 8080`; the RAM watchdog on the server ([scripts/ram_watchdog.sh](../../../scripts/ram_watchdog.sh)) |
| Requests | `/v1/chat/completions`, not streamed: `temperature` 0, `top_k` 1, `top_p` 1, `min_p` 0, `seed` 1546, `cache_prompt` false, `max_tokens` 8,192 (the room left in the context was larger for both prompts). The model's chat template adds its default system block ("Reasoning effort is set to xhigh"); the server splits the reasoning from the answer (`--reasoning-format deepseek`) |
| Prompts | `recall_depth.py`, seed 1546: the control (target 8,000), then target 56,460; 60,240 was next and not sent |
| Times | server started 09:22:10, answering at 09:26:02 (the model loaded after 225 s); the control 09:27:57–09:29:15; the 56,437-token prompt 09:29:15–09:38:59 and 09:38:59–09:48:33; the server stopped 09:48:51 |
| Host memory | the lowest MemAvailable over the load, 365 MiB (the watchdog's table; no kill) |
| Client | `recall_depth.py` on another machine on the same network, through an ssh tunnel to the server's loopback port |

A first start of the client was stopped during the control's request and its files discarded: its prompts put every
fact about 43 tokens nearer the end than planned (255 tokens for 300, 957 for 1,000), because the model's template
puts its system block at the start of every prompt and the first version counted it at the end. The tool was fixed
(`costs()` measures the question from its text to the end) and the whole run started again; every number here is
from that second start.

## Files

| File | What it is |
|---|---|
| [table.md](table.md) | `recall_depth.py table`: one row a request, each fact found or not by value |
| [results.tsv](results.tsv) | the client's table: one row a request (prompt length as counted by the server and by the client, fact depths, facts found in the answer and in the reasoning, finish, reply tokens, prompt and reply speed, wall time, the prompt's SHA-256, the answer's first line) |
| [reasoning_items_56460.tsv](reasoning_items_56460.tsv) | the 56,437-token reply's reasoning as its 385 listed lines, each beside the user message at that place in the prompt, and whether they are the same |
| [run/r183.sh](run/r183.sh) | R183's steps: `load` (copy, start, wait), `run` (the client), `end` (stop, collect, clean up) |
| [table_r190.md](table_r190.md) | R190's table, as `table.md`: one row a request, each fact found or not by value |
| [results_r190.tsv](results_r190.tsv) | R190's client table, as `results.tsv`, with two more columns: the question (`kinds`) and the sampling of each try (`greedy`) |
| [run/r190.sh](run/r190.sh) | R190's steps, as `r183.sh`, with `QUESTION=kinds ON_MISS=sample NAME=kinds` |

The tool: [bench/recall_depth](../../../bench/recall_depth/README.md) (`recall_depth.py`, `serve.sh`, `remote.sh`).

## Commands

R190:

```bash
export RIG=rig BIN=<v4.0's bin folder on the rig> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf \
  DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=<results here>
bash docs/data/2026-11-17_r183_recall_depth/run/r190.sh load
ssh -N -L 18080:127.0.0.1:8080 "$RIG" &
bash docs/data/2026-11-17_r183_recall_depth/run/r190.sh run      # TARGETS="8000 56460 60240"; all three passed
bash docs/data/2026-11-17_r183_recall_depth/run/r190.sh end
OUT=$OUT/client python3 bench/recall_depth/recall_depth.py table > table_r190.md
```

`QUESTION=kinds URL=<server> python3 bench/recall_depth/recall_depth.py show 60240` prints the 60,207-token prompt and
its SHA-256 (`864c3b28db24815f…`).

R183 (`r183.sh` sets `QUESTION=list`):

```bash
export RIG=rig BIN=<v4.0's bin folder on the rig> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-pmix-fast-00001-of-00003.gguf \
  DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf OUT=<results here>
bash docs/data/2026-11-17_r183_recall_depth/run/r183.sh load
ssh -N -L 18080:127.0.0.1:8080 "$RIG" &
bash docs/data/2026-11-17_r183_recall_depth/run/r183.sh run      # TARGETS="8000 56460 60240"; stopped after 56460
bash docs/data/2026-11-17_r183_recall_depth/run/r183.sh end
OUT=$OUT/client python3 bench/recall_depth/recall_depth.py table > table.md
```

The same seed, question and model give the same prompts:
`QUESTION=list URL=<server> python3 bench/recall_depth/recall_depth.py show 56460` prints the 56,437-token prompt and
its SHA-256 (`4602df5b45c07c42…` here, in [results.tsv](results.tsv)).

## Provenance and licence

The prompts' text (small talk, facts, questions) is this project's own, written for the tool. The model's replies are
summarized (R190's short answers are quoted whole); the raw replies, the server's log and the watchdog's table are not
published. Upstream's issue was read
only; nothing of its code was run.
