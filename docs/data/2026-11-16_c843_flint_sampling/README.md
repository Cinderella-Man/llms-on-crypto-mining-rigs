# Whether v4's Flint preset changes what Flint samples (C843, Q1488; 2026-11-16)

What this checked: whether Miner Llama v4's Flint preset ([release/flint-v4](../../../release/flint-v4/README.md))
draws other tokens than v3.0's when both sample the same agent request with the same seed. In Flint's release agent
check ([2026-11-14_r159_flint_release_agent](../2026-11-14_r159_flint_release_agent/README.md)) v4's two runs
generated at v3.0's speed but wrote and read about twice as much (decode 120 and 112 s a task against 49 and 73), and
two runs each at temperature 1.0 cannot tell chance from a switch that changes the sampled tokens. Correctness work: no
speed is read here. Registry row Q1488 in [quants/experiments.md](../../../quants/experiments.md).

## Result

**No. v4's Flint preset draws exactly the tokens v3.0 draws, and none of its three switches changes what Flint
samples.** On the same 26 agent requests at fixed seeds, v4 with its preset (`v4`) and v4 without
`LLAMA_SAMPLING_TOP_K_ROWS=1` (`v4n`) wrote the same token ids as v3.0 in every reply: 16,290 generated tokens in each
arm, the same at every position, the same 26 lengths and the same 26 ends (19 tool calls and 7 replies stopped at the
2,048-token limit). Every arm rendered and read the same prompts (the same md5 and token count for every request), the
same lazy tool-call grammar, and its replies report the same sampler chain with sampling on the cards. The server's
own counts per request agree too: in all three arms, 487 tokens through the sampling step, all drawn on the cards, 461
of them under the grammar, none rejected or forced, and 41,318 pipelined guesses, 5,285 right and 9,011 wrong. Each
load's two repeats wrote their first reply again, token for token.

So v4's longer agent runs in R159 were chance, not a switch: the agent sends no seed, so every run draws its own
random stream, and at temperature 1.0 the same request can end in a short tool call or in 2,048 tokens of reasoning
and code (B1 here: 2,048, 1,461 and 2,048 tokens at seeds 11, 12 and 13; B3: 24, 24 and 2,048). With the same seed,
v4 and v3.0 write the same reply. This agrees with R161's deep lane, where the three switches gave the same sampled
reply as v3.0 at every depth ([2026-11-14_r161_flint_release_deep](../2026-11-14_r161_flint_release_deep/README.md)),
and it extends that check to the agent's own requests: the chat template, the tool-call grammar, natural ends and
replies up to 2,048 tokens.

| | `v3` | `v4` | `v4n` |
|---|---|---|---|
| requests (24 + 2 repeats) | 26 | 26 | 26 |
| the same token ids as `v3` | (control) | 26 of 26 | 26 of 26 |
| generated tokens; the same as `v3` | 16,290 | 16,290; 16,290 | 16,290; 16,290 |
| ends: tool call, length limit | 19, 7 | 19, 7 | 19, 7 |
| the same prompt (md5 and tokens) as `v3` | | 26 of 26 | 26 of 26 |
| repeats the same as the load's first reply | 2 of 2 | 2 of 2 | 2 of 2 |
| switches set (the server's log and environment) | none | 3 | 2 |
| load; lowest MemAvailable | 575 s; 1,042 MiB | 575 s; 1,040 MiB | 576 s; 1,031 MiB |

Per request: [table.md](table.md). Run on one rig on 2026-11-16, 14:50:22-15:50:26, the arms in the order `v3`, `v4`,
`v4n`; every load passed its quiet check, and the RAM watchdog did not act.

## The arms

One load each, on one rig (nine P106-090 cards), Flint Pascal Q8 GGUF (`p065`, 40.4 GB) at 262,144 tokens, in this
order:

| Arm | Build | Settings |
|---|---|---|
| `v3` (the control) | the v3.0 release, `pascal-p106-v3-b12076-ladder5` `1551f5fe1` (tag `miner-llama-v3.0`) | [release/flint-v3/settings.env](../../../release/flint-v3/settings.env) as shipped: no switch |
| `v4` | `v4-rc2`, `pascal-p106-v4` `eb82916d3` (the rig build's tree `pascal-p106-v4-b12360-c828`) | [release/flint-v4/settings.env](../../../release/flint-v4/settings.env): v3.0's line with `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1` |
| `v4n` | `v4-rc2` | the v4 preset without `LLAMA_SAMPLING_TOP_K_ROWS=1` |

Every line is the release file's `PRESET_ARGS` (the same in both files: `-bs`, `--pipe-depth 4`, `--temp 1.0
--top-p 0.95 --top-k 20 --min-p 0`) plus what the release agent check adds: `--jinja --reasoning-format deepseek
--metrics --no-webui`.

## The requests

Eight requests recorded in R159's agent runs, as the agent sent them (the system prompt, the task, the tool calls and
results so far, the four tools), picked from the three tasks where v4's runs wrote the most, plus one more:

| Request | Task | Turn | Messages | From | Recorded reply |
|---|---|---|---|---|---|
| A1 | JavaScript `variable-length-quantity` | 1 | 2 | v3.0, run 2 | 25 tokens, a tool call |
| A2 | Python `pig-latin` | 1 | 2 | v3.0, run 2 | 21 tokens, a tool call |
| A3 | Python `list-ops` | 1 | 2 | v3.0, run 2 | 34 tokens, a tool call |
| A4 | Python `scale-generator` | 1 | 2 | v3.0, run 2 | 36 tokens, a tool call |
| B1 | JavaScript `variable-length-quantity` | 2 | 5 | v4, run 1 | 10,793 tokens, a tool call |
| B2 | Python `pig-latin` | 3 | 6 | v3.0, run 2 | 2,404 tokens, a tool call |
| B3 | Python `list-ops` | 2 | 4 | v3.0, run 2 | 2,945 tokens, a tool call |
| B4 | Python `scale-generator` | 2 | 5 | v4, run 1 | 2,779 tokens, a tool call |

The A requests are first turns, which end in a short tool call under the tool-call grammar; the B requests are the
turns where the agent writes its fix, with long reasoning first.

## Method

- **Per load:** a quiet check (no llama process, at most 100 MiB on any card), the server under the RAM watchdog
  (`scripts/ram_watchdog.sh`, default limits), its `LLAMA_`/`GGML_` variables read from the running process, then
  `run/c843_client.py` on the loopback: every request with seeds 11, 12 and 13 (24 requests), then A1 and B1 at seed 11
  again (the load's repeats), then the server stopped.
- **Each request** is the recorded body with five changes: `stream` off, `max_tokens` 2,048, the `seed`, `cache_prompt`
  false (every request reads its whole prompt from an empty cache, so it starts from the same state in every arm and in a
  repeat), `return_tokens` and `verbose` on (the generated token ids and the end reason come back in the reply; neither
  changes what is sampled; `logprobs` is not asked for, since it turns the card's sampling off). The sampling fields are
  the recorded ones, Flint's: temperature 1, top-p 0.95, top-k 20, min-p 0.
- **Compared** per request and seed against `v3`: the generated token ids (the same, or the first token where they
  differ), the reply's length, its end (`finish_reason` and the server's stop type), and the prompt as the server
  rendered and counted it (md5 of the rendered text, tokens read). Each reply also reports the settings it ran with (the
  seed echoed, the sampler chain, card sampling on, the chat format, the lazy grammar), and each server log's per-request
  line counts the tokens drawn on the card, under the grammar, rejected and forced, and the pipelined guesses.

## Commands

From the repository's root on a machine that runs only ssh and tar; the machine values in the environment:

```bash
python3 -I docs/data/2026-11-16_c843_flint_sampling/run/c843_pick.py --out requests.jsonl \
  --bodies <R159 run>/proxy.jsonl.bodies.gz ... --pick A1=<request id> ... --pick B4=<request id>
RIG=<rig alias> WORK=<run folder on the rig> BIN_V3=<v3.0 tree>/bin/llama-server BIN_V4=<v4-rc2 tree>/bin/llama-server \
  MODEL=<models>/flint-pascal-Q8.gguf REQUESTS=requests.jsonl N_PREDICT=2048 SEEDS="11 12 13" REPEAT="A1:11 B1:11" \
  ARMS="v3 v4 v4n" bash docs/data/2026-11-16_c843_flint_sampling/run/c843.sh start
RIG=<rig alias> WORK=<run folder on the rig> bash docs/data/2026-11-16_c843_flint_sampling/run/c843.sh wait
RIG=<rig alias> WORK=<run folder on the rig> OUT=<folder> bash docs/data/2026-11-16_c843_flint_sampling/run/c843.sh collect
python3 -I docs/data/2026-11-16_c843_flint_sampling/run/c843_table.py <folder> --control v3 --out table.md --tsv streams.tsv
```

## Files

| File | Content |
|---|---|
| [table.md](table.md) | `c843_table.py`'s tables: every request and seed per arm (length, end, the same token ids as `v3` or the first token that differs), the totals, the repeats, the sampler path from the server logs, the settings each reply reports |
| [streams.tsv](streams.tsv) | one row per arm and request: prompt tokens and md5, the body's md5, generated tokens, end, tool calls, the md5s of the token ids, the reasoning, the text, the tool calls and the grammar, the comparison with `v3`, the server's sampling counts |
| [loads.tsv](loads.tsv) | per arm: start, load time, build, variables and line as they ran |
| [run/c843.sh](run/c843.sh) | start, wait, status and collect from the operator's machine |
| [run/c843_rig.sh](run/c843_rig.sh) | the run on the rig: the loads, the quiet checks, the watchdog, the client |
| [run/c843_client.py](run/c843_client.py) | the client: the requests at fixed seeds, one JSON line per reply |
| [run/c843_pick.py](run/c843_pick.py) | picks the requests from recorded request bodies |
| [run/c843_table.py](run/c843_table.py) | the tables |

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), `v4-rc2` and v3.0. Model: [Flint](https://huggingface.co/RecursiveSI/flint)
(Apache-2.0) as my Flint Pascal Q8 GGUF ([models/flint](../../../models/flint/README.md)). The requests are the agent
check's own sessions ([bench/agent_check](../../../bench/agent_check/README.md)): its system prompt and bug-fix tasks;
the tables hold counts, md5s and token positions, no text and no token ids. Measured on a Manli M-P106L9-N6G (nine
P106-090).
