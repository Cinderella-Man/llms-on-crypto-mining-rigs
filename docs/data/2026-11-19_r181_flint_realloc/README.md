# Why Flint's compute graph is re-planned in agent requests (R181, Q1522, 2026-11-19)

What this measured: why Flint Pascal Q8 GGUF re-plans its compute buffers in nearly every agent request (250 and
290 times in the release check's two runs, R159e and R159f; 191 and 200 in R17's two runs) and 0 times in the deep
lane, and what that costs. A re-plan in the middle of a prompt makes the host wait for all nine cards before it moves
the buffers, so the cards' pipeline empties partway through the prompt. Registry row
[Q1522](../../../quants/experiments.md).

One agent check (12 bug-fix tasks) of Flint's v5 preset ([release/flint-v5/settings.env](../../../release/flint-v5/settings.env))
on the v5 engine (`pascal-p106-v5` `602691153`), nine P106-090 cards, every card at stock, with one variable added:
`LLAMA_UBATCH_TRACE=1`. That switch is already in the engine. At the server's default log level it writes a line for
every re-plan (whether a node changed card, and which; the wait for the cards, `drain`; the plan's own time), a line
for the tensor that no longer fit the plan (its name, shape, the bytes it needs and the bytes its place was planned
for), and a line for every micro-batch. It is a trace run, not a timing run.

## Result

**In plain words:** at load, the scheduler that reads prompts is planned once for the largest prompt piece (128 tokens)
over the whole 262,144-token context, so every prompt fits that plan. But the same scheduler also runs the tokens
Flint writes while the tool-call grammar applies, one at a time without pipelined guesses, and a one-token graph
has a different shape (3,392 nodes against 3,898). The scheduler then plans again from the graph in hand, sized for
the context of that moment, and the plan over the whole context is gone. The next prompt plans once more from its
own first piece. From then on, each time the prompt grows the context past another 256-token step, the attention
mask no longer fits its place in that plan, and the host stops to re-plan while the cards still hold earlier pieces
of the prompt. A re-plan like that costs a request **est. 308.0 ms** of prompt time, and this run had 96 of them:
**est. 29.6 s of the run's 162.1 s of prompt reading, 2.7 % of the 18.0 minutes the 12 tasks took.**

The deep lane writes no token under a grammar, so its prompt scheduler keeps the load-time plan: 0 re-plans.

The preset already holds a switch for this, `LLAMA_SCHED_PER_SIZE_RESERVE=2` (C818): after a re-plan, plan over the
whole memory again. **On Flint it does nothing.** It needs per-size schedulers, which only Flash-Next's draft turns
on, and the server says so at load: `LLAMA_SCHED_PER_SIZE_RESERVE=2 needs per-size schedulers (LLAMA_MTP_GRAPH_KEEP=1
or LLAMA_SCHED_PER_SIZE=1), ignored`. It said the same in R17's runs and in the release check's.

### Every re-plan, by cause

The run: 12 of 12 tasks passed, 76 requests, 34,502 prompt tokens read in 162.1 s (213 tok/s), generation 48.6 tok/s
([f5ut_compare.md](f5ut_compare.md)). The release count (the server's `graph reallocs` line, which the agent check's
report sums) is 200 re-plans at an unchanged graph size and 0 at a changed one, in 75 of 76 requests. The trace
finds 718 re-plans in all. The release count leaves out re-plans after a node changed card (128 here) and the
pipelined slots' own schedulers (390), which report no count.

| Scheduler | Pass | What no longer fit | Cards busy | In the release count | Re-plans | Requests | Drain ms | Plan ms |
|---|---|---|---|---|---:|---:|---:|---:|
| prompts | a prompt micro-batch (128 tokens 76, 64 tokens 16, 63 and 60 one each) | the attention mask (`attn_inp_kq_mask`), one 256-token step past the plan | yes | yes | 94 | 40 | 74,604.6 | 335.2 |
| prompts | the prompt's last 4 tokens, with the output | the output index (`out_ids`): the plan came from a piece without one | no | yes | 75 | 75 | 5.4 | 252.6 |
| prompts | a 1-token pass under the grammar | a node on another card (`l_out-5`'s copy, card 0 → card 1): the one-token graph's shape | no | no | 64 | 64 | 4.4 | 172.4 |
| prompts | the next prompt's first micro-batch | a node on another card (`ffn_moe_weights-5`, card 1 → card 0): the prompt graph's shape again | no | no | 64 | 64 | 4.3 | 233.0 |
| prompts | a 1-token pass under the grammar | the attention mask, one step past the plan | no | yes | 18 | 17 | 1.2 | 48.9 |
| prompts | a prompt micro-batch | the token rows (`embd`): the plan came from the 4-token pass | no | yes | 11 | 11 | 0.9 | 37.4 |
| prompts | a prompt micro-batch (67 and 72 tokens) | the token rows, after a 64-token piece | yes | yes | 2 | 2 | 1,184.6 | 7.0 |
| pipelined slots (10) | a pipelined token, the first in the run | the sampler's random input on the host where the load's plan put it on card 8 | no | no | 10 | 1 | 72.7 | 39.9 |
| pipelined slots (10) | a pipelined token | the attention mask, past the longest context so far | no | no | 380 | 7 | 7,046.4 | 1,029.2 |

("Cards busy": the host waited for micro-batches still in flight. In this run every such drain took at least
490.37 ms and every other drain at most 34.08 ms. The busy mask re-plans at 128-token micro-batches drained for a
median 852.78 ms, the longest 958.07 ms.) The 200 counted re-plans are the rows marked "yes" in that column:
94 + 75 + 18 + 11 + 2. In the whole run the prompt scheduler's buffers never grew. They were sized at load for the
whole context, so every re-plan only moved tensors inside room that was already there.

**One request** (task 11484 in [replans.tsv](replans.tsv), 1,467 new tokens in micro-batches of 128): a re-plan at
the first micro-batch (the previous reply's grammar tokens had replaced the plan), then 5 re-plans with the cards
busy, one per 256 tokens, which drained for 867.84, 830.36, 809.95, 844.51 and 860.97 ms; then the 4-token output
pass's re-plan, then the reply's first grammar token's. The prompt took 6,279.61 ms.

### What a busy re-plan costs

Least squares over the 76 requests ([fit.tsv](fit.tsv)), prompt ms = a + b × prompt tokens + c × busy re-plans:
a = 381.0 ms a request (standard error 31.2), b = 3.003 ms a token (0.084), **c = 308.0 ms a re-plan (25.1)**. The
96 busy re-plans (94 mask, 2 token rows) then cost est. 29.6 s (standard error 2.4 s) of the 162.1 s of prompt time.
The drains themselves sum to 75.79 s. A drain is not all lost, since the cards keep working through it; what is lost
is the pipeline refilling afterwards. The pipelined slots' 390 re-plans drained 7.12 s and planned 1.07 s, an upper
bound (est.) on their cost. This run's trace lines add host work of their own, so its speeds are not compared with
other runs.

### Why the release count grows with the prompt

Each request counts one re-plan for its 4-token output pass, plus one for every 256-token step its prompt crosses
after the plan was replaced. That is why R17b's 20-token requests counted 1 and its 1,653-token request 7
([its table](../2026-11-17_r17_flint_textreuse/f4trb_requests_reallocs.tsv)). The first request counts 0: its
prompt runs on the load-time plan, and its one re-plan, at its first grammar token, is a change of card, which the
count leaves out.

## What follows

- **Q1627 (R216), a rig run, no code:** Flint's v5 preset plus `LLAMA_SCHED_PER_SIZE=1`. That switch already exists
  (C151). Passes of up to 8 tokens (the grammar's one-token passes, the 4-token output pass) then get schedulers of
  their own, and the preset's `LLAMA_SCHED_PER_SIZE_RESERVE=2`, which turns on with it, plans each of them once over
  the whole memory. The prompt scheduler then never sees another graph shape, and should keep its load-time plan: 0
  re-plans. A one-token scheduler's buffers over the whole memory should be about what a pipelined slot's are (est.):
  0.65 MiB a card and 0.53 MiB of host memory in this load; the 4-token scheduler's more.
- **Q1628 (C900), code, if Q1627 fails:** `LLAMA_SCHED_PER_SIZE_RESERVE=2` working without per-size schedulers in a
  context with no draft model. After any re-plan of the prompt scheduler, plan again over the whole memory, in the
  shape of the batch at hand, not only the 128-token start batch. A one-token plan then holds every later one-token
  graph and a prompt plan every later prompt; each switch between the two costs one plan with the cards idle
  (about 3.5 ms in this trace, est.).
- **Q1629 (C901), code:** the pipelined slots planned at load with the sampler's random input where a live step puts
  it (the host), or planned again over the whole memory after their first change of card. Their re-plans also join
  the release count. Est. at most 8.2 s a run.

## Files

| File | What |
|---|---|
| [causes.tsv](causes.tsv) | the re-plans grouped by scheduler, pass, cause and whether the cards were busy: counts, requests, drain and plan ms, micro-batch sizes (the table above) |
| [replans.tsv](replans.tsv) | every re-plan: time, request (task id), the micro-batch it happened in (index, tokens, outputs, why its graph was rebuilt), the node that changed card, graph sizes, nodes, leafs, splits, drain and plan us, the trigger lines |
| [requests.tsv](requests.tsv) | every request: prompt and generated tokens and ms, micro-batches, re-plans (all, counted, busy), the release count's own per-request difference, drain and plan ms |
| [fit.tsv](fit.tsv) | the least-squares fit of prompt time on tokens and busy re-plans, with standard errors |
| [f5ut_compare.md](f5ut_compare.md), [f5ut_report_table.md](f5ut_report_table.md), [f5ut_gonogo.md](f5ut_gonogo.md) | the agent check's own report of this run |
| [agent/f5ut/](agent/f5ut/) | the run's `attempts.csv` and `turns.csv` |
| [run/r181.sh](run/r181.sh) | starts the run: the release harness's custom arm with Flint's v5 preset and `LLAMA_UBATCH_TRACE=1` |
| [run/r181_replans.py](run/r181_replans.py) | the four tables from the server log |

## Commands

From the repository's root, on the machine that drives the rig and is the client too (Docker, the sandbox image
`agent-check:0.87.1`). `BIN` is the v5 engine's `llama-server` (`pascal-p106-v5` `602691153`, md5
`86607661d5a19d1a90d19ee4db10c90b`), `MODEL` Flint Pascal Q8 GGUF (`flint-near-q8-p065.gguf`, 40,431,236,544 bytes).
`PREFIX` gives the client's containers names of their own, since another rig's check shared the client.

```bash
RIG=rig CLIENT=local MODEL=$MODEL BIN=$BIN RIG_LOGDIR=$WORK/agent-logs \
  RIG_PORT=8282 RELAY_PORT=8286 PROXY_PORT=18783 CLIENT_DIR=$CLIENT_DIR CLIENT_OUT=$CLIENT_OUT OUT=$OUT \
  PREFIX=acr-r181 bash docs/data/2026-11-19_r181_flint_realloc/run/r181.sh
```

The started command (the harness's line): `LLAMA_KV_SEQ_RM_BOUND=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
GGML_CUDA_GRAPHS_FORCE=1 LLAMA_SCHED_PER_SIZE_RESERVE=2 LLAMA_SERVER_TEXT_REUSE=1 LLAMA_UBATCH_TRACE=1 llama-server
-ngl 99 -fa on -c 262144 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek --temp 1.0
--top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --alias flint -b 512 -ub 128 -ts 6,6,5,5,6,5,5,6,6
--pipe-depth 4 --ctx-checkpoints-vram 4 --ubatch-taper short=600:64`. The model was loaded after 587 s and the 12
tasks ran in 18.3 minutes. Nothing else ran on the rig meanwhile.

The tables, from the server log the harness copies to `$CLIENT_OUT/rig/server-f5ut.log` (46,170,784 bytes; it stays
private, since it holds the session's text):

```bash
python3 -I docs/data/2026-11-19_r181_flint_realloc/run/r181_replans.py docs/data/2026-11-19_r181_flint_realloc \
  $CLIENT_OUT/rig/server-f5ut.log
```

## Provenance and licence

Code: Miner Llama (MIT, like llama.cpp), the v5 engine `pascal-p106-v5` `602691153`. Model:
[Flint](https://huggingface.co/RecursiveSI/flint) (Apache-2.0) as my Flint Pascal Q8 GGUF
([models/flint](../../../models/flint/README.md)). The tasks are the agent check's bug-fix set
([bench/agent_check](../../../bench/agent_check/README.md)); the tables hold times, counts, tensor names and task
ids, no text. Measured on a Manli M-P106L9-N6G (nine P106-090, 4 GB host) on 2026-11-19, every card at stock (75 W,
no clock offset); the tables were made with the script above.
