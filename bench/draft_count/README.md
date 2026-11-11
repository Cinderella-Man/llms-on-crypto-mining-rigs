# Draft count: what a draft of 3 to 7 tokens yields, by draft position (Q1395)

Flash-Next decodes with its MTP draft: the draft proposes a few tokens and the main model checks them all in one verify
pass of draft + 1 tokens. The release preset drafts 3 tokens. A longer draft pays only if the tokens it adds are
accepted often enough to cover the wider verify pass and the extra drafting passes, so the first question is how often
each draft position is accepted. This tool counts it on recorded agent requests, for several draft lengths in one run,
with the counts only (no time): acceptance depends on the models, the sampler and the text, not on the card, so the run
belongs on any machine that holds the model. The time per step at each width is the rig's job (a step map such as
Q1393's); `draft_table.py --cost` combines the two into an estimate.

The same loads can dump the draft's most probable candidates at every drafted position (`LLAMA_SPEC_DUMP`), which
[bench/tree_replay](../tree_replay/README.md) reads to price token trees against plain drafts of the same verify width.

## What runs

`run_draft_count.sh` is one chain on one set of cards. Every load is one llama-server in a container on the line of a
preset (default [release/flash-next-v3/settings.env](../../release/flash-next-v3/settings.env)) with:

- the arm's draft length: an arm named `d<N>` drafts N tokens (`--spec-draft-n-max N` in place of the preset's);
- probabilistic verification (`--spec-draft-sampling probabilistic --spec-draft-p-min 0`, `COMMON_ARGS`) and the
  end-of-turn stop of token-by-token rejection (`LLAMA_SPEC_REJECT_STOP=1`, `COMMON_ENV`);
- `--metrics`, `LLAMA_TRACE=1` (one log line per verify step, nothing else changes) and, with `DUMP=1` (the default),
  `LLAMA_SPEC_DUMP` with the draft's `DUMP_K` = 3 most probable candidates per position;
- without the preset's nine-card split (`-ts`, `-otd`) and `GGML_CUDA_KEEPALIVE`; `TS=` gives the cards' split, faster
  cards first (CUDA lists the cards by speed, not by bus).

[bench/agent_check/replay.py](../agent_check/replay.py) then re-sends the recorded requests of one agent-check arm, one
at a time, with the same prompts, seeds and reply limits in every load. After the last request the chain saves the
server's `/metrics`. The MTP line sizes its recurrent snapshots to the draft length (`n_rs_seq` = `--spec-draft-n-max`),
so a draft of 7 is rolled back without a checkpoint restore, as a draft of 3 is; the tables count any restore.

## The counts

`draft_table.py` reads every load three ways and checks them against each other:

| Source | What it gives |
|---|---|
| the replay rows (the server's `timings`) | per request: generated tokens, drafted (`draft_n`), accepted (`draft_n_accepted`), tokens drawn on the device, grammar rejections, the reply's MD5 |
| the trace lines (`accepted a/d draft tokens`) | per verify step: the accepted a of the d drafted tokens, and the request (task) it belongs to |
| `/metrics` | per load: the verify steps, the drafted and accepted totals and the server's own counter of accepted tokens per draft position (`llamacpp:spec_decode_num_accepted_tokens_per_pos_total`) |
| the dump (with `DUMP=1`) | per verify step: the draft, the target's tokens, `n_acc`, the draft's candidates and their probabilities |

- **Tokens a step:** generated tokens over steps, a step being every token the main model sampled itself (generated
  minus accepted), pooled over the load's requests; FN43 (Q1173) used the same.
- **By position** (from the trace lines): at position i, *drafted_i* steps drafted i or more tokens, *reached_i* of
  them accepted every earlier drafted token, *accepted_i* accepted i or more. **P_i** = accepted_i / drafted_i is the
  probability that the first i are all accepted, **c_i** = accepted_i / reached_i the conditional acceptance of
  position i. Near the end of a reply the server drafts fewer tokens (at most the tokens left), so the denominators are
  counted per position.
- **Cut to a shorter draft** (`cut.tsv`): accepted + 1 per verify step if each step of a load had verified only its
  first L drafted tokens, an estimate on the same steps to read against a `dL` load's own count.
- **Arms and comparisons:** each load by value and each arm pooled over its loads; every arm against `--ref` (default
  the first arm loaded) as the pooled difference in tokens a step with 95 % bootstrap intervals over requests and over
  traces, beside both arms' spreads between loads. Two loads of one arm write different replies on this line (FN43), so
  an arm needs two or more loads and a difference is read against their spread.
- **By context length** (`by_context.tsv`, `--prompt-bins`, default 4,096 and 8,192): tokens a step, P_i and c_i
  per arm for the requests whose context (prompt tokens, cached or not) falls in each bin.
- **Checks** (`check.tsv`): the trace's sums equal the timings'; `/metrics` per position equals the trace's
  accepted_i, and its steps and sums equal the trace's; the dump's steps equal the trace's step for step; no step
  restored a checkpoint; every request matched its task.
- `--cost FILE` (columns `n_max` and `ms_step`: an estimated time per step at each draft length) adds est. tokens per
  second per arm.

## Use

```
WORK=<work folder> REPO=<checkout> BIN=<llama-server> MODEL=<first shard> MODEL_DRAFT=<MTP draft> \
  LOG=<recorded proxy.jsonl, its .bodies.gz next to it> GPUS=<docker device list, or cpu> MOUNTS="<folders>" \
  [LOADS="d3 d4 d5 d6 d7 d7 d6 d5 d4 d3"] [TS=<split>] [TAG=^t1swift/] [DUMP=1] [DUMP_K=3] \
  setsid -f nohup nice -n 19 bash run_draft_count.sh > console.txt 2>&1 < /dev/null
python3 draft_table.py WORK [WORK ...] --out DIR [--ref d3] [--cost COST.tsv]
python3 draft_table.py --selftest
python3 ../tree_replay/tree_replay.py WORK/dump-d7-1.jsonl WORK/dump-d7-2.jsonl --out TREE --mode factorized
```

More settings are listed at the top of `run_draft_count.sh`: `TRACES_N` and `REQUESTS` take a subset of the recorded
requests, `REPLY_LIMIT` caps the replies, `ARGS_<arm>` and `ENV_<arm>` add flags and variables to one arm
(`ENV_<arm>=LLAMA_SPEC_DUMP=` turns its dump off), `START_BY` lets no load start after a given time, and a finished
load is skipped when the chain starts again. `GPUS=cpu` runs the containers without cards, for a CPU build and a preset
written for it.

Output in DIR: `loads.tsv`, `positions.tsv`, `arms.tsv`, `compare.tsv`, `by_context.tsv`, `cut.tsv`, `requests.tsv`,
`same_reply.tsv`, `check.tsv` and `summary.md`. Standard library only. `--selftest` builds synthetic loads with known counts, checks
every table against them, and checks that a per-position counter or a dump that disagrees with the trace is caught.

The CPU dry run that checked the tools end to end, on a random-weight model with an MTP block:
[docs/data/2026-11-11_c798_draft_count_tools](../../docs/data/2026-11-11_c798_draft_count_tools/README.md).
