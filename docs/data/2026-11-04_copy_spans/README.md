# Q617 and Q624: how much of Flash-Next's agent output is copied from its context, and what copy chains would gain (2026-11-04)

Idea 2 of the [MoE speculation note](../../research/notes/moe_speculation_union.md) is long copy chains: 16–32-token
drafts from an earlier match, est. +20–45 % on agent sessions if 25–40 % of the output is copied. It is disproved if the
copy share of Flash-Next's own agent output is under 15 %, or if a match continues correctly less than 85 % of the time
inside copied text. This folder measures both on Flash-Next's agent-check transcripts, with the tool in
[bench/copy_spans](../../../bench/copy_spans/README.md), and replays a prompt-lookup guesser and a perfect copy guesser
on the same transcripts. Registry rows: Q617 (chains), Q624 (the kernel beyond 8 tokens), new ideas Q785–Q789 in
[quants/experiments.md](../../../quants/experiments.md). Nothing ran on a GPU or a rig.

## Result: a quarter to a third of the output is copied, in short runs, and a copy chain adds nothing to DFlash2

- **Neither of the note's disproving tests fires.** The share of the generated tokens that lies in a verbatim repeat of
  at least 8 tokens is **25.1 % for the prompt alone (95 % interval 23.0–27.4) and 34.0 % for the prompt plus the
  response's own earlier text (31.7–36.3)**; at 16 tokens it is 11.3 % and 14.2 %. A match of 6 or more tokens is
  followed by the right token 93–99 % of the time inside copied text.
- **Its estimate does not hold.** Copy spans are short: median 11 tokens, mean 15, 6 % of them 32 or longer. A
  prompt-lookup guesser with 16-token drafts accepts **2.8 draft tokens per draft step** (match of 4 or more tokens; 4.3
  for 8 or more). A guesser that always picks the best earlier occurrence of its 4-token key and drafts exactly what is
  right accepts 5.0, and one that knows each span from its 4th token on, 9.1.
- **On top of DFlash2 a copy chain gains nothing.** No lookup setting beats DFlash2 alone: the best keeps 0.995 of its
  speed after the multi-token kernels and 0.945 at today's verify cost. The two perfect guessers gain +5–6 % today and
  +13–15 % after the kernels (decode steps only; +10 % and +24 % if one also knew where every span starts). Decode is about
  70 % of a task.
- **Chains beyond 8 columns do not pay.** Tokens per step of the perfect guessers at 8, 16 and 32 drafted tokens (key of
  4): 1.305, 1.325, 1.332 (rest of the span) and 1.373, 1.390, 1.395 (best occurrence). Q624's 16-column kernel buys
  at most 1.5 % of the tokens per step. Choosing a better occurrence than the newest one has 1.6 % to gain (1.368 to
  1.390 at 16 tokens).
- **Where the copies are.** 64 % of the output is reasoning, and 25 % of that is copied, which is 48 % of all copied
  tokens. Tool-call arguments (12.6 % of the output) are 49 % copied; an edit's `oldText` is 82 % copied and 76 % of those
  values start copying at their first token. But only 3.1 % of the spans start right after a marker, so a start at the
  marker would not remove the 4–8 tokens a guesser needs to find its source.

### Share of the generated tokens in copy spans

419 responses from 81 sessions (an arm's attempt at one of the 12 bug-fix tasks), 115,794 generated tokens, Flash-Next at
temperature 1.0. Intervals: bootstrap over sessions.

| Minimum span length | prompt only (`ctx`) | prompt and own earlier text (`own`) |
|---:|---:|---:|
| 4 tokens | 40.9 % (38.3–43.7) | 54.2 % (51.6–56.8) |
| **8 tokens** | **25.1 % (23.0–27.4)** | **34.0 % (31.7–36.3)** |
| 16 tokens | 11.3 % (10.1–12.7) | 14.2 % (13.0–15.4) |
| 8 tokens, without the call scaffolding (10 % of the output) | 20.1 % | 29.6 % |
| 8 tokens, with JSON-escaped tool results added as a source (`esc`, `own+esc`) | 25.6 % (23.4–27.9) | 34.3 % (31.9–36.6) |

([share_overall.csv](tables/share_overall.csv), [share_without_scaffolding.csv](tables/share_without_scaffolding.csv))

- **Consistent across arms and tasks.** At 8 tokens, `own` is 31.5–37.7 % in each of the nine arms with more than 3,000
  tokens (`ctx` 20.9–28.1 %), and 24.8–46.9 % across the 12 tasks ([share_by_task.csv](tables/share_by_task.csv)).
  Half of the responses have 35 % or more (`own`, 8 tokens), 95 % have 15 % or more
  ([share_per_response.csv](tables/share_per_response.csv)).
- **By turn** ([share_by_turn.csv](tables/share_by_turn.csv)): the long reasoning after the first tool results (turn 2,
  45 % of all tokens, 92 % of it reasoning) copies the least from the prompt (15 %) and 31 % with its own text; turns
  3–4, which write the edit, copy 32–43 % from the prompt.
- **The escape-aware source adds almost nothing.** Inside a JSON string a newline is `\n`, so text that the model writes
  into an `edits` argument does not match the raw file text; adding every tool result in its escaped form as a source
  gains 0.5 points at 8 tokens and 1.2 at 16. The edits in this task set are short.

### Which part of the response is copied

`own`, spans of 8 or more tokens ([share_by_region.csv](tables/share_by_region.csv)); `ctx` in brackets.

| Part of the response | share of the output | of which copied | share of all copied tokens |
|---|---:|---:|---:|
| reasoning | 64.4 % | 25.5 % (13.9 %) | 48.3 % |
| answer text | 13.1 % | 31.1 % (29.8 %) | 12.0 % |
| call scaffolding (`</think>`, `<tool_call>`, `<parameter=…>`, the closing tags) | 10.0 % | 73.6 % (70.2 %) | 21.6 % |
| all tool-call arguments | 12.6 % | 49.2 % (42.0 %) | 18.2 % |
| `bash.command` | 7.1 % | 35.5 % (32.5 %) | 7.4 % |
| all file-edit arguments (`edit.*`) | 5.3 % | 67.8 % (55.2 %) | 10.6 % |
| `edit` `oldText` | 1.9 % | 81.6 % (81.6 %) | 4.4 % |
| `edit` `newText` | 2.1 % | 81.8 % (57.3 %) | 4.9 % |

No `write` call occurs: all 12 tasks are small fixes. A `newText` is mostly the model's own reasoning draft of the fix,
which is why it needs `own`.

### How long the copied runs are, and where they start

`own`, 8 or more tokens: 2,554 spans, mean 15.4 tokens, **median 11**, 90th percentile 27, longest 326
([span_lengths.csv](tables/span_lengths.csv)).

| Span length | spans | share of the copied tokens |
|---:|---:|---:|
| 8–15 | 69.5 % | 47.2 % |
| 16–31 | 24.7 % | 33.8 % |
| 32–63 | 5.1 % | 13.8 % |
| 64–127 | 0.6 % | 3.3 % |
| 128 and longer | 0.1 % | 1.8 % |

- **Few spans start at a marker.** Of the 2,554 spans, 59 % start in reasoning (no marker yet), 10 % in answer text,
  16 % in call scaffolding and 14 % inside an argument; only 3.1 % start exactly after a marker and 4.0 % within two
  tokens ([span_starts.csv](tables/span_starts.csv)). Per argument value the picture is clearer
  ([argument_values.csv](tables/argument_values.csv), `own`, 8 or more tokens):

| Argument | values | mean tokens | with a copy span | copy starts at its first token | copy runs in from before | copy starts later | no copy |
|---|---:|---:|---:|---:|---:|---:|---:|
| `edit` `oldText` | 93 | 23.0 | 87 % | **76 %** | 5 % | 5 % | 13 % |
| `edit` `newText` | 93 | 25.6 | 90 % | 3 % | 75 % | 12 % | 10 % |
| `bash` `command` | 278 | 29.6 | 45 % | 0 % | 34 % | 11 % | 55 % |
| `edit` `path` | 86 | 4.9 | 52 % | 0 % | 33 % | 20 % | 48 % |

  An `oldText` quotes the file the agent just printed from its first token; a `newText` continues the copy of its
  `oldText` scaffolding; a command is mostly new text.
- **How far a match goes** ([continuation_by_match_length.csv](tables/continuation_by_match_length.csv); every position
  with an earlier match, `own`): a match of 8–11 tokens is followed by the right token 79 % of the time (93 % inside a
  copy span, 5 % outside), then by 4.9 right tokens in a row on average, 20 % of the time by 8 or more, 7.6 % by 16 or
  more. A match of 24–47 tokens: 91 %, 10.1 tokens, 45 %, 22 %. Only matches of 48 or more reach 21 tokens in a row.
  Text that has been repeated for 24 tokens is usually an edit's `oldText` or an earlier command, and the copy stops
  where the new edit differs.

### The guesser

`own` source, newest occurrence of the longest match, draft of `n` tokens after a match of at least `m` (the key);
speed-up over one-token decoding at r = 0.40 (the rig today) and r = 0.10 (est. after the multi-token kernels), 95 %
intervals from sessions ([guesser.csv](tables/guesser.csv)). Two bounds on any copy guesser: the *span oracle* drafts
the rest of the current span of 8 or more copied tokens once `m` of them were generated; the *occurrence oracle* takes,
among the earlier occurrences of the last `m` tokens (the newest 400), the one whose continuation is right for longest.
Both draft exactly the tokens that are right and take a plain step otherwise.

| Guesser | `n` / `m` | steps with a draft | accepted per draft step | tokens per step | columns per step | r = 0.40 | r = 0.10 |
|---|---|---:|---:|---:|---:|---:|---:|
| lookup | 2 / 4 | 20.5 % | 1.24 | 1.252 | 1.41 | 1.08 | 1.20 |
| lookup | 8 / 4 | 14.2 % | 2.49 | 1.351 | 2.13 | 0.93 (0.92–0.94) | 1.21 (1.20–1.23) |
| lookup | 8 / 8 | 4.5 % | 3.50 | 1.154 | 1.36 | 1.01 (1.00–1.02) | 1.12 (1.11–1.12) |
| lookup | **16 / 4** | 13.2 % | **2.82** | 1.368 | 3.05 | 0.75 (0.73–0.77) | 1.14 (1.12–1.15) |
| lookup | 16 / 8 | 3.8 % | 4.29 | 1.162 | 1.61 | 0.94 (0.92–0.95) | 1.10 (1.09–1.11) |
| lookup | 32 / 4 | 12.9 % | 2.92 | 1.372 | 4.83 | 0.54 (0.52–0.57) | 0.99 (0.98–1.01) |
| lookup | 32 / 8 | 3.6 % | 4.61 | 1.165 | 2.11 | 0.81 (0.79–0.83) | 1.05 (1.04–1.06) |
| lookup | adaptive (2 × match, ≤ 32) / 4 | 13.7 % | 2.65 | 1.360 | 2.45 | 0.86 | 1.19 |
| span oracle | 8 / 4 | 4.8 % | 6.36 | 1.305 | 1.31 | 1.16 (1.15–1.18) | 1.27 (1.24–1.29) |
| span oracle | 16 / 4 | 3.6 % | 9.14 | 1.325 | 1.33 | 1.17 (1.16–1.19) | 1.28 (1.26–1.31) |
| span oracle | 32 / 4 | 3.1 % | 10.76 | 1.332 | 1.33 | 1.18 (1.16–1.19) | 1.29 (1.26–1.32) |
| span oracle | 16 / 0 | 4.4 % | 11.26 | 1.494 | 1.50 | 1.25 | 1.42 |
| occurrence oracle | 8 / 4 | 8.9 % | 4.20 | 1.373 | 1.38 | 1.19 (1.18–1.21) | 1.32 (1.30–1.35) |
| occurrence oracle | 16 / 4 | 7.9 % | 4.98 | 1.390 | 1.39 | 1.20 (1.19–1.22) | 1.34 (1.31–1.37) |
| occurrence oracle | 16 / 8 | 2.9 % | 5.82 | 1.165 | 1.17 | 1.09 (1.09–1.10) | 1.15 (1.13–1.16) |
| occurrence oracle | 32 / 4 | 7.6 % | 5.25 | 1.395 | 1.40 | 1.20 (1.19–1.22) | 1.34 (1.31–1.37) |

- **A 16-token draft is accepted 2.8 tokens deep**, because only 18 % of drafted tokens are right: the draft is cut at
  the first difference. A key of 8 tokens raises the depth to 4.3 and removes 71 % of the drafts. The copies a lookup at
  `n` = 16, `m` = 4 reaches are 27 % of the tokens ([guesser_by_region.csv](tables/guesser_by_region.csv): 49 % of the
  accepted tokens are reasoning, 20 % scaffolding, 13 % answer text, 8 % commands, 10 % edit text).
- **Picking another occurrence cannot do much.** The occurrence oracle sees every candidate and still gets 1.390
  tokens per step at `n` = 16, `m` = 4 against the newest-occurrence lookup's 1.368: the repeats themselves are short
  or end where the new text differs.
- **Draft lengths past 8 add nothing** (tokens per step 1.351, 1.368, 1.372 for the lookup at 8, 16, 32; 1.373, 1.390,
  1.395 for the occurrence oracle; 1.305, 1.325, 1.332 for the span oracle) and cost columns.

### With DFlash2

DFlash2 on the rig gives 2.56 tokens per step with two drafted tokens (Q561). Its per-request acceptance falls with
the request's copy share: a line through the 60 requests of the `dflash2b` arm
([dflash2_vs_copy_share.csv](tables/dflash2_vs_copy_share.csv)) gives **2.35 tokens per step on text with no copy
(2.28–2.45) and 2.92 on text that is all copy (2.78–3.04)**, so DFlash2 already takes 2.9 of the 3 tokens a step can
give inside copies. The table puts a copy guesser in front of DFlash2: a step with a match of `m` or more tokens is a
copy draft, any other step is a DFlash2 step at 2.35 tokens. Gain is the speed-up over DFlash2 alone (modelled
1.31× at r = 0.40, 1.80× at r = 0.10), decode steps only ([combined.csv](tables/combined.csv)).

| Copy guesser in front of DFlash2 | `n` / `m` | tokens per pass | columns per pass | gain, r = 0.40 | gain, r = 0.10 |
|---|---|---:|---:|---:|---:|
| lookup | 2 / 4 | 2.31 | 3.00 | 0.93 | 0.96 |
| lookup | 8 / 6 | 2.65 | 3.89 | 0.88 | 0.995 |
| lookup | 16 / 4 | 2.74 | 6.57 | 0.62 | 0.88 |
| lookup | 16 / 8 | 2.60 | 4.18 | 0.82 | 0.95 |
| span oracle | 16 / 4 | 2.98 | 3.58 | 1.05 | 1.13 |
| span oracle | 16 / 0 | 3.33 | 3.91 | 1.10 | 1.24 |
| occurrence oracle | 16 / 4 | 2.96 | 3.50 | 1.06 | 1.15 |
| occurrence oracle | 16 / 8 | 2.64 | 3.25 | 0.99 | 1.03 |

A bound draft is only what is right, so its copy steps waste no columns. With DFlash2 at its overall 2.56 tokens outside
the copies, the span oracle at 16 / 0 gains 1.18 / 1.33 ([combined.csv](tables/combined.csv)). The decode
part of a task is 69–73 % of its time in the baseline and retype runs' tables, so a session gains about 70 % of each
figure.

### What this does to the note's idea 2

The copy share (25–34 % at 8 tokens) is inside the note's 25–40 % range, and the continuation rate (93–99 %) is above its
85 % line, so neither disproving test fires. The "4.5–7× inside a span" does not hold on this workload: spans are short
(median 11 tokens), DFlash2 already gets 2.9 of 3 tokens inside them, and a lookup guesser keeps 0.88–0.995 of DFlash2's
speed. Registry Q617 is parked and the 16-column kernel (Q624) is not needed for chains. New rows: a draft started at a
marker (Q785, parked: 3 % of spans start at one), a better occurrence than the newest (Q786, parked: 1.6 % headroom),
copy spans on workloads with longer rewrites (Q787, the one that could reopen Q617), a JSON-escape-aware source (Q788,
parked: +0.5 points), and this table redone from per-step DFlash2 data (Q789).

## Method

- **Data.** Four agent-check runs of Swift-1.5 GSQ-RCO Q2_0 (a Flash-Next fine-tune; its BF16 tensors retyped in the `hcq8` arm) on the rig, nine arms with more
  than 3,000 generated tokens and one short one, all temperature 1.0 (top-k 20, top-p 0.95), the pi agent fixing the 12
  bug-fix tasks of [bench/agent_check](../../../bench/agent_check/README.md) with its tools `read`, `bash`, `edit` and
  `write`: [flash_next_baseline](../2026-11-03_flash_next_baseline/README.md) (`stock`: llama.cpp `d77256999`; `rel`:
  Miner Llama v3), [retype_agent](../2026-11-03_retype_agent/README.md) (`hcq8`),
  [flash_next_dflash2](../2026-11-03_flash_next_dflash2/README.md) (`nodraft`, `mtp`, `mtp2`, `dflash2b`) and
  [flash_next_pipelined](../2026-11-03_flash_next_pipelined/README.md) (`pipe0`, `pipe3`, `pipe3b`). Speculative arms
  sample the same distribution as plain decoding, so every arm is a sample of the model's output. One request in the
  logs had no finished reply (the DFlash2 arm stopped by the watchdog) and is left out. The tasks repeat across arms
  (81 sessions, 12 tasks), which is why intervals are bootstrapped over sessions.
- **Rebuild of the prompt and the response** ([bench/copy_spans](../../../bench/copy_spans/README.md)): the prompt
  text from each logged request through the chat template (hand-written, equal to the Jinja template byte for byte in
  all 419 requests), the response from the server's parsed reply in the model's tool-call format. Tokenized with
  `llama-tokenize` on a tokenizer-only GGUF made from the model file (pre-tokenizer `qwen35`, 248,320 tokens). Checks:
  the rebuilt prompt has the server's token count in **419 of 419** requests; the rebuilt response is within 2 tokens of
  the server's in 419 of 419 (115,794 tokens against 115,791); every tokenized text decodes back to its bytes
  ([render_check.csv](tables/render_check.csv)).
- **Matching.** A suffix automaton over the prompt (newest end of every state) and a hash index of n-grams (up to 48)
  over the response's own earlier text, the last 16 prompt tokens counting as part of the response's stream. Checked
  against brute force on 1,200 random cases, the span definition on 1,500, the guesser's steps against a plain
  re-implementation on 1,800 ([selftest.py](../../../bench/copy_spans/selftest.py)).
- **DFlash2 model.** Q561's 2.56 tokens per step, two drafted, 3 columns, 15 % of a step for the draft pass (22 % after
  the kernels); a pass of k columns costs 1 + r (k − 1) one-token steps. The model overstates DFlash2 today (it gives
  1.31× where the rig measured 1.15–1.24×); the gains in the table are ratios of two model values.

## Limits

- The task set is 12 small bug fixes, five turns each, contexts of 2.8–10 thousand tokens, edits of a few lines and no
  file written from scratch. Q64 found 99 % of Flint's whole-file rewrites copied; a workload of rewrites and long
  sessions is Q787.
- The 2.35 and 2.92 come from 60 requests and one line; Q789 replaces them with per-step data from Q688's dump.
- The cost model's r is a model; at r = 0.40 every long draft loses, at r = 0.10 they break even. Neither is measured
  for drafts of 8–32 columns on the rig (Q591's kernels are timed to 8).

## Files

| File | What it holds |
|---|---|
| [tables/share_overall.csv](tables/share_overall.csv) | share of tokens in copy spans by scope (all, each arm), source and minimum length, with 95 % intervals |
| [tables/share_per_response.csv](tables/share_per_response.csv), [share_by_task.csv](tables/share_by_task.csv), [share_by_turn.csv](tables/share_by_turn.csv), [share_without_scaffolding.csv](tables/share_without_scaffolding.csv) | the same by response, task and turn, and without the call scaffolding |
| [tables/share_by_region.csv](tables/share_by_region.csv) | by part of the response, with the aggregate rows for arguments and file edits |
| [tables/span_lengths.csv](tables/span_lengths.csv), [span_starts.csv](tables/span_starts.csv), [argument_values.csv](tables/argument_values.csv) | span lengths; the marker before each span; per tool-argument value |
| [tables/continuation_by_match_length.csv](tables/continuation_by_match_length.csv) | next-token and run-length accuracy of a match by its length |
| [tables/guesser.csv](tables/guesser.csv), [guesser_by_region.csv](tables/guesser_by_region.csv), [combined.csv](tables/combined.csv) | the simulation: lookup, oracle, with DFlash2 as the fallback |
| [tables/dflash2_vs_copy_share.csv](tables/dflash2_vs_copy_share.csv), [tau_out.txt](tables/tau_out.txt), [model.json](tables/model.json) | the DFlash2 line and the cost model |
| [tables/per_request.csv](tables/per_request.csv), [render_check.csv](tables/render_check.csv) | one row per response: tokens, copied tokens per setting; the rebuild checks |
| [run/copy_spans.sh](run/copy_spans.sh) | the commands that produced the tables |

No transcript text is published. The text samples and the token files stay with the raw logs.

## Commands

[run/copy_spans.sh](run/copy_spans.sh) is the whole run; it takes the machine (`RIG`, the ssh command of the analysis
machine), the work folder (`WORK`), the build with `llama-tokenize` (`TOKENIZE`), a model file (`MODEL`) and the run
folders from the environment. It copies the four runs' `proxy.jsonl` and `proxy.jsonl.bodies.gz` (md5 sums printed),
runs `bench/copy_spans/run.sh` with the DFlash2 fit on the `dflash2b` arm, and writes the tables. About 25 seconds on
six cores; no GPU.

## Provenance

The transcripts are the agent check's own logs (tasks: Exercism's Python and JavaScript tracks, MIT, selected by
[bench/agent_check](../../../bench/agent_check/README.md)); the model is Swift-1.5 (Qwen3.8-Flash-Next fine-tune; its
licence is in the model repository). The tables hold counts only.
