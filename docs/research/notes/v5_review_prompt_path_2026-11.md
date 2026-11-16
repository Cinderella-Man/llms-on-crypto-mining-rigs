# The prompt path on v4.0, three models: where the wait is now and what v5 can take (2026-11-16)

A v5 review of prompt reading and the first-token wait on Miner Llama v4.0: Flash-Next (UkisAI's Swift-1.5 GSQ-RCO
Q2_0 with its MTP draft), Flint (Flint Pascal Q8 GGUF, `p065`; Flint Pascal Q5 GGUF) and Ornith (Ornith-1.5-122B
Pascal N2P GGUF, `valid1`). As in [the third review](flash_next_review3_2026-11.md), "est." marks my arithmetic or
inference with its basis; other numbers are measured, one run a value, from the linked folder or
[the registry](../../../quants/experiments.md). Nothing ran on any machine.

## The short answer

- **Reading is a fifth of an agent task on all three models** (20 to 23 % in the five runs without a cut reply). A
  1 % cut of prompt time is est. 0.2 % of a task; what it still decides is the first-token wait.
- **The largest item is work, not speed.** A reply that comes back split into other tokens is read again: 30.4 and
  47.3 % of Flint Q8's prompt seconds, 46.1 % of Flint Q5's, 7.7 and 15.4 % of Ornith's. The fix is in the build,
  off, classed negligible by a measure that cannot see it.
- **Flash-Next and Ornith pay a fixed est. 4.3 s and 6.8 to 7.2 s on every turn of 512 to 1,999 new tokens.** The
  rule against it was judged on a line that rebuilt its graph every request.
- **Flint reads an agent's prompts at under half its deep lane's speed** and reallocates its graph 250 and 290 times
  a run, never in the deep lane.
- **Ornith's taper question is est. a kernel question:** small micro-batches move its expert products off a prompt
  kernel known to round N2P (Q307).

## Where an agent task's time goes on v4.0

Seconds a task, means over each run's 12 tasks, the release preset on `v4-rc2`:

| Model, run | All | Generation | Prompt reading | The rest (tools, the client, fixed costs) |
|---|---:|---:|---:|---|
| Flash-Next, [R156a](../../data/2026-11-13_r156_agent_v4r/README.md) | 168 | 131 (78 %) | 33 (20 %) | 4 (2 %) |
| Flash-Next, R156b | 153 | 78 (51 %) | 23 (15 %) | 52 (34 %): a reply cut at the 12-minute cap |
| Flint Q8, [R159e](../../data/2026-11-14_r159_flint_release_agent/README.md) | 144.0 | 64.0 (44.4 %) | 18.5 (12.9 %) | 61.5 (42.7 %): a cut reply, 705.5 s |
| Flint Q8, R159f | 116.3 | 86.3 (74.2 %) | 25.0 (21.5 %) | 5.1 (4.4 %) |
| Flint Q5, [R170a](../../data/2026-11-16_r170_flint_q5/README.md) | 115.1 | 86.0 (74.7 %) | 25.9 (22.5 %) | 3.2 (2.8 %) |
| Ornith, [R166a](../../data/2026-11-16_r166_ornith_stack_no_taper/README.md) | 145.3 | 110.0 (75.7 %) | 33.1 (22.8 %) | 2.2 (1.5 %) |
| Ornith, R166b | 184.0 | 142.7 (77.5 %) | 39.0 (21.2 %) | 2.4 (1.3 %) |

How: Flint and Ornith are sums over each run's `agent/<arm>/attempts.csv` (`wall_s`, `prefill_s`, `decode_s`,
`fixed_s` + `other_s`), divided by 12. Flash-Next's rows are the report's whole seconds (`v4r2_report_table.md`);
its per-request rows are not published. A cut reply is generation the server never timed. Runs did different work:
compare shares, not seconds.

Prompt tok/s by the size of a turn's new prompt (runs in the table's order; `prompt_buckets.tsv`,
`prompt_by_size.tsv`, Ornith summed from `turns.csv`):

| New prompt tokens | Flash-Next | Flint Q8 | Ornith |
|---|---|---|---|
| under 512 | 93.0, 96.4 | 172.7, 183.8 | 46.1, 45.5 |
| 512 to 1,023 | 112.3, 123.0 | 220.3, 217.5 | 86.7, 82.6 |
| 1,024 to 1,999 | 152.4, 185.5 | 236.9, 230.7 | 143.9, 143.9 |
| 2,000 and more | 181.7, 177.5 | 236.2, 196.0 | 149.8, 162.2 |
| share of prompt seconds under 512; at 2,000 and more | 30.5, 28.3 %; 25.4, 24.9 % | 22.3, 14.8 %; 38.8, 51.2 % | 29.3, 31.0 %; 21.5, 27.0 % |
| first token, s: median; p90; first turn | 2.42, 2.48; 9.00, 8.65; 6.68, 6.65 | 1.74, 1.82; 6.61, 6.35; 3.24, 3.18 | 6.69, 6.06; 11.29, 11.32; 8.43, 8.45 |

Where each is still slow:
- **First turns and turns of 512 to 1,999 tokens** carry a fixed cost: Ornith est. 6.81 s + 2.29 ms a token and
  7.19 s + 2.03 ms (least squares, 21 requests a run); Flash-Next est. 4.29 s + 3.38 ms and 4.32 s + 2.39 ms (the
  two buckets' means); Flint est. 0.74 s + 3.63 ms and 0.65 s + 3.79 ms. So Flint adds tokens at 275 and 264 tok/s,
  against 549.3 and 566.8 at 4,096 tokens in its deep lane
  ([R168](../../data/2026-11-16_r168_flint_f4t_deep/README.md)).
- **Under 512 tokens** a request takes 1.82 and 1.81 s (Flash-Next), 1.04 and 1.03 s (Flint), 3.14 and 3.15 s
  (Ornith); Ornith's 257 to 511-token ones 6.95 and 7.09 s, 3.87 s with the taper
  ([R160b](../../data/2026-11-14_r160_ornith_release_agent/README.md)).
- **At 2,000 and more** Flint's seconds are re-reads (the report's rule, on `turns.csv`): four turns of R159e take
  67.6 of 222.6 prompt seconds, five of R159f 141.7 of 299.5, one alone 93.4 s.

## What v4 proved and what lost here

Proved:
- CUDA graphs for prompt micro-batches: −44.45 s of a 300.53 s pass over 62 agent prompts; the slimmer catch-up
  −9.62 s; the overlapped catch-up and `-b 4096` −2.90 and −2.83 s
  ([R123a](../../data/2026-11-12_r123a_prompt_ladder/README.md),
  [R123b](../../data/2026-11-12_r123b_prompt_ladder/README.md)).
- One graph shape in every batch: 0 reallocations since; agent prompts from 73.2 to 136.0 and 147.6 tok/s
  ([FN84](../../data/2026-11-12_fn84_graph_realloc/README.md), [R72](../../data/2026-11-12_r72_agent_v4c2/README.md)).
- Checkpoints on the cards and in the call: 338.96 and 331.09 s a pass against 509.16 and 499.93
  ([R43](../../data/2026-11-10_prompt_lane/README.md), [R85](../../data/2026-11-11_r85_c217_p106/README.md)); Flint
  restores in 2.0 to 2.2 ms against 292. The host no longer limits `-b 4096` (lowest free memory 312 and 419 MiB on
  Flash-Next, 1,225 and 1,228 on Ornith); Flint stays at `-b 512`.
- The short taper: Ornith's first-token median 3.05 and 3.27 s against 6.69 and 6.06; Flint under 512 tokens 172.7
  and 183.8 tok/s against 123.5 and 121.5.

Lost:
- The n-gram prefetch, +5.09 s a pass; the doubled launch queue, −0.80 s for 222 MiB of host memory; the prompt
  split, +1.39 s ([R141](../../data/2026-11-13_r141_ppsplit_alt/README.md)).
- The keep-alive on Flint: prompts −16.5 % ([R165](../../data/2026-11-16_r165_flint_stack_split/README.md)); at
  depth prompts read 2.3 % faster without it on Flash-Next and 2.6 % on Ornith
  ([R167](../../data/2026-11-16_r167_keepalive_depth/README.md)).
- Ornith's taper: a KLD of 0.047, so opt-in ([FN93c](../../data/2026-11-16_fn93c_ornith_taper_kld/README.md)).

Stale or blind:
- **The taper's head rule lost** 5.60, 7.03 and 1.43 s from 512 tokens up
  ([R60](../../data/2026-11-12_r60_taper_v4/README.md)), on a line with 1,547 reallocations in 373 requests.
- **Text reuse was "negligible"** at 2.59 against 2.60 minutes at equal work while reading 2,982 prompt tokens a
  task against 5,847 ([v3.0's run](../../data/2026-11-06_flint_side_rig2/README.md)): equal work holds the tokens
  fixed.

## Candidates for v5

Worth is est. % of that model's prompt time, then of an agent task. Cost is code, then rig time.

| # | Row | Model | What | Evidence | Worth | Cost | Needs | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | Q260 | Flint, Ornith | `LLAMA_SERVER_TEXT_REUSE=1` in the presets | measured: re-reads take 67.6 of 222.6 s and 141.7 of 299.5 s (Flint Q8), 30.6 of 396.6 s and 71.9 of 467.8 s (Ornith) | est. Flint 29.1, 44.2 %; 3.8, 9.5 %. Ornith 7.6, 15.0 %; 1.7, 3.2 % | none; 3 h a model | row 2 | first |
| 2 | NEW | all | Agent tables count prompt tokens read, reply tokens re-read, the longest wait | equal work missed row 1 | makes rows 1, 3, 4 classable | small; none | — | first |
| 3 | NEW | Flint | Why its graph reallocates in agent requests, then the fix | measured: 250 and 290 a run (v3.0: 188, 237), 0 in the deep lane; the same fault cost Flash-Next 192.99 against 112.12 s | est. up to half of the 512 to 1,999 buckets: 19.5, 17.0 %; 3.6 % | none, 20 min; fix small to large | — | first |
| 4 | Q775, Q777, Q1235 | Flash-Next, Ornith | Calls up to 2,048 tokens in micro-batches of 128 or 64 | est.: the fixed costs above | est. ceiling: Flash-Next 24.0, 29.3 %; 4.7, 4.5 %. Ornith 36.1, 32.3 %; 8.2, 6.8 % | none; 1 h replay, an agent pair | Ornith: row 5 | first |
| 5 | FN93d | Ornith | The taper against the Q8_0 reference, by chunk | measured: prompts +9.2 %, a task −2.4 %; Q1127: builds 0.0217 apart were −0.00068 ± 0.00069 against it | as measured | none; 1 h | base logits on a rig | first |
| 6 | Q307, Q351, FN93e | Ornith | N2P's prompt kernel exact, its tiles tuned | est., below; Q189: prompts ~16 % slower than UD-Q2_K_XL's | est. up to 13.8 %; 3.0 % | large; 2 h | row 5 | next |
| 7 | Q1418 | Flash-Next | Name the 444.1 to 559.1 ms before a slot's first evaluation | measured 2026-11-12 ([FN85](../../data/2026-11-12_fn85_server_gap/README.md)) | est. 0.5 s a request: 12.0, 12.1 %; 2.4, 1.9 % | reading; none | — | first |
| 8 | Q1215 again | Flash-Next | The draft's share of reading on v4.0 | measured on v3.0's line: 101.4, 101.8 tok/s against 145.4, 145.2 without it ([R76](../../data/2026-11-11_r76_c620_p106/README.md)) | est. up to 30.3 %; 6 % | none; 40 min | — | next |
| 9 | C830, R167c | Flint, Ornith | The keep-alive during prompt chunks ([R137](../../data/2026-11-13_r137_ka250/README.md)) | measured: −16.5 % (Flint), −2.4 % with its eager stop (Ornith; [R169](../../data/2026-11-16_r169_ornith_no_eager/README.md)) | Ornith est. 2.4 %; 0.5 % | small; 1 h 40 min | — | later |
| 10 | Q1450, Q1368 | all | Prompt graphs replayed, not captured anew | est. 0.2 s of a 4,097-token prompt (Q1368), which takes 24.9 and 24.3 s | est. 0.8 % of a long prompt | done; 1 h 40 min | — | later |

Row 1's worth is each run's re-read seconds times the reply's share of those turns' tokens. Rows overlap (1 and 3
on Flint; 4, 7 and 8 on Flash-Next) and do not add up.

**Ornith's taper (rows 5 and 6).** FN93d settles the direction: d = KLD(taper) − KLD(plain) by chunk, worse only if
d − 2 se > 0. Could a taper leave the arithmetic alone? Est. yes. Ornith uses 8 of 256 experts, so a 512-token
micro-batch gives an expert about 16 tokens and a 64-token one about 2. Above 8 tokens a product takes the prompt
kernel, which rounds N2P to one scale per 32 weights where the format has one per 8 (Q307); at 8 or fewer the
few-token kernels use the format's own. Q307 is parked until "a prompt-processing quality check shows it"; FN93c is
one. With that kernel exact, both sizes compute the same products.

## Rows to close, merge or re-rank

- R106 (Q1351, why the n-gram prefetch lost, +2.09 %): park; est. under 4 % of a cold long prompt.
- C819 (Q1431, no swap from a load's page cache): park; the release runs read ten common tasks at 140.4 and 139.9
  tok/s with swap 21 → 21 and 36 → 404 MiB.
- C11 (Q264), Q363, short prompts: close; the short taper is in the presets.
- C12 (Q362, parallel checkpoint copies): close; restores take 2 ms.
- R30, S27, C154r, R96 (Q549, Q609, Q704, Q1337, host memory): close; the checkpoints left the host.
- R66, R67 (Q1247, Q1249, the sparse prompt path): close; its rule is in the preset. R65, R68: park.
- R62 (Q1239, the launch queue), R93 (Q1327, in-call checkpoints with the overlap): close; R123b and R123a ran them.
- R61 (Q1236, fewer tail checkpoints): into row 1. C642 (Q1128, `-ub 256`): into row 4.
- C13 (Q361), C15 (Q430), the card mask: park; the mask is off.
- R98, FN87, R151 (Q1341, Q1428, Q1457): park; diagnostics with no decision behind them.
- C14 (Q364, text reuse upstream): after row 1. C832 (Q1464, the cards' clocks): rides on row 4's replay.
- C10, C668 (the screened head, a reflection-word bias): not prompt path.

## New ideas

- Count the work, not only the speed, in every agent table (row 2): equal work hid a halving of prompt tokens.
- Trace Flint's reallocations by cause (row 3): v3.0 has them too.
- An FN93e arm with the expert products forced onto one kernel: row 6's cause in one load.
- Flash-Next: emit the first token before the draft's last catch-up; it needs only the main model.
- Stop the keep-alive only for prompt calls over a size: Flint keeps its gain on short prompts (+27.3 % at 64
  tokens, R165).

## What I could not settle from the files

- **Flash-Next's re-reads in seconds:** its per-request rows are not published (its reports count 33
  re-reads beside 544 exact matches over nine arms).
- **Cards busy on v4.0:** 1.416 of nine on v3.0 ([R42](../../data/2026-11-10_prompt_timeline/README.md)), 1.711 and
  1.759 with prompt graphs ([R82](../../data/2026-11-11_r82_c702_p106/README.md)); not counted since, never for Flint
  or Ornith.
- **Why Flint reallocates, and why the keep-alive costs its deep prompts:** no trace exists.
- **Whether text reuse removes every re-read on v4.0:** row 1 assumes it, as on v2.0 (Q260).
- **What the fixed 4.3 s and 7 s hold:** they are fits. [R81](../../data/2026-11-11_r81_ubatch_trace/README.md)
  traced 291.3 ms of host time before card 0 on a cold 512-token micro-batch on v3.0's line;
  [the prefill note](flash_next_prefill.md) prices the fill.
