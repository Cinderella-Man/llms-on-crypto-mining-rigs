# Q168 (E3): a Flint file within noise of Q8_0 at fewer bytes (2026-10-27)

The data behind registry row Q168 in [quants/experiments.md](../../../quants/experiments.md) and item 3 of "what to
borrow" in [slq.md](../../research/notes/slq.md). The allocator of SLQ, applied to this project's fast formats. The
starting file is Unsloth's Flint Q8_0 with Flint-Pascal-Q5's F16 routers ("q8r"). Each "switch" takes one tensor
class of one layer group, or the output head, or the token embedding, from Flint-Pascal-Q5 (Q5_0 where that file has
it). Every switch was scored alone. The costs were added, and dynamic programming over the bytes each switch saves
per token found the least predicted KLD at each size. Three picks were then composed and scored as files. No speed
was measured here; speed is only measured on the rig.

## Result

KLD against BF16, 80 × 512 tokens of wikitext-2, one GPU set (two RTX 3080 and an RTX 3090), the rescale on:

| File | GB read per token | vs Q8_0 | File GB | KLD (predicted) | KLD measured | Same top token |
|---|---|---|---|---|---|---|
| Unsloth Q8_0 (BF16 routers) | 2.696 | — | 42.30 | — | 0.0627 ± 0.0032 | 92.39 ± 0.19 % |
| q8r: Q8_0 with Pascal-Q5's F16 routers | 2.696 | 0 % | 42.30 | — | 0.0580 ± 0.0029 | 92.81 ± 0.18 % |
| **p061**: q8r + attention q in 33–48, attention output in 41–48, the shared expert in 33–40, the Q5_0 head | 2.487 | −7.8 % | 42.09 | 0.0610 | **0.0611 ± 0.0028** | 92.14 ± 0.19 % |
| **p065**: p061 + attention k/v in 33–48, attention output in 33–40, the shared expert in 41–48, experts' gate/up in 41–48 | 2.400 | −11.0 % | 40.43 | 0.0645 | **0.0656 ± 0.0029** | 91.78 ± 0.19 % |
| **p068**: p065 + attention q in 25–32, experts' gate/up in 33–40 | 2.337 | −13.3 % | 38.79 | 0.0678 | **0.0684 ± 0.0028** | 91.02 ± 0.20 % |
| Flint-Pascal-Q5 (all 23 switches), for reference | 2.090 | −22.5 % | 29.39 | 0.129 | 0.1108 ± 0.0037 | 88.44 ± 0.22 % |

- **The costs add near Q8_0.** The three composed files land within 0.0011 of their predictions.
- **p061 is within noise of Q8_0** on both measures, reading 7.8 % fewer bytes per token.
- **p065** is within two standard errors on KLD (+0.0029). Its same top token is 0.6 points lower, a little over two
  standard errors of a difference between two files (~0.27 points).
- **p068** is at the edge on KLD (+0.0057) and 1.4 points lower on same top token: beyond noise. It sits at the 91.3 %
  same-token level that Q21 measured between two GPU types running the same file.
- **The sum of all 23 costs over-predicts Pascal-Q5** (0.129 against 0.1108 measured on this GPU set): far from Q8_0
  the costs add to more than the whole, as Q77 found near Flint's bar. So the dynamic program is exact enough near
  Q8_0 and pessimistic far from it.

**On agent text** (12 × 8,192 tokens of held-out agent sessions; KLD against Q8_0's own logits, same GPU set):

| File | KLD | Same top token |
|---|---|---|
| q8r (Q8_0 with F16 routers) | 0.2273 ± 0.0056 | 93.78 ± 0.11 % |
| p061 | 0.2281 ± 0.0056 | 93.56 ± 0.11 % |
| p065 | 0.2292 ± 0.0056 | 93.42 ± 0.11 % |

- On long agent text the routers' type alone, BF16 in Unsloth's file against F16 in q8r, moves KLD by 0.227. This
  is the routing-flip divergence of Q21, which grows with context.
- p061 and p065 add 0.0008 and 0.0018 on top of it (0.1 and 0.2 standard errors of a difference).
- Their same top token is 0.22 and 0.37 points below q8r's (1.4 and 2.4 standard errors of a difference, ~0.155
  points).
- Both are as close to Q8_0 as Q8_0 with F16 routers is. p065's same top token is again at the edge of noise.

## On the rig (2026-10-28)

The rig: nine P106-090 cards. The build: Miner Llama v2.0 (`37e717ff6`), release line `-b 512 -ub 128
-ts 6,6,5,5,6,5,5,6,6 --pipe-depth 4`, 262,144 tokens of context. Tests through
[bench/agent_check/release_check.sh](../../../bench/agent_check/release_check.sh) (`run/rig_test.sh`).

**Paired replay.** The six recorded agent-check traces of the answer store's control arm were replayed to Q8_0 and then
to p065. The prompts and seeds were the same, and replies were capped at their recorded lengths. 36 requests per arm;
45,093 and 45,091 prompt tokens processed; 20,544 and 21,196 tokens generated (the files differ, so the replies do
too). `replay.md`, ratios to Q8_0 with 95 % intervals over traces:

| | ms per generated token | ms per prompt token | minutes for the same work |
|---|---|---|---|
| Q8_0 | 22.92 | 5.14 | 11.8 |
| p065 | 21.58 | 5.15 | 11.4 |
| **p065 / Q8_0** | **0.941 (0.930–0.980)** | 1.002 (0.997–1.008) | 0.963 (0.950–0.992) |

- **p065 decodes 6.3 % faster than Q8_0** (1 / 0.941) for 11.0 % fewer bytes per token: weight reading is a little
  over half of a decoded token's time on these cards.
- Prompt processing is unchanged.
- Agent sessions take 3.7 % less time for the same work, since decode is part of each session and prompts the rest.

**Live agent check of p065** (12 bug-fix tasks with real tool runs, `live-table.md`): 12 / 12 passed, 47.1 tok/s
generation, 25.9 minutes for all tasks, 0 malformed of 91 tool calls, no runaways. Q8_0 on the same build and settings
has scored 12 / 12 at 46.5 tok/s and 45.0 tok/s in earlier live runs (registry Q201's same-slot control and the v2.0
release).
Live runs vary by about ±4 %, which is why the paired replay carries the speed result.

**What the switches cost** (KLD change against q8r; scores in `kld.tsv`, bytes in `deltas.tsv`):

| Group (layers) | 0–8 | 9–16 | 17–24 | 25–32 | 33–40 | 41–48 |
|---|---|---|---|---|---|---|
| attention q (Q5_0 in Pascal-Q5 except 9–24) | +0.0062 | — | — | +0.0014 | −0.0000 | −0.0001 |
| attention k/v | — | — | — | — | +0.0003 | +0.0002 |
| attention output | — | — | — | — | +0.0006 | +0.0006 |
| shared expert | — | — | — | — | +0.0001 | +0.0006 |
| experts' gate/up (Q8_0 in Pascal-Q5 in 2–8) | — | +0.0152 | +0.0061 | +0.0042 | +0.0020 | +0.0018 |
| experts' down (Q8_0 in Pascal-Q5 in 46–48) | +0.0118 | +0.0095 | +0.0038 | +0.0019 | +0.0018 | +0.0006 |

Output head: +0.0023 for 115 MB per token. Token embedding: +0.0120, and it saves no bytes per token, so it stays
Q8_0. The early layers carry the cost, as Q36 found for 4-bit; the late layers' attention, shared expert and experts
are nearly free. The routers' type alone moves KLD by 0.005 (Q8_0 against q8r), the routing-flip noise Q113 described,
so single costs under ~0.002 are within that noise.

## Files

| File | What |
|---|---|
| `deltas.tsv` | per switch: tensors, file bytes and bytes read per token it changes (experts count 6 of 256, `token_cost.py`'s rule; the embedding none), and q8r's totals |
| `kld.tsv` | every scored file: KLD ± SE, same top token ± SE, PPL ratio, 99th-percentile KLD (Q8_0, q8r, 23 switches, 3 picks and Pascal-Q5 against BF16 on wikitext; `*.agent` rows against Q8_0 on agent text) |
| `solve.tsv` | the frontier over bytes read per token: for each saving (MB), the least predicted KLD and its switches |
| `solve-file.tsv` | the same over file bytes |
| `solve-picks.tsv` | the picks at KLD ≤ q8r + 0.003 / 0.0065 / 0.010 and at files ≤ 33 / 36 / 39 GB |
| `rig/replay.md` | the paired replay's summary: per arm the cost per prompt and generated token, ratios to Q8_0 with 95 % intervals |
| `rig/replay.jsonl` | every replayed request: trace, index, sampling, prompt and cache tokens, generated tokens, timings (no text) |
| `rig/live-table.md`, `rig/live-attempts.csv`, `rig/live-turns.csv` | the live agent check of p065: the summary row, one row per task, one row per turn |
| `run/rig_test.sh` | the rig run: the paired replay (Q8_0, then p065) and the live arm |
| `run/run.sh` | the runs: push, deltas, agent_base, score (detached), collect, solve, fetch, relay |
| `run/score.sh` | compose each item from the base and donor, score it with the flags of every Flint mix score, delete it |
| `run/near_q8.py` | the switches, composing, byte deltas, log parsing and the dynamic program |

## How it was made

- **Files:** Unsloth's `flint-Q8_0.gguf` (`unsloth/flint-GGUF`) and this project's Flint-Pascal-Q5 (= K2h-r16,
  [release recipe](../../../release/flint-v2/reproduce/model.md)). Every composed tensor is byte for byte from one of
  them ([scripts/compose_gguf.py](../../../scripts/compose_gguf.py)).
- **KLD:** llama.cpp `d77256999` with CUDA, `llama-perplexity -c 512 --chunks 80 -ngl 99 -fa on --kl-divergence`
  against Flint BF16's logits on wikitext-2 test (the reference of
  [scripts/quality_kld.sh](../../../scripts/quality_kld.sh)), the MoE rescale on (llama.cpp's default), on two
  RTX 3080 and one RTX 3090 of a separate machine. Every number in the tables comes from this one GPU set; other sets
  move Flint's KLD by up to ~0.005 (Q36) and other GPU types by more (Q21).
- **Groups:** layers 0–8, 9–16, 17–24, 25–32, 33–40, 41–48 (Q36's). The routers are always Pascal-Q5's F16 (Q113).
- **Licences:** Flint is Apache-2.0; the scores and scripts are this project's.

Commands (with `QM` the quality machine's SSH alias, `WORK` a scratch folder on a fast disk there, `BASE`, `DONOR`,
`REF` (the folder with `wikitext-2-raw/wiki.test.raw` and `flint-BF16.c512x80.kld`), `BIN` (llama-perplexity),
`GGUFPY` and `GPUS` set):

```bash
run/run.sh push
run/run.sh deltas
ITEMS="q8_0@$BASE q8r= head q0 q3 q4 q5 o4 o5 kv4 kv5 sh4 sh5 embd dn0 dn1 dn2 dn3 dn4 dn5 gu1 gu2 gu3 gu4 gu5" TAG=night run/run.sh score
run/run.sh collect && run/run.sh solve
ITEMS="p061=q4,q5,o5,sh4,head p065=q4,q5,kv4,kv5,o4,o5,sh4,sh5,gu5,head p068=q3,q4,q5,kv4,kv5,o4,o5,sh4,sh5,gu4,gu5,head" TAG=picks run/run.sh score
run/run.sh agent_base                    # AGENT_TXT: held-out agent sessions (bench/agentic_data run 1), AGENT_BASE: output
TEXT=agent ITEMS="q8r= p061=q4,q5,o5,sh4,head p065=q4,q5,kv4,kv5,o4,o5,sh4,sh5,gu5,head" TAG=agent run/run.sh score
ITEMS="pascal-q5@$DONOR" TAG=anchor run/run.sh score
run/run.sh collect && run/run.sh fetch
```

The agent text is the held-out part of agentic data run 1 (Q165, Flint's own sessions on licence-checked code,
[2026-10-26_agentic_data](../2026-10-26_agentic_data/README.md)); the KLD base built from it (14.8 GB) is not
published.
