# Q167 (E1): Ornith-1.5-122B's N2P/N3P mix allocated per 8-layer group (2026-10-27/28)

The data behind registry row Q167 in [quants/experiments.md](../../../quants/experiments.md) and E1 of
[slq.md](../../research/notes/slq.md): SLQ's allocator applied to Miner Llama's 2- and 3-bit formats
([pascal_lowbit_format.md](../../research/notes/pascal_lowbit_format.md)).

- **Start:** Q164's file c (routed experts' gate/up N2P, down N3P, every other tensor Q5_0, attention k/v, output head
  and token embedding Q8_0).
- **Switches:** each moves one class of one 8-layer group to another file's format. Expert gate/up go to N3P and
  expert down to N2P (from file d, c's recipe with the expert formats swapped); everything else in the group goes to
  Q8_0 (from file b, c's experts with every other tensor Q8_0).
- **Scoring:** every one of the 18 switches was scored alone against the Q8_0's three KLD bases.
- **Allocation:** the costs were added, and every one of the 2^18 sets was tried at each byte budget. The pick at file
  c's bytes, and the pick at +0.9 GB, were then composed and scored as files.

No speed was measured here: per weight, N2P and N3P decode in about the same time on the P106, so the mix changes only
quality and bytes.

## Result

KLD against Ornith-1.5-122B Q8_0 on two V100s:
- wikitext-2: 80 × 512 tokens;
- held-out agent sessions: 12 × 8,192 tokens;
- the agent check's Exercism code: 80 × 512 tokens.

| File | Bytes on the cards | File | wiki KLD | agent KLD | code KLD | wiki same top |
|---|---|---|---|---|---|---|
| AD-IQ3_XXS-IQ2_S (published) | — | 47.64 GB | 0.2332 | 0.0657 | 0.0807 | 81.43 % |
| UD-Q2_K_XL (published) | — | 44.53 GB | 0.1843 | 0.0541 | 0.0713 | 83.55 % |
| file b: c's experts, every other tensor Q8_0 | — | 49.31 GB | 0.1951 | 0.0545 | 0.0701 | 83.18 % |
| file c: gate/up N2P, down N3P, dense Q5_0 | 45.81 GB | 47.58 GB | 0.2033 ± 0.0030 | 0.0572 ± 0.0012 | 0.0721 ± 0.0028 | 82.85 % |
| **valid0**: c with gate/up N3P in layers 24–39, down N2P in layers 0–31 (predicted 0.1860 / 0.0566 / 0.0600) | **45.81 GB** | 47.58 GB | **0.1897 ± 0.0029** | **0.0550 ± 0.0011** | **0.0646 ± 0.0026** | 83.53 % |
| **valid1**: valid0 + gate/up N3P in layers 40–47 (predicted 0.1695 / 0.0538 / 0.0552) | 46.72 GB | 48.48 GB | **0.1747 ± 0.0028** | **0.0513 ± 0.0010** | **0.0613 ± 0.0026** | 84.28 % |

Differences below are in standard errors of a difference between two files, √(SE₁² + SE₂²). That is conservative:
both files are scored on the same tokens, so the paired noise is smaller.

- **At file c's bytes, the allocation beats c:**
  - wikitext by 0.0136 (−6.7 %, 3.2 standard errors);
  - code by 0.0075 (−10 %, 2.0);
  - agent text by 0.0022 (−3.9 %, 1.4, within noise).
  - Same top token is +0.7 points on wikitext (1.9).
- **The costs add roughly.** On wikitext and code, each validation file lands 0.004–0.006 worse than its prediction
  (1.3–2.3 of one file's standard errors). On agent text it lands 0.002 better than predicted.
- **Against UD-Q2_K_XL:**
  - valid0 is level: wikitext +0.005, agent +0.001, code −0.007 (0.6–1.8 standard errors either way). It is 3.05 GB
    larger, in formats est. ~3× faster per expert weight on the P106 (Q164).
  - valid1 is better than UD on all three texts: −0.0096, −0.0028 and −0.0100 (2.5, 1.8 and 2.6 standard errors).
    It is 3.95 GB larger.
- **Ornith's sensitive layers are the late ones, the reverse of Flint (Q168):**
  - Gate/up to N3P lowers wikitext KLD by 0.016–0.018 per group in layers 24–47, by 0.007 in 16–23, and by under
    0.001 in 0–15.
  - Down to N2P costs +0.009 per group in layers 24–47 and 0.0005–0.004 in 0–23.
  - The dense tensors at Q8_0 move KLD by at most ~one standard error per group (+0.28 GB each). So file b, which
    spends its extra 1.7 GB there, is worse than valid0 on wikitext and code.

Cost of each switch (KLD change against file c; bytes on the cards):

| Group (layers) | 0–7 | 8–15 | 16–23 | 24–31 | 32–39 | 40–47 |
|---|---|---|---|---|---|---|
| gate/up N2P → N3P (+0.906 GB), wiki / agent / code | −0.0002 / +0.0004 / −0.0019 | −0.0008 / −0.0004 / −0.0038 | −0.0067 / −0.0028 / −0.0060 | −0.0156 / −0.0033 / −0.0072 | −0.0178 / −0.0032 / −0.0076 | −0.0165 / −0.0028 / −0.0048 |
| down N3P → N2P (−0.453 GB) | +0.0023 / +0.0007 / +0.0004 | +0.0005 / +0.0011 / +0.0009 | +0.0043 / +0.0015 / +0.0000 | +0.0090 / +0.0026 / +0.0013 | +0.0099 / +0.0021 / +0.0033 | +0.0091 / +0.0018 / +0.0024 |
| dense Q5_0 → Q8_0 (+0.284 GB) | +0.0012 / −0.0001 / −0.0007 | +0.0002 / +0.0002 / +0.0001 | −0.0004 / −0.0003 / −0.0013 | −0.0019 / −0.0001 / +0.0004 | −0.0021 / −0.0013 / −0.0007 | −0.0027 / −0.0009 / +0.0005 |

The knapsack's picks at other budgets (`solve.tsv`; predicted wiki / agent / code):

| Budget | Switches | Predicted |
|---|---|---|
| +1 GB | gate/up N3P in 24–47, down N2P in 0–31 (= valid1) | 0.170 / 0.054 / 0.055 |
| −1 GB | gate/up N3P in 32–39, down N2P in 0–31 and 40–47, dense Q8_0 in 32–39 | 0.209 / 0.060 / 0.069 |
| −2 GB | down N2P in 0–31 and 40–47 | 0.229 / 0.065 / 0.077 |

UD-Q2_K_XL's 44.53 GB is not reachable with these switches: all six down groups at N2P free 2.72 GB, leaving c at
44.86 GB.

## Files

| File | What |
|---|---|
| `deltas.tsv` | per switch: tensors and the bytes it adds on the cards (from the three files' headers); file c's bytes on the cards (token embedding and the MTP block excluded) |
| `kld.tsv` | mean KLD ± SE and same top token ± SE on each text for file c, the 18 switches and the two validation files |
| `solve.tsv` | the best set of switches at each budget (bytes on the cards relative to file c) with its predicted KLD per text |
| `solve-switches.tsv` | each switch's bytes, KLD change per text and damage (the sum of relative KLD changes over the three texts) |
| `run/run.sh` | the runs: stage (file c onto the fast disk), push, deltas, score (detached), collect, solve, fetch |
| `run/score.sh` | compose each item from file c with the switched groups from b or d, score it on the three bases, delete it |
| `run/alloc.py` | the switches, composing, byte deltas, log parsing and the exhaustive search |

## How it was made

- **Files:** Q164's files b, c and d ([quantize.sh](../2026-10-27_lowbit/run/quantize.sh) `FILE=b|c|d`: llama-quantize
  of fork branch `pascal-p106-v3-lowbit` from ornith-ai's Q8_0 with Unsloth's importance matrix, `--pure`). Every
  composed tensor is byte for byte from one of them ([scripts/compose_gguf.py](../../../scripts/compose_gguf.py), with
  that branch's gguf-py, which knows N2P and N3P).
- **KLD:** the same branch's llama-perplexity (build `5fd8ffeec`, CUDA), `-c 512 -b 512 -ub 512 --chunks 80` for
  wikitext and code and `-c 8192 -b 8192 -ub 512 --chunks 12` for the agent text, `-ngl 99 -fa on`, against the
  Q8_0's bases from [the Ornith benchmark](../../benchmarks/2026-10-27-ornith-122b.md), on two V100s. File c's own
  scores are the same build's on the same GPUs.
- **Groups:** layers 0–7, 8–15, …, 40–47; the MTP block (48) and the token embedding stay as in file c.
- **Licences:** Ornith-1.5 is MIT; Unsloth's importance matrix is Apache-2.0; the scores and scripts are this
  project's.

Commands (with `QM`, `WORK` (scratch on a fast disk), `SRC_C`, `FILE_C`, `FILE_B`, `FILE_D`, `KLD` (the bases and
texts), `WIKI`, `BIN`, `GGUFPY`, `GPUS`, `C_LOGS` (file c's three logs) set):

```bash
run/run.sh stage && run/run.sh push && run/run.sh deltas
PREFETCH=0 ITEMS="dense0 dense1 dense2 dense3 dense4 dense5" TAG=dense run/run.sh score
PREFETCH=0 ITEMS="gu0 dn0 gu1 dn1 gu2 dn2 gu3 dn3 gu4 dn4 gu5 dn5" TAG=experts run/run.sh score
run/run.sh collect && run/run.sh solve
PREFETCH=0 ITEMS="valid0=gu3,gu4,dn0,dn1,dn2,dn3 valid1=gu3,gu4,gu5,dn0,dn1,dn2,dn3" TAG=valid run/run.sh score
EXTRA_NAMES="valid0 valid1" run/run.sh collect && run/run.sh fetch
```
