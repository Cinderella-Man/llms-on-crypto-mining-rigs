# Q384: the heads offline on the agent check's own text

The six recorded agent-check traces of the paired replay (`replay.sh`'s `TRACES`: three JavaScript and three Python
Exercism bug fixes, 36 requests, 27,284 reply tokens), turned into rows by `agent_tokens.py --traces --id-base 900000`
(Flint's chat template; the server generated 27,321), dumped like the training data (`llama-staircase-dump`, the 26-layer
grid, 232,326 tokens) and scored by `score.py` at the nine-card windows: g9 (the head right after card 1's stage) and
g9miss (the first guess after a miss). Metric: the served sampler's inverse-CDF coupling (`uniform`), 95 % intervals by
request. α is the sharpening of the head's distribution: the engine applies the GGUF's α 3; `score.py`'s default is 1.
"Head positions" are the positions the head takes in the server's order (`--pipe-guess staircase`): those where neither
the copy lookup nor a 2-token continuation guesses (`source_split.py`'s cont1 and none).

## Served against offline, the head that was served (nine-card, 1,000 minutes)

| | Head positions, g9 | Head positions, g9miss |
|---|---|---|
| offline, α 1 | 0.256 (0.231–0.307) | 0.234 (0.210–0.281) |
| offline, α 3 (as served) | **0.309** (0.283–0.361) | **0.287** (0.262–0.333) |
| served on the rig (paired replay, Q193) | 0.248 (1,761 of 7,104 guesses right) | |

The same head on the held-out sessions of agentic run 1 at α 1: 0.314 at the 1-token continuations' positions, where the
agent check's text gives 0.243.

- **The text:** at α 1 the agent check's requests score 0.07 below the held-out sessions at the same kind of position
  (0.243 against 0.314 at cont1). Not because of the language: the three JavaScript traces, a language the training
  sessions do not have, score highest (table below). One Python trace, scale-generator, holds half of the head's
  positions (long reasoning) and scores lowest.
- **The engine:** at α 3, as served, the head is right 0.287–0.309 offline at its positions and 0.248 on the rig: 0.04–0.06
  are lost in serving. Candidates: the engine keeps the head's top 16 ids, the answer store takes some of the positions,
  the served placement (split 6/6/5/5/6/5/5/6/5) differs from the even split of g9, guesses that build on unverified
  guesses, and the P106's numerics. Q195 measures them.
- **Sharpening:** α 3 adds +0.05 at the head's positions (0.256 → 0.309), as on validation.
- **Longer training:** the 1,480-minute nine-card head adds +0.005 here (0.314 at g9, α 3).

## Where the head is right (served head, g9, α 3, head positions)

| Split | Positions | Head right |
|---|---|---|
| reasoning | 8,542 (87 %) | 0.286 |
| answer | 856 (9 %) | 0.358 |
| tool call | 386 (4 %) | 0.700 |
| Flint's top probability 0.99–1 | 4,172 (43 %) | cont1 0.392, none 0.548 |
| Flint's top probability 0.5–0.7 | 1,626 (17 %) | cont1 0.206, none 0.228 |
| context 0–4k tokens | 1,808 (18 %) | 0.348 |
| context 4k–8k tokens | 4,531 (46 %) | 0.320 |
| context 8k–16k tokens | 2,435 (25 %) | 0.270 |
| context 16k–32k tokens | 1,010 (10 %) | 0.280 |

| Trace | Head positions | Head right, g9 | g9miss |
|---|---|---|---|
| javascript-complex-numbers | 637 (7 %) | 0.369 | 0.335 |
| javascript-phone-number | 606 (6 %) | 0.398 | 0.368 |
| javascript-space-age | 522 (5 %) | 0.371 | 0.334 |
| python-affine-cipher | 1,116 (11 %) | 0.329 | 0.310 |
| python-list-ops | 1,854 (19 %) | 0.322 | 0.300 |
| python-scale-generator | 5,049 (52 %) | 0.275 | 0.256 |

Reasoning is most of what the head guesses, and it is right on under a third of it. Where Flint is nearly certain (top
probability above 0.99, 43 % of the head's positions) the head is right 0.39–0.55 of the time: it misses tokens the target
model finds obvious. Acceptance falls as the context grows, and the trace with the most reasoning scores lowest. Each
request is one row here (the prompt is the whole conversation so far), so the request-number table in the breakdowns has
a single row.

## All heads

Coupled acceptance at g9 over all scored positions, all / fresh text:

| Head | α 1 | α 3 |
|---|---|---|
| nine-card, 1,000 minutes (served) | 0.296 / 0.263 | 0.353 / 0.322 |
| 3–12-card, 1,000 minutes | 0.280 / 0.251 | 0.337 / 0.310 |
| nine-card, 1,480 minutes | 0.309 / 0.276 | 0.358 / 0.328 |
| 3–12-card, 1,480 minutes | 0.292 / 0.263 | 0.345 / 0.319 |

Files in `q384/`: `scores-<head>[-a3].md` (score.py), `split-<head>[-a3].md` (source_split.py, with the confidence-gate
table), `breakdown-<head>[-a3].md` (breakdown.py: by source, segment, Flint's top probability, context length, source ×
top probability, both windows), `rows.txt` and `dump.txt` (the counts). `$RUN` is the training run's folder, `$WORK` the
work folder.
