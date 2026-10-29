# Q385: where the head is right on the held-out sessions

`score.py --only-windows g9,g9miss` on the 40 held-out sessions of agentic run 1 (`--spans`: generated tokens only,
681,208 positions; 680,600 have a guess source), the nine-card heads after 1,000 and 1,480 minutes, at α 1 and at
α 3 (the engine's sharpening), then `breakdown.py`. Metric: the served sampler's inverse-CDF coupling (`uniform`); 95 %
intervals by session. "Head positions": where neither the copy lookup nor a 2-token continuation guesses (cont1 and
none), the positions the head takes in the server's order. Overlap: Σ min(p, q), the most any coupling of the two
distributions could match.

## From held-out to served (the served head: nine-card, 1,000 minutes; g9)

| | All positions | Head positions | Reasoning at the head's positions |
|---|---|---|---|
| held-out, α 1 | 0.362 | 0.322 | 0.293 |
| held-out, α 3 | 0.408 | **0.384** | 0.357 |
| agent check's text, α 3 (Q384) | 0.340 | **0.309** | 0.286 |
| served on the rig (Q193) | | **0.248** | |

- **Text:** −0.075 from the held-out sessions to the agent check's text at the head's positions (α 3). Reasoning is 85 %
  of the head's positions in both (84.6 % and 87.3 %); the check's reasoning is harder (0.286 against 0.357), and the
  check has few tool-call tokens (5 % of its positions against 32 %), where the head does best.
- **Serving:** −0.04–0.06 from the check's text offline to the rig (Q384, Q195).
- **Sharpening:** α 3 adds +0.06 at the head's positions (0.322 → 0.384).
- **Coupling:** at the head's positions the coupled match (0.384) is 93 % of the overlap (0.411): the sampler's coupling
  leaves little; the head's distribution is what is off.
- **Longer training:** the 1,480-minute head is +0.009 at the head's positions (0.393, α 3) and +0.012 at α 1 (0.334).

## Breakdowns (served head, g9, α 3, head positions unless said)

| Split | Positions | Head right | Lookup right |
|---|---|---|---|
| source: copy (all positions) | 354,878 (52 %) | 0.429 | 0.815 |
| source: cont2 (all positions) | 167,738 (25 %) | 0.388 | 0.434 |
| source: cont1 | 128,194 (19 %) | 0.375 | 0.191 |
| source: none | 29,790 (4 %) | 0.422 | 0 |
| reasoning | 133,646 (85 %) | 0.357 | |
| answer | 9,134 (6 %) | 0.446 | |
| tool call | 15,204 (10 %) | 0.586 | |
| Flint's top probability 0.99–1 | 76,940 (49 %) | 0.520 | |
| 0.9–0.99 | 13,413 (8 %) | 0.366 | |
| 0.7–0.9 | 26,055 (16 %) | 0.294 | |
| 0.5–0.7 | 24,956 (16 %) | 0.224 | |
| under 0.5 | 16,620 (11 %) | 0.147 | |
| context 0–4k tokens | 3,473 (2 %) | 0.622 | |
| 4k–16k | 25,625 (16 %) | 0.414 | |
| 16k–32k | 80,473 (51 %) | 0.363 | |
| 32k–64k | 48,413 (31 %) | 0.386 | |
| request 1 of the session | 1,379 (1 %) | 0.866 | |
| requests 2–20 | 133,618 (85 %) | 0.369–0.393 | |
| requests over 20 | 22,987 (15 %) | 0.371 | |

- **Where Flint is sure the head is not:** half the head's positions have Flint's top probability above 0.99, and the
  head is right on 0.52 of them. On the copy positions with Flint above 0.99 the copy lookup is right 0.880, the head
  0.448: it does not retrieve from the context what the target model finds obvious.
- **After a miss** (g9miss) the head is 0.021 lower at its positions (0.363 against 0.384; the 1,480-minute head 0.378
  against 0.393).
- **Session position:** past the first request (0.866) acceptance is flat across requests; by context it falls from 0.62
  under 4k tokens to 0.36–0.39 from 16k on.
- **Where training spends its loss:** the head's positions are 23 % of the generated positions (Q372 weights them).

Files in `q385/`: `scores-<head>[-a3].md` (score.py at g9 and g9miss), `breakdown-<head>[-a3].md` (every table, both
windows, with intervals and overlap). `$RUN` is the training run's folder, `$WORK` the work folder.
