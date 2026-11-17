# The class tables count the work: 13 published tables rebuilt, R17a classed (C855, Q1526; 2026-11-17)

[bench/agent_check](../../../bench/agent_check/README.md)'s two class tools now carry the work a run did beside its
speed ([the section](../../../bench/agent_check/README.md#classing-runs-agenttsv-and-classtsv)).
`agent_tables.py` ends each run's row of `agent.tsv` with prompt tokens read a task, reply tokens re-read a task and
those turns' prompt seconds a task, the number of such turns, and the longest first-token wait. `class_arms.py`
classes the four work measures with the same rule as the time measures: a gain or a loss where the difference of the
means is beyond the larger of the two arms' spreads, negligible otherwise. This folder holds the check that every
older column and class line comes out the same, the new tables of 13 published tables, and the classes they give. No
model ran: the tables are arithmetic on per-request rows already in this repository.

Registry row Q1526 in [quants/experiments.md](../../../quants/experiments.md).

## Why

"Minutes a task at equal work" prices one fixed amount of work at each run's speeds. It shows a faster engine, but not
a change that leaves work out. Text reuse (`LLAMA_SERVER_TEXT_REUSE=1`) leaves work out: the server stops reading a
reply again when the reply comes back split into other tokens. C852 (Q1521) made the agent check's report count that
work ([2026-11-17_c852_agent_work_counts](../2026-11-17_c852_agent_work_counts/README.md)); the class tables still
rested on equal work and speeds alone, so text reuse had no class of its own.

## The check: every older column and class line the same

[run/check.sh](run/check.sh) rebuilds each published table whose rows are all published, from those rows, once with
the script that made it ([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)) and once with
[bench/agent_check/agent_tables.py](../../../bench/agent_check/agent_tables.py). Its output is [check.txt](check.txt).

| Table | The new table's first 14 columns against the old script's | The old script's table against the published one |
|---|---|---|
| [R159](../2026-11-14_r159_flint_release_agent/README.md) `agent.tsv` (7 runs, Flint Pascal Q8 GGUF) | the same | the same |
| [R17](../2026-11-17_r17_flint_textreuse/README.md) `agent.tsv` (4 runs) | the same | the same |
| [R160](../2026-11-14_r160_ornith_release_agent/README.md) `agent.tsv` (7 runs, Ornith-1.5-122B Pascal N2P GGUF), `agent_o3aref.tsv` (6) | the same | the same |
| [R166](../2026-11-16_r166_ornith_stack_no_taper/README.md) `agent.tsv` (9 runs) | the same | the same |
| [R169](../2026-11-16_r169_ornith_no_eager/README.md) `agent.tsv` (11 runs) | the same | the same |
| [R170](../2026-11-16_r170_flint_q5/README.md) `agent.tsv` (3 runs, Flint Pascal Q5 GGUF), `agent_ref_q5f3.tsv` (2) | the same | the same |
| [v3.0's text-reuse run](../2026-11-06_flint_side_rig2/README.md) `agent_t2d.tsv` (2 runs, Unsloth Q8_0) | the same | the same |
| [Ornith's MTP draft](../2026-11-06_ornith_mtp_rig2/README.md) `agent_part1.tsv`, `agent_part1_9tasks.tsv`, `agent_mtpadapt.tsv`, `agent_mtpadapt_11tasks.tsv` (`NMAX=2`, two with `SKIP_TASKS`) | the same | the same except `acceptance` and `tokens_per_step`, which come from server logs that are not published |

R159's class ([class.tsv](../2026-11-14_r159_flint_release_agent/class.tsv)): the old and the new `class_arms.py` on
the published `agent.tsv` give it byte for byte, and the new one on the new table gives its 13 lines first, then the
work measures' 8 lines. In every rebuilt run a task's count from `turns.csv` equals its attempts in `attempts.csv`, so
"a task" in the work columns is the same task as in equal work. The work cells equal `report.py work`'s on the same
rows: `f4ta` 48,907 prompt tokens over 12 tasks gives 4,076 a task, 14,632 reply tokens re-read 1,219, 67.6 s 5.63 s,
the longest wait 24.70 s ([C852's work.tsv](../2026-11-17_c852_agent_work_counts/work.tsv)).

The self-test's new step ([selftest.sh](../../../bench/agent_check/selftest.sh): a fixed table folder through both
tools) was run on its own and passes; the full self-test, which drives the agent in its sandbox, was not run.

## R17a: text reuse, classed on the work

[tables/r17_class.tsv](tables/r17_class.tsv): `f4tr` (R17a, one run, text reuse on) against `f4t` (R159e `f4ta` and
R159f `f4tb`, the tuned v4 preset without it), Flint Pascal Q8 GGUF. One run has no spread of its own, so each class
rests on `f4t`'s spread.

| Measure | `f4tr` | `f4ta`, `f4tb` | Difference of the means (`f4t`'s spread) | Class |
|---|---|---|---|---|
| minutes a task at equal work | 1.81 | 1.72, 1.81 | +0.045 (0.09) | negligible |
| generation tok/s | 46.74 | 49.32, 46.99 | −1.415 (2.33) | negligible |
| prompt tok/s | 210.0 | 219.7, 204.1 | −1.9 (15.6) | negligible |
| first token median s | 1.76 | 1.74, 1.82 | −0.02 (0.08) | negligible |
| first token p90 s | 5.31 | 6.61, 6.35 | −1.17 (0.26) | gain |
| first token on a task's first turn, median s | 3.21 | 3.24, 3.18 | 0 (0.06) | negligible |
| **prompt tokens read a task** | **2,970** | 4,076, 5,093 | −1,614 (1,017) | **gain** |
| **reply tokens re-read a task** | **0** | 1,219, 2,063 | −1,641 (844) | **gain** |
| **prompt s of the turns that re-read a reply, a task** | **0** | 5.63, 11.81 | −8.72 (6.18) | **gain** |
| longest first-token wait s | 11.68 | 24.70, 93.88 | −47.61 (69.18) | negligible |

- **Equal work says negligible; the work says gain.** The server read no reply a second time in 77 turns, against 4
  and 5 turns in the two runs without the switch, and read 2,970 prompt tokens a task against 4,076 and 5,093.
- **R17b had not landed when this ran,** so the arm is one run. With R17b's rows added to the table folder,
  `class_arms.py agent.tsv f4tr=<R17a's run>,<R17b's run> f4t=f4ta,f4tb` gives the class on two runs each.

## Two runs an arm are a narrow guard for the work

The agent sends no seed, so each run writes different replies and does different work. The spread of two runs of one
arm is often smaller than the difference chance makes between two arms:

| Arms (two runs each) | Prompt tokens read a task | Reply tokens re-read a task | Their prompt s a task | Longest wait s | Equal work |
|---|---|---|---|---|---|
| `f4` (v4-rc2's preset) against v3.0 ([r159_class_f4.tsv](tables/r159_class_f4.tsv)) | 6,007, 5,560 against 3,111, 4,099: loss | 3,064, 2,637 against 322, 1,372: loss | 17.64, 13.99 against 2.17, 7.19: loss | 88.86, 78.05 against 13.04, 55.47: loss | negligible |
| `f4t` against `f4` ([r159_class.tsv](tables/r159_class.tsv)) | 4,076, 5,093 against 6,007, 5,560: gain | 1,219, 2,063 against 3,064, 2,637: gain | 5.63, 11.81 against 17.64, 13.99: gain | negligible | negligible |
| `f4t` against v3.0 | negligible | negligible | negligible | negligible | negligible |

v4-rc2's preset and v3.0 draw exactly the same tokens for the same request and seed
([Q1488](../2026-11-16_c843_flint_sampling/README.md)), and `f4t` adds speed and memory switches to `f4`. So the
first two rows' work classes are chance, which two runs an arm cannot tell from a cause. A work gain counts when the
change has a cause that removes the tokens, as text reuse does (0 reply tokens re-read on every turn), or when it
holds on more runs. A wider band for the work measures is Q1575: the spread of every run that draws alike, and the
re-read turns counted as turns. An estimate of what that gives: the seven Flint Pascal Q8 GGUF runs above re-read a
reply on 41 of their 425 turns after a task's first; at that rate, R17a's 0 in 65 such turns has a chance of
(1 − 41/425)^65 ≈ 0.14 % (est.).

On Ornith ([r160_class.tsv](tables/r160_class.tsv), [r166_class.tsv](tables/r166_class.tsv),
[r169_class.tsv](tables/r169_class.tsv)) the stacked preset `o4s` against `o4` and v3.0 is negligible on prompt tokens
read, reply tokens re-read and their seconds, and a gain on the longest wait (28.09 and 28.50 s against 37.93 and
37.56, and 37.40 and 38.41), where it reads prompts at 114.5 and 113.7 tok/s against 56.6 and 65.4. `o4sn` against
`o4s` and `o4ne` against `o4sn` are negligible on all four work measures.

## Commands

From the repository's root, with Python 3 (no packages; on a laptop it takes a few seconds):

```bash
WORK=<scratch folder> bash docs/data/2026-11-17_c855_class_work_counts/run/check.sh > docs/data/2026-11-17_c855_class_work_counts/check.txt
```

It builds each table folder from published rows with
[R17's merge_rows.py](../2026-11-17_r17_flint_textreuse/run/merge_rows.py) (the reference runs: `v3p065` from
[flint-p065-v3base](../2026-11-05_release_regression/agent/flint-p065-v3base/), `v3ka` from
[ornith-ka](../2026-11-06_ornith_mtp_rig2/agent/ornith-ka/), `v3q5` from
[flint-pq5-v3base](../2026-11-05_release_regression/agent/flint-pq5-v3base/), each arm `rel` renamed), runs both
scripts with each table's published arguments, compares, and writes the new tables and classes to `tables/`.

## Files

| File | Content |
|---|---|
| [check.txt](check.txt) | the check's output: per table, new against old and old against published; R159's class three ways |
| [tables/](tables/) | the new `agent.tsv` of each rebuilt table (`<run>_agent*.tsv`: the old columns, then `prompt_read_per_task`, `reread_per_task`, `reread_s_per_task`, `reread_turns`, `ttft_max_s`) and the classes: `r159_class.tsv` (R159's class with the work measures), `r159_class_f4.tsv` (`f4` against v3.0), `r17_class.tsv` (R17a against `f4t`), `r160_class.tsv`, `r166_class.tsv`, `r169_class.tsv` (Ornith's pairs) |
| [run/check.sh](run/check.sh) | the check |

The tools are [bench/agent_check/agent_tables.py](../../../bench/agent_check/agent_tables.py) and
[bench/agent_check/class_arms.py](../../../bench/agent_check/class_arms.py).

## Provenance and licence

The tables are computed from per-request and per-attempt rows published in this repository: times, token counts and
task ids, no text. The tasks are the agent check's 12 bug fixes, derived from Exercism's Python and JavaScript
exercises (MIT; source, licences and commits in [bench/agent_check](../../../bench/agent_check/README.md)). The runs
are those of the linked folders, on a Manli M-P106L9-N6G (nine P106-090). The tables summarize this project's own runs
and are published under the repository's licence.
