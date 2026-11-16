# The agent check's work counts on ten published runs (C852, Q1521; 2026-11-17)

[bench/agent_check](../../../bench/agent_check/README.md)'s report now counts what a run did beside how fast it did
it: prompt tokens read, reply tokens re-read and the longest first-token wait, per run
([the columns](../../../bench/agent_check/README.md#the-work-a-run-did-beside-its-speed)). This folder holds those
counts for ten published runs and the check of the new code against a count made by hand. No model ran: the counts
are arithmetic on per-request tables already in this repository.

Registry row Q1521 in [quants/experiments.md](../../../quants/experiments.md).

## Why

"Minutes a task at equal work" prices one run's token counts at another run's speeds, so it holds the tokens fixed.
On v3.0 it classed text reuse (`LLAMA_SERVER_TEXT_REUSE=1`) negligible, 2.59 against 2.60 minutes a task, in a run
that read 2,982 prompt tokens a task against 5,847
([2026-11-06_flint_side_rig2](../2026-11-06_flint_side_rig2/README.md)). The
[prompt-path review of v4.0](../../research/notes/v5_review_prompt_path_2026-11.md) asked for tables that count the
work before text reuse is run again on v4.0.

## The check: re-read seconds against the review's

The review counted, by hand from `turns.csv`, the prompt seconds of the turns that read a reply again in four runs
on `v4-rc2`. `report.py work` on the same tables gives the same seconds and the same turns:

| Run | Model | The review: re-read s of prompt s | `report.py work` | Turns |
|---|---|---|---|---|
| R159e `f4ta` | Flint Pascal Q8 GGUF (`p065`) | 67.6 of 222.6 | 67.6 of 222.6 | 4 |
| R159f `f4tb` | Flint Pascal Q8 GGUF | 141.7 of 299.5 | 141.7 of 299.5 | 5 |
| R166a `o4sna` | Ornith-1.5-122B Pascal N2P GGUF (`valid1`) | 30.6 of 396.6 | 30.6 of 396.6 | 2 |
| R166b `o4snb` | Ornith-1.5-122B Pascal N2P GGUF | 71.9 of 467.8 | 71.9 of 467.8 | 3 |

To three decimals they are 67.612 of 222.579, 141.722 of 299.467, 30.604 of 396.630 and 71.880 of 467.817 s
([work.tsv](work.tsv)). The review's "four turns of R159e" and "five of R159f, one alone 93.4 s" are the rows of
[work.md](work.md)'s second table (the one turn: `f4tb`, Python `scale-generator`, turn 4, 93.41 s).

The changed `table` and `compare` reports were also run on the raw logs of R159's six runs and of R166's two (each
attempt's `result.json` and the proxy's `proxy.jsonl`, which are not published):
- their three new columns equal `report.py work`'s on the published tables, run for run;
- with the new cells removed, `table`'s rows equal
  [r159_six_report_table.md](../2026-11-14_r159_flint_release_agent/r159_six_report_table.md) and `compare`'s report
  equals [r159_six_compare.md](../2026-11-14_r159_flint_release_agent/r159_six_compare.md), without its
  graph-reallocation block, which needs the server logs;
- the older columns of `attempts.csv` and `turns.csv` equal the published tables of all eight runs, row for row.

The full self-test ([selftest.sh](../../../bench/agent_check/selftest.sh)) was not run: its new step, a fixed
per-request table with every cache path through `report.py work`, was run on its own and passes.

## Eight runs on v4-rc2 and v3.0

Flint Pascal Q8 GGUF, six runs of [R159](../2026-11-14_r159_flint_release_agent/README.md): `f3*` is v3.0, `f4*` the
v4 preset, `f4t*` the tuned v4 preset. Ornith-1.5-122B Pascal N2P GGUF, two runs of
[R166](../2026-11-16_r166_ornith_stack_no_taper/README.md). 12 tasks a run. From [work.md](work.md):

| Run | Turns | Prompt tokens read (per task) | Tokens generated (per task) | Reply tokens re-read (turns; their prompt s of all prompt s) | Longest first-token wait s (task, turn) |
|---|---|---|---|---|---|
| `f3a` | 66 | 37,335 (3,111) | 28,677 (2,390) | 3,868 (6; 26.0 of 208.3) | 13.04 (JavaScript `complex-numbers`, 2) |
| `f3b` | 68 | 49,185 (4,099) | 41,976 (3,498) | 16,470 (6; 86.3 of 271.2) | 55.47 (Python `scale-generator`, 4, a re-read) |
| `f4a` | 73 | 72,086 (6,007) | 68,241 (5,687) | 36,767 (9; 211.7 of 407.5) | 88.86 (Python `scale-generator`, 6, a re-read) |
| `f4b` | 80 | 66,718 (5,560) | 64,569 (5,381) | 31,643 (5; 167.9 of 373.0) | 78.05 (Python `scale-generator`, 8, a re-read) |
| `f4ta` | 73 | 48,907 (4,076) | 37,846 (3,154) | 14,632 (4; 67.6 of 222.6) | 24.70 (Python `affine-cipher`, 3, a re-read) |
| `f4tb` | 71 | 61,112 (5,093) | 48,656 (4,055) | 24,755 (5; 141.7 of 299.5) | 93.88 (Python `scale-generator`, 4, a re-read) |
| `o4sna` | 63 | 40,863 (3,405) | 21,810 (1,818) | 4,831 (2; 30.6 of 396.6) | 28.14 (Python `affine-cipher`, 1) |
| `o4snb` | 73 | 49,555 (4,130) | 27,605 (2,300) | 12,332 (3; 71.9 of 467.8) | 32.24 (Python `scale-generator`, 7, a re-read) |

- **Every run read a reply again:** 6, 6, 9, 5, 4 and 5 turns in Flint's runs and 2 and 3 in Ornith's, none from an
  older checkpoint. The longest first-token wait is a re-read in six of the eight runs.
- **Two runs of one arm do different work.** The agent sends no seed, and the same request ends in a short tool call
  or a long reply by chance ([2026-11-16_c843_flint_sampling](../2026-11-16_c843_flint_sampling/README.md)): `f4ta`
  and `f4tb` read 4,076 and 5,093 prompt tokens a task, `f3a` and `f3b` 3,111 and 4,099. A change is classed on two
  runs an arm, each run by value.
- **`f3a` and `f4ta` each hold one request without counters:** the reply that the 12-minute cap cut in Python
  `scale-generator`. It has a first token (3.02 and 3.25 s) and no token counts, so it adds no tokens to its run's
  rows and its cache path is unknown.

## Text reuse on v3.0, counted again

The two runs of [2026-11-06_flint_side_rig2](../2026-11-06_flint_side_rig2/README.md) on Flint Pascal Q8 GGUF, one run
an arm: `v3base` (every switch at its default) and `reread`
(`LLAMA_SERVER_TEXT_REUSE=1 LLAMA_SERVER_TAIL_CKPT_MIN=4096`). From [work_text_reuse_v3.md](work_text_reuse_v3.md):

| Run | Turns | Prompt tokens read (per task) | Tokens generated (per task) | Reply tokens re-read (turns; their prompt s of all prompt s) | Longest first-token wait s (task, turn) | Minutes a task at equal work |
|---|---|---|---|---|---|---|
| `v3base` | 75 | 70,159 (5,847) | 65,307 (5,442) | 33,728 (7; 191.7 of 390.5) | 73.65 (Python `list-ops`, 5, a re-read) | 2.60 |
| `reread` | 77 | 35,780 (2,982) | 55,801 (4,650) | 0 (0; 0.0 of 200.1) | 12.47 (JavaScript `complex-numbers`, 2) | 2.59 |

The new columns show what equal work could not: with text reuse no turn read a reply again, where `v3base` read
33,728 reply tokens again in 7 turns that took 191.7 of its 390.5 prompt seconds, and the longest first-token wait
fell from 73.65 to 12.47 s. The per-task prompt tokens are the published 5,847 and 2,982. It is one run an arm, so
this is a count, not a class.

## Files

| File | What it is |
|---|---|
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on the eight runs: one row a run, then every turn that read a reply again (40 turns) with its cache path, the previous prompt's end, the previous reply's tokens, the cached tokens, the reply tokens re-read, the prompt tokens and seconds, and the first token |
| [work_text_reuse_v3.md](work_text_reuse_v3.md), [work_text_reuse_v3.tsv](work_text_reuse_v3.tsv) | the same for v3.0's two runs, named `v3base` and `reread` (both tables call their arm `rel`) |

`work.tsv`'s columns: `arm`, `tasks`, `turns` (requests), `prompt_read` and `generated` (tokens), `reread_turns`,
`older_checkpoint_turns` (of them), `reread_n` (reply tokens re-read), `reread_s` and `prompt_s` (the re-read turns'
prompt seconds and the run's), `ttft_max_s` with `ttft_max_task` and `ttft_max_turn`, and `ttft_max_reread` (1 when
that turn read a reply again).

## Commands

From the repository's root, with Python 3 and no packages:

```bash
D=docs/data/2026-11-17_c852_agent_work_counts
python3 -I bench/agent_check/report.py work \
  docs/data/2026-11-14_r159_flint_release_agent/agent/*/turns.csv \
  docs/data/2026-11-16_r166_ornith_stack_no_taper/agent/*/turns.csv --tsv $D/work.tsv > $D/work.md
A=docs/data/2026-11-06_flint_side_rig2/agent
python3 -I bench/agent_check/report.py work v3base=$A/flint-v3base/turns.csv reread=$A/flint-reread/turns.csv \
  --tsv $D/work_text_reuse_v3.tsv > $D/work_text_reuse_v3.md
```

On a run's raw logs the same columns come with the usual reports (`CLIENT_OUT` is the release check's folder on the
client):

```bash
python3 -I bench/agent_check/report.py table $CLIENT_OUT/res --log $CLIENT_OUT/proxy.jsonl
python3 -I bench/agent_check/report.py compare $CLIENT_OUT/res --log $CLIENT_OUT/proxy.jsonl --arms f4ta,f4tb
```

## Provenance and licence

Derived from this repository's own tables: the per-request rows (`agent/<arm>/turns.csv`) of the three data folders
linked above, which hold llama-server's counters and the client's times for each request of the agent check, and no
task text. The counting code is [bench/agent_check/report.py](../../../bench/agent_check/report.py). Same licence as
the repository.
