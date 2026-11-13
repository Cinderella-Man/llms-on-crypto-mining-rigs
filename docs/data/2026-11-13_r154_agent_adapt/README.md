# The agent check of `v4c3` with the draft length decided while drafting (R154, Q1465; 2026-11-13)

R154 runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md), 12 bug-fix
tasks) on `v4c3`, v4's candidate line after the code freeze ([R142](../2026-11-13_r142_agent_v4c3/README.md)), with the
MTP draft's length chosen per step by `LLAMA_SPEC_ADAPT` and decided while drafting (C815,
[2026-11-12_c815_adapt_while_drafting](../2026-11-12_c815_adapt_while_drafting/README.md)), on one rig of nine
P106-090 cards with Flash-Next's v3 preset (Swift Q2_0 and the MTP draft). Two runs, one arm each: R154a (`v4c3ad`, this
page) and R154b (`v4c3ad2`, open). Registry row Q1465 in [quants/experiments.md](../../../quants/experiments.md).

**R154a: `v4c3ad` takes 2.46 minutes a task at equal work with 12 of 12 passed, and generates at 25.26 tok/s, below
both `v4c3` runs (28.03 and 26.16; without `bugfix-javascript-complex-numbers` 25.26 against 28.00 and 27.79).
Generation did not reach 30 tok/s.** The rule drafted 6 tokens on 51.0 % of the steps, and a step gave 4.049 tokens
against `v4c3`'s 3.29, but the longer steps cost more than they gave: est. 160.3 ms a step against `v4c3`'s 117.4. At
equal work `v4c3`'s two runs took 2.35 and 2.44 minutes a task (2.45 and 2.43 without that task, against 2.55 here).
C815's estimate from recorded drafts was +0.946 and +0.989 tok/s, priced with FN82's estimated v4 cost table; on the
P106 this first run is 2.77 tok/s below R142a. The arm is not classed before its second run.

In plain words: the draft now guesses up to six tokens ahead instead of three, and more of each guess is kept, but
checking the longer guesses on these cards takes more time than the extra tokens save.

Every value below is one run; `v4c3`'s two runs are listed each by its own values. "Equal work" is R90's `v3` work (its
prompt and generated tokens, and its time outside the server) at each arm's own prompt and generation speeds
([agent_tables.py](../2026-11-05_keepalive/run/agent_tables.py)): it compares the engines on the same work, while the
minutes for all tasks depend on what each attempt generated.

## The arm

`v4c3`'s command exactly ([R142's driver](../2026-11-13_r142_agent_v4c3/run/r142.sh): the build, the model files, the
12 tasks, the client and every switch), with two changes and nothing else ([run/r154.sh](run/r154.sh), the arm
`v4c3ad`):

| Change | What it is |
|---|---|
| `LLAMA_SPEC_ADAPT=1 LLAMA_SPEC_ADAPT_ONLINE=2` | the draft's length chosen per step by the switch's rule, decided while drafting (C815's level 2: it also stops when the next allowed length cannot pay even if every token still to draft is accepted) |
| `LLAMA_SPEC_ADAPT_COST=87.597,94.940,105.432,117.985,139.227,152.703,166.181` | FN82's est. v4 cost table, V(1..7) in ms ([2026-11-12_fn82_adaptive_draft](../2026-11-12_fn82_adaptive_draft/README.md)) |
| `LLAMA_SPEC_ADAPT_DRAFT_MS=21.87` | the est. cost of one drafting pass |
| `GGML_CUDA_MMVQ_PASCAL_RO8=1` | C729's read-once kernels at 5–8 tokens with C809 (passed on a P106 in R134), since verify passes now reach 7 tokens |
| `--spec-draft-n-max 6` in place of `--spec-draft-n-max 3` | a draft of up to 6 tokens |

`GGML_CUDA_MMVQ_PASCAL_KCOL_EXACT=1`, probabilistic verification, `--spec-draft-p-min 0` (which C815 needs) and C675's
stop on a rejected draft (`LLAMA_SPEC_REJECT_STOP=1`) are `v4c3`'s already. The switch's verify widths were its
defaults, 1, 3, 5 and 7 (`LLAMA_SPEC_ADAPT_WIDTHS` unset), so a step drafts 0, 2, 4 or 6 tokens.

## The run

- **Build:** C825's tarball of `pascal-p106-v4`'s tip `65ac9a47d` (which holds C815 as `14710289b`), the tree R142
  ran, `llama-server` md5 `9cbea9859c17e3355c9eb51d14a4a863`.
- **Checked before the first task:** the started command (environment, binary and arguments, 1,839 characters) equals
  R142b's (1,658) with exactly the two changes applied, as a string comparison; 44 environment variables against 39.
  The server log, read after the run: `switches: 44 set`; `LLAMA_SPEC_ADAPT: on; verify widths 1,3,5,7 (n_max 6),
  T(1) 87.6 ms, T(7) 166.2 ms, draft 21.9 ms, S run`; `LLAMA_SPEC_ADAPT_ONLINE=2: the rule is taken while drafting
  (C815)`; `load_model` names Swift Q2_0's first shard.
- **Timeline (2026-11-13):** server started 12:24:15, up after 311 s (R142a 309, R142b 313); 12 tasks 12:29:29–12:54:39;
  report 12:54:52–12:54:54; the rig clean at 12:54:55. The client ran on another machine's CPU at `--cpu-shares 256`.

## The numbers ([agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv))

| | R142a `v4c3` | R142b `v4c32` | **R154a `v4c3ad`** |
|---|---|---|---|
| passed | 12/12 | 12/12 | **12/12** |
| equal work, min a task (12) | 2.35 | 2.44 | **2.46** (−20.1 % against R90's `v3`, 3.08) |
| equal work, R90's 11 | 1.49 | 1.61 | **1.61** |
| equal work without `complex-numbers` | 2.45 | 2.43 | **2.55** |
| turns | 66 | 98 | **68** (63 without `complex-numbers`) |
| generation tok/s | 28.03 | 26.16 | **25.26** |
| generation tok/s without `complex-numbers` | 28.00 | 27.79 | **25.26** |
| prompt tok/s | 145.5 | 136.6 | **143.7** (without `complex-numbers` 140.6; 141.6, 152.0) |
| first token median, p90 s | 2.45, 8.39 | 2.45, 9.77 | **2.58, 8.41** (without `complex-numbers` 2.50, 8.38; 2.45, 8.26; 2.50, 8.81) |
| first token of a task's first turn, median s | 6.64 | 6.62 | **6.56** |
| acceptance | 0.765 | 0.723 | **0.759** (22,035 of 29,014 drafted; 0.760 without `complex-numbers`; 0.763, 0.759) |
| tokens a step | 3.29 | 3.17 | **4.049** (from the server log; see below) |
| est. ms a step (tokens a step ÷ generation tok/s) | 117.4 | 121.2 | **160.3** |
| graph reallocations | 0 of 66 | 0 of 98 | **0 of 68** |
| lowest MemAvailable ([host.tsv](host.tsv)) | 478 MiB | 376 MiB | **275 MiB** (12:45:13) |
| swap | 5 → 475 MiB | 8 → 8 MiB | **8 → 8 MiB** |
| minutes, all 12 tasks (wall) | 16.4 | 32.2 | **24.9** |

- **Generation is below both `v4c3` runs** on all 12 tasks and without `complex-numbers` (25.26 against 28.00 and
  27.79), by more than `v4c3`'s two runs differ. At equal work `v4c3ad` is level with R142b on all 12 (2.46 against
  2.35 and 2.44) only because R142b's `complex-numbers` took the 37-turn path; without that task it is slower than both
  (2.55 against 2.45 and 2.43). Not classed before R154b.
- **The tables' "tokens a step" column is 1 + 3 × acceptance**, right for a fixed draft of 3 (`v4c3`) but not for this
  arm, where it reads 3.28. The 4.049 above is the server's own count, (accepted + steps) ÷ steps over 7,227 verify
  steps ([draft_widths.tsv](draft_widths.tsv)). The ms a step are est.: each arm's tokens a step divided by its
  whole-session generation speed.
- **Prompts** ([prompt_buckets.tsv](prompt_buckets.tsv)), by a turn's new prompt tokens, tok/s (R142a, R142b):
  0–511 88.9 (94.0, 93.0), 512–1,023 122.9 (124.4, 111.2), 1,024–1,999 181.1 (186.5, 162.1), 2,000+ 170.7 (180.6,
  181.0): between `v4c3`'s two runs at 512–1,999 new tokens, below both at 0–511 and at 2,000+.

## The draft's lengths ([draft_widths.tsv](draft_widths.tsv))

From the server's per-request lines (`adaptive draft (on) = … steps by verify width (…)`, `draft acceptance = …`),
summed by [run/draft_widths.sh](run/draft_widths.sh):

| Verify width (tokens drafted) | Steps | Share |
|---|---|---|
| 1 (0) | 761 | 10.5 % |
| 3 (2) | 2,114 | 29.3 % |
| 5 (4) | 663 | 9.2 % |
| 7 (6) | 3,689 | 51.0 % |
| all | 7,227 | 4.015 drafted tokens a step, 4.049 tokens a step |

- **C815's closing line:** 7,227 drafts under the rule taken while drafting, 3,424 stopped early, 11,819 drafting passes
  left out (of 43,362 at 6 a draft, so 31,543 made: 4.36 a step against `v4c3`'s 3), 11,819 sampler draws made for
  them, 2,873 last passes kept. C738 prepared 2,873 verify passes during the last drafting pass, 0 not prepared,
  0 dropped. The draft length cap (`LLAMA_MTP_DRAFT_CAP=1`) left out 0 passes.
- **Acceptance by task** ([task_acceptance.tsv](task_acceptance.tsv)): lowest 0.679 (`python-pig-latin`), highest 0.817
  (`javascript-grade-school`), 0.759 over all 12; the tasks with the most drafted tokens, `python-variable-length-quantity`
  (6,502 drafted, 0.793; `v4c3` 0.815 and 0.788) and `javascript-variable-length-quantity` (4,448, 0.791; 0.769 and 0.787).
- **Why it can be slower with more tokens a step:** the rule chose lengths by FN82's estimated table, and on these cards
  the long steps did not pay that table's way: 0.76 more tokens a step for an est. 42.9 ms more a step, where `v4c3`'s
  whole step is 35.7 ms a token (est.). That table was built from v3's profile scaled by single-kernel timings; the
  P106 map of the v4 step at verify widths 1 to 8 (R121, Q1393) is what C797 (Q1394) feeds into the table instead.

## The host and the cards

- **Host:** MemAvailable lowest 275 MiB at 12:45:13 (during `python-scale-generator`), 284 MiB at the end; swap stayed
  at 8 MiB. The run's watchdog did not act. 275 MiB is 101 MiB below R142b's lowest and the lowest of the agent arms in
  R72's client folder (the report's `meminfo.log` reads 276 MB), so the report's memory row reads NO-GO on it.
- **Cards** (peak MiB, 12:24:09–12:54:55): 0:5,175 1:5,207 2:5,075 3:5,223 4:4,319 5:5,091 6:4,465 7:5,103 8:5,561;
  R142b 0:5,117 1:5,159 2:5,019 3:5,175 4:4,275 5:5,033 6:4,429 7:5,045 8:5,701.
- **Server log:** 13 checkpoint restores, none erased, no full re-processing, no NaN, no watchdog kill; graph
  reallocations 0 of 68 requests.
- **Report** ([v4c3ad_gonogo.md](v4c3ad_gonogo.md), [v4c3ad_report_table.md](v4c3ad_report_table.md)):
  infrastructure GO (84/84), tool calls 0 malformed of 494 (`v4c3ad` 65, 1 failed), first token vs prefill GO; proxy
  rows NO-GO only for R72b's runaway, memory NO-GO with this run's 275 MiB.

## Per task ([v4c3ad_compare.md](v4c3ad_compare.md))

Minutes from the agent's start to its exit, turns, against R142a and R142b:

| Task | R142a `v4c3` | R142b `v4c32` | R154a `v4c3ad` |
|---|---|---|---|
| python-affine-cipher | 1.8, 6 | 2.2, 6 | 3.1, 8 |
| javascript-space-age | 0.8, 5 | 0.8, 5 | 0.8, 5 |
| python-phone-number | 0.9, 6 | 1.0, 6 | 0.8, 5 |
| javascript-triangle | 0.7, 5 | 0.7, 5 | 0.8, 5 |
| python-variable-length-quantity | 2.2, 6 | 4.4, 6 | 4.8, 5 |
| javascript-phone-number | 1.5, 7 | 0.9, 5 | 1.1, 5 |
| python-list-ops | 1.3, 5 | 2.1, 5 | 2.1, 5 |
| javascript-grade-school | 0.8, 5 | 0.9, 7 | 1.0, 7 |
| python-scale-generator | 1.9, 5 | 2.3, 5 | 3.2, 6 |
| javascript-complex-numbers | 0.9, 5 | 10.8, 37 | 1.3, 5 |
| python-pig-latin | 1.9, 5 | 2.3, 6 | 2.6, 7 |
| javascript-variable-length-quantity | 1.6, 6 | 4.0, 5 | 3.3, 5 |

Quicker than both `v4c3` runs in 1 task (`python-phone-number`), slower than both in 6, between or equal to one of them in 5.

## Commands

R154a, from the repository root on a machine that only runs ssh, with R72's client folders (R72a's prep; R142a brought
its bench files up to date) and C825's tree on the rig:

```bash
RIG=<rig alias> CLIENT=<client alias> \
  MODEL=<folder>/Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf DRAFT=<folder>/mtp-Qwen3.8-Flash-Next-draft.gguf \
  NEW=llama.cpp-20261113-65ac9a47d-pascal-p106-v4 REF=llama.cpp-20261111-d097d6d88-pascal-p106-v4-b12246-kern \
  RIGDIR=<folder on the rig> OUT=<local folder> STEPS="agent report" AGENT_ORDER=v4c3ad \
  REPORT_ARMS="gains v4c2 v4c22 v4c2g v4c3 v4c32 v4c3ad" \
  RIG_PORT=<port> RELAY_PORT=<port> CLIENT_PORT=<port> PROXY_PORT=<port> \
  CLIENT_DIR=<folder on the client> CLIENT_OUT=<folder on the client> CPU_SHARES=256 \
  setsid nohup bash docs/data/2026-11-13_r154_agent_adapt/run/r154.sh > r154a.log 2>&1 < /dev/null &
```

The report rewrites the client folder's `res/turns.csv` with exactly the arms it is given, and the tables need R72's
and R142's arms there, so `REPORT_ARMS` names all of them. R154b is the same command with `AGENT_ORDER=v4c3ad2` and
`v4c3ad2` added to `REPORT_ARMS`.

The tables, on the client's CPU in an empty folder ([run/tables.sh](run/tables.sh), R142's, unchanged), the draft's
lengths from the collected server log, and the cards' peaks:

```bash
REPO=<this repository> R90=<R90's client results folder> R72=<R72's client results folder> NEWARMS="v4c3 v4c32 v4c3ad" \
  bash <this repository>/docs/data/2026-11-13_r154_agent_adapt/run/tables.sh
nice -n 19 bash docs/data/2026-11-13_r154_agent_adapt/run/draft_widths.sh <R72's client results folder>/rig/server-v4c3ad.log > draft_widths.tsv
awk -F', ' -v a='2026/11/13 12:24:09' -v b='2026/11/13 12:54:55' '$1 >= a && $1 <= b { v = $6 + 0; if (v > p[$2]) p[$2] = v }
  END { for (i = 0; i < 9; i++) printf " %d:%d", i, p[i]; print "" }' <R72's client results folder>/rig/gpu.csv
```

## Files

| File | What |
|---|---|
| [agent.tsv](agent.tsv), [agent_11.tsv](agent_11.tsv), [agent_nocx.tsv](agent_nocx.tsv) | every arm against R90's `v3`: all 12 tasks, R90's 11, the 12 without `complex-numbers` (R72's and R142's rows equal their published ones) |
| [prompt_buckets.tsv](prompt_buckets.tsv) | prompt speed and first token by a turn's new prompt tokens |
| [task_acceptance.tsv](task_acceptance.tsv) | the draft's acceptance by task |
| [host.tsv](host.tsv) | each `v4c3` arm's lowest MemAvailable and swap, from the rig's RAM watchdog |
| [draft_widths.tsv](draft_widths.tsv) | per request: verify steps, accepted and drafted tokens, tokens a step, steps by verify width, graph reallocations; total and shares |
| [v4c3ad_compare.md](v4c3ad_compare.md), [v4c3ad_report_table.md](v4c3ad_report_table.md), [v4c3ad_gonogo.md](v4c3ad_gonogo.md) | the run's own report over R72's, R142's and this arm |
| [run/r154.sh](run/r154.sh) | the driver: R142's with the arm `v4c3ad` (and `v4c3ad2`) |
| [run/tables.sh](run/tables.sh) | R142's tables script, unchanged |
| [run/draft_widths.sh](run/draft_widths.sh) | the draft table from a server log |

Provenance: measured on a Manli M-P106L9-N6G (nine P106-090) with the build and files named above; R90's, R72's and
R142's rows are their own measurements. The prompts are the agent sessions of bench/agent_check's bugfix set; no prompt
text is in this folder. The tables are this repository's measurements, under its licence.
