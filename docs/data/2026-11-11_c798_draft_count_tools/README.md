# C798: the counting tools for longer drafts and token trees with the MTP draft, checked on a CPU

Registry: [Q1395](../../../quants/experiments.md) (C798), for Q1396 (FN80: the MTP draft at 3 to 7 tokens) and Q1397
(FN81: token trees priced offline); idea row Q1398 (C799: a siblings tree for the MTP draft). Tools:
[bench/draft_count](../../../bench/draft_count/README.md) and the sequential-drafter tables of
[bench/tree_replay](../../../bench/tree_replay/README.md).

## Result

- **The engine already counts both, so the tools needed no engine change.** `pascal-p106-v4` keeps a per-slot count of
  accepted tokens per draft position and exports it on `/metrics`
  (`llamacpp:spec_decode_num_accepted_tokens_per_pos_total`). `LLAMA_TRACE=1` logs every verify step's accepted a of d
  drafted tokens. `LLAMA_SPEC_DUMP` (FN29's dump, part of `pascal-p106-v4`) writes each step's draft, the main model's
  tokens and the MTP draft's most probable candidates at every drafted position, with their probabilities.
  `--spec-draft-n-max N` also sizes the recurrent snapshots (`n_rs_seq` = N), so a 7-token draft is rolled back
  without a checkpoint restore.
- **`bench/draft_count`** runs one load per draft length on recorded agent requests (probabilistic verification and
  C675's stop on every load) and reads each load three ways (the replay's `timings`, the trace lines, `/metrics`), plus
  the dump. It checks them against each other, then gives tokens a step, P_i and c_i by position, a split by context
  length, a cut-to-shorter estimate, and every length against the 3-token draft with intervals.
- **`bench/tree_replay`** reads the MTP dump as it is (`--mode factorized`). It now prices **siblings** trees: the walk
  plus w − 1 leaves beside each of its first m tokens. They need no drafting pass beyond the walk's, and their value is
  exact under probabilistic verification: the walk is verified by rejection, and the residual's draw is matched against
  the leaves. It also prices **full** trees of width w over the first m positions (estimates). Each is set against the
  chain with as many verified tokens, on the same steps.
- **The CPU dry run passed every check.**
  - On a random-weight model with an MTP block: all 34 of 34 count checks passed. The dump changed nothing: the 7-token
    loads with and without it wrote the same 6 replies with the same counts. The CPU loads repeat exactly.
  - With the server's synthetic acceptance (known rates per position), every P_i came back at its set rate, and the
    7-token draft cut to 4 matched the 4-token load. Details below.

## What ran

All on a CPU (a build machine, Docker without cards, `--cpu-shares 256`, `nice -n 19`), with no GPU. The binaries were
`pascal-p106-v4` `08634ce00`'s CPU build (`llama-server`, `test-llama-archs`). The model was a random-weight qwen4exp
MoE model with one MTP block and its draft file (`test-llama-archs -o --mtp-draft`), with Flash-Next's tokenizer and
chat template (`LLAMA_ARCHS_VOCAB`) and a 65,536-token context. It ran on [run/cpu_preset.env](run/cpu_preset.env):
the release preset's context, sampling and speculative flags, without the CUDA settings. The requests were the first
two requests of the first three traces of the release check's `t1swift` arm (6 requests, contexts of 1,397 to 3,124
tokens, 802 to 1,825 of them new since the cache carries over), re-sent by `bench/agent_check/replay.py` with their
recorded seeds.

| Step | What | Result |
|---|---|---|
| selftest | `draft_table.py --selftest` (synthetic loads with known counts; a broken `/metrics` counter and a broken dump must be caught) and `tree_replay.py --selftest` (its earlier checks, plus two hand-made MTP steps and 4,000 random steps with known sibling values) | both pass ([selftest-draft_table.txt](selftest-draft_table.txt), [selftest-tree_replay.txt](selftest-tree_replay.txt)) |
| chain | `run_draft_count.sh`, loads `d3 d5 d7 d7x d3`, 128-token replies; `d7x` is `d7` with the dump off | 34 of 34 checks OK ([tables/check.tsv](tables/check.tsv)). The random model accepts nothing: its draft's top 10 are near-uniform (about 0.10 each) and its main model spreads its probability over the whole vocabulary, so all 756 verify steps of every load keep 0 drafted tokens. `d7` and `d7x`: 6 of 6 replies the same, and the same generated, drafted and accepted counts, verify steps and P_1. The two `d3` loads: the same. `d7`'s dump: 756 steps, type `draft-mtp`, verify `reject`, 5,292 candidate rows (7 per step: the top 3 and the drafted token where it is not among them) ([verdict.txt](verdict.txt)) |
| synth | a second chain, loads `d7s d4s`, with `--spec-synth-rates 0.9,0.8,0.7,0.6,0.5,0.4,0.3` (and the first four for `d4s`): the server accepts the first i drafted tokens with those probabilities. 1,024-token replies, dump off (synthetic steps are not dumped) | 12 of 12 checks OK; every P_i is its set rate within sampling error (table below); tokens a step 5.1980 (`d7s`, set 5.2) and 4.0157 (`d4s`, set 4.0); the 7-token draft cut to 4 gives 4.0204 tokens a verify step against `d4s`'s own 4.0315 (set 4.0000) ([verdict-synth.txt](verdict-synth.txt), [tables-synth/](tables-synth/summary.md)) |
| tree | `tree_replay.py --check` and the analysis (`--mode factorized`, shapes) on `d7`'s dump | 756 steps, 0 problems ([tree-check.txt](tree-check.txt)); 720 steps scored (the others' continuation ends before 7 tokens); the shape tables are written ([tree/shapes.csv](tree/shapes.csv), [tree/siblings.csv](tree/siblings.csv)), every value 1.0 since nothing is accepted |

Synthetic acceptance, P_i (the first i drafted tokens all accepted) against the set rate:

| Load | Steps | P_1 | P_2 | P_3 | P_4 | P_5 | P_6 | P_7 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| set | | 0.9 | 0.8 | 0.7 | 0.6 | 0.5 | 0.4 | 0.3 |
| `d7s` | 1,175 | 0.8987 | 0.8075 | 0.7118 | 0.6063 | 0.5060 | 0.4046 | 0.2979 |
| `d4s` | 1,522 | 0.9054 | 0.8101 | 0.7163 | 0.6028 | | | |

The largest |z| is 1.39 (`d4s`, P_3). The steps drafting position i run from 1,175 to 1,168 for `d7s` and from 1,522
to 1,518 for `d4s`: a reply's last steps draft fewer tokens.

## What it means for FN80 and FN81

- **FN80 (Q1396)** runs `run_draft_count.sh` on three quant cards with the real model on `t1swift`'s 66 whole recorded
  requests, two loads of each length from 3 to 7, est. 1.3 h. Its loads also write the dump. The commands are in the
  registry row.
- **FN81 (Q1397)** needs no cards: `tree_replay.py` on FN80's 7-token dumps, est. 10 minutes on a CPU.
- **Not checked here:**
  - real acceptance on the real model (the random model accepts nothing, the synthetic rates are set);
  - the CUDA build. The counters and the dump are host code, the same in every build.

## Files

| File | What |
|---|---|
| [run/dry_run.sh](run/dry_run.sh) | the dry run's steps: `model`, `selftest`, `chain`, `tables`, `synth`, `tree` |
| [run/cpu_preset.env](run/cpu_preset.env) | the CPU stand-in for the release preset |
| [summary.tsv](summary.tsv) | each step's verdict (the model files' MD5s; `WORK` stands for the work folder) |
| [verdict.txt](verdict.txt), [verdict-synth.txt](verdict-synth.txt) | the two chains' verdicts |
| [tables/](tables/summary.md) | `draft_table.py` on the main chain: `loads.tsv`, `positions.tsv`, `arms.tsv`, `compare.tsv`, `by_context.tsv`, `cut.tsv`, `requests.tsv`, `same_reply.tsv`, `check.tsv`, `summary.md` |
| [tables-synth/](tables-synth/summary.md) | the same for the synthetic chain |
| [tree/](tree/summary.md) | `tree_replay.py` on `d7`'s dump: `shapes.csv`, `siblings.csv` and its other tables |
| [tree-check.txt](tree-check.txt) | `tree_replay.py --check` on that dump |
| [dump_sample.jsonl](dump_sample.jsonl) | the first five lines of `d7`'s dump: the format for the MTP draft (random-weight model, Flash-Next's token ids) |
| [selftest-draft_table.txt](selftest-draft_table.txt), [selftest-tree_replay.txt](selftest-tree_replay.txt) | the self-tests' output |

## Commands

On the build machine, from a work folder holding this repository's checkout in `repo/`, the two binaries, the recorded
requests and an env file with the variables `dry_run.sh` names (`WORK`, `REPO`, `ARCHS`, `BIN`, `VOCAB`, `LOG`,
`MOUNTS`, `IMAGE`):

```bash
. ./dry.env && bash repo/docs/data/2026-11-11_c798_draft_count_tools/run/dry_run.sh
```

The steps can be run one by one with `STEPS="model selftest chain tables synth tree"` (the default order). `VOCAB` was
Swift Q2_0's first shard: only its header (tokenizer and chat template) is read. `LOG` was the release check's
`t1swift` proxy log and its bodies.

## Provenance and licence

The requests come from this repository's own agent check (the v3 release check's `t1swift` arm, 12 bug-fix tasks). The
model is random weights written by llama.cpp's test tool; only the tokenizer and chat template come from Flash-Next.
The tables are this repository's tools' output, under the repository's licence.
