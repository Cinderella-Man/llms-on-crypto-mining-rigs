# CPU checks of the full-block dump and the by-position tables (2026-11-04)

[run/cpu_check.sh](run/cpu_check.sh) and [run/cpu_dflash2_check.sh](run/cpu_dflash2_check.sh) on the quant machine, in
Docker without a GPU: fork branch `pascal-p106-v3-b12076-specdump2` `3e22fd7c1`, CPU backend only, `-j 6` under the
shared build lock.

| Check | Verdict | Detail |
|---|---|---|
| build | OK | `llama-server`, `test-spec-dump`, `test-llama-archs`, static; 2 warnings in the whole tree (`common/sampling.cpp`, not changed), none in the changed files |
| unit | OK | `test-spec-dump`: 211 checks, 0 failed (155 for the lattice reduction as before, 56 new: the selector's greedy walk against a hand-made reference on three lattices; the second block's record equal to the reduction along that walk, the draft's own record untouched, candidates kept and the walk among them, the first position scored from the anchor, a one-position block, `clear()`, a short block's walk a prefix of a longer one's) |
| selftest | OK | `tree_replay.py --selftest`: 42 of 42 (below) |
| smoke | OK | random-weight qwen4exp as target and as its own draft-simple drafter, three servers (no dump, the dump, the dump with `LLAMA_SPEC_DUMP_FULL_BLOCK=1`): the same tokens on all 3 requests, the two dumps line for line equal but for the server start (a drafter without a second block adds none), 37 lines with candidates, the target's rows and `gram`, `--check` 0 problems |
| real pair | OK | Swift-1.5 Q2_0 and Inco AI's DFlash2 Q8_0 on the CPU backend, `--spec-draft-n-max 2`, four servers: below |

## The real pair on the CPU backend

Two 24-token requests (greedy, "def fibonacci(n): …"; temperature 1.0 with seed 7, "The three laws of thermodynamics
state that") to four servers started one after another: no dump; `LLAMA_SPEC_DUMP`; with `LLAMA_SPEC_DUMP_FULL_BLOCK=1`;
with `=4`. The real drafter's second block goes through the real selector, which no random-weight model exercises.

| Check | Result |
|---|---|
| tokens | the same in the four servers on both requests (24 and 24 tokens) |
| drafts | 18, 18 and 18 dump lines (dump, full, full4); every line equal but for the run and the second block |
| second block | 18 of 18 lines of `full` carry 8 tokens and 7 positions with a candidate list and a walk each; 18 of 18 of `full4` carry 4 tokens and 3 positions; none of the plain dump's carry one |
| `--check` | 0 problems on both dumps; 18 steps with a second block |
| tables | `tree_replay.py` on the full dump: `block*.csv` written, 7 positions |

The first request accepted 15 of 15 drafted tokens and the second 13 of 19 (`draft_n_accepted`), the same in all four
servers. [cpu_dflash2/](cpu_dflash2/) holds the verdict table, the 18 dump lines and the tables.

## The replay's self-test

| Test | Result |
|---|---|
| the earlier 16: hand-made 2 × 2 lattice (continuation rebuilt, best-first order, path probabilities, τ at budgets, the n-gram chain, factorized probabilities), a broken dump caught by `--check`, 6,000 random lattices (rank values equal path probabilities, largest \|z\| 2.58; τ at 3 / 8 / 16 against the prediction), monotonic chain, tree at least the chain | pass |
| a hand-made request of three steps with second blocks (`11 99 31 41 51 61 71` rebuilt by position across steps): steps and censored per position (3 0 / 3 0 / 2 1 / 2 1), P_1…P_4 = 1, 2/3, 1/2, 1/2, conditionals 1, 2/3, 1, 1, strict P_j on the two full-length steps, τ(k) = 2, 2.667, 3.167, 3.667, S(4) = τ / (1 + d + 4r), the width table (same 1.0 and 1.0; P_2 of the draft and of the block 0.5 and 0.5), the grammar split, the tree tables on the second block's lattice (4 positions) and on the draft's own (2) | pass |
| segments from token markers: reasoning until its close (a request that begins inside it), a tool call between its markers, markers of one kind alone, none; a step takes its anchor's segment | pass |
| 8,000 synthetic second blocks, two groups (tool-call anchors at 1,000 tokens of context with conditional acceptance 0.95 0.95 0.90 0.90 0.85 0.85 0.80; prose anchors at 20,000 with 0.70 0.60 0.50 0.50 0.40 0.40 0.30), a quarter of the continuations cut at 3–6 tokens (a request's end): each group's P_1…P_7 and conditionals within sampling error (largest \|z\| 2.08 and 1.96 for the tool group, 1.12 and 1.81 for prose), the context bins, the selector's predicted P_j equal to the truth, censoring at positions 4–7 only, the strict set agreeing with the per-position set, the draft equal to the second block's walk on 0.90 of the steps (0.902), τ(6) 4.035 against 4.020, the reference tails' best k (the note's est. ≥ 5 at r = 0.08, the collapse ≤ 4) | pass |
| the marker helper reading a vocabulary from a synthetic GGUF header past scalar and array values | pass |

The earlier tree-replay dump sample ([../2026-11-04_tree_replay/smoke_dump.jsonl](../2026-11-04_tree_replay/smoke_dump.jsonl),
draft-simple, no second block) still passes `--check` (0 problems) and writes its tables, now with the by-position ones
from the draft's own lattice.
