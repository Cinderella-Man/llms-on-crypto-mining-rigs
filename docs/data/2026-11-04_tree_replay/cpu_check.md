# CPU check of the verify-step dump and the tree-value replay (2026-11-04)

[run/cpu_check.sh](run/cpu_check.sh) on the quant machine, in Docker without a GPU: fork branch
`pascal-p106-v3-b12076-specdump` `dafb09a4a`, CPU backend only, `-j 6` under the shared build lock.

| Check | Verdict | Detail |
|---|---|---|
| build | OK | `llama-server`, `test-spec-dump`, `test-llama-archs`, static, no warning in the changed files |
| unit | OK | `test-spec-dump`: 155 checks, 0 failed (synthetic DFlash2 lattices: softmax rows and forward marginals against a double-precision reference, every path summing to 1 when all candidates are kept; a cut to 2 of 6 candidates with the walked one added, rows as sub-rows of the full softmax, the kept mass, a stopped walk; the anchor row as the only one read at the first position) |
| selftest | OK | `tree_replay.py --selftest`: 16 of 16 (below) |
| smoke | OK | random-weight qwen4exp as target and as its own draft-simple drafter: 37 dump lines, all with candidates, target rows and positions; `--check` 0 problems; the same tokens with and without the dump on all 3 requests |

## The replay's self-test

| Test | Result |
|---|---|
| a hand-made 2 × 2 lattice: the continuation rebuilt from the next step | (12, 21) |
| best-first order and path probabilities | 0.6, 0.42, 0.4, 0.36, 0.18, 0.04, in that order |
| the tree's τ at budgets 1, 2, 3, 4, 6 against a known truth | 1, 1, 2, 3, 3 |
| the drafter's walk and the copy chain on the same step | 0 and 2 tokens accepted |
| factorized path probabilities (marginals 0.6/0.4, 0.78/0.22) | 0.6, 0.468, 0.4, 0.312, 0.132, 0.088 |
| a dump whose accepted count contradicts the target's tokens | caught by `--check` |
| 6,000 random lattices (4 candidates, 5 positions, 10 % of each row's mass outside the candidates) with continuations drawn from the lattice | each of 16 ranks' value equals its mean path probability (largest \|z\| 2.58); τ at budgets 3 / 8 / 16: 2.001 / 2.603 / 3.041 against 2.009 / 2.616 / 3.072 predicted |
| monotonic chain acceptance; the tree accepts at least the chain within the walk's depth | yes |

## The smoke test

Requests (token prompts of 130 tokens with a 40-token block repeated, so the context match finds it): R1 greedy, 48
tokens; R2 temperature 1.0, seed 42, 48 tokens; R3 greedy, R1's prompt and reply plus 30 tokens, 32 tokens. Four
drafted tokens (`--spec-draft-n-max 4`), `LLAMA_SPEC_DUMP_K=4`, `_TGT_K=4`, `_NGRAM=8`.

| Request | tokens | drafted / accepted (server) | same tokens without the dump |
|---|---:|---|---|
| R1 (greedy) | 48 | 37 / 37 | yes |
| R2 (temperature 1.0) | 48 | 82 / 4 | yes |
| R3 (greedy, cached prefix) | 32 | 24 / 24 | yes |

The dump: 37 steps, all `draft-simple` with candidates and the target's rows, 22 with a context match, 143 drafted
and 65 accepted tokens; 20 steps follow a checkpoint replay (the random recurrent model verifies more tokens than its
rollback snapshots hold, so a partly accepted draft is replayed, and the replay is not written: the next step starts
one token later). The drafter is the target itself, so greedy drafts are always accepted and sampled ones almost never
(the random model's distribution is nearly flat). [smoke_dump.jsonl](smoke_dump.jsonl) is that dump, as a sample of
the format; it carries no lattice (draft-simple records marginals), the DFlash2 lattice path is covered by the unit
test and first runs on the real model in [run/run_quant.sh](run/run_quant.sh)'s smoke stage.
