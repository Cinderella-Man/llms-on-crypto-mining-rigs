# CPU check of `pascal-p106-v3-b12076-blockver` (2026-11-04)

[run/cpu_check.sh](run/cpu_check.sh) on a build machine, in Docker (`nvidia/cuda:12.9.2-devel-ubuntu24.04`, CPU
backend, static), at the branch head `b8f42a3ff`; the base build is `a330672c1`. The stages as last run:
`STEPS="build unit models smoke stat"`, the statistical stage's arms in two runs on the same models (the first with
every arm but the drafter SD2's, the second `STAT_ARMS="plain-rej2 plain-block2 plain-rejkeep2 gram-rejkeep"`), the
analysis over all of them.

| Check | Verdict | Detail |
|---|---|---|
| build | OK | `llama-server`, `test-spec-block`, `test-llama-archs` at `b8f42a3ff`; the base's `llama-server` at `a330672c1` |
| unit | OK | `test-spec-block` 47/47 |
| models | OK | random qwen4exp pairs: Flash-Next's tokenizer (smoke), GPT-2's (stat) |
| smoke-off | OK | 6/6 replies of the new build without switches with the base build's tokens and draft counts |
| smoke-lossless | OK | 21/21 greedy replies of the drafting arms equal to plain decoding |
| smoke-json | OK | 32/32 grammar replies whole (JSON or the XML tool call) |
| smoke-forced | OK | drafts cut to 0: 39 steps, 128 forced tokens appended, 55 accepted |
| smoke-bad | OK | a malformed value stops the server |
| stat | OK | 19/19 exactness tests pass at p ≥ 1e-4; the control (temperature 0.8) told apart at p = 1.7e-21 |

## Statistical stage

Every arm against plain decoding, two-sample chi-square tests of homogeneity (cells pooled to expected counts ≥ 5),
every arm with its own seeds; all rows in [stat.tsv](stat.tsv), acceptance in [acceptance.tsv](acceptance.tsv).

| Arm | Verification | Drafter | Tests | Smallest p | Accepted / drafted |
|---|---|---|---:|---:|---:|
| plain block | block | SD (stdev 0.8) | 5 | 0.221 | 0.0275 |
| plain block2 | block | SD2 (stdev 0.95) | 5 | 0.086 | 0.2207 |
| plain rejkeep2 | token by token, `LLAMA_SPEC_KEEP_DRAWS=1` | SD2 | 5 | 0.014 (token 1, drawn before any draft) | 0.2023 |
| plain rej | token by token (the base build's path) | SD | 5 | 0.058 | 0.0268 |
| plain rej2 | token by token (the base build's path) | SD2 | 5 | 0.0049 | 0.2017 |
| gram block | block | SD | 1 | 0.808 | 0 |
| gram force | block, forced tokens | SD | 1 | 0.073 | 0 |
| gram keep0 | block, forced tokens only | — | 1 | 0.653 | 0.4153 |
| gram rejkeep | token by token, `LLAMA_SPEC_KEEP_DRAWS=1` | SD | 1 | 0.238 | 0 |
| **gram rej** | **token by token (the base build's path)** | SD | 1 | **6.0e-09 (TV 0.095)** | 0 |
| plain t08 (control) | plain decoding at temperature 0.8 | — | 1 | 1.7e-21: told apart | — |

The plain arms test tokens 1, 2 and 3 and tokens 1–2 and 1–3 jointly over 4,000 requests each (five tokens after a
text prompt, temperature 1, top-k 8); the grammar arms the tool call's outcome (tool and digit) over 1,500 requests.
The random drafter's own tokens are never ones the grammar allows, so the grammar arms with a drafter reject at the
first position every step: they test the residual under a grammar (and, for `gram force`, that nothing is appended
after a rejected token); `gram keep0` tests forced tokens verified by block verification.
