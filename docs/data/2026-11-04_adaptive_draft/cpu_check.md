# Q687's CPU check (2026-11-04, quant machine, Docker, no GPU)

[run/cpu_check.sh](run/cpu_check.sh) on fork branch `pascal-p106-v3-b12076-adraft` `b52064398`, against its parent
`pascal-p106-v3-b12076-ladder2` `a330672c1` built the same way. The model is a random-weight qwen4exp (Flash-Next's
architecture, from llama.cpp's `test-llama-archs`) as target and as its own draft-simple drafter,
`--spec-draft-n-max 4` (8 for `on8`), one slot, 6 CPU threads.

## Verdicts

| Check | Verdict | Detail |
|---|---|---|
| build | OK | CPU backend, static: `llama-server`, `test-spec-adapt`, `test-llama-archs` of `b52064398`, `llama-server` of `a330672c1` |
| unit | OK | `test-spec-adapt` 13,507 / 13,507: 4,000 random drafts and convex cost tables (rule = exhaustive search of its objective), 1,440 of them with every width allowed (rule = the literal per-position rule) |
| smoke-off | OK | 9 / 9 replies of `off`, `log` and `on-all` with the base build's tokens and draft counts |
| smoke-lossless | OK | 16 / 16 greedy replies of the eight drafting arms equal to plain decoding (`nospec`) |
| smoke-log | OK | 6 / 6 logs pass `adapt_log.py --check` (828 steps, every choice and running S recomputed, 0 problems); the switched-on arms verify at widths 1, 3 and 5 only |
| smoke-bad | OK | `LLAMA_SPEC_ADAPT_COST=136,abc` stops the server at its start ("'abc' is not a number") |

The build stage is the script's previous run (the same build code); the unit and smoke stages ran again with
`STEPS="unit smoke"`, after the `on-all` arm got a fixed S: at the running S, scale 0.001 alone still cut some drafts.

## The requests

| Arm | R1 greedy: tokens, drafted, accepted | R2 temperature 1.0 | R3 greedy, longer prompt |
|---|---|---|---|
| `nospec` (plain decoding) | 96, –, – | 48, –, – | 64, –, – |
| `base` (`a330672c1`) | 96, 76, 76 | 48, 82, 4 | 64, 50, 50 |
| `off`, `log`, `on-all` | 96, 76, 76 | 48, 82, 4 | 64, 50, 50 |
| `on`, `on-step`, `on-fixed` | 96, 0, 0 | 48, 0, 0 | 64, 0, 0 |
| `on8` | 96, 38, 38 | 48, 42, 3 | 64, 24, 24 |

"Drafted" is the server's `draft_n`, the drafted tokens verified. Every arm's R2 reply also equals plain decoding's
here.

## The logs (`adapt_log.py`)

| Arm | Switch | S | Steps | Steps by verify width | Verified per step | Accepted per step | Model speed-up | The rule on the same drafts |
|---|---|---|---|---|---|---|---|---|
| `log` | off | run | 53 | 3: 2, 5: 51 | 3.92 | 2.45 | 1.25 | 1: 53 (0.87) |
| `on` | on | run | 202 | 1: 202 | 0 | 0 | 0.87 | the same |
| `on-step` | on | step, cheap table | 202 | 1: 202 | 0 | 0 | 0.83 | the same |
| `on-fixed` | on | 0.5, scale 0.5 | 202 | 1: 202 | 0 | 0 | 0.87 | the same |
| `on8` | on | run, cheaper table | 116 | 1: 64, 3: 52 | 0.90 | 0.56 | 1.51 | the same |
| `on-all` | on | 0.001, scale 0.001 | 53 | 3: 2, 5: 51 | 3.92 | 2.45 | 1.25 | the same |

The model speed-up is T(1) × tokens per step / modelled step cost, from each arm's own cost table (the default
table for `log`, `on`, `on-fixed` and `on-all`). Width 3 in `log` and `on-all` is two steps near the end of R3,
where the context leaves room for two drafted tokens only. The random drafter's probabilities sit at 0.10–0.104 (its
top 10 are nearly tied), so the rule keeps nothing on the default table, while the drafter, being the target, is
right on the greedy requests: on these drafts the rule would turn a modelled 1.25× into 0.87×. A drafter's
probabilities have to track its acceptance for the rule to gain (Q760); DFlash2's and the MTP draft's are measured on
the rig (C134r's logs).

Requests: R1 is a 130-token prompt (a 40-token block repeated, so a drafter has something to copy), 96 tokens; R2 the
same prompt at temperature 1.0, top-k 20, top-p 0.95, seed 42, 48 tokens; R3 a 160-token prompt, 64 tokens. The model's
context is 256 tokens.
