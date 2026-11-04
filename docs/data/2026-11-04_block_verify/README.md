# Q797 and Q802: block verification and the tool-call grammar's forced tokens (2026-11-04)

Two exact changes to how llama-server verifies a draft, from the [acceptance note](../../research/notes/flash_next_acceptance.md)
(its ideas 7 and 13, section 4.1 and 4.5):

- **Block verification (Q797).** With probabilistic draft sampling (`--spec-draft-sampling probabilistic`, Q469, Q731)
  the drafted tokens are verified by rejection, one at a time: the first rejected token ends the step. Block
  verification ([Sun et al., "Block Verification Accelerates Speculative Decoding", arXiv:2403.10444](https://arxiv.org/abs/2403.10444),
  Algorithm 2) decides the drafted block jointly. It keeps the target's output distribution exact, and its expected
  number of accepted tokens is never below token-by-token verification's (the paper proves it optimal among exact
  verifiers that see the draft's and the target's distributions along the drafted block). The note estimates
  +3–6 % accepted length for DFlash2 on top of rejection.
- **Forced tokens (Q802).** While a tool-call grammar applies, much of what comes next is fixed by it: the XML tags of
  Qwen3.8's tool calls, a parameter's name, a tool's name once its first letters are written. Those tokens cost nothing
  to guess, so llama-server appends them to the draft and the verify pass checks them with the drafted ones, also when
  the drafter drafted fewer tokens than the width, or none. Est. +1–2 %.
- **A third change came out of the check (Q808).** After a checkpoint replay (a draft longer than a recurrent target's
  rollback snapshots, or a target that cannot remove part of a sequence) llama-server restores the sampler copied
  before verification, its random draws included, so token-by-token rejection draws again the uniforms of the step it
  replays. Block verification keeps them used; `LLAMA_SPEC_KEEP_DRAWS=1` does the same for token-by-token rejection.

All three are behind switches that are off by default; with them off the server's tokens and draft counts are the base
build's. Greedy requests (temperature 0, verified by sample-and-match) are never verified differently. Registry rows
Q797, Q802, Q808 (and Q809, Q810, ideas from this work) in [quants/experiments.md](../../../quants/experiments.md).

**Status:** built for the rig, checked on the CPU (below). The rig run, S18's DFlash2 line on the agent check with
probabilistic sampling, against block verification, against forced tokens, is [run/rig_blockver.sh](run/rig_blockver.sh).

## The switches

| Variable | Effect |
|---|---|
| `LLAMA_SPEC_BLOCK_VERIFY=1` | sampled drafts (probabilistic draft sampling, temperature > 0) are verified as one block; greedy requests and requests with backend sampling keep their verification |
| `LLAMA_SPEC_FORCE=1` | while a grammar applies, the tokens of the text it forces after the draft are appended, up to the drafter's width (`--spec-draft-n-max`) |
| `LLAMA_SPEC_FORCE_CHARS=<n>` | the forced text read per step, in characters (default 64) |
| `LLAMA_SPEC_FORCE_MAX=<n>` | at most n forced tokens per step (default: the width) |
| `LLAMA_SPEC_DRAFT_KEEP=<n>` | every draft cut to its first n drafted tokens, so forced tokens get the rest of the width (default: no cut) |
| `LLAMA_SPEC_KEEP_DRAWS=1` | token-by-token rejection keeps its random draws used across a checkpoint replay (block verification always does) |

A malformed value stops the server at its start. The server logs one line with the settings when any is set, and a
per-task line `forced tokens = <steps> steps, <n> tokens the grammar forced appended to drafts, <k> accepted`.

## Block verification

For a drafted block X_1 … X_g with draft distributions M_s(· | X^i) and target distributions M_b(· | X^i) (after the
target's sampler chain and the grammar's mask, as token-by-token rejection uses them), Algorithm 2 computes the weights

    p_0 = 1,   p_i = min(1, p_(i-1) · M_b(X_i | X^(i-1)) / M_s(X_i | X^(i-1)))        (Equation 9)

and the acceptance probabilities

    h_g = p_g,   h_i = S_i / (S_i + 1 − p_i)  with  S_i = Σ_x max(0, p_i · M_b(x | X^i) − M_s(x | X^i))   (Equation 5)

then accepts the first τ drafted tokens, τ the largest i with η_i < h_i for independent uniforms η_i (0 if none), and
draws the next token from M_b(· | X^g) when τ = g, else from the residual max(0, p_τ M_b(· | X^τ) − M_s(· | X^τ)),
normalized (Equation 4). Where token-by-token rejection stops at the first rejected token, block verification can
accept a later prefix: a token the target likes less than the drafter is carried by the ones after it that the target
likes more. The paper's worked example (vocabulary {a, b}, M_b = (1/3, 2/3), M_s = (2/3, 1/3), two drafted tokens)
accepts 11/9 tokens in expectation against 10/9.

**In llama-server** (`common_sampler_sample_and_accept_n_block` in `common/sampling.cpp`): the decision needs the
target's distribution at every drafted position, each after the drafted tokens before it, because the penalties, the
reasoning budget and the grammar follow the accepted text. The function walks the block: at each position it samples
(which leaves the candidates, as token-by-token rejection reads them), masks and renormalizes them by the grammar, then
accepts the drafted token into the sampler and moves on; it stops at a token the target cannot produce (p_i = 0: no
later position can be accepted). Then it draws τ, puts the sampler's state back and accepts only X^τ and the next token.
Two details keep it exact:

- **The draws stay used.** The uniforms come from the sampler's own stream, and the restore keeps the chain's draws
  (its `dist` and `xtc` samplers) where the walk left them. With a grammar, which candidates a position has depends on
  its draw (the drawn token is checked first, and the masked row is resampled on a rejection), so the decision depends
  on the draw; a draw handed back would tie a later token to it. The server's checkpoint replay (a recurrent target
  whose draft exceeds its rollback snapshots) keeps them too when the switch is on.
- **The arithmetic** (`common_spec_block_compute`, `common_spec_block_residual`) is separate from the sampler and is
  what the unit test checks. Where p_i = 1 and S_i = 0 (target and draft agree at position i) h_i takes its limit 1;
  a residual without mass, which only rounding leaves, falls back to M_b(· | X^τ), its limit.

The cost over token-by-token rejection is the walk past the first rejection (one sampling pass per drafted position,
which token-by-token rejection skips after a rejection) and a copy of the sampler's state and its restore (the
samplers, the grammar and the history, without the candidate array).

## Forced tokens

`llama_grammar_forced_text` (`src/llama-grammar.cpp`) walks the grammar's stacks while every stack requires the same
single character or the same token, and returns that text; it stops where the stacks disagree, at a character class,
where the grammar may end, mid-way through a multi-byte character, and returns nothing for a lazy grammar before its
trigger. `common_sampler_grammar_forced_text` runs it on a copy of the sampler that has taken the drafted tokens (so
the grammar's state is the one after the draft); llama-server's `spec_force_extend` then:

1. cuts the draft to `LLAMA_SPEC_DRAFT_KEEP` tokens, if set;
2. returns unless the grammar applies after the history (a lazy grammar before its trigger, most of the output, costs
   nothing), the draft is shorter than the width, and the grammar accepts every drafted token (a token it rejects is
   certain to be rejected, and the draft stays as it was);
3. tokenizes the forced text as plain text (special tokens parsed), checks that the pieces spell it, and holds back
   the last token unless the text ends at a newline or with a special token: the text after it may join it into a
   longer token;
4. appends up to the width (`--spec-draft-n-max`, which sizes the target's outputs per step and a recurrent model's
   rollback snapshots), with a point-mass proposal for each appended token when the drafter gave its distributions.

**Exact under every verification:** whether and what is appended depends only on the text before the appended
position (the history and the drafted tokens kept), so each appended token is a proposal whose distribution, given
the prefix, is a point mass on it; the cut is a fixed one. Rejection and block verification are exact for such
proposals, and a draft length that depends on the drafted prefix (unit test: "length set by the prefix") as well.

**What it can gain on the rig:** with `--spec-draft-n-max 2` DFlash2 always drafts two tokens, so forced tokens have
no room. The rig arm runs `--spec-draft-n-max 4` with every draft cut to two (`LLAMA_SPEC_DRAFT_KEEP=2`): S18's verify
width of 3 outside tool calls, up to 5 inside, where the forced tokens fill it. Its control is the same line without
`LLAMA_SPEC_FORCE` (the same buffers, the same widths outside tool calls).

## CPU check

[run/cpu_check.sh](run/cpu_check.sh) on a build machine in Docker, CPU backend, no GPU and no real model; the verdicts
are in [cpu_check.md](cpu_check.md). The models are random-weight qwen4exp files (Flash-Next's architecture) from
`test-llama-archs`, which gains `LLAMA_ARCHS_VOCAB`: the smoke stage's pair carries Flash-Next's own tokenizer and chat
template (248,320 tokens, read from a Flash-Next GGUF's header), the statistical stage's pair GPT-2's tokenizer (50,257
tokens: under a grammar every request checks whole rows the sampler rejects, several times cheaper with the smaller
vocabulary). Each drafter is its target's weights at another scale (the same seed): a different distribution,
correlated with the target's (smoke: stdev 0.13 against 0.1; statistical stage: 0.8 and 0.95 against 1.0). Every
drafting arm runs draft-simple at `--spec-draft-n-max 4` with `--spec-draft-sampling probabilistic
--spec-draft-p-min 0`.

### Unit test: `test-spec-block`, 47/47

**The paper's worked example** (vocabulary {a, b}, M_b = (1/3, 2/3), M_s = (2/3, 1/3), two drafted tokens) as the
paper states it: draft aa is accepted whole with probability 1/4 and otherwise corrected to b; ab and bb are accepted
whole; ba keeps b always and a with probability 1/2, else the second token is corrected to b. Expected accepted tokens
11/9 against token verification's 10/9.

**Exactness by exact enumeration.** For each instance (random context-dependent targets and drafts over 3–4 tokens),
every draft is enumerated with its probability, every accepted length with its probability (τ = i with probability
h_i · Π_(j>i) (1 − h_j)), the next token from M_b or the residual, then the target's own continuation to g + 1 tokens:
the result must be the target's joint distribution of g + 1 tokens. Max error is float rounding (the distributions are
stored as `llama_token_data`, in float):

| Instances | g | max \|P_out − P_target\| | E[accepted], block | token by token | Gain |
|---|---:|---:|---:|---:|---:|
| dense, 3 tokens, 300 | 1 | 9.2e-08 | 0.509 | 0.509 | 0 (the same rule at g = 1) |
| dense, 3 tokens, 300 | 2 | 9.2e-08 | 0.808 | 0.777 | +4.0 % |
| dense, 3 tokens, 300 | 3 | 1.2e-07 | 1.044 | 0.958 | +8.9 % |
| dense, 3 tokens, 100 | 4 | 9.3e-08 | 1.120 | 0.993 | +12.8 % |
| dense, 4 tokens, 100 | 3 | 7.7e-08 | 0.924 | 0.844 | +9.4 % |
| target zeros (a grammar's mask), 200 | 3 | 9.0e-08 | 0.609 | 0.568 | +7.2 % |
| drafts of top-2 support, 200 | 3 | 1.1e-07 | 0.674 | 0.647 | +4.2 % |
| zeros and top-2, 200 | 3 | 9.5e-08 | 0.597 | 0.573 | +4.3 % |
| point-mass proposals (forced tokens), 200 | 3 | 9.8e-08 | 0.698 | 0.661 | +5.5 % |
| draft length set by the prefix, 200 | ≤ 4 | 1.0e-07 | 0.764 | 0.719 | +6.3 % |
| all of these, 200 | ≤ 4 | 1.2e-07 | 0.548 | 0.533 | +2.8 % |

Block verification accepted no fewer tokens in expectation than token-by-token rejection on every instance (the
smallest gap −1.6e-08, rounding). The gains are for random toy pairs, not a forecast: they grow with the block, as the
paper's do. **The grammar's forced text** on small grammars: the fixed prefix, a tool name forced after its first
letter, where two names are left, up to where the grammar may end, a character class, multi-byte characters, mid-way
through a character, and a lazy grammar before its trigger.

### Smoke stage: llama-server on six requests per arm

Requests: R1 greedy text, R2 greedy JSON tool call under a grammar ([run/json_tool.gbnf](run/json_tool.gbnf)), R3
greedy tool call in Qwen3.8's XML format with the special token `<tool_call>` ([run/xml_tool.gbnf](run/xml_tool.gbnf));
R4–R6 the same at temperature 1 (top-k 8). Arms: plain decoding, the base build `a330672c1`, the new build without
switches, block verification, forced tokens without and with it, drafts cut to 0 and to 2 with both.

| Check | Result |
|---|---|
| switches off against the base build | 6/6 replies with the base build's tokens and draft counts, greedy and sampled |
| greedy against plain decoding | 21/21 replies of the seven drafting arms equal plain decoding |
| grammar replies | 32/32 whole (valid JSON for R2/R5, the whole XML tool call for R3/R6) |
| forced tokens | drafts cut to 0: 39 steps, 128 forced tokens appended, 55 accepted over the six requests (with the random drafter's own tokens, which a grammar almost never allows, nothing is appended after them) |
| malformed value | `LLAMA_SPEC_FORCE_CHARS=abc` stops the server at its start |

Draft acceptance at temperature 1 on R4 (text): token-by-token rejection 21 of 56 drafted tokens over the request,
block verification 24 of 49 (fewer, longer steps).

### Statistical stage: sampled outputs against plain decoding

Each arm is a llama-server with its own seeds (so the arms' samples are independent), its outputs compared with plain
decoding's by two-sample chi-square tests ([run/stat_check.py](run/stat_check.py), cells pooled to expected counts of
at least 5): the first five tokens after a text prompt, 4,000 requests per arm (tokens 1, 2, 3 and 1–2, 1–3 jointly;
the first token comes from the prompt pass, the rest from drafts of three), and a short JSON tool call under a grammar
([run/stat_tool.gbnf](run/stat_tool.gbnf), its outcome: three tools by ten digits), 1,500 requests per arm. The models
have GPT-2's tokenizer and weights of stdev 1.0, whose rows are peaked (at the smoke stage's 0.1 the top 8 of a row
are near uniform, and temperature 0.8 could not be told from 1.0: the first run of this stage showed it). Two drafters:
SD (stdev 0.8, accepts few: the residual path) and SD2 (stdev 0.95: drafts accepted in part, so the weights p_i lie
between 0 and 1). Draft-simple on this hybrid target restores a checkpoint and replays after every partial acceptance
(it has no rollback snapshots). All rows: [stat.tsv](stat.tsv); verdicts: [cpu_check.md](cpu_check.md).

| Arm | Smallest p (tests) | Accepted / drafted |
|---|---:|---:|
| block verification, drafter SD | 0.22 (5) | 0.0275 |
| block verification, drafter SD2 | 0.086 (5) | 0.2207 |
| token by token with `LLAMA_SPEC_KEEP_DRAWS=1`, SD2 | 0.014 (5; token 1, drawn before any draft) | 0.2023 |
| block verification under the grammar | 0.81 (1) | 0 |
| forced tokens with block verification under the grammar | 0.073 (1) | 0 |
| forced tokens only, block verification, under the grammar | 0.65 (1) | 0.4153 |
| token by token with `LLAMA_SPEC_KEEP_DRAWS=1` under the grammar | 0.24 (1) | 0 |
| token by token as the base build has it, SD / SD2 | 0.058 / 0.0049 (5 / 5) | 0.0268 / 0.2017 |
| **token by token as the base build has it, under the grammar** | **6.0e-09 (1), total variation 0.095** | 0 |
| control: plain decoding at temperature 0.8 | 1.7e-21: told apart | — |

**Every exact arm passes** (19 of 19 tests at p ≥ 1e-4), and the control is told apart, so the test sees a shift of
that size. Block verification accepted 9 % more drafted tokens than token-by-token rejection with the drafter SD2
(0.2207 against 0.2017 per drafted token), as the toy pairs of the unit test predict at this block and acceptance.

**The finding: token-by-token rejection as the base build has it is not exact after a checkpoint replay** (Q808).
The grammar arm shows it at p = 6e-9: the replay restores the sampler copied before verification, its uniforms and
the chain's draws included, and under a grammar which candidates a position has depends on its draw (the drawn token
is checked first, the masked row resampled on a rejection), so the step after the replay repeats draws whose outcome
the output already depends on. With `LLAMA_SPEC_KEEP_DRAWS=1` the same arm passes (p = 0.24), and block verification
always keeps its draws. The plain arms do not show it at this sample size (their smallest p 0.0049). The rig's DFlash2
and MTP lines do not replay (their drafts stay within the target's rollback snapshots, `--spec-draft-n-max` of them),
so its arms are not affected; draft-simple and the n-gram drafters on a hybrid model, and any target that cannot
remove part of a sequence, are.

## The rig run

[run/rig_blockver.sh](run/rig_blockver.sh): S18's DFlash2 line ([Q561](../2026-11-03_flash_next_dflash2/README.md):
Swift-1.5 GSQ-RCO Q2_0 with Inco AI's DFlash2 drafter, `-b 512`, nine cards) on the agent check, every arm with
`--spec-draft-sampling probabilistic --spec-draft-p-min 0` (the agent samples at temperature 1.0, top-p 0.95,
top-k 20, so every arm verifies by rejection), each arm smoke-tested first, then 40 minutes of the timed check:

| Arm | `--spec-draft-n-max` | Switches | Compares |
|---|---|---|---|
| `prob` | 2 | none: token-by-token rejection | Q731's arm, against S18's `dflash2b` (sample-and-match) |
| `block` | 2 | `LLAMA_SPEC_BLOCK_VERIFY=1` | block against token-by-token verification |
| `keep4k2` | 4, drafts cut to 2 | `LLAMA_SPEC_BLOCK_VERIFY=1 LLAMA_SPEC_DRAFT_KEEP=2` | the control for `force4k2`: `block`'s widths on buffers sized for 4 |
| `force4k2` | 4, drafts cut to 2 | the same plus `LLAMA_SPEC_FORCE=1` | forced tokens filling the width inside tool calls (verify width up to 5) |

What decides: minutes per task at equal tokens (S18's method) and accepted tokens per verify pass, from the report
against S18's `nodraft` and `dflash2b` arms; for `force4k2` also `forced.tsv` (steps with forced tokens, tokens
appended and accepted per task). If the n-max-4 arms meet the RAM watchdog in their smoke tests (S18's smoke at n-max 4
did, with `-b 2048`), `ARMS="keep3k2 force3k2"`. Its usage is in the script's header.

## Files

| File | What |
|---|---|
| [cpu_check.md](cpu_check.md) | the CPU check's verdicts and the statistical stage per arm |
| [stat.tsv](stat.tsv) | every chi-square test: arm kind (exact, the base build's path, control), samples, statistic, degrees of freedom, p, total variation |
| [acceptance.tsv](acceptance.tsv) | drafted and accepted tokens per arm of the statistical stage |
| [run/cpu_check.sh](run/cpu_check.sh) | the CPU check: build, unit test, the random models, smoke and statistical stages, verdicts |
| [run/smoke.sh](run/smoke.sh), [run/stat.sh](run/stat.sh), [run/serve_lib.sh](run/serve_lib.sh) | its stages inside the container |
| [run/stat_check.py](run/stat_check.py) | the chi-square tests of the statistical stage |
| [run/json_tool.gbnf](run/json_tool.gbnf), [run/xml_tool.gbnf](run/xml_tool.gbnf), [run/stat_tool.gbnf](run/stat_tool.gbnf) | the grammars of the smoke and statistical stages |
| [run/rig_blockver.sh](run/rig_blockver.sh) | the rig run on the agent check |

The unit test is the fork's `tests/test-spec-block.cpp`; `test-llama-archs` gained `LLAMA_ARCHS_VOCAB` and
`LLAMA_ARCHS_N_CTX` for these models.

## Provenance and licence

The fork's branch `pascal-p106-v3-b12076-blockver` at `b8f42a3ff`, two commits on `pascal-p106-v3-b12076-ladder2`
`a330672c1`: `308cd232b` (block verification, forced tokens) and `b8f42a3ff` (`LLAMA_SPEC_KEEP_DRAWS`). The CPU check
uses random-weight models with Qwen3.8-Flash-Next's tokenizer and chat template, read from the header of a Flash-Next
GGUF, and with GPT-2's tokenizer (`models/ggml-vocab-gpt-2.gguf` of llama.cpp); no model output or weights are
published. The tables here are this project's measurements (MIT, as the repository).
