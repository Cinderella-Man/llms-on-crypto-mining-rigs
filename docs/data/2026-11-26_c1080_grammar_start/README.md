# C1080: when b12674's tool-call grammar starts, Flash-Next (2026-11-26)

Registry row [Q1813](../../../quants/experiments.md), from Flash-Next's coding-agent check on upstream b12674
([2026-11-24_c912_flash_agent](../2026-11-24_c912_flash_agent/README.md), `grammar.tsv`). On the b12674 candidate the
server's sampling line counted every token Flash-Next sampled as sampled while the tool-call grammar applied (32,088 of
32,088), where `pascal-p106-v5` `af00bb13b` counted only the tokens after a tool call opened (6,715 of 26,635). The
question: does the grammar now constrain the reasoning, or did only the count move?

**Only the count moved.** Upstream's token-aware parsing (#29307, in b12674) rewrote the Qwen3-Coder chat handler that
Flash-Next's template uses. Its tool-call grammar is no longer lazy for the sampler: the grammar itself scans for the
`<tool_call>` token, so the sampler runs it from the first token. The fork's sampling line counts every token the
grammar checks, which is now every token.
- **Before the first `<tool_call>` the grammar holds nothing back that is text.** It lets through every token that
  spells valid text and every end-of-turn token. At every step it leaves out the same 407 of the vocabulary's 248,320
  tokens: 163 bytes that cannot start a UTF-8 character, one NUL byte, and 243 ids with no text at all. Right after the
  first bytes of a character split over tokens, it lets through only the bytes that finish that character.
- **The replies and the drafts do not change.** On three recorded agent-check turns, the agent's request (the grammar
  on) and the same request with `"tool_choice": "none"` (the same prompt, no grammar) give the same greedy tokens on
  each build: 75, 1,138 and 114 tokens on the candidate. A repeat of the request and the request without the grammar
  draft and accept the same number of tokens. In 2,654 grammar steps on the candidate the grammar never rejected the
  reply's own token.
- **Today's engine does the same.** `pascal-p106-v5` `2d839da1d` gives the candidate's replies, drafts and grammar
  states token for token. The b12723 candidate's Qwen3-Coder handler is the same file.

Ornith's template has the same Qwen3-Coder markers, so its count changed for the same reason. Its run needs three
cards and is registry row Q1883.

## What runs

| Label | Engine | Rig tarball md5 | `llama-server` md5 |
|---|---|---|---|
| `cand` | the b12674 candidate `pascal-p106-v5-cand-b12674` `eee615cdb` (C912) | `575ef3c702f69f14d7d3022c1287c38f` | `2727faf28b5b34dfe42399a9d431da67` |
| `ref` | `pascal-p106-v5` `af00bb13b` (before b12674; C912's control) | `b51ad2740d5822e7e14af656c413ec22` | `f5cca34ea6bd4f3712cd6193bf0605be` |
| `land3` | `pascal-p106-v5` `2d839da1d` (2026-11-25, with b12674) | `7d774f05c29460a6993b19d38796047c` | `ab3bc3b42fbdbaae5f43869970c05736` |

- **Model:** the maker's Swift GSQ-RCO Q2_0 of Qwen3.8-Flash-Next (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf`,
  the v5 preset's file, as in C912) with Swift's own MTP head as the draft
  (`mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf`, md5 `e0aa166bbd5328f39026f2f42b458ff6`; the rigs' preset uses
  another MTP draft file, as in C947's build-machine checks).
- **Settings:** Flash-Next's v5 preset ([release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env))
  as it is, with the layer split changed to two cards (`-ts 1,1`) and verbose logging (`-v`). The fork's compute-6.1-only
  paths stay at their defaults, off on these cards: this checks the server and the sampler, not the kernels.
- **Cards:** two RTX 3090s of the build machine, which run the rig build's compute 6.1 PTX. Correctness only, nothing timed.
  Each server loaded in 41 s.
- **Requests:** three turns of C912's agent runs, as the agent harness's proxy recorded them
  ([bench/agent_check](../../../bench/agent_check/README.md)), all from the task `bugfix-python-affine-cipher`:

  | Name | Recorded in | Turn | What the model did there in C912 |
  |---|---|---|---|
  | `t1_first` | `f5cand` and `f5ref` (the same body) | 1 (2,780 prompt tokens) | two tool calls after a one-line reasoning |
  | `t2_reason` | `f5ref` | 2 (3,757 prompt tokens) | 788 tokens of reasoning and text, then one call |
  | `t5_final` | `f5ref` | 5 | a final answer with no tool call |

  The bodies (the harness's system prompt, the task's text and the model's earlier replies) are not in this folder.
  Their md5s: `f4a0e4be6825eda4ee486e2c61c2a3da`, `5b34f46e32bd956102d88ac1f4a12a64`,
  `2345e3a29d34c1f89fccb557b4f2218e`.
- **Arms** ([run/run.sh](run/run.sh)): every request is greedy (temperature 0, top-k 1, seed 1, no prompt cache, at most
  4,096 tokens), in this order on each server:

  | Arm | Request |
  |---|---|
  | `chat` | the agent's request with its four tools: the tool-call grammar on |
  | `chat2` | the same request again |
  | `none` | the same request with `"tool_choice": "none"`: the same chat path and prompt (the template still lists the tools), no grammar |
  | `plain` | the prompt the chat template renders (`/apply-template`), sent to `/completion` with no grammar |

## The grammar each build builds

[params.tsv](params.tsv), [grammar_cand.gbnf](grammar_cand.gbnf), [grammar_ref.gbnf](grammar_ref.gbnf). Both builds
take llama.cpp's Qwen3-Coder handler for Flash-Next's template (the server's log: "Using specialized template:
Qwen3-Coder"). The grammar depends only on the tools, so each build built one grammar for all three requests.

| | `ref` (`af00bb13b`) | `cand` and `land3` (b12674) |
|---|---|---|
| lazy (the sampler waits for a trigger) | yes, trigger token 248058 (`<tool_call>`) | no |
| how the grammar starts | `root ::= tool-call-root`: the call itself, applied once the trigger is seen | `root ::= trigger`, with `trigger ::= \| <[248058]> tool-calls-rest \| !<[248058]> trigger`: any token but `<tool_call>` keeps scanning, `<tool_call>` starts the call, and the grammar may end anywhere |
| the generation prompt fed to the grammar first | none (a lazy grammar is not fed it) | `<\|im_start\|>assistant\n<think>\n` |
| when the server applies it | after the reasoning ends (the reasoning-budget sampler holds a lazy grammar off inside `<think>`), waiting for the trigger | from the first token |

## The grammar's state at every token

[run/grammar_steps.cpp](run/grammar_steps.cpp) walks each chat reply through the grammar the server built, as the
server's grammar sampler does: before each token it applies the grammar to the whole vocabulary and counts the tokens
it lets through, then accepts the reply's token. [steps.tsv](steps.tsv) sums each reply,
[left_out.tsv](left_out.tsv) classifies the tokens left out, and [tokens.tsv.gz](tokens.tsv.gz) has every step.

| Build | Request | Tokens | First `<tool_call>` at | Steps before it: whole vocabulary / fewer | Steps from it on: whole vocabulary / fewer | Reply's token rejected |
|---|---|---|---|---|---|---|
| `cand` | `t1_first` | 75 | 10 | 0 / 10 | 0 / 65 | 0 |
| `cand` | `t2_reason` | 1,138 | 1,011 | 0 / 1,011 | 0 / 127 | 0 |
| `cand` | `t5_final` | 114 | 85 | 0 / 85 | 0 / 29 | 0 |
| `ref` | `t1_first` | 83 | 10 | 10 / 0 | 1 / 72 | 0 |
| `ref` | `t2_reason` | 417 | 387 | 387 / 0 | 1 / 29 | 0 |
| `ref` | `t5_final` | 348 | none | 348 / 0 | none | 0 |
| `land3` | (each request) | the candidate's | | | | 0 |

- **On the candidate, what "fewer" means before the call.** At 1,101 of the 1,106 steps before the first call the grammar
  left out the same 407 tokens: 163 single bytes or byte runs that cannot start a UTF-8 character, the NUL byte, and
  243 ids with no text. It left out no end-of-turn token and no token that spells valid text. At the other 5 steps it
  let through only 83 tokens: each came right after the token `' \xe2\x89'`, the first two bytes of `≡`, where only
  a byte that finishes the character is valid, and the model's next token (`\xa1`) was one.
- **From the call on, both builds constrain the call's syntax:** at the call's fixed text the grammar lets through
  one to a few tokens; inside an argument's free text it leaves out 414 to 447 tokens (the malformed bytes, the NUL
  byte and the ids with no text, the 5 end-of-turn tokens, and where the argument may close, 33 runs of whitespace
  with two or more newlines).
- **The old build's server confirms its own state** ([server_lines.tsv](server_lines.tsv)): its debug log shows the
  grammar fed only after the reasoning ended, waiting for the trigger (3, 3 and 316 tokens), then triggered on token
  248058, and its sampling line counts 72 and 30 tokens after the trigger. The candidate's line counts every token
  (75, 1,138, 114). No build's line counts a rejected token.
- `chat2` walks through the same states as `chat` on every reply.

## The replies and the draft

[arms.tsv](arms.tsv): tokens, how the reply ended, the draft's tokens and acceptance, and where each arm's reply first
differs from the build's `chat` reply.

| Build | Request | `chat` tokens | `chat2` | `none` | `plain` differs from token | drafted / accepted: `chat`; `chat2`; `none` |
|---|---|---|---|---|---|---|
| `cand` | `t1_first` | 75 | same | same | 31 | 63 / 53; 63 / 53; 63 / 53 |
| `cand` | `t2_reason` | 1,138 | same | same | 6 | 1,080 / 777; 1,083 / 776; 1,083 / 776 |
| `cand` | `t5_final` | 114 | same | same | 14 | 120 / 73; 123 / 72; 123 / 72 |
| `ref` | `t1_first` | 83 | same | same | 31 | 66 / 60; 66 / 60; 66 / 60 |
| `ref` | `t2_reason` | 417 | same | same | 6 | 381 / 290; 372 / 293; 372 / 293 |
| `ref` | `t5_final` | 348 | same | same | 32 | 321 / 240; 312 / 243; 312 / 243 |
| `land3` | each | the candidate's | same | same | the candidate's | the candidate's |

- **With and without the grammar the reply is the same** on each build (`chat` against `none`), and so are the draft's
  counts (`chat2` against `none`, run one after the other). So the grammar changes neither the text nor the draft's
  acceptance. C912's higher acceptance on the candidate (0.781 against 0.742) comes from the text sampled at
  temperature 1.0 and from the engine, not from the grammar.
- **`chat` drafts a few tokens differently from its repeat** when the request before it was on another prompt
  (`t2_reason`, `t5_final`; on `t1_first`, the first request after the load, they agree), with the same tokens, on both
  builds. The grammar plays no part (the repeat and `none` agree). Registry row Q1885 finds out why.
- **`plain` is not a grammar comparison:** the chat path reads a prompt in calls cut at its user messages (for its
  checkpoints), `/completion` does not, so the numbers differ slightly and greedy replies part at a near-tie (tokens 6
  to 32). The builds' replies part from each other the same way (the upstream merge changed the engine's arithmetic):
  `ref`'s `chat` replies leave `cand`'s at tokens 31, 6 and 14.

## Which upstream commit

**#29307, "chat : implement token aware parsing" (upstream `1014775c8a`, in b12674).** In `common/parsers/qwen3-coder.cpp`
it replaced `data.grammar_lazy = !has_response_format && has_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO`
and the word triggers with `data.grammar_lazy = false`, under the comment "The grammar scans for the triggers itself, so
the sampler runs it from the first token", and builds the grammar with `trigger_rule(name, start, rest)`. Upstream's
`docs/development/parsing.md` (same commit) says such a grammar "runs from the first token, so `grammar_lazy` is false
and no trigger patterns are needed". It is the only commit between `af00bb13b` and `eee615cdb` that touches
`qwen3-coder.cpp` or changes a `grammar_lazy` line in `common/`, and the candidate's file is upstream b12674's byte for
byte. **#32497** adds a pattern trigger only in the autoparser's path for bare JSON calls with no start marker, which
Flash-Next's template does not take. The code and the logged parameters settle it, so no bisect build was made.

## What else changes with it

- **Well-formed UTF-8 from the first token.** The grammar now rejects, also in the reasoning, a token that would start
  malformed UTF-8, a NUL byte or an id with no text; before b12674 it did so only after a call opened. No reply here
  needed it. In C912's runs the server rejected 1 of the candidate's 32,088 sampled tokens and 1 of the control's 6,715
  constrained ones.
- **No reasoning-budget sampler for these requests.** The server builds it for an unlimited budget only when the grammar
  is lazy (to hold the grammar off inside `<think>`). With the budget unlimited it forced no token on either build.
- **The grammar runs on the host for every token.** The rigs' decode step was not slower in C912 (117.02 against 118.80
  ms over all requests); this folder times nothing.
- **The count in the agent tables** ([grammar.tsv](../2026-11-24_c912_flash_agent/grammar.tsv)) no longer says where
  calls start. Registry row Q1884 gives the sampling line a count from the first `<tool_call>` token on.

## Exact commands

On the build machine, with its paths in the environment (one container per server, labelled, `--cpu-shares=256`, the
drives mounted at the same paths):

```bash
# the three servers and the four arms per request, detached
WORK=<out> BUILDS='cand=<eee615cdb>/bin ref=<af00bb13b>/bin land3=<2d839da1d>/bin' \
REQS='<req>/t1_first.json <req>/t2_reason.json <req>/t5_final.json' REPO=<a copy of this repository> \
FN=<Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00003.gguf> DRAFT=<mtp-Swift-1.5-Qwen3.8-Flash-Next-draft.gguf> \
GPUS=0,1 TS=1,1 MOUNTS='<drives>' DOCKER_ARGS='--label <labels> --cpu-shares=256' PREFLIGHT=<cuda_preflight.sh> \
JIT_SEED='cand=<jit cache> ref=<jit cache> land3=<jit cache>' \
  setsid nohup bash run/run.sh > driver.log 2>&1 < /dev/null &
# the grammar walker, against the candidate's build tree (its libllama.a: md5 a93d6a02f56699db254f54c3706fb48d)
TREE=<the candidate's build folder> OUT=<tool> LOCK=<build lock> DOCKER_ARGS='...' bash run/build.sh
# every chat reply through it (grammar-steps md5 f17ab8dc83bedbbd38da3c49144ee078)
WORK=<out> TOOL=<tool>/grammar-steps MODEL=<FN> MOUNTS='<drives>' DOCKER_ARGS='...' bash run/steps.sh
```

On any machine, on a copy of `<out>`:

```bash
python3 -I run/tables.py <out> docs/data/2026-11-26_c1080_grammar_start
```

## Files

| File | What it holds |
|---|---|
| [arms.tsv](arms.tsv) | per build, request and arm: tokens, end, drafted and accepted tokens, the server's grammar counts, the first token that differs from the build's `chat` reply |
| [steps.tsv](steps.tsv) | per build and request: where the first `<tool_call>` is, the steps before and after it at which the grammar let the whole vocabulary through, rejections, whether `chat2` walked the same states |
| [left_out.tsv](left_out.tsv) | per build, request and phase: each count of tokens left out, at how many steps, and of what kind (malformed UTF-8 start, NUL byte, no text, end of turn, valid text; "-" where the list was longer than 1,024 tokens and so not named) |
| [server_lines.tsv](server_lines.tsv) | per build and chat request: the server's sampling line and the old build's own trigger lines |
| [params.tsv](params.tsv) | per build: the chat handler, lazy or not, triggers, generation prompt, the grammar's size, md5 and first rule |
| [grammar_cand.gbnf](grammar_cand.gbnf), [grammar_ref.gbnf](grammar_ref.gbnf), [grammar_land3.gbnf](grammar_land3.gbnf) | the tool-call grammar each build built for the agent's four tools (`land3`'s is `cand`'s byte for byte) |
| [tokens.tsv.gz](tokens.tsv.gz) | every step of the `chat` replies: allowed, tokens let through, waiting for a trigger, phase; the token's id and text only for the model's special tokens (`<tool_call>`, `</think>` …) |
| [run/run.sh](run/run.sh) | the servers and the arms |
| [run/grammar_steps.cpp](run/grammar_steps.cpp), [run/build.sh](run/build.sh) | the grammar walker and its build against a rig build's tree |
| [run/steps.sh](run/steps.sh) | the slot parameters from the server's log, then the walker over every chat reply |
| [run/tables.py](run/tables.py) | these tables |

## Provenance and licence

Run on 2026-11-26 00:41–00:52 (the servers) and 00:55–01:10 (the walker) on the build machine's GPUs 0 and 1. The
request bodies come from C912's recorded runs and hold the task's text from Exercism's exercises (the polyglot
benchmark's, MIT) and the harness's prompt, so they are not published; the replies' text (and the ids that spell it)
is left out of `tokens.tsv.gz` for the same reason. The grammars hold only the four tools' names and parameter names. The tables and
scripts here are under the repository's licence.
