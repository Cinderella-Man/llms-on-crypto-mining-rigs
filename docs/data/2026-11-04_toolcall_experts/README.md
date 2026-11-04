# Keeping the experts tool calls need in a usage cut (2026-11-04)

Registry row Q897 ([quants/experiments.md](../../../quants/experiments.md)). Qwen3.8-Flash-Next has 512 routed experts per
layer and picks 10 per token. Q232 ranked them by how often they are picked on agent text
([expert usage](../2026-11-04_expert_usage/README.md)), and Q890 cut Swift IQ3_S to its 464, 448 and 416 most-used
experts per layer ([gguf_prune](../2026-11-04_gguf_prune/README.md)) so that it fits nine 6 GB cards. An expert that is
rare overall can still be the one tool-call syntax needs. agentionai's Gyro-Coder v1.2 found three such experts missing
from its 256-expert cut. Putting them back took broken tool-call tags from 0.69 % to 0 per `<parameter=` site, at the
same KLD. This rig runs a coding agent, so tool-call integrity is part of quality.

This folder holds the tools that find those experts and keep them, their checks, the token classes of the two agent
texts, and today's broken-tag rate on the agent check. **The GPU pass that counts the experts is not run yet**; its
command is below.

## Result so far

**A per-span expert count in `llama-imatrix`.** Miner Llama's branch `pascal-p106-v3-b12076-spancount` (`f6c157897`, on
`-kshape` `39c3ab5f1`) adds two options:
- `--span-dump FNAME` writes the tokenized input (index, token id, the bytes of each piece). It loads only the
  vocabulary and tokenizes exactly as a run does.
- `--span-mask FNAME` gives every token a class and, beside the usual matrix, collects one matrix per class
  (`<output>.span<c>.gguf`). Each collected row also goes to its token's class, in the same order, so the class
  matrices add up to the usual one. The usual matrix is computed exactly as without a mask.

In each class matrix, the `.counts` of a layer's `ffn_gate_exps` are how often that class's tokens picked each expert.

**The checks pass on a CPU** (21 of 21, [cpu_checks.tsv](cpu_checks.tsv); `run/cpu_check.sh` on a random qwen4exp model
from `test-llama-archs`: 32 experts, 8 used, four chunks of 512 tokens):

| Check | What it shows |
|---|---|
| every token in class 1 | class 1's matrix equals the usual one bit for bit, class 0's is zero, and the usual matrix equals a run without a mask |
| every token in class 0 | the same the other way round |
| random classes 0–3 | the four add up to the usual matrix: counts exactly, `in_sum2` within 1.1e-5, `in_sum` within 2.2e-6; each class counts its own tokens and 8 routings per token |
| class 1 = chunk 3 of 4 | its counts are a 4-chunk run's minus a 3-chunk run's exactly, and class 0 equals the 3-chunk run bit for bit |
| class 1 = the last token of a repeated 8-token sentence, class 2 = the token after | layer 0's n-gram projections (`ple_key`, `ple_value`) read the same input on all 158 class-1 rows (relative variance 1.05e-6) and different inputs on class 2's (above 0.47), in one batch per chunk and in two |
| a mask with one token id changed; a batch over two ubatches | both refused |
| span counts → force list → `gguf_prune.py --force-ids` on the random model | 4 experts forced in at 24 of 32 kept, all kept, 24 a layer, `verify` passes |
| the sm_86 build on the CPU (`-ngl 0`, no GPU visible; not in the table) | the same dump and the same five matrices of the random classes as the CPU build, bit for bit |

The fourth row checks the right chunk, the fifth the right token within a chunk. Qwen4exp's first layer reads each
token's 5 predecessors through its n-gram embeddings, so "the same input" needs the same 6 tokens, not just the same id.

**FN4's agent text is mostly tool results.** Q232's ranking counts every token of the 200 chunks FN4's matrix read. By
the span finder ([spans_plain.tsv](spans_plain.tsv), Swift's tokenizer), 89.6 % of those 102,400 tokens are tool
results (file contents the agent read). Only 8.1 % are the model's own writing: 6,780 tokens of reasoning and answers,
673 of tool-call syntax and 884 of argument values, over 98 calls. FN4's text writes calls as `name({json})` lines,
not in the chat template's `<tool_call>` format, so the tags the model emits at work are not in it at all. The
ranking the cut uses therefore mostly reflects reading.

**So the GPU pass reads the same sessions in the chat template.** `run/template_text.py` renders the 125 sessions FN4's
text holds through Flash-Next's chat template: `<tool_call><function=…><parameter=…>` tags, `<tool_response>` blocks,
`<think>`. Each tool result is cut to 1,500 characters so more turns fit. In its first 200 chunks
([spans_template.tsv](spans_template.tsv)) there are 275 calls with 385 parameters: 7,554 tokens of call syntax, 4,127 of
argument values, 39,641 of reasoning and answers, and 48,158 of tool results. That is 75,540 routings per layer on call
syntax, enough to see an expert that carries 1 % of them (755 routings).

The span finder gives each token a class:

| Class | Template text (the chat template) | Plain text (FN4's `name({json})` lines) |
|---|---|---|
| 0 other | user messages, the rest | the same |
| 1 call | a `<tool_call>` block but its values: the tags, the function and parameter names, the newlines | the name, the parentheses, the JSON's braces, keys, colons, commas and quotes |
| 2 argument | a parameter's value | the JSON's string contents and scalars |
| 3 response | each `<tool_response>` block | the body of a `### tool` message |
| 4 assistant | the rest of an assistant message: reasoning with its `<think>` tags, the answer | a `### assistant` body but its call lines |

A token that covers bytes of more than one class takes the first of 1, 2, 3, 4, 0. So a token that merges a tag's end
with a value's start (`' "/'`) counts as syntax. `run/span_mask.py find --show 3` prints calls token by token, e.g.
`1:'<tool_call>' 1:'\n' 1:'<' 1:'function' 1:'=read' 1:'_file' 1:'>' 1:'\n' 1:'<' 1:'parameter' 1:'=path' 1:'>' 1:'\n'
2:'std' 2:'mods' 2:'/calendar' 2:'.py' 1:'\n' 1:'</' 1:'parameter' 1:'>' …`.

**Today's broken-tag rate is 0 in 539 `<parameter=` sites** ([broken_tags_baseline.tsv](broken_tags_baseline.tsv),
`run/broken_tags.py`). That covers every Flash-Next agent-check transcript on hand: Swift Q2_0 without a draft and
with MTP and DFlash2 drafts (S12, S15, S18, S33), its retyped `hcq8` file (R28), the pipelined lines (R29a) and S34's
`pmix-q2-fast` arms (the no-draft arm and the first 17 requests of the MTP arm). At 539 sites, 0 bounds the rate at
about 0.56 % (95 %), just below Gyro-Coder v1.2's 0.69 %. A pruned file's agent check, or the generation check Q937
plans, decides whether a cut breaks tags. Every file here is uncut.

## What runs next

1. **The GPU pass** (`run/gpu_pass.sh`, C336g): the template text through Swift IQ3_S, the file the cuts are made from,
   on three 24 GB cards with `-fit on`, 200 chunks. Swift-1.5's Q8_0 needs about 134 GB on the cards for its routed
   and dense tensors and does not fit five. Est. 1–1.5 h: FN4's pass took 17.4 s a chunk, and the class matrices add
   about as much CPU work again. Then `analyze` writes the tables and force lists without a GPU:
   - `span_cut_summary.tsv`: the share of each class's routings that the 464, 448 and 416 cuts drop, against the 0.22,
     0.39 and 0.93 % of all routings;
   - `force_<N>.tsv`: the experts each cut drops that carry at least X % of a layer's call and argument routings, with
     X the smallest in a grid from 0.05 to 10 % that forces at most 8 a layer. `call_only/` takes the syntax alone,
     `written/` everything the model writes.
2. **A cut with the forced ids** (`gguf_prune.py prune --keep 464 --experts experts.tsv.gz --force-ids force_464.tsv`):
   the same size and fit as Q890's e464, because each forced expert takes the place of the lowest-ranked kept one.
   Then its KLD beside Q891's cuts, and its broken-tag rate in the agent check (Q902's fit, then the rig).

## Files

| File | What |
|---|---|
| [cpu_checks.tsv](cpu_checks.tsv) | one verdict per CPU check (`run/cpu_check.sh`) |
| [spans_template.tsv](spans_template.tsv), [spans_plain.tsv](spans_plain.tsv) | tokens per class, calls, parameters and tool results, in the whole text and in the 200 chunks a pass reads |
| [broken_tags_baseline.tsv](broken_tags_baseline.tsv) | per agent-check arm: requests, calls, `<parameter=` sites, broken sites by kind, broken per site, calls to unknown tools, other tag text |
| `run/template_text.py` | FN4's sessions through the chat template (`bench/copy_spans/common.py`'s renderer) |
| `run/span_mask.py` | `find`: token classes from a `--span-dump` (template or plain format); `synth`: the checks' masks; `text`: the checks' text |
| `run/span_check.py` | `same`, `zero`, `sum`, `diff`, `variance`: the checks on imatrix files |
| `run/span_counts.py` | per layer and expert, the routings of each class, and what a cut drops of them |
| `run/force_keep.py` | the force-keep list per cut, the sweep over X, the displaced experts |
| `run/broken_tags.py` | broken tool-call tags per `<parameter=` site in agent-check proxy logs |
| `run/build.sh`, `run/cpu_check.sh`, `run/gpu_pass.sh` | the builds (CPU and sm_86), the CPU checks, the GPU pass and its analysis |
| [scripts/gguf_prune.py](../../../scripts/gguf_prune.py) | `--force-ids`: keep these ids whatever their rank, fill the rest by the ranking (self-test case 5) |

## Commands

```bash
# builds (Docker, no GPU): CPU tools in $WORK/cpu/bin, the sm_86 llama-imatrix tarball in $WORK
WORK=<work> SRC=<checkout of pascal-p106-v3-b12076-spancount> JOBS=8 bash run/build.sh
# the CPU checks (seconds each; a random model, no download)
WORK=<work> BIN=$WORK/cpu/bin REPO=<this repository> bash run/cpu_check.sh
# the GPU pass: text, mask (CPU), the pass on three cards, then the tables
WORK=<work> OUT=<out> MODEL=<Swift IQ3_S first shard> CPU_BIN=$WORK/cpu/bin GPU_BIN=<tarball>/bin REPO=<this repository> \
EXPERTS=docs/data/2026-11-04_expert_usage/experts.tsv.gz SESSIONS=<run 1 export>/hf/data/sessions.jsonl \
TEST_ROWS=<run 1>/seqs/test-rows.jsonl GPUS=0,1,2 MOUNTS="<folders of the model and the sessions>" \
PREFLIGHT=scripts/cuda_preflight.sh bash run/gpu_pass.sh
# the same for FN4's own text (no --parse-special): FORMAT=plain TEXT=calib-agent-train.txt
# broken tags in agent-check logs (folders with proxy.jsonl and proxy.jsonl.bodies.gz)
python3 -I run/broken_tags.py <results>/out ... --out broken_tags
```

`gpu_pass.sh` checks that the dump's pieces rebuild the text byte for byte (less the last newline, which llama.cpp's
`-f` drops), and that the class matrices add up to the usual one. The sessions file is the export of agentic run 1;
`template_text.py` picks the sessions `calib_text.py` picks
([2026-10-28_ornith35_kld/run/calib_text.py](../2026-10-28_ornith35_kld/README.md)). FN4's text has md5
`e1c1e8ec97c4accddc740dfb637ad29c`, the template text with 1,500-character results `f8c50b4c5928483caeaf467f7b713f14`.

## Provenance

The texts are agentic run 1's sessions (Flint-written, gated, publishable), outside the 40 held-out sessions FN2
scores KLD on; neither text is stored here, both rebuild from the sessions. Token counts used Swift IQ3_S's
vocabulary ([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
the same tokenizer as the Q8_0). The broken-tag counts come from the agent check's proxy logs of exercism tasks (MIT);
the tables hold counts only. The random model comes from Miner Llama's `test-llama-archs`. Tables and scripts of this
repository (MIT).
