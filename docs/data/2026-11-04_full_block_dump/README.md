# Q804: DFlash2's acceptance at drafted positions 1–7, from a full-block dump (2026-11-04)

How far down its block does Inco AI's DFlash2 drafter keep predicting what Qwen3.8-Flash-Next will say on coding-agent
text? The line drafts 2 tokens a step (`--spec-draft-n-max 2`), so the drafter's positions 3–7 are never verified and
nobody has measured them. They decide the largest speed-up the research note
[flash_next_acceptance.md](../../research/notes/flash_next_acceptance.md) finds (idea 1: after the small-batch kernels,
Q591, a 5–6-token draft costs little more than a 2-token one): est. 9.5 → 24–26 tok/s if the tail holds (P_4…P_7 =
0.39 / 0.28 / 0.20 / 0.145), S ≈ 1.75 at k = 3–4 if it collapses as a community INT4 port's does (0.14 / 0.10 / 0.02 /
0.00). This folder holds the tools that measure it on recorded agent sessions without verifying anything beyond the
line's own 2 tokens, their checks, and the run. Registry row Q804 in [quants/experiments.md](../../../quants/experiments.md).

**Status:** built and checked on the CPU, including a real-model check (below). The measurement itself runs on the quant
machine with [run/run_full_block.sh](run/run_full_block.sh) once its GPUs are back; its tables land here.

## The pieces

| Piece | Where | What |
|---|---|---|
| The dump | fork branch `pascal-p106-v3-b12076-specdump2`, one commit (`3e22fd7c1`) on `-specdump` | `LLAMA_SPEC_DUMP_FULL_BLOCK=1` (or `=n`) beside `LLAMA_SPEC_DUMP=<file>`: after the draft is made, DFlash2 decodes its whole trained block a second time and the dump line carries that block's lattice as `fb` (7 positions after the anchor for Inco AI's drafter); a `gram` flag says whether the tool-call grammar was active |
| The tables | [bench/tree_replay](../../../bench/tree_replay/README.md) (`block*.csv`, `summary.md`) | P_j and the conditional acceptance at positions 1–7, overall and by context length, by segment (tool call, reasoning, text), by the grammar flag and by content (copy); τ(k) and S(k) for drafting k tokens, read against the note's two tails; the block width's effect on the first positions; `gguf_markers.py` for the token ids of the segment markers |
| The checks | [run/cpu_check.sh](run/cpu_check.sh), [run/cpu_dflash2_check.sh](run/cpu_dflash2_check.sh) | build, unit test, the tables' self-test, a random-weight smoke test, and the real drafter and target on the CPU backend |
| The run | [run/run_full_block.sh](run/run_full_block.sh) (a wrapper of [run_quant.sh](../2026-11-04_tree_replay/run/run_quant.sh)) | the line at `--spec-draft-n-max 2` on S18's recorded agent sessions with the full-block dump, then the tables |

**What a block is.** DFlash2 fills a block of mask tokens after the anchor (the last sampled token) in one pass and
scores, for each position, its 16 candidates against each of the previous position's 16; the server walks that lattice
greedily. Inco AI's drafter was trained on blocks of 8 (`dflash.block_size`): the anchor and 7 positions. That is the
whole block, so a 7-position dump covers everything it predicts; 16 is the selector's candidates per position, not a
block width, and a block longer than the trained one is not decoded. The line decodes a block of `n_max + 1` tokens (3
at `n_max` 2); the drafter's attention across the block is not causal, so a block of 8 need not predict positions 1 and
2 as a block of 3 does.

**Why a second decode.** The draft, the verify and the output must stay the line's own, or the dump measures another
line. The server therefore takes the draft from the block it decodes anyway, drops that block's cells in the draft
context (as it does after every draft), and decodes the wide block at the same positions for the record only. With
the switch off nothing is decoded or stored. Each step costs one more drafter pass, so a dump run is acceptance only:
no timing from it is a speed result.

**Scoring positions beyond the verified ones.** Truth is the target's own continuation, rebuilt by position from the
same request's later steps (the next step's anchor and sampled tokens follow on), so a step's block is scored at
positions 3–7 although only 2 were verified. A position counts for a step when the block reaches it and the
continuation is known to it; a request's last steps count at the positions they can score. Sample-and-match accepts a
drafted token while the target's sample equals it, so a block's walk is accepted for exactly the leading positions
where it equals the continuation, whatever was verified, and the mean over steps estimates P_j without bias
([bench/tree_replay](../../../bench/tree_replay/README.md), Method). The steps are anchored where the live line put
them (at most 2 drafted tokens and their acceptances), not where a 6-token draft would have landed.

## Checks (CPU, quant machine, 2026-11-04)

All pass ([cpu_check.md](cpu_check.md)): the CPU build; `test-spec-dump` 211 of 211 (the lattice reduction, the
selector's greedy walk against a hand-made reference, the second block's record leaving the draft's own untouched);
`tree_replay.py --selftest` 42 of 42 (a hand-made request with exact by-position acceptance, censoring, τ(k), S(k) and
width table; 8,000 synthetic second blocks with a known conditional acceptance at each of 7 positions in two groups,
every split within sampling error; segment labels; the marker helper); a random-weight qwen4exp as target and as its
own drafter with the same tokens with the dump off, on, and on with the switch (a drafter without a second block has
none to add); and **the real pair on the CPU backend**: Swift-1.5 Q2_0 with Inco AI's Q8_0 drafter, `--spec-draft-n-max
2`, two 24-token requests (greedy and sampled) to four servers (no dump, the dump, `FULL_BLOCK=1`, `FULL_BLOCK=4`):

| Check | Result |
|---|---|
| tokens | the same in all four servers on both requests |
| drafts | all 18 dump lines equal with and without the switch but for the run and the second block (the drafts, the candidates, the walk, the target's rows) |
| second block | 18 of 18 lines carry 8 tokens (7 positions) with `FULL_BLOCK=1`, 18 of 18 carry 4 tokens (3 positions) with `=4`, none carry one without the switch |
| check | `tree_replay.py --check`: 0 problems on both dumps |
| tables | `tree_replay.py` writes `block*.csv` from the dump |

The 18 steps are a sample of the tool's output, not a measurement ([cpu_dflash2/](cpu_dflash2/)); they show one
thing worth a number: on 16 steps the draft's block of 3 and the block of 8 gave the same first two tokens on 13, and
the draft's two tokens were accepted slightly more often than the wide block's (P_2 0.69 against 0.63, a difference of
one step in sixteen), so the block width does change the drafter's first positions. `block_width.csv` measures it on
the run's steps.

## The run (quant machine, after the reboot)

On three RTX 3090s through the rig build's compute 6.1 PTX (the first start JIT-compiles every kernel, minutes):
Swift-1.5 GSQ-RCO Q2_0 (the rig's file) with Inco AI's DFlash2 Q8_0 drafter, `--spec-draft-n-max 2` and the release
line's sampling (temperature 1.0, top-p 0.95, top-k 20), host sampling, S18's recorded agent sessions
([Q561](../2026-11-03_flash_next_dflash2/README.md): 175 requests, ~48,600 generated tokens) re-sent with
`bench/agent_check/replay.py` (the recorded messages, tools and sampling, each reply's recorded length, a fixed seed per
request). A greedy smoke request first: its dump must pass `--check` and carry a second block of 8 on every step.

```
BIN=<llama.cpp-…-specdump2>/bin MODEL=<Swift Q2_0 first shard> DRAFT=<Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf> \
OUT=<out dir> LOG=<agent-check out>/proxy.jsonl REPO=<this repository> GPUS=2,3,4 \
MOUNTS="-v <bin dir>:<bin dir>:ro -v <model dir>:<model dir>:ro -v <drafter dir>:<drafter dir>:ro" \
bash docs/data/2026-11-04_full_block_dump/run/run_full_block.sh
```

Est. 1–3 h, acceptance only. The same server setup serves [Q688](../2026-11-04_tree_replay/README.md)'s run: the tree
tables walk the second block's 7-position lattice by default, so one run at `N_MAX=2` and `FULL_BLOCK=1` can replace
that run at `N_MAX=7` (the verify then has 3 columns, as the line's). On a rig the same dump runs on S18's line: set
`LLAMA_SPEC_DUMP=<file>` and `LLAMA_SPEC_DUMP_FULL_BLOCK=1` in `llama-server`'s environment beside
`--spec-draft-n-max 2`, then run `tree_replay.py` on the file.

**What the result decides.** `summary.md`'s "Acceptance by position" section gives P_j with its standard error,
τ(k) and S(k) at r = 0.08–0.40 and the best k at each r, next to the same arithmetic on the note's two tails. Idea 1
goes ahead (kernels, then 5–6 drafted tokens) if the best k at r = 0.08–0.10 is 5 or 6 and S is about 2.2 or more;
it stays at 3–4 tokens if P_4 falls to about 0.15 and the best k is 3–4 at S ≈ 1.75. The tables by context length and
by segment say where the tail is long (tool-call bodies, copied spans) and where it is short (reasoning), which is the
input to the per-step draft length (Q687).

## Files

| File | What |
|---|---|
| `run/run_full_block.sh` | the run: `run_quant.sh` at `N_MAX=2`, `FULL_BLOCK=1` |
| `run/cpu_check.sh` | build (CPU backend), `test-spec-dump`, the replay's self-test, the random-model smoke test with and without the dump and the switch |
| `run/cpu_dflash2_check.sh` | the real target and drafter on the CPU backend, four servers, the same requests |
| `cpu_check.md` | the checks' verdicts |
| `cpu_dflash2/` | the real-model check's `summary.tsv`, `dump_sample.jsonl` (its 18 dump lines, the format of `fb`) and `block*.csv` from it |

## Provenance

Code: Miner Llama's fork, branch `pascal-p106-v3-b12076-specdump2`; rig tarball built with
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) (`TESTS=1`, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`). The model and the drafter are those of [Q561](../2026-11-03_flash_next_dflash2/README.md)
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
[incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF), Apache-2.0); the
requests are the bugfix task set of [bench/agent_check](../../../bench/agent_check/README.md).
