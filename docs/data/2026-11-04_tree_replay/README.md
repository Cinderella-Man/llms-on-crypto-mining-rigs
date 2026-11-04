# Q688: the tree-value replay that decides Flash-Next's token trees (2026-11-04)

Should the rig verify a token tree (several candidate continuations in one pass of the model) instead of one chain of
drafted tokens? The owner's question (Q614) is whether branches raise the chance of accepting 5, 10, 15 or 20 tokens a
step. On nine P106 cards every tree node is one more column of the verify pass, and a node pays only if the
probability that its whole path is accepted exceeds p* = r·S: est. 0.15–0.25 after the small-batch kernels (Q591),
0.4–0.5 today. Published trees add 0.01–0.03 accepted tokens per extra node
([moe_speculation_union.md](../../research/notes/moe_speculation_union.md), sections 3.2 and 3.3, and idea 5). This
folder holds the tools that measure it for Flash-Next's own DFlash2 drafter on agent sessions, before any tree code is
written, and tomorrow's run. Registry rows Q688 (this), Q614 (the tree it gates) and Q730–Q734 (ideas from it) in
[quants/experiments.md](../../../quants/experiments.md).

**Status:** written, built for the rig, checked on the CPU (below). The measurement itself runs on the quant machine
with [run/run_quant.sh](run/run_quant.sh); its tables land here as `tables-markov/` and `tables-factorized/`.

## The pieces

| Piece | Where | What |
|---|---|---|
| The dump | fork branch `pascal-p106-v3-b12076-specdump` `dafb09a4a` (one commit on `pascal-p106-v3-b12076-ladder2` `a330672c1`) | `LLAMA_SPEC_DUMP=<file>`: llama-server writes one JSON line per speculative verify step (the fields: [bench/tree_replay](../../../bench/tree_replay/README.md)); off by default, and with the variable unset no candidate is recorded |
| The replay | [bench/tree_replay/tree_replay.py](../../../bench/tree_replay/tree_replay.py) | chain acceptance by position; best-first trees (DDTree's expansion by path probability) at budgets 3, 4, 6, 8, 12 and 16; τ and Δτ per extra node against p*; speed-ups from a cost model; tables by context length and by content (copy spans: the context's suffix occurs earlier); the draft length chosen per step (Q687) and a copy-gated policy (Q617) on the same steps |
| CPU check | [run/cpu_check.sh](run/cpu_check.sh) | build, unit test, the replay's self-test, and a smoke test on a random-weight model |
| The run | [run/run_quant.sh](run/run_quant.sh) | Flash-Next with the DFlash2 drafter drafting its whole block, S18's recorded agent sessions re-sent in order, the dump, the tables |

**What the dump records, for DFlash2.** Inco AI's drafter (block 8, selector top-k 16) decodes a block of mask
tokens in one pass and scores, for each position, its 16 candidates against each of the previous position's 16: a
first-order lattice over the block, which the server walks greedily. The dump keeps, per position, the 8 candidates
with the highest marginal probability under that lattice (plus the walked one), the conditional probabilities between
kept candidates from the selector's softmax over all 16, and the kept mass. With 7 drafted tokens the lattice covers
every position the drafter predicts; `LLAMA_SPEC_DUMP_FULL_BLOCK=1` gets the same lattice from a server that verifies
fewer (the rig's 4 GB host holds 2). The target's tokens, its 8 most probable tokens at each sampled position and the
copy drafter's proposal (the tokens after the context's longest earlier match, up to 16) are in the same line.

**Why the replay is exact in expectation.** The server verifies greedy drafts by sample-and-match: it samples each
position from the target and keeps a drafted token while the sample equals it. A tree verified the same way accepts
the longest root path that matches the target's continuation, so a node is worth 1 if its whole path matches the
tokens the target went on to generate, and the tree's accepted length is 1 plus its accepted nodes. The target's
continuation does not depend on what was drafted, so its realised tokens score every tree on the same step without
bias. The steps are anchored where the live chain put them; a teacher-forced dump at every position (Q730) would
remove that choice too, if it ever matters.

## CPU check (quant machine, no GPU, 2026-11-04)

All four checks pass ([cpu_check.md](cpu_check.md)): the CPU build; `test-spec-dump` 155 of 155 (the DFlash2 lattice
reduction against a reference on synthetic lattices); `tree_replay.py --selftest` 16 of 16 (on 6,000 random lattices
whose continuations are drawn from the lattice itself, each best-first rank's value equals its mean path probability,
largest |z| 2.58); and a smoke test on a random-weight qwen4exp (Flash-Next's architecture) as its own draft-simple
drafter, whose 37 dumped steps pass `--check` while the server returns the same tokens with and without the dump. The
rig tarball `llama.cpp-20261104-dafb09a4a-pascal-p106-v3-b12076-specdump.tar.gz` (md5
`00788be38f858552754eec05a7c5dac3`, `TESTS=1`, no AVX, glibc 2.38) is built.


## The run (tomorrow, quant machine)

On the quant machine after FN1c (its seven GPUs are busy until ~07:00), on three RTX 3090s through the rig build's
compute 6.1 PTX: Swift-1.5 GSQ-RCO Q2_0 (the rig's file) with Inco AI's DFlash2 Q8_0 drafter, `--spec-draft-n-max 7`,
the release line's sampling (temperature 1.0, top-p 0.95, top-k 20) and context settings, host sampling. The
requests are S18's agent-check sessions ([Q561](../2026-11-03_flash_next_dflash2/README.md): the `dflash2b`, `mtp2` and
`nodraft` arms, 31 attempts, 175 requests, ~48,600 generated tokens), re-sent with `bench/agent_check/replay.py`:
the recorded messages, tools and sampling, each reply's recorded length and a fixed seed per request. First a greedy
smoke request, whose dump must pass `tree_replay.py --check` with full 8-candidate lattices of 7 positions.

```
BIN=<llama.cpp-…-specdump>/bin MODEL=<Swift Q2_0 first shard> DRAFT=<Qwen3.8-Flash-Next-DFlash2-Q8_0.gguf> \
OUT=<out dir> LOG=<agent-check out>/proxy.jsonl REPO=<this repository> GPUS=2,3,4 \
MOUNTS="-v <bin dir>:<bin dir>:ro -v <model dir>:<model dir>:ro -v <drafter dir>:<drafter dir>:ro" \
bash docs/data/2026-11-04_tree_replay/run/run_quant.sh
```

Est. 1–3 h (acceptance only: a shared machine gives no speed results, and S comes from the cost model). On a rig the
same dump runs on S18's line with `LLAMA_SPEC_DUMP_FULL_BLOCK=1` beside `--spec-draft-n-max 2`.

**What decides Q614:** a tree goes ahead only if a node off the drafter's walk adds ≥ 0.15 tokens at a budget of 8
nodes or fewer (`summary.md` states it), and the best tree's S beats the best chain's at r = 0.08–0.12.

## Files

| File | What |
|---|---|
| `run/cpu_check.sh` | the CPU check: build (CPU backend), `test-spec-dump`, `tree_replay.py --selftest`, the random-model smoke test with and without the dump |
| `run/run_quant.sh` | the run on a GPU machine: server, smoke check, replay, tables |
| `cpu_check.md` | the CPU check's results: the verdicts and the smoke test's dump statistics |
| `smoke_dump.jsonl` | the smoke test's dump (37 steps from a random-weight model): a sample of the line format |

## Provenance

Code: Miner Llama's fork, branch `pascal-p106-v3-b12076-specdump`; rig tarball built with
[scripts/build_llamacpp_docker.sh](../../../scripts/build_llamacpp_docker.sh) (`TESTS=1`, image
`nvidia/cuda:12.9.2-devel-ubuntu24.04`). The CPU check's model is random weights from the fork's `test-llama-archs`.
The run's model and drafter are those of [Q561](../2026-11-03_flash_next_dflash2/README.md)
([ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF](https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF),
[incoai/Qwen3.8-Flash-Next-DFlash2-GGUF](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2-GGUF), Apache-2.0);
the requests are the bugfix task set of [bench/agent_check](../../../bench/agent_check/README.md).
