# Tree-value replay: what each extra node of a token tree is worth (Q688)

A token tree verifies several candidate continuations in one pass of the model. On the P106 rig every node of the
tree is one more column of that pass, so a node pays only if the tokens it adds exceed its cost: with a pass of k
tokens costing T1·(1 + (k − 1)·r), a node pays only when the probability that its whole path is accepted exceeds
p* = r·S, where S is the speed-up already reached. After the small-batch kernels (Q591) that is est. 0.15–0.25 on the
rig, against 0.01–0.03 accepted tokens per extra node in published trees
([moe_speculation_union.md](../../docs/research/notes/moe_speculation_union.md), sections 3.2 and 3.3). This tool
measures the value of each node offline, from what the drafter proposed and what the target generated, before any
tree code is written. Registry row Q688 in [quants/experiments.md](../../quants/experiments.md); it gates Q614.

## What goes in: llama-server's verify-step dump

Miner Llama's branch `pascal-p106-v3-b12076-specdump` writes one JSON line per speculative verify step when
`LLAMA_SPEC_DUMP=<file>` is set (off by default; nothing is computed without it):

| Field | What |
|---|---|
| `run`, `task`, `slot`, `step` | the server start (microseconds), the request, the slot, a running step count |
| `pos`, `n_prompt` | the position of the last sampled token (the context length before the step) and the request's prompt length |
| `last`, `draft`, `sampled`, `n_acc` | the last sampled token, the drafted tokens, the target's tokens (the accepted drafts, then its own), the accepted count |
| `verify`, `temp`, `type` | sample-and-match or rejection, the request's temperature, the drafter |
| `cand`, `p`, `pick`, `lattice`, `kept` | the drafter's candidates per position (`LLAMA_SPEC_DUMP_K`, default 8), their probabilities, the drafter's own walk; for DFlash2 (`lattice: true`) the selector's conditional rows: `p[0]` from the anchor, `p[i]` row-major with one row per candidate of position i − 1, over the selector's whole top-k (16 for Inco AI's drafter), so rows of kept candidates sum to less than 1, and `kept` is the selector's marginal mass of the kept candidates |
| `tgt_ids`, `tgt_p`, `tgt_ps` | the target's most probable tokens (`LLAMA_SPEC_DUMP_TGT_K`, default 8) at temperature 1 at each sampled position, and the probability of the token it sampled |
| `ng_len`, `ng` | the longest suffix of the context (up to `LLAMA_SPEC_DUMP_NGRAM`, default 16) that occurs earlier in it, and the tokens that followed its last occurrence: what a copy drafter would propose |

`LLAMA_SPEC_DUMP_FULL_BLOCK=1` makes DFlash2 decode its whole trained block, so the lattice covers every position the
drafter predicts while the draft stays at `--spec-draft-n-max` (for a host that cannot verify the whole block, like the
rig's 4 GB). Steps that replay accepted tokens after a checkpoint restore, and synthetic acceptance, are not written.

## Method

- **Truth.** A step's truth is the target's own continuation after it: the tokens of the request's later steps, rebuilt
  by position. The server verifies greedy drafts by sample-and-match: it samples each position from the target and
  keeps a drafted token while the sample equals it. A tree verified the same way accepts the longest root path that
  matches the target's continuation, so each node is worth 1 if its whole path matches and 0 otherwise, and the
  accepted length is 1 plus the accepted nodes. At temperature 1 the continuation is one sample, and the target's
  samples have the same distribution whatever was drafted, so the mean over steps estimates a tree's accepted length
  without bias. The steps at the end of a request, whose continuation is shorter than the lattice, are left out.
- **Trees.** Best-first by path probability, as DDTree builds them: the B nodes with the highest path probability, which
  always form a tree (a child is never more probable than its parent). `--mode markov` multiplies DFlash2's
  conditional rows along the path, the drafter's own model of its block; `--mode factorized` multiplies per-position
  marginals, as DDTree does with DFlash's marginals, and is the only choice for sequential drafters (draft model, MTP,
  EAGLE-3), whose candidates are conditional on their own drafted prefix. The chain is the drafter's own walk.
- **Values.** The B-th node of the best-first order adds its acceptance rate to the tree's accepted length: that rate is
  the Δτ per extra node at budget B, to compare with p*. For a node whose parent is on the accepted part of the verified
  chain the target's own probability of the node's token is known, which gives a lower-variance estimate of the same
  value (`value_expected`; it uses the target's distribution at temperature 1, before top-k and top-p).
- **Speed-ups.** S = τ / (1 + d + c·r): d the draft pass and c the verified columns beyond the anchor, both in one-token
  steps, r the cost of one more column (defaults: d = 0.22, the est. 20 ms DFlash2 pass on a 90 ms step; r = 0.08,
  0.10, 0.12 after the kernels and 0.40 today, from Q583 and the note's section 3.1).
- **Slices.** By context length (the step's position) and by content: "copy" where the context's suffix occurs earlier
  for `--copy-min` (default 4) tokens or more. The copy drafter's own chain (`ng`) and a copy-gated policy (the copy
  chain on copy steps without a draft pass, the tree elsewhere) are scored on the same steps.

## Use

```
python3 tree_replay.py --selftest
python3 tree_replay.py --check DUMP.jsonl[.gz] ...
python3 tree_replay.py DUMP.jsonl[.gz] ... --out DIR [--budgets 3,4,6,8,12,16] [--r 0.08,0.10,0.12,0.40] \
    [--draft-cost 0.22] [--mode markov|factorized] [--copy-min 4] [--ctx-bins 4096,8192,16384,32768]
```

`--check` tests the dump against itself: the accepted drafts are the target's tokens, the drafted tokens are the
walk's candidates, the accepted counts match the rebuilt continuation, the lattice rows match the candidate counts and
no two steps disagree about a position. `--selftest` runs a hand-made lattice with a known best-first order and known
values, a broken dump that `--check` must catch, and 6,000 random lattices whose continuations are drawn from the
lattice itself, where each rank's value must equal its mean path probability within sampling error.

Output in DIR: `summary.md` (every table and Q614's gate), `chain.csv` (acceptance by position, drafter's walk and
the verified draft), `budgets.csv` (τ of chain and tree, their difference, off-walk nodes and their value, S at each
r, Δτ per node), `curve.csv` (every budget from 1 to the largest, for the best budget at each r), `ranks.csv` (value,
expected value, predicted probability, off-walk share and depth of the r-th node), `calibration.csv`, `by_ctx.csv`,
`by_content.csv`, `adaptive.csv` (the walk cut where its path probability falls below a threshold: the draft length
chosen per step, Q687), `copy_gated.csv`, `check.txt`. Standard library only.

The run on Flash-Next's agent sessions: [docs/data/2026-11-04_tree_replay](../../docs/data/2026-11-04_tree_replay/README.md).
