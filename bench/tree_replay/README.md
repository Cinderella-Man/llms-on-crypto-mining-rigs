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

Miner Llama's branch `pascal-p106-v3-b12076-specdump` (and `-specdump2`, one commit on it, which adds the last two
fields) writes one JSON line per speculative verify step when `LLAMA_SPEC_DUMP=<file>` is set (off by default; nothing
is computed without it):

| Field | What |
|---|---|
| `run`, `task`, `slot`, `step` | the server start (microseconds), the request, the slot, a running step count |
| `pos`, `n_prompt` | the position of the last sampled token (the context length before the step) and the request's prompt length |
| `last`, `draft`, `sampled`, `n_acc` | the last sampled token, the drafted tokens, the target's tokens (the accepted drafts, then its own), the accepted count |
| `verify`, `temp`, `type` | sample-and-match or rejection, the request's temperature, the drafter |
| `cand`, `p`, `pick`, `lattice`, `kept` | the drafter's candidates per position (`LLAMA_SPEC_DUMP_K`, default 8), their probabilities, the drafter's own walk; for DFlash2 (`lattice: true`) the selector's conditional rows: `p[0]` from the anchor, `p[i]` row-major with one row per candidate of position i − 1, over the selector's whole top-k (16 for Inco AI's drafter), so rows of kept candidates sum to less than 1, and `kept` is the selector's marginal mass of the kept candidates |
| `tgt_ids`, `tgt_p`, `tgt_ps` | the target's most probable tokens (`LLAMA_SPEC_DUMP_TGT_K`, default 8) at temperature 1 at each sampled position, and the probability of the token it sampled |
| `ng_len`, `ng` | the longest suffix of the context (up to `LLAMA_SPEC_DUMP_NGRAM`, default 16) that occurs earlier in it, and the tokens that followed its last occurrence: what a copy drafter would propose |
| `gram` | 1 when the sampler's tool-call grammar was active as the draft was made, else 0 (branch `pascal-p106-v3-b12076-specdump2`) |
| `fb` | with `LLAMA_SPEC_DUMP_FULL_BLOCK`: the lattice of a second block the drafter decoded for the record, `{w, cand, p, pick, kept}` in the form of the fields above, one entry per position after the anchor along the selector's greedy walk (`w` tokens in the block, `w − 1` positions; 7 for Inco AI's drafter) |

`LLAMA_SPEC_DUMP_FULL_BLOCK=1` (branch `pascal-p106-v3-b12076-specdump2`) makes DFlash2 decode its whole trained block a
second time, after the draft has been taken from the block the line decodes anyway (`--spec-draft-n-max` + 1 tokens):
the draft, the verify and the output are the same with it on or off, `cand`, `p`, `pick` and `kept` stay the draft's
own, and `fb` holds the drafter's proposals at every position it can predict, for a host that cannot verify them all
(the rig's 4 GB holds 2). `=n` (2 ≤ n ≤ the trained block) decodes a block of n tokens. The drafter's attention across
the block is not causal (Inco AI's drafter), so a block of 8 does not predict positions 1 and 2 as a block of 3 does:
`block_width.csv` compares the two on the same steps. The first branch (`-specdump`) fed the whole block to the draft
itself, which changed the draft; dumps from it carry the block in `cand` and are read as before. Steps that replay
accepted tokens after a checkpoint restore, and synthetic acceptance, are not written.

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
- **Acceptance by position (`block*.csv`).** Along the drafter's greedy walk through the lattice (the second block's where
  a step has one): P_j, the probability that the first j drafted tokens are all accepted, and the conditional
  acceptance P_j / P_(j−1), at every position j the block reaches. A step counts at position j when its walk reaches j
  and the target's continuation is known to j, so a request's last steps count at the positions they can score
  (`censored` counts the others) and every position uses all its data; `strict_cumulative` repeats P_j on the steps
  whose continuation is known to the whole block. The standard error of P_j is clustered by request. τ(k) = 1 + P_1 +
  … + P_k and S(k) = τ(k) / (1 + d + k·r) price drafting k tokens in one verify pass of k + 1 columns, next to the same
  arithmetic on the research note's two tails (est. rig-anchored and the INT4 port's collapse), so the measured tail
  reads against them. The tables split by context length, by segment of the anchor token, by the server's `gram` flag
  and by content. Segments need the token ids of the markers that open and close a tool call and the reasoning:
  `gguf_markers.py MODEL.gguf` prints them as the arguments `--tool-open --tool-close --think-open --think-close`; a
  request's first tokens are classified by the first marker seen (a closing marker first means the request began
  inside it, as when the chat template opens the reasoning in the prompt). `block_width.csv` compares the draft's own
  block with the second block on the same steps: how often the draft's first j tokens equal the second block's walk,
  and each walk's P_j against the same continuation.

## Use

```
python3 tree_replay.py --selftest
python3 tree_replay.py --check DUMP.jsonl[.gz] ...
python3 tree_replay.py DUMP.jsonl[.gz] ... --out DIR [--budgets 3,4,6,8,12,16] [--r 0.08,0.10,0.12,0.40] \
    [--draft-cost 0.22] [--mode markov|factorized] [--copy-min 4] [--ctx-bins 4096,8192,16384,32768] \
    [--lattice auto|fb|prod] [$(python3 gguf_markers.py MODEL.gguf)]
```

`--lattice` picks the lattice every table walks: the second block's (`fb`), the draft's own (`prod`), or the second
block where a step has one (`auto`, the default).

`--check` tests the dump against itself: the accepted drafts are the target's tokens, the drafted tokens are the
walk's candidates, the accepted counts match the rebuilt continuation, the lattice rows (and the second block's) match
the candidate counts and no two steps disagree about a position. `--selftest` runs a hand-made lattice with a known
best-first order and known values, a broken dump that `--check` must catch, 6,000 random lattices whose continuations
are drawn from the lattice itself (each rank's value must equal its mean path probability within sampling error), a
hand-made request whose by-position acceptance, censoring, τ(k), S(k) and width comparison are known exactly, the
segment labels of a token stream, 8,000 random second blocks with a known conditional acceptance at each of 7 positions
in two groups (P_j, the conditionals, the context, segment, grammar and content splits, the selector's predicted
probability and a request's end censoring must come out within sampling error), and the marker helper on a synthetic
GGUF header.

Output in DIR: `summary.md` (every table and Q614's gate), `chain.csv` (acceptance by position, drafter's walk and
the verified draft), `budgets.csv` (τ of chain and tree, their difference, off-walk nodes and their value, S at each
r, Δτ per node), `curve.csv` (every budget from 1 to the largest, for the best budget at each r), `ranks.csv` (value,
expected value, predicted probability, off-walk share and depth of the r-th node), `calibration.csv`, `by_ctx.csv`,
`by_content.csv`, `adaptive.csv` (the walk cut where its path probability falls below a threshold: the draft length
chosen per step, Q687), `copy_gated.csv`, `check.txt`. Acceptance by position: `block.csv` (steps, censored, P_j with its
standard error, conditional, strict, the selector's predicted P_j), `block_speed.csv` (τ(k) and S(k) at each r),
`block_vs_reference.csv` (the best k and its S, measured and on the note's two tails), `block_by_ctx.csv`,
`block_by_segment.csv`, `block_by_gram.csv`, `block_by_content.csv` (P_j and the conditionals per group, with the steps
counted at each position) and `block_width.csv` (the draft's block against the second block). Standard library only.

The run on Flash-Next's agent sessions: [docs/data/2026-11-04_tree_replay](../../docs/data/2026-11-04_tree_replay/README.md).
