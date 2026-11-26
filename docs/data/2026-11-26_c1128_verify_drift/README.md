# C1128: how far a verify pass's logits drift from one-token passes, over whole replies

What this is: registry [Q1861](../../../quants/experiments.md) (C1128), on the build machine's RTX 3090s running the
rig build through its compute 6.1 PTX with the fork's Pascal card paths forced, 2026-11-26. Correctness only, nothing
timed. On a P106, [C964](../2026-11-25_c964_flash_mtp_greedy_p106/README.md) found that Flash-Next's greedy replies
with the MTP draft leave plain decoding's text on all three of its requests, each time at a near tie between two
tokens that the draft's verify pass (four tokens a call) ranks the other way round from a one-token pass. This row
measures how far a four-token pass moves the next-token distributions from one-token passes at every position of
those replies, against a second one-token load as the floor, to tell whether those three flips are the rounding every
position shows or a few positions that drift much further.

*(Draft: the method and the reading rule, fixed before the run. The result follows.)*

## Method

- **Text:** C964's three requests (its coding request, 27 prompt tokens; its prose paragraph, 19; the deep prompt,
  8,189), each followed by plain decoding's 256-token reply on the P106 (C964's `tokens.tsv`, arm `plain`), as token
  ids. `llama-perplexity` cannot read this text: it tokenizes a text file without parsing special tokens (the replies
  hold `<think>`-like special tokens), and it scores only the second half of each window, which would leave out the
  first ~120 reply tokens of the two short requests. So a small tool, [run/verify_drift.cpp](run/verify_drift.cpp),
  does what the row asks of `llama-perplexity` with the exact token ids: one load per arm reads every token after the
  prompt *S* tokens a call, keeps every position's logits, and writes them (the reference load) or compares them
  with the reference's, position by position.
- **Two readings per request**, each on a cleared memory:
  - `reply`: the prompt read as llama-server reads a prompt call (calls of 4,096 tokens, the last one cut at
    *N* − 4 − 512 and *N* − 4 with in-call checkpoints, the server's hints), then the reply *S* tokens a call. Row 0 is
    the prompt's last position; row *i* is reply token *i* − 1's position and predicts reply token *i*. This is how the
    two servers read C964's replies: the prompt alike, then one token a call (plain decoding) or four (the draft's
    verify pass).
  - `whole`: the prompt and the reply from the first token, *S* tokens a call: `llama-perplexity` at `-ub S`, the
    deep prompt's 8,189 positions included.
- **Arms (one load each):**

  | Arm | Tokens a call | `n_rs_seq` | Role |
  |---|---:|---:|---|
  | `base` | 1 | 0 | the reference: plain decoding's server (`--kl-divergence-base` at `-ub 1`) |
  | `four` | 4 | 3 | the drafting server's verify pass (`--kl-divergence` at `-ub 4`) |
  | `floor` | 1 | 0 | `base` again in a fresh load: the load-to-load floor |
  | `one_rs3` | 1 | 3 | the snapshot setting alone |
  | `four_rs0` | 4 | 0 | the call size alone |

  `n_rs_seq` is the number of recurrent-state snapshots a context keeps for rolling back drafted tokens: llama-server
  sets 3 with the preset's MTP draft of three tokens and 0 without a draft.
- **Per position** (against `base`): the KL divergence of the arm's distribution from the reference's (in double, over
  the whole 248,320-token vocabulary), whether the top token agrees, whether the logit rows are the same bits, and how
  far the gap between the reference's two most likely tokens *a* and *b* moved: in probability,
  (p(a) − p(b)) − (p_ref(a) − p_ref(b)), C964's measure, and in logits, (x(a) − x(b)) − (x_ref(a) − x_ref(b)), which
  does not depend on how likely the two tokens are.
- **Line:** [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)'s `PRESET_ENV` with
  [scripts/pascal_forced.sh](../../../scripts/pascal_forced.sh)'s `pascal_force_preset` (every compute-6.1-only path
  at `2`, a P106's launch shapes, `LLAMA_NO_MOE_DOWN_RESCALE=tg`), and the preset's arguments that act on a load or
  on the arithmetic: `-ngl 99 -fa on -c 65536 -b 4096 -ub 512 -fit off -lm dio --no-host`, the taper
  `short=1400:64`, the layer split `-ts 1,1` on two RTX 3090s. Engine: Miner Llama `pascal-p106-v5` at `2d839da1d`
  (the rig build `llama.cpp-20261125-2d839da1d-pascal-p106-v5-land-batch3`; the tool is linked against that build's
  own tree, whose `llama-perplexity` has the tarball's md5). Target: Swift GSQ-RCO Q2_0, the preset's file.

## The reading rule (fixed before the run)

C964's three flips moved the probability gap between the two candidates by 0.0219 (code), 0.0849 (prose) and 0.0190
(deep), and the logit gap by 0.0467, 0.2459 and 0.0943 (from the probabilities in C964's table). The rule reads the
`four` arm's |logit gap move| over the 768 verify-pass positions of the `reply` reading (rows 1–256 of the three
requests, every output of a four-token call):

1. **The floor first:** the `floor` arm's 99th percentile must be below a tenth of the `four` arm's median, or the
   drift is not resolved from load-to-load noise and nothing below is concluded.
2. **Exact:** if the `four` arm's rows are the same bits as the reference's at 99 % of the positions or more, the
   four-token pass is exact on these card paths, and C964's flips come from something this run does not reproduce.
3. **Rounding everywhere:** if the `four` arm's median is at least a third of the smallest flip's move (0.0467 / 3 =
   0.0156), a typical position moves by the size of the flips, and the flips are that rounding meeting a near tie.
4. **A tail:** if the median is below a tenth of it (0.0047) and all three flips' moves lie above the `four` arm's
   99th percentile, the flips are a few positions that drift much further than the rest (Q971's starting point).
5. **Otherwise** the result is between the two, reported with each flip's percentile rank.

The KL divergence, the same-top-token rate, the positions where C964 flipped (code row 124, prose 67, deep 21, read on
these cards) and the other arms are reported, not judged.
