# Does the taper's `cuts=1` change Flash-Next's next-token numbers? (C981, Q1714; 2026-11-25)

DRAFT: written before the run; results follow.

## The rule, fixed before the run

`cuts=1` passes if its mean KL divergence against `short=1400:64` is at most **0.028**, a tenth of Swift GSQ-RCO
Q2_0's own distance from Swift's Q8_0 on this agent text (0.285104, [FN2](../2026-11-03_flash_next_quality/README.md);
FN95's rule, [2026-11-16_fn95_flashnext_taper_kld](../2026-11-16_fn95_flashnext_taper_kld/README.md)). The mean is
taken over the positions `cuts=1` can change: in each of the 14 call sizes whose micro-batches it changes, every
position from the call's first micro-batch that differs between the two plans to the call's end, plus the 16 tokens
decoded after the call, over the fresh calls (8 per size).

Reported beside it, not judged: the same mean for `short=1400:64` against the taper off (what the preset's taper
already moves), the share of positions with the same top token, the log-probability of the text's next token
(perplexity), the repeat of `short=1400:64` (the floor), and the two control sizes whose plan `cuts=1` leaves as it is
(expected: the same bits).
