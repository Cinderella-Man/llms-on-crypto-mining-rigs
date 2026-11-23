# Q1707: does a Q8_0 cache keep Ornith-1.5-122B's quality? KLD at 8,192 to 49,152 tokens of agent text (2026-11-23)

Running; this page gets the result when the runs end.

## Pass band (fixed before the run)

Each band is wider than the measurement's own error, so noise alone cannot fail or pass it.

- **(A) Against Ornith-1.5-122B's own Q8_0 model** (its reference log-probabilities on the agent text, 12 chunks of
  8,192 tokens, from [2026-10-27_ornith_kld](../2026-10-27_ornith_kld/README.md)): the Q8_0 cache's mean KLD at most
  0.003 above the F16 cache's, and its same top token at most 0.3 points below. One file's standard error on this text
  is about 0.001 in KLD and 0.1 points in same top token, so the bands are three of them, and more than twice the error
  of a difference of two independent runs.
- **(B) The Q8_0 cache against the F16 cache**, at each depth (8,192, 16,384, 32,768, 49,152): mean KLD at most
  the larger of twice the rounding yardstick's (the F16 cache read in micro-batches of 128 instead of 512, the same sums
  in another order) and 0.003; the share of tokens whose top token differs at most the larger of twice the yardstick's
  and 0.3 points; the mean log-ratio of the perplexities within ±0.01 (1 %).
- **(C) The decode path at 8,192** (every token its own pass): the Q8_0 cache through the base's kernels and through
  pull request 9's decode kernel, each against the F16 cache in the same passes, by (B)'s rule, with F16 read in
  micro-batches of 512 as the yardstick.
- **Validity:** a second load of the F16 reference must repeat it (KLD ≤ 0.0001, same top token 100 %), and the Q8_0
  cache with the Pascal prompt kernel's output scaled by 1.05 (its fault switch) must land outside (B)'s band at 8,192.

The Q8_0 cache keeps Ornith's quality if (A) holds and (B) holds at every depth; (C) says whether the decode kernel
that the rig run uses keeps it too.
