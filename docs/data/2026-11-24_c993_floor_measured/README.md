# Q1726: the per-stretch check's floor on Ornith-1.5-122B, measured (2026-11-24)

[C989](../2026-11-24_c989_attn_error_floor/README.md) estimated how small an error in the Pascal prompt attention
kernel the per-stretch quality check on Ornith-1.5-122B Pascal N2P GGUF (`valid1`) sees against the F16 cache: est.
3.43 % in KLD and 1.37 % in top-token disagreement, scaled down from one fault of 5 %. This folder measures it instead:
the same check with the kernel's output multiplied by 1.01, 1.02, 1.03 and 1.05, set by the fault switch's new size
([2026-11-24_c993_fault_scale](../2026-11-24_c993_fault_scale/README.md), `GGML_CUDA_FATTN_TILE_PASCAL_FAULT_SCALE`)
([Q1726](../../../quants/experiments.md)). It is a correctness check on three RTX 3090s through the rig build's compute
6.1 PTX; no time on these cards means anything for a P106.

**Result:** pending (the run started 2026-11-24 19:45).

## The rule, fixed before the run

- **Pairs:** for each factor f, the fault arm (`q8f101`, `q8f102`, `q8f103`, `q8f105`) minus the unfaulted Q8_0 arm
  (`q8`), stretch by stretch over C976's 12 stretches, both scored against the F16 cache's reference written in the
  same run. That is C989's `q8f1 − q8` form.
- **Seen:** a factor is seen when the mean per-stretch difference is at least 3 standard errors above zero in KLD or in
  top-token disagreement (both one-sided). The ln PPL ratio difference is reported two-sided and is not part of the
  answer.
- **The answer:** the smallest factor seen, by KLD, by top-token disagreement and by either, against C989's est. 3.43 %
  and 1.37 %.
- **Validity:** every log names the compute 6.1 prompt attention kernel at head 256 with its cache type; every fault arm
  logs `GGML_CUDA_FATTN_TILE_PASCAL_FAULT=1 (tests of the tests): output x <its factor>` and no other arm logs it.
  Expected, not required: `q8` and `q8f105` repeat C989's `q8` and `q8f1` on every stretch and the reference repeats
  C989's perplexity (this build is `pascal-p106-v5` `af00bb13b` plus the switch, C989's was `3dfad876f`; with the
  switch unset the kernels are the base's bytes). If they differ, the comparison still stands within this run.
- **Expected from C989's and [C994](../2026-11-24_c994_attn_fault_selfref/README.md)'s numbers (est.):** the
  difference against the F16 cache has two parts. The fault's own KLD grows as the square of its size (C994: 0.011030
  at 5 %), and a cross term with the Q8_0 cache's own error grows in proportion to it (C994: −0.006750 at 5 %, the part
  of the fault that the F16 reference hides). If both hold at smaller sizes, the mean KLD difference is est. −0.00091 at
  × 1.01, −0.00094 at × 1.02, −0.00008 at × 1.03 and +0.00428 at × 1.05, so no factor below 1.05 would be seen in KLD,
  and C989's est. 3.43 % (which scaled the whole difference as the square) would be too low.
