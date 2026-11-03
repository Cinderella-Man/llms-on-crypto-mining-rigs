# The k-column kernels' decoding against ggml's reference, on a CPU

`bench/mmvq_pascal/verify/kcol_unit_check.sh 40` (2026-11-03, `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no GPU): each
type's unit function from `src/kcol_mm.cuh` (fork `pascal-p106-v3-b12076-kcol` at `e1fa4c573`), built for the host,
against a double-precision product of ggml's reference dequantization and the activations of the records, which the
kernels' quantizer, written for the host, makes from random activations. Per type 40 trials of 1, 3 and 8 columns at
1 and 2 rows, 4 lane units per row, odd units placed at the smallest misalignment the staging buffer gives the type
(and a third at 2 + 4 bytes for the 2-byte-aligned types); every fourth trial with activations of scale 3000, whose
32-value sums pass FP16's range; one all-zero block per column. Tolerance 1e-5 of the sum of the absolute terms.

```
q8_0     240 cases  worst err/tol 0.0029  PASS
q2_0     240 cases  worst err/tol 0.0029  PASS
q3_K     240 cases  worst err/tol 0.0025  PASS
q4_K     240 cases  worst err/tol 0.0044  PASS
q5_K     240 cases  worst err/tol 0.0041  PASS
q6_K     240 cases  worst err/tol 0.0036  PASS
iq4_xs   240 cases  worst err/tol 0.0028  PASS
decoding: every type matches ggml's reference
```

The same check, 8 trials, on a copy with two planted errors (the misaligned units' 2-byte shift dropped,
`u.sel = 0x3210`; Q4_K's high-nibble scale 1/8 instead of 1/16) fails exactly the types they touch:

```
q8_0      48 cases  worst err/tol 0.0029  PASS
q2_0      48 cases  worst err/tol 0.0013  PASS
q3_K      48 cases  worst err/tol 1000000000000000019884624838656.0000  FAIL
q4_K      48 cases  worst err/tol 13790.2225  FAIL
q5_K      48 cases  worst err/tol 0.0026  PASS
q6_K      48 cases  worst err/tol 1000000000000000019884624838656.0000  FAIL
iq4_xs    48 cases  worst err/tol 0.0024  PASS
decoding: FAILURES
```

(1e30 marks a non-finite result: the unshifted words of a misaligned unit put garbage into its FP16 scale.)
