# The k-column kernels' decoding against ggml's reference, on a CPU (with Q4_0, Q5_0, IQ4_NL)

`bench/mmvq_pascal/verify/kcol_unit_check.sh 40` (2026-11-04, `nvidia/cuda:12.9.2-devel-ubuntu24.04`, no GPU): each
type's unit function from `src/kcol_mm.cuh` (fork `pascal-p106-v3-b12076-kcol` with Q630's commit), built for the host,
against a double-precision product of ggml's reference dequantization and the activations of the records, which the
kernels' quantizer, written for the host, makes from random activations. The method is
[Q591's](../2026-11-04_kcol_dense/decoding_check.md): per type 40 trials of 1, 3 and 8 columns at 1 and 2 rows, 4 lane
units per row (here 256 weights for the three new types), odd units at the smallest misalignment the staging buffer
gives the type (4 bytes for the new types); every fourth trial with activations of scale 3000, whose 32-value sums pass
FP16's range; one all-zero block per column. Tolerance 1e-5 of the sum of the absolute terms.

```
q8_0     240 cases  worst err/tol 0.0029  PASS
q2_0     240 cases  worst err/tol 0.0029  PASS
q3_K     240 cases  worst err/tol 0.0025  PASS
q4_K     240 cases  worst err/tol 0.0044  PASS
q5_K     240 cases  worst err/tol 0.0041  PASS
q6_K     240 cases  worst err/tol 0.0036  PASS
iq4_xs   240 cases  worst err/tol 0.0028  PASS
q4_0     240 cases  worst err/tol 0.0030  PASS
q5_0     240 cases  worst err/tol 0.0030  PASS
iq4_nl   240 cases  worst err/tol 0.0037  PASS
decoding: every type matches ggml's reference
```

The first seven lines are unchanged from Q591's run: IQ4_XS's lookup, now one helper shared with IQ4_NL, gives the
same results.

[run/mutation_check.sh](run/mutation_check.sh), 8 trials, on a copy with three planted errors (Q4_0's offset −7
instead of −8; Q5_0's first-block `qh` read without the 2-byte shift; IQ4_NL's code words 0/1 and 2/3 swapped) fails
exactly the types they touch:

```
q8_0      48 cases  worst err/tol 0.0029  PASS
q2_0      48 cases  worst err/tol 0.0013  PASS
q3_K      48 cases  worst err/tol 0.0025  PASS
q4_K      48 cases  worst err/tol 0.0033  PASS
q5_K      48 cases  worst err/tol 0.0026  PASS
q6_K      48 cases  worst err/tol 0.0036  PASS
iq4_xs    48 cases  worst err/tol 0.0024  PASS
q4_0      48 cases  worst err/tol 5454.3075  FAIL
q5_0      48 cases  worst err/tol 24867.4288  FAIL
iq4_nl    48 cases  worst err/tol 39606.0179  FAIL
decoding: FAILURES
```
