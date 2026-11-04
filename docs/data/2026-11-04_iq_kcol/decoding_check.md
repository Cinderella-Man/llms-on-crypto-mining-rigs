# Decoding check of the Q713 kernels on a CPU

[bench/mmvq_pascal/verify/iq_kcol_check.sh](../../../bench/mmvq_pascal/verify/iq_kcol_check.sh) on the tree of fork
commit `0fd6ce5c1` (100 trials per type), in the CUDA 12.9 container on a build machine's CPU, ggml's `to_float` from
the same tree's `libggml-base`. Per type and trial: the k-column unit function at 1, 3 and 8 columns and 1 and 2 rows
(6 cases), and the multi-token expert kernel's arithmetic with each super-block at both half-word parities (2 cases).
Tolerance: 1e-5 of the sum of the absolute terms of a double-precision product of ggml's dequantized weights and the
same Q8_1 activations.

```
iq1s_grid_gpu: the CPU grid plus one as nibbles (weights 0-3 low, 4-7 high)
iq2_xxs  kcol  600 cases worst err/tol 0.0041; mtok  200 cases worst 0.0015, 200/200 bit-identical to the one-token function  PASS
iq2_xs   kcol  600 cases worst err/tol 0.0038; mtok  200 cases worst 0.0013, 200/200 bit-identical to the one-token function  PASS
iq2_s    kcol  600 cases worst err/tol 0.0052; mtok  200 cases worst 0.0014  PASS
iq3_xxs  kcol  600 cases worst err/tol 0.0046; mtok  200 cases worst 0.0011  PASS
iq3_s    kcol  600 cases worst err/tol 0.0041; mtok  200 cases worst 0.0013, 200/200 bit-identical to the one-token function  PASS
iq1_m    kcol  600 cases worst err/tol 0.0065; mtok  200 cases worst 0.0016  PASS
decoding: every type matches ggml's dequantization
```

`MUTATE=1`: a copy of `mmvq-pascal-iq.cuh` with three planted faults in `decode32` (IQ2_S's grid index without its
second high bit, IQ3_S's second index of a pair given the first one's ninth bit, IQ1_M's negative-delta table entry for
grid value +1 set to 9 instead of 7). Summary lines (2,400 case-failure lines omitted):

```
iq2_xxs  kcol  600 cases worst err/tol 0.0041; mtok  200 cases worst 0.0015, 200/200 bit-identical to the one-token function  PASS
iq2_xs   kcol  600 cases worst err/tol 0.0038; mtok  200 cases worst 0.0013, 200/200 bit-identical to the one-token function  PASS
iq2_s    kcol  600 cases worst err/tol 12254.2976; mtok  200 cases worst 9063.3540  FAIL
iq3_xxs  kcol  600 cases worst err/tol 0.0046; mtok  200 cases worst 0.0011  PASS
iq3_s    kcol  600 cases worst err/tol 9373.0274; mtok  200 cases worst 7233.0533, 0/200 bit-identical to the one-token function  FAIL
iq1_m    kcol  600 cases worst err/tol 2785.6178; mtok  200 cases worst 2469.7606  FAIL
decoding: FAILURES
mutation: fails exactly iq2_s, iq3_s and iq1_m, as it must
```

The check covers the bit layouts, the grids, the signs, the scales and the record layout (IQ1_M's activation order);
the kernels' staging, lane groups, group search and reductions are checked on a GPU
([run/gpu_checks.sh](run/gpu_checks.sh)).
