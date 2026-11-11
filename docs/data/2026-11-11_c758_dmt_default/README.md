# C758 (Q1335): C722's read-once kernel at one row per lane group by default (2026-11-11)

What this changed and checked: the default of `GGML_CUDA_MMVQ_PASCAL_DMT=1`, the switch for C722's read-once IQ4_XS and
Q4_K kernel ([design](../2026-11-11_c722_short_batch_products/README.md), [docs/switches.md](../../switches.md)). It now
launches the setting R79 found fastest on a P106 ([R79](../2026-11-11_r79_c722_p106/README.md)). The checks ran on a
build machine's CPU, without a GPU. They show that every kernel the new default launches is one R79 ran, the same
SASS byte for byte, with the same launch shape, and that nothing else in the build changed. The registry row is Q1335 in
[quants/experiments.md](../../../quants/experiments.md).

## Result

**`DMT=1` alone now runs what R79 measured as `DMT=1 DMT_R=1`, except IQ4_XS's 2-token products at K = 2,560, which
run R79's faster `DMT=2 DMT_R=1`. Each of those kernels has the base build's SASS, byte for byte.** Nothing else
changed: all 193 objects of the CUDA backend compile to the base's SASS, the random-model logits of 146 architectures are
the base's, and ctest passes 70 of 70. R79's kernel times and its served verify pass stand for the new default (below).

- **Fork:** `pascal-p106-v4-b12246-c758` at `59a147df5`, one commit on `pascal-p106-v4-b12246-c748` `5b329df16`. It changes
  host code only, in `ggml/src/ggml-cuda/mmvq-pascal-dmt.cu` and its header; the device code and its line numbers are
  untouched.
- **Rig tarball:** `llama.cpp-20261111-59a147df5-pascal-p106-v4-b12246-c758.tar.gz`, 656,338,683 B, md5
  `9bc26b0b222e16e5df5f36b033834f0a`, built with `TESTS=1`. Its `test-backend-ops` md5 is
  `236247c9094f169d6cdae8b0c99bb475`. No AVX instructions; the highest glibc symbol is GLIBC_2.38.

### The new default and the old settings

R = 1 is the default for both types and every K: one row per lane group, as in the one-token kernel. The register
prefetch of each warp's next weight segments stays on, except at R = 1 for IQ4_XS's 2-token products at K = 2,560.
`_R=0` keeps C722's choice per product. `_PF=1` is new: it turns the prefetch on for every product. Every earlier
setting still runs exactly what it ran ([pick_spellings.tsv](pick_spellings.tsv), each pair 0 lines apart over 384
products):

| Before C758 | Since C758 |
|---|---|
| `DMT=1` (R chosen per product) | `DMT=1 DMT_R=0` |
| `DMT=2` | `DMT=2 DMT_R=0` |
| `DMT=1 DMT_R=1` (R79's `r1`, its served `on`) | `DMT=1 DMT_PF=1` |
| `DMT=1 DMT_R=2` | `DMT=1 DMT_R=2` |
| `DMT=2 DMT_R=1` (R79's `r1nopf`) | `DMT=2` or `DMT=2 DMT_R=1` |
| `DMT=2 DMT_R=2` | `DMT=2 DMT_R=2` |
| `DMT=1 DMT_NW=2` | `DMT=1 DMT_R=0 DMT_NW=2` |

Why this default, from R79's `test-backend-ops perf` on one P106-090 (µs per product, pass 1 / pass 2):

| Product | Tokens | C722's rule (`DMT=1`) | R = 1 (`r1`) | R = 1 without the prefetch (`r1nopf`) |
|---|---:|---:|---:|---:|
| IQ4_XS 10,240 × 2,560 (`attn_qkv`) | 4 | 240.21 / 237.59 | 146.31 / 147.45 | 171.38 / 174.03 |
| IQ4_XS 2,560 × 6,144 | 4 | 228.20 / 230.01 | 113.15 / 113.08 | 142.57 / 142.69 |
| IQ4_XS 6,144 × 2,560 | 2 | 126.21 / 127.33 | 94.00 / 95.42 | 75.74 / 75.90 |
| IQ4_XS 10,240 × 2,560 | 2 | 194.49 / 194.50 | 153.56 / 151.83 | 117.74 / 118.73 |
| IQ4_XS 12,288 × 2,560 | 2 | 228.62 / 228.20 | 180.54 / 180.89 | 135.00 / 136.04 |
| IQ4_XS 2,560 × 6,144 | 2 | 120.03 / 121.02 | 80.35 / 80.70 | 82.55 / 81.83 |
| Q4_K 6,144 × 2,560 | 2 | 80.73 / 79.23 | 64.04 / 64.13 | 63.46 / 63.45 |
| Q4_K 10,240 × 2,560 | 2 | 120.89 / 120.74 | 97.83 / 97.92 | 97.45 / 97.47 |

- **R = 2 is slower wherever C722's rule took it:** both types at K = 2,560 and IQ4_XS at K = 6,144.
- **The prefetch costs time only at IQ4_XS's 2 tokens at K = 2,560.** At 3 and 4 tokens, at K = 6,144 and at R = 2 it
  is faster. At Q4_K's 2 tokens at K = 2,560 the two are within the spread (2.0 % and 2.6 %), so the prefetch stays
  on there.
- **IQ4_XS's 2 tokens at K = 2,560 is the only place the new default parts from `r1`.** Every other product runs the
  kernel R79 served as `DMT=1 DMT_R=1`, with the same launch shape. R79's served verify pass therefore stands for
  `DMT=1`: the IQ4_XS and Q4_K products 8.291, 8.305, 8.314 and 8.313 ms a pass against 16.212, 16.221, 16.202 and
  16.214 without the switch, and the warm step 114.01 and 113.62 ms against 122.10 and 122.03 (second request). A
  verify pass of 2 tokens runs its IQ4_XS products at K = 2,560 on `r1nopf`, which R79 timed faster than `r1` at the
  three such shapes it measured (the table).

### Checks ([summary.tsv](summary.tsv))

| Check | Result |
|---|---|
| SASS of every object | 193 objects of the CUDA backend, each one's `cuobjdump -sass` with the anonymous namespaces' hashes masked: 193 the base's, none new ([sass_objects.tsv](sass_objects.tsv)) |
| SASS of each read-once kernel | the 24 instantiations `mul_mat_vec_q_pascal_dmt<type, NT, R, PF>` in `mmvq-pascal-dmt.cu.o`, one by one: 24 the base's; registers 72–204, no stack or local memory ([kernels.tsv](kernels.tsv)) |
| What each product launches | `run/pick.cu` built against each tree's own `mmvq-pascal-dmt.cu` and run under 17 settings, 384 products each (both types, 2–4 tokens, rows of 1 to 64 super-blocks). The tip's `DMT=1` takes the same 219 products as before and declines the same 165. It launches the base's `DMT=1 DMT_R=1` kernel, warps, shared memory and staging stride for all of them except IQ4_XS, 2 tokens, K = 2,560, where it launches the base's `DMT=2 DMT_R=1` ([pick_default.tsv](pick_default.tsv)) |
| Flash-Next's shapes | the 12 products at K = 2,560 and 6,144: each kernel the new default launches, the R79 arm that ran it, and its SASS in both builds ([pick_flashnext.tsv](pick_flashnext.tsv)) |
| Every switch off | `test-llama-archs -b CPU -s 1` (a static CPU build of the tip against the base's): 146 architectures, the same logits with no switch and with `GGML_CUDA_MMVQ_PASCAL_DMT=1` |
| ctest | 70 of 70 in the tip's CPU build (without the tests that need a download, the network or torch) |
| Harness | `bench/mmvq_pascal/verify/multitoken_bitid` linked against the tip's build tree, md5 `889e4817fabc1761a1634b1d3eced18a`, for the rig check below |

### Not checked here: one kernel's bit identity on a P106

`mul_mat_vec_q_pascal_dmt<iq4_xs, 2, 1, false>` is the kernel the new default runs for IQ4_XS's 2 tokens at K = 2,560.
R79's `multitoken_bitid dmt` never ran it. None of its four settings reached this kernel at its shapes (K up to 8,192):
under `DMT=2` the old rule takes it only from K = 9,728 up (`run/pick.cu` built with `-DC758_BASE` against the base
tree, run with `GGML_CUDA_MMVQ_PASCAL_DMT=2`). R79 did time this kernel (`r1nopf`), and on a P106 the night merge's
replay run `dmt-r1-nopf` (`DMT=2 DMT_R=1`) passed 315 of 315 cases with it
([2026-11-11_v4night_replay](../2026-11-11_v4night_replay/README.md)).
Its sums are the one-token kernel's by construction: the prefetch moves when the next segment's weights are loaded,
not what is computed. The rig check closes it: on one P106, with the harness above,

```bash
env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 \
    GGML_CUDA_MMVQ_PASCAL_DMT=1 ./multitoken_bitid dmt        # expect 0 of 60 with differences
env CUDA_VISIBLE_DEVICES=0 GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768 GGML_CUDA_MMVQ_PASCAL_KCOL_TABLE=0 \
    ./multitoken_bitid dmt                                    # the control (the k-column kernel): 50 of 60, as R79
```

## Files

| File | What |
|---|---|
| [summary.tsv](summary.tsv) | every check's verdict and detail |
| [sass_objects.tsv](sass_objects.tsv) | per object of the CUDA backend: the masked SASS md5 in the base and the tip |
| [kernels.tsv](kernels.tsv) | per read-once kernel: the masked SASS md5 in the base and the tip, registers, stack, local memory |
| [pick_default.tsv](pick_default.tsv) | the tip's `DMT=1` per product (type, tokens, K): R, prefetch, warps, lanes per row, shared memory, staging stride, kernel, and the base setting it equals |
| [pick_spellings.tsv](pick_spellings.tsv) | each earlier setting against its spelling on the tip: lines that differ over 384 products |
| [pick_flashnext.tsv](pick_flashnext.tsv) | Flash-Next's 12 products: the kernel, R79's arm, the SASS in both builds |
| [run/checks.sh](run/checks.sh) | the checks: build, SASS, pick, CPU build, `test-llama-archs`, ctest (its header) |
| [run/pick.cu](run/pick.cu) | the launch choice of a tree's `mmvq-pascal-dmt.cu`, printed per product without a GPU |

## Commands

On the build machine, all in containers without GPUs at low CPU priority (`--cpu-shares=256`, nice 19), one at a time
under a lock. `WORK/src` is a clone of `5b329df16`'s llama.cpp tree with the fork as remote `fork`. `BASE_BUILD` is
`5b329df16`'s rig build tree, from the same build script. `BASE_CPU` is the `bin/` of `5b329df16`'s static native CPU
build.

```bash
WORK=<work dir> REPO=<this repository> BRANCH=pascal-p106-v4-b12246-c758 BASE=5b329df16 \
    BASE_BUILD=<5b329df16's rig build tree>/build BASE_CPU=<5b329df16's CPU build>/bin LOCK=<lock file> JOBS=8 \
    CPU_CCACHE_SEED=<a ccache folder> bash docs/data/2026-11-11_c758_dmt_default/run/checks.sh
# the harness for the rig check
bash bench/mmvq_pascal/verify/build_multitoken_bitid.sh "$WORK/src" "$WORK/src"
```

The first run's `pick` step linked the harness with unresolved symbols ignored. That binary did not load ("unexpected
PLT reloc type"). The committed script links with section garbage collection instead, which leaves no unresolved
symbol. Only that step was run again (`STEPS=pick`), after the other steps had passed; `summary.tsv` holds the second
run's `pick` lines.

## Provenance and licence

The code is Miner Llama's fork of llama.cpp (MIT). The tables were generated from the build of that code by the
scripts in `run/`; no model or data file was used apart from `test-llama-archs`' random models.
