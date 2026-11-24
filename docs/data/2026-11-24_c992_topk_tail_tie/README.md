# The router-tail test without tie failures (C992, Q1725; 2026-11-24)

What this is: registry row Q1725 in [quants/experiments.md](../../../quants/experiments.md), and its check.
`test-backend-ops` has a fork-only case, `TOPK_MOE_TAIL`, that runs a mixture-of-experts router with the fork's tail
cut (`GGML_OP_MOE_TAIL`: the low-weight end of each token's chosen experts gets weight 0, or no expert at all) and
then an expert product that reads the cut weights and ids. On a card it runs as one fused launch, and the test
compares the card's cut weights, cut ids and product with the processor's.

In [C968's card suite](../2026-11-23_c968_pascal_mmvq_suite/README.md) one such case failed:
`TOPK_MOE_TAIL(n_expert=256,n_tokens=9,n_expert_used=8,with_norm=1,gating_func=1,scale_w=2.5,p=0.8,w=0,renorm=1,n_skip=8,repeat=0)`.
Its cut ids differed (normalized squared error 3.5e-5 against a limit of 1e-7) while its cut weights and its product
matched. Its kept record showed a router tie: with sigmoid gating of logits up to 6, one token's 8th expert and the
first one left out were 1 float32 ulp apart, the card picked the other one, and the tail cuts that 8th slot to weight
0, so the product never reads it. The same happened with the fork's Pascal kernels off. Which random draws tie is
chance, so every suite run could hit it and its failure had to be explained away.

The fork branch `pascal-p106-v5-topk-tail-tie` (`b24c9a675`, one commit on `pascal-p106-v5` at `af00bb13b`, in
`tests/test-backend-ops.cpp` only, no kernel) changes the test in two ways:

- **The check:** in a batch past `n_skip` tokens every slot keeps its own expert and a cut slot gets weight 0, so the
  id of a slot that both the card and the processor cut to weight 0 is not compared there. It must still be a valid
  expert (0 to `n_expert` − 1). Every other id is compared as before: a kept slot's, a skipped slot's −1, and with
  `repeat` the last kept expert. This is what makes the kept record pass.
- **The draw:** each token's logits are drawn again until its `n_expert_used` + 1 best probabilities, computed on the
  host in double precision, are each at least a relative 2^-16 apart. A tie between a kept slot and a cut one would
  send the token to another expert with a non-zero weight; the product then differs and no check on the outputs can
  excuse it, so the draw avoids it. Upstream llama.cpp fixed the same kind of failure in its own router test
  (`TOPK_MOE`) by drawing its logits without ties
  ([#31426](https://github.com/ggml-org/llama.cpp/pull/31426)); upstream has no tail operator, so there is nothing
  of this change to offer it.

Built for the rig (`TESTS=1`) and checked on a build machine with RTX 3090s, which run the rig build's compute 6.1
PTX. Correctness only: nothing here is timed.

## Result in plain words

- **The kept case now passes.** Replayed on its stored inputs, the base fails it with no switch and with the Pascal
  kernels forced (cut ids 3.5e-5), and the branch passes it in both (cut ids 0). The cut weights (5.2e-15) and the
  product (5.7e-15) give the same errors in all four. A live check on the restored inputs gives the same verdicts, and
  the processor's reference is the stored one bit for bit ([kept.tsv](kept.tsv)).
- **No other case changes its verdict.** All 55 `TOPK_MOE_TAIL` cases and the 130 `MOE_TAIL` cases (the tail operator
  alone) were recorded once by each build and replayed by both, with no switch and with the Pascal kernels forced. All
  185 pass in all eight replays, so every verdict is the same for the base and the branch
  ([replay.tsv](replay.tsv), [verdicts.tsv](verdicts.tsv)). On the same inputs the two settings give the same error
  for every output of every case: the Pascal kernels do not touch the router.
- **Live, on new random inputs each run, the base failed 12 of 380 runs and the branch none of 380.** Base: 0 of 40
  with no switch, 3 of 40 with the Pascal kernels forced, 9 of 300 in the soak with no switch. Branch: 0 of 40, 0 of 40
  and 0 of 300. Each run checks all 55 cases ([live.tsv](live.tsv)).
- **Both halves of the change are needed.** Of the base's 12 failures, 4 were a cut slot's id, which the new check
  alone excuses. The other 8 were a near tie between a kept slot and a cut one: the card sent the token to another
  expert with a non-zero weight, so the product itself differed. One of those was a softmax case, the rest sigmoid.
  Only the tie-free draw prevents those. Each failure's errors are in [failures.tsv](failures.tsv).
- **The draw does what it says.** In the base's record, 24 of 473 tokens had two of their best probabilities closer
  than a relative 2^-16, the closest 13 float32 ulps apart. In the branch's record no token did, and the closest pair
  was 299 ulps apart. C968's kept case had a pair 1 ulp apart ([draws.tsv](draws.tsv)).
- **The build:** rig tarball `llama.cpp-20261124-b24c9a675-pascal-p106-v5-topk-tail-tie.tar.gz`, 1,385,276,969 B, md5
  `4e5525495779dc348253c819b0532c28`, `test-backend-ops` md5 `1dd8fd204b9bb521c9c12b3ff406c696`. It has the base's 21
  binaries and web page (b12483) and the same 32 compiler warning lines, none new. Every binary's md5 differs from the
  base's because each one carries the commit id.

## Files

| File | What it holds |
|---|---|
| [kept.tsv](kept.tsv) | C968's kept case replayed with the base and the branch, with no switch and with the fork's Pascal kernels forced: the verdict, each compared output's error, and a live check on the restored inputs |
| [replay.tsv](replay.tsv) | every `TOPK_MOE_TAIL` and `MOE_TAIL` case recorded once by each build and replayed by both, in both settings: cases, passes, failures |
| [verdicts.tsv](verdicts.tsv) | per record and setting, the cases whose verdict differs between the base and the branch |
| [live.tsv](live.tsv) | the live runs of every `TOPK_MOE_TAIL` case, new random inputs each run, per build and setting (live: 40 runs an arm; soak: 300 more with no switch) |
| [failures.tsv](failures.tsv) | each failing live case: which outputs failed, and whether a cut slot's id or the product differed |
| [draws.tsv](draws.tsv) | the routers of every recorded `TOPK_MOE_TAIL` case: tokens whose best probabilities nearly tie |
| [run/run.sh](run/run.sh) | the driver: sources, the rig build, the checks on one card |
| [run/c992_tables.py](run/c992_tables.py) | the tables from the driver's outputs and the records |

## Commands

```bash
# on the build machine, detached; SEED is a llama.cpp clone holding the base, BASE the base's rig build (TESTS=1)
W=<work folder> SEED=<clone> REPO=<copy of this repository> BASE=<base rig build folder> BASE_LOG=<its build log> \
  RIG_CCACHE=<compiler cache to seed> JIT_SEED=<PTX JIT cache to seed> GPU=2 PREFLIGHT='<CUDA check, run as $PREFLIGHT -g 2>' \
  LOCK=<build lock file> JOBS=8 setsid nohup bash run/run.sh > <work folder>/logs/run.out 2>&1 < /dev/null &
# (it ran as two invocations: STEPS='src rig check tables', then STEPS='soak tables' with the same environment)
# the tables from a copy of the work folder, with C968's topk_moe_tail folder copied to <copy>/kept
python3 -I run/c992_tables.py <copy of the work folder> <this folder>
# the failing case replayed alone on any CUDA card, from C968's folder (with this branch's test-backend-ops it passes)
test-backend-ops test -b CUDA0 -o TOPK_MOE_TAIL --replay topk_moe_tail --replay-allow-missing --cpu-check 1 \
    -p '^n_expert=256,n_tokens=9,n_expert_used=8,with_norm=1,gating_func=1,scale_w=2\.500000,p=0\.800000'
```

The p106 setting forces the fork's Pascal matrix-vector kernels with a P106's launch shapes, as C968 did:
`GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2 GGML_CUDA_MMVQ_PASCAL_NSM=5 GGML_CUDA_MMVF_PASCAL_NSM=5
GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5 GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`.

Provenance: the inputs are `test-backend-ops`' own random draws (`std::random_device`), stored in the records; C968's
kept record is in [its folder](../2026-11-23_c968_pascal_mmvq_suite/topk_moe_tail/). Code MIT (llama.cpp's licence).
