# The fork's tools on the new batch calls, on three RTX 3090s (C957, Q1687 point 11; 2026-11-23)

What this is: the GPU half of the check of [C880](../2026-11-21_c880_batch_api/README.md) (registry row Q1589 in
[quants/experiments.md](../../../quants/experiments.md); fork branch `pascal-p106-v5-batch-api` `aebff903e`, one commit
on `pascal-p106-v5` `3dfad876f`). That commit moves the fork's own tools and tests off upstream's deprecated batch
calls (`llama_batch_init`, `llama_decode`, `llama_batch_free`) onto the new batch object and its one call
(`llama_batch_ext`, `llama_process`). C880 checked it on a processor. Here the eight changed tools and tests run on a
build machine's three RTX 3090s, which run the rig build's compute 6.1 PTX, with the random models' layers split over
the three cards. It is the last of the eleven branches of registry row Q1687, the GPU half of the branches built on the
night of 2026-11-20/21. Correctness only: nothing here is timed.

## Result in plain words

- **The branch gives the base's output in every comparison.** On the cards, with the same random models, the eight
  tools and tests gave the same exit codes, the same result lines and the same logs line for line on both builds: 129
  of 129 comparisons with every switch at its default (the same 129 C880 made on the processor), 81 of 81 with the
  fork's Pascal kernels forced on, and 77 of 77 on the same models quantized to Q8_0 with those kernels forced.
- **Every test passes as on the base.** `test-moe-ids` (24 matrix products by expert on the cards, no repeated or
  out-of-range id), `test-hop-cast` (216 round-trip checks on the processor and each card, 236 checks of the model
  split over the three cards with four ways of crossing between cards), the 24 arms of the three pool tests, and
  `llama-screen-check` greedy and sampled: all pass, on both builds, with the same counts.
- **The runs repeat:** the base run twice gave the same output in all 129 comparisons, so "the same" above is a
  verdict, not luck.
- **The fork's Pascal kernels ran where they can.** On a 3090 they stay off unless forced, so the forced runs set them
  on with a P106's launch shapes, as [C953](../2026-11-23_c953_ornith_kcol_gpu/README.md) found is needed, and a
  one-token probe listed `MMVQ_PASCAL = 1` and `MMVF_PASCAL = 1` among the cards' features (none without). C880's random
  models hold only F32 weights, which those matrix-vector kernels never take (they take quantized weights, and BF16 for
  the F16 twin): forcing them changed nothing on those models. On the Q8_0 copies they computed the products: forcing
  them changed the logits, KV state and probe hashes of all 56 pipespec job lines (plain decoding's references
  included), and not one token.
- **Two older faults show the same on both builds, as on the processor.** Every pipelined pipespec job differs from
  plain decoding in its logits (C932's stale delta-net histories, fixed on its own branch and checked on these cards by
  [C948](../2026-11-21_c948_pipe_fix_gpu/README.md)), so the five pipespec checks exit 1 and
  `llama-pipe-recurrent-check` exits 2 on both builds. On the F32 models the pipelined tokens still equal plain
  decoding's; on the Q8_0 models the sampled jobs' tokens part from it too (26 of 50 jobs keep plain decoding's
  tokens, every greedy one among them).
- **The warnings:** with the six extra tools compiled, the base's rig build gives 56 warning lines, 24 of them the eight
  files' deprecated batch calls; the branch gives 32, none of them new.
- **Found on the way (Q1716):** `llama-quantize`'s rules quantize Flash-Next's `ple_norm_key`, `ple_norm_query` and
  `ple_norm_conv` tensors, which the graph multiplies elementwise, and no backend multiplies by a Q8_0 operand: a
  random Flash-Next file quantized with the default rules stops at its first pass (`binary_op: unsupported types`).
  Kept F32 (`--tensor-type ple_norm=f32`), the file runs. The release files keep those tensors F32.

## What ran

Two rig builds made the same way on the build machine's processor (`scripts/build_llamacpp_docker.sh`, `TESTS=1`,
`JOBS=8`, the web page pinned to `b12483`, the six tools the rig build otherwise leaves out as `EXTRA_TARGETS`:
`llama-pipe-recurrent-check test-moe-ids test-hop-cast test-kpool-dev test-pool-mask-dev test-kpool-stage
test-llama-archs test-cuda-keepalive`):

| Build | Commit | Tarball md5 | Note |
|---|---|---|---|
| branch | `aebff903e` (`pascal-p106-v5-batch-api`) | `483648fae76a2ba26ea1fd15d9a8743e` | `llama-pipespec` `3988f23b6096e8e62acc207e3407a628` |
| base | `3dfad876f` (`pascal-p106-v5-pipe-grammar-off`, = C908's) | `d5dd167fd6e695496b1c974431b0c2e5` | `llama-server`, `llama-pipespec` and `test-backend-ops` bit for bit C908's tarball's |

Each run is [run/gpu_tools.sh](run/gpu_tools.sh) (C880's `tools.sh` with the models' layers on the cards: `-ngl 99
-ts 1,1,1`) with one build, one kernel setting and one set of models, in its own container with the three cards
([run/run.sh](run/run.sh)):

| Run | Models | Setting | Parts |
|---|---|---|---|
| f32-stock-base, f32-stock-head, f32-stock-base2 | F32 | stock | all: pipespec, pipe-recurrent-check, screen-check, moe-ids, hop-cast, the pool tests |
| f32-p106-base, f32-p106-head | F32 | p106 | all but the pool tests |
| q8-p106-base, q8-p106-head | Q8_0 | p106 | pipespec, pipe-recurrent-check, moe-ids, hop-cast |
| q8-stock-base | Q8_0 | stock | the same four |

- **Models:** C880's random models (`test-llama-archs -a qwen4exp -s 1` with a 4,096-token vocabulary: a Flash-Next MoE
  model, the same with 32 experts and 8 used, and the first with a screened head; md5s the same as C880's,
  [models.md5](models.md5)), every weight F32. The Q8_0 copies of the first two: `llama-quantize` with
  `--tensor-type hc_=f32 --tensor-type ple_norm=f32` (24 tensors Q8_0, among them every attention, delta-net and
  expert matrix; [model_types.txt](model_types.txt)). Only the `ple_norm` exception is needed (a file with only
  `hc_*` kept F32 stops as above).
- **Settings:** stock = every switch at its default (a 3090 runs llama.cpp's own matrix-vector kernels); p106 =
  `GGML_CUDA_MMVQ_PASCAL=2 GGML_CUDA_MMVF_PASCAL=2`, a P106's launch shapes (`GGML_CUDA_MMVQ_PASCAL_NSM=5`,
  `GGML_CUDA_MMVF_PASCAL_NSM=5`, `GGML_CUDA_MMVQ_PASCAL_KCOL_NSM=5`, `GGML_CUDA_MMVQ_PASCAL_KCOL_SMSM=98304`) and
  `GGML_CUDA_FATTN_SPARSE_PASCAL=2`.
- **The tools:** `bench/pipespec/check_identity.sh` (greedy at depths 1 and 3 with 0 and 50 % wrong guesses, the prompt
  in calls of 7 tokens, seeded sampling, a race of two samples) and `check_streams.sh` (two conversations at once);
  `llama-pipe-recurrent-check` (three pipe slots with forced rollbacks); `llama-screen-check` greedy and sampled;
  `test-moe-ids` (every layer on the cards, each expert product's ids read back after it ran); `test-hop-cast
  --unit-only` (round trips through every narrow type on the processor and each card, each card against the
  processor) and with the model split over the three cards in four arms (off as the reference, run twice; the residual
  cast to Q8_0 at each card boundary; every hop through a pinned host ring, which must be exact; every hop through
  mapped host memory, which must be exact); `test-kpool-dev`, `test-pool-mask-dev`, `test-kpool-stage` in 8 arms each
  (these three pin their model to the processor backend by design, as on C880's run; here they run from the CUDA
  build).
- **Compared** with C880's [compare.py](../2026-11-21_c880_batch_api/run/compare.py): every exit code, every pipespec
  result line in all fields but the timings (tokens, logits, KV state and probe as hashes), every other log line for
  line after dropping the loader's and the clock's lines, and `test-hop-cast`'s table.

## The comparisons ([checks.tsv](checks.tsv))

| Run a | Run b | What it shows | Same | Differ |
|---|---|---|---|---|
| f32-stock-base | f32-stock-head | the check, every switch at its default | 129 of 129 | 0 |
| f32-p106-base | f32-p106-head | the check, Pascal kernels forced | 81 of 81 | 0 |
| q8-p106-base | q8-p106-head | the check, Q8_0 models, Pascal kernels forced | 77 of 77 | 0 |
| f32-stock-base | f32-stock-base2 | the runs repeat | 129 of 129 | 0 |
| f32-stock-base | f32-p106-base | the setting's reach on F32 weights | 81 of 81 run in both | 48: the pool tests, run only in the first |
| q8-stock-base | q8-p106-base | the setting's reach on Q8_0 weights | 21 of 77 | 56: logits, KV state and probe hashes of every pipespec job line (7 also the race's hash); no token |

Each test's verdict per run is in [tests.tsv](tests.tsv), every exit code in [rc.tsv](rc.tsv), and every pipespec
job against its own reference (plain decoding, or the conversation's plain run) in [pipespec.tsv](pipespec.tsv):

| Runs | Pipelined jobs | Tokens as plain decoding's | Logits, state, probe as plain decoding's |
|---|---|---|---|
| f32 (all five runs, each) | 50 | 50 | 0 |
| q8 (all three runs, each) | 50 | 26: the 13 greedy, 6 prompt-in-calls and 7 of 14 conversation jobs; none of the 17 seeded or raced | 0 |

## Files

| File | What |
|---|---|
| [checks.tsv](checks.tsv) | one row per compared pair of runs: what it shows, compare.py's counts, what differs |
| [pairs/](pairs/) | compare.py's table of each pair, item by item |
| [rc.tsv](rc.tsv) | every tool's and test's exit code in every run |
| [tests.tsv](tests.tsv) | every test's verdict line in every run |
| [pipespec.tsv](pipespec.tsv) | every pipespec job of every run: tokens, hashes, and whether they equal the job's reference's |
| [models.md5](models.md5), [model_types.txt](model_types.txt) | the models' md5s and tensor types |
| [rig-warnings.txt](rig-warnings.txt) | the two rig builds' warning lines, branch against base |
| [run/run.sh](run/run.sh) | the build machine's driver: the Q8_0 copies, the eight runs, the probe, the comparisons |
| [run/gpu_tools.sh](run/gpu_tools.sh) | one run: the eight tools and tests with one build |
| [run/summary.py](run/summary.py), [run/model_types.py](run/model_types.py) | the tables above; a GGUF file's tensor types from its header |

## Commands

On the build machine, the two rig builds (from a llama.cpp clone with the fork as remote `fork`, in `$WORK/rig/src`):

```bash
for ref in fork/pascal-p106-v5-batch-api fork/pascal-p106-v5-pipe-grammar-off; do
  WORK=$WORK/rig TESTS=1 JOBS=8 HF_UI_VERSION=b12483 DOCKER_ARGS='--label <label> --cpu-shares=256' \
    EXTRA_TARGETS="llama-pipe-recurrent-check test-moe-ids test-hop-cast test-kpool-dev test-pool-mask-dev test-kpool-stage test-llama-archs test-cuda-keepalive" \
    bash scripts/build_llamacpp_docker.sh $ref
done
```

Then, with both tarballs unpacked, C880's random models in `$MD` (its `models` step), a copy of this repository in
`$WORK/repo`, and a processor build's `llama-quantize`:

```bash
WORK=<work dir> NEW=<branch build>/bin REF=<base build>/bin REPO=$WORK/repo MD=<models> QUANT=<llama-quantize> \
  DOCKER_ARGS='--label <label> --cpu-shares=256' PREFLIGHT=<CUDA check> \
  setsid nohup bash $WORK/repo/docs/data/2026-11-23_c957_batch_api_gpu/run/run.sh > $WORK/driver.log 2>&1 < /dev/null &
python3 -I run/summary.py <WORK> .
```

It ran on 2026-11-23: the builds 18:33–18:37 (the branch, then the base), the eight runs 18:37–18:49 (about 1.5
minutes a run with the pool tests, 40 seconds without). The Q8_0 runs ran twice: the first Q8_0 copies, made with only
`llama-quantize`'s default rules, stopped every tool that loads them on both builds alike (the `ple_norm` finding
above), and the copies used here were made with the two exceptions, followed by the probe and the comparisons.

## Provenance and licence

The engine is Miner Llama (llama.cpp's MIT licence). The models are random-weight test models that `test-llama-archs`
makes, and Q8_0 copies of them; the prompts are this repository's `bench/pipespec/prompts/`. No model file, download
or private data. Tables and scripts here: the repository's licence.
