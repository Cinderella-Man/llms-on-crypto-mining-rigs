# What a speculative step costs beyond the cards' GPU time (C338 step 1, 2026-11-04)

What this measured: one speculative step of Flash-Next's MTP draft line on a 4 GB rig of nine P106-090, split into
the main model's verify pass on the cards, the draft's passes, and the time no card's clock holds. The question
([Q900](../../../quants/experiments.md)) was how much of a real step `llama-bench`'s k-token passes miss. A verify
pass copies one row of logits per verified position to the host (248,320 F32, 0.99 MB) over the last card's
PCIe 1.0 x1 link and samples them there; `llama-bench` asks for one row only.

The line is S34's MTP arm ([2026-11-04_stacked_gains](../2026-11-04_stacked_gains/README.md)): pascal-mix's pmix-q2-fast
on Miner Llama's `-ladder3` with `GGML_CUDA_MMVQ_PASCAL_KCOL_SMEM=32768`, the Gyro MTP draft at
`--spec-draft-n-max 3` (`-otd token_embd=CUDA4 --spec-draft-n-ubatch 128 --no-host`). One load, one short request at
the line's sampling: 501 generated tokens in 34.3 s (14.6 tok/s), 561 drafted and 313 accepted (0.558), so 187 steps of
2.68 tokens each. The fork's host profile (`GGML_CUDA_PROFILE_HOST`) wrote, per card and graph evaluation, the host's
entry to and exit from `graph_compute` and the GPU time between two events around its work. It adds two events per
evaluation and no synchronize, unlike the kernel profiler, which slows the pass.

## Result

Per step, over 186 steps after the first three ([steps.md](steps.md), [run/host_steps.py](run/host_steps.py)):

| Part | Median, ms | p10 | p90 |
|---|---:|---:|---:|
| the step, from one verify pass's first card to the next's | 179.2 | 175.7 | 190.2 |
| the verify pass's GPU time, summed over the nine cards (four tokens) | 113.6 | 111.4 | 119.1 |
| the draft's GPU time (eight evaluations on cards 4 and 8: four draft passes) | 24.8 | 24.6 | 32.3 |
| **neither** (the step minus both) | **~41** | | |

- **About 41 ms of each 179 ms step (23 %) runs on no card's clock.** That is the eight hops between cards, the
  copy of the four verify rows of logits to the host (3.97 MB over PCIe 1.0 x1, est. ~20 ms at the link's ~190 MB/s),
  the draft's own full-vocabulary rows, sampling, the acceptance test and the server's bookkeeping.
- **The host profile cannot time the copy itself:** `graph_compute` returns once the work is queued (each card's host
  span is ~0.5 ms; all nine take 4.1 ms), so the wait for the GPUs, the copy and the sampling land in the same interval
  after the last card (138 ms median). The split between hops and the copy is an estimate from the link's speed and the
  hop bench ([2026-11-04_megakernel_rig](../2026-11-04_megakernel_rig/README.md): a 4-token F32 hop ~1.1–1.3 ms).
- **Does it change anything?** Yes: sampling verify batches on the last card and copying back only the accepted ids and
  the rule's probabilities (C338 step 2) would remove most of the ~20 ms copy, about 11 % of a step. The draft's
  full-vocabulary rows are a second copy of the same kind; the shortlist computes the draft's head over 32,801 of its
  248,320 rows (7.6× fewer), the likeliest reason it paid 10 % in S34, while its output row keeps the full width.

## Files

| File | What it is |
|---|---|
| [steps.md](steps.md) | per step: the step, the verify pass's host span and GPU time, the interval after the last card, the draft's host span and GPU time and the interval before the next verify pass: median, p10, p90, totals |
| [run/host_steps.py](run/host_steps.py) | the split, from a `GGML_CUDA_PROFILE_HOST` file |
| [run/c338_step1.sh](run/c338_step1.sh) | the run: one load and one request with the host profile, then the split |

## Commands

```bash
ENVFILE=s12.env MODEL=<pmix-q2-fast shard 1 on the rig> BUILD=llama.cpp-20261104-7bc1552f4-pascal-p106-v3-b12076-ladder3 \
  OUT=/tmp/c338 bash docs/data/2026-11-04_verify_host/run/c338_step1.sh
```

## Provenance

The host profile is the fork's (`ggml/src/ggml-cuda/ggml-cuda.cu`, `GGML_CUDA_PROFILE_HOST`); the request is S12's smoke
request. The table summarizes the profile; the profile file and the server log are not published.
