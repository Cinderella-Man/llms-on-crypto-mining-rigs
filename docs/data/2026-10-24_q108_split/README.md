# Q108 after Q117: does decode lose speed, and do the cards hit the power cap? (2026-10-24)

Q117 measured the combined v2 build decoding 1–3 tok/s slower than the v2 base, and turning Q108 (upstream PR #31047,
MoE prefill tiles sized by tokens per expert) off brought decode back. Two questions followed: does any PR #31047 code
run in a one-token step, and do faster prompts leave the cards hotter or more power-capped for the decode that follows?
The first is answered from the code, the second from every nvidia-smi log of the rig with clock event reasons (the Q38
runs and the rig step-1 runs of 2026-10-23: llama-bench pp512 then tg128 in one process, four to nine cards). Results:
registry rows Q108 and Q117 in [quants/experiments.md](../../../quants/experiments.md),
[2026-10-24_rig_q117](../2026-10-24_rig_q117/README.md) (the tg-only check) and
[power_and_tuning.md](../../research/notes/power_and_tuning.md).

| File | What |
|---|---|
| `phases.txt` | [scripts/smi_phases.py](../../../scripts/smi_phases.py) on each of the 24 logs: the cards in use, then per busy phase (consecutive seconds with the cards' mean utilization ≥ 10 %, at least 2 s) its start, length, utilization, mean and lowest SM clock, mean power, hottest card and the union of clock event reasons over its samples |
| `cap_share.csv` | per log, the power cap (reason 0x4) counted two ways: as a share of the samples of the cards in use (all samples, and busy ones with utilization ≥ 10 %), and as the compute phases (mean SM clock ≥ 1,300 MHz, the loading phases at the 139 MHz idle clock left out) of at least 10 s in which one sample of one card has it; plus the share of capped samples inside the compute phases, their mean SM clock and power, and any slowdown bit seen (none) |
| `run/cap_share.py` | the script that made `cap_share.csv` |

## What it shows

- **No PR #31047 code runs in a one-token step.** `ggml_cuda_mul_mat_id` sends a quantized `MUL_MAT_ID` with at most
  `get_mmvq_mmid_max_batch(type, cc)` tokens (4–6 on compute 6.1, 8 elsewhere) to the matrix-vector path before it
  considers MMQ, and the fused paths call it directly. The PR changes only the MMQ, MMF and ids-helper code they share;
  `GGML_CUDA_MMQ_MOE_LEGACY` is read only inside `ggml_cuda_mul_mat_q`. llama-bench gives every test a fresh context,
  so nothing of the prompt tests' allocations carries into tg128. A split (the PR's tiles for prompt batches only) was
  written and parked once the tg-only check found no gap: Q117's decode difference was an order effect.
- **No thermal limit.** No slowdown bit (0x8, 0x20, 0x40, 0x80) in any sample; the hottest card in a busy phase read
  40 °C.
- **The power cap flickers in nearly every busy phase but in few samples.** Of the 25 compute phases of 10 s or more,
  19 have the cap in at least one sample; in the four- to six-card runs 19 of 21, in the nine-card runs (Flint Q8_0,
  ~36 W per card) none of 4. Inside those phases the cap is set in 0–3.6 % of the card samples, and over whole logs in
  0.02–0.34 % of the samples of the cards in use (0.1–1.8 % of the busy ones). Compute phases average 33.8–38.3 W per
  card at 1,673–1,872 MHz.
- **Decode clocks vary between identical runs.** The decode phases that follow a prompt run at 1,780–1,868 MHz mean SM
  clock; identical runs differ by about 5 % (step-1 runs A/B 1,780–1,784, C/D 1,868), the size of Q117's decode gap.
  A power-capped boost that varies from run to run, not heat, is the likelier cause of the 1–3 tok/s tg128 spread.

The two counts agree with [power_and_tuning.md](../../research/notes/power_and_tuning.md)'s 1 % of samples, which counts
all 266,202 card samples of the same and earlier logs, idle ones included: a busy card touches the 75 W limit briefly
(single readings above 75 W) and the cap engages for a sample or two, often enough to show in almost every phase of a
four- to six-card run, too rarely to move the phase's mean clock.

## How it was made

On 2026-10-24 (the phase table) and 2026-10-27 (the cap counts), from the nvidia-smi logs that
[scripts/bench.sh](../../../scripts/bench.sh) wrote beside llama-bench in the Q38 and step-1 runs, every card about every
0.54 s; the logs themselves stay out:

```bash
nvidia-smi --query-gpu=timestamp,index,pstate,clocks.sm,clocks.mem,power.draw,temperature.gpu,utilization.gpu,memory.used,clocks_event_reasons.active \
    --format=csv -lms 500 > smi.csv &                      # scripts/bench.sh, during each run
for f in <logs>/*/smi.csv; do python3 -I scripts/smi_phases.py $f; echo; done > phases.txt
python3 -I docs/data/2026-10-24_q108_split/run/cap_share.py <logs>/*/smi.csv > cap_share.csv
```
