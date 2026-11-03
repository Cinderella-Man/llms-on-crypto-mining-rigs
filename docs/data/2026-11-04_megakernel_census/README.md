# Launch census of a Flash-Next token on nine P106s, for one persistent kernel per card (2026-11-04)

What a persistent kernel per card (a "megakernel", registry row Q670) could save on Qwen3.8-Flash-Next (Swift-1.5
GSQ-RCO Q2_0) on nine P106-090 cards, bounded from the kernel traces already recorded, with no new rig run: per decoded
token and per card, the GPU time against the wall time, the launch groups and est. kernels and their sizes by op, how
much of the ~33 ms of "operations without weights" in R13's profile is fixed per-launch cost, and how the fixed part
behaves over k-token passes. Result page: the design note [megakernel.md](../../research/notes/megakernel.md) and
registry row Q670 in [quants/experiments.md](../../../quants/experiments.md). The microbenchmark that measures the two
costs the traces cannot give, [bench/megakernel_p106](../../../bench/megakernel_p106/README.md), is built; its rig
command is below.

## Result

Per decoded token, nine cards, medians over the trace's tokens:

| | R13 (v3, Swift Q2_0) | R28 (v3, hyper-connections in Q8_0) |
|---|---:|---:|
| Decoded tokens in the trace (wall measurable) | 94 (89) | 94 (89) |
| Launch groups per token | 2,520 | 2,520 |
| Est. kernels per token (a quantized product adds its activation quantization) | 2,898 | 3,188 |
| Groups without weights; their profiled ms | 1,658; 30.9 | 1,658; 31.0 |
| of which near-zero work (elementwise, norms, routing, hyper-connection mixing) | 1,265 groups, 14.5 ms | 1,265, 14.8 ms |
| of which moving a state, a cache or attention (GDN, flash attention, `GET_ROWS`, `CONT`, `CONCAT`, `SSM_CONV`, `CPY`, `SET_ROWS`) | 393 groups, 16.4 ms | 393, 16.2 ms |
| GPU ms per token (profiled) | 129.1 | 113.3 |
| Hops between cards / host before card 0 / wall (profiled, ms) | 15.6 / 11.5 / 155.4 | 14.3 / 11.1 / 139.8 |
| GPU busy, share of the profiled wall | 83 % | 81 % |
| Groups by duration < 10 / 10–20 / 20–50 / ≥ 50 µs | 619 / 1,075 / 215 / 612 | 626 / 1,058 / 319 / 516 |

The cost of a small kernel without the profiler, from Q88's rig bench on a P106 ([2026-10-24_rig_q88](../2026-10-24_rig_q88/README.md)):
an empty CUDA-graph node 1.84 µs; the small kernels of a MoE layer inside a graph 2.12–2.40 µs (median 2.31). The
bound from these, per token ([r13/bound.csv](r13/bound.csv), [r28/bound.csv](r28/bound.csv)):

| | R13 | R28 |
|---|---:|---:|
| Profiled µs per near-zero group (mean; the elementwise groups' median is 9.2–10.2) | 11.5 | 11.7 |
| The profiler's own cost per group (est.: profiled near-zero minus a graph kernel) | 6.9–9.2 µs | 6.9–9.4 µs |
| The profiler's cost over all groups (est.) | 17–23 ms | 17–24 ms |
| The near-zero groups without the profiler (est.) | 2.9 ms | 2.9 ms |
| **Ceiling: every est. kernel's launch removed, barriers free** | **6.7 ms** | **7.4 ms** |
| Net with a 0.5 / 1 / 1.5 / 2 µs barrier on every stage | 5.2 / 3.8 / 2.4 / 0.9 ms | 5.8 / 4.2 / 2.6 / 1.0 ms |
| Net with those barriers on 70 % of the stages | 5.7 / 4.7 / 3.7 / 2.6 ms | 6.3 / 5.1 / 4.0 / 2.9 ms |

k-token passes (Q583's llama-bench trace, [s21/kpass.csv](s21/kpass.csv), medians of five passes):

| k | Launch groups per pass | without weights | GPU ms per pass | without weights, ms | their median µs |
|---:|---:|---:|---:|---:|---:|
| 1 | 2,590 | 1,729 | 139.6 | 33.7 | 10.2 |
| 2 | 2,642 | 1,773 | 182.0 | 39.0 | 11.2 |
| 3 | 2,642 | 1,773 | 234.1 | 40.6 | 11.3 |
| 4 | 2,642 | 1,773 | 286.9 | 41.7 | 11.3 |
| 5 | 2,642 | 1,773 | 323.0 | 42.8 | 11.3 |

A least-squares line through k = 1–5: the groups without weights take 33.8 ms per pass + 2.1 ms per token (k = 2–5:
37.4 + 1.2), the whole pass 92.3 + 47.2 ms per token ([s21/kpass_fit.csv](s21/kpass_fit.csv)).

What it shows:
- **A per-card persistent kernel can save at most ~7 ms of a ~121 ms token (~6 %)**, and a grid barrier on every
  stage takes most of it back: a P100's `grid.sync()` costs 1.8–2.3 µs at one block per SM, about a P106 graph node.
- **Most of R13's "operations without weights" is not launch cost.** The event profiler adds est. 7–9 µs to every
  launch group (17–23 ms per token); the groups that move a state, a cache or attention are 16 ms profiled (what
  Q436–Q438 fuse away); the near-zero groups cost ~3 ms without the profiler.
- **The fixed part is per pass.** The weightless time barely grows with k (33.7 → 42.8 ms from 1 to 5 tokens), so with
  a drafter accepting ~3 tokens per step the per-token share of any launch saving falls by ~3×.
- **The profiled hops (1.7–2.2 ms per card boundary, [r13/cards.csv](r13/cards.csv)) are larger than the residual's
  bytes need** (40 KB at the rig's ~200 MB/s, ≥ 0.4 ms), but they include the profiler's synchronize and CSV write;
  the bench's `--hop` mode measures an unprofiled hop.

What the traces leave unknown: `GGML_CUDA_PROFILE` records one duration per launch group, the time between the events
before and after it, so a launch gap and a kernel's run are one number, and the events' own cost is inside every
number. Start and end times per kernel need `%globaltimer` stamps written by the kernels, a CUPTI activity trace (the
CMP cards refuse NVIDIA's profilers), or the event cost measured on the same card and subtracted: the bench's
`graph+events` against `graph`. Whether decode replays captured CUDA graphs or runs eagerly is not in the trace either
(registry Q680).

## Files

| File | What |
|---|---|
| `r13/`, `r28/` | census of R13's and R28's traces: `summary.md` (the headline lines), `tokens.csv` (per token: launch groups, GPU ms, hops ms, host ms before card 0, wall ms; profiled), `cards.csv` (per card: groups, groups without weights, est. kernels, groups under 10 µs, GPU and weightless ms per token, median hop), `ops.csv` (per op class or weight role: groups and est. kernels per token, ms per token, duration p10/median/p90, groups and ms per size bucket), `labels.csv` (every launch-group label, layer numbers folded: groups and ms per token, min and median µs), `fixed.csv` (the time below an assumed fixed cost of 2–8 µs per group), `bound.csv` (the bound above) |
| `s21/kpass.csv`, `s21/kpass_fit.csv` | the k-token passes of Q583's llama-bench trace: per k the median pass, and the line through them |
| `run/census.py` | the census of one `GGML_CUDA_PROFILE` trace |
| `run/kpass.py` | the census of a llama-bench trace of k-token passes |
| `run/bound.py` | the bound, from a census, Q88's `q88_s4l0.csv` and `kpass_fit.csv` |

## How it was made

On the quant machine (`nice -n 19`, Python standard library, no GPU), from the raw traces of R13
([2026-11-03_flash_next_baseline](../2026-11-03_flash_next_baseline/README.md): `GGML_CUDA_PROFILE` of the diagnostic
v3 line, 17.7 MB), R28 ([2026-11-03_retype_agent](../2026-11-03_retype_agent/README.md), its profiled line, 17.7 MB)
and Q583 ([2026-11-03_verify_cost](../2026-11-03_verify_cost/README.md): `prof-bench.csv` and `bench-prof.json`), which
are not published (raw output), and Q88's public `q88_s4l0.csv`:

```bash
# WORK holds raw/ (the traces) and run/ (this folder's scripts)
cd "$WORK"
mkdir -p out/r13 out/r28
for t in r13 r28; do
    python3 -I run/census.py raw/$t-prof.csv $t out/$t > out/$t/summary.md
done
python3 -I run/kpass.py raw/s21-prof-bench.csv raw/s21-bench-prof.json out/s21
for t in r13 r28; do
    python3 -I run/bound.py out/$t raw/q88_s4l0.csv out/s21/kpass_fit.csv
done
```

## Next: the microbenchmark on a rig

[bench/megakernel_p106](../../../bench/megakernel_p106/README.md), built on 2026-11-04 with CUDA 12.9 for sm_61
(`bench/megakernel_p106/build.sh`; binary md5 in its `build/BUILDINFO.txt`). On a rig with nothing else on the cards,
~10 minutes, card 0 for the stages and cards 0 and 1 for the hop:

```bash
RIG=${RIG:-rig}
ssh "$RIG" 'mkdir -p /tmp/mk'
scp bench/megakernel_p106/build/megakernel_p106 bench/megakernel_p106/run_rig.sh "$RIG":/tmp/mk/
ssh "$RIG" 'cd /tmp/mk && setsid nohup ./run_rig.sh > console.log 2>&1 < /dev/null &'
ssh "$RIG" 'test -e /tmp/mk/out/DONE && cat /tmp/mk/out/run.log'      # until DONE exists
scp -r "$RIG":/tmp/mk/out ./megakernel_rig_out
ssh "$RIG" 'rm -rf /tmp/mk'
```

What decides stage 1 (Q670): on the `census6` mix, the persistent variants (`barrier`, `barrier-graph`,
`barrier-local`, `coop`) at ≤ 50 % of `graph`'s µs per stage, a saving of ≥ ~1.2 µs per stage (≥ ~3.5 ms per token
over ~2,900 kernels). `graph+events` against `graph` gives the profiler's cost per group (Q681); `hop.csv` the
unprofiled hop and what a mailbox saves (Q682).

## Licence

The tables and scripts are this repository's (MIT). They summarize kernel timings of the public Qwen3.8-Flash-Next
model files; no model data is in them.
