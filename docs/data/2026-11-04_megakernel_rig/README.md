# C140: the megakernel microbenchmark and the card-to-card hop on a P106 (2026-11-04)

Q670 stage 0 (does one persistent kernel per card pay?), Q681's calibration (what one `GGML_CUDA_PROFILE` event costs)
and Q682's hop (how long the residual takes between two cards without a profiler). `bench/megakernel_p106` on the
first rig, idle, ~8 minutes after a `kho=off` boot; no model loaded. The census and the design are in
[2026-11-04_megakernel_census](../2026-11-04_megakernel_census/README.md) and the
[megakernel note](../../research/notes/megakernel.md); the hop switches in [2026-11-04_card_hops](../2026-11-04_card_hops/README.md).

## Outcome

- **The persistent-kernel track stays stopped.** On `census6` (one recurrent layer's weightless chain: 228 stages of
  elementwise, norm and small matrix-vector work) the best persistent variant takes **72 % of a CUDA graph's time per
  stage** (1.98 against 2.74 µs; the gate to reopen stage 1 was ≤ 50 %). With a barrier between every stage it is
  89–91 %. At ~2,900–3,200 kernels a token that is at most ~2.3 ms of a ~121 ms token (~2 %), below the census's
  ~7 ms bound.
- **The profiler's event costs ~7.6–8.5 µs a group** (`graph+events` − `graph`: 8.2 µs on empty stages, 7.6 on
  `census6`, 8.5 on `census1`, 6.6 on the 48-row matrix-vector stage), inside the census's 7–9 µs estimate. A profiled
  elementwise group records ~9.2 µs where it costs ~1.7–2.7 µs in a graph: the light profiler (C142) subtracts this.
- **One token's hop between cards is 0.455 ms unprofiled** (40,960 B of F32 residual, the driver's peer copy, the
  path every build takes today): ~3.6 ms a token over the eight boundaries, against 1.7–2.2 ms a hop under the
  profiler (R13). It is bandwidth-bound, not latency-bound: ~26 µs fixed plus ~10 µs per KB (both legs of PCIe 1.0 x1
  in series). **A mailbox does not help** (436 µs, the same as a host-staged copy); fewer bytes and overlapped legs
  do: the fork's staged hop in 4 chunks takes 317 µs (−30 %), and **one token in Q8_0 (10,880 B) through the staged
  hop in 2 chunks takes 127 µs (−72 %): ~1.0 ms a token instead of ~3.6.** Every card pair measures the same within
  3 % (two cards behind one switch, a switch to the card on its own port and back, two switches).

## The stage timings (card 0, µs per stage; `timing.txt`, `timing.csv`)

| Chain | stream (eager) | stream + events | graph | graph + events | best persistent (variant, grid) | persistent / graph |
|---|---:|---:|---:|---:|---|---:|
| empty × 256 | 6.81 | 12.25 | 1.70 | 9.92 | 1.05 (coop, 5) | 0.62 |
| ew2560 × 256 | 6.81 | 12.25 | 1.71 | 9.92 | 0.42 (barrier-local, 10) | 0.24 |
| ew10240 × 256 | 6.81 | 12.26 | 1.72 | 9.93 | 0.72 (barrier-local, 20) | 0.42 |
| norm2560 × 256 | 6.83 | 12.29 | 3.18 | 10.23 | 3.31 (coop-graph, 5) | 1.04 |
| mv10240x4 × 64 | 6.85 | 12.44 | 3.71 | 10.52 | 3.59 (barrier-graph, 5) | 0.97 |
| mv2560x48 × 64 | 6.89 | 12.49 | 4.75 | 11.31 | 6.03 (barrier-local, 20) | 1.27 |
| census1 (38 stages) | 6.82 | 12.46 | 1.76 | 10.24 | 1.83 (barrier-local, 20) | 1.04 |
| **census6 (228 stages)** | 6.83 | 12.38 | **2.74** | 10.39 | **1.98 (barrier-local, 20)**; 2.44 (barrier, 10) | **0.72**; 0.89 |

- Empty kernel: 1.72 µs a graph node, 6.43 µs an eager launch; a grid barrier on 5–20 resident blocks costs
  1.05–1.32 µs (cooperative `grid.sync()` and the hand-written atomic barrier within ~0.1 µs of each other).
  Streaming read 178 GB/s.
- Persistent kernels win only where a stage is mostly launch cost and needs no barrier (`barrier-local`: same-size
  elementwise stages, each thread reading only what it wrote). Norms and matrix-vector stages need the whole grid
  and lose to the graph. A real layer mixes both, so the saving is a quarter of a stage at best.
- `checks: all passed` (`check.txt`): every persistent variant and grid bit-identical to the stream launches; the stream
  path against the CPU within 8.8e-07. `smoke.txt`: every timing path ran.

## The hop (cards 0 → 1, µs per hop, median of 300; `hops/hop-*.txt`, `hops/hop-*.csv`)

| Bytes | What | host-staged | peer copy (today) | mailbox | staged, 1 chunk | staged, 2 chunks | staged, 4 chunks |
|---:|---|---:|---:|---:|---:|---:|---:|
| 10,240 | | 128.4 | 143.4 | 137.2 | 135.9 | 118.4 | 122.5 |
| 10,880 | one token in Q8_0 | 134.8 | 141.4 | 143.4 | 144.0 | **126.7** | 130.2 |
| 20,480 | one token in F16 | 232.5 | 237.6 | 237.0 | 242.5 | 200.7 | 186.3 |
| 40,960 | **one token in F32 (today)** | 435.7 | **455.1** | 436.4 | 447.2 | 356.2 | 316.9 |
| 43,520 | four tokens in Q8_0 | 461.2 | 478.9 | 461.7 | 471.9 | 377.7 | 336.9 |
| 81,920 | four tokens in F16 | 844.3 | 864.7 | 835.4 | 857.5 | 660.5 | 583.0 |
| 163,840 | four tokens in F32 | 1,662.5 | 1,684.6 | 1,631.3 | 1,674.4 | 1,291.1 | 1,113.2 |

The other pairs (3 → 4, 4 → 5, 3 → 5) are within 3 % of these (3 → 4 slightly faster). "manual hop data: every byte
arrived (7 sizes x 3 chunk counts)" and 0 mailbox timeouts on every pair. Peer access is not possible between these
cards, so the driver's "peer copy" stages through the host too.

**Per token on nine cards (eight boundaries):** F32 peer copy 3.64 ms; F32 staged in 4 chunks 2.54 ms; F16 staged in 4
chunks 1.49 ms; **Q8_0 staged in 2 chunks 1.01 ms**. The staged hop's default is 2 chunks (`GGML_CUDA_HOP_CHUNKS`); 4
chunks win from ~20 KB up (by 7–14 %), so the F32 path and prompt-sized hops want 4 and the Q8_0 decode hop 2 (Q820).
The cast's quality is C200k's KLD; the agent-check arm is C200r.

## Run

Card 0 for the stages, the pairs above for the hop, nothing else on the rig. `run/c140.sh` runs C140's bundle
(`bench/megakernel_p106` built from `1b344d6`, binary md5 `b041450c76a43245f27c0930467d1236`, `run_rig.sh` with
`HOP=` empty) and then the card-hops bundle's hop part (the same bench with `--hop-chunks` and the staged hop, md5
`f47c545dd6b0f9f9d6fe194e16877ae7`; `rig_hops.sh` with `LINES=0`, which runs
`megakernel_p106 --hop A,B --hop-only --csv hop-A-B.csv` for `HOPS="0,1 3,4 4,5 3,5"`). Both bundles' `MD5SUMS`
checked on the rig. Driver 580.178.04, the cards idle at P8 before the run.

```bash
# from a folder W holding mk/ (C140's bundle) and hops/ (the card-hops bundle), on the rig:
W=<work folder> setsid nohup bash run/c140.sh > c140.console 2>&1 < /dev/null &   # ~1.5 min; ends with $W/ALLDONE
```
