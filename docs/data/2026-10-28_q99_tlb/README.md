# Q99: do GPU page walks slow the Pascal kernels inside the model? (2026-10-28)

Inside Flint's model the ported Pascal kernels run 13–33 µs slower per call than in `bench/mmvq_pascal`, and a card
sweeps ~4.7–5.4 GB of weights per token while the benchmark's working sets stayed under ~1.2 GB. The hypothesis
(registry Q99, [Pascal forks and quants](../../research/notes/pascal_forks_and_quants.md) section 3): the GPU's
second-level TLB covers ~2 GiB, so a call pays ~1.3–2.0 µs per 2 MiB page beyond that, est. 2.7–4.1 ms per token and
+5–12 % if fixed. Kill criterion set before the run: MoE calls at 5 GB less than 10 µs slower than at 0.5 GB.

## Result: no page-walk cost; Q99 is closed

- **The pointer chase shows no TLB knee.** A dependent load through slots 2 MiB apart costs 175–184 ns per step from a
  16 MiB to a 5 GiB footprint (237–249 cycles, an L2 hit), with the four other SMs idle; under a streaming load
  336–341 ns, again flat across footprints (`tlb.csv`). The rise from 2.5 GiB on is ~4 ns (2 %).
- **The spread sweep shows no cost either.** With every matrix's weights spread over 4.6–5.0 GB and fresh expert ids for
  every call, the MoE calls take −0.7 to +1.7 µs against their 0.3–0.6 GB runs (gate/up 53–88 µs, down 28–49 µs per call;
  `spread_compare.tsv`), far under the 10 µs kill line. The attention products move by −1.2 to +0.8 µs, the output head
  (1.2–2.2 ms) by −27 to +25 µs (≤ 1.5 %, noise at that size), and the whole token's modelled time by 0.1–1.2 %.
- **What it means:** the in-model excess of 13–33 µs per call comes from something else (candidates: card clock state
  between turns, the cache state left by the other operations of a layer, launch scheduling inside the CUDA graph).
  Q99's est. +5–12 % is off the table.

## Files

| File | What |
|---|---|
| `tlb.csv` | `--tlb-probe`: strided and random-page chases, 16 MiB to 5 GiB, alone and under a 4-SM streaming load |
| `spread0.5.csv` … `spread5.csv` | the matrix-vector sweep with weights spread over 0.5–5 GB (Q5_0, Q8_0, Q4_1; three kernel variants; Flint's MoE, attention and output shapes); every variant checked against the CPU reference |
| `spread_compare.tsv` | per shape, type and variant: median µs per call at 0.5 GB and at 5 GB, their difference, the bytes swept |
| `run/run.sh` | the run, from a machine that reaches the rig (`RIG`) with a `bench/mmvq_pascal` build (`BIN`) |

## Made with

`bench/mmvq_pascal` built with its `build.sh` (CUDA 12.9 container, `sm_61` code, static runtime) from this repository
at `e2d0df8`, run on one P106-090 (card 0) of the nine-card rig with nothing else on the rig:

```bash
RIG=rig BIN=bench/mmvq_pascal/build/mmvq_pascal bash run/run.sh
```

The sweep took ~3 minutes in all.

## Provenance and licence

Measurements on this project's rig; the tables and script are under this repository's licence.
