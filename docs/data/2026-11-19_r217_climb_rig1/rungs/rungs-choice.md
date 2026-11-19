Rungs from profile.json, by `climb.py rungs`, failures from failures-before.json.

| Card | Profile (MT/s, MHz) | Memory cap | Core cap | Rung 1 | Rung 2 | Rung 3 | Core (rung 4) |
|---|---|---|---|---|---|---|---|
| 0 | +1,200, +0 | +1,200 MT/s (2 steps under +1,400, Q133, 2026-11-17: card 0 alone with the core at 0; the greedy check's probabilities differed on 50 of 512 tokens (kernel checks, text and cuda_memtest clean)) | none | +1,000 | +1,100 | +1,200 | +0 |
| 1 | +1,200, +0 | +1,200 MT/s (2 steps under +1,400, profile.json: failed at +1,400 MT/s) | none | +1,000 | +1,100 | +1,200 | +0 |
| 2 | +1,100, +0 | +1,100 MT/s (2 steps under +1,300, profile.json: failed at +1,300 MT/s) | +244 MHz (2 steps under +300, R193, 2026-11-17: Xid 31 at +1,200 MT/s and +300 MHz (a 2,202 MHz peak), read as a core error: card 2 held +1,200 MT/s with +249 MHz (2,138 MHz) in R194) | +900 | +1,000 | +1,100 | +0 |
| 3 | +1,200, +0 | +1,200 MT/s (2 steps under +1,400, profile.json: failed at +1,400 MT/s) | none | +1,000 | +1,100 | +1,200 | +0 |
| 4 | +1,300, +0 | +1,000 MT/s (2 steps under +1,200, R194, 2026-11-17: two wrong kernel results, then a hang, at +1,200 MT/s and +275 MHz (a 2,138 MHz peak); memory and core not separated) | +218 MHz (2 steps under +275, R194, 2026-11-17: two wrong kernel results, then a hang, at +1,200 MT/s and +275 MHz (a 2,138 MHz peak); memory and core not separated) | +800 | +900 | +1,000 | +0 |
| 5 | +1,300, +0 | none | none | +1,100 | +1,200 | +1,300 | +0 |
| 6 | +1,200, +0 | +1,200 MT/s (2 steps under +1,400, profile.json: failed at +1,400 MT/s) | none | +1,000 | +1,100 | +1,200 | +0 |
| 7 | +1,100, +0 | +1,100 MT/s (2 steps under +1,300, profile.json: failed at +1,300 MT/s) | none | +900 | +1,000 | +1,100 | +0 |
| 8 | +900, +0 | +900 MT/s (2 steps under +1,100, profile.json: failed at +1,100 MT/s) | none | +700 | +800 | +900 | +0 |
