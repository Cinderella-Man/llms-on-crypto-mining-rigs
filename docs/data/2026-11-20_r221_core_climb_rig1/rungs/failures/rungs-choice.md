Rungs from profile.json, by `climb.py rungs`, failures from use.json.

| Card | Profile (MT/s, MHz) | Memory cap | Core cap | Rung 1 | Rung 2 | Rung 3 | Core (rung 4) |
|---|---|---|---|---|---|---|---|
| 0 | +900, +269 | +900 MT/s (2 steps under +1,100, soak 2026-11-20, rung 2: 2 differing checks at +1,100 MT/s, +0 MHz) | +269 MHz (2 steps under +320, profile.json: failed at +320 MHz) | +700 | +800 | +900 | +269 |
| 1 | +1,000, +256 | +1,000 MT/s (2 steps under +1,200, soak 2026-11-20, rung 3: 3 differing checks at +1,200 MT/s, +0 MHz) | +256 MHz (2 steps under +307, profile.json: failed at +307 MHz) | +800 | +900 | +1,000 | +256 |
| 2 | +1,100, +256 | +1,100 MT/s (2 steps under +1,300, R217, the memory climb, 2026-11-19: greedy output differed at +1,300 MT/s with the core at 0) | +256 MHz (2 steps under +307, profile.json: failed at +307 MHz) | +900 | +1,000 | +1,100 | +256 |
| 3 | +1,200, +206 | +1,200 MT/s (2 steps under +1,400, R217, the memory climb, 2026-11-19: greedy output differed at +1,400 MT/s with the core at 0) | +206 MHz (2 steps under +256, profile.json: failed at +256 MHz) | +1,000 | +1,100 | +1,200 | +206 |
| 4 | +1,100, +206 | +1,100 MT/s (2 steps under +1,300, soak 2026-11-20, rung 3: 1 differing check at +1,300 MT/s, +0 MHz) | +206 MHz (2 steps under +256, profile.json: failed at +256 MHz) | +900 | +1,000 | +1,100 | +206 |
| 5 | +1,300, +231 | none | +231 MHz (2 steps under +282, profile.json: failed at +282 MHz) | +1,100 | +1,200 | +1,300 | +231 |
| 6 | +1,000, +0 | +1,000 MT/s (2 steps under +1,200, soak 2026-11-20, rung 3: 3 differing checks at +1,200 MT/s, +0 MHz) | +0 MHz (2 steps under +28, profile.json: failed at +28 MHz) | +800 | +900 | +1,000 | +0 |
| 7 | +1,100, +282 | +1,100 MT/s (2 steps under +1,300, R217, the memory climb, 2026-11-19: greedy output differed at +1,300 MT/s with the core at 0) | +282 MHz (2 steps under +332, profile.json: failed at +332 MHz) | +900 | +1,000 | +1,100 | +282 |
| 8 | +900, +231 | +900 MT/s (2 steps under +1,100, R217, the memory climb, 2026-11-19: greedy output differed at +1,100 MT/s with the core at 0) | +231 MHz (2 steps under +282, profile.json: failed at +282 MHz) | +700 | +800 | +900 | +231 |
