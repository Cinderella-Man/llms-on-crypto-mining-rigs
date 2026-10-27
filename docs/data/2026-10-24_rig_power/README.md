# The nine cards' power and clock limits at idle (2026-10-24)

One `nvidia-smi` query of all nine P106-090 cards with nothing running, taken before the power and tuning work of
[power_and_tuning.md](../../research/notes/power_and_tuning.md), which uses it for the power limits (75 W default and
enforced, 60–75 W settable), the top SM clock (1,911 MHz) and the idle state.

| File | What |
|---|---|
| `power_summary_idle.csv` | one row per card: power limit, default, minimum, maximum and enforced limit (W), power draw (W), current and maximum SM clock, current memory clock (MHz), temperature (°C), fan speed, P-state, active clock-event reasons |

What it shows: every card has a 75 W default and enforced limit, settable from 60 to 75 W, and a 1,911 MHz top SM
clock. Seven cards idled in P8 at 139 MHz SM and 405 MHz memory, 5.4–6.5 W; cards 4 and 5 sat in P0 at 1,493 MHz SM
and 4,006 MHz memory, ~27 W, at the moment of the query. Every card reported only the idle reason (`0x1`).
Temperatures were 22–30 °C; the cards have no fans of their own (0 %).

Made on 2026-10-24 at 12:22 with driver 580.178.04, on the rig, from a machine that reaches it over ssh:

```bash
RIG=${RIG:-rig}
ssh "$RIG" 'nvidia-smi --query-gpu=index,power.limit,power.default_limit,power.min_limit,power.max_limit,enforced.power.limit,power.draw,clocks.current.sm,clocks.max.sm,clocks.current.memory,temperature.gpu,fan.speed,pstate,clocks_event_reasons.active --format=csv' > power_summary_idle.csv
# the same counters in full, for reading (not kept: raw output)
ssh "$RIG" 'nvidia-smi -q -d PERFORMANCE,TEMPERATURE,POWER,CLOCK'
```

The cumulative "SW Power Capping" counter of the full query is not a usable signal on these cards (see the note,
section 1.5). Measured data of this repository (MIT).
