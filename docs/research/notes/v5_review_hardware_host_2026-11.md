# v5 review, hardware and host: clocks, power, links, storage and the 4 GB host (2026-11-16)

One of five reviews for the v5 order, on the conventions of
[the third Flash-Next review](flash_next_review3_2026-11.md): "est." marks my arithmetic, with its basis; every
other number is quoted from the linked folder or [registry](../../../quants/experiments.md) row. Nothing ran on any
machine.

## The short answer

- **No clock offset has ever been set on a card.** The memory offset (Q133) was approved for one card on 2026-10-24
  and never ran, not even its read-only probe.
- **The keep-alive is the clock measurement I have.** Holding 1,873 to 1,911 MHz against 1,354 took a plain
  Flash-Next token from 62.75 and 59.62 ms to 51.06 and 51.08. So est. half of that token follows the core clock,
  and the core offset (Q135) ranks beside the memory offset.
- **No setting replaces the keep-alive.** Application clocks, locked clocks and "prefer maximum performance" do not
  hold the clock (Q141, Q154). What can change is when its block runs: at depth it costs every model prompt speed.
- **Only Flash-Next is short of host memory.** One untested setting has a measured price: the launch queue.
- **A tuned column for three models costs est. 10 to 11 rig hours on one rig** with memory offsets alone; core
  offsets add est. 3 h 25 min to 4 h 45 min. Storage and the links need no rig row.

## What is measured and what is only estimated

**Clocks and power, measured:**
- Each card: 75 W (settable 60 to 75), top clock 1,911 MHz
  ([2026-10-24_rig_power](../../data/2026-10-24_rig_power/README.md)).
- The keep-alive, against cards left at base
  ([2026-11-05_card_clocks](../../data/2026-11-05_card_clocks/README.md)): +72 W over the nine cards; 16.5 and
  16.7 J a token against 15.1 and 15.8; on the agent check 20.75 against 19.19 tok/s
  ([2026-11-05_keepalive](../../data/2026-11-05_keepalive/README.md), Q1021).
- At depth without it ([R167](../../data/2026-11-16_r167_keepalive_depth/README.md)): Flash-Next's prompts 219.4 and
  218.2 tok/s against 212.6 and 215.3, decode 25.72 and 25.86 against 26.15 and 26.11; Ornith's prompts +2.6 %,
  decode −3.4 %.
- Flint with it ([R165](../../data/2026-11-16_r165_flint_stack_split/README.md)): prompts 267.7 and 266.9 against
  321.1 and 319.0, decode 39.06 and 38.94 against 40.55 and 39.86. Its one gain is the 64-token prompt at depth 0:
  96.1 and 105.7 against 85.3 and 73.2.

**Clocks, only estimated:**
- **Memory offset:** +3–5 % decode at +10 % clock, from a v2 Flint token (14.9 of its 26.8 ms stream weights;
  Q133).
- **Core share (est.):** the keep-alive's clock ratio (1.38 to 1.41) and time ratio (0.835, on the means) put
  est. half of a plain Flash-Next token at boost on the core clock. So +5 % core clock is est. +2.4 % plain decode
  and est. +1.1 % with the draft (the agent check's token fell 7.5 %, the plain one est. 16.5 %). Flint: est.
  +1.7 %, from 38.5 against 34.0 tok/s at boost and base (Q154). The same half caps a +10 % memory offset on
  Flash-Next at est. +4.5 %.
- **The protocol** ([bench/mem_offset](../../../bench/mem_offset/README.md), [hardware tuning](hardware_tuning.md)
  §5.2) never ran: one card at a time; +200 MT/s steps with a correctness sweep and an Xid scan; then a pattern
  test, the exactness check (a one-token KLD run equals a stock rerun: KLD 0 when stable) and an hour's soak under
  70 °C; one step back. Offset 0 undoes it; a hung card needs a reboot at the console.

**The host, measured:**
- `kho=off` is on both rigs: 3,209 and 3,213 MiB available at idle
  ([2026-11-04_rig_host_ram](../../data/2026-11-04_rig_host_ram/README.md), Q692).
- Lowest free memory in the v4 agent runs: Flash-Next 312 and 419 MiB, Flint 1,084 and 1,100, Ornith 1,225 and
  1,228 ([flash-next-v4](../../../release/flash-next-v4/README.md), [flint-v4](../../../release/flint-v4/README.md),
  [ornith-v4](../../../release/ornith-v4/README.md)). At depth Flash-Next has 247 and 268 (R167).
- Flint keeps `-b 512 -ub 128`: a larger batch pins about 1 GiB at 262,144 tokens (its page). `-b 4096` adds no
  host memory on Flash-Next ([R123b](../../data/2026-11-12_r123b_prompt_ladder/README.md)). Checkpoints on the cards
  leave Flint 1,165 and 1,144 MiB against 949 and 932 (R165).
- The launch queue at `2x` costs 327 MiB, 36 MiB a card (R123b, Q1239). Est.: `0.5x` returns 164 (Q693).
- Swap is erratic: one preset went from 36 to 404 MiB on one rig and stayed at 21 on the other
  ([R156](../../data/2026-11-13_r156_agent_v4r/README.md)); the one swapping load of
  [R138](../../data/2026-11-12_r138_swap_cause/README.md) started 17 s after an unpack. Its cost in speed is not
  measured (Q1431).
- The [watchdog](../../data/2026-11-04_ram_watchdog/README.md) did not act in R156, R165 or R167.

**Storage, measured:** the n-gram table needs flash: 6,039 random reads a second on the SSD, 95 on the USB hard
disk ([2026-11-03_table_storage](../../data/2026-11-03_table_storage/README.md)). A generated token reads 5.70
pages ([2026-11-03_ngram_replay](../../data/2026-11-03_ngram_replay/README.md)): est. at most 0.87 ms at the SSD's
152 µs. Flash-Next's server is up after 226 s (R138). No v5 rig row: the one open read (Q728) rides on any load.

**Links, measured:** one Q8_0 hop takes 126.7 µs ([interconnect note](diy_card_interconnect.md)). Staged hops give
Flint 40.99 and 41.03 tok/s against 40.42 and 40.40; its 8,192-token chunk reads at 446.1 and 437.4 against 478.2
and 479.2 (R165).

## Candidates for v5

| # | Row | What | Evidence | Worth | Cost | Needs | Verdict |
|---|---|---|---|---|---|---|---|
| 1 | Q1464, widened | every driver logs each card's clock, power and throttle reasons | R165 and R167 hold no such log | prices rows 4–6 | small; 0 h | — | first |
| 2 | Q133 | the read-only probe; the memory offset on card 0; then card by card | est., a v2 Flint token | decode: Flint est. +3–5 %, Flash-Next est. ≤ +4.5 %; prompts est. 0–3 % | none; 1 min, est. 1 h 50 min, then est. 20–30 min a card | the owner at the console | first |
| 3 | Q135 | the core offset, same sitting | est., the keep-alive's data | decode: Flash-Next est. +1.1 % served, +2.4 % plain; Flint est. +1.7 %; prompts est. +3–5 % | none; est. +45 min on card 0 | the owner at the console | first |
| 4 | NEW (with Q1461, R167c) | the block paused during long prompt calls; wake-only for Flint | measured: R167, R165 | prompts at depth: Flash-Next est. +2.3 %, Ornith est. +2.6 % | small; est. 1 h 15 min to 2 h 25 min a model | — | next |
| 5 | Q1065 | the pulse keep-alive in the engine | bench: 45.133, 45.129, 45.128 ms a pass against 45.879, 45.878, 45.881 | est. 1.6 % of a one-token pass; Ornith est. +2.4 % prompts at depth | small; est. 1 h | — | next |
| 6 | Q1045 | `idle`: cards held out of P8 between turns | measured: 64-token prompts −9.0 % (Flash-Next), −17.8 % (Ornith) without the block | first token est. up to 1 s sooner | none; est. 40 min an arm | the owner: est. +20 W a card while tools run | next |
| 7 | NEW | the tuned column: validated offsets on the released presets; same-day stock and tuned agent runs, three models | — | the README's promise | none; est. 4 h (six arms of est. 40 min) | rows 2, 3; the owner: one rig or two | next |
| 8 | Q693 | the launch queue at `0.5x`, `0.25x` on Flash-Next | est., Q1239's 36 MiB a card | est. +164 or +245 MiB on a 312 MiB floor | none; est. 1 h | — | next |
| 9 | Q1431 | does swap cost agent speed | measured: erratic swap (R156, R167) | unknown | none; est. 40 min | the owner: `vm.swappiness` | next |
| 10 | NEW | staged hops on decode graphs only (Flint) | measured: R165 | Flint decode est. +1.5 % | small; est. 1 h 40 min | — | later |
| 11 | Q991, Q992, Q994 | peer verdict (read-only); forced peer copies; Gen2 links | est. (interconnect note) | decode est. −0.57 ms a token; prompts est. +4–6 % | none; minutes, then est. 1 h each | the owner present: module reload, register writes | Q991 next; rest later |

## Decisions that are the owner's

1. **Offsets on card 0.** May the kit sweep the memory clock, then the core clock, on card 0 of one rig for est.
   2 h 35 min, you at its console? Est. +3–5 % decode against a possible hung card (a reboot and the passphrase).
   The 2026-10-24 approval covered that day.
2. **The other eight cards.** Est. 20–30 min each and offset on later days; with a nine-card check and an hour's
   soak, est. 6 h to 7 h 20 min a rig for the memory offset. Is one rig's column enough for v5?
3. **Persistence.** After 24 clean hours, may a boot-time unit re-apply each card's offset, a standing change to
   the rig?
4. **Idle power.** May a preset hold the cards awake between turns: est. +20 W a card while tools run, for a first
   token est. up to 1 s sooner?
5. **`vm.swappiness`.** May one agent arm (est. 40 min, one rig) lower it, restored afterwards?
6. **Peer copies and Gen2.** One rig, you present, est. 1 h each: est. 1 % of decode and est. +4–6 % prompts
   against a card off the bus or wrong copies. I would wait for rows 2 and 3.

## Rows to close, merge or re-rank

- Q133: keep, first. Its approval is stale; its estimate rests on a v2 token.
- Q135: up beside Q133. Its gate (power data from busy cards) is met: 34.8 to 39.8 W a card at boost (Q1021).
- Q134: close: the setup turns automatic upgrades off.
- Q83, Q692, Q511, Q1021: done; close.
- Q86: stays parked; its overclock half duplicates Q133.
- Q1075: close; Q1417 and Q1426 fixed it.
- Q1045: reopen `idle`; est. the eager stop removes the wait that made it a loss.
- Q1464 (C832): row 1. Q1461 (C830) and R167c: merge into row 4.
- Q1431 (C819): widen to the deep lane, where R167's loads swapped.
- Hardware hacks: C508 is a purchase: park. C503 waits on C502.
- Parked list: C510, P15, P20 need other hardware: close. C513, C515 to C517, C519 to C521 reopen on "nothing":
  close.
- Housekeeping: no hardware row.
- Q287, Q694, Q695, Q571 (host settings): later; est. 0–2 % of served tokens, est. up to 56, 150 and 100 MiB.
  Q694 is a boot parameter, the owner's.
- Rigs 3–5: C49 (`rig_setup.sh`) before the third rig, with the offset probe added; C48 waits on the rigs.

## New ideas

- **The block paused during long prompt calls,** the request's wake kept: long prompts are faster without it,
  short ones slower (R167).
- **A wake-only keep-alive for Flint:** with the block its short prompt gains 27.3 %, its decode loses 3.0 % (R165).
- **Staged hops on decode graphs only:** Flint's decode gain without its prompt loss.
- **The core offset as Flint's keep-alive:** est. it lifts the 1,354 MHz base clock too, since the driver applies
  an offset to every performance level.
- **J a token beside tok/s in v5's table,** from row 1's log.
- **A tuned preset file:** each card's offsets, the exactness check, a reset on exit.

## What I could not settle from the files

- Why the keep-alive costs Flint its deep prompts: no run logged clocks, power or temperature at depth.
- Whether Flash-Next's 9.9 % decode loss at 49,152 tokens without the keep-alive is the clock or the swap: those
  loads moved 819 and 726 MiB to swap.
- Whether the driver accepts offsets on a P106-090, their range, and whether a core offset lifts the base clock.
- Any v4 token's split into core-bound, memory-bound and fixed time; the cards' temperatures with the keep-alive.
- Whether the release runs ran headless, and whether the NVIDIA packages are held.
- What part of the 226 s load is the disk.
