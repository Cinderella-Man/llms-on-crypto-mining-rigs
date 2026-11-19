# FN118: do a card's wrong answers come soon after its clock offset changes? (2026-11-19)

**The answer in plain words.** No sign of it. Every published soak of the second nine-card rig was read again, round
by round: 12,438 card-rounds in seven soaks on two days, 96 offset changes and 8 wrong answers (checks that differed
from the card's stock answer). **None of the 8 came in the first 5 rounds (about a minute) after its card's own offset
changed, and none within 5 minutes of it.** Two came within 20 rounds, both in soaks of the first day that stopped at
those very errors, so they could not have come later. Had each card's wrong answers been spread evenly over the
rounds it ran at that value, 3.58 would have fallen within 20 rounds of its own change; 2 did. **Counted from the last
change of any card on the rig**, 4 of the 8 fell within 20 rounds, where an even spread gives 3.80: 4 or more happen
by chance 59.7 % of the time. The two errors that raised the question ([R201](../2026-11-18_r201_soak_ladder/README.md):
cards 2 and 6 in the 6th and 12th round after the ladder's last raise, a raise that did not change those two cards)
look rare only when a card's rounds are taken from one boot: both cards had already run the same values for 133 and
157 rounds in [R200](../2026-11-17_r200_climb_rig2/README.md)'s soaks, most of them soon after an `apply`.

**What it means.**

- **A soak's first minutes are not special on this evidence: count every round of a rung,** as the kit does
  ([bench/mem_offset](../../../bench/mem_offset/README.md#holding-a-profile-soakpy-the-ladder-and-the-whole-memory-check)).
  Nothing here says a profile set once and left alone errs less often than its soak showed.
- **What 8 wrong answers can and cannot tell.** Six of them could have fallen early or late (the other two ended
  soaks that never got past round 13). They rule out, at 95 %, that a round in the first 20 after a card's own change
  is more than 1.88 times as likely to hold a wrong answer as a later round. For the last change of any card the
  95 % interval runs from 0.0776 to 11.1 times: a rig-wide effect of the size R201 suggested is neither shown nor
  ruled out.
- **If anything the wrong answers came late.** On a scale where 0 is the first round after the card's own change and
  1 its last round at that value, they sat at 0.689 on average, where an even spread gives 0.500; a place this late
  or later comes by chance 3.5 % of the time. That was found while looking for the opposite, so it is a lead, not a
  finding. One possible reason is heat: a card warms for some minutes after its check server starts, and a warmer card
  may err more; that is a new desk check on the watchdog's samples ([Q1615](../../../quants/experiments.md), FN119).
- **The clean soaks agree.** At values that never erred, 1,511 card-rounds fell within 20 rounds of one of 80 own
  changes (in R200's tries, [R201](../2026-11-18_r201_soak_ladder/README.md), R201's second boot and
  [R202b](../2026-11-18_r202b_core_soak/README.md)), and none differed: at those values a round right after a change
  errs less often than once in 505 (95 %).
- **What limits a soak is its length, not its start.** At the three values that erred in long runs the cards erred
  once in 438 rounds (card 2, +1,200 MT/s), once in 462 (card 6, +1,100) and twice in 305 (card 3, +1,300). A card
  that errs once in 438 rounds passes a 30-minute rung of 152 rounds clean 70.7 % of the time (est., from one error),
  and seeing its error with a 95 % chance takes 1,311 rounds, 4 h 19 min at 11.84 s a round (est.). That is why the
  use profile sits two steps under each card's lowest failure: 30 clean minutes alone pass such a value seven times in
  ten.
- **Why R201's estimate said once in 650.** It took the two errors' rounds from R201 alone and counted only rung 4's
  first 12 rounds as "after a change", with the window of 12 chosen after seeing the errors. Rung 3's first 12 rounds
  also followed a raise; with them the same estimate is once in 162, and with R200's rounds at the same values (48 of
  card 2's 438 rounds and 71 of card 6's 462 within 12 rounds of a change) once in 59. With every error and every
  round, the table below finds nothing to explain.

The next part of the question stays on the rig: a rung in a later ladder where nothing changes, which would show a
rig-wide effect directly ([Q1596](../../../quants/experiments.md)). This script reruns on every later soak's work
folder at no rig cost, so the count grows with each soak.

## What was done

At a desk, on data already copied off the rig; nothing ran on a rig or a GPU. The soaks, all on the second nine-card
P106-090 rig with the kit's greedy check (Granite 3.1 3B-A800M Instruct, two prompts of 256 tokens, every token's top-10
log-probabilities hashed, against the card's stock hash):

| Soak | Day | Rounds | What changed, and when | Wrong answers |
|---|---|---|---|---|
| R200 try 1 to try 4 | 2026-11-17 | 13, 11, 108 and 25 | each try: `apply` on a fresh boot set every card, then a soak that stopped at the first wrong answer | 4: one per try |
| R201 | 2026-11-18 | 608 | `apply` (rung 1), then three raises 30 minutes apart; rungs 2 and 3 raised all nine cards' memory, rung 4 only the cores of cards 0, 4 and 8 | 4: card 3 twice in rung 3, cards 2 and 6 once each in rung 4 |
| R201's second boot (R201b) | 2026-11-18 | 152 | `apply` of the use profile | 0 |
| R202b | 2026-11-18 | 465 | `apply` (rung 1), then two raises of the cores of cards 1, 2, 3, 5, 6 and 7 | 0 |

R200's stock control (148 rounds, every offset 0) is left out: no card had an offset.

**A card's own change** is the last time its memory or core offset was set to a new value: by `apply`, or by a ladder
raise that changed that card. The kit's ladder sets only the cards whose values go up
([bench/mem_offset/soak.py](../../../bench/mem_offset/soak.py), `raise_card`), so in R201's rung 4 cards 2 and 6 were
not set at all. **Any card's change** is the last time any card on the rig was set. Rounds and minutes are counted
per card from that moment: round 1 is the card's first soak round after it, and the minutes run to the moment the
round's result was written. After an `apply` the check servers load before the first round (R200 try 1's first round
came 4.20 minutes after its change); after a ladder raise the next round starts at once.

**The test.** A stratum is one card at one memory and core value, pooled over every soak in which it ran (card 6 at
+1,100 MT/s ran 462 rounds in R200's four tries and R201's rungs 3 and 4). If a card's chance of a wrong answer at a
value were the same in every round, the rounds holding its wrong answers would be rounds drawn at random from all the
rounds it ran at that value. Three numbers come from that, each counted from the card's own change and from any card's:

- **Early count:** wrong answers within the first 20 rounds (or 5 minutes), against its exact distribution under even
  spreading, one hypergeometric draw per stratum. A stratum whose every round was early (R200's tries 1 and 2 stopped
  at their errors in rounds 13 and 11) cannot fall otherwise and moves nothing.
- **Rate ratio:** how many times as likely a wrong answer is in an early round as in a later one, the most likely
  value and its exact 95 % interval (every ratio under which the early count seen is not in either 2.5 % tail).
- **Average place:** each wrong answer's place among its stratum's rounds ordered by minutes since the change (0 the
  earliest, 1 the latest), averaged; its distribution from 100,000 random draws (seed 118). It needs no window.

R200's tries stopped at their first wrong answer, so their strata end at an error, and drawing rounds evenly stands in
for a fixed chance a round only approximately. The strata it touches most, R200 tries 1 and 2, are early in every
round and move nothing; at the values that erred in long runs the chance a round is one in 132 or less.

## Results

### The eight wrong answers ([tables/errors.csv](tables/errors.csv))

| Soak | Card | Round | Memory, core | Rounds, minutes since its own change | Since any card's change | Rounds the card ran at that value | Of them within 20 rounds of its own change, of any change |
|---|---|---|---|---|---|---|---|
| R200 try 1 | 4 | 13 | +1,300 MT/s, +231 MHz | 13, 6.53 | 13, 6.48 | 13 (try 1) | 13, 13 |
| R200 try 2 | 2 | 11 | +1,300, 0 | 11, 5.78 | 11, 5.70 | 24 (tries 1 and 2) | 24, 24 |
| R200 try 3 | 7 | 108 | +1,200, 0 | 108, 24.83 | 108, 24.83 | 132 (tries 1, 2 and 3) | 44, 44 |
| R200 try 4 | 7 | 25 | +1,100, 0 | 25, 8.37 | 25, 8.35 | 25 (try 4) | 20, 20 |
| R201 | 3 | 330 | +1,300, 0 | 27, 5.37 | 27, 5.37 | 305 (rungs 3 and 4) | 20, 40 |
| R201 | 3 | 384 | +1,300, 0 | 81, 16.08 | 81, 16.08 | the same 305 | 20, 40 |
| R201 | 2 | 461 | +1,200, 0 | 158, 31.40 | 6, 1.18 | 438 (R200 tries 3 and 4, R201 rungs 3 and 4) | 60, 80 |
| R201 | 6 | 467 | +1,100, 0 | 164, 32.58 | 12, 2.37 | 462 (R200 tries 1 to 4, R201 rungs 3 and 4) | 84, 104 |

### Every round by time since the change ([tables/buckets.csv](tables/buckets.csv))

Each cell: card-rounds, wrong answers. **Since the card's own change**, in rounds:

| Soak | Rounds 1 to 5 | 6 to 20 | 21 to 60 | 61 and later |
|---|---|---|---|---|
| R200 try 1 | 45, 0 | 72, 1 | 0, 0 | 0, 0 |
| R200 try 2 | 45, 0 | 54, 1 | 0, 0 | 0, 0 |
| R200 try 3 | 45, 0 | 135, 0 | 360, 0 | 432, 1 |
| R200 try 4 | 45, 0 | 135, 0 | 45, 1 | 0, 0 |
| R201 | 150, 0 | 450, 0 | 1,200, 1 | 3,672, 3 |
| R201b | 45, 0 | 135, 0 | 360, 0 | 828, 0 |
| R202b | 105, 0 | 315, 0 | 840, 0 | 2,925, 0 |
| **All** | **480, 0** | **1,296, 2** | **2,805, 2** | **7,857, 4** |

**Since any card's change**, in rounds (R200's tries and R201b are the same as above: every change was one `apply`):

| Soak | Rounds 1 to 5 | 6 to 20 | 21 to 60 | 61 and later |
|---|---|---|---|---|
| R201 | 180, 0 | 540, 2 | 1,440, 1 | 3,312, 1 |
| R202b | 135, 0 | 405, 0 | 1,080, 0 | 2,565, 0 |
| **All** | **540, 0** | **1,476, 4** | **3,285, 2** | **7,137, 2** |

In minutes, all soaks together:

| Since | Under 2 minutes | 2 to 5 | 5 to 15 | 15 and later |
|---|---|---|---|---|
| the card's own change | 330, 0 | 928, 0 | 3,784, 4 | 7,396, 4 |
| any card's change | 450, 1 | 1,125, 1 | 4,392, 4 | 6,471, 2 |

### The chance of the clustering ([tables/tests.csv](tables/tests.csv))

| Counted from | Early | Early wrong answers: seen, even spread | Chance of as many or more | Rate ratio, early to later (95 % interval) | Average place (even: 0.500); chance as early or earlier; as late or later |
|---|---|---|---|---|---|
| the card's own change | first 20 rounds | 2, 3.58 | 100.0 % | 0 (0 to 1.88) | 0.689; 96.5 %; 3.5 % |
| the card's own change | first 5 minutes | 0, 1.67 | 100.0 % | 0 (0 to 2.14) | 0.689; 96.5 %; 3.5 % |
| any card's change | first 20 rounds | 4, 3.80 | 59.7 % | 1.23 (0.0776 to 11.1) | 0.572; 75.3 %; 24.7 % |
| any card's change | first 5 minutes | 2, 1.99 | 64.2 % | 1.01 (0.0923 to 6.17) | 0.572; 75.0 %; 25.0 % |

Without R201's cards 2 and 6, the two that raised the question, the other six wrong answers give 2 early against 3.26
(own change, 20 rounds) and 2 against 3.40 (any change, 20 rounds); in both the 2 are the R200 errors that could not
have come later. Their average place is 0.692 and 0.751, as late or later by chance 5.9 % and 1.8 %.

### The values that erred ([tables/strata.csv](tables/strata.csv))

| Card | Memory, core | Rounds at it | Wrong answers | Once in | A 152-round rung clean at that rate | Rounds to see one with a 95 % chance |
|---|---|---|---|---|---|---|
| 2 | +1,200 MT/s, 0 | 438 | 1 | 438 | 70.7 % | 1,311 |
| 6 | +1,100, 0 | 462 | 1 | 462 | 71.9 % | 1,383 |
| 3 | +1,300, 0 | 305 | 2 | 152 | 36.8 % | 456 |
| 7 | +1,200, 0 | 132 | 1 | 132 | 31.5 % | 394 |
| 7 | +1,100, 0 | 25 | 1 | 25 | 0.2 % | 74 |
| 2 | +1,300, 0 | 24 | 1 | 24 | 0.2 % | 71 |
| 4 | +1,300, +231 MHz | 13 | 1 | 13 | 0.0 % | 38 |

The last four rest on soaks that stopped at their error (R200), so their rates run high; every rate here comes from
one or two errors and is an estimate.

## Files

| File | What it holds |
|---|---|
| [tables/rounds.csv.gz](tables/rounds.csv.gz) | Every card's every soak round (12,438): soak, card, round, time, rung, memory and core offset, whether it differed, the card's own last change and the rounds and minutes since it, the rig's last change and the rounds and minutes since it |
| [tables/changes.csv](tables/changes.csv) | Every offset change (96): soak, card, time, the new memory and core offset, `apply` or which raise |
| [tables/errors.csv](tables/errors.csv) | Each wrong answer: its rounds and minutes since both changes, its stratum's rounds, wrong answers and soaks, how many of the stratum's rounds were within 20 rounds of each change, and its place by minutes |
| [tables/buckets.csv](tables/buckets.csv) | Card-rounds and wrong answers by rounds (1 to 5, 6 to 20, 21 to 60, 61 and later) and by minutes (under 2, 2 to 5, 5 to 15, 15 and later) since each change, per soak and in all |
| [tables/tests.csv](tables/tests.csv) | The early count against even spreading, the rate ratio and its interval, and the average place, for every error and without R201's cards 2 and 6 |
| [tables/strata.csv](tables/strata.csv) | Each card at a value that erred: rounds, wrong answers, the soaks, the chance a 152-round rung is clean at that rate, the rounds to see one with a 95 % chance |
| [run/errors_after_change.py](run/errors_after_change.py) | `build` makes `rounds.csv.gz` and `changes.csv` from the kit's work folders; `table` makes the rest from `rounds.csv.gz` |

## Commands

On any machine, with `python3 -I` (standard library only). The tables from the published `rounds.csv.gz`:

```bash
python3 -I run/errors_after_change.py table tables/rounds.csv.gz tables
```

`rounds.csv.gz` and `changes.csv` from the work folders as the kit left them on the rig and as they were copied off
it (each with `out/climb.log`, `soak/soak.log` and `soak/soak.csv`). R200's soaks ran the kit before the ladder, whose
`soak.csv` has no offset columns; those come from R200's published `tables/soak.csv`:

```bash
D=docs/data
python3 -I run/errors_after_change.py build --out tables \
  --r200-offsets $D/2026-11-17_r200_climb_rig2/tables/soak.csv \
  --r200 try1=R200_TRY1 --r200 try2=R200_TRY2 --r200 try3=R200_TRY3 --r200 try4=R200_TRY4 \
  --kit r201=R201_WORK --kit r201b=R201B_WORK --kit r202b=R202B_WORK
```

R201's, R201b's and R202b's `soak.csv` are the published
[R201 tables/soak.csv](../2026-11-18_r201_soak_ladder/tables/soak.csv),
[R201 use-profile/soak.csv](../2026-11-18_r201_soak_ladder/use-profile/soak.csv) and
[R202b tables/soak.csv](../2026-11-18_r202b_core_soak/tables/soak.csv), byte for byte. A later soak adds one more
`--kit NAME=WORK`.

## Provenance and licence

Computed on 2026-11-19 from the second rig's soaks of 2026-11-17 and 2026-11-18 (engine build
`llama.cpp-20261113-eb312b7bc-pascal-p106-v4-b12360-c828`, the `mem_offset` kit as published with each run). The check
model is IBM's Granite 3.1 3B-A800M Instruct (Apache 2.0); nothing of it is here. The tables are this repository's
licence.
