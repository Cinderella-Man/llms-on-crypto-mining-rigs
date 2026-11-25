# R230 on a P106: no full-head drafting round after a shortlist miss against one (Q1643; 2026-11-25)

**No full-head round is slower: the draft loses the second piece of rare words.** Qwen3.8-Flash-Next's MTP draft reads
a shortlist of its output head: the ids below 32,768, the special tokens and every token the server has read or
written. When the main model's latest token is outside that list, the token joins the list, and with
`LLAMA_DRAFT_SHORTLIST=32768,8192,1` the next drafting round reads the full head. [R175](../2026-11-20_r175_flash_shortlist/README.md)
found that one round beats the v5 preset's two. This test drops the round too (`…,0`): the draft then reads the
shortlist always.
- **Where rare words go on, a loss.** The first request at 8,192 tokens writes 14 tokens new to the list. With one
  full-head round the draft guessed 3 of their next pieces (`bled` then `uncated`, `Bug` then `:**`, a partial
  character then `¹`); with none it cannot, so it accepted 170 of 254 draft tokens against 173 of 245 and took 86 steps against 83. Each step was
  shorter (117.74 and 117.87 ms against 119.42 and 119.52), but the request took 10,125.5 and 10,136.5 ms of
  generation against 9,911.5 and 9,920.1 (+215.1 ms, +2.2 %; spreads 8.6 and 11.0 ms).
- **Where the new token is a word of its own, the same.** The first request at depth 0 writes 7 new tokens and none
  continues into another new piece: the same 148 of 318 accepted and 108 steps in all four loads, 10,948.9 and
  10,954.5 ms against 10,888.4 and 10,999.7 (+7.7 ms, inside the control's 111.3 ms spread).
- **Where nothing is new, the same.** The repeats (pass 2) write tokens already in the list: +42.2, +4.2 and +3.8 ms a
  request at 0, 4,096 and 8,192 tokens, each inside its spread.
- **The replies are the same.** Every greedy reply is the same token ids in all four loads. The sampled replies are
  the same in all four loads at 0 and 8,192 tokens; at 4,096 each arm writes its own from token 209 on, the same in
  its two loads.
- **The server's own counts** say the same: per load, 31 target tokens outside the list with one round, 35 with none,
  on the same 37 new tokens (2 of them are the special tokens `</think>` and `<tool_call>`, in the list from the
  start). With one round the full head drafted 4 of the new tokens, so they never missed; with none, every new token
  that is not special is a miss.

Over the greedy requests a load took 49,042.2 and 49,050.8 ms of generation with no round against 48,706.6 and
48,840.4 with one (+273.0 ms, +0.6 %; the control's spread 133.8 ms), leaving out the first request at 4,096 tokens
(below). **The preset should not go to `…,0`.** One round stays the candidate; its coding-agent pair against the
preset's two is Q1642. Registry [Q1643](../../../quants/experiments.md) (this test), [Q1507](../../../quants/experiments.md)
(R175), [Q1863](../../../quants/experiments.md) (a way to keep the second pieces without the full head).

## What was run

- **Rig and settings:** rig 1, nine P106-090 cards with that rig's memory offsets on (+900 to +1,300 MT/s on eight
  cards, card 6 at +0, every core at stock, the 75 W default power limit) and the kit's watchdog running, both set on
  the day's boot ([bench/mem_offset](../../../bench/mem_offset/README.md)). The offsets read the same before and after
  the lane, with no watchdog trip, no step-down and no GPU Xid.
- **Model and preset:** Qwen3.8-Flash-Next as [release/flash-next-v5/settings.env](../../../release/flash-next-v5/settings.env)
  runs it (md5 `b8e950f2de8c7906c156d582416731bf`: the sparse ratio 0.75, `--ubatch-taper short=1400:64`): UkisAI's
  Swift-1.5 GSQ-RCO Q2_0 (`Swift-Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-0000{1,2,3}-of-00003.gguf`; the third shard is its
  n-gram table) and the MTP draft (`mtp-Qwen3.8-Flash-Next-draft.gguf`), on, three tokens a step. Context 65,536.
- **Build:** one for both arms, fork `pascal-p106-v5` at `eee615cdb` (the candidate `pascal-p106-v5-cand-b12674`,
  upstream b12674 merged); `llama-server` md5 `2727faf28b5b34dfe42399a9d431da67`.
- **The setting** (`common/draft-shortlist.h` at `eee615cdb`): `LLAMA_DRAFT_SHORTLIST=<k_low>,<m_cap>,<rounds>`.
  When the main model's latest token is outside the list, the next `<rounds>` drafting rounds read the full head, the
  round right after the miss included; 0 never falls back. The token joins the list at once either way, and so do the
  context's tokens (`common/speculative.cpp`). A drafting round is one step's three drafting passes, all on one head.

| Arm | Setting | Loads |
|---|---|---|
| `c1` (control) | `LLAMA_DRAFT_SHORTLIST=32768,8192,1`, R175's winner | `c1a`, `c1b` |
| `s0` | `LLAMA_DRAFT_SHORTLIST=32768,8192,0` | `s0a`, `s0b` |

The run script is R175's [run/r175.sh](../2026-11-20_r175_flash_shortlist/run/r175.sh) with `VARIANT=r230`, which
sets these two arms and their loads and nothing else; it drops the preset's own value (`32768,8192,2`), so the two
server lines differ in that value only, and every server printed its rounds in its set-up line.

The deep lane ([bench/deep_lane](../../../bench/deep_lane/README.md)) ran 18:09:02–18:45:10.
- **Loads:** a warm-up load of `c1` (untimed), then `c1a s0a s0b c1b`, each a fresh server start; the second round
  runs the arms in reverse order (Q1253). Every load read the files in 306 or 307 s (the warm-up 309 s).
- **Quiet checks:** the check before every load passed: no llama process, 0 MiB on every card.
- **Requests:** at 0, 4,096 and 8,192 tokens each load read the prompt (a `prime` request, from 4,096 on), ran 256
  tokens at temperature 0 twice (pass 1 and pass 2), then once with the preset's own sampler and a fixed seed. The
  prompts are the same ids on every load, cut from this project's recorded agent text.

## Result

### Generation per request, ms (`requests.csv`)

Each value is one load. Spread: the difference of each arm's two loads. Gain when the arm's mean is below the
control's by more than the larger spread, loss when above by more, else negligible.

| Request | Depth | New tokens | `c1` | `s0` | Arm − control | Spreads (`c1`, `s0`) | Class |
|---|---:|---:|---|---|---:|---|---|
| greedy, pass 1 | 0 | 7 | 10,888.4, 10,999.7 | 10,948.9, 10,954.5 | +7.7 | 111.3, 5.7 | negligible |
| greedy, pass 1 | 4,096 | 1 | 8,939.7, 8,918.3 | 9,219.8, 9,771.9 | +566.9 | 21.5, 552.1 | not the setting's (below) |
| greedy, pass 1 | 8,192 | 14 | 9,911.5, 9,920.1 | 10,125.5, 10,136.5 | **+215.1** | 8.5, 11.0 | **loss** |
| greedy, pass 2 | 0 | 0 | 10,472.8, 10,490.8 | 10,549.4, 10,498.7 | +42.2 | 18.0, 50.7 | negligible |
| greedy, pass 2 | 4,096 | 0 | 8,593.4, 8,597.0 | 8,587.1, 8,611.6 | +4.2 | 3.5, 24.5 | negligible |
| greedy, pass 2 | 8,192 | 0 | 8,840.4, 8,832.9 | 8,831.3, 8,849.5 | +3.8 | 7.6, 18.2 | negligible |
| sampled | 0 | 9 | 11,302.6, 11,368.6 | 11,295.8, 11,291.8 | −41.7 | 66.0, 4.0 | negligible |
| sampled | 4,096 | 3 | 8,527.7, 8,491.9 | 8,696.7, 8,720.3 | other replies | — | — |
| sampled | 8,192 | 1 | 9,005.1, 8,648.7 | 8,643.4, 8,709.7 | −150.3 | 356.4, 66.3 | negligible |

### The step and the draft ([steps.md](steps.md), [rare.md](rare.md))

A step is one pass of the main model: its generation time over its steps (the generated tokens less the accepted draft
tokens). Per load: ms (steps; accepted of drafted).

| Request | Depth | `c1` | `s0` | Step: arm − control |
|---|---:|---|---|---:|
| greedy, pass 1 | 0 | 100.82, 101.85 (108; 148/318) | 101.38, 101.43 (108; 148/318) | +0.07 |
| greedy, pass 1 | 4,096 | 113.16, 112.89 (79; 177/231) | 116.71, 123.70 (79; 177/231) | +7.18 |
| greedy, pass 1 | 8,192 | 119.42, 119.52 (83; 173/245) | 117.74, 117.87 (**86; 170/254**) | −1.67 |
| greedy, pass 2 | 0 | 97.88, 98.04 (107; 149/315) | 98.59, 98.12 (107; 149/315) | +0.39 |
| greedy, pass 2 | 4,096 | 108.78, 108.82 (79; 177/231) | 108.70, 109.01 (79; 177/231) | +0.05 |
| greedy, pass 2 | 8,192 | 113.34, 113.24 (78; 178/230) | 113.22, 113.46 (78; 178/230) | +0.05 |
| sampled | 0 | 101.83, 102.42 (111; 145/329) | 101.76, 101.73 (111; 145/329) | −0.38 |
| sampled | 4,096 | 110.75, 110.28 (77; 179/228) | 111.50, 111.80 (78; 178/230) | other replies |
| sampled | 8,192 | 120.07, 115.32 (75; 181/220) | 115.24, 116.13 (75; 181/220) | −2.00 |

- **At 8,192 tokens the step reads as a gain and the request as a loss.** Without the full head every drafting round
  is cheaper: `c1` ran 11 full-head rounds in that request (14 new tokens less the 3 its full head drafted), so its
  step is 1.67 ms longer (est. 12.6 ms a full-head round: `c1`'s time less 83 of `s0`'s steps, over 11 rounds). But
  `s0` could not draft the 3 second pieces, which became 3 steps of their own at about 118 ms each. The request time
  is the measure here, since the two arms' steps yield different numbers of tokens.
- **The new tokens** ([rare.md](rare.md), made with [run/rare.py](run/rare.py)): the replies' tokens at 32,768 or
  above that no earlier prompt or reply of the load held, replayed request by request. A token the full head can
  draft sits at most three places after a miss, and at 8,192 exactly three do: `uncated`, `:**` and `¹`, the 3
  accepted tokens `c1` has more. The 14 are `bled`,
  `uncated`, ` summarize`, `Bug`, `:**`, a partial character and `¹`, ` multiplying`, `Secondary`, `):**`, ` relied`,
  ` sentinel`, `whose` and `placed`. At depth 0 none of the 7 is followed by another new piece within a drafting
  round, so the full head had nothing to guess, and the two arms accepted the same draft tokens.
- **The sampled reply at 4,096 tokens** draws the same 3 new tokens in both arms (`(py`, then `/site` and `-packages`),
  and `c1`'s full head drafted `-packages`: one accepted token more, after which the sampler's draws part at token
  209, so the two arms write different texts from there.

### The first request at 4,096 tokens

Both `s0` loads took longer on this request (9,219.8 and 9,771.9 ms against 8,939.7 and 8,918.3) with the same reply,
the same 177 of 231 accepted draft tokens and the same 79 steps; their repeat (pass 2) was the same as the control's.
The reply holds one new token (` oddly`), so the setting changes one drafting round of it, worth est. 12.6 ms and in
`s0`'s favour. The card log shows the same clocks in all four loads with no throttle reason, and the server reported
0 graph reallocations. The host swapped out during both `s0` requests (5 and 64 MiB within a 10 s window of the RAM
watchdog's samples) and not during either `c1` request, so the time is read as the 4 GB host's, not the setting's.
The lane's own tables ([table.md](table.md)) class depth 4,096 a loss (−3.1 %) and all depths a loss (26.36 and 26.11
tok/s against 26.65 and 26.59, −1.4 %) with this request in them; without it the greedy requests are still a loss
(+273.0 ms a load, above).

### The shortlist's counts (the server log's last line per load)

| Load | Full-head rounds after a miss | Drafts with the shortlist | With the full head | Target tokens outside it | Ids in it at the end |
|---|---:|---:|---:|---:|---:|
| `c1a`, `c1b` | 1 | 768 | 31 | 31 | 33,031 |
| `s0a`, `s0b` | 0 | 803 | 0 | 35 | 33,030 |

### Checks

- **Replies** ([replies.tsv](replies.tsv), made with R69's
  [replies.py](../2026-11-20_r69_sparse_switch/run/replies.py)): every greedy reply and every prime's token the same
  ids in all four loads; the sampled replies the same in all four loads at 0 and 8,192 tokens, one group per arm at
  4,096.
- **Prompt reading** ([table.md](table.md)): the primes took 20,374.2, 20,759.8, 20,358.6 and 20,496.8 ms at 4,096
  tokens and 21,791.8, 21,138.7, 21,366.5 and 21,094.2 ms at 8,192 (`c1a`, `s0a`, `s0b`, `c1b`); a prime drafts
  nothing.
- **Memory and health:** every load's lowest host MemAvailable 198 to 266 MiB, no RAM watchdog stop, 0 graph
  reallocations in all requests; the card log ([cards.md](cards.md)): median busy clock 1,847 to 1,911 MHz, power cap
  in 14 to 42 of each card's busy samples, highest single power sample 80.53 W (card 5), no watchdog trip.

## Files

| File | What |
|---|---|
| [steps.md](steps.md), [steps.tsv](steps.tsv) | the step of every load per depth and pass type, both classes, and the shortlist's counts per load (R175's [steps.py](../2026-11-20_r175_flash_shortlist/run/steps.py)) |
| [rare.md](rare.md), [rare.tsv](rare.tsv) | per request the new tokens and each load's accepted of drafted, steps and step; per load the new tokens against the server's misses; the new tokens with their place in the reply |
| [table.md](table.md), [classes.tsv](classes.tsv), [classes_pass1.tsv](classes_pass1.tsv), [classes_pass2.tsv](classes_pass2.tsv) | the lane's own tables and classes (on the load, pass 1 and pass 2); `table.md` is the rig's byte for byte |
| [replies.tsv](replies.tsv) | per request and load, the reply's token-id and text md5 and the first token where it differs from `c1a`'s |
| [requests.csv](requests.csv) | one line per request (deep_client.py) |
| [loads.tsv](loads.tsv) | one line per load: its start, load time and the full server line |
| [cards.md](cards.md) | the card log's summary (`scripts/card_table.py`) |
| [run/rare.py](run/rare.py) | the new tokens per request, replayed from the replies, and the draft's acceptance on them |

## Commands

```bash
# on a machine that runs only ssh and tar, from the repository's root; RIG, WORK and the paths are yours
VARIANT=r230 RIG=rig WORK=/var/tmp/r230 BIN=<llama-server on the rig> MODEL=<first shard on the rig> \
  MODEL_DRAFT=<MTP draft on the rig> TEXT_LOCAL=<proxy.jsonl.bodies.gz> STEP=dry \
  bash docs/data/2026-11-20_r175_flash_shortlist/run/r175.sh
# the same with STEP=start, then STEP=wait, then STEP=collect OUT=<folder>
python3 -I bench/deep_lane/deep_table.py OUT --control c1 --out table.md --tsv classes.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c1 --class-on pass1 --tsv classes_pass1.tsv
python3 -I bench/deep_lane/deep_table.py OUT --control c1 --class-on pass2 --tsv classes_pass2.tsv
python3 -I docs/data/2026-11-20_r175_flash_shortlist/run/steps.py OUT --control c1 --arm s0 --tsv steps.tsv > steps.md
python3 -I docs/data/2026-11-25_r230_flash_shortlist0/run/rare.py OUT --control c1 --arm s0 --tsv rare.tsv > rare.md
python3 -I docs/data/2026-11-20_r69_sparse_switch/run/replies.py OUT --control-load c1a --loads c1a,s0a,s0b,c1b > replies.tsv
python3 -I scripts/card_table.py OUT/cards.csv > cards.md
```

## Provenance and licence

Measured by this project on its own rig; the tables and scripts are this repository's. The prompts are cut from this
project's own recorded agent sessions (the v3.0 release check), sent as token ids; their text is not reproduced here,
apart from the single new tokens quoted above. The model is UkisAI's Swift-1.5 GSQ-RCO Q2_0 of Qwen3.8-Flash-Next with
its MTP draft, under the makers' licences.
