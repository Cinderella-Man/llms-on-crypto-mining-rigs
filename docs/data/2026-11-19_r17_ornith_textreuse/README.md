# Text reuse in Ornith's v5 preset: four alternated agent checks on the second rig (R17's Ornith arm, Q260; 2026-11-19)

**In plain words.** On a coding agent's later turns the server sometimes reads the agent's previous reply again as
prompt: the model wrote a word in other tokens than its tokenizer would choose, the agent sends the reply back as text,
the server splits it its own way, and its cache match stops inside the reply. Text reuse (`LLAMA_SERVER_TEXT_REUSE=1`)
keeps the cached tokens wherever the new prompt spells the same text, so the reply is not read again. In Flint's two
runs it removed every such re-read, a gain on the work and neutral on time
([R17a and R17b](../2026-11-17_r17_flint_textreuse/README.md)), and it is in Flint's and Flash-Next's v5 presets. This
folder decides it for Ornith: four agent checks on one boot of the second rig, every card at its soaked clock profile,
alternating Ornith's v5 preset as it is (control) and the same plus text reuse, so the two settings share the boot, the
clocks and the hour.

**The class on two runs against two ([class.tsv](class.tsv), [below](#the-class-two-runs-against-two)): negligible on
every measure, the work included, and slower on none.** Neither text-reuse run read an earlier reply again (0 and 0
tokens a task), but control run b did not either, so the controls' own spread on re-reads (443 tokens a task, all of
control run a's) is larger than the difference of the means (221.5). Prompt tokens read a task: 3,080 and 2,886 against
3,407 and 3,008 (−224.5; the controls' spread 399). On time: 1.89 and 1.90 minutes a task at equal work against 1.87
and 1.89; 19.41 and 19.01 tok/s generation against 19.12 and 19.50; the 19 requests every run read on the same prompt
181.01 and 181.52 s against 179.73 and 181.42 s; a decode step at the runs' shared mix of contexts 51.35 and 51.28 ms
against 51.33 and 51.29 ms. The switch did its work in both runs: the server logged it keeping cached tokens on 4 and
5 later turns, 8,706 and 20,270 tokens that would otherwise have been read again (est.). Ornith spells a reply in other
tokens than its tokenizer would on few turns, so two runs a side cannot show that removal beyond the controls' spread;
on time the switch costs nothing measurable. Flint's and Flash-Next's v5 presets set it; Ornith's v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)) does not set it as of these runs.

**Control run a (R17c, `o5t0a`, 2026-11-19 18:21–18:58): 12 of 12 tasks passed, 1.87 minutes a task at equal work,
19.12 tok/s generation, and the server read two replies again: 5,321 tokens, 29.9 of the run's 370.4 seconds of
prompt reading.** As context only, the same setting on the same rig on the morning's boot (R209e's `o5oc1`): 1.90
minutes a task, 19.26 tok/s, and 2,078 reply tokens read again in 3 turns (23.4 of 370.6 s).

**Text reuse run a (R17d, `o5t1a`, 2026-11-19 19:07–19:40): 12 of 12 tasks passed, 1.89 minutes a task at equal work,
19.41 tok/s generation, and no reply was read again: 0 tokens, 0.0 of 360.4 seconds of prompt reading.** The switch
kept 2,020, 2,110, 2,225 and 2,351 cached tokens on four later turns of the two tasks whose replies control run a read
again (Python `scale-generator` and `pig-latin`), and those turns read 28, 52, 32 and 36 new tokens. Prompt tokens
read fell from 3,407 to 3,080 a task. On the measures of time this one run sits beside control run a (1.89 against
1.87 minutes a task at equal work; 19.43 against 19.45 tok/s at the runs' shared mix of contexts). No class before the
fourth run.

**Control run b (R17e, `o5t0b`, 2026-11-19 19:50–20:22): 12 of 12 tasks passed, 1.89 minutes a task at equal work,
19.50 tok/s generation, and this time the server read no reply again: 0 tokens, 0.0 of 357.9 seconds of prompt
reading.** Every later turn found the whole previous prompt and reply in the cache, as in text reuse run a. So the two
controls differ by 443 reply tokens a task on that measure (5,321 tokens in control a, none in control b), as much as
control a's whole count. On time the two controls are 0.02 minutes a task apart at equal work (1.87 and 1.89), and 1.73
seconds (0.91 %) on the 20 requests every run read on the same prompt; text reuse run a has control b's 1.89 minutes
and lies between the two controls on those requests (190.52 s against 189.23 and 190.96). The
controls' spread on every measure is in [the section below](#control-run-b-o5t0b-r17e-2026-11-19). No class before
the fourth run.

**Text reuse run b (R17f, `o5t1b`, 2026-11-19 20:32–21:18): 11 of 12 tasks passed, 1.90 minutes a task at equal work,
19.01 tok/s generation, and no reply was read again: 0 tokens, 0.0 of 326.8 seconds of prompt reading.** The switch
kept 1,297, 7,020 and 7,149 cached tokens on Python `list-ops` turns 3, 4 and 5 and 2,342 and 2,462 on `pig-latin`
turns 4 and 5; turns 4 and 5 of `list-ops`, at 12,316 and 12,451 prompt tokens, read 30 and 36 new tokens and gave a
first token after 1.42 and 1.46 s. Prompt tokens read: 2,886 a task, the fewest of the four runs. The task that failed,
Python `scale-generator`, hit the 12-minute cap on its third turn while still writing one reply (10,192 tokens when
the client stopped it); the switch acted on none of that task's turns, and Ornith has hit the cap on this task before
without it ([2026-10-27](../2026-10-27_ornith/agent/v2/compare.md),
[2026-11-05](../2026-11-05_release_regression/agent/ornith-ud-stock/compare.md)). At the shared mix of contexts a decode step took 51.28 ms, control run b's 51.29.

## The runs

R17's Ornith arm runs the release check's coding-agent arm ([bench/agent_check](../../../bench/agent_check/README.md):
12 bug-fix tasks, Python and JavaScript alternating, a 12-minute cap a task, a 50-minute budget a run) on
Ornith-1.5-122B Pascal N2P GGUF (valid1, `Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf`) at 49,152 tokens on nine
P106-090 cards, on the v5 engine (`pascal-p106-v5` at `602691153`, upstream b12483 merged) with Ornith's v5 preset
([release/ornith-v5/settings.env](../../../release/ornith-v5/settings.env)). The four runs differ only in the arm's name
and one variable:

| Run | Arm | Server variables added to the v5 preset | Status |
|---|---|---|---|
| R17c, control a | `o5t0a` | none | done 18:21–18:58, 12/12 |
| R17d, text reuse a | `o5t1a` | `LLAMA_SERVER_TEXT_REUSE=1` | done 19:07–19:40, 12/12 |
| R17e, control b | `o5t0b` | none | done 19:50–20:22, 12/12 |
| R17f, text reuse b | `o5t1b` | `LLAMA_SERVER_TEXT_REUSE=1` | done 20:32–21:18, 11/12 |

The class is two runs against two by [class_arms.py](../../../bench/agent_check/class_arms.py), as for Flint: a
measure is a gain or a loss only where the difference of the two arms' means is beyond the larger of their spreads
([the class](#the-class-two-runs-against-two)). Beside them, as context, R209e's `o5oc1`: the
same rig, clock profile, engine, preset and started command on the morning's boot
([R209](../2026-11-19_r209_oc_v5/README.md#ornith-with-the-offsets-o5oc1-2026-11-19)).

## Control run a: `o5t0a` (R17c, 2026-11-19)

| | R209e `o5oc1` (context: the same setting, the morning's boot) | **R17c `o5t0a`** | change |
|---|---|---|---|
| passed | 12/12 | **12/12** | |
| minutes a task at equal work (`v3ka`'s work) | 1.90 | **1.87** | −1.58 % |
| generation tok/s, session | 19.26 | **19.12** | −0.73 % |
| ms a decode step at the two runs' shared mix of contexts; tok/s at it | 52.37; 19.10 | **52.00; 19.23** | −0.71 % |
| prompt tok/s, session | 101.6 | **110.4** | +8.66 % (fewer short reads) |
| the 22 requests read on the same prompt in both, prompt s | 211.80 | **210.42** | −0.65 % |
| of them the 12 first turns, prompt s | 110.65 | **109.19** | −1.32 % |
| first token median; p90 s | 5.37; 10.04 | **7.03; 10.60** | +30.91 %; +5.58 % (the turn mix) |
| first token on a task's first turn, median s | 7.94 | **7.95** | |
| prompt tokens read (a task) | 37,657 (3,138) | **40,888 (3,407)** | |
| reply tokens re-read (turns; their prompt s of all prompt s) | 2,078 (3; 23.4 of 370.6) | **5,321 (2; 29.9 of 370.4)** | |
| reply tokens re-read a task; their prompt s a task | 173; 1.95 | **443; 2.49** | |
| longest first-token wait s (task, turn) | 26.88 (Python `affine-cipher`, 1) | **26.47 (Python `affine-cipher`, 1)** | |
| tokens generated (a task) | 20,222 (1,685) | **20,554 (1,713)** | |
| minutes, all 12 tasks; turns | 24.1; 67 | **24.5; 58** | |
| tool calls / failed / malformed | 67 / 1 / 0 | **58 / 1 / 0** | |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 14 / 0 / 0; 0; 0 in 67 | **13 / 0 / 0; 0; 0 in 58** | |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **the same on every card (2,127.0)** | |
| hottest card; highest power | 45 °C (card 0); 89.80 W (card 7) | **47 °C (card 0); 85.04 W (card 6)** | |
| watchdog trips; GPU Xid | 0; 0 | **0; 0** | |
| lowest MemAvailable MiB; swap MiB during the load | 1,183; 52 → 362 | **1,233; 52 → 362** | |

- **The two re-read turns** ([work.md](work.md)): Python `pig-latin` turn 4 (1,441 reply tokens, 9.87 s of prompt) and
  Python `scale-generator` turn 5 (3,880 reply tokens, 20.07 s, a first token after 20.17 s). In both the cache match
  stopped 4 tokens before the previous prompt's end, so the whole reply was read again. 44 of the 46 later turns found
  the whole previous prompt and reply in the cache (median hit 94.2 %, [o5t0a_gonogo.md](o5t0a_gonogo.md)).
- **The longest wait was not a re-read:** Python `affine-cipher`'s first turn reads a 2,742-token prompt with nothing
  cached, 26.47 s here and 26.88 s in `o5oc1`; text reuse cannot shorten it.
- **Like for like the speed is `o5oc1`'s:** a decode step at the shared mix 0.71 % shorter, the 22 same-prompt
  requests 0.65 % faster ([decode.md](decode.md), [same_prompts.md](same_prompts.md)), every card's busy SM clock
  median the same to the MHz ([clocks.tsv](clocks.tsv)). Session prompts look faster (110.4 against 101.6 tok/s) only
  through the mix: 33 of 58 requests read fewer than 512 new tokens, against 43 of 67 ([reallocs.tsv](reallocs.tsv)).
  The first-token median is higher for the same reason: 46 later turns against 55.
- **Answers sound:** every task passed, also its hidden tests, test files untouched; 0 of 58 calls malformed, none
  inside the reasoning; the one failed call (Python `affine-cipher`) is one the tool answered with an error. No NaN or
  inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load, as for
  every Ornith run of this preset).
- **Heat and power** ([cards.md](cards.md), the card log every 0.5 s over the tasks): above 80 W in 3 4 4 1 5 5 4 4 2
  single samples on cards 0 to 8; the clock watchdog (70 °C, 80 W for 5 s) never tripped and the boot logged no GPU
  Xid.

The same-prompt and shared-mix figures in this section are those of `o5oc1` and `o5t0a` alone (22 shared requests,
five context buckets). From text reuse run a on, [same_prompts.md](same_prompts.md) and [decode.md](decode.md) compare
every run so far over what all of them share (21 requests and four buckets with three runs).

## Text reuse run a: `o5t1a` (R17d, 2026-11-19)

| | R17c `o5t0a` (control a) | **R17d `o5t1a`** | change |
|---|---|---|---|
| passed | 12/12 | **12/12** | |
| minutes a task at equal work (`v3ka`'s work) | 1.87 | **1.89** | +1.07 % |
| generation tok/s, session | 19.12 | **19.41** | +1.52 % |
| ms a decode step at the three runs' shared mix of contexts; tok/s at it | 51.42; 19.45 | **51.47; 19.43** | +0.10 % |
| prompt tok/s, session | 110.4 | **102.5** | −7.16 % (the mix) |
| the 21 requests read on the same prompt in all three runs, prompt s | 199.77 | **201.05** | +0.64 % |
| of them the 12 first turns, prompt s | 109.19 | **110.39** | +1.10 % |
| first token median; p90 s | 7.03; 10.60 | **6.13; 9.58** | −12.80 %; −9.62 % |
| first token on a task's first turn, median s | 7.95 | **8.00** | +0.63 % |
| prompt tokens read (a task) | 40,888 (3,407) | **36,956 (3,080)** | −9.62 % |
| reply tokens re-read (turns; their prompt s of all prompt s) | 5,321 (2; 29.9 of 370.4) | **0 (0; 0.0 of 360.4)** | −100 % |
| reply tokens re-read a task; their prompt s a task | 443; 2.49 | **0; 0.00** | |
| turns where text reuse kept cached tokens | | **4** | |
| longest first-token wait s (task, turn) | 26.47 (Python `affine-cipher`, 1) | **26.70 (Python `affine-cipher`, 1)** | |
| tokens generated (a task) | 20,554 (1,713) | **17,458 (1,455)** | −15.06 % |
| minutes, all 12 tasks; turns | 24.5; 58 | **21.4; 64** | |
| tool calls / failed / malformed | 58 / 1 / 0 | **64 / 2 / 0** | |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 13 / 0 / 0; 0; 0 in 58 | **11 / 0 / 0; 0; 0 in 64** | |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | **2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8)** | cards 2, 3, 4 one step lower |
| hottest card; highest power | 47 °C (card 0); 85.04 W (card 6) | **47 °C (card 0); 85.84 W (card 7)** | |
| watchdog trips; GPU Xid | 0; 0 | **0; 0** | |
| lowest MemAvailable MiB; swap MiB during the load | 1,233; 52 → 362 | **1,189; 53 → 363** | |

- **The switch's four turns** (the server's log line `prompt ids differ from the cache at <n>, same text: <k> more
  cached tokens kept`): Python `scale-generator` turns 4 and 5 (ids differ at token 5,511; 2,020 and 2,110 more tokens
  kept; 28 and 52 new tokens read in 1.09 and 1.63 s; first token after 1.18 and 1.70 s) and Python `pig-latin` turns
  4 and 5 (at 3,798; 2,225 and 2,351 kept; 32 and 36 read in 1.19 and 1.22 s; first token after 1.27 and 1.28 s). In
  each pair the second turn meets the same point again: the cache holds the model's own tokens for the earlier reply,
  and the agent sends that reply back as the same text in other tokens. These are the two tasks whose replies control
  run a read again (`pig-latin` turn 4, 1,441 tokens; `scale-generator` turn 5, 3,880 tokens). All 52 later turns
  found the whole previous prompt and reply in the cache (median hit 96.9 %, [o5t1a_gonogo.md](o5t1a_gonogo.md)).
- **Fewer tokens, not faster ones:** the run generated 17,458 tokens against 20,554, so the 12 tasks took 21.4 minutes
  against 24.5; at equal work (each run scaled to `v3ka`'s tokens) 1.89 against 1.87 minutes a task. Like for like the
  decode step at the shared mix is 51.47 against 51.42 ms and the 21 same-prompt requests 201.05 against 199.77 s; for
  scale, the same setting on two boots (`o5oc1` and `o5t0a`) differed by 0.71 % and 0.65 % on those two measures. The
  class waits for the fourth run.
- **First tokens:** the median and p90 are lower (6.13 and 9.58 s against 7.03 and 10.60): no turn read 1,441 or 3,880
  reply tokens, and the turn mix differs (52 later turns against 46). The longest wait is again Python
  `affine-cipher`'s first turn (2,742 prompt tokens, nothing cached), 26.70 s; text reuse cannot shorten it.
- **Answers sound:** every task passed, also its hidden tests, test files untouched; 0 of 64 calls malformed, none
  inside the reasoning; the two failed calls (Python `affine-cipher`) are calls the tool answered with an error. No
  NaN or inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load).
- **Heat and power** ([cards.md](cards.md)): above 80 W in 1 3 4 2 3 0 2 3 2 single samples on cards 0 to 8; the
  peak temperature 1 °C above control a's on cards 1 to 8 and the same on card 0; the clock watchdog never tripped and
  the boot logged no GPU Xid. The busy SM clock median is control a's on six cards and one driver step lower on cards 2,
  3 and 4 ([clocks.tsv](clocks.tsv)).
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,189 MiB (19:40:37, as the run ended); each card's
  peak memory 22 18 22 18 18 18 18 14 14 MiB below control a's on cards 0 to 8, with a longest context of 7,916 tokens
  against 10,752.

The same-prompt and shared-mix figures in this section are those of the first three runs (21 shared requests, four
context buckets). From control run b on, [same_prompts.md](same_prompts.md) and [decode.md](decode.md) compare the
four runs so far over what all of them share (20 requests, the same four buckets weighted by the four runs' tokens).

## Control run b: `o5t0b` (R17e, 2026-11-19)

Control run b beside control run a and text reuse run a. The last column is the two controls' spread: the difference
between them, and its share of their mean. The class after the fourth run calls a difference between the two settings
a gain or a loss only where it is beyond the larger of the two settings' spreads.

| | R17c `o5t0a` (control a) | R17d `o5t1a` (text reuse a) | **R17e `o5t0b` (control b)** | the controls' spread |
|---|---|---|---|---|
| passed | 12/12 | 12/12 | **12/12** | |
| minutes a task at equal work (`v3ka`'s work) | 1.87 | 1.89 | **1.89** | 0.02 (1.06 %) |
| generation tok/s, session | 19.12 | 19.41 | **19.50** | 0.38 (1.97 %) |
| ms a decode step at the four runs' shared mix of contexts; tok/s at it | 51.40; 19.46 | 51.44; 19.44 | **51.37; 19.47** | 0.03 ms (0.06 %) |
| prompt tok/s, session | 110.4 | 102.5 | **100.9** | 9.5 (8.99 %; the mix) |
| the 20 requests read on the same prompt in all four runs, prompt s | 189.23 | 190.52 | **190.96** | 1.73 (0.91 %) |
| of them the 12 first turns, prompt s | 109.19 | 110.39 | **110.74** | 1.55 (1.41 %) |
| first token median s | 7.03 | 6.13 | **5.64** | 1.39 (21.94 %) |
| first token p90 s | 10.60 | 9.58 | **9.62** | 0.98 (9.69 %) |
| first token on a task's first turn, median s | 7.95 | 8.00 | **8.03** | 0.08 (1.00 %) |
| prompt tokens read (a task) | 40,888 (3,407) | 36,956 (3,080) | **36,100 (3,008)** | 399 a task (12.44 %) |
| reply tokens re-read (turns; their prompt s of all prompt s) | 5,321 (2; 29.9 of 370.4) | 0 (0; 0.0 of 360.4) | **0 (0; 0.0 of 357.9)** | |
| reply tokens re-read a task; their prompt s a task | 443; 2.49 | 0; 0.00 | **0; 0.00** | 443; 2.49 (all of control a's) |
| longest first-token wait s (task, turn) | 26.47 (Python `affine-cipher`, 1) | 26.70 (the same) | **26.96 (the same)** | 0.49 (1.83 %) |
| tokens generated (a task) | 20,554 (1,713) | 17,458 (1,455) | **16,209 (1,351)** | 362 a task (23.64 %) |
| minutes, all 12 tasks; turns | 24.5; 58 | 21.4; 64 | **20.3; 66** | 4.2 (18.75 %); 8 |
| tool calls / failed / malformed | 58 / 1 / 0 | 64 / 2 / 0 | **66 / 2 / 0** | |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 13 / 0 / 0; 0; 0 in 58 | 11 / 0 / 0; 0; 0 in 64 | **11 / 0 / 0; 0; 0 in 66** | |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | 2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8) | **2113 2138 2138 2088 2100 2088 2151 2151 2113 (2,120.0)** | one driver step on cards 2, 3, 4, 5, 8 |
| hottest card; highest power | 47 °C (card 0); 85.04 W (card 6) | 47 °C (card 0); 85.84 W (card 7) | **47 °C (card 0); 88.43 W (card 3)** | |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | **0; 0** | |
| lowest MemAvailable MiB; swap MiB during the load | 1,233; 52 → 362 | 1,189; 53 → 363 | **1,225; 53 → 363** | |

- **No reply read again without the switch, this time:** all 54 later turns found the whole previous prompt and reply
  in the cache (median hit 98.4 %, [o5t0b_gonogo.md](o5t0b_gonogo.md)); the lowest hit, 44.6 %, is a second turn
  (JavaScript `complex-numbers`), which reads the first tool result. Control run a's two re-reads were Python
  `pig-latin` turn 4 and `scale-generator` turn 5; here those tasks ran 6 and 5 turns without one. A re-read happens
  only when the model spells a word of its reply in other tokens than the tokenizer would, so a run can have none.
- **What the spread means for the class:** on reply tokens re-read the two controls' spread (443 a task) is control a's
  whole count. The control setting's mean is 221.5 a task, so even with no re-read in text reuse run b the difference
  of the means (221.5) stays inside the controls' spread. On the time measures the controls' spread is 0.02 minutes a
  task at equal work, 0.38 tok/s of generation, 0.03 ms a decode step at the shared mix and 1.73 s on the 20
  same-prompt requests. Text reuse run a is at or between the two controls on three of these (1.89 minutes a task,
  19.41 tok/s, 190.52 s) and 0.04 ms above the slower control on the decode step (51.44 against 51.40 ms).
- **Fewer tokens again:** control run b generated 16,209 tokens, the fewest of the four runs so far, in 20.3 minutes
  for the 12 tasks; at equal work it is 1.89 minutes a task, text reuse run a's figure. The first-token median (5.64 s)
  is the lowest so far: no re-read turn, and 54 later turns against 46 in control a.
- **Answers sound:** every task passed, also its hidden tests, test files untouched; 0 of 66 calls malformed, none
  inside the reasoning; the two failed calls (Python `affine-cipher`) are calls the tool answered with an error. No
  NaN or inf, no watchdog kill. The report's gates: all GO except memory (NO-GO on the swap growth during the load).
- **Heat and power** ([cards.md](cards.md)): above 80 W in 1 6 1 4 1 1 2 3 3 single samples on cards 0 to 8; the
  highest, 88.43 W, on card 3; peak temperature the same as control a's on cards 0 and 7 and 1 to 2 °C higher on the
  others (card 2 39 °C against 37). The clock watchdog (70 °C, 80 W for 5 s) never tripped and the boot logged no GPU
  Xid. The busy SM clock median is control a's on cards 0, 1, 6 and 7 and one driver step lower on cards 2, 3, 4, 5 and
  8 ([clocks.tsv](clocks.tsv)).
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,225 MiB (20:21:44, as the run ended); each card's
  peak memory the same as text reuse run a's, 22 18 22 18 18 18 18 14 14 MiB below control a's on cards 0 to 8, with
  a longest prompt of 7,160 tokens (control a 10,752).

## Text reuse run b: `o5t1b` (R17f, 2026-11-19)

The fourth run beside the other three. From this run on, [same_prompts.md](same_prompts.md) and
[decode.md](decode.md) compare the five runs (with `o5oc1`) over what all of them share: 19 requests, and the same four
context buckets weighted by the five runs' tokens. The sections above keep their own figures.

| | R17c `o5t0a` (control a) | R17d `o5t1a` (text reuse a) | R17e `o5t0b` (control b) | **R17f `o5t1b` (text reuse b)** |
|---|---|---|---|---|
| passed | 12/12 | 12/12 | 12/12 | **11/12** (Python `scale-generator` at the 12-minute cap) |
| minutes a task at equal work (`v3ka`'s work) | 1.87 | 1.89 | 1.89 | **1.90** |
| generation tok/s, session | 19.12 | 19.41 | 19.50 | **19.01** |
| ms a decode step at the five runs' shared mix of contexts; tok/s at it | 51.33; 19.48 | 51.35; 19.48 | 51.29; 19.50 | **51.28; 19.50** |
| prompt tok/s, session | 110.4 | 102.5 | 100.9 | **106.0** |
| the 19 requests read on the same prompt in all five runs, prompt s | 179.73 | 181.01 | 181.42 | **181.52** |
| of them the 12 first turns, prompt s | 109.19 | 110.39 | 110.74 | **110.77** |
| first token median; p90 s | 7.03; 10.60 | 6.13; 9.58 | 5.64; 9.62 | **6.94; 10.31** |
| first token on a task's first turn, median s | 7.95 | 8.00 | 8.03 | **7.98** |
| prompt tokens read (a task) | 40,888 (3,407) | 36,956 (3,080) | 36,100 (3,008) | **34,628 (2,886)** |
| reply tokens re-read (turns; their prompt s of all prompt s) | 5,321 (2; 29.9 of 370.4) | 0 (0; 0.0 of 360.4) | 0 (0; 0.0 of 357.9) | **0 (0; 0.0 of 326.8)** |
| turns where text reuse kept cached tokens; the tokens kept | | 4; 8,706 | | **5; 20,270** |
| longest first-token wait s (task, turn) | 26.47 (Python `affine-cipher`, 1) | 26.70 (the same) | 26.96 (the same) | **26.89 (the same)** |
| tokens generated (a task) | 20,554 (1,713) | 17,458 (1,455) | 16,209 (1,351) | **21,478 (1,790)**, and 10,192 in the capped reply |
| minutes, all 12 tasks; turns | 24.5; 58 | 21.4; 64 | 20.3; 66 | **34.8; 55** |
| tool calls / failed / malformed | 58 / 1 / 0 | 64 / 2 / 0 | 66 / 2 / 0 | **55 / 1 / 0** |
| checkpoints restored / created / erased; full re-processing; graph reallocations | 13 / 0 / 0; 0; 0 in 58 | 11 / 0 / 0; 0; 0 in 64 | 11 / 0 / 0; 0; 0 in 66 | **11 / 0 / 0; 0; 0 in 54** |
| busy SM clock medians, cards 0–8, MHz (mean) | 2113 2138 2151 2100 2113 2100 2151 2151 2126 (2,127.0) | 2113 2138 2138 2088 2100 2100 2151 2151 2126 (2,122.8) | 2113 2138 2138 2088 2100 2088 2151 2151 2113 (2,120.0) | **control b's on every card (2,120.0)** |
| hottest card; highest power | 47 °C (card 0); 85.04 W (card 6) | 47 °C (card 0); 85.84 W (card 7) | 47 °C (card 0); 88.43 W (card 3) | **47 °C (card 0); 87.85 W (card 1)** |
| watchdog trips; GPU Xid | 0; 0 | 0; 0 | 0; 0 | **0; 0** |
| lowest MemAvailable MiB; swap MiB during the load | 1,233; 52 → 362 | 1,189; 53 → 363 | 1,225; 53 → 363 | **1,227; 53 → 363** |

- **The switch's five turns** (`grep 'same text:' rig/server-o5t1b.log`): Python `list-ops` turns 3, 4 and 5 (ids
  differ at token 5,266 each time; 1,297, 7,020 and 7,149 more tokens kept; 361, 30 and 36 new tokens read; first
  token after 6.94, 1.42 and 1.46 s) and Python `pig-latin` turns 4 and 5 (at 3,749; 2,342 and 2,462 kept; 32 and 36
  read; first token after 1.39 and 1.29 s). As in text reuse run a, the later turn of a pair meets the same point again:
  the cache holds the model's own tokens for an earlier reply, and the agent sends that reply back as the same text in
  other tokens. All later turns with a timed reply found the whole previous prompt and reply in the cache (median hit
  94.8 %, [o5t1b_gonogo.md](o5t1b_gonogo.md)).
- **The capped task:** Python `scale-generator`'s third turn gave its first token after 7.26 s and was still writing
  one reply, 10,192 tokens at 17.01 tok/s with its context grown to 16,497 tokens, when the 12-minute cap stopped it
  (720.25 s for the task). The switch kept no token on any of its three turns. The capped reply has no timings, so the
  report's proxy-row check is NO-GO for it (3 rows, 2 replies) and the run's tables count 54 timed requests of 55. The
  agent sends no seed, so each run writes different replies; this task took 157 s in control run b and Ornith has hit
  the cap on it before (2026-10-27's `v2` and 2026-11-05's `ornith-ud-stock` runs, neither with the switch).
- **Generation and the mix:** 19.01 tok/s over the session, the lowest of the four, because 5,697 of its 21,478 tokens
  were decoded at 8,000 tokens of context and more (Python `list-ops`, to 12,451 tokens), against 4,770 of 20,554 in
  control run a and none in text reuse run a or control run b; at the shared mix a decode step took 51.28 ms, control run b's 51.29 ([decode.md](decode.md)). The wall time of 34.8
  minutes holds the capped task's 12.0 and `list-ops`' 8.6 (5,363 tokens in one reply); at equal work it is 1.90
  minutes a task.
- **Answers sound:** 11 tasks passed, also their hidden tests, test files untouched; 0 of 55 calls malformed, none
  inside the reasoning; the one failed call (Python `affine-cipher`) is one the tool answered with an error. No NaN or
  inf, no watchdog kill. The report's gates: GO except memory (the swap growth during the load, as always) and the proxy
  rows (the capped reply).
- **Heat and power** ([cards.md](cards.md)): above 80 W in 1 5 4 2 1 0 3 3 5 single samples on cards 0 to 8, the
  highest 87.85 W on card 1; peak temperature control run b's on every card but card 7 (41 °C against 40). The clock
  watchdog (70 °C, 80 W for 5 s) never tripped and the boot logged no GPU Xid. The busy SM clock median is control run
  b's on every card ([clocks.tsv](clocks.tsv)).
- **Host and cards** ([host.tsv](host.tsv)): lowest MemAvailable 1,227 MiB (21:01:18); each card's peak memory control
  run a's to the MiB, with a longest prompt of 12,451 tokens.

## The class: two runs against two

`class_arms.py agent.tsv o5t1=o5t1a,o5t1b o5t0=o5t0a,o5t0b` ([class.tsv](class.tsv)). A measure is a gain or a loss
only where the difference of the two settings' means is beyond the larger of their spreads (each setting's highest run
minus its lowest); otherwise it is negligible. The last three rows are not in `class_arms.py`; they are the same rule
applied to [same_prompts.md](same_prompts.md) and [decode.md](decode.md).

| Measure | text reuse: R17d, R17f | control: R17c, R17e | Difference of the means (the larger spread) | Class |
|---|---|---|---|---|
| minutes a task at equal work | 1.89, 1.90 | 1.87, 1.89 | +0.015, +0.8 % (0.02) | negligible |
| generation tok/s | 19.41, 19.01 | 19.12, 19.50 | −0.10, −0.5 % (0.40) | negligible |
| prompt tok/s | 102.5, 106.0 | 110.4, 100.9 | −1.4, −1.3 % (9.5) | negligible |
| first token median s | 6.13, 6.94 | 7.03, 5.64 | +0.20, +3.2 % (1.39) | negligible |
| first token p90 s | 9.58, 10.31 | 10.60, 9.62 | −0.165, −1.6 % (0.98) | negligible |
| first token on a task's first turn, median s | 8.00, 7.98 | 7.95, 8.03 | 0.00 (0.08) | negligible |
| prompt tokens read a task | 3,080, 2,886 | 3,407, 3,008 | −224.5, −7.0 % (399) | negligible |
| reply tokens re-read a task | 0, 0 | 443, 0 | −221.5, −100 % (443) | negligible |
| prompt s of the turns that re-read a reply, a task | 0, 0 | 2.49, 0 | −1.245, −100 % (2.49) | negligible |
| longest first-token wait s | 26.70, 26.89 | 26.47, 26.96 | +0.08, +0.3 % (0.49) | negligible |
| the 19 same-prompt requests, prompt s | 181.01, 181.52 | 179.73, 181.42 | +0.69, +0.4 % (1.69) | negligible |
| of them the 12 first turns, prompt s | 110.39, 110.77 | 109.19, 110.74 | +0.615, +0.6 % (1.55) | negligible |
| ms a decode step at the shared mix | 51.35, 51.28 | 51.33, 51.29 | +0.005, +0.0 % (0.07) | negligible |

- **No measure is slower beyond the spread,** so by the project's rule the switch is a negligible change that costs
  nothing measurable, not a loss.
- **On the work the rule cannot see the switch on two runs a side.** Ornith re-read a reply on 2 of control run a's 46
  later turns and on none of control run b's 54, so the controls' own spread on re-reads is control run a's whole count.
  The server's log shows what the switch removed in its own runs: on 4 and 5 later turns the new prompt left the
  cache's tokens at a word the model had spelled in other tokens, and the switch kept 8,706 and 20,270 cached tokens
  there. Without it those tokens would have been read again as prompt (est.; at control run a's rate for its re-reads,
  5,321 tokens in 29.9 s, that is 49 s and 114 s of prompt reading, est.). Flint, whose replies re-read on far more
  turns, showed the same removal as a gain ([R17a and R17b](../2026-11-17_r17_flint_textreuse/README.md#the-class-two-runs-against-two)).
- **Prompt tokens read a task** fell in both text-reuse runs (3,080 and 2,886) below both controls' (3,407 and 3,008),
  but the controls differ by 399 a task, more than the difference of the means (224.5).

## How it ran

- **The rig:** the second rig, nine P106-090, booted at 14:04 and set to its soaked clock profile at 14:13 by the
  offset kit's `apply` ([R211](../2026-11-19_r209_oc_v5/README.md#the-profile-again-on-the-boot-of-1404)): memory
  +1,300 +1,000 +1,000 +1,100 +1,100 +1,200 +900 +900 +1,300 MT/s and core +256 +256 +269 +231 +206 +206 +282 +256 +231
  MHz on cards 0 to 8, the 75 W power limit, the kit's watchdog running. Before and after each run every card read back
  at those values, no load process, 0 Xid, 0 step-downs. Nothing was set or reset during the four runs; after the
  fourth, at 21:19, every offset was set to 0 and read back and the watchdog stopped, after 0 trips since 14:07.
- **The build:** the v5 engine `602691153` (`pascal-p106-v5`, upstream b12483 merged), the rig's tarball
  `llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483` (`llama-server` md5 `86607661d5a19d1a90d19ee4db10c90b`,
  checked before and after).
- **The started command** (549 bytes with its newline, md5 `a0a97e752b2acbfef50bbaf86250e1e5`) is R209e's `o5oc1`
  byte for byte, and the server's `switches:` line the same after `switches:` (7 set, the v5 preset's):

  ```
  GGML_CUDA_KEEPALIVE=decode GGML_CUDA_KEEPALIVE_EAGER=1 LLAMA_SERVER_YIELD=2 LLAMA_SAMPLING_TOP_K_ROWS=1
    GGML_CUDA_HOP_STAGED=1 GGML_CUDA_GRAPHS_FORCE=1 GGML_CUDA_MMID_GLU_UNFUSE=2
  llama-server -ngl 99 -fa on -c 49152 -np 1 -bs --ctx-checkpoints 4 --cache-ram 0 --jinja --reasoning-format deepseek
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 --metrics --no-webui --temp 0.6 -b 4096 -ub 512
    -ts 6,6,6,6,5,5,5,5,6 -fit off --ctx-checkpoints-vram 4 --ctx-checkpoints-incall
  ```

  The text-reuse runs' command, dry-run through the same launcher with `EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1` (no load:
  the rig alias did not resolve, so the harness printed it and stopped), is this one with ` LLAMA_SERVER_TEXT_REUSE=1`
  after `GGML_CUDA_MMID_GLU_UNFUSE=2`, 575 bytes, nothing else changed (`cmp`). Text reuse run a started exactly that
  line (575 bytes, md5 `67aae3dba6542ab62ac9f096e24d3ae3`, `cmp` against the dry run), and its server logged
  `switches: 8 set`, the preset's seven and `LLAMA_SERVER_TEXT_REUSE=1`. Control run b started control run a's line
  byte for byte (549 bytes, `cmp`) and logged control run a's `switches:` line (7 set). Text reuse run b started text
  reuse run a's line byte for byte (575 bytes, md5 `67aae3dba6542ab62ac9f096e24d3ae3`, `cmp`) and logged text reuse
  run a's `switches:` line (8 set).
- **The client** ran on the machine that drives the rig (`CLIENT=local`) in the sandbox image `agent-check:0.87.1` at
  `--cpu-shares 256`, with the bench files R209's runs used (the agent check as of C842, 2026-11-14; since then only
  comments changed in the client's code). Each run has its own client folder and container prefix
  (`PREFIX=acr-<run>`), so another check on the same client at the same time cannot remove its containers. The work
  tables come from v5's [report.py](../../../bench/agent_check/report.py) `work` (C852) on the published rows.
- **Timeline:** control run a's server started at 18:21:56 and the client reached it at 18:32:54 (658 s; R209e:
  661 s); the 12 tasks ran 18:32:54–18:57:39; the rig's logs went to the client at 18:57:47. Text reuse run a's server
  started at 19:07:54 and was up after 653 s; the client reached it at 19:18:49, the 12 tasks ran 19:18:49–19:40:25 and
  the logs went to the client at 19:40:43. Control run b's server started at 19:50:40 and was up after 660 s; the
  client reached it at 20:01:41, the 12 tasks ran 20:01:41–20:22:11 and the logs went to the client at 20:22:19.
  Text reuse run b's server started at 20:32:28 and was up after 662 s; the client reached it at 20:43:32, the 12
  tasks ran 20:43:32–21:18:33 and the logs went to the client at 21:18:43. Nothing else ran on the rig.

## Commands

The run, from the repository's root on the machine that drives the rig:

```bash
RIG=<rig alias> CLIENT=local ARM=o5t0a EXTRA_ENV= PREFIX=acr-r17c \
  MODEL=<rig path>/Ornith-1.5-122B-N2P-N3P-valid1-headQ5_0.gguf \
  BIN=<rig path>/llama.cpp-20261117-602691153-pascal-p106-v5-cand-b12483/bin/llama-server \
  RIG_LOGDIR=<rig folder>/agent-logs RIG_PORT=8282 RELAY_PORT=8296 PROXY_PORT=18793 \
  CLIENT_DIR=<client folder>/bench CLIENT_OUT=<client folder>/r17c OUT=<out>/agent-o5t0a \
  bash docs/data/2026-11-19_r17_ornith_textreuse/run/ornith_switch.sh
# the report: the same with STEPS="report collect" OUT=<out>/agent
# a text-reuse run: ARM=o5t1a EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1, its own rig folder, client folder and PREFIX
# control run b: ARM=o5t0b EXTRA_ENV= PREFIX=acr-r17e, its own rig folder and client folder
# text reuse run b: ARM=o5t1b EXTRA_ENV=LLAMA_SERVER_TEXT_REUSE=1 PREFIX=acr-r17f, its own rig folder and client folder
```

The tables, with Python 3 (no packages; a few seconds), in a folder holding `res/turns.csv` and `res/attempts.csv`
with the rows of `v3ka` and `o5oc1` (R209's tables folder: `v3ka`'s rows as [R203](../2026-11-18_r203_ornith_offsets/README.md)
published them, `o5oc1`'s = [R209's agent/o5oc1/](../2026-11-19_r209_oc_v5/agent/o5oc1/)) and each run's rows
([agent/o5t0a/](agent/o5t0a/), [agent/o5t1a/](agent/o5t1a/), [agent/o5t0b/](agent/o5t0b/),
[agent/o5t1b/](agent/o5t1b/)), each run's server log as
`rig/server-<arm>.log`, and in `rig_<arm>/` its `meminfo.log`, `cards.csv` and `window` (`o5oc1`'s from R209's
tables; from the driver's log, `o5t0a`'s window `2026-11-19 18:32:54|2026-11-19 18:57:39`, `o5t1a`'s
`2026-11-19 19:18:49|2026-11-19 19:40:25`, `o5t0b`'s `2026-11-19 20:01:41|2026-11-19 20:22:11` and `o5t1b`'s
`2026-11-19 20:43:32|2026-11-19 21:18:33`):

```bash
bash docs/data/2026-11-19_r17_ornith_textreuse/run/tables.sh <tables folder> o5t0a o5t1a o5t0b o5t1b
python3 -I bench/agent_check/class_arms.py <tables folder>/agent.tsv o5t1=o5t1a,o5t1b o5t0=o5t0a,o5t0b > class.tsv
```

The controls' spread in control run b's section is the difference between `o5t0a`'s and `o5t0b`'s rows of
[agent.tsv](agent.tsv), [decode.md](decode.md) and [same_prompts.md](same_prompts.md), and its share of their mean. The
last three rows of the class apply `class_arms.py`'s rule to the rows of [same_prompts.md](same_prompts.md) and
[decode.md](decode.md) by hand.

The server's text-reuse lines: `grep 'same text:' rig/server-o5t1a.log rig/server-o5t1b.log` (each names the token where the prompt's ids
leave the cache's and how many more cached tokens it kept; the turn is the one whose prompt has that many tokens).

## Files

| File | Content |
|---|---|
| [class.tsv](class.tsv) | the class of text reuse (`o5t1a`, `o5t1b`) against the control (`o5t0a`, `o5t0b`) on each measure |
| [agent.tsv](agent.tsv) | per run (`v3ka`, `o5oc1`, `o5t0a`, `o5t1a`, `o5t0b`, `o5t1b`): passed, turns, prompt and generation tok/s, first token median, p90 and first turn, minutes, minutes a task at equal work against `v3ka`, then the work: prompt tokens read a task, reply tokens re-read a task and their prompt s a task, the re-read turns, the longest first-token wait |
| [work.md](work.md), [work.tsv](work.tsv) | `report.py work` on `o5oc1`, `o5t0a`, `o5t1a`, `o5t0b` and `o5t1b`: prompt tokens read, tokens generated, reply tokens re-read and their prompt seconds, the longest first-token wait; then every turn that read a reply again |
| [same_prompts.md](same_prompts.md) | the requests read on the same prompt in every run (task, turn, new tokens, cached tokens, each run's prompt ms); each run's generation over all its requests |
| [decode.md](decode.md) | each run by the context at a request's middle (requests, tokens, decode s, tok/s, ms a token), and at their shared mix of contexts |
| [reallocs.tsv](reallocs.tsv), [o5t0a_requests_reallocs.tsv](o5t0a_requests_reallocs.tsv), [o5t1a_requests_reallocs.tsv](o5t1a_requests_reallocs.tsv), [o5t0b_requests_reallocs.tsv](o5t0b_requests_reallocs.tsv), [o5t1b_requests_reallocs.tsv](o5t1b_requests_reallocs.tsv) | graph reallocations and prompt seconds per prompt size, and per request |
| [host.tsv](host.tsv) | the host's lowest MemAvailable, swap, and each card's peak memory |
| [clocks.tsv](clocks.tsv), [cards.md](cards.md) | each card's busy SM clock median; its highest temperature and power, samples above 80 W and busy memory clock median, over each run's tasks |
| [o5t0a_compare.md](o5t0a_compare.md), [o5t0a_report_table.md](o5t0a_report_table.md), [o5t0a_gonogo.md](o5t0a_gonogo.md); the same for `o5t1a` ([o5t1a_compare.md](o5t1a_compare.md), [o5t1a_report_table.md](o5t1a_report_table.md), [o5t1a_gonogo.md](o5t1a_gonogo.md)) `o5t0b` ([o5t0b_compare.md](o5t0b_compare.md), [o5t0b_report_table.md](o5t0b_report_table.md), [o5t0b_gonogo.md](o5t0b_gonogo.md)) and `o5t1b` ([o5t1b_compare.md](o5t1b_compare.md), [o5t1b_report_table.md](o5t1b_report_table.md), [o5t1b_gonogo.md](o5t1b_gonogo.md)) | the harness's report of each run: per task, per turn and the go/no-go |
| [agent/o5t0a/](agent/o5t0a/), [agent/o5t1a/](agent/o5t1a/), [agent/o5t0b/](agent/o5t0b/), [agent/o5t1b/](agent/o5t1b/) | per request (`turns.csv`) and per attempt (`attempts.csv`) rows of each run |
| [run/ornith_switch.sh](run/ornith_switch.sh), [run/tables.sh](run/tables.sh) | the launcher of the four runs and the tables |

## Provenance and licence

Measured on the project's own rig on 2026-11-19 with the repository's own scripts and Miner Llama's v5 engine (MIT, like
llama.cpp); the agent check's tasks are [bench/agent_check](../../../bench/agent_check/README.md)'s (their task IDs are
in `agent/<arm>/attempts.csv`); the tables hold times, counts and task ids, no text. The tables and scripts here are
under the repository's licence.
